# -*- coding: utf-8 -*-
"""GBDTシグナルモデルの学習と C++ ヘッダへのコンパイル

dump_features.exe が出した特徴量CSVで LightGBM 回帰を学習し、全決定木を
if-else 直書きの model.hpp に変換する。生成コードは分岐だけでヒープ確保が無く、
backtest_ml.cpp のホットパスにそのまま埋め込める。

検証用に verify.csv (特徴量 + Python側の予測値) も出力し、
backtest_ml.exe --verify で C++ 生成コードとの一致を確認する。
学習と推論で数値がズレていないことを機械的に保証するための工程で、
これが無いと「Pythonでは効いていたのにC++では効かない」を見逃す。

使い方:
  python train_export.py --features features.csv [--trees 300] [--fire-pct 95]

--features はカンマ区切りで複数CSVを受け取り、その順に連結する
(複数seedのダンプを時系列ブロックとして扱う。前80%が学習、後20%が検証)。

--null-features を渡すと、同じハイパーパラメータで「オーダーフローに情報の無い
データ」にも学習させ、両者の検証成績を並べて出す。合成フィードには板がフェアに
追随する遅れによる見かけの予測力が必ず乗るので、この差だけが本当の情報量。
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def emit_node(node: dict, lines: list, indent: str) -> None:
    if "leaf_value" in node:
        lines.append(f"{indent}return {node['leaf_value']!r};")
        return
    if node.get("decision_type") != "<=":
        raise SystemExit(f"未対応の分岐型: {node.get('decision_type')}")
    thr = float(node["threshold"])
    lines.append(f"{indent}if (f[{node['split_feature']}] <= {thr!r}) {{")
    emit_node(node["left_child"], lines, indent + "    ")
    lines.append(f"{indent}}} else {{")
    emit_node(node["right_child"], lines, indent + "    ")
    lines.append(f"{indent}}}")


def codegen(booster: lgb.Booster, threshold: float, n_feat: int, names) -> str:
    dump = booster.dump_model()
    trees = dump["tree_info"]
    lines = [
        "#pragma once",
        "// model.hpp — train_export.py が生成した GBDT 推論コード。手編集しない。",
        f"// trees={len(trees)}  features={list(names)}",
        "",
        "namespace stkmodel {",
        "",
        f"inline constexpr int N_FEATURES = {n_feat};",
        f"inline constexpr double THRESHOLD = {threshold!r};   // |pred| がこれを超えたら発火",
        "",
    ]
    for k, t in enumerate(trees):
        lines.append(f"inline double tree_{k}(const double* f) {{")
        emit_node(t["tree_structure"], lines, "    ")
        lines.append("}")
        lines.append("")
    lines.append("inline double predict(const double* f) {")
    lines.append("    double s = 0.0;")
    for k in range(len(trees)):
        lines.append(f"    s += tree_{k}(f);")
    lines.append("    return s;")
    lines.append("}")
    lines.append("")
    lines.append("}  // namespace stkmodel")
    return "\n".join(lines) + "\n"


def load(paths: str):
    df = pd.concat([pd.read_csv(p.strip()) for p in paths.split(",")], ignore_index=True)
    feat_cols = [c for c in df.columns if c not in ("event_idx", "dmid")]
    X = df[feat_cols].to_numpy(dtype=np.float64)
    y = df["dmid"].to_numpy(dtype=np.float64)
    return df, feat_cols, X, y


def train(X, y, args):
    split = int(len(y) * 0.8)  # 時系列順に前80%で学習
    train_set = lgb.Dataset(X[:split], label=y[:split])
    booster = lgb.train(
        {
            "objective": "regression",
            "learning_rate": args.lr,
            "num_leaves": args.leaves,
            "min_data_in_leaf": 200,
            "feature_fraction": 1.0,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": 7,
        },
        train_set,
        num_boost_round=args.trees,
    )
    return booster, split


def evaluate(booster, X, y, split, fire_pct):
    pred = booster.predict(X[split:])
    yv = y[split:]
    corr = float(np.corrcoef(pred, yv)[0, 1]) if yv.std() > 0 else float("nan")
    nz = yv != 0
    dir_acc = float((np.sign(pred[nz]) == np.sign(yv[nz])).mean()) if nz.any() else float("nan")
    thr = float(np.percentile(np.abs(pred), fire_pct))
    fired = np.abs(pred) > thr
    fnz = fired & nz
    fired_acc = float((np.sign(pred[fnz]) == np.sign(yv[fnz])).mean()) if fnz.any() else float("nan")
    # 発火時の平均獲得(方向 x 実際の変動)。ティック単位。
    fired_cap = float((np.sign(pred[fired]) * yv[fired]).mean()) if fired.any() else float("nan")
    return {"corr": corr, "dir_acc": dir_acc, "thr": thr,
            "fired_acc": fired_acc, "fired_cap": fired_cap, "n_fired": int(fired.sum())}


def main() -> int:
    p = argparse.ArgumentParser(description="GBDTシグナル学習 + C++コード生成")
    p.add_argument("--features", default=str(HERE / "features.csv"))
    p.add_argument("--null-features", default=None,
                   help="帰無データ(--null で出したCSV)。渡すと本番と並べて評価する")
    p.add_argument("--trees", type=int, default=300)
    p.add_argument("--leaves", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--fire-pct", type=float, default=95.0,
                   help="発火閾値 = 検証データ|pred|のこのパーセンタイル")
    p.add_argument("--out", default=str(HERE / "model.hpp"))
    p.add_argument("--verify-out", default=str(HERE / "verify.csv"))
    args = p.parse_args()

    df, feat_cols, X, y = load(args.features)
    split = int(len(df) * 0.8)
    print(f"行数: 学習={split} 検証={len(df) - split}  特徴量={len(feat_cols)}種")
    print(f"      {feat_cols}")

    booster, split = train(X, y, args)
    m = evaluate(booster, X, y, split, args.fire_pct)

    print("\n[本番データ]")
    print(f"  検証 corr(pred, dmid)      : {m['corr']:+.4f}")
    print(f"  検証 方向的中率(全サンプル): {m['dir_acc']:.1%}")
    print(f"  発火閾値 (p{args.fire_pct:.0f})           : {m['thr']:.5f}")
    print(f"  発火時の方向的中率         : {m['fired_acc']:.1%} ({m['n_fired']} 件)")
    print(f"  発火時の平均獲得           : {m['fired_cap']:+.4f} ティック")

    if args.null_features:
        _, _, Xn, yn = load(args.null_features)
        bn, sn = train(Xn, yn, args)
        mn = evaluate(bn, Xn, yn, sn, args.fire_pct)
        print("\n[帰無データ: オーダーフローに情報なし]")
        print(f"  検証 corr(pred, dmid)      : {mn['corr']:+.4f}")
        print(f"  発火時の方向的中率         : {mn['fired_acc']:.1%}")
        print(f"  発火時の平均獲得           : {mn['fired_cap']:+.4f} ティック")
        print("\n[差分 = モデルが本当に読み取った情報]")
        print(f"  corr         : {m['corr'] - mn['corr']:+.4f}")
        print(f"  方向的中率   : {(m['fired_acc'] - mn['fired_acc']) * 100:+.1f} pt")
        print(f"  平均獲得     : {m['fired_cap'] - mn['fired_cap']:+.4f} ティック")
        if m["fired_acc"] - mn["fired_acc"] < 0.01:
            print("  -> 差がほぼありません。モデルはフィードの構造を学んだだけです。")

    imp = sorted(zip(feat_cols, booster.feature_importance("gain")), key=lambda t: -t[1])
    print("\n特徴量の重要度 (gain):")
    for n, g in imp:
        print(f"  {n:<15} {g:>12.0f}")
    zero = [n for n, g in imp if g == 0]
    if zero:
        print(f"  ({', '.join(zero)} は一度も使われていません。削除候補)")

    Path(args.out).write_text(
        codegen(booster, m["thr"], len(feat_cols), feat_cols), encoding="utf-8")
    print(f"\n生成: {args.out}")

    vi = np.linspace(split, len(df) - 1, min(500, len(df) - split), dtype=int)
    vdf = pd.DataFrame(X[vi], columns=[f"f{i}" for i in range(len(feat_cols))])
    vdf["pred"] = booster.predict(X[vi])
    vdf.to_csv(args.verify_out, index=False, float_format="%.17g")
    print(f"生成: {args.verify_out} ({len(vdf)} 行) -- backtest_ml.exe --verify で照合")
    return 0


if __name__ == "__main__":
    sys.exit(main())
