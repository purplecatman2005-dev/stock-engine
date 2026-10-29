# -*- coding: utf-8 -*-
"""scan_stocks.py - 「どの銘柄に繋ぐべきか」を予測しやすさ x 儲かるかでランク付けする

hft-engine の adapters/scan_products.py の株版。

  無料・登録不要(既定):
    python scan_stocks.py
    python scan_stocks.py --tickers 7203,6758,9984 --days 5 --horizon 5

  kabuステーションの実板(口座取得後):
    python scan_stocks.py --source kabu --password <APIパスワード> --minutes 10

  ネット不要の自己検証:
    python scan_stocks.py --selftest

--------------------------------------------------------------------------
無料データでの妥協点 -- ここを理解せずに結果を読むと必ず誤解する

日本株の無料データ(yfinance)には板情報が無く、返ってくるのは1分足の
始値・高値・安値・終値・出来高だけ。だから板の非対称性である本物の OFI は
原理的に計算できない。代わりに使うのが「符号付き出来高」:

    signed_volume_t = sign(close_t - close_{t-1}) * volume_t

値上がりした足の出来高は買い主導、値下がりは売り主導と分類する(ティックルール、
Lee-Ready 法の粗い版)。これを過去 W 本ぶん合計したものを OFI の代用とする。

★ この代用指標は「過去のリターン」と機械的に相関する。だから将来リターンとの
   相関(ic)が出たとしても、それはリターンの自己相関を測っているだけかもしれない。
   そのため必ずシャッフル検定(ic_shuf)を並べて出す。ic が ic_shuf と同程度なら
   何も見つけていない。

コストの扱い:
   スプレッドが観測できないので、その価格帯の呼値(1ティック)をスプレッドの
   下限として使う。実際のスプレッドは必ずこれ以上なので、edge_bps は
   「最も楽観的な見積もり」。それでも負なら実際は確実に負ける。

--source kabu にすると板が取れるので、本物の best-level OFI に切り替わる。
--------------------------------------------------------------------------
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wire  # noqa: E402

# 東証の主要流動銘柄(既定の調査対象)。コード:名称
DEFAULT_TICKERS = {
    "7203": "トヨタ自動車",
    "6758": "ソニーグループ",
    "8306": "三菱UFJフィナンシャル・グループ",
    "9984": "ソフトバンクグループ",
    "6861": "キーエンス",
    "8035": "東京エレクトロン",
    "9432": "日本電信電話",
    "6098": "リクルートホールディングス",
    "4063": "信越化学工業",
    "7974": "任天堂",
    "6501": "日立製作所",
    "8058": "三菱商事",
    "6902": "デンソー",
    "4502": "武田薬品工業",
    "9433": "KDDI",
}

# TOPIX100構成銘柄は呼値が縮小される(1,000円以下=0.1円 など)。
# 呼値を間違えるとコストの見積もりが桁で狂うので、既定銘柄は全て縮小対象として扱う。
TOPIX100_CODES = set(DEFAULT_TICKERS)


# =====================================================================
#  指標の計算(データ源に依存しない部分)
# =====================================================================
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """順位相関。scipy を使わず順位付け -> ピアソンで計算する。"""
    if len(a) < 8:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def evaluate_signal(signal: np.ndarray, fwd_ret: np.ndarray, rng: np.random.Generator,
                    overlap: float = 1.0) -> dict:
    """シグナルと将来リターンから ic / hit / シャッフル検定 / サンプル誤差を出す。

    signal, fwd_ret は同じ長さで、同じ時点に対応していること。

    overlap は「1つのラベルが何サンプルぶん重なっているか」= ホライズンの長さ。
    毎時点で評価すると、隣り合うサンプルは同じ将来リターンをほぼ共有するので
    独立ではない。n をそのまま使うと誤差を過小評価し、ノイズを実力と誤認する。
    """
    ok = np.isfinite(signal) & np.isfinite(fwd_ret)
    s, r = signal[ok], fwd_ret[ok]
    out = {"n": int(len(s)), "n_eff": 0, "hit_se": float("nan"),
           "ic": float("nan"), "ic_shuf": float("nan"),
           "hit": float("nan"), "hit_shuf": float("nan")}
    if len(s) < 32:
        return out

    out["ic"] = spearman(s, r)
    # シャッフル検定: 将来リターンだけを並べ替える。ic はここでほぼ0になるべき。
    out["ic_shuf"] = spearman(s, rng.permutation(r))

    # |signal| 上位1/4での方向的中率(シグナルが強いときだけ張る想定)
    thr = np.quantile(np.abs(s), 0.75)
    fired = np.abs(s) >= thr
    nz = fired & (r != 0)
    if nz.any():
        out["hit"] = float((np.sign(s[nz]) == np.sign(r[nz])).mean())
        # 帰無: 方向をコイン投げにする
        coin = rng.choice((-1.0, 1.0), size=int(nz.sum()))
        out["hit_shuf"] = float((coin == np.sign(r[nz])).mean())
        # 実質の独立サンプル数と、そこから出る的中率の標準誤差(ポイント)
        n_eff = max(1, int(nz.sum() / max(1.0, overlap)))
        out["n_eff"] = n_eff
        out["hit_se"] = float(np.sqrt(0.25 / n_eff) * 100)
    return out


def breakeven_hit(spread_bps: float, vol_bps: float, fee_bps: float) -> float:
    """損益分岐の的中率(%)。これを超えないとコスト負けする。

        期待獲得 = (2*hit - 1) * vol_bps
        往復コスト = spread_bps + 2*fee_bps
        釣り合う点 -> hit = 0.5 + (spread + 2*fee) / (2*vol)

    決めているのは vol_bps / spread_bps (値幅とコストの比)であって、
    シグナルの賢さではない。この比が小さい銘柄は、どんなモデルでも勝てない。
    呼値が株価に対して大きい銘柄(値がさでない低位株や、値動きの小さい大型株)が
    これに当たる。銘柄選択がモデル改善より先に来る理由。
    """
    if not np.isfinite(vol_bps) or vol_bps <= 0:
        return float("nan")
    return 50.0 + (spread_bps + 2 * fee_bps) / (2 * vol_bps) * 100


def cost_bps(price_yen: float, code: str, fee_bps: float, tick_table=None) -> tuple:
    """呼値ベースの最小コストを返す -> (spread_bps, tick_yen)

    スプレッドは観測できないので、その価格帯の呼値をスプレッドの下限として使う。
    実際のスプレッドは必ずこれ以上なので、これを使った edge は最も楽観的な値。
    """
    if tick_table is None:
        tick_table = wire.TABLE_TOPIX100 if code in TOPIX100_CODES else wire.TABLE_STANDARD
    grid = wire.TickGrid(tick_table)
    tick_sen = grid.step_at_price(wire.yen_to_sen(price_yen))
    tick_yen = wire.sen_to_yen(tick_sen)
    return (tick_yen / price_yen * 1e4 if price_yen > 0 else float("nan")), tick_yen


def make_row(code: str, name: str, price: float, m: dict, vol_bps: float,
             fee_bps: float, tick_table=None) -> dict:
    sp_bps, tick_yen = cost_bps(price, code, fee_bps, tick_table)
    hit = m["hit"]
    hit_pct = hit * 100 if np.isfinite(hit) else float("nan")
    # 期待エッジ: 的中率から出る期待獲得 - スプレッド下限 - 手数料往復
    edge = ((2 * hit - 1) * vol_bps - sp_bps - 2 * fee_bps) if np.isfinite(hit) else float("nan")

    be = breakeven_hit(sp_bps, vol_bps, fee_bps)
    be0 = breakeven_hit(sp_bps, vol_bps, 0.0)          # 手数料ゼロの証券会社を使う場合
    margin = hit_pct - be if np.isfinite(hit_pct) else float("nan")
    se = m["hit_se"]

    # 判定: 実測が損益分岐をどれだけ上回っているかを、サンプル誤差と比べる。
    # 誤差の2倍を超えて初めて「たまたま」では説明しにくくなる。
    if not np.isfinite(be):
        verdict = "測定不能"
    elif be >= 65.0:
        verdict = "対象外"       # 呼値が大きすぎて原理的に届かない
    elif not np.isfinite(margin) or not np.isfinite(se):
        verdict = "測定不能"
    elif margin <= 0:
        verdict = "コスト負け"
    elif margin > 2 * se:
        verdict = "有望"
    else:
        verdict = "ノイズ内"     # 上回って見えるが誤差で説明できる

    return {
        "code": code, "name": name, "price": price,
        "n": m["n"], "n_eff": m["n_eff"], "hit_se_pt": se,
        "ic": m["ic"], "ic_shuf": m["ic_shuf"],
        "hit": hit, "hit_shuf": m["hit_shuf"],
        "vol_bps": vol_bps, "tick_yen": tick_yen, "spread_bps": sp_bps,
        "vol_sp": vol_bps / sp_bps if sp_bps > 0 else float("nan"),
        "breakeven_hit": be, "breakeven_hit_nofee": be0,
        "margin_pt": margin, "verdict": verdict,
        "edge_bps": edge,
        # 本当の情報量 = 帰無との差
        "ic_net": m["ic"] - m["ic_shuf"] if np.isfinite(m["ic"]) else float("nan"),
        "hit_net_pt": (hit - m["hit_shuf"]) * 100 if np.isfinite(hit) else float("nan"),
    }


# =====================================================================
#  データ源A: yfinance 1分足(無料・登録不要)
# =====================================================================
def scan_yfinance(args) -> list:
    import yfinance as yf

    tickers = {}
    if args.tickers:
        for t in args.tickers.split(","):
            t = t.strip()
            if t:
                tickers[t] = DEFAULT_TICKERS.get(t, t)
    else:
        tickers = dict(DEFAULT_TICKERS)

    symbols = [f"{c}.T" for c in tickers]
    print(f"[yfinance] {len(symbols)} 銘柄の1分足を取得中 (period={args.days}d) ...", flush=True)
    # yfinance の1分足は直近7日程度しか取れない
    data = yf.download(symbols, period=f"{args.days}d", interval="1m",
                       group_by="ticker", auto_adjust=False, progress=False, threads=True)
    if data is None or len(data) == 0:
        print("[yfinance] データが取れませんでした(市場が閉じている/期間が長すぎる/通信不可)")
        return []

    rng = np.random.default_rng(args.seed)
    rows = []
    for code, name in tickers.items():
        sym = f"{code}.T"
        try:
            df = data[sym] if sym in data.columns.get_level_values(0) else None
        except (KeyError, AttributeError):
            df = None
        if df is None:
            print(f"  {code} {name}: データなし")
            continue
        df = df.dropna(subset=["Close", "Volume"])
        if len(df) < args.window + args.horizon + 64:
            print(f"  {code} {name}: 足が少なすぎます ({len(df)} 本)")
            continue

        close = df["Close"].to_numpy(dtype=np.float64)
        volume = df["Volume"].to_numpy(dtype=np.float64)

        # 符号付き出来高(ティックルール)。板が無いので OFI の代用。
        dclose = np.diff(close, prepend=close[0])
        signed = np.sign(dclose) * volume
        # 過去 window 本の合計 = OFI の窓合計に相当
        csum = np.cumsum(signed)
        sig = np.full(len(close), np.nan)
        sig[args.window:] = csum[args.window:] - csum[:-args.window]

        # horizon 本先のリターン(bps)
        fwd = np.full(len(close), np.nan)
        fwd[:-args.horizon] = (close[args.horizon:] / close[:-args.horizon] - 1.0) * 1e4

        # 毎足で評価するのでラベルは horizon 本ぶん重なる -> 独立サンプルはその1/horizon
        m = evaluate_signal(sig, fwd, rng, overlap=args.horizon)
        # ボラ: horizon 本先リターンの標準偏差 = 取りに行く値幅の大きさ
        vol_bps = float(np.nanstd(fwd))
        rows.append(make_row(code, name, float(close[-1]), m, vol_bps, args.fee_bps,
                             wire.TABLE_TOPIX100 if args.tick_table == "topix100"
                             else (None if args.tick_table == "auto" else wire.TABLE_STANDARD)))
    return rows


# =====================================================================
#  データ源B: kabuステーションの実板(口座取得後)
# =====================================================================
def scan_kabu(args) -> list:
    """kabuステーションの板を --minutes 分観測し、本物の best-level OFI で評価する。"""
    import websocket  # websocket-client

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kabu_adapter import (KABU_HOST, KABU_PORT_LIVE, KABU_PORT_TEST,
                              kabu_login, kabu_register, kabu_symbol_info)

    if not args.password:
        raise SystemExit("--source kabu には --password が必要です。")
    codes = [t.strip() for t in (args.tickers or ",".join(DEFAULT_TICKERS)).split(",") if t.strip()]

    port = KABU_PORT_TEST if args.test_env else KABU_PORT_LIVE
    base_url = f"http://{KABU_HOST}:{port}/kabusapi"
    token = kabu_login(base_url, args.password)
    info = {}
    for c in codes:
        info[c] = kabu_symbol_info(base_url, token, c, 1)
        kabu_register(base_url, token, c, 1)
        time.sleep(0.2)
    print(f"[kabu] {len(codes)} 銘柄を登録。{args.minutes:.1f} 分観測します...", flush=True)

    # 銘柄ごとに (時刻, mid, OFI寄与) を溜める
    series = {c: {"ts": [], "mid": [], "ofi": []} for c in codes}
    prev = {c: None for c in codes}   # (bid_px, bid_qty, ask_px, ask_qty)

    ws = websocket.create_connection(f"ws://{KABU_HOST}:{port}/kabusapi/websocket", timeout=30)
    t_end = time.time() + args.minutes * 60
    try:
        while time.time() < t_end:
            raw = ws.recv()
            if not raw:
                continue
            try:
                b = json.loads(raw)
            except json.JSONDecodeError:
                continue
            code = str(b.get("Symbol"))
            if code not in series:
                continue
            bp, bq = b.get("BidPrice"), b.get("BidQty")
            ap, aq = b.get("AskPrice"), b.get("AskQty")
            if not bp or not ap or bq is None or aq is None:
                continue
            cur = (float(bp), float(bq), float(ap), float(aq))
            p = prev[code]
            if p is not None:
                # エンジンと同じ best-level OFI
                e = 0.0
                if cur[0] >= p[0]:
                    e += cur[1]
                if cur[0] <= p[0]:
                    e -= p[1]
                if cur[2] <= p[2]:
                    e -= cur[3]
                if cur[2] >= p[2]:
                    e += p[3]
                series[code]["ts"].append(time.time())
                series[code]["mid"].append(0.5 * (cur[0] + cur[2]))
                series[code]["ofi"].append(e)
            prev[code] = cur
    except KeyboardInterrupt:
        print("\n[kabu] 観測を打ち切りました", flush=True)
    finally:
        ws.close()

    rng = np.random.default_rng(args.seed)
    rows = []
    for code in codes:
        s = series[code]
        n = len(s["mid"])
        name = info.get(code, {}).get("SymbolName", code)
        if n < args.window + 64:
            print(f"  {code} {name}: 観測が少なすぎます ({n} 件)")
            continue
        ts = np.array(s["ts"])
        mid = np.array(s["mid"])
        ofi = np.array(s["ofi"])

        # 過去 window 秒の OFI 合計 / horizon 秒先の mid リターン
        sig = np.full(n, np.nan)
        fwd = np.full(n, np.nan)
        csum = np.cumsum(ofi)
        j = 0
        k = 0
        for i in range(n):
            while j < n and ts[j] < ts[i] - args.window:
                j += 1
            if j < i:
                sig[i] = csum[i] - csum[j]
            while k < n and ts[k] < ts[i] + args.horizon:
                k += 1
            if k < n:
                fwd[i] = (mid[k] / mid[i] - 1.0) * 1e4
        # 板の更新ごとに評価するので、ホライズン(秒)の間に入るイベント数だけ重なる
        duration = float(ts[-1] - ts[0]) if n > 1 else 1.0
        overlap = max(1.0, n / max(duration, 1e-9) * args.horizon)
        m = evaluate_signal(sig, fwd, rng, overlap=overlap)
        vol_bps = float(np.nanstd(fwd))
        rows.append(make_row(code, name, float(mid[-1]), m, vol_bps, args.fee_bps,
                             wire.TABLE_TOPIX100 if args.tick_table == "topix100"
                             else (None if args.tick_table == "auto" else wire.TABLE_STANDARD)))
    return rows


# =====================================================================
#  出力
# =====================================================================
SORT_KEYS = {
    # 既定。銘柄の「構造的な勝ちやすさ」。モデルを変えても動かない量なので、
    # 銘柄選択はここで決まる。
    "vol_sp": lambda r: -r["vol_sp"] if np.isfinite(r["vol_sp"]) else 1e9,
    # 損益分岐をどれだけ上回ったか。誤差が大きいので単体で信用しないこと。
    "margin": lambda r: -r["margin_pt"] if np.isfinite(r["margin_pt"]) else 1e9,
    "edge": lambda r: -r["edge_bps"] if np.isfinite(r["edge_bps"]) else 1e9,
    "hit": lambda r: -r["hit"] if np.isfinite(r["hit"]) else 1e9,
}


def report(rows: list, out_csv: str, source: str, sort_key: str, fee_bps: float):
    if not rows:
        print("\n結果なし。")
        return
    rows = sorted(rows, key=SORT_KEYS[sort_key])

    print(f"\n=== ランキング ({source}) ===")
    print(f"並び順: {sort_key}  /  手数料 {fee_bps:.2f} bps(片道)")
    print()
    print("必要hit = 損益分岐の的中率。(2*hit-1)*vol_bps = spread_bps + 2*fee_bps を解いた値。")
    print("         これを超えないとコスト負けする。決めているのは vol/sp(値幅とコストの比)で、")
    print("         シグナルの賢さではない。ここが低い銘柄はモデルを変えても勝てない。")
    print("誤差    = 実測hitの標準誤差。ラベルが重なるぶんを割り引いた独立サンプル数から算出。")
    print("余裕    = 実測hit - 必要hit。誤差の2倍を超えて初めて『有望』と表示する。")
    print()

    hdr = (f"{'code':<6} {'銘柄':<14} {'株価':>9} {'呼値':>6} {'vol/sp':>7} "
           f"{'必要hit':>8} {'(手数料0)':>9} {'実測hit':>8} {'誤差':>7} {'余裕':>8} "
           f"{'n_eff':>6}  判定")
    print(hdr)
    print("-" * (len(hdr) + 4))
    for r in rows:
        name = r["name"][:12]
        be = r["breakeven_hit"]
        print(f"{r['code']:<6} {name:<14} {r['price']:>9.1f} {r['tick_yen']:>6.2f} "
              f"{r['vol_sp']:>7.1f} "
              f"{be:>7.1f}% {r['breakeven_hit_nofee']:>8.1f}% "
              f"{r['hit'] * 100:>7.1f}% {r['hit_se_pt']:>6.1f}pt {r['margin_pt']:>+7.1f}pt "
              f"{r['n_eff']:>6}  {r['verdict']}")

    promising = [r for r in rows if r["verdict"] == "有望"]
    noise = [r for r in rows if r["verdict"] == "ノイズ内"]
    impossible = [r for r in rows if r["verdict"] == "対象外"]
    med_se = float(np.nanmedian([r["hit_se_pt"] for r in rows]))
    med_neff = int(np.nanmedian([r["n_eff"] for r in rows]))

    print()
    print(f"有望(余裕 > 誤差x2) : {len(promising)} / {len(rows)}")
    print(f"ノイズ内            : {len(noise)} / {len(rows)}")
    print(f"対象外(必要hit>=65%): {len(impossible)} / {len(rows)}  <- 呼値が株価に対して大きすぎる")
    print(f"独立サンプル 中央値 : {med_neff} 件 -> 的中率の誤差 ±{med_se:.1f}pt")

    if med_se >= 3.0:
        # 誤差が大きいときは、順位そのものが偶然で入れ替わる。
        # 情報ゼロを仮定したシミュレーションで「最高値がどこまで出るか」を示す。
        rng = np.random.default_rng(0)
        draws = rng.binomial(med_neff, 0.5, size=(20000, len(rows))) / med_neff * 100
        mx = float(np.median(draws.max(axis=1)))
        rng_pt = float(np.median(draws.max(axis=1) - draws.min(axis=1)))
        obs_hi = float(np.nanmax([r["hit"] * 100 for r in rows]))
        obs_rng = obs_hi - float(np.nanmin([r["hit"] * 100 for r in rows]))
        print()
        print("[警告] サンプルが少なく、銘柄間の的中率の差は偶然で説明できます。")
        print(f"       全銘柄の真の的中率が50%(=情報ゼロ)だと仮定して{len(rows)}銘柄スキャンすると:")
        print(f"         最高の銘柄の的中率  シミュレーション {mx:.1f}%  /  実測 {obs_hi:.1f}%")
        print(f"         最高-最低の幅       シミュレーション {rng_pt:.1f}pt /  実測 {obs_rng:.1f}pt")
        print("       実測がシミュレーションと同程度なら、順位に意味はありません。")
        print("       --days を増やすか銘柄数を増やして独立サンプルを稼いでください。")

    if not promising:
        print()
        print("統計的に損益分岐を超えている銘柄はありません。これは普通のことです。")
        print("しかもここでのスプレッドは呼値=理論下限なので、実際はもっと負けます。")
    print()
    print(f"手数料を0にすると必要hitが平均 "
          f"{float(np.nanmean([r['breakeven_hit'] - r['breakeven_hit_nofee'] for r in rows])):.1f}pt "
          f"下がります(現物ゼロ革命の証券会社)。モデル改善より効くことがあります。")

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\ncsv: {out_csv}")


# =====================================================================
#  自己検証
# =====================================================================
def selftest() -> int:
    fails = 0

    def check(cond, label):
        nonlocal fails
        if not cond:
            fails += 1
            print(f"  FAIL {label}")

    rng = np.random.default_rng(7)

    # 順位相関: 完全一致 / 完全逆 / 無相関
    x = np.arange(200.0)
    check(abs(spearman(x, x) - 1.0) < 1e-9, "spearman(x, x) == 1")
    check(abs(spearman(x, -x) + 1.0) < 1e-9, "spearman(x, -x) == -1")
    check(abs(spearman(x, rng.permutation(x))) < 0.2, "無相関はおおよそ0")

    # 情報の無いシグナル -> ic も ic_net もほぼ0
    n = 4000
    noise = rng.normal(size=n)
    fwd = rng.normal(size=n)
    m = evaluate_signal(noise, fwd, rng)
    check(abs(m["ic"]) < 0.06, f"情報なしの ic はほぼ0 (実際 {m['ic']:+.3f})")
    check(abs(m["hit"] - 0.5) < 0.06, f"情報なしの hit はほぼ50% (実際 {m['hit']:.1%})")

    # 情報のあるシグナル -> ic 正、hit > 50%、シャッフルは0付近
    sig = rng.normal(size=n)
    fwd2 = 0.5 * sig + rng.normal(size=n)
    m2 = evaluate_signal(sig, fwd2, rng)
    check(m2["ic"] > 0.3, f"情報ありの ic が正 (実際 {m2['ic']:+.3f})")
    check(m2["hit"] > 0.6, f"情報ありの hit > 60% (実際 {m2['hit']:.1%})")
    check(abs(m2["ic_shuf"]) < 0.06, f"シャッフルは0付近 (実際 {m2['ic_shuf']:+.3f})")
    check(m2["ic"] - m2["ic_shuf"] > 0.3, "ic_net が情報量を捉えている")

    # 呼値ベースのコスト
    sp, tick = cost_bps(2000.0, "7203", 0.0)   # TOPIX100 -> 2000円は0.5円刻み
    check(abs(tick - 0.5) < 1e-9, f"TOPIX100 2000円の呼値=0.5円 (実際 {tick})")
    check(abs(sp - 0.5 / 2000 * 1e4) < 1e-6, "spread_bps = 呼値/価格*1e4")
    sp2, tick2 = cost_bps(2000.0, "9999", 0.0)  # 非TOPIX100 -> 1円刻み
    check(abs(tick2 - 1.0) < 1e-9, f"通常銘柄 2000円の呼値=1円 (実際 {tick2})")

    # make_row: 的中率50%なら edge はコストぶんだけマイナス
    base = {"n": 100, "n_eff": 100, "hit_se": 5.0, "ic": 0.0, "ic_shuf": 0.0,
            "hit": 0.5, "hit_shuf": 0.5}
    row = make_row("7203", "トヨタ", 2000.0, base, vol_bps=10.0, fee_bps=1.0)
    check(abs(row["edge_bps"] - (0.0 - sp - 2.0)) < 1e-6,
          f"hit=50% の edge = -(spread + 手数料x2) (実際 {row['edge_bps']:+.3f})")

    # 損益分岐: 定義どおりか、edge=0 の点と一致するか
    check(abs(breakeven_hit(4.0, 100.0, 1.0) - (50 + 6.0 / 200 * 100)) < 1e-9,
          "breakeven_hit = 50 + (spread + 2*fee)/(2*vol)*100")
    check(breakeven_hit(4.0, 100.0, 0.0) < breakeven_hit(4.0, 100.0, 1.0),
          "手数料を下げると必要的中率も下がる")
    check(breakeven_hit(4.0, 200.0, 1.0) < breakeven_hit(4.0, 100.0, 1.0),
          "ボラが大きいほど必要的中率は下がる")
    check(breakeven_hit(8.0, 100.0, 1.0) > breakeven_hit(4.0, 100.0, 1.0),
          "スプレッドが大きいほど必要的中率は上がる")
    # 損益分岐ちょうどの的中率を入れたら「余裕0」かつ「edge 0」になること。
    # make_row は株価から呼値を引き直すので、期待値もその呼値から計算すること。
    sp_test, _ = cost_bps(2000.0, "9999", 1.0, wire.TABLE_STANDARD)
    be = breakeven_hit(sp_test, 50.0, 1.0)
    r2 = make_row("9999", "テスト", 2000.0, dict(base, hit=be / 100), vol_bps=50.0,
                  fee_bps=1.0, tick_table=wire.TABLE_STANDARD)
    check(abs(r2["margin_pt"]) < 1e-9,
          f"損益分岐ちょうどなら余裕0 (実際 {r2['margin_pt']:+.6f}pt)")
    check(abs(r2["edge_bps"]) < 1e-9,
          f"損益分岐ちょうどなら edge も0 (実際 {r2['edge_bps']:+.6f}bps)")

    # 判定ロジック: 余裕が誤差の2倍を超えたときだけ「有望」
    r_noise = make_row("9999", "T", 2000.0,
                       dict(base, hit=0.56, hit_se=5.0), vol_bps=100.0, fee_bps=0.0,
                       tick_table=wire.TABLE_STANDARD)
    check(r_noise["verdict"] == "ノイズ内",
          f"余裕 < 誤差x2 は『ノイズ内』(実際 {r_noise['verdict']}, 余裕 {r_noise['margin_pt']:+.1f}pt)")
    r_good = make_row("9999", "T", 2000.0,
                      dict(base, hit=0.70, hit_se=1.0), vol_bps=100.0, fee_bps=0.0,
                      tick_table=wire.TABLE_STANDARD)
    check(r_good["verdict"] == "有望",
          f"余裕 > 誤差x2 は『有望』(実際 {r_good['verdict']})")
    r_lose = make_row("9999", "T", 2000.0,
                      dict(base, hit=0.48, hit_se=1.0), vol_bps=100.0, fee_bps=0.0,
                      tick_table=wire.TABLE_STANDARD)
    check(r_lose["verdict"] == "コスト負け", f"損益分岐未満 (実際 {r_lose['verdict']})")
    # 呼値が大きすぎて原理的に届かない銘柄
    r_imp = make_row("9999", "T", 2000.0, dict(base, hit=0.99, hit_se=0.1),
                     vol_bps=2.0, fee_bps=0.0, tick_table=wire.TABLE_STANDARD)
    check(r_imp["verdict"] == "対象外",
          f"必要hit>=65% は『対象外』(必要 {r_imp['breakeven_hit']:.1f}%, 実際 {r_imp['verdict']})")

    # 重なりを考慮した独立サンプル数と誤差
    n = 4000
    m3 = evaluate_signal(rng.normal(size=n), rng.normal(size=n), rng, overlap=5.0)
    check(m3["n_eff"] == int((n // 4) / 5), f"独立サンプル = 発火数/重なり (実際 {m3['n_eff']})")
    m4 = evaluate_signal(rng.normal(size=n), rng.normal(size=n), rng, overlap=1.0)
    check(m4["hit_se"] < m3["hit_se"], "重なりが大きいほど誤差も大きくなる")

    if wire.selftest():
        print("  FAIL wire.py の自己検証が失敗")
        fails += 1

    print(f"selftest: {'OK - 全項目一致' if fails == 0 else f'NG - {fails} 件失敗'}")
    return fails


# =====================================================================
def main() -> int:
    p = argparse.ArgumentParser(
        description="どの銘柄に繋ぐべきかを予測しやすさ x 儲かるかでランク付けする",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=("yfinance", "kabu"), default="yfinance",
                   help="yfinance=無料1分足(板なし) / kabu=実板(要口座)")
    p.add_argument("--selftest", action="store_true", help="ネット不要の指標計算の検証")
    p.add_argument("--tickers", default=None,
                   help="銘柄コードのカンマ区切り(既定: 主要15銘柄)")
    p.add_argument("--days", type=int, default=5,
                   help="yfinance: 取得日数(1分足は7日程度まで)")
    p.add_argument("--window", type=float, default=10,
                   help="シグナルの集計窓。yfinance=足数 / kabu=秒")
    p.add_argument("--horizon", type=float, default=5,
                   help="予測ホライズン。yfinance=足数 / kabu=秒")
    p.add_argument("--minutes", type=float, default=10.0, help="kabu: 観測時間(分)")
    p.add_argument("--fee-bps", type=float, default=1.0, help="片道手数料(bps)")
    p.add_argument("--tick-table", choices=("auto", "standard", "topix100"), default="auto",
                   help="呼値テーブル。auto=主要銘柄はTOPIX100扱い")
    p.add_argument("--password", default=None, help="kabuステーションのAPIパスワード")
    p.add_argument("--test-env", action="store_true", help="kabuの検証環境を使う")
    p.add_argument("--sort", choices=tuple(SORT_KEYS), default="vol_sp",
                   help="並び順。vol_sp=値幅とコストの比(既定・構造的な勝ちやすさ) / "
                        "margin=損益分岐からの余裕 / edge=期待エッジ / hit=実測的中率")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "scan_stocks.csv"))
    p.add_argument("--seed", type=int, default=42, help="シャッフル検定の乱数シード")
    args = p.parse_args()

    if args.selftest:
        return 1 if selftest() else 0

    if args.source == "kabu":
        args.window = args.window if args.window > 1 else 10
        rows = scan_kabu(args)
        report(rows, args.out, "kabuステーション実板 / best-level OFI", args.sort, args.fee_bps)
    else:
        args.window = int(args.window)
        args.horizon = int(args.horizon)
        rows = scan_yfinance(args)
        report(rows, args.out,
               f"yfinance 1分足 / 符号付き出来高(板なしの代用), 窓{args.window}本 -> {args.horizon}本先",
               args.sort, args.fee_bps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
