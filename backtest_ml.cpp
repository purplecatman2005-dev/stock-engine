// backtest_ml.cpp — GBDTモデルによるシグナルのバックテスト + Python予測との一致検証
//
// 使い方:
//   backtest_ml.exe [1日あたりイベント数=200000] [日数=20] [seed=777] [基準値段=2500] [手数料bps=1.0]
//   backtest_ml.exe --verify verify.csv    Python側の予測値と完全一致するか検証
//   backtest_ml.exe --no-null ...          帰無仮説の測定を省く
//
// backtest.cpp(OFI単体)と同じ枠組みで、シグナルの判定だけを GBDT に差し替えたもの。
// 帰無仮説(オーダーフローに情報の無いフィード)も必ず同時に測る。理由は backtest.cpp
// の冒頭コメントを参照 -- 合成フィードには構造由来の見かけの的中率が必ず乗るため。
//
// --verify は「学習(Python)と推論(C++)が同じ数値を出すこと」の機械的な保証。
// train_export.py が吐いた verify.csv の各行で model.hpp の predict() を呼び、
// Python の予測値との差が許容誤差内かを確認する。ここがズレていると
// 「Pythonでは効いていたのに実運用で効かない」を見逃す。
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "book.hpp"
#include "features.hpp"
#include "feed.hpp"
#include "jp_market.hpp"
#include "model.hpp"

using namespace stk;

namespace {

constexpr std::size_t H = 500;  // 評価ホライズン(ザラ場イベント数)
constexpr std::size_t WARMUP = 5'000;

struct Result {
    std::size_t cont_events = 0;
    std::size_t fired = 0, scored = 0, hits = 0, zeros = 0, skipped = 0;
    double captured_bps = 0, net_bps = 0;
    double avg_spread_yen = 0, avg_spread_bps = 0;
    double hit_rate = 0;
};

Result measure(const StockFeed::Config& cfg, double fee_bps, const char* csv_path) {
    StockFeed feed(cfg);
    std::vector<u8> buf;
    buf.reserve(cfg.events_per_day * std::size_t(cfg.days) * 12);
    feed.generate(buf);

    jpx::TickGrid grid(cfg.table);
    jpx::PriceWindow win{};
    StockBook book;
    FeatureEngine fe;
    BestState prev{};
    jpx::Session sess = jpx::Session::Off;
    i64 ms_of_day = 0;

    struct Signal { std::size_t idx; double pred; int dir; double mid_yen; double spread_yen; };
    std::vector<Signal> signals;
    std::vector<double> mid_yens;
    std::vector<u32> seg_of;
    mid_yens.reserve(cfg.events_per_day * std::size_t(cfg.days));
    seg_of.reserve(cfg.events_per_day * std::size_t(cfg.days));

    std::size_t last_fire = 0, k = 0;
    u32 seg = 0;
    bool was_cont = false;
    double spread_yen_sum = 0, mid_yen_sum = 0;
    std::size_t spread_cnt = 0;
    double f[ML_NFEAT];

    for (std::size_t off = 0; off < buf.size();) {
        NormMsg m;
        off += decode(buf.data() + off, m);
        switch (m.type) {
        case 'R':
            win = jpx::make_window(grid, m.base_sen);
            book.reset(m.idx);
            fe.reset();
            prev = BestState{};
            continue;
        case 'S':
            sess = static_cast<jpx::Session>(m.side);
            ms_of_day = i64(m.idx) * 1000;
            continue;
        case 'T': continue;
        case 'A': book.add(m.id, static_cast<Side>(m.side), m.idx, m.qty); break;
        case 'X': case 'E': book.reduce(m.id, m.qty); break;
        case 'D': book.remove(m.id); break;
        default: continue;
        }

        const BestState cur = snap_best(book);
        const bool ok = fe.update(prev, cur, sess, ms_of_day, f);
        prev = cur;

        const bool cont = jpx::is_continuous(sess);
        if (!cont) { was_cont = false; continue; }
        if (!was_cont) { ++seg; was_cont = true; }
        if (!cur.valid) continue;

        const double bid_yen = jpx::sen_to_yen(jpx::idx_to_sen(grid, win, cur.pb));
        const double ask_yen = jpx::sen_to_yen(jpx::idx_to_sen(grid, win, cur.pa));
        const double mid_yen = 0.5 * (bid_yen + ask_yen);
        const double spread_yen = ask_yen - bid_yen;
        spread_yen_sum += spread_yen;
        mid_yen_sum += mid_yen;
        ++spread_cnt;
        mid_yens.push_back(mid_yen);
        seg_of.push_back(seg);

        if (ok && k >= WARMUP && k - last_fire >= H) {
            const double pred = stkmodel::predict(f);
            if (std::abs(pred) > stkmodel::THRESHOLD) {
                signals.push_back({k, pred, pred > 0 ? +1 : -1, mid_yen, spread_yen});
                last_fire = k;
            }
        }
        ++k;
    }

    std::FILE* fp = csv_path ? std::fopen(csv_path, "w") : nullptr;
    if (fp) std::fprintf(fp, "event_idx,pred,dir,mid_yen,fwd_mid_yen,dmid_yen,captured_bps,net_bps\n");

    Result r;
    r.cont_events = mid_yens.size();
    r.fired = signals.size();
    double cap_sum = 0, net_sum = 0;
    for (const Signal& s : signals) {
        if (s.idx + H >= mid_yens.size()) break;
        if (seg_of[s.idx + H] != seg_of[s.idx]) { ++r.skipped; continue; }
        const double fwd = mid_yens[s.idx + H];
        const double dmid = fwd - s.mid_yen;
        const double cap = dmid * s.dir;
        const double cap_bps = s.mid_yen > 0 ? cap / s.mid_yen * 1e4 : 0.0;
        const double cost_bps = (s.mid_yen > 0 ? s.spread_yen / s.mid_yen * 1e4 : 0.0) + 2 * fee_bps;
        if (fp)
            std::fprintf(fp, "%zu,%.6f,%d,%.2f,%.2f,%+.2f,%+.3f,%+.3f\n",
                         s.idx, s.pred, s.dir, s.mid_yen, fwd, dmid, cap_bps, cap_bps - cost_bps);
        ++r.scored;
        if (dmid == 0) ++r.zeros;
        else if (cap > 0) ++r.hits;
        cap_sum += cap_bps;
        net_sum += cap_bps - cost_bps;
    }
    if (fp) std::fclose(fp);

    if (r.scored) {
        r.captured_bps = cap_sum / double(r.scored);
        r.net_bps = net_sum / double(r.scored);
        const std::size_t decisive = r.scored - r.zeros;
        r.hit_rate = decisive ? 100.0 * double(r.hits) / double(decisive) : 0.0;
    }
    if (spread_cnt) {
        r.avg_spread_yen = spread_yen_sum / double(spread_cnt);
        const double avg_mid = mid_yen_sum / double(spread_cnt);
        r.avg_spread_bps = avg_mid > 0 ? r.avg_spread_yen / avg_mid * 1e4 : 0;
    }
    return r;
}

void print_block(const char* title, const Result& r) {
    std::printf("%s\n", title);
    std::printf("  ザラ場イベント : %zu\n", r.cont_events);
    std::printf("  シグナル       : %zu 発火 / %zu 採点 (%zu 件は休憩を跨ぐため除外)\n",
                r.fired, r.scored, r.skipped);
    if (!r.scored) { std::printf("  採点できるシグナルなし\n"); return; }
    std::printf("  方向的中率     : %.1f%% (%zu/%zu, mid不変 %zu 件を除く)\n",
                r.hit_rate, r.hits, r.scored - r.zeros, r.zeros);
    std::printf("  平均獲得       : %+.3f bps/シグナル\n", r.captured_bps);
    std::printf("  平均スプレッド : %.3f 円 = %.3f bps\n", r.avg_spread_yen, r.avg_spread_bps);
    std::printf("  平均ネット     : %+.3f bps/シグナル\n", r.net_bps);
}

// train_export.py が吐いた verify.csv と C++ の predict() を突き合わせる
int verify(const char* path) {
    std::FILE* fp = std::fopen(path, "r");
    if (!fp) { std::printf("開けません: %s\n", path); return 1; }
    char line[8192];
    if (!std::fgets(line, sizeof line, fp)) { std::fclose(fp); return 1; }  // ヘッダ

    std::size_t n = 0, bad = 0;
    double max_abs_diff = 0, max_rel_diff = 0;
    while (std::fgets(line, sizeof line, fp)) {
        double v[ML_NFEAT + 1];
        int got = 0;
        char* p = line;
        while (got < ML_NFEAT + 1 && *p) {
            char* end = nullptr;
            v[got] = std::strtod(p, &end);
            if (end == p) break;
            ++got;
            p = end;
            while (*p == ',' || *p == ' ') ++p;
        }
        if (got != ML_NFEAT + 1) continue;
        const double want = v[ML_NFEAT];
        const double have = stkmodel::predict(v);
        const double ad = std::abs(have - want);
        const double rd = std::abs(want) > 1e-12 ? ad / std::abs(want) : ad;
        if (ad > max_abs_diff) max_abs_diff = ad;
        if (rd > max_rel_diff) max_rel_diff = rd;
        // 木の重みを double で合計しているので、順序差による丸め誤差だけを許容する
        if (ad > 1e-9 && rd > 1e-9) ++bad;
        ++n;
    }
    std::fclose(fp);

    std::printf("=== verify: %s ===\n", path);
    std::printf("特徴量数     : %d (model.hpp: %d)\n", ML_NFEAT, stkmodel::N_FEATURES);
    if (ML_NFEAT != stkmodel::N_FEATURES) {
        std::printf("NG — features.hpp と model.hpp の特徴量数が違う。\n");
        std::printf("     dump_features.exe をやり直して再学習してください。\n");
        return 1;
    }
    std::printf("照合行数     : %zu\n", n);
    std::printf("最大絶対差   : %.3g\n", max_abs_diff);
    std::printf("最大相対差   : %.3g\n", max_rel_diff);
    if (!n) { std::printf("NG — 1行も照合できませんでした\n"); return 1; }
    std::printf("%s\n", bad == 0 ? "OK — Python の予測と C++ の predict() が一致"
                                 : "NG — 一致しない行があります");
    if (bad) std::printf("不一致 %zu 行\n", bad);
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool with_null = true;
    std::vector<char*> pos;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--verify") == 0 && i + 1 < argc) return verify(argv[i + 1]);
        if (std::strcmp(argv[i], "--no-null") == 0) with_null = false;
        else pos.push_back(argv[i]);
    }

    StockFeed::Config cfg;
    cfg.events_per_day = 200'000;
    cfg.days = 20;
    cfg.seed = 777;   // 学習(42..)とは別のseed = アウトオブサンプル
    cfg.base_yen = 2500.0;
    double fee_bps = 1.0;
    if (pos.size() > 0) cfg.events_per_day = std::stoull(pos[0]);
    if (pos.size() > 1) cfg.days = std::stoi(pos[1]);
    if (pos.size() > 2) cfg.seed = std::stoull(pos[2]);
    if (pos.size() > 3) cfg.base_yen = std::stod(pos[3]);
    if (pos.size() > 4) fee_bps = std::stod(pos[4]);

    std::printf("=== GBDT signal backtest ===\n");
    std::printf("イベント %zu/日 x %d日, seed=%llu (学習と別seed = アウトオブサンプル)\n",
                cfg.events_per_day, cfg.days, (unsigned long long)cfg.seed);
    std::printf("基準値段 %.0f円, 手数料 %.2f bps(片道), 特徴量 %d種, 発火閾値 %.5f\n\n",
                cfg.base_yen, fee_bps, ML_NFEAT, stkmodel::THRESHOLD);

    const Result real = measure(cfg, fee_bps, "backtest_ml_signals.csv");
    print_block("[本番] オーダーフローに情報あり", real);

    if (!with_null) return 0;

    StockFeed::Config null_cfg = cfg;
    null_cfg.drift_strength = 0.0;
    null_cfg.flow_coupling = 0.0;
    std::printf("\n");
    const Result null = measure(null_cfg, fee_bps, nullptr);
    print_block("[帰無] オーダーフローに情報なし (drift=0, coupling=0)", null);

    std::printf("\n--- 判定 ---\n");
    std::printf("的中率の差 : %+.1f pt  (本番 %.1f%% - 帰無 %.1f%%)\n",
                real.hit_rate - null.hit_rate, real.hit_rate, null.hit_rate);
    std::printf("獲得の差   : %+.3f bps (本番 %+.3f - 帰無 %+.3f)\n",
                real.captured_bps - null.captured_bps, real.captured_bps, null.captured_bps);
    std::printf("\n%s\n", real.hit_rate - null.hit_rate > 1.0
                              ? "モデルはオーダーフローの情報を読めている。"
                              : "モデルはフィードの構造を読んでいるだけ。情報は取れていない。");
    std::printf("csv: backtest_ml_signals.csv\n");
    std::printf("\n注意: これは合成フィードです。実市場のエッジの証明ではありません。\n");
    return 0;
}
