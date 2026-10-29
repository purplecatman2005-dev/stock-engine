// backtest.cpp — OFI(オーダーフロー不均衡)シグナルのイベント駆動バックテスト
//
// 使い方:
//   backtest.exe [1日あたりイベント数=200000] [日数=20] [seed=42] [基準値段(円)=2500] [片道手数料bps=1.0]
//   backtest.exe --no-null ...    帰無仮説の測定を省く(速いが結果の解釈ができない)
//
// ロジック:
//   各イベントで best 差分から OFI 寄与を出し、直近 W=64 イベントの窓合計を z-score 化。
//   |z| > 2.5 でシグナル発火(発火後 H イベントはクールダウン)、H イベント先の
//   mid 変動で採点する。発火はザラ場(連続約定)の間だけ。板寄せ中の板はクロスして
//   いるのが正常で、その間の mid は価格として意味を持たないため除外する。
//
// ★ このバックテストが必ず2回走る理由 — 帰無仮説
//
//   合成フィードには「板がフェアバリューに追いつくまでの遅れ」が必ず残る。OFI はこの
//   遅れを読めてしまうので、オーダーフローに情報を一切入れていないフィードでも
//   的中率が60%を超える。開発中に実測で、情報ゼロのフィードで88%が出たことがある
//   (無関係な位置の z を使うシャッフル検定は49.3%だったので、計測ミスではなく
//    フィードの構造そのものが原因)。
//
//   だから「的中率○○%」という数字は単体では何も意味しない。同じ測定を
//   「情報ゼロのフィード(drift=0, coupling=0)」でも走らせ、その差だけが
//   シグナルが本当に読み取った情報量である。実際の市場データを使うときも同じで、
//   必ず何らかの帰無(ランダム方向、ラベルシャッフル)と比べること。
//
// 株ならではの採点:
//   結果を「ティック」ではなく「bps(万分率)」と「円」で出す。呼値が価格帯で変わるので
//   ティックのままでは銘柄間・価格帯間で比較できないため。
//   往復コスト = スプレッド + 手数料x2。主要ネット証券の現物手数料はゼロ革命で
//   0円が現実的だが、信用の金利・貸株料などが残るので既定 1.0 bps を置く。
//
// 出力: backtest_signals.csv (情報ありの側のみ)
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "book.hpp"
#include "feed.hpp"
#include "jp_market.hpp"
#include "signals.hpp"

using namespace stk;

namespace {

constexpr std::size_t W = 64;      // OFI集計窓(イベント数)
constexpr double Z_TH = 2.5;       // 発火閾値
constexpr std::size_t H = 500;     // 評価ホライズン(イベント数)
constexpr std::size_t WARMUP = 5'000;

struct Result {
    std::size_t cont_events = 0;
    std::size_t fired = 0, scored = 0, hits = 0, zeros = 0, skipped = 0;
    double captured_bps = 0;   // 平均
    double net_bps = 0;        // 平均
    double avg_spread_yen = 0;
    double avg_spread_bps = 0;
    double hit_rate = 0;       // mid不変を除いた方向的中率(%)
};

Result measure(const StockFeed::Config& cfg, double fee_bps, const char* csv_path) {
    StockFeed feed(cfg);
    std::vector<u8> buf;
    buf.reserve(cfg.events_per_day * std::size_t(cfg.days) * 12);
    feed.generate(buf);

    jpx::TickGrid grid(cfg.table);
    jpx::PriceWindow win{};
    StockBook book;
    RollingZ roll(W);
    BestState prev{};
    jpx::Session sess = jpx::Session::Off;

    struct Signal { std::size_t idx; double z; int dir; double mid_yen; double spread_yen; };
    std::vector<Signal> signals;

    // 採点用の時系列はザラ場イベントだけで作る
    std::vector<double> mid_yens;
    std::vector<u32> seg_of;  // 属する連続約定セグメント番号(昼休みや翌日で切れる)
    mid_yens.reserve(cfg.events_per_day * std::size_t(cfg.days));
    seg_of.reserve(cfg.events_per_day * std::size_t(cfg.days));

    std::size_t last_fire = 0, k = 0;
    u32 seg = 0;
    bool was_continuous = false;
    double spread_yen_sum = 0, mid_yen_sum = 0;
    std::size_t spread_cnt = 0;

    for (std::size_t off = 0; off < buf.size();) {
        NormMsg m;
        off += decode(buf.data() + off, m);
        switch (m.type) {
        case 'R':
            win = jpx::make_window(grid, m.base_sen);
            book.reset(m.idx);
            roll.clear();
            prev = BestState{};
            continue;
        case 'S': sess = static_cast<jpx::Session>(m.side); continue;
        case 'T': continue;
        case 'A': book.add(m.id, static_cast<Side>(m.side), m.idx, m.qty); break;
        case 'X': case 'E': book.reduce(m.id, m.qty); break;
        case 'D': book.remove(m.id); break;
        default: continue;
        }

        const BestState cur = snap_best(book);
        const double z = roll.push(ofi_contribution(prev, cur));
        prev = cur;

        const bool cont = jpx::is_continuous(sess);
        if (!cont) { was_continuous = false; continue; }
        if (!was_continuous) { ++seg; was_continuous = true; }
        if (!cur.valid) continue;

        // 呼値が価格帯で変わるので、円換算は必ず price_of() を通す
        const double bid_yen = jpx::sen_to_yen(jpx::idx_to_sen(grid, win, cur.pb));
        const double ask_yen = jpx::sen_to_yen(jpx::idx_to_sen(grid, win, cur.pa));
        const double mid_yen = 0.5 * (bid_yen + ask_yen);
        const double spread_yen = ask_yen - bid_yen;
        spread_yen_sum += spread_yen;
        mid_yen_sum += mid_yen;
        ++spread_cnt;
        mid_yens.push_back(mid_yen);
        seg_of.push_back(seg);

        if (k >= WARMUP && roll.warmed() && (z > Z_TH || z < -Z_TH) && k - last_fire >= H) {
            signals.push_back({k, z, z > 0 ? +1 : -1, mid_yen, spread_yen});
            last_fire = k;
        }
        ++k;
    }

    // ---- 前方 mid 変動で採点 ----
    std::FILE* f = csv_path ? std::fopen(csv_path, "w") : nullptr;
    if (f) std::fprintf(f, "event_idx,z,dir,mid_yen,fwd_mid_yen,dmid_yen,captured_yen,captured_bps,net_bps\n");

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
        const double net_bps = cap_bps - cost_bps;
        if (f)
            std::fprintf(f, "%zu,%.3f,%d,%.2f,%.2f,%+.2f,%+.2f,%+.3f,%+.3f\n",
                         s.idx, s.z, s.dir, s.mid_yen, fwd, dmid, cap, cap_bps, net_bps);
        ++r.scored;
        if (dmid == 0) ++r.zeros;
        else if (cap > 0) ++r.hits;
        cap_sum += cap_bps;
        net_sum += net_bps;
    }
    if (f) std::fclose(f);

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
    std::printf("  平均ネット     : %+.3f bps/シグナル (獲得 - スプレッド - 手数料x2)\n", r.net_bps);
}

}  // namespace

int main(int argc, char** argv) {
    bool with_null = true;
    std::vector<char*> pos;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-null") == 0) with_null = false;
        else pos.push_back(argv[i]);
    }

    StockFeed::Config cfg;
    cfg.events_per_day = 200'000;
    cfg.days = 20;
    cfg.seed = 42;
    cfg.base_yen = 2500.0;
    double fee_bps = 1.0;
    if (pos.size() > 0) cfg.events_per_day = std::stoull(pos[0]);
    if (pos.size() > 1) cfg.days = std::stoi(pos[1]);
    if (pos.size() > 2) cfg.seed = std::stoull(pos[2]);
    if (pos.size() > 3) cfg.base_yen = std::stod(pos[3]);
    if (pos.size() > 4) fee_bps = std::stod(pos[4]);

    std::printf("=== OFI signal backtest ===\n");
    std::printf("イベント %zu/日 x %d日, seed=%llu, 基準値段 %.0f円, 手数料 %.2f bps(片道)\n",
                cfg.events_per_day, cfg.days, (unsigned long long)cfg.seed, cfg.base_yen, fee_bps);
    std::printf("シグナル: |OFI z-score(窓%zu)| > %.1f, ホライズン %zu イベント\n\n",
                W, Z_TH, H);

    const Result real = measure(cfg, fee_bps, "backtest_signals.csv");
    print_block("[本番] オーダーフローに情報あり (drift/coupling = 既定値)", real);

    if (!with_null) {
        std::printf("\n--no-null を付けたので帰無仮説を測っていません。\n");
        std::printf("この的中率の大半はフィードの構造由来かもしれず、単体では解釈できません。\n");
        return 0;
    }

    StockFeed::Config null_cfg = cfg;
    null_cfg.drift_strength = 0.0;   // フェアバリューはただのランダムウォーク
    null_cfg.flow_coupling = 0.0;    // 発注サイドは完全なコイン投げ
    std::printf("\n");
    const Result null = measure(null_cfg, fee_bps, nullptr);
    print_block("[帰無] オーダーフローに情報なし (drift=0, coupling=0)", null);

    std::printf("\n--- 判定 ---\n");
    const double edge_pt = real.hit_rate - null.hit_rate;
    const double edge_bps = real.captured_bps - null.captured_bps;
    std::printf("的中率の差 : %+.1f pt  (本番 %.1f%% - 帰無 %.1f%%)\n",
                edge_pt, real.hit_rate, null.hit_rate);
    std::printf("獲得の差   : %+.3f bps (本番 %+.3f - 帰無 %+.3f)\n",
                edge_bps, real.captured_bps, null.captured_bps);
    std::printf("\n");
    if (edge_pt > 1.0)
        std::printf("シグナルはオーダーフローの情報を読めている(差が帰無より明確に大きい)。\n");
    else
        std::printf("シグナルはフィードの構造を読んでいるだけ。情報は取れていない。\n");
    std::printf("なお本番の平均ネットが %+.3f bps でも、その大半は帰無側にも出る見かけの\n",
                real.net_bps);
    std::printf("エッジ(帰無の平均ネット %+.3f bps)なので、実運用の期待値ではない。\n", null.net_bps);
    std::printf("\ncsv: backtest_signals.csv (本番側のシグナル明細)\n");
    std::printf("\n注意: これは合成フィードです。「エンジンとシグナル計算が正しく動く」ことの\n");
    std::printf("      確認であって、実市場で儲かることの証明ではありません。\n");
    return 0;
}
