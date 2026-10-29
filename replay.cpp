// replay.cpp — 収録データのリプレイ検証と実時間採点
//
// 使い方:
//   replay.exe captures\capture_YYYYMMDD_HHMMSS.bin [ホライズン秒=30] [片道手数料bps=1.0]
//
// やること:
//   1. 収録フレームを順に流して板を再構築し、std::map 参照実装と全イベントで照合する
//      (アダプタが吐いたデータでエンジンが壊れないかの検証)
//   2. ザラ場中の OFI z-score でシグナルを出し、「N秒後の mid」で採点する
//      イベント数ではなく実時間のホライズンを使う。株はイベント密度が時間帯で
//      10倍以上変わるので(日中U字)、イベント数ホライズンだと寄り付きと昼で
//      まったく違う時間長を測ってしまう
//   3. 収録データの統計(セッション別イベント数、スプレッド、出来高)を出す
//
// 出力: replay_signals.csv
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "book.hpp"
#include "feed.hpp"
#include "jp_market.hpp"
#include "signals.hpp"

using namespace stk;

// 参照実装(bench.cpp と同じ考え方: 明らかに正しいことを優先した std::map 板)
class RefBook {
public:
    void reset() { lv_[0].clear(); lv_[1].clear(); ord_.clear(); }
    void add(u32 id, int side, i32 idx, u32 qty) { ord_[id] = {idx, qty, side}; lv_[side][idx] += qty; }
    void reduce(u32 id, u32 dec) {
        auto it = ord_.find(id);
        if (it == ord_.end() || it->second.qty == 0) return;
        if (dec > it->second.qty) dec = it->second.qty;
        it->second.qty -= dec;
        auto& m = lv_[it->second.side];
        auto lit = m.find(it->second.idx);
        lit->second -= dec;
        if (lit->second == 0) m.erase(lit);
    }
    void remove(u32 id) {
        auto it = ord_.find(id);
        if (it != ord_.end()) reduce(id, it->second.qty);
    }
    bool has(int side) const { return !lv_[side].empty(); }
    i32 best(int side) const { return side == BID ? lv_[0].rbegin()->first : lv_[1].begin()->first; }
    i64 best_qty(int side) const { return side == BID ? lv_[0].rbegin()->second : lv_[1].begin()->second; }

private:
    struct O { i32 idx; u32 qty; int side; };
    std::map<i32, i64> lv_[2];
    std::map<u32, O> ord_;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("使い方: replay.exe <capture.bin> [ホライズン秒=30] [片道手数料bps=1.0]\n");
        return 1;
    }
    const char* path = argv[1];
    double horizon_sec = 30.0;
    double fee_bps = 1.0;
    if (argc > 2) horizon_sec = std::stod(argv[2]);
    if (argc > 3) fee_bps = std::stod(argv[3]);

    std::FILE* fp = std::fopen(path, "rb");
    if (!fp) { std::printf("開けません: %s\n", path); return 1; }
    std::fseek(fp, 0, SEEK_END);
    const long sz = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    std::vector<u8> buf(static_cast<std::size_t>(sz));
    if (std::fread(buf.data(), 1, buf.size(), fp) != buf.size()) {
        std::printf("読み込み失敗\n");
        std::fclose(fp);
        return 1;
    }
    std::fclose(fp);

    constexpr std::size_t W = 64;
    constexpr double Z_TH = 2.5;
    const u64 horizon_ns = static_cast<u64>(horizon_sec * 1e9);

    std::printf("=== replay: %s (%.2f MB) ===\n", path, double(sz) / 1e6);
    std::printf("ホライズン %.1f 秒 (実時間), 手数料 %.2f bps(片道)\n\n", horizon_sec, fee_bps);

    jpx::TickGrid grid(jpx::TickTable::Standard);
    jpx::PriceWindow win{};
    StockBook book;
    RefBook ref;
    RollingZ roll(W);
    BestState prev{};
    jpx::Session sess = jpx::Session::Off;

    struct Sample { u64 ts; double mid_yen; u32 seg; };
    std::vector<Sample> samples;   // ザラ場のみ
    struct Signal { std::size_t si; double z; int dir; double spread_yen; };
    std::vector<Signal> signals;

    std::size_t checked = 0, mismatches = 0, unknown = 0;
    std::size_t n_reset = 0, n_trade = 0;
    i64 traded_shares = 0;
    std::size_t sess_count[10] = {};
    double spread_yen_sum = 0, mid_yen_sum = 0;
    std::size_t spread_cnt = 0;
    u32 seg = 0;
    bool was_cont = false;
    u64 last_fire_ts = 0, first_ts = 0, last_ts = 0;

    for (std::size_t off = 0; off + 9 <= buf.size();) {
        u64 ts;
        std::memcpy(&ts, buf.data() + off, 8);
        const u8 type = buf[off + 8];
        const std::size_t msz = msg_size(type);
        if (msz == 0) { ++unknown; ++off; continue; }
        if (off + 8 + msz > buf.size()) break;
        if (!first_ts) first_ts = ts;
        last_ts = ts;

        NormMsg m;
        decode(buf.data() + off + 8, m);
        off += 8 + msz;

        if (m.type == 'R') {
            grid = jpx::TickGrid(m.table ? jpx::TickTable::Topix100 : jpx::TickTable::Standard);
            win = jpx::make_window(grid, m.base_sen);
            book.reset(win.n_ticks);
            ref.reset();
            roll.clear();
            prev = BestState{};
            ++n_reset;
            std::printf("[reset] 基準値段 %.1f円 -> 窓 [%.1f, %.1f]円 %d本\n",
                        jpx::sen_to_yen(m.base_sen), jpx::sen_to_yen(win.low_sen),
                        jpx::sen_to_yen(win.high_sen), win.n_ticks);
            continue;
        }
        if (m.type == 'S') { sess = static_cast<jpx::Session>(m.side); continue; }
        if (m.type == 'T') { ++n_trade; traded_shares += m.qty; continue; }

        switch (m.type) {
        case 'A':
            if (m.idx < 0 || m.idx >= book.n_ticks()) { ++unknown; continue; }
            book.add(m.id, static_cast<Side>(m.side), m.idx, m.qty);
            ref.add(m.id, m.side, m.idx, m.qty);
            break;
        case 'X': case 'E': book.reduce(m.id, m.qty); ref.reduce(m.id, m.qty); break;
        case 'D': book.remove(m.id); ref.remove(m.id); break;
        default: continue;
        }
        ++checked;
        sess_count[static_cast<int>(sess)]++;

        // 参照実装との照合
        for (int s = 0; s < 2; ++s) {
            const Side side = static_cast<Side>(s);
            if (book.has_side(side) != ref.has(s)) { ++mismatches; break; }
            if (!ref.has(s)) continue;
            if (book.best_idx(side) != ref.best(s) || book.best_qty(side) != ref.best_qty(s)) {
                if (mismatches < 3)
                    std::printf("      MISMATCH ev=%zu side=%d flat=(%d,%lld) ref=(%d,%lld)\n",
                                checked, s, book.best_idx(side), (long long)book.best_qty(side),
                                ref.best(s), (long long)ref.best_qty(s));
                ++mismatches;
                break;
            }
        }

        const BestState cur = snap_best(book);
        const double z = roll.push(ofi_contribution(prev, cur));
        prev = cur;

        const bool cont = jpx::is_continuous(sess);
        if (!cont) { was_cont = false; continue; }
        if (!was_cont) { ++seg; was_cont = true; }
        if (!cur.valid) continue;

        const double bid_yen = jpx::sen_to_yen(jpx::idx_to_sen(grid, win, cur.pb));
        const double ask_yen = jpx::sen_to_yen(jpx::idx_to_sen(grid, win, cur.pa));
        const double mid_yen = 0.5 * (bid_yen + ask_yen);
        spread_yen_sum += ask_yen - bid_yen;
        mid_yen_sum += mid_yen;
        ++spread_cnt;
        samples.push_back({ts, mid_yen, seg});

        if (roll.warmed() && (z > Z_TH || z < -Z_TH) && ts - last_fire_ts >= horizon_ns) {
            signals.push_back({samples.size() - 1, z, z > 0 ? +1 : -1, ask_yen - bid_yen});
            last_fire_ts = ts;
        }
    }

    // ---- 検証結果 ----
    std::printf("\n[検証] 板イベント %zu 件を参照実装と照合\n", checked);
    std::printf("       %s\n", mismatches == 0 ? "OK — 全イベントで best bid/ask が一致"
                                               : "NG — 不一致あり");
    if (mismatches) std::printf("       不一致 %zu 件\n", mismatches);
    if (unknown) std::printf("       スキップした不正フレーム: %zu\n", unknown);

    std::printf("\n[統計]\n");
    if (first_ts && last_ts)
        std::printf("       収録時間     : %.1f 秒\n", double(last_ts - first_ts) / 1e9);
    std::printf("       reset        : %zu\n", n_reset);
    std::printf("       約定          : %zu 件 / %lld 株\n", n_trade, (long long)traded_shares);
    for (int i = 0; i < 10; ++i)
        if (sess_count[i])
            std::printf("       %-17s %zu イベント\n", jpx::SESSION_NAMES[i], sess_count[i]);
    const double avg_spread = spread_cnt ? spread_yen_sum / double(spread_cnt) : 0;
    const double avg_mid = spread_cnt ? mid_yen_sum / double(spread_cnt) : 0;
    if (spread_cnt)
        std::printf("       平均スプレッド: %.3f 円 = %.2f bps (mid %.1f円)\n",
                    avg_spread, avg_mid > 0 ? avg_spread / avg_mid * 1e4 : 0, avg_mid);

    // ---- 実時間ホライズンで採点 ----
    std::FILE* out = std::fopen("replay_signals.csv", "w");
    if (out) std::fprintf(out, "ts_ns,z,dir,mid_yen,fwd_mid_yen,dmid_yen,captured_bps,net_bps\n");
    std::size_t scored = 0, hits = 0, zeros = 0, skipped = 0;
    double cap_sum = 0, net_sum = 0;
    std::size_t search = 0;
    for (const Signal& s : signals) {
        const Sample& a = samples[s.si];
        // a.ts + horizon_ns 以降の最初のサンプルを探す(サンプルは時刻順なので線形前進)
        if (search < s.si) search = s.si;
        while (search < samples.size() && samples[search].ts < a.ts + horizon_ns) ++search;
        if (search >= samples.size()) break;
        if (samples[search].seg != a.seg) { ++skipped; continue; }  // 休憩を跨いだ
        const double fwd = samples[search].mid_yen;
        const double dmid = fwd - a.mid_yen;
        const double cap = dmid * s.dir;
        const double cap_bps = a.mid_yen > 0 ? cap / a.mid_yen * 1e4 : 0;
        const double cost_bps = (a.mid_yen > 0 ? s.spread_yen / a.mid_yen * 1e4 : 0) + 2 * fee_bps;
        if (out)
            std::fprintf(out, "%llu,%.3f,%d,%.2f,%.2f,%+.2f,%+.3f,%+.3f\n",
                         (unsigned long long)a.ts, s.z, s.dir, a.mid_yen, fwd, dmid,
                         cap_bps, cap_bps - cost_bps);
        ++scored;
        if (dmid == 0) ++zeros;
        else if (cap > 0) ++hits;
        cap_sum += cap_bps;
        net_sum += cap_bps - cost_bps;
    }
    if (out) std::fclose(out);

    std::printf("\n[採点] シグナル %zu 発火 / %zu 採点 (%zu 件は休憩を跨ぐため除外)\n",
                signals.size(), scored, skipped);
    if (scored) {
        const std::size_t decisive = scored - zeros;
        std::printf("       方向的中率   : %.1f%% (%zu/%zu, mid不変 %zu 件を除く)\n",
                    decisive ? 100.0 * double(hits) / double(decisive) : 0.0, hits, decisive, zeros);
        std::printf("       平均獲得     : %+.3f bps/シグナル\n", cap_sum / double(scored));
        std::printf("       平均ネット   : %+.3f bps/シグナル (スプレッド・手数料控除後)\n",
                    net_sum / double(scored));
        std::printf("       csv          : replay_signals.csv\n");
        std::printf("\n       この的中率だけでは判断できません。ランダムな方向で同じ採点を\n");
        std::printf("       したときの値と比べてください(backtest.exe が帰無仮説を測る理由)。\n");
    } else {
        std::printf("       採点できるシグナルがありません(収録が短いかホライズンが長すぎます)\n");
    }
    return mismatches ? 1 : 0;
}
