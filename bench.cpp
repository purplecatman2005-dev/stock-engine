// bench.cpp — 正しさの検証 + 性能計測
//
// 使い方:
//   bench.exe [1日あたりイベント数=200000] [日数=5] [seed=42]
//
// 3段階で確かめる:
//   [1] 呼値テーブルの自己検証 — 円 <-> ティック番号 の往復が壊れていないか
//   [2] 板の整合性検証 — フラット配列板と std::map 参照実装が全イベントで一致するか
//   [3] 性能計測 — rdtsc で 1メッセージあたりの処理時間を測る
//
// [2] が「速いだけで間違っている」を防ぐ安全網。速度最適化はここが緑のときだけ意味がある。
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <string>
#include <vector>
#include <x86intrin.h>

#include "book.hpp"
#include "feed.hpp"
#include "jp_market.hpp"
#include "signals.hpp"

using namespace stk;

// ---------------------------------------------------------------------
// 参照実装: std::map の素直な板。速さは度外視して「明らかに正しい」ことを優先する。
// ---------------------------------------------------------------------
class RefBook {
public:
    void reset() { lv_[0].clear(); lv_[1].clear(); ord_.clear(); }

    void add(u32 id, int side, i32 idx, u32 qty) {
        ord_[id] = {idx, qty, side};
        lv_[side][idx] += qty;
    }
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

// ---------------------------------------------------------------------
static int check_tick_grid() {
    std::printf("[1/3] 呼値テーブルの自己検証\n");
    int fails = 0;
    struct Case { const char* label; jpx::TickTable table; double yen; double want_step_yen; };
    const Case cases[] = {
        {"通常 800円",       jpx::TickTable::Standard,     800.0,     1.0},
        {"通常 2999円",      jpx::TickTable::Standard,    2999.0,     1.0},
        {"通常 4000円",      jpx::TickTable::Standard,    4000.0,     5.0},
        {"通常 12000円",     jpx::TickTable::Standard,   12000.0,    10.0},
        {"通常 40000円",     jpx::TickTable::Standard,   40000.0,    50.0},
        {"TOPIX100 800円",   jpx::TickTable::Topix100,     800.0,     0.1},
        {"TOPIX100 2000円",  jpx::TickTable::Topix100,    2000.0,     0.5},
        {"TOPIX100 8000円",  jpx::TickTable::Topix100,    8000.0,     1.0},
        {"TOPIX100 20000円", jpx::TickTable::Topix100,   20000.0,     5.0},
    };
    for (const Case& c : cases) {
        const jpx::TickGrid g(c.table);
        const i64 sen = jpx::yen_to_sen(c.yen);
        const i64 step = g.step_at_price(sen);
        const bool ok_step = step == jpx::yen_to_sen(c.want_step_yen);
        // 往復: 価格 -> ティック -> 価格 は呼値に乗っている価格なら不動点
        const i64 rt = g.price_of(g.tick_of(sen));
        const bool ok_rt = rt == sen;
        // 1ティック上げたら呼値ぶん上がる
        const i64 up = g.price_of(g.tick_of(sen) + 1);
        const bool ok_up = up - sen == step;
        if (!ok_step || !ok_rt || !ok_up) {
            ++fails;
            std::printf("      FAIL %-18s step=%.2f円(期待 %.2f) 往復=%.2f円 +1tick=%.2f円\n",
                        c.label, jpx::sen_to_yen(step), c.want_step_yen,
                        jpx::sen_to_yen(rt), jpx::sen_to_yen(up));
        }
    }
    // 通しティック番号の単調性: 全域をスキャンして price_of が厳密増加か
    {
        const jpx::TickGrid g(jpx::TickTable::Standard);
        i64 prev = -1;
        for (i64 t = 0; t < g.n_ticks(); t += 7) {
            const i64 p = g.price_of(t);
            if (p <= prev) { ++fails; std::printf("      FAIL 単調性 tick=%lld\n", (long long)t); break; }
            prev = p;
        }
    }
    // 値幅制限と価格窓
    {
        const jpx::TickGrid g(jpx::TickTable::Standard);
        const jpx::PriceWindow w = jpx::make_window(g, jpx::yen_to_sen(2500));
        // 基準値段2500円 -> 値幅±500円 -> [2000, 3000]円、呼値1円なので1001本
        if (w.low_sen != jpx::yen_to_sen(2000) || w.high_sen != jpx::yen_to_sen(3000) || w.n_ticks != 1001) {
            ++fails;
            std::printf("      FAIL 価格窓 base=2500円 -> [%.0f, %.0f]円 %dticks (期待 [2000,3000] 1001)\n",
                        jpx::sen_to_yen(w.low_sen), jpx::sen_to_yen(w.high_sen), w.n_ticks);
        }
        // 窓内インデックスの往復
        for (i32 i = 0; i < w.n_ticks; i += 13) {
            if (jpx::sen_to_idx(g, w, jpx::idx_to_sen(g, w, i)) != i) {
                ++fails; std::printf("      FAIL 窓内往復 idx=%d\n", i); break;
            }
        }
    }
    std::printf("      %s\n", fails == 0 ? "OK — 呼値・値幅制限・価格窓すべて整合"
                                         : "NG — 上の失敗を修正すること");
    return fails;
}

// ---------------------------------------------------------------------
int main(int argc, char** argv) {
    std::size_t per_day = 200'000;
    int days = 5;
    u64 seed = 42;
    if (argc > 1) per_day = std::stoull(argv[1]);
    if (argc > 2) days = std::stoi(argv[2]);
    if (argc > 3) seed = std::stoull(argv[3]);

    std::printf("=== stock-engine bench (%zu events/day x %d days, seed=%llu) ===\n\n",
                per_day, days, (unsigned long long)seed);

    if (check_tick_grid() != 0) return 1;

    StockFeed::Config cfg;
    cfg.events_per_day = per_day;
    cfg.days = days;
    cfg.seed = seed;
    StockFeed feed(cfg);
    std::vector<u8> buf;
    buf.reserve(per_day * days * 12);
    feed.generate(buf);
    std::printf("\nフィード生成: %zu メッセージ / %zu バイト\n",
                feed.messages_emitted(), buf.size());

    // ---- [2] 整合性検証 ----
    std::printf("\n[2/3] 板の整合性検証 (フラット配列 vs std::map 参照実装)\n");
    {
        StockBook book;
        RefBook ref;
        jpx::Session sess = jpx::Session::Off;
        std::size_t checked = 0, mismatches = 0, n_reset = 0, n_trade = 0, n_sess = 0;
        std::size_t cont_events = 0, crossed_cont = 0;
        for (std::size_t off = 0; off < buf.size();) {
            NormMsg m;
            off += decode(buf.data() + off, m);
            switch (m.type) {
            case 'R': book.reset(m.idx); ref.reset(); ++n_reset; continue;
            case 'T': ++n_trade; continue;
            case 'S': sess = static_cast<jpx::Session>(m.side); ++n_sess; continue;
            case 'A': book.add(m.id, static_cast<Side>(m.side), m.idx, m.qty);
                      ref.add(m.id, m.side, m.idx, m.qty); break;
            case 'X': case 'E': book.reduce(m.id, m.qty); ref.reduce(m.id, m.qty); break;
            case 'D': book.remove(m.id); ref.remove(m.id); break;
            default: continue;
            }
            ++checked;

            // ザラ場ではクロス(best_bid >= best_ask)が起きてはいけない。
            // 板寄せ前の気配はクロスして当然なので、そこは対象外。
            // 板寄せの約定処理を壊すとここが赤くなる(実際に一度そうなった)。
            if (jpx::is_continuous(sess) && book.two_sided()) {
                ++cont_events;
                if (book.spread_ticks() <= 0) {
                    if (crossed_cont < 3)
                        std::printf("      CROSSED ev=%zu bid=%d ask=%d (ザラ場でクロス)\n",
                                    checked, book.best_idx(BID), book.best_idx(ASK));
                    ++crossed_cont;
                }
            }

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
        }
        std::printf("      板イベント %zu 件を照合 (reset %zu / 約定 %zu / セッション %zu)\n",
                    checked, n_reset, n_trade, n_sess);
        std::printf("      %s\n", mismatches == 0 ? "OK — 全イベントで best bid/ask が一致"
                                                  : "NG — 不一致あり");
        if (mismatches) { std::printf("      不一致 %zu 件\n", mismatches); return 1; }
        std::printf("      ザラ場 %zu 件のクロス検査: %s\n", cont_events,
                    crossed_cont == 0 ? "OK — クロスなし" : "NG — クロスあり");
        if (crossed_cont) { std::printf("      クロス %zu 件\n", crossed_cont); return 1; }
    }

    // ---- [3] 性能計測 ----
    std::printf("\n[3/3] 性能計測 (rdtsc)\n");
    {
        const double ghz = [] {
            using clock = std::chrono::steady_clock;
            const auto t0 = clock::now();
            const unsigned long long c0 = __rdtsc();
            while (clock::now() - t0 < std::chrono::milliseconds(100)) {}
            const unsigned long long c1 = __rdtsc();
            const double ns = std::chrono::duration<double, std::nano>(clock::now() - t0).count();
            return double(c1 - c0) / ns;
        }();

        StockBook book;
        RollingZ roll(64);
        BestState prev{};
        std::vector<unsigned long long> lat;
        lat.reserve(1 << 20);
        std::size_t n_ev = 0;
        volatile double sink = 0;

        const unsigned long long t_start = __rdtsc();
        for (std::size_t off = 0; off < buf.size();) {
            const unsigned long long a = __rdtsc();
            NormMsg m;
            off += decode(buf.data() + off, m);
            switch (m.type) {
            case 'R': book.reset(m.idx); roll.clear(); prev = BestState{}; continue;
            case 'T': case 'S': continue;
            case 'A': book.add(m.id, static_cast<Side>(m.side), m.idx, m.qty); break;
            case 'X': case 'E': book.reduce(m.id, m.qty); break;
            case 'D': book.remove(m.id); break;
            default: continue;
            }
            const BestState cur = snap_best(book);
            sink += roll.push(ofi_contribution(prev, cur));
            prev = cur;
            const unsigned long long b = __rdtsc();
            if (lat.size() < lat.capacity()) lat.push_back(b - a);
            ++n_ev;
        }
        const unsigned long long t_end = __rdtsc();
        (void)sink;

        std::sort(lat.begin(), lat.end());
        auto pct = [&](double p) { return lat.empty() ? 0.0 : lat[std::size_t(p * (lat.size() - 1))] / ghz; };
        const double total_ns = double(t_end - t_start) / ghz;
        std::printf("      TSC          : %.3f GHz\n", ghz);
        std::printf("      処理イベント : %zu\n", n_ev);
        std::printf("      スループット : %.2f M events/sec\n", double(n_ev) * 1e3 / total_ns);
        std::printf("      レイテンシ   : p50 %.1f ns / p99 %.1f ns / p99.9 %.1f ns\n",
                    pct(0.50), pct(0.99), pct(0.999));
        std::printf("      (デコード+板更新+OFI+z-score を1イベントとして計測。rdtsc の\n");
        std::printf("       読み出しコスト約20-30サイクルを含むので実質はこれより速い)\n");
    }

    std::printf("\n=== 全チェック通過 ===\n");
    return 0;
}
