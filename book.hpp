#pragma once
// book.hpp — 呼値ティック番号で引くフラット配列L2オーダーブック(株版)
//
// hft-engine/orderbook.hpp と同じ定石を踏襲しつつ、株の事情に合わせて変えた点:
//
//  - 価格窓が「当日の値幅制限(ストップ高安)」から実行時に決まる。
//    hft-engine はコンパイル時定数 -DHFT_PX_WINDOW=... だったが、株は銘柄ごとに
//    基準値段が違うので実行時。窓が数千ティックなので配列は数十KB、L2に常駐する。
//  - 板が扱うのは「窓内インデックス」だけ。円との変換は jp_market.hpp の
//    TickGrid が境界(アダプタ/フィード)で1回だけ行い、ホットパスには出さない。
//  - 板寄せ(始値決定)のための約定価格計算 itayose_price() を持つ。ザラ場しかない
//    暗号資産には無い、株特有の処理。
//
// 変わっていない定石:
//  - 価格→配列添字は減算1回。std::map のポインタチェイスを排除
//  - 注文IDは密な連番前提で配列直引き(ハッシュ表すら使わない)
//  - best bid/ask は増分維持。消滅時だけ touch 近傍を線形走査
//  - ホットパスでヒープ確保なし
#include <cassert>
#include <cstdint>
#include <vector>

#include "jp_market.hpp"

namespace stk {

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

#if defined(__GNUC__)
#define STK_LIKELY(x) __builtin_expect(!!(x), 1)
#define STK_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define STK_LIKELY(x) (x)
#define STK_UNLIKELY(x) (x)
#endif

enum Side : u8 { BID = 0, ASK = 1 };

struct Order {
    u32 px_idx;  // 窓内インデックス
    u32 qty;     // 残株数(0 = 消滅)
    u8 side;
    u8 _pad[3];
};
static_assert(sizeof(Order) == 12);

class StockBook {
public:
    StockBook() = default;
    explicit StockBook(i32 n_ticks) { reset(n_ticks); }

    // 窓を張り直す(寄付・基準値段変更・アダプタからの 'R' 受信時)
    void reset(i32 n_ticks) {
        n_ = n_ticks;
        qty_[BID].assign(static_cast<std::size_t>(n_), 0);
        qty_[ASK].assign(static_cast<std::size_t>(n_), 0);
        orders_.clear();
        orders_.reserve(1 << 18);
        best_[BID] = -1;
        best_[ASK] = n_;
    }

    // ---- フィードイベント適用 ----
    void add(u32 id, Side s, i32 idx, u32 qty) {
        assert(idx >= 0 && idx < n_);
        if (id >= orders_.size()) orders_.resize(id + 1, Order{0, 0, 0, {}});
        orders_[id] = Order{static_cast<u32>(idx), qty, static_cast<u8>(s), {}};
        qty_[s][static_cast<std::size_t>(idx)] += qty;
        if (s == BID) { if (idx > best_[BID]) best_[BID] = idx; }
        else          { if (idx < best_[ASK]) best_[ASK] = idx; }
    }

    // 部分取消・全量取消・約定はすべて「数量削減」として同一パスで処理する
    void reduce(u32 id, u32 dec) {
        if (STK_UNLIKELY(id >= orders_.size())) return;
        Order& o = orders_[id];
        if (STK_UNLIKELY(o.qty == 0)) return;  // 二重取消は無視(実フィードでも防御する)
        if (dec > o.qty) dec = o.qty;
        o.qty -= dec;
        const Side s = static_cast<Side>(o.side);
        qty_[s][o.px_idx] -= dec;
        if (qty_[s][o.px_idx] == 0 && static_cast<i32>(o.px_idx) == best_[s]) repair_best(s);
    }

    void remove(u32 id) {
        if (STK_UNLIKELY(id >= orders_.size())) return;
        reduce(id, orders_[id].qty);
    }

    // ---- L2アクセス ----
    i32 best_idx(Side s) const { return best_[s]; }
    bool has_side(Side s) const { return s == BID ? best_[BID] >= 0 : best_[ASK] < n_; }
    bool two_sided() const { return has_side(BID) && has_side(ASK); }
    i64 qty_at(Side s, i32 idx) const { return qty_[s][static_cast<std::size_t>(idx)]; }
    i64 best_qty(Side s) const { return qty_[s][static_cast<std::size_t>(best_[s])]; }
    double mid_idx() const { return 0.5 * (best_[BID] + best_[ASK]); }
    i32 spread_ticks() const { return best_[ASK] - best_[BID]; }
    i32 n_ticks() const { return n_; }
    const Order& order(u32 id) const { return orders_[id]; }
    std::size_t order_capacity() const { return orders_.size(); }

    // best から levels 段ぶんの累積株数。kabuステーションAPIが返す「10本値」に対応し、
    // best 1段だけでは見えない厚みの偏りを捉える(株では見せ板/アイスバーグの影響で
    // best だけの不均衡が騙しになりやすいため、多段の厚みが効く)。
    i64 depth_qty(Side s, i32 levels) const {
        if (!has_side(s)) return 0;
        i64 sum = 0;
        if (s == BID) {
            for (i32 i = best_[BID], k = 0; i >= 0 && k < levels; --i, ++k) sum += qty_[BID][static_cast<std::size_t>(i)];
        } else {
            for (i32 i = best_[ASK], k = 0; i < n_ && k < levels; ++i, ++k) sum += qty_[ASK][static_cast<std::size_t>(i)];
        }
        return sum;
    }

    // ---- 板寄せ(itayose): 始値・引け値の決定 ----
    // 約定株数が最大になる価格を選ぶ(同数なら中央寄り)。東証の板寄せは
    // 実際にはさらに細かい優先ルールがあるが、約定数量最大化が本質。
    // 板が両サイド無い/クロスしていない場合は -1 を返す。
    //
    // 呼ばれるのは1日に数回(寄付・前引け・後場寄り・引け)だけなのでホットパスではない。
    i32 itayose_price_idx() const {
        if (!two_sided() || best_[BID] < best_[ASK]) return -1;  // クロスしていない
        const i32 lo = best_[ASK], hi = best_[BID];
        const std::size_t m = static_cast<std::size_t>(hi - lo + 1);

        // cum_bid[k] = 価格 lo+k 以上の買い総数(降順累積)
        std::vector<i64> cum_bid(m, 0);
        i64 acc = 0;
        for (std::size_t k = m; k-- > 0;) {
            acc += qty_[BID][static_cast<std::size_t>(lo) + k];
            cum_bid[k] = acc;
        }
        // 各候補価格 p について、約定株数 = min(p以上の買い, p以下の売り)
        i64 cum_ask = 0, best_vol = -1;
        i32 best_p = -1;
        for (std::size_t k = 0; k < m; ++k) {
            cum_ask += qty_[ASK][static_cast<std::size_t>(lo) + k];
            const i64 vol = cum_bid[k] < cum_ask ? cum_bid[k] : cum_ask;
            if (vol > best_vol) { best_vol = vol; best_p = lo + static_cast<i32>(k); }
        }
        return best_p;
    }

private:
    // best が消えたときだけ、悪い方向へ空レベルを走査する。
    // 板の性質上ほぼ touch 直下で見つかるので分岐予測が効く。
    void repair_best(Side s) {
        i32 i = best_[s];
        if (s == BID) { while (i >= 0 && qty_[BID][static_cast<std::size_t>(i)] == 0) --i; best_[BID] = i; }
        else          { while (i < n_ && qty_[ASK][static_cast<std::size_t>(i)] == 0) ++i; best_[ASK] = i; }
    }

    std::vector<i64> qty_[2];   // [side][idx] -> 合計株数
    std::vector<Order> orders_; // 注文ID直引き
    i32 n_ = 0;
    i32 best_[2] = {-1, 0};     // 空サイドの番兵: BID=-1, ASK=n_
};

}  // namespace stk
