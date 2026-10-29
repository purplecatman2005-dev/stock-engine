#pragma once
// signals.hpp — マイクロストラクチャ・シグナル(株版)
//
//  - OFI (Order Flow Imbalance, Cont-Kukanov-Stoikov 2014):
//    best 両サイドの価格・数量の変化から「買い圧 - 売り圧」の純流入を1イベント毎に出す。
//    板の上げ下げと積み増し/引き剥がしを1つの符号付き量にまとめた指標で、
//    短期のmid変動に対する説明力が実証的に最も高い板指標。
//  - microprice: 数量加重mid。次のmid変動の不偏推定として板読みの基本。
//  - ローリングz-score: 固定長リングバッファ + 累積和で O(1)/イベント更新。
//
// hft-engine/signals.hpp からの変更点は「価格が窓内インデックス(=呼値の段数)」に
// なったことだけ。OFI の式自体は価格の絶対値ではなく前後比較しか使わないので、
// 呼値が価格帯で変わる株でもそのまま成立する ---
// むしろ「1ティック = 1呼値」で正規化されているぶん株の方が素直。
#include <algorithm>
#include <cmath>
#include <vector>

#include "book.hpp"

namespace stk {

// 1イベント分のOFI寄与を出すための best スナップショット。
// 価格は窓内インデックス、数量は株数。
struct BestState {
    i32 pb = 0, pa = 0;  // best bid/ask (窓内インデックス)
    i64 qb = 0, qa = 0;  // best 株数
    i64 db = 0, da = 0;  // 上位N段の累積株数(厚み)
    bool valid = false;
};

inline constexpr i32 DEPTH_LEVELS = 5;  // kabuの10本値の半分。見せ板の影響を薄めつつ厚みを見る

inline BestState snap_best(const StockBook& b) {
    BestState s;
    if (b.two_sided()) {
        s.pb = b.best_idx(BID);
        s.qb = b.best_qty(BID);
        s.pa = b.best_idx(ASK);
        s.qa = b.best_qty(ASK);
        s.db = b.depth_qty(BID, DEPTH_LEVELS);
        s.da = b.depth_qty(ASK, DEPTH_LEVELS);
        s.valid = true;
    }
    return s;
}

// OFI 寄与 e_n。
//   買い圧: bid が上がる or 同値で積まれる  -> +
//   売り圧: ask が下がる or 同値で積まれる  -> -
inline double ofi_contribution(const BestState& prev, const BestState& cur) {
    if (!prev.valid || !cur.valid) return 0.0;
    double e = 0.0;
    if (cur.pb >= prev.pb) e += double(cur.qb);
    if (cur.pb <= prev.pb) e -= double(prev.qb);
    if (cur.pa <= prev.pa) e -= double(cur.qa);
    if (cur.pa >= prev.pa) e += double(prev.qa);
    return e;
}

// 数量加重mid(窓内インデックス単位)。
// bid側が厚いほど ask 寄りになる = 次に上がりやすい、という読み。
inline double microprice(const BestState& s) {
    const double d = double(s.qb + s.qa);
    if (d <= 0) return 0.5 * (s.pb + s.pa);
    return (s.pb * double(s.qa) + s.pa * double(s.qb)) / d;
}

// best 1段の板不均衡 [-1, +1]
inline double best_imbalance(const BestState& s) {
    const double d = double(s.qb + s.qa);
    return d > 0 ? double(s.qb - s.qa) / d : 0.0;
}

// 上位N段の厚み不均衡 [-1, +1]。株では best だけの不均衡が見せ板で騙されやすいので、
// 多段の厚みを併せて見るのが実務的。
inline double depth_imbalance(const BestState& s) {
    const double d = double(s.db + s.da);
    return d > 0 ? double(s.db - s.da) / d : 0.0;
}

// 固定窓ローリング平均/分散(リングバッファ+累積和; O(1)更新)
class RollingZ {
public:
    explicit RollingZ(std::size_t window) : buf_(window, 0.0), w_(window) {}

    // 新しい値を入れ、窓合計のz-scoreを返す(窓が埋まるまでは0)
    double push(double x) {
        sum_ -= buf_[i_];
        sumsq_ -= buf_[i_] * buf_[i_];
        buf_[i_] = x;
        sum_ += x;
        sumsq_ += x * x;
        if (++i_ == w_) i_ = 0;
        if (n_ < w_) ++n_;
        if (n_ < w_) return 0.0;
        const double mean = sum_ / double(w_);
        const double var = sumsq_ / double(w_) - mean * mean;
        // 窓「合計」の偏り: sum ~ N(0, w*var) を仮定して正規化(OFIは中心対称)
        if (var <= 1e-12) return 0.0;
        return sum_ / std::sqrt(double(w_) * var);
    }

    void clear() {
        std::fill(buf_.begin(), buf_.end(), 0.0);
        i_ = n_ = 0;
        sum_ = sumsq_ = 0.0;
    }

    bool warmed() const { return n_ >= w_; }

private:
    std::vector<double> buf_;
    std::size_t w_, i_ = 0, n_ = 0;
    double sum_ = 0.0, sumsq_ = 0.0;
};

}  // namespace stk
