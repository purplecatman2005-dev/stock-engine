#pragma once
// features.hpp — GBDT用の特徴量エンジン(株版)
//
// dump_features.cpp(学習データ生成)と backtest_ml.cpp(推論バックテスト)で
// 完全に同一の特徴量を計算するための共有実装。ホットパスで割当ゼロ。
//
// hft-engine/ml_features.hpp の12種に、株で意味を持つ4種を足した16種:
//
//   板の形(hft-engine と共通)
//     ofi_z16 / ofi_z64 / ofi_z256 : OFIのローリングz-score(短期/中期/長期)
//     micro_prem                   : マイクロプライス - mid (次のmid変動の推定)
//     micro_prem_rel               : 上をスプレッドの半分で割った正規化版
//     spread                       : スプレッド(ティック)
//     spread_dev64                 : スプレッドの直近64イベント平均からの偏差
//     imbalance                    : best 1段の板不均衡
//     ret8 / ret32 / ret128        : 過去8/32/128イベントのmid変化
//     rvol64                       : 直近64イベントの実現ボラ
//
//   株で追加した4種
//     depth_imb   : 上位5段の厚み不均衡。株は見せ板やアイスバーグで best 1段の
//                   不均衡が騙しになりやすく、多段の厚みを併せて見るのが実務的
//     tod         : 立会の進捗 0.0(寄り)〜1.0(引け)。株のボラ・出来高・スプレッドは
//                   日中U字なので、同じ板の形でも時間帯で意味が変わる
//     u_shape     : その時刻のU字強度。tod の非線形変換をモデルに与えて木の分割を助ける
//     is_pm       : 後場かどうか(0/1)。昼休みを挟むと情報が溜まって寄りが跳ねる
//
// 「時間帯」を入れられるのが株版の利点。24時間動く暗号資産には無い構造。
#include <algorithm>
#include <cmath>
#include <cstddef>

#include "jp_market.hpp"
#include "signals.hpp"

namespace stk {

inline constexpr int ML_NFEAT = 16;
inline constexpr const char* ML_FEAT_NAMES[ML_NFEAT] = {
    "ofi_z64", "ofi_z256", "micro_prem", "spread", "imbalance", "ret32", "ret128",
    "ofi_z16", "spread_dev64", "rvol64", "ret8", "micro_prem_rel",
    "depth_imb", "tod", "u_shape", "is_pm",
};

// イベント毎に update() を呼ぶ。ウォームアップが済むと true を返し f[] が有効になる。
class FeatureEngine {
public:
    FeatureEngine() : z16_(16), z64_(64), z256_(256) {}

    // 'R'(板リセット)を受けたら作り直す。日をまたぐと価格窓が変わるので、
    // 窓内インデックスで計算しているリターンやボラは連続性を失う。
    void reset() { *this = FeatureEngine(); }

    // sess / ms_of_day は 'S' メッセージから来る現在のセッションと時刻。
    bool update(const BestState& prev, const BestState& cur, jpx::Session sess,
                i64 ms_of_day, double* f) {
        const double ofi = ofi_contribution(prev, cur);
        const double zs = z16_.push(ofi);
        const double za = z64_.push(ofi);
        const double zb = z256_.push(ofi);
        if (!cur.valid) return false;

        const double mid = 0.5 * (cur.pb + cur.pa);
        const double spread = double(cur.pa - cur.pb);

        // 実現ボラ: 1イベントmid差分の2乗の直近VW個移動和
        if (has_prev_mid_) {
            const double d = mid - prev_mid_;
            vol_sum_ -= vol_[vhead_];
            vol_[vhead_] = d * d;
            vol_sum_ += d * d;
            vhead_ = (vhead_ + 1) & VMASK;
            if (vn_ < VW) ++vn_;
        }
        prev_mid_ = mid;
        has_prev_mid_ = true;

        // スプレッド移動平均(現在値との偏差を特徴量にする)
        sp_sum_ -= sp_[shead_];
        sp_[shead_] = spread;
        sp_sum_ += spread;
        shead_ = (shead_ + 1) & VMASK;
        if (sn_ < VW) ++sn_;

        mids_[head_] = mid;
        head_ = (head_ + 1) & MASK;
        if (n_ < RING) ++n_;

        if (!z256_.warmed() || n_ < RING || vn_ < VW || sn_ < VW) return false;

        const double micro_prem = microprice(cur) - mid;
        f[0] = za;
        f[1] = zb;
        f[2] = micro_prem;
        f[3] = spread;
        f[4] = best_imbalance(cur);
        f[5] = mid - at_ago(32);
        f[6] = mid - at_ago(128);
        f[7] = zs;
        f[8] = spread - sp_sum_ / double(VW);
        // vol_sum_ は移動和なのでFP誤差で微小負になりうる(sqrtのNaN防止)
        f[9] = std::sqrt(std::max(0.0, vol_sum_) / double(VW));
        f[10] = mid - at_ago(8);
        // 有効な板では pa>pb (整数ティック) のため spread>=1 だが、0除算を防御
        f[11] = micro_prem / (0.5 * std::max(spread, 1.0));
        f[12] = depth_imbalance(cur);
        const double prog = jpx::day_progress(ms_of_day);
        f[13] = prog;
        f[14] = jpx::u_shape(prog);
        f[15] = (sess == jpx::Session::Afternoon || sess == jpx::Session::AfternoonAuction ||
                 sess == jpx::Session::ClosingAuction || sess == jpx::Session::Close)
                    ? 1.0 : 0.0;
        return true;
    }

private:
    static constexpr std::size_t RING = 256;  // 2冪, ret128より大きい
    static constexpr std::size_t MASK = RING - 1;
    static constexpr std::size_t VW = 64;     // 実現ボラ/スプレッドMAの窓
    static constexpr std::size_t VMASK = VW - 1;

    double at_ago(std::size_t k) const {
        // head_ は「次に書く位置」= 最新は head_-1。k個前 = head_-1-k
        return mids_[(head_ + RING - 1 - k) & MASK];
    }

    RollingZ z16_, z64_, z256_;
    double mids_[RING] = {};
    std::size_t head_ = 0, n_ = 0;

    double vol_[VW] = {}, vol_sum_ = 0.0;
    std::size_t vhead_ = 0, vn_ = 0;
    double prev_mid_ = 0.0;
    bool has_prev_mid_ = false;

    double sp_[VW] = {}, sp_sum_ = 0.0;
    std::size_t shead_ = 0, sn_ = 0;
};

}  // namespace stk
