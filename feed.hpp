#pragma once
// feed.hpp — ワイヤフォーマットと合成フィード生成器(株版)
//
// ワイヤフォーマット(リトルエンディアン, パック):
//   'A' Add     : u8 type, u32 id, u8 side, i32 idx(窓内インデックス), u32 qty  (14B)
//   'X' Cancel  : u8 type, u32 id, u32 dec_qty                                  ( 9B)
//   'E' Exec    : u8 type, u32 id, u32 exec_qty                                 ( 9B)
//   'D' Delete  : u8 type, u32 id                                               ( 5B)
// 実データ用の拡張(live/replay とアダプタのみが解釈):
//   'R' Reset   : u8 type, i64 base_sen, i32 n_ticks, u8 table                  (14B)
//                 板全消去 + 当日の価格窓を張り直す。base_sen はストップ安価格(銭)
//   'T' Trade   : u8 type, u8 side, i32 idx, u32 qty                            (10B)
//                 約定(板には適用しない。記録・出来高集計用)
//   'S' Session : u8 type, u8 session, u32 ms_of_day_x1000... -> u8 + u8 + i32   ( 6B)
//                 セッション遷移(寄付板寄せ/前引け/後場寄り/引け)。
//                 i32 は 0時からの経過秒。ザラ場外でシグナルを出さないために使う。
//
// ストリーム/収録の共通フレーム: [u64 ts_ns (epoch)] + [ワイヤメッセージ]
// (合成フィードはタイムスタンプ無しの生メッセージ列を作る。bench/backtest 用)
//
// 数量単位: 株数(必ず100株単位)。価格: 窓内インデックス = ストップ安からの呼値段数。
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "book.hpp"
#include "jp_market.hpp"

namespace stk {

#pragma pack(push, 1)
struct MsgAdd     { u8 type; u32 id; u8 side; i32 idx; u32 qty; };
struct MsgCancel  { u8 type; u32 id; u32 dec; };
struct MsgExec    { u8 type; u32 id; u32 qty; };
struct MsgDelete  { u8 type; u32 id; };
struct MsgReset   { u8 type; i64 base_sen; i32 n_ticks; u8 table; };
struct MsgTrade   { u8 type; u8 side; i32 idx; u32 qty; };
struct MsgSession { u8 type; u8 session; i32 sec_of_day; };
#pragma pack(pop)
static_assert(sizeof(MsgAdd) == 14 && sizeof(MsgCancel) == 9 && sizeof(MsgDelete) == 5);
static_assert(sizeof(MsgReset) == 14 && sizeof(MsgTrade) == 10 && sizeof(MsgSession) == 6);

// メッセージ種別ごとのワイヤ長。0 = 不明タイプ
inline std::size_t msg_size(u8 t) {
    switch (t) {
    case 'A': case 'R': return 14;
    case 'X': case 'E': return 9;
    case 'D': return 5;
    case 'T': return 10;
    case 'S': return 6;
    default:  return 0;
    }
}

// パーサが使う正規化メッセージ
struct NormMsg {
    u8 type;      // 'A' 'X' 'E' 'D' 'R' 'T' 'S'
    u8 side;      // A/T: サイド、S: セッション
    u32 id;
    i32 idx;      // A/T: 窓内インデックス、R: n_ticks、S: 0時からの経過秒
    u32 qty;
    i64 base_sen; // R のみ
    u8 table;     // R のみ (0=Standard, 1=Topix100)
};

// バッファ先頭のメッセージを1件デコードし、消費バイト数を返す
inline std::size_t decode(const u8* p, NormMsg& m) {
    m = NormMsg{};
    switch (*p) {
    case 'A': { MsgAdd a;     std::memcpy(&a, p, sizeof a); m.type = a.type; m.side = a.side; m.id = a.id; m.idx = a.idx; m.qty = a.qty; return sizeof a; }
    case 'X': { MsgCancel c;  std::memcpy(&c, p, sizeof c); m.type = c.type; m.id = c.id; m.qty = c.dec;  return sizeof c; }
    case 'E': { MsgExec e;    std::memcpy(&e, p, sizeof e); m.type = e.type; m.id = e.id; m.qty = e.qty;  return sizeof e; }
    case 'D': { MsgDelete d;  std::memcpy(&d, p, sizeof d); m.type = d.type; m.id = d.id;                 return sizeof d; }
    case 'R': { MsgReset r;   std::memcpy(&r, p, sizeof r); m.type = r.type; m.base_sen = r.base_sen; m.idx = r.n_ticks; m.table = r.table; return sizeof r; }
    case 'T': { MsgTrade t;   std::memcpy(&t, p, sizeof t); m.type = t.type; m.side = t.side; m.idx = t.idx; m.qty = t.qty; return sizeof t; }
    case 'S': { MsgSession s; std::memcpy(&s, p, sizeof s); m.type = s.type; m.side = s.session; m.idx = s.sec_of_day;    return sizeof s; }
    default:  m.type = 0; return 1;
    }
}

// =====================================================================
//  合成フィード生成器
// =====================================================================
// hft-engine の合成フィードに、株の1日を足したもの。
//
//  - 潜在ドリフト(マルコフ切替)が「発注サイドの偏り」と「フェアバリューの移動」の
//    両方を駆動する。これでオーダーフローが将来のmid変動を弱く先行し、OFIの
//    検出力がバックテストで測れるようになる(ここは hft-engine と同じ)。
//  - 日中U字: 寄りと引けでイベント密度・ボラ・スプレッドが跳ね、昼前後で凪ぐ。
//    実際の株の出来高分布はこの形。時間帯が特徴量として意味を持つ理由。
//  - セッション: 09:00 寄付板寄せ / 11:30 前引け / 12:30 後場寄り / 15:30 引け。
//    板寄せでは注文を溜めてから単一価格で一括約定する。
//  - 売買単位100株。価格は呼値グリッドに乗る(窓内インデックスなので自動)。
//
// これは実市場のエッジの証明ではなく、「エンジンとシグナル計算が正しく動く」
// ことを確認するための構造。
class StockFeed {
public:
    struct Config {
        double base_yen = 2500.0;              // 前日終値(基準値段)
        jpx::TickTable table = jpx::TickTable::Standard;
        std::size_t events_per_day = 200'000;  // 1日あたりのイベント数
        int days = 1;
        u64 seed = 42;
        // 板に生かしておく注文数の目標。新規/取消の比率をこれに合わせて自動調整する。
        //
        // これが無いと板が単調に厚くなり、best が巨大な滞留注文で固定されて
        // mid が凍る(hft-engine の合成フィードは 1.4M イベントでこれに当たっていた)。
        // 目標厚みに張り付かせることで何イベント流しても定常な板になる。
        std::size_t target_live_orders = 800;

        // --- シグナルの「見つけやすさ」を決める2つのつまみ ---
        //
        // 潜在ドリフト状態(平均500イベントで ±1/0 に切り替わる)が、
        //   (a) 発注サイドの偏り = flow_coupling
        //   (b) フェアバリューの移動 = drift_strength
        // の両方を駆動する。だからオーダーフローが将来のmid変動を先行する。
        //
        // 両方を0にすると「オーダーフローに情報が一切ない板」になる。
        // backtest.exe はこれを帰無仮説として毎回同時に測り、差分を本当のエッジとして
        // 報告する。合成フィードには板がフェアに追随する遅れによる見かけの的中率
        // (既定値で約64%)が必ず乗るので、帰無との差でしか判断してはいけない。
        //
        // 既定値の実測(200k events/日 x 8日): 帰無 63.8% -> 情報あり 71.8%(+8.0pt)、
        // 平均スプレッド1.45ティック(2500円銘柄で約6bps = 実際の流動株に近い水準)。
        double flow_coupling = 0.15;     // ドリフトが発注サイドを偏らせる強さ
        double drift_strength = 0.008;   // ドリフト1単位あたりのフェアバリュー移動(ティック/イベント)
        double noise_strength = 0.15;    // フェアバリューのランダム成分(ティック/イベント)
    };

    explicit StockFeed(const Config& cfg)
        : cfg_(cfg), rng_(cfg.seed), grid_(cfg.table) {
        base_sen_ = jpx::yen_to_sen(cfg.base_yen);
        window_ = jpx::make_window(grid_, base_sen_);
    }

    const jpx::TickGrid& grid() const { return grid_; }
    const jpx::PriceWindow& window() const { return window_; }
    std::size_t messages_emitted() const { return n_msgs_; }

    // 全日分のメッセージ列を生成する
    void generate(std::vector<u8>& out) {
        out_ = &out;
        for (int d = 0; d < cfg_.days; ++d) run_day();
        out_ = nullptr;
    }

private:
    // ---- 1立会日 ----
    void run_day() {
        // 寄付: 板をまっさらにして当日の窓を張る
        book_.reset(window_.n_ticks);
        lvl_[BID].assign(static_cast<std::size_t>(window_.n_ticks), std::vector<u32>());
        lvl_[ASK].assign(static_cast<std::size_t>(window_.n_ticks), std::vector<u32>());
        live_.clear();
        live_pos_.clear();
        next_id_ = 0;
        emit_reset();

        // フェアバリューは窓の中央(=前日終値)から始める
        fair_idx_ = 0.5 * (window_.n_ticks - 1);
        drift_ = 0;

        // 08:00-09:00 寄前注文受付: 注文だけ溜める(約定なし)
        emit_session(jpx::Session::PreOpen, jpx::T_PREOPEN * 60);
        const std::size_t n_preopen = cfg_.events_per_day / 20;
        for (std::size_t i = 0; i < n_preopen; ++i) { drift_step(0.3); do_add(/*allow_cross=*/true); }

        // 09:00 寄付板寄せ: クロスを一括約定させて始値を決める
        emit_session(jpx::Session::MorningAuction, jpx::T_OPEN * 60);
        run_itayose();

        // 前場ザラ場 09:00-11:30
        run_continuous(jpx::Session::Morning, jpx::T_OPEN, jpx::T_AM_CLOSE);

        // 11:30 前引け板寄せ
        emit_session(jpx::Session::MorningClose, jpx::T_AM_CLOSE * 60);
        run_itayose();

        // 昼休み: 注文の出入りだけ(約定なし)
        emit_session(jpx::Session::Lunch, jpx::T_AM_CLOSE * 60 + 1);
        const std::size_t n_lunch = cfg_.events_per_day / 50;
        for (std::size_t i = 0; i < n_lunch; ++i) { drift_step(0.2); if (u01() < 0.6) do_add(true); else do_cancel(); }

        // 12:30 後場寄り板寄せ
        emit_session(jpx::Session::AfternoonAuction, jpx::T_PM_OPEN * 60);
        run_itayose();

        // 後場ザラ場 12:30-15:25
        run_continuous(jpx::Session::Afternoon, jpx::T_PM_OPEN, jpx::T_CA_START);

        // 15:25-15:30 クロージング・オークション: 注文受付のみ、約定なし
        emit_session(jpx::Session::ClosingAuction, jpx::T_CA_START * 60);
        const std::size_t n_ca = cfg_.events_per_day / 25;
        for (std::size_t i = 0; i < n_ca; ++i) { drift_step(0.5); if (u01() < 0.75) do_add(true); else do_cancel(); }

        // 15:30 引け板寄せ
        emit_session(jpx::Session::Close, jpx::T_CLOSE * 60);
        run_itayose();
        emit_session(jpx::Session::Off, jpx::T_CLOSE * 60 + 1);

        // 翌日の基準値段 = 当日終値。値幅制限も窓も張り直しになる
        base_sen_ = jpx::idx_to_sen(grid_, window_, clamp_idx(static_cast<i32>(fair_idx_ + 0.5)));
        window_ = jpx::make_window(grid_, base_sen_);
    }

    // ザラ場: [from_min, to_min) を日中U字に沿ってイベントで埋める
    void run_continuous(jpx::Session sess, i32 from_min, i32 to_min) {
        emit_session(sess, from_min * 60);
        // ザラ場全体のイベント数を、この区間の長さで按分
        const double frac = double(to_min - from_min) / jpx::TRADING_MINUTES;
        const std::size_t n = static_cast<std::size_t>(cfg_.events_per_day * 0.85 * frac);
        i32 last_emit_min = from_min;
        for (std::size_t i = 0; i < n; ++i) {
            // 進捗に応じたU字の強度。density が高い時間帯ほどボラも大きくする
            const double t = double(i) / double(n ? n : 1);
            const i32 min_now = from_min + static_cast<i32>(t * (to_min - from_min));
            const double u = jpx::u_shape(jpx::day_progress(i64(min_now) * 60'000));
            if (min_now / 30 != last_emit_min / 30) {  // 30分毎に時刻を刻む
                emit_session(sess, min_now * 60);
                last_emit_min = min_now;
            }
            drift_step(u);
            // フェアバリューから乖離した気配は真っ先に食われる。これで mid が
            // フェアに張り付き、板の遅れが将来のmid変動を予測してしまう見せかけの
            // エッジが消える(詳細は arbitrage() のコメント)。
            if (arbitrage()) continue;
            // 新規/取消の比率を板の厚みで自動調整する(定常な板を保つ)
            const double p_add = add_probability();
            const double r = u01();
            if (r < p_add || live_.empty()) do_add(false);
            else if (r < 0.85)             do_cancel();
            else                           do_exec(u);
        }
    }

    // 板寄せ: クロスしている注文を単一価格で一括約定させる
    void run_itayose() {
        // 約定価格は最初に一度だけ決まる。以降はその価格で FIFO に食い合わせる。
        const i32 p = book_.itayose_price_idx();
        if (p >= 0)
            for (int guard = 0; guard < 1'000'000 && match_at(p); ++guard) {}
        // 板寄せ後のフェアバリューは約定価格(=新しいmid)に寄る
        const BestSnapshot s = snapshot();
        if (s.valid) fair_idx_ = 0.5 * (s.pb + s.pa);
    }

    // 板寄せ価格 p で1回分の食い合わせを行う。約定が起きたら true。
    //
    // 板寄せは「クロスしている注文をすべて単一価格 p で約定させる」処理なので、
    // 停止条件は「クロスが解消したこと」= best_bid < best_ask。
    // p 以上の買いだけ、p 以下の売りだけ、を見て止めると反対側に約定すべき注文が
    // 残ってクロスしたままになる(板寄せ後の板がクロスするのは仕様違反)。
    bool match_at(i32 p) {
        if (!book_.two_sided()) return false;
        const i32 bi = book_.best_idx(BID);
        const i32 ai = book_.best_idx(ASK);
        if (bi < ai) return false;  // クロス解消済み
        auto& bq = lvl_[BID][static_cast<std::size_t>(bi)];
        auto& aq = lvl_[ASK][static_cast<std::size_t>(ai)];
        if (bq.empty() || aq.empty()) return false;
        const u32 bid_id = bq.front(), ask_id = aq.front();
        const u32 fill = std::min(book_.order(bid_id).qty, book_.order(ask_id).qty);
        if (fill == 0) return false;
        emit_reduce('E', bid_id, fill);
        emit_reduce('E', ask_id, fill);
        emit_trade(BID, p, fill);
        return true;
    }

    // ---- 個別イベント ----
    void drift_step(double intensity) {
        if (u01() < 1.0 / 500.0) drift_ = static_cast<int>(rng_() % 3) - 1;
        // ボラは日中U字に比例。1イベントあたりの動きは呼値の数%程度
        fair_idx_ += intensity * (cfg_.drift_strength * drift_ + cfg_.noise_strength * norm());
        const double margin = 8.0;
        if (fair_idx_ < margin) fair_idx_ = margin;
        if (fair_idx_ > window_.n_ticks - margin) fair_idx_ = window_.n_ticks - margin;
    }

    void do_add(bool allow_cross) {
        // ドリフト上向き -> 買い注文が増える(オーダーフローに情報が乗る)
        const Side s = (u01() < 0.5 + cfg_.flow_coupling * drift_) ? BID : ASK;
        const i32 f = static_cast<i32>(fair_idx_);
        i32 idx = (s == BID) ? f - 1 - geometric(0.45, 12) : f + 1 + geometric(0.45, 12);
        if (!allow_cross) {
            // ザラ場では板をクロスさせない(クロス分は約定として現れるため)
            if (s == BID && book_.has_side(ASK) && idx >= book_.best_idx(ASK)) idx = book_.best_idx(ASK) - 1;
            if (s == ASK && book_.has_side(BID) && idx <= book_.best_idx(BID)) idx = book_.best_idx(BID) + 1;
        }
        idx = clamp_idx(idx);
        // 売買単位100株、1〜10単位
        emit_add(s, idx, u32(jpx::LOT) * (1 + u32(rng_() % 10)));
    }

    // 新規注文を出す確率。板が目標より厚ければ下げ、薄ければ上げる。
    // 約定は 0.15 で固定(価格発見を駆動する要素なので厚みで変えたくない)。
    double add_probability() const {
        const double load = double(live_.size()) / double(cfg_.target_live_orders);
        const double p = 0.55 - 0.40 * std::tanh(2.0 * (load - 1.0));
        return 0.85 * (p < 0.05 ? 0.05 : (p > 0.95 ? 0.95 : p));
    }

    void do_cancel() {
        if (live_.empty()) return;
        // トーナメント選択: 候補を数本引いて「フェアバリューから最も遠い」注文を取消す。
        //
        // 一律ランダムに取消すと、フェアが動いたあとに取り残された注文(stale quote)が
        // いつまでも板に残り、best がそれに固定されて mid が動かなくなる。
        // 実際のマーケットメイカーもフェアが動けば遠くなった気配を引っ込めるので、
        // 遠い注文を優先的に消すのが構造的に正しい。
        u32 id = live_[rng_() % live_.size()];
        double worst = std::abs(double(book_.order(id).px_idx) - fair_idx_);
        for (int t = 1; t < 4; ++t) {
            const u32 c = live_[rng_() % live_.size()];
            const double d = std::abs(double(book_.order(c).px_idx) - fair_idx_);
            if (d > worst) { worst = d; id = c; }
        }
        const Order& o = book_.order(id);
        if (o.qty == 0) return;
        if (u01() < 0.3 && o.qty > u32(jpx::LOT)) {  // 部分取消(100株単位)
            const u32 units = o.qty / u32(jpx::LOT);
            const u32 dec = u32(jpx::LOT) * (1 + u32(rng_() % units));
            emit_reduce('X', id, dec >= o.qty ? o.qty : dec);
        } else {
            emit_reduce('D', id, o.qty);
        }
    }

    // 裁定: フェアバリューを跨いだ気配水準を丸ごと食う。約定が起きたら true。
    //
    // なぜ「1本ずつ」ではなく「水準ごと」なのか:
    //   1本ずつ食うと touch がフェアに追いつくまで何十イベントもかかり、板がフェアに
    //   遅れて追随する状態が続く。すると OFI は「まだ埋まっていない乖離」を読むだけで
    //   将来の mid を高精度に当ててしまう。実測で、オーダーフローに情報を一切入れない
    //   (drift=0, flow_coupling=0) フィードでも的中率88%が出た。無関係な位置の z を
    //   使ったシャッフル検定は49.3%だったので、計測ではなく板の遅れが原因だと確定した。
    //   実市場では割安な気配は一瞬で剥がされるので、水準ごと食うのが構造的に正しい。
    bool arbitrage() {
        if (book_.has_side(ASK) && double(book_.best_idx(ASK)) < fair_idx_ - 0.5) { sweep(ASK); return true; }
        if (book_.has_side(BID) && double(book_.best_idx(BID)) > fair_idx_ + 0.5) { sweep(BID); return true; }
        return false;
    }

    // best 水準の注文を全部約定させる
    void sweep(Side s) {
        const i32 idx = book_.best_idx(s);
        auto& q = lvl_[s][static_cast<std::size_t>(idx)];
        // emit_reduce が q から要素を消すので、先頭を食い続ける形にする
        for (int guard = 0; guard < 4096 && !q.empty(); ++guard) {
            const u32 id = q.front();
            const u32 fill = book_.order(id).qty;
            if (fill == 0) break;
            emit_reduce('E', id, fill);
            emit_trade(s == ASK ? BID : ASK, idx, fill);  // ask を食った = 買い主導
        }
    }

    void do_exec(double intensity) {
        // どちらのサイドが叩かれるかは2つの力で決まる:
        //
        //  (1) 裁定 — フェアバリューより安い ask は買われ、高い bid は売られる。
        //      これが無いと板がフェアに追いつかず、mid が数百イベント遅れて動く。
        //      すると OFI は「進行中の遅れ」を読むだけで的中率が96%まで跳ね上がり、
        //      実市場ではありえない数字が出る(実際にそうなった)。
        //      市場参加者は割安な気配を放置しないので、これが構造的に正しい。
        //  (2) ドリフト — 情報を持った参加者の方向性(flow_coupling)。
        //      オーダーフローが将来の値動きを先行する成分はここから出る。
        Side hit;
        const bool ask_cheap = book_.has_side(ASK) && double(book_.best_idx(ASK)) <= fair_idx_;
        const bool bid_rich  = book_.has_side(BID) && double(book_.best_idx(BID)) >= fair_idx_;
        if (ask_cheap && !bid_rich)      hit = ASK;   // 割安な売り気配を買う
        else if (bid_rich && !ask_cheap) hit = BID;   // 割高な買い気配に売る
        else hit = (u01() < 0.5 + cfg_.flow_coupling * drift_) ? ASK : BID;
        if (!book_.has_side(hit)) { do_add(false); return; }
        auto& q = lvl_[hit][static_cast<std::size_t>(book_.best_idx(hit))];
        if (q.empty()) { do_add(false); return; }
        const u32 id = q.front();  // 価格優先・時間優先(FIFO)
        const Order& o = book_.order(id);
        const i32 px = static_cast<i32>(o.px_idx);
        u32 fill;
        if (u01() < 0.6 * (0.5 + 0.5 * intensity)) {
            fill = o.qty;
        } else {
            const u32 units = o.qty / u32(jpx::LOT);
            fill = u32(jpx::LOT) * (1 + u32(rng_() % (units ? units : 1)));
        }
        if (fill > o.qty) fill = o.qty;
        emit_reduce('E', id, fill);
        emit_trade(hit == ASK ? BID : ASK, px, fill);  // ask を叩いた = 買い主導
    }

    // ---- 内部ミラー板の更新 + ワイヤ書き出し ----
    void emit_add(Side s, i32 idx, u32 qty) {
        const u32 id = next_id_++;
        book_.add(id, s, idx, qty);
        lvl_[s][static_cast<std::size_t>(idx)].push_back(id);
        live_pos_.resize(next_id_, UINT32_MAX);
        live_pos_[id] = static_cast<u32>(live_.size());
        live_.push_back(id);
        MsgAdd m{'A', id, static_cast<u8>(s), idx, qty};
        write(&m, sizeof m);
    }

    void emit_reduce(char type, u32 id, u32 dec) {
        const Order o = book_.order(id);
        if (o.qty == 0) return;
        if (dec > o.qty) dec = o.qty;
        book_.reduce(id, dec);
        if (dec >= o.qty) drop_from_indexes(id, static_cast<Side>(o.side), o.px_idx);
        if (type == 'X')      { MsgCancel m{'X', id, dec}; write(&m, sizeof m); }
        else if (type == 'E') { MsgExec m{'E', id, dec};   write(&m, sizeof m); }
        else                  { MsgDelete m{'D', id};      write(&m, sizeof m); }
    }

    void emit_trade(Side aggressor, i32 idx, u32 qty) {
        MsgTrade m{'T', static_cast<u8>(aggressor), idx, qty};
        write(&m, sizeof m);
    }

    void emit_reset() {
        MsgReset m{'R', base_sen_, window_.n_ticks,
                   static_cast<u8>(cfg_.table == jpx::TickTable::Topix100 ? 1 : 0)};
        write(&m, sizeof m);
    }

    void emit_session(jpx::Session s, i32 sec_of_day) {
        MsgSession m{'S', static_cast<u8>(s), sec_of_day};
        write(&m, sizeof m);
    }

    void drop_from_indexes(u32 id, Side s, u32 px_idx) {
        auto& q = lvl_[s][px_idx];
        for (std::size_t i = 0; i < q.size(); ++i)
            if (q[i] == id) { q.erase(q.begin() + static_cast<std::ptrdiff_t>(i)); break; }
        const u32 pos = live_pos_[id];
        if (pos == UINT32_MAX) return;
        live_pos_[live_.back()] = pos;
        live_[pos] = live_.back();
        live_.pop_back();
        live_pos_[id] = UINT32_MAX;
    }

    void write(const void* p, std::size_t n) {
        if (!out_) return;
        const u8* b = static_cast<const u8*>(p);
        out_->insert(out_->end(), b, b + n);
        ++n_msgs_;
    }

    // ---- 補助 ----
    struct BestSnapshot { i32 pb, pa; bool valid; };
    BestSnapshot snapshot() const {
        if (!book_.two_sided()) return {0, 0, false};
        return {book_.best_idx(BID), book_.best_idx(ASK), true};
    }

    i32 clamp_idx(i32 idx) const {
        if (idx < 1) return 1;
        if (idx > window_.n_ticks - 2) return window_.n_ticks - 2;
        return idx;
    }

    double u01() { return std::uniform_real_distribution<double>(0, 1)(rng_); }
    double norm() { return std::normal_distribution<double>(0, 1)(rng_); }
    i32 geometric(double p, i32 cap) {
        i32 k = 0;
        while (k < cap && u01() > p) ++k;
        return k;
    }

    Config cfg_;
    std::mt19937_64 rng_;
    jpx::TickGrid grid_;
    jpx::PriceWindow window_;
    i64 base_sen_ = 0;

    StockBook book_;                              // 生成器自身のミラー板
    std::vector<std::vector<u32>> lvl_[2];        // レベル別FIFO(時間優先の再現用)
    std::vector<u32> live_;                       // 生存注文ID(取消抽選用)
    std::vector<u32> live_pos_;                   // id -> live_ 内位置
    u32 next_id_ = 0;
    double fair_idx_ = 0;
    int drift_ = 0;
    std::vector<u8>* out_ = nullptr;
    std::size_t n_msgs_ = 0;
};

}  // namespace stk
