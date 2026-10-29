#pragma once
// jp_market.hpp — 東証(TSE)のマイクロストラクチャ定義
//
// 株版エンジンの核心。暗号資産(hft-engine)と日本株の決定的な違いはここに集約される:
//
//  1. 呼値(よびね)が価格帯で変わる。BTCの「1円=1ティック」固定は使えない。
//     板エンジンの速さは「価格→配列添字」が足し算1回で済むことに依存するので、
//     ここで 円 <-> ティック番号 の相互変換を用意し、板本体はティック番号だけを扱う。
//  2. 値幅制限(ストップ高安)が当日の価格範囲を物理的に閉じる。
//     これがそのまま板の価格窓になる。BTC版の1,048,576ティックに対し株は数千で足り、
//     板全体がL2キャッシュに載るぶん株版の方が速い。
//  3. 板寄せ/ザラ場/引けというセッション構造がある。時間帯によって板の性質が変わる。
//
// 価格の内部表現は「銭(せん) = 0.01円」の整数。TOPIX100銘柄の呼値0.1円/0.5円を
// 浮動小数点なしで扱うため。円との変換は yen_to_sen() / sen_to_yen().
#include <cstdint>
#include <cstddef>

// ---- Windowsコンソールを UTF-8 表示にする(日本語出力の文字化け防止) ----
// windows.h はマクロ汚染が大きいので、使う関数1つだけを宣言する
#ifdef _WIN32
extern "C" __declspec(dllimport) int __stdcall SetConsoleOutputCP(unsigned int);
namespace jpx::detail {
inline const bool console_utf8 = (SetConsoleOutputCP(65001), true);
}
#endif

namespace jpx {

using i32 = std::int32_t;
using i64 = std::int64_t;

// ---- 価格単位 ----
inline constexpr i64 SEN_PER_YEN = 100;
inline constexpr i64 yen_to_sen(double yen) { return static_cast<i64>(yen * SEN_PER_YEN + 0.5); }
inline constexpr double sen_to_yen(i64 sen) { return static_cast<double>(sen) / SEN_PER_YEN; }

// ---- 売買単位 ----
// 2018年10月に全上場銘柄が100株単位に統一された。
inline constexpr i32 LOT = 100;

// =====================================================================
//  呼値(tick size)
// =====================================================================
// 東証の呼値は価格帯ごとの階段。TOPIX100構成銘柄は「呼値の単位の縮小」対象で
// より細かい刻みが適用される(流動性が高いためスプレッドを詰められる)。
//
// 各バンドは [lo, hi) 銭で、刻み step 銭。バンド境界は必ず step の倍数なので
// バンド内のティック本数は整数になる。

struct TickBand {
    i64 hi_sen;    // このバンドの上限(この値未満に適用)。最終バンドは番兵
    i64 step_sen;  // 呼値(銭)
};

enum class TickTable { Standard, Topix100 };

// 通常銘柄: 3,000円以下=1円、5,000円以下=5円、30,000円以下=10円 …
inline constexpr TickBand STANDARD_BANDS[] = {
    {       300'000,        100},  // ~3,000円        : 1円
    {       500'000,        500},  // ~5,000円        : 5円
    {     3'000'000,      1'000},  // ~30,000円       : 10円
    {     5'000'000,      5'000},  // ~50,000円       : 50円
    {    30'000'000,     10'000},  // ~300,000円      : 100円
    {    50'000'000,     50'000},  // ~500,000円      : 500円
    {   300'000'000,    100'000},  // ~3,000,000円    : 1,000円
    {   500'000'000,    500'000},  // ~5,000,000円    : 5,000円
    { 3'000'000'000,  1'000'000},  // ~30,000,000円   : 10,000円
    { 5'000'000'000,  5'000'000},  // ~50,000,000円   : 50,000円
    {50'000'000'000, 10'000'000},  // それ以上        : 100,000円 (番兵)
};

// TOPIX100(呼値縮小): 1,000円以下=0.1円、3,000円以下=0.5円、10,000円以下=1円 …
inline constexpr TickBand TOPIX100_BANDS[] = {
    {       100'000,         10},  // ~1,000円        : 0.1円
    {       300'000,         50},  // ~3,000円        : 0.5円
    {     1'000'000,        100},  // ~10,000円       : 1円
    {     3'000'000,        500},  // ~30,000円       : 5円
    {    10'000'000,      1'000},  // ~100,000円      : 10円
    {    30'000'000,      5'000},  // ~300,000円      : 50円
    {   100'000'000,     10'000},  // ~1,000,000円    : 100円
    {   300'000'000,     50'000},  // ~3,000,000円    : 500円
    { 1'000'000'000,    100'000},  // ~10,000,000円   : 1,000円
    { 3'000'000'000,    500'000},  // ~30,000,000円   : 5,000円
    {50'000'000'000,  1'000'000},  // それ以上        : 10,000円 (番兵)
};

inline constexpr std::size_t N_BANDS = sizeof(STANDARD_BANDS) / sizeof(TickBand);
static_assert(sizeof(TOPIX100_BANDS) / sizeof(TickBand) == N_BANDS);

// 呼値グリッド: 価格(銭) <-> 通し番号のティック番号 を相互変換する。
//
// ティック番号は「価格0からその価格まで、呼値階段を何段登ったか」。
// バンド数は11しかないので走査はキャッシュに全部載る線形ループ。実際の売買は
// 1銘柄1日で1バンド内に収まる(値幅制限があるため)ので分岐予測はほぼ常に当たる。
class TickGrid {
public:
    explicit TickGrid(TickTable table = TickTable::Standard) {
        const TickBand* b = (table == TickTable::Topix100) ? TOPIX100_BANDS : STANDARD_BANDS;
        i64 lo = 0, cum = 0;
        for (std::size_t i = 0; i < N_BANDS; ++i) {
            lo_[i] = lo;
            step_[i] = b[i].step_sen;
            cum_[i] = cum;
            cum += (b[i].hi_sen - lo) / b[i].step_sen;
            lo = b[i].hi_sen;
        }
        n_ticks_ = cum;
    }

    // 価格(銭) -> ティック番号。呼値に乗っていない価格は切り下げる。
    i64 tick_of(i64 px_sen) const {
        if (px_sen < 0) px_sen = 0;
        const std::size_t i = band_of_price(px_sen);
        return cum_[i] + (px_sen - lo_[i]) / step_[i];
    }

    // ティック番号 -> 価格(銭)
    i64 price_of(i64 tick) const {
        if (tick < 0) tick = 0;
        if (tick >= n_ticks_) tick = n_ticks_ - 1;
        const std::size_t i = band_of_tick(tick);
        return lo_[i] + (tick - cum_[i]) * step_[i];
    }

    // その価格帯の呼値(銭)。1ティック動いたときの円換算に使う。
    i64 step_at_price(i64 px_sen) const { return step_[band_of_price(px_sen)]; }
    i64 step_at_tick(i64 tick) const { return step_[band_of_tick(tick)]; }

    // 呼値に乗るよう切り下げ/切り上げした価格(銭)
    i64 floor_to_tick(i64 px_sen) const { return price_of(tick_of(px_sen)); }
    i64 ceil_to_tick(i64 px_sen) const {
        const i64 f = floor_to_tick(px_sen);
        return f == px_sen ? f : price_of(tick_of(px_sen) + 1);
    }

    i64 n_ticks() const { return n_ticks_; }

private:
    std::size_t band_of_price(i64 px_sen) const {
        std::size_t i = 0;
        while (i + 1 < N_BANDS && px_sen >= lo_[i + 1]) ++i;
        return i;
    }
    std::size_t band_of_tick(i64 tick) const {
        std::size_t i = 0;
        while (i + 1 < N_BANDS && tick >= cum_[i + 1]) ++i;
        return i;
    }

    i64 lo_[N_BANDS]{}, step_[N_BANDS]{}, cum_[N_BANDS]{};
    i64 n_ticks_ = 0;
};

// =====================================================================
//  値幅制限(ストップ高・ストップ安)
// =====================================================================
// 前日終値(基準値段)に応じて当日動ける値幅が決まる。この幅が板の価格窓になる。

struct LimitBand {
    i64 base_hi_sen;   // 基準値段がこの値未満なら
    i64 width_sen;     // 上下この幅まで
};

inline constexpr LimitBand LIMIT_BANDS[] = {
    {        10'000,        3'000},  // 100円未満        : ±30円
    {        20'000,        5'000},  // 200円未満        : ±50円
    {        50'000,        8'000},  // 500円未満        : ±80円
    {        70'000,       10'000},  // 700円未満        : ±100円
    {       100'000,       15'000},  // 1,000円未満      : ±150円
    {       150'000,       30'000},  // 1,500円未満      : ±300円
    {       200'000,       40'000},  // 2,000円未満      : ±400円
    {       300'000,       50'000},  // 3,000円未満      : ±500円
    {       500'000,       70'000},  // 5,000円未満      : ±700円
    {       700'000,      100'000},  // 7,000円未満      : ±1,000円
    {     1'000'000,      150'000},  // 10,000円未満     : ±1,500円
    {     1'500'000,      300'000},  // 15,000円未満     : ±3,000円
    {     2'000'000,      400'000},  // 20,000円未満     : ±4,000円
    {     3'000'000,      500'000},  // 30,000円未満     : ±5,000円
    {     5'000'000,      700'000},  // 50,000円未満     : ±7,000円
    {     7'000'000,    1'000'000},  // 70,000円未満     : ±10,000円
    {    10'000'000,    1'500'000},  // 100,000円未満    : ±15,000円
    {    15'000'000,    3'000'000},  // 150,000円未満    : ±30,000円
    {    20'000'000,    4'000'000},  // 200,000円未満    : ±40,000円
    {    30'000'000,    5'000'000},  // 300,000円未満    : ±50,000円
    {    50'000'000,    7'000'000},  // 500,000円未満    : ±70,000円
    {    70'000'000,   10'000'000},  // 700,000円未満    : ±100,000円
    {   100'000'000,   15'000'000},  // 1,000,000円未満  : ±150,000円
    {   150'000'000,   30'000'000},  // 1,500,000円未満  : ±300,000円
    {   200'000'000,   40'000'000},  // 2,000,000円未満  : ±400,000円
    {   300'000'000,   50'000'000},  // 3,000,000円未満  : ±500,000円
    {   500'000'000,   70'000'000},  // 5,000,000円未満  : ±700,000円
    {   700'000'000,  100'000'000},  // 7,000,000円未満  : ±1,000,000円
    { 1'000'000'000,  150'000'000},  // 10,000,000円未満 : ±1,500,000円
    { 1'500'000'000,  300'000'000},  // 15,000,000円未満 : ±3,000,000円
    { 2'000'000'000,  400'000'000},  // 20,000,000円未満 : ±4,000,000円
    { 3'000'000'000,  500'000'000},  // 30,000,000円未満 : ±5,000,000円
    { 5'000'000'000,  700'000'000},  // 50,000,000円未満 : ±7,000,000円
    {50'000'000'000, 1'000'000'000}, // それ以上         : ±10,000,000円 (番兵)
};
inline constexpr std::size_t N_LIMIT_BANDS = sizeof(LIMIT_BANDS) / sizeof(LimitBand);

// 基準値段(銭)に対する当日の値幅(銭)
inline i64 daily_limit_sen(i64 base_sen) {
    for (std::size_t i = 0; i < N_LIMIT_BANDS; ++i)
        if (base_sen < LIMIT_BANDS[i].base_hi_sen) return LIMIT_BANDS[i].width_sen;
    return LIMIT_BANDS[N_LIMIT_BANDS - 1].width_sen;
}

// 当日の価格窓。ストップ安を窓の下端(ティック番号0)に置く。
//
// 板エンジンはこの窓の中だけを配列で持つ。ストップ高安を外れる注文は
// そもそも取引所が受け付けないので、窓外を扱う必要がない ---
// これが暗号資産に対する株の構造的な利点。
struct PriceWindow {
    i64 low_sen = 0;      // ストップ安(銭)
    i64 high_sen = 0;     // ストップ高(銭)
    i64 low_tick = 0;     // ストップ安の通しティック番号
    i32 n_ticks = 0;      // 窓の幅(ティック本数)
};

inline PriceWindow make_window(const TickGrid& g, i64 base_sen) {
    const i64 w = daily_limit_sen(base_sen);
    PriceWindow pw;
    pw.low_sen = g.ceil_to_tick(base_sen - w > 0 ? base_sen - w : g.step_at_price(1));
    pw.high_sen = g.floor_to_tick(base_sen + w);
    pw.low_tick = g.tick_of(pw.low_sen);
    pw.n_ticks = static_cast<i32>(g.tick_of(pw.high_sen) - pw.low_tick + 1);
    return pw;
}

// 窓内インデックス <-> 価格(銭)。板エンジンが扱うのは常に窓内インデックス。
inline i32 sen_to_idx(const TickGrid& g, const PriceWindow& w, i64 px_sen) {
    return static_cast<i32>(g.tick_of(px_sen) - w.low_tick);
}
inline i64 idx_to_sen(const TickGrid& g, const PriceWindow& w, i32 idx) {
    return g.price_of(w.low_tick + idx);
}

// =====================================================================
//  立会時間とセッション (2024-11-05 の取引時間延長後)
// =====================================================================
//   08:00-09:00  寄前注文受付 (板寄せの気配を形成)
//   09:00        寄付板寄せ  -> 前場ザラ場開始
//   09:00-11:30  前場ザラ場
//   11:30        前引け板寄せ
//   11:30-12:30  昼休み (12:05 から後場の注文受付)
//   12:30        後場寄り板寄せ -> 後場ザラ場開始
//   12:30-15:25  後場ザラ場
//   15:25-15:30  クロージング・オークション (注文受付のみ、約定なし)
//   15:30        引け板寄せ

enum class Session : std::uint8_t {
    PreOpen = 0,      // 寄前注文受付
    MorningAuction,   // 09:00 寄付板寄せ
    Morning,          // 前場ザラ場
    MorningClose,     // 11:30 前引け板寄せ
    Lunch,            // 昼休み
    AfternoonAuction, // 12:30 後場寄り板寄せ
    Afternoon,        // 後場ザラ場
    ClosingAuction,   // 15:25-15:30 クロージング・オークション
    Close,            // 15:30 引け板寄せ
    Off,              // 立会時間外
};

inline constexpr const char* SESSION_NAMES[] = {
    "PreOpen", "MorningAuction", "Morning", "MorningClose", "Lunch",
    "AfternoonAuction", "Afternoon", "ClosingAuction", "Close", "Off",
};

inline constexpr i32 hm(i32 h, i32 m) { return h * 60 + m; }

// 立会日の各節目(0時からの分)
inline constexpr i32 T_PREOPEN  = hm( 8,  0);
inline constexpr i32 T_OPEN     = hm( 9,  0);
inline constexpr i32 T_AM_CLOSE = hm(11, 30);
inline constexpr i32 T_PM_OPEN  = hm(12, 30);
inline constexpr i32 T_CA_START = hm(15, 25);
inline constexpr i32 T_CLOSE    = hm(15, 30);

// 0時からの経過ミリ秒 -> セッション。板寄せの瞬間だけを別扱いにするため、
// 節目ちょうど(その分の先頭1秒)を板寄せ扱いにしている。
inline Session session_at(i64 ms_of_day) {
    const i32 min_of_day = static_cast<i32>(ms_of_day / 60'000);
    const bool at_edge = (ms_of_day % 60'000) < 1'000;
    if (min_of_day < T_PREOPEN)  return Session::Off;
    if (min_of_day < T_OPEN)     return Session::PreOpen;
    if (min_of_day == T_OPEN && at_edge)     return Session::MorningAuction;
    if (min_of_day < T_AM_CLOSE) return Session::Morning;
    if (min_of_day == T_AM_CLOSE && at_edge) return Session::MorningClose;
    if (min_of_day < T_PM_OPEN)  return Session::Lunch;
    if (min_of_day == T_PM_OPEN && at_edge)  return Session::AfternoonAuction;
    if (min_of_day < T_CA_START) return Session::Afternoon;
    if (min_of_day < T_CLOSE)    return Session::ClosingAuction;
    if (min_of_day == T_CLOSE && at_edge)    return Session::Close;
    return Session::Off;
}

// ザラ場(連続約定)かどうか。シグナルを出してよいのはここだけ。
inline bool is_continuous(Session s) {
    return s == Session::Morning || s == Session::Afternoon;
}

// 立会中の進捗 0.0(寄り) ～ 1.0(引け)。日中U字を作るときと、
// ML特徴量 "tod" (time of day) に使う。ザラ場合計は 150分 + 175分 = 325分。
inline constexpr double TRADING_MINUTES = double((T_AM_CLOSE - T_OPEN) + (T_CA_START - T_PM_OPEN));

inline double day_progress(i64 ms_of_day) {
    const double m = double(ms_of_day) / 60'000.0;
    double elapsed;
    if (m <= T_OPEN)          elapsed = 0.0;
    else if (m < T_AM_CLOSE)  elapsed = m - T_OPEN;
    else if (m < T_PM_OPEN)   elapsed = T_AM_CLOSE - T_OPEN;
    else if (m < T_CA_START)  elapsed = (T_AM_CLOSE - T_OPEN) + (m - T_PM_OPEN);
    else                      elapsed = TRADING_MINUTES;
    return elapsed / TRADING_MINUTES;
}

// 日中U字の強度。寄り付きと引けで活動が跳ね、昼前後で最も凪ぐ。
// 実証的に出来高・ボラ・スプレッドすべてがこの形をとる。
inline double u_shape(double progress) {
    const double x = 2.0 * progress - 1.0;   // -1(寄り) .. +1(引け)
    return 0.35 + 1.30 * x * x;
}

}  // namespace jpx
