# wire.py — C++エンジン(feed.hpp)のワイヤ形式エンコーダ + 東証マイクロストラクチャ
#
# 取引所に依存しない層。kabuステーションでも他のデータ源でも、
# 「気配の一覧」を渡せば同じバイナリに変換できる。
#
# ストリーム/収録の共通フレーム: [u64 ts_ns (epoch)] + [ワイヤメッセージ]
#
# ワイヤメッセージ(リトルエンディアン, パック):
#   'A' Add     : u8 type, u32 id, u8 side, i32 idx, u32 qty                 (14B)
#   'X' Cancel  : u8 type, u32 id, u32 dec_qty                               ( 9B)
#   'E' Exec    : u8 type, u32 id, u32 exec_qty                              ( 9B)
#   'D' Delete  : u8 type, u32 id                                            ( 5B)
#   'R' Reset   : u8 type, i64 base_sen, i32 n_ticks, u8 table               (14B)
#   'T' Trade   : u8 type, u8 side, i32 idx, u32 qty                         (10B)
#   'S' Session : u8 type, u8 session, i32 sec_of_day                        ( 6B)
#
# 価格は「窓内インデックス」= ストップ安からの呼値の段数。円との変換は TickGrid が行う。
# jp_market.hpp の C++ 実装と数値が完全に一致していること(--selftest で確認できる)。
import struct
import time

BID, ASK = 0, 1

SEN_PER_YEN = 100
LOT = 100  # 売買単位(2018年10月に全銘柄100株に統一)

_A = struct.Struct("<BIBiI")
_X = struct.Struct("<BII")
_E = struct.Struct("<BII")
_D = struct.Struct("<BI")
_R = struct.Struct("<BqiB")
_T = struct.Struct("<BBiI")
_S = struct.Struct("<BBi")
_TS = struct.Struct("<Q")


def yen_to_sen(yen: float) -> int:
    return int(yen * SEN_PER_YEN + 0.5)


def sen_to_yen(sen: int) -> float:
    return sen / SEN_PER_YEN


# =====================================================================
#  呼値(tick size) — jp_market.hpp の STANDARD_BANDS / TOPIX100_BANDS と同じ表
# =====================================================================
# 各要素は (バンド上限(銭), 呼値(銭))。上限は「この値未満」に適用。
STANDARD_BANDS = [
    (300_000, 100),          # ~3,000円     : 1円
    (500_000, 500),          # ~5,000円     : 5円
    (3_000_000, 1_000),      # ~30,000円    : 10円
    (5_000_000, 5_000),      # ~50,000円    : 50円
    (30_000_000, 10_000),    # ~300,000円   : 100円
    (50_000_000, 50_000),    # ~500,000円   : 500円
    (300_000_000, 100_000),  # ~3,000,000円 : 1,000円
    (500_000_000, 500_000),  # ~5,000,000円 : 5,000円
    (3_000_000_000, 1_000_000),
    (5_000_000_000, 5_000_000),
    (50_000_000_000, 10_000_000),
]

# TOPIX100構成銘柄は「呼値の単位の縮小」対象でより細かい刻みになる
TOPIX100_BANDS = [
    (100_000, 10),           # ~1,000円     : 0.1円
    (300_000, 50),           # ~3,000円     : 0.5円
    (1_000_000, 100),        # ~10,000円    : 1円
    (3_000_000, 500),        # ~30,000円    : 5円
    (10_000_000, 1_000),     # ~100,000円   : 10円
    (30_000_000, 5_000),     # ~300,000円   : 50円
    (100_000_000, 10_000),   # ~1,000,000円 : 100円
    (300_000_000, 50_000),   # ~3,000,000円 : 500円
    (1_000_000_000, 100_000),
    (3_000_000_000, 500_000),
    (50_000_000_000, 1_000_000),
]

TABLE_STANDARD = 0
TABLE_TOPIX100 = 1


class TickGrid:
    """価格(銭) <-> 通しティック番号 の相互変換。jp_market.hpp::TickGrid と同じ。"""

    def __init__(self, table: int = TABLE_STANDARD):
        bands = TOPIX100_BANDS if table == TABLE_TOPIX100 else STANDARD_BANDS
        self.table = table
        self._lo, self._step, self._cum = [], [], []
        lo, cum = 0, 0
        for hi, step in bands:
            self._lo.append(lo)
            self._step.append(step)
            self._cum.append(cum)
            cum += (hi - lo) // step
            lo = hi
        self.n_ticks = cum

    def _band_of_price(self, px_sen: int) -> int:
        i = 0
        while i + 1 < len(self._lo) and px_sen >= self._lo[i + 1]:
            i += 1
        return i

    def _band_of_tick(self, tick: int) -> int:
        i = 0
        while i + 1 < len(self._cum) and tick >= self._cum[i + 1]:
            i += 1
        return i

    def tick_of(self, px_sen: int) -> int:
        px_sen = max(0, px_sen)
        i = self._band_of_price(px_sen)
        return self._cum[i] + (px_sen - self._lo[i]) // self._step[i]

    def price_of(self, tick: int) -> int:
        tick = min(max(0, tick), self.n_ticks - 1)
        i = self._band_of_tick(tick)
        return self._lo[i] + (tick - self._cum[i]) * self._step[i]

    def step_at_price(self, px_sen: int) -> int:
        return self._step[self._band_of_price(px_sen)]

    def floor_to_tick(self, px_sen: int) -> int:
        return self.price_of(self.tick_of(px_sen))

    def ceil_to_tick(self, px_sen: int) -> int:
        f = self.floor_to_tick(px_sen)
        return f if f == px_sen else self.price_of(self.tick_of(px_sen) + 1)


# =====================================================================
#  値幅制限(ストップ高・ストップ安) — jp_market.hpp::LIMIT_BANDS と同じ表
# =====================================================================
# (基準値段の上限(銭), 値幅(銭))
LIMIT_BANDS = [
    (10_000, 3_000), (20_000, 5_000), (50_000, 8_000), (70_000, 10_000),
    (100_000, 15_000), (150_000, 30_000), (200_000, 40_000), (300_000, 50_000),
    (500_000, 70_000), (700_000, 100_000), (1_000_000, 150_000), (1_500_000, 300_000),
    (2_000_000, 400_000), (3_000_000, 500_000), (5_000_000, 700_000), (7_000_000, 1_000_000),
    (10_000_000, 1_500_000), (15_000_000, 3_000_000), (20_000_000, 4_000_000),
    (30_000_000, 5_000_000), (50_000_000, 7_000_000), (70_000_000, 10_000_000),
    (100_000_000, 15_000_000), (150_000_000, 30_000_000), (200_000_000, 40_000_000),
    (300_000_000, 50_000_000), (500_000_000, 70_000_000), (700_000_000, 100_000_000),
    (1_000_000_000, 150_000_000), (1_500_000_000, 300_000_000), (2_000_000_000, 400_000_000),
    (3_000_000_000, 500_000_000), (5_000_000_000, 700_000_000), (50_000_000_000, 1_000_000_000),
]


def daily_limit_sen(base_sen: int) -> int:
    for hi, width in LIMIT_BANDS:
        if base_sen < hi:
            return width
    return LIMIT_BANDS[-1][1]


class PriceWindow:
    """当日の価格窓。ストップ安を窓内インデックス0に置く。"""

    def __init__(self, grid: TickGrid, base_sen: int):
        w = daily_limit_sen(base_sen)
        lo = base_sen - w
        self.grid = grid
        self.base_sen = base_sen
        self.low_sen = grid.ceil_to_tick(lo if lo > 0 else grid.step_at_price(1))
        self.high_sen = grid.floor_to_tick(base_sen + w)
        self.low_tick = grid.tick_of(self.low_sen)
        self.n_ticks = grid.tick_of(self.high_sen) - self.low_tick + 1

    def idx_of_sen(self, px_sen: int) -> int:
        return self.grid.tick_of(px_sen) - self.low_tick

    def sen_of_idx(self, idx: int) -> int:
        return self.grid.price_of(self.low_tick + idx)

    def contains(self, px_sen: int) -> bool:
        return self.low_sen <= px_sen <= self.high_sen


# =====================================================================
#  セッション — jp_market.hpp::Session と同じ順序・同じ時刻
# =====================================================================
(PRE_OPEN, MORNING_AUCTION, MORNING, MORNING_CLOSE, LUNCH,
 AFTERNOON_AUCTION, AFTERNOON, CLOSING_AUCTION, CLOSE, OFF) = range(10)

SESSION_NAMES = [
    "PreOpen", "MorningAuction", "Morning", "MorningClose", "Lunch",
    "AfternoonAuction", "Afternoon", "ClosingAuction", "Close", "Off",
]

T_PREOPEN = 8 * 60
T_OPEN = 9 * 60
T_AM_CLOSE = 11 * 60 + 30
T_PM_OPEN = 12 * 60 + 30
T_CA_START = 15 * 60 + 25
T_CLOSE = 15 * 60 + 30


def session_at(ms_of_day: int) -> int:
    """0時からの経過ミリ秒 -> セッション。jp_market.hpp::session_at と同じ判定。"""
    m = ms_of_day // 60_000
    at_edge = (ms_of_day % 60_000) < 1_000
    if m < T_PREOPEN:
        return OFF
    if m < T_OPEN:
        return PRE_OPEN
    if m == T_OPEN and at_edge:
        return MORNING_AUCTION
    if m < T_AM_CLOSE:
        return MORNING
    if m == T_AM_CLOSE and at_edge:
        return MORNING_CLOSE
    if m < T_PM_OPEN:
        return LUNCH
    if m == T_PM_OPEN and at_edge:
        return AFTERNOON_AUCTION
    if m < T_CA_START:
        return AFTERNOON
    if m < T_CLOSE:
        return CLOSING_AUCTION
    if m == T_CLOSE and at_edge:
        return CLOSE
    return OFF


def is_continuous(sess: int) -> bool:
    return sess in (MORNING, AFTERNOON)


# =====================================================================
#  エンコーダ
# =====================================================================
def _frame(payload: bytes, ts_ns) -> bytes:
    return _TS.pack(time.time_ns() if ts_ns is None else ts_ns) + payload


def add(oid: int, side: int, idx: int, qty: int, ts_ns=None) -> bytes:
    return _frame(_A.pack(ord("A"), oid, side, idx, qty), ts_ns)


def cancel(oid: int, dec: int, ts_ns=None) -> bytes:
    return _frame(_X.pack(ord("X"), oid, dec), ts_ns)


def execute(oid: int, qty: int, ts_ns=None) -> bytes:
    return _frame(_E.pack(ord("E"), oid, qty), ts_ns)


def delete(oid: int, ts_ns=None) -> bytes:
    return _frame(_D.pack(ord("D"), oid), ts_ns)


def reset(base_sen: int, n_ticks: int, table: int = TABLE_STANDARD, ts_ns=None) -> bytes:
    return _frame(_R.pack(ord("R"), base_sen, n_ticks, table), ts_ns)


def trade(side: int, idx: int, qty: int, ts_ns=None) -> bytes:
    return _frame(_T.pack(ord("T"), side, idx, qty), ts_ns)


def session(sess: int, sec_of_day: int, ts_ns=None) -> bytes:
    return _frame(_S.pack(ord("S"), sess, sec_of_day), ts_ns)


# =====================================================================
#  板の差分エンジン
# =====================================================================
class BookDiff:
    """気配のスナップショットを受け取り、合成注文(A/X/D)の差分に変換する。

    kabuステーションAPIは「今の板の10本値」をPushしてくるだけで、
    「どの注文が増えた/減った」という情報は来ない。エンジン側は注文単位の
    イベント列を期待しているので、ここでレベルごとの数量差分を
    「1レベル = 1本の合成注文」に見立てて A/X/D に翻訳する。

    このやり方だと個別注文の時間優先は再現できないが、OFI や板の厚みといった
    レベル単位の指標には影響しない(OFI は best の価格と数量しか見ない)。
    """

    def __init__(self, window: PriceWindow):
        self.window = window
        self.next_id = 0
        # (side, idx) -> [[order_id, qty], ...]  1レベルに複数の合成注文を持つ
        self.level = {}

    def apply(self, side: int, idx: int, qty: int, ts_ns=None) -> list:
        """あるレベルの合計数量を qty にする差分メッセージを返す。

        数量が増えたときは「新しい注文が列の後ろに加わった」として増分だけを A で出す。
        レベルを D してから作り直してはいけない: best の水準が一瞬空になるので
        エンジン側の best bid/ask が下がって戻る往復を見てしまい、OFI が打ち消し合って
        分散だけが膨らみ、z-score が ±0.1 程度に潰れてシグナルが出なくなる
        (開発中に実測。実フィードでも数量増加は新規注文として現れる)。
        """
        out = []
        key = (side, idx)
        orders = self.level.get(key, [])
        total = sum(q for _, q in orders)

        if qty <= 0:
            for oid, _ in orders:
                out.append(delete(oid, ts_ns))
            self.level.pop(key, None)
            return out

        if qty > total:
            oid = self.next_id
            self.next_id += 1
            orders = orders + [[oid, qty - total]]
            out.append(add(oid, side, idx, qty - total, ts_ns))
            self.level[key] = orders
        elif qty < total:
            # 減少は新しい注文から取り消す(既存の注文IDを保てるので板が安定する)
            need = total - qty
            while need > 0 and orders:
                oid, q = orders[-1]
                if q <= need:
                    out.append(delete(oid, ts_ns))
                    need -= q
                    orders.pop()
                else:
                    out.append(cancel(oid, need, ts_ns))
                    orders[-1][1] = q - need
                    need = 0
            if orders:
                self.level[key] = orders
            else:
                self.level.pop(key, None)
        return out

    def snapshot(self, bids: list, asks: list, ts_ns=None) -> bytes:
        """気配一覧 [(価格(銭), 株数), ...] を現在の板に反映する差分を返す。

        渡されなかったレベルは「消えた」とみなして削除する。
        """
        out = []
        want = {}
        for side, levels in ((BID, bids), (ASK, asks)):
            for px_sen, qty in levels:
                if not self.window.contains(px_sen):
                    continue  # 値幅制限の外は取引所が受け付けないので無視
                want[(side, self.window.idx_of_sen(px_sen))] = qty
        # 消えたレベル
        for key in list(self.level.keys()):
            if key not in want:
                out.extend(self.apply(key[0], key[1], 0, ts_ns))
        # 増減したレベル
        for (side, idx), qty in want.items():
            out.extend(self.apply(side, idx, qty, ts_ns))
        return b"".join(out)

    def clear(self):
        self.level.clear()


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

    # 呼値
    g = TickGrid(TABLE_STANDARD)
    check(g.step_at_price(yen_to_sen(800)) == 100, "通常800円の呼値=1円")
    check(g.step_at_price(yen_to_sen(2999)) == 100, "通常2999円の呼値=1円")
    check(g.step_at_price(yen_to_sen(4000)) == 500, "通常4000円の呼値=5円")
    check(g.step_at_price(yen_to_sen(12000)) == 1000, "通常12000円の呼値=10円")
    gt = TickGrid(TABLE_TOPIX100)
    check(gt.step_at_price(yen_to_sen(800)) == 10, "TOPIX100 800円の呼値=0.1円")
    check(gt.step_at_price(yen_to_sen(2000)) == 50, "TOPIX100 2000円の呼値=0.5円")

    # 往復
    for table, yens in ((TABLE_STANDARD, [100, 800, 2999, 3000, 4000, 12000, 40000]),
                        (TABLE_TOPIX100, [100.5, 800.1, 2000.5, 8000, 20000])):
        gg = TickGrid(table)
        for y in yens:
            s = yen_to_sen(y)
            check(gg.price_of(gg.tick_of(s)) == s, f"往復 table={table} {y}円")
            check(gg.price_of(gg.tick_of(s) + 1) - s == gg.step_at_price(s),
                  f"+1tick table={table} {y}円")

    # 値幅制限と価格窓
    w = PriceWindow(TickGrid(), yen_to_sen(2500))
    check(w.low_sen == yen_to_sen(2000), "2500円の値幅下限=2000円")
    check(w.high_sen == yen_to_sen(3000), "2500円の値幅上限=3000円")
    check(w.n_ticks == 1001, f"2500円の窓=1001本 (実際 {w.n_ticks})")
    for i in range(0, w.n_ticks, 37):
        check(w.idx_of_sen(w.sen_of_idx(i)) == i, f"窓内往復 idx={i}")

    # セッション
    check(session_at((9 * 60 + 30) * 60_000) == MORNING, "09:30 = 前場")
    check(session_at((12 * 60) * 60_000) == LUNCH, "12:00 = 昼休み")
    check(session_at((13 * 60) * 60_000) == AFTERNOON, "13:00 = 後場")
    check(session_at((15 * 60 + 27) * 60_000) == CLOSING_AUCTION, "15:27 = クロージング")
    check(session_at((16 * 60) * 60_000) == OFF, "16:00 = 時間外")
    check(is_continuous(MORNING) and is_continuous(AFTERNOON), "ザラ場判定")
    check(not is_continuous(LUNCH) and not is_continuous(MORNING_AUCTION), "非ザラ場判定")

    # フレーム長
    check(len(add(0, BID, 0, 100)) == 8 + 14, "A のフレーム長")
    check(len(cancel(0, 100)) == 8 + 9, "X のフレーム長")
    check(len(delete(0)) == 8 + 5, "D のフレーム長")
    check(len(reset(0, 1, 0)) == 8 + 14, "R のフレーム長")
    check(len(trade(BID, 0, 100)) == 8 + 10, "T のフレーム長")
    check(len(session(MORNING, 0)) == 8 + 6, "S のフレーム長")

    # BookDiff: 数量の増減が A/X/D に正しく落ちるか
    bd = BookDiff(w)
    msgs = bd.snapshot([(yen_to_sen(2499), 300)], [(yen_to_sen(2500), 200)])
    check(len(msgs) == 2 * (8 + 14), "初回スナップショットは A 2本")
    msgs = bd.snapshot([(yen_to_sen(2499), 100)], [(yen_to_sen(2500), 200)])
    check(len(msgs) == 8 + 9, "数量減は X 1本")
    # 数量増は「増分だけの A 1本」。D+A の往復にしてはいけない(OFIが潰れる)
    msgs = bd.snapshot([(yen_to_sen(2499), 400)], [(yen_to_sen(2500), 200)])
    check(len(msgs) == 8 + 14, f"数量増は A 1本のみ (実際 {len(msgs)} バイト)")
    check(bd.level[(BID, w.idx_of_sen(yen_to_sen(2499)))][0][1] == 100,
          "元の注文の数量が保たれている")
    msgs = bd.snapshot([(yen_to_sen(2499), 100)], [(yen_to_sen(2500), 200)])
    check(len(msgs) == 8 + 5, "増分ぶんの減少は D 1本(増分の注文が消える)")
    msgs = bd.snapshot([(yen_to_sen(2499), 100)], [])
    check(len(msgs) == 8 + 5, "レベル消滅は D 1本")
    # 合計数量が板と一致し続けること
    bd2 = BookDiff(w)
    total = 0
    for q in (500, 700, 300, 1200, 100, 900):
        bd2.snapshot([(yen_to_sen(2499), q)], [])
        total = sum(x for _, x in bd2.level[(BID, w.idx_of_sen(yen_to_sen(2499)))])
        if total != q:
            break
    check(total == 900, f"増減を繰り返しても合計が一致 (実際 {total})")

    # 注意: Windowsのコンソールは cp932 なので、出力する文字列に
    # em dash(—) や ✓ のような cp932 に無い文字を使うと UnicodeEncodeError で落ちる。
    # ソース中のコメントはUTF-8で読まれるだけなので問題ない。
    print(f"selftest: {'OK - 全項目一致' if fails == 0 else f'NG - {fails} 件失敗'}")
    return fails


if __name__ == "__main__":
    import sys
    sys.exit(1 if selftest() else 0)
