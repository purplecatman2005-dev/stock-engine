# -*- coding: utf-8 -*-
"""kabu_adapter.py - kabuステーションAPI -> ワイヤ形式 -> live.exe のブリッジ

kabuステーションAPI(auカブコム証券)は個人が日本株の板情報(気配10本値)を
リアルタイムで取れる実質唯一の無料経路。ただし証券口座 + kabuステーションアプリ +
API利用申請が必要なので、口座が無いうちは --mock で合成板を流して全経路を検証する。

  口座なし(今すぐ動く):
    python kabu_adapter.py --mock --speed 120
        1立会日を120倍速で再生する。板寄せ -> ザラ場 -> 昼休み -> 引けが1分程度で流れる。

  口座取得後(本番):
    python kabu_adapter.py --symbol 7203 --password <APIパスワード>
        --mock を外すだけ。C++側は一切変更しない。

  ネット・口座なしの検証:
    python kabu_adapter.py --selftest

--------------------------------------------------------------------------
本番経路(kabuステーションが localhost:18080 で待ち受けている前提):

  1. POST http://localhost:18080/kabusapi/token
       {"APIPassword": "..."}                      -> {"ResultCode":0, "Token":"..."}
  2. GET  /kabusapi/symbol/{symbol}@{exchange}
       ヘッダ X-API-KEY: <token>                    -> PreviousClose / UpperLimit / LowerLimit
  3. PUT  /kabusapi/register
       {"Symbols":[{"Symbol":"7203","Exchange":1}]} -> 銘柄登録
  4. WS   ws://localhost:18080/kabusapi/websocket
       登録した銘柄の板が変化するたびにJSONがPushされる
         Buy1..Buy10 / Sell1..Sell10 = {"Price": 2500.0, "Qty": 300, ...}
         CurrentPrice / TradingVolume / BidPrice / AskPrice ...

JSONの解釈から下流(ワイヤ変換 -> TCP -> live.exe -> 収録 -> replay.exe)は
--mock と本番で完全に同じコードを通る。--mock で検証できないのは上記1-4の
HTTP/WSプラミング部分だけ。
--------------------------------------------------------------------------
"""
import argparse
import json
import random
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wire  # noqa: E402

KABU_HOST = "localhost"
KABU_PORT_LIVE = 18080   # 本番
KABU_PORT_TEST = 18081   # 検証環境
EXCHANGE_TOSHO = 1       # 東証


# =====================================================================
#  live.exe への TCP 送信(切断しても待ち続ける)
# =====================================================================
class EngineLink:
    def __init__(self, host="127.0.0.1", port=9501, quiet=False):
        self.addr = (host, port)
        self.sock = None
        self.quiet = quiet
        self.sent_bytes = 0
        self.on_reconnect = None   # 再接続時に板を作り直すためのコールバック

    def _log(self, msg):
        if not self.quiet:
            print(msg, flush=True)

    def connect(self, retry_sec=2.0):
        while self.sock is None:
            try:
                s = socket.create_connection(self.addr, timeout=5)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock = s
                self._log(f"[link] live.exe に接続 {self.addr[0]}:{self.addr[1]}")
            except OSError as e:
                self._log(f"[link] 接続待ち ({e}) - live.exe を起動してください")
                time.sleep(retry_sec)

    def send(self, data: bytes):
        if not data:
            return
        while True:
            if self.sock is None:
                self.connect()
                if self.on_reconnect:
                    self.on_reconnect()
                # 再接続時は板を作り直すので、今回のフレームは捨てて次から送る
                return
            try:
                self.sock.sendall(data)
                self.sent_bytes += len(data)
                return
            except OSError as e:
                self._log(f"[link] 送信失敗 ({e}) - 再接続します")
                try:
                    self.sock.close()
                except OSError:
                    pass
                self.sock = None

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None


# =====================================================================
#  板JSON -> ワイヤ形式
# =====================================================================
class BoardAdapter:
    """kabuステーションの板JSONを1件受け取り、ワイヤメッセージに変換する。

    本番と --mock で共通に使う唯一の変換器。ここが正しければ、
    データ源が何であってもエンジン側は同じものを受け取る。
    """

    def __init__(self, link: EngineLink, table=wire.TABLE_STANDARD, quiet=False):
        self.link = link
        self.grid = wire.TickGrid(table)
        self.table = table
        self.window = None
        self.diff = None
        self.session = None
        self.last_volume = 0
        self.last_price_sen = None
        self.n_boards = 0
        self.n_trades = 0
        self.out_of_window = 0
        self.quiet = quiet
        link.on_reconnect = self._invalidate

    def _log(self, msg):
        if not self.quiet:
            print(msg, flush=True)

    def _invalidate(self):
        """live.exe が再接続したら板をゼロから作り直す。"""
        self.window = None
        self.diff = None
        self.session = None
        self.last_volume = 0

    # ---- 当日の価格窓を張る ----
    def anchor(self, base_yen: float, ts_ns=None):
        base_sen = wire.yen_to_sen(base_yen)
        self.window = wire.PriceWindow(self.grid, base_sen)
        self.diff = wire.BookDiff(self.window)
        self.link.send(wire.reset(base_sen, self.window.n_ticks, self.table, ts_ns))
        self._log(
            f"[anchor] 基準値段 {base_yen:.1f}円 -> 窓 "
            f"[{wire.sen_to_yen(self.window.low_sen):.1f}, "
            f"{wire.sen_to_yen(self.window.high_sen):.1f}]円 "
            f"{self.window.n_ticks}本 (呼値 {wire.sen_to_yen(self.grid.step_at_price(base_sen)):.2f}円)"
        )

    # ---- セッション遷移 ----
    def update_session(self, ms_of_day: int, ts_ns=None):
        sess = wire.session_at(ms_of_day)
        if sess != self.session:
            self.session = sess
            self.link.send(wire.session(sess, ms_of_day // 1000, ts_ns))
            self._log(f"[session] {wire.SESSION_NAMES[sess]}")

    # ---- 板1件 ----
    def on_board(self, board: dict, ms_of_day: int = None, ts_ns=None):
        """kabuステーションの板JSON(1件)を処理する。"""
        if ts_ns is None:
            ts_ns = time.time_ns()

        # 初回は基準値段(前日終値)で窓を張る。PreviousClose が無ければ現値で代用。
        if self.window is None:
            base = board.get("PreviousClose") or board.get("CurrentPrice")
            if not base:
                return
            self.anchor(float(base), ts_ns)

        if ms_of_day is None:
            lt = time.localtime(ts_ns / 1e9)
            ms_of_day = (lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec) * 1000
        self.update_session(ms_of_day, ts_ns)

        # 気配10本値を取り出す。kabu は Buy1..Buy10 / Sell1..Sell10 という
        # キー名で1本ずつ入れてくる(欠けていることもある)。
        bids, asks = [], []
        for i in range(1, 11):
            for key, out in (("Buy", bids), ("Sell", asks)):
                lv = board.get(f"{key}{i}")
                if not lv:
                    continue
                px, qty = lv.get("Price"), lv.get("Qty")
                if not px or qty is None:
                    continue
                px_sen = wire.yen_to_sen(float(px))
                if not self.window.contains(px_sen):
                    self.out_of_window += 1
                    continue
                out.append((px_sen, int(qty)))

        if bids or asks:
            self.link.send(self.diff.snapshot(bids, asks, ts_ns))
            self.n_boards += 1

        # 出来高の増分を約定として記録する('T' は板には適用されない)。
        # 約定の売り買い方向は kabu からは分からないので、現値と直前の mid の
        # 位置関係で推定する(Lee-Ready 法の簡易版)。
        vol = board.get("TradingVolume")
        price = board.get("CurrentPrice")
        if vol is not None and price:
            vol = int(vol)
            if self.last_volume and vol > self.last_volume:
                px_sen = wire.yen_to_sen(float(price))
                if self.window.contains(px_sen):
                    mid_sen = None
                    if bids and asks:
                        mid_sen = (max(p for p, _ in bids) + min(p for p, _ in asks)) / 2
                    aggressor = wire.BID if (mid_sen is None or px_sen >= mid_sen) else wire.ASK
                    self.link.send(wire.trade(
                        aggressor, self.window.idx_of_sen(px_sen), vol - self.last_volume, ts_ns))
                    self.n_trades += 1
            self.last_volume = vol


# =====================================================================
#  モック: 合成板を kabuステーションと同じJSON形式で作る
# =====================================================================
class MockBoard:
    """kabuステーションAPIが返すのと同じ形の板JSONを合成する。

    C++側の合成フィード(feed.hpp)と同じ考え方:
      - フェアバリューがランダムウォーク + ドリフト
      - 気配10本値をフェアの周りに敷く
      - 日中U字(寄りと引けでボラと出来高が跳ねる)
      - 板寄せ中(寄前/昼休み/クロージング)は約定させない
    ただしこちらは「注文単位」ではなく「レベル単位の数量」を作る。
    kabuステーションが返すのがレベル単位だから、本番と同じ形になる。
    """

    LEVELS = 10             # kabuが返す気配の本数
    DRIFT_COUPLING = 0.35   # ドリフトが板の厚みを偏らせる強さ
    MAX_LEVEL_QTY = wire.LOT * 60   # 1水準の厚みの上限(6,000株)

    def __init__(self, symbol="7203", base_yen=2500.0, table=wire.TABLE_STANDARD, seed=42):
        self.rng = random.Random(seed)
        self.symbol = symbol
        self.base_yen = base_yen
        self.grid = wire.TickGrid(table)
        self.base_sen = wire.yen_to_sen(base_yen)
        self.window = wire.PriceWindow(self.grid, self.base_sen)
        self.fair_sen = float(self.base_sen)
        self.drift = 0
        self.volume = 0
        self.last_px_sen = self.base_sen
        self.step = self.grid.step_at_price(self.base_sen)
        # 各サイドの板を「ティック番号 -> 株数」で持ち続ける。
        #
        # ここを毎回引き直してはいけない。実際の板は数量が持続して少しずつ増減する。
        # 毎ティック全レベルをランダムに作り直すと OFI(best の数量変化の累積)が
        # ノイズで埋まり、z-score が ±1 を超えなくなってシグナルが1本も出ない
        # (開発中に実測。20秒の収録で発火0件だった)。
        self.qty = {wire.BID: {}, wire.ASK: {}}

    def tick(self, ms_of_day: int) -> dict:
        sess = wire.session_at(ms_of_day)
        # 日中U字: 寄りと引けで活動が跳ね、昼前後で凪ぐ
        prog = self._progress(ms_of_day)
        u = 0.35 + 1.30 * (2.0 * prog - 1.0) ** 2

        if self.rng.random() < 1.0 / 200.0:
            self.drift = self.rng.choice((-1, 0, 1))
        self.fair_sen += u * self.step * (0.02 * self.drift + 0.25 * self.rng.gauss(0, 1))
        lo = self.window.low_sen + (self.LEVELS + 2) * self.step
        hi = self.window.high_sen - (self.LEVELS + 2) * self.step
        self.fair_sen = min(max(self.fair_sen, lo), hi)

        fair_tick = self.grid.tick_of(int(self.fair_sen))
        # best bid = fair の1つ下、best ask = fair。流動株らしくスプレッド1ティック。
        top = {wire.BID: fair_tick - 1, wire.ASK: fair_tick}
        sign = {wire.BID: -1, wire.ASK: +1}

        for side in (wire.BID, wire.ASK):
            book = self.qty[side]
            want = {top[side] + sign[side] * i for i in range(self.LEVELS)}
            # 範囲外になった水準は引っ込める(フェアが動いたら遠い気配は消える)
            for t in [t for t in book if t not in want]:
                del book[t]
            # 新しく範囲に入った水準は初期の厚みを置く。touch は薄く外側ほど厚い。
            for i in range(self.LEVELS):
                t = top[side] + sign[side] * i
                if t not in book:
                    book[t] = wire.LOT * (1 + i) * (1 + self.rng.randrange(3))
            # 既存の水準を少しだけ増減させる。ドリフト方向のサイドが厚くなる
            # ことでオーダーフローに情報が乗る。
            bias = self.DRIFT_COUPLING * self.drift * (1 if side == wire.BID else -1)
            for _ in range(3):
                t = self.rng.choice(list(book))
                delta = wire.LOT * self.rng.randrange(1, 4)
                if self.rng.random() < 0.5 + bias:
                    # 上限を置く。置かないと偏りが積み上がって best が数万株まで
                    # 膨れ、実際の板とかけ離れた形になる。
                    book[t] = min(book[t] + delta, self.MAX_LEVEL_QTY)
                else:
                    book[t] = max(wire.LOT, book[t] - delta)

        board = {
            "Symbol": self.symbol,
            "SymbolName": "MOCK",
            "Exchange": EXCHANGE_TOSHO,
            "PreviousClose": self.base_yen,
            "UpperLimit": wire.sen_to_yen(self.window.high_sen),
            "LowerLimit": wire.sen_to_yen(self.window.low_sen),
        }
        for side, key in ((wire.BID, "Buy"), (wire.ASK, "Sell")):
            for i in range(self.LEVELS):
                t = top[side] + sign[side] * i
                board[f"{key}{i + 1}"] = {"Price": wire.sen_to_yen(self.grid.price_of(t)),
                                          "Qty": self.qty[side][t]}
        board["BidPrice"] = board["Buy1"]["Price"]
        board["BidQty"] = board["Buy1"]["Qty"]
        board["AskPrice"] = board["Sell1"]["Price"]
        board["AskQty"] = board["Sell1"]["Qty"]

        # 約定はザラ場だけ。板寄せ中は注文が溜まるだけで約定しない。
        if wire.is_continuous(sess) and self.rng.random() < 0.6:
            hit_ask = self.rng.random() < 0.5 + 0.10 * self.drift
            px_tick = top[wire.ASK] if hit_ask else top[wire.BID]
            self.last_px_sen = self.grid.price_of(px_tick)
            self.volume += wire.LOT * (1 + self.rng.randrange(int(1 + 8 * u)))
            # 叩かれた側の best は薄くなる(約定した分が減る)
            book = self.qty[wire.ASK if hit_ask else wire.BID]
            book[px_tick] = max(wire.LOT, book[px_tick] - wire.LOT * self.rng.randrange(1, 4))
        board["CurrentPrice"] = wire.sen_to_yen(self.last_px_sen)
        board["TradingVolume"] = self.volume
        return board

    @staticmethod
    def _progress(ms_of_day: int) -> float:
        m = ms_of_day / 60_000.0
        total = (wire.T_AM_CLOSE - wire.T_OPEN) + (wire.T_CA_START - wire.T_PM_OPEN)
        if m <= wire.T_OPEN:
            e = 0.0
        elif m < wire.T_AM_CLOSE:
            e = m - wire.T_OPEN
        elif m < wire.T_PM_OPEN:
            e = wire.T_AM_CLOSE - wire.T_OPEN
        elif m < wire.T_CA_START:
            e = (wire.T_AM_CLOSE - wire.T_OPEN) + (m - wire.T_PM_OPEN)
        else:
            e = total
        return e / total


def run_mock(args):
    """合成板を live.exe に流す。実時間の speed 倍速で1立会日を再生する。"""
    link = EngineLink(args.host, args.port, args.quiet)
    link.connect()
    table = wire.TABLE_TOPIX100 if args.tick_table == "topix100" else wire.TABLE_STANDARD
    adapter = BoardAdapter(link, table, args.quiet)
    mock = MockBoard(args.symbol, args.base_yen, table, args.seed)

    # 08:00 から 15:31 までを再生。1ステップ = 実時間 interval 秒 / speed 倍速
    start_ms = wire.T_PREOPEN * 60_000
    end_ms = (wire.T_CLOSE + 1) * 60_000
    sim_ms = start_ms
    step_ms = int(args.interval * 1000 * args.speed)
    print(f"[mock] {args.symbol} 基準値段 {args.base_yen:.0f}円 / {args.speed:.0f}倍速 / "
          f"{args.interval * 1000:.0f}ms 間隔で送信 (Ctrl+C で終了)", flush=True)
    t0 = time.time()
    n = 0
    try:
        while sim_ms < end_ms:
            board = mock.tick(sim_ms)
            adapter.on_board(board, ms_of_day=sim_ms)
            n += 1
            sim_ms += step_ms
            if args.days_loop and sim_ms >= end_ms:
                sim_ms = start_ms
                mock.base_yen = wire.sen_to_yen(int(mock.fair_sen))
                mock.base_sen = wire.yen_to_sen(mock.base_yen)
                mock.window = wire.PriceWindow(mock.grid, mock.base_sen)
                adapter.anchor(mock.base_yen)
            target = t0 + n * args.interval
            sleep = target - time.time()
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n[mock] 停止", flush=True)
    finally:
        link.close()
    print(f"[mock] 板 {adapter.n_boards} 件 / 約定 {adapter.n_trades} 件 / "
          f"{link.sent_bytes / 1e6:.2f} MB 送信", flush=True)
    if adapter.out_of_window:
        print(f"[mock] 値幅制限の外で無視した気配: {adapter.out_of_window} 件", flush=True)
    return 0


# =====================================================================
#  本番: kabuステーションAPI
# =====================================================================
def kabu_login(base_url: str, password: str) -> str:
    import requests
    r = requests.post(f"{base_url}/token", json={"APIPassword": password}, timeout=10)
    r.raise_for_status()
    js = r.json()
    if js.get("ResultCode") != 0:
        raise SystemExit(f"トークン取得に失敗: {js}")
    return js["Token"]


def kabu_symbol_info(base_url: str, token: str, symbol: str, exchange: int) -> dict:
    import requests
    r = requests.get(f"{base_url}/symbol/{symbol}@{exchange}",
                     headers={"X-API-KEY": token}, timeout=10)
    r.raise_for_status()
    return r.json()


def kabu_register(base_url: str, token: str, symbol: str, exchange: int) -> dict:
    import requests
    r = requests.put(f"{base_url}/register", headers={"X-API-KEY": token},
                     json={"Symbols": [{"Symbol": symbol, "Exchange": exchange}]}, timeout=10)
    r.raise_for_status()
    return r.json()


def run_live(args):
    if not args.password:
        raise SystemExit("--password が必要です(kabuステーションのAPIパスワード)。"
                         "口座が無い場合は --mock を使ってください。")
    import websocket  # websocket-client

    port = KABU_PORT_TEST if args.test_env else KABU_PORT_LIVE
    base_url = f"http://{KABU_HOST}:{port}/kabusapi"
    ws_url = f"ws://{KABU_HOST}:{port}/kabusapi/websocket"

    print(f"[kabu] トークン取得 {base_url}", flush=True)
    token = kabu_login(base_url, args.password)
    info = kabu_symbol_info(base_url, token, args.symbol, args.exchange)
    prev_close = info.get("PreviousClose")
    print(f"[kabu] {args.symbol} {info.get('SymbolName')} 前日終値 {prev_close} / "
          f"値幅 {info.get('LowerLimit')} - {info.get('UpperLimit')} / "
          f"売買単位 {info.get('TradingUnit')}", flush=True)
    kabu_register(base_url, token, args.symbol, args.exchange)
    print("[kabu] 銘柄登録 完了", flush=True)

    link = EngineLink(args.host, args.port, args.quiet)
    link.connect()
    table = wire.TABLE_TOPIX100 if args.tick_table == "topix100" else wire.TABLE_STANDARD
    adapter = BoardAdapter(link, table, args.quiet)
    if prev_close:
        adapter.anchor(float(prev_close))

    # WS は板が変化するたびにPushしてくる。無通知が続くのは板が動いていないだけの
    # こともあるので、ウォッチドッグは「立会時間中に限って」警告する。
    last_msg = time.time()
    print(f"[kabu] WS 接続 {ws_url}", flush=True)
    while True:
        try:
            ws = websocket.create_connection(ws_url, timeout=args.watchdog)
        except Exception as e:  # noqa: BLE001
            print(f"[kabu] WS 接続失敗 ({e}) - 5秒後に再試行", flush=True)
            time.sleep(5)
            continue
        try:
            while True:
                try:
                    raw = ws.recv()
                except Exception as e:  # noqa: BLE001
                    print(f"[kabu] WS 受信エラー ({e}) - 再接続", flush=True)
                    break
                if not raw:
                    continue
                last_msg = time.time()
                try:
                    board = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if args.symbol and str(board.get("Symbol")) != str(args.symbol):
                    continue
                adapter.on_board(board)
                if adapter.n_boards % 500 == 0:
                    print(f"[kabu] 板 {adapter.n_boards} 件 / 約定 {adapter.n_trades} 件",
                          flush=True)
        except KeyboardInterrupt:
            print("\n[kabu] 停止", flush=True)
            ws.close()
            link.close()
            return 0
        finally:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass
        if time.time() - last_msg > args.watchdog:
            print(f"[kabu] {args.watchdog:.0f} 秒間無通知 - 再接続します", flush=True)


# =====================================================================
#  自己検証
# =====================================================================
class _FakeLink:
    """EngineLink の差し替え。送られたフレームを解析して数える。"""

    def __init__(self):
        self.frames = []
        self.on_reconnect = None
        self.sent_bytes = 0

    def send(self, data: bytes):
        self.sent_bytes += len(data)
        off = 0
        while off + 9 <= len(data):
            t = data[off + 8:off + 9].decode("latin1")
            size = {"A": 14, "X": 9, "E": 9, "D": 5, "R": 14, "T": 10, "S": 6}.get(t)
            if size is None:
                raise AssertionError(f"未知のメッセージ種別 {t!r}")
            self.frames.append(t)
            off += 8 + size
        assert off == len(data), "フレーム境界が合わない"

    def connect(self, retry_sec=2.0):
        pass

    def close(self):
        pass

    def count(self, t):
        return sum(1 for f in self.frames if f == t)


def selftest() -> int:
    fails = 0

    def check(cond, label):
        nonlocal fails
        if not cond:
            fails += 1
            print(f"  FAIL {label}")

    if wire.selftest():
        print("  FAIL wire.py の自己検証が失敗")
        return 1

    # モック板 -> ワイヤ変換の一巡
    link = _FakeLink()
    adapter = BoardAdapter(link, wire.TABLE_STANDARD, quiet=True)
    mock = MockBoard("7203", 2500.0, wire.TABLE_STANDARD, seed=7)

    # 09:30(ザラ場)を1000ステップ
    ms = (9 * 60 + 30) * 60_000
    for i in range(1000):
        adapter.on_board(mock.tick(ms + i * 100), ms_of_day=ms + i * 100)

    check(link.count("R") == 1, f"R(リセット)は1回 (実際 {link.count('R')})")
    check(link.count("S") >= 1, "S(セッション)が出ている")
    check(link.count("A") > 0, "A(新規)が出ている")
    check(link.count("D") + link.count("X") > 0, "取消/削除が出ている")
    check(link.count("T") > 0, "T(約定)が出ている")
    check(adapter.out_of_window == 0, f"値幅制限外の気配なし (実際 {adapter.out_of_window})")
    check(adapter.n_boards == 1000, f"板1000件を処理 (実際 {adapter.n_boards})")

    # 板寄せ中(昼休み)は約定が出ないこと
    link2 = _FakeLink()
    ad2 = BoardAdapter(link2, wire.TABLE_STANDARD, quiet=True)
    mock2 = MockBoard("7203", 2500.0, wire.TABLE_STANDARD, seed=7)
    ms = (12 * 60) * 60_000   # 昼休み
    for i in range(300):
        ad2.on_board(mock2.tick(ms + i * 100), ms_of_day=ms + i * 100)
    check(link2.count("T") == 0, f"昼休みに約定なし (実際 {link2.count('T')})")

    # TOPIX100(呼値0.5円)でも窓に収まること
    link3 = _FakeLink()
    ad3 = BoardAdapter(link3, wire.TABLE_TOPIX100, quiet=True)
    mock3 = MockBoard("6758", 2000.0, wire.TABLE_TOPIX100, seed=7)
    ms = (10 * 60) * 60_000
    for i in range(500):
        ad3.on_board(mock3.tick(ms + i * 100), ms_of_day=ms + i * 100)
    check(ad3.out_of_window == 0, f"TOPIX100でも窓内 (実際 {ad3.out_of_window})")
    check(ad3.grid.step_at_price(wire.yen_to_sen(2000)) == 50, "TOPIX100 2000円の呼値=0.5円")

    # 値幅制限の外の気配は無視されること
    link4 = _FakeLink()
    ad4 = BoardAdapter(link4, wire.TABLE_STANDARD, quiet=True)
    ad4.on_board({"PreviousClose": 2500.0, "CurrentPrice": 2500.0, "TradingVolume": 0,
                  "Buy1": {"Price": 2499.0, "Qty": 100},
                  "Sell1": {"Price": 9999.0, "Qty": 100}},  # ストップ高(3000円)の外
                 ms_of_day=(10 * 60) * 60_000)
    check(ad4.out_of_window == 1, f"窓外の気配を1件無視 (実際 {ad4.out_of_window})")

    print(f"selftest: {'OK - 全項目一致' if fails == 0 else f'NG - {fails} 件失敗'}")
    return fails


# =====================================================================
def main() -> int:
    p = argparse.ArgumentParser(
        description="kabuステーションAPI -> ワイヤ形式 -> live.exe のブリッジ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="口座が無いうちは --mock で全経路を検証できます。")
    p.add_argument("--mock", action="store_true",
                   help="合成板を流す(口座不要・今すぐ動く)")
    p.add_argument("--selftest", action="store_true",
                   help="ネット・口座なしで変換ロジックを検証する")
    p.add_argument("--symbol", default="7203", help="銘柄コード(既定 7203 トヨタ)")
    p.add_argument("--exchange", type=int, default=EXCHANGE_TOSHO,
                   help="市場コード(1=東証)")
    p.add_argument("--password", default=None, help="kabuステーションのAPIパスワード")
    p.add_argument("--test-env", action="store_true",
                   help="kabuステーションの検証環境(ポート18081)を使う")
    p.add_argument("--tick-table", choices=("standard", "topix100"), default="standard",
                   help="呼値テーブル。TOPIX100構成銘柄は topix100")
    p.add_argument("--base-yen", type=float, default=2500.0,
                   help="--mock の基準値段(前日終値)")
    p.add_argument("--host", default="127.0.0.1", help="live.exe のホスト")
    p.add_argument("--port", type=int, default=9501, help="live.exe のポート")
    p.add_argument("--speed", type=float, default=120.0,
                   help="--mock の再生速度倍率(120 なら1立会日が約4分)")
    p.add_argument("--interval", type=float, default=0.05,
                   help="--mock の送信間隔(実時間・秒)")
    p.add_argument("--days-loop", action="store_true",
                   help="--mock で立会日を繰り返す(長時間の収録用)")
    p.add_argument("--seed", type=int, default=42, help="--mock の乱数シード")
    p.add_argument("--watchdog", type=float, default=90.0,
                   help="本番: 無通知が続いたら再接続する秒数")
    p.add_argument("--quiet", action="store_true", help="ログを抑える")
    args = p.parse_args()

    if args.selftest:
        return 1 if selftest() else 0
    if args.mock:
        return run_mock(args)
    return run_live(args)


if __name__ == "__main__":
    sys.exit(main())
