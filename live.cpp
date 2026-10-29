// live.cpp — 実データ受信エンジン
//
// アダプタ(adapters/kabu_adapter.py)から localhost TCP でバイナリフィードを受信し、
// 板更新 → リアルタイム OFI z-score の表示 → 全フレームの収録を行う。
//
// 使い方:
//   live.exe [port=9501]
//   (別の端末で) python adapters\kabu_adapter.py --mock --port 9501
//
// ストリーム形式(adapters/wire.py と対): [u64 ts_ns] + ワイヤメッセージ
//   A/X/E/D は feed.hpp と同一。拡張:
//   'R' 板リセット + 当日の価格窓(基準値段・呼値テーブル)
//   'T' 約定(板には適用しない。記録・出来高集計のみ)
//   'S' セッション遷移(ザラ場かどうかの判定に使う)
//
// 収録: captures/capture_YYYYMMDD_HHMMSS.bin に受信フレームをそのまま書く。
// Ctrl+C で終了(収録は1秒毎に flush 済み)。
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <direct.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "book.hpp"
#include "feed.hpp"
#include "jp_market.hpp"
#include "signals.hpp"

using namespace stk;

static std::atomic<bool> g_stop{false};
static BOOL WINAPI on_ctrl(DWORD) { g_stop.store(true); return TRUE; }

static void fmt_time(u64 ts_ns, char* buf, std::size_t n) {
    const std::time_t sec = static_cast<std::time_t>(ts_ns / 1'000'000'000ull);
    std::tm tm_{};
    localtime_s(&tm_, &sec);
    std::strftime(buf, n, "%H:%M:%S", &tm_);
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // リダイレクト時もログを即時に出す
    int port = 9501;
    if (argc > 1) port = std::stoi(argv[1]);

    constexpr std::size_t W = 64;        // OFI集計窓(イベント数)
    constexpr double Z_TH = 2.5;         // 発火閾値
    constexpr std::size_t COOLDOWN = W;  // 発火後の最小間隔(イベント数)

    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) { std::printf("WSAStartup failed\n"); return 1; }
    SetConsoleCtrlHandler(on_ctrl, TRUE);

    SOCKET lst = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<u_short>(port));
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    BOOL yes = TRUE;
    setsockopt(lst, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<char*>(&yes), sizeof yes);
    if (bind(lst, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0 || listen(lst, 1) != 0) {
        std::printf("bind/listen failed on port %d (WSA %d)\n", port, WSAGetLastError());
        return 1;
    }

    _mkdir("captures");
    char fname[128];
    {
        const std::time_t now = std::time(nullptr);
        std::tm tm_{};
        localtime_s(&tm_, &now);
        std::strftime(fname, sizeof fname, "captures/capture_%Y%m%d_%H%M%S.bin", &tm_);
    }
    std::FILE* cap = std::fopen(fname, "wb");
    if (!cap) { std::printf("cannot open %s\n", fname); return 1; }

    std::printf("=== stock-engine live ===\n");
    std::printf("listening 127.0.0.1:%d | capture: %s\n", port, fname);
    std::printf("アダプタ起動: python adapters\\kabu_adapter.py --mock --port %d\n\n", port);

    jpx::TickGrid grid(jpx::TickTable::Standard);
    jpx::PriceWindow win{};
    StockBook book;
    RollingZ roll(W);
    BestState prev{};
    jpx::Session sess = jpx::Session::Off;
    bool anchored = false;

    u64 n_msgs = 0, n_trades = 0, n_resets = 0, n_signals = 0, msgs_this_sec = 0;
    i64 traded_shares = 0;
    std::size_t last_fire = 0, n_events = 0;
    double last_z = 0;
    u64 last_ts = 0;

    std::vector<u8> acc;
    acc.reserve(1 << 20);
    std::vector<char> rbuf(1 << 16);
    SOCKET cli = INVALID_SOCKET;
    auto last_flush = std::chrono::steady_clock::now();

    while (!g_stop.load()) {
        // 接続待ち/受信を select でタイムアウト付き監視(Ctrl+C に応答するため)
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(cli == INVALID_SOCKET ? lst : cli, &fds);
        timeval tv{0, 200'000};
        const int r = select(0, &fds, nullptr, nullptr, &tv);
        if (r <= 0) goto tick;

        if (cli == INVALID_SOCKET) {
            cli = accept(lst, nullptr, nullptr);
            std::printf("[tcp] adapter connected\n");
            acc.clear();
            goto tick;
        }
        {
            const int got = recv(cli, rbuf.data(), static_cast<int>(rbuf.size()), 0);
            if (got <= 0) {
                std::printf("[tcp] adapter disconnected; waiting for reconnect\n");
                closesocket(cli);
                cli = INVALID_SOCKET;
                goto tick;
            }
            acc.insert(acc.end(), rbuf.data(), rbuf.data() + got);
        }

        // フレーム消費
        {
            std::size_t off = 0;
            while (acc.size() - off >= 9) {
                const u8 type = acc[off + 8];
                const std::size_t msz = msg_size(type);
                if (msz == 0) {
                    std::printf("[fatal] unknown message type 0x%02x — protocol mismatch\n", type);
                    g_stop.store(true);
                    break;
                }
                if (acc.size() - off < 8 + msz) break;
                const u8* p = acc.data() + off;
                std::memcpy(&last_ts, p, 8);
                std::fwrite(p, 1, 8 + msz, cap);  // 収録(受信フレームそのまま)
                ++n_msgs;
                ++msgs_this_sec;

                NormMsg m;
                decode(p + 8, m);
                if (m.type == 'R') {
                    grid = jpx::TickGrid(m.table ? jpx::TickTable::Topix100 : jpx::TickTable::Standard);
                    win = jpx::make_window(grid, m.base_sen);
                    if (win.n_ticks != m.idx) {
                        std::printf("[warn] 価格窓の本数が食い違う: アダプタ %d / エンジン %d\n",
                                    m.idx, win.n_ticks);
                    }
                    book.reset(win.n_ticks);
                    roll.clear();
                    prev = BestState{};
                    anchored = true;
                    ++n_resets;
                    char tbuf[16];
                    fmt_time(last_ts, tbuf, sizeof tbuf);
                    std::printf("[%s] RESET 基準値段 %.1f円 -> 窓 [%.1f, %.1f]円 %d本 (呼値 %.2f円)\n",
                                tbuf, jpx::sen_to_yen(m.base_sen),
                                jpx::sen_to_yen(win.low_sen), jpx::sen_to_yen(win.high_sen),
                                win.n_ticks, jpx::sen_to_yen(grid.step_at_price(m.base_sen)));
                } else if (m.type == 'S') {
                    const jpx::Session ns = static_cast<jpx::Session>(m.side);
                    if (ns != sess) {
                        char tbuf[16];
                        fmt_time(last_ts, tbuf, sizeof tbuf);
                        std::printf("[%s] SESSION %s\n", tbuf,
                                    jpx::SESSION_NAMES[static_cast<int>(ns)]);
                    }
                    sess = ns;
                } else if (m.type == 'T') {
                    ++n_trades;
                    traded_shares += m.qty;
                } else {
                    if (m.type == 'A' && (m.idx < 0 || m.idx >= book.n_ticks())) {
                        std::printf("[fatal] idx %d が価格窓 [0,%d) の外 — アダプタとの不整合\n",
                                    m.idx, book.n_ticks());
                        g_stop.store(true);
                        break;
                    }
                    switch (m.type) {
                    case 'A': book.add(m.id, static_cast<Side>(m.side), m.idx, m.qty); break;
                    case 'X': case 'E': book.reduce(m.id, m.qty); break;
                    case 'D': book.remove(m.id); break;
                    default: break;
                    }
                    const BestState cur = snap_best(book);
                    last_z = roll.push(ofi_contribution(prev, cur));
                    prev = cur;
                    ++n_events;
                    // シグナルはザラ場だけ。板寄せ中は価格形成の仕組みが違う。
                    if (roll.warmed() && cur.valid && jpx::is_continuous(sess) &&
                        (last_z > Z_TH || last_z < -Z_TH) && n_events - last_fire >= COOLDOWN) {
                        last_fire = n_events;
                        ++n_signals;
                        char tbuf[16];
                        fmt_time(last_ts, tbuf, sizeof tbuf);
                        const double bid = jpx::sen_to_yen(jpx::idx_to_sen(grid, win, cur.pb));
                        const double ask = jpx::sen_to_yen(jpx::idx_to_sen(grid, win, cur.pa));
                        std::printf("[%s] SIGNAL %s z=%+.2f mid=%.2f円 spread=%.2f円 厚み比 %+.2f\n",
                                    tbuf, last_z > 0 ? "BUY " : "SELL", last_z,
                                    0.5 * (bid + ask), ask - bid, depth_imbalance(cur));
                    }
                }
                off += 8 + msz;
            }
            if (off > 0) acc.erase(acc.begin(), acc.begin() + static_cast<std::ptrdiff_t>(off));
        }

    tick:
        const auto now = std::chrono::steady_clock::now();
        if (now - last_flush >= std::chrono::seconds(1)) {
            std::fflush(cap);
            last_flush = now;
            if (anchored && book.two_sided()) {
                const double bid = jpx::sen_to_yen(jpx::idx_to_sen(grid, win, book.best_idx(BID)));
                const double ask = jpx::sen_to_yen(jpx::idx_to_sen(grid, win, book.best_idx(ASK)));
                std::printf("[stat] %llu msg/s | %s | bid %.2f x %lld | ask %.2f x %lld | "
                            "spread %.2f | z %+.2f | 出来高 %lld株 | signals %llu\n",
                            static_cast<unsigned long long>(msgs_this_sec),
                            jpx::SESSION_NAMES[static_cast<int>(sess)],
                            bid, static_cast<long long>(book.best_qty(BID)),
                            ask, static_cast<long long>(book.best_qty(ASK)),
                            ask - bid, last_z, static_cast<long long>(traded_shares),
                            static_cast<unsigned long long>(n_signals));
            }
            msgs_this_sec = 0;
        }
    }

    std::fclose(cap);
    if (cli != INVALID_SOCKET) closesocket(cli);
    closesocket(lst);
    WSACleanup();
    std::printf("\n=== done: %llu msgs, %llu resets, %llu trades (%lld株), %llu signals -> %s ===\n",
                static_cast<unsigned long long>(n_msgs),
                static_cast<unsigned long long>(n_resets),
                static_cast<unsigned long long>(n_trades),
                static_cast<long long>(traded_shares),
                static_cast<unsigned long long>(n_signals), fname);
    return 0;
}
