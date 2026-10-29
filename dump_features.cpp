// dump_features.cpp — GBDT学習用の特徴量 + 前方ラベルをCSVに出す
//
// 使い方:
//   dump_features.exe [1日あたりイベント数=200000] [日数=10] [seed=42] [出力=features.csv]
//   dump_features.exe ... --null    オーダーフローに情報のないフィードで出す(帰無データ)
//
// 合成フィードをリプレイし、ザラ場のイベントを STRIDE 間隔でサンプリングして
// 「特徴量16種 + H イベント先の mid 変動(ティック)」を1行として書き出す。
//
// 株ならではの扱い:
//   - ザラ場(連続約定)のイベントだけを対象にする。板寄せ中の板はクロスしているのが
//     正常で、その間の mid は価格ではない
//   - ラベルは同じザラ場セグメント内でしか取らない。昼休みや翌日を跨いだ変動を
//     ラベルにすると、寄り付きのギャップを予測させることになり短期板シグナルの
//     学習として成立しない
//   - 価格窓は日ごとに張り直されるので、窓が変わったらリターン系の特徴量も
//     作り直す(FeatureEngine::reset)
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "book.hpp"
#include "features.hpp"
#include "feed.hpp"
#include "jp_market.hpp"

using namespace stk;

int main(int argc, char** argv) {
    StockFeed::Config cfg;
    cfg.events_per_day = 200'000;
    cfg.days = 10;
    cfg.seed = 42;
    const char* out_path = "features.csv";
    bool null_mode = false;

    std::vector<char*> pos;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--null") == 0) null_mode = true;
        else pos.push_back(argv[i]);
    }
    if (pos.size() > 0) cfg.events_per_day = std::stoull(pos[0]);
    if (pos.size() > 1) cfg.days = std::stoi(pos[1]);
    if (pos.size() > 2) cfg.seed = std::stoull(pos[2]);
    if (pos.size() > 3) out_path = pos[3];
    if (null_mode) { cfg.drift_strength = 0.0; cfg.flow_coupling = 0.0; }

    constexpr std::size_t H = 500;      // ラベルの評価ホライズン(ザラ場イベント数)
    constexpr std::size_t STRIDE = 16;  // サンプリング間隔(自己相関の低減)
    constexpr std::size_t WARMUP = 5'000;

    std::printf("=== dump features (%zu events/day x %d days, seed=%llu%s) -> %s ===\n",
                cfg.events_per_day, cfg.days, (unsigned long long)cfg.seed,
                null_mode ? ", NULL" : "", out_path);

    StockFeed feed(cfg);
    std::vector<u8> buf;
    buf.reserve(cfg.events_per_day * std::size_t(cfg.days) * 12);
    feed.generate(buf);

    StockBook book;
    FeatureEngine fe;
    BestState prev{};
    jpx::Session sess = jpx::Session::Off;
    i64 ms_of_day = 0;

    struct Row { std::size_t si; double f[ML_NFEAT]; };
    std::vector<Row> rows;
    std::vector<double> mids;   // ザラ場のみの mid(窓内インデックス)
    std::vector<u32> seg_of;
    mids.reserve(cfg.events_per_day * std::size_t(cfg.days));
    seg_of.reserve(cfg.events_per_day * std::size_t(cfg.days));

    u32 seg = 0;
    bool was_cont = false;
    std::size_t k = 0;
    double f[ML_NFEAT];

    for (std::size_t off = 0; off < buf.size();) {
        NormMsg m;
        off += decode(buf.data() + off, m);
        switch (m.type) {
        case 'R':
            book.reset(m.idx);
            fe.reset();          // 価格窓が変わるのでリターン系を作り直す
            prev = BestState{};
            continue;
        case 'S':
            sess = static_cast<jpx::Session>(m.side);
            ms_of_day = i64(m.idx) * 1000;
            continue;
        case 'T': continue;
        case 'A': book.add(m.id, static_cast<Side>(m.side), m.idx, m.qty); break;
        case 'X': case 'E': book.reduce(m.id, m.qty); break;
        case 'D': book.remove(m.id); break;
        default: continue;
        }

        const BestState cur = snap_best(book);
        const bool ok = fe.update(prev, cur, sess, ms_of_day, f);
        prev = cur;

        const bool cont = jpx::is_continuous(sess);
        if (!cont) { was_cont = false; continue; }
        if (!was_cont) { ++seg; was_cont = true; }
        if (!cur.valid) continue;

        mids.push_back(0.5 * (cur.pb + cur.pa));
        seg_of.push_back(seg);

        if (ok && k >= WARMUP && k % STRIDE == 0) {
            Row r;
            r.si = mids.size() - 1;
            for (int i = 0; i < ML_NFEAT; ++i) r.f[i] = f[i];
            rows.push_back(r);
        }
        ++k;
    }

    std::FILE* fp = std::fopen(out_path, "w");
    if (!fp) { std::fprintf(stderr, "%s を開けません\n", out_path); return 1; }
    std::fprintf(fp, "event_idx");
    for (const char* name : ML_FEAT_NAMES) std::fprintf(fp, ",%s", name);
    std::fprintf(fp, ",dmid\n");
    std::size_t written = 0, skipped = 0;
    for (const Row& r : rows) {
        if (r.si + H >= mids.size()) break;
        if (seg_of[r.si + H] != seg_of[r.si]) { ++skipped; continue; }
        std::fprintf(fp, "%zu", r.si);
        for (int i = 0; i < ML_NFEAT; ++i) std::fprintf(fp, ",%.10g", r.f[i]);
        std::fprintf(fp, ",%.4f\n", mids[r.si + H] - mids[r.si]);
        ++written;
    }
    std::fclose(fp);
    std::printf("ザラ場イベント: %zu / 行: %zu (%zu 行は休憩を跨ぐため除外)\n",
                mids.size(), written, skipped);
    return 0;
}
