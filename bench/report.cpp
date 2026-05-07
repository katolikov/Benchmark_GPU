// Render results JSONL into a Markdown report. Replaces make_report.py.
//
// Usage:
//   report < results.jsonl > REPORT.md
// or:
//   report path/to/results.jsonl path/to/REPORT.md
//
// Each input line is one of the JSON objects emitted by emit_json() in
// common.h. The parser is a tiny key-extractor — it relies on our exact
// emitted format (single line, no extra whitespace, fixed key spelling).

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Row {
    std::string framework, backend, variant, label;
    int    threads = 0, warmup_runs = 0, runs = 0;
    long long seed = 0;
    double create_ms = 0, first_inference_ms = 0;
    double warm_mean = 0, warm_min = 0, warm_max = 0;
    double inf_mean = 0, inf_stdev = 0, inf_min = 0, inf_max = 0;
    double inf_p50 = 0, inf_p90 = 0, inf_p99 = 0;
    long long output_elements = 0;
    bool   have_cmp = false;
    double cos = 0, max_abs = 0, mean_abs = 0, rel_l2 = 0;
    int    argmax_a = -1, argmax_ref = -1;
};

// Find `"key":` at or after `from`, optionally restricted to the substring
// inside parent object `parent`. Returns position immediately after the colon.
size_t find_key(const std::string &s, const std::string &key,
                const std::string &parent = "", size_t from = 0) {
    size_t scan_from = from;
    size_t scan_end = s.size();
    if (!parent.empty()) {
        std::string p = "\"" + parent + "\":{";
        size_t pp = s.find(p, from);
        if (pp == std::string::npos) return std::string::npos;
        scan_from = pp + p.size();
        // find matching closing brace; depth=1 starting after '{'
        int depth = 1;
        scan_end = scan_from;
        while (scan_end < s.size() && depth > 0) {
            char c = s[scan_end++];
            if (c == '{') ++depth;
            else if (c == '}') --depth;
        }
    }
    std::string pat = "\"" + key + "\":";
    size_t p = s.find(pat, scan_from);
    if (p == std::string::npos || p >= scan_end) return std::string::npos;
    return p + pat.size();
}

std::string read_str(const std::string &s, const std::string &key,
                     const std::string &parent = "") {
    size_t p = find_key(s, key, parent);
    if (p == std::string::npos || s[p] != '"') return "";
    size_t e = s.find('"', p + 1);
    return s.substr(p + 1, e - p - 1);
}

double read_num(const std::string &s, const std::string &key,
                const std::string &parent = "") {
    size_t p = find_key(s, key, parent);
    if (p == std::string::npos) return 0;
    return std::strtod(s.c_str() + p, nullptr);
}

long long read_int(const std::string &s, const std::string &key,
                   const std::string &parent = "") {
    size_t p = find_key(s, key, parent);
    if (p == std::string::npos) return 0;
    return std::strtoll(s.c_str() + p, nullptr, 10);
}

Row parse_row(const std::string &line) {
    Row r;
    r.framework         = read_str(line, "framework");
    r.backend           = read_str(line, "backend");
    r.variant           = read_str(line, "variant");
    r.label             = read_str(line, "label");
    r.threads           = (int)read_int(line, "threads");
    r.warmup_runs       = (int)read_int(line, "warmup_runs");
    r.runs              = (int)read_int(line, "runs");
    r.seed              = read_int(line, "seed");
    r.create_ms         = read_num(line, "create_ms");
    r.first_inference_ms= read_num(line, "first_inference_ms");
    r.warm_mean         = read_num(line, "mean", "warmup");
    r.warm_min          = read_num(line, "min",  "warmup");
    r.warm_max          = read_num(line, "max",  "warmup");
    r.inf_mean          = read_num(line, "mean",  "inference");
    r.inf_stdev         = read_num(line, "stdev", "inference");
    r.inf_min           = read_num(line, "min",   "inference");
    r.inf_max           = read_num(line, "max",   "inference");
    r.inf_p50           = read_num(line, "p50",   "inference");
    r.inf_p90           = read_num(line, "p90",   "inference");
    r.inf_p99           = read_num(line, "p99",   "inference");
    r.output_elements   = read_int(line, "output_elements");
    if (find_key(line, "cosine", "vs_ref") != std::string::npos) {
        r.have_cmp  = true;
        r.cos       = read_num(line, "cosine",   "vs_ref");
        r.max_abs   = read_num(line, "max_abs",  "vs_ref");
        r.mean_abs  = read_num(line, "mean_abs", "vs_ref");
        r.rel_l2    = read_num(line, "rel_l2",   "vs_ref");
        r.argmax_a  = (int)read_int(line, "argmax_a",   "vs_ref");
        r.argmax_ref= (int)read_int(line, "argmax_ref", "vs_ref");
    }
    return r;
}

std::string fmt(double v, int prec = 2) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.*f", prec, v);
    return buf;
}

std::string render(const std::vector<Row> &rows) {
    std::ostringstream o;
    o << "# MobileNetV2 GPU/CPU benchmark — MNN vs TFLite (LiteRT)\n\n";
    o << "**Device:** Samsung Galaxy Z Fold7 (SM-F766B, Exynos 2500, "
         "Xclipse 950 GPU)  \n";
    o << "**ABI:** arm64-v8a · Android 16 · NDK 27.0.12077973  \n";
    o << "**Model:** MobileNetV2 (1.0_224, fp32, ImageNet weights)  \n";
    o << "**Input:** seed=" << (rows.empty() ? 42 : rows.front().seed)
      << " uniform_real(-1, 1) generated in C++ from `std::mt19937_64`.\n\n";
    o << "Each backend was compared to a reference output dumped from a "
         "previous CPU-FP32 run of the SAME framework — TFLite XNNPACK CPU "
         "for the TFLite side, MNN CPU for the MNN side.\n\n";
    if (!rows.empty()) {
        o << "- Warmup runs: " << rows.front().warmup_runs
          << " · Timed runs: " << rows.front().runs << "\n";
    }
    o << "- Each MNN GPU run starts cold (cache deleted before run).\n\n";

    // -------- Headline ----------
    o << "## Headline numbers (all times in milliseconds)\n\n";
    o << "| Framework | Backend | Variant | Create | First inf. | Warm mean | "
         "Inf. p50 | Inf. p90 | Inf. mean ± stdev | Cosine vs ref | Argmax match |\n";
    o << "|---|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|\n";
    for (const auto &r : rows) {
        std::string match = r.have_cmp
            ? (r.argmax_a == r.argmax_ref ? "✓" : "✗")
            : "-";
        std::string cos = r.have_cmp ? fmt(r.cos, 4) : "-";
        o << "| " << r.framework
          << " | " << r.backend
          << " | " << (r.variant.empty() ? "-" : r.variant)
          << " | " << fmt(r.create_ms, 1)
          << " | " << fmt(r.first_inference_ms, 1)
          << " | " << fmt(r.warm_mean)
          << " | " << fmt(r.inf_p50)
          << " | " << fmt(r.inf_p90)
          << " | " << fmt(r.inf_mean) << " ± " << fmt(r.inf_stdev)
          << " | " << cos
          << " | " << match << " |\n";
    }
    o << "\n";

    // -------- CPU section ----------
    o << "## CPU comparison (4 threads, fp32)\n\n";
    o << "| Path | Inf. p50 | Inf. mean | Notes |\n";
    o << "|---|---:|---:|---|\n";
    for (const auto &r : rows) {
        if (r.backend != "cpu" && r.backend != "xnnpack") continue;
        const char *note = "";
        if      (r.framework == "tflite" && r.backend == "cpu")     note = "TFLite C API w/o delegate (basic kernels — not used in production).";
        else if (r.backend == "xnnpack")                            note = "TFLite + XNNPACK delegate (TFLite mobile production default).";
        else if (r.framework == "mnn" && r.backend == "cpu")        note = "MNN CPU (NEON / SVE / i8mm / fp16-dot).";
        o << "| " << r.label
          << " | " << fmt(r.inf_p50)
          << " | " << fmt(r.inf_mean)
          << " | " << note << " |\n";
    }
    o << "\n";

    // -------- GPU section ----------
    o << "## GPU comparison (Xclipse 950)\n\n";
    o << "| Path | Create | First inf. | Inf. p50 | Inf. mean | Cosine |\n";
    o << "|---|---:|---:|---:|---:|---:|\n";
    for (const auto &r : rows) {
        if (r.backend != "gpu" && r.backend != "opencl" && r.backend != "vulkan") continue;
        std::string cos = r.have_cmp ? fmt(r.cos, 4) : "-";
        o << "| " << r.label
          << " | " << fmt(r.create_ms, 1)
          << " | " << fmt(r.first_inference_ms, 1)
          << " | " << fmt(r.inf_p50)
          << " | " << fmt(r.inf_mean)
          << " | " << cos << " |\n";
    }
    o << "\n";

    // -------- Cold start ----------
    o << "## Cold-start cost (only GPU paths have JIT-time)\n\n";
    o << "MNN does kernel JIT on **first inference**; TFLite does it during "
         "**create**. The cold totals are similar but the latency is split "
         "very differently.\n\n";
    o << "| Path | Create | First inf. | Cold total |\n";
    o << "|---|---:|---:|---:|\n";
    for (const auto &r : rows) {
        if (r.backend != "gpu" && r.backend != "opencl" && r.backend != "vulkan") continue;
        double total = r.create_ms + r.first_inference_ms;
        o << "| " << r.label
          << " | " << fmt(r.create_ms, 1)
          << " | " << fmt(r.first_inference_ms, 1)
          << " | **" << fmt(total, 1) << "** |\n";
    }
    o << "\n";

    // -------- Accuracy ----------
    o << "## Accuracy vs same-framework CPU-FP32 reference\n\n";
    o << "| Path | Cosine | Max abs | Mean abs | Rel. L2 | Argmax (got / ref) |\n";
    o << "|---|---:|---:|---:|---:|:---:|\n";
    for (const auto &r : rows) {
        if (!r.have_cmp) continue;
        char arg[32];
        std::snprintf(arg, sizeof arg, "%d / %d", r.argmax_a, r.argmax_ref);
        o << "| " << r.label
          << " | " << fmt(r.cos, 6)
          << " | " << fmt(r.max_abs, 4)
          << " | " << fmt(r.mean_abs, 5)
          << " | " << fmt(r.rel_l2, 5)
          << " | " << arg << " |\n";
    }
    o << "\n";

    // -------- Best per framework ----------
    auto best = [&](const std::vector<std::string> &backends) -> const Row * {
        const Row *best = nullptr;
        for (const auto &r : rows) {
            if (std::find(backends.begin(), backends.end(), r.backend) == backends.end()) continue;
            if (!best || r.inf_p50 < best->inf_p50) best = &r;
        }
        return best;
    };
    auto best_cpu     = best({"cpu", "xnnpack"});
    auto best_gpu     = best({"gpu", "opencl", "vulkan"});
    auto best_mnn_gpu = best({"opencl", "vulkan"});
    auto best_tfl_gpu = best({"gpu"});
    o << "## Best configuration per framework\n\n";
    if (best_cpu)     o << "- **Fastest CPU overall**: `"     << best_cpu->label
                        << "` — p50 " << fmt(best_cpu->inf_p50)     << " ms.\n";
    if (best_gpu)     o << "- **Fastest GPU overall**: `"     << best_gpu->label
                        << "` — p50 " << fmt(best_gpu->inf_p50)     << " ms.\n";
    if (best_mnn_gpu) o << "- **Best MNN GPU config**: `"     << best_mnn_gpu->label
                        << "` — p50 " << fmt(best_mnn_gpu->inf_p50) << " ms.\n";
    if (best_tfl_gpu) o << "- **Best TFLite GPU config**: `"  << best_tfl_gpu->label
                        << "` — p50 " << fmt(best_tfl_gpu->inf_p50) << " ms.\n";
    o << "\n";
    return o.str();
}

} // namespace

int main(int argc, char **argv) {
    std::istream *in = &std::cin;
    std::ostream *out = &std::cout;
    std::ifstream fin;
    std::ofstream fout;
    if (argc >= 2) {
        fin.open(argv[1]);
        if (!fin) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
        in = &fin;
    }
    if (argc >= 3) {
        fout.open(argv[2]);
        if (!fout) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
        out = &fout;
    }

    std::vector<Row> rows;
    std::string line;
    while (std::getline(*in, line)) {
        // strip "RESULT: " prefix if present
        const char *prefix = "RESULT: ";
        if (line.compare(0, std::strlen(prefix), prefix) == 0) {
            line = line.substr(std::strlen(prefix));
        }
        // skip non-JSON / blank lines
        size_t p = line.find_first_not_of(" \t\r");
        if (p == std::string::npos || line[p] != '{') continue;
        rows.push_back(parse_row(line));
    }
    *out << render(rows);
    return 0;
}
