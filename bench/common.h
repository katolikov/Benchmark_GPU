// Common timing / IO / comparison helpers shared by bench_mnn and bench_tflite.
// Designed for stand-alone Android arm64-v8a binaries — no dependencies beyond
// libc++/libc.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace bench {

using clk = std::chrono::steady_clock;

inline double now_ms() {
    auto t = clk::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t).count() / 1e6;
}

inline std::vector<uint8_t> read_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", path.c_str());
        std::exit(2);
    }
    auto sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    f.read(reinterpret_cast<char *>(buf.data()), sz);
    return buf;
}

inline void write_file(const std::string &path, const void *data, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(data), bytes);
}

struct Stats {
    double mean = 0, stdev = 0, min = 0, max = 0, p50 = 0, p90 = 0, p99 = 0;
};

inline Stats compute_stats(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / v.size();
    double sse = 0;
    for (double x : v) sse += (x - s.mean) * (x - s.mean);
    s.stdev = std::sqrt(sse / v.size());
    s.min = v.front();
    s.max = v.back();
    auto pick = [&](double q) { return v[std::min<size_t>(v.size() - 1, (size_t)(q * v.size()))]; };
    s.p50 = pick(0.50);
    s.p90 = pick(0.90);
    s.p99 = pick(0.99);
    return s;
}

struct CmpResult {
    double cos = 0;
    double max_abs = 0;
    double mean_abs = 0;
    double rel_l2 = 0;
    int    argmax_a = -1, argmax_b = -1;
};

inline CmpResult compare_fp32(const float *a, const float *b, size_t n) {
    CmpResult r;
    double dot = 0, na = 0, nb = 0, mae = 0, sse = 0, ref_sse = 0;
    float ma = -1e30f, mb = -1e30f;
    int ia = -1, ib = -1;
    for (size_t i = 0; i < n; ++i) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
        double d = (double)a[i] - (double)b[i];
        if (std::fabs(d) > r.max_abs) r.max_abs = std::fabs(d);
        mae += std::fabs(d);
        sse += d * d;
        ref_sse += (double)b[i] * (double)b[i];
        if (a[i] > ma) { ma = a[i]; ia = (int)i; }
        if (b[i] > mb) { mb = b[i]; ib = (int)i; }
    }
    r.cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    r.mean_abs = mae / (double)n;
    r.rel_l2 = std::sqrt(sse) / (std::sqrt(ref_sse) + 1e-30);
    r.argmax_a = ia;
    r.argmax_b = ib;
    return r;
}

struct Args {
    std::string model;
    std::string input;            // optional; if empty, generate random fp32 from --seed
    std::string output_bin;
    std::string ref_bin;
    std::string backend;          // backend name ("cpu","opencl","vulkan","gpu","xnnpack")
    std::string variant;          // sub-variant ("fp32","fp16","buffer","image","cl")
    int warmup = 5;
    int runs = 30;
    uint64_t seed = 42;           // PRNG seed for synthetic input
    std::string label;            // free-form label, echoed in JSON
    int threads = 4;
};

// Deterministic across runs of binaries built with the same libc++. Both
// bench_mnn and bench_tflite share this, so they feed the same bytes.
inline std::vector<float> generate_input_fp32(uint64_t seed, size_t count) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(count);
    for (size_t i = 0; i < count; ++i) v[i] = dist(rng);
    return v;
}

inline void parse_args(int argc, char **argv, Args &a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing arg for %s\n", k.c_str()); std::exit(2); }
            return std::string(argv[++i]);
        };
        if      (k == "--model")    a.model = next();
        else if (k == "--input")    a.input = next();
        else if (k == "--output")   a.output_bin = next();
        else if (k == "--ref")      a.ref_bin = next();
        else if (k == "--backend")  a.backend = next();
        else if (k == "--variant")  a.variant = next();
        else if (k == "--warmup")   a.warmup = std::atoi(next().c_str());
        else if (k == "--runs")     a.runs = std::atoi(next().c_str());
        else if (k == "--threads")  a.threads = std::atoi(next().c_str());
        else if (k == "--seed")     a.seed = std::strtoull(next().c_str(), nullptr, 10);
        else if (k == "--label")    a.label = next();
        else { std::fprintf(stderr, "unknown arg: %s\n", k.c_str()); std::exit(2); }
    }
    if (a.model.empty() || a.backend.empty()) {
        std::fprintf(stderr,
            "usage: %s --model M --backend cpu|opencl|vulkan|gpu|xnnpack "
            "[--variant fp32|fp16|...] [--input I.bin | --seed N] "
            "[--ref REF.bin] [--output OUT.bin] "
            "[--warmup N] [--runs N] [--threads N] [--label TAG]\n",
            argv[0]);
        std::exit(2);
    }
}

// Print one JSON line: machine-parseable, easy to grep.
inline void emit_json(const std::string &framework,
                      const Args &a,
                      double t_create_ms,
                      double t_first_ms,
                      const Stats &s_warm,
                      const Stats &s_run,
                      const CmpResult *cmp_ref,
                      size_t output_count) {
    std::printf("{");
    std::printf("\"framework\":\"%s\"", framework.c_str());
    std::printf(",\"backend\":\"%s\"", a.backend.c_str());
    std::printf(",\"variant\":\"%s\"", a.variant.c_str());
    std::printf(",\"label\":\"%s\"", a.label.c_str());
    std::printf(",\"threads\":%d", a.threads);
    std::printf(",\"warmup_runs\":%d", a.warmup);
    std::printf(",\"runs\":%d", a.runs);
    std::printf(",\"seed\":%llu", (unsigned long long)a.seed);
    std::printf(",\"create_ms\":%.4f", t_create_ms);
    std::printf(",\"first_inference_ms\":%.4f", t_first_ms);
    std::printf(",\"warmup\":{\"mean\":%.4f,\"min\":%.4f,\"max\":%.4f}",
                s_warm.mean, s_warm.min, s_warm.max);
    std::printf(",\"inference\":{\"mean\":%.4f,\"stdev\":%.4f,\"min\":%.4f,\"max\":%.4f,"
                "\"p50\":%.4f,\"p90\":%.4f,\"p99\":%.4f}",
                s_run.mean, s_run.stdev, s_run.min, s_run.max, s_run.p50, s_run.p90, s_run.p99);
    std::printf(",\"output_elements\":%zu", output_count);
    if (cmp_ref) {
        std::printf(",\"vs_ref\":{\"cosine\":%.6f,\"max_abs\":%.6f,\"mean_abs\":%.6f,"
                    "\"rel_l2\":%.6f,\"argmax_a\":%d,\"argmax_ref\":%d}",
                    cmp_ref->cos, cmp_ref->max_abs, cmp_ref->mean_abs,
                    cmp_ref->rel_l2, cmp_ref->argmax_a, cmp_ref->argmax_b);
    }
    std::printf("}\n");
}

} // namespace bench
