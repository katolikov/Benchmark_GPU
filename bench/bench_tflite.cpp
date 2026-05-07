// TFLite benchmark binary using the C API. Mirrors bench_mnn.cpp.
//
// Backends: cpu, gpu
// Variants for GPU: "fp32" (precision=MAX), "fp16" (precision-loss allowed,
//                   priority=MIN_LATENCY) — TFLite's recommended fast config.
//                   "cl" forces OpenCL backend, "vulkan" attempts Vulkan via
//                   experimental_flags GL_ONLY=0 (Vulkan not selectable from
//                   public C API; GPU delegate auto-picks CL/GL on Android).
#include "common.h"

#include <tflite/c/c_api.h>
#include <tflite/c/c_api_types.h>
#include <tflite/c/common.h>
#include <tflite/delegates/gpu/delegate.h>
#include <tflite/delegates/gpu/delegate_options.h>

// XNNPACK delegate is exported by libtensorflowlite_jni.so but its public C API
// header is not bundled in the litert AAR. Declare the small surface we use.
extern "C" {
    typedef struct {
        int num_threads;
        uint64_t flags;
        struct TfLiteXNNPackDelegateWeightsCache *weights_cache;
        const char *experimental_weight_cache_file_path;
        void *handle_backend_context_callback;
        void *handle_backend_context_user_data;
        const char *runtime_flags;
    } TfLiteXNNPackDelegateOptions;
    TfLiteXNNPackDelegateOptions TfLiteXNNPackDelegateOptionsDefault();
    TfLiteDelegate *TfLiteXNNPackDelegateCreate(const TfLiteXNNPackDelegateOptions *options);
    void TfLiteXNNPackDelegateDelete(TfLiteDelegate *delegate);
}

#include <cstdio>
#include <cstring>
#include <memory>

using namespace bench;

static void error_reporter(void *, const char *fmt, va_list args) {
    std::fprintf(stderr, "[tflite] ");
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
}

int main(int argc, char **argv) {
    Args a;
    parse_args(argc, argv, a);

    bool use_gpu = (a.backend == "gpu");
    bool use_xnn = (a.backend == "xnnpack");
    if (!use_gpu && !use_xnn && a.backend != "cpu") {
        std::fprintf(stderr, "TFLite: unknown backend '%s'\n", a.backend.c_str());
        return 2;
    }

    double t0 = now_ms();

    // ---- Model + options ------------------------------------------------
    TfLiteModel *model = TfLiteModelCreateFromFile(a.model.c_str());
    if (!model) { std::fprintf(stderr, "TFLite: load failed\n"); return 1; }

    TfLiteInterpreterOptions *opts = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(opts, a.threads);
    TfLiteInterpreterOptionsSetErrorReporter(opts, error_reporter, nullptr);

    TfLiteDelegate *delegate = nullptr;
    TfLiteGpuDelegateOptionsV2 gpu_opts;
    if (use_gpu) {
        gpu_opts = TfLiteGpuDelegateOptionsV2Default();
        // Recommended best-perf: priority1=MIN_LATENCY (FP16 allowed),
        //                       inference_preference=SUSTAINED_SPEED for repeated runs.
        if (a.variant == "fp32") {
            gpu_opts.is_precision_loss_allowed = 0;
            gpu_opts.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
            gpu_opts.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
            gpu_opts.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
        } else {
            // fp16 fast (default per TFLite docs for GPU performance)
            gpu_opts.is_precision_loss_allowed = 1;
            gpu_opts.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
            gpu_opts.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
            gpu_opts.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
        }
        gpu_opts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
        gpu_opts.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;
        if (a.variant == "cl") {
            gpu_opts.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY;
        }
        gpu_opts.max_delegated_partitions = 1;

        delegate = TfLiteGpuDelegateV2Create(&gpu_opts);
        if (!delegate) {
            std::fprintf(stderr, "TFLite: GPU delegate create failed\n");
            return 1;
        }
        TfLiteInterpreterOptionsAddDelegate(opts, delegate);
    } else if (use_xnn) {
        TfLiteXNNPackDelegateOptions xopts = TfLiteXNNPackDelegateOptionsDefault();
        xopts.num_threads = a.threads;
        delegate = TfLiteXNNPackDelegateCreate(&xopts);
        if (!delegate) { std::fprintf(stderr, "TFLite: XNNPACK delegate create failed\n"); return 1; }
        TfLiteInterpreterOptionsAddDelegate(opts, delegate);
    }

    TfLiteInterpreter *interp = TfLiteInterpreterCreate(model, opts);
    if (!interp) {
        std::fprintf(stderr, "TFLite: interpreter create failed\n");
        return 1;
    }
    if (TfLiteInterpreterAllocateTensors(interp) != kTfLiteOk) {
        std::fprintf(stderr, "TFLite: AllocateTensors failed\n");
        return 1;
    }
    double t_create = now_ms() - t0;

    TfLiteTensor *in_t  = TfLiteInterpreterGetInputTensor(interp, 0);
    const TfLiteTensor *out_t = TfLiteInterpreterGetOutputTensor(interp, 0);

    size_t in_bytes_n = TfLiteTensorByteSize(in_t);
    std::vector<uint8_t> in_buf(in_bytes_n);
    if (!a.input.empty()) {
        auto file_bytes = read_file(a.input);
        if (file_bytes.size() != in_bytes_n) {
            std::fprintf(stderr, "TFLite: input %zu B != tensor %zu B\n",
                         file_bytes.size(), in_bytes_n);
            return 1;
        }
        std::memcpy(in_buf.data(), file_bytes.data(), in_bytes_n);
    } else {
        size_t fp32_n = in_bytes_n / sizeof(float);
        auto v = generate_input_fp32(a.seed, fp32_n);
        std::memcpy(in_buf.data(), v.data(), in_bytes_n);
    }
    if (TfLiteTensorCopyFromBuffer(in_t, in_buf.data(), in_bytes_n) != kTfLiteOk) {
        std::fprintf(stderr, "TFLite: copy in failed\n");
        return 1;
    }

    // ---- First inference (cold; GPU compiles kernels here) -------------
    double tf0 = now_ms();
    if (TfLiteInterpreterInvoke(interp) != kTfLiteOk) {
        std::fprintf(stderr, "TFLite: first invoke failed\n");
        return 1;
    }
    double t_first = now_ms() - tf0;

    std::vector<double> warm_v;
    warm_v.reserve(a.warmup);
    for (int i = 0; i < a.warmup; ++i) {
        double s = now_ms();
        if (TfLiteInterpreterInvoke(interp) != kTfLiteOk) return 1;
        warm_v.push_back(now_ms() - s);
    }
    std::vector<double> run_v;
    run_v.reserve(a.runs);
    for (int i = 0; i < a.runs; ++i) {
        double s = now_ms();
        if (TfLiteInterpreterInvoke(interp) != kTfLiteOk) return 1;
        run_v.push_back(now_ms() - s);
    }

    size_t out_bytes = TfLiteTensorByteSize(out_t);
    std::vector<uint8_t> out_buf(out_bytes);
    TfLiteTensorCopyToBuffer(out_t, out_buf.data(), out_bytes);
    size_t out_count = out_bytes / sizeof(float);

    if (!a.output_bin.empty()) {
        write_file(a.output_bin, out_buf.data(), out_bytes);
    }

    CmpResult cmp;
    bool have_cmp = false;
    if (!a.ref_bin.empty()) {
        auto rb = read_file(a.ref_bin);
        if (rb.size() == out_bytes) {
            cmp = compare_fp32(reinterpret_cast<const float *>(out_buf.data()),
                               reinterpret_cast<const float *>(rb.data()),
                               out_count);
            have_cmp = true;
        } else {
            std::fprintf(stderr, "TFLite: ref %zu != out %zu\n", rb.size(), out_bytes);
        }
    }

    Stats s_warm = compute_stats(warm_v);
    Stats s_run = compute_stats(run_v);
    emit_json("tflite", a, t_create, t_first, s_warm, s_run,
              have_cmp ? &cmp : nullptr, out_count);

    TfLiteInterpreterDelete(interp);
    TfLiteInterpreterOptionsDelete(opts);
    TfLiteModelDelete(model);
    if (delegate) {
        if (use_xnn) TfLiteXNNPackDelegateDelete(delegate);
        else         TfLiteGpuDelegateV2Delete(delegate);
    }
    return 0;
}
