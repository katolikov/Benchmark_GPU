# Benchmark_GPU — MNN vs TFLite on Android (CPU / OpenCL / Vulkan)

A small, reproducible harness for running CNN inference on an Android device
through both **MNN** and **TFLite (LiteRT)** and comparing the two — timing
(create / warmup / inference percentiles) and numeric output (cosine, max
absolute error, argmax). Pure C++ on-device; only optional Python is the
one-time `ONNX → TFLite` model converter.

## What's included

| Path | Purpose |
|---|---|
| [bench/bench_tflite.cpp](bench/bench_tflite.cpp) | TFLite C-API bench (CPU, XNNPACK CPU, GPU FP16/FP32, GPU CL_ONLY) |
| [bench/bench_mnn.cpp](bench/bench_mnn.cpp) | MNN bench using `Express::Module` (CPU, OpenCL buffer/image, Vulkan, FP16/FP32) |
| [bench/common.h](bench/common.h) | Args parser, deterministic input generator, timing/comparison/JSON emit |
| [bench/report.cpp](bench/report.cpp) | Reads `results.jsonl`, emits Markdown report |
| [bench/CMakeLists.txt](bench/CMakeLists.txt) | One CMake project, two Android binaries + report tool |
| [scripts/run_bench.sh](scripts/run_bench.sh) | **Config-driven single-run benchmark** (the main entry point) |
| [scripts/run_sweep.sh](scripts/run_sweep.sh) | Run all 11 backend/variant combinations and render report |
| [scripts/convert_onnx_to_tflite.py](scripts/convert_onnx_to_tflite.py) | Optional one-time ONNX→TFLite step (only Python in the project) |
| [examples/configs/](examples/configs/) | Sample JSON configs for each backend |
| [libs/](libs/) | Prebuilt MNN + LiteRT 1.4.2 native libraries and C/C++ headers |

The runtime benchmark — input generation, model loading, inference loop,
output capture, comparison, JSON emission, Markdown rendering — is **all
C++**. The Python step is required only when you hand the runner a `.onnx`
and want to convert it to `.tflite` on the host (TFLite has no on-device
ONNX importer the way MNN does).

## How it's used (60-second tour)

```bash
# Build the on-device binaries once.
cmake -S bench -B build/android-arm64 \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 \
  -DCMAKE_BUILD_TYPE=Release && cmake --build build/android-arm64 -j8

# Run a single config-driven benchmark (TFLite GPU FP16):
scripts/run_bench.sh \
  --config examples/configs/tflite_gpu_fp16.json \
  --output-dir results/runs/$(date +%Y%m%d-%H%M%S)

# Run all 11 backend/variant combinations and render the comparison:
scripts/run_sweep.sh
```

Each `run_bench.sh` invocation produces:

```
<output-dir>/
├── results.json            single JSON line with timing + accuracy
├── output.bin              raw fp32 output tensor pulled from device
├── log.txt                 full ADB log of the run
├── config.resolved.json    flattened config that was actually used
└── REPORT.md               single-run markdown report
```

## Latest results (Galaxy Z Fold7, Exynos 2500, Xclipse 950)

MobileNetV2 (1.0_224, fp32, ImageNet weights). 5 warmup + 30 timed runs.
Reference for cosine = same-framework CPU FP32 dump.

| Path | Create | First inf. | Inf. p50 | Cosine vs CPU |
|---|---:|---:|---:|---:|
| tflite xnnpack fp32 | 20.7 | 22.8 | **3.68** | (reference) |
| tflite cpu fp32 (no delegate) | 4.7 | 87.9 | 115.94 | 1.0000 |
| **tflite gpu fp16** | 1212.6 | 14.0 | **4.37** | 0.9998 |
| tflite gpu fp32 | 1312.6 | 17.8 | 5.47 | 1.0000 |
| tflite gpu cl_only fp16 | 1260.5 | 14.8 | 4.42 | 0.9998 |
| mnn cpu fp32 | 21.8 | 21.7 | 6.13 | (reference) |
| **mnn opencl buffer fp16** | 251.4 | 2716.8 | **5.89** | 0.9999 |
| mnn opencl image fp16 | 381.4 | 1681.7 | 6.96 | 0.9999 |
| mnn opencl buffer fp32 | 252.2 | 2784.1 | 6.70 | 1.0000 |
| mnn vulkan fp16 | 302.4 | 5486.6 | 11.26 | 0.9998 |
| mnn vulkan fp32 | 331.5 | 5780.4 | 13.70 | 0.9998 |

Headline takeaways:

- **Steady-state**: TFLite GPU FP16 (4.37 ms p50) edges out MNN OpenCL buffer
  FP16 (5.89 ms). On this Xclipse 950 the two OpenCL paths are within ~1.5 ms.
- **Vulkan is ~3× slower than OpenCL** in MNN on this device — for Mali/
  Xclipse class GPUs the OpenCL path is clearly preferred for both frameworks.
- **CPU**: MNN's CPU (6.13 ms) is competitive with TFLite-XNNPACK (3.68 ms);
  the TFLite C API *without* the XNNPACK delegate is ~30× slower because it
  falls back to reference kernels — that path is for compatibility, not perf.
- **Cold start**: TFLite frontloads kernel compilation into `create()`
  (~1.25 s); MNN does it on **first inference** (~2-6 s for OpenCL/Vulkan).
  Cold totals are similar but the latency is split very differently.
- **Accuracy**: every backend predicts the same Top-1 class as its
  framework's CPU reference. FP16 GPU paths show the expected ~0.1 max-abs
  deviation (cosine ≥ 0.9998).

Full report at [results/REPORT.md](results/REPORT.md).

## Hardware tested

- **Device**: Samsung Galaxy Z Fold7 (SM-F766B)
- **SoC**: Samsung Exynos 2500 (s5e9955) — 1× Cortex-X5 + 5× A725 + 2× A720 + 1× A520
- **GPU**: Xclipse 950 (Vulkan + OpenCL, fp16 + i8mm + sve2)
- **OS**: Android 16
- **NDK**: 27.0.12077973 (clang)
- **MNN libs**: pulled from a separate MNN build; see [libs/mnn/](libs/mnn/)
- **TFLite libs**: LiteRT 1.4.2 (`libtensorflowlite_jni.so` + `libtensorflowlite_gpu_jni.so`)

## Reproducing on your device

See [HOWTO.md](HOWTO.md) for the full step-by-step. Short version:

```bash
git clone <this-repo> Benchmark_GPU && cd Benchmark_GPU
export ANDROID_NDK=$HOME/Library/Android/sdk/ndk/27.0.12077973   # adjust path
cmake -S bench -B build/android-arm64 \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 \
  -DCMAKE_BUILD_TYPE=Release && cmake --build build/android-arm64 -j8
scripts/run_sweep.sh
```

## Why two parallel models?

The MNN side starts from `torchvision.mobilenet_v2(IMAGENET1K_V1)` exported
to ONNX (NCHW). The TFLite side starts from `tf.keras.applications.MobileNetV2`
(NHWC). Both are MobileNetV2 (1.0_224, fp32) with valid ImageNet weights, but
they are not bit-identical — they were trained by different people on
different pipelines. We compare each backend to **its own framework's CPU
output** rather than across frameworks for that reason. Cross-framework
timing comparison is still apples-to-apples (same architecture, same input
distribution).

The model files live in [models/](models/); regenerate them only if you
want to swap MobileNetV2 for something else.

## License

Apache 2.0 (matches MNN and TFLite/LiteRT). Native libraries under [libs/](libs/)
are redistributed from their respective upstream projects under their original
licenses (MNN — Apache 2.0; LiteRT — Apache 2.0).
