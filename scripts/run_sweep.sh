#!/usr/bin/env bash
# Run the full benchmark sweep on the connected ADB device.
#
# Pure-C++ pipeline: input is generated deterministically inside each bench
# binary from --seed, and the *first* run for each framework dumps its CPU
# output to be used as the reference for the framework's GPU runs.
#
# Outputs:
#   results/results.jsonl   — one JSON line per (framework, backend, variant)
#   results/REPORT.md       — rendered by ./build/host/report

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WARMUP=${WARMUP:-5}
RUNS=${RUNS:-30}
SEED=${SEED:-42}
DEV_DIR=/data/local/tmp/bench
ENV="cd $DEV_DIR && LD_LIBRARY_PATH=."

mkdir -p "$PROJECT_ROOT/results"
JSONL="$PROJECT_ROOT/results/results.jsonl"
LOG="$PROJECT_ROOT/results/sweep.log"
: > "$JSONL"
: > "$LOG"

# Push references back to the host so we can stash them, but they live on
# device. On-device names are fixed.
TFLITE_CPU_REF="$DEV_DIR/ref_tflite_cpu.bin"
MNN_CPU_REF="$DEV_DIR/ref_mnn_cpu.bin"

run_one() {
    local cmd="$1"
    local label="$2"
    {
        echo "----- $label -----"
        echo "+ $cmd"
    } >> "$LOG"
    out=$(adb shell "$cmd" 2>>"$LOG" | tr -d '\r' | tee -a "$LOG" | grep '^{' | tail -1 || true)
    if [ -z "$out" ]; then
        echo "{\"framework\":\"?\",\"label\":\"$label\",\"error\":\"no JSON\"}" >> "$JSONL"
        echo "  FAILED $label (see $LOG)"
        return
    fi
    out=$(echo "$out" | sed -e "s/\"label\":\"[^\"]*\"/\"label\":\"$label\"/")
    echo "$out" >> "$JSONL"
    echo "  OK     $label"
}

echo "=== sweep: warmup=$WARMUP runs=$RUNS seed=$SEED ==="
adb shell "rm -f $DEV_DIR/*.cache $TFLITE_CPU_REF $MNN_CPU_REF" >/dev/null

# ---------- TFLite side -----------------------------------------------------
# 1) XNNPACK CPU FP32 — production-default mobile CPU; dumps the reference.
run_one "$ENV ./bench_tflite --model mobilenet_v2.tflite --backend xnnpack --threads 4 \
              --seed $SEED --warmup $WARMUP --runs $RUNS \
              --output $TFLITE_CPU_REF --label tflite_xnnpack_fp32" tflite_xnnpack_fp32

# 2) TFLite CPU FP32 (basic, no delegate) — sanity baseline.
run_one "$ENV ./bench_tflite --model mobilenet_v2.tflite --backend cpu --variant fp32 \
              --threads 4 --seed $SEED --warmup $WARMUP --runs $RUNS \
              --ref $TFLITE_CPU_REF --label tflite_cpu_fp32" tflite_cpu_fp32

# 3) TFLite GPU FP16 — TFLite's recommended best-perf GPU config.
run_one "$ENV ./bench_tflite --model mobilenet_v2.tflite --backend gpu --variant fp16 \
              --seed $SEED --warmup $WARMUP --runs $RUNS \
              --ref $TFLITE_CPU_REF --label tflite_gpu_fp16" tflite_gpu_fp16

# 4) TFLite GPU FP32 — max precision.
run_one "$ENV ./bench_tflite --model mobilenet_v2.tflite --backend gpu --variant fp32 \
              --seed $SEED --warmup $WARMUP --runs $RUNS \
              --ref $TFLITE_CPU_REF --label tflite_gpu_fp32" tflite_gpu_fp32

# 5) TFLite GPU FP16 with CL_ONLY flag — force OpenCL.
run_one "$ENV ./bench_tflite --model mobilenet_v2.tflite --backend gpu --variant cl \
              --seed $SEED --warmup $WARMUP --runs $RUNS \
              --ref $TFLITE_CPU_REF --label tflite_gpu_cl_fp16" tflite_gpu_cl_fp16

# ---------- MNN side --------------------------------------------------------
adb shell "rm -f $DEV_DIR/*.cache" >/dev/null

# 6) MNN CPU FP32 — dumps the reference for MNN GPU runs.
run_one "$ENV ./bench_mnn --model mobilenet_v2.mnn --backend cpu --variant fp32 \
              --threads 4 --seed $SEED --warmup $WARMUP --runs $RUNS \
              --output $MNN_CPU_REF --label mnn_cpu_fp32" mnn_cpu_fp32

# 7) MNN OpenCL FP16 buffer — MNN's recommended best-perf GPU config.
adb shell "rm -f $DEV_DIR/*.cache" >/dev/null
run_one "$ENV ./bench_mnn --model mobilenet_v2.mnn --backend opencl --variant buffer \
              --seed $SEED --warmup $WARMUP --runs $RUNS \
              --ref $MNN_CPU_REF --label mnn_opencl_buffer_fp16" mnn_opencl_buffer_fp16

# 8) MNN OpenCL FP16 image.
adb shell "rm -f $DEV_DIR/*.cache" >/dev/null
run_one "$ENV ./bench_mnn --model mobilenet_v2.mnn --backend opencl --variant image \
              --seed $SEED --warmup $WARMUP --runs $RUNS \
              --ref $MNN_CPU_REF --label mnn_opencl_image_fp16" mnn_opencl_image_fp16

# 9) MNN OpenCL FP32 buffer.
adb shell "rm -f $DEV_DIR/*.cache" >/dev/null
run_one "$ENV ./bench_mnn --model mobilenet_v2.mnn --backend opencl --variant fp32 \
              --seed $SEED --warmup $WARMUP --runs $RUNS \
              --ref $MNN_CPU_REF --label mnn_opencl_buffer_fp32" mnn_opencl_buffer_fp32

# 10) MNN Vulkan FP16.
adb shell "rm -f $DEV_DIR/*.cache" >/dev/null
run_one "$ENV ./bench_mnn --model mobilenet_v2.mnn --backend vulkan --variant fp16 \
              --seed $SEED --warmup $WARMUP --runs $RUNS \
              --ref $MNN_CPU_REF --label mnn_vulkan_fp16" mnn_vulkan_fp16

# 11) MNN Vulkan FP32.
adb shell "rm -f $DEV_DIR/*.cache" >/dev/null
run_one "$ENV ./bench_mnn --model mobilenet_v2.mnn --backend vulkan --variant fp32 \
              --seed $SEED --warmup $WARMUP --runs $RUNS \
              --ref $MNN_CPU_REF --label mnn_vulkan_fp32" mnn_vulkan_fp32

# ---------- Render report (Android-built C++ binary, run on device) --------
# We build report.cpp once for arm64-v8a alongside bench_*; on macOS hosts
# the system C++ toolchain is often broken so we just use the device.
adb push "$PROJECT_ROOT/build/android-arm64/report" "$DEV_DIR/" >/dev/null
adb push "$JSONL" "$DEV_DIR/results.jsonl" >/dev/null
adb shell "$ENV ./report results.jsonl REPORT.md" >/dev/null
adb pull "$DEV_DIR/REPORT.md" "$PROJECT_ROOT/results/REPORT.md" 2>/dev/null
echo "=== done. JSONL: $JSONL ; REPORT: $PROJECT_ROOT/results/REPORT.md ==="
