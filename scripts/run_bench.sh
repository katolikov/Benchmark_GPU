#!/usr/bin/env bash
# Config-driven single-run benchmark on an Android device via ADB.
#
# Inspired by AMNN's LLM_Benchmark/run_benchmark.py — same shape (config JSON +
# output dir), but TFLite-focused and pure C++ on-device.
#
# Usage:
#   scripts/run_bench.sh --config examples/configs/tflite_gpu_fp16.json \
#                        --output-dir results/runs/$(date +%Y%m%d-%H%M%S)
#
#   # Override individual fields without editing the JSON:
#   scripts/run_bench.sh --config <cfg> --output-dir <dir> \
#       --model path/to/model.tflite --backend gpu --variant fp16 --runs 50
#
# What it does:
#   1. Reads JSON config (and CLI overrides).
#   2. If model is .onnx, calls scripts/convert_onnx_to_tflite.py to produce
#      a .tflite (the only Python touchpoint; convert once, reuse forever).
#   3. Pushes model + bench binaries + libs to /data/local/tmp/<remote_dir>.
#   4. Runs the chosen bench (bench_tflite or bench_mnn) on device with the
#      backend / variant / threads / runs / seed taken from the config.
#   5. Pulls back: results.json (one JSON line), output.bin (model output),
#      log.txt (full device log), config.resolved.json (flattened config).
#   6. Renders REPORT.md for the run via build/android-arm64/report (also
#      runs on the device, so no host C++ toolchain needed).
#
# Config schema (see examples/configs/*.json):
#   framework        : "tflite"  (TFLite + GPU/XNNPACK/CPU) or "mnn"
#   model.path       : path to .onnx or .tflite or .mnn (host)
#   inputs           : [optional list of host paths to .bin tensor files]
#   engine.backend   : tflite: cpu | xnnpack | gpu      mnn: cpu | opencl | vulkan
#   engine.variant   : tflite-gpu: fp16 | fp32 | cl     mnn-gpu: buffer | image | fp16 | fp32
#   engine.threads   : int (CPU thread count)
#   benchmark.warmup : int
#   benchmark.runs   : int
#   benchmark.seed   : uint64 — used when no --inputs file is supplied
#   device.remote_dir: subdir under /data/local/tmp/ (will be wiped each run)
#   device.adb_serial: optional ADB serial (defaults to $ANDROID_SERIAL or
#                       single connected device)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NDK="${ANDROID_NDK:-/Users/artemkatolikov/Library/Android/sdk/ndk/27.0.12077973}"

# ---------- args -----------------------------------------------------------
CONFIG=""
OUT_DIR=""
OVR_MODEL=""
OVR_BACKEND=""
OVR_VARIANT=""
OVR_THREADS=""
OVR_WARMUP=""
OVR_RUNS=""
OVR_SEED=""
OVR_FRAMEWORK=""
OVR_INPUTS=()
OVR_LABEL=""
OVR_REMOTE_DIR=""
OVR_SERIAL=""

usage() {
    cat <<EOF
Usage: $0 --config CFG.json --output-dir DIR [overrides...]

Required:
  --config FILE        JSON config (see examples/configs/)
  --output-dir DIR     Where to write results (created if missing)

Optional overrides (any of these wins over the config):
  --framework {tflite|mnn}
  --model PATH         .onnx | .tflite | .mnn  (host path)
  --backend NAME       tflite: cpu|xnnpack|gpu     mnn: cpu|opencl|vulkan
  --variant NAME       tflite-gpu: fp16|fp32|cl    mnn-gpu: buffer|image|fp16|fp32
  --threads N
  --warmup N
  --runs N
  --seed N             PRNG seed when --input is not provided
  --input FILE         Repeatable. Host path to a .bin tensor (fp32 little-endian).
  --label TAG          Free-form label echoed in JSON / used in filenames.
  --remote-dir NAME    Subdir under /data/local/tmp (default: bench_run)
  --adb-serial S       ADB serial for the target device.

Examples:
  $0 --config examples/configs/tflite_gpu_fp16.json --output-dir results/runs/gpu_fp16
  $0 --config examples/configs/tflite_xnnpack.json  --output-dir out --runs 100
EOF
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --config)       CONFIG="$2"; shift 2;;
        --output-dir)   OUT_DIR="$2"; shift 2;;
        --framework)    OVR_FRAMEWORK="$2"; shift 2;;
        --model)        OVR_MODEL="$2"; shift 2;;
        --backend)      OVR_BACKEND="$2"; shift 2;;
        --variant)      OVR_VARIANT="$2"; shift 2;;
        --threads)      OVR_THREADS="$2"; shift 2;;
        --warmup)       OVR_WARMUP="$2"; shift 2;;
        --runs)         OVR_RUNS="$2"; shift 2;;
        --seed)         OVR_SEED="$2"; shift 2;;
        --input)        OVR_INPUTS+=("$2"); shift 2;;
        --label)        OVR_LABEL="$2"; shift 2;;
        --remote-dir)   OVR_REMOTE_DIR="$2"; shift 2;;
        --adb-serial)   OVR_SERIAL="$2"; shift 2;;
        -h|--help)      usage;;
        *) echo "unknown arg: $1" >&2; usage;;
    esac
done

[ -n "$CONFIG"  ] || { echo "missing --config" >&2; usage; }
[ -n "$OUT_DIR" ] || { echo "missing --output-dir" >&2; usage; }
[ -f "$CONFIG"  ] || { echo "config not found: $CONFIG" >&2; exit 1; }

mkdir -p "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd)
LOG="$OUT_DIR/log.txt"
: > "$LOG"

CFG_PY="$PROJECT_ROOT/scripts/cfg.py"
cfg_get() { python3 "$CFG_PY" "$CONFIG" "$1" 2>/dev/null; }
pick()    { [ -n "${1:-}" ] && echo "$1" || echo "${2:-}"; }

FRAMEWORK=$(pick "$OVR_FRAMEWORK"  "$(cfg_get '.framework')")
MODEL=$(    pick "$OVR_MODEL"      "$(cfg_get '.model.path')")
BACKEND=$(  pick "$OVR_BACKEND"    "$(cfg_get '.engine.backend')")
VARIANT=$(  pick "$OVR_VARIANT"    "$(cfg_get '.engine.variant')")
THREADS=$(  pick "$OVR_THREADS"    "$(cfg_get '.engine.threads')")
WARMUP=$(   pick "$OVR_WARMUP"     "$(cfg_get '.benchmark.warmup')")
RUNS=$(     pick "$OVR_RUNS"       "$(cfg_get '.benchmark.runs')")
SEED=$(     pick "$OVR_SEED"       "$(cfg_get '.benchmark.seed')")
LABEL=$(    pick "$OVR_LABEL"      "$(cfg_get '.label')")
REMOTE_DIR=$(pick "$OVR_REMOTE_DIR" "$(cfg_get '.device.remote_dir')")
SERIAL=$(   pick "$OVR_SERIAL"     "$(cfg_get '.device.adb_serial')")

# defaults
FRAMEWORK=${FRAMEWORK:-tflite}
THREADS=${THREADS:-4}
WARMUP=${WARMUP:-5}
RUNS=${RUNS:-30}
SEED=${SEED:-42}
REMOTE_DIR=${REMOTE_DIR:-bench_run}
LABEL=${LABEL:-${FRAMEWORK}_${BACKEND:-cpu}_${VARIANT:-default}}

# inputs: CLI overrides > config list
INPUTS=("${OVR_INPUTS[@]}")
if [ ${#INPUTS[@]} -eq 0 ]; then
    while IFS= read -r line; do
        [ -n "$line" ] && INPUTS+=("$line")
    done < <(cfg_get '.inputs[]')
fi

[ -n "$MODEL"   ] || { echo "model.path missing in config (or --model override)" >&2; exit 1; }
[ -n "$BACKEND" ] || { echo "engine.backend missing in config (or --backend override)" >&2; exit 1; }
[ -f "$MODEL"   ] || { echo "model file not found: $MODEL" >&2; exit 1; }

# ---------- adb wrapper ----------------------------------------------------
ADB="adb"
[ -n "$SERIAL" ] && ADB="adb -s $SERIAL"

# ---------- model conversion (only if framework=tflite and model=.onnx) ----
ext="${MODEL##*.}"
TFLITE=""
ONNX=""
MNN=""
case "$FRAMEWORK" in
    tflite)
        if [ "$ext" = "tflite" ]; then
            TFLITE="$MODEL"
        elif [ "$ext" = "onnx" ]; then
            cache_dir="$OUT_DIR/converted"
            mkdir -p "$cache_dir"
            base=$(basename "$MODEL" .onnx)
            TFLITE="$cache_dir/${base}.tflite"
            if [ ! -f "$TFLITE" ]; then
                echo "[convert] $MODEL -> $TFLITE  (one-time, via onnx2tf)" | tee -a "$LOG"
                # Prefer the project's local venv (.venv-convert) for onnx2tf —
                # see HOWTO.md for set-up. Fallback to system python3.
                PY="python3"
                if [ -x "$PROJECT_ROOT/.venv-convert/bin/python3" ]; then
                    PY="$PROJECT_ROOT/.venv-convert/bin/python3"
                fi
                OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES TF_CPP_MIN_LOG_LEVEL=2 \
                    "$PY" "$PROJECT_ROOT/scripts/convert_onnx_to_tflite.py" "$MODEL" "$cache_dir" \
                    --name "$base" 2>&1 | tee -a "$LOG"
            fi
        else
            echo "TFLite expects .tflite or .onnx, got .$ext" >&2; exit 1
        fi
        ;;
    mnn)
        if [ "$ext" = "mnn" ]; then
            MNN="$MODEL"
        elif [ "$ext" = "onnx" ]; then
            ONNX="$MODEL"
            cache_dir="$OUT_DIR/converted"
            mkdir -p "$cache_dir"
            base=$(basename "$MODEL" .onnx)
            MNN="$cache_dir/${base}.mnn"
            if [ ! -f "$MNN" ]; then
                # Use the on-device MNNConvert (in /data/local/tmp/MNN/) so we
                # don't need a host MNN build. Push -> convert -> pull.
                if ! $ADB shell "[ -x /data/local/tmp/MNN/MNNConvert ]" 2>/dev/null; then
                    echo "ERROR: /data/local/tmp/MNN/MNNConvert missing on device. " \
                         "Push MNN's prebuilt MNNConvert binary there or supply a .mnn directly." >&2
                    exit 1
                fi
                echo "[convert] $MODEL -> $MNN  (via on-device MNNConvert)" | tee -a "$LOG"
                $ADB push "$MODEL" /data/local/tmp/MNN/_in.onnx >>"$LOG" 2>&1
                $ADB shell "cd /data/local/tmp/MNN && LD_LIBRARY_PATH=. ./MNNConvert -f ONNX \
                    --modelFile _in.onnx --MNNModel _out.mnn --bizCode bench" 2>&1 | tee -a "$LOG"
                $ADB pull /data/local/tmp/MNN/_out.mnn "$MNN" >>"$LOG" 2>&1
                $ADB shell "rm -f /data/local/tmp/MNN/_in.onnx /data/local/tmp/MNN/_out.mnn" \
                    >>"$LOG" 2>&1
            fi
        else
            echo "MNN expects .mnn or .onnx, got .$ext" >&2; exit 1
        fi
        ;;
    *) echo "unknown framework: $FRAMEWORK" >&2; exit 1;;
esac

# ---------- push artifacts -------------------------------------------------
DEV_DIR=/data/local/tmp/$REMOTE_DIR
$ADB shell "rm -rf $DEV_DIR && mkdir -p $DEV_DIR"
echo "[push] $DEV_DIR" | tee -a "$LOG"

LIBS_HOST="$PROJECT_ROOT/libs"
BINS_HOST="$PROJECT_ROOT/build/android-arm64"
if [ ! -x "$BINS_HOST/bench_tflite" ] || [ ! -x "$BINS_HOST/bench_mnn" ] || [ ! -x "$BINS_HOST/report" ]; then
    echo "Bench binaries missing under $BINS_HOST. Run:" >&2
    echo "  cmake -S bench -B build/android-arm64 \\" >&2
    echo "    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \\" >&2
    echo "    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 \\" >&2
    echo "    -DCMAKE_BUILD_TYPE=Release && cmake --build build/android-arm64 -j8" >&2
    exit 1
fi

LIBCPP="$NDK/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"

push_one() { $ADB push "$1" "$DEV_DIR/$(basename ${2:-$1})" 2>&1 | tail -1 | tee -a "$LOG"; }

case "$FRAMEWORK" in
    tflite)
        push_one "$BINS_HOST/bench_tflite"
        push_one "$LIBS_HOST/tflite/arm64-v8a/libtensorflowlite_jni.so"
        push_one "$LIBS_HOST/tflite/arm64-v8a/libtensorflowlite_gpu_jni.so"
        push_one "$TFLITE" "model.tflite"
        ;;
    mnn)
        push_one "$BINS_HOST/bench_mnn"
        push_one "$LIBS_HOST/mnn/arm64-v8a/libMNN.so"
        push_one "$LIBS_HOST/mnn/arm64-v8a/libMNN_Express.so"
        push_one "$LIBS_HOST/mnn/arm64-v8a/libMNN_CL.so"
        push_one "$LIBS_HOST/mnn/arm64-v8a/libMNN_Vulkan.so"
        push_one "$MNN" "model.mnn"
        ;;
esac
push_one "$BINS_HOST/report"
push_one "$LIBCPP"

# Push any host inputs into device. Numbered: input_0.bin, input_1.bin, ...
INPUT_ARGS=""
i=0
for inp in "${INPUTS[@]}"; do
    [ -f "$inp" ] || { echo "input not found: $inp" >&2; exit 1; }
    push_one "$inp" "input_${i}.bin"
    if [ $i -eq 0 ]; then INPUT_ARGS="--input input_0.bin"; fi
    i=$((i+1))
done
if [ -z "$INPUT_ARGS" ]; then
    INPUT_ARGS="--seed $SEED"   # generate on-device from PRNG
fi

# ---------- run ------------------------------------------------------------
case "$FRAMEWORK" in
    tflite) BIN=bench_tflite; MODEL_DEV=model.tflite;;
    mnn)    BIN=bench_mnn;    MODEL_DEV=model.mnn;;
esac

CMD="cd $DEV_DIR && LD_LIBRARY_PATH=. ./$BIN \
    --model $MODEL_DEV \
    $INPUT_ARGS \
    --backend $BACKEND \
    ${VARIANT:+--variant $VARIANT} \
    --threads $THREADS \
    --warmup $WARMUP \
    --runs $RUNS \
    --output output.bin \
    --label $LABEL"

# Add --ref if a reference was provided in the config or as input_1.bin
REF=$(cfg_get '.reference // empty')
if [ -n "$REF" ]; then
    if [ "$REF" = "self" ]; then
        :  # no-op; first run dumps its own output as reference is meaningless
    elif [ -f "$REF" ]; then
        push_one "$REF" "ref.bin"
        CMD="$CMD --ref ref.bin"
    fi
fi

echo "[run] $CMD" | tee -a "$LOG"
JSON_LINE=$($ADB shell "$CMD" 2>>"$LOG" | tr -d '\r' | tee -a "$LOG" | grep '^{' | tail -1 || true)

if [ -z "$JSON_LINE" ]; then
    echo "FAIL: bench did not emit JSON. See $LOG" >&2
    exit 3
fi
echo "$JSON_LINE" > "$OUT_DIR/results.json"

# Pull results
$ADB pull "$DEV_DIR/output.bin" "$OUT_DIR/output.bin" 2>>"$LOG" || true

# Render single-run report on device (no host C++ toolchain needed)
$ADB push "$OUT_DIR/results.json" "$DEV_DIR/results.jsonl" >>"$LOG" 2>&1
$ADB shell "cd $DEV_DIR && LD_LIBRARY_PATH=. ./report results.jsonl REPORT.md" >>"$LOG" 2>&1
$ADB pull "$DEV_DIR/REPORT.md" "$OUT_DIR/REPORT.md" 2>>"$LOG" || true

# Save the resolved configuration for reproducibility
cat > "$OUT_DIR/config.resolved.json" <<JSON
{
  "framework": "$FRAMEWORK",
  "model": {"path": "$MODEL", "device_path": "$DEV_DIR/$MODEL_DEV"},
  "engine": {"backend": "$BACKEND", "variant": "$VARIANT", "threads": $THREADS},
  "benchmark": {"warmup": $WARMUP, "runs": $RUNS, "seed": $SEED},
  "label": "$LABEL",
  "device": {"remote_dir": "$DEV_DIR", "adb_serial": "$SERIAL"}
}
JSON

echo
echo "==========================================================="
echo "  output:   $OUT_DIR"
echo "  results:  $OUT_DIR/results.json"
echo "  output:   $OUT_DIR/output.bin"
echo "  log:      $OUT_DIR/log.txt"
echo "  report:   $OUT_DIR/REPORT.md"
echo "==========================================================="
python3 "$CFG_PY" "$OUT_DIR/results.json" summarize
