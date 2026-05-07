# HOWTO — running the benchmark on your device

## 0. Prerequisites

- macOS or Linux host with `adb`, `cmake ≥ 3.22`, `bash`, `python3`
  (stdlib only — used by `scripts/run_bench.sh` to read JSON configs)
- Android NDK r27 or newer (export `ANDROID_NDK=...`)
- Android device connected via ADB, with `arm64-v8a` ABI
- For the *optional* ONNX→TFLite conversion: Python 3.12 and the deps in
  `requirements-convert.txt` (only needed if you give the runner a `.onnx`)

```bash
which adb cmake bash python3     # all should resolve
adb devices                      # at least one device shown
echo $ANDROID_NDK                # path to your NDK
```

## 1. Clone and build the on-device binaries

```bash
git clone <this-repo> Benchmark_GPU
cd Benchmark_GPU

cmake -S bench -B build/android-arm64 \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-30 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/android-arm64 -j8
```

This produces three binaries:

- `build/android-arm64/bench_tflite` — TFLite (LiteRT) bench, links against
  the prebuilt `libtensorflowlite_jni.so` / `libtensorflowlite_gpu_jni.so`
  in [libs/tflite/](libs/tflite/).
- `build/android-arm64/bench_mnn` — MNN bench using the modern
  `Express::Module` API. Links against the libs in [libs/mnn/](libs/mnn/).
- `build/android-arm64/report` — Markdown renderer; reads `results.jsonl`,
  emits `REPORT.md`. Runs on-device too (so you don't need a host C++
  toolchain to render the report).

## 2. The single-run runner — `scripts/run_bench.sh`

This is the main entry point. It mirrors the shape of MNN's
`LLM_Benchmark/run_benchmark.py` — pass a JSON config plus an output dir,
get back per-run results.

```bash
scripts/run_bench.sh \
  --config examples/configs/tflite_gpu_fp16.json \
  --output-dir results/runs/$(date +%Y%m%d-%H%M%S)
```

What it does, in order:

1. Reads the JSON config (with optional CLI overrides).
2. If `model.path` is `.onnx`, converts to `.tflite` via
   `scripts/convert_onnx_to_tflite.py` (one-time per model; cached under
   `<output-dir>/converted/`).
3. Pushes the model + chosen bench binary + libs to `/data/local/tmp/<remote_dir>`.
4. Runs the bench with the resolved config.
5. Pulls back: `results.json`, `output.bin`, `log.txt`, `REPORT.md`.
6. Writes `config.resolved.json` for reproducibility.

### CLI overrides

Every JSON field can be overridden on the command line — handy for sweeping
one variable without editing the file:

```bash
scripts/run_bench.sh \
  --config examples/configs/tflite_gpu_fp16.json \
  --output-dir out/threads4 \
  --threads 4 --runs 100 --seed 7
```

| Override flag | Replaces | Notes |
|---|---|---|
| `--framework {tflite\|mnn}` | `framework` | which bench binary to use |
| `--model PATH` | `model.path` | `.onnx`, `.tflite`, or `.mnn` (host path) |
| `--backend NAME` | `engine.backend` | tflite: `cpu`/`xnnpack`/`gpu`. mnn: `cpu`/`opencl`/`vulkan` |
| `--variant NAME` | `engine.variant` | tflite-gpu: `fp16`/`fp32`/`cl`. mnn-gpu: `buffer`/`image`/`fp16`/`fp32` |
| `--threads N` | `engine.threads` | CPU thread count |
| `--warmup N` | `benchmark.warmup` | number of warmup invocations |
| `--runs N` | `benchmark.runs` | number of timed invocations |
| `--seed N` | `benchmark.seed` | PRNG seed when no `--input` is supplied |
| `--input FILE` | `inputs[]` | repeatable. fp32 little-endian raw tensor |
| `--label TAG` | `label` | echoed in JSON; useful for grep/dashboarding |
| `--remote-dir NAME` | `device.remote_dir` | subdir under `/data/local/tmp/` |
| `--adb-serial S` | `device.adb_serial` | for multi-device hosts |

### Config schema

```jsonc
{
  "framework": "tflite",          // "tflite" or "mnn"

  "model": {
    "path": "models/mobilenet_v2.tflite"   // .onnx, .tflite, or .mnn
  },

  "inputs": [],                   // optional list of host paths to .bin tensors
                                  // (fp32 little-endian, length must match the
                                  // model's input tensor byte size). When empty,
                                  // the bench generates input on-device from
                                  // `benchmark.seed` using std::mt19937_64.

  "engine": {
    "backend": "gpu",             // see table above
    "variant": "fp16",            // see table above
    "threads": 4
  },

  "benchmark": {
    "warmup": 5,
    "runs": 30,
    "seed": 42
  },

  "device": {
    "remote_dir": "bench_run",    // path: /data/local/tmp/<remote_dir>
    "adb_serial": ""              // empty = default device
  },

  "label": "tflite_gpu_fp16",
  "reference": ""                 // optional: host path to reference output;
                                  // if set, bench will report cosine/max-abs
                                  // against this tensor.
}
```

### Sample configs

The [examples/configs/](examples/configs/) directory has ready-to-use configs
for each backend variant:

- `tflite_gpu_fp16.json` — TFLite GPU delegate, FP16 (recommended fast)
- `tflite_xnnpack.json` — TFLite + XNNPACK CPU delegate (mobile production CPU default)
- `tflite_from_onnx.json` — same as above but with an `.onnx` source — exercises the converter
- `mnn_opencl_fp16.json` — MNN OpenCL buffer FP16 (best MNN GPU config)

### Outputs

For each run the output directory will contain:

```
<output-dir>/
├── results.json           # one line of JSON; the same format emit_json() produces
├── output.bin             # the model's raw fp32 output tensor
├── log.txt                # full ADB-side log (linker warnings, MNN debug, errors)
├── config.resolved.json   # merged config (defaults + JSON + CLI overrides)
├── REPORT.md              # rendered single-run Markdown report
└── converted/             # only present if you supplied an .onnx
    ├── mobilenet_v2.tflite
    └── .onnx2tf_work/     # temp dir, safe to delete
```

## 3. The full sweep — `scripts/run_sweep.sh`

Runs all 11 (framework, backend, variant) combinations and renders a
side-by-side comparison report.

```bash
scripts/run_sweep.sh
```

Tweakables (env vars):

```bash
WARMUP=10 RUNS=100 SEED=1234 scripts/run_sweep.sh
```

Outputs:

- `results/results.jsonl` — one JSON line per backend
- `results/REPORT.md` — full comparison report
- `results/sweep.log` — full device log of the run

## 4. Optional: ONNX→TFLite converter setup

Only needed if you supply a `.onnx` to `run_bench.sh` instead of a `.tflite`.
Skip this section if you already have a `.tflite`.

```bash
# Create an isolated venv just for the conversion step.
python3.12 -m venv .venv-convert
source .venv-convert/bin/activate
pip install --upgrade pip
pip install onnx2tf onnx onnxsim sng4onnx onnx_graphsurgeon \
            tf_keras tensorflow==2.18.0 \
            'numpy<2' psutil 'ml_dtypes>=0.5'
deactivate
```

`run_bench.sh` automatically uses `.venv-convert/bin/python3` if it exists,
falling back to system `python3`. The conversion runs once and the result is
cached in `<output-dir>/converted/<basename>.tflite`.

The converter is `scripts/convert_onnx_to_tflite.py` and can be invoked
directly:

```bash
.venv-convert/bin/python scripts/convert_onnx_to_tflite.py \
    path/to/model.onnx path/to/out_dir
# ↑ produces path/to/out_dir/<basename>.tflite
```

## 5. Adding a new model

1. Drop the `.tflite` (or `.onnx`) into `models/`. For MNN, use either an
   `.mnn` directly or convert ONNX with the on-device `MNNConvert`:

   ```bash
   adb push my_model.onnx /data/local/tmp/MNN/
   adb shell "cd /data/local/tmp/MNN && LD_LIBRARY_PATH=. ./MNNConvert \
       -f ONNX --modelFile my_model.onnx --MNNModel my_model.mnn --bizCode bench"
   adb pull /data/local/tmp/MNN/my_model.mnn models/
   ```

2. Copy one of the example configs to `examples/configs/my_model.json` and
   update `model.path` and `label`.

3. Run it:

   ```bash
   scripts/run_bench.sh --config examples/configs/my_model.json \
                        --output-dir results/runs/my_model
   ```

The bench infers the model's input shape and either accepts your `--input`
file (must match the byte size of `tensor[0]`) or generates random fp32
data of the right size from the seed.

## 6. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ERROR: cannot stat 'build/android-arm64/bench_tflite'` | Run the CMake build in step 1 first. |
| `cannot locate symbol "_ZN3fmt..." referenced by /system/lib64/libinput.so` | Don't put `/system/lib64` on `LD_LIBRARY_PATH`. The runner already uses `LD_LIBRARY_PATH=.` only. |
| Cosine vs ref ≪ 1 on MNN CPU | Make sure you're using the `Express::Module` API (this repo does). The older Interpreter API mis-handles MobileNetV2 NCHW→NC4HW4 transitions. |
| `Module 'onnx2tf' has no attribute 'convert'` | The deprecated `ai-edge-torch` overrode `onnx2tf`. Recreate the venv (see §4). |
| `TypeError: ... ConcatV2 ... types [int32, int64] that don't all match` | ONNX permits mixed int32/int64 in `Concat` (commonly Shape outputs vs int32 constants); TF doesn't. The converter automatically (a) folds constants via `onnxsim` and (b) inserts `Cast` nodes before any remaining mixed Concat / Where. If your model still trips this, try `--skip-simplify` (sometimes onnxsim re-introduces mismatches) or inspect the failing op in Netron and add a manual Cast. |
| `WARNING: linker: Warning: unable to normalize "\/data/local/tmp/..."` | Cosmetic noise from Android's linker, ignore. |
| `Can't open file:mobilenet_v2.mnn.cache` | First-time MNN GPU runs print this before they create the cache; the next run will be faster. |
| `TFLite GPU delegate create failed` on Mali < G77 | Older Mali GPUs sometimes require `experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY`. Set `--variant cl` for OpenCL-only. |

## 7. Where to look in the code

- Add a new TFLite delegate option: [bench/bench_tflite.cpp](bench/bench_tflite.cpp), look for `gpu_opts.inference_priority1`.
- Add a new MNN backend option: [bench/bench_mnn.cpp](bench/bench_mnn.cpp), see the `MNN::ScheduleConfig` block.
- Change input generation: [bench/common.h](bench/common.h) — `generate_input_fp32()`.
- Change report layout: [bench/report.cpp](bench/report.cpp) — `render()`.
- Change which configs the sweep covers: [scripts/run_sweep.sh](scripts/run_sweep.sh) — list of `run_one` calls.
