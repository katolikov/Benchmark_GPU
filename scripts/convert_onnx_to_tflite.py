#!/usr/bin/env python3
"""ONNX -> TFLite (fp32) one-time conversion helper.

This is the ONLY Python step in the project. The benchmark itself runs as
a pure-C++ binary on the Android device. Use this script once per model
to obtain a `.tflite` file from your `.onnx`; everything after that is C++.

Pipeline: ONNX -> TF SavedModel (via onnx2tf) -> TFLite (fp32).

Usage:
  python3 convert_onnx_to_tflite.py model.onnx out_dir

Requirements:
  python -m venv .venv && source .venv/bin/activate
  pip install onnx2tf tensorflow tf_keras onnx onnxsim "ml_dtypes>=0.5"
"""
from __future__ import annotations

import argparse
import os
import sys
import shutil

# onnx2tf unconditionally calls download_test_image_data() during conversion
# of any 4D-input model — and the bundled URL returns a stale/corrupt file
# in current numpy versions. We don't need that sample data (we don't enable
# `check_onnx_tf_outputs_elementwise_close`), so override that one function
# in both the utils module AND the onnx2tf module that imported it by name.
import numpy as _np
import onnx2tf
import onnx2tf.utils.common_functions as _cf

def _dummy_test_data():
    return _np.zeros((1, 224, 224, 3), dtype=_np.float32)

_cf.download_test_image_data = _dummy_test_data
# onnx2tf.onnx2tf does `from onnx2tf.utils.common_functions import (...
# download_test_image_data ...)` at import time, so patching the utils module
# isn't enough — patch the local binding too.
import onnx2tf.onnx2tf as _o2t
_o2t.download_test_image_data = _dummy_test_data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", help="Path to input .onnx file")
    ap.add_argument("out_dir", help="Output directory (will be created)")
    ap.add_argument("--name", default=None,
                    help="Output file basename (default: derived from onnx)")
    args = ap.parse_args()

    if not os.path.exists(args.onnx):
        print(f"ERROR: {args.onnx} not found", file=sys.stderr)
        return 1

    name = args.name or os.path.splitext(os.path.basename(args.onnx))[0]
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    work_dir = os.path.join(out_dir, ".onnx2tf_work")
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)

    print(f"[1/2] Converting ONNX -> TFLite via onnx2tf...")
    onnx2tf.convert(
        input_onnx_file_path=args.onnx,
        output_folder_path=work_dir,
        output_signaturedefs=True,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )

    # onnx2tf emits *_float32.tflite among other variants. Pick the fp32 one.
    fp32_tflite = None
    for f in os.listdir(work_dir):
        if f.endswith("_float32.tflite") or f == "model_float32.tflite":
            fp32_tflite = os.path.join(work_dir, f)
            break
    if not fp32_tflite:
        # fallback: any .tflite
        for f in os.listdir(work_dir):
            if f.endswith(".tflite"):
                fp32_tflite = os.path.join(work_dir, f)
                break
    if not fp32_tflite:
        print("ERROR: onnx2tf did not produce any .tflite", file=sys.stderr)
        return 2

    final = os.path.join(out_dir, f"{name}.tflite")
    shutil.copyfile(fp32_tflite, final)
    print(f"[2/2] Wrote {final} ({os.path.getsize(final):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
