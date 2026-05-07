#!/usr/bin/env python3
"""ONNX -> TFLite (fp32) one-time conversion helper.

This is the ONLY Python step in the project. The benchmark itself runs as
a pure-C++ binary on the Android device. Use this script once per model
to obtain a `.tflite` file from your `.onnx`; everything after that is C++.

Pipeline:  ONNX
       -> [pre]  fold constants with onnx-simplifier
       -> [pre]  patch any int32/int64 dtype mismatch in Concat / Where / If
                 inputs (TF refuses; ONNX permits it)
       -> [convert] TF SavedModel via onnx2tf
       -> [export]  TFLite fp32 (no quant)

Usage:
  python3 convert_onnx_to_tflite.py MODEL.onnx OUT_DIR [--name NAME]
                                  [--skip-simplify] [--skip-patch]

Requirements (see ../requirements-convert.txt):
  python -m venv .venv-convert && source .venv-convert/bin/activate
  pip install -r requirements-convert.txt
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import List

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


# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------
def simplify_onnx(model_path: str, out_path: str) -> str:
    """Fold constants and resolve shape ops via onnx-simplifier.

    Most int32/int64 mismatches come from Shape -> Cast -> Concat chains
    where one branch is an int64 Shape output and the other is an int32
    constant. Constant folding usually resolves these into a single
    consistent dtype, so this step is run before the type-coercion patch
    below.
    """
    import onnx
    from onnxsim import simplify

    model = onnx.load(model_path)
    simplified, ok = simplify(model)
    if not ok:
        # Simplifier failed — keep going with the un-simplified model.
        print("  (onnxsim: simplify failed — keeping original graph)")
        onnx.save(model, out_path)
        return out_path
    onnx.save(simplified, out_path)
    return out_path


def _make_unique_name(graph, prefix: str) -> str:
    used = {n.name for n in graph.node}
    used.update({n.name for n in graph.initializer})
    used.update({i.name for i in graph.input})
    used.update({o.name for o in graph.output})
    i = 0
    while True:
        cand = f"{prefix}_{i}"
        if cand not in used:
            return cand
        i += 1


def patch_concat_dtype_mismatch(model_path: str, out_path: str) -> str:
    """Insert Cast nodes to coerce int64 inputs of mixed-dtype Concat / Where /
    If branches to match their siblings.

    This is the actual fix for the user-reported error::

        TypeError: Tensors in list passed to 'values' of 'ConcatV2' Op have
        types [int32, int64] that don't all match.

    Strategy:
      1. Run shape+type inference on the model.
      2. For each Concat / Where node, gather the inferred dtypes of its
         inputs.
      3. If they disagree AND the disagreement is int32 ↔ int64, insert a
         Cast node so every input matches the dominant dtype (we pick int32
         when one branch is int32, since TF tends to prefer int32 for shapes).
    """
    import onnx
    from onnx import TensorProto, helper, shape_inference

    model = onnx.load(model_path)
    try:
        model = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    except Exception as e:
        print(f"  (shape_inference failed: {e}; trying without data_prop)")
        try:
            model = shape_inference.infer_shapes(model)
        except Exception as ee:
            print(f"  (shape_inference failed entirely: {ee}; skipping dtype patch)")
            onnx.save(model, out_path)
            return out_path

    graph = model.graph

    # Build a name -> elem_type table from initializers, inputs, value_info, outputs.
    dtype_of: dict = {}

    def _add(name: str, elem_type: int):
        if name and elem_type:
            dtype_of[name] = elem_type

    for init in graph.initializer:
        _add(init.name, init.data_type)
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if vi.type.tensor_type.elem_type:
            _add(vi.name, vi.type.tensor_type.elem_type)

    # Constant nodes — record their output dtype too.
    for n in graph.node:
        if n.op_type == "Constant":
            for attr in n.attribute:
                if attr.name == "value" and attr.t and attr.t.data_type:
                    _add(n.output[0], attr.t.data_type)

    # Walk nodes; emit a patched node list with Casts injected as needed.
    new_nodes: List = []
    patches = 0

    INT32 = TensorProto.INT32
    INT64 = TensorProto.INT64

    for node in graph.node:
        if node.op_type not in ("Concat", "Where"):
            new_nodes.append(node)
            continue

        in_types = [dtype_of.get(i, 0) for i in node.input]
        # If we couldn't infer at least one type, leave it alone.
        if not all(in_types):
            new_nodes.append(node)
            continue

        # Only patch the int32 ↔ int64 case. Anything else is a real bug.
        unique = set(in_types)
        if unique == {INT32, INT64}:
            target = INT32   # TF prefers int32 for shape-like tensors
            new_inputs = []
            for in_name, in_dt in zip(node.input, in_types):
                if in_dt == target:
                    new_inputs.append(in_name)
                    continue
                cast_out = _make_unique_name(graph, f"{in_name}_cast_i32")
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[in_name],
                    outputs=[cast_out],
                    to=target,
                    name=_make_unique_name(graph, "Cast_dtype_patch"),
                )
                new_nodes.append(cast_node)
                new_inputs.append(cast_out)
                _add(cast_out, target)
                patches += 1
            patched = helper.make_node(
                node.op_type,
                inputs=new_inputs,
                outputs=list(node.output),
                name=node.name,
                **{a.name: helper.get_attribute_value(a) for a in node.attribute},
            )
            new_nodes.append(patched)
        else:
            new_nodes.append(node)

    if patches:
        # rebuild graph node list
        del graph.node[:]
        graph.node.extend(new_nodes)
        print(f"  inserted {patches} Cast node(s) to fix int32/int64 mismatches")

    onnx.save(model, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", help="Path to input .onnx file")
    ap.add_argument("out_dir", help="Output directory (will be created)")
    ap.add_argument("--name", default=None,
                    help="Output file basename (default: derived from onnx)")
    ap.add_argument("--skip-simplify", action="store_true",
                    help="Skip onnxsim constant folding pass")
    ap.add_argument("--skip-patch", action="store_true",
                    help="Skip int32/int64 Concat/Where coercion pass")
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
    os.makedirs(work_dir, exist_ok=True)

    onnx_in = args.onnx

    if not args.skip_simplify:
        print("[1/3] Simplifying ONNX (onnxsim — fold constants)...")
        sim = os.path.join(work_dir, f"{name}.simplified.onnx")
        onnx_in = simplify_onnx(onnx_in, sim)
    else:
        print("[1/3] Skipping onnxsim (per --skip-simplify)")

    if not args.skip_patch:
        print("[2/3] Patching int32/int64 Concat/Where dtype mismatches...")
        patched = os.path.join(work_dir, f"{name}.patched.onnx")
        onnx_in = patch_concat_dtype_mismatch(onnx_in, patched)
    else:
        print("[2/3] Skipping dtype patch (per --skip-patch)")

    print("[3/3] Converting ONNX -> TFLite via onnx2tf...")
    try:
        onnx2tf.convert(
            input_onnx_file_path=onnx_in,
            output_folder_path=work_dir,
            output_signaturedefs=True,
            copy_onnx_input_output_names_to_tflite=True,
            disable_strict_mode=True,
            non_verbose=True,
        )
    except Exception as e:
        # Common message we want to give actionable help on:
        msg = str(e)
        if "ConcatV2" in msg or "don't all match" in msg:
            print()
            print("=" * 68, file=sys.stderr)
            print("CONVERSION FAILED with a Concat dtype-mismatch error.", file=sys.stderr)
            print("Even after onnxsim + the int32/int64 patch, the TF graph still",
                  file=sys.stderr)
            print("had mismatched dtypes. Possible next steps:", file=sys.stderr)
            print("  - Try `--skip-simplify` (sometimes onnxsim re-introduces them)",
                  file=sys.stderr)
            print("  - Inspect the failing op in Netron and add a Cast manually",
                  file=sys.stderr)
            print("  - Pass a `param_replacement_file` to onnx2tf for that op",
                  file=sys.stderr)
            print("=" * 68, file=sys.stderr)
        raise

    # onnx2tf emits *_float32.tflite among other variants. Pick the fp32 one.
    fp32_tflite = None
    for f in sorted(os.listdir(work_dir)):
        if f.endswith("_float32.tflite") or f == "model_float32.tflite":
            fp32_tflite = os.path.join(work_dir, f)
            break
    if not fp32_tflite:
        for f in sorted(os.listdir(work_dir)):
            if f.endswith(".tflite"):
                fp32_tflite = os.path.join(work_dir, f)
                break
    if not fp32_tflite:
        print("ERROR: onnx2tf did not produce any .tflite", file=sys.stderr)
        return 2

    final = os.path.join(out_dir, f"{name}.tflite")
    shutil.copyfile(fp32_tflite, final)
    print(f"OK    Wrote {final} ({os.path.getsize(final):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
