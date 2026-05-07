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


# Ops that require all (relevant) inputs to share a dtype in TF/TFLite.
# We only look at these when scanning for mismatches — touching anything
# else (Cast, Reshape, ...) would be wrong.
_DTYPE_UNIFY_OPS = {
    # binary arithmetic
    "Add", "Sub", "Mul", "Div", "Pow", "Mod",
    # variadic / reductions across inputs
    "Min", "Max", "Mean", "Sum",
    # collection / control-flow
    "Concat", "Where",
    # element-wise comparisons (output is bool but inputs must match)
    "Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual",
    # logical
    "And", "Or", "Xor",
    # matmul + Gemm — also dtype-strict in TF
    "MatMul",
}

# For most ops, every input participates in dtype unification. Where is the
# exception: input[0] is the bool condition; only inputs[1:3] should match.
_UNIFY_INDICES = {
    "Where": [1, 2],
}


def _build_dtype_table(graph) -> dict:
    """Walk the graph in topological order and return a name -> elem_type map.

    More forgiving than `onnx.shape_inference.infer_shapes` alone — that pass
    sometimes drops the dtype of Cast outputs (especially when the producing
    Cast precedes a node it can't fully shape-infer). We always trust:
      * graph initializers
      * graph inputs / outputs / value_info
      * Cast node attribute `to`
      * Constant / ConstantOfShape node `value` tensor's data_type
      * Shape / Size / ArgMax / ArgMin / NonZero -> int64
      * comparison / logical ops -> bool
      * everything else: inherit from first known input dtype
    """
    import onnx
    from onnx import TensorProto

    dt: dict = {}

    def add(name: str, dtype: int):
        if name and dtype:
            dt[name] = dtype

    for init in graph.initializer:
        add(init.name, init.data_type)
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if vi.type.tensor_type.elem_type:
            add(vi.name, vi.type.tensor_type.elem_type)

    BOOL_OUT = {"Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual",
                "And", "Or", "Xor", "Not", "IsInf", "IsNaN"}
    INT64_OUT = {"Shape", "Size", "ArgMax", "ArgMin", "NonZero"}

    for node in graph.node:
        if not node.output:
            continue
        out_dt = 0

        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to":
                    out_dt = attr.i
                    break
        elif node.op_type in ("Constant", "ConstantOfShape"):
            for attr in node.attribute:
                if attr.name == "value" and attr.t and attr.t.data_type:
                    out_dt = attr.t.data_type
                    break
        elif node.op_type in INT64_OUT:
            out_dt = TensorProto.INT64
        elif node.op_type in BOOL_OUT:
            out_dt = TensorProto.BOOL
        elif node.op_type == "Range":
            if node.input and node.input[0] in dt:
                out_dt = dt[node.input[0]]
        elif node.op_type == "TopK":
            # outputs[0] is values (= input[0] dtype), outputs[1] is int64 indices
            if node.input and node.input[0] in dt:
                add(node.output[0], dt[node.input[0]])
            if len(node.output) > 1:
                add(node.output[1], TensorProto.INT64)
            continue
        else:
            # Inherit from first input we have a dtype for. Covers Conv,
            # MatMul, Add, Mul, Concat, Reshape, Transpose, BatchNorm, …
            for inp in node.input:
                if inp and inp in dt:
                    out_dt = dt[inp]
                    break

        if out_dt:
            for outname in node.output:
                add(outname, out_dt)

    return dt


def force_fp32(model_path: str, out_path: str) -> str:
    """Aggressively strip fp16 / bf16 from the ONNX graph.

    Use this when the dtype-coercion patch alone isn't enough — most often
    when onnx2tf re-introduces fp16 internally for a partial-precision
    model. Effects on the saved model:

      * fp16 initializers -> fp32 initializers (lossless upcast)
      * Cast(to=FLOAT16) / Cast(to=BFLOAT16) -> Cast(to=FLOAT)
      * graph.input / output / value_info entries with fp16 -> fp32

    This eliminates every fp16 boundary, so any subsequent fp16/fp32
    Concat/Div/Mul mismatch literally cannot exist. The performance cost
    is paid only by the converter — TFLite's GPU delegate will still run
    in fp16 at runtime when `is_precision_loss_allowed = true`.
    """
    import onnx
    import numpy as np
    from onnx import TensorProto, numpy_helper

    model = onnx.load(model_path)
    graph = model.graph

    # 1. Initializers
    converted_inits = 0
    for init in graph.initializer:
        if init.data_type in (TensorProto.FLOAT16, TensorProto.BFLOAT16):
            arr = numpy_helper.to_array(init).astype(np.float32)
            new_init = numpy_helper.from_array(arr, name=init.name)
            init.CopyFrom(new_init)
            converted_inits += 1

    # 2. Cast nodes targeting fp16/bf16 -> retarget to fp32
    converted_casts = 0
    for node in graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i in (TensorProto.FLOAT16, TensorProto.BFLOAT16):
                    attr.i = TensorProto.FLOAT
                    converted_casts += 1
                    break
        elif node.op_type == "Constant":
            # Constant nodes can have a "value" attribute holding a fp16 tensor.
            for attr in node.attribute:
                if attr.name == "value" and attr.t and attr.t.data_type in (
                        TensorProto.FLOAT16, TensorProto.BFLOAT16):
                    arr = numpy_helper.to_array(attr.t).astype(np.float32)
                    attr.t.CopyFrom(numpy_helper.from_array(arr))
                    converted_inits += 1

    # 3. Graph IO / value_info
    converted_vi = 0
    for vi_list in (graph.input, graph.output, graph.value_info):
        for vi in vi_list:
            if vi.type.tensor_type.elem_type in (TensorProto.FLOAT16, TensorProto.BFLOAT16):
                vi.type.tensor_type.elem_type = TensorProto.FLOAT
                converted_vi += 1

    print(f"  force-fp32: rewrote {converted_inits} initializer(s), "
          f"{converted_casts} Cast(to=fp16), {converted_vi} value_info entries")
    onnx.save(model, out_path)
    return out_path

def patch_concat_dtype_mismatch(model_path: str, out_path: str,
                                verbose: bool = False) -> str:
    """Insert Cast nodes so any mixed-dtype binary/variadic op has matching
    input dtypes. Handles two common ONNX→TF stumbling blocks:

      * int32 vs int64 — usually Shape (int64) meeting an int32 constant.
      * float16 vs float32 — model uses partial fp16 (an FP16 region inside
        an otherwise FP32 graph); ONNX permits the boundary, TF doesn't.

    For each Concat / Add / Mul / Div / Where / ... node whose participating
    inputs disagree, insert a Cast on the minority branch:

        int32  ↔ int64     →  cast int64 → int32  (TF prefers int32 for shape ops)
        float16 ↔ float32  →  cast float16 → float32
        otherwise          →  cast all to the widest type (numpy promotion)

    If we can't infer a dtype for at least one input, leave the node alone —
    silently doing the wrong thing is worse than letting TF raise.
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

    # Topological dtype propagation (handles Cast / Constant / Shape / etc.).
    # This is more reliable than reading value_info directly — onnx2tf often
    # converts models whose intermediate tensors don't have full type info.
    dtype_of = _build_dtype_table(graph)

    def _add(name: str, elem_type: int):
        if name and elem_type:
            dtype_of[name] = elem_type

    # NumPy-style promotion ranks. Higher wins. Special-case: when an op
    # only mixes int32+int64 the override below picks int32 instead.
    INT32 = TensorProto.INT32
    INT64 = TensorProto.INT64
    FLOAT16 = TensorProto.FLOAT16
    BFLOAT16 = TensorProto.BFLOAT16
    FLOAT = TensorProto.FLOAT
    DOUBLE = TensorProto.DOUBLE
    promote_rank = {
        TensorProto.BOOL: 0,
        TensorProto.UINT8: 10, TensorProto.INT8: 11,
        TensorProto.UINT16: 20, TensorProto.INT16: 21,
        TensorProto.UINT32: 30, INT32: 31,
        TensorProto.UINT64: 40, INT64: 41,
        BFLOAT16: 50, FLOAT16: 51,
        FLOAT: 60,
        DOUBLE: 70,
    }

    def _pick_target(op_type: str, in_types: List[int]) -> int:
        types = set(in_types)
        if not types:
            return 0
        # TF expects int32 shape tensors for Concat/Where, even though int64
        # has higher rank.
        if op_type in ("Concat", "Where") and types == {INT32, INT64}:
            return INT32
        # FP16 mixed with FP32 → upcast to FP32. Keeps TFLite happy and
        # avoids silent precision loss.
        if FLOAT in types and (FLOAT16 in types or BFLOAT16 in types):
            return FLOAT
        # General case: NumPy-style promotion to the widest dtype.
        return max(types, key=lambda t: promote_rank.get(t, 0))

    new_nodes: List = []
    patches = 0

    for node in graph.node:
        if node.op_type not in _DTYPE_UNIFY_OPS:
            new_nodes.append(node)
            continue

        # Which input indices participate in dtype unification?
        all_indices = list(range(len(node.input)))
        unify_idxs = _UNIFY_INDICES.get(node.op_type, all_indices)
        # Filter out empty optional inputs (ONNX uses "" for "not provided").
        unify_idxs = [i for i in unify_idxs if i < len(node.input) and node.input[i]]

        in_types = [dtype_of.get(node.input[i], 0) for i in unify_idxs]
        if not all(in_types) or len(set(in_types)) <= 1:
            new_nodes.append(node)
            continue

        target = _pick_target(node.op_type, in_types)
        if not target:
            new_nodes.append(node)
            continue

        new_inputs = list(node.input)
        for idx, in_dt in zip(unify_idxs, in_types):
            if in_dt == target:
                continue
            in_name = node.input[idx]
            cast_out = _make_unique_name(graph, f"{in_name}_cast_{target}")
            cast_node = helper.make_node(
                "Cast",
                inputs=[in_name],
                outputs=[cast_out],
                to=target,
                name=_make_unique_name(graph, "Cast_dtype_patch"),
            )
            new_nodes.append(cast_node)
            new_inputs[idx] = cast_out
            _add(cast_out, target)
            patches += 1
            if verbose:
                print(f"    [patch] {node.op_type:8} {node.name or '<unnamed>':40}"
                      f" cast {in_name} ({in_dt}) -> {target}")

        patched = helper.make_node(
            node.op_type,
            inputs=new_inputs,
            outputs=list(node.output),
            name=node.name,
            **{a.name: helper.get_attribute_value(a) for a in node.attribute},
        )
        new_nodes.append(patched)

    if patches:
        del graph.node[:]
        graph.node.extend(new_nodes)
        print(f"  inserted {patches} Cast node(s) to fix dtype mismatches")

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
                    help="Skip int32/int64 + fp16/fp32 dtype coercion pass")
    ap.add_argument("--force-fp32", action="store_true",
                    help="Strip all fp16/bf16 from the ONNX graph (lossless "
                         "upcast). Use when dtype-patch isn't enough — onnx2tf "
                         "occasionally introduces fp16 boundaries internally")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Print every Cast inserted by the dtype-patch")
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
        print("[1/4] Simplifying ONNX (onnxsim — fold constants)...")
        sim = os.path.join(work_dir, f"{name}.simplified.onnx")
        onnx_in = simplify_onnx(onnx_in, sim)
    else:
        print("[1/4] Skipping onnxsim (per --skip-simplify)")

    if args.force_fp32:
        print("[2/4] Forcing all fp16/bf16 -> fp32...")
        forced = os.path.join(work_dir, f"{name}.fp32.onnx")
        onnx_in = force_fp32(onnx_in, forced)
    else:
        print("[2/4] Keeping mixed precision as-is (use --force-fp32 to upcast)")

    if not args.skip_patch:
        print("[3/4] Patching mixed-dtype binary/variadic ops...")
        patched = os.path.join(work_dir, f"{name}.patched.onnx")
        onnx_in = patch_concat_dtype_mismatch(onnx_in, patched, verbose=args.verbose)
    else:
        print("[3/4] Skipping dtype patch (per --skip-patch)")

    print("[4/4] Converting ONNX -> TFLite via onnx2tf...")
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
        msg = str(e)
        if "don't all match" in msg or "must have the same dtype" in msg:
            print()
            print("=" * 68, file=sys.stderr)
            print("CONVERSION FAILED with a dtype-mismatch error.", file=sys.stderr)
            print("This usually means onnx2tf introduces a fp16 boundary",
                  file=sys.stderr)
            print("internally that the patch can't reach. Next steps:", file=sys.stderr)
            print("  1. Re-run with `--force-fp32` — upcasts every fp16/bf16",
                  file=sys.stderr)
            print("     in the ONNX graph to fp32 before conversion. This is",
                  file=sys.stderr)
            print("     the surest fix for `tf.math.divide / tf.concat / tf.add",
                  file=sys.stderr)
            print("     ... must have the same dtype, got fp16 != fp32` errors.",
                  file=sys.stderr)
            print("  2. If still failing, run with `-v` to see which Casts the",
                  file=sys.stderr)
            print("     patch inserted, then open the failing op in Netron and",
                  file=sys.stderr)
            print("     check whether its op_type is in `_DTYPE_UNIFY_OPS`.",
                  file=sys.stderr)
            print("  3. As a last resort: --skip-simplify (sometimes onnxsim",
                  file=sys.stderr)
            print("     re-introduces a Cast that was earlier folded out).",
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
