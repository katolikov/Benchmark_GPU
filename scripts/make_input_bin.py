#!/usr/bin/env python3
"""Turn an image / .npy / raw-CSV into a `.bin` ready for `bench_tflite --input`.

The bench binaries read raw little-endian fp32 with no header. Total byte
count must match the model's input tensor (= prod(shape) * 4).

Usage:
  make_input_bin.py SOURCE OUTPUT_BIN --shape 1x224x224x3 --layout NHWC
                                       [--mean R,G,B] [--std R,G,B]
                                       [--scale 1.0]    [--bgr]

  make_input_bin.py img.png   in.bin --shape 1x224x224x3 --layout NHWC \\
       --mean 0.485,0.456,0.406 --std 0.229,0.224,0.225        # ImageNet stats

  make_input_bin.py x.npy     in.bin --shape 1x3x224x224 --layout NCHW

  make_input_bin.py raw.bin   in.bin --shape 1x3x224x224 --layout NCHW \\
       --no-preprocess                                          # passthrough copy
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _parse_shape(s: str):
    return tuple(int(x) for x in s.lower().replace(",", "x").split("x") if x)


def _parse_triple(s: str):
    return tuple(float(x) for x in s.split(","))


def load_array(path: str, target_shape, layout: str, args) -> np.ndarray:
    """Load any source into a fp32 numpy array of `target_shape` in `layout`."""
    layout = layout.upper()
    if layout not in ("NHWC", "NCHW"):
        raise ValueError(f"layout must be NHWC or NCHW, got {layout}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
        # If the npy already matches target_shape exactly, return as-is.
        if arr.shape == tuple(target_shape):
            return arr
        # Else assume the .npy is in the natural layout for the source and
        # let the user fix it themselves — we can't safely re-shape blind.
        raise ValueError(
            f".npy shape {arr.shape} != target {tuple(target_shape)}; "
            "transpose / reshape it yourself or use --no-preprocess"
        )

    if ext == ".bin" and args.no_preprocess:
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size != int(np.prod(target_shape)):
            raise ValueError(
                f".bin has {raw.size} fp32 elements, target needs "
                f"{int(np.prod(target_shape))}"
            )
        return raw.reshape(target_shape)

    # Treat anything else as an image. We rely on Pillow for decoding.
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow not installed (pip install pillow)", file=sys.stderr)
        sys.exit(1)

    if layout == "NHWC":
        n, h, w, c = target_shape
    else:
        n, c, h, w = target_shape
    if n != 1 or c not in (1, 3):
        raise ValueError(f"image source supports N=1 and C=1 or 3 only, got {target_shape}")

    img = Image.open(path)
    if c == 1:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    img = img.resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)              # HWC, uint8 -> float
    if c == 1:
        arr = arr[..., None]                              # HW -> HW1

    # Pixel scale: image is 0..255 by default. --scale 1/255 -> 0..1.
    if args.scale != 1.0:
        arr = arr * args.scale

    if args.bgr and c == 3:
        arr = arr[..., ::-1].copy()

    if args.mean is not None:
        mean = np.asarray(args.mean, dtype=np.float32)
        if mean.shape != (c,):
            raise ValueError(f"--mean expects {c} values, got {len(args.mean)}")
        arr = arr - mean.reshape(1, 1, c)

    if args.std is not None:
        std = np.asarray(args.std, dtype=np.float32)
        if std.shape != (c,):
            raise ValueError(f"--std expects {c} values, got {len(args.std)}")
        arr = arr / std.reshape(1, 1, c)

    arr = arr[None, ...]                                  # add N
    if layout == "NCHW":
        arr = np.transpose(arr, (0, 3, 1, 2)).copy()      # N H W C -> N C H W
    return arr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="Image / .npy / .bin file (.bin only with --no-preprocess)")
    ap.add_argument("output", help="Output .bin path")
    ap.add_argument("--shape", required=True,
                    help="Target tensor shape, e.g. 1x224x224x3 or 1x3x224x224")
    ap.add_argument("--layout", default="NHWC", choices=["NHWC", "NCHW", "nhwc", "nchw"],
                    help="Target tensor layout (default: NHWC)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help='Pixel scale, e.g. 0.00392156862 to map 0..255 -> 0..1, '
                         'or 0.0078125 to map 0..255 -> 0..2 then subtract 1 with --mean (default: 1.0)')
    ap.add_argument("--mean", type=_parse_triple, default=None,
                    help="Per-channel mean subtracted *after* scale, e.g. 0.485,0.456,0.406")
    ap.add_argument("--std", type=_parse_triple, default=None,
                    help="Per-channel std divided *after* mean, e.g. 0.229,0.224,0.225")
    ap.add_argument("--bgr", action="store_true",
                    help="Swap RGB channels to BGR (some Caffe-style models)")
    ap.add_argument("--no-preprocess", action="store_true",
                    help="If source is a .bin, copy it through verbatim "
                         "(only validates element count)")
    args = ap.parse_args()

    target = _parse_shape(args.shape)
    arr = load_array(args.source, target, args.layout, args)

    if arr.shape != tuple(target):
        print(f"ERROR: produced shape {arr.shape} != target {tuple(target)}",
              file=sys.stderr)
        return 1

    arr.astype(np.float32, copy=False).tofile(args.output)
    print(f"OK    wrote {args.output} ({arr.nbytes:,} bytes, shape={arr.shape}, "
          f"layout={args.layout.upper()}, dtype=fp32)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
