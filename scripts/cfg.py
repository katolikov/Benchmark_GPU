#!/usr/bin/env python3
"""Tiny `jq -r` substitute, stdlib-only — no pip deps.

Used by scripts/run_bench.sh to read JSON config files. Implements the small
subset of jq syntax we actually need.

Usage:
  cfg.py FILE .key.sub          # print scalar (empty line if missing/null)
  cfg.py FILE .array[]          # print each array element on its own line
  cfg.py FILE summarize         # read a results.json and print a 1-screen summary
                                 # (used to pretty-print the final bench result)
"""
from __future__ import annotations

import json
import sys


def walk(d, path: str):
    """Walk a dotted path. Returns the value or None if missing."""
    for p in (x for x in path.split(".") if x):
        if isinstance(d, dict):
            d = d.get(p)
        elif isinstance(d, list) and p.isdigit():
            i = int(p)
            d = d[i] if 0 <= i < len(d) else None
        else:
            return None
        if d is None:
            return None
    return d


def emit(v) -> None:
    if v is None:
        return
    if isinstance(v, bool):
        print("true" if v else "false")
    elif isinstance(v, (dict, list)):
        print(json.dumps(v))
    elif v == "":
        return
    else:
        print(v)


def cmd_query(path_to_file: str, expr: str) -> None:
    with open(path_to_file) as f:
        d = json.load(f)
    explode = expr.endswith("[]")
    path = expr[:-2] if explode else expr
    v = walk(d, path)
    if explode:
        if isinstance(v, list):
            for x in v:
                emit(x)
        return
    emit(v)


def cmd_summarize(path_to_file: str) -> None:
    """Pretty-print the headline fields from a single bench results JSON."""
    with open(path_to_file) as f:
        d = json.load(f)
    inf = d.get("inference", {})
    cos = d.get("vs_ref", {}).get("cosine") if d.get("vs_ref") else None
    rows = [
        ("label",      d.get("label", "")),
        ("framework",  f'{d.get("framework","")}/{d.get("backend","")}/{d.get("variant","")}'),
        ("create_ms",  d.get("create_ms")),
        ("first_ms",   d.get("first_inference_ms")),
        ("p50_ms",     inf.get("p50")),
        ("p90_ms",     inf.get("p90")),
        ("mean_ms",    inf.get("mean")),
        ("stdev_ms",   inf.get("stdev")),
    ]
    if cos is not None:
        rows.append(("cosine", cos))
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        if v is None:
            continue
        if isinstance(v, float):
            print(f"  {k:<{width}}  {v:.4f}")
        else:
            print(f"  {k:<{width}}  {v}")


def main() -> int:
    args = sys.argv[1:]
    if len(args) == 2 and args[1] == "summarize":
        cmd_summarize(args[0])
        return 0
    if len(args) == 2:
        cmd_query(args[0], args[1])
        return 0
    sys.stderr.write(
        "usage: cfg.py FILE PATH\n"
        "       cfg.py FILE summarize\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
