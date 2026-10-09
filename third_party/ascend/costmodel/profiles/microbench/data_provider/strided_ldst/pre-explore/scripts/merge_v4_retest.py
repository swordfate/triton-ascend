#!/usr/bin/env python3
"""Merge the main and missing-stride 5-allocation SIMT retest JSONs.

Expected input shape: {"results": [{num_warps, block, stride, targets_ns, ...}]}.
Keys must be disjoint; the output keeps the union sorted by (W, block, stride).
"""
import argparse
import json
from pathlib import Path


def load_results(path: Path):
    payload = json.loads(path.read_text())
    return payload.get("results", payload)


def key(row):
    return (int(row["num_warps"]), int(row["block"]), int(row["stride"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--missing", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    seen = {}
    for source in (args.main, args.missing):
        for row in load_results(source):
            k = key(row)
            if k in seen:
                raise SystemExit(f"duplicate retest key {k}: {source}")
            seen[k] = row
            rows.append(row)
    rows.sort(key=key)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": rows}, indent=2), encoding="utf-8")
    print(f"wrote {args.out}: {len(rows)} cases")


if __name__ == "__main__":
    main()
