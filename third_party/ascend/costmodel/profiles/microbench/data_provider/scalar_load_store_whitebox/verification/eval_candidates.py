#!/usr/bin/env python3
"""Evaluate candidate profile parameter sets against the 3-kernel CAModel data."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from make_three_kernel_table import MEASURED  # noqa: E402

TAGS = ["base", "simt_fill470_l48", "simt_fill480_l48", "simt_fill440_l0", "simd_fill410_l10", "simd_fill430_l10", "combined_470_48_430_10"]


def load_pred(path):
    d = json.load(open(path))
    out = {}
    for stage in d["stage_model"]["logical_stages"]:
        for impl in stage["implementations"]:
            im = impl["implementation"]
            if im.get("superblock_factor", 1) != 1:
                continue
            out[(im["mode"].upper(), stage["id"])] = impl["total_system_cycles"]
    return out


def main():
    rows = []
    for tag in TAGS:
        preds = {}
        for kernel in ["padded_copy_gather", "padded_copy_scatter", "padded_copy_wgrad"]:
            path = ROOT / "candidate_reports" / f"costmodel_{kernel}__{tag}.json"
            preds.update({(mode, kernel, sid): val for (mode, sid), val in load_pred(path).items()})
        per_mode = {}
        for mode in ["SIMT", "SIMD"]:
            errs = []
            for m, kernel, stage, kind, window, ids in MEASURED:
                if m != mode:
                    continue
                sid = f"{stage}_scalar_load"
                pred = preds[(mode, kernel, sid)]
                errs.append((pred - window) / window)
            per_mode[mode] = (sum(abs(e) for e in errs) / len(errs) * 100,
                              sum(errs) / len(errs) * 100)
        all_errs = [per_mode[m][0] for m in per_mode]
        rows.append((tag, per_mode["SIMT"][0], per_mode["SIMT"][1],
                     per_mode["SIMD"][0], per_mode["SIMD"][1],
                     sum([per_mode["SIMT"][0], per_mode["SIMD"][0]]) / 2))
    print(f"{'tag':22s} {'SIMT_MAPE%':>10s} {'SIMT_bias%':>10s} {'SIMD_MAPE%':>10s} {'SIMD_bias%':>10s} {'avg_MAPE%':>10s}")
    for r in rows:
        print(f"{r[0]:22s} {r[1]:10.1f} {r[2]:10.1f} {r[3]:10.1f} {r[4]:10.1f} {r[5]:10.1f}")


if __name__ == "__main__":
    main()
