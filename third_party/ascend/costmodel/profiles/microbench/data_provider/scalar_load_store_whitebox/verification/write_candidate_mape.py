#!/usr/bin/env python3
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from make_three_kernel_table import MEASURED  # noqa: E402

TAGS = ["base", "simt_fill470_l48", "simt_fill480_l48", "simt_fill440_l0",
        "simd_fill430_l10", "simd_fill410_l10", "combined_470_48_430_10"]


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


rows = []
for tag in TAGS:
    preds = {}
    for kernel in ["padded_copy_gather", "padded_copy_scatter", "padded_copy_wgrad"]:
        p = ROOT / "candidate_reports" / f"costmodel_{kernel}__{tag}.json"
        preds.update({(m, kernel, sid): v for (m, sid), v in load_pred(p).items()})
    row = {"tag": tag}
    for mode in ["SIMT", "SIMD"]:
        errs = []
        for m, kernel, stage, kind, window, ids in MEASURED:
            if m != mode:
                continue
            pred = preds[(mode, kernel, f"{stage}_scalar_load")]
            errs.append((pred - window) / window)
        row[f"{mode}_mape_pct"] = round(sum(abs(e) for e in errs) / len(errs) * 100, 2)
        row[f"{mode}_bias_pct"] = round(sum(errs) / len(errs) * 100, 2)
    row["avg_mape_pct"] = round((row["SIMT_mape_pct"] + row["SIMD_mape_pct"]) / 2, 2)
    rows.append(row)

out = ROOT / "candidate_mape.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("wrote", out)
