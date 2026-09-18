#!/usr/bin/env python3
"""Three padded kernels: costmodel vs CAModel active windows (after exposure fix)."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (mode, kernel, stage, kind, CAModel window, instruction ids)
MEASURED = [
    ("SIMT", "padded_copy_gather", "stage_4", "direct", 517, "511"),
    ("SIMT", "padded_copy_gather", "stage_5", "direct", 516, "517"),
    ("SIMT", "padded_copy_gather", "stage_7", "pair", 473, "596,599"),
    ("SIMT", "padded_copy_scatter", "stage_4", "direct", 476, "554"),
    ("SIMT", "padded_copy_scatter", "stage_5", "direct", 475, "558"),
    ("SIMT", "padded_copy_scatter", "stage_7", "pair", 417, "613,616"),
    ("SIMT", "padded_copy_scatter", "stage_11", "indirect1", 438, "650"),
    ("SIMT", "padded_copy_wgrad", "stage_4", "direct", 559, "582"),
    ("SIMT", "padded_copy_wgrad", "stage_5", "direct", 558, "588"),
    ("SIMT", "padded_copy_wgrad", "stage_7", "pair", 548, "651,654"),
    ("SIMD", "padded_copy_gather", "stage_4", "direct", 458, "554"),
    ("SIMD", "padded_copy_gather", "stage_5", "direct", 405, "357"),
    ("SIMD", "padded_copy_gather", "stage_7", "pair", 500, "544,545"),
    ("SIMD", "padded_copy_scatter", "stage_4", "direct", 532, "561"),
    ("SIMD", "padded_copy_scatter", "stage_5", "direct", 489, "339"),
    ("SIMD", "padded_copy_scatter", "stage_7", "pair", 528, "554,555"),
    ("SIMD", "padded_copy_scatter", "stage_11", "indirect1", 361, "574"),
    ("SIMD", "padded_copy_wgrad", "stage_4", "direct", 380, "512"),
    ("SIMD", "padded_copy_wgrad", "stage_5", "direct", 371, "414"),
    ("SIMD", "padded_copy_wgrad", "stage_7", "pair", 486, "493,494"),
]

REPORT_FILES = {
    "padded_copy_gather": "costmodel_padded_gather.json",
    "padded_copy_scatter": "costmodel_padded_copy_scatter.json",
    "padded_copy_wgrad": "costmodel_padded_copy_wgrad.json",
}


def load_model_values():
    values = {}
    for kernel, name in REPORT_FILES.items():
        d = json.load(open(ROOT / name))
        for stage in d["stage_model"]["logical_stages"]:
            sid = stage["id"]
            for impl in stage["implementations"]:
                mode = impl["implementation"]["mode"].upper()
                if mode not in ("SIMD", "SIMT"):
                    continue
                key = (mode, kernel, sid)
                existing = values.get(key)
                if existing is None or impl["implementation"].get("superblock_factor") == 1:
                    values[key] = {
                        "total": impl["total_system_cycles"],
                        "load": impl["resource_system_cycles"].get("load_per_iteration", 0.0),
                    }
    return values


def main():
    model = load_model_values()
    rows = []
    for mode, kernel, stage, kind, window, ids in MEASURED:
        sid = f"{stage}_scalar_load"
        mv = model.get((mode, kernel, sid))
        if mv is None:
            raise SystemExit(f"missing model for {mode} {kernel} {sid}")
        total = mv["total"]
        load = mv["load"]
        rows.append({
            "mode": mode,
            "kernel": kernel,
            "stage": stage,
            "kind": kind,
            "costmodel_total": round(total, 3),
            "costmodel_load": round(load, 3),
            "costmodel_store": "",
            "camodel_window": window,
            "err_total_pct": round((total - window) / window * 100, 1),
            "err_load_pct": round((load - window) / window * 100, 1),
            "err_store_pct": "",
            "camodel_ids": ids,
            "note": "scalar load",
        })
    # Merge the store table when present so one CSV contains both load and
    # store comparisons for the three padded kernels.
    store_path = ROOT / "three_kernel_store_compare.csv"
    if store_path.exists():
        for s in csv.DictReader(open(store_path)):
            rows.append({
                "mode": s["mode"],
                "kernel": s["kernel"],
                "stage": s["stage"],
                "kind": s["kind"],
                "costmodel_total": s["costmodel_total"],
                "costmodel_load": "",
                "costmodel_store": s["costmodel_store"],
                "camodel_window": s["camodel_window"],
                "err_total_pct": s["err_store_pct"],
                "err_load_pct": "",
                "err_store_pct": s["err_store_pct"],
                "camodel_ids": s["camodel_ids"],
                "note": s["note"],
            })
    out = ROOT / "three_kernel_compare.csv"
    fieldnames = ["mode", "kernel", "stage", "kind", "costmodel_total",
                  "costmodel_load", "costmodel_store", "camodel_window",
                  "err_total_pct", "err_load_pct", "err_store_pct",
                  "camodel_ids", "note"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
