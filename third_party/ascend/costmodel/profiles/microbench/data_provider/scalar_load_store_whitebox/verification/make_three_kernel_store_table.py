#!/usr/bin/env python3
"""Store-cycle comparison for the three padded kernels (2026-09-17).

ScalarStore rows use the newly integrated MainScalar/SIMT white-box store
model; IndirectGatherMemory rows are reference rows that still use the
indirect transaction model and are not covered by the scalar-store formulas.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (mode, kernel, stage, kind, camodel_window, instruction ids, note)
MEASURED = [
    ("SIMT", "padded_copy_gather", "stage_15", "tile_store", 458, "660", "SIMT_STG tile store (loop)"),
    ("SIMD", "padded_copy_gather", "stage_15", "tile_store", 1171, "698", "MTE3 UB->OUT tile store"),
    ("SIMT", "padded_copy_scatter", "stage_17", "tile_store", 525, "716", "SIMT_STG tile store (loop)"),
    ("SIMD", "padded_copy_scatter", "stage_17", "tile_store", 1185, "927", "MTE3 UB->OUT tile store"),
    ("SIMT", "padded_copy_wgrad", "stage_22", "scalar_store", 420, "867", "SIMT_STG scalar wgrad store"),
    ("SIMD", "padded_copy_wgrad", "stage_22", "scalar_store", 436, "1258", "MTE3 UB->OUT scalar store (core0)"),
]

REPORTS = {
    "padded_copy_gather": "costmodel_padded_copy_gather.json",
    "padded_copy_scatter": "costmodel_padded_copy_scatter.json",
    "padded_copy_wgrad": "costmodel_padded_copy_wgrad.json",
}
INDIRECT_LATENCY = {"SIMT": 65.4, "SIMD": 1.95}


def load_reports():
    out = {}
    for kernel, name in REPORTS.items():
        d = json.load(open(ROOT / name))
        for stage in d["stage_model"]["logical_stages"]:
            sid = stage["id"]
            for impl in stage["implementations"]:
                mode = impl["implementation"]["mode"].upper()
                if impl["implementation"].get("superblock_factor") != 1:
                    continue
                key = (mode, kernel, sid)
                if key in out:
                    continue
                workload = stage.get("workload", {})
                out[key] = {
                    "total": impl["total_system_cycles"],
                    "store": impl["resource_system_cycles"].get("store_per_iteration", 0.0),
                    "load": impl["resource_system_cycles"].get("load_per_iteration", 0.0),
                    "store_exposure": workload.get("indirect_scalar_store_exposure_count_per_iteration", 0.0),
                    "workload": workload,
                }
    return out


def main():
    model = load_reports()
    rows = []
    for mode, kernel, stage, kind, window, ids, note in MEASURED:
        sid = f"{stage}_{'indirect_gather_memory' if kind == 'tile_store' else 'scalar_store'}"
        mv = model.get((mode, kernel, sid))
        if mv is None:
            raise SystemExit(f"missing model for {mode} {kernel} {sid}")
        store = mv["store"]
        total = mv["total"]
        base = store
        if kind == "scalar_store":
            base = store - mv["store_exposure"] * INDIRECT_LATENCY[mode]
        row = {
            "mode": mode,
            "kernel": kernel,
            "stage": stage,
            "kind": kind,
            "costmodel_store": round(store, 3),
            "costmodel_store_base": round(base, 3),
            "costmodel_total": round(total, 3),
            "camodel_window": window,
            "err_store_pct": round((store - window) / window * 100, 1),
            "err_base_pct": round((base - window) / window * 100, 1) if kind == "scalar_store" else "",
            "camodel_ids": ids,
            "note": note,
        }
        rows.append(row)
    out = ROOT / "three_kernel_store_compare.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
