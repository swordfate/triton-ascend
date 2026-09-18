#!/usr/bin/env python3
"""Inspect scalar store stages in the padded three-kernel reports."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = {
    "gather": ROOT / "costmodel_padded_gather.json",
    "scatter": ROOT / "costmodel_padded_scatter.json".replace("padded_scatter", "padded_copy_scatter"),
    "wgrad": ROOT / "costmodel_padded_copy_wgrad.json",
}

for name, path in FILES.items():
    print("=" * 30, name, path.name)
    d = json.load(open(path))
    print("decision", d["decision_kind"], d["candidate_costs"])
    for s in d["stage_model"]["logical_stages"]:
        w = s.get("workload", {})
        store_related = any(
            ("store" in k.lower() and v) or ("store" in s["id"].lower())
            for k, v in w.items()
        )
        if not store_related:
            continue
        print("\n---", s["id"], "model=", s["model"], "iter=", s.get("iteration_count"))
        print("  source=", s.get("source_locations"))
        for k, v in sorted(w.items()):
            if v:
                print("   w", k, "=", v)
        for impl in s.get("implementations", []):
            im = impl["implementation"]
            if impl["total_system_cycles"] > 0:
                print("   impl", im["mode"], "F=", im.get("superblock_factor"),
                      "total=", round(impl["total_system_cycles"], 3),
                      "resources=", {k: round(v, 3) for k, v in impl.get("resource_system_cycles", {}).items() if v})
