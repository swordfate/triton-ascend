#!/usr/bin/env python3
"""Create candidate SIMD/SIMT scalar_memory profiles for parameter sweeps."""
import json
from pathlib import Path

SRC = Path("/home/c00946898/triton-whitebox-latest/third_party/ascend/costmodel/profiles/simd_simt/david_v100_simd_simt_v1.json")
OUT = Path("/home/c00946898/stage_vs_camodel_20260917/profiles")
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATES = {
    # name: (simt_fill, simt_indirect_latency, simd_fill, simd_indirect_latency)
    "base": (None, None, None, None),
    "simt_fill470_l48": (470, 48, None, None),
    "simt_fill480_l48": (480, 48, None, None),
    "simt_fill470_l65": (470, 65.4, None, None),
    "simt_fill440_l0": (440, 0.0, None, None),
    "simd_fill410_l10": (None, None, 410, 10.0),
    "simd_fill430_l10": (None, None, 430, 10.0),
    "combined_470_48_430_10": (470, 48.0, 430, 10.0),
}


def main():
    base = json.loads(SRC.read_text())
    for name, (sf, sl, df, dl) in CANDIDATES.items():
        d = json.loads(json.dumps(base))
        if sf is not None:
            d["simt"]["stage_resources"]["scalar_memory"]["uniform_load_fill_system_cycles"] = sf
        if sl is not None:
            d["simt"]["stage_resources"]["scalar_memory"]["indirect_dependency_latency_system_cycles"] = sl
        if df is not None:
            d["simd"]["stage_resources"]["scalar_memory"]["main_load_fill_system_cycles"] = df
        if dl is not None:
            d["simd"]["stage_resources"]["scalar_memory"]["indirect_dependency_latency_system_cycles"] = dl
        out = OUT / f"{name}.json"
        out.write_text(json.dumps(d, indent=2) + "\n")
        print("wrote", out)


if __name__ == "__main__":
    main()
