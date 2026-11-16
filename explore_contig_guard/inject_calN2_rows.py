#!/usr/bin/env python3
"""inject_calN2_rows.py - insert calN2 n1_s1 low-grid rows into the v2 profile.

Reads /root/work/explore_contig/calN2_rows.json (keys n1_s1_b<elements>_k<K>_g<G>_m0),
maps element-block -> byte key (b<elements*4>), inserts into BOTH the installed
profile JSON and the repo source profile JSON (if present).
"""
import json
import sys
from pathlib import Path

ROWS = Path("/root/work/explore_contig/calN2_rows.json")
INSTALLED = Path("/usr/local/python3.11.15/lib/python3.11/site-packages/"
                 "triton/_C/ascend/costmodel_profiles/simd_simt/"
                 "ascend_950pr_simd_simt_v1.json")
REPO = Path("/root/work/triton-ascend/third_party/ascend/costmodel/"
            "profiles/simd_simt/ascend_950pr_simd_simt_v1.json")


def main():
    rows = json.loads(ROWS.read_text())
    remapped = {}
    for mode, table in rows.items():
        remapped[mode] = {}
        for key, val in table.items():
            if not isinstance(val, (int, float)) or val != val:
                print("SKIP non-finite", mode, key, val)
                continue
            # n1_s1_b<elements>_k<K>_g<G>_m0  ->  b<elements*4> (bytes)
            parts = key.split("_")
            bi = [i for i, p in enumerate(parts) if p.startswith("b")][0]
            elems = int(parts[bi][1:])
            parts[bi] = "b%d" % (elems * 4)
            newkey = "_".join(parts)
            remapped[mode][newkey] = round(float(val), 1)
            print(mode, key, "->", newkey, val)

    for path in (INSTALLED, REPO):
        if not path.exists():
            print("skip missing", path)
            continue
        prof = json.loads(path.read_text())
        for mode in ("simd", "simt"):
            direct = prof[mode]["memory"]["contiguous_ldst_v2"]["direct"]
            before = len(direct)
            for key, val in remapped[mode].items():
                direct[key] = val
            print(path.name, mode, "rows", before, "->", len(direct))
        path.write_text(json.dumps(prof, indent=2, ensure_ascii=False))
        print("wrote", path)


main()
