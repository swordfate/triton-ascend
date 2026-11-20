#!/usr/bin/env python3
"""Rebuild loop_envelope_fit JSON sections from the E27b expanded campaigns
and write to BOTH the repo profile and the installed wheel's profile."""
import json
from pathlib import Path

REPO_J = Path("/root/work/triton-ascend/third_party/ascend/costmodel/"
              "profiles/simd_simt/ascend_950pr_simd_simt_v1.json")
INST_J = Path("/usr/local/python3.11.15/lib/python3.11/site-packages/triton/"
              "_C/ascend/costmodel_profiles/simd_simt/"
              "ascend_950pr_simd_simt_v1.json")
DTYPES = ("int8", "fp16", "fp32", "int64")


def main():
    secs = {}
    for mode, eng in (("simd", "simd"), ("simt_only", "simt")):
        dt = {}
        for dtype in DTYPES:
            p = Path("/tmp/e27b_%s_%s.jsonl" % (dtype, mode))
            rep = data = None
            for line in p.read_text().splitlines():
                o = json.loads(line)
                if "report" in o:
                    rep = o["report"]
                if "data" in o:
                    data = o["data"]
            if not rep or not data:
                print("MISSING", p)
                continue
            lbmax = max((d["lB"] for d in data if d["kind"] == "L"),
                        default=32768)
            sbmax = max((d["sB"] for d in data if d["kind"] == "S"),
                        default=16384)
            dt[dtype] = {
                "stepL": rep["stepL"],
                "load_l0": rep["byte_load"]["l0"],
                "load_rate": rep["byte_load"]["B_ns"],
                "load_bytes_max": lbmax,
                "stepS": rep["stepS"],
                "store_l0": rep["byte_store"]["l0"],
                "store_rate": rep["byte_store"]["B_ns"],
                "store_bytes_max": sbmax,
                "kappa": rep["kappa"],
                "excess": rep["excess_T0_deltas"],
                "chain": rep["chain_T1_deltas"],
                "mape_all_M3": rep["MAPE_all"]["M3"],
                "mape_all_M0": rep["MAPE_all"]["M0"],
                "mape_loads_M3": rep["MAPE_loads"]["M3"],
                "mape_tiny_M3": rep["MAPE_tiny"]["M3"],
            }
            print(eng, dtype, "lbmax", lbmax, "sbmax", sbmax,
                  "M3", rep["MAPE_all"]["M3"])
        if dt:
            secs[eng] = {
                "enabled": 1,
                "source": ("E27b expanded fit campaign e27b_fit2.py "
                           "(paired-acc structure; nL/nS/bytes/dom matrix, "
                           "le to 4096 elems, totals to 64KB+; grid=1, "
                           "K=64, SYS_CNT windows; simt via clock64 hook)"),
                "dtypes": dt,
            }
    for path in (REPO_J, INST_J):
        j = json.loads(path.read_text())
        for eng, sec in secs.items():
            j[eng]["memory"]["contiguous_ldst_v2"]["loop_envelope_fit"] = sec
        path.write_text(json.dumps(j, indent=1, ensure_ascii=False))
        json.loads(path.read_text())
        print("wrote", path)


main()
