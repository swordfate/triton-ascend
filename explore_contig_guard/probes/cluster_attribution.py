#!/usr/bin/env python3
"""Cluster attribution: rows-on vs rows-off per-stage for the 5 blocker
clusters of row retirement (reduce_sum, rowred, permute, p112,
fused_softmax, strided_scatter)."""
import json
import os
import subprocess
import sys

MANIFEST = ("/root/w/cc-harness/costmodel_generalization/"
            "generalization_matrix_v140_all_100.json")
MEAS = {}
for line in open("/root/work/table_100case.md", errors="ignore"):
    if not line.startswith("|"):
        continue
    cols = [c.strip() for c in line.rstrip("\n").split("|")]
    if len(cols) < 6 or cols[1] in ("#", "---:"):
        continue
    if cols[4] and cols[4] != "-":
        try:
            MEAS[cols[2]] = float(cols[4])
        except ValueError:
            pass

CASES = ["pytest_reduce_sum_epilogue_f32_8x8x512",
         "rowred_r16_c128_k16",
         "pytest_permute_reshape_fused_f32",
         "grid_stride_dot_p112",
         "pytest_fused_softmax_f32_128x257",
         "pytest_strided_scatter_f32_s16"]

ENV = dict(os.environ)
ENV.update({
    "PATH": "/home/c00946898/bishengir-install/bin:" + ENV.get("PATH", ""),
    "PYTHONPATH": ("/usr/local/Ascend/ascend-toolkit/python/site-packages:"
                   "/usr/local/Ascend/cann-9.1.0/python/site-packages"),
    "ASCEND_RT_VISIBLE_DEVICES": "3",
    "TRITON_ASCEND_COSTMODEL_DETAILED_REPORT": "1",
})

PROFILE = ("/usr/local/python3.11.15/lib/python3.11/site-packages/triton/"
           "_C/ascend/costmodel_profiles/simd_simt/"
           "ascend_950pr_simd_simt_v1.json")


def run_case(case, skip_rows):
    out = "/tmp/CLUSTER_%s" % ("norows" if skip_rows else "rows")
    env = dict(ENV)
    if skip_rows:
        env["ASCEND_V2_SKIP_ROWS"] = "1"
    else:
        env.pop("ASCEND_V2_SKIP_ROWS", None)
    cmd = ["python3", "run_generalization_matrix.py",
           "--manifest", MANIFEST,
           "--output-root", out,
           "--profile", PROFILE,
           "--expected-bishengir",
           "/home/c00946898/bishengir-install/bin/bishengir-compile",
           "--measurement-processes", "1", "--disable-profiler",
           "--case", case]
    subprocess.run(cmd, cwd="/root/w/cc-harness/costmodel_generalization",
                   env=env, capture_output=True, timeout=600)
    path = os.path.join(out, case + "__auto_f1.route.json")
    s = open(path, errors="ignore").read().strip()
    d = json.JSONDecoder().raw_decode(s)[0]
    kind = d.get("effective_decision_kind")
    pred = (d.get("candidate_elapsed_costs") or {}).get(kind, 0) / 1000.0
    stages = []
    for st in (d.get("stage_model") or {}).get("logical_stages") or []:
        best = None
        for im in (st.get("implementations") or []):
            i = im.get("implementation") or {}
            if i.get("superblock_factor") != 1:
                continue
            mode = i.get("mode")
            if kind == "all_simd" and mode != "simd":
                continue
            if kind == "all_simt_only" and mode != "simt":
                continue
            if kind == "mixed_simd_simt" and mode == "simt":
                continue
            best = (st.get("id"),
                    round((im.get("total_system_cycles") or 0) / 1000.0, 3))
        if best:
            stages.append(best)
    return kind, pred, stages


for case in CASES:
    meas = MEAS.get(case)
    kind_r, pred_r, st_r = run_case(case, False)
    kind_n, pred_n, st_n = run_case(case, True)
    print("== %s  measured=%s us" % (case, meas))
    print("   rows: %s %.3f us (%+.1f%%)" %
          (kind_r, pred_r, (pred_r - meas) / meas * 100 if meas else 0))
    print("   norows: %s %.3f us (%+.1f%%)" %
          (kind_n, pred_n, (pred_n - meas) / meas * 100 if meas else 0))
    sr = dict(st_r)
    sn = dict(st_n)
    for sid in sr:
        a, b = sr[sid], sn.get(sid, 0)
        if abs(a - b) > 0.001:
            print("   %-38s %8.3f -> %8.3f  (%+8.3f us)" %
                  (sid, a, b, b - a))
    sys.stdout.flush()
print("DONE")
