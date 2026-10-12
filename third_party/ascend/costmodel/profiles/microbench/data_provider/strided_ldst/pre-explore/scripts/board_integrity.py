#!/usr/bin/env python3
import os, glob, json, statistics
PRE = os.path.expanduser("~/strided-template-calibration-v2")
D = PRE + "/results/dtype_calibration_v10_expand2d/raw"
fs = sorted(glob.glob(D + "/*.json"))
print("完成文件: %d" % len(fs))
for p in fs:
    try:
        j = json.load(open(p))
    except Exception as e:
        print("  bad", p, e); continue
    cs = j.get("cases", [])
    sp = []
    for c in cs:
        a = c.get("attempts") or []
        for x in a:
            sp.append(x.get("spread", 0))
    ok = sum(1 for c in cs if c.get("valid"))
    print("  %-46s cases=%3d valid=%3d spread_med=%.3f spread_p90=%.3f" % (
        os.path.basename(p), len(cs), ok,
        statistics.median(sp) if sp else 0, sorted(sp)[int(len(sp) * .9)] if sp else 0))
