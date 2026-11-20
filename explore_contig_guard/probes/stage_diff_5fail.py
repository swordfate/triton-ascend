#!/usr/bin/env python3
"""Stage-level diff between 4902 baseline (/tmp/RUN_PATH) and fresh 1119
(/tmp/RUN_1119CLEAN) for the 5 new-FAIL cases.
Per stage: id, iters, has_contig, and the winning-route implementation's
total_system_cycles + formula_evidence."""
import json
import glob
import sys

CASES = ["muls_add_n1048576", "memory_stream_l4_n2097152",
         "pytest_permute_reshape_fused_f32", "pytest_fused_softmax_f32_128x257",
         "pytest_strided_scatter_f32_s16"]
ROOTS = {"4902": "/tmp/RUN_PATH", "1119f": "/tmp/RUN_1119CLEAN"}


def find(root, case):
    fs = glob.glob(root + "/s*/*%s__auto_f1.route.json" % case)
    return json.load(open(fs[0])) if fs else None


def impl_for(stage, mode, factor):
    for im in stage.get("implementations", []):
        i = im.get("implementation", {})
        if i.get("mode") == mode and i.get("superblock_factor", 1) == factor:
            return im
    return None


for case in CASES:
    print("#### %s" % case)
    data = {tag: find(root, case) for tag, root in ROOTS.items()}
    stages4 = (data["4902"] or {}).get("stage_model", {}).get("logical_stages", [])
    stagesF = (data["1119f"] or {}).get("stage_model", {}).get("logical_stages", [])
    d = data["1119f"]
    route = d.get("effective_decision_kind", "?")
    mode = "simt" if "simt" in route else "simd"
    if "mixed" in route:
        mode = "mixed"
    factor = d.get("selected_superblock_factor") or 1
    print("  route=%s mode=%s factor=%s  total: 4902=%.3fus fresh=%.3fus" % (
        route, mode, factor,
        (data["4902"].get("candidate_elapsed_costs", {}).get(
            data["4902"].get("effective_decision_kind"), 0) / 1000.0) if data["4902"] else -1,
        d.get("candidate_elapsed_costs", {}).get(route, 0) / 1000.0))
    for a, b in zip(stages4, stagesF):
        im4 = impl_for(a, mode, factor)
        imF = impl_for(b, mode, factor)
        f4 = a.get("features", {})
        cyc4 = (im4 or {}).get("total_system_cycles")
        cycF = (imF or {}).get("total_system_cycles")
        ev4 = (im4 or {}).get("formula_evidence", "-")[:80]
        evF = (imF or {}).get("formula_evidence", "-")[:80]
        mark = "" if cyc4 == cycF else "   <<< CHANGED"
        print("  %-34s it=%-3s cont=%-5s 4902=%9.1f fresh=%9.1f %s%s" % (
            b.get("id", "?")[:34], b.get("iteration_count"),
            b.get("features", {}).get("has_contiguous_memory"),
            cyc4 if cyc4 is not None else -1,
            cycF if cycF is not None else -1, evF, mark))
        if cyc4 != cycF:
            print("      4902 evidence: %s" % ev4)
    print()
