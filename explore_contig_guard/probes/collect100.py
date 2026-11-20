#!/usr/bin/env python3
"""Collect 100-case route.json dumps into a TSV: name, route, pred_us.

Usage: python3 collect100.py <RESULT_DIR> <OUT_TSV>
Prediction = candidate_elapsed_costs[effective_decision_kind] (us).
"""
import json
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    rows = []
    for f in sorted(root.glob("*.route.json")):
        name = f.name.replace("__auto_f1.route.json", "")
        try:
            d = json.load(open(f))
        except Exception as exc:  # noqa: BLE001
            rows.append((name, "PARSE_ERR", "", str(exc)[:60]))
            continue
        route = d.get("effective_decision_kind") or d.get("decision_kind") \
            or "none"
        costs = d.get("candidate_elapsed_costs") or {}
        pred = costs.get(route)
        rows.append((name, route, "" if pred is None else "%.4f" % pred,
                     " ".join("%s=%s" % (k, "%.4f" % v)
                              for k, v in sorted(costs.items()))))
    with open(out, "w") as h:
        h.write("name\troute\tpred_us\tall_candidates\n")
        for r in rows:
            h.write("\t".join(r) + "\n")
    print("wrote %d rows -> %s" % (len(rows), out))


main()
