#!/usr/bin/env python3
"""Per-SIMT-LDG active windows for one CAModel run (diagnostic helper)."""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

CORE = "core0.veccore0"


def main():
    run_dir = Path(sys.argv[1])
    manifest = json.loads((run_dir / "launch_manifest.json").read_text())
    profile = sorted(run_dir.glob("OPPROF_*"))[0]
    kernel_dir = profile / "strided_load_kernel"
    if not kernel_dir.is_dir():
        kernel_dir = profile
    for case in manifest["cases"]:
        if case["mode"] != "simt_only":
            continue
        dump = kernel_dir / str(case["order"]) / "dump"
        if not dump.is_dir():
            dump = kernel_dir / "dump"
        text = (dump / f"{CORE}.rvec.simt.lsu.dump").read_text(errors="replace")
        events = defaultdict(lambda: {"issue": [], "retire": []})
        for line in text.splitlines():
            m = re.search(r"\[(\d+)\] (ISSUE_INSTR|RETIRE_INSTR) name=SIMT_LDG id=(\d+)", line)
            if not m:
                continue
            t, kind, iid = int(m.group(1)), m.group(2), int(m.group(3))
            events[iid]["issue" if kind == "ISSUE_INSTR" else "retire"].append(t)
        rows = []
        for iid, rec in sorted(events.items()):
            issue = min(rec["issue"])
            retire = max(rec["retire"])
            rows.append((iid, issue, retire, retire - issue))
        window = (min(r[1] for r in rows), max(r[2] for r in rows)) if rows else (None, None)
        print(
            f'{case["order"]:>3} stride={case["stride"]:<4} block={case["block"]:<4} '
            f'n_ldg={len(rows)} window={window[1] - window[0] if rows else None} '
            f'per_ldg={[r[3] for r in rows]} issues={[r[1] for r in rows]} retires={[r[2] for r in rows]}'
        )


if __name__ == "__main__":
    main()
