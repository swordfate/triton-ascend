#!/usr/bin/env python3
"""Batch driver: run a list of contiguous-probe points, serially.

Rules this driver enforces, all of them from 04_PITFALLS.md / 02_LESSONS.md:
  * one measurement process at a time on the device -- two concurrent
    processes raise device error 507035 and poison the context;
  * every point repeated --repeat times, each repeat a separate process;
  * the load average is recorded before and after every point;
  * a failed point is written to the manifest with status=fail and is never
    interpolated over.

Each point is one JSON object in --points.  Keys mirror probe_contiguous.py.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBE = HERE / "probe_contiguous.py"

# Every knob the two probes accept.  Each point only passes the ones it sets, so
# a single flag table serves both the stage-1 probe and the sweep probe.
FLAGS = ["kind", "mode", "direction", "dtype", "block", "ops", "groups",
         "group", "outer", "factor", "logical_programs", "grid", "masked",
         "rows", "active_elements", "real_rows", "real_cols", "row_stride",
         "num_warps", "program_stride"]


def load_average():
    with open("/proc/loadavg") as stream:
        return [float(x) for x in stream.read().split()[:3]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--active", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--probe", default="probe_contiguous.py",
                        help="probe script in this directory (stage-1 reproduction or sweep)")
    parser.add_argument("--only", default="", help="substring filter on tag")
    parser.add_argument("--resume", action="store_true",
                        help="skip points already present in the manifest")
    parser.add_argument("--stop-on-fail", action="store_true")
    args = parser.parse_args()

    points = json.loads(args.points.read_text())
    if args.only:
        points = [p for p in points if args.only in p.get("tag", "")]

    done = set()
    if args.resume and args.manifest.exists():
        for line in args.manifest.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Successful measurements carry no "status" key at all; only
            # failures are annotated.
            if record.get("status", "ok") == "ok":
                done.add((record.get("tag", ""), record.get("repeat_index", 0)))

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    total = len(points) * args.repeat
    index = 0
    failures = 0
    for point in points:
        tag = point.get("tag") or "-".join(
            "%s%s" % (key, point[key]) for key in FLAGS if key in point)
        for repeat in range(args.repeat):
            index += 1
            if (tag, repeat) in done:
                print("[%d/%d] skip   %s r%d" % (index, total, tag, repeat), flush=True)
                continue
            command = [sys.executable, str(HERE / args.probe),
                       "--manifest", str(args.manifest),
                       "--repeat", str(args.repeat),
                       "--repeat-index", str(repeat),
                       "--active", str(args.active),
                       "--warmup", str(args.warmup),
                       "--tag", tag]
            for key in FLAGS:
                if key in point:
                    command += ["--" + key.replace("_", "-"), str(point[key])]
            load_before = load_average()
            started = time.time()
            proc = subprocess.run(command, capture_output=True, text=True)
            elapsed = time.time() - started
            if proc.returncode != 0:
                failures += 1
                record = {"status": "fail", "tag": tag,
                          "repeat_index": repeat, "repeat_total": args.repeat,
                          "returncode": proc.returncode,
                          "stderr": proc.stderr[-4000:],
                          "stdout": proc.stdout[-2000:],
                          "load_before": load_before,
                          "load_after": load_average(),
                          "wall_s": round(elapsed, 2),
                          "point": point,
                          "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
                with args.manifest.open("a") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                print("[%d/%d] FAIL   %s r%d (rc=%d) %.1fs" % (index, total, tag,
                      repeat, proc.returncode, elapsed), flush=True)
                print(proc.stderr[-1200:], flush=True)
                if args.stop_on_fail:
                    return 1
            else:
                print("[%d/%d] ok     %s r%d %.1fs" % (index, total, tag, repeat, elapsed),
                      flush=True)
                print("      " + proc.stdout.strip()[-800:], flush=True)
    print("done: %d points x %d repeats, %d failures" % (len(points), args.repeat, failures))
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
