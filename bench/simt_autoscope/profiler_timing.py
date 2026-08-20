#!/usr/bin/env python3
"""Kernel-only latency via torch_npu profiler (kernel_details.csv Duration).

The Event-based timing used elsewhere is launch-inclusive: back-to-back
launches between two Events include the device-side launch gap (~8-13k
SYS_CNT cycles on A5).  The costmodel scores kernel-only cycles, matching
the run_costmodel.sh measurements which read Duration(us) from
ASCEND_PROFILER_OUTPUT/kernel_details.csv.  This helper reproduces that.

Usage:

    from profiler_timing import measure_profiler_latency_ms
    latency_ms = measure_profiler_latency_ms(
        launch_fn,          # callable launching the kernel exactly once
        name_hint="kernel", # substring of the CSV Name column, optional
        reps=20, warmup=2,
    )
"""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
from pathlib import Path


def measure_profiler_latency_ms(launch_fn, name_hint=None, reps=20,
                                warmup=2):
    """Median kernel Duration(us)/1000 across reps traced launches."""
    import torch
    try:
        import torch_npu
    except ImportError:
        raise RuntimeError(
            "torch_npu unavailable; profiler timing requires NPU")
    if not hasattr(torch_npu, "profiler"):
        raise RuntimeError("torch_npu.profiler unavailable in this build")

    prof_dir = Path(tempfile.mkdtemp(prefix="npu_prof_"))
    try:
        schedule = torch_npu.profiler.schedule(
            wait=0, warmup=warmup, active=reps, repeat=1, skip_first=0)
        handler = torch_npu.profiler.tensorboard_trace_handler(str(prof_dir))
        profile_kwargs = {
            "activities": [torch_npu.profiler.ProfilerActivity.NPU],
            "schedule": schedule,
            "on_trace_ready": handler,
        }
        try:
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
                l2_cache=False,
                data_simplification=False,
            )
            profile_kwargs["experimental_config"] = experimental_config
        except Exception:
            pass

        with torch_npu.profiler.profile(**profile_kwargs) as prof:
            for _ in range(warmup + reps):
                launch_fn()
                prof.step()
            torch.npu.synchronize()

        subdirs = [os.path.join(prof_dir, d) for d in os.listdir(prof_dir)
                   if os.path.isdir(os.path.join(prof_dir, d))]
        if not subdirs:
            raise RuntimeError("profiler produced no output directory")
        kernel_dir = max(subdirs, key=os.path.getmtime)
        csv_path = os.path.join(kernel_dir, "ASCEND_PROFILER_OUTPUT",
                                "kernel_details.csv")

        def read_durations(csv_path, name_hint):
            durations = []
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("Name", "")
                    if name_hint and name_hint not in name:
                        continue
                    value = row.get("Duration(us)", "").strip()
                    if value and value.upper() != "N/A":
                        durations.append(float(value))
            return durations

        durations = read_durations(csv_path, name_hint)
        if not durations and name_hint:
            durations = read_durations(csv_path, None)
        if not durations:
            raise RuntimeError(
                f"no kernel durations in {csv_path} (hint={name_hint})")
        durations.sort()
        return durations[len(durations) // 2] / 1000.0
    finally:
        shutil.rmtree(prof_dir, ignore_errors=True)
