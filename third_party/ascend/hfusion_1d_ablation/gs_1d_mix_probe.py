#!/usr/bin/env python3
"""Standalone probe for the 1D gather/scatter hfusion ablation.

Usage:
  python gs_1d_mix_probe.py <gather|scatter> <warmup|run> <BLOCK>
                            <dtype: f16|f32|i8|i16>
                            <auto|plain|fallback|keep_tt_load> [num_warps]

auto  : compile_mode=simd_simt, auto_simt_scope_mode=auto.  The C++ cost
        model selects the route and materializes local SIMT scopes.
plain : compile_mode=simd_simt, auto_simt_scope_mode=off, no scope.  Legacy
        backend_default recognition lowers the indirect access globally.
fallback / keep_tt_load: same compile options as auto; they differ only in
        which ablation patch is compiled into the installed triton wheel
        (scoped op -> scf.for fallback, or scoped op -> original tt.load).

warmup compiles and dumps the ttadapter IR to one file under
/tmp/triton_hfusion_1d_ablation_ir (override with
TRITON_HFUSION_ABLATION_IR_DIR); run launches the kernel and checks numerics.
Set GS_TAG to force a cache miss and regenerate the route report.  Apply
patches/fallback_in_simt_scope.patch for the fallback mode or
patches/keep_tt_load_in_simt_scope.patch for the keep_tt_load mode, rebuild
the triton wheel, then run route=fallback/keep_tt_load.  In both modes the
cost-model scope.scope remains; plain is unaffected and keeps legacy global
hfusion.
"""
import csv
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver

VEC = int(driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"])
DTYPES = {"f16": torch.float16, "f32": torch.float32, "i8": torch.int8, "i16": torch.int16}


@triton.jit
def gather_1d(x_ptr, idx_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    idx = tl.load(idx_ptr + offs)
    x = tl.load(x_ptr + idx)
    y = x * 2 + 1
    tl.store(out_ptr + offs, y)


@triton.jit
def scatter_1d(x_ptr, idx_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    idx = tl.load(idx_ptr + offs)
    y = x * 2 + 1
    tl.store(out_ptr + idx, y)


def _dump_ttadapter(compiled, kind, route, dtype_name, block):
    text = compiled.asm.get("ttadapter", "")
    if isinstance(text, bytes):
        text = text.decode("utf-8", "ignore")
    if not text:
        return
    out_dir = Path(os.environ.get(
        "TRITON_HFUSION_ABLATION_IR_DIR",
        str(Path(tempfile.gettempdir()) / "triton_hfusion_1d_ablation_ir")))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = route
    dump_path = out_dir / f"{kind}_{tag}_{dtype_name}_b{block}.ttadapter.mlir"
    dump_path.write_text(text)
    print(f"  ttadapter IR -> {dump_path}", flush=True)


def _measure_performance(case, launch, profile_root):
    """Return the median kernel duration (us) from the NPU profiler."""
    for _ in range(20):
        launch()
    torch.npu.synchronize()
    config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )
    skip_first, warmup, active = 5, 3, 20
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(
                wait=0, warmup=warmup, active=active, repeat=1,
                skip_first=skip_first),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_root)),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=config,
    ) as profiler:
        for _ in range(skip_first + warmup + active):
            launch()
            profiler.step()
    torch.npu.synchronize()

    detail_files = list(Path(profile_root).rglob("kernel_details.csv"))
    durations = []
    for detail_file in detail_files:
        with detail_file.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("Duration(us)"):
                    durations.append(float(row["Duration(us)"]))
    if not durations:
        print(f"PERF {case}: no kernel duration found", flush=True)
        return None
    median_us = statistics.median(durations)
    print(f"PERF {case}: median {median_us:.3f} us (n={len(durations)})", flush=True)
    return median_us


def main():
    kind, mode, block, dtype_name, route, *rest = sys.argv[1:7]
    block = int(block)
    num_warps = int(rest[0]) if rest else 4
    dtype = DTYPES[dtype_name]
    total = VEC * block

    x = (torch.randn((total,), dtype=torch.float32) * 0.1).to(dtype).npu()
    out = torch.zeros((total,), dtype=dtype).npu()
    idx_cpu = (torch.arange(total, dtype=torch.int64) * 37) % total
    idx = idx_cpu.to(torch.int32).npu()

    report_path = f"/tmp/gs1d_{kind}_{route}_{dtype_name}_{block}{os.getenv('GS_TAG', '')}.json"
    if os.path.exists(report_path):
        os.remove(report_path)
    options = {"num_warps": num_warps, "compile_mode": "simd_simt",
               "auto_simt_scope_mode": "off" if route == "plain" else "auto",
               "auto_simt_scope_dump": report_path,
               "logical_program_count_hint": VEC,
               "physical_vector_core_count_hint": VEC}
    kernel = gather_1d if kind == "gather" else scatter_1d
    print(f"##### {kind} {route} dtype={dtype_name} BLOCK={block} warps={num_warps}", flush=True)

    if mode == "warmup":
        try:
            compiled = kernel.warmup(x, idx, out, BLOCK=block, grid=(VEC,), **options)
            print("WARMUP OK", flush=True)
            print(f"  compiled.hash = {compiled.hash}", flush=True)
            for key in ("ttir", "ttadapter", "bcmlir"):
                text = compiled.asm.get(key, "")
                if isinstance(text, bytes):
                    text = text.decode("utf-8", "ignore")
                if text:
                    print(f"  {key}: hfusion.gather_load={text.count('hfusion.gather_load')} "
                          f"hfusion.scatter_store={text.count('hfusion.scatter_store')} "
                          f"scope.scope={text.count('scope.scope')} scf.for={text.count('scf.for')}",
                          flush=True)
            _dump_ttadapter(compiled, kind, route, dtype_name, block)
        except Exception as exc:
            print(f"WARMUP FAILED {type(exc).__name__}", flush=True)
            print("---- tail ----", flush=True)
            print(str(exc)[-1600:], flush=True)
    else:
        kernel[(VEC,)](x, idx, out, BLOCK=block, **options)
        torch.npu.synchronize()
        if kind == "gather":
            ref = torch.gather(x.float(), 0, idx.long()) * 2.0 + 1.0
        else:
            ref = torch.zeros_like(x, dtype=torch.float32)
            ref[idx.long()] = x.float() * 2.0 + 1.0
        diff = float((out.float() - ref).abs().max().item())
        print(f"RUN OK max_abs_diff={diff:.3e}", flush=True)
        profile_root = Path(tempfile.mkdtemp(
            prefix=f"gs1d_{kind}_{route}_{dtype_name}_b{block}_profile_"))
        _measure_performance(
            f"{kind}_{route}",
            lambda: kernel[(VEC,)](x, idx, out, BLOCK=block, **options),
            profile_root)

    if os.path.exists(report_path):
        lines = [ln for ln in open(report_path).read().splitlines() if ln.strip()]
        if lines:
            report = json.loads(lines[-1])
            print("decision:", report.get("effective_decision_kind"),
                  "| anchors:", report.get("materialized_simt_anchor_count"),
                  "| costs:", json.dumps(report.get("candidate_costs", {})), flush=True)


if __name__ == "__main__":
    main()
