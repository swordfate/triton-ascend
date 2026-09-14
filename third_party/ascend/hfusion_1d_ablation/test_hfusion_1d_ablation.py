#!/usr/bin/env python3
"""1D gather/scatter hfusion ablation cases.

Two tiny kernels exercise the triton-ascend unstructured-access lowering:

  * auto (route="auto"): compile_mode="simd_simt",
    auto_simt_scope_mode="auto".  The C++ cost model selects
    mixed_simd_simt, materializes the indirect stage as a local SIMT scope
    and hfusion.gather_load / hfusion.scatter_store appears inside it.

  * plain (route="plain"): compile_mode="simd_simt",
    auto_simt_scope_mode="off" and no scope at all.  The legacy
    backend_default recognition lowers the indirect access globally to the
    same hfusion ops, without any scope.scope.

Two scoped ablations are supported (both keep the cost-model scope):

  * fallback (patches/fallback_in_simt_scope.patch):
    `useUnstructuredOp` is forced false for ops inside vector_mode="simt",
    so the scoped access goes through UnstructuredMemAccessConverter's legacy
    scalar-loop fallback: scope.scope + scf.for + scalar tt.load/store.
  * keep_tt_load (patches/keep_tt_load_in_simt_scope.patch):
    UnstructuredMemAccessConverter returns failure for scoped ops, leaving
    the original tile tt.load/tt.store for TritonToLinalg.  No unstructured
    op, no scalar fallback.

TRITON_HFUSION_ABLATION selects which patched wheel this test expects:

  * on (default): no patch; auto and plain both produce hfusion ops.
  * fallback: auto keeps scope.scope but has no hfusion and does contain the
    scf.for scalar fallback; plain keeps legacy global hfusion.
  * keep_tt_load: auto keeps scope.scope and the original tile tt.load/store,
    with no hfusion and no scf.for; plain keeps legacy global hfusion.

Each compiled case writes its ttadapter IR to a standalone file and prints its
path (default /tmp/triton_hfusion_1d_ablation_ir, override with
TRITON_HFUSION_ABLATION_IR_DIR).

"""
import csv
import json
import os
import statistics
import tempfile
import warnings
from pathlib import Path

import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver
from triton.backends.ascend.utils import is_compile_on_910_95

simd_simt_910_95_only = pytest.mark.xfail(
    not is_compile_on_910_95(),
    reason="SIMD/SIMT cost model only supports 910_95",
    run=False,
)

BLOCK = 8192
NUM_WARPS = 4
HFUSION_ABLATION = os.getenv("TRITON_HFUSION_ABLATION", "on").lower()
HFUSION_FALLBACK = HFUSION_ABLATION == "fallback"          # scoped op -> scalar fallback
HFUSION_KEEP_TT_LOAD = HFUSION_ABLATION == "keep_tt_load"  # scoped op -> original tt.load


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


def _vector_core_count():
    properties = driver.active.utils.get_device_properties(torch.npu.current_device())
    return int(properties["num_vectorcore"])


def _hfusion_token(kind):
    return "hfusion.gather_load" if kind == "gather" else "hfusion.scatter_store"


def _kernel(kind):
    return gather_1d if kind == "gather" else scatter_1d


def _make_inputs(kind, dtype, block, programs):
    total = programs * block
    x = (torch.randn((total,), dtype=torch.float32) * 0.1).to(dtype).npu()
    out = torch.zeros((total,), dtype=dtype).npu()
    # 37 is coprime with programs*block for programs=56, block=8192.
    idx = ((torch.arange(total, dtype=torch.int64) * 37) % total).to(torch.int32).npu()
    return x, idx, out


def _compile(kind, x, idx, out, programs, route, report_path,
             block=BLOCK, num_warps=NUM_WARPS):
    kernel = _kernel(kind)
    options = {
        "num_warps": num_warps,
        "compile_mode": "simd_simt",
        "auto_simt_scope_mode": "auto" if route == "auto" else "off",
        "auto_simt_scope_dump": report_path,
        "logical_program_count_hint": programs,
        "physical_vector_core_count_hint": programs,
    }
    compiled = kernel.warmup(x, idx, out, BLOCK=block, grid=(programs,), **options)
    return kernel, compiled, options


def _launch(kernel, x, idx, out, programs, options, block=BLOCK):
    kernel[(programs,)](x, idx, out, BLOCK=block, **options)
    torch.npu.synchronize()


def _asm_text(compiled, key="ttadapter"):
    text = compiled.asm.get(key, "")
    if isinstance(text, bytes):
        text = text.decode("utf-8", "ignore")
    return str(text)


def _ablation_ir_dir():
    default_dir = Path(tempfile.gettempdir()) / "triton_hfusion_1d_ablation_ir"
    return Path(os.environ.get("TRITON_HFUSION_ABLATION_IR_DIR", str(default_dir)))


def _dump_ttadapter(compiled, kind, route):
    """Write the ttadapter IR to one self-contained file and print its path."""
    text = _asm_text(compiled, "ttadapter")
    out_dir = _ablation_ir_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    if route == "auto" and HFUSION_FALLBACK:
        tag = route + "_fallback"
    elif route == "auto" and HFUSION_KEEP_TT_LOAD:
        tag = route + "_keep_tt_load"
    else:
        tag = route
    path = out_dir / f"{kind}_{tag}.ttadapter.mlir"
    path.write_text(text)
    message = f"[hfusion-ablation] {kind}/{route}: ttadapter IR -> {path}"
    print(message, flush=True)
    warnings.warn(message, stacklevel=1)
    return path


def _scope_body(text):
    """Return the first scope.scope body (up to its scope.return)."""
    start = text.find("scope.scope")
    if start < 0:
        return ""
    end = text.find("scope.return", start)
    return text[start:end if end >= 0 else len(text)]


def _assert_numeric(kind, x, idx, out):
    if kind == "gather":
        reference = torch.gather(x.float(), 0, idx.long()) * 2.0 + 1.0
        max_abs_diff = (out.float() - reference).abs().max().item()
        assert max_abs_diff <= 1e-2, f"gather max_abs_diff={max_abs_diff}"
    else:
        reference = torch.zeros_like(x, dtype=torch.float32)
        reference[idx.long()] = x.float() * 2.0 + 1.0
        max_abs_diff = (out.float() - reference).abs().max().item()
        assert max_abs_diff == 0.0, f"scatter max_abs_diff={max_abs_diff}"


def _measure_performance(case, launch, profile_root):
    """Mirror of test_simd_simt_costmodel_cases._assert_performance.

    Returns the median kernel duration (us) reported by the NPU profiler.
    """
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
    assert detail_files, f"{case}: profiler did not generate kernel_details.csv"
    durations = []
    for detail_file in detail_files:
        with detail_file.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("Duration(us)"):
                    durations.append(float(row["Duration(us)"]))
    assert durations, f"{case}: profiler generated no kernel duration"
    median_us = statistics.median(durations)
    message = f"[hfusion-ablation] {case}: median kernel time {median_us:.3f} us (n={len(durations)})"
    print(message, flush=True)
    warnings.warn(message, stacklevel=1)
    return median_us


@simd_simt_910_95_only
@pytest.mark.parametrize("kind,dtype", [("gather", torch.float16), ("scatter", torch.int8)],
                         ids=["gather_f16", "scatter_i8"])
def test_costmodel_mixed_triggers_hfusion(kind, dtype, tmp_path):
    programs = _vector_core_count()
    x, idx, out = _make_inputs(kind, dtype, BLOCK, programs)
    report_path = tmp_path / f"{kind}_auto_route.json"
    kernel, compiled, options = _compile(kind, x, idx, out, programs, "auto",
                                         str(report_path))
    _dump_ttadapter(compiled, kind, "auto")
    text = _asm_text(compiled)
    token = _hfusion_token(kind)
    scope_body = _scope_body(text)
    if HFUSION_FALLBACK:
        # scope.scope stays, but the scoped access is scalarized by the
        # UnstructuredMemAccessConverter legacy fallback.
        assert token not in text, f"{token} must be gone in fallback mode"
        assert "scope.scope" in text, "cost-model scope must still be materialized"
        assert "ascend.unstructured_load" not in text
        assert "ascend.unstructured_store" not in text
        assert "scf.for" in scope_body, "fallback must produce a scoped scf.for loop"
    elif HFUSION_KEEP_TT_LOAD:
        # scope.scope stays and the original tile tt.load/store is untouched.
        assert token not in text, f"{token} must be gone in keep_tt_load mode"
        assert "scope.scope" in text, "cost-model scope must still be materialized"
        assert "ascend.unstructured_load" not in text
        assert "ascend.unstructured_store" not in text
        assert "tt.load" in scope_body, "original tile tt.load must stay inside the scope"
        assert "scf.for" not in scope_body, "keep_tt_load must not create a scalar loop"
    else:
        assert token in text, f"expected {token} in auto ttadapter"
        assert "scope.scope" in text, "costmodel mixed route must materialize a SIMT scope"

    report = json.loads(report_path.read_text().splitlines()[-1])
    assert report["effective_decision_kind"] == "mixed_simd_simt"
    assert report["materialized_simt_anchor_count"] == 1

    _launch(kernel, x, idx, out, programs, options)
    _assert_numeric(kind, x, idx, out)
    _measure_performance(
        f"{kind}_auto",
        lambda: kernel[(programs,)](x, idx, out, BLOCK=BLOCK, **options),
        tmp_path / f"{kind}_auto_profile")


@simd_simt_910_95_only
@pytest.mark.parametrize("kind,dtype", [("gather", torch.float16), ("scatter", torch.int8)],
                         ids=["gather_f16", "scatter_i8"])
def test_plain_legacy_hfusion_and_ablation(kind, dtype, tmp_path):
    programs = _vector_core_count()
    x, idx, out = _make_inputs(kind, dtype, BLOCK, programs)
    report_path = tmp_path / f"{kind}_plain_route.json"
    kernel, compiled, options = _compile(kind, x, idx, out, programs, "plain",
                                         str(report_path))
    _dump_ttadapter(compiled, kind, "plain")
    text = _asm_text(compiled)
    token = _hfusion_token(kind)

    # plain has no scope, so the scope-only ablation does not affect it:
    # legacy backend_default recognition still lowers the indirect access
    # globally, without creating a scope.
    assert token in text, f"expected {token} in plain ttadapter"
    assert "scope.scope" not in text, "legacy plain path must not create a scope"

    _launch(kernel, x, idx, out, programs, options)
    _assert_numeric(kind, x, idx, out)
    _measure_performance(
        f"{kind}_plain",
        lambda: kernel[(programs,)](x, idx, out, BLOCK=BLOCK, **options),
        tmp_path / f"{kind}_plain_profile")
