#!/usr/bin/env python3
"""calH: boundary-hiding law campaign.

Question: how much of the route-fixed boundary (launch + wave dispatch)
overlaps the contiguous memory stream at multi-wave launches?

Probe bodies (guard-identical lcopy/lload/lstore structure, straight-line):
  - store-only: 1 x 4KB store   (boundary can't hide — reference point)
  - copy:       1 x 4KB load + 1 x 4KB store
  - copy_big:   1 x 64KB load + 1 x 64KB store
  - add:        2 x 4KB loads + 1 x 4KB store
grids: 8, 56, 224, 1024.  Per-geometry sessions, harness Level1 protocol.

Output: whole-kernel medians + the model's candidate for the same geometry
is compared offline; the law is derived as
    measured = alpha * boundary + beta * mem
fits per (grid, body).
"""
import os, sys, statistics, tempfile, csv, shutil, json
from pathlib import Path
sys.path.insert(0, '/root/work/explore_contig')
os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '6')
import torch
import torch_npu
import triton
import triton.language as tl
torch.npu.set_device(0)


@triton.jit
def hcopy_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < n)
    tl.store(y_ptr + offs, v, mask=offs < n)


@triton.jit
def hadd_kernel(x_ptr, z_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(x_ptr + offs, mask=offs < n)
    b = tl.load(z_ptr + offs, mask=offs < n)
    tl.store(y_ptr + offs, a + b, mask=offs < n)


@triton.jit
def hstore_kernel(y_ptr, v, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(y_ptr + offs, v, mask=offs < n)


def session(kn, args_fn, grid, tag, kernel_name):
    import torch_npu.profiler as prof
    total = 14
    trace = Path(tempfile.mkdtemp(prefix='calH_' + tag))

    def fn():
        kn[(grid,)](*args_fn(), BLOCK=BLOCK, num_warps=4,
                    compile_mode='simd', auto_simt_scope_mode='off',
                    logical_program_count_hint=grid,
                    physical_vector_core_count_hint=56)

    for _ in range(5):
        fn()
    torch.npu.synchronize()
    with prof.profile(activities=[prof.ProfilerActivity.NPU],
                      schedule=prof.schedule(wait=1, warmup=2, active=total),
                      on_trace_ready=prof.tensorboard_trace_handler(str(trace))) as p:
        for _ in range(4):
            fn()
        torch.npu.synchronize()
        for _ in range(total):
            fn()
            p.step()
    torch.npu.synchronize()
    durs = []
    for path in sorted(trace.rglob('kernel_details.csv')):
        with path.open(newline='') as h:
            for row in csv.DictReader(h):
                if kernel_name in (row.get('Name') or ''):
                    durs.append(float(row['Duration(us)']))
    shutil.rmtree(trace, ignore_errors=True)
    med = statistics.median(durs) * 1000 if durs else float('nan')
    waves = max(1, (grid + 55) // 56)
    return med, med / waves


BLOCK = 1024  # 4KB f32 tile per program

cases = []
for grid in (8, 56, 224, 1024):
    n = grid * BLOCK
    x = None  # tensors created per grid to bound memory
    def mk(grid=grid, n=grid * BLOCK):
        def args():
            return (X, Y, n)
        return args
    cases.append(('copy', hcopy_kernel, grid, n))

res = {}
for grid in (8, 56, 224, 1024):
    n = grid * BLOCK
    X = torch.randn((n,), dtype=torch.float32, device='npu')
    Z = torch.randn((n,), dtype=torch.float32, device='npu')
    Y = torch.empty((n,), dtype=torch.float32, device='npu')

    whole, pp = session(hstore_kernel, lambda: (Y, 1.0, n), grid,
                        'st%d' % grid, 'hstore_kernel')
    res['store_b4k_g%d' % grid] = whole
    print('store  g%-5d whole=%9.1f perprog=%9.1f' % (grid, whole, pp))
    sys.stdout.flush()

    whole, pp = session(hcopy_kernel, lambda: (X, Y, n), grid,
                        'cp%d' % grid, 'hcopy_kernel')
    res['copy_b4k_g%d' % grid] = whole
    print('copy   g%-5d whole=%9.1f perprog=%9.1f' % (grid, whole, pp))
    sys.stdout.flush()

    whole, pp = session(hadd_kernel, lambda: (X, Z, Y, n), grid,
                        'ad%d' % grid, 'hadd_kernel')
    res['add_b4k_g%d' % grid] = whole
    print('add    g%-5d whole=%9.1f perprog=%9.1f' % (grid, whole, pp))
    sys.stdout.flush()

out = Path('/root/work/explore_contig/calH_boundary_hide.json')
out.write_text(json.dumps(res, indent=1))
print('WROTE', out)
