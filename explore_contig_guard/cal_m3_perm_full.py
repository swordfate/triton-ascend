#!/usr/bin/env python3
"""calM3: permute FULL body (x,y contiguous loads + z strided load + strided
store, exact original kernel) at k{8,12,16} g32 - the in-context total."""
import os, sys, statistics, tempfile, csv, shutil, json
from pathlib import Path
os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '2')
import torch
import torch_npu
import triton
import triton.language as tl
torch.npu.set_device(0)


def session(kn, args, grid, tag, kname, warps, hint, total=14):
    import torch_npu.profiler as prof
    trace = Path(tempfile.mkdtemp(prefix='calM3_' + tag))

    def fn():
        kn[(grid,)](*args, num_warps=warps, compile_mode='simd',
                    auto_simt_scope_mode='off',
                    logical_program_count_hint=hint,
                    physical_vector_core_count_hint=56, superblock_factor=1)
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
                if kname in (row.get('Name') or ''):
                    durs.append(float(row['Duration(us)']))
    shutil.rmtree(trace, ignore_errors=True)
    return statistics.median(durs) * 1000 if durs else float('nan')


def measure(kn, args, grid, tag, kname, warps, hint):
    vals = [session(kn, args, grid, tag + str(i), kname, warps, hint)
            for i in range(2)]
    m = statistics.median(vals)
    print('%-24s %s -> %.1f' % (tag, ['%.0f' % v for v in vals], m))
    sys.stdout.flush()
    return m


@triton.jit
def cm_perm_full_kernel(a, b, c, out, BLOCK1: tl.constexpr,
                        BLOCK1_SUB: tl.constexpr, BLOCK2: tl.constexpr,
                        S: tl.constexpr, N: tl.constexpr, D: tl.constexpr):
    offset: tl.constexpr = tl.program_id(0) * BLOCK1
    sbase = tl.arange(0, BLOCK1_SUB)
    dbase = tl.arange(0, BLOCK2)
    for loop1 in range(BLOCK1 // BLOCK1_SUB):
        s = offset + loop1 * BLOCK1_SUB + sbase[:, None]
        for n in range(N):
            for loop2 in range(D // BLOCK2):
                d = loop2 * BLOCK2 + dbase[None, :]
                x = tl.load(a + 32768 * n + 8 * s + d)
                y = tl.load(b + 32768 * n + 8 * s + d)
                z = tl.load(c + 8 * n + d + 96 * s)
                tl.store(out + 8 * n + d + 96 * s, x + y + z)


a_full = torch.randn((16 * 32768,), dtype=torch.float32, device='npu')
b_full = torch.randn_like(a_full)
c = torch.randn((4096, 96), dtype=torch.float32, device='npu')
o = torch.empty_like(c)
rows = {}
for N in (8, 12, 16):
    m = measure(cm_perm_full_kernel, (a_full, b_full, c, o, 128, 64, 8, 4096,
                                      N, 8),
                32, 'permFULL_N%d_g32' % N, 'cm_perm_full_kernel', 4, 32)
    rows['attr_perm_full_k%d_g32' % N] = round(m, 1)

out = Path('/root/work/explore_contig/calM3_rows.json')
out.write_text(json.dumps(rows, indent=1))
print('WROTE', out)
