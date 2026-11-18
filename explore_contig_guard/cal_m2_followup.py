#!/usr/bin/env python3
"""calM2 (final): the exact row/ablation points still missing.
  rowred simt loads-only mbTrue: g8 k{4,16,64} (final-cost bracket),
                                g56 k8 (trend continuity)
  fused FULL-math body: g8 k{1,4,16,64} (final-cost rows, calJ convention)
  masked80 bucket body g8 k64 (complete the g8 trend, replace poisoned row)
  permute z-strided body N=12 g32 (legacy-charge attribution)
  permute x/y loads-only N=12 g8/g56 confirm (bracket sanity)
"""
import os, sys, statistics, tempfile, csv, shutil, json
from pathlib import Path
os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '2')
import torch
import torch_npu
import triton
import triton.language as tl
torch.npu.set_device(0)


def session(kn, args, grid, tag, kname, warps, hint, mode='simd',
            multibuffer=None, total=14):
    import torch_npu.profiler as prof
    trace = Path(tempfile.mkdtemp(prefix='calM2_' + tag))

    def fn():
        kw = dict(num_warps=warps, compile_mode=mode,
                  auto_simt_scope_mode='off',
                  logical_program_count_hint=hint,
                  physical_vector_core_count_hint=56,
                  superblock_factor=1)
        if multibuffer is not None:
            kw['multibuffer'] = multibuffer
        kn[(grid,)](*args, **kw)
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


def measure(kn, args, grid, tag, kname, warps, hint, mode='simd',
            multibuffer=None):
    vals = [session(kn, args, grid, tag + str(i), kname, warps, hint, mode,
                    multibuffer) for i in range(2)]
    m = statistics.median(vals)
    waves = max(1, (grid + 55) // 56)
    print('%-30s %s -> %.1f (perprog %.1f)' % (tag, ['%.0f' % v for v in vals],
                                               m, m / waves))
    sys.stdout.flush()
    return m, m / waves


@triton.jit
def cm_rowred_loads_kernel(input_ptr, keep, rows, RPP: tl.constexpr,
                           COLS: tl.constexpr, REPEATS: tl.constexpr):
    pid = tl.program_id(0)
    row_lane = tl.arange(0, RPP)[:, None]
    col = tl.arange(0, COLS)[None, :]
    acc = tl.zeros((RPP, 1), dtype=tl.float32)
    for repeat in range(REPEATS):
        row = (pid * REPEATS + repeat) * RPP + row_lane
        mask = row < rows
        value = tl.load(input_ptr + row * COLS + col, mask=mask, other=0.0)
        acc += tl.sum(value, axis=1).reshape(RPP, 1)
    tl.store(keep + pid * RPP + tl.arange(0, RPP), acc.reshape(RPP))


@triton.jit
def cm_zstrd_kernel(a, b, c, keep, BLOCK1: tl.constexpr,
                    BLOCK1_SUB: tl.constexpr, BLOCK2: tl.constexpr,
                    S: tl.constexpr, N: tl.constexpr, D: tl.constexpr):
    offset: tl.constexpr = tl.program_id(0) * BLOCK1
    sbase = tl.arange(0, BLOCK1_SUB)
    dbase = tl.arange(0, BLOCK2)
    acc = tl.zeros((BLOCK1_SUB, BLOCK2), dtype=tl.float32)
    for loop1 in range(BLOCK1 // BLOCK1_SUB):
        s = offset + loop1 * BLOCK1_SUB + sbase[:, None]
        for n in range(N):
            for loop2 in range(D // BLOCK2):
                d = loop2 * BLOCK2 + dbase[None, :]
                z = tl.load(c + 8 * n + d + 96 * s)
                acc += z
    tl.store(keep + tl.program_id(0), tl.sum(acc))


@triton.jit
def cm_fsoft_full_kernel(out, inp, input_stride, output_stride, n_rows,
                         n_cols, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row in tl.range(row_start, n_rows, row_step):
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        value = tl.load(inp + row * input_stride + cols, mask=mask,
                        other=-float('inf'))
        numerator = tl.exp(value - tl.max(value, axis=0))
        result = numerator / tl.sum(numerator, axis=0)
        tl.store(out + row * output_stride + cols, result, mask=mask)


@triton.jit
def cm_perm_kernel(a, b, keep, BLOCK1: tl.constexpr, BLOCK1_SUB: tl.constexpr,
                   BLOCK2: tl.constexpr, S: tl.constexpr, N: tl.constexpr,
                   D: tl.constexpr):
    offset: tl.constexpr = tl.program_id(0) * BLOCK1
    sbase = tl.arange(0, BLOCK1_SUB)
    dbase = tl.arange(0, BLOCK2)
    acc = tl.zeros((BLOCK1_SUB, BLOCK2), dtype=tl.float32)
    for loop1 in range(BLOCK1 // BLOCK1_SUB):
        s = offset + loop1 * BLOCK1_SUB + sbase[:, None]
        for n in range(N):
            for loop2 in range(D // BLOCK2):
                d = loop2 * BLOCK2 + dbase[None, :]
                x = tl.load(a + 32768 * n + 8 * s + d)
                y = tl.load(b + 32768 * n + 8 * s + d)
                acc += x + y
    tl.store(keep + tl.program_id(0), tl.sum(acc))


@triton.jit
def cm_bucket_kernel(x_ptr, y_ptr, base, CH: tl.constexpr,
                     LOOPS: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, CH)
    for repeat in range(LOOPS):
        off = base + (pid * LOOPS + repeat) * CH + cols
        v = tl.load(x_ptr + off)
        tl.store(y_ptr + off, v + 1.0)


@triton.jit
def cm_msm_full_kernel(inp, out, rows, RPP: tl.constexpr, LC: tl.constexpr,
                       SC: tl.constexpr, LOOPS: tl.constexpr):
    pid = tl.program_id(0)
    row_lane = tl.arange(0, RPP)[:, None]
    col = tl.arange(0, LC)[None, :]
    for repeat in range(LOOPS):
        row = (pid * LOOPS + repeat) * RPP + row_lane
        row_mask = row < rows
        mask = row_mask & (col < LC)
        offsets = row * SC + col
        value = tl.load(inp + offsets, mask=mask, other=-float('inf'))
        maximum = tl.max(value, axis=1).reshape(RPP, 1)
        numerator = tl.exp(value - maximum)
        denominator = tl.sum(numerator, axis=1).reshape(RPP, 1)
        tl.store(out + offsets, numerator / denominator, mask=mask)


rows = {}

# 1. rowred simt loads-only mbTrue at g8 (final-cost bracket) + g56 k8
for K in (4, 16, 64):
    total_rows = 8 * K * 16
    x = torch.randn((total_rows, 128), dtype=torch.float32, device='npu')
    keep = torch.zeros((8 * 16,), dtype=torch.float32, device='npu')
    m, pp = measure(cm_rowred_loads_kernel, (x, keep, total_rows, 16, 128, K),
                    8, 'rowredL_k%d_g8' % K, 'cm_rowred_loads_kernel', 4, 8,
                    mode='simt_only', multibuffer=True)
    rows['n1_s0_b8192_k%d_g8_m0' % K] = round(pp, 1)
total_rows = 56 * 8 * 16
x = torch.randn((total_rows, 128), dtype=torch.float32, device='npu')
keep = torch.zeros((56 * 16,), dtype=torch.float32, device='npu')
m, pp = measure(cm_rowred_loads_kernel, (x, keep, total_rows, 16, 128, 8),
                56, 'rowredL_k8_g56', 'cm_rowred_loads_kernel', 4, 56,
                mode='simt_only', multibuffer=True)
rows['n1_s0_b8192_k8_g56_m0'] = round(pp, 1)

# 2. fused FULL-math g8 rows (calJ convention: whole-body measurement)
for K in (1, 4, 16, 64):
    n_rows = K * 8
    xx = torch.randn((n_rows, 257), dtype=torch.float32, device='npu')
    o = torch.empty_like(xx)
    m, pp = measure(cm_fsoft_full_kernel, (o, xx, 257, 257, n_rows, 257, 512),
                    8, 'fsoftFULL_k%d_g8' % K, 'cm_fsoft_full_kernel', 4, 8)
    rows['n1_s1_b2048_k%d_g8_m0' % K] = round(pp, 1)
# g32 k4 full (route-eval sanity)
xx = torch.randn((128, 257), dtype=torch.float32, device='npu')
o = torch.empty_like(xx)
m, pp = measure(cm_fsoft_full_kernel, (o, xx, 257, 257, 128, 257, 512),
                32, 'fsoftFULL_k4_g32', 'cm_fsoft_full_kernel', 4, 32)
rows['n1_s1_b2048_k4_g32_m0'] = round(pp, 1)

# 3. masked80 bucket body g8 k64
CH = 16384
chunks = 8 * 64
x4 = torch.randn((chunks * CH,), dtype=torch.float32, device='npu')
y4 = torch.empty_like(x4)
m, pp = measure(cm_bucket_kernel, (x4, y4, 0, CH, 64),
                8, 'bucket_k64_g8', 'cm_bucket_kernel', 4, 8)
rows['n1_s1_b65536_k64_g8_m0'] = round(pp, 1)

# 4. permute z-strided body (attribution) + x/y bracket sanity
c = torch.randn((4096, 96), dtype=torch.float32, device='npu')
keep32 = torch.zeros((32,), dtype=torch.float32, device='npu')
m, pp = measure(cm_zstrd_kernel, (c, c, c, keep32, 128, 64, 8, 4096, 12, 8),
                32, 'zstrd_N12_g32', 'cm_zstrd_kernel', 4, 32)
rows['attr_perm_zstrd_k12_g32'] = round(pp, 1)
a_full = torch.randn((12 * 32768,), dtype=torch.float32, device='npu')
b_full = torch.randn_like(a_full)
for g in (8, 56):
    m, pp = measure(cm_perm_kernel, (a_full, b_full, keep32, 128, 64, 8,
                                     4096, 12, 8),
                    g, 'perm_N12_g%d' % g, 'cm_perm_kernel', 4, g)
    rows['n2_s0_b2048_k12_g%d_m0' % g] = round(pp, 1)

# 5. masked80 FULL kernel (softmax) at g8, k{4,8,16,18} — real-body rows
for K in (4, 8, 16, 18):
    total_rows = 8 * K * 80
    x5 = torch.randn((total_rows, 150), dtype=torch.float32, device='npu')
    o5 = torch.empty_like(x5)
    m, pp = measure(cm_msm_full_kernel, (x5, o5, total_rows, 80, 150, 150, K),
                    8, 'msmFULL_k%d_g8' % K, 'cm_msm_full_kernel', 4, 8,
                    multibuffer=True)
    rows['n1_s1_b48000full_k%d_g8_m0' % K] = round(pp, 1)

out = Path('/root/work/explore_contig/calM2_rows.json')
out.write_text(json.dumps(rows, indent=1))
print('WROTE', out)
