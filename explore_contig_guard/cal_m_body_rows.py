#!/usr/bin/env python3
"""calM (extended): body-exact rows + in-context ablations for the 4
row-gap cases.  Card 2 only.  Profiler-median whole-kernel ns (calJ method).

Rows:
  permute   n2_s0_b2048(f32)   k{4,8,12,16}  g{8,32,56}  simd loads-only
  fused_sm  n1_s1_b2048(f32)   k{1,4,16,64}  g{8,32,56}  simd masked ldst
  rowred    n1_s0_b8192(f32)   k{4,16,64}    g56         simt loads-only
  masked_sm n1_s1_b65536(f32)  k{4,8,16}     g{8,56}     simd 64KB bucket ldst
            n1_s1_b48000-ish real body (80,150) k{4,8,16,18} g56 (context ref)
Ablations (full kernels, multibuffer True/False):
  fused full (math) k4 g32;  rowred full k16 g56 simt;  masked80 full k18 g56
  rowred loads+reduce (no stores) k16 g56 simt."""
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
    trace = Path(tempfile.mkdtemp(prefix='calM_' + tag))

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


# ---- A. permute loads-only (x,y contiguous 2048B; exact loop nest) ----
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


# ---- B/B2. fused_softmax: masked 512-wide (257 active); MATH switch ----
@triton.jit
def cm_fsoft_kernel(out, inp, input_stride, output_stride, n_rows, n_cols,
                    BLOCK_SIZE: tl.constexpr, MATH: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row in tl.range(row_start, n_rows, row_step):
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        value = tl.load(inp + row * input_stride + cols, mask=mask,
                        other=-float('inf'))
        if MATH == 1:
            numerator = tl.exp(value - tl.max(value, axis=0))
            result = numerator / tl.sum(numerator, axis=0)
        else:
            result = value
        tl.store(out + row * output_stride + cols, result, mask=mask)


# ---- C/C2/C3. rowred bodies (SIMT) ----
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
def cm_rowred_full_kernel(input_ptr, output_ptr, rows, RPP: tl.constexpr,
                          COLS: tl.constexpr, REPEATS: tl.constexpr):
    pid = tl.program_id(0)
    row_lane = tl.arange(0, RPP)[:, None]
    col = tl.arange(0, COLS)[None, :]
    for repeat in range(REPEATS):
        row = (pid * REPEATS + repeat) * RPP + row_lane
        mask = row < rows
        value = tl.load(input_ptr + row * COLS + col, mask=mask, other=0.0)
        maximum = tl.max(value, axis=1).reshape(RPP, 1)
        total = tl.sum(value, axis=1).reshape(RPP, 1)
        tl.store(output_ptr + row * 2, maximum, mask=mask)
        tl.store(output_ptr + row * 2 + 1, total, mask=mask)


@triton.jit
def cm_rowred_nostore_kernel(input_ptr, keep, rows, RPP: tl.constexpr,
                             COLS: tl.constexpr, REPEATS: tl.constexpr):
    pid = tl.program_id(0)
    row_lane = tl.arange(0, RPP)[:, None]
    col = tl.arange(0, COLS)[None, :]
    acc = tl.zeros((RPP, 1), dtype=tl.float32)
    for repeat in range(REPEATS):
        row = (pid * REPEATS + repeat) * RPP + row_lane
        mask = row < rows
        value = tl.load(input_ptr + row * COLS + col, mask=mask, other=0.0)
        maximum = tl.max(value, axis=1).reshape(RPP, 1)
        total = tl.sum(value, axis=1).reshape(RPP, 1)
        acc += maximum + total
    tl.store(keep + pid * RPP + tl.arange(0, RPP), acc.reshape(RPP))


# ---- D/D2/D3. masked_sm bodies ----
@triton.jit
def cm_msm_kernel(inp, out, rows, RPP: tl.constexpr, LC: tl.constexpr,
                  SC: tl.constexpr, LOOPS: tl.constexpr):
    pid = tl.program_id(0)
    row_lane = tl.arange(0, RPP)[:, None]
    col = tl.arange(0, LC)[None, :]
    for repeat in range(LOOPS):
        row = (pid * LOOPS + repeat) * RPP + row_lane
        mask = row < rows
        offsets = row * SC + col
        value = tl.load(inp + offsets, mask=mask, other=0.0)
        tl.store(out + offsets, value + 1.0, mask=mask)


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

# A. permute
a_full = torch.randn((12 * 32768,), dtype=torch.float32, device='npu')
b_full = torch.randn_like(a_full)
keep32 = torch.zeros((32,), dtype=torch.float32, device='npu')
for N in (4, 8, 12, 16):
    for g in (8, 32, 56):
        m, pp = measure(cm_perm_kernel, (a_full, b_full, keep32, 128, 64, 8,
                                         4096, N, 8),
                        g, 'perm_N%d_g%d' % (N, g), 'cm_perm_kernel', 4, g)
        rows['n2_s0_b2048_k%d_g%d_m0' % (N, g)] = round(pp, 1)

# B. fused memory-only rows
for K in (1, 4, 16, 64):
    for g in (8, 32, 56):
        n_rows = K * g
        x = torch.randn((n_rows, 257), dtype=torch.float32, device='npu')
        o = torch.empty_like(x)
        m, pp = measure(cm_fsoft_kernel,
                        (o, x, 257, 257, n_rows, 257, 512, 0),
                        g, 'fsoft_k%d_g%d' % (K, g), 'cm_fsoft_kernel', 4, g)
        rows['n1_s1_b2048_k%d_g%d_m0' % (K, g)] = round(pp, 1)

# B2. fused full (math) ablation: multibuffer on/off, case geometry k4 g32
x = torch.randn((128, 257), dtype=torch.float32, device='npu')
o = torch.empty_like(x)
for mb in (True, False):
    m, pp = measure(cm_fsoft_kernel, (o, x, 257, 257, 128, 257, 512, 1),
                    32, 'fsoftFULL_mb%d' % mb, 'cm_fsoft_kernel', 4, 32,
                    multibuffer=mb)
    rows['attr_fsoft_full_k4_g32_mb%s' % mb] = round(pp, 1)

# C. rowred loads-only rows (simt)
for K in (4, 16, 64):
    total_rows = 56 * K * 16
    xx = torch.randn((total_rows, 128), dtype=torch.float32, device='npu')
    keep = torch.zeros((56 * 16,), dtype=torch.float32, device='npu')
    m, pp = measure(cm_rowred_loads_kernel, (xx, keep, total_rows, 16, 128, K),
                    56, 'rowredL_k%d_g56' % K, 'cm_rowred_loads_kernel', 4, 56,
                    mode='simt_only', multibuffer=True)
    rows['n1_s0_b8192_k%d_g56_m0' % K] = round(pp, 1)

# C3. rowred loads+reduce (no stores) — attribution
total_rows = 56 * 16 * 16
xx = torch.randn((total_rows, 128), dtype=torch.float32, device='npu')
keep = torch.zeros((56 * 16,), dtype=torch.float32, device='npu')
m, pp = measure(cm_rowred_nostore_kernel, (xx, keep, total_rows, 16, 128, 16),
                56, 'rowredLR_k16_g56', 'cm_rowred_nostore_kernel', 4, 56,
                mode='simt_only', multibuffer=True)
rows['attr_rowred_loadreduce_k16_g56'] = round(pp, 1)

# C2. rowred full ablation
o2 = torch.empty((total_rows * 2,), dtype=torch.float32, device='npu')
for mb in (True, False):
    m, pp = measure(cm_rowred_full_kernel, (xx, o2, total_rows, 16, 128, 16),
                    56, 'rowredF_mb%d' % mb, 'cm_rowred_full_kernel', 4, 56,
                    mode='simt_only', multibuffer=mb)
    rows['attr_rowred_full_k16_g56_mb%s' % mb] = round(pp, 1)

# D. masked_sm real-body context rows
for K in (4, 8, 16, 18):
    total_rows = 56 * K * 80
    x3 = torch.randn((total_rows, 150), dtype=torch.float32, device='npu')
    o3 = torch.empty_like(x3)
    m, pp = measure(cm_msm_kernel, (x3, o3, total_rows, 80, 150, 150, K),
                    56, 'msm_k%d_g56' % K, 'cm_msm_kernel', 4, 56,
                    multibuffer=True)
    rows['n1_s1_b48000_k%d_g56_m0' % K] = round(pp, 1)

# D2. masked_sm bucket body (64KB 1D chunks) rows
CH = 16384
for K in (4, 8, 16):
    for g in (8, 56):
        chunks = g * K
        x4 = torch.randn((chunks * CH,), dtype=torch.float32, device='npu')
        y4 = torch.empty_like(x4)
        m, pp = measure(cm_bucket_kernel, (x4, y4, 0, CH, K),
                        g, 'bucket_k%d_g%d' % (K, g), 'cm_bucket_kernel', 4, g)
        rows['n1_s1_b65536_k%d_g%d_m0' % (K, g)] = round(pp, 1)

# D3. masked80 full (softmax) ablation
total_rows = 56 * 18 * 80
x5 = torch.randn((total_rows, 150), dtype=torch.float32, device='npu')
o5 = torch.empty_like(x5)
for mb in (True, False):
    m, pp = measure(cm_msm_full_kernel, (x5, o5, total_rows, 80, 150, 150, 18),
                    56, 'msmFULL_mb%d' % mb, 'cm_msm_full_kernel', 4, 56,
                    multibuffer=mb)
    rows['attr_msm_full_k18_g56_mb%s' % mb] = round(pp, 1)

out = Path('/root/work/explore_contig/calM_rows.json')
out.write_text(json.dumps(rows, indent=1))
print('WROTE', out)
