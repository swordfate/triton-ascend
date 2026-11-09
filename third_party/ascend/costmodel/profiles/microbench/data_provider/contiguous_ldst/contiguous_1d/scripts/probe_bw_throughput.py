#!/usr/bin/env python3
"""带宽型探针 v5：地址窗口与迭代次数解耦，足迹压在 L2 内。"""
import argparse, json, csv, shutil, tempfile
from pathlib import Path
import torch, torch_npu, triton
import triton.language as tl

DT = {'int8': torch.int8, 'fp16': torch.float16, 'fp32': torch.float32,
      'int64': torch.int64}
EB = {'int8': 1, 'fp16': 2, 'fp32': 4, 'int64': 8}


@triton.jit(do_not_specialize=['iters', 'm_dyn'])
def bw_kernel(src, dst, iters, m_dyn, BLOCK: tl.constexpr, WINMASK: tl.constexpr,
              UNROLL: tl.constexpr, STORE: tl.constexpr, FORM: tl.constexpr,
              M_STATIC: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    span = (WINMASK + 1) * BLOCK * UNROLL
    base = pid * span
    if FORM == 1:
        keep = lane < m_dyn
    elif M_STATIC < BLOCK:
        keep = lane < M_STATIC
    else:
        keep = lane < BLOCK
    if STORE == 0:
        acc = tl.zeros((BLOCK,), src.dtype.element_ty)
        for step in range(iters):
            row = base + (step & WINMASK) * BLOCK * UNROLL
            v = tl.zeros((BLOCK,), src.dtype.element_ty)
            for j in tl.static_range(UNROLL):
                v += tl.load(src + row + j * BLOCK + lane, mask=keep, other=0)
            acc += v
        tl.store(dst + pid * BLOCK + lane, acc)
    else:
        val = tl.full((BLOCK,), 1, dst.dtype.element_ty)
        for step in range(iters):
            row = base + (step & WINMASK) * BLOCK * UNROLL
            for j in tl.static_range(UNROLL):
                tl.store(dst + row + j * BLOCK + lane, val, mask=keep)


def prof_time(fn, k, repeats):
    fn(k); torch.npu.synchronize()
    trace = Path(tempfile.mkdtemp(prefix='bwt_'))
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=1, warmup=2, active=repeats),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace))) as prof:
        for _ in range(repeats + 3):
            fn(k); prof.step()
    torch.npu.synchronize()
    durs = []
    for p in sorted(trace.rglob('kernel_details.csv')):
        with p.open(newline='') as h:
            for row in csv.DictReader(h):
                if 'bw_kernel' in (row.get('Name') or '') and row.get('Duration(us)'):
                    durs.append(float(row['Duration(us)']))
    shutil.rmtree(trace, ignore_errors=True)
    return min(durs) if durs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('simd', 'simt_only'), default='simt_only')
    ap.add_argument('--direction', choices=('load', 'store'), default='load')
    ap.add_argument('--dtype', default='fp32')
    ap.add_argument('--bytes', type=int, default=1024)
    ap.add_argument('--valid', type=int, default=0)
    ap.add_argument('--form', choices=('a', 'b'), default='a')
    ap.add_argument('--warps', type=int, default=4)
    ap.add_argument('--grid', type=int, default=56)
    ap.add_argument('--unroll', type=int, default=4)
    ap.add_argument('--window-blocks', type=int, default=0)
    ap.add_argument('--k-low', type=int, default=2048)
    ap.add_argument('--k-high', type=int, default=8192)
    ap.add_argument('--repeats', type=int, default=4)
    ap.add_argument('--manifest', type=Path, required=True)
    a = ap.parse_args()
    if a.mode == 'simd':
        a.warps = 1
    eb = EB[a.dtype]
    nreal = a.valid if a.valid > 0 else a.bytes // eb
    # SIMD 支持非 2 幂 arange，直接按真实长度编译，不做填充也不加掩码
    if a.mode == 'simd':
        block = nreal
    else:
        block = 1
        while block < nreal:
            block *= 2
    if a.mode == 'simt_only' and nreal != block:
        print('RESULT skip nonpow2-simt')
        return
    by = nreal * eb
    if a.window_blocks > 0:
        wb = a.window_blocks
    else:
        wb = max(8, int(64 * 1024 * 1024 / (block * eb * a.unroll * a.grid)))
    wb = max(2, min(wb, a.k_low))
    winmask = wb - 1
    span = (winmask + 1) * block * a.unroll
    n = a.grid * span + block * 8
    if a.dtype == 'fp16':
        src = torch.rand(n, dtype=torch.float16, device='npu')
    else:
        lo_, hi_ = (-120, 120) if a.dtype == 'int8' else (-1024, 1024)
        src = torch.randint(lo_, hi_, (n,), dtype=DT[a.dtype], device='npu')
    dst = torch.empty(n, dtype=DT[a.dtype], device='npu')
    opts = {'compile_mode': a.mode, 'num_warps': a.warps}
    if a.mode == 'simt_only':
        opts['superblock_factor'] = 1
    m_dyn = nreal
    m_static = nreal if nreal < block else block   # SIMD 下 block==nreal，掩码恒真

    def launch(k):
        bw_kernel[(a.grid,)](src, dst, k, m_dyn, block, winmask, a.unroll,
                             0 if a.direction == 'load' else 1,
                             1 if a.form == 'b' else 0, m_static, **opts)

    # 多点线性回归求斜率，比两点差分稳得多；R^2 低就判废
    ks = [a.k_low, 2 * a.k_low, a.k_high]
    ts = []
    for k in ks:
        v = prof_time(launch, k, a.repeats)
        if not v:
            print('RESULT fail'); return
        ts.append(v)
    import statistics
    mx = statistics.fmean(ks)
    my = statistics.fmean(ts)
    sxx = sum((x - mx) ** 2 for x in ks)
    sxy = sum((x - mx) * (y - my) for x, y in zip(ks, ts))
    slope = sxy / sxx if sxx else 0.0
    ss_tot = sum((y - my) ** 2 for y in ts)
    ss_res = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(ks, ts))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    per_iter = slope * 1000.0
    if per_iter <= 0 or r2 < 0.98:
        print('RESULT reject slope=%.1fns r2=%.4f' % (per_iter, r2)); return
    per_acc = per_iter / a.unroll
    mlp = a.grid * a.unroll * by
    rec = dict(status='ok', mode=a.mode, direction=a.direction, dtype=a.dtype,
               bytes=by, nreal=nreal, block=block, form=a.form, warps=a.warps,
               grid=a.grid, unroll=a.unroll, window_blocks=winmask + 1,
               footprint_mb=a.grid * span * eb / 1e6,
               per_iter_ns=per_iter, per_acc_ns=per_acc, mlp_bytes=mlp, r2=r2)
    with a.manifest.open('a') as f:
        f.write(json.dumps(rec) + '\n')
    print('RESULT %-9s %-5s %-5s %-5s b%-6d W%-2d u%-2d wb%-4d | perAcc=%8.1fns foot=%5.1fMB r2=%.4f' % (
        a.mode, a.direction, a.dtype, a.form, by, a.warps, a.unroll,
        winmask + 1, per_acc, rec['footprint_mb'], r2))


if __name__ == '__main__':
    main()
