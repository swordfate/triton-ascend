#!/usr/bin/env python3
"""calF2 tiny-store body rows — complete + stability pass.
Body per iter: load 4KB tile (2x512 f32) + tiny bias load (2 f32, masked)
+ sum + tiny masked store (2 f32).  K in {4,16,64} via XBLOCK=2K, grid in
{8,56,224}.  2 sessions per geometry for stability."""
import os, sys, statistics, tempfile, csv, shutil, json
from pathlib import Path
sys.path.insert(0, '/root/work/explore_contig')
os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '2')
import torch
import torch_npu
import triton
import triton.language as tl
torch.npu.set_device(0)


@triton.jit
def tiny_body_kernel(out, inp, bias, xnumel, rnumel, XBLOCK: tl.constexpr,
                     XBLOCK_SUB: tl.constexpr, RBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    for sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + sub + tl.arange(0, XBLOCK_SUB)
        xmask = xindex[:, None] < xnumel
        value = tl.load(inp + rindex + RBLOCK * xindex[:, None], mask=xmask & rmask)
        value = tl.reshape(value, [XBLOCK_SUB, RBLOCK])
        b = tl.load(bias + xindex, mask=xindex < xnumel)
        r = tl.sum(value, axis=1) + b
        tl.store(out + xindex, r)


def session(K, G, tag):
    import torch_npu.profiler as prof
    XB = 2 * K
    total = 14
    trace = Path(tempfile.mkdtemp(prefix='f2t_' + tag))
    xnumel = G * XB
    inp = torch.randn((G, K, 512), dtype=torch.float32, device='npu')
    bias = torch.randn((xnumel,), dtype=torch.float32, device='npu')
    out = torch.empty((xnumel,), dtype=torch.float32, device='npu')

    def fn():
        tiny_body_kernel[(G,)](out, inp, bias, xnumel, 512, XBLOCK=XB,
                               XBLOCK_SUB=2, RBLOCK=512, num_warps=4,
                               compile_mode='simd',
                               auto_simt_scope_mode='off',
                               logical_program_count_hint=G,
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
                if 'tiny_body_kernel' in (row.get('Name') or ''):
                    durs.append(float(row['Duration(us)']))
    shutil.rmtree(trace, ignore_errors=True)
    med = statistics.median(durs) * 1000 if durs else float('nan')
    waves = max(1, (G + 55) // 56)
    return med / waves


rows = {}
for G in (8, 56, 224):
    for K in (4, 16, 64):
        a = session(K, G, 'a')
        b = session(K, G, 'b')
        m = (a + b) / 2.0
        rows['n2_s1_b4096_t8_k%d_g%d' % (K, G)] = round(m, 1)
        print('K=%3d G=%4d  a=%8.1f b=%8.1f  mean=%8.1f' % (K, G, a, b, m))
        sys.stdout.flush()

out = Path('/root/work/explore_contig/tiny_direct_rows.json')
out.write_text(json.dumps(rows, indent=1))
print('WROTE', out)
