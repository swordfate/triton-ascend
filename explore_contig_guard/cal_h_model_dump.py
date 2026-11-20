#!/usr/bin/env python3
"""calH model side: same calH launches with auto route dumps -> candidate.
Writes /root/work/explore_contig/calH_model.json {key: candidate_ns}."""
import os, sys, json
from pathlib import Path
sys.path.insert(0, '/root/work/explore_contig')
os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '6')
import torch
import torch_npu
import triton
import triton.language as tl
torch.npu.set_device(0)
sys.path.insert(0, '/root/work/explore_contig')
import cal_h_boundary_hide as H  # reuse kernels/BLOCK

out = {}
for grid in (8, 56, 224, 1024):
    n = grid * H.BLOCK
    X = torch.randn((n,), dtype=torch.float32, device='npu')
    Z = torch.randn((n,), dtype=torch.float32, device='npu')
    Y = torch.empty((n,), dtype=torch.float32, device='npu')
    for name, kn, args in (
            ('store', H.hstore_kernel, lambda: (Y, 1.0, n)),
            ('copy', H.hcopy_kernel, lambda: (X, Y, n)),
            ('add', H.hadd_kernel, lambda: (X, Z, Y, n))):
        dump = '/tmp/calH_%s_g%d.route.json' % (name, grid)
        kn[(grid,)](*args(), BLOCK=H.BLOCK, num_warps=4,
                    compile_mode='simd_simt', auto_simt_scope_mode='auto',
                    auto_simt_scope_dump=dump,
                    logical_program_count_hint=grid,
                    physical_vector_core_count_hint=56)
        torch.npu.synchronize()
        out['%s_b4k_g%d' % (name, grid)] = dump

res = {}
dec = json.JSONDecoder()


def docs(p):
    s = open(p).read()
    i = 0
    o = []
    while i < len(s):
        r = s[i:].lstrip()
        if not r:
            break
        off = len(s) - i - len(r)
        try:
            x, e = dec.raw_decode(r)
            o.append(x)
            i = i + off + e
        except Exception:
            break
    return o


for k, f in out.items():
    d = docs(f)[-1]
    kind = d.get('effective_decision_kind')
    ce = (d.get('candidate_elapsed_costs') or {}).get(kind, 0)
    sm = d.get('stage_model') or {}
    rt = (sm.get('routes') or {}).get(kind) or {}
    res[k] = {'cand': ce, 'route': rt.get('total_system_cycles', 0)}
    print('%-18s cand=%9.1f route=%9.1f' % (k, ce, rt.get('total_system_cycles', 0)))

Path('/root/work/explore_contig/calH_model.json').write_text(json.dumps(res, indent=1))
print('WROTE calH_model.json')
