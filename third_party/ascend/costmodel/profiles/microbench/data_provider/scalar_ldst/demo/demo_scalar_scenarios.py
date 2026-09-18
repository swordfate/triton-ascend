import torch
import torch_npu
import triton
import triton.language as tl
import statistics
import argparse


@triton.jit
def scalar_ldst_kernel(a, b, out, N_LD: tl.constexpr, N_ST: tl.constexpr):
    pid = tl.program_id(0)
    s = 0
    for i in tl.static_range(N_LD):
        s += tl.load(a + pid * N_LD + i)
    for i in tl.static_range(N_ST):
        tl.store(b + pid * N_ST + i, s + i)
    tl.store(out + pid, s)


@triton.jit
def scalar_compute_kernel(a, out, N: tl.constexpr):
    pid = tl.program_id(0)
    s = tl.load(a + pid)
    for i in tl.static_range(N):
        s = s * 3 + i
        s = s ^ 0x2c6b
    tl.store(out + pid, s)


@triton.jit
def scalar_control_kernel(a, out, N: tl.constexpr):
    pid = tl.program_id(0)
    s = tl.load(a + pid)
    acc = 0
    for i in tl.static_range(N):
        if (s & 1) == 0:
            acc += i
        else:
            acc -= i
        s = (s >> 1) ^ 0x2c6b
    tl.store(out + pid, acc)


def opts_for(mode, sf, grid):
    if mode == "simd":
        return {"num_warps": 1, "compile_mode": "simd", "auto_simt_scope_mode": "off",
                "enable_auto_blockify": False, "superblock_factor": 1,
                "logical_program_count_hint": grid}
    return {"num_warps": 1, "compile_mode": mode, "auto_simt_scope_mode": "off",
            "enable_auto_blockify": True, "superblock_factor": sf,
            "logical_program_count_hint": grid}


def measure_launch(launch, reps=8):
    launch()
    torch.npu.synchronize()
    ts = []
    for _ in range(reps):
        st = torch.npu.Event(enable_timing=True)
        en = torch.npu.Event(enable_timing=True)
        st.record()
        launch()
        en.record()
        torch.npu.synchronize()
        ts.append(st.elapsed_time(en))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', type=int, default=2048)
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--mode', choices=['simd', 'simt_only', 'all'], default='all')
    args = ap.parse_args()
    grid = args.grid
    n = args.n

    # scalar load/store
    a = torch.arange(grid * n, dtype=torch.int32, device='npu') % 7
    b = torch.zeros(grid * n, dtype=torch.int32, device='npu')
    out = torch.zeros(grid, dtype=torch.int32, device='npu')

    a2 = torch.randint(0, 100, (grid,), dtype=torch.int32, device='npu')
    out2 = torch.zeros(grid, dtype=torch.int32, device='npu')

    a3 = torch.randint(0, 1 << 30, (grid,), dtype=torch.int32, device='npu')
    out3 = torch.zeros(grid, dtype=torch.int32, device='npu')

    cases = [
        ('ldst', lambda mode, sf: scalar_ldst_kernel[(grid,)](
            a, b, out, N_LD=n, N_ST=n, **opts_for(mode, sf, grid))),
        ('compute', lambda mode, sf: scalar_compute_kernel[(grid,)](
            a2, out2, N=n, **opts_for(mode, sf, grid))),
        ('control', lambda mode, sf: scalar_control_kernel[(grid,)](
            a3, out3, N=n, **opts_for(mode, sf, grid))),
    ]

    print('case,mode,sf,median_ms')
    for name, launch in cases:
        # SIMD is a single baseline; superblock factor only applies to SIMT.
        try:
            t = measure_launch(lambda: launch('simd', 1), reps=8)
            print(f'{name},simd,1,{t:.6f}', flush=True)
        except Exception as e:
            print(f'{name},simd,1,ERROR:{type(e).__name__}:{e}', flush=True)

        if args.mode in ('all', 'simt_only'):
            for sf in [1, 4, 16, 64]:
                try:
                    t = measure_launch(lambda: launch('simt_only', sf), reps=8)
                    print(f'{name},simt_only,{sf},{t:.6f}', flush=True)
                except Exception as e:
                    print(f'{name},simt_only,{sf},ERROR:{type(e).__name__}:{e}', flush=True)


if __name__ == '__main__':
    main()
