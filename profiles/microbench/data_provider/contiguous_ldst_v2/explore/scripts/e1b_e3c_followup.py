"""E1b + E3c: follow-ups answering the reviewer's questions.

E1b - decompose the K=1 loop overhead (Q1).
  Single-shot measured 149 ns (warm); the same load inside a runtime-trip-count
  `for step in range(iters)` with iters=1 measured 244 ns.  What is the +95 ns?
  Variants (all warm, 4 KiB fp32, grid=1):
    a  loop-free load+store                 (reference)
    b  runtime for, K=1, acc chain          (= the 244 ns point)
    c  runtime for, K=1, NO acc             (body: x=load; store  -  isolates
                                             the acc add + acc init)
    d  tl.static_range(1) with acc          (constexpr trip count - the
                                             compiler folds the loop; isolates
                                             runtime loop control)
  Deltas: b-a = whole scaffolding; b-c = acc add/init; b-d = runtime loop
  control (cmp+branch).  TTIR of a/b/d is dumped and op-counted so the
  structural difference is on record.

E3c - truly independent loads (Q2a) + SIMT retest with noise guards (Q2b).
  E3's "independent" variant accumulated every load into one acc, which
  serializes the ADDS.  Here each load lands in its OWN variable and its own
  store, generated per U:
      x_j = tl.load(src + j*BLOCK + lane)
      tl.store(dst + j*BLOCK + lane, x_j + 1)
  The control kernel performs the SAME stores with tl.full constants instead of
  loads, so diff(U) prices exactly the U loads (stores cancel by construction).
  Each point runs 3 independent profiler sessions; we report min-of-medians.
  Chain variant (single multiplied consumption chain) is kept for comparison.

Usage:
  bash run_exp.sh e1b_e3c_followup.py <card> e1b_e3c_c<card>.jsonl \
      --modes simd simt_only
"""
import argparse
from pathlib import Path

import torch
import triton
import triton.language as tl

from common import (JsonlOut, add_common_args, gate_or_die, health_check,
                    pair_diff_ns, prof_session, stats_us, std_opts)

BLOCK = 1024


# ---------------- E1b kernels ----------------------------------------------
@triton.jit
def e1b_once(src, dst, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    x = tl.load(src + lane)
    tl.store(dst + lane, x)


@triton.jit(do_not_specialize=["iters", "winmask"])
def e1b_loop_acc(src, dst, iters, winmask, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), src.dtype.element_ty)
    for step in range(iters):
        acc += tl.load(src + (step & winmask) * BLOCK + lane)
    tl.store(dst + lane, acc)


@triton.jit(do_not_specialize=["iters", "winmask"])
def e1b_loop_noacc(src, dst, iters, winmask, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    for step in range(iters):
        x = tl.load(src + (step & winmask) * BLOCK + lane)
        tl.store(dst + lane, x)


@triton.jit
def e1b_static_acc(src, dst, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), src.dtype.element_ty)
    for step in tl.static_range(1):
        acc += tl.load(src + (step & 0) * BLOCK + lane)
    tl.store(dst + lane, acc)


@triton.jit
def e1b_ctl(dst, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    tl.store(dst + lane, tl.full((BLOCK,), 3, dst.dtype.element_ty))


# ---------------- E3c kernels (generated per U) ----------------------------
E3C_TEMPLATE = """
@triton.jit
def e3c_ind_u{U}(src, dst, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
{loads}
{stores}
"""

E3C_CTL_TEMPLATE = """
@triton.jit
def e3c_ctl_u{U}(dst, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
{stores}
"""

_GEN_HEADER = ("import triton\nimport triton.language as tl\n\n")


def gen_e3c_module(path):
    """Write all U variants to a real .py file - triton's @jit requires
    functions to be defined in an importable Python file (inspect reads the
    source), so exec-generated functions do not work."""
    lines = [_GEN_HEADER, "BLOCK = 1024\n"]
    for u in (1, 2, 3, 4, 6, 8):
        loads = "\n".join(
            "    x%d = tl.load(src + %d * BLOCK + lane)" % (j, j)
            for j in range(u))
        stores = "\n".join(
            "    tl.store(dst + %d * BLOCK + lane, x%d + 1)" % (j, j)
            for j in range(u))
        ctl_stores = "\n".join(
            "    tl.store(dst + %d * BLOCK + lane, "
            "tl.full((BLOCK,), 1, dst.dtype.element_ty))" % j
            for j in range(u))
        lines.append(E3C_TEMPLATE.format(U=u, loads=loads, stores=stores))
        lines.append(E3C_CTL_TEMPLATE.format(U=u, stores=ctl_stores))
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def load_e3c_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e3c", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# chain: multiply-consumption chain (from E3), single final store
@triton.jit
def e3c_chain(src, dst, U: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    c = tl.load(src + lane)
    for j in tl.static_range(1, U):
        c = c * tl.load(src + j * BLOCK + lane)
    tl.store(dst + lane, c)


@triton.jit
def e1b_ctl_tiny(dst):
    tl.store(dst, 0)


def measure(tag, fns, names, pairs, sessions, out, extra):
    """Run `sessions` independent profiler sessions; report each and the
    min-of-medians so a transiently slow session cannot fake a number."""
    per_session = []
    for s in range(sessions):
        rows = prof_session(fns, names, pairs)
        per_session.append(rows)
    rec = {"tag": tag, **extra, "sessions": []}
    best_diff = None
    for rows in per_session:
        d = pair_diff_ns(rows[names[0]], rows[names[1]])
        rec["sessions"].append({
            "probe": stats_us(rows[names[0]]),
            "ctl": stats_us(rows[names[1]]),
            "diff": d,
        })
        if d and d["cv"] <= 0.4:
            cand = d["diff_med_ns"]
            if best_diff is None or cand < best_diff:
                best_diff = cand
    rec["diff_med_ns_min"] = best_diff
    out.write(**rec)
    print(tag, "min_diff=%s" % best_diff, flush=True)
    return best_diff


def main():
    ap = argparse.ArgumentParser()
    ap = add_common_args(ap)
    ap.add_argument("--modes", default="simd simt_only", nargs="+")
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--skip-e1b", action="store_true")
    args = ap.parse_args()

    if args.health and not args.skip_health:
        rec = health_check(args.health)
        gate_or_die(rec, args.health)
        print("health:", rec)

    out = JsonlOut(args.out)
    src = torch.randn(BLOCK * 64, device="npu")
    dst = torch.zeros(BLOCK * 16, device="npu")
    gen_path = str(Path(args.out).parent / "_gen_e3c.py")
    gen_e3c_module(gen_path)
    e3c = load_e3c_module(gen_path)

    for mode in args.modes:
        opts = std_opts(mode, args.warps)
        # ---------------- E1b ------------------------------------------
        variants = [
            ("a_once", lambda: e1b_once[(1,)](src, dst, BLOCK, **opts),
             "e1b_once"),
            ("b_loop_acc_k1",
             lambda: e1b_loop_acc[(1,)](src, dst, 1, 0, BLOCK, **opts),
             "e1b_loop_acc"),
            ("c_loop_noacc_k1",
             lambda: e1b_loop_noacc[(1,)](src, dst, 1, 0, BLOCK, **opts),
             "e1b_loop_noacc"),
            ("d_static_acc",
             lambda: e1b_static_acc[(1,)](src, dst, BLOCK, **opts),
             "e1b_static_acc"),
        ]
        if not args.skip_e1b:
            for tag, fn, kname in variants:
                measure("e1b_" + tag,
                        [fn, lambda: e1b_ctl[(1,)](dst, BLOCK, **opts)],
                        [kname, "e1b_ctl"], args.pairs, args.sessions, out,
                        {"exp": "e1b", "mode": mode})
        # TTIR structure record (once, simd only)
        if mode == "simd" and not args.skip_e1b:
            for kname, handle in (
                    ("once", e1b_once[(1,)](src, dst, BLOCK, **opts)),
                    ("loop_acc", e1b_loop_acc[(1,)](src, dst, 1, 0, BLOCK, **opts)),
                    ("static_acc", e1b_static_acc[(1,)](src, dst, BLOCK, **opts))):
                ttir = handle.asm.get("ttir", "") or ""
                rec = {
                    "exp": "e1b_ttir", "kernel": kname, "mode": mode,
                    "scf_for": ttir.count("scf.for"),
                    "scf_yield": ttir.count("scf.yield"),
                    "addf": ttir.count("arith.addf"),
                    "muli": ttir.count("arith.muli"),
                    "andi": ttir.count("arith.andi"),
                    "addi": ttir.count("arith.addi"),
                    "tt_load": ttir.count("tt.load"),
                    "tt_store": ttir.count("tt.store"),
                }
                out.write(**rec)
                print("ttir", kname, rec, flush=True)

        # ---------------- E3c ------------------------------------------
        for U in (1, 2, 3, 4, 6, 8):
            ind = getattr(e3c, "e3c_ind_u%d" % U)
            ctl = getattr(e3c, "e3c_ctl_u%d" % U)
            measure("e3c_ind_u%d" % U,
                    [lambda U=U, i=ind, c=ctl: i[(1,)](src, dst, BLOCK, **opts),
                     lambda U=U, c=ctl: c[(1,)](dst, BLOCK, **opts)],
                    ["e3c_ind_u%d" % U, "e3c_ctl_u%d" % U],
                    args.pairs, args.sessions, out,
                    {"exp": "e3c", "mode": mode, "kind": "load", "U": U,
                     "grid": 1, "chain": False})
            if U >= 2:
                measure("e3c_chain_u%d" % U,
                        [lambda U=U: e3c_chain[(1,)](src, dst, U, BLOCK, **opts),
                         lambda: e1b_ctl[(1,)](dst, BLOCK, **opts)],
                        ["e3c_chain", "e1b_ctl"],
                        args.pairs, args.sessions, out,
                        {"exp": "e3c", "mode": mode, "kind": "load", "U": U,
                         "grid": 1, "chain": True})
    print("E1B+E3C DONE", flush=True)


if __name__ == "__main__":
    main()
