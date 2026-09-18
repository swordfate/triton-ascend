#!/usr/bin/env python3
"""Direct-scalar-load validation for the 6 kernels.

Direct part = the first `K=1, indirect=0` ScalarLoad stages (all 6 kernels have
two of them).  Measured side:
  SIMD: first N_direct scalar LD/LDP windows with window >= 20 cycles and a
        non-parameter address (removes prologue/hit tile loads).
  SIMT: first N_direct SIMT_LDG windows by issue time.
Indirect part = remaining ScalarLoad stages vs the remaining measured loads
(reported separately because branch execution and same-line effects dominate).

For the direct error we only sum predicted stages that have a matched measured
window (`matched`).  A stage inside an unexecuted `scf.if` (e.g.
`binned_copy_gather` `if expert_idx > 0` sampled at expert_idx==0) has no
measured counterpart and must not be added to the compared sum.  The old
static worst-case sum is still emitted as `direct_pred_static` /
`direct_static_err_pct` for reference.
"""
import argparse
import csv
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_scalar_loads import parse_simd, parse_simt  # noqa: E402

KERNELS = [
    "padded_copy_gather", "padded_copy_scatter", "padded_copy_wgrad",
    "binned_copy_gather", "binned_copy_scatter", "binned_copy_wgrad",
]
REPORTS = [("baseline", "out/costmodel_baseline_{k}.json"),
           ("tuned", "out/costmodel_{k}.json")]


def report_stages(path):
    d = json.load(open(path))
    direct, indirect = [], []
    for st in d["stage_model"]["logical_stages"]:
        if st.get("model") != "scalar_load":
            continue
        w = st["workload"]
        entry = None
        for impl in st["implementations"]:
            im = impl["implementation"]
            if im["mode"] == "simd" and im["superblock_factor"] == 1:
                entry = ("simd", impl["resource_system_cycles"].get("load_per_iteration", 0.0))
            elif im["mode"] == "simt" and im["superblock_factor"] == 1 and entry is None:
                entry = ("simt", impl["resource_system_cycles"].get("load_per_iteration", 0.0))
        loads = {}
        for impl in st["implementations"]:
            im = impl["implementation"]
            if im["superblock_factor"] == 1:
                loads[im["mode"]] = impl["resource_system_cycles"].get("load_per_iteration", 0.0)
        item = {
            "id": st["id"],
            "K": int(w.get("scalar_load_count_per_iteration", 0)),
            "indirect": int(w.get("indirect_scalar_load_count_per_iteration", 0)),
            "loads": loads,
        }
        if item["K"] == 1 and item["indirect"] == 0:
            direct.append(item)
        else:
            indirect.append(item)
    return direct, indirect


def find_dump(k, mode):
    log = HERE / "out" / f"{k}_{mode}.run.log"
    if not log.exists():
        return None
    m = re.findall(r"Profiling results saved in (\S+)", log.read_text(errors="replace"))
    if not m:
        return None
    d = pathlib.Path(m[-1]) / "dump"
    return d if d.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="out/direct_scalar_eval.csv")
    args = ap.parse_args()
    rows = []
    for tag, pat in REPORTS:
        for k in KERNELS:
            rp = HERE / pat.format(k=k)
            if not rp.exists():
                continue
            direct, indirect = report_stages(rp)
            for mode in ["simd", "simt_only"]:
                dump = find_dump(k, mode)
                if dump is None:
                    continue
                impl_mode = "simd" if mode == "simd" else "simt"
                mr = parse_simd(dump) if impl_mode == "simd" else parse_simt(dump)
                if impl_mode == "simd":
                    mr = [r for r in mr if r["window"] >= 20]
                nd = len(direct)
                m_direct = mr[:nd]
                # matched per-load direct errors (pair in program order)
                matched = []
                for i in range(min(len(direct), len(m_direct))):
                    pv = direct[i]["loads"].get(impl_mode, 0.0)
                    mv = m_direct[i]["window"]
                    if mv:
                        matched.append((direct[i]["id"], pv, mv, (pv - mv) / mv * 100.0))
                pred_direct_static = sum(x["loads"].get(impl_mode, 0.0) for x in direct)
                # Only sum the predicted stages that actually have a matched
                # measured instruction.  A conditional stage may not execute
                # in the sampled program (e.g. `if expert_idx > 0`), and the
                # static worst-case sum must not be compared against a
                # single-program measured window.
                pred_direct_matched = sum(pv for _, pv, _, _ in matched)
                meas_direct_matched = sum(mv for _, _, mv, _ in matched)
                pred_ind = sum(x["loads"].get(impl_mode, 0.0) for x in indirect)
                k_ind = sum(x["K"] for x in indirect)
                m_ind = mr[nd:nd + k_ind]
                meas_ind = sum(x["window"] for x in m_ind)
                unmatched_pred_cycles = pred_direct_static - pred_direct_matched
                coverage_pct = (100.0 * len(matched) / nd) if nd else 0.0
                rows.append({
                    "profile": tag, "kernel": k, "mode": mode,
                    "direct_pred": round(pred_direct_matched, 2),
                    "direct_pred_static": round(pred_direct_static, 2),
                    "direct_meas": meas_direct_matched,
                    "direct_err_pct": round((pred_direct_matched - meas_direct_matched) / meas_direct_matched * 100.0, 1) if meas_direct_matched else "",
                    "direct_static_err_pct": round((pred_direct_static - meas_direct_matched) / meas_direct_matched * 100.0, 1) if meas_direct_matched else "",
                    "matched_mape_pct": round(sum(abs(e) for _, _, _, e in matched) / len(matched), 1) if matched else "",
                    "matched_bias_pct": round(sum(e for _, _, _, e in matched) / len(matched), 1) if matched else "",
                    "matched_n": f"{len(matched)}/{nd}",
                    "matched_detail": ";".join(f"{sid}:{pv:.0f}vs{mv}:{e:+.1f}%" for sid, pv, mv, e in matched),
                    "unmatched_pred_cycles": round(unmatched_pred_cycles, 2),
                    "coverage_pct": round(coverage_pct, 1),
                    "indirect_pred": round(pred_ind, 2),
                    "indirect_meas": meas_ind,
                    "indirect_err_pct": round((pred_ind - meas_ind) / meas_ind * 100.0, 1) if meas_ind else "",
                    "n_indirect": f"{len(m_ind)}/{k_ind}",
                })
    out = HERE / args.csv
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)
    # matched MAPE must be computed over all matched load pairs, not by
    # averaging per-case signed biases.
    import re as _re
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        for part in r["matched_detail"].split(";"):
            if not part:
                continue
            m = _re.search(r":([+-]?[0-9.]+)%$", part)
            if m:
                agg[(r["profile"], r["mode"])].append(float(m.group(1)))
    for tag in ["baseline", "tuned"]:
        for mode in ["simd", "simt_only"]:
            vals = agg.get((tag, mode), [])
            if vals:
                mape = sum(abs(v) for v in vals) / len(vals)
                bias = sum(vals) / len(vals)
                print(f"{tag:8s} matched-direct {mode:9s} n_loads={len(vals)} MAPE={mape:.1f}% bias={bias:+.1f}%")
            matched_e = [r["direct_err_pct"] for r in rows
                         if r["profile"] == tag and r["mode"] == mode and r["direct_err_pct"] != ""]
            if matched_e:
                mae = sum(abs(v) for v in matched_e) / len(matched_e)
                bias = sum(matched_e) / len(matched_e)
                print(f"{tag:8s} sum-direct matched   {mode:9s} n_cases={len(matched_e)} MAPE={mae:.1f}% bias={bias:+.1f}%")
            static_e = [r["direct_static_err_pct"] for r in rows
                        if r["profile"] == tag and r["mode"] == mode and r["direct_static_err_pct"] != ""]
            if static_e:
                mae = sum(abs(v) for v in static_e) / len(static_e)
                bias = sum(static_e) / len(static_e)
                print(f"{tag:8s} sum-direct static    {mode:9s} n_cases={len(static_e)} MAPE={mae:.1f}% bias={bias:+.1f}%")
    print()
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
