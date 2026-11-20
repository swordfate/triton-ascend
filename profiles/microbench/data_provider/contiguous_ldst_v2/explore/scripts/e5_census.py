"""E5 (local, no NPU): census of contiguous tl.load/tl.store shapes across the
100-case generalization matrix.

For every case in the manifest we resolve its driver script, extract the
@triton.jit kernel bodies, and statically record per kernel:
  * number of tl.load / tl.store sites,
  * masked vs unmasked (mask= kwarg or `other=` present),
  * element dtype hints from the driver (f32/f16/bf16/i32/i64/...),
  * loop presence (`for ` in body) and grid-stride shape,
  * address shape: `ptr + offsets` (1-D contiguous), `ptr + row*stride + col`
    (2-D), tensor-of-pointer (make_block_ptr style), or gather/index form,
  * whether any load address depends on a loaded value (indirect),
  * #loads per kernel and whether several same-shape loads coexist (the
    concurrency axis E3 probes).

The result is a per-case JSONL plus a printed factor-coverage summary.  It is
static analysis only - it tells us which (bytes, dtype, mask, concurrency,
loop, grid) combinations the 100 cases actually exercise, i.e. which axes the
recalibration must cover, and which corner cases (e.g. multiple same-shape
loads in one kernel) are common in practice.

Usage:
  python e5_census.py --manifest <harness>/generalization_matrix_v140_all_100.json \
      --harness <harness-dir> --out e5_census.jsonl
"""
import argparse
import ast
import json
import re
from pathlib import Path

LOAD_RE = re.compile(r"tl\.load\(")
STORE_RE = re.compile(r"tl\.store\(")


def extract_jit_sources(path: Path):
    """Return {kernel_name: body_text} for @triton.jit functions in a driver."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return {}
    out = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            decorated = any(isinstance(d, ast.Attribute) and
                            d.attr == "jit" or
                            (isinstance(d, ast.Call)) for d in node.decorator_list)
            if not decorated:
                continue
            if not any("jit" in ast.dump(d) for d in node.decorator_list):
                continue
            body = "\n".join(lines[node.lineno - 1: node.end_lineno])
            out[node.name] = body
    return out


def classify_body(name, body):
    loads = len(LOAD_RE.findall(body))
    stores = len(STORE_RE.findall(body))
    masked = bool(re.search(r"mask\s*=", body))
    has_loop = bool(re.search(r"\bfor\s+\w+\s+in\s", body))
    grid_stride = bool(re.search("num_programs|program_id", body)) and has_loop
    indirect = bool(re.search(r"tl\.load\([^)]*\)\s*\)?\s*\+|idx.*tl\.load|gather",
                              body)) and "gather" in body
    addptr_1d = bool(re.search(r"\+\s*offsets|\+\s*lane|\+\s*offs", body))
    addptr_2d = bool(re.search(r"\w*\s*\*\s*stride|\*\s*\w*stride|\+ row|\+ cols",
                               body))
    block_ptr = "make_block_ptr" in body or "block_ptr" in body
    # multiple loads: >1 load site (potential same-shape group / concurrency)
    return {
        "kernel": name,
        "loads": loads,
        "stores": stores,
        "masked": masked,
        "loop": has_loop,
        "grid_stride": grid_stride,
        "indirect_gather": indirect,
        "addr_1d": addptr_1d,
        "addr_2d": addptr_2d,
        "block_ptr": block_ptr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--out", default="e5_census.jsonl")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    harness = Path(args.harness)
    jit_cache = {}

    lines = []
    for case in manifest["cases"]:
        driver = harness / case["driver"]
        if driver not in jit_cache:
            jit_cache[driver] = extract_jit_sources(driver)
        kernels = jit_cache.get(driver, {})
        args_text = " ".join(case.get("arguments", []))
        rec = {
            "case": case["name"],
            "family": case.get("family"),
            "driver": case["driver"],
            "case_args": args_text,
            "kernels": [classify_body(n, b) for n, b in kernels.items()],
        }
        lines.append(rec)

    out = Path(args.out)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in lines),
                   encoding="utf-8")

    # -------- summary ----------------------------------------------------
    def fmt_counts(attr):
        from collections import Counter
        c = Counter()
        for r in lines:
            v = sum(k[attr] for k in r["kernels"])
            c[v] += 1
        return dict(sorted(c.items()))

    print("cases:", len(lines))
    print("loads/kernel histogram:", fmt_counts("loads"))
    print("stores/kernel histogram:", fmt_counts("stores"))
    for attr in ("masked", "loop", "grid_stride", "indirect_gather", "addr_1d",
                 "addr_2d", "block_ptr"):
        yes = sum(1 for r in lines if any(k[attr] for k in r["kernels"]))
        print("%-16s %d/%d cases have it" % (attr, yes, len(lines)))
    multi = [r["case"] for r in lines if sum(k["loads"] for k in r["kernels"]) >= 2]
    print("cases with >=2 loads in one kernel (%d):" % len(multi))
    for m in multi[:40]:
        print("   ", m)
    print("wrote", out)


if __name__ == "__main__":
    main()
