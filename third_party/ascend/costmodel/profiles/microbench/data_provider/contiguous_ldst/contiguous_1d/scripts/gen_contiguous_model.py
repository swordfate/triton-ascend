#!/usr/bin/env python3
"""Emit the simd.memory.contiguous_ldst profile block from the fitted JSONs.

The cost model reads its coefficients from the stage profile, so this script
produces JSON only: no generated C++ is kept in the tree.

    python3 scripts/gen_contiguous_model.py            # print the block
    python3 scripts/gen_contiguous_model.py --write     # patch the profile
"""
import argparse
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(BASE)))))
PROFILE = os.path.join(ROOT, "profiles", "simd_simt",
                       "ascend_950pr_simd_simt_v1.json")
EVIDENCE = ("third_party/ascend/costmodel/profiles/microbench/data_provider/"
            "contiguous_ldst/contiguous_1d")
MAX_ELEMENTS = 16384
MAX_WARPS = 64

UNMASKED_LOAD = [("unmasked.load.lines_le_2_i8", "simt.load|lines<=2,i8"),
          ("unmasked.load.lines_le_2_eb2", "simt.load|lines<=2,eb2"),
          ("unmasked.load.lines_le_2_eb4", "simt.load|lines<=2,eb4"),
          ("unmasked.load.lines_le_2_eb8", "simt.load|lines<=2,eb8"),
          ("unmasked.load.lines_2_32_i8", "simt.load|2<lines<=32,i8"),
          ("unmasked.load.lines_2_32_eb2", "simt.load|2<lines<=32,eb2"),
          ("unmasked.load.lines_2_32_eb4", "simt.load|2<lines<=32,eb4"),
          ("unmasked.load.lines_2_32_eb8", "simt.load|2<lines<=32,eb8"),
          ]
UNMASKED_STORE = [
           ("unmasked.store.te_ge_1_int8", "simt.store|te>=1,int8"),
           ("unmasked.store.te_ge_1_wide_w_le_4", "simt.store|te>=1,wide,W<=4"),
           ("unmasked.store.te_ge_1_wide_w_gt_4_eb2", "simt.store|te>=1,wide,W>4,eb2"),
           ("unmasked.store.te_ge_1_wide_w_gt_4_eb4", "simt.store|te>=1,wide,W>4,eb4"),
           ("unmasked.store.te_ge_1_wide_w_gt_4_eb8", "simt.store|te>=1,wide,W>4,eb8")]
MASKED_LOAD = [("masked.load.lines_le_2", "load|lines<=2"),
          ("masked.load.lines_2_32", "load|2<l<=32"),
          ("masked.load.l_gt_32_nospill", "load|l>32_nospill"),
          ("masked.load.l_gt_32_spill", "load|l>32_spill")]
MASKED_STORE = [("masked.store.te_lt_1", "store|te<1"),
           ("masked.store.narrow_l_le_8", "store|narrow_l<=8"),
           ("masked.store.narrow_8_l_32", "store|narrow_8<l<=32"),
           ("masked.store.narrow_l_gt_32", "store|narrow_l>32"),
           ("masked.store.eb4_w_le_4", "store|eb4_W<=4"),
           ("masked.store.eb4_w_gt_4", "store|eb4_W>4")]


def load(name):
    with open(os.path.join(BASE, "fitted", name)) as handle:
        return json.load(handle)


def form_a(formula):
    """Form A keeps the intercept outside the feature list."""
    return ["one"] + list(formula["feats"]), [float(formula["coefs"][0])] + \
        [float(c) for c in formula["coefs"][1:]]


def form_b(formula):
    """Form B carries "one" inside the feature list."""
    return list(formula["feats"]), [float(c) for c in formula["coefs"]]


def build():
    spec, formb, simd = (load("FINAL_SPEC.json"), load("FORMB_FINAL.json"),
                         load("SIMD_CONTIGUOUS.json"))
    formulas = {
        "simd.load": {"features": simd["load"]["features"],
                      "coefficients": simd["load"]["coefficients"]},
        "simd.store": {"features": simd["store"]["features"],
                       "coefficients": simd["store"]["coefficients"]},
    }
    for name, key in UNMASKED_LOAD + UNMASKED_STORE:
        features, coefficients = form_a(spec["segments"][key])
        formulas[name] = {"features": features,
                          "coefficients": coefficients}
    for name, key in MASKED_LOAD + MASKED_STORE:
        features, coefficients = form_b(formb["segments"][key])
        formulas[name] = {"features": features,
                          "coefficients": coefficients}
    # large blocks (>64 KiB) are out of scope for now: the model routes them
    # through the SIMD path, whose loads do not compile at that size.
    smallPath = os.path.join(BASE, "fitted", "SMALL_SPEC.json")
    if os.path.exists(smallPath):
        with open(smallPath) as handle:
            for name, entry in json.load(handle).items():
                formulas[name] = entry
    tables = {}
    tablesPath = os.path.join(BASE, "fitted", "TABLES.json")
    if os.path.exists(tablesPath):
        with open(tablesPath) as handle:
            tables = json.load(handle)
    return {"evidence": EVIDENCE,
            "max_elements": MAX_ELEMENTS,
            "max_warps": MAX_WARPS,
            "formulas": formulas,
            "tables": tables}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="patch the stage profile in place")
    parser.add_argument("--out", default=PROFILE)
    args = parser.parse_args()
    block = build()
    if not args.write:
        print(json.dumps(block, indent=2, sort_keys=True))
        return
    with open(args.out) as handle:
        profile = json.load(handle)
    # Split by route: the SIMD curves price the vector path, the unmasked /
    # masked ones price the SIMT segments.
    simdFormulas = {k: v for k, v in block["formulas"].items()
                    if k.startswith(("simd.", "bw.simd."))}
    simtFormulas = {k: v for k, v in block["formulas"].items()
                    if not k.startswith("simd.")}
    for side, formulas in (("simd", simdFormulas), ("simt", simtFormulas)):
        if not formulas:
            continue
        entry = {
            "evidence": block["evidence"],
            "max_elements": block["max_elements"],
            "max_warps": block["max_warps"],
            "formulas": formulas,
        }
        if side == "simt" and block.get("tables"):
            entry["tables"] = block["tables"]
        profile[side].setdefault("memory", {})["contiguous_ldst"] = entry
    with open(args.out, "w") as handle:
        json.dump(profile, handle, indent=2)
        handle.write("\n")
    print("patched %s with %d formulas" % (args.out, len(block["formulas"])))


if __name__ == "__main__":
    main()
