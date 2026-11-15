#!/usr/bin/env python3
"""Assemble fitted/FINAL_SPEC.json from the merged SIMD model and the SIMT segments.

Two sources:

  fitted/SIMD_JOINT.json   SIMD fitted on aligned *and* non-power-of-two points together.
                           The non-power-of-two load points are divided by TAIL_FACTOR
                           before fitting, so the stored geometry is always the padded one
                           and the factor is applied again at prediction time.
  fitted/SIMT_SEGMENTS.json  SIMT fitted on the aligned grid only (its non-power-of-two
                           behaviour is not modelled; see PR_SUMMARY 5b.6).

The output is one flat dict: segment name -> {feats, coefs}, plus "tail_factor".
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EB = {"int8": 1, "fp16": 2, "fp32": 4, "int64": 8}
# 每线程寄存器数按线程数分档（AIV SIMT RegFile 128 KB）：≤256→127, ≤512→64, ≤1024→32, ≤2048→16
REGB = {1: 508.0, 2: 508.0, 4: 508.0, 8: 508.0, 16: 256.0, 32: 128.0, 64: 64.0}


def pad(n):
    return 1 << max(0, (n - 1).bit_length())


def kof(x, cap=2):
    return 2.0 ** min(cap, math.floor(math.log2(max(1.0, x))))


def feats(n, eb, w, tail):
    """Full feature set, shared with the fitting scripts.

    tail=True means n is the real element count of a masked (non power of two)
    access; tail=False means n is already the padded block size.
    """
    e = n if tail else pad(n)
    by = e * eb
    lines = by / 128.0
    te = e / (32.0 * w)
    lpW = lines / w
    slow = 1.0 if (n % 32 and n / (32.0 * w) >= 4.0) else 0.0
    slowK = 1.0 if (n % 32 and n / (32.0 * w) >= 8.0 / eb) else 0.0
    part = 1.0 if (n * eb) % 128 else 0.0
    return dict(by=by, blk=by / 32.0, blk_part=1.0 if by % 32 else 0.0,
                pow2=1.0 if (n & (n - 1)) == 0 else 0.0,
                lines=lines, te=te, lpW=lpW, W=w, invW=1.0 / w, W2=w * w,
                rounds=math.ceil(te), K1=by * kof(lpW),
                K2=by * 2.0 ** min(2, math.floor(math.log2(max(1.0, te / 16)))),
                LN=by * math.log(1 + te), lin=1.0 if by >= 128 else 0.0,
                st=1.0 if te < 1 else 0.0, stW=(1.0 if te < 1 else 0.0) * w,
                lnW=math.log(1 + w), WlnW=w * math.log(1 + w),
                byLnW=by * math.log(1 + w), i8=1.0 if eb == 1 else 0.0,
                i8K=(by if eb == 1 else 0.0) * kof(lpW), g=max(0.0, te - 32),
                g2By=by * max(0.0, te - 64), waves_by=math.ceil(te) * by,
                lanes=min(e, 32.0 * w), s32=1.0 if e < 32 else 0.0,
                s32W=(1.0 if e < 32 else 0.0) * w, slow=slow, fast=1.0 - slow,
                slow_by=by * slow, fast_by=by * (1.0 - slow),
                slowK=slowK, fastK=1.0 - slowK,
                slowK_by=by * slowK, fastK_by=by * (1.0 - slowK),
                teK=math.ceil(te / (8.0 / eb)),
                part=part, part_by=by * part, half=1.0 if eb == 2 else 0.0,
                half_fast=((1.0 - slow) if eb == 2 else 0.0),
                half_fast_by=((by * (1.0 - slow)) if eb == 2 else 0.0),
                half_slow_by=((by * slow) if eb == 2 else 0.0),
                fill=(32.0 * eb) / 256.0,
                wpl_by=by * max(1.0, 128.0 / (32.0 * eb)),
                warpB=32.0 * eb, schedR=math.ceil(w / 4.0), invWby=by / w, by2=by * by,
                linW=(1.0 if by >= 128 else 0.0) * w, lines_part=part,
                lines_full=1.0 - part, half_slow=(slow if eb == 2 else 0.0),
                fast_lnW=math.log(1 + w) * (1.0 - slow),
                slow_lnW=math.log(1 + w) * slow, byLnW2=by * math.log(1 + w),
                spill=max(0.0, (n * eb) / (32.0 * w) - REGB[w]) / REGB[w],
                spillL=by / 128.0 * (1.0 if (n * eb) / (32.0 * w) > REGB[w] else 0.0),
                wpl=max(1.0, 128.0 / (32.0 * eb)),
                linwpl=lines / max(1.0, 128.0 / (32.0 * eb)),
                rsqrtW=1.0 / math.sqrt(w),
                Wh=max(0.0, w - 16.0), Wh8=max(0.0, w - 8.0),
                Wh_by=by * max(0.0, w - 16.0), Wh8_by=by * max(0.0, w - 8.0),
                Wlin=(w if w <= 16 else 16.0),
                byh1k=max(0.0, by - 1024.0), r2=math.ceil(te) ** 2,
                burst=math.ceil(by / 16384.0) - 1.0)


def geom(e, dtype, w):
    """Features of an aligned access whose padded block size is e."""
    return feats(e, EB[dtype], w, False)


def segment(mode, direction, g):
    if mode == "simd":
        return mode + "." + direction
    if direction == "load":
        if g["lines"] <= 2:
            if g["i8"] > 0:
                return "simt.load|lines<=2,i8"
            if g["half"] > 0:
                return "simt.load|lines<=2,eb2"
            if g["warpB"] == 128.0:
                return "simt.load|lines<=2,eb4"
            return "simt.load|lines<=2,eb8"
        if g["lines"] <= 32:
            # 同字节数下 1/2 字节与 4/8 字节元素的指令形态不同，实测 max 相差 2 倍。
            if g["i8"] > 0:
                return "simt.load|2<lines<=32,i8"
            if g["half"] > 0:
                return "simt.load|2<lines<=32,eb2"
            if g["warpB"] == 128.0:
                return "simt.load|2<lines<=32,eb4"
            return "simt.load|2<lines<=32,eb8"
        return "simt.load|l>32"
    if g["st"] > 0:
        return "simt.store|te<1"
    if g["i8"] > 0:
        return "simt.store|te>=1,int8"
    if g["W"] <= 4:
        return "simt.store|te>=1,wide,W<=4"
    # 大 warp 的 store 里 fp16 与 4/8 字节元素是两条不同的曲线（实测差 2 倍）。
    if g["half"] > 0:
        return "simt.store|te>=1,wide,W>4,eb2"
    return ("simt.store|te>=1,wide,W>4,eb4" if g["by"] > 0 and g["i8"] == 0 and g["warpB"] == 128.0
            else "simt.store|te>=1,wide,W>4,eb8")


def build():
    simd = json.loads((ROOT / "fitted" / "SIMD_JOINT.json").read_text())
    simt = json.loads((ROOT / "fitted" / "SIMT_SEGMENTS.json").read_text())
    tail = json.loads((ROOT / "fitted" / "SIMT_TAIL_SEGMENTS.json").read_text())
    spec = {k: {"feats": v["feats"], "coefs": v["coefs"]} for k, v in simd["spec"].items()}
    for k, v in simt.items():
        # ③ 的 SIMT load 段统一由 fit_physical.py（STAGED_FIT.json）提供，旧的不用
        if k.startswith("simt") and not k.startswith("simt.load"):
            spec[k] = {"feats": v["feats"], "coefs": v["coefs"]}
    for k, v in tail.items():
        spec[k] = {"feats": v["feats"], "coefs": v["coefs"]}
    # ①②③ 用「只用物理量」的拟合结果（见 fit/fit_staged.py）
    staged = json.loads((ROOT / "fitted" / "STAGED_FIT.json").read_text())
    for k, v in staged.items():
        if v["stage"] == 1 or k in ("simt.load|l>32,W<=4", "simt.load|l>32,W>4", "simt.load|lines<=2", "simt.load|2<lines<=32"):
            spec[k] = {"feats": v["feats"], "coefs": v["coefs"]}
    bt = json.loads((ROOT / "fitted" / "LOAD_BOTTLENECK.json").read_text())
    # SIMD 两条公式按真实字节数（不再按补齐后的块宽），4 系数，覆盖 2 幂与非 2 幂
    simd_new = json.loads((ROOT / "fitted" / "SIMD_CONTIGUOUS.json").read_text())
    for direction in ("load", "store"):
        v = simd_new[direction]
        spec["simd." + direction] = {
            "feats": ["by", "blk_part", "burst"],
            "coefs": [v["a"], v["b"], v["c"], v["d"]],
        }
    return {"tail_factor": simd["tail_factor"], "segments": spec,
            "bottleneck": {k: {kk: v[kk] for kk in ("a", "b", "c", "d", "e1", "e2")}
                           for k, v in bt.items()}}


def tail_segment(direction, g):
    """SIMT 尾部：load 走 bottleneck 形式（见 eval_bottleneck），这里只处理 store。"""
    if g["te"] < 1:
        return "tail|store|te<1"
    base = "narrow" if g["i8"] + g["half"] > 0 else "wide"
    w = g["W"]
    wgd = "W1" if w <= 4 else ("W2_4" if w <= 16 else "Wgt4")
    return "tail|store|te>=1,%s,%s" % (base, wgd)


def evaluate(seg, g):
    v = seg["coefs"][0]
    for name, c in zip(seg["feats"], seg["coefs"][1:]):
        v += c * g[name]
    return v


def eval_bottleneck(seg, g):
    """max( warp 发射 , 循环展开 , 数据通路 , 寄存器溢出 ) —— 每个分支一个硬件量。"""
    w, x, lines, part = g["W"], g["iters"], g["lines"], g["part"]
    wpl = max(1.0, 128.0 / (32.0 * g["eb"]))
    return max(seg["a"] * w,
               seg["b"] + seg["c"] * x,
               seg["d"] * wpl * lines * part,
               seg["e1"] * lines * g["sp1"],
               seg["e2"] * lines * g["sp2"])


def load_bottleneck_key(dtype):
    return "tail|load|int64" if dtype == "int64" else "tail|load|narrow"


def predict(spec, mode, direction, dtype, elements, warps):
    """Predict one access.

    A masked (non power of two) access is predicted by the tail segments, which
    are fitted in terms of the REAL element count.  Everything else uses the
    aligned segments, which are fitted in terms of the PADDED block size.
    """
    if mode == "simd":
        # SIMD 的两条公式是按「真实元素数」标定的（覆盖 2 幂与非 2 幂），不补齐、不乘 tail_factor
        g = feats(elements, EB[dtype], warps, True)
        key = "simd." + direction
        return evaluate(spec["segments"][key], g), key
    masked = pad(elements) != elements
    if masked and pad(elements) * EB[dtype] >= 128:
        g = feats(elements, EB[dtype], warps, True)
        g["eb"] = EB[dtype]
        g["iters"] = math.ceil(elements / (32.0 * warps))
        g["part"] = 1.0 if elements % 32 else 0.0
        _teb = (elements / (32.0 * warps)) * EB[dtype]
        g["sp1"] = 1.0 if _teb > REGB[warps] / 2 else 0.0
        g["sp2"] = 1.0 if _teb > REGB[warps] else 0.0
        if direction == "load":
            key = load_bottleneck_key(dtype)
            return eval_bottleneck(spec["bottleneck"][key], g), key
        key = tail_segment(direction, g)
        return evaluate(spec["segments"][key], g), key
    e = pad(elements)
    g = geom(e, dtype, warps)
    key = segment(mode, direction, g)
    return evaluate(spec["segments"][key], g), key


def main():
    spec = build()
    out = ROOT / "fitted" / "FINAL_SPEC.json"
    out.write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n")
    points = [json.loads(line) for line in (ROOT / "raw" / "raw_pure_unified.jsonl").read_text().splitlines() if line.strip()]
    errs = {}
    for p in points:
        dom = 2048 if p["dir"] == "load" else 8192
        if p["mode"] == "simt_only" and p["el"] > dom:
            continue
        if p["mode"] == "simd" and p["el"] > (8192 if p["dir"] == "load" else 16384):
            continue
        v, _ = predict(spec, p["mode"], p["dir"], p["dtype"], p["el"], p["W"])
        errs.setdefault(p["mode"].replace("_only", "") + "." + p["dir"], []).append(
            abs(v - p["cost"]) / p["cost"] * 100)
    total = []
    for k, v in sorted(errs.items()):
        print("%-12s n=%3d  MAPE %5.2f%%" % (k, len(v), sum(v) / len(v)))
        total += v
    print("aligned total  n=%d  MAPE %.2f%%" % (len(total), sum(total) / len(total)))
    print("wrote", out, "segments:", len(spec["segments"]),
          "coefficients:", sum(len(s["coefs"]) for s in spec["segments"].values()))


if __name__ == "__main__":
    main()
