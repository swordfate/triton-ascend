import math

def min_(a, b): return lambda r: min(a(r), b)
POOL = {
    "1":          lambda r: 1.0,
    "L":          lambda r: r["L"],                       # 128B lines touched
    "Lf":         lambda r: r["Lf"],
    "pages":      lambda r: r["pages"],
    "n":          lambda r: r["n"],                       # elements
    "span":       lambda r: r["span"] / 128.0,
    "eb":         lambda r: r["eb"],
    "ebL":        lambda r: r["eb"] * r["L"],             # bytes actually moved
    "ebn":        lambda r: r["eb"] * r["n"],
    "nW":         lambda r: r["n"] / r["W"],
    "iters":      lambda r: r["iters"],
    "perlane":    lambda r: r["per_lane"],
    "minL32":     lambda r: min(r["L"], 32.0),
    "maxL32":     lambda r: max(0.0, r["L"] - 32.0),
    "minL128":    lambda r: min(r["L"], 128.0),
    "W":          lambda r: r["W"],
    "lowW":       lambda r: max(0.0, 4.0 - r["W"]) * r["L"],
    "rowgap":     lambda r: r["rowgap"] / 128.0,
    "cmd":        lambda r: r["cmd"],
    "lpe":        lambda r: r["lines_per_elem"],
    "itersL":     lambda r: r["iters"] * min(r["L"], 32.0),
    "Wn":         lambda r: r["W"] * r["n"] / 32.0,
    "hL32":      lambda r, k=32: max(0.0, r["L"] - k),
    "hL128":      lambda r, k=128: max(0.0, r["L"] - k),
    "hL512":      lambda r, k=512: max(0.0, r["L"] - k),
    "hL2048":      lambda r, k=2048: max(0.0, r["L"] - k),
    "hL8192":      lambda r, k=8192: max(0.0, r["L"] - k),
    "hP1":      lambda r, k=1: max(0.0, r["pages"] - k),
    "hP4":      lambda r, k=4: max(0.0, r["pages"] - k),
    "hP16":      lambda r, k=16: max(0.0, r["pages"] - k),
    "hP64":      lambda r, k=64: max(0.0, r["pages"] - k),
    "pagecross": lambda r: float(r["f_page_cross"]),
    "cross":     lambda r: float(r["f_cross"]),
    "spanMB":    lambda r: r["span"] / (1024.0*1024.0),
    "hspan128k": lambda r: max(0.0, r["span"] - 131072.0)/131072.0,
    # --- rank-2 comb structure -------------------------------------------
    "rows":       lambda r: float(r.get("f_rows") or 0.0),
    "cols":       lambda r: float(r.get("f_cols") or 0.0),
    "rowspan":    lambda r: float(r.get("f_row_span_bytes") or 0.0)/128.0,
    "rowlines":   lambda r: float(r.get("f_row_lines") or 0.0),
    "gaplines":   lambda r: float(r.get("f_gap_lines") or 0.0),
    "comblines":  lambda r: float(r.get("f_comb_lines") or 0.0),
    "combover":   lambda r: float(r.get("f_comb_overlap") or 0.0),
    "rowsshare":  lambda r: float(r.get("f_rows_share_line") or 0.0),
    "rowseg":     lambda r: float(r.get("f_row_segments") or 0.0),
    "gapline":    lambda r: float(r.get("f_gap_per_line") or 0.0),
    "rsdivg":     lambda r: float(r.get("f_row_span_div_gap") or 0.0),
    "rowsegL":    lambda r: float(r.get("f_row_segments") or 0.0) * r["L"],
    "rowsshareL": lambda r: float(r.get("f_rows_share_line") or 0.0) * r["L"],
    "gaplineL":   lambda r: float(r.get("f_gap_per_line") or 0.0) * r["L"],
    # --- read-modify-write / warp line footprint (white box) ---
    "epl":        lambda r: max(1.0, math.floor(128.0 / max(1e-9, r["ISB"]))),   # elements per line
    "fullline":   lambda r: r["n"] / max(1.0, math.floor(128.0 / max(1e-9, r["ISB"]))),
    "rmw":        lambda r: max(0.0, r["L"] - r["n"] * r["eb"] / 128.0),         # partially written lines
    "warpL":      lambda r: min(32.0, max(1.0, math.ceil(32.0 * r["ISB"] / 128.0))),
    "itersWarpL": lambda r: r["iters"] * min(32.0, max(1.0, math.ceil(32.0 * r["ISB"] / 128.0))),
    "rmwEb":      lambda r: max(0.0, r["L"] - r["n"] * r["eb"] / 128.0) * r["eb"],
    "LmRmw":      lambda r: r["L"] - max(0.0, r["L"] - r["n"] * r["eb"] / 128.0),
    "nOverEpl":   lambda r: r["n"] / max(1.0, math.floor(128.0 / max(1e-9, r["ISB"]))) * r["eb"],
    "spanOverL":  lambda r: r["span"] / max(1.0, r["L"]),
    # --- physics of partial lines / wasted fetch ---
    "density":    lambda r: (r["n"] * r["eb"]) / max(1.0, r["span"]),      # useful byte fraction
    "waste":      lambda r: max(0.0, r["L"] * 128.0 - r["n"] * r["eb"]) / 128.0,
    "nbytes":     lambda r: r["n"] * r["eb"] / 128.0,                      # packed line count
    "perline":    lambda r: r["n"] / max(1.0, r["L"]),                     # elements per line
    "expline":    lambda r: 128.0 / r["eb"],                               # elements that fit a line
    "strideL":    lambda r: r["ISB"] / 128.0,                              # lines per stride step
    "cmdB":       lambda r: r["L"] / max(1.0, math.floor(128.0 / r["eb"])),
    "linv":       lambda r: r["L"] * (1.0 - (r["n"] * r["eb"]) / max(1.0, r["span"])),
    "LdivEb":     lambda r: r["L"] / r["eb"],
    "LmEb":       lambda r: r["L"] * r["eb"],
    "nWl":        lambda r: (r["n"] / r["W"]) * r["L"],
    "itersW":     lambda r: r["iters"] * r["W"],
    "Wl":         lambda r: r["W"] * r["L"],
    "minL32n":    lambda r: min(r["L"], 32.0) * r["eb"],
}
