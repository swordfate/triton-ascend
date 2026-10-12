#!/usr/bin/env bash
set -uo pipefail
PRE="$HOME/strided-template-calibration-v2"
cd "$PRE"
for d in runs/simt_camodel/w2 runs/simt_camodel/w16; do
  echo "=== $d ==="
  bash scripts/board_env.sh python scripts/analyze_strided_camodel.py "$d" \
      -o "$d/parsed.json" --output-csv "$d/parsed.csv" 2>&1 | tail -2
  bash scripts/board_env.sh python - "$d" <<'PY'
import json,sys
d=sys.argv[1]
j=json.load(open(d+"/parsed.json"))
print("  %-6s %-8s %-9s %-10s %-9s %-9s %-9s"%("stride","L128","simt_ldg","dc_hit","dc_miss","biu_send","biu_bytes"))
for c in j["cases"]:
    lsu=c.get("simt_lsu_counts") or {}
    ldg=lsu.get("SIMT_LDG") or lsu.get("n_SIMT_LDG") or lsu
    dc=c.get("cache") or {}
    biu=c.get("biu") or {}
    print("  %-6s %-8s %-9s %-10s %-9s %-9s %-9s"%(
        c.get("stride"), c.get("unique_lines_128_aligned"), str(ldg)[:28],
        dc.get("hit"), dc.get("miss"), biu.get("send_count"), biu.get("send_bytes")))
PY
done
