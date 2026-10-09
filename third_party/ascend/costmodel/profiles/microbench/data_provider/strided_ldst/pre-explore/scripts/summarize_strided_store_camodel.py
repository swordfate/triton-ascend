#!/usr/bin/env python3
"""Print a compact table from parsed strided-store CAModel JSON files."""
import argparse
import json
from pathlib import Path


def hist(hist_dict):
    return ",".join(f"{k}:{v}" for k, v in sorted(hist_dict.items(), key=lambda kv: int(kv[0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parsed", nargs="+", type=Path)
    args = ap.parse_args()
    for path in args.parsed:
        data = json.loads(path.read_text())
        print(f"===== {path}")
        print("idx mode      stride block active store pre post "
              "i2b span b2r store_instr biu_cmds biu_sizes dc_reqs dc_sizes")
        for c in data["cases"]:
            biu = c.get("biu", {})
            dc = c.get("dc", {}) or {}
            print(f'{c["index"]:>3} {c["mode"]:<9} {c["stride"]:>6} {c["block"]:>5} '
                  f'{c.get("instr_active_cycles"):>6} {c.get("store_active_cycles"):>5} '
                  f'{c.get("pre_store_cycles"):>4} {c.get("post_store_cycles"):>4} '
                  f'{c.get("store_issue_to_first_biu"):>4} {c.get("biu_write_span"):>4} '
                  f'{c.get("biu_to_retire"):>3} {c.get("store_engine"):>30} '
                  f'{biu.get("send_aw_count", "-"):>6} {hist(biu.get("send_aw_sizes", {})):>12} '
                  f'{dc.get("write_requests", "-"):>6} {hist(dc.get("write_sizes", {})):>12}')
        print()


if __name__ == "__main__":
    main()
