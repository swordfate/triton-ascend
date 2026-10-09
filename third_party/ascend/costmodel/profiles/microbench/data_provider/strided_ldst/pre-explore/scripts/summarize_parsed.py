#!/usr/bin/env python3
"""Print a compact text table from one or more parsed strided-CAModel JSONs."""
import argparse
import json
from pathlib import Path


def fmt(value, width=5):
    if value is None:
        return "-".rjust(width)
    return str(value).rjust(width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parsed", nargs="+", type=Path)
    args = ap.parse_args()
    for path in args.parsed:
        data = json.loads(path.read_text())
        print(f"===== {path}")
        print(
            "idx mode      stride blk active task load nload pre post "
            "i->biu fill biu->ret | biu_cmds sizes recv | dc_reqs sizes"
        )
        for c in data["cases"]:
            biu = c.get("biu", {})
            dc = c.get("cache", {}) or {}
            sizes = ",".join(f"{k}:{v}" for k, v in sorted(biu.get("send_sizes", {}).items(),
                                                          key=lambda kv: int(kv[0])))
            dc_sizes = ",".join(f"{k}:{v}" for k, v in sorted(dc.get("read_sizes", {}).items(),
                                                              key=lambda kv: int(kv[0])))
            print(
                f'{c["index"]:>3} {c["mode"]:<9} {c["stride"]:>6} {c["block"]:>3} '
                f'{fmt(c.get("instr_active_cycles"))} {fmt(c.get("task_cycles"))} '
                f'{fmt(c.get("load_active_cycles"))} {fmt(c.get("load_instr_count"))} '
                f'{fmt(c.get("pre_load_cycles"))} {fmt(c.get("post_load_cycles"))} '
                f'{fmt(c.get("load_issue_to_first_biu"))} {fmt(c.get("biu_fill_span"))} '
                f'{fmt(c.get("last_biu_to_retire"))} | '
                f'{fmt(biu.get("send_count"),4)} {sizes:<14} {fmt(biu.get("recv_count"),4)} | '
                f'{fmt(dc.get("read_requests"),4)} {dc_sizes}'
            )
        print()


if __name__ == "__main__":
    main()
