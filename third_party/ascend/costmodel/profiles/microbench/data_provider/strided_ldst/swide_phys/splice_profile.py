# -*- coding: utf-8 -*-
"""把 strided_wide_completion 段写进 profile。

两种模式：
  默认（插入）：profile 里没有该键时，在 strided_short_completion 之后插入；
  SWIDE_REPLACE=1（替换）：profile 里已有该键时，用 gen 目录的合并版（_v2）替换对象内容。

用法：
  python splice_profile.py <profile.json> [gen目录]
"""
import json, io, os, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding='utf-8')

NL = chr(10)


def object_end(text, start):
    """start 指向对象开 '{'；用标准库词法返回收 '}' 偏移（字符串安全）。"""
    _, end = json.JSONDecoder().raw_decode(text[start:])
    return start + end - 1


def inner_lines(obj):
    """dump 对象并去掉最外层大括号行，行首缩进 4 空格（对齐宿主文件 1-space 步进）。"""
    text = json.dumps(obj, indent=1, ensure_ascii=False)
    ls = text.split(NL)
    assert ls[0] == '{' and ls[-1] == '}'
    return ['    ' + ln for ln in ls[1:-1]]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    path = args[0]
    gen_dir = args[1] if len(args) > 1 else os.path.join(HERE, 'gen')
    text = io.open(path, encoding='utf-8').read()
    lines = text.split(NL)
    replace_mode = os.environ.get('SWIDE_REPLACE') == '1'

    if replace_mode:
        v2 = json.load(io.open(os.path.join(gen_dir, 'profile_strided_wide_completion_v2.json'),
                               encoding='utf-8'), object_pairs_hook=collections.OrderedDict)
        sites = []
        pos = 0
        while True:
            try:
                koff = text.index('"strided_wide_completion"', pos)
            except ValueError:
                break
            ooff = text.index('{', koff)
            eoff = object_end(text, ooff)
            sites.append((koff, eoff))
            pos = eoff + 1
        assert len(sites) == 2, sites
        # 反向替换：第二个（simt）先改，避免行号漂移
        for (koff, eoff), mode in zip(sorted(sites, reverse=True), ('simt', 'simd')):
            kline = text.count(NL, 0, koff)
            eline = text.count(NL, 0, eoff)
            lines[kline + 1:eline] = inner_lines(v2[mode])
        io.open(path, 'w', encoding='utf-8', newline=NL).write(NL.join(lines))
        print('replaced:', [(mode, text.count(NL, 0, k) + 1)
                            for (k, _), mode in zip(sorted(sites), ('simd', 'simt'))])
        return

    if '"strided_wide_completion"' in text:
        print('已存在 strided_wide_completion，跳过')
        return

    gen = json.load(io.open(os.path.join(gen_dir, 'profile_strided_wide_completion.json'),
                            encoding='utf-8'), object_pairs_hook=collections.OrderedDict)
    anchors = [i for i, ln in enumerate(lines) if '"strided_short_completion": {' in ln]
    assert len(anchors) == 2, anchors
    inserts = []
    for a in anchors:
        window = NL.join(lines[a:a + 500])
        mode = 'simt' if 'simt_load_r1' in window else 'simd'
        ooff = text.index('{', sum(len(x) + 1 for x in lines[:a]))
        eoff = object_end(text, ooff)
        inserts.append((mode, text.count(NL, 0, eoff)))
    for mode, end_line in sorted(inserts, key=lambda x: -x[1]):
        close = lines[end_line].rstrip()
        assert close.strip() in ('}', '},'), repr(close)
        if close.strip() == '}':
            lines[end_line] = close + ','
        blk = ['   "strided_wide_completion": {'] + inner_lines(gen[mode]) + ['   }']
        lines[end_line + 1:end_line + 1] = blk
    io.open(path, 'w', encoding='utf-8', newline=NL).write(NL.join(lines))
    print('inserted:', [(m, e + 1) for m, e in sorted(inserts)])


if __name__ == '__main__':
    main()
