# -*- coding: utf-8 -*-
"""从 phys_cap18.json 生成 C++/profile 集成产物，并做 C++ 特征口径的逐点 parity 验证。

产物（写入本目录 gen/）:
  profile_strided_wide_completion.json  两个模式的 strided_wide_completion 对象（待合并进 profile）
  slot_terms.json                       slot -> 有序特征词表（profile 数组顺序即此顺序）
  golden_cases.json                     UT 黄金用例（几何 -> 期望 ns）
  swide_switch.txt                      StridedWideCostModel.cpp 的 switch 代码体（供粘贴）

验证:
  1) 本脚本的 cpp 式特征计算 == feats_v2/enrich 的 Python 特征（全数据逐点、逐特征）
  2) cpp 式特征 · profile 系数 == refit 模型预测（全数据逐点）
运行: python gen_integration.py --check   （需能访问 D:\\work_wj\\mask_vs_nomask）
"""
import sys, os, json, math, io, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8')

CAP = r'D:\work_wj\63-workspace\63-workspace\strided-calib\03-featurize'
DATA = os.environ.get('SWIDE_DATA', r'D:\work_wj\mask_vs_nomask')
WINDOWS = [32, 64, 128, 256, 512, 1024, 2048, 4096]

SLOTS = {
    'load|simd':        [('rows<=4',  'simd_load_swide_rows_le4'),
                         ('rows>4',   'simd_load_swide_rows_gt4')],
    'store|simd':       [('rows<=2',  'simd_store_swide_rows_le2'),
                         ('rows>2',   'simd_store_swide_rows_gt2')],
    'load|simt_only':   [('rows=2',      'simt_load_swide_rows2'),
                         ('rows=8',      'simt_load_swide_rows8'),
                         ('rows=64,W<=2', 'simt_load_swide_rows64_wle2'),
                         ('rows=64,3-8',  'simt_load_swide_rows64_w3_8'),
                         ('rows=64,W>8',  'simt_load_swide_rows64_wgt8')],
    'store|simt_only':  [('rows=2',      'simt_store_swide_rows2'),
                         ('rows=8',      'simt_store_swide_rows8'),
                         ('rows=64,W<=2', 'simt_store_swide_rows64_wle2'),
                         ('rows=64,3-8',  'simt_store_swide_rows64_w3_8'),
                         ('rows=64,W>8',  'simt_store_swide_rows64_wgt8')],
}

# ------------------------------------------------- C++ 式特征（独立实现）
def cpp_block_stats(rows, cols, pitch_bytes, inner_bytes, eb):
    """复刻 StageCostModels.cpp describeStridedShortAddresses 的块统计。
    地址 = row*pitch_bytes + col*inner_bytes (字节)；块 = addr // window。"""
    n = rows * cols
    out = {}
    for w in WINDOWS:
        counts = {}
        for flat in range(n):
            row, col = divmod(flat, cols)
            addr = row * pitch_bytes + col * inner_bytes
            b = addr // w
            counts[b] = counts.get(b, 0) + 1
        out[w] = (max(counts.values()),               # mx
                  sum(v * v for v in counts.values()),  # S
                  len(counts))                          # L
    return out

def cpp_features(r, warps):
    """从原始 case 字段（R,C,IS,OS,eb,run_bytes,gap_bytes,rows,W）按 C++ 将有的算法造特征。
    inner: IS=1(元素)。pitch = OS*eb 字节 = run+gap。"""
    rows = int(r['rows']); cols = int(r['C']); eb = int(r['element_bytes'])
    run = int(r['run_bytes']); gap = int(r['gap_bytes'])
    assert cols * eb == run, (cols, eb, run)
    pitch = run + gap
    n = rows * cols
    nbytes = n * eb
    W = float(max(1, warps))
    span = (rows - 1) * pitch + run
    duty = run / float(pitch)
    f = {}
    f['n'] = float(n); f['nbytes'] = float(nbytes); f['W'] = W; f['invW'] = 1.0 / W
    f['rows'] = float(rows); f['run'] = float(run); f['gap'] = float(gap)
    f['span'] = float(span); f['duty'] = duty; f['duty2'] = duty * duty
    f['mis32'] = 1.0 if pitch % 32 else 0.0
    it32 = 32.0 * W
    f['iters'] = float(math.ceil(n / it32)); f['q'] = n / it32
    h = max(float(n) - it32, 0.0)
    f['h'] = h; f['h_div_w'] = h / W; f['h_pos'] = 1.0 if h > 0 else 0.0
    f['elem_per_lane'] = n / it32
    f['bytes_per_lane'] = float(nbytes) / it32
    f['elem_per_warp'] = n / W; f['bytes_per_warp'] = float(nbytes) / W
    lps = math.ceil(run / 128.0)
    f['lines_per_seg'] = float(lps)
    tl = float(rows) * lps
    f['total_lines'] = tl; f['elem_per_line'] = n / tl
    f['rows_x_duty'] = float(rows) * duty; f['rows2'] = float(rows) * rows
    f['inv_rows'] = 1.0 / rows; f['inv_n'] = 1.0 / n
    f['sqrt_n'] = math.sqrt(float(n)); f['sqrt_rows'] = math.sqrt(float(rows))
    f['minW8'] = min(W, 8.0); f['minW16'] = min(W, 16.0)
    f['maxW4'] = max(0.0, W - 4.0); f['maxW8'] = max(0.0, W - 8.0); f['maxW16'] = max(0.0, W - 16.0)
    f['n2W'] = float(n) * n / W
    f['cseg'] = float(cols); f['cseg_mod32'] = float(cols % 32)
    f['cseg_x_invW'] = float(cols) / W
    bs = cpp_block_stats(rows, cols, pitch, eb, eb)
    for w in WINDOWS:
        mx, s, L = bs[w]
        f['mx%d' % w] = float(mx); f['S%d' % w] = float(s); f['L%d' % w] = float(L)
    # feats_v2.build 把 L128 覆盖为 ceil(nbytes/128)（粗略 line 计数），拟合用的正是该口径；
    # L256 及以上才是地址集统计（cmn.feats）。
    f['L128'] = float(math.ceil(nbytes / 128.0))
    # masked 槽位额外读取的派生量（与 refit_simt.enrich / feats_v2.build 逐项一致）
    f['W2'] = W * W
    f['minW2'] = min(W, 2.0)
    f['inv_sectors'] = 1.0 / math.ceil(run / 32.0) if run > 0 else 0.0
    f['is32'] = 1.0 if run == 32 else 0.0
    f['log2run'] = math.log2(max(1.0, run))
    f['logW'] = math.log(W)
    f['log_n'] = math.log(n)
    f['iters_x_W'] = f['iters'] * W
    f['nbytes_per_W'] = nbytes / W
    f['run_x_W'] = run * W
    val_span = (rows - 1) * pitch + run
    f['span_div_run'] = val_span / run
    f['stride_mod32'] = float(pitch % 32)
    f['cseg_div32'] = math.floor(cols / 32.0)
    f['cseg_r32'] = (cols % 32) / 32.0
    f['cseg_x_minW4'] = cols * min(W, 4.0)
    f['r16_x_segW'] = max(rows - 16, 0) * math.ceil(rows / W)
    lps = math.ceil(run / 128.0)
    f['bytes_per_segline'] = run / lps if lps > 0 else 0.0
    f['seg_lanes_r'] = cols / 32.0 - math.floor(cols / 32.0)
    f['conc1k'] = f['L1024']
    return f

# ------------------------------------------------- 段选择（C++ selectSwideFormula 同规则）
def cpp_slot(r, warps, op):
    rows = int(r['rows']); eb = int(r['element_bytes'])
    run = int(r['run_bytes']); gap = int(r['gap_bytes'])
    pitch = run + gap
    if eb not in (1, 2, 4): return None
    if r['order'] != 'inner' or run < 32 or run > 2048: return None
    if gap < 2 or gap > 8192: return None
    if (rows - 1) * pitch + run > 512 * 1024: return None
    if r['path'] == 'simd':
        if rows < 2 or rows > 256: return None
        if op == 'load':
            return 'simd_load_swide_rows_le4' if rows <= 4 else 'simd_load_swide_rows_gt4'
        return 'simd_store_swide_rows_le2' if rows <= 2 else 'simd_store_swide_rows_gt2'
    if rows not in (2, 8, 64): return None
    if warps < 1 or warps > 64: return None
    if op == 'load' and run > 128: return None  # load 只标到 128B
    seg = ('rows=2' if rows == 2 else 'rows=8' if rows == 8 else
           ('rows=64,W<=2' if warps <= 2 else 'rows=64,3-8' if warps <= 8 else 'rows=64,W>8'))
    for want, name in SLOTS['%s|simt_only' % op]:
        if want == seg:
            return name
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    phys_name = 'phys_cap_mixed.json' if os.path.exists(
        os.path.join(HERE, 'phys_cap_mixed.json')) else 'phys_cap18.json'
    cap = json.load(io.open(os.path.join(HERE, phys_name), encoding='utf-8'))

    # 组装槽位表：coef[0]=截距，keys 对齐 coef[1:]；'one' 若被选中则折叠进截距
    slot_tab = {}
    for slot, v in cap.items():
        for seg, name in SLOTS[slot]:
            s = v['segments'][seg]
            keys = list(s['keys']); coefs = list(s['coef'])
            assert len(keys) + 1 == len(coefs), (name, len(keys), len(coefs))
            intercept = coefs[0]
            terms, coeffs = [], []
            for k, c in zip(keys, coefs[1:]):
                if k == 'one':
                    intercept += c
                    continue
                terms.append(k); coeffs.append(c)
            slot_tab[name] = {'terms': terms, 'coef': coeffs,
                              'intercept': intercept, 'n': s['n'], 'mape': s['mape']}

    os.makedirs(os.path.join(HERE, 'gen'), exist_ok=True)
    # profile JSON
    prof = {}
    for mode in ('simd', 'simt'):
        obj = {'description':
               'Closed-form StridedWide (rank-2, one unit-stride axis, run>=32B, '
               'unmasked, inner) completion fits.  Calibrated on the WIDE2/LOADV1 '
               'sweeps; a slot is a steady-state per-iteration cost in SYS_CNT '
               'nanoseconds (== system cycles at 1 GHz).  Admission is bounded to '
               'the swept domain; anything outside keeps the fallback mechanisms.',
               'evidence': 'swide_phys/' + phys_name + '; '
                           'data_provider/strided_ldst/pre-explore README section 5'}
        for slot, v in cap.items():
            if not (slot.endswith('|' + mode) or slot.endswith('|' + mode + '_only')):
                continue
            for seg, name in SLOTS[slot]:
                t = slot_tab[name]
                obj[name] = {
                    'terms': t['terms'],
                    'intercept_system_cycles': t['intercept'],
                    'coefficient_system_cycles': t['coef'],
                    'evidence': 'swide_phys/' + phys_name + ' (%s, n=%d, MAPE %.2f%%)'
                                % (seg, t['n'], t['mape']),
                }
        prof[mode] = obj
    json.dump(prof, io.open(os.path.join(HERE, 'gen', 'profile_strided_wide_completion.json'),
                            'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump({k: v['terms'] for k, v in slot_tab.items()},
              io.open(os.path.join(HERE, 'gen', 'slot_terms.json'), 'w', encoding='utf-8'),
              indent=1, ensure_ascii=False)

    # C++ switch 体
    with io.open(os.path.join(HERE, 'gen', 'swide_switch.txt'), 'w', encoding='utf-8') as fh:
        for name in [s[1] for segs in SLOTS.values() for s in segs]:
            t = slot_tab[name]
            enum_name = ''.join(p.capitalize() for p in name.split('_'))
            rhs = 'fit.intercept'
            for i, k in enumerate(t['terms']):
                rhs += ' + c[%d] * f.%s' % (i, k)
            fh.write('  case SwideFormula::%s:  // terms: %s\n    return %s;\n'
                     % (enum_name, ', '.join(t['terms']), rhs))

    if not args.check:
        print('生成完成: gen/{profile_strided_wide_completion,slot_terms,golden*}.json, swide_switch.txt')
        return

    # ---------------- parity 验证 ----------------
    sys.path.insert(0, CAP); sys.path.insert(0, DATA)
    from feats_v2 import load_all_v2  # noqa
    from refit_simt import enrich  # noqa: 拟合时的特征口径(enrich 加 cseg 等物理因子)
    S, L = load_all_v2()
    worst = 0.0; worst_feat = ('', 0.0, 0.0)
    n_ok = 0; n_mismatch = 0
    for dat, op in ((S, 'store'), (L, 'load')):
        for r in dat:
            warps = int(r['num_warps'])
            slot = cpp_slot(r, warps, op)
            if r['order'] != 'inner' or int(r['run_bytes']) < 32:
                if slot is not None: raise AssertionError('gate mismatch')
                continue
            assert slot is not None, r
            fcpp = cpp_features(r, warps)
            enrich(r)   # Python 侧补齐拟合时新增的物理因子
            fpy = r['_f']
            for k in slot_tab[slot]['terms']:
                a, b = fcpp[k], fpy[k]
                d = abs(a - b) / max(1.0, abs(b))
                if d > worst:
                    worst, worst_feat = d, (k, a, b)
            pred = slot_tab[slot]['intercept'] + sum(
                c * fcpp[k] for k, c in zip(slot_tab[slot]['terms'], slot_tab[slot]['coef']))
            py_pred = slot_tab[slot]['intercept'] + sum(
                c * fpy[k] for k, c in zip(slot_tab[slot]['terms'], slot_tab[slot]['coef']))
            n_ok += 1
            # 判据: cpp 特征预测 vs python(拟合口径)特征预测 —— 应逐位一致
            if abs(pred - py_pred) > 1e-9 * max(1.0, abs(py_pred)):
                n_mismatch += 1
    print('parity: points=%d  最大特征相对偏差=%.3g (feat=%s cpp=%r py=%r)  大偏差点=%d'
          % (n_ok, worst, worst_feat[0], worst_feat[1], worst_feat[2], n_mismatch))

    # 黄金用例：每槽位取前 2 个数据点
    picked = {}
    for dat, op in ((S, 'store'), (L, 'load')):
        for r in dat:
            warps = int(r['num_warps'])
            slot = cpp_slot(r, warps, op)
            if slot is None: continue
            f = cpp_features(r, warps)
            pred = slot_tab[slot]['intercept'] + sum(
                c * f[k] for k, c in zip(slot_tab[slot]['terms'], slot_tab[slot]['coef']))
            item = {'rows': int(r['rows']), 'cols': int(r['C']), 'eb': int(r['element_bytes']),
                    'run_bytes': int(r['run_bytes']), 'gap_bytes': int(r['gap_bytes']),
                    'warps': warps, 'order': 'inner', 'op': op, 'path': r['path'],
                    'expected_ns': pred}
            picked.setdefault(slot, [])
            if len(picked[slot]) < 2:
                picked[slot].append(item)
    json.dump(picked, io.open(os.path.join(HERE, 'gen', 'golden_cases.json'), 'w',
                              encoding='utf-8'), indent=1)
    print('golden cases: %d slots' % len(picked))

    # UT 黄金用例 C++ 片段（系数与用例同源，检验 switch 项序）
    with io.open(os.path.join(HERE, 'gen', 'golden_ut.txt'), 'w', encoding='utf-8') as fh:
        fh.write('// 由 swide_phys/gen_integration.py 生成：黄金拟合系数 + 用例（几何 -> 期望 ns）。' + chr(10))
        fh.write('// 若 StridedWideCostModel.cpp 的 switch 项序与 profile terms 数组失配，此处数值即失配。' + chr(10))
        order = [x[1] for segs in SLOTS.values() for x in segs]
        fh.write('struct SwideGoldenFitRec { const char* slot; double intercept; std::vector<double> coef; };' + chr(10))
        fh.write('struct SwideGoldenCaseRec { const char* slot; int64_t rows, cols, eb, run, gap, warps; double expectedNs; };' + chr(10) + chr(10))
        fh.write('static const SwideGoldenFitRec kSwideGoldenFits[] = {' + chr(10))
        for name in order:
            t = slot_tab[name]
            coef = ', '.join('%.17g' % c for c in t['coef'])
            fh.write('    {"%s", %.17g, {%s}},' % (name, t['intercept'], coef) + chr(10))
        fh.write('};' + chr(10) + chr(10))
        fh.write('static const SwideGoldenCaseRec kSwideGoldenCases[] = {' + chr(10))
        for slot, cases in picked.items():
            for cse in cases:
                fh.write('    {"%s", %d, %d, %d, %d, %d, %d, %.17g},' % (
                    slot, cse['rows'], cse['cols'], cse['eb'], cse['run_bytes'],
                    cse['gap_bytes'], cse['warps'], cse['expected_ns']) + chr(10))
        fh.write('};' + chr(10))


if __name__ == '__main__':
    main()
