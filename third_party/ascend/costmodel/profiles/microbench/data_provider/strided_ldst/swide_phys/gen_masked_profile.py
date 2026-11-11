# -*- coding: utf-8 -*-
"""生成 masked SIMT 槽位的集成产物：profile 片段（合并进 strided_wide_completion）、
C++ switch 代码体、UT 黄金用例，并做 C++ 口径的特征/prediction 逐点 parity。

输入:
  out/masked_slots.json                    # fit_masked_slots.py --cap 20 的产物
  gen/profile_strided_wide_completion.json # 现有 unmasked 片段（simd+simt）
  D:\\work_wj\\mask_vs_nomask\\{WIDE2_final,LOADV1_v1,STORE_simt_rows16_v1,LOAD_simt_rows16_v1}.jsonl

输出:
  gen/profile_strided_wide_completion_v2.json   # unmasked + masked 合并
  gen/swide_switch_masked.txt                   # 16 个 masked switch case
  gen/golden_masked_ut.txt                      # masked 黄金拟合+用例
  parity: cpp_features 口径 vs 拟合口径（逐点）
"""
import sys, os, json, io, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r'D:\work_wj\63-workspace\63-workspace\strided-calib\03-featurize')
sys.path.insert(0, r'D:\work_wj\mask_vs_nomask')
sys.stdout.reconfigure(encoding='utf-8')

DATA = os.environ.get('SWIDE_DATA', r'D:\work_wj\mask_vs_nomask')
sys.path.insert(0, DATA)
from gen_integration import cpp_features, SLOTS  # noqa: E402
from feats_v2 import build as fv2_build  # noqa: E402
from cmn import feats as cmn_feats  # noqa: E402
from refit_simt import enrich, predict_seg  # noqa: E402

CAP = 20

MASKED_SEGS = ('rows2', 'rows8', 'rows64_wle2', 'rows64_w3_8', 'rows64_wgt8',
               'rows16_wle2', 'rows16_w3_8', 'rows16_wgt8')


def load_jsonl(path, want_var=None):
    out = []
    with io.open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if want_var is not None and int(r.get('var', -1)) != want_var:
                continue
            if r.get('error') or not r.get('valid'):
                continue
            if (r.get('spread') or 1) > 0.2:
                continue
            out.append(r)
    return out


def prep(rows):
    for r in rows:
        if '_f' not in r:
            r['_f'] = cmn_feats(r)
            fv2_build(r)
        enrich(r)
    return rows


def camel(name):
    return ''.join(p.capitalize() for p in name.split('_'))


def eval_model(r, fit):
    v = fit['coef'][0]
    for k, c in zip(fit['keys'], fit['coef'][1:]):
        v += c * r['_f'][k]
    return max(v, 1e-9)


def eval_cpp(r, fit):
    f = cpp_features(r, int(r['num_warps']))
    v = fit['coef'][0]
    for k, c in zip(fit['keys'], fit['coef'][1:]):
        v += c * f[k]
    return max(v, 1e-9)


def main():
    masked = json.load(io.open(os.path.join(HERE, 'out', 'masked_slots.json'), encoding='utf-8'))
    # 常量项 'one' 折叠进截距（与 unmasked 的 SLOTS 组装同规则）
    for name, t in masked.items():
        intercept = t['coef'][0]
        keys, coeffs = [], []
        for k, c in zip(t['keys'], t['coef'][1:]):
            if k == 'one':
                intercept += c
            else:
                keys.append(k)
                coeffs.append(c)
        t['keys'] = keys
        t['coef'] = [intercept] + coeffs

    base = json.load(io.open(os.path.join(HERE, 'gen', 'profile_strided_wide_completion.json'),
                             encoding='utf-8'))
    simt = base['simt']
    simt['description'] = (simt['description'].split(' Masked SIMT slots')[0] +
                           ' Masked SIMT slots (store and load, rows 2/8/64/W-split and '
                           'rows16/W-split) are separate linear fits: the measured masked '
                           'unmasked ratio is only uniform for SIMD (median 1.000, 99.2% '
                           'within 5%), while the SIMT masked store/load relationship is '
                           'not a smooth offset, so sharing one formula degraded '
                           'segment MAPE to 12-22%.')
    for name in sorted(masked):
        t = masked[name]
        simt[name] = {
            'terms': t['keys'],
            'intercept_system_cycles': t['coef'][0],
            'coefficient_system_cycles': t['coef'][1:],
            'evidence': 'swide_phys masked refit: %d points, in-sample MAPE %.2f%%'
                        % (t['n'], t['mape']),
        }
    out = {'simd': base['simd'], 'simt': simt}
    os.makedirs(os.path.join(HERE, 'gen'), exist_ok=True)
    json.dump(out, io.open(os.path.join(HERE, 'gen', 'profile_strided_wide_completion_v2.json'),
                           'w', encoding='utf-8'), indent=1, ensure_ascii=False)

    # C++ switch 体
    with io.open(os.path.join(HERE, 'gen', 'swide_switch_masked.txt'), 'w', encoding='utf-8') as fh:
        for name in sorted(masked):
            t = masked[name]
            terms = [k for k in t['keys']]
            rhs = 'fit.intercept'
            for i, k in enumerate(terms):
                rhs += ' + c[%d] * f.%s' % (i, k)
            fh.write('  case SwideFormula::%s:  // terms: %s\n    return %s;\n'
                     % (camel(name), ', '.join(terms), rhs))

    # parity + 黄金用例
    store_v1 = prep(load_jsonl(os.path.join(DATA, 'WIDE2_final.jsonl'), 1) +
                    load_jsonl(os.path.join(DATA, 'STORE_simt_rows16_v1.jsonl')))
    load_v1 = prep(load_jsonl(os.path.join(DATA, 'LOADV1_v1.jsonl')) +
                   load_jsonl(os.path.join(DATA, 'LOAD_simt_rows16_v1.jsonl')))
    worst = 0.0; npts = 0; bad = 0
    golden = {}
    for dat, opname in ((store_v1, 'store'), (load_v1, 'load')):
        for r in dat:
            if r['path'] != 'simt_only' or r['order'] != 'inner' or int(r['run_bytes']) < 32:
                continue
            key = 'simt_%s_swide_masked_%s' % (opname, None)
            # 找对应槽位（与 fit_masked_slots.seg_name 同规则）
            rows = int(r['rows']); W = r['_f']['W']
            if rows == 16:
                seg = 'rows16_wle2' if W <= 2 else ('rows16_w3_8' if W <= 8 else 'rows16_wgt8')
            elif rows == 2:
                seg = 'rows2'
            elif rows == 8:
                seg = 'rows8'
            elif rows == 64:
                seg = 'rows64_wle2' if W <= 2 else ('rows64_w3_8' if W <= 8 else 'rows64_wgt8')
            else:
                continue
            slot = 'simt_%s_swide_masked_%s' % (opname, seg)
            if slot not in masked:
                continue
            fit = masked[slot]
            py = eval_model(r, fit); cc = eval_cpp(r, fit)
            exp = {k: r['_f'][k] for k in fit['keys']}
            py2 = fit['coef'][0] + sum(c * exp[k] for k, c in zip(fit['keys'], fit['coef'][1:]))
            d = abs(cc - py2) / max(1.0, abs(py2))
            worst = max(worst, d)
            npts += 1
            if d > 1e-9:
                bad += 1
            golden.setdefault(slot, [])
            if len(golden[slot]) < 2:
                golden[slot].append({
                    'rows': rows, 'cols': int(r['C']), 'eb': int(r['element_bytes']),
                    'run_bytes': int(r['run_bytes']), 'gap_bytes': int(r['gap_bytes']),
                    'warps': int(r['num_warps']), 'op': opname,
                    'expected_ns': py2,
                })
    print('parity: points=%d  max feature/prediction deviation=%.3g  mismatches=%d'
          % (npts, worst, bad))

    with io.open(os.path.join(HERE, 'gen', 'golden_masked_ut.txt'), 'w', encoding='utf-8') as fh:
        fh.write('// masked SIMT 槽位黄金拟合与用例（由 gen_masked_profile.py 生成）\n')
        fh.write('static const SwideGoldenFitRec kSwideMaskedGoldenFits[] = {\n')
        for name in sorted(masked):
            t = masked[name]
            coef = ', '.join('%.17g' % c for c in t['coef'][1:])
            fh.write('    {"%s", %.17g, {%s}},\n' % (name, t['coef'][0], coef))
        fh.write('};\n\n')
        fh.write('static const SwideGoldenCaseRec kSwideMaskedGoldenCases[] = {\n')
        for slot in sorted(golden):
            for c in golden[slot]:
                fh.write('    {"%s", %d, %d, %d, %d, %d, %d, %.17g},\n'
                         % (slot, c['rows'], c['cols'], c['eb'], c['run_bytes'],
                            c['gap_bytes'], c['warps'], c['expected_ns']))
        fh.write('};\n')
    print('wrote gen/{profile_strided_wide_completion_v2.json, swide_switch_masked.txt, golden_masked_ut.txt}')


if __name__ == '__main__':
    main()
