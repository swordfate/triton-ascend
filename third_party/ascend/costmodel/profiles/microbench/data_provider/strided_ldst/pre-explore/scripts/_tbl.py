import sys, json, io, collections
sys.path.insert(0, 'scripts')
from fit_v9 import load
I = lambda x: int(float(x))
CV = lambda r: float(r.get('f_cols') or r['n']) if int(r['rank'])==2 else float(r['n'])
RV = lambda r: float(r.get('f_rows') or 0.0)
SP = lambda r: float(r.get('target_spread') or 0.0)
FP = lambda r: (((RV(r)-1)*float(r['f_row_pitch_bytes']) + (CV(r)-1)*float(r['ISB']) + float(r['eb'])) / max(1.0, RV(r)*CV(r)*float(r['eb']))) if int(r['rank'])==2 else 1.0
EB = {1:'int8', 2:'float16', 4:'float32', 8:'int64'}
rows = load('results/final/dataset_new_only_v5.csv')
rr0 = [r for r in rows if str(r.get('valid'))=='1']
SCOPE = {
 'simd_load_wide_r1':  [lambda r: float(r['eb'])<=4, lambda r: SP(r)<=0.2],
 'simd_load_wide_r2':  [lambda r: float(r['eb'])<=4, lambda r: RV(r)>=2, lambda r: CV(r)>=4, lambda r: SP(r)<=0.2],
 'simd_load_gather_r1':[lambda r: I(r['n'])>=32, lambda r: float(r['ISB'])<=128, lambda r: SP(r)<=0.2],
 'simd_load_gather_r2':[lambda r: RV(r)>=2, lambda r: SP(r)<=0.2, lambda r: FP(r)<=32],
 'simd_store_r1':      [lambda r: float(r['eb'])<=4, lambda r: float(r['ISB'])<=128],
 'simd_store_r2':      [lambda r: RV(r)>=2, lambda r: CV(r)*float(r['eb'])>=32, lambda r: float(r['eb'])<=4],
 'simt_load_r1':       [lambda r: float(r['eb'])<=4, lambda r: float(r['ISB'])<=128, lambda r: SP(r)<=0.2],
 'simt_load_r2':       [lambda r: float(r['eb'])<=4, lambda r: RV(r)>=2, lambda r: SP(r)<=0.2],
 'simt_store_r1':      [],
 'simt_store_r2':      [lambda r: RV(r)>=2, lambda r: CV(r)*float(r['eb'])>=32, lambda r: CV(r)>=8, lambda r: I(r['n'])>=64],
}
def raw(k):
    sl, rk = k.rsplit('_r', 1); rk = int(rk)
    if sl == 'simd_load_wide':   return [r for r in rr0 if r['path']=='simd' and r['op']=='load' and r['ISB']<=2.0*r['eb'] and int(r['rank'])==rk]
    if sl == 'simd_load_gather': return [r for r in rr0 if r['path']=='simd' and r['op']=='load' and r['ISB']>2.0*r['eb'] and int(r['rank'])==rk]
    if sl == 'simd_store':       return [r for r in rr0 if r['path']=='simd' and r['op']=='store' and int(r['rank'])==rk]
    return [r for r in rr0 if r['path']=='simt_only' and r['op']==sl.split('_')[1] and int(r['rank'])==rk]
EV = json.load(io.open('_eval.json', encoding='utf-8'))
SC = {'simd_load_wide':('1d/simd/load','2d/simd/load'), 'simd_load_gather':('1d/simd/load','2d/simd/load'),
      'simd_store':('1d/simd/store','2d/simd/store'), 'simt_load':('1d/simt/load','2d/simt/load'),
      'simt_store':('1d/simt/store','2d/simt/store')}
ORDER = ['simd_load_wide_r1','simd_load_gather_r1','simd_store_r1','simt_load_r1','simt_store_r1',
         'simd_load_wide_r2','simd_load_gather_r2','simd_store_r2','simt_load_r2','simt_store_r2']
NAME = {'simd_load_wide':'simd_load_wide','simd_load_gather':'simd_load_gather','simd_store':'simd_store',
        'simt_load':'simt_load','simt_store':'simt_store'}
res = []
for k in ORDER:
    sl, rk = k.rsplit('_r', 1); rk = int(rk)
    rr = raw(k)
    for f in SCOPE[k]: rr = [r for r in rr if f(r)]
    T = [float(r['target_ns']) for r in rr]
    dts = sorted({EB[int(float(r['eb']))] for r in rr}, key=lambda z: ['int8','float16','float32','int64'].index(z))
    sts = sorted({round(float(r['ISB'])/float(r['eb'])) for r in rr})
    if rk == 1:
        ns = sorted({I(r['n']) for r in rr})
        cfg = 'dtype=%s; n=%s; stride=%s' % ('/'.join(dts), '/'.join(str(x) for x in ns), '/'.join(str(x) for x in sts))
        case = '%s/stride%s' % (SC[sl][0], ('>=2' if min(sts)>=2 else '<=2'))
    else:
        sh = sorted({'%dx%d' % (I(RV(r)), I(CV(r))) for r in rr}, key=lambda z:(int(z.split('x')[0])*int(z.split('x')[1]), int(z.split('x')[0])))
        rs = sorted({round(float(r['f_row_pitch_bytes'])/float(r['eb'])) for r in rr})
        cfg = 'dtype=%s; R×C=%s; stride=%s; row_stride=%s' % ('/'.join(dts), '/'.join(sh), '/'.join(str(x) for x in sts), '/'.join(str(x) for x in rs))
        case = '%s/stride%s (R×C)' % (SC[sl][1], ('<=2' if max(sts)<=2 else '>=2'))
    ws = sorted({I(r['W']) for r in rr})
    if len(ws) > 1: cfg += '; num_warps=%s' % '/'.join(str(x) for x in ws)
    e = EV[k]
    mape = e['mape_calc']
    res.append({'k':k,'case':case,'cfg':cfg,'est_us':e['mean_est_ns']/1000.0,'meas_us':e['mean_meas_ns']/1000.0,
                'mape':mape,'verdict':'PASS' if mape<10 else 'FAILED','n':e['n']})
io.open('_readme_tbl.json','w',encoding='utf-8').write(json.dumps(res, indent=1, ensure_ascii=False))
for r in res: print('%-22s %-28s est=%.4f us meas=%.4f us mape=%.2f %s' % (r['k'], r['case'], r['est_us'], r['meas_us'], r['mape'], r['verdict']))