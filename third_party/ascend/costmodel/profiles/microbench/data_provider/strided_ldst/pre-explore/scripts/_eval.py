import sys, json, math, io, collections
sys.path.insert(0, 'scripts')
from fit_v9 import load
I = lambda x: int(float(x))
CV = lambda r: float(r.get('f_cols') or r['n']) if int(r['rank'])==2 else float(r['n'])
RV = lambda r: float(r.get('f_rows') or 0.0)
SP = lambda r: float(r.get('target_spread') or 0.0)
FP = lambda r: (((RV(r)-1)*float(r['f_row_pitch_bytes']) + (CV(r)-1)*float(r['ISB']) + float(r['eb'])) / max(1.0, RV(r)*CV(r)*float(r['eb']))) if int(r['rank'])==2 else 1.0

_cache = {}
def geo(r):
    key = (I(r['n']), I(CV(r)), I(RV(r)), int(r['ISB']), int(float(r['f_row_pitch_bytes'])), int(float(r['eb'])))
    if key in _cache: return _cache[key]
    n, C, R, isb, pitch, eb = key
    if R <= 1:
        addrs = [i*isb for i in range(n)]
    else:
        addrs = [i*pitch + j*isb for i in range(R) for j in range(max(C,1))]
    d = {'addrs': addrs, 'R': max(R,1), 'C': C, 'isb': isb, 'pitch': pitch, 'eb': eb, 'n': n}
    _cache[key] = d; return d

def cntb(g, B):
    c = collections.Counter(a//B for a in g['addrs']); return c
def Lb(g, B): return len(cntb(g, B))
def mx(g, B):
    c = cntb(g, B); return max(c.values()) if c else 0
def Sq(g, B):
    c = cntb(g, B); return sum(v*v for v in c.values())
def rmax1k(g):
    per = collections.defaultdict(set)
    R, C, isb, pitch = g['R'], g['C'], g['isb'], g['pitch']
    for i in range(R):
        for j in range(C): per[(i*pitch + j*isb)//1024].add(i)
    return max(len(v) for v in per.values()) if per else 0
def multi(g, B):
    per = collections.defaultdict(set)
    R, C, isb, pitch = g['R'], g['C'], g['isb'], g['pitch']
    for i in range(R):
        for j in range(C): per[(i*pitch + j*isb)//B].add(i)
    return sum(1 for v in per.values() if len(v) >= 2)
def wrow(g, B):
    R, C, isb, pitch = g['R'], g['C'], g['isb'], g['pitch']
    best = 0
    for i in range(R):
        c = collections.Counter((i*pitch + j*isb)//B for j in range(C))
        if c: best = max(best, max(c.values()))
    return best
def conc1k(g):
    per = collections.defaultdict(set)
    for a in g['addrs']: per[a//1024].add(a//128)
    return max(len(v) for v in per.values()) if per else 0

def predict(k, r):
    g = geo(r); n = g['n']; C = g['C']; R = g['R']; isb = g['isb']; eb = g['eb']
    W = I(r['W']); iters = math.ceil(n/(32.0*W))
    mwl = float(r.get('f_mean_warp_lines') or 1.0)
    if k == 'simd_load_wide_r1':
        span = (n-1)*isb + eb
        return 65.1033 + 0.00847947*span - 0.00478897*max(0.0, span-1536)
    if k == 'simd_load_wide_r2':
        rb = (C-1)*isb + eb; src = C*isb
        if src >= 32: return 68.6263 + 1.1068*R + 3.5929*rmax1k(g) + 0.5032*R*(1.0 if rb < 32 else 0.0)
        return 131.6284 + 3.0088*n + 54.4293*Lb(g, 64)
    if k == 'simd_load_gather_r1':
        return 75.61 + 0.36184*n + 0.079356*(n*eb) + 16.5852*mx(g,64) + 1.2845*mx(g,512) + 0.014506*Sq(g,1024)
    if k == 'simd_load_gather_r2':
        bs = 1.0 if C*eb < 32 else 0.0
        return (96.77 + 37.5624*bs*Lb(g,64) + 0.15665*(1-bs)*n*wrow(g,64)
                + 2.69141*(1-bs)*mx(g,1024) + 52.5296*bs*Lb(g,256) + 36.7907*bs*multi(g,256)
                + 0.000286488*(n*n*eb))
    if k == 'simd_store_r1':
        return 23.5488 + 0.185763*(n*eb) + 85.459976*mx(g,128) + 0.438898*Sq(g,64) + 0.057264*Sq(g,512)
    if k == 'simd_store_r2':
        Q4 = min(C, 4*32.0/isb)
        return 76.05 + 76.57*Q4 + 0.3458*n*Q4 + 1.115*max(0.0, n-128)
    if k == 'simt_load_r1':
        return max(2.33422*W, 0.770909*W*iters*mwl, 100.629 + 5.98993*min(mwl,8) + 9.13077*min(mwl,4) + 16.74857*iters + 0.2284797*Lb(g,128))
    if k == 'simt_load_r2':
        return max(2.424697*W, 0.6631113*W*iters*mwl, 109.283 + 7.55672*conc1k(g) + 9.524156*iters + 6.493768*min(mwl,4))
    if k == 'simt_store_r1':
        L = Lb(g,128)
        return max(2.18151*W, 14.9242 + 9.19181*iters,
                   4.77174*L*(1 if eb==1 else 0) + 4.73181*L*(1 if eb==2 else 0) + 1.44931*L*(1 if eb==4 else 0) + 1.93634*L*(1 if eb==8 else 0))
    if k == 'simt_store_r2':
        L = Lb(g,128); b4 = 1 if eb < 4 else 0
        return max(4.88754*L*b4 + 1.36564*L*(1 if eb==4 else 0) + 1.94354*L*(1 if eb==8 else 0),
                   0.135207*n*min(R,32)*b4 + 0.0456752*n*min(R,32)*(1 if eb==4 else 0) + 0.0613001*n*min(R,32)*(1 if eb==8 else 0),
                   18.3383*max(0.0, iters-2), 33.4824)
    return None

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

ORDER = ['simd_load_wide_r1','simd_load_gather_r1','simd_store_r1','simt_load_r1','simt_store_r1',
         'simd_load_wide_r2','simd_load_gather_r2','simd_store_r2','simt_load_r2','simt_store_r2']
m = {}
import glob
from pathlib import Path
fs_ = sorted(glob.glob('results/final/PHYS*_*.json'))
fs_.sort(key=lambda x: (2 if Path(x).name.startswith('PHYS3_') else 1 if Path(x).name.startswith('PHYS2_') else 0, x))
for p in fs_:
    j = json.load(io.open(p, encoding='utf-8'))
    for k, v in j.items():
        if not k.startswith('_') and isinstance(v, dict) and 'mape' in v: m[k] = v
out = {}
for k in ORDER:
    rr = raw(k)
    for f in SCOPE[k]: rr = [r for r in rr if f(r)]
    T = [float(r['target_ns']) for r in rr]
    P = [predict(k, r) for r in rr]
    mape = sum(abs(p-t)/t for p, t in zip(P, T))/len(T)*100
    out[k] = {'n': len(rr), 'mape_calc': round(mape,2), 'mape_json': m[k]['mape'],
              'mean_est_ns': sum(P)/len(P), 'mean_meas_ns': sum(T)/len(T)}
    print('%-22s n=%-5d  我算的MAPE=%6.2f%%  JSON=%-6s  %s' % (k, len(rr), mape, m[k]['mape'],
          'OK' if abs(mape-m[k]['mape'])<0.15 else '*** MISMATCH ***'))
io.open('_eval.json','w',encoding='utf-8').write(json.dumps(out, indent=1, ensure_ascii=False))