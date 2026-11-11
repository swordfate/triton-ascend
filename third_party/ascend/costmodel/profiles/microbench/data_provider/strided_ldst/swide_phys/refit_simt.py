# -*- coding: utf-8 -*-
"""StridedWide SIMT 重拟合：允许分段 + 更复杂特征，目标 MAPE < 10%。

数据: D:\\work_wj\\mask_vs_nomask\\{WIDE2_final,LOADV1}.jsonl (原始实测, 未提交 git)
特征: 复用 strided-calib/03-featurize 的 cmn.feats + feats_v2.build, 再叠加 hinge/交互特征
方法: 逐段贪心前向选特征(numpy 加权LS+MAPE) + Nelder-Mead MAPE 精修(3 重启)
评估: 段内 in-sample MAPE -> 加权 blend; 另有 dtype 留一 OOS 与 dense_rows 域外探针

用法:
  python refit_simt.py eval-baseline     # 用 models/final_v5.json 系数复现已交付 MAPE
  python refit_simt.py fit --scheme K8|rows3|K8_W|rows3_W|rows3_W2 [--cap 12] [--oos]
"""
import sys, os, math, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r'D:\work_wj\63-workspace\63-workspace\strided-calib\03-featurize')
sys.path.insert(0, r'D:\work_wj\mask_vs_nomask')
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8')
from feats_v2 import load_all_v2  # noqa: E402

DATA = r'D:\work_wj\mask_vs_nomask'
MODELS = r'D:\work_wj\63-workspace\63-workspace\strided-calib\models'
OUT = os.path.join(HERE, 'out')

# ------------------------------------------------------------ 物理因子池
# 每个因子都有明确硬件含义; SIMD 池不含任何 warp/线程量(线程是 SIMT 才有的)

LABELS = {
    'n': '总元素数', 'nbytes': '总字节数',
    'W': 'warp 数', 'invW': '1/W',
    'elem_per_lane': '每线程元素数 n/(32W)', 'bytes_per_lane': '每线程字节数 nbytes/(32W)',
    'iters': '每线程访存轮数 ceil(n/(32W))', 'q': 'n/(32W) 精确值',
    'elem_per_warp': '每 warp 元素数 n/W', 'bytes_per_warp': '每 warp 字节数 nbytes/W',
    'line_per_warp': '每 warp 摊到的 128B line 数', 'seg_per_warp': '每 warp 摊到的连续段数',
    'h': '超出单轮(32W)的元素数', 'h_pos': '是否有超出', 'rounds': 'ceil(n/(32W))',
    'rows': '连续段数(描述符遍历)', 'span': '访问足迹字节', 'run': '段长字节', 'gap': '间隔字节',
    'stride': '段距字节', 'duty': '段占空比 run/(run+gap)', 'duty2': '占空比平方',
    'L128': '触达的不同 128B line 数', 'conc1k': '1KB 窗内 line 数', 'mx128': '单 128B 块最多元素数',
    'total_lines': 'rows x 每段line数', 'lines_per_seg': '每段 128B line 数',
    'sectors': '每段 32B sector 数', 'mis32': '段起始未对齐 32B',
    'cseg': '每段元素数 run/eb', 'cseg_mod32': '每段元素数 mod 32 (lane 覆盖锯齿)',
    'cseg_div32': 'floor(每段元素数/32)', 'cseg_x_invW': '每段元素数/W',
    'seg_lanes': '每段占的 32-lane 组数', 'seg_lanes_r': '组数破整部分',
    'elem_per_line': '每 line 元素数',
    'L256': '触达的不同 256B 块数(L2 slice 分散度)', 'L512': '触达的不同 512B 块数',
    'L1024': '触达的不同 1KB 块数', 'L2048': '触达的不同 2KB 块数', 'L4096': '触达的不同 4KB 块数',
    'mx64': '单 64B 块最多元素数', 'mx256': '单 256B 块最多元素数', 'mx512': '单 512B 块最多元素数',
    'mx1024': '单 1KB 块最多元素数', 'mx2048': '单 2KB 块最多元素数', 'mx4096': '单 4KB 块最多元素数',
    'S32': '同 32B 块元素集中度 Σk²', 'S64': '同 64B 块元素集中度 Σk²',
    'S128': '同 128B 块元素集中度 Σk²', 'S256': '同 256B 块集中度 Σk²',
    'S512': '同 512B 块集中度 Σk²', 'S1024': '同 1KB 块集中度 Σk²',
    'S2048': '同 2KB 块集中度 Σk²', 'S4096': '同 4KB 块集中度 Σk²',
    'r16invW': 'max(0,rows-16)/W 大段数稀释', 'r16minW4': 'max(0,rows-16)x min(W,4) 大段数x小W区',
    'r16_x_segW': 'max(0,rows-16)x 每warp段数', 'n2W': 'n x 每warp元素数 (每线程消费加权)',
    'h_div_w': '超出单轮元素数/W',
    'minW2': 'min(W,2) 小W延迟拐点', 'minW4': 'min(W,4)', 'minW8': 'min(W,8)', 'minW16': 'min(W,16)',
    'maxW2': 'max(0,W-2)', 'maxW4': 'max(0,W-4) 大W吞吐拐点', 'maxW8': 'max(0,W-8)', 'maxW16': 'max(0,W-16)',
    'minrows8': 'min(rows,8)', 'minrows16': 'min(rows,16)',
    'maxrows8': 'max(0,rows-8) 段数台阶', 'maxrows16': 'max(0,rows-16) 段数台阶',
    'rows_x_duty': 'rows x 占空比', 'rows2': 'rows^2', 'inv_rows': '1/rows',
    'sqrt_n': 'sqrt(元素数)', 'sqrt_rows': 'sqrt(段数)', 'inv_n': '1/n',
    'n_div_rows': '每段平均元素数', 'elem_per_warp_x_rows': '每warp元素 x 段数',
}
# SIMT: 线程/warp 结构 + 内存系统 + 拐点 hinge(物理: 小W延迟/大W吞吐)
SIMT_PHYS = ['n', 'nbytes', 'W', 'invW', 'elem_per_lane', 'bytes_per_lane', 'iters', 'q',
             'elem_per_warp', 'bytes_per_warp', 'line_per_warp', 'seg_per_warp',
             'h', 'h_pos', 'rounds',
             'rows', 'span', 'run', 'gap', 'stride', 'duty', 'duty2',
             'L128', 'L256', 'L512', 'L1024', 'L2048', 'L4096',
             'mx64', 'mx128', 'mx256', 'mx512', 'mx1024', 'mx2048', 'mx4096',
             'S32', 'S64', 'S128', 'S256', 'S512', 'S1024', 'S2048', 'S4096',
             'total_lines', 'lines_per_seg', 'sectors', 'mis32',
             'cseg', 'cseg_mod32', 'cseg_div32', 'cseg_x_invW', 'seg_lanes', 'seg_lanes_r',
             'elem_per_line', 'r16invW', 'r16minW4', 'r16_x_segW', 'n2W', 'h_div_w',
             'minW2', 'minW4', 'minW8', 'minW16', 'maxW2', 'maxW4', 'maxW8', 'maxW16']
# SIMD: 纯几何/字节/line, 无任何线程量
SIMD_PHYS = ['n', 'nbytes', 'rows', 'run', 'gap', 'stride', 'duty', 'duty2', 'span',
             'L128', 'L256', 'L512', 'L1024', 'L2048', 'L4096',
             'mx128', 'mx256', 'mx512', 'mx1024', 'mx2048', 'mx4096',
             'S128', 'S256', 'S512', 'S1024', 'S2048', 'S4096',
             'total_lines', 'lines_per_seg', 'sectors', 'mis32',
             'cseg', 'cseg_mod32', 'elem_per_line', 'inv_rows', 'sqrt_n', 'sqrt_rows',
             'minrows8', 'minrows16', 'maxrows8', 'maxrows16', 'rows_x_duty', 'rows2', 'inv_n', 'n_div_rows']

# 运行期覆盖: fit_segment 只能从这个池选
POOL_OVERRIDE = None

# ---------------------------------------------------------------- features

def enrich(r):
    f = r['_f']
    w = max(1.0, f['W']); rows = f['rows']; n = f['n']; eb = float(int(r['element_bytes']))
    f['n'] = n; f['nbytes'] = n * eb; f['iters'] = float(math.ceil(n / (32.0 * w))); f['W'] = w
    q = n / (32.0 * w); h = max(n - 32.0 * w, 0.0); rd = max(1.0, math.ceil(q))
    f['q'] = q; f['h'] = h; f['h_div_w'] = h / w; f['rounds'] = rd; f['h_pos'] = 1.0 if h > 0 else 0.0
    f['r_x_h'] = rd * f['h_pos']
    f['seg_per_warp'] = float(math.ceil(rows / w))
    f['line_per_warp'] = float(math.ceil(f['total_lines'] / w))
    f['span_line'] = f['span'] / 128.0
    f['bytes_per_warp'] = f['nbytes'] / w
    f['elem_per_warp'] = n / w
    # hinge 特征: 捕捉 max(A/W, B + C*W) 的拐点与 rows 台阶
    for t in (2, 4, 8, 16):
        f['minW%d' % t] = min(w, float(t))
        f['maxW%d' % t] = max(0.0, w - t)
    f['invW'] = 1.0 / w
    f['sat_invW2'] = max(f['invW'], 1.0 / 2)
    f['sat_invW4'] = max(f['invW'], 1.0 / 4)
    f['sat_invW8'] = max(f['invW'], 1.0 / 8)
    f['min_invW8'] = min(f['invW'], 1.0 / 8)
    f['minrows8'] = min(rows, 8.0); f['minrows16'] = min(rows, 16.0)
    f['maxrows8'] = max(0.0, rows - 8.0); f['maxrows16'] = max(0.0, rows - 16.0)
    f['r16invW'] = f['maxrows16'] * f['invW']          # 大段数 x 每 warp 稀释
    f['r16minW4'] = f['maxrows16'] * f['minW4']        # 大段数 x 小W(饱和)
    f['r16maxW4'] = f['maxrows16'] * f['maxW4']
    f['r16_x_segW'] = f['maxrows16'] * f['seg_per_warp']
    f['log2gap'] = math.log2(max(1.0, f['gap'])); f['log2run'] = math.log2(max(1.0, f['run']))
    f['one'] = 1.0
    # 每线程/每 lane 工作量(SIMT 物理因子)
    f['elem_per_lane'] = n / (32.0 * w)
    f['bytes_per_lane'] = n * eb / (32.0 * w)
    f['elem_per_warp'] = n / w
    # C 相关: 每段元素数与 32 lane 的整除关系(README 5.8: 同字节下 f16 与 int8 差 4.2x)
    cseg = f['run'] / eb
    f['cseg'] = cseg; f['cseg_mod32'] = float(cseg % 32.0)
    f['cseg_div32'] = math.floor(cseg / 32.0); f['cseg_r32'] = (cseg % 32.0) / 32.0
    f['cseg_x_invW'] = cseg / w; f['cseg_x_minW4'] = cseg * f['minW4']
    f['cseg_x_r16'] = cseg * f['maxrows16']
    f['seg_lanes'] = cseg / 32.0  # 每段占多少 warp-lane 组
    f['seg_lanes_r'] = f['seg_lanes'] - math.floor(f['seg_lanes'])
    return f


# 全部候选特征(先收集, 再过滤掉本段常量/近常量列)
def pool_of(seg):
    keys = set()
    for r in seg:
        keys.update(r['_f'].keys())
    keys.discard('one'); keys.discard('_f')
    return ['one'] + sorted(keys - {'one'})


def design(seg, keys):
    X = np.array([[r['_f'][k] for k in keys] for r in seg], dtype=np.float64)
    y = np.array([r['target_ns'] for r in seg], dtype=np.float64)
    return X, y


def mape(X, y, p):
    pred = X @ p
    pred = np.maximum(pred, 1e-9)
    return float(np.mean(np.abs(pred - y) / y) * 100.0)


def fit_ls(X, y):
    w = 1.0 / y
    sw = np.sqrt(w)
    A = X * sw[:, None]; b = y * sw
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    return p


def nm_polish(X, y, p0, iters=6000, rng=None, jitter=0.0):
    n = len(p0)
    x0 = p0.copy()
    if jitter > 0 and rng is not None:
        x0 = x0 * (1.0 + jitter * (rng.random(n) - 0.5))
    step = np.maximum(np.abs(x0) * 0.30, 1e-7); step[0] = max(abs(x0[0]) * 0.3, 1.0)
    pts = [x0]
    for i in range(n):
        p = x0.copy(); p[i] += step[i]; pts.append(p)
    vals = [mape(X, y, p) for p in pts]
    for _ in range(iters):
        o = np.argsort(vals); pts = [pts[i] for i in o]; vals = [vals[i] for i in o]
        if vals[-1] - vals[0] < 1e-10:
            break
        cen = np.mean(pts[:-1], axis=0)
        xr = cen + (cen - pts[-1]); fr = mape(X, y, xr)
        if fr < vals[0]:
            xe = cen + 2 * (cen - pts[-1]); fe = mape(X, y, xe)
            if fe < fr: pts[-1], vals[-1] = xe, fe
            else: pts[-1], vals[-1] = xr, fr
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = cen + 0.5 * (pts[-1] - cen); fc = mape(X, y, xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0]); vals[i] = mape(X, y, pts[i])
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def ridge_fit(X, y, lam_rel):
    """标准化空间里的加权 ridge; X,y 已标准化/中心化由调用方处理."""
    w = 1.0 / y
    sw = np.sqrt(w)
    A = X * sw[:, None]; b = y * sw
    m = X.shape[1]
    AtA = A.T @ A
    tr = float(np.trace(AtA))
    G = AtA + (lam_rel * tr / m if tr > 0 else 1e-9 * max(1.0, m)) * np.eye(m)
    try:
        return np.linalg.solve(G, A.T @ b)
    except np.linalg.LinAlgError:
        # 秩亏时退化为最小二乘（ridge 的数值兜底，不改变主路径）
        return np.linalg.lstsq(G, A.T @ b, rcond=None)[0]


class Scaler:
    def __init__(self, X):
        self.mu = X.mean(axis=0); self.sd = X.std(axis=0)
        self.sd = np.where(self.sd < 1e-12, 1.0, self.sd)
    def t(self, X):
        return (X - self.mu) / self.sd


def fit_segment(seg, cap=12, polish=True, seed=8888, stop_tol=0.005):
    keys = pool_of(seg)
    if POOL_OVERRIDE is not None:
        keys = ['one'] + [k for k in keys if k in POOL_OVERRIDE]
    Xall, y = design(seg, keys)
    keep = ['one']
    for j in range(1, len(keys)):
        col = Xall[:, j]
        if col.std() > 1e-9 * max(1.0, abs(col.mean())):
            keep.append(keys[j])
    keys = keep
    n = len(seg)
    # 5 折 CV(固定划分), 用于选择项数 —— 防共线系数在 OOS 上爆炸
    rng0 = np.random.default_rng(2024)
    fold = np.zeros(n, dtype=int)
    perm = rng0.permutation(n)
    for i, pi in enumerate(perm):
        fold[pi] = i % 5

    def cv_mape(sel):
        ks = sel
        if len(ks) == 0:
            p = np.array([np.median(y)])
            return mape(design(seg, ['one'])[0], y, p)
        es = []
        for k in range(5):
            tr = [r for r, fi in zip(seg, fold) if fi != k]
            te = [r for r, fi in zip(seg, fold) if fi == k]
            Xtr, ytr = design(tr, ks); Xte, yte = design(te, ks)
            sc = Scaler(Xtr)
            A = np.column_stack([np.ones(len(tr)), sc.t(Xtr)])
            p = ridge_fit(A, ytr, 5e-3 if n < 100 else 1e-3)
            At = np.column_stack([np.ones(len(te)), sc.t(Xte)])
            es.append(mape(At, yte, p))
        return float(np.mean(es))

    # 前向选择: 按 CV MAPE 选, 项数也由 CV 决定
    sel = []; best_cv = cv_mape(sel); curve = [(0, best_cv)]
    while len(sel) < cap:
        cand = None
        for kk in keys:
            if kk in sel: continue
            c = cv_mape(sel + [kk])
            if cand is None or c < cand[0]: cand = (c, kk)
        if cand is None: break
        sel.append(cand[1]); best_cv = cand[0]; curve.append((len(sel), best_cv))
    # 在 CV 最优项数处截断(宽松 1% 容差取更省项数)
    best_k = min(curve, key=lambda t: t[1])[0]
    tol = best_cv * 1.01 + 0.05
    for k, c in curve:
        if k > 0 and c <= tol:
            best_k = min(best_k, k); break
    sel = sel[:best_k]

    # 全量重拟合(标准化+ridge), NM 只做温和精修且以 CV 不劣化为门槛
    X, _ = design(seg, sel)
    sc = Scaler(X)
    A = np.column_stack([np.ones(n), sc.t(X)])
    p_std = ridge_fit(A, y, 5e-3 if n < 100 else 1e-3)
    def to_raw(p_std):
        c = np.zeros(len(sel) + 1)
        c[0] = p_std[0] - np.sum(p_std[1:] * sc.mu / sc.sd)
        c[1:] = p_std[1:] / sc.sd
        return c
    p_raw = to_raw(p_std)
    X1 = np.column_stack([np.ones(n), X])   # 原始空间评估统一带截距列
    train_m = mape(X1, y, p_raw)
    if polish and sel:
        rng = np.random.default_rng(seed)
        bp, bm = nm_polish(X1, y, p_raw, iters=3000, rng=rng)
        cv_p = []
        for k in range(5):
            te = [r for r, fi in zip(seg, fold) if fi == k]
            Xte, yte = design(te, sel)
            cv_p.append(mape(np.column_stack([np.ones(len(te)), Xte]), yte, bp))
        if float(np.mean(cv_p)) <= best_cv * 1.05:
            p_raw, train_m = bp, bm
    return {'keys': sel, 'coef': [float(x) for x in p_raw], 'mape': train_m,
            'cv_mape': float(best_cv), 'n': len(seg)}


# ---------------------------------------------------------------- schemes

SEG_K = 4  # simdK 方案的段阈值(由调用方按槽位设置: load=4, store=2)


def segments(ss, scheme):
    W = lambda r: r['_f']['W']
    if scheme == 'simdK':
        return [('rows<=%d' % SEG_K, [r for r in ss if r['rows'] <= SEG_K]),
                ('rows>%d' % SEG_K, [r for r in ss if r['rows'] > SEG_K])]
    if scheme == 'K8':
        return [('rows<=8', [r for r in ss if r['rows'] <= 8]),
                ('rows>8',  [r for r in ss if r['rows'] > 8])]
    if scheme == 'rows3':
        return [('rows=2', [r for r in ss if r['rows'] == 2]),
                ('rows=8', [r for r in ss if r['rows'] == 8]),
                ('rows=64', [r for r in ss if r['rows'] == 64])]
    if scheme == 'K8_W':
        return [('rows<=8', [r for r in ss if r['rows'] <= 8]),
                ('rows>8,W<=4', [r for r in ss if r['rows'] > 8 and W(r) <= 4]),
                ('rows>8,W>4', [r for r in ss if r['rows'] > 8 and W(r) > 4])]
    if scheme == 'rows3_W':
        return [('rows=2', [r for r in ss if r['rows'] == 2]),
                ('rows=8', [r for r in ss if r['rows'] == 8]),
                ('rows=64,W<=4', [r for r in ss if r['rows'] == 64 and W(r) <= 4]),
                ('rows=64,W>4', [r for r in ss if r['rows'] == 64 and W(r) > 4])]
    if scheme == 'rows3_W2':
        return [('rows=2', [r for r in ss if r['rows'] == 2]),
                ('rows=8', [r for r in ss if r['rows'] == 8]),
                ('rows=64,W<=2', [r for r in ss if r['rows'] == 64 and W(r) <= 2]),
                ('rows=64,3-8', [r for r in ss if r['rows'] == 64 and 3 <= W(r) <= 8]),
                ('rows=64,W>8', [r for r in ss if r['rows'] == 64 and W(r) > 8])]
    if scheme == 'rows3_W2b':
        return [('rows=2', [r for r in ss if r['rows'] == 2]),
                ('rows=8', [r for r in ss if r['rows'] == 8]),
                ('rows=64,W=1', [r for r in ss if r['rows'] == 64 and W(r) == 1]),
                ('rows=64,W=2', [r for r in ss if r['rows'] == 64 and W(r) == 2]),
                ('rows=64,3-8', [r for r in ss if r['rows'] == 64 and 3 <= W(r) <= 8]),
                ('rows=64,W>8', [r for r in ss if r['rows'] == 64 and W(r) > 8])]
    raise ValueError(scheme)


def slot_points(dat, path):
    ss = [r for r in dat if r['path'] == path and r['order'] == 'inner' and r['run_bytes'] >= 32]
    for r in ss: enrich(r)
    return ss


def predict_seg(r, seg_model):
    v = seg_model['coef'][0]
    for k, c in zip(seg_model['keys'], seg_model['coef'][1:]):
        v += c * r['_f'][k]
    return max(v, 1e-9)


def eval_segments(ss, scheme, rec):
    """用已拟合的段模型评估任意点集(返回逐点误差%)."""
    errs = []
    for name, part in segments(ss, scheme):
        m = rec[name]
        for r in part:
            e = abs(predict_seg(r, m) - r['target_ns']) / r['target_ns'] * 100
            errs.append((e, r))
    return errs


def oos_mode(S, L, scheme, cap):
    """dtype 留一: 在其余 dtype 上重拟合, 在被留出 dtype 上评估."""
    print('=== dtype 留一 OOS  scheme=%s cap=%d ===' % (scheme, cap))
    for slot in ('load|simt_only', 'store|simt_only'):
        dat = L if slot.startswith('load') else S
        ss = slot_points(dat, slot.split('|')[1])
        dtypes = sorted(set(r['dtype'] for r in ss))
        tot = 0.0; n = 0
        print('--- %s (n=%d, dtypes=%s)' % (slot, len(ss), dtypes))
        for hold in dtypes:
            train = [r for r in ss if r['dtype'] != hold]
            test = [r for r in ss if r['dtype'] == hold]
            rec = {}
            for name, part in segments(train, scheme):
                rec[name] = fit_segment(part, cap=cap, seed=1234)
            errs = [e for e, _ in eval_segments(test, scheme, rec)]
            mm = float(np.mean(errs))
            tot += mm * len(test); n += len(test)
            print('    留出 %-9s n=%-4d OOS MAPE=%6.2f%%  (max %6.1f%%)' % (hold, len(test), mm, max(errs)))
        print('    => %s OOS 加权 %.2f%%' % (slot, tot / n))


def diagnose(S, L, scheme, cap):
    """全量拟合后打印每段最差点."""
    for slot in ('load|simt_only', 'store|simt_only'):
        dat = L if slot.startswith('load') else S
        ss = slot_points(dat, slot.split('|')[1])
        print('--- %s worst points ---' % slot)
        rec = {}
        for name, part in segments(ss, scheme):
            rec[name] = fit_segment(part, cap=cap)
        errs = eval_segments(ss, scheme, rec)
        errs.sort(key=lambda x: -x[0])
        for e, r in errs[:8]:
            print('   %6.1f%%  %s run=%s gap=%s rows=%s C=%s W=%s iters=%d' % (
                e, r['dtype'], r['run_bytes'], r['gap_bytes'], r['rows'],
                int(r['run_bytes']) // int(r['element_bytes']), r['num_warps'], r['_f']['iters']))


# ---------------------------------------------------------------- modes

def eval_baseline(S, L):
    d = json.load(open(os.path.join(MODELS, 'final_v5.json'), encoding='utf-8'))
    print('=== 已交付 final_v5 复核(同一数据/同一过滤) ===')
    for key, dat in (('load|simt_only|inner', L), ('store|simt_only|inner', S),
                     ('load|simd|inner', L), ('store|simd|inner', S)):
        path = key.split('|')[1]; op = key.split('|')[0]
        ss = [r for r in dat if r['path'] == path and r['order'] == 'inner' and r['run_bytes'] >= 32]
        for r in ss: enrich(r)
        K = d[key]['K']
        tot = 0.0; n = 0
        for tag, cond in (('small', lambda r: r['rows'] <= K), ('large', lambda r: r['rows'] > K)):
            part = [r for r in ss if cond(r)]; m = d[key][tag]
            errs = [abs(predict_seg(r, m) - r['target_ns']) / r['target_ns'] * 100 for r in part]
            mm = float(np.mean(errs))
            tot += mm * len(part); n += len(part)
            print('  %-22s %-5s n=%-4d MAPE=%6.2f%%  (交付: %.2f%%)' % (key, tag, len(part), mm, m['mape']))
        print('  %-22s blend  n=%-4d MAPE=%6.2f%%' % (key, n, tot / n))


def fit_slot(S, L, op_path, scheme, cap, stop_tol=0.005, seed=8888):
    dat = L if op_path.startswith('load') else S
    ss = slot_points(dat, op_path.split('|')[1])
    segs = segments(ss, scheme)
    tot = 0.0; n = 0; rec = {}
    for name, part in segs:
        t0 = time.time()
        m = fit_segment(part, cap=cap, stop_tol=stop_tol, seed=seed)
        m['seg'] = name
        rec[name] = m
        tot += m['mape'] * len(part); n += len(part)
        print('  %-14s n=%-4d %6.2f%%  %2d项  %4.1fs  %s' % (
            name, len(part), m['mape'], len(m['keys']), time.time() - t0, m['keys']), flush=True)
    blend = tot / n
    print('  => %d 段  加权 %.2f%%  %s' % (len(segs), blend, 'PASS' if blend <= 10 else 'FAIL'), flush=True)
    return {'blend': blend, 'n': n, 'segments': rec}


def baseline_oos(S, L):
    """已交付 final_v5 固定系数在留出 dtype 上的误差(不重拟合, 供对比)."""
    d = json.load(open(os.path.join(MODELS, 'final_v5.json'), encoding='utf-8'))
    print('=== 已交付公式的 dtype 留一误差(固定系数, 不重拟合) ===')
    for slot in ('load|simt_only|inner', 'store|simt_only|inner'):
        dat = L if slot.startswith('load') else S
        ss = slot_points(dat, slot.split('|')[1])
        K = d[slot]['K']; m = d[slot]
        dtypes = sorted(set(r['dtype'] for r in ss))
        tot = 0.0; n = 0
        for hold in dtypes:
            test = [r for r in ss if r['dtype'] == hold]
            errs = []
            for r in test:
                seg = m['small'] if r['rows'] <= K else m['large']
                errs.append(abs(predict_seg(r, seg) - r['target_ns']) / r['target_ns'] * 100)
            mm = float(np.mean(errs)); tot += mm * len(test); n += len(test)
            print('    留出 %-9s n=%-4d OOS MAPE=%6.2f%%  (max %6.1f%%)' % (hold, len(test), mm, max(errs)))
        print('    => %s OOS 加权 %.2f%%' % (slot, tot / n))


def fit_phys(S, L, cap):
    """4 个槽位全部用物理因子池: SIMT=rows3_W2 分段, SIMD=交付版 K 分段(K=4/2)."""
    global POOL_OVERRIDE, SEG_K
    plan = (('load|simt_only', 'rows3_W2', SIMT_PHYS, L),
            ('store|simt_only', 'rows3_W2', SIMT_PHYS, S),
            ('load|simd', 'simdK', SIMD_PHYS, L),
            ('store|simd', 'simdK', SIMD_PHYS, S))
    out = {}
    for slot, scheme, pool, dat in plan:
        POOL_OVERRIDE = pool
        SEG_K = 4 if slot.startswith('load') else 2
        print('=== %s  scheme=%s  pool=%d 个物理因子 ===' % (slot, scheme, len(pool)), flush=True)
        ss = slot_points(dat, slot.split('|')[1])
        tot = 0.0; n = 0; rec = {}
        for name, part in segments(ss, scheme):
            m = fit_segment(part, cap=cap, seed=8888)
            m['seg'] = name
            rec[name] = m
            tot += m['mape'] * len(part); n += len(part)
            print('  %-14s n=%-4d %6.2f%%  %2d项' % (name, len(part), m['mape'], len(m['keys'])), flush=True)
            for k, c in zip(m['keys'], m['coef'][1:]):
                print('      %+12.5g * %-14s  # %s' % (c, k, LABELS.get(k, '?')), flush=True)
        blend = tot / n
        print('  => 加权 %.2f%%  %s' % (blend, 'PASS' if blend <= 10 else 'FAIL'), flush=True)
        out[slot] = {'blend': blend, 'n': n, 'segments': rec}
    POOL_OVERRIDE = None
    wp = sum(out[s]['blend'] * out[s]['n'] for s in out); wn = sum(out[s]['n'] for s in out)
    print('=== 四槽位总加权 %.2f%% ===' % (wp / wn))
    path = os.path.join(OUT, 'phys_cap%d.json' % cap)
    json.dump(out, open(path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('已存', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode')
    ap.add_argument('--scheme', default='rows3_W')
    ap.add_argument('--cap', type=int, default=12)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print('加载数据...', flush=True)
    S, L = load_all_v2()
    print('store %d 点, load %d 点' % (len(S), len(L)))
    if args.mode == 'eval-baseline':
        eval_baseline(S, L); return
    if args.mode == 'fit':
        out = {}
        for slot in ('load|simt_only', 'store|simt_only'):
            print('=== %s  scheme=%s cap=%d ===' % (slot, args.scheme, args.cap), flush=True)
            out[slot] = fit_slot(S, L, slot, args.scheme, args.cap)
        wp = (out['load|simt_only']['blend'] * out['load|simt_only']['n'] +
              out['store|simt_only']['blend'] * out['store|simt_only']['n'])
        wn = out['load|simt_only']['n'] + out['store|simt_only']['n']
        print('=== SIMT 加权 %.2f%% ===' % (wp / wn))
        path = os.path.join(OUT, 'simt_%s_cap%d.json' % (args.scheme, args.cap))
        json.dump(out, open(path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
        print('已存', path)
        return
    if args.mode == 'oos':
        oos_mode(S, L, args.scheme, args.cap); return
    if args.mode == 'fit-phys':
        fit_phys(S, L, args.cap); return
    if args.mode == 'baseline-oos':
        baseline_oos(S, L); return
    if args.mode == 'diagnose':
        diagnose(S, L, args.scheme, args.cap); return
    raise ValueError(args.mode)


if __name__ == '__main__':
    main()
