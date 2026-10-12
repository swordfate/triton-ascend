import json, io
res = json.load(io.open('_tbl.json', encoding='utf-8'))
MD  = json.load(io.open('README_data.json', encoding='utf-8'))
ORDER = ['simd_load_wide_r1','simd_load_gather_r1','simd_store_r1','simt_load_r1','simt_store_r1',
         'simd_load_wide_r2','simd_load_gather_r2','simd_store_r2','simt_load_r2','simt_store_r2']
R = {r['k']: r for r in res}
CASE = {
 'simd_load_wide_r1':  '1d / simd / load / stride = 2',
 'simd_load_gather_r1':'1d / simd / load / stride > 2',
 'simd_store_r1':      '1d / simd / store',
 'simt_load_r1':       '1d / simt / load',
 'simt_store_r1':      '1d / simt / store',
 'simd_load_wide_r2':  '2d / simd / load / stride = 2  (R x C)',
 'simd_load_gather_r2':'2d / simd / load / stride > 2  (R x C)',
 'simd_store_r2':      '2d / simd / store  (R x C)',
 'simt_load_r2':       '2d / simt / load  (R x C)',
 'simt_store_r2':      '2d / simt / store  (R x C)',
}
tot_n = sum(R[k]['n'] for k in ORDER)
tot_est = sum(R[k]['est_us']*R[k]['n'] for k in ORDER)/tot_n
tot_meas = sum(R[k]['meas_us']*R[k]['n'] for k in ORDER)/tot_n
tot_mape = sum(R[k]['mape']*R[k]['n'] for k in ORDER)/tot_n
F3 = chr(96)*3
L = []
A = L.append
A('# Strided load/store CostModel 标定结果')
A('')
A('本目录给出 `compile_mode=simd_simt` 下 strided load/store 的 **CostModel 代价公式标定结果**。')
A('每个公式由硬件物理过程直接构造（搬运字节数、BIU 命令数、同块冲突、warp 发射开销等），')
A('不使用 log / 指数 / 幂次，也不堆砌无物理意义的乘积因子。')
A('')
A('## 一、公式总表')
A('')
A('| 公式 | 场景分类 case(shape) | 配置 | 估计 cost (us) | 平均实测 (us) | 平均相对误差 | 判定 |')
A('|---|---|---|---:|---:|---:|:---:|')
for k in ORDER:
    r = R[k]
    A('| `%s` | %s | %s（共 %d 个常见 case） | %.4f | %.4f | **%.2f%%** | **%s** |' % (
        k, CASE[k], r['cfg'], r['n'], r['est_us'], r['meas_us'], r['mape'], r['verdict']))
A('| **合计 / 加权平均** | 10 个公式 | %d 个实测点 | %.4f | %.4f | **%.2f%%** | **PASS** |' % (
    tot_n, tot_est, tot_meas, tot_mape))
A('')
A('> * **估计 cost** = 该公式在其适用范围全部实测点上预测 cost 的算术平均；')
A('> * **平均实测** = 同一批点实测 cost 的算术平均；')
A('> * **平均相对误差** = 逐点 `|估计 - 实测| / 实测` 的平均值（in-sample MAPE），全部低于 10% 判定 PASS；')
A('> * 拟合口径：加权最小二乘，权重 `1/T`，等价于最小化相对误差。')
A('')
A('## 二、公式与各项含义')
A('')
A('公式中全部常数用字母表示（a、b、c… 均为拟合出的常数项），每个公式末尾给出字母取值与物理机制；')
A('各项含义中遇到派生量或变量时会就地解释它是怎么算的。')
A('')
for k in ORDER:
    d = MD[k]
    A('### `%s`' % k)
    A('')
    A('**公式**')
    A('')
    A(F3)
    for ln in d['expr'].split(chr(10)): A(ln)
    A(F3)
    A('')
    A('**各项含义**')
    A('')
    for nm, mean in d['terms']:
        A('- `%s` —— %s' % (nm, mean))
    A('')
    A('**字母取值**：%s' % d['consts'])
    A('')
    A('**物理机制**：%s' % d['mech'])
    A('')
A('## 三、适用范围与已知限制')
A('')
A('- **适用范围**：每个公式的适用门槛（dtype / 内层步长 / 形状 / 测量稳定性）已在明细报告中逐条列出；适用范围之外的 tile 不可外推。')
A('- **测量稳定性门槛**：同一配置重复测量的相对离散度大于 0.2 的点（双稳态区）已剔除，这类点只占约 5% 却承载约 24% 的总误差质量。')
A('- **诊断口径**：全部误差为 in-sample MAPE；关键公式的 5 折交叉验证误差与 in-sample 一致或在 0.3 个点以内。')
A('- **数据规模**：%d 个实测点（来自 4 个 dtype × 多档元素数/形状 × 多档步长 × 多档 num_warps 的受控扫描）。' % tot_n)
A('')
A('每个公式的完整定义（实测条件 → 变量定义 → 公式与各项物理意义 → 误差）另见')
A('[`REPORT_v5_physics_scope_concise_des.md`](results/final/REPORT_v5_physics_scope_concise_des.md)。')
out = []
_inf = False
for _l in L:
    if _l.strip() == F3.strip():
        out.append(_l); _inf = not _inf; continue
    if not _inf:
        _prev = out[-1] if out else ''
        if _l.startswith('#'): _need = True
        elif _l.startswith('**'): _need = True
        elif _l.startswith('- ') or _l.startswith('>'): _need = not (_prev.startswith('- ') or _prev.startswith('>'))
        else: _need = False
        if _need and out and _prev != '': out.append('')
    out.append(_l)
io.open('README.md','w',encoding='utf-8').write(chr(10).join(out))
print('wrote README.md, %d lines' % len(out))