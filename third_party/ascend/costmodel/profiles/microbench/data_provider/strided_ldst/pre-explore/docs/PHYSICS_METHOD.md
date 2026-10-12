# 物理第一性拟合方法论（共享文档）

## 0. 环境

- 工作目录：`D:\work_wj\triton-ascend\third_party\ascend\costmodel\profiles\microbench\data_provider\strided_ldst\pre-explore`
- Python：`C:\Users\admin\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe`（有 numpy，无 scipy）
- 运行方式：`& '<python路径>' -c "<代码>"` 或写脚本文件后运行（PowerShell 环境，注意引号）
- **不要用 javascript 模板字符串里的反引号**（会破坏 JS）

## 1. 数据

`results/final/dataset_new_only_v5.csv`（10063 行；`valid=1` 的 9541 行用于拟合）

用现成的 loader：
```python
import sys; sys.path.insert(0, "scripts")
from fit_v9 import load, mape
rows = load("results/final/dataset_new_only_v5.csv")
rr = [r for r in rows if str(r.get("valid")) == "1"]
```

每行的字段（`load()` 会把列名简化）：
`op, path, dtype, eb(元素字节), W(num_warps), rank, n(元素数), ISB(内层步长字节),
 T(实测 ns/iter), L(128B line 数), span(地址跨度/128), pages, iters, cmd,
 rows(R), cols(C), row_pitch_bytes, row_span_bytes, row_lines, gap_lines,
 mean_warp_lines, mwl, ... 以及大量 f_* 原始特征`

**槽位（公式）划分**：
```python
def slot(rr0, sl, rk):
    if sl == "simd_load_wide":
        return [r for r in rr0 if r["path"]=="simd" and r["op"]=="load"
                and r["ISB"] <= 2.0*r["eb"] and int(r["rank"])==rk]
    if sl == "simd_load_gather":
        return [r for r in rr0 if r["path"]=="simd" and r["op"]=="load"
                and r["ISB"] > 2.0*r["eb"] and int(r["rank"])==rk]
    if sl == "simd_store":
        return [r for r in rr0 if r["path"]=="simd" and r["op"]=="store" and int(r["rank"])==rk]
    return [r for r in rr0 if r["path"]=="simt_only" and r["op"]==sl.split("_")[1]
            and int(r["rank"])==rk]
```

**重要**：rank-2 数据里 R=1 的形状（1x16/1x32/1x64）是 1D 形状，**应剔除**；R≥2 才是真 2D。

## 2. 拟合口径

- **加权最小二乘，权重 = 1/T**（等价于最小化相对误差 = MAPE）
- 报告 **in-sample MAPE**
- 允许 `max`/`min`；**不允许** log/指数/幂次
- 目标：项数尽量少，每一项有明确物理意义

## 3. 物理第一性方法（关键）

**步骤**：
1. 先画出 T 对各个物理量的**二维表**（固定其他量，只变一个），肉眼找结构
2. 提出物理假设（成本 = 哪些物理过程相加/相乘）
3. 用物理量构造特征并拟合，看 MAPE
4. 若残差有结构 → 回到 1；若无结构 → 完成

**构造惩罚项的常用手法**（用户明确许可）：
- **乘积项**：`冲突度 × 元素数`、`每块搬运量 × 块数`
- **hinge（max/min）**：`max(0, 阈值 − 量)` 表示"低于阈值才有的惩罚"
- **乘性因子**：`基准代价 × (1 + 冲突度)`
- **减轻惩罚的选项**：如果惩罚过强，引入"是否同一缓存/同一次取回"的指示或比例来抵扣，
  例如 `− 系数 × 可合并的元素数`

## 4. 已确认的物理发现（务必使用）

| 发现 | 内容 | 验证方式 |
|---|---|---|
| **MTE2 vs 128B** | **128B 是我自己设的假设，对 MTE2 不成立** | 用户指出 + 实测冲突粒度是 **256B** |
| **4K page** | 也是我设的假设（CPU 概念）；4KB 附近的现象是**双稳态**，不是干净阈值 | 两次测量同配置差 2.6 倍 |
| **BIU 命令数** | **每元素一条**（CAModel: `biu_send_count = n`，`biu_send_sizes={eb:n}`） | CAModel 白盒 |
| **同块冲突粒度** | **256 字节**（不是 128） | 扫描 B∈{16..4096} 得最优 B=256 |
| **冲突模型形式** | `T = 固定 + n·(c0 + c1·(w−1)) + ...`，w=最挤 256B 块的元素数 | gather_r1 拟合 |
| **wide 分支** | 内层 stride≤2 时编译器做**连续块搬运 + extract_slice**，extract 代价 ≈ 0 | CAModel + stride1/2 对照实测 |
| **SIMT 双瓶颈** | `T = max(k·num_warps, 访存代价)`，k≈2.18 ns/warp（W≥16 时 T 与元素数无关） | 受控 n×W 扫描 |
| **C<32 跳变** | 一行装不满一个 warp（32 线程）时 T 跳升 1.5–2 倍 | 44 个跳变点分析 |

## 5. 已完成的公式（参考范例）

### `simd_load_wide_r1`（2 项，MAPE 4.11%）
```
span_bytes = (n − 1) × stride × eb + eb      # 连续搬运的字节数
T = 65.34 + 0.008652 × span_bytes
```
- 65.34 ns = MTE2 固定启动/首字节延迟
- 0.008652 ns/B ≈ 115 GB/s 等效带宽
- **数据退化警告**：这 32 个点的 stride 全是 2，所以 span 与 n·eb 共线，无法分离
- 实测 T(s2)/T(s1) = 0.96–1.13 → **extract_slice 代价 ≈ 0**

### `simd_load_gather_r1`（6 项，MAPE 20.18%）
```
w = 最挤的 256B 块里的元素数（由 ISB 和 n 算出）
L256 = 触及的 256B 块数
T = 96.03 + 4.241·(w−1) + 0.718·n + 0.0557·n·(w−1) + 0.133·L256 + 0.655·eb − 0.00983·ISB
```
- 物理：每元素一条 BIU 命令，代价被同块拥挤度抬高
- **对照：贪心 78 项 = 12.14%**，所以物理形式已经很有竞争力

## 6. 输出格式

每个公式产出一个 JSON 片段（写入 `results/final/PHYS_<slot>.json`）：

```json
{
  "slot_rk": {
    "n": 点数,
    "mape": 误差,
    "n_terms": 项数,
    "formula": "T = ... 的可读文本",
    "terms": [{"name": "...", "coef": 1.23, "meaning": "物理意义"}],
    "derived": ["中间量的定义，例如 span_bytes = ..."],
    "notes": "物理洞察、数据局限、与贪心的对比"
  }
}
```

同时打印一份人类可读的总结到 stdout。


---

# 7. 第二轮：真实算子范围（v5）—— 用户已授权

用户明确授权：**物理公式误差太大时，可以剔除"真实 AI 算子中不常见"或"跨度太大导致测量不稳定"的数据**，
前提是**报告里必须写清楚剔除了什么、为什么、剩多少点**。

## 7.1 目标

**平均 in-sample MAPE < 10%**，且公式必须保持**物理可解释**（不许堆砌几十个胡乱相乘的因子）。

## 7.2 建议的"真实算子范围"门槛（需各自论证，不必全用）

真实 Triton 算子（matmul / softmax / layernorm / attention / elementwise）的典型特征：

| 维度 | 典型范围 | 数据中占比 | 建议门槛 |
|---|---|---|---|
| 元素数 n | 128–4096 | n<32 占 23% | `n >= 32`（小于一个 warp 的 tile 不是真实计算 tile） |
| 内层步长 ISB | 连续或 ≤ 128B | ISB>128 占 5% | `ISB <= 128` |
| num_warps | 4, 8（偶尔 1/2/16） | W>16 占 7% | `W <= 16` |
| dtype | int8/f16/f32 为主，int64 罕见 | int64 占 22% | `eb <= 4`（可选） |
| rank-2 内维 C | ≥ 8（要能向量化） | C<4 占 6% | `C >= 4` |
| rank-2 行数 R | ≥ 2 | R=1 占 13% | `R >= 2`（R=1 是伪 2D） |

全部门槛叠加后保留 **4632 / 9541 = 48.5%**。

**注意**：这些门槛是"真实算子范围"的主张，必须逐条给出理由，并报告剔除前后的误差对比。
也可以只剔除**真正造成大部分误差的那个子集**（用残差分组分析找出），这样保留的数据更多。

## 7.3 方法要求

1. **先用残差分组分析找出误差集中在哪些子集**（按 n / ISB / eb / W / C / R / dtype / campaign 分组看 MAPE 和 bias）
2. 只剔除**有物理理由的**子集：要么"真实算子不这样用"，要么"测量本身不稳定"（例如 `target_spread` 大、双稳态）
3. 在保留集上重新拟合**物理第一性公式**
4. 如果加一个门槛后误差没有明显改善，**不要加**（避免无谓地丢数据）
5. 报告要给出：门槛列表 + 每条的物理理由 + 保留点数 + 保留率 + MAPE

## 7.4 输出格式（在原有基础上增加）

```json
{
  "slot_rk": {
    "...": "原有字段",
    "scope": {
      "filters": ["n >= 32：小于一个 warp 的 tile 不是真实计算 tile", "..."],
      "n_all": 9541, "n_kept": 4632, "keep_ratio": 0.485,
      "removed_reason_breakdown": {"n<32": 2202, "...": 0}
    }
  }
}
```
