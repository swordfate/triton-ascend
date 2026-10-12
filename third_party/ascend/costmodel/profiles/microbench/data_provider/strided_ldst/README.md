# strided load/store CostModel 标定 —— 复现指南

> 目录：`third_party/ascend/costmodel/profiles/microbench/data_provider/strided_ldst/`
> 读者：之后接手这份标定工作的人。照着本文件，可以从零把全部 strided load/store 测量与拟合复现出来。
> 全部中间产物在 `pre-explore/` 下；本文件在顶层。

---

## 〇、两个提交：本分支怎么用（先读这一节）

> **本分支 `feature/strided-template-calibration-v2` 上有两个提交，分工不同。先确认你要的是哪一个。**

| 提交 | 哈希 | 内容 | 适合谁 |
|---|---|---|---|
| **提交 1** | `5539c4831` | **全量标定工作台**：37 个 campaign 目录（各 campaign 自己的 `raw/` 副本按 `.gitignore` 未提交；**能用 git 追溯的原始数据**是 `results/final/raw/` 的 231 个合并 JSON，29 MB）、全部 623 个脚本文件、`results/final/_*` 的 44 个拟合中间状态 | **想复现全流程的人**（`raw -> dataset -> 拟合 -> 报告`） |
| **提交 2**（本提交） | 本提交（在 `5539c4831` 之上） | **只精简数据**：删掉 37 个 campaign 目录 + `results/final/raw/`(231 个文件) + `results/final/_*`(44 个) + `fit_v5/`+`fit_newonly/`(6 个)，共 **1678 个文件**；`results/` 由 **2381 文件 / 168.3 MB 减到 111 文件 / 38.2 MB**；**脚本一个都没删** | **想直接用最终数据集拟合 / 校验公式的人** |

提交 2 的删减口径说明（三种数都对，看你数哪一种）：

| 口径 | 文件数 | 大小 |
|---|---:|---:|
| 提交 1 **工作区**实测（含 `.gitignore` 忽略、从未提交的 592 个文件） | 2381 | 168.3 MB |
| 提交 1 **git 跟踪**（`git ls-tree -r HEAD -- pre-explore/results`） | 1789 | 138.6 MB |
| 提交 2（本提交） | **111** | **38.2 MB** |

也就是说：提交 1 里**能用 git 追溯的原始数据**是 `results/final/raw/` 的 231 个 JSON（29 MB）与各 campaign 自己已提交的部分；
各代 campaign 的 `raw/` 副本按 `.gitignore` 设计未提交。提交 2 把这些原始数据整体移出本提交，只保留最终数据集、模型、报告与全部脚本。

**怎么选**：

- **想从板上 raw JSON 重跑特征化、拟合、报告** -> **用提交 1**：`git checkout 5539c4831`，然后照本文件 §四 走；
  特别是 `run_final_pipeline.py --build` 的前三步（`expand2d/raw`、`expand2d_b/raw`、`_build/`）**只在提交 1 里有数据**。
- **只想用最终数据集重拟 / 校验公式 / 读结论** -> **用提交 2（本提交）**：
  `pre-explore/results/final/dataset_new_only_v5.csv`（10063 行，`valid=1` 的 9541 行是拟合集）与 16 个 `PHYS*.json` 都在。
- **想读最终结论** -> [`pre-explore/README.md`](pre-explore/README.md)（10 式总表 + PASS/FAIL）与
  [`pre-explore/results/final/REPORT_v5_physics_scope_concise_des.md`](pre-explore/results/final/REPORT_v5_physics_scope_concise_des.md)。

**为什么最终放弃数据驱动的贪心拟合、改用物理第一性建模**（三条路线实测对比 + 6 个可写进 CostModel 的硬件量）：
见 [`pre-explore/docs/WHY-PHYSICS-OVER-GREEDY.md`](pre-explore/docs/WHY-PHYSICS-OVER-GREEDY.md)。

> **阅读约定**：本文件下面凡出现 `results/<campaign>/...` 形式的路径（campaign 名如 `dtype_calibration_v1`、
> `dtype_calibration_v2_rotate512`、`dtype_calibration_v4_final`、`expand2d`、`expand2d_b`、`v4raw`、`v5raw`、`v8raw`、
> `v9raw`、`model_store_v1`、`model_template_stride_v1/v2`、`model_v2/v3_numwarps/v4_numwarps/v4_simple/v7_simple`、
> `dir23`、`wide12`、`_build`、`merged` 等），以及 `results/final/raw/` 与 `results/final/_*`，
> **都只在提交 1 里存在（见提交 1）**；提交 2 里只剩 `results/final/` 的数据集 / 模型 / 报告和 `pre-explore/scripts/` 的全部脚本。
> 目录级清单见 [`pre-explore/results/README.md`](pre-explore/results/README.md)（已按提交 2 重写）。

---

## 一、这是什么、结论是什么

本目录为 **triton-ascend CostModel** 标定 `compile_mode=simd_simt` 下 **strided load/store 的代价公式**：
覆盖 1D / 2D 两种访问秩、`simd` / `simt_only` 两条 lowering 路径、load / store 两种操作、
int8 / float16 / float32 / int64 四种 dtype，多档元素数 / 形状、多档内层步长 stride 与 num_warps。

全部实测在板卡 `ascend-950pr-63` 上用 **Event 斜率法（rotate-loop）** 完成；拟合用**加权最小二乘（权重 `1/T`，
等价于最小化相对误差即 MAPE）**；公式**全部由硬件物理过程直接构造**（搬运字节数、BIU 命令数、同块冲突、
warp 发射开销……），**不使用 log / 指数 / 幂次**，也不堆砌无物理意义的乘积因子。

**最终结论：10 个公式在各自适用范围内全部 in-sample MAPE < 10%，在 5118 个实测点上加权平均 7.22%，判定全部 PASS。**

| 公式 | 场景 | 点数 | MAPE |
|---|---|---:|---:|
| `simd_load_wide_r1` | 1d / simd / load / stride = 2 | 21 | 1.22% |
| `simd_load_gather_r1` | 1d / simd / load / stride > 2 | 136 | 8.70% |
| `simd_store_r1` | 1d / simd / store | 189 | 5.33% |
| `simt_load_r1` | 1d / simt / load | 506 | 4.95% |
| `simt_store_r1` | 1d / simt / store | 962 | 6.27% |
| `simd_load_wide_r2` | 2d / simd / load / stride = 2 | 120 | 4.48% |
| `simd_load_gather_r2` | 2d / simd / load / stride > 2 | 438 | 8.44% |
| `simd_store_r2` | 2d / simd / store | 325 | 8.12% |
| `simt_load_r2` | 2d / simt / load | 1211 | 6.34% |
| `simt_store_r2` | 2d / simt / store | 1210 | 9.64% |
| **合计 / 加权平均** | 10 个公式 | **5118** | **7.22%** |

完整的公式总表、每一项的物理含义、字母取值与物理机制见
[`pre-explore/README.md`](pre-explore/README.md)；
每个公式的逐条完整定义（实测条件 → 变量定义 → 公式与各项物理意义 → 误差 → 适用范围门槛）见
[`pre-explore/results/final/REPORT_v5_physics_scope_concise_des.md`](pre-explore/results/final/REPORT_v5_physics_scope_concise_des.md)。

本标定已经被 C++ / profile 消费（`feature/strided-load-store-costmodel-v4` 分支），
route-level 的 A/B 验证报告见 [`pre-explore/docs/ROUTE-LEVEL-VALIDATION-v8.md`](pre-explore/docs/ROUTE-LEVEL-VALIDATION-v8.md)。

---

## 二、目录结构

```
strided_ldst/
├── README.md                        ← 本文件：顶层复现指南
└── pre-explore/                     ← 整个标定工作的根目录
    ├── README.md                    ← 最终物理模型报告（10 个公式总表 + 逐项物理含义）
    ├── README_data.json             ← 上面 README 的数据源；报告生成器读它
    ├── .gitignore                   ← 忽略 triton_cache*/、__pycache__/、results/dtype_calibration_v*/raw/ 等
    ├── docs/
    │   ├── PHYSICS_METHOD.md        ← 物理第一性拟合方法论 + 已确认的物理发现 + 真实算子范围门槛
    │   ├── ROUTE-LEVEL-VALIDATION-v8.md  ← v8 SIMT load 公式的 route-level A/B 验证
    │   ├── COMMIT_MSG.txt           ← 最终 campaign 的提交说明（网格规模、脚本、修过的坑）
    │   ├── handoff/                 ← 3 份给接手人的接力提示词 NEXT-AI-*.md（含环境与口径约定）
    │   └── agent-logs/              ← 11 份子代理推导过程记录 p3_*.txt（拟合怎么一步步收敛的）
    ├── scripts/                     ← 现存 106 个脚本 = 顶层 49 .py + 47 .sh，camodel/ 3 .py + 7 .sh
    │                                  （清理前是 622 个；被删的 518 个是子代理推导的中间产物，
    │                                   推导记录见 docs/agent-logs/ 与 git 历史，索引见 scripts/INDEX.md）
    │   ├── board_env.sh             ★ 板上环境包装器：所有板上命令都套它
    │   ├── dtype_stride_measure_probe.py   ★ 1D 探针（唯一被最终 1D 数据使用的探针）
    │   ├── stride2d_measure_probe.py       ★ 2D 探针（唯一被最终 2D 数据使用的探针）
    │   ├── template_stride_common.py       ← 探针共享库：Event 计时、斜率、四道门、asm 落盘
    │   ├── stride_feature_reference.py     ← 全部特征的唯一权威定义（C++ 移植照它写）
    │   ├── build_dtype_dataset.py          ← 1D raw -> dataset.csv + dataset_meta.json
    │   ├── build_2d_dataset.py             ← 2D raw -> dataset_2d.csv + dataset_2d_meta.json
    │   ├── build_unified_dataset.py        ← 把 1D/2D、历次 campaign 统一特征化成一个 CSV
    │   ├── fit_v9.py / fit_v11_stepwise.py / run_final_pipeline.py  ← 拟合骨架（NNLS / 前向选择 / 贪心 ≤10 项）
    │   ├── _eval.py / _tbl.py / _readme4.py        ← 评估 -> 总表 -> README.md 生成链
    │   ├── gen_report_concise.py           ← 生成 REPORT_v5_physics_scope_concise_des.md
    │   ├── run_*.sh                        ← campaign 驱动（定义网格、窗口、reps/attempts），现存 30 个
    │   ├── board_*.sh                      ← 板端辅助，现存 17 个：board_env.sh / board_env_latest.sh
    │   │                                     （环境包装，★所有板端调用的入口）+ 15 个定向单点实验
    │   │                                     （原 45 个运维脚本里的状态/等待/watchdog/asm 巡检已删除）
    │   └── camodel/                        ← CAModel 白盒（msopprof simulator）探针与驱动
    └── results/                     ← 约 2400 个文件 / 约 170 MB，历次 campaign 的原始测量数据
        ├── dtype_calibration_v1/raw/        1D dtype×stride 首轮（自动 64 MB 窗口）
        ├── dtype_calibration_v2_rotate512/  1D 对齐历史 512 MB 窗口 + stepiso 窗口隔离实验
        ├── dtype_calibration_v4_final/raw/  最终 1D + 2D 大网格（含 num_warps sweep）
        ├── expand2d/ , expand2d_b/          2D 加密 campaign（v3_2d / v4_2d 两个来源）
        ├── model_template_stride_v1/v2/     template 路径（`--exclude-template`，不参与最终 10 公式）
        ├── model_store_v1/ , model_v2/ , model_v4_numwarps/  更早的历史 1D 数据（被并入 unified）
        ├── final/                           ★ 合并视图 + 数据集 + 公式 JSON + 报告（最重要的目录）
        └── merged/                          合并视图（与 final 重叠，是镜像，统计时不要重复计数）
```

> **提交 2 提示**：上面这棵树里，除 `final/` 之外的目录（各 campaign、`model_*`、`_build`、`merged` 等）**只在提交 1 里存在（见提交 1）**。
> 提交 2 的 `pre-explore/results/` 只剩 `final/` 与 12 个根目录文件，共 111 个文件。

`results/final/` 里最关键的几件东西：

| 文件 | 作用 |
|---|---|
| `raw/`（231 个 json） | **全部 campaign 的原始测量合并视图**（.gitignore 只提交这一份，各 campaign 自己的 raw/ 被忽略）。**提交 2 已删除此目录 —— 见提交 1** |
| `dataset.csv` / `dataset_2d.csv` | 1D（2560 行）/ 2D（3432 行）数据集 |
| `dataset_unified_v5.csv` | 统一特征化后的全集（12040 行） |
| `dataset_new_only_v5.csv` | **拟合实际使用的数据集**（10063 行，其中 `valid=1` 的 9541 行） |
| `PHYS3_<slot>.json` | **10 个公式的最终冻结定义**（formula / terms / derived / consts / mape / n_terms / scope） |
| `REPORT_v5_physics_scope_concise_des.md` | 逐公式明细报告 |

---

## 三、环境与前置

### 3.1 本地（Windows）

- 操作系统：Windows + PowerShell。
- Python（本工作唯一使用的解释器）：

```
C:/Users/admin/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/python.exe
```

  这个环境**有 numpy，没有 scipy** —— 所有拟合代码必须是纯 numpy 实现（`scripts/fit_v9.py` 里的
  `nnls()` 就是手写的 Lawson-Hanson NNLS，不要改成 `scipy.optimize.nnls`）。
- 运行脚本：

```powershell
& "C:/Users/admin/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/python.exe" scripts/_eval.py
```

- 用 run_code 驱动时：**绝不要用 JS 模板字符串里的反引号**（会破坏 JS 源码）。写文件统一 UTF-8。

### 3.2 板上（`ascend-950pr-63`）

| 项 | 值 |
|---|---|
| 登录 | `ssh ascend-950pr-63` |
| 用户 | `c00946898` |
| 工作目录 | `~/strided-template-calibration-v2/`（内容 = 本地 `pre-explore/`） |
| 核数 | 256 |
| conda 环境 | `wj_autoscope` |

环境必须这样进（**`scripts/board_env.sh` 已经封装好，正常情况下不需要手敲**）：

```bash
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
ulimit -n 1048576
export ASCEND_RT_VISIBLE_DEVICES=0
```

> **不要** `export PYTHONPATH=$HOME/sb64_latest_env`。必须用可 launch 的 `wj_autoscope` 安装包，
> 并保留 CANN `set_env.sh` 导出的 `PYTHONPATH`；否则 acl / `is_compile_on_910_95` 检测会失败，
> torch_npu 自定义 kernel 起不来。`board_env.sh` 特意不去 unset `PYTHONPATH`，就是为了这个。
>
> 另一个容易踩的点：CANN 的 `set_env.sh` 会访问未定义变量，所以 `board_env.sh` 里 source 期间
> 是 `set +u`（先关掉 nounset 再开回来）。

### 3.3 代码与数据的同步

脚本在本地编辑，**用 `scp` 送到板上**；结果 JSON 在板上生成，**用 `scp` 拉回本地**。
（早期也用 `rsync` 同步过 `scripts/`，效果等价。）

```powershell
# 上行：把 scripts 与（首次需要的）results 目录结构送上去
scp -r ./pre-explore/scripts ascend-950pr-63:~/strided-template-calibration-v2/

# 上行：只送改动过的单个脚本
scp ./pre-explore/scripts/run_final_campaign.sh ascend-950pr-63:~/strided-template-calibration-v2/scripts/

# 下行：把一轮 campaign 的 raw 结果拉回来
#   （注意：results/dtype_calibration_v*/raw/ 只在提交 1 里有 —— 见提交 1；提交 2 里这些目录已删）
scp -r "ascend-950pr-63:~/strided-template-calibration-v2/results/dtype_calibration_v4_final/raw/*.json" ./pre-explore/results/dtype_calibration_v4_final/raw/
```

约定：

- **板上只跑测量，本地只跑构建/拟合/报告**。板上不跑 numpy 拟合，本地不连板。
- 每个 campaign 在板上写到独立目录 `results/<campaign>/raw/`，本地同名目录不要混用。
  **提交 2 里这些目录已全部删除 —— 见提交 1**（提交 2 只保留 `results/final/` 的最终数据集、模型与报告）。
- `TRITON_CACHE_DIR` 由驱动脚本设置（例如 `$PRE/triton_cache_v4`）并在板上保留，
  这样重跑同一网格时编译可复用；它被 `.gitignore` 忽略，不要往版本库里带。

---

## 四、怎么跑（完整复现流程）

### 4.0 全景

```
[板上]  bash scripts/board_env.sh python scripts/<探针>.py --op ... --dtype ... --blocks ... --strides ... --out ...
          ↓  每 case 约 5-9 秒，一个 campaign 几百到上千 case，多个 campaign 并发
[板上]  results/<campaign>/raw/*.json          （每个 json = 一个 (op, path, dtype, W, 网格) 批次）
#          ^ 这些 campaign 目录只在提交 1 里有（见提交 1）
          ↓  scp 拉回本地同名路径
[本地]  build_dtype_dataset.py / build_2d_dataset.py     raw -> dataset.csv（中位数聚合 + 质量标记）
          ↓
[本地]  build_unified_dataset.py                          统一特征化 -> dataset_unified_v5.csv
          ↓                                                      -> dataset_new_only_v5.csv（拟合用）
[本地]  fit（run_final_pipeline.py 或各 PHYS3 脚本）       逐点 NNLS（权重 1/T）-> PHYS3_<slot>.json
          ↓
[本地]  _eval.py -> _tbl.py -> _readme4.py                评估 -> 总表 -> pre-explore/README.md
[本地]  gen_report_concise.py                             -> results/final/REPORT_v5_physics_scope_concise_des.md
```

### 4.1 第一步：板上测量（典型命令）

**所有板上命令都套 `scripts/board_env.sh`**，它负责 source 环境、`conda activate wj_autoscope`、
`ulimit -n`、设 `ASCEND_RT_VISIBLE_DEVICES`，然后 `exec` 后面的命令。

#### 1D 探针（`scripts/dtype_stride_measure_probe.py`）

```bash
cd ~/strided-template-calibration-v2

bash scripts/board_env.sh python scripts/dtype_stride_measure_probe.py \
  --op load \
  --path simd \
  --dtype float32 \
  --blocks 4 8 16 32 64 128 256 512 \
  --strides 2 3 4 6 8 16 32 64 128 256 \
  --num-warps 1 \
  --rotate-step 8192 --rotate-mask 16383 \
  --reps 3 --max-attempts 3 \
  --asm-dir results/dtype_calibration_v4_final/measure_asm \
  --out     results/dtype_calibration_v4_final/raw/load_simd_float32_W1_core.json
#   ^ results/dtype_calibration_v4_final/ 只在提交 1 里有（见提交 1）
```

参数逐个说清：

| 参数 | 含义 |
|---|---|
| `--op load|store` | 被测操作 |
| `--path simd|simt_only|template` | 强制 lowering 路径。`simd` 对应 `compile_mode=simd`（MTE 结构化搬运 / 逐元素 gather）；`simt_only` 对应 `compile_mode=simt_only`（SIMT_LDG/STG + 128B DCache）；`template` 对应 `compile_mode=simd_simt_template`（调用 `triton_stride_load/store`，**不参与最终 10 公式**） |
| `--dtype int8|float16|bfloat16|float32|int32|int64` | 元素类型；最终公式只用 int8/float16/float32/int64 |
| `--blocks` | 1D tile 的元素数 n（探针内部 `BLOCK`） |
| `--strides` | 元素域内层步长（相邻两个被访问元素相隔的元素个数）；**不允许 1** |
| `--num-warps` | SIMT 的 warp 数；`simd` 路径恒为 1 |
| `--rotate-step` / `--rotate-mask` | rotate 窗口：第 i 次迭代访问 `(i & mask) * step` 起的一段。显式传就是历史口径 512 MB 窗口；不传则按 **64 MB 字节预算自动**（见 §5.1） |
| `--iters 2000 8000` | 每个点用两个迭代数，取最小 Event 后做**一元线性回归的斜率**（ns/iter） |
| `--reps` | 每个 iters 点重复几次，取最小 Event（Event 只会被竞争抬高） |
| `--max-attempts` | 整个 case 最多尝试几次；只有通过四道门的 attempt 才算 valid，取其中 target 最小的一次 |
| `--asm-dir` | 把编译产物（含 TTAdapter）落盘，用于事后确认 lowering 真的走对了 |
| `--out` | 本批次的 raw JSON 输出路径 |

#### 2D 探针（`scripts/stride2d_measure_probe.py`）

地址模型是 `addr(r, c) = r * OS + c * IS`，`r in [0,R)`、`c in [0,C)`。
校准使用的是 **row-stride 参数化**：`--row-strides rs` + `--inner-strides cs` 给出 `OS = rs * C * cs`，
即从一个稠密的 `(R*rs) x (C*cs)` 行主序矩阵里按 `rs` 行、`cs` 列抽样。

```bash
bash scripts/board_env.sh python scripts/stride2d_measure_probe.py \
  --op load \
  --path simd \
  --dtype float32 \
  --shapes 4x4 4x8 4x16 8x8 8x16 16x16 \
  --row-strides 2 3 4 6 8 16 \
  --inner-strides 2 3 4 8 16 64 \
  --num-warps 1 \
  --rotate-step 8192 --rotate-mask 16383 \
  --reps 3 --max-attempts 3 \
  --asm-dir results/dtype_calibration_v4_final/measure_asm \
  --out     results/dtype_calibration_v4_final/raw/2d_load_simd_float32_W1_core.json
#   ^ results/dtype_calibration_v4_final/ 只在提交 1 里有（见提交 1）
```

额外注意：

- `--include-contig` 才会测完全连续的锚点；**默认不测**，保证数据集里永远不混入连续访问（另一套代价律）。
- 2D 的 **load** 探针把 `tl.sum(acc)` 写进 **float32** 的 sink buffer，不写回元素 dtype 的 buffer ——
  这是修过的一个真 bug（见 §6）。

#### 用驱动脚本跑一整个 campaign

单条命令只测一个批次。整轮网格由 `run_*.sh` 驱动，例如最终轮：

```bash
PASS=core1d     bash scripts/run_final_campaign.sh   # 1D 全网格，SIMD W=1 + SIMT W=4
PASS=wsweep1d   bash scripts/run_final_campaign.sh   # 1D 缩减网格，SIMT W in 1..64
PASS=core2d     bash scripts/run_final_campaign.sh   # 2D 主网格
PASS=wsweep2d   bash scripts/run_final_campaign.sh   # 2D 缩减网格，SIMT W in 1..64
PASS=n1         bash scripts/run_final_campaign.sh   # 2D n x 1 单列族，全部 7 档 W
```

每个驱动脚本里都定义了网格、`WIN="--rotate-step 8192 --rotate-mask 16383"`、`REPS`、`ATT`，
并把输出文件名写成 `"$OUT/raw/${op}_${path}_${dtype}_W${warps}_${tag}.json"` ——
**文件名自带 block/stride 签名**，这是刻意设计的（见 §5.6）。

### 4.2 每次测量要多久

**单 case 典型 5–9 秒**（含该 case 的编译、warmup、correctness、多次 attempt）。
下面是从 raw JSON 的 `elapsed_seconds` 字段统计出来的真实数字：

| campaign | raw 文件 | case 数 | 板上总耗时 | 平均 s/case |
|---|---:|---:|---:|---:|
| `dtype_calibration_v4_final` | 21 | 870 | 1.9 h | 7.9 |
| `expand2d` | 20 | 2400 | 4.5 h | 6.7 |
| `expand2d_b` | 22 | 1952 | 3.0 h | 5.5 |
| `dtype_calibration_v2_rotate512` | 83 | 642 | 0.9 h | 4.9 |
| `dtype_calibration_v1` | 48 | 402 | 0.4 h | 3.4 |
| `model_template_stride_v2` | 4 | 1480 | 2.3 h | 5.5 |
| `model_store_v1` | 3 | 744 | 0.05 h | 0.2 |
| `final`（合并镜像，**不要重复计数**） | 231 | 7512 | 12.0 h | 5.7 |
| `merged`（合并镜像，**不要重复计数**） | 181 | 5200 | 8.3 h | 5.8 |

> **提交 2 提示**：上表里的 campaign 目录（`dtype_calibration_*`、`expand2d*`、`model_*`、`merged`）**都只在提交 1 里存在（见提交 1）**；
> `final` 那一行对应的是 `results/final/raw/`（231 个 JSON，也已在本提交删除）。本表用于估算板上重跑时间，仍然有效。

单文件例子（**见提交 1**）：`results/dtype_calibration_v4_final/raw/2d_load_simd_float16_W1_core.json`
= **180 个 case / 1801.9 秒 / 10.0 s per case**。

范围：最快约 **0.2 s/case**（`model_store_v1` 的小 case 批），最慢约 **43 s/case**（含首次编译的长尾）。
**规划用 5-9 s/case 估总时间是安全的**：一个 2000 case 的 campaign 大约 3-5 小时。

### 4.3 结果落在哪里

```
results/<campaign>/raw/<op>_<path>_<dtype>_W<warps>_<tag>.json        # 1D
results/<campaign>/raw/2d_<op>_<path>_<dtype>_W<warps>_<tag>.json     # 2D
results/<campaign>/measure_asm/<label>.<key>                          # 对应的编译产物
```

> **提交 2 提示**：以上 `results/<campaign>/...` 布局**只在提交 1 里存在（见提交 1）**。
> 提交 2 里 `results/` 只有 `final/` 的数据集 / 模型 / 报告，共 111 个文件。

每个 JSON 的顶层结构：

```json
{
  "kind": "dtype_stride_measure" | "stride2d_measure",
  "op": "load", "path": "simd", "dtype": "float32", "element_bytes": 4,
  "cases": [
    { "block": 256, "stride": 4, "num_warps": 1,
      "target_ns": 512.0,
      "target_kind": "rotate_loop_min_event_slope_ns_per_iter",
      "valid": true, "correctness_ok": true, "contains_template_call": false,
      "attempts": [ { "rows": [...], "target_ns": ..., "spread": 0.03,
                      "witness_ok": true, "frequency_ok": true,
                      "post_freq_mhz": 1800, "stable_ok": true } ] }
  ],
  "buffer": { "step": 8192, "mask": 16383, "total": ..., "window_bytes": ... },
  "frequency_check": { "inflight_freq_mhz": ..., "reference_witness_ns": ... },
  "elapsed_seconds": 1801.9
}
```

`attempts` 一定要保留 —— 数据集里的 `target_spread` 质量标记就是从它算出来的。

### 4.4 第二步：把结果拉回本地

```powershell
# 注意：results/dtype_calibration_v4_final/ 只在提交 1 里有（见提交 1）
scp -r "ascend-950pr-63:~/strided-template-calibration-v2/results/dtype_calibration_v4_final/raw/*.json" \
       ./pre-explore/results/dtype_calibration_v4_final/raw/
```

拉回后**先用 `results/<campaign>/raw/*.json` 与板上的文件数对一遍**（**这些目录见提交 1**），
再进下一步；缺文件会让数据集静默变小（见 §5.6）。

### 4.5 第三步：构建数据集

聚合规则（`build_dtype_dataset.py` / `build_2d_dataset.py` 一致）：

1. 一个 case 只有在**至少一次 attempt 同时通过** witness / frequency / spread 三道门，
   并且**数值正确性校验通过**时，才算 usable；
2. 同一个 `(op, path, dtype, block, stride, num_warps)` 键的所有正确测量，
   **取中位数**作为 `target_ns`（同时记下 `target_min_ns` / `target_max_ns` / `target_spread` / `quality`）；
3. `stride == 1` 出现就**直接报错退出**（断言，不是过滤）；
4. 所有特征由 `stride_feature_reference.py` 统一算，保证将来 C++ 移植只有一份定义。

```powershell
$PY = "C:/Users/admin/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/python.exe"
cd pre-explore

# 注意：下面用到的 raw 目录（dtype_calibration_v1、dtype_calibration_v2_rotate512、expand2d、expand2d_b）
#       以及中间目录 _build/ 都只在提交 1 里有数据（见提交 1）。提交 2 可从 unified/new_only 起步。

# 1D
& $PY scripts/build_dtype_dataset.py --raw-dir results/dtype_calibration_v1/raw --out-dir results/dtype_calibration_v1
& $PY scripts/build_dtype_dataset.py --raw-dir results/dtype_calibration_v2_rotate512/raw --out-dir results/dtype_calibration_v2_rotate512

# 2D（每个 campaign 一个 build 目录）
& $PY scripts/build_2d_dataset.py --raw-dir results/expand2d/raw   --out-dir results/_build/build_c1
& $PY scripts/build_2d_dataset.py --raw-dir results/expand2d_b/raw --out-dir results/_build/build_c2

# 统一特征化（把 1D/2D、历史 campaign 全部拉到一个特征空间）
& $PY scripts/build_unified_dataset.py --out results/final/dataset_unified_v5.csv --exclude-template \
    --v1-1d results/final/dataset.csv --v1-2d results/final/dataset_2d.csv \
    --v2-1d results/dtype_calibration_v2_rotate512/dataset.csv \
    --v3-2d results/_build/build_c1/dataset_2d.csv \
    --v4-2d results/_build/build_c2/dataset_2d.csv
```

`--exclude-template` 是必须的：已经验证 `compile_mode=simd_simt` / `simt_only` 永远不会 emit
`triton_stride_load/store`，template 路径在范围之外。

拟合真正读的是 **new-only 视图**（只保留 `v1_/v2_/v3_/v4_` 这几轮受控测量）：

```powershell
& $PY scripts/run_final_pipeline.py --build     # 会重建 2D + unified，并派生 dataset_new_only_v5.csv
```

### 4.6 第四步：拟合

**拟合口径（不要改）**：

- 加权最小二乘，**权重 `1/T`** —— 等价于最小化相对误差，也就是 MAPE；
- 只报 **in-sample MAPE**（关键公式另做过 5 折交叉验证，结论与 in-sample 一致或在 0.3 个点以内）；
- 允许 `max` / `min`；**不允许** log / 指数 / 幂次；
- 项数尽量少，每一项都要有明确物理意义。

数据先按 **slot（公式）× rank** 划分（定义在 `PHYSICS_METHOD.md` §1，代码里在 `run_final_pipeline.py` 的 `slot_rows()`）：

```
simd_load_wide   : path==simd, op==load, ISB <= 2*eb
simd_load_gather : path==simd, op==load, ISB >  2*eb
simd_store       : path==simd, op==store
simt_load        : path==simt_only, op==load
simt_store       : path==simt_only, op==store
再各自按 rank（1 / 2）分成两个公式 —— 共 10 个。
```

**再加一层适用范围门槛**（`_tbl.py` / `gen_report_concise.py` 里的 `SCOPE`，就是最终 5118 个点的来源）：

| 公式 | 门槛 |
|---|---|
| `simd_load_wide_r1` | `eb<=4`；`spread<=0.2` |
| `simd_load_wide_r2` | `eb<=4`；`R>=2`；`C>=4`；`spread<=0.2` |
| `simd_load_gather_r1` | `n>=32`；`ISB<=128`；`spread<=0.2` |
| `simd_load_gather_r2` | `R>=2`；`spread<=0.2`；`footprint/payload<=32` |
| `simd_store_r1` | `eb<=4`；`ISB<=128` |
| `simd_store_r2` | `R>=2`；`C*eb>=32`；`eb<=4` |
| `simt_load_r1` | `eb<=4`；`ISB<=128`；`spread<=0.2` |
| `simt_load_r2` | `eb<=4`；`R>=2`；`spread<=0.2` |
| `simt_store_r1` | （无） |
| `simt_store_r2` | `R>=2`；`C*eb>=32`；`C>=8`；`n>=64` |

其中 `footprint/payload = ((R-1)*pitch + (C-1)*ISB + eb) / (R*C*eb)`（2D 地址跨度相对有效载荷的膨胀比）。

跑拟合：

```powershell
& $PY scripts/run_final_pipeline.py            # 每个 slot 贪心选 <=10 项，线性或 max(a*W, 线性)
# -> results/final/MODEL_FINAL_V7.json + 逐公式 MAPE 打印
```

最终交付的 10 个公式（含物理含义、派生量定义、字母取值）已经**冻结**在
`results/final/PHYS3_<slot>.json`。文件名有优先级：`PHYS3_*` > `PHYS2_*` > `PHYS_*`，
报告链读的是优先级最高的那份。

### 4.7 第五步：生成报告

**以下脚本都用相对路径，必须在 `pre-explore/` 目录下运行**：

```powershell
cd pre-explore

& $PY scripts/_eval.py                  # 逐点复算 10 个公式，与 JSON 里的 mape 自检 -> _eval.json
& $PY scripts/_tbl.py                   # 汇总表（n / est_us / meas_us / mape / PASS）-> _readme_tbl.json
Copy-Item _readme_tbl.json _tbl.json    # ⚠ 见下方说明
& $PY scripts/_readme4.py               # 读 _tbl.json + README_data.json -> README.md
& $PY scripts/gen_report_concise.py     # -> results/final/REPORT_v5_physics_scope_concise_des.md
```

> ⚠ **一个真实的复现坑**：`_tbl.py` 写出的文件叫 `_readme_tbl.json`，而 `_readme4.py` 读的文件叫
> `_tbl.json`。中间这一步 `Copy-Item` 不能省（这两个中间文件已被清理，不在仓库里）。

> ⚠ `_eval.py` 会把自算的 MAPE 与 `PHYS3_*.json` 里记录的 `mape` 对拍，不一致会打印
> `*** MISMATCH ***`。**看到 MISMATCH 就不要继续往下生成报告**，说明公式或门槛被改过。

### 4.8 一键复现清单

```text
板上：
  1. scp 上行 scripts/
  2. 对每个 PASS 执行：setsid nohup bash -c "PASS=<p> bash scripts/run_final_campaign.sh > runs/<p>.log 2>&1" &
  3. 监控（board_status.sh / board_wait_campaigns.sh 已删除，等价做法见 §5.5）：
     ps -ef | grep -c '[d]type_stride_measure_probe'  +  ls results/<campaign>/raw/*.json | wc -l
     直到 raw/*.json 数量达标、日志出现 ALLDONE
  4. scp 下行 results/<campaign>/raw/

本地：
  5. build_dtype_dataset.py / build_2d_dataset.py
  6. build_unified_dataset.py --exclude-template ...
  7. run_final_pipeline.py --build 然后 run_final_pipeline.py
  8. _eval.py -> _tbl.py -> Copy-Item _readme_tbl.json _tbl.json -> _readme4.py -> gen_report_concise.py
  9. 核对：10 个公式 MAPE 全 < 10%、加权平均约 7.22%、点数合计 5118
```

---

## 五、注意事项 / 为什么这样写（重点）

### 5.1 rotate 窗口：`--rotate-step` / `--rotate-mask` 是最容易测错的地方

探针的测量内核是一个**循环**，第 i 次迭代访问从 `(i & MASK) * STEP` 开始的同一块 tile。
这样做的目的是把「单发延迟」变成「流水化后的边际吞吐（斜率）」—— CostModel 要预测的是
**真实 kernel 循环里一次 dynamic access 的代价**，斜率才是主口径。

但这里有一个硬约束：

> **当 `n * stride > STEP`（元素域）时，相邻两次迭代的地址区间会重叠**，
> 第二次迭代命中的是第一次刚读过、还在 cache 里的 line，
> **测出来的斜率被缓存命中压低**，不再是无冲突的流式吞吐。

实测证据（同一配置，只换窗口）：

| 配置 | 窗口 | 实测 target |
|---|---|---|
| SIMT load, int64, n=256, stride=4, W=4（span 8192 B） | `--rotate-step 1024  --rotate-mask 8191`（约 67 MB） | **236.9 ns** |
| 同上 | `--rotate-step 8192  --rotate-mask 16383`（约 1073 MB） | **444.5 ns** |

**同一配置换大窗口，结果差 1.88 倍**；在所有历史数据里这类差异落在 **1.5–2.5 倍**区间。
（历史上还记录过 `step 4096 / 8192` 两个 SIMT int8 点相差 70% / 156%，
为此专门写了 `PASS=stepiso` 把 STEP 与 WINDOW 的影响分离开，见
[`pre-explore/scripts/run_rotate512_campaign.sh`](pre-explore/scripts/run_rotate512_campaign.sh) 的注释。）

**两个官方窗口口径，选哪个必须写清楚**：

| 口径 | 参数 | 用途 |
|---|---|---|
| 历史口径 512 MB | `--rotate-step 8192 --rotate-mask 16383` | 与 v8 SIMT load / store / template 的既有 profile 系数**同口径**，最终数据用的是它 |
| 自动口径 64 MB | 不传参数（`step = 1 << max(10, bitlen(max_tile-1))`，`mask` 按 64 MB 字节预算算） | 探针默认值；**自动模式下 step 恒 >= tile 跨度，不会重叠**，但与历史口径不可逐格比较 |

结论与纪律：

1. **默认用 `--rotate-step 8192 --rotate-mask 16383`**，与已发布系数同口径。
2. 若测大 tile（`n * stride > 8192`），**必须把 step 调到大于等于 tile 跨度**，否则斜率被低估。
3. **不要跨口径比较**：64 MB 与 512 MB 的同一格点不能混进同一个数据集。

### 5.2 探针主动拒绝 stride = 1

**连续访问是完全不同的一套代价律**（宽路径整块搬运、整块写），把它混进 strided 拟合会污染系数。
所以：

- `dtype_stride_measure_probe.py` 在 `main()` 里对 `--strides` 逐个检查，`stride < 2` 直接
  `SystemExit("stride == 1 must not be measured (contiguous)")`；
- 2D 探针的 `contig` 形态只在显式传 `--include-contig` 时才生成，**默认不测**；
- `build_dtype_dataset.py` 结尾断言 `stride == 1` 没有泄漏进数据集，泄漏就报错退出。

要研究连续访问，必须**另起一套探针/口径绕过这个闸门**（例如 CAModel 的 `--include-contig` 或
单独的 store 连续写实验），**不要**为了图方便去改断言。

### 5.3 重复测量取中位数；`spread > 0.2` 的点必须剔除

两个不同层次的离散度，不要混淆：

| 量 | 定义 | 阈值 | 作用 |
|---|---|---|
| 探针内的 `spread` | 同一次 attempt 内，两个 `iters` 点各自的「每迭代 Event」之间的相对离散度 `(max-min)/min` | `<= 0.40`（`--max-spread`） | **门**：超过就不算 valid attempt |
| 数据集里的 `target_spread` | 同一格点所有正确测量之间 target 的相对离散度 | **`> 0.2` 的点在适用范围里剔除** | **质量标记**：双稳态区 |

**为什么取中位数而不是取最小**：同一个 case 会被多个 pass / 多个 campaign 重复测到，
Event 只会被竞争抬高不会压低，但单个极小值可能是侥幸；**中位数是稳健估计**，
同时 `target_min/max/spread` 一起落盘，让下游能自己判断。

**为什么必须剔除 `spread > 0.2`**：这些点处于**双稳态区**（板上两次测同一配置能差 2.6 倍，见 §6），
它们**只占约 5% 的点，却承载约 24% 的总误差质量**。剔除它们不是「挑好看的数」，
而是把**测量本身不可重复**的格点从适用范围里排除；报告里已经写明了剔除了什么、为什么、剩多少点。

### 5.4 每个 case 都要做 correctness 校验

探针在进入计时之前会**真的跑一遍并写回、比对**：

- load：把 tile 的和写进 sink，与 `x[addr].sum()` 的参考值比（容差 `max(1e-3, |ref|*1e-4)`）；
- store：把输出缓冲区清零后写入 `(arange % 7)`，再按 strided 视图取回，要求**逐元素完全相等**。

校验失败的 case 在数据集里**直接丢弃**（不是打标记）。这一步抓到过真 bug：
2D int8 load 曾经因为 sink buffer 用了元素 dtype，`tl.sum(acc)`（float32）在 tile 约 43 个元素以上
就溢出 int8，**448 个 case 被误判为错误**；改成 float32 sink 后重测 296/296 通过。

另外每个 attempt 前后还有三道物理门（实现都在 `template_stride_common.py` / 探针里）：

- **ALU witness**：先测一次纯 ALU busy kernel 作为参考，attempt 之后再测一次，
  偏差必须 `<= 15%`（`--witness-tol`）——防止降频/抢占把时间算歪；
- **Aicore 频率**：读 `npu-smi info -t common -i 0` 的 `Aicore curFreq(MHZ)`，必须 `>= 1500 MHz`；
- 计时前还有一次 ALU busy burst（ramp）把频率顶到 boost，避免测量落在升频过程中。

### 5.5 板上并发：256 核，多个 campaign 后台跑

板上有 256 个核，一轮 campaign 要几小时，所以**必须并发 + 后台**：

```bash
# 启动（setsid + nohup 脱离终端，退出 ssh 也不会被杀）
setsid nohup bash -c "PASS=core1d bash scripts/run_final_campaign.sh > runs/core1d.log 2>&1" &

# 或者用 tmux（原 board_launch_final.sh 已删除，下面这行就是它的等价展开）
tmux new-session -d -s final -c "$PRE" "for p in core1d wsweep1d core2d wsweep2d n1; do PASS=$p bash $PRE/scripts/run_final_campaign.sh > $PRE/final_$p.log 2>&1; done; echo ALLDONE > $PRE/final_done.txt"
```

监控（⚠ **这一组运维脚本已在脚本清理中删除**：`board_status.sh`、`board_wait_campaigns.sh`、`board_progress.sh`、`board_eta.sh`、`board_watchdog.sh`。
它们的推导/用法记录见 [`pre-explore/docs/agent-logs/`](pre-explore/docs/agent-logs/) 与 git 历史 `git show HEAD:pre-explore/scripts/board_status.sh`）。等价做法：

- 板上时间与进程数：`date; ps -ef | grep -c '[d]type_stride_measure_probe'`；
- 完成度：`ls results/<campaign>/raw/*.json | wc -l`（对比 §4.3 的预期数）+ `tail -40 runs/<p>.log` 看 `WROTE ... valid=x/y correct=z/y`；
- valid / spread 分布：`python scripts/board_integrity.py`（**注意第 4 行的结果目录写死为 `v10_expand2d`，换 campaign 要先改**）；
- ETA：按日志里相邻两行 `----- ... (时间戳) -----` 的间隔乘以剩余 case 数心算；
- 看门狗：`while` 循环 + 上面两条命令自己拼一个即可（原 `board_watchdog.sh` 是每 30 分钟审计一次并告警空 case 文件）。

纪律：

1. **并发数不要超过物理核能承受的量**。历史经验是同时跑 2-4 个 campaign（每个 campaign 内部是串行的）比较稳；
   并发过多会让 witness / frequency 门频繁失败，反而拖慢进度。
2. **看日志里的 `WROTE ... valid=x/y correct=z/y`**：`valid` 比例明显低于 90% 就说明板上不干净
   （被别人占用 / 降频），这时候停下来重跑比硬撑划算。
3. 长跑一定要 `setsid nohup` 或 `tmux`，否则 ssh 断线会带走整轮测量。

### 5.6 结果文件名必须带足够区分度（曾经因此丢过数据）

**真实事故**：两个驱动脚本对同一个 `(op, path, dtype)` 的两次调用复用了同一个输出文件名，
后一次把前一次的结果**静默覆盖**了，直到对账时才发现少了一个批次。

现在的约定是文件名必须带上**网格签名**：

```
<op>_<path>_<dtype>_W<num_warps>_<tag>.json              # 1D
2d_<op>_<path>_<dtype>_W<num_warps>_<tag>.json           # 2D
其中 tag 描述这一批的 block/stride 或 shapes/forms 签名，例如 core / wsweep / n1 /
  1x256-8x16-16x32_both_s3_tight-both_s3_gap / pass1 / pass2
```

新写驱动脚本时请遵守：**同一个输出路径只能被一个批次写**。
拉回本地后，用文件数对账（`ls raw/*.json | wc -l` vs 预期）再进下一步。

### 5.7 其它约定

- **`num_warps` 是真实因子，不是常数**。SIMT 路径在 W in {1,2,4,8,16,32,64} 上都测过；
  公式里的 `max(a*W, ...)` 分支就是从数据里读出来的（W >= 16 时 T 与元素数无关）。
  SIMD 路径 W 恒为 1。
- **每个 campaign 一个 `--out` 目录**，不要往别人的目录里写；`results/final/raw/` 是合并视图，由构建流程生成（**该目录已在提交 2 删除，见提交 1**）。
- **编译产物要留**（`--asm-dir`）。`assert_lowering()` 会在计时前检查 asm 里到底有没有/该不该有
  `triton_stride_load/store`，事后还能复查 lowering 是否被编译器换了路径。
- **不要放宽任何一道门**。想提高 valid 率就减少并发 / 重跑，不要动 `--max-spread` / `--witness-tol` / `--min-freq-mhz`。

---

## 六、已知的数据陷阱

下面每一条都是**踩过并已确认**的，用数据时必须带上这个背景。

| 陷阱 | 现象 | 处理 |
|---|---|---|
| **128 B 是我自己设的假设，对 MTE2 不成立** | 曾把 128 B cacheline 当成 MTE2 的冲突粒度来构造特征，误差下不去 | 用户指出后实测扫描 `B in {16..4096}` 得**最优 B = 256 B**；同块冲突粒度是 256 B，BIU 命令的 payload 粒度才是 64 B/512 B/1 KB 窗口 |
| **4 KB page 也是自设假设（CPU 概念）** | 4 KB 附近看起来像阈值 | 实测是**双稳态**，不是干净阈值：同一配置两次测量能差 **2.6 倍**；最终公式里**没有** 4 KB 页面项 |
| **BIU 命令数 = 元素数** | gather 路径每元素一条命令 | CAModel 白盒确认 `biu_send_count = n`、`biu_send_sizes = {eb: n}`；这条是 `simd_load_gather_r1` 里 `b*n` 项的依据 |
| **rank-2 里 R = 1 的形状是伪 2D** | 1x16 / 1x32 / 1x64 在数据里被算作 rank-2 | 拟合前**必须剔除 `R < 2`**（`SCOPE` 里的 `F_r2`）；它们本质是 1D，混进去会污染 2D 公式 |
| **int8 的 sink buffer 截断** | 2D int8 load 在 tile 约 43 个元素以上全部被判「结果错误」（448 个 case） | 根因是 `tl.sum(acc)` 是 float32 而 sink 用了元素 dtype；改成 float32 sink 后重测 296/296 通过 |
| **输出文件名撞车静默覆盖** | 一次 campaign 少了一个批次的数据 | 文件名带 block/stride 签名；每轮结束对账文件数（见 §5.6） |
| **rotate 窗口重叠压低斜率** | `n*stride > step` 时同一配置换窗口结果差 1.5–2.5 倍 | 见 §5.1；大 tile 必须把 step 调到 >= tile 跨度 |
| **双稳态格点** | `target_spread > 0.2` 的点重复测量差 2 倍以上 | 剔除；它们占 5% 的点却贡献 24% 的误差质量（见 §5.3） |
| **单发口径 vs 斜率口径** | 同一个点单发测 263.5 ns，斜率测 68.9 ns，差 4-5 倍 | 两者是**不同的物理量**（单发延迟 vs 流水化吞吐）。只有 SIMD load 的历史数据是单发口径；本目录最终数据统一用**斜率**。**不要混用** |
| **template 路径在范围之外** | `triton_stride_load/store` 的数据会让同一批公式误差变大 | `build_unified_dataset.py --exclude-template`；已验证 `simd_simt` / `simt_only` 永不 emit 这两个符号 |
| **SIMD load 内部还分 wide / gather 两个区制** | 用一条线性公式拟合整个 SIMD load，误差下不去 | 按 `ISB <= 2*eb` 切分：`<=` 走连续块搬运 + extract_slice（extract 代价约 0），`>` 走逐元素 gather（每元素一条 BIU 命令） |
| **`_tbl.py` 与 `_readme4.py` 的文件名不匹配** | 报告链在中间断掉 | `_readme_tbl.json` 需要复制成 `_tbl.json`（见 §4.7） |

### 误差与适用范围的口径

- 全部误差都是 **in-sample MAPE**；关键公式的 5 折交叉验证误差与 in-sample 一致或在 0.3 个点以内。
- 最终 10 个公式覆盖 **5118 个实测点**；拟合输入是 `dataset_new_only_v5.csv` 里 `valid=1` 的 **9541 行**，
  再叠加 §4.6 的 10 条适用范围门槛。
- **适用范围之外的 tile 不可外推**：门槛之外（`eb > 4`、`ISB > 128`、`R < 2`、`n < 32`、`spread > 0.2`、
  `C*eb < 32` 等）没有标定，CostModel 侧必须有对应的兜底策略。
- 已知的残余风险（见 `ROUTE-LEVEL-VALIDATION-v8.md` §Residual risks）：对齐 / mask / 负步长 / 运行时步长 /
  非 4 字节 dtype 都在模型的事实域之外；b512 单/双 warp 大跨度仍有高误差尾巴。

---

## 七、排错 / FAQ

| 症状 | 可能原因 | 怎么办 |
|---|---|---|
| 板上 `import torch_npu` 或 acl 检测失败 | 环境没按 `board_env.sh` 进，或者 `PYTHONPATH` 被清掉了 | 统一用 `bash scripts/board_env.sh python ...`；不要 export `PYTHONPATH=$HOME/sb64_latest_env` |
| `valid` 比例很低 | 板上被别人占用 / 降频 / 并发太高 | `npu-smi info` 看当前频率、`ps -ef` 看进程与并发（原 `board_status.sh` 已删除）；降并发重跑 |
| `correctness_ok` 大面积 false | 网格本身有问题（例如把连续访问当 strided 测），或 dtype 溢出 | 先看失败是否集中在某个 dtype / 形状；参考 §6 的 int8 sink 案例 |
| 日志里 `COMPILE/ASSERT FAIL` | `assert_lowering` 发现 lowering 路径不对（template 没触发 / 非 template 却触发了） | 检查 `compile_mode` 与相关选项；template 还需要 `parallel_mode=mix_simd_simt` + `compile_on_910_95=True` |
| `_eval.py` 打印 `*** MISMATCH ***` | 公式 JSON 或适用范围门槛被改过 | 停下来核对，不要继续生成报告 |
| 数据集行数比预期少 | scp 拉回不完整 / 文件名撞车覆盖 | 对账 `raw/*.json` 文件数与预期 |
| 拟合报 scipy 缺失 | 用了 scipy | 本环境**没有 scipy**，用 `fit_v9.py` 里的纯 numpy NNLS |

---

## 八、相关文档与交接

| 文档 | 内容 |
|---|---|
| [`pre-explore/README.md`](pre-explore/README.md) | **最终物理模型报告**：10 个公式总表、逐项物理含义、字母取值、物理机制、适用范围与已知限制 |
| [`pre-explore/results/final/REPORT_v5_physics_scope_concise_des.md`](pre-explore/results/final/REPORT_v5_physics_scope_concise_des.md) | 逐公式明细：实测条件 → 变量定义 → 公式与各项物理意义 → in-sample 误差 |
| [`pre-explore/docs/PHYSICS_METHOD.md`](pre-explore/docs/PHYSICS_METHOD.md) | 物理第一性方法论、已确认的物理发现、真实算子范围门槛、输出 JSON 格式 |
| [`pre-explore/docs/ROUTE-LEVEL-VALIDATION-v8.md`](pre-explore/docs/ROUTE-LEVEL-VALIDATION-v8.md) | v8 公式的 route-level A/B 验证（决策、候选分值、残余风险） |
| [`pre-explore/docs/COMMIT_MSG.txt`](pre-explore/docs/COMMIT_MSG.txt) | 最终测量 campaign 的提交说明：网格、脚本、修过的坑 |
| [`pre-explore/docs/handoff/`](pre-explore/docs/handoff/) | 3 份接力提示词，含历次口径讨论（尤其 v6 那份讲清了 rotate 窗口的来龙去脉） |
| [`pre-explore/docs/agent-logs/`](pre-explore/docs/agent-logs/) | 11 份推导过程记录（`p3_*.txt`），拟合是怎么一步步收敛的；被删除的 `_p2_*`/`_p3_*`/`phys_*`/`phys3_*` 等探索脚本正是这些日志的产物 |
| [`pre-explore/scripts/INDEX.md`](pre-explore/scripts/INDEX.md) | **现存 106 个脚本的分类索引**（探针 / 数据集构建 / 路径检查 / 拟合与库 / 报告生成 / 板上驱动 / CAModel）+ 完整复现调用链 + 「测量批次 campaign」术语表。覆盖 622 个脚本的旧版见 `git show HEAD:pre-explore/scripts/INDEX.md` |

---

## 九、一分钟速查

```bash
# 板上：测一批 1D（注意窗口必须与历史口径一致）
bash scripts/board_env.sh python scripts/dtype_stride_measure_probe.py \
  --op load --path simd --dtype float32 \
  --blocks 4 8 16 32 64 128 256 512 --strides 2 3 4 6 8 16 32 64 128 256 \
  --num-warps 1 --rotate-step 8192 --rotate-mask 16383 \
  --reps 3 --max-attempts 3 \
  --asm-dir results/<campaign>/measure_asm \
  --out     results/<campaign>/raw/load_simd_float32_W1_core.json
```

```powershell
# 本地：raw -> 数据集 -> 拟合 -> 报告
cd pre-explore
& $PY scripts/build_dtype_dataset.py --raw-dir results/<campaign>/raw --out-dir results/<campaign>
& $PY scripts/run_final_pipeline.py --build
& $PY scripts/run_final_pipeline.py
& $PY scripts/_eval.py; & $PY scripts/_tbl.py; Copy-Item _readme_tbl.json _tbl.json
& $PY scripts/_readme4.py; & $PY scripts/gen_report_concise.py
```

三条铁律：**窗口要够大（step >= n*stride）、stride=1 不进来、spread>0.2 的点剔除**。
