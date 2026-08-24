# Cost Model 文档索引（SIMD/SIMT Route Model）

本目录汇集 SIMD/SIMT 路由代价模型（autoscope）的全部设计与分析文档。五份文档的定位、
相互关系与时效状态如下，按角色分为三类：**设计规范 / 代码导读 / 分析与路线图**。

## 文档清单与关系图

```
                    ┌─────────────────────────────────────┐
                    │ simt_costmodel_structured_design-v2 │  设计规范
                    │ （Stage 化重构的设计定义 + 三用例实验 +│  （写代码前的"应该是什么"）
                    │   GAP 与下一步计划）                  │
                    └──────────────┬──────────────────────┘
                                   │ 规范 ←→ 实现互为对照
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
 ┌─────────────────────┐ ┌──────────────────┐ ┌──────────────────────────┐
 │ simt_costmodel_      │ │ autoscope_       │ │ simt_costmodel_kind_     │
 │ code_guide           │ │ costmodel_first_ │ │ coverage_and_taxonomy    │
 │ （按调用顺序的参考手册）│ │ principles      │ │ （28 算子覆盖证据 + 分类  │
 │                      │ │ （why→how 推导式  │ │  重组提案 v1）            │
 │                      │ │  精读，教学向）    │ │                          │
 └─────────────────────┘ └──────────────────┘ └───────────┬──────────────┘
                                                          │ 其分类提案被下文修订
                                                          ▼
                                            ┌─────────────────────────────┐
                                            │ costmodel_taxonomy_review_  │
                                            │ and_extension_guide         │
                                            │ （31 算子再精读 + 两轴一格    │
                                            │  重组定案 + 四工单扩展路线图） │
                                            └─────────────────────────────┘
```

## 各文档说明

### 1. `simt_costmodel_structured_design-v2.md` —— 设计规范【现行】

当前 Stage-based 实现的权威设计文档：五个核心改造、组件架构（StagePartitioner /
StageCostEvaluator / Registry / SuperBlock / TransitionCost / RouteSolver）、三个用例
（solve_tril、gather-dot-min、FBGEMM gather-scale+FP8）2026-08-21 的实验数据，
以及第 10 节的 GAP 清单与 P0–P2 下一步计划。

**时效提示**：第 10 节部分条目已随代码演化变化——profile 已删除未实测的 182-cycle
定向切换假数（`scope_handoff.fixed_directional_system_cycles` 现为 0 并标注
unmeasured）；"Mixed scope F2/F4 未打通""Materializer 能力小于 Solver 表达力"
两条经 selector pass 的 `action_supported` 检查链路确认**仍未解决**。

### 2. `simt_costmodel_code_guide.md` —— 参考手册型代码导读【现行】

按调用顺序逐组件讲解实现（LayoutMerge/V1 前置 → 锚点 → 特征 → 切分 → 计价 → DP →
决策落地），带行号速查，适合当作开发时的手册翻。第 10 节从旧模型两个结构性错误出发
推导五个设计决策，与本目录《first_principles》互补但不重复——一个按"地图"组织，
一个按"推导链"组织。

### 3. `autoscope_costmodel_first_principles.md` —— 教学型精读【现行，本仓库产出】

以第一性原理推导链组织的完整讲解：为什么需要解析模型 → 为什么必须用 transform 后的
TTIR → 为什么有锚点/不可变计划 → 为什么 Stage 化 → 为什么 DP → 如何物化，附调用链
全景图、11 个组件精读、solve_tril 实例走查与 FAQ。适合新人入门或向他人讲解；
查具体函数时配合《code_guide》使用。

### 4. `simt_costmodel_kind_coverage_and_taxonomy.md` —— 覆盖分析【已被修订，仅作证据参考】

对 28 个目标算子的逐段覆盖分析与第一版分类重组提案（7 族）。**其"三公理"方法论
可取，但 7 族划分仍混用分类轴，判定树与族边界已被下文取代**；保留价值在于其第 5 节
逐算子"哪段代码不属于已有范式"的证据表。

### 5. `costmodel_taxonomy_review_and_extension_guide.md` —— 分类定案与扩展路线图【现行，最新】

基于 31 个 kernel 的四份并行精读报告：诊断现有 20-kind 的三个病灶（轴混用、粒度与
公式脱钩、机制整族缺席），给出两轴一格重组定案（7 族 14 类 + 属性旗标 + 约束强度
判定树），并给出四个可并行的工程工单（枚举重构 / 新域扩展 / 聚合公式补价 /
profile v18 校准）与 ScatterCompaction 域的七步完整示例。

## 阅读路径建议

| 目的 | 路径 |
|---|---|
| 新人理解这套系统 | 3 → 2 →（需要时查）1 |
| 动手扩展覆盖面 | 5（路线图与示例）→ 1 第 10 节（校准侧 GAP 对照）|
| 评审分类体系 | 5 第一部分 → 4 的证据表交叉验证 |
| 查某个函数的行为 | 2 的行号索引 → 源码 |

## 维护约定

- 设计变更先改 1，实现后同步 2/3 的对应章节；分类体系变更以 5 为准。
- 文档中的行号对应撰写时的源码版本，漂移后以函数名检索为准。
