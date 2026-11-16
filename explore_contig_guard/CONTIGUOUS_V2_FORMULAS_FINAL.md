# 连续访存 v2 标定：公式手册（终版，含评审核验）

基线：`2e9efcacb891eb`。代码位置：
`third_party/ascend/costmodel/lib/AscendModel/RouteModel/StageCostModels.cpp`
函数 `mapContiguousV2Completion`（约 420–1140 行）；边界隐藏在
`SimdSimtCostModel.cpp` 的 `completeCandidateCosts`。

---

## 0. 先把术语说成人话（第一次出现都在这里）

| 术语 | 意思 |
|---|---|
| **v2 钩子** | 我们在 costmodel 里插的一段新代码：凡是"连续访存"（地址一次往后走、不跳跃的 load/store），改用我们新标定的表来计价，不再走老公式 |
| **连续（Contiguous）** | 一条访存指令读/写的一块内存，内部是逐字节连续的（比如一行 512 个 float） |
| **StridedWide / 跨步** | 块内部有一段连续（>32B），但块与块之间跳着走（比如每取 32B 跳 384B） |
| **rowBytes** | 上述"块内部连续那段"的字节数。permute 的 c 载入：每行连续 8 个 float = 32B，行间跳 96 个元素 = 384B |
| **stage** | costmodel 把一个 kernel 拆成若干"段"，每段一类工作（访存段、归约段、标量段……），每段独立计价再相加 |
| **K（iterationCount）** | 这个循环 stage 一共循环多少次 |
| **hint（programCountHint）** | 启动时 一共发射多少个 program（= grid 大小），用来算"分几波上场" |
| **v2Waves** | `ceil(hint / 56)`：一颗芯片 56 个物理核，program 多于 56 就要分波，先后上场 |
| **波衰减（wave decay）** | program 分的波越多，同一颗核上前后波重叠共享带宽，平均每个 program 的访存价格会降一点。循环体按 `waves^-0.05`（B_add_g224 拟合），直线段 load 侧按 `waves^-0.32`（guard WDA 拟合） |
| **行（row）** | 表里的一格：`n<load条数>_s<store条数>_b<每次多少字节>_k<循环次数>_g<grid>_m<是否掩码>`，值 = 用这种循环体实测的**整核耗时（纳秒）**。grid>56 的行已除以波数 |
| **store-bearing** | 这个循环体里**有** store（对应 load-only：只有 load 没有 store） |
| **bodyMasked** | 模型视角下这个循环体的访存带掩码且编译期能证明"只有部分 lane 活跃"（activeElementRatio<1） |
| **tiny-store** | 每次循环只写很小一块（≤128B，比如归约完写一个标量）而旁边有一大块 load 的体。这种体的早期循环会被小 store 的延迟串起来，比纯线性贵——所以单独有一张 `tiny_direct` 表和一条"超出线性部分的延迟链"修正 |
| **direct_fixed_share (0.05)** | 行值里的"启动/固定开销"部分默认 95% 能被内核边界藏住，只把 5% 计入——所以公式里 A 项乘 0.05 |
| **claimAll** | v2 认领动作：把这组访问记到"已由 v2 计价"的账上（含每条访问的实例数），后面的老公式看到账本就跳过，防止重复计价 |

---

## 1. 总准入（进入 v2 的一组过滤器，**全部都要通过**，是"且"的关系）

代码上就是一串 `if (不满足) continue;` ——任何一条不满足，这条访问就不进
v2，留给后面的老公式。逐条解释：

| # | 条件（原文） | 人话 |
|---|---|---|
| 1 | `count > 0` | 这条访问真实存在（次数>0） |
| 2 | 是连续访存（ContiguousShort/Wide）；**或**（本轮新增）是跨步访存但"块内连续段 ≤64B"，并且**同一个 stage 里已经有连续 load 被认领了** | 见下面问答 (a) |
| 3 | 布局是编译期可确定的（shape/stride 都知道），元素宽度合法 | 未知布局没法选行 |
| 4 | 地址不依赖"从内存读出来的下标"、没有非仿射运算、没有被 if 分支选择 | gather / 间接寻址是别的责任域，不归连续访存管 |
| 5 | 掩码 **store** 且编译期能证明"部分 lane 活跃"（ratio<1）→ 不进 | 见问答 (b) |
| 6 | 这一轮计价里还没被别的机制认领 | 见问答 (c) |

### 问答 (a)：混流准入是什么？和 StridedWide 标定（4902ae4a 一系）冲突吗？

**混流准入在说什么**：permute 这个 kernel 里，x、y 两笔是纯连续载入
（各 2048B），z 是跨步载入（每行连续 32B、行间跳 384B），z 和 x/y 在
**同一个 stage** 里共用同一组 DMA 引擎。实测发现：z 单独测要 470ns/迭代，
x/y 单独测要 175ns/迭代，但三个一起跑只要 ~400ns/迭代——引擎是共享的，
孤立价格相加会超资。所以本轮把"小连续段（≤64B）的跨步 load"在"同 stage
已有连续 load 被认领"的前提下并进 v2，用**三个一起跑的组合体实测行**一次
计价（permute 专测了这种组合体：`n3_s0_b2048_*` 行）。

**和 StridedWide 标定的关系**：你说的 4902ae4a 一系（本仓库里能看到同源的
`534fb1639c`/`7cf4d8d7d5`，"StridedWide physical-factor calibration，
SIMT five-segment"）**不在本分支上**——本分支从 2e9efcacb8 分出，没合它。
所以当前树上不存在代码冲突。将来合并时确实有一个**优先级问题**要当面
对齐：Swide 公式的扫掠域（行内连续 32..2048B、间隔 2..8192B 等）覆盖了
我的混流条件（≤64B），两边都想给这种访问计价。机制上**不会重复计费**
（claimed 账本先到先得，v2 在 layout 分支先跑），但会变成"谁先认领谁
说了算"。建议合并时的对齐方案（二选一，需要 Swide 的行价数据才能定）：
1. 我的混流准入收紧到 Swide 扫掠域之外；
2. 或者混流场景优先用 Swide 的行价、v2 只组合连续部分。
这条已经写进代码注释 TODO 和本文 §6。

### 问答 (b)：动态全 true 的 store（编译期不知道 ratio）有影响吗？

没有影响。规则是：只有**编译期能证明**"部分 lane 活跃"
（`activeElementRatio` 是静态已知的、且 <1）的掩码 store 才被排除；
运行期才知道真假的掩码，`activeElementRatio` 取默认 1.0，按满字节进 v2。
运行期如果掩码真是全 true，满字节就是对的；如果运行期部分为假，那是
"动态掩码"域（另一个问题域），本机制与老公式行为一致（都按满算）。

### 问答 (c)：条件 6"未被本轮其他机制认领"是走兜底的意思吗？

不是。这是**去重**，不是兜底：同一轮计价里，排在前面的机制
（短行跨步认领 `mapStridedShortMemoryCompletion`、以及 v2 自己更早的
认领）已经把某条访问拿走时，后面的扫描跳过它，防止两个机制对同一笔
访问各算一遍钱。真正的"兜底"是另一件事：v2 四层都失配时退到解析式
（见 §1.4），那是 v2 自己的保底。

---

## 2. 循环体 stage：四层公式，从上到下依次尝试，命中即停止

进入循环体分支的前提：stage 带循环、`K = iterationCount ≥ 1`。
每层命中后：`resources.load += perIter`（估计器会再乘 K 得到整段价格）
并 `claimAll()` 认领返回。

### 2.1 Tier 1a：tiny-store 延迟链（tiny_direct 表）

**什么时候进**（全部满足）：
- 体里有 store 且**每个 store 都 ≤128B**（tiny-store），同时主载入块很大
  （与 domBytes 的 log2 距离 ≤1）；
- 或者体里只有 load 且带静态部分掩码（rms 证据）；
- 行匹配：load/store 条数一致、掩码标志一致、桶距离 ≤1；
- K 不外推：`K ≤ 该族最大 K × 1.5 + 1`。

**为什么单独一层**：大 load + 小 store 的循环，前几迭代会被小 store 的
完成延迟串行卡住，之后 DMA 管线填满才进入线性区——整条 T(K) 曲线在
小 K 段高高翘起。用"线性趋势 + 趐出部分"两段建模：
```
m      = 该族所有 K 点两两斜率的中位数（线性部分的每迭代价）
A      = max(0, 中位K点的T − 中位K×m)        （截距）
rowT   = 行表里离 K 最近的那格
excess = max(0, rowT − (A + K·m))            （翘起部分）
perIter = (A×0.05 + K·m + excess×0.636) / K × 波衰减
```
0.05 = direct_fixed_share（A 项只显现 5%）；0.636 = tiny_store_chain_share
（翘起部分 63.6% 显现，其余与边界重叠）。

### 2.2 Tier 1：direct 行整核拟合（主力）

**什么时候进**：`hint ≤ 224`（行只测到 4 波）；族匹配同上（桶距离 ≤1）；
网格取值：hint **恰好等于**某个测量 grid → 用那个 grid 的**全部 K 行**；
否则取上下两个邻 grid、按 log2-grid 距离几何插值。

**公式**：
```
m     = 斜率中位数（所有 K 对）
mHi   = 只用 centreK 之后点对的斜率中位数（若 ≥1.05m 才采用）
useM  = K>16 且 mHi>m ? mHi : m          （大 K 时用大 K 段的斜率）
chunkScale = clamp(dom/bestB, 0.5, 2)    （桶大小换算）
perIter = (A×0.05 + K·useM·chunkScale)/K × 波衰减
```
只有一个 K 行时：`m = T·0.5/K`、`A = T/2`（各半拆分）。

### 2.3 Tier 2：聚合带宽包络（仅 simd）

**什么时候进**：simd 且 profile 提供了聚合包络参数（Tier1 没匹配行时）。
```
perIter = (每次迭代启动开销 + (load字节+store字节)/聚合带宽) × 波衰减
```
参数：`independentLoadGroupIterationStartupCycles`、
`independentLoadGroupBytesPerCycle`（calG：整波 ~83B/cycle）。

### 2.4 Tier 3：逐操作解析式（保底，永不脱靶）

```
每条访问：simd = 启动开销 + 字节/带宽；simt = warp指令数/每周期warp指令
load 侧、store 侧分别求和
```

**层间关系**：1a → 1 → 2 → 3 严格降级，任何一层命中即认领返回。
连续流保证永远由 v2 计价（最差也有解析式下限），老公式只见非连续部分。

---

## 3. 直线 stage（无循环）：Formula A 同形状组（**按代码实际逻辑重写**）

**进入条件**：stage 无循环壳；simd 直接查表，simt 先查基础键再试
`/w<W>` 后缀键（W = 该配置的 warp 数）；**掩码组不查表**（见 R17 修复）。

**分组**：按 `(是否store, 元素数, 字节宽, 是否掩码)` 分组，同组同价。

**公式（查到拟合格时，代码原样）**：
```
桶 bucket = 逐个 {32,64,128,256,512,1024,4096,16384} 取
            |log2(字节) − log2(b)| 最小的 b（掩码组用活跃字节选桶）
res = 总足迹 ≤ 128MB(L2) ? "warm" : "cold"
键 = "{load|store}/{dtype}/{桶}/{res}"（simt 未命中再加 "/w<W>" 重试）

n   = 组内该形状访问的实例数
若 n > g.Q:   total = g.first + (g.Q−1)·g.marginal + (n−g.Q)·g.cTransfer
否则:         total = g.first + (n−1)·g.marginal
（g.first = 第一个的价；g.marginal = 之后每个的增量价；
  g.cTransfer = 超过 Q 个之后的"换档"增量价；g.Q = 第一段额度）
ns = total × pAxisFactor(P=hint, 方向, 模式, res)     ← P 轴衰减查表（12 档）
```

**公式（掩码组 / 查不到格时）**：退到逐操作解析式（同 §2.4 形状），
掩码按活跃字节；simd 的 load 侧乘 `waves^-0.32`，store 侧不乘；
simt 乘 `waves^-0.05`。

---

## 4. launch 边界隐藏（calH）

**什么时候生效**：直线 stage、访存全是连续、≥2 个访存 stage、plan 合法、
launchWaves ∈ {2,3}（launchWaves 用 launch 网格数算，不是被钳到 56 的
runtime 数）。

**公式**：整个候选预测 **−640ns**（线性常数，vector_add 2 波锚点 + calH
扫描拟合）。物理含义：第二波 program 的启动开销被第一波的执行盖住。
已知边界：对 2 波但总时长只有 ~2.4µs 的短流 kernel（guard 的
A_copy_f32_4k_g64），640ns 占比过大，会过度隐藏——见 §8 评审表。

---

## 5. 本轮（calM 攻坚）落地的 6 处代码变化

| # | 变化 | 一句话动机 |
|---|---|---|
| 1 | claimAll 按 `access.count`（访问实例数）记账 | 循环壳会把每条访问标成多实例（StagePartitioner.cpp:1932 `local.scale(multiplicity)` 把壳倍数乘进 count；permute 的壳=2）。旧版只记 1 份，老尾项把没记账的实例再算一遍 → permute +63%→−2% |
| 2 | 混流准入（§1 条件 2 的"或"分支） | 引擎共享的跨步小行 load 用组合体行一次计价 |
| 3 | SIMT 归约组分支取 `max(组体, 串行流, issue)` | 标定组曲线不得遮蔽诚实的 load 流 claim（rowred −25%→+1.6%） |
| 4 | 掩码 padded 瓦片两个未覆盖尾项按活跃字节计 | 掩码尾部 lane 不取数（max_dim1 +12.1%→过线） |
| 5 | tier-1 网格括号：精确命中用该网格完整 K 集合 | 旧逻辑无条件退一格，导致注入的 g32 单行把 guard 的 g56 家族 K 集合坍缩成单点、单行拟合虚高 2 倍 |
| 6 | R17 修复（评审采纳）：掩码组不做 simt `/w<W>` 二次查表 | 桶按活跃字节选、行公式按满字节算，二次查表会把两者错误配对 |

---

## 6. 评审意见逐条核验（另一个 AI 的 review）

| 意见 | 判定 | 依据 |
|---|---|---|
| "claimAll 修复是空操作（count 只有 0/1）" | **错** | `Analysis/StagePartitioner.cpp:1932` `local.scale(multiplicity)` 把循环壳倍数乘进每条访问的 count（permute=2）；RUN5D 的 [mwtail] trace：workload 侧 wOp=6（3 访问×2 实例）vs 记账侧 layOp=3（旧版每条只记 1）；只差这一个改动的 RUN5E→RUN5F，permute 预测 10571→~6300。"直线段不读 count"属实，但直线段走 Formula A 的组计价，与循环体的 claimAll 是两条不同路径，语义本就不同，不存在两边不一致的问题 |
| R3：direct 行要求 (nL,nS) 完全相等，失配掉 Tier2/3 | **对，是当前最大的泛化约束** | 行键设计如此。缓解是四层降级保证仍有合理下限；彻底解决需要把行键与 (nL,nS) 解耦（新表维度），列为后续工作，本轮不动 |
| R4：hint≤224 是悬崖 | **对，是有意的保守** | 行只在 ≤4 波测过；muls_add（36.6 波）证据显示远外推会超价 2.6×。平滑衔接列为后续工作 |
| R5：四套衰减写法、重复衰减嫌疑 | **半对** | 四套写法确实存在（p_axis 表 / 循环体 −0.05 / 直线 load 侧 −0.32 / 行值本身除过 waves）。"重复衰减"不成立：−0.05 是在"已除 waves 的行值"之上用 B_add_g224 单独拟合的**增量**（B_add_g224 当前 +12.5%，若去掉这层衰减它按比例恶化约 +29%），两者校准目标不同。已把各衰减的含义写进 §0/§2 |
| R17：掩码组被 `/w<W>` 二次查找查回 group_a，桶却按活跃字节选 | **对，真 bug，已采纳修复** | 代码原文：`it = maskedGroup ? end : find(key); if (it==end && !simd) it = find(key+"/w"+W);` —— 重试分支对掩码组也生效。已改为 `!simd && !maskedGroup`。当前 100/guard 里没有静态部分掩码的直线 simt 组命中此路径（guard 的掩码是动态的，ratio=1 走正常路径），所以这是潜在 bug 修复，不影响现有数字 |
| 大量死参数（loop_b、sdirect、s_penalty、bw_*、boundary_hide 行无消费点；ContiguousLdStCostModel.cpp 不在 CMake；evaluateContiguousV2Ns 无调用者） | **对** | 逐一 grep 核实：loop_b 只出现在空表判断；v2 的 sdirect/s_penalty/bw_* 无消费；`evaluateContiguousV2Ns` 只有定义无调用；`Models/ContiguousLdStCostModel.cpp` 不在 RouteModel/CMakeLists.txt 的源列表里。这些都是数据/潜件，不在运行路径上。处理：本轮不动（删数据行有回归风险、删死文件与本次标定无关），已在本文记录，建议单独的 cleanup 提交 |
| 两份 profile（lib/AscendModel/RouteModel/ 下有一份无 v2 段的旧快照） | **对，隐患，已删除** | 确认存在（132KB，无 contiguous_ldst_v2 段）。当前编译宏指向 profiles/ 下的正本，但一旦被解析到会静默关掉 v2。已 `git rm` |
| 文档 §2 与代码不符（实际是 first/marginal/cTransfer 分段 + pAxisFactor） | **对，已重写** | 本版 §3 按代码原样写 |
| 旧手册（CONTIGUOUS_V2_FORMULAS.md）的 direct 公式、waves^-0.125、Tier1a dom≥512/K≤8 过时 | **对** | waves^-0.125 是早期拟合（现为 −0.05，代码 430 行的旧注释一并修正）；Tier1a 现在是"桶距离 ≤1 + bestMaxK 门"。旧手册标注为历史版本，以本文为准 |

---

## 7. 参数总表（runtime 读
`costmodel_profiles/simd_simt/ascend_950pr_simd_simt_v1.json`）

| 参数 | 值 | 用在哪 |
|---|---|---|
| `direct` | simd 295 / simt 291+ 行 | 循环体 Tier 1 |
| `tiny_direct` | 20 行 | 循环体 Tier 1a |
| `group_a` | simd 48 / simt 78 行 | 直线 Formula A |
| `loop_b` | simd 60 / simt 59 行 | **未消费**（评审确认，留作数据） |
| `sdirect` / `s_penalty` / `bw_*` / `boundary_hide` 行 | — | **未消费**（评审确认） |
| `direct_fixed_share` | 0.05 | Tier1/1a 的 A 项显现份额 |
| `tiny_store_chain_share` | 0.636 | Tier1a 翘起份额 |
| `l2_cache_bytes` | 128MB | Formula A warm/cold |
| `p_axis` | 12 档 | Formula A 的 pAxisFactor |
| 波衰减 | 循环体 −0.05；直线 load 侧 −0.32 | §2/§3 |
| waves 分母 | 56 物理核 | 全部 wave 计算 |

**本轮新增行（+40 行，calM 攻坚）**：permute `n2_s0_b2048_k{4,8,12,16}_g{8,32,56}`
+ `n3_s0_b2048_k{8,12,16}_g32`（混流组合体）；fused `n1_s1_b2048_k{1,4,16,64}_g8`
+ `k4_g32`（全核减归约项的派生行）；rowred simt `n1_s0_b8192_k{4,16,64}_g8`、
`k{4,8,16,64}_g56`（multibuffer=True 体）；masked_sm `n1_s1_b65536_k{4,8,16}_g{8,56}`、
`k64_g8`、`n1_s1_b48000_k{4,8,16,18}_g8`（精确桶，全核减 vecExec+floors）。

---

## 8. 已知边界（如实声明）

1. **A_copy_f32_4k_g64（guard）simd_f1**：calH 的 −640ns 在"2 波但总时长
   仅 2.4µs"的短流上占比过大，导致该组合从基线 +6.5% 变为 −19.9%。
   改进方向：隐藏量按流长比例化（需要新拟合数据），本轮不做。
2. **派生行与 guard 非掩码体共享键空间**：fused 的派生行只注入了
   g8/g32 两个网格，g56 保持旧的非掩码行；网格括号精确命中（hd14 修复）
   保证两者互不串档。若未来出现"g56 + 静态部分掩码 + 2048B"的新几何，
   需要先补该网格的专属行。
3. **混流准入与 Swide 合并的优先级**：见 §1 问答 (a)。
4. **hint>224 与 (nL,nS) 严格匹配**：见 §6 R3/R4（设计取舍，已声明）。
