
---

## 覆盖扩展战役补记(2026-11-14)

### bf16 别名验证
calD_bf16.jsonl(SIMD/SIMT × {1024,4096,16384}B × grid{1,56}):grid=1 口径下
bf16/fp16 边际比 0.95~1.00,确认 bf16 硬件成本与 fp16 一致;StageMemoryAccess
仅含 elementBits=16,两种 dtype 天然共用 fp16 cell,别名成立,无需独立标定。

### 记账修复(全程适用)
- G-double: v2 认领须同步标记 layout warp-instruction 覆盖,否则 simt 解析
  兜底重复收费(cumprod 560.2+486.4=1046.6 案例)。
- G5: 平坦 load 组公式(independent_loop_load_group)自扫几何不查认领列表,
  须在首个被 v2 认领的 access 处截断(exp 93.8+62.0=155.8 案例)。

### 覆盖架构(最终态)
| 连续访存场景 | 计价 | 依据 |
|---|---|---|
| N≥2 同形组 | Formula A 实测表(first+(N-1)*marginal,P 轴=E13) | 组摊薄赢 legacy |
| N=1 单访问 | v2 路径内置 legacy 一致解析式(startup+B/rate;simt 按 warp-instr/rate) | 探针 first 含发射开销,legacy 把它单列;对齐后数值=基线,零回归 |
| K≥32 循环体 | Formula B 实测表(P+K*m,m-斜率 P 轴) | BW 主导域 |
| K<32 / K>128 循环 | legacy 接管(数值=基线) | 启动主导域,m 斜率轴不适用 |
| bf16 | fp16 cell 别名 | 实测一致 |

RUNV17(全开): 95 case 比较,82 PASS 与基线逐位一致,仅 exp 一例经 Formula B
(+7.1%,仍 PASS)。单访问解析覆盖经 RUNV17 验证预测逐位不变。
