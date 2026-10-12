# 版本 1：MODSEL —— 修饰因子逐槽位选择（推荐版本）

每个公式独立前向选择**修饰因子**（dtype 指示、num_warps 指示、物理 regime 标志），每加入一个修饰因子就让它乘遍该公式的所有基特征。项数因此按公式的复杂度自然分化（6-113 项）。

## 汇总表

| 公式 | 场景 | 路径/操作/秩 | 点数 | 项数 | **in-sample MAPE** |
|---|---|---|---:|---:|---:|
| `simd_load_wide_r1` | SIMD load 宽分支（1D） | simd / load / rank-1 | 32 | 6 | **3.46%** |
| `simd_load_wide_r2` | SIMD load 宽分支（2D） | simd / load / rank-2 | 246 | 17 | **17.10%** |
| `simd_load_gather_r1` | SIMD load 聚集分支（1D） | simd / load / rank-1 | 284 | 78 | **12.14%** |
| `simd_load_gather_r2` | SIMD load 聚集分支（2D） | simd / load / rank-2 | 750 | 73 | **11.08%** |
| `simd_store_r1` | SIMD store（1D） | simd / store / rank-1 | 291 | 105 | **4.38%** |
| `simd_store_r2` | SIMD store（2D） | simd / store / rank-2 | 943 | 113 | **5.63%** |
| `simt_load_r1` | 纯 SIMT load（1D） | simt_only / load / rank-1 | 855 | 89 | **7.50%** |
| `simt_load_r2` | 纯 SIMT load（2D） | simt_only / load / rank-2 | 1762 | 81 | **7.24%** |
| `simt_store_r1` | 纯 SIMT store（1D） | simt_only / store / rank-1 | 962 | 113 | **5.47%** |
| `simt_store_r2` | 纯 SIMT store（2D） | simt_only / store / rank-2 | 3416 | 49 | **17.75%** |
| | | | **9541** | | **11.29%** |

> 拟合口径：加权最小二乘，权重 1/T（等价于最小化相对误差，即 MAPE）；in-sample。

---

## `simd_load_wide_r1` — SIMD load 宽分支（1D）

### 1. 目标场景与实测条件构成

compile_mode=simd，tl.load，元素域 stride <= 2，rank-1。CAModel 白盒验证：此时 BIU 连续搬运整块再 extract 抽取，biu_cmds 从元素数降到 1-9，bytes/cmd 从 4B 升到 28-512B。

- **路径 / 操作 / 秩**：simd / load / rank-1
- **dtype**（4 种）：float16(2B)、float32(4B)、int64(8B)、int8(1B)
- **num_warps**（1 种）：1
- **elements**（8 种）：8、16、32、64、128、256、512、4
- **内层步长（字节）**（4 种）：2、4、8、16
- **总点数**：**32**

### 2. 公式与各项物理意义

```
T = 65.4052
  + 2.36499 * ov_L_pages*1
  + -1.88413 * ov_L_pages*smISB
  + 0.695759 * ov_L_pages*e1
  + 0.361965 * ov_L_pages*Lgt4
  + -1.31318 * ov_L_pages*mwlg2
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 65.4052 | 固定开销（循环发射、启动） |
| `ov_L_pages*1` | 2.36499 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [无（常数项）] |
| `ov_L_pages*smISB` | -1.88413 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [inner stride <= 8 bytes] |
| `ov_L_pages*e1` | 0.695759 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [dtype = int8] |
| `ov_L_pages*Lgt4` | 0.361965 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [L > 4] |
| `ov_L_pages*mwlg2` | -1.31318 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [mean_warp_lines > 2] |

### 3. in-sample 拟合误差

- 点数：**32**
- 项数：**6**
- **MAPE：3.46%**

---

## `simd_load_wide_r2` — SIMD load 宽分支（2D）

### 1. 目标场景与实测条件构成

compile_mode=simd，tl.load，内层元素域 stride <= 2，rank-2。CAModel 验证同样的连续块搬运（IS=2 tight 时 biu_cmds=1, B/cmd=512）。

- **路径 / 操作 / 秩**：simd / load / rank-2
- **dtype**（4 种）：float16(2B)、float32(4B)、int64(8B)、int8(1B)
- **num_warps**（1 种）：1
- **形状 (R×C)**（23 种）：4x4、4x8、4x16、8x8、8x16、4x1、16x1、16x16、1x16、1x32、1x64、2x8、2x16、2x32、8x4、16x4、16x8、8x32、4x32、2x64、32x8、16x32、32x16
- **内层步长（字节）**（5 种）：1、2、4、8、16
- **总点数**：**246**

### 2. 公式与各项物理意义

```
T = 75.375
  + -0.073661 * wuSeg*1
  + 0.202536 * wuSeg*smISB
  + 2e-06 * p_R_spanb*1
  + -9.3e-05 * p_R_spanb*smISB
  + -38.3767 * r_R_L*1
  + 42.1779 * r_R_L*smISB
  + -0.406618 * r_C_it*1
  + -0.163069 * r_C_it*smISB
  + -0.86639 * comb_lines*1
  + 1.96123 * comb_lines*smISB
  + -347.872 * r_L_rs*1
  + 335.467 * r_L_rs*smISB
  + 170.418 * r_R_eb*1
  + -168.209 * r_R_eb*smISB
  + 0.019748 * R2*1
  + -0.082181 * R2*smISB
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 75.375 | 固定开销（循环发射、启动） |
| `wuSeg*1` | -0.073661 | wuSeg × [无（常数项）] |
| `wuSeg*smISB` | 0.202536 | wuSeg × [inner stride <= 8 bytes] |
| `p_R_spanb*1` | 2e-06 | p_R_spanb × [无（常数项）] |
| `p_R_spanb*smISB` | -9.3e-05 | p_R_spanb × [inner stride <= 8 bytes] |
| `r_R_L*1` | -38.3767 | r_R_L × [无（常数项）] |
| `r_R_L*smISB` | 42.1779 | r_R_L × [inner stride <= 8 bytes] |
| `r_C_it*1` | -0.406618 | r_C_it × [无（常数项）] |
| `r_C_it*smISB` | -0.163069 | r_C_it × [inner stride <= 8 bytes] |
| `comb_lines*1` | -0.86639 | R × 每行 line 数 × [无（常数项）] |
| `comb_lines*smISB` | 1.96123 | R × 每行 line 数 × [inner stride <= 8 bytes] |
| `r_L_rs*1` | -347.872 | r_L_rs × [无（常数项）] |
| `r_L_rs*smISB` | 335.467 | r_L_rs × [inner stride <= 8 bytes] |
| `r_R_eb*1` | 170.418 | r_R_eb × [无（常数项）] |
| `r_R_eb*smISB` | -168.209 | r_R_eb × [inner stride <= 8 bytes] |
| `R2*1` | 0.019748 | R² × [无（常数项）] |
| `R2*smISB` | -0.082181 | R² × [inner stride <= 8 bytes] |

### 3. in-sample 拟合误差

- 点数：**246**
- 项数：**17**
- **MAPE：17.10%**

---

## `simd_load_gather_r1` — SIMD load 聚集分支（1D）

### 1. 目标场景与实测条件构成

compile_mode=simd，tl.load，元素域 stride > 2，rank-1。每个元素一条 BIU 命令（biu_send_sizes = {eb: n}）。

- **路径 / 操作 / 秩**：simd / load / rank-1
- **dtype**（4 种）：float16(2B)、float32(4B)、int64(8B)、int8(1B)
- **num_warps**（1 种）：1
- **elements**（8 种）：16、128、8、256、4、512、32、64
- **内层步长（字节）**（19 种）：3、4、5、6、8、10、12、16、20、24、32、40、48、64、128、256、512、1024、2048
- **总点数**：**284**

### 2. 公式与各项物理意义

```
T = -31.3963
  + -1.63807 * n2L*1
  + 0.076786 * n2L*mwlg8
  + 2.19132 * n2L*e1
  + -0.006371 * n2L*Lgt16
  + -5.042 * n2L*e8
  + 0.171428 * n2L*Clt32
  + -0.083103 * n2L*smISB
  + 1.91566 * n2L*e2
  + 0.141212 * n2L*Clt64
  + 0.04637 * n2L*nge256
  + -0.038867 * n2L*Lgt4
  + 4.82583 * ov_n_mwl*1
  + 1.72746 * ov_n_mwl*mwlg8
  + 0.341237 * ov_n_mwl*e1
  + -14.043 * ov_n_mwl*Lgt16
  + -1.86461 * ov_n_mwl*e8
  + -4.81602 * ov_n_mwl*Clt32
  + 26.2899 * ov_n_mwl*smISB
  + 0.53184 * ov_n_mwl*e2
  + -3.72338 * ov_n_mwl*Clt64
  + 0.774539 * ov_n_mwl*nge256
  + 3.40798 * ov_n_mwl*Lgt4
  + 312.794 * p_T_it*1
  + -1.20519 * p_T_it*mwlg8
  + 3.1386 * p_T_it*e1
  + -313.979 * p_T_it*Lgt16
  + -1.35927 * p_T_it*e8
  + -310.838 * p_T_it*Clt32
  + -3.23297 * p_T_it*smISB
  + 2.49015 * p_T_it*e2
  + 3.30436 * p_T_it*Clt64
  + -0.736949 * p_T_it*nge256
  + -0.408009 * p_T_it*Lgt4
  + 3.00755 * Lepl*1
  + -9.32735 * Lepl*mwlg8
  + 1.88384 * Lepl*e1
  + -7.07642 * Lepl*Lgt16
  + -5.15794 * Lepl*e8
  + -9.51215 * Lepl*Clt32
  + 11.2246 * Lepl*smISB
  + 1.82008 * Lepl*e2
  + -5.10177 * Lepl*Clt64
  + 0.102952 * Lepl*nge256
  + 11.9171 * Lepl*Lgt4
  + -321.386 * bucket_worst_2k8*1
  + 1.00801 * bucket_worst_2k8*mwlg8
  + 1.9412 * bucket_worst_2k8*e1
  + 338.244 * bucket_worst_2k8*Lgt16
  + -0.28358 * bucket_worst_2k8*e8
  + 335.033 * bucket_worst_2k8*Clt32
  + 1.67203 * bucket_worst_2k8*smISB
  + 0.703224 * bucket_worst_2k8*e2
  + 5.36083 * bucket_worst_2k8*Clt64
  + -0.959231 * bucket_worst_2k8*nge256
  + -16.9183 * bucket_worst_2k8*Lgt4
  + -60.9254 * Wn*1
  + 276.573 * Wn*mwlg8
  + -200.955 * Wn*e1
  + -9.81184 * Wn*Lgt16
  + 262.381 * Wn*e8
  + -80.4139 * Wn*Clt32
  + -1080.06 * Wn*smISB
  + -160.848 * Wn*e2
  + -70.5969 * Wn*Clt64
  + -0.02303 * Wn*nge256
  + 90.7374 * Wn*Lgt4
  + 0.477382 * nEpl_eb*1
  + -0.002241 * nEpl_eb*mwlg8
  + -0.833842 * nEpl_eb*e1
  + -0.026108 * nEpl_eb*Lgt16
  + 0.410781 * nEpl_eb*e8
  + 0.088239 * nEpl_eb*Clt32
  + -0.001419 * nEpl_eb*smISB
  + -0.506645 * nEpl_eb*e2
  + -0.018937 * nEpl_eb*Clt64
  + -0.016861 * nEpl_eb*nge256
  + 0.016745 * nEpl_eb*Lgt4
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | -31.3963 | 固定开销（循环发射、启动） |
| `n2L*1` | -1.63807 | n² / L × [无（常数项）] |
| `n2L*mwlg8` | 0.076786 | n² / L × [mean_warp_lines > 8] |
| `n2L*e1` | 2.19132 | n² / L × [dtype = int8] |
| `n2L*Lgt16` | -0.006371 | n² / L × [L > 16] |
| `n2L*e8` | -5.042 | n² / L × [dtype = int64] |
| `n2L*Clt32` | 0.171428 | n² / L × [C < 32] |
| `n2L*smISB` | -0.083103 | n² / L × [inner stride <= 8 bytes] |
| `n2L*e2` | 1.91566 | n² / L × [dtype = float16] |
| `n2L*Clt64` | 0.141212 | n² / L × [C < 64] |
| `n2L*nge256` | 0.04637 | n² / L × [n >= 256] |
| `n2L*Lgt4` | -0.038867 | n² / L × [L > 4] |
| `ov_n_mwl*1` | 4.82583 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [无（常数项）] |
| `ov_n_mwl*mwlg8` | 1.72746 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [mean_warp_lines > 8] |
| `ov_n_mwl*e1` | 0.341237 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = int8] |
| `ov_n_mwl*Lgt16` | -14.043 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [L > 16] |
| `ov_n_mwl*e8` | -1.86461 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = int64] |
| `ov_n_mwl*Clt32` | -4.81602 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [C < 32] |
| `ov_n_mwl*smISB` | 26.2899 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [inner stride <= 8 bytes] |
| `ov_n_mwl*e2` | 0.53184 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = float16] |
| `ov_n_mwl*Clt64` | -3.72338 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [C < 64] |
| `ov_n_mwl*nge256` | 0.774539 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [n >= 256] |
| `ov_n_mwl*Lgt4` | 3.40798 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [L > 4] |
| `p_T_it*1` | 312.794 | p_T_it × [无（常数项）] |
| `p_T_it*mwlg8` | -1.20519 | p_T_it × [mean_warp_lines > 8] |
| `p_T_it*e1` | 3.1386 | p_T_it × [dtype = int8] |
| `p_T_it*Lgt16` | -313.979 | p_T_it × [L > 16] |
| `p_T_it*e8` | -1.35927 | p_T_it × [dtype = int64] |
| `p_T_it*Clt32` | -310.838 | p_T_it × [C < 32] |
| `p_T_it*smISB` | -3.23297 | p_T_it × [inner stride <= 8 bytes] |
| `p_T_it*e2` | 2.49015 | p_T_it × [dtype = float16] |
| `p_T_it*Clt64` | 3.30436 | p_T_it × [C < 64] |
| `p_T_it*nge256` | -0.736949 | p_T_it × [n >= 256] |
| `p_T_it*Lgt4` | -0.408009 | p_T_it × [L > 4] |
| `Lepl*1` | 3.00755 | L · epl × [无（常数项）] |
| `Lepl*mwlg8` | -9.32735 | L · epl × [mean_warp_lines > 8] |
| `Lepl*e1` | 1.88384 | L · epl × [dtype = int8] |
| `Lepl*Lgt16` | -7.07642 | L · epl × [L > 16] |
| `Lepl*e8` | -5.15794 | L · epl × [dtype = int64] |
| `Lepl*Clt32` | -9.51215 | L · epl × [C < 32] |
| `Lepl*smISB` | 11.2246 | L · epl × [inner stride <= 8 bytes] |
| `Lepl*e2` | 1.82008 | L · epl × [dtype = float16] |
| `Lepl*Clt64` | -5.10177 | L · epl × [C < 64] |
| `Lepl*nge256` | 0.102952 | L · epl × [n >= 256] |
| `Lepl*Lgt4` | 11.9171 | L · epl × [L > 4] |
| `bucket_worst_2k8*1` | -321.386 | 最差 line 桶 (2048..8192) × [无（常数项）] |
| `bucket_worst_2k8*mwlg8` | 1.00801 | 最差 line 桶 (2048..8192) × [mean_warp_lines > 8] |
| `bucket_worst_2k8*e1` | 1.9412 | 最差 line 桶 (2048..8192) × [dtype = int8] |
| `bucket_worst_2k8*Lgt16` | 338.244 | 最差 line 桶 (2048..8192) × [L > 16] |
| `bucket_worst_2k8*e8` | -0.28358 | 最差 line 桶 (2048..8192) × [dtype = int64] |
| `bucket_worst_2k8*Clt32` | 335.033 | 最差 line 桶 (2048..8192) × [C < 32] |
| `bucket_worst_2k8*smISB` | 1.67203 | 最差 line 桶 (2048..8192) × [inner stride <= 8 bytes] |
| `bucket_worst_2k8*e2` | 0.703224 | 最差 line 桶 (2048..8192) × [dtype = float16] |
| `bucket_worst_2k8*Clt64` | 5.36083 | 最差 line 桶 (2048..8192) × [C < 64] |
| `bucket_worst_2k8*nge256` | -0.959231 | 最差 line 桶 (2048..8192) × [n >= 256] |
| `bucket_worst_2k8*Lgt4` | -16.9183 | 最差 line 桶 (2048..8192) × [L > 4] |
| `Wn*1` | -60.9254 | W / n × [无（常数项）] |
| `Wn*mwlg8` | 276.573 | W / n × [mean_warp_lines > 8] |
| `Wn*e1` | -200.955 | W / n × [dtype = int8] |
| `Wn*Lgt16` | -9.81184 | W / n × [L > 16] |
| `Wn*e8` | 262.381 | W / n × [dtype = int64] |
| `Wn*Clt32` | -80.4139 | W / n × [C < 32] |
| `Wn*smISB` | -1080.06 | W / n × [inner stride <= 8 bytes] |
| `Wn*e2` | -160.848 | W / n × [dtype = float16] |
| `Wn*Clt64` | -70.5969 | W / n × [C < 64] |
| `Wn*nge256` | -0.02303 | W / n × [n >= 256] |
| `Wn*Lgt4` | 90.7374 | W / n × [L > 4] |
| `nEpl_eb*1` | 0.477382 | n·epl·eb × [无（常数项）] |
| `nEpl_eb*mwlg8` | -0.002241 | n·epl·eb × [mean_warp_lines > 8] |
| `nEpl_eb*e1` | -0.833842 | n·epl·eb × [dtype = int8] |
| `nEpl_eb*Lgt16` | -0.026108 | n·epl·eb × [L > 16] |
| `nEpl_eb*e8` | 0.410781 | n·epl·eb × [dtype = int64] |
| `nEpl_eb*Clt32` | 0.088239 | n·epl·eb × [C < 32] |
| `nEpl_eb*smISB` | -0.001419 | n·epl·eb × [inner stride <= 8 bytes] |
| `nEpl_eb*e2` | -0.506645 | n·epl·eb × [dtype = float16] |
| `nEpl_eb*Clt64` | -0.018937 | n·epl·eb × [C < 64] |
| `nEpl_eb*nge256` | -0.016861 | n·epl·eb × [n >= 256] |
| `nEpl_eb*Lgt4` | 0.016745 | n·epl·eb × [L > 4] |

### 3. in-sample 拟合误差

- 点数：**284**
- 项数：**78**
- **MAPE：12.14%**

---

## `simd_load_gather_r2` — SIMD load 聚集分支（2D）

### 1. 目标场景与实测条件构成

compile_mode=simd，tl.load，内层元素域 stride > 2，rank-2。每元素一条 BIU 命令。

- **路径 / 操作 / 秩**：simd / load / rank-2
- **dtype**（4 种）：float16(2B)、float32(4B)、int64(8B)、int8(1B)
- **num_warps**（1 种）：1
- **形状 (R×C)**（21 种）：4x4、4x8、4x16、8x8、8x16、16x16、1x16、1x32、1x64、2x8、2x16、2x32、8x4、2x64、4x32、16x32、8x32、32x16、16x4、16x8、32x8
- **内层步长（字节）**（10 种）：3、4、6、8、12、16、24、32、64、128
- **总点数**：**750**

### 2. 公式与各项物理意义

```
T = 102.262
  + 55.7247 * mn_L_cshortN*1
  + -66.6933 * mn_L_cshortN*mwlg8
  + -11.7117 * mn_L_cshortN*smISB
  + 55.7247 * mn_L_cshortN*Clt32
  + -19.0815 * mn_L_cshortN*Rgt8
  + 27.3476 * mn_L_cshortN*e4
  + -19.2611 * mn_L_cshortN*Lgt16
  + -15.4806 * mn_L_cshortN*Clt8
  + 18.2957 * mn_L_cshortN*nlt64
  + 0.211496 * n2L*1
  + 0.214888 * n2L*mwlg8
  + -0.042822 * n2L*smISB
  + -0.105959 * n2L*Clt32
  + 0.031853 * n2L*Rgt8
  + -0.226706 * n2L*e4
  + -0.026911 * n2L*Lgt16
  + -0.425475 * n2L*Clt8
  + 0.182099 * n2L*nlt64
  + -0.293513 * nhe4*1
  + 0.147027 * nhe4*smISB
  + 0.135184 * nhe4*Clt32
  + 0.078195 * nhe4*Rgt8
  + 0.020356 * nhe4*e4
  + 0.022985 * nhe4*Lgt16
  + -0.001662 * nhe4*Clt8
  + -0.104492 * nhe4*nlt64
  + 0.010173 * nEpl_eb*1
  + 0.021915 * nEpl_eb*mwlg8
  + 0.002654 * nEpl_eb*smISB
  + -0.009766 * nEpl_eb*Clt32
  + 0.008905 * nEpl_eb*Rgt8
  + 0.048526 * nEpl_eb*e4
  + -1e-05 * nEpl_eb*Lgt16
  + 0.069232 * nEpl_eb*Clt8
  + 0.005605 * nEpl_eb*nlt64
  + 33.2539 * r_C_ISB*1
  + 16.9886 * r_C_ISB*mwlg8
  + -23.4159 * r_C_ISB*smISB
  + -37.7153 * r_C_ISB*Clt32
  + -196.087 * r_C_ISB*Rgt8
  + -4.90479 * r_C_ISB*e4
  + -37.8574 * r_C_ISB*Lgt16
  + 36.2744 * r_C_ISB*Clt8
  + 15.05 * r_C_ISB*nlt64
  + -47.5495 * burstShort*1
  + 548.506 * burstShort*mwlg8
  + -68.0956 * burstShort*smISB
  + -47.5495 * burstShort*Clt32
  + 250.653 * burstShort*Rgt8
  + -166.583 * burstShort*e4
  + 234.037 * burstShort*Lgt16
  + 3.09829 * burstShort*Clt8
  + 10.2037 * burstShort*nlt64
  + 12.186 * ov_seg_mwl*1
  + -5.15386 * ov_seg_mwl*mwlg8
  + -9.04299 * ov_seg_mwl*smISB
  + -9.16665 * ov_seg_mwl*Clt32
  + 1.62695 * ov_seg_mwl*Rgt8
  + 4.02644 * ov_seg_mwl*e4
  + 5.40678 * ov_seg_mwl*Lgt16
  + 9.05499 * ov_seg_mwl*Clt8
  + 55.6958 * ov_seg_mwl*nlt64
  + 2.49388 * line_elems_max*1
  + -16.2788 * line_elems_max*mwlg8
  + 3.67402 * line_elems_max*smISB
  + 8.48199 * line_elems_max*Clt32
  + 7.71295 * line_elems_max*Rgt8
  + 0.887634 * line_elems_max*e4
  + 7.45362 * line_elems_max*Lgt16
  + -5.11679 * line_elems_max*Clt8
  + -7.29933 * line_elems_max*nlt64
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 102.262 | 固定开销（循环发射、启动） |
| `mn_L_cshortN*1` | 55.7247 | min(触及的 128B line 数, cshort · n / 32) × [无（常数项）] |
| `mn_L_cshortN*mwlg8` | -66.6933 | min(触及的 128B line 数, cshort · n / 32) × [mean_warp_lines > 8] |
| `mn_L_cshortN*smISB` | -11.7117 | min(触及的 128B line 数, cshort · n / 32) × [inner stride <= 8 bytes] |
| `mn_L_cshortN*Clt32` | 55.7247 | min(触及的 128B line 数, cshort · n / 32) × [C < 32] |
| `mn_L_cshortN*Rgt8` | -19.0815 | min(触及的 128B line 数, cshort · n / 32) × [R > 8] |
| `mn_L_cshortN*e4` | 27.3476 | min(触及的 128B line 数, cshort · n / 32) × [dtype = float32] |
| `mn_L_cshortN*Lgt16` | -19.2611 | min(触及的 128B line 数, cshort · n / 32) × [L > 16] |
| `mn_L_cshortN*Clt8` | -15.4806 | min(触及的 128B line 数, cshort · n / 32) × [C < 8] |
| `mn_L_cshortN*nlt64` | 18.2957 | min(触及的 128B line 数, cshort · n / 32) × [n < 64] |
| `n2L*1` | 0.211496 | n² / L × [无（常数项）] |
| `n2L*mwlg8` | 0.214888 | n² / L × [mean_warp_lines > 8] |
| `n2L*smISB` | -0.042822 | n² / L × [inner stride <= 8 bytes] |
| `n2L*Clt32` | -0.105959 | n² / L × [C < 32] |
| `n2L*Rgt8` | 0.031853 | n² / L × [R > 8] |
| `n2L*e4` | -0.226706 | n² / L × [dtype = float32] |
| `n2L*Lgt16` | -0.026911 | n² / L × [L > 16] |
| `n2L*Clt8` | -0.425475 | n² / L × [C < 8] |
| `n2L*nlt64` | 0.182099 | n² / L × [n < 64] |
| `nhe4*1` | -0.293513 | n · max(0, 128/ISB − 4) × [无（常数项）] |
| `nhe4*smISB` | 0.147027 | n · max(0, 128/ISB − 4) × [inner stride <= 8 bytes] |
| `nhe4*Clt32` | 0.135184 | n · max(0, 128/ISB − 4) × [C < 32] |
| `nhe4*Rgt8` | 0.078195 | n · max(0, 128/ISB − 4) × [R > 8] |
| `nhe4*e4` | 0.020356 | n · max(0, 128/ISB − 4) × [dtype = float32] |
| `nhe4*Lgt16` | 0.022985 | n · max(0, 128/ISB − 4) × [L > 16] |
| `nhe4*Clt8` | -0.001662 | n · max(0, 128/ISB − 4) × [C < 8] |
| `nhe4*nlt64` | -0.104492 | n · max(0, 128/ISB − 4) × [n < 64] |
| `nEpl_eb*1` | 0.010173 | n·epl·eb × [无（常数项）] |
| `nEpl_eb*mwlg8` | 0.021915 | n·epl·eb × [mean_warp_lines > 8] |
| `nEpl_eb*smISB` | 0.002654 | n·epl·eb × [inner stride <= 8 bytes] |
| `nEpl_eb*Clt32` | -0.009766 | n·epl·eb × [C < 32] |
| `nEpl_eb*Rgt8` | 0.008905 | n·epl·eb × [R > 8] |
| `nEpl_eb*e4` | 0.048526 | n·epl·eb × [dtype = float32] |
| `nEpl_eb*Lgt16` | -1e-05 | n·epl·eb × [L > 16] |
| `nEpl_eb*Clt8` | 0.069232 | n·epl·eb × [C < 8] |
| `nEpl_eb*nlt64` | 0.005605 | n·epl·eb × [n < 64] |
| `r_C_ISB*1` | 33.2539 | r_C_ISB × [无（常数项）] |
| `r_C_ISB*mwlg8` | 16.9886 | r_C_ISB × [mean_warp_lines > 8] |
| `r_C_ISB*smISB` | -23.4159 | r_C_ISB × [inner stride <= 8 bytes] |
| `r_C_ISB*Clt32` | -37.7153 | r_C_ISB × [C < 32] |
| `r_C_ISB*Rgt8` | -196.087 | r_C_ISB × [R > 8] |
| `r_C_ISB*e4` | -4.90479 | r_C_ISB × [dtype = float32] |
| `r_C_ISB*Lgt16` | -37.8574 | r_C_ISB × [L > 16] |
| `r_C_ISB*Clt8` | 36.2744 | r_C_ISB × [C < 8] |
| `r_C_ISB*nlt64` | 15.05 | r_C_ISB × [n < 64] |
| `burstShort*1` | -47.5495 | [C·eb < 32] 指示 × [无（常数项）] |
| `burstShort*mwlg8` | 548.506 | [C·eb < 32] 指示 × [mean_warp_lines > 8] |
| `burstShort*smISB` | -68.0956 | [C·eb < 32] 指示 × [inner stride <= 8 bytes] |
| `burstShort*Clt32` | -47.5495 | [C·eb < 32] 指示 × [C < 32] |
| `burstShort*Rgt8` | 250.653 | [C·eb < 32] 指示 × [R > 8] |
| `burstShort*e4` | -166.583 | [C·eb < 32] 指示 × [dtype = float32] |
| `burstShort*Lgt16` | 234.037 | [C·eb < 32] 指示 × [L > 16] |
| `burstShort*Clt8` | 3.09829 | [C·eb < 32] 指示 × [C < 8] |
| `burstShort*nlt64` | 10.2037 | [C·eb < 32] 指示 × [n < 64] |
| `ov_seg_mwl*1` | 12.186 | ov_seg_mwl × [无（常数项）] |
| `ov_seg_mwl*mwlg8` | -5.15386 | ov_seg_mwl × [mean_warp_lines > 8] |
| `ov_seg_mwl*smISB` | -9.04299 | ov_seg_mwl × [inner stride <= 8 bytes] |
| `ov_seg_mwl*Clt32` | -9.16665 | ov_seg_mwl × [C < 32] |
| `ov_seg_mwl*Rgt8` | 1.62695 | ov_seg_mwl × [R > 8] |
| `ov_seg_mwl*e4` | 4.02644 | ov_seg_mwl × [dtype = float32] |
| `ov_seg_mwl*Lgt16` | 5.40678 | ov_seg_mwl × [L > 16] |
| `ov_seg_mwl*Clt8` | 9.05499 | ov_seg_mwl × [C < 8] |
| `ov_seg_mwl*nlt64` | 55.6958 | ov_seg_mwl × [n < 64] |
| `line_elems_max*1` | 2.49388 | 单 line 内最多元素数 × [无（常数项）] |
| `line_elems_max*mwlg8` | -16.2788 | 单 line 内最多元素数 × [mean_warp_lines > 8] |
| `line_elems_max*smISB` | 3.67402 | 单 line 内最多元素数 × [inner stride <= 8 bytes] |
| `line_elems_max*Clt32` | 8.48199 | 单 line 内最多元素数 × [C < 32] |
| `line_elems_max*Rgt8` | 7.71295 | 单 line 内最多元素数 × [R > 8] |
| `line_elems_max*e4` | 0.887634 | 单 line 内最多元素数 × [dtype = float32] |
| `line_elems_max*Lgt16` | 7.45362 | 单 line 内最多元素数 × [L > 16] |
| `line_elems_max*Clt8` | -5.11679 | 单 line 内最多元素数 × [C < 8] |
| `line_elems_max*nlt64` | -7.29933 | 单 line 内最多元素数 × [n < 64] |

### 3. in-sample 拟合误差

- 点数：**750**
- 项数：**73**
- **MAPE：11.08%**

---

## `simd_store_r1` — SIMD store（1D）

### 1. 目标场景与实测条件构成

compile_mode=simd，tl.store，rank-1（MTE3 写路径）。

- **路径 / 操作 / 秩**：simd / store / rank-1
- **dtype**（4 种）：float16(2B)、float32(4B)、int64(8B)、int8(1B)
- **num_warps**（1 种）：1
- **elements**（8 种）：16、4、8、32、256、64、512、128
- **内层步长（字节）**（19 种）：2、3、4、5、6、8、10、12、16、24、32、40、48、64、128、256、512、1024、2048
- **总点数**：**291**

### 2. 公式与各项物理意义

```
T = 503.798
  + 94.6964 * line_elems_max*1
  + 29.6889 * line_elems_max*e8
  + -42.544 * line_elems_max*nge256
  + -43.8383 * line_elems_max*e1
  + 14.6839 * line_elems_max*mwlg2
  + -135.806 * line_elems_max*Lgt4
  + 111.847 * line_elems_max*mwlg8
  + -250.355 * line_elems_max*Lgt64
  + 54.1501 * line_elems_max*Lgt16
  + 38.6607 * line_elems_max*nge128
  + -18.6924 * line_elems_max*Clt16
  + -10.3297 * line_elems_max*Clt8
  + 37.6398 * line_elems_max*smISB
  + 2.00104 * n2L*1
  + -1.40997 * n2L*e8
  + -0.690993 * n2L*nge256
  + 0.069023 * n2L*e1
  + -2.59444 * n2L*mwlg2
  + 0.232267 * n2L*Lgt4
  + 0.21913 * n2L*mwlg8
  + 0.264168 * n2L*Lgt64
  + 0.121129 * n2L*Lgt16
  + 1.22483 * n2L*nge128
  + 15.7605 * n2L*Clt16
  + -12.6635 * n2L*Clt8
  + -2.35825 * n2L*smISB
  + 9.48899 * C*1
  + 1.20494 * C*e8
  + 1.25235 * C*nge256
  + -0.564107 * C*e1
  + -0.010717 * C*mwlg2
  + -13.0231 * C*Lgt4
  + 5.45344 * C*mwlg8
  + -2.4302 * C*Lgt64
  + -3.31708 * C*Lgt16
  + 2.4025 * C*nge128
  + -0.119301 * C*Clt16
  + 0.03969 * C*Clt8
  + 3.63763 * C*smISB
  + -3.28568 * r_C_rl*1
  + -25.9449 * r_C_rl*e8
  + 120.85 * r_C_rl*nge256
  + 44.1768 * r_C_rl*e1
  + -17.6497 * r_C_rl*mwlg2
  + 186.274 * r_C_rl*Lgt4
  + -183.24 * r_C_rl*mwlg8
  + 254.985 * r_C_rl*Lgt64
  + -61.2086 * r_C_rl*Lgt16
  + -171.195 * r_C_rl*nge128
  + -120.294 * r_C_rl*Clt16
  + 164.915 * r_C_rl*Clt8
  + -19.4899 * r_C_rl*smISB
  + 0.000665 * nEpl_eb*1
  + 0.142781 * nEpl_eb*e8
  + 0.030528 * nEpl_eb*nge256
  + -0.027001 * nEpl_eb*e1
  + 0.021885 * nEpl_eb*mwlg2
  + -0.023255 * nEpl_eb*Lgt4
  + -0.043822 * nEpl_eb*mwlg8
  + 0.004521 * nEpl_eb*Lgt64
  + -0.02225 * nEpl_eb*Lgt16
  + 0.017575 * nEpl_eb*nge128
  + 0.026417 * nEpl_eb*Clt16
  + 0.048714 * nEpl_eb*Clt8
  + -0.007531 * nEpl_eb*smISB
  + -22.5412 * le_isb32xN*1
  + -1.01468 * le_isb32xN*e8
  + -1.10716 * le_isb32xN*nge256
  + 0.265902 * le_isb32xN*e1
  + 14.11 * le_isb32xN*mwlg2
  + 11.4668 * le_isb32xN*Lgt4
  + 2.80497 * le_isb32xN*Lgt64
  + 5.67082 * le_isb32xN*Lgt16
  + 0.219045 * le_isb32xN*nge128
  + 24.1449 * le_isb32xN*Clt16
  + -19.6181 * le_isb32xN*Clt8
  + 3.63763 * le_isb32xN*smISB
  + -20.3585 * r_T_it*1
  + -0.70691 * r_T_it*e8
  + -50.0687 * r_T_it*nge256
  + 0.43173 * r_T_it*e1
  + 2.06408 * r_T_it*mwlg2
  + 4.86633 * r_T_it*Lgt4
  + -2.26898 * r_T_it*mwlg8
  + 67.8974 * r_T_it*Lgt64
  + 5.64257 * r_T_it*Lgt16
  + -49.9968 * r_T_it*nge128
  + -0.318446 * r_T_it*Clt16
  + 0.317519 * r_T_it*Clt8
  + -2.66106 * r_T_it*smISB
  + 9.50956 * le_L128xN*1
  + 0.766592 * le_L128xN*e8
  + 1.27293 * le_L128xN*nge256
  + -0.532149 * le_L128xN*e1
  + 0.009858 * le_L128xN*mwlg2
  + -13.0025 * le_L128xN*Lgt4
  + 5.47401 * le_L128xN*mwlg8
  + -2.40963 * le_L128xN*Lgt64
  + -3.29651 * le_L128xN*Lgt16
  + 2.42307 * le_L128xN*nge128
  + -0.119301 * le_L128xN*Clt16
  + 0.03969 * le_L128xN*Clt8
  + 3.63763 * le_L128xN*smISB
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 503.798 | 固定开销（循环发射、启动） |
| `line_elems_max*1` | 94.6964 | 单 line 内最多元素数 × [无（常数项）] |
| `line_elems_max*e8` | 29.6889 | 单 line 内最多元素数 × [dtype = int64] |
| `line_elems_max*nge256` | -42.544 | 单 line 内最多元素数 × [n >= 256] |
| `line_elems_max*e1` | -43.8383 | 单 line 内最多元素数 × [dtype = int8] |
| `line_elems_max*mwlg2` | 14.6839 | 单 line 内最多元素数 × [mean_warp_lines > 2] |
| `line_elems_max*Lgt4` | -135.806 | 单 line 内最多元素数 × [L > 4] |
| `line_elems_max*mwlg8` | 111.847 | 单 line 内最多元素数 × [mean_warp_lines > 8] |
| `line_elems_max*Lgt64` | -250.355 | 单 line 内最多元素数 × [L > 64] |
| `line_elems_max*Lgt16` | 54.1501 | 单 line 内最多元素数 × [L > 16] |
| `line_elems_max*nge128` | 38.6607 | 单 line 内最多元素数 × [n >= 128] |
| `line_elems_max*Clt16` | -18.6924 | 单 line 内最多元素数 × [C < 16] |
| `line_elems_max*Clt8` | -10.3297 | 单 line 内最多元素数 × [C < 8] |
| `line_elems_max*smISB` | 37.6398 | 单 line 内最多元素数 × [inner stride <= 8 bytes] |
| `n2L*1` | 2.00104 | n² / L × [无（常数项）] |
| `n2L*e8` | -1.40997 | n² / L × [dtype = int64] |
| `n2L*nge256` | -0.690993 | n² / L × [n >= 256] |
| `n2L*e1` | 0.069023 | n² / L × [dtype = int8] |
| `n2L*mwlg2` | -2.59444 | n² / L × [mean_warp_lines > 2] |
| `n2L*Lgt4` | 0.232267 | n² / L × [L > 4] |
| `n2L*mwlg8` | 0.21913 | n² / L × [mean_warp_lines > 8] |
| `n2L*Lgt64` | 0.264168 | n² / L × [L > 64] |
| `n2L*Lgt16` | 0.121129 | n² / L × [L > 16] |
| `n2L*nge128` | 1.22483 | n² / L × [n >= 128] |
| `n2L*Clt16` | 15.7605 | n² / L × [C < 16] |
| `n2L*Clt8` | -12.6635 | n² / L × [C < 8] |
| `n2L*smISB` | -2.35825 | n² / L × [inner stride <= 8 bytes] |
| `C*1` | 9.48899 | C × [无（常数项）] |
| `C*e8` | 1.20494 | C × [dtype = int64] |
| `C*nge256` | 1.25235 | C × [n >= 256] |
| `C*e1` | -0.564107 | C × [dtype = int8] |
| `C*mwlg2` | -0.010717 | C × [mean_warp_lines > 2] |
| `C*Lgt4` | -13.0231 | C × [L > 4] |
| `C*mwlg8` | 5.45344 | C × [mean_warp_lines > 8] |
| `C*Lgt64` | -2.4302 | C × [L > 64] |
| `C*Lgt16` | -3.31708 | C × [L > 16] |
| `C*nge128` | 2.4025 | C × [n >= 128] |
| `C*Clt16` | -0.119301 | C × [C < 16] |
| `C*Clt8` | 0.03969 | C × [C < 8] |
| `C*smISB` | 3.63763 | C × [inner stride <= 8 bytes] |
| `r_C_rl*1` | -3.28568 | r_C_rl × [无（常数项）] |
| `r_C_rl*e8` | -25.9449 | r_C_rl × [dtype = int64] |
| `r_C_rl*nge256` | 120.85 | r_C_rl × [n >= 256] |
| `r_C_rl*e1` | 44.1768 | r_C_rl × [dtype = int8] |
| `r_C_rl*mwlg2` | -17.6497 | r_C_rl × [mean_warp_lines > 2] |
| `r_C_rl*Lgt4` | 186.274 | r_C_rl × [L > 4] |
| `r_C_rl*mwlg8` | -183.24 | r_C_rl × [mean_warp_lines > 8] |
| `r_C_rl*Lgt64` | 254.985 | r_C_rl × [L > 64] |
| `r_C_rl*Lgt16` | -61.2086 | r_C_rl × [L > 16] |
| `r_C_rl*nge128` | -171.195 | r_C_rl × [n >= 128] |
| `r_C_rl*Clt16` | -120.294 | r_C_rl × [C < 16] |
| `r_C_rl*Clt8` | 164.915 | r_C_rl × [C < 8] |
| `r_C_rl*smISB` | -19.4899 | r_C_rl × [inner stride <= 8 bytes] |
| `nEpl_eb*1` | 0.000665 | n·epl·eb × [无（常数项）] |
| `nEpl_eb*e8` | 0.142781 | n·epl·eb × [dtype = int64] |
| `nEpl_eb*nge256` | 0.030528 | n·epl·eb × [n >= 256] |
| `nEpl_eb*e1` | -0.027001 | n·epl·eb × [dtype = int8] |
| `nEpl_eb*mwlg2` | 0.021885 | n·epl·eb × [mean_warp_lines > 2] |
| `nEpl_eb*Lgt4` | -0.023255 | n·epl·eb × [L > 4] |
| `nEpl_eb*mwlg8` | -0.043822 | n·epl·eb × [mean_warp_lines > 8] |
| `nEpl_eb*Lgt64` | 0.004521 | n·epl·eb × [L > 64] |
| `nEpl_eb*Lgt16` | -0.02225 | n·epl·eb × [L > 16] |
| `nEpl_eb*nge128` | 0.017575 | n·epl·eb × [n >= 128] |
| `nEpl_eb*Clt16` | 0.026417 | n·epl·eb × [C < 16] |
| `nEpl_eb*Clt8` | 0.048714 | n·epl·eb × [C < 8] |
| `nEpl_eb*smISB` | -0.007531 | n·epl·eb × [inner stride <= 8 bytes] |
| `le_isb32xN*1` | -22.5412 | le_isb32xN × [无（常数项）] |
| `le_isb32xN*e8` | -1.01468 | le_isb32xN × [dtype = int64] |
| `le_isb32xN*nge256` | -1.10716 | le_isb32xN × [n >= 256] |
| `le_isb32xN*e1` | 0.265902 | le_isb32xN × [dtype = int8] |
| `le_isb32xN*mwlg2` | 14.11 | le_isb32xN × [mean_warp_lines > 2] |
| `le_isb32xN*Lgt4` | 11.4668 | le_isb32xN × [L > 4] |
| `le_isb32xN*Lgt64` | 2.80497 | le_isb32xN × [L > 64] |
| `le_isb32xN*Lgt16` | 5.67082 | le_isb32xN × [L > 16] |
| `le_isb32xN*nge128` | 0.219045 | le_isb32xN × [n >= 128] |
| `le_isb32xN*Clt16` | 24.1449 | le_isb32xN × [C < 16] |
| `le_isb32xN*Clt8` | -19.6181 | le_isb32xN × [C < 8] |
| `le_isb32xN*smISB` | 3.63763 | le_isb32xN × [inner stride <= 8 bytes] |
| `r_T_it*1` | -20.3585 | r_T_it × [无（常数项）] |
| `r_T_it*e8` | -0.70691 | r_T_it × [dtype = int64] |
| `r_T_it*nge256` | -50.0687 | r_T_it × [n >= 256] |
| `r_T_it*e1` | 0.43173 | r_T_it × [dtype = int8] |
| `r_T_it*mwlg2` | 2.06408 | r_T_it × [mean_warp_lines > 2] |
| `r_T_it*Lgt4` | 4.86633 | r_T_it × [L > 4] |
| `r_T_it*mwlg8` | -2.26898 | r_T_it × [mean_warp_lines > 8] |
| `r_T_it*Lgt64` | 67.8974 | r_T_it × [L > 64] |
| `r_T_it*Lgt16` | 5.64257 | r_T_it × [L > 16] |
| `r_T_it*nge128` | -49.9968 | r_T_it × [n >= 128] |
| `r_T_it*Clt16` | -0.318446 | r_T_it × [C < 16] |
| `r_T_it*Clt8` | 0.317519 | r_T_it × [C < 8] |
| `r_T_it*smISB` | -2.66106 | r_T_it × [inner stride <= 8 bytes] |
| `le_L128xN*1` | 9.50956 | le_L128xN × [无（常数项）] |
| `le_L128xN*e8` | 0.766592 | le_L128xN × [dtype = int64] |
| `le_L128xN*nge256` | 1.27293 | le_L128xN × [n >= 256] |
| `le_L128xN*e1` | -0.532149 | le_L128xN × [dtype = int8] |
| `le_L128xN*mwlg2` | 0.009858 | le_L128xN × [mean_warp_lines > 2] |
| `le_L128xN*Lgt4` | -13.0025 | le_L128xN × [L > 4] |
| `le_L128xN*mwlg8` | 5.47401 | le_L128xN × [mean_warp_lines > 8] |
| `le_L128xN*Lgt64` | -2.40963 | le_L128xN × [L > 64] |
| `le_L128xN*Lgt16` | -3.29651 | le_L128xN × [L > 16] |
| `le_L128xN*nge128` | 2.42307 | le_L128xN × [n >= 128] |
| `le_L128xN*Clt16` | -0.119301 | le_L128xN × [C < 16] |
| `le_L128xN*Clt8` | 0.03969 | le_L128xN × [C < 8] |
| `le_L128xN*smISB` | 3.63763 | le_L128xN × [inner stride <= 8 bytes] |

### 3. in-sample 拟合误差

- 点数：**291**
- 项数：**105**
- **MAPE：4.38%**

---

## `simd_store_r2` — SIMD store（2D）

### 1. 目标场景与实测条件构成

compile_mode=simd，tl.store，rank-2。

- **路径 / 操作 / 秩**：simd / store / rank-2
- **dtype**（4 种）：float16(2B)、float32(4B)、int64(8B)、int8(1B)
- **num_warps**（1 种）：1
- **形状 (R×C)**（23 种）：4x4、4x8、4x16、8x8、16x16、8x16、1x16、2x8、32x16、2x16、1x32、1x64、2x32、8x4、16x8、8x32、4x32、2x64、16x32、16x4、32x8、4x1、16x1
- **内层步长（字节）**（12 种）：1、2、3、4、6、8、12、16、24、32、64、128
- **总点数**：**943**

### 2. 公式与各项物理意义

```
T = 70.0806
  + 31.5398 * line_elems_max*1
  + -26.9588 * line_elems_max*nlt64
  + -15.4915 * line_elems_max*nge256
  + -7.00002 * line_elems_max*Seggt1
  + -6.53203 * line_elems_max*mwlg2
  + -14.1411 * line_elems_max*e8
  + 45.1875 * line_elems_max*Clt16
  + 0.235519 * line_elems_max*smISB
  + 30.3742 * line_elems_max*nge128
  + -14.7808 * line_elems_max*e4
  + 8.92044 * line_elems_max*Clt64
  + -13.6011 * line_elems_max*mwlg8
  + 39.6252 * line_elems_max*Lgt4
  + 12.9572 * line_elems_max*Rgt8
  + 0.992858 * hepl_n32*1
  + 0.016409 * hepl_n32*nlt64
  + 0.122748 * hepl_n32*nge256
  + -0.201268 * hepl_n32*Seggt1
  + -0.131832 * hepl_n32*mwlg2
  + -0.072561 * hepl_n32*e8
  + 0.219477 * hepl_n32*Clt16
  + -0.461653 * hepl_n32*smISB
  + -0.014087 * hepl_n32*nge128
  + 0.016887 * hepl_n32*e4
  + -0.315217 * hepl_n32*Clt64
  + 0.004746 * hepl_n32*mwlg8
  + 0.095538 * hepl_n32*Lgt4
  + -0.159018 * hepl_n32*Rgt8
  + 25.1753 * bsN*1
  + -8.94465 * bsN*nlt64
  + 2.08989 * bsN*nge256
  + -76.0155 * bsN*Seggt1
  + -2.92846 * bsN*mwlg2
  + 3.01359 * bsN*e8
  + 67.8485 * bsN*Clt16
  + -0.701587 * bsN*smISB
  + 15.2198 * bsN*nge128
  + -4.68246 * bsN*e4
  + 25.1753 * bsN*Clt64
  + -2.88421 * bsN*mwlg8
  + -2.79761 * bsN*Lgt4
  + 3.94095 * bsN*Rgt8
  + 0.090775 * bucket_pairs_32k16*1
  + 0.042696 * bucket_pairs_32k16*nlt64
  + -0.006034 * bucket_pairs_32k16*nge256
  + -0.025657 * bucket_pairs_32k16*Seggt1
  + 0.005241 * bucket_pairs_32k16*mwlg2
  + 0.0084 * bucket_pairs_32k16*e8
  + 0.001325 * bucket_pairs_32k16*Clt16
  + 0.021921 * bucket_pairs_32k16*smISB
  + -0.00971 * bucket_pairs_32k16*nge128
  + 0.001057 * bucket_pairs_32k16*e4
  + -0.000246 * bucket_pairs_32k16*Clt64
  + 0.007743 * bucket_pairs_32k16*mwlg8
  + -0.051307 * bucket_pairs_32k16*Lgt4
  + -0.002017 * bucket_pairs_32k16*Rgt8
  + -35.1324 * ep_cap*1
  + 20.3961 * ep_cap*nlt64
  + 22.5105 * ep_cap*nge256
  + 12.5825 * ep_cap*Seggt1
  + -22.9863 * ep_cap*mwlg2
  + 14.7779 * ep_cap*e8
  + 13.6984 * ep_cap*Clt16
  + 4.93978 * ep_cap*smISB
  + -34.5233 * ep_cap*nge128
  + 8.39917 * ep_cap*e4
  + -4.72335 * ep_cap*Clt64
  + 5.90624 * ep_cap*mwlg8
  + -9.00355 * ep_cap*Lgt4
  + -19.198 * ep_cap*Rgt8
  + 42.2271 * r_C_rl*1
  + 13.6653 * r_C_rl*nlt64
  + 0.05461 * r_C_rl*nge256
  + 3.68725 * r_C_rl*Seggt1
  + 27.7931 * r_C_rl*mwlg2
  + 5.33448 * r_C_rl*e8
  + -74.1427 * r_C_rl*Clt16
  + 2.23553 * r_C_rl*smISB
  + -20.4534 * r_C_rl*nge128
  + 9.93376 * r_C_rl*e4
  + 19.6077 * r_C_rl*Clt64
  + -13.6011 * r_C_rl*mwlg8
  + -37.6456 * r_C_rl*Lgt4
  + 43.4433 * r_C_rl*Rgt8
  + 2.43606 * ov_cshortN_L*1
  + -29.7904 * ov_cshortN_L*nlt64
  + 23.6389 * ov_cshortN_L*nge256
  + 346.756 * ov_cshortN_L*Seggt1
  + -103.066 * ov_cshortN_L*mwlg2
  + -14.1008 * ov_cshortN_L*e8
  + -147.326 * ov_cshortN_L*Clt16
  + 2.11694 * ov_cshortN_L*smISB
  + 32.8707 * ov_cshortN_L*nge128
  + -7.69071 * ov_cshortN_L*e4
  + 2.43606 * ov_cshortN_L*Clt64
  + -36.4815 * ov_cshortN_L*Lgt4
  + 4.21279 * ov_cshortN_L*Rgt8
  + -42.8064 * cshortN*1
  + 27.7025 * cshortN*nlt64
  + -24.5383 * cshortN*nge256
  + -93.4018 * cshortN*Seggt1
  + 105.332 * cshortN*mwlg2
  + 2.2602 * cshortN*e8
  + -21.4698 * cshortN*Clt16
  + -0.997242 * cshortN*smISB
  + -44.6373 * cshortN*nge128
  + 11.1694 * cshortN*e4
  + -42.8064 * cshortN*Clt64
  + 9.70493 * cshortN*mwlg8
  + 31.5214 * cshortN*Lgt4
  + -7.65537 * cshortN*Rgt8
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 70.0806 | 固定开销（循环发射、启动） |
| `line_elems_max*1` | 31.5398 | 单 line 内最多元素数 × [无（常数项）] |
| `line_elems_max*nlt64` | -26.9588 | 单 line 内最多元素数 × [n < 64] |
| `line_elems_max*nge256` | -15.4915 | 单 line 内最多元素数 × [n >= 256] |
| `line_elems_max*Seggt1` | -7.00002 | 单 line 内最多元素数 × [R > 1] |
| `line_elems_max*mwlg2` | -6.53203 | 单 line 内最多元素数 × [mean_warp_lines > 2] |
| `line_elems_max*e8` | -14.1411 | 单 line 内最多元素数 × [dtype = int64] |
| `line_elems_max*Clt16` | 45.1875 | 单 line 内最多元素数 × [C < 16] |
| `line_elems_max*smISB` | 0.235519 | 单 line 内最多元素数 × [inner stride <= 8 bytes] |
| `line_elems_max*nge128` | 30.3742 | 单 line 内最多元素数 × [n >= 128] |
| `line_elems_max*e4` | -14.7808 | 单 line 内最多元素数 × [dtype = float32] |
| `line_elems_max*Clt64` | 8.92044 | 单 line 内最多元素数 × [C < 64] |
| `line_elems_max*mwlg8` | -13.6011 | 单 line 内最多元素数 × [mean_warp_lines > 8] |
| `line_elems_max*Lgt4` | 39.6252 | 单 line 内最多元素数 × [L > 4] |
| `line_elems_max*Rgt8` | 12.9572 | 单 line 内最多元素数 × [R > 8] |
| `hepl_n32*1` | 0.992858 | n · min(128/ISB, 32) × [无（常数项）] |
| `hepl_n32*nlt64` | 0.016409 | n · min(128/ISB, 32) × [n < 64] |
| `hepl_n32*nge256` | 0.122748 | n · min(128/ISB, 32) × [n >= 256] |
| `hepl_n32*Seggt1` | -0.201268 | n · min(128/ISB, 32) × [R > 1] |
| `hepl_n32*mwlg2` | -0.131832 | n · min(128/ISB, 32) × [mean_warp_lines > 2] |
| `hepl_n32*e8` | -0.072561 | n · min(128/ISB, 32) × [dtype = int64] |
| `hepl_n32*Clt16` | 0.219477 | n · min(128/ISB, 32) × [C < 16] |
| `hepl_n32*smISB` | -0.461653 | n · min(128/ISB, 32) × [inner stride <= 8 bytes] |
| `hepl_n32*nge128` | -0.014087 | n · min(128/ISB, 32) × [n >= 128] |
| `hepl_n32*e4` | 0.016887 | n · min(128/ISB, 32) × [dtype = float32] |
| `hepl_n32*Clt64` | -0.315217 | n · min(128/ISB, 32) × [C < 64] |
| `hepl_n32*mwlg8` | 0.004746 | n · min(128/ISB, 32) × [mean_warp_lines > 8] |
| `hepl_n32*Lgt4` | 0.095538 | n · min(128/ISB, 32) × [L > 4] |
| `hepl_n32*Rgt8` | -0.159018 | n · min(128/ISB, 32) × [R > 8] |
| `bsN*1` | 25.1753 | [C·eb<32] · n × [无（常数项）] |
| `bsN*nlt64` | -8.94465 | [C·eb<32] · n × [n < 64] |
| `bsN*nge256` | 2.08989 | [C·eb<32] · n × [n >= 256] |
| `bsN*Seggt1` | -76.0155 | [C·eb<32] · n × [R > 1] |
| `bsN*mwlg2` | -2.92846 | [C·eb<32] · n × [mean_warp_lines > 2] |
| `bsN*e8` | 3.01359 | [C·eb<32] · n × [dtype = int64] |
| `bsN*Clt16` | 67.8485 | [C·eb<32] · n × [C < 16] |
| `bsN*smISB` | -0.701587 | [C·eb<32] · n × [inner stride <= 8 bytes] |
| `bsN*nge128` | 15.2198 | [C·eb<32] · n × [n >= 128] |
| `bsN*e4` | -4.68246 | [C·eb<32] · n × [dtype = float32] |
| `bsN*Clt64` | 25.1753 | [C·eb<32] · n × [C < 64] |
| `bsN*mwlg8` | -2.88421 | [C·eb<32] · n × [mean_warp_lines > 8] |
| `bsN*Lgt4` | -2.79761 | [C·eb<32] · n × [L > 4] |
| `bsN*Rgt8` | 3.94095 | [C·eb<32] · n × [R > 8] |
| `bucket_pairs_32k16*1` | 0.090775 | line 桶 (32768..16384) 对数 × [无（常数项）] |
| `bucket_pairs_32k16*nlt64` | 0.042696 | line 桶 (32768..16384) 对数 × [n < 64] |
| `bucket_pairs_32k16*nge256` | -0.006034 | line 桶 (32768..16384) 对数 × [n >= 256] |
| `bucket_pairs_32k16*Seggt1` | -0.025657 | line 桶 (32768..16384) 对数 × [R > 1] |
| `bucket_pairs_32k16*mwlg2` | 0.005241 | line 桶 (32768..16384) 对数 × [mean_warp_lines > 2] |
| `bucket_pairs_32k16*e8` | 0.0084 | line 桶 (32768..16384) 对数 × [dtype = int64] |
| `bucket_pairs_32k16*Clt16` | 0.001325 | line 桶 (32768..16384) 对数 × [C < 16] |
| `bucket_pairs_32k16*smISB` | 0.021921 | line 桶 (32768..16384) 对数 × [inner stride <= 8 bytes] |
| `bucket_pairs_32k16*nge128` | -0.00971 | line 桶 (32768..16384) 对数 × [n >= 128] |
| `bucket_pairs_32k16*e4` | 0.001057 | line 桶 (32768..16384) 对数 × [dtype = float32] |
| `bucket_pairs_32k16*Clt64` | -0.000246 | line 桶 (32768..16384) 对数 × [C < 64] |
| `bucket_pairs_32k16*mwlg8` | 0.007743 | line 桶 (32768..16384) 对数 × [mean_warp_lines > 8] |
| `bucket_pairs_32k16*Lgt4` | -0.051307 | line 桶 (32768..16384) 对数 × [L > 4] |
| `bucket_pairs_32k16*Rgt8` | -0.002017 | line 桶 (32768..16384) 对数 × [R > 8] |
| `ep_cap*1` | -35.1324 | epl 上限 × [无（常数项）] |
| `ep_cap*nlt64` | 20.3961 | epl 上限 × [n < 64] |
| `ep_cap*nge256` | 22.5105 | epl 上限 × [n >= 256] |
| `ep_cap*Seggt1` | 12.5825 | epl 上限 × [R > 1] |
| `ep_cap*mwlg2` | -22.9863 | epl 上限 × [mean_warp_lines > 2] |
| `ep_cap*e8` | 14.7779 | epl 上限 × [dtype = int64] |
| `ep_cap*Clt16` | 13.6984 | epl 上限 × [C < 16] |
| `ep_cap*smISB` | 4.93978 | epl 上限 × [inner stride <= 8 bytes] |
| `ep_cap*nge128` | -34.5233 | epl 上限 × [n >= 128] |
| `ep_cap*e4` | 8.39917 | epl 上限 × [dtype = float32] |
| `ep_cap*Clt64` | -4.72335 | epl 上限 × [C < 64] |
| `ep_cap*mwlg8` | 5.90624 | epl 上限 × [mean_warp_lines > 8] |
| `ep_cap*Lgt4` | -9.00355 | epl 上限 × [L > 4] |
| `ep_cap*Rgt8` | -19.198 | epl 上限 × [R > 8] |
| `r_C_rl*1` | 42.2271 | r_C_rl × [无（常数项）] |
| `r_C_rl*nlt64` | 13.6653 | r_C_rl × [n < 64] |
| `r_C_rl*nge256` | 0.05461 | r_C_rl × [n >= 256] |
| `r_C_rl*Seggt1` | 3.68725 | r_C_rl × [R > 1] |
| `r_C_rl*mwlg2` | 27.7931 | r_C_rl × [mean_warp_lines > 2] |
| `r_C_rl*e8` | 5.33448 | r_C_rl × [dtype = int64] |
| `r_C_rl*Clt16` | -74.1427 | r_C_rl × [C < 16] |
| `r_C_rl*smISB` | 2.23553 | r_C_rl × [inner stride <= 8 bytes] |
| `r_C_rl*nge128` | -20.4534 | r_C_rl × [n >= 128] |
| `r_C_rl*e4` | 9.93376 | r_C_rl × [dtype = float32] |
| `r_C_rl*Clt64` | 19.6077 | r_C_rl × [C < 64] |
| `r_C_rl*mwlg8` | -13.6011 | r_C_rl × [mean_warp_lines > 8] |
| `r_C_rl*Lgt4` | -37.6456 | r_C_rl × [L > 4] |
| `r_C_rl*Rgt8` | 43.4433 | r_C_rl × [R > 8] |
| `ov_cshortN_L*1` | 2.43606 | max(0, cshort · n / 32 − 触及的 128B line 数) × [无（常数项）] |
| `ov_cshortN_L*nlt64` | -29.7904 | max(0, cshort · n / 32 − 触及的 128B line 数) × [n < 64] |
| `ov_cshortN_L*nge256` | 23.6389 | max(0, cshort · n / 32 − 触及的 128B line 数) × [n >= 256] |
| `ov_cshortN_L*Seggt1` | 346.756 | max(0, cshort · n / 32 − 触及的 128B line 数) × [R > 1] |
| `ov_cshortN_L*mwlg2` | -103.066 | max(0, cshort · n / 32 − 触及的 128B line 数) × [mean_warp_lines > 2] |
| `ov_cshortN_L*e8` | -14.1008 | max(0, cshort · n / 32 − 触及的 128B line 数) × [dtype = int64] |
| `ov_cshortN_L*Clt16` | -147.326 | max(0, cshort · n / 32 − 触及的 128B line 数) × [C < 16] |
| `ov_cshortN_L*smISB` | 2.11694 | max(0, cshort · n / 32 − 触及的 128B line 数) × [inner stride <= 8 bytes] |
| `ov_cshortN_L*nge128` | 32.8707 | max(0, cshort · n / 32 − 触及的 128B line 数) × [n >= 128] |
| `ov_cshortN_L*e4` | -7.69071 | max(0, cshort · n / 32 − 触及的 128B line 数) × [dtype = float32] |
| `ov_cshortN_L*Clt64` | 2.43606 | max(0, cshort · n / 32 − 触及的 128B line 数) × [C < 64] |
| `ov_cshortN_L*Lgt4` | -36.4815 | max(0, cshort · n / 32 − 触及的 128B line 数) × [L > 4] |
| `ov_cshortN_L*Rgt8` | 4.21279 | max(0, cshort · n / 32 − 触及的 128B line 数) × [R > 8] |
| `cshortN*1` | -42.8064 | cshort · n / 32 × [无（常数项）] |
| `cshortN*nlt64` | 27.7025 | cshort · n / 32 × [n < 64] |
| `cshortN*nge256` | -24.5383 | cshort · n / 32 × [n >= 256] |
| `cshortN*Seggt1` | -93.4018 | cshort · n / 32 × [R > 1] |
| `cshortN*mwlg2` | 105.332 | cshort · n / 32 × [mean_warp_lines > 2] |
| `cshortN*e8` | 2.2602 | cshort · n / 32 × [dtype = int64] |
| `cshortN*Clt16` | -21.4698 | cshort · n / 32 × [C < 16] |
| `cshortN*smISB` | -0.997242 | cshort · n / 32 × [inner stride <= 8 bytes] |
| `cshortN*nge128` | -44.6373 | cshort · n / 32 × [n >= 128] |
| `cshortN*e4` | 11.1694 | cshort · n / 32 × [dtype = float32] |
| `cshortN*Clt64` | -42.8064 | cshort · n / 32 × [C < 64] |
| `cshortN*mwlg8` | 9.70493 | cshort · n / 32 × [mean_warp_lines > 8] |
| `cshortN*Lgt4` | 31.5214 | cshort · n / 32 × [L > 4] |
| `cshortN*Rgt8` | -7.65537 | cshort · n / 32 × [R > 8] |

### 3. in-sample 拟合误差

- 点数：**943**
- 项数：**113**
- **MAPE：5.63%**

---

## `simt_load_r1` — 纯 SIMT load（1D）

### 1. 目标场景与实测条件构成

compile_mode=simt_only，tl.load，rank-1，走 LSU 路径（已验证不走 template）。

- **路径 / 操作 / 秩**：simt_only / load / rank-1
- **dtype**（4 种）：float16(2B)、float32(4B)、int64(8B)、int8(1B)
- **num_warps**（7 种）：1、2、4、8、16、32、64
- **elements**（8 种）：4、16、64、256、8、32、512、128
- **内层步长（字节）**（18 种）：2、3、4、5、6、8、10、12、16、24、32、48、64、128、256、512、1024、2048
- **总点数**：**855**

### 2. 公式与各项物理意义

```
T = 113.645
  + 2.47705 * minL64*1
  + -4.26871 * minL64*e1
  + 0.683628 * minL64*W64
  + 1.14918 * minL64*W32
  + -0.578114 * minL64*Wgt1
  + 0.043955 * minL64*mwlg8
  + -0.340536 * minL64*Clt16
  + 0.964427 * minL64*W16
  + 0.527231 * minL64*wide
  + 2.41782 * minL64*e8
  + 2.62463 * minL64*mwlg2
  + 5.8006 * txBytes*1
  + -0.353594 * txBytes*e1
  + 2.36607 * txBytes*W64
  + 0.849517 * txBytes*W32
  + -4.7753 * txBytes*Wgt1
  + 0.145726 * txBytes*mwlg8
  + -0.19706 * txBytes*Clt16
  + 0.307243 * txBytes*W16
  + 0.297334 * txBytes*wide
  + 0.091642 * txBytes*e8
  + -1.3306 * txBytes*mwlg2
  + 3.72472 * r_R_mwl*1
  + 2.17527 * r_R_mwl*e1
  + -13.6297 * r_R_mwl*W64
  + -4.71037 * r_R_mwl*W32
  + -0.394519 * r_R_mwl*Wgt1
  + 548.567 * r_R_mwl*mwlg8
  + 2.71477 * r_R_mwl*Clt16
  + 4.41985 * r_R_mwl*W16
  + 3.77181 * r_R_mwl*wide
  + -0.65547 * r_R_mwl*e8
  + 82.8983 * r_R_mwl*mwlg2
  + -4.08471 * minL32*1
  + 5.64763 * minL32*e1
  + -3.98419 * minL32*W64
  + -1.93784 * minL32*W32
  + 3.02279 * minL32*Wgt1
  + 0.536606 * minL32*mwlg8
  + -0.340536 * minL32*Clt16
  + -0.794279 * minL32*W16
  + 3.38938 * minL32*wide
  + -3.38378 * minL32*e8
  + -3.93713 * minL32*mwlg2
  + 0.943292 * itersL*1
  + -0.305002 * itersL*e1
  + -3.98419 * itersL*W64
  + -1.93784 * itersL*W32
  + -0.733409 * itersL*Wgt1
  + -0.017902 * itersL*mwlg8
  + -0.340536 * itersL*Clt16
  + -0.794279 * itersL*W16
  + -0.979224 * itersL*wide
  + 0.619679 * itersL*e8
  + 0.350415 * itersL*mwlg2
  + -5.42313 * LmEb*1
  + 0.324569 * LmEb*e1
  + 0.319754 * LmEb*W64
  + 0.016682 * LmEb*W32
  + 4.79993 * LmEb*Wgt1
  + -0.03506 * LmEb*mwlg8
  + -0.19706 * LmEb*Clt16
  + -0.019886 * LmEb*W16
  + -0.234629 * LmEb*wide
  + -0.036859 * LmEb*e8
  + 0.74269 * LmEb*mwlg2
  + -0.231799 * minL128*1
  + -0.433033 * minL128*e1
  + 3.94273 * minL128*W64
  + 1.11688 * minL128*W32
  + 0.149685 * minL128*Wgt1
  + -0.01048 * minL128*mwlg8
  + -0.340536 * minL128*Clt16
  + -0.043452 * minL128*W16
  + 0.527231 * minL128*wide
  + -0.529176 * minL128*e8
  + -0.084219 * minL128*mwlg2
  + 15.6889 * mx_iters_mwl*1
  + -1.24766 * mx_iters_mwl*e1
  + 18.794 * mx_iters_mwl*W64
  + 7.55985 * mx_iters_mwl*W32
  + -2.09564 * mx_iters_mwl*Wgt1
  + -6.41607 * mx_iters_mwl*mwlg8
  + -0.340536 * mx_iters_mwl*Clt16
  + 2.31917 * mx_iters_mwl*W16
  + -7.8407 * mx_iters_mwl*wide
  + 2.86846 * mx_iters_mwl*e8
  + -3.98733 * mx_iters_mwl*mwlg2
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 113.645 | 固定开销（循环发射、启动） |
| `minL64*1` | 2.47705 | min(L,64) × [无（常数项）] |
| `minL64*e1` | -4.26871 | min(L,64) × [dtype = int8] |
| `minL64*W64` | 0.683628 | min(L,64) × [num_warps = 64] |
| `minL64*W32` | 1.14918 | min(L,64) × [num_warps = 32] |
| `minL64*Wgt1` | -0.578114 | min(L,64) × [num_warps > 1] |
| `minL64*mwlg8` | 0.043955 | min(L,64) × [mean_warp_lines > 8] |
| `minL64*Clt16` | -0.340536 | min(L,64) × [C < 16] |
| `minL64*W16` | 0.964427 | min(L,64) × [num_warps = 16] |
| `minL64*wide` | 0.527231 | min(L,64) × [inner stride <= 2*element_bytes] |
| `minL64*e8` | 2.41782 | min(L,64) × [dtype = int64] |
| `minL64*mwlg2` | 2.62463 | min(L,64) × [mean_warp_lines > 2] |
| `txBytes*1` | 5.8006 | iters · mwl · eb × [无（常数项）] |
| `txBytes*e1` | -0.353594 | iters · mwl · eb × [dtype = int8] |
| `txBytes*W64` | 2.36607 | iters · mwl · eb × [num_warps = 64] |
| `txBytes*W32` | 0.849517 | iters · mwl · eb × [num_warps = 32] |
| `txBytes*Wgt1` | -4.7753 | iters · mwl · eb × [num_warps > 1] |
| `txBytes*mwlg8` | 0.145726 | iters · mwl · eb × [mean_warp_lines > 8] |
| `txBytes*Clt16` | -0.19706 | iters · mwl · eb × [C < 16] |
| `txBytes*W16` | 0.307243 | iters · mwl · eb × [num_warps = 16] |
| `txBytes*wide` | 0.297334 | iters · mwl · eb × [inner stride <= 2*element_bytes] |
| `txBytes*e8` | 0.091642 | iters · mwl · eb × [dtype = int64] |
| `txBytes*mwlg2` | -1.3306 | iters · mwl · eb × [mean_warp_lines > 2] |
| `r_R_mwl*1` | 3.72472 | r_R_mwl × [无（常数项）] |
| `r_R_mwl*e1` | 2.17527 | r_R_mwl × [dtype = int8] |
| `r_R_mwl*W64` | -13.6297 | r_R_mwl × [num_warps = 64] |
| `r_R_mwl*W32` | -4.71037 | r_R_mwl × [num_warps = 32] |
| `r_R_mwl*Wgt1` | -0.394519 | r_R_mwl × [num_warps > 1] |
| `r_R_mwl*mwlg8` | 548.567 | r_R_mwl × [mean_warp_lines > 8] |
| `r_R_mwl*Clt16` | 2.71477 | r_R_mwl × [C < 16] |
| `r_R_mwl*W16` | 4.41985 | r_R_mwl × [num_warps = 16] |
| `r_R_mwl*wide` | 3.77181 | r_R_mwl × [inner stride <= 2*element_bytes] |
| `r_R_mwl*e8` | -0.65547 | r_R_mwl × [dtype = int64] |
| `r_R_mwl*mwlg2` | 82.8983 | r_R_mwl × [mean_warp_lines > 2] |
| `minL32*1` | -4.08471 | min(L,32) × [无（常数项）] |
| `minL32*e1` | 5.64763 | min(L,32) × [dtype = int8] |
| `minL32*W64` | -3.98419 | min(L,32) × [num_warps = 64] |
| `minL32*W32` | -1.93784 | min(L,32) × [num_warps = 32] |
| `minL32*Wgt1` | 3.02279 | min(L,32) × [num_warps > 1] |
| `minL32*mwlg8` | 0.536606 | min(L,32) × [mean_warp_lines > 8] |
| `minL32*Clt16` | -0.340536 | min(L,32) × [C < 16] |
| `minL32*W16` | -0.794279 | min(L,32) × [num_warps = 16] |
| `minL32*wide` | 3.38938 | min(L,32) × [inner stride <= 2*element_bytes] |
| `minL32*e8` | -3.38378 | min(L,32) × [dtype = int64] |
| `minL32*mwlg2` | -3.93713 | min(L,32) × [mean_warp_lines > 2] |
| `itersL*1` | 0.943292 | iters · L × [无（常数项）] |
| `itersL*e1` | -0.305002 | iters · L × [dtype = int8] |
| `itersL*W64` | -3.98419 | iters · L × [num_warps = 64] |
| `itersL*W32` | -1.93784 | iters · L × [num_warps = 32] |
| `itersL*Wgt1` | -0.733409 | iters · L × [num_warps > 1] |
| `itersL*mwlg8` | -0.017902 | iters · L × [mean_warp_lines > 8] |
| `itersL*Clt16` | -0.340536 | iters · L × [C < 16] |
| `itersL*W16` | -0.794279 | iters · L × [num_warps = 16] |
| `itersL*wide` | -0.979224 | iters · L × [inner stride <= 2*element_bytes] |
| `itersL*e8` | 0.619679 | iters · L × [dtype = int64] |
| `itersL*mwlg2` | 0.350415 | iters · L × [mean_warp_lines > 2] |
| `LmEb*1` | -5.42313 | LmEb × [无（常数项）] |
| `LmEb*e1` | 0.324569 | LmEb × [dtype = int8] |
| `LmEb*W64` | 0.319754 | LmEb × [num_warps = 64] |
| `LmEb*W32` | 0.016682 | LmEb × [num_warps = 32] |
| `LmEb*Wgt1` | 4.79993 | LmEb × [num_warps > 1] |
| `LmEb*mwlg8` | -0.03506 | LmEb × [mean_warp_lines > 8] |
| `LmEb*Clt16` | -0.19706 | LmEb × [C < 16] |
| `LmEb*W16` | -0.019886 | LmEb × [num_warps = 16] |
| `LmEb*wide` | -0.234629 | LmEb × [inner stride <= 2*element_bytes] |
| `LmEb*e8` | -0.036859 | LmEb × [dtype = int64] |
| `LmEb*mwlg2` | 0.74269 | LmEb × [mean_warp_lines > 2] |
| `minL128*1` | -0.231799 | min(L,128) × [无（常数项）] |
| `minL128*e1` | -0.433033 | min(L,128) × [dtype = int8] |
| `minL128*W64` | 3.94273 | min(L,128) × [num_warps = 64] |
| `minL128*W32` | 1.11688 | min(L,128) × [num_warps = 32] |
| `minL128*Wgt1` | 0.149685 | min(L,128) × [num_warps > 1] |
| `minL128*mwlg8` | -0.01048 | min(L,128) × [mean_warp_lines > 8] |
| `minL128*Clt16` | -0.340536 | min(L,128) × [C < 16] |
| `minL128*W16` | -0.043452 | min(L,128) × [num_warps = 16] |
| `minL128*wide` | 0.527231 | min(L,128) × [inner stride <= 2*element_bytes] |
| `minL128*e8` | -0.529176 | min(L,128) × [dtype = int64] |
| `minL128*mwlg2` | -0.084219 | min(L,128) × [mean_warp_lines > 2] |
| `mx_iters_mwl*1` | 15.6889 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [无（常数项）] |
| `mx_iters_mwl*e1` | -1.24766 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = int8] |
| `mx_iters_mwl*W64` | 18.794 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 64] |
| `mx_iters_mwl*W32` | 7.55985 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 32] |
| `mx_iters_mwl*Wgt1` | -2.09564 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps > 1] |
| `mx_iters_mwl*mwlg8` | -6.41607 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [mean_warp_lines > 8] |
| `mx_iters_mwl*Clt16` | -0.340536 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [C < 16] |
| `mx_iters_mwl*W16` | 2.31917 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 16] |
| `mx_iters_mwl*wide` | -7.8407 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [inner stride <= 2*element_bytes] |
| `mx_iters_mwl*e8` | 2.86846 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = int64] |
| `mx_iters_mwl*mwlg2` | -3.98733 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [mean_warp_lines > 2] |

### 3. in-sample 拟合误差

- 点数：**855**
- 项数：**89**
- **MAPE：7.50%**

---

## `simt_load_r2` — 纯 SIMT load（2D）

### 1. 目标场景与实测条件构成

compile_mode=simt_only，tl.load，rank-2，LSU 路径。

- **路径 / 操作 / 秩**：simt_only / load / rank-2
- **dtype**（4 种）：float16(2B)、float32(4B)、int64(8B)、int8(1B)
- **num_warps**（7 种）：1、2、4、8、16、32、64
- **形状 (R×C)**（34 种）：4x8、4x4、16x16、4x16、8x8、8x16、16x1、1x16、1x32、1x64、2x8、2x16、2x32、8x4、2x64、32x16、4x1、4x32、16x32、32x8、8x2、32x1、16x4、16x2、64x8、8x32、64x1、2x128、4x64、32x2、16x8、64x2、8x64、64x4
- **内层步长（字节）**（12 种）：1、2、3、4、6、8、12、16、24、32、64、128
- **总点数**：**1762**

### 2. 公式与各项物理意义

```
T = 114.096
  + 10.6071 * transX*1
  + 3.40379 * transX*Rgt8
  + 0.87068 * transX*smISB
  + 12.008 * transX*W64
  + -7.51639 * transX*Lgt4
  + 4.19428 * transX*W32
  + -6.51575 * transX*nge256
  + -3.28693 * transX*Rgt16
  + -0.564627 * transX*e8
  + -0.880645 * transX*e4
  + -0.573206 * r_L_rl*1
  + 2.80569 * r_L_rl*Rgt8
  + 3.29331 * r_L_rl*smISB
  + -6.19375 * r_L_rl*W64
  + 0.206327 * r_L_rl*Lgt4
  + -7.12478 * r_L_rl*W32
  + 1.00399 * r_L_rl*nge256
  + -0.09491 * r_L_rl*Rgt16
  + -0.497046 * r_L_rl*e8
  + -1.41843 * r_L_rl*e4
  + -0.775262 * mx_iters_mwl*1
  + 1.53979 * mx_iters_mwl*Rgt8
  + -0.872245 * mx_iters_mwl*smISB
  + 12.008 * mx_iters_mwl*W64
  + 0.23351 * mx_iters_mwl*Lgt4
  + 4.19428 * mx_iters_mwl*W32
  + 3.844 * mx_iters_mwl*nge256
  + -0.256027 * mx_iters_mwl*Rgt16
  + -0.009104 * mx_iters_mwl*e8
  + 0.757842 * mx_iters_mwl*e4
  + 1.98871 * p_eb_it*1
  + 3.44053 * p_eb_it*Rgt8
  + -1.28741 * p_eb_it*smISB
  + 10.1601 * p_eb_it*W64
  + -1.24821 * p_eb_it*Lgt4
  + 3.68877 * p_eb_it*W32
  + 1.00693 * p_eb_it*nge256
  + 1.24164 * p_eb_it*Rgt16
  + -1.22606 * p_eb_it*e8
  + 0.296813 * p_eb_it*e4
  + -36.2974 * le_L128xL*1
  + -0.658656 * le_L128xL*Rgt8
  + -9.05002 * le_L128xL*smISB
  + -8.32501 * le_L128xL*W64
  + 42.0349 * le_L128xL*Lgt4
  + -1.9911 * le_L128xL*W32
  + -5.97883 * le_L128xL*nge256
  + 0.517927 * le_L128xL*Rgt16
  + 0.6072 * le_L128xL*e8
  + 0.378346 * le_L128xL*e4
  + -0.838808 * itersL*1
  + -0.062547 * itersL*Rgt8
  + 0.245885 * itersL*smISB
  + 1.2033 * itersL*W64
  + 0.545925 * itersL*Lgt4
  + 2.66608 * itersL*W32
  + 0.252765 * itersL*nge256
  + 0.132708 * itersL*Rgt16
  + 0.165294 * itersL*e8
  + -0.103988 * itersL*e4
  + 42.4466 * ov_L_pages*1
  + -8.87196 * ov_L_pages*Rgt8
  + 8.51643 * ov_L_pages*smISB
  + -2.81718 * ov_L_pages*W64
  + -35.0742 * ov_L_pages*Lgt4
  + -2.76031 * ov_L_pages*W32
  + 7.44542 * ov_L_pages*nge256
  + -1.25805 * ov_L_pages*Rgt16
  + 1.34613 * ov_L_pages*e8
  + -0.386163 * ov_L_pages*e4
  + -6.35549 * hL4*1
  + 3.39468 * hL4*Rgt8
  + -2.92115 * hL4*smISB
  + 12.097 * hL4*W64
  + -6.35549 * hL4*Lgt4
  + 5.08144 * hL4*W32
  + 4.11142 * hL4*nge256
  + 4.02901 * hL4*Rgt16
  + -0.419936 * hL4*e8
  + 0.841458 * hL4*e4
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 114.096 | 固定开销（循环发射、启动） |
| `transX*1` | 10.6071 | iters · max_warp_lines × [无（常数项）] |
| `transX*Rgt8` | 3.40379 | iters · max_warp_lines × [R > 8] |
| `transX*smISB` | 0.87068 | iters · max_warp_lines × [inner stride <= 8 bytes] |
| `transX*W64` | 12.008 | iters · max_warp_lines × [num_warps = 64] |
| `transX*Lgt4` | -7.51639 | iters · max_warp_lines × [L > 4] |
| `transX*W32` | 4.19428 | iters · max_warp_lines × [num_warps = 32] |
| `transX*nge256` | -6.51575 | iters · max_warp_lines × [n >= 256] |
| `transX*Rgt16` | -3.28693 | iters · max_warp_lines × [R > 16] |
| `transX*e8` | -0.564627 | iters · max_warp_lines × [dtype = int64] |
| `transX*e4` | -0.880645 | iters · max_warp_lines × [dtype = float32] |
| `r_L_rl*1` | -0.573206 | r_L_rl × [无（常数项）] |
| `r_L_rl*Rgt8` | 2.80569 | r_L_rl × [R > 8] |
| `r_L_rl*smISB` | 3.29331 | r_L_rl × [inner stride <= 8 bytes] |
| `r_L_rl*W64` | -6.19375 | r_L_rl × [num_warps = 64] |
| `r_L_rl*Lgt4` | 0.206327 | r_L_rl × [L > 4] |
| `r_L_rl*W32` | -7.12478 | r_L_rl × [num_warps = 32] |
| `r_L_rl*nge256` | 1.00399 | r_L_rl × [n >= 256] |
| `r_L_rl*Rgt16` | -0.09491 | r_L_rl × [R > 16] |
| `r_L_rl*e8` | -0.497046 | r_L_rl × [dtype = int64] |
| `r_L_rl*e4` | -1.41843 | r_L_rl × [dtype = float32] |
| `mx_iters_mwl*1` | -0.775262 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [无（常数项）] |
| `mx_iters_mwl*Rgt8` | 1.53979 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [R > 8] |
| `mx_iters_mwl*smISB` | -0.872245 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [inner stride <= 8 bytes] |
| `mx_iters_mwl*W64` | 12.008 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 64] |
| `mx_iters_mwl*Lgt4` | 0.23351 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [L > 4] |
| `mx_iters_mwl*W32` | 4.19428 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 32] |
| `mx_iters_mwl*nge256` | 3.844 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [n >= 256] |
| `mx_iters_mwl*Rgt16` | -0.256027 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [R > 16] |
| `mx_iters_mwl*e8` | -0.009104 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = int64] |
| `mx_iters_mwl*e4` | 0.757842 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = float32] |
| `p_eb_it*1` | 1.98871 | p_eb_it × [无（常数项）] |
| `p_eb_it*Rgt8` | 3.44053 | p_eb_it × [R > 8] |
| `p_eb_it*smISB` | -1.28741 | p_eb_it × [inner stride <= 8 bytes] |
| `p_eb_it*W64` | 10.1601 | p_eb_it × [num_warps = 64] |
| `p_eb_it*Lgt4` | -1.24821 | p_eb_it × [L > 4] |
| `p_eb_it*W32` | 3.68877 | p_eb_it × [num_warps = 32] |
| `p_eb_it*nge256` | 1.00693 | p_eb_it × [n >= 256] |
| `p_eb_it*Rgt16` | 1.24164 | p_eb_it × [R > 16] |
| `p_eb_it*e8` | -1.22606 | p_eb_it × [dtype = int64] |
| `p_eb_it*e4` | 0.296813 | p_eb_it × [dtype = float32] |
| `le_L128xL*1` | -36.2974 | le_L128xL × [无（常数项）] |
| `le_L128xL*Rgt8` | -0.658656 | le_L128xL × [R > 8] |
| `le_L128xL*smISB` | -9.05002 | le_L128xL × [inner stride <= 8 bytes] |
| `le_L128xL*W64` | -8.32501 | le_L128xL × [num_warps = 64] |
| `le_L128xL*Lgt4` | 42.0349 | le_L128xL × [L > 4] |
| `le_L128xL*W32` | -1.9911 | le_L128xL × [num_warps = 32] |
| `le_L128xL*nge256` | -5.97883 | le_L128xL × [n >= 256] |
| `le_L128xL*Rgt16` | 0.517927 | le_L128xL × [R > 16] |
| `le_L128xL*e8` | 0.6072 | le_L128xL × [dtype = int64] |
| `le_L128xL*e4` | 0.378346 | le_L128xL × [dtype = float32] |
| `itersL*1` | -0.838808 | iters · L × [无（常数项）] |
| `itersL*Rgt8` | -0.062547 | iters · L × [R > 8] |
| `itersL*smISB` | 0.245885 | iters · L × [inner stride <= 8 bytes] |
| `itersL*W64` | 1.2033 | iters · L × [num_warps = 64] |
| `itersL*Lgt4` | 0.545925 | iters · L × [L > 4] |
| `itersL*W32` | 2.66608 | iters · L × [num_warps = 32] |
| `itersL*nge256` | 0.252765 | iters · L × [n >= 256] |
| `itersL*Rgt16` | 0.132708 | iters · L × [R > 16] |
| `itersL*e8` | 0.165294 | iters · L × [dtype = int64] |
| `itersL*e4` | -0.103988 | iters · L × [dtype = float32] |
| `ov_L_pages*1` | 42.4466 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [无（常数项）] |
| `ov_L_pages*Rgt8` | -8.87196 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [R > 8] |
| `ov_L_pages*smISB` | 8.51643 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [inner stride <= 8 bytes] |
| `ov_L_pages*W64` | -2.81718 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [num_warps = 64] |
| `ov_L_pages*Lgt4` | -35.0742 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [L > 4] |
| `ov_L_pages*W32` | -2.76031 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [num_warps = 32] |
| `ov_L_pages*nge256` | 7.44542 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [n >= 256] |
| `ov_L_pages*Rgt16` | -1.25805 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [R > 16] |
| `ov_L_pages*e8` | 1.34613 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [dtype = int64] |
| `ov_L_pages*e4` | -0.386163 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [dtype = float32] |
| `hL4*1` | -6.35549 | max(0, L−4) × [无（常数项）] |
| `hL4*Rgt8` | 3.39468 | max(0, L−4) × [R > 8] |
| `hL4*smISB` | -2.92115 | max(0, L−4) × [inner stride <= 8 bytes] |
| `hL4*W64` | 12.097 | max(0, L−4) × [num_warps = 64] |
| `hL4*Lgt4` | -6.35549 | max(0, L−4) × [L > 4] |
| `hL4*W32` | 5.08144 | max(0, L−4) × [num_warps = 32] |
| `hL4*nge256` | 4.11142 | max(0, L−4) × [n >= 256] |
| `hL4*Rgt16` | 4.02901 | max(0, L−4) × [R > 16] |
| `hL4*e8` | -0.419936 | max(0, L−4) × [dtype = int64] |
| `hL4*e4` | 0.841458 | max(0, L−4) × [dtype = float32] |

### 3. in-sample 拟合误差

- 点数：**1762**
- 项数：**81**
- **MAPE：7.24%**

---

## `simt_store_r1` — 纯 SIMT store（1D）

### 1. 目标场景与实测条件构成

compile_mode=simt_only，tl.store，rank-1，LSU 路径。

- **路径 / 操作 / 秩**：simt_only / store / rank-1
- **dtype**（4 种）：float16(2B)、float32(4B)、int64(8B)、int8(1B)
- **num_warps**（7 种）：1、2、4、8、16、32、64
- **elements**（8 种）：256、64、4、16、128、32、8、512
- **内层步长（字节）**（17 种）：2、3、4、6、8、10、12、16、24、32、48、64、128、256、512、1024、2048
- **总点数**：**962**

### 2. 公式与各项物理意义

```
T = -5.71488
  + -7.14675 * mx_L_W*1
  + 0.107962 * mx_L_W*e2
  + 8.88804 * mx_L_W*Wgt1
  + -1.32702 * mx_L_W*W16
  + -1.23653 * mx_L_W*W8
  + 0.059378 * mx_L_W*e1
  + -0.420797 * mx_L_W*mwlg8
  + 1.0366 * mx_L_W*nge256
  + 0.073507 * mx_L_W*e8
  + 0.179104 * mx_L_W*Lgt4
  + -1.97351 * mx_L_W*W4
  + -0.369235 * mx_L_W*W32
  + 0.173969 * mx_L_W*Clt8
  + -0.99594 * mx_L_W*nge128
  + 1.71913 * hLeb4*1
  + 1.00266 * hLeb4*e2
  + 0.164096 * hLeb4*Wgt1
  + 0.202589 * hLeb4*W16
  + 0.050621 * hLeb4*W8
  + -1.70607 * hLeb4*e1
  + 1.10992 * hLeb4*mwlg8
  + 0.123498 * hLeb4*nge256
  + 0.159895 * hLeb4*e8
  + 1.71913 * hLeb4*Lgt4
  + 0.076289 * hLeb4*W4
  + -0.483648 * hLeb4*W32
  + 0.151561 * hLeb4*nge128
  + 23.0263 * cmd*1
  + 1.6642 * cmd*e2
  + -2.48065 * cmd*Wgt1
  + 4.44246 * cmd*W16
  + -1.29369 * cmd*W8
  + 3.74106 * cmd*e1
  + -1.32505 * cmd*mwlg8
  + 3.87509 * cmd*nge256
  + 0.465509 * cmd*e8
  + -3.95036 * cmd*Lgt4
  + 1.15363 * cmd*W4
  + -0.504223 * cmd*W32
  + -0.38867 * cmd*Clt8
  + -1.49883 * cmd*nge128
  + -14.4196 * ov_L_iters*1
  + 3.74053 * ov_L_iters*e2
  + 5.31605 * ov_L_iters*Wgt1
  + 0.44748 * ov_L_iters*W16
  + 0.425644 * ov_L_iters*W8
  + 4.00635 * ov_L_iters*e1
  + -1.66467 * ov_L_iters*mwlg8
  + -5.07014 * ov_L_iters*nge256
  + 2.57828 * ov_L_iters*e8
  + 3.82819 * ov_L_iters*Lgt4
  + -1.33557 * ov_L_iters*W4
  + -0.002061 * ov_L_iters*W32
  + 0.092006 * ov_L_iters*Clt8
  + 3.40676 * ov_L_iters*nge128
  + 15.8052 * mx_L_iters*1
  + -4.0414 * mx_L_iters*e2
  + -7.31468 * mx_L_iters*Wgt1
  + 0.383588 * mx_L_iters*W16
  + 0.325682 * mx_L_iters*W8
  + -5.19829 * mx_L_iters*e1
  + 2.66349 * mx_L_iters*mwlg8
  + -2.16304 * mx_L_iters*nge256
  + -2.26977 * mx_L_iters*e8
  + -3.63167 * mx_L_iters*Lgt4
  + 2.30726 * mx_L_iters*W4
  + -0.00151 * mx_L_iters*W32
  + -0.296665 * mx_L_iters*Clt8
  + 4.16263 * mx_L_iters*nge128
  + 6.48095 * hLW*1
  + -0.456457 * hLW*e2
  + -6.35767 * hLW*Wgt1
  + -0.304743 * hLW*W16
  + -0.436835 * hLW*W8
  + -0.265447 * hLW*e1
  + 0.92768 * hLW*mwlg8
  + 1.83711 * hLW*nge256
  + 0.518938 * hLW*e8
  + -0.835423 * hLW*Lgt4
  + -0.137551 * hLW*W4
  + -0.386881 * hLW*W32
  + 0.049131 * hLW*Clt8
  + -2.02731 * hLW*nge128
  + -1.1402 * ov_L_mwl*1
  + 0.593229 * ov_L_mwl*e2
  + 0.010986 * ov_L_mwl*Wgt1
  + 0.513154 * ov_L_mwl*W16
  + 0.577254 * ov_L_mwl*W8
  + 1.22469 * ov_L_mwl*e1
  + 0.078928 * ov_L_mwl*mwlg8
  + 8.55356 * ov_L_mwl*nge256
  + 0.117505 * ov_L_mwl*e8
  + 0.229536 * ov_L_mwl*Lgt4
  + 0.385883 * ov_L_mwl*W4
  + 1.28627 * ov_L_mwl*W32
  + -9.02651 * ov_L_mwl*nge128
  + -1.91077 * hLW05*1
  + 2.31134 * hLW05*e2
  + -0.312691 * hLW05*Wgt1
  + 0.240789 * hLW05*W16
  + 0.667324 * hLW05*W8
  + 2.37576 * hLW05*e1
  + -1.7505 * hLW05*mwlg8
  + -2.39687 * hLW05*nge256
  + -0.440262 * hLW05*e8
  + 2.22603 * hLW05*Lgt4
  + 0.868529 * hLW05*W4
  + -0.306721 * hLW05*W32
  + 0.707432 * hLW05*Clt8
  + 2.26842 * hLW05*nge128
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | -5.71488 | 固定开销（循环发射、启动） |
| `mx_L_W*1` | -7.14675 | max(触及的 128B line 数, num_warps) × [无（常数项）] |
| `mx_L_W*e2` | 0.107962 | max(触及的 128B line 数, num_warps) × [dtype = float16] |
| `mx_L_W*Wgt1` | 8.88804 | max(触及的 128B line 数, num_warps) × [num_warps > 1] |
| `mx_L_W*W16` | -1.32702 | max(触及的 128B line 数, num_warps) × [num_warps = 16] |
| `mx_L_W*W8` | -1.23653 | max(触及的 128B line 数, num_warps) × [num_warps = 8] |
| `mx_L_W*e1` | 0.059378 | max(触及的 128B line 数, num_warps) × [dtype = int8] |
| `mx_L_W*mwlg8` | -0.420797 | max(触及的 128B line 数, num_warps) × [mean_warp_lines > 8] |
| `mx_L_W*nge256` | 1.0366 | max(触及的 128B line 数, num_warps) × [n >= 256] |
| `mx_L_W*e8` | 0.073507 | max(触及的 128B line 数, num_warps) × [dtype = int64] |
| `mx_L_W*Lgt4` | 0.179104 | max(触及的 128B line 数, num_warps) × [L > 4] |
| `mx_L_W*W4` | -1.97351 | max(触及的 128B line 数, num_warps) × [num_warps = 4] |
| `mx_L_W*W32` | -0.369235 | max(触及的 128B line 数, num_warps) × [num_warps = 32] |
| `mx_L_W*Clt8` | 0.173969 | max(触及的 128B line 数, num_warps) × [C < 8] |
| `mx_L_W*nge128` | -0.99594 | max(触及的 128B line 数, num_warps) × [n >= 128] |
| `hLeb4*1` | 1.71913 | max(0, L/eb − 4) × [无（常数项）] |
| `hLeb4*e2` | 1.00266 | max(0, L/eb − 4) × [dtype = float16] |
| `hLeb4*Wgt1` | 0.164096 | max(0, L/eb − 4) × [num_warps > 1] |
| `hLeb4*W16` | 0.202589 | max(0, L/eb − 4) × [num_warps = 16] |
| `hLeb4*W8` | 0.050621 | max(0, L/eb − 4) × [num_warps = 8] |
| `hLeb4*e1` | -1.70607 | max(0, L/eb − 4) × [dtype = int8] |
| `hLeb4*mwlg8` | 1.10992 | max(0, L/eb − 4) × [mean_warp_lines > 8] |
| `hLeb4*nge256` | 0.123498 | max(0, L/eb − 4) × [n >= 256] |
| `hLeb4*e8` | 0.159895 | max(0, L/eb − 4) × [dtype = int64] |
| `hLeb4*Lgt4` | 1.71913 | max(0, L/eb − 4) × [L > 4] |
| `hLeb4*W4` | 0.076289 | max(0, L/eb − 4) × [num_warps = 4] |
| `hLeb4*W32` | -0.483648 | max(0, L/eb − 4) × [num_warps = 32] |
| `hLeb4*nge128` | 0.151561 | max(0, L/eb − 4) × [n >= 128] |
| `cmd*1` | 23.0263 | ceil(L / floor(128/eb)) = MTE 命令数 × [无（常数项）] |
| `cmd*e2` | 1.6642 | ceil(L / floor(128/eb)) = MTE 命令数 × [dtype = float16] |
| `cmd*Wgt1` | -2.48065 | ceil(L / floor(128/eb)) = MTE 命令数 × [num_warps > 1] |
| `cmd*W16` | 4.44246 | ceil(L / floor(128/eb)) = MTE 命令数 × [num_warps = 16] |
| `cmd*W8` | -1.29369 | ceil(L / floor(128/eb)) = MTE 命令数 × [num_warps = 8] |
| `cmd*e1` | 3.74106 | ceil(L / floor(128/eb)) = MTE 命令数 × [dtype = int8] |
| `cmd*mwlg8` | -1.32505 | ceil(L / floor(128/eb)) = MTE 命令数 × [mean_warp_lines > 8] |
| `cmd*nge256` | 3.87509 | ceil(L / floor(128/eb)) = MTE 命令数 × [n >= 256] |
| `cmd*e8` | 0.465509 | ceil(L / floor(128/eb)) = MTE 命令数 × [dtype = int64] |
| `cmd*Lgt4` | -3.95036 | ceil(L / floor(128/eb)) = MTE 命令数 × [L > 4] |
| `cmd*W4` | 1.15363 | ceil(L / floor(128/eb)) = MTE 命令数 × [num_warps = 4] |
| `cmd*W32` | -0.504223 | ceil(L / floor(128/eb)) = MTE 命令数 × [num_warps = 32] |
| `cmd*Clt8` | -0.38867 | ceil(L / floor(128/eb)) = MTE 命令数 × [C < 8] |
| `cmd*nge128` | -1.49883 | ceil(L / floor(128/eb)) = MTE 命令数 × [n >= 128] |
| `ov_L_iters*1` | -14.4196 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [无（常数项）] |
| `ov_L_iters*e2` | 3.74053 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [dtype = float16] |
| `ov_L_iters*Wgt1` | 5.31605 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [num_warps > 1] |
| `ov_L_iters*W16` | 0.44748 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [num_warps = 16] |
| `ov_L_iters*W8` | 0.425644 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [num_warps = 8] |
| `ov_L_iters*e1` | 4.00635 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [dtype = int8] |
| `ov_L_iters*mwlg8` | -1.66467 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [mean_warp_lines > 8] |
| `ov_L_iters*nge256` | -5.07014 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [n >= 256] |
| `ov_L_iters*e8` | 2.57828 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [dtype = int64] |
| `ov_L_iters*Lgt4` | 3.82819 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [L > 4] |
| `ov_L_iters*W4` | -1.33557 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [num_warps = 4] |
| `ov_L_iters*W32` | -0.002061 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [num_warps = 32] |
| `ov_L_iters*Clt8` | 0.092006 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [C < 8] |
| `ov_L_iters*nge128` | 3.40676 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [n >= 128] |
| `mx_L_iters*1` | 15.8052 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [无（常数项）] |
| `mx_L_iters*e2` | -4.0414 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [dtype = float16] |
| `mx_L_iters*Wgt1` | -7.31468 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [num_warps > 1] |
| `mx_L_iters*W16` | 0.383588 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [num_warps = 16] |
| `mx_L_iters*W8` | 0.325682 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [num_warps = 8] |
| `mx_L_iters*e1` | -5.19829 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [dtype = int8] |
| `mx_L_iters*mwlg8` | 2.66349 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [mean_warp_lines > 8] |
| `mx_L_iters*nge256` | -2.16304 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [n >= 256] |
| `mx_L_iters*e8` | -2.26977 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [dtype = int64] |
| `mx_L_iters*Lgt4` | -3.63167 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [L > 4] |
| `mx_L_iters*W4` | 2.30726 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [num_warps = 4] |
| `mx_L_iters*W32` | -0.00151 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [num_warps = 32] |
| `mx_L_iters*Clt8` | -0.296665 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [C < 8] |
| `mx_L_iters*nge128` | 4.16263 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [n >= 128] |
| `hLW*1` | 6.48095 | hLW × [无（常数项）] |
| `hLW*e2` | -0.456457 | hLW × [dtype = float16] |
| `hLW*Wgt1` | -6.35767 | hLW × [num_warps > 1] |
| `hLW*W16` | -0.304743 | hLW × [num_warps = 16] |
| `hLW*W8` | -0.436835 | hLW × [num_warps = 8] |
| `hLW*e1` | -0.265447 | hLW × [dtype = int8] |
| `hLW*mwlg8` | 0.92768 | hLW × [mean_warp_lines > 8] |
| `hLW*nge256` | 1.83711 | hLW × [n >= 256] |
| `hLW*e8` | 0.518938 | hLW × [dtype = int64] |
| `hLW*Lgt4` | -0.835423 | hLW × [L > 4] |
| `hLW*W4` | -0.137551 | hLW × [num_warps = 4] |
| `hLW*W32` | -0.386881 | hLW × [num_warps = 32] |
| `hLW*Clt8` | 0.049131 | hLW × [C < 8] |
| `hLW*nge128` | -2.02731 | hLW × [n >= 128] |
| `ov_L_mwl*1` | -1.1402 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [无（常数项）] |
| `ov_L_mwl*e2` | 0.593229 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = float16] |
| `ov_L_mwl*Wgt1` | 0.010986 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps > 1] |
| `ov_L_mwl*W16` | 0.513154 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 16] |
| `ov_L_mwl*W8` | 0.577254 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 8] |
| `ov_L_mwl*e1` | 1.22469 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = int8] |
| `ov_L_mwl*mwlg8` | 0.078928 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [mean_warp_lines > 8] |
| `ov_L_mwl*nge256` | 8.55356 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [n >= 256] |
| `ov_L_mwl*e8` | 0.117505 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = int64] |
| `ov_L_mwl*Lgt4` | 0.229536 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [L > 4] |
| `ov_L_mwl*W4` | 0.385883 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 4] |
| `ov_L_mwl*W32` | 1.28627 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 32] |
| `ov_L_mwl*nge128` | -9.02651 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [n >= 128] |
| `hLW05*1` | -1.91077 | max(0, L − 0.5W) × [无（常数项）] |
| `hLW05*e2` | 2.31134 | max(0, L − 0.5W) × [dtype = float16] |
| `hLW05*Wgt1` | -0.312691 | max(0, L − 0.5W) × [num_warps > 1] |
| `hLW05*W16` | 0.240789 | max(0, L − 0.5W) × [num_warps = 16] |
| `hLW05*W8` | 0.667324 | max(0, L − 0.5W) × [num_warps = 8] |
| `hLW05*e1` | 2.37576 | max(0, L − 0.5W) × [dtype = int8] |
| `hLW05*mwlg8` | -1.7505 | max(0, L − 0.5W) × [mean_warp_lines > 8] |
| `hLW05*nge256` | -2.39687 | max(0, L − 0.5W) × [n >= 256] |
| `hLW05*e8` | -0.440262 | max(0, L − 0.5W) × [dtype = int64] |
| `hLW05*Lgt4` | 2.22603 | max(0, L − 0.5W) × [L > 4] |
| `hLW05*W4` | 0.868529 | max(0, L − 0.5W) × [num_warps = 4] |
| `hLW05*W32` | -0.306721 | max(0, L − 0.5W) × [num_warps = 32] |
| `hLW05*Clt8` | 0.707432 | max(0, L − 0.5W) × [C < 8] |
| `hLW05*nge128` | 2.26842 | max(0, L − 0.5W) × [n >= 128] |

### 3. in-sample 拟合误差

- 点数：**962**
- 项数：**113**
- **MAPE：5.47%**

---

## `simt_store_r2` — 纯 SIMT store（2D）

### 1. 目标场景与实测条件构成

compile_mode=simt_only，tl.store，rank-2，LSU 路径。

- **路径 / 操作 / 秩**：simt_only / store / rank-2
- **dtype**（4 种）：float16(2B)、float32(4B)、int64(8B)、int8(1B)
- **num_warps**（7 种）：1、2、4、8、16、32、64
- **形状 (R×C)**（33 种）：4x8、16x16、4x4、4x16、8x8、8x16、2x16、8x4、16x4、16x8、1x16、1x32、1x64、2x8、2x32、8x32、16x32、32x8、32x16、4x32、2x64、8x2、16x1、16x2、32x1、32x2、64x1、64x2、2x128、4x64、64x4、8x64、64x8
- **内层步长（字节）**（11 种）：2、3、4、6、8、12、16、24、32、64、128
- **总点数**：**3416**

### 2. 公式与各项物理意义

```
T = 21.923
  + 1.89692 * mx_L_W*1
  + 0.100843 * mx_L_W*e4
  + 0.157758 * mx_L_W*Rgt8
  + -1.01474 * mx_L_W*Clt32
  + 0.099358 * mx_L_W*e8
  + -0.183761 * mx_L_W*nlt64
  + -0.51838 * mx_L_W*e2
  + -0.461368 * mx_L_W*mwlg2
  + 0.00268 * bucket_pairs_32k16*1
  + -0.001645 * bucket_pairs_32k16*e4
  + 0.002858 * bucket_pairs_32k16*Rgt8
  + 0.002358 * bucket_pairs_32k16*Clt32
  + -0.001897 * bucket_pairs_32k16*e8
  + -0.004319 * bucket_pairs_32k16*nlt64
  + 0.000418 * bucket_pairs_32k16*e2
  + 8.6e-05 * bucket_pairs_32k16*mwlg2
  + -0.151072 * hLeb2*1
  + -0.158826 * hLeb2*e4
  + 1.86323 * hLeb2*Rgt8
  + 1.6584 * hLeb2*Clt32
  + 11.0642 * hLeb2*e8
  + 0.661586 * hLeb2*nlt64
  + 5.46554 * hLeb2*e2
  + -4.3151 * hLeb2*mwlg2
  + -0.001051 * p_n_T*1
  + -0.00021 * p_n_T*e4
  + 0.000104 * p_n_T*Rgt8
  + 0.000987 * p_n_T*Clt32
  + -0.000125 * p_n_T*e8
  + 0.000759 * p_n_T*nlt64
  + 0.000237 * p_n_T*e2
  + 0.00016 * p_n_T*mwlg2
  + 7.66414 * hLeb4*1
  + -4.04018 * hLeb4*e4
  + 0.826387 * hLeb4*Rgt8
  + -0.280978 * hLeb4*Clt32
  + -7.9723 * hLeb4*e8
  + -3.36191 * hLeb4*nlt64
  + -0.722937 * hLeb4*e2
  + -0.571865 * hLeb4*mwlg2
  + -0.393079 * mx_L_cshortN*1
  + -0.265608 * mx_L_cshortN*e4
  + -0.319038 * mx_L_cshortN*Rgt8
  + 0.412072 * mx_L_cshortN*Clt32
  + -0.266124 * mx_L_cshortN*e8
  + 0.018058 * mx_L_cshortN*nlt64
  + 0.045238 * mx_L_cshortN*e2
  + 0.308598 * mx_L_cshortN*mwlg2
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 21.923 | 固定开销（循环发射、启动） |
| `mx_L_W*1` | 1.89692 | max(触及的 128B line 数, num_warps) × [无（常数项）] |
| `mx_L_W*e4` | 0.100843 | max(触及的 128B line 数, num_warps) × [dtype = float32] |
| `mx_L_W*Rgt8` | 0.157758 | max(触及的 128B line 数, num_warps) × [R > 8] |
| `mx_L_W*Clt32` | -1.01474 | max(触及的 128B line 数, num_warps) × [C < 32] |
| `mx_L_W*e8` | 0.099358 | max(触及的 128B line 数, num_warps) × [dtype = int64] |
| `mx_L_W*nlt64` | -0.183761 | max(触及的 128B line 数, num_warps) × [n < 64] |
| `mx_L_W*e2` | -0.51838 | max(触及的 128B line 数, num_warps) × [dtype = float16] |
| `mx_L_W*mwlg2` | -0.461368 | max(触及的 128B line 数, num_warps) × [mean_warp_lines > 2] |
| `bucket_pairs_32k16*1` | 0.00268 | line 桶 (32768..16384) 对数 × [无（常数项）] |
| `bucket_pairs_32k16*e4` | -0.001645 | line 桶 (32768..16384) 对数 × [dtype = float32] |
| `bucket_pairs_32k16*Rgt8` | 0.002858 | line 桶 (32768..16384) 对数 × [R > 8] |
| `bucket_pairs_32k16*Clt32` | 0.002358 | line 桶 (32768..16384) 对数 × [C < 32] |
| `bucket_pairs_32k16*e8` | -0.001897 | line 桶 (32768..16384) 对数 × [dtype = int64] |
| `bucket_pairs_32k16*nlt64` | -0.004319 | line 桶 (32768..16384) 对数 × [n < 64] |
| `bucket_pairs_32k16*e2` | 0.000418 | line 桶 (32768..16384) 对数 × [dtype = float16] |
| `bucket_pairs_32k16*mwlg2` | 8.6e-05 | line 桶 (32768..16384) 对数 × [mean_warp_lines > 2] |
| `hLeb2*1` | -0.151072 | max(0, L/eb − 2) × [无（常数项）] |
| `hLeb2*e4` | -0.158826 | max(0, L/eb − 2) × [dtype = float32] |
| `hLeb2*Rgt8` | 1.86323 | max(0, L/eb − 2) × [R > 8] |
| `hLeb2*Clt32` | 1.6584 | max(0, L/eb − 2) × [C < 32] |
| `hLeb2*e8` | 11.0642 | max(0, L/eb − 2) × [dtype = int64] |
| `hLeb2*nlt64` | 0.661586 | max(0, L/eb − 2) × [n < 64] |
| `hLeb2*e2` | 5.46554 | max(0, L/eb − 2) × [dtype = float16] |
| `hLeb2*mwlg2` | -4.3151 | max(0, L/eb − 2) × [mean_warp_lines > 2] |
| `p_n_T*1` | -0.001051 | p_n_T × [无（常数项）] |
| `p_n_T*e4` | -0.00021 | p_n_T × [dtype = float32] |
| `p_n_T*Rgt8` | 0.000104 | p_n_T × [R > 8] |
| `p_n_T*Clt32` | 0.000987 | p_n_T × [C < 32] |
| `p_n_T*e8` | -0.000125 | p_n_T × [dtype = int64] |
| `p_n_T*nlt64` | 0.000759 | p_n_T × [n < 64] |
| `p_n_T*e2` | 0.000237 | p_n_T × [dtype = float16] |
| `p_n_T*mwlg2` | 0.00016 | p_n_T × [mean_warp_lines > 2] |
| `hLeb4*1` | 7.66414 | max(0, L/eb − 4) × [无（常数项）] |
| `hLeb4*e4` | -4.04018 | max(0, L/eb − 4) × [dtype = float32] |
| `hLeb4*Rgt8` | 0.826387 | max(0, L/eb − 4) × [R > 8] |
| `hLeb4*Clt32` | -0.280978 | max(0, L/eb − 4) × [C < 32] |
| `hLeb4*e8` | -7.9723 | max(0, L/eb − 4) × [dtype = int64] |
| `hLeb4*nlt64` | -3.36191 | max(0, L/eb − 4) × [n < 64] |
| `hLeb4*e2` | -0.722937 | max(0, L/eb − 4) × [dtype = float16] |
| `hLeb4*mwlg2` | -0.571865 | max(0, L/eb − 4) × [mean_warp_lines > 2] |
| `mx_L_cshortN*1` | -0.393079 | max(触及的 128B line 数, cshort · n / 32) × [无（常数项）] |
| `mx_L_cshortN*e4` | -0.265608 | max(触及的 128B line 数, cshort · n / 32) × [dtype = float32] |
| `mx_L_cshortN*Rgt8` | -0.319038 | max(触及的 128B line 数, cshort · n / 32) × [R > 8] |
| `mx_L_cshortN*Clt32` | 0.412072 | max(触及的 128B line 数, cshort · n / 32) × [C < 32] |
| `mx_L_cshortN*e8` | -0.266124 | max(触及的 128B line 数, cshort · n / 32) × [dtype = int64] |
| `mx_L_cshortN*nlt64` | 0.018058 | max(触及的 128B line 数, cshort · n / 32) × [n < 64] |
| `mx_L_cshortN*e2` | 0.045238 | max(触及的 128B line 数, cshort · n / 32) × [dtype = float16] |
| `mx_L_cshortN*mwlg2` | 0.308598 | max(触及的 128B line 数, cshort · n / 32) × [mean_warp_lines > 2] |

### 3. in-sample 拟合误差

- 点数：**3416**
- 项数：**49**
- **MAPE：17.75%**

---
