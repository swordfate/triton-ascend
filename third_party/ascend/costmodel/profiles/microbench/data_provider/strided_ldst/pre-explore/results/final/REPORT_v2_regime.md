# 版本 2：REGIME —— 全部修饰因子（精度最高）

每个公式都使用**全部 14 个物理 regime 标志 + 4 个 dtype 指示 + 7 个 num_warps 指示**，共 26 个修饰因子乘遍基特征。项数最多，但 in-sample 误差最小。

## 汇总表

| 公式 | 场景 | 路径/操作/秩 | 点数 | 项数 | **in-sample MAPE** |
|---|---|---|---:|---:|---:|
| `simd_load_wide_r1` | SIMD load 宽分支（1D） | simd / load / rank-1 | 32 | 28 | **3.48%** |
| `simd_load_wide_r2` | SIMD load 宽分支（2D） | simd / load / rank-2 | 246 | 217 | **10.06%** |
| `simd_load_gather_r1` | SIMD load 聚集分支（1D） | simd / load / rank-1 | 284 | 190 | **11.59%** |
| `simd_load_gather_r2` | SIMD load 聚集分支（2D） | simd / load / rank-2 | 750 | 217 | **9.90%** |
| `simd_store_r1` | SIMD store（1D） | simd / store / rank-1 | 291 | 217 | **4.24%** |
| `simd_store_r2` | SIMD store（2D） | simd / store / rank-2 | 943 | 217 | **5.45%** |
| `simt_load_r1` | 纯 SIMT load（1D） | simt_only / load / rank-1 | 855 | 217 | **7.37%** |
| `simt_load_r2` | 纯 SIMT load（2D） | simt_only / load / rank-2 | 1762 | 217 | **6.16%** |
| `simt_store_r1` | 纯 SIMT store（1D） | simt_only / store / rank-1 | 962 | 217 | **5.10%** |
| `simt_store_r2` | 纯 SIMT store（2D） | simt_only / store / rank-2 | 3416 | 163 | **15.17%** |
| | | | **9541** | | **9.81%** |

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
T = 64.2028
  + 0.683423 * ov_L_pages*const
  + 0.380701 * ov_L_pages*eb1
  + -0.21846 * ov_L_pages*eb2
  + -0.365352 * ov_L_pages*eb4
  + 0.886535 * ov_L_pages*eb8
  + 0.683423 * ov_L_pages*W1
  + 1.12156 * ov_L_pages*Clt8
  + 0.195059 * ov_L_pages*Clt16
  + 0.682782 * ov_L_pages*Clt32
  + -0.000612 * ov_L_pages*Clt64
  + -0.000612 * ov_L_pages*nlt64
  + -0.159143 * ov_L_pages*nge128
  + 0.20096 * ov_L_pages*nge256
  + 0.126723 * ov_L_pages*Lgt4
  + -0.146808 * ov_L_pages*Lgt16
  + -0.815327 * ov_L_pages*mwlg2
  + -0.203112 * ov_L_pages*smISB
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 64.2028 | 固定开销（循环发射、启动） |
| `ov_L_pages*const` | 0.683423 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [无（常数项）] |
| `ov_L_pages*eb1` | 0.380701 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [dtype = int8] |
| `ov_L_pages*eb2` | -0.21846 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [dtype = float16] |
| `ov_L_pages*eb4` | -0.365352 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [dtype = float32] |
| `ov_L_pages*eb8` | 0.886535 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [dtype = int64] |
| `ov_L_pages*W1` | 0.683423 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [num_warps = 1] |
| `ov_L_pages*Clt8` | 1.12156 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [C < 8] |
| `ov_L_pages*Clt16` | 0.195059 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [C < 16] |
| `ov_L_pages*Clt32` | 0.682782 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [C < 32] |
| `ov_L_pages*Clt64` | -0.000612 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [C < 64] |
| `ov_L_pages*nlt64` | -0.000612 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [n < 64] |
| `ov_L_pages*nge128` | -0.159143 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [n >= 128] |
| `ov_L_pages*nge256` | 0.20096 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [n >= 256] |
| `ov_L_pages*Lgt4` | 0.126723 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [L > 4] |
| `ov_L_pages*Lgt16` | -0.146808 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [L > 16] |
| `ov_L_pages*mwlg2` | -0.815327 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [mean_warp_lines > 2] |
| `ov_L_pages*smISB` | -0.203112 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [inner stride <= 8 bytes] |

### 3. in-sample 拟合误差

- 点数：**32**
- 项数：**28**
- **MAPE：3.48%**

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
T = 0.997081
  + 0.954488 * wuSeg*const
  + -7.9714 * wuSeg*eb1
  + -1.12767 * wuSeg*eb2
  + 3.89383 * wuSeg*eb4
  + 6.15976 * wuSeg*eb8
  + 0.954528 * wuSeg*W1
  + 4e-06 * wuSeg*W2
  + -3e-06 * wuSeg*W4
  + 1e-06 * wuSeg*W8
  + -2e-06 * wuSeg*W16
  + -4e-06 * wuSeg*W32
  + 4e-06 * wuSeg*W64
  + -2.70588 * wuSeg*Clt8
  + -0.320831 * wuSeg*Clt16
  + 0.954518 * wuSeg*Clt32
  + 0.954512 * wuSeg*Clt64
  + -5.08127 * wuSeg*nlt64
  + -3.27455 * wuSeg*nge128
  + 11.3768 * wuSeg*nge256
  + 2.68799 * wuSeg*Rgt8
  + 10.7942 * wuSeg*Rgt16
  + -0.117988 * wuSeg*Lgt4
  + 0.512328 * wuSeg*Lgt16
  + 1e-06 * wuSeg*Lgt64
  + -2.13385 * wuSeg*mwlg2
  + 7e-06 * wuSeg*mwlg8
  + -1.71797 * wuSeg*smISB
  + -0.002257 * p_R_spanb*const
  + -0.002014 * p_R_spanb*eb1
  + -0.002194 * p_R_spanb*eb2
  + -0.002058 * p_R_spanb*eb4
  + 0.003994 * p_R_spanb*eb8
  + -0.002252 * p_R_spanb*W1
  + 0 * p_R_spanb*W4
  + -0 * p_R_spanb*W64
  + -0.000554 * p_R_spanb*Clt8
  + 8e-06 * p_R_spanb*Clt16
  + 3e-06 * p_R_spanb*Clt32
  + 4e-06 * p_R_spanb*Clt64
  + 0.000409 * p_R_spanb*nlt64
  + -9.9e-05 * p_R_spanb*nge128
  + 0.000122 * p_R_spanb*nge256
  + 9.6e-05 * p_R_spanb*Rgt8
  + 1e-06 * p_R_spanb*Rgt16
  + 0.000626 * p_R_spanb*Lgt4
  + -0.000188 * p_R_spanb*Lgt16
  + -4.9e-05 * p_R_spanb*mwlg2
  + -0.002011 * p_R_spanb*mwlg8
  + 0.005896 * p_R_spanb*smISB
  + -30.24 * r_R_L*const
  + -41.2054 * r_R_L*eb1
  + 7.69766 * r_R_L*eb2
  + 7.38103 * r_R_L*eb4
  + -4.11329 * r_R_L*eb8
  + -30.24 * r_R_L*W1
  + 57.3814 * r_R_L*Clt8
  + -65.4532 * r_R_L*Clt16
  + -77.4065 * r_R_L*Clt32
  + -6.29601 * r_R_L*Clt64
  + 91.9133 * r_R_L*nlt64
  + -9.96583 * r_R_L*nge128
  + 141.118 * r_R_L*nge256
  + 2.37779 * r_R_L*Rgt8
  + -867.65 * r_R_L*Rgt16
  + -112.108 * r_R_L*Lgt4
  + -63.921 * r_R_L*Lgt16
  + -36.887 * r_R_L*mwlg2
  + -0 * r_R_L*mwlg8
  + 65.5552 * r_R_L*smISB
  + 0.811712 * r_C_it*const
  + -0.835982 * r_C_it*eb1
  + -1.47925 * r_C_it*eb2
  + -0.782286 * r_C_it*eb4
  + 3.90923 * r_C_it*eb8
  + 0.811712 * r_C_it*W1
  + 8.2993 * r_C_it*Clt8
  + 3.21748 * r_C_it*Clt16
  + 3.69447 * r_C_it*Clt32
  + 9.48 * r_C_it*Clt64
  + -8.11686 * r_C_it*nlt64
  + -11.3 * r_C_it*nge128
  + -0.821313 * r_C_it*nge256
  + 0.132002 * r_C_it*Rgt8
  + -0.001839 * r_C_it*Rgt16
  + -0.223408 * r_C_it*Lgt4
  + -132.354 * r_C_it*Lgt16
  + -8.25084 * r_C_it*mwlg2
  + -0 * r_C_it*mwlg8
  + -3.00777 * r_C_it*smISB
  + 2.36811 * comb_lines*const
  + 79.8627 * comb_lines*eb1
  + 13.047 * comb_lines*eb2
  + -24.9373 * comb_lines*eb4
  + -65.6043 * comb_lines*eb8
  + 2.36811 * comb_lines*W1
  + 1.16267 * comb_lines*Clt8
  + 1.26205 * comb_lines*Clt16
  + -31.2553 * comb_lines*Clt32
  + -20.2619 * comb_lines*Clt64
  + 30.7659 * comb_lines*nlt64
  + -5.98669 * comb_lines*nge128
  + -4.71612 * comb_lines*nge256
  + -16.586 * comb_lines*Rgt8
  + 39.4306 * comb_lines*Rgt16
  + 5.27122 * comb_lines*Lgt4
  + 25.4953 * comb_lines*Lgt16
  + 79.5637 * comb_lines*mwlg2
  + -0 * comb_lines*mwlg8
  + 68.0849 * comb_lines*smISB
  + -311.225 * r_L_rs*const
  + -120.279 * r_L_rs*eb1
  + 39.2649 * r_L_rs*eb2
  + 176.436 * r_L_rs*eb4
  + -406.647 * r_L_rs*eb8
  + -311.225 * r_L_rs*W1
  + 1377.76 * r_L_rs*Clt8
  + -563.384 * r_L_rs*Clt16
  + -308.743 * r_L_rs*Clt32
  + -311.197 * r_L_rs*Clt64
  + 861.428 * r_L_rs*nlt64
  + 141.583 * r_L_rs*nge128
  + 1178.41 * r_L_rs*nge256
  + -166.197 * r_L_rs*Rgt8
  + -1376.67 * r_L_rs*Rgt16
  + -42.24 * r_L_rs*Lgt4
  + -1524.78 * r_L_rs*Lgt16
  + 36.7505 * r_L_rs*mwlg2
  + -0 * r_L_rs*mwlg8
  + -261.893 * r_L_rs*smISB
  + 4.12287 * r_R_eb*const
  + 79.8627 * r_R_eb*eb1
  + 12.05 * r_R_eb*eb2
  + -150.437 * r_R_eb*eb4
  + 62.6473 * r_R_eb*eb8
  + 4.12287 * r_R_eb*W1
  + 0.962169 * r_R_eb*Clt8
  + 239.088 * r_R_eb*Clt16
  + 234.135 * r_R_eb*Clt32
  + -152.29 * r_R_eb*Clt64
  + 0.646874 * r_R_eb*nlt64
  + -23.6586 * r_R_eb*nge128
  + -38.6054 * r_R_eb*nge256
  + 17.9861 * r_R_eb*Rgt8
  + 131.597 * r_R_eb*Rgt16
  + 23.2618 * r_R_eb*Lgt4
  + 6.50956 * r_R_eb*Lgt16
  + 5.1418 * r_R_eb*mwlg2
  + -0 * r_R_eb*mwlg8
  + -58.5104 * r_R_eb*smISB
  + 0.653515 * R2*const
  + -0.344529 * R2*eb1
  + 1.12953 * R2*eb2
  + 0.931621 * R2*eb4
  + -1.06311 * R2*eb8
  + 0.653515 * R2*W1
  + 9.84795 * R2*Clt8
  + -7.14082 * R2*Clt16
  + -16.3594 * R2*Clt32
  + 8.4391 * R2*Clt64
  + 4.30017 * R2*nlt64
  + 12.9202 * R2*nge128
  + -13.5167 * R2*nge256
  + -3.82128 * R2*Rgt8
  + -1.88337 * R2*Rgt16
  + -0.405605 * R2*Lgt4
  + 0.4768 * R2*Lgt16
  + -0.661669 * R2*mwlg2
  + -2e-06 * R2*mwlg8
  + -1.77769 * R2*smISB
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 0.997081 | 固定开销（循环发射、启动） |
| `wuSeg*const` | 0.954488 | wuSeg × [无（常数项）] |
| `wuSeg*eb1` | -7.9714 | wuSeg × [dtype = int8] |
| `wuSeg*eb2` | -1.12767 | wuSeg × [dtype = float16] |
| `wuSeg*eb4` | 3.89383 | wuSeg × [dtype = float32] |
| `wuSeg*eb8` | 6.15976 | wuSeg × [dtype = int64] |
| `wuSeg*W1` | 0.954528 | wuSeg × [num_warps = 1] |
| `wuSeg*W2` | 4e-06 | wuSeg × [num_warps = 2] |
| `wuSeg*W4` | -3e-06 | wuSeg × [num_warps = 4] |
| `wuSeg*W8` | 1e-06 | wuSeg × [num_warps = 8] |
| `wuSeg*W16` | -2e-06 | wuSeg × [num_warps = 16] |
| `wuSeg*W32` | -4e-06 | wuSeg × [num_warps = 32] |
| `wuSeg*W64` | 4e-06 | wuSeg × [num_warps = 64] |
| `wuSeg*Clt8` | -2.70588 | wuSeg × [C < 8] |
| `wuSeg*Clt16` | -0.320831 | wuSeg × [C < 16] |
| `wuSeg*Clt32` | 0.954518 | wuSeg × [C < 32] |
| `wuSeg*Clt64` | 0.954512 | wuSeg × [C < 64] |
| `wuSeg*nlt64` | -5.08127 | wuSeg × [n < 64] |
| `wuSeg*nge128` | -3.27455 | wuSeg × [n >= 128] |
| `wuSeg*nge256` | 11.3768 | wuSeg × [n >= 256] |
| `wuSeg*Rgt8` | 2.68799 | wuSeg × [R > 8] |
| `wuSeg*Rgt16` | 10.7942 | wuSeg × [R > 16] |
| `wuSeg*Lgt4` | -0.117988 | wuSeg × [L > 4] |
| `wuSeg*Lgt16` | 0.512328 | wuSeg × [L > 16] |
| `wuSeg*Lgt64` | 1e-06 | wuSeg × [L > 64] |
| `wuSeg*mwlg2` | -2.13385 | wuSeg × [mean_warp_lines > 2] |
| `wuSeg*mwlg8` | 7e-06 | wuSeg × [mean_warp_lines > 8] |
| `wuSeg*smISB` | -1.71797 | wuSeg × [inner stride <= 8 bytes] |
| `p_R_spanb*const` | -0.002257 | p_R_spanb × [无（常数项）] |
| `p_R_spanb*eb1` | -0.002014 | p_R_spanb × [dtype = int8] |
| `p_R_spanb*eb2` | -0.002194 | p_R_spanb × [dtype = float16] |
| `p_R_spanb*eb4` | -0.002058 | p_R_spanb × [dtype = float32] |
| `p_R_spanb*eb8` | 0.003994 | p_R_spanb × [dtype = int64] |
| `p_R_spanb*W1` | -0.002252 | p_R_spanb × [num_warps = 1] |
| `p_R_spanb*W4` | 0 | p_R_spanb × [num_warps = 4] |
| `p_R_spanb*W64` | -0 | p_R_spanb × [num_warps = 64] |
| `p_R_spanb*Clt8` | -0.000554 | p_R_spanb × [C < 8] |
| `p_R_spanb*Clt16` | 8e-06 | p_R_spanb × [C < 16] |
| `p_R_spanb*Clt32` | 3e-06 | p_R_spanb × [C < 32] |
| `p_R_spanb*Clt64` | 4e-06 | p_R_spanb × [C < 64] |
| `p_R_spanb*nlt64` | 0.000409 | p_R_spanb × [n < 64] |
| `p_R_spanb*nge128` | -9.9e-05 | p_R_spanb × [n >= 128] |
| `p_R_spanb*nge256` | 0.000122 | p_R_spanb × [n >= 256] |
| `p_R_spanb*Rgt8` | 9.6e-05 | p_R_spanb × [R > 8] |
| `p_R_spanb*Rgt16` | 1e-06 | p_R_spanb × [R > 16] |
| `p_R_spanb*Lgt4` | 0.000626 | p_R_spanb × [L > 4] |
| `p_R_spanb*Lgt16` | -0.000188 | p_R_spanb × [L > 16] |
| `p_R_spanb*mwlg2` | -4.9e-05 | p_R_spanb × [mean_warp_lines > 2] |
| `p_R_spanb*mwlg8` | -0.002011 | p_R_spanb × [mean_warp_lines > 8] |
| `p_R_spanb*smISB` | 0.005896 | p_R_spanb × [inner stride <= 8 bytes] |
| `r_R_L*const` | -30.24 | r_R_L × [无（常数项）] |
| `r_R_L*eb1` | -41.2054 | r_R_L × [dtype = int8] |
| `r_R_L*eb2` | 7.69766 | r_R_L × [dtype = float16] |
| `r_R_L*eb4` | 7.38103 | r_R_L × [dtype = float32] |
| `r_R_L*eb8` | -4.11329 | r_R_L × [dtype = int64] |
| `r_R_L*W1` | -30.24 | r_R_L × [num_warps = 1] |
| `r_R_L*Clt8` | 57.3814 | r_R_L × [C < 8] |
| `r_R_L*Clt16` | -65.4532 | r_R_L × [C < 16] |
| `r_R_L*Clt32` | -77.4065 | r_R_L × [C < 32] |
| `r_R_L*Clt64` | -6.29601 | r_R_L × [C < 64] |
| `r_R_L*nlt64` | 91.9133 | r_R_L × [n < 64] |
| `r_R_L*nge128` | -9.96583 | r_R_L × [n >= 128] |
| `r_R_L*nge256` | 141.118 | r_R_L × [n >= 256] |
| `r_R_L*Rgt8` | 2.37779 | r_R_L × [R > 8] |
| `r_R_L*Rgt16` | -867.65 | r_R_L × [R > 16] |
| `r_R_L*Lgt4` | -112.108 | r_R_L × [L > 4] |
| `r_R_L*Lgt16` | -63.921 | r_R_L × [L > 16] |
| `r_R_L*mwlg2` | -36.887 | r_R_L × [mean_warp_lines > 2] |
| `r_R_L*mwlg8` | -0 | r_R_L × [mean_warp_lines > 8] |
| `r_R_L*smISB` | 65.5552 | r_R_L × [inner stride <= 8 bytes] |
| `r_C_it*const` | 0.811712 | r_C_it × [无（常数项）] |
| `r_C_it*eb1` | -0.835982 | r_C_it × [dtype = int8] |
| `r_C_it*eb2` | -1.47925 | r_C_it × [dtype = float16] |
| `r_C_it*eb4` | -0.782286 | r_C_it × [dtype = float32] |
| `r_C_it*eb8` | 3.90923 | r_C_it × [dtype = int64] |
| `r_C_it*W1` | 0.811712 | r_C_it × [num_warps = 1] |
| `r_C_it*Clt8` | 8.2993 | r_C_it × [C < 8] |
| `r_C_it*Clt16` | 3.21748 | r_C_it × [C < 16] |
| `r_C_it*Clt32` | 3.69447 | r_C_it × [C < 32] |
| `r_C_it*Clt64` | 9.48 | r_C_it × [C < 64] |
| `r_C_it*nlt64` | -8.11686 | r_C_it × [n < 64] |
| `r_C_it*nge128` | -11.3 | r_C_it × [n >= 128] |
| `r_C_it*nge256` | -0.821313 | r_C_it × [n >= 256] |
| `r_C_it*Rgt8` | 0.132002 | r_C_it × [R > 8] |
| `r_C_it*Rgt16` | -0.001839 | r_C_it × [R > 16] |
| `r_C_it*Lgt4` | -0.223408 | r_C_it × [L > 4] |
| `r_C_it*Lgt16` | -132.354 | r_C_it × [L > 16] |
| `r_C_it*mwlg2` | -8.25084 | r_C_it × [mean_warp_lines > 2] |
| `r_C_it*mwlg8` | -0 | r_C_it × [mean_warp_lines > 8] |
| `r_C_it*smISB` | -3.00777 | r_C_it × [inner stride <= 8 bytes] |
| `comb_lines*const` | 2.36811 | R × 每行 line 数 × [无（常数项）] |
| `comb_lines*eb1` | 79.8627 | R × 每行 line 数 × [dtype = int8] |
| `comb_lines*eb2` | 13.047 | R × 每行 line 数 × [dtype = float16] |
| `comb_lines*eb4` | -24.9373 | R × 每行 line 数 × [dtype = float32] |
| `comb_lines*eb8` | -65.6043 | R × 每行 line 数 × [dtype = int64] |
| `comb_lines*W1` | 2.36811 | R × 每行 line 数 × [num_warps = 1] |
| `comb_lines*Clt8` | 1.16267 | R × 每行 line 数 × [C < 8] |
| `comb_lines*Clt16` | 1.26205 | R × 每行 line 数 × [C < 16] |
| `comb_lines*Clt32` | -31.2553 | R × 每行 line 数 × [C < 32] |
| `comb_lines*Clt64` | -20.2619 | R × 每行 line 数 × [C < 64] |
| `comb_lines*nlt64` | 30.7659 | R × 每行 line 数 × [n < 64] |
| `comb_lines*nge128` | -5.98669 | R × 每行 line 数 × [n >= 128] |
| `comb_lines*nge256` | -4.71612 | R × 每行 line 数 × [n >= 256] |
| `comb_lines*Rgt8` | -16.586 | R × 每行 line 数 × [R > 8] |
| `comb_lines*Rgt16` | 39.4306 | R × 每行 line 数 × [R > 16] |
| `comb_lines*Lgt4` | 5.27122 | R × 每行 line 数 × [L > 4] |
| `comb_lines*Lgt16` | 25.4953 | R × 每行 line 数 × [L > 16] |
| `comb_lines*mwlg2` | 79.5637 | R × 每行 line 数 × [mean_warp_lines > 2] |
| `comb_lines*mwlg8` | -0 | R × 每行 line 数 × [mean_warp_lines > 8] |
| `comb_lines*smISB` | 68.0849 | R × 每行 line 数 × [inner stride <= 8 bytes] |
| `r_L_rs*const` | -311.225 | r_L_rs × [无（常数项）] |
| `r_L_rs*eb1` | -120.279 | r_L_rs × [dtype = int8] |
| `r_L_rs*eb2` | 39.2649 | r_L_rs × [dtype = float16] |
| `r_L_rs*eb4` | 176.436 | r_L_rs × [dtype = float32] |
| `r_L_rs*eb8` | -406.647 | r_L_rs × [dtype = int64] |
| `r_L_rs*W1` | -311.225 | r_L_rs × [num_warps = 1] |
| `r_L_rs*Clt8` | 1377.76 | r_L_rs × [C < 8] |
| `r_L_rs*Clt16` | -563.384 | r_L_rs × [C < 16] |
| `r_L_rs*Clt32` | -308.743 | r_L_rs × [C < 32] |
| `r_L_rs*Clt64` | -311.197 | r_L_rs × [C < 64] |
| `r_L_rs*nlt64` | 861.428 | r_L_rs × [n < 64] |
| `r_L_rs*nge128` | 141.583 | r_L_rs × [n >= 128] |
| `r_L_rs*nge256` | 1178.41 | r_L_rs × [n >= 256] |
| `r_L_rs*Rgt8` | -166.197 | r_L_rs × [R > 8] |
| `r_L_rs*Rgt16` | -1376.67 | r_L_rs × [R > 16] |
| `r_L_rs*Lgt4` | -42.24 | r_L_rs × [L > 4] |
| `r_L_rs*Lgt16` | -1524.78 | r_L_rs × [L > 16] |
| `r_L_rs*mwlg2` | 36.7505 | r_L_rs × [mean_warp_lines > 2] |
| `r_L_rs*mwlg8` | -0 | r_L_rs × [mean_warp_lines > 8] |
| `r_L_rs*smISB` | -261.893 | r_L_rs × [inner stride <= 8 bytes] |
| `r_R_eb*const` | 4.12287 | r_R_eb × [无（常数项）] |
| `r_R_eb*eb1` | 79.8627 | r_R_eb × [dtype = int8] |
| `r_R_eb*eb2` | 12.05 | r_R_eb × [dtype = float16] |
| `r_R_eb*eb4` | -150.437 | r_R_eb × [dtype = float32] |
| `r_R_eb*eb8` | 62.6473 | r_R_eb × [dtype = int64] |
| `r_R_eb*W1` | 4.12287 | r_R_eb × [num_warps = 1] |
| `r_R_eb*Clt8` | 0.962169 | r_R_eb × [C < 8] |
| `r_R_eb*Clt16` | 239.088 | r_R_eb × [C < 16] |
| `r_R_eb*Clt32` | 234.135 | r_R_eb × [C < 32] |
| `r_R_eb*Clt64` | -152.29 | r_R_eb × [C < 64] |
| `r_R_eb*nlt64` | 0.646874 | r_R_eb × [n < 64] |
| `r_R_eb*nge128` | -23.6586 | r_R_eb × [n >= 128] |
| `r_R_eb*nge256` | -38.6054 | r_R_eb × [n >= 256] |
| `r_R_eb*Rgt8` | 17.9861 | r_R_eb × [R > 8] |
| `r_R_eb*Rgt16` | 131.597 | r_R_eb × [R > 16] |
| `r_R_eb*Lgt4` | 23.2618 | r_R_eb × [L > 4] |
| `r_R_eb*Lgt16` | 6.50956 | r_R_eb × [L > 16] |
| `r_R_eb*mwlg2` | 5.1418 | r_R_eb × [mean_warp_lines > 2] |
| `r_R_eb*mwlg8` | -0 | r_R_eb × [mean_warp_lines > 8] |
| `r_R_eb*smISB` | -58.5104 | r_R_eb × [inner stride <= 8 bytes] |
| `R2*const` | 0.653515 | R² × [无（常数项）] |
| `R2*eb1` | -0.344529 | R² × [dtype = int8] |
| `R2*eb2` | 1.12953 | R² × [dtype = float16] |
| `R2*eb4` | 0.931621 | R² × [dtype = float32] |
| `R2*eb8` | -1.06311 | R² × [dtype = int64] |
| `R2*W1` | 0.653515 | R² × [num_warps = 1] |
| `R2*Clt8` | 9.84795 | R² × [C < 8] |
| `R2*Clt16` | -7.14082 | R² × [C < 16] |
| `R2*Clt32` | -16.3594 | R² × [C < 32] |
| `R2*Clt64` | 8.4391 | R² × [C < 64] |
| `R2*nlt64` | 4.30017 | R² × [n < 64] |
| `R2*nge128` | 12.9202 | R² × [n >= 128] |
| `R2*nge256` | -13.5167 | R² × [n >= 256] |
| `R2*Rgt8` | -3.82128 | R² × [R > 8] |
| `R2*Rgt16` | -1.88337 | R² × [R > 16] |
| `R2*Lgt4` | -0.405605 | R² × [L > 4] |
| `R2*Lgt16` | 0.4768 | R² × [L > 16] |
| `R2*mwlg2` | -0.661669 | R² × [mean_warp_lines > 2] |
| `R2*mwlg8` | -2e-06 | R² × [mean_warp_lines > 8] |
| `R2*smISB` | -1.77769 | R² × [inner stride <= 8 bytes] |

### 3. in-sample 拟合误差

- 点数：**246**
- 项数：**217**
- **MAPE：10.06%**

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
T = -77.4057
  + 2.00757 * n2L*const
  + 1.66288 * n2L*eb1
  + 1.59235 * n2L*eb2
  + -0.100566 * n2L*eb4
  + -1.14708 * n2L*eb8
  + 2.00757 * n2L*W1
  + -0 * n2L*W2
  + 0 * n2L*W4
  + 0 * n2L*W8
  + -0 * n2L*W16
  + 0 * n2L*W32
  + -0 * n2L*W64
  + 6.25566 * n2L*Clt8
  + -5.13736 * n2L*Clt16
  + -0.732244 * n2L*Clt32
  + 0.091563 * n2L*Clt64
  + 0.091563 * n2L*nlt64
  + -0.061367 * n2L*nge128
  + 0.034427 * n2L*nge256
  + 0 * n2L*Rgt8
  + -0 * n2L*Rgt16
  + -0.019749 * n2L*Lgt4
  + 0.013776 * n2L*Lgt16
  + 0.123475 * n2L*Lgt64
  + -4.47827 * n2L*mwlg2
  + 0.013541 * n2L*mwlg8
  + -4.55433 * n2L*smISB
  + 3.54324 * ov_n_mwl*const
  + 1.47005 * ov_n_mwl*eb1
  + 1.81774 * ov_n_mwl*eb2
  + 1.14045 * ov_n_mwl*eb4
  + -0.884998 * ov_n_mwl*eb8
  + 3.54324 * ov_n_mwl*W1
  + 0 * ov_n_mwl*W8
  + -105.185 * ov_n_mwl*Clt8
  + 8.72331 * ov_n_mwl*Clt16
  + 4.7933 * ov_n_mwl*Clt32
  + -1.75396 * ov_n_mwl*Clt64
  + -1.75396 * ov_n_mwl*nlt64
  + -3.54114 * ov_n_mwl*nge128
  + 3.09511 * ov_n_mwl*nge256
  + 0.275024 * ov_n_mwl*Lgt4
  + -14.0407 * ov_n_mwl*Lgt16
  + 1.95926 * ov_n_mwl*Lgt64
  + -3.82353 * ov_n_mwl*mwlg2
  + 3.16938 * ov_n_mwl*mwlg8
  + 34.534 * ov_n_mwl*smISB
  + 86.0896 * p_T_it*const
  + 24.3746 * p_T_it*eb1
  + 23.2847 * p_T_it*eb2
  + 20.0471 * p_T_it*eb4
  + 18.3832 * p_T_it*eb8
  + 86.0896 * p_T_it*W1
  + 8.01055 * p_T_it*Clt8
  + 48.0601 * p_T_it*Clt16
  + -235.724 * p_T_it*Clt32
  + 0.128918 * p_T_it*Clt64
  + 0.128918 * p_T_it*nlt64
  + 2.07088 * p_T_it*nge128
  + -2.26699 * p_T_it*nge256
  + -0.780404 * p_T_it*Lgt4
  + -296.83 * p_T_it*Lgt16
  + -0.803756 * p_T_it*Lgt64
  + -10.3886 * p_T_it*mwlg2
  + 3.11512 * p_T_it*mwlg8
  + -5.3268 * p_T_it*smISB
  + 7.00385 * Lepl*const
  + 2.77409 * Lepl*eb1
  + 2.88149 * Lepl*eb2
  + 1.30409 * Lepl*eb4
  + 0.044184 * Lepl*eb8
  + 7.00385 * Lepl*W1
  + -4.63148 * Lepl*Clt8
  + -4.42736 * Lepl*Clt16
  + -11.7246 * Lepl*Clt32
  + -3.57507 * Lepl*Clt64
  + -3.57507 * Lepl*nlt64
  + 1.82796 * Lepl*nge128
  + 0.031026 * Lepl*nge256
  + 4.76904 * Lepl*Lgt4
  + 48.6471 * Lepl*Lgt16
  + -1.04668 * Lepl*Lgt64
  + 4.56054 * Lepl*mwlg2
  + -74.4259 * Lepl*mwlg8
  + 12.5932 * Lepl*smISB
  + -116.865 * bucket_worst_2k8*const
  + -27.8073 * bucket_worst_2k8*eb1
  + -29.0505 * bucket_worst_2k8*eb2
  + -29.7526 * bucket_worst_2k8*eb4
  + -30.255 * bucket_worst_2k8*eb8
  + -116.865 * bucket_worst_2k8*W1
  + -17.8583 * bucket_worst_2k8*Clt8
  + -62.4576 * bucket_worst_2k8*Clt16
  + 317.296 * bucket_worst_2k8*Clt32
  + 4.73796 * bucket_worst_2k8*Clt64
  + 4.73796 * bucket_worst_2k8*nlt64
  + -0.800404 * bucket_worst_2k8*nge128
  + -2.18412 * bucket_worst_2k8*nge256
  + -27.5489 * bucket_worst_2k8*Lgt4
  + 265.179 * bucket_worst_2k8*Lgt16
  + -3.21692 * bucket_worst_2k8*Lgt64
  + 27.9661 * bucket_worst_2k8*mwlg2
  + 0.694471 * bucket_worst_2k8*mwlg8
  + 30.1675 * bucket_worst_2k8*smISB
  + 5.96673 * Wn*const
  + -160.017 * Wn*eb1
  + -115.63 * Wn*eb2
  + 65.8317 * Wn*eb4
  + 215.782 * Wn*eb8
  + 5.96673 * Wn*W1
  + 0.031291 * Wn*Clt8
  + 0.344179 * Wn*Clt16
  + -4.08995 * Wn*Clt32
  + 3.28046 * Wn*Clt64
  + 3.28046 * Wn*nlt64
  + 0.064715 * Wn*nge128
  + -0.070844 * Wn*nge256
  + 764.192 * Wn*Lgt4
  + -9.27594 * Wn*Lgt16
  + -0.025117 * Wn*Lgt64
  + 825.408 * Wn*mwlg2
  + 2184.53 * Wn*mwlg8
  + -814.072 * Wn*smISB
  + -0.11592 * nEpl_eb*const
  + -0.766048 * nEpl_eb*eb1
  + -0.237167 * nEpl_eb*eb2
  + 0.419127 * nEpl_eb*eb4
  + 0.468167 * nEpl_eb*eb8
  + -0.11592 * nEpl_eb*W1
  + 0.435974 * nEpl_eb*Clt8
  + 0.075043 * nEpl_eb*Clt16
  + 0.131661 * nEpl_eb*Clt32
  + -0.009596 * nEpl_eb*Clt64
  + -0.009596 * nEpl_eb*nlt64
  + 0.00393 * nEpl_eb*nge128
  + -0.006613 * nEpl_eb*nge256
  + 0.007413 * nEpl_eb*Lgt4
  + -0.036813 * nEpl_eb*Lgt16
  + -0.013559 * nEpl_eb*Lgt64
  + 0.037584 * nEpl_eb*mwlg2
  + 0.010729 * nEpl_eb*mwlg8
  + 0.033418 * nEpl_eb*smISB
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | -77.4057 | 固定开销（循环发射、启动） |
| `n2L*const` | 2.00757 | n² / L × [无（常数项）] |
| `n2L*eb1` | 1.66288 | n² / L × [dtype = int8] |
| `n2L*eb2` | 1.59235 | n² / L × [dtype = float16] |
| `n2L*eb4` | -0.100566 | n² / L × [dtype = float32] |
| `n2L*eb8` | -1.14708 | n² / L × [dtype = int64] |
| `n2L*W1` | 2.00757 | n² / L × [num_warps = 1] |
| `n2L*W2` | -0 | n² / L × [num_warps = 2] |
| `n2L*W4` | 0 | n² / L × [num_warps = 4] |
| `n2L*W8` | 0 | n² / L × [num_warps = 8] |
| `n2L*W16` | -0 | n² / L × [num_warps = 16] |
| `n2L*W32` | 0 | n² / L × [num_warps = 32] |
| `n2L*W64` | -0 | n² / L × [num_warps = 64] |
| `n2L*Clt8` | 6.25566 | n² / L × [C < 8] |
| `n2L*Clt16` | -5.13736 | n² / L × [C < 16] |
| `n2L*Clt32` | -0.732244 | n² / L × [C < 32] |
| `n2L*Clt64` | 0.091563 | n² / L × [C < 64] |
| `n2L*nlt64` | 0.091563 | n² / L × [n < 64] |
| `n2L*nge128` | -0.061367 | n² / L × [n >= 128] |
| `n2L*nge256` | 0.034427 | n² / L × [n >= 256] |
| `n2L*Rgt8` | 0 | n² / L × [R > 8] |
| `n2L*Rgt16` | -0 | n² / L × [R > 16] |
| `n2L*Lgt4` | -0.019749 | n² / L × [L > 4] |
| `n2L*Lgt16` | 0.013776 | n² / L × [L > 16] |
| `n2L*Lgt64` | 0.123475 | n² / L × [L > 64] |
| `n2L*mwlg2` | -4.47827 | n² / L × [mean_warp_lines > 2] |
| `n2L*mwlg8` | 0.013541 | n² / L × [mean_warp_lines > 8] |
| `n2L*smISB` | -4.55433 | n² / L × [inner stride <= 8 bytes] |
| `ov_n_mwl*const` | 3.54324 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [无（常数项）] |
| `ov_n_mwl*eb1` | 1.47005 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = int8] |
| `ov_n_mwl*eb2` | 1.81774 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = float16] |
| `ov_n_mwl*eb4` | 1.14045 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = float32] |
| `ov_n_mwl*eb8` | -0.884998 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = int64] |
| `ov_n_mwl*W1` | 3.54324 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 1] |
| `ov_n_mwl*W8` | 0 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 8] |
| `ov_n_mwl*Clt8` | -105.185 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [C < 8] |
| `ov_n_mwl*Clt16` | 8.72331 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [C < 16] |
| `ov_n_mwl*Clt32` | 4.7933 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [C < 32] |
| `ov_n_mwl*Clt64` | -1.75396 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [C < 64] |
| `ov_n_mwl*nlt64` | -1.75396 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [n < 64] |
| `ov_n_mwl*nge128` | -3.54114 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [n >= 128] |
| `ov_n_mwl*nge256` | 3.09511 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [n >= 256] |
| `ov_n_mwl*Lgt4` | 0.275024 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [L > 4] |
| `ov_n_mwl*Lgt16` | -14.0407 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [L > 16] |
| `ov_n_mwl*Lgt64` | 1.95926 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [L > 64] |
| `ov_n_mwl*mwlg2` | -3.82353 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [mean_warp_lines > 2] |
| `ov_n_mwl*mwlg8` | 3.16938 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [mean_warp_lines > 8] |
| `ov_n_mwl*smISB` | 34.534 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [inner stride <= 8 bytes] |
| `p_T_it*const` | 86.0896 | p_T_it × [无（常数项）] |
| `p_T_it*eb1` | 24.3746 | p_T_it × [dtype = int8] |
| `p_T_it*eb2` | 23.2847 | p_T_it × [dtype = float16] |
| `p_T_it*eb4` | 20.0471 | p_T_it × [dtype = float32] |
| `p_T_it*eb8` | 18.3832 | p_T_it × [dtype = int64] |
| `p_T_it*W1` | 86.0896 | p_T_it × [num_warps = 1] |
| `p_T_it*Clt8` | 8.01055 | p_T_it × [C < 8] |
| `p_T_it*Clt16` | 48.0601 | p_T_it × [C < 16] |
| `p_T_it*Clt32` | -235.724 | p_T_it × [C < 32] |
| `p_T_it*Clt64` | 0.128918 | p_T_it × [C < 64] |
| `p_T_it*nlt64` | 0.128918 | p_T_it × [n < 64] |
| `p_T_it*nge128` | 2.07088 | p_T_it × [n >= 128] |
| `p_T_it*nge256` | -2.26699 | p_T_it × [n >= 256] |
| `p_T_it*Lgt4` | -0.780404 | p_T_it × [L > 4] |
| `p_T_it*Lgt16` | -296.83 | p_T_it × [L > 16] |
| `p_T_it*Lgt64` | -0.803756 | p_T_it × [L > 64] |
| `p_T_it*mwlg2` | -10.3886 | p_T_it × [mean_warp_lines > 2] |
| `p_T_it*mwlg8` | 3.11512 | p_T_it × [mean_warp_lines > 8] |
| `p_T_it*smISB` | -5.3268 | p_T_it × [inner stride <= 8 bytes] |
| `Lepl*const` | 7.00385 | L · epl × [无（常数项）] |
| `Lepl*eb1` | 2.77409 | L · epl × [dtype = int8] |
| `Lepl*eb2` | 2.88149 | L · epl × [dtype = float16] |
| `Lepl*eb4` | 1.30409 | L · epl × [dtype = float32] |
| `Lepl*eb8` | 0.044184 | L · epl × [dtype = int64] |
| `Lepl*W1` | 7.00385 | L · epl × [num_warps = 1] |
| `Lepl*Clt8` | -4.63148 | L · epl × [C < 8] |
| `Lepl*Clt16` | -4.42736 | L · epl × [C < 16] |
| `Lepl*Clt32` | -11.7246 | L · epl × [C < 32] |
| `Lepl*Clt64` | -3.57507 | L · epl × [C < 64] |
| `Lepl*nlt64` | -3.57507 | L · epl × [n < 64] |
| `Lepl*nge128` | 1.82796 | L · epl × [n >= 128] |
| `Lepl*nge256` | 0.031026 | L · epl × [n >= 256] |
| `Lepl*Lgt4` | 4.76904 | L · epl × [L > 4] |
| `Lepl*Lgt16` | 48.6471 | L · epl × [L > 16] |
| `Lepl*Lgt64` | -1.04668 | L · epl × [L > 64] |
| `Lepl*mwlg2` | 4.56054 | L · epl × [mean_warp_lines > 2] |
| `Lepl*mwlg8` | -74.4259 | L · epl × [mean_warp_lines > 8] |
| `Lepl*smISB` | 12.5932 | L · epl × [inner stride <= 8 bytes] |
| `bucket_worst_2k8*const` | -116.865 | 最差 line 桶 (2048..8192) × [无（常数项）] |
| `bucket_worst_2k8*eb1` | -27.8073 | 最差 line 桶 (2048..8192) × [dtype = int8] |
| `bucket_worst_2k8*eb2` | -29.0505 | 最差 line 桶 (2048..8192) × [dtype = float16] |
| `bucket_worst_2k8*eb4` | -29.7526 | 最差 line 桶 (2048..8192) × [dtype = float32] |
| `bucket_worst_2k8*eb8` | -30.255 | 最差 line 桶 (2048..8192) × [dtype = int64] |
| `bucket_worst_2k8*W1` | -116.865 | 最差 line 桶 (2048..8192) × [num_warps = 1] |
| `bucket_worst_2k8*Clt8` | -17.8583 | 最差 line 桶 (2048..8192) × [C < 8] |
| `bucket_worst_2k8*Clt16` | -62.4576 | 最差 line 桶 (2048..8192) × [C < 16] |
| `bucket_worst_2k8*Clt32` | 317.296 | 最差 line 桶 (2048..8192) × [C < 32] |
| `bucket_worst_2k8*Clt64` | 4.73796 | 最差 line 桶 (2048..8192) × [C < 64] |
| `bucket_worst_2k8*nlt64` | 4.73796 | 最差 line 桶 (2048..8192) × [n < 64] |
| `bucket_worst_2k8*nge128` | -0.800404 | 最差 line 桶 (2048..8192) × [n >= 128] |
| `bucket_worst_2k8*nge256` | -2.18412 | 最差 line 桶 (2048..8192) × [n >= 256] |
| `bucket_worst_2k8*Lgt4` | -27.5489 | 最差 line 桶 (2048..8192) × [L > 4] |
| `bucket_worst_2k8*Lgt16` | 265.179 | 最差 line 桶 (2048..8192) × [L > 16] |
| `bucket_worst_2k8*Lgt64` | -3.21692 | 最差 line 桶 (2048..8192) × [L > 64] |
| `bucket_worst_2k8*mwlg2` | 27.9661 | 最差 line 桶 (2048..8192) × [mean_warp_lines > 2] |
| `bucket_worst_2k8*mwlg8` | 0.694471 | 最差 line 桶 (2048..8192) × [mean_warp_lines > 8] |
| `bucket_worst_2k8*smISB` | 30.1675 | 最差 line 桶 (2048..8192) × [inner stride <= 8 bytes] |
| `Wn*const` | 5.96673 | W / n × [无（常数项）] |
| `Wn*eb1` | -160.017 | W / n × [dtype = int8] |
| `Wn*eb2` | -115.63 | W / n × [dtype = float16] |
| `Wn*eb4` | 65.8317 | W / n × [dtype = float32] |
| `Wn*eb8` | 215.782 | W / n × [dtype = int64] |
| `Wn*W1` | 5.96673 | W / n × [num_warps = 1] |
| `Wn*Clt8` | 0.031291 | W / n × [C < 8] |
| `Wn*Clt16` | 0.344179 | W / n × [C < 16] |
| `Wn*Clt32` | -4.08995 | W / n × [C < 32] |
| `Wn*Clt64` | 3.28046 | W / n × [C < 64] |
| `Wn*nlt64` | 3.28046 | W / n × [n < 64] |
| `Wn*nge128` | 0.064715 | W / n × [n >= 128] |
| `Wn*nge256` | -0.070844 | W / n × [n >= 256] |
| `Wn*Lgt4` | 764.192 | W / n × [L > 4] |
| `Wn*Lgt16` | -9.27594 | W / n × [L > 16] |
| `Wn*Lgt64` | -0.025117 | W / n × [L > 64] |
| `Wn*mwlg2` | 825.408 | W / n × [mean_warp_lines > 2] |
| `Wn*mwlg8` | 2184.53 | W / n × [mean_warp_lines > 8] |
| `Wn*smISB` | -814.072 | W / n × [inner stride <= 8 bytes] |
| `nEpl_eb*const` | -0.11592 | n·epl·eb × [无（常数项）] |
| `nEpl_eb*eb1` | -0.766048 | n·epl·eb × [dtype = int8] |
| `nEpl_eb*eb2` | -0.237167 | n·epl·eb × [dtype = float16] |
| `nEpl_eb*eb4` | 0.419127 | n·epl·eb × [dtype = float32] |
| `nEpl_eb*eb8` | 0.468167 | n·epl·eb × [dtype = int64] |
| `nEpl_eb*W1` | -0.11592 | n·epl·eb × [num_warps = 1] |
| `nEpl_eb*Clt8` | 0.435974 | n·epl·eb × [C < 8] |
| `nEpl_eb*Clt16` | 0.075043 | n·epl·eb × [C < 16] |
| `nEpl_eb*Clt32` | 0.131661 | n·epl·eb × [C < 32] |
| `nEpl_eb*Clt64` | -0.009596 | n·epl·eb × [C < 64] |
| `nEpl_eb*nlt64` | -0.009596 | n·epl·eb × [n < 64] |
| `nEpl_eb*nge128` | 0.00393 | n·epl·eb × [n >= 128] |
| `nEpl_eb*nge256` | -0.006613 | n·epl·eb × [n >= 256] |
| `nEpl_eb*Lgt4` | 0.007413 | n·epl·eb × [L > 4] |
| `nEpl_eb*Lgt16` | -0.036813 | n·epl·eb × [L > 16] |
| `nEpl_eb*Lgt64` | -0.013559 | n·epl·eb × [L > 64] |
| `nEpl_eb*mwlg2` | 0.037584 | n·epl·eb × [mean_warp_lines > 2] |
| `nEpl_eb*mwlg8` | 0.010729 | n·epl·eb × [mean_warp_lines > 8] |
| `nEpl_eb*smISB` | 0.033418 | n·epl·eb × [inner stride <= 8 bytes] |

### 3. in-sample 拟合误差

- 点数：**284**
- 项数：**190**
- **MAPE：11.59%**

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
T = 110.651
  + 25.3541 * mn_L_cshortN*const
  + -2.85569 * mn_L_cshortN*eb1
  + 5.62467 * mn_L_cshortN*eb2
  + 22.5851 * mn_L_cshortN*eb4
  + -0 * mn_L_cshortN*eb8
  + 25.3541 * mn_L_cshortN*W1
  + 0 * mn_L_cshortN*W2
  + 0 * mn_L_cshortN*W4
  + 0 * mn_L_cshortN*W8
  + 0 * mn_L_cshortN*W16
  + -0 * mn_L_cshortN*W32
  + -0 * mn_L_cshortN*W64
  + -14.4309 * mn_L_cshortN*Clt8
  + -6.87418 * mn_L_cshortN*Clt16
  + 25.3541 * mn_L_cshortN*Clt32
  + 25.3541 * mn_L_cshortN*Clt64
  + 42.8052 * mn_L_cshortN*nlt64
  + -17.9625 * mn_L_cshortN*nge128
  + -62.8351 * mn_L_cshortN*nge256
  + 58.0196 * mn_L_cshortN*Rgt8
  + -25.9547 * mn_L_cshortN*Rgt16
  + 29.7513 * mn_L_cshortN*Lgt4
  + 4.80774 * mn_L_cshortN*Lgt16
  + -0 * mn_L_cshortN*Lgt64
  + -24.3189 * mn_L_cshortN*mwlg2
  + -65.339 * mn_L_cshortN*mwlg8
  + -34.5257 * mn_L_cshortN*smISB
  + 0.11036 * n2L*const
  + 0.47438 * n2L*eb1
  + 0.593723 * n2L*eb2
  + -0.363895 * n2L*eb4
  + -0.593846 * n2L*eb8
  + 0.11036 * n2L*W1
  + 0 * n2L*W2
  + -0 * n2L*W4
  + -0 * n2L*W16
  + -0 * n2L*W32
  + 0 * n2L*W64
  + -0.161351 * n2L*Clt8
  + 0.081581 * n2L*Clt16
  + -0.004444 * n2L*Clt32
  + 0.034841 * n2L*Clt64
  + 0.08205 * n2L*nlt64
  + -0.002149 * n2L*nge128
  + 0.845911 * n2L*nge256
  + -0.592577 * n2L*Rgt8
  + -0.041007 * n2L*Rgt16
  + -0.150469 * n2L*Lgt4
  + -0.065376 * n2L*Lgt16
  + -0.072244 * n2L*Lgt64
  + -0.036317 * n2L*mwlg2
  + 0.222845 * n2L*mwlg8
  + -0.629821 * n2L*smISB
  + -0.170526 * nhe4*const
  + -0.8468 * nhe4*eb1
  + 0.202148 * nhe4*eb2
  + 0.215136 * nhe4*eb4
  + 0.258991 * nhe4*eb8
  + -0.170526 * nhe4*W1
  + 0 * nhe4*W4
  + -0.106096 * nhe4*Clt8
  + -0.082362 * nhe4*Clt16
  + 0.04753 * nhe4*Clt32
  + -0.058314 * nhe4*Clt64
  + 0.077755 * nhe4*nlt64
  + -0.104735 * nhe4*nge128
  + 0.055014 * nhe4*nge256
  + 0.087555 * nhe4*Rgt8
  + 0.75339 * nhe4*Rgt16
  + 0.063645 * nhe4*Lgt4
  + 0.052511 * nhe4*Lgt16
  + -0.096253 * nhe4*mwlg2
  + 1.34953 * nhe4*smISB
  + -0.029652 * nEpl_eb*const
  + 0.087146 * nEpl_eb*eb1
  + -0.291322 * nEpl_eb*eb2
  + 0.090703 * nEpl_eb*eb4
  + 0.08382 * nEpl_eb*eb8
  + -0.029652 * nEpl_eb*W1
  + 0.024904 * nEpl_eb*Clt8
  + -0.009755 * nEpl_eb*Clt16
  + -0.017093 * nEpl_eb*Clt32
  + -0.007087 * nEpl_eb*Clt64
  + 0.02601 * nEpl_eb*nlt64
  + -0.008762 * nEpl_eb*nge128
  + 0.004349 * nEpl_eb*nge256
  + 0.020332 * nEpl_eb*Rgt8
  + 0.026918 * nEpl_eb*Rgt16
  + 0.028623 * nEpl_eb*Lgt4
  + -0.002481 * nEpl_eb*Lgt16
  + -0.005527 * nEpl_eb*Lgt64
  + 0.051002 * nEpl_eb*mwlg2
  + 0.019172 * nEpl_eb*mwlg8
  + -0.198998 * nEpl_eb*smISB
  + -34.1283 * r_C_ISB*const
  + 1.41691 * r_C_ISB*eb1
  + -7.38808 * r_C_ISB*eb2
  + -16.8084 * r_C_ISB*eb4
  + -11.3487 * r_C_ISB*eb8
  + -34.1283 * r_C_ISB*W1
  + 82.6799 * r_C_ISB*Clt8
  + -1.15072 * r_C_ISB*Clt16
  + -35.9996 * r_C_ISB*Clt32
  + -33.1756 * r_C_ISB*Clt64
  + -1.18815 * r_C_ISB*nlt64
  + -6.92412 * r_C_ISB*nge128
  + -294.764 * r_C_ISB*nge256
  + -55.5429 * r_C_ISB*Rgt8
  + -3225.9 * r_C_ISB*Rgt16
  + 34.781 * r_C_ISB*Lgt4
  + -20.0701 * r_C_ISB*Lgt16
  + 379.393 * r_C_ISB*Lgt64
  + 76.3506 * r_C_ISB*mwlg2
  + -3.28733 * r_C_ISB*mwlg8
  + 86.1985 * r_C_ISB*smISB
  + -11.1695 * burstShort*const
  + 47.9283 * burstShort*eb1
  + 44.3408 * burstShort*eb2
  + -103.439 * burstShort*eb4
  + -11.1695 * burstShort*W1
  + -23.9839 * burstShort*Clt8
  + 36.9331 * burstShort*Clt16
  + -11.1695 * burstShort*Clt32
  + -11.1695 * burstShort*Clt64
  + -132.156 * burstShort*nlt64
  + 201.477 * burstShort*nge128
  + 1192.08 * burstShort*nge256
  + -910.514 * burstShort*Rgt8
  + 536.538 * burstShort*Rgt16
  + -143.424 * burstShort*Lgt4
  + -303.368 * burstShort*Lgt16
  + 96.244 * burstShort*mwlg2
  + 394.281 * burstShort*mwlg8
  + 66.6103 * burstShort*smISB
  + -18.4246 * ov_seg_mwl*const
  + 1.91639 * ov_seg_mwl*eb1
  + -3.31272 * ov_seg_mwl*eb2
  + -3.38349 * ov_seg_mwl*eb4
  + -13.6448 * ov_seg_mwl*eb8
  + -18.4246 * ov_seg_mwl*W1
  + -25.8413 * ov_seg_mwl*Clt8
  + -15.4426 * ov_seg_mwl*Clt16
  + -6.82226 * ov_seg_mwl*Clt32
  + 43.2669 * ov_seg_mwl*Clt64
  + 59.735 * ov_seg_mwl*nlt64
  + -24.857 * ov_seg_mwl*nge128
  + -27.0462 * ov_seg_mwl*nge256
  + 4.55763 * ov_seg_mwl*Rgt8
  + 74.1266 * ov_seg_mwl*Rgt16
  + 22.846 * ov_seg_mwl*Lgt4
  + 22.5667 * ov_seg_mwl*Lgt16
  + 7.10221 * ov_seg_mwl*Lgt64
  + 11.9269 * ov_seg_mwl*mwlg2
  + -25.2801 * ov_seg_mwl*mwlg8
  + 38.6556 * ov_seg_mwl*smISB
  + 8.45338 * line_elems_max*const
  + -1.28422 * line_elems_max*eb1
  + 3.23674 * line_elems_max*eb2
  + 4.99035 * line_elems_max*eb4
  + 1.51052 * line_elems_max*eb8
  + 8.45338 * line_elems_max*W1
  + -4.98808 * line_elems_max*Clt8
  + -7.46908 * line_elems_max*Clt16
  + -0.937236 * line_elems_max*Clt32
  + 10.3589 * line_elems_max*Clt64
  + -6.64289 * line_elems_max*nlt64
  + 15.4854 * line_elems_max*nge128
  + -122.117 * line_elems_max*nge256
  + 102.188 * line_elems_max*Rgt8
  + -85.0138 * line_elems_max*Rgt16
  + -4.65881 * line_elems_max*Lgt4
  + 4.82824 * line_elems_max*Lgt16
  + -44.9332 * line_elems_max*Lgt64
  + -9.76911 * line_elems_max*mwlg2
  + -13.9035 * line_elems_max*mwlg8
  + -10.8276 * line_elems_max*smISB
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 110.651 | 固定开销（循环发射、启动） |
| `mn_L_cshortN*const` | 25.3541 | min(触及的 128B line 数, cshort · n / 32) × [无（常数项）] |
| `mn_L_cshortN*eb1` | -2.85569 | min(触及的 128B line 数, cshort · n / 32) × [dtype = int8] |
| `mn_L_cshortN*eb2` | 5.62467 | min(触及的 128B line 数, cshort · n / 32) × [dtype = float16] |
| `mn_L_cshortN*eb4` | 22.5851 | min(触及的 128B line 数, cshort · n / 32) × [dtype = float32] |
| `mn_L_cshortN*eb8` | -0 | min(触及的 128B line 数, cshort · n / 32) × [dtype = int64] |
| `mn_L_cshortN*W1` | 25.3541 | min(触及的 128B line 数, cshort · n / 32) × [num_warps = 1] |
| `mn_L_cshortN*W2` | 0 | min(触及的 128B line 数, cshort · n / 32) × [num_warps = 2] |
| `mn_L_cshortN*W4` | 0 | min(触及的 128B line 数, cshort · n / 32) × [num_warps = 4] |
| `mn_L_cshortN*W8` | 0 | min(触及的 128B line 数, cshort · n / 32) × [num_warps = 8] |
| `mn_L_cshortN*W16` | 0 | min(触及的 128B line 数, cshort · n / 32) × [num_warps = 16] |
| `mn_L_cshortN*W32` | -0 | min(触及的 128B line 数, cshort · n / 32) × [num_warps = 32] |
| `mn_L_cshortN*W64` | -0 | min(触及的 128B line 数, cshort · n / 32) × [num_warps = 64] |
| `mn_L_cshortN*Clt8` | -14.4309 | min(触及的 128B line 数, cshort · n / 32) × [C < 8] |
| `mn_L_cshortN*Clt16` | -6.87418 | min(触及的 128B line 数, cshort · n / 32) × [C < 16] |
| `mn_L_cshortN*Clt32` | 25.3541 | min(触及的 128B line 数, cshort · n / 32) × [C < 32] |
| `mn_L_cshortN*Clt64` | 25.3541 | min(触及的 128B line 数, cshort · n / 32) × [C < 64] |
| `mn_L_cshortN*nlt64` | 42.8052 | min(触及的 128B line 数, cshort · n / 32) × [n < 64] |
| `mn_L_cshortN*nge128` | -17.9625 | min(触及的 128B line 数, cshort · n / 32) × [n >= 128] |
| `mn_L_cshortN*nge256` | -62.8351 | min(触及的 128B line 数, cshort · n / 32) × [n >= 256] |
| `mn_L_cshortN*Rgt8` | 58.0196 | min(触及的 128B line 数, cshort · n / 32) × [R > 8] |
| `mn_L_cshortN*Rgt16` | -25.9547 | min(触及的 128B line 数, cshort · n / 32) × [R > 16] |
| `mn_L_cshortN*Lgt4` | 29.7513 | min(触及的 128B line 数, cshort · n / 32) × [L > 4] |
| `mn_L_cshortN*Lgt16` | 4.80774 | min(触及的 128B line 数, cshort · n / 32) × [L > 16] |
| `mn_L_cshortN*Lgt64` | -0 | min(触及的 128B line 数, cshort · n / 32) × [L > 64] |
| `mn_L_cshortN*mwlg2` | -24.3189 | min(触及的 128B line 数, cshort · n / 32) × [mean_warp_lines > 2] |
| `mn_L_cshortN*mwlg8` | -65.339 | min(触及的 128B line 数, cshort · n / 32) × [mean_warp_lines > 8] |
| `mn_L_cshortN*smISB` | -34.5257 | min(触及的 128B line 数, cshort · n / 32) × [inner stride <= 8 bytes] |
| `n2L*const` | 0.11036 | n² / L × [无（常数项）] |
| `n2L*eb1` | 0.47438 | n² / L × [dtype = int8] |
| `n2L*eb2` | 0.593723 | n² / L × [dtype = float16] |
| `n2L*eb4` | -0.363895 | n² / L × [dtype = float32] |
| `n2L*eb8` | -0.593846 | n² / L × [dtype = int64] |
| `n2L*W1` | 0.11036 | n² / L × [num_warps = 1] |
| `n2L*W2` | 0 | n² / L × [num_warps = 2] |
| `n2L*W4` | -0 | n² / L × [num_warps = 4] |
| `n2L*W16` | -0 | n² / L × [num_warps = 16] |
| `n2L*W32` | -0 | n² / L × [num_warps = 32] |
| `n2L*W64` | 0 | n² / L × [num_warps = 64] |
| `n2L*Clt8` | -0.161351 | n² / L × [C < 8] |
| `n2L*Clt16` | 0.081581 | n² / L × [C < 16] |
| `n2L*Clt32` | -0.004444 | n² / L × [C < 32] |
| `n2L*Clt64` | 0.034841 | n² / L × [C < 64] |
| `n2L*nlt64` | 0.08205 | n² / L × [n < 64] |
| `n2L*nge128` | -0.002149 | n² / L × [n >= 128] |
| `n2L*nge256` | 0.845911 | n² / L × [n >= 256] |
| `n2L*Rgt8` | -0.592577 | n² / L × [R > 8] |
| `n2L*Rgt16` | -0.041007 | n² / L × [R > 16] |
| `n2L*Lgt4` | -0.150469 | n² / L × [L > 4] |
| `n2L*Lgt16` | -0.065376 | n² / L × [L > 16] |
| `n2L*Lgt64` | -0.072244 | n² / L × [L > 64] |
| `n2L*mwlg2` | -0.036317 | n² / L × [mean_warp_lines > 2] |
| `n2L*mwlg8` | 0.222845 | n² / L × [mean_warp_lines > 8] |
| `n2L*smISB` | -0.629821 | n² / L × [inner stride <= 8 bytes] |
| `nhe4*const` | -0.170526 | n · max(0, 128/ISB − 4) × [无（常数项）] |
| `nhe4*eb1` | -0.8468 | n · max(0, 128/ISB − 4) × [dtype = int8] |
| `nhe4*eb2` | 0.202148 | n · max(0, 128/ISB − 4) × [dtype = float16] |
| `nhe4*eb4` | 0.215136 | n · max(0, 128/ISB − 4) × [dtype = float32] |
| `nhe4*eb8` | 0.258991 | n · max(0, 128/ISB − 4) × [dtype = int64] |
| `nhe4*W1` | -0.170526 | n · max(0, 128/ISB − 4) × [num_warps = 1] |
| `nhe4*W4` | 0 | n · max(0, 128/ISB − 4) × [num_warps = 4] |
| `nhe4*Clt8` | -0.106096 | n · max(0, 128/ISB − 4) × [C < 8] |
| `nhe4*Clt16` | -0.082362 | n · max(0, 128/ISB − 4) × [C < 16] |
| `nhe4*Clt32` | 0.04753 | n · max(0, 128/ISB − 4) × [C < 32] |
| `nhe4*Clt64` | -0.058314 | n · max(0, 128/ISB − 4) × [C < 64] |
| `nhe4*nlt64` | 0.077755 | n · max(0, 128/ISB − 4) × [n < 64] |
| `nhe4*nge128` | -0.104735 | n · max(0, 128/ISB − 4) × [n >= 128] |
| `nhe4*nge256` | 0.055014 | n · max(0, 128/ISB − 4) × [n >= 256] |
| `nhe4*Rgt8` | 0.087555 | n · max(0, 128/ISB − 4) × [R > 8] |
| `nhe4*Rgt16` | 0.75339 | n · max(0, 128/ISB − 4) × [R > 16] |
| `nhe4*Lgt4` | 0.063645 | n · max(0, 128/ISB − 4) × [L > 4] |
| `nhe4*Lgt16` | 0.052511 | n · max(0, 128/ISB − 4) × [L > 16] |
| `nhe4*mwlg2` | -0.096253 | n · max(0, 128/ISB − 4) × [mean_warp_lines > 2] |
| `nhe4*smISB` | 1.34953 | n · max(0, 128/ISB − 4) × [inner stride <= 8 bytes] |
| `nEpl_eb*const` | -0.029652 | n·epl·eb × [无（常数项）] |
| `nEpl_eb*eb1` | 0.087146 | n·epl·eb × [dtype = int8] |
| `nEpl_eb*eb2` | -0.291322 | n·epl·eb × [dtype = float16] |
| `nEpl_eb*eb4` | 0.090703 | n·epl·eb × [dtype = float32] |
| `nEpl_eb*eb8` | 0.08382 | n·epl·eb × [dtype = int64] |
| `nEpl_eb*W1` | -0.029652 | n·epl·eb × [num_warps = 1] |
| `nEpl_eb*Clt8` | 0.024904 | n·epl·eb × [C < 8] |
| `nEpl_eb*Clt16` | -0.009755 | n·epl·eb × [C < 16] |
| `nEpl_eb*Clt32` | -0.017093 | n·epl·eb × [C < 32] |
| `nEpl_eb*Clt64` | -0.007087 | n·epl·eb × [C < 64] |
| `nEpl_eb*nlt64` | 0.02601 | n·epl·eb × [n < 64] |
| `nEpl_eb*nge128` | -0.008762 | n·epl·eb × [n >= 128] |
| `nEpl_eb*nge256` | 0.004349 | n·epl·eb × [n >= 256] |
| `nEpl_eb*Rgt8` | 0.020332 | n·epl·eb × [R > 8] |
| `nEpl_eb*Rgt16` | 0.026918 | n·epl·eb × [R > 16] |
| `nEpl_eb*Lgt4` | 0.028623 | n·epl·eb × [L > 4] |
| `nEpl_eb*Lgt16` | -0.002481 | n·epl·eb × [L > 16] |
| `nEpl_eb*Lgt64` | -0.005527 | n·epl·eb × [L > 64] |
| `nEpl_eb*mwlg2` | 0.051002 | n·epl·eb × [mean_warp_lines > 2] |
| `nEpl_eb*mwlg8` | 0.019172 | n·epl·eb × [mean_warp_lines > 8] |
| `nEpl_eb*smISB` | -0.198998 | n·epl·eb × [inner stride <= 8 bytes] |
| `r_C_ISB*const` | -34.1283 | r_C_ISB × [无（常数项）] |
| `r_C_ISB*eb1` | 1.41691 | r_C_ISB × [dtype = int8] |
| `r_C_ISB*eb2` | -7.38808 | r_C_ISB × [dtype = float16] |
| `r_C_ISB*eb4` | -16.8084 | r_C_ISB × [dtype = float32] |
| `r_C_ISB*eb8` | -11.3487 | r_C_ISB × [dtype = int64] |
| `r_C_ISB*W1` | -34.1283 | r_C_ISB × [num_warps = 1] |
| `r_C_ISB*Clt8` | 82.6799 | r_C_ISB × [C < 8] |
| `r_C_ISB*Clt16` | -1.15072 | r_C_ISB × [C < 16] |
| `r_C_ISB*Clt32` | -35.9996 | r_C_ISB × [C < 32] |
| `r_C_ISB*Clt64` | -33.1756 | r_C_ISB × [C < 64] |
| `r_C_ISB*nlt64` | -1.18815 | r_C_ISB × [n < 64] |
| `r_C_ISB*nge128` | -6.92412 | r_C_ISB × [n >= 128] |
| `r_C_ISB*nge256` | -294.764 | r_C_ISB × [n >= 256] |
| `r_C_ISB*Rgt8` | -55.5429 | r_C_ISB × [R > 8] |
| `r_C_ISB*Rgt16` | -3225.9 | r_C_ISB × [R > 16] |
| `r_C_ISB*Lgt4` | 34.781 | r_C_ISB × [L > 4] |
| `r_C_ISB*Lgt16` | -20.0701 | r_C_ISB × [L > 16] |
| `r_C_ISB*Lgt64` | 379.393 | r_C_ISB × [L > 64] |
| `r_C_ISB*mwlg2` | 76.3506 | r_C_ISB × [mean_warp_lines > 2] |
| `r_C_ISB*mwlg8` | -3.28733 | r_C_ISB × [mean_warp_lines > 8] |
| `r_C_ISB*smISB` | 86.1985 | r_C_ISB × [inner stride <= 8 bytes] |
| `burstShort*const` | -11.1695 | [C·eb < 32] 指示 × [无（常数项）] |
| `burstShort*eb1` | 47.9283 | [C·eb < 32] 指示 × [dtype = int8] |
| `burstShort*eb2` | 44.3408 | [C·eb < 32] 指示 × [dtype = float16] |
| `burstShort*eb4` | -103.439 | [C·eb < 32] 指示 × [dtype = float32] |
| `burstShort*W1` | -11.1695 | [C·eb < 32] 指示 × [num_warps = 1] |
| `burstShort*Clt8` | -23.9839 | [C·eb < 32] 指示 × [C < 8] |
| `burstShort*Clt16` | 36.9331 | [C·eb < 32] 指示 × [C < 16] |
| `burstShort*Clt32` | -11.1695 | [C·eb < 32] 指示 × [C < 32] |
| `burstShort*Clt64` | -11.1695 | [C·eb < 32] 指示 × [C < 64] |
| `burstShort*nlt64` | -132.156 | [C·eb < 32] 指示 × [n < 64] |
| `burstShort*nge128` | 201.477 | [C·eb < 32] 指示 × [n >= 128] |
| `burstShort*nge256` | 1192.08 | [C·eb < 32] 指示 × [n >= 256] |
| `burstShort*Rgt8` | -910.514 | [C·eb < 32] 指示 × [R > 8] |
| `burstShort*Rgt16` | 536.538 | [C·eb < 32] 指示 × [R > 16] |
| `burstShort*Lgt4` | -143.424 | [C·eb < 32] 指示 × [L > 4] |
| `burstShort*Lgt16` | -303.368 | [C·eb < 32] 指示 × [L > 16] |
| `burstShort*mwlg2` | 96.244 | [C·eb < 32] 指示 × [mean_warp_lines > 2] |
| `burstShort*mwlg8` | 394.281 | [C·eb < 32] 指示 × [mean_warp_lines > 8] |
| `burstShort*smISB` | 66.6103 | [C·eb < 32] 指示 × [inner stride <= 8 bytes] |
| `ov_seg_mwl*const` | -18.4246 | ov_seg_mwl × [无（常数项）] |
| `ov_seg_mwl*eb1` | 1.91639 | ov_seg_mwl × [dtype = int8] |
| `ov_seg_mwl*eb2` | -3.31272 | ov_seg_mwl × [dtype = float16] |
| `ov_seg_mwl*eb4` | -3.38349 | ov_seg_mwl × [dtype = float32] |
| `ov_seg_mwl*eb8` | -13.6448 | ov_seg_mwl × [dtype = int64] |
| `ov_seg_mwl*W1` | -18.4246 | ov_seg_mwl × [num_warps = 1] |
| `ov_seg_mwl*Clt8` | -25.8413 | ov_seg_mwl × [C < 8] |
| `ov_seg_mwl*Clt16` | -15.4426 | ov_seg_mwl × [C < 16] |
| `ov_seg_mwl*Clt32` | -6.82226 | ov_seg_mwl × [C < 32] |
| `ov_seg_mwl*Clt64` | 43.2669 | ov_seg_mwl × [C < 64] |
| `ov_seg_mwl*nlt64` | 59.735 | ov_seg_mwl × [n < 64] |
| `ov_seg_mwl*nge128` | -24.857 | ov_seg_mwl × [n >= 128] |
| `ov_seg_mwl*nge256` | -27.0462 | ov_seg_mwl × [n >= 256] |
| `ov_seg_mwl*Rgt8` | 4.55763 | ov_seg_mwl × [R > 8] |
| `ov_seg_mwl*Rgt16` | 74.1266 | ov_seg_mwl × [R > 16] |
| `ov_seg_mwl*Lgt4` | 22.846 | ov_seg_mwl × [L > 4] |
| `ov_seg_mwl*Lgt16` | 22.5667 | ov_seg_mwl × [L > 16] |
| `ov_seg_mwl*Lgt64` | 7.10221 | ov_seg_mwl × [L > 64] |
| `ov_seg_mwl*mwlg2` | 11.9269 | ov_seg_mwl × [mean_warp_lines > 2] |
| `ov_seg_mwl*mwlg8` | -25.2801 | ov_seg_mwl × [mean_warp_lines > 8] |
| `ov_seg_mwl*smISB` | 38.6556 | ov_seg_mwl × [inner stride <= 8 bytes] |
| `line_elems_max*const` | 8.45338 | 单 line 内最多元素数 × [无（常数项）] |
| `line_elems_max*eb1` | -1.28422 | 单 line 内最多元素数 × [dtype = int8] |
| `line_elems_max*eb2` | 3.23674 | 单 line 内最多元素数 × [dtype = float16] |
| `line_elems_max*eb4` | 4.99035 | 单 line 内最多元素数 × [dtype = float32] |
| `line_elems_max*eb8` | 1.51052 | 单 line 内最多元素数 × [dtype = int64] |
| `line_elems_max*W1` | 8.45338 | 单 line 内最多元素数 × [num_warps = 1] |
| `line_elems_max*Clt8` | -4.98808 | 单 line 内最多元素数 × [C < 8] |
| `line_elems_max*Clt16` | -7.46908 | 单 line 内最多元素数 × [C < 16] |
| `line_elems_max*Clt32` | -0.937236 | 单 line 内最多元素数 × [C < 32] |
| `line_elems_max*Clt64` | 10.3589 | 单 line 内最多元素数 × [C < 64] |
| `line_elems_max*nlt64` | -6.64289 | 单 line 内最多元素数 × [n < 64] |
| `line_elems_max*nge128` | 15.4854 | 单 line 内最多元素数 × [n >= 128] |
| `line_elems_max*nge256` | -122.117 | 单 line 内最多元素数 × [n >= 256] |
| `line_elems_max*Rgt8` | 102.188 | 单 line 内最多元素数 × [R > 8] |
| `line_elems_max*Rgt16` | -85.0138 | 单 line 内最多元素数 × [R > 16] |
| `line_elems_max*Lgt4` | -4.65881 | 单 line 内最多元素数 × [L > 4] |
| `line_elems_max*Lgt16` | 4.82824 | 单 line 内最多元素数 × [L > 16] |
| `line_elems_max*Lgt64` | -44.9332 | 单 line 内最多元素数 × [L > 64] |
| `line_elems_max*mwlg2` | -9.76911 | 单 line 内最多元素数 × [mean_warp_lines > 2] |
| `line_elems_max*mwlg8` | -13.9035 | 单 line 内最多元素数 × [mean_warp_lines > 8] |
| `line_elems_max*smISB` | -10.8276 | 单 line 内最多元素数 × [inner stride <= 8 bytes] |

### 3. in-sample 拟合误差

- 点数：**750**
- 项数：**217**
- **MAPE：9.90%**

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
T = -2.16106
  + 53.1454 * line_elems_max*const
  + -31.2667 * line_elems_max*eb1
  + 9.26607 * line_elems_max*eb2
  + 25.5474 * line_elems_max*eb4
  + 49.5986 * line_elems_max*eb8
  + 53.1454 * line_elems_max*W1
  + -0 * line_elems_max*W16
  + -12.8225 * line_elems_max*Clt8
  + -19.9381 * line_elems_max*Clt16
  + -19.8428 * line_elems_max*Clt32
  + -4.57306 * line_elems_max*Clt64
  + -4.57306 * line_elems_max*nlt64
  + 86.8823 * line_elems_max*nge128
  + -25.9683 * line_elems_max*nge256
  + -125.567 * line_elems_max*Lgt4
  + -29.914 * line_elems_max*Lgt16
  + -228.86 * line_elems_max*Lgt64
  + -2.68602 * line_elems_max*mwlg2
  + 179.973 * line_elems_max*mwlg8
  + 14.9645 * line_elems_max*smISB
  + -0.222675 * n2L*const
  + 0.746086 * n2L*eb1
  + 0.391891 * n2L*eb2
  + -0.239185 * n2L*eb4
  + -1.12147 * n2L*eb8
  + -0.222675 * n2L*W1
  + -6.12058 * n2L*Clt8
  + 23.2232 * n2L*Clt16
  + -7.29901 * n2L*Clt32
  + 1.02065 * n2L*Clt64
  + 1.02065 * n2L*nlt64
  + 1.14678 * n2L*nge128
  + 0.088611 * n2L*nge256
  + 0.215948 * n2L*Lgt4
  + -0.084344 * n2L*Lgt16
  + 0.442365 * n2L*Lgt64
  + -1.68543 * n2L*mwlg2
  + 0.294931 * n2L*mwlg8
  + -1.52665 * n2L*smISB
  + 8.9473 * C*const
  + 1.25273 * C*eb1
  + 1.99299 * C*eb2
  + 2.41929 * C*eb4
  + 3.28229 * C*eb8
  + 8.9473 * C*W1
  + 0.34722 * C*Clt8
  + 0.732947 * C*Clt16
  + -3.70946 * C*Clt32
  + 4.71263 * C*Clt64
  + 4.71263 * C*nlt64
  + -3.03197 * C*nge128
  + 0.993058 * C*nge256
  + -11.4774 * C*Lgt4
  + 20.4767 * C*Lgt16
  + -3.43372 * C*Lgt64
  + -6.93736 * C*mwlg2
  + -15.7958 * C*mwlg8
  + 1.04359 * C*smISB
  + 12.1326 * r_C_rl*const
  + 40.7984 * r_C_rl*eb1
  + 1.45689 * r_C_rl*eb2
  + -9.71279 * r_C_rl*eb4
  + -20.4099 * r_C_rl*eb8
  + 12.1326 * r_C_rl*W1
  + 149.38 * r_C_rl*Clt8
  + -223.907 * r_C_rl*Clt16
  + 159.954 * r_C_rl*Clt32
  + -38.9232 * r_C_rl*Clt64
  + -38.9232 * r_C_rl*nlt64
  + -168.253 * r_C_rl*nge128
  + 13.777 * r_C_rl*nge256
  + 171.162 * r_C_rl*Lgt4
  + 60.228 * r_C_rl*Lgt16
  + 182.227 * r_C_rl*Lgt64
  + -2.15478 * r_C_rl*mwlg2
  + -288.672 * r_C_rl*mwlg8
  + -2.14615 * r_C_rl*smISB
  + 0.103361 * nEpl_eb*const
  + 0.003698 * nEpl_eb*eb1
  + 0.047376 * nEpl_eb*eb2
  + 0.049609 * nEpl_eb*eb4
  + 0.002679 * nEpl_eb*eb8
  + 0.103361 * nEpl_eb*W1
  + -0.024911 * nEpl_eb*Clt8
  + -0.081975 * nEpl_eb*Clt16
  + -0.087389 * nEpl_eb*Clt32
  + -0.025051 * nEpl_eb*Clt64
  + -0.025051 * nEpl_eb*nlt64
  + 0.030455 * nEpl_eb*nge128
  + 0.043217 * nEpl_eb*nge256
  + -0.032517 * nEpl_eb*Lgt4
  + -0.022955 * nEpl_eb*Lgt16
  + 0.004449 * nEpl_eb*Lgt64
  + 0.01774 * nEpl_eb*mwlg2
  + -0.070739 * nEpl_eb*mwlg8
  + 0.001638 * nEpl_eb*smISB
  + -12.5786 * le_isb32xN*const
  + -2.56927 * le_isb32xN*eb1
  + -2.73804 * le_isb32xN*eb2
  + -2.9457 * le_isb32xN*eb4
  + -4.32558 * le_isb32xN*eb8
  + -12.5786 * le_isb32xN*W1
  + -17.8497 * le_isb32xN*Clt8
  + 22.6212 * le_isb32xN*Clt16
  + 22.1057 * le_isb32xN*Clt32
  + -9.92094 * le_isb32xN*Clt64
  + -9.92094 * le_isb32xN*nlt64
  + 5.54422 * le_isb32xN*nge128
  + -3.34755 * le_isb32xN*nge256
  + 9.95558 * le_isb32xN*Lgt4
  + -39.1024 * le_isb32xN*Lgt16
  + 4.11362 * le_isb32xN*Lgt64
  + 18.618 * le_isb32xN*mwlg2
  + 1.04359 * le_isb32xN*smISB
  + -8.76702 * r_T_it*const
  + -1.64889 * r_T_it*eb1
  + -1.95852 * r_T_it*eb2
  + -2.23397 * r_T_it*eb4
  + -2.92564 * r_T_it*eb8
  + -8.76702 * r_T_it*W1
  + 2.77776 * r_T_it*Clt8
  + 4.32067 * r_T_it*Clt16
  + -4.56415 * r_T_it*Clt32
  + 3.85795 * r_T_it*Clt64
  + 3.85795 * r_T_it*nlt64
  + -14.4416 * r_T_it*nge128
  + -14.1901 * r_T_it*nge256
  + 4.46555 * r_T_it*Lgt4
  + -44.2845 * r_T_it*Lgt16
  + 102.098 * r_T_it*Lgt64
  + 4.02905 * r_T_it*mwlg2
  + 20.0597 * r_T_it*mwlg8
  + -1.11597 * r_T_it*smISB
  + 8.8657 * le_L128xN*const
  + 1.71041 * le_L128xN*eb1
  + 1.92224 * le_L128xN*eb2
  + 2.16364 * le_L128xN*eb4
  + 3.0694 * le_L128xN*eb8
  + 8.8657 * le_L128xN*W1
  + 0.34722 * le_L128xN*Clt8
  + 0.732947 * le_L128xN*Clt16
  + -3.70946 * le_L128xN*Clt32
  + 4.71263 * le_L128xN*Clt64
  + 4.71263 * le_L128xN*nlt64
  + -3.11356 * le_L128xN*nge128
  + 0.911461 * le_L128xN*nge256
  + -11.559 * le_L128xN*Lgt4
  + 20.3951 * le_L128xN*Lgt16
  + -3.51532 * le_L128xN*Lgt64
  + -7.01896 * le_L128xN*mwlg2
  + -15.8774 * le_L128xN*mwlg8
  + 1.04359 * le_L128xN*smISB
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | -2.16106 | 固定开销（循环发射、启动） |
| `line_elems_max*const` | 53.1454 | 单 line 内最多元素数 × [无（常数项）] |
| `line_elems_max*eb1` | -31.2667 | 单 line 内最多元素数 × [dtype = int8] |
| `line_elems_max*eb2` | 9.26607 | 单 line 内最多元素数 × [dtype = float16] |
| `line_elems_max*eb4` | 25.5474 | 单 line 内最多元素数 × [dtype = float32] |
| `line_elems_max*eb8` | 49.5986 | 单 line 内最多元素数 × [dtype = int64] |
| `line_elems_max*W1` | 53.1454 | 单 line 内最多元素数 × [num_warps = 1] |
| `line_elems_max*W16` | -0 | 单 line 内最多元素数 × [num_warps = 16] |
| `line_elems_max*Clt8` | -12.8225 | 单 line 内最多元素数 × [C < 8] |
| `line_elems_max*Clt16` | -19.9381 | 单 line 内最多元素数 × [C < 16] |
| `line_elems_max*Clt32` | -19.8428 | 单 line 内最多元素数 × [C < 32] |
| `line_elems_max*Clt64` | -4.57306 | 单 line 内最多元素数 × [C < 64] |
| `line_elems_max*nlt64` | -4.57306 | 单 line 内最多元素数 × [n < 64] |
| `line_elems_max*nge128` | 86.8823 | 单 line 内最多元素数 × [n >= 128] |
| `line_elems_max*nge256` | -25.9683 | 单 line 内最多元素数 × [n >= 256] |
| `line_elems_max*Lgt4` | -125.567 | 单 line 内最多元素数 × [L > 4] |
| `line_elems_max*Lgt16` | -29.914 | 单 line 内最多元素数 × [L > 16] |
| `line_elems_max*Lgt64` | -228.86 | 单 line 内最多元素数 × [L > 64] |
| `line_elems_max*mwlg2` | -2.68602 | 单 line 内最多元素数 × [mean_warp_lines > 2] |
| `line_elems_max*mwlg8` | 179.973 | 单 line 内最多元素数 × [mean_warp_lines > 8] |
| `line_elems_max*smISB` | 14.9645 | 单 line 内最多元素数 × [inner stride <= 8 bytes] |
| `n2L*const` | -0.222675 | n² / L × [无（常数项）] |
| `n2L*eb1` | 0.746086 | n² / L × [dtype = int8] |
| `n2L*eb2` | 0.391891 | n² / L × [dtype = float16] |
| `n2L*eb4` | -0.239185 | n² / L × [dtype = float32] |
| `n2L*eb8` | -1.12147 | n² / L × [dtype = int64] |
| `n2L*W1` | -0.222675 | n² / L × [num_warps = 1] |
| `n2L*Clt8` | -6.12058 | n² / L × [C < 8] |
| `n2L*Clt16` | 23.2232 | n² / L × [C < 16] |
| `n2L*Clt32` | -7.29901 | n² / L × [C < 32] |
| `n2L*Clt64` | 1.02065 | n² / L × [C < 64] |
| `n2L*nlt64` | 1.02065 | n² / L × [n < 64] |
| `n2L*nge128` | 1.14678 | n² / L × [n >= 128] |
| `n2L*nge256` | 0.088611 | n² / L × [n >= 256] |
| `n2L*Lgt4` | 0.215948 | n² / L × [L > 4] |
| `n2L*Lgt16` | -0.084344 | n² / L × [L > 16] |
| `n2L*Lgt64` | 0.442365 | n² / L × [L > 64] |
| `n2L*mwlg2` | -1.68543 | n² / L × [mean_warp_lines > 2] |
| `n2L*mwlg8` | 0.294931 | n² / L × [mean_warp_lines > 8] |
| `n2L*smISB` | -1.52665 | n² / L × [inner stride <= 8 bytes] |
| `C*const` | 8.9473 | C × [无（常数项）] |
| `C*eb1` | 1.25273 | C × [dtype = int8] |
| `C*eb2` | 1.99299 | C × [dtype = float16] |
| `C*eb4` | 2.41929 | C × [dtype = float32] |
| `C*eb8` | 3.28229 | C × [dtype = int64] |
| `C*W1` | 8.9473 | C × [num_warps = 1] |
| `C*Clt8` | 0.34722 | C × [C < 8] |
| `C*Clt16` | 0.732947 | C × [C < 16] |
| `C*Clt32` | -3.70946 | C × [C < 32] |
| `C*Clt64` | 4.71263 | C × [C < 64] |
| `C*nlt64` | 4.71263 | C × [n < 64] |
| `C*nge128` | -3.03197 | C × [n >= 128] |
| `C*nge256` | 0.993058 | C × [n >= 256] |
| `C*Lgt4` | -11.4774 | C × [L > 4] |
| `C*Lgt16` | 20.4767 | C × [L > 16] |
| `C*Lgt64` | -3.43372 | C × [L > 64] |
| `C*mwlg2` | -6.93736 | C × [mean_warp_lines > 2] |
| `C*mwlg8` | -15.7958 | C × [mean_warp_lines > 8] |
| `C*smISB` | 1.04359 | C × [inner stride <= 8 bytes] |
| `r_C_rl*const` | 12.1326 | r_C_rl × [无（常数项）] |
| `r_C_rl*eb1` | 40.7984 | r_C_rl × [dtype = int8] |
| `r_C_rl*eb2` | 1.45689 | r_C_rl × [dtype = float16] |
| `r_C_rl*eb4` | -9.71279 | r_C_rl × [dtype = float32] |
| `r_C_rl*eb8` | -20.4099 | r_C_rl × [dtype = int64] |
| `r_C_rl*W1` | 12.1326 | r_C_rl × [num_warps = 1] |
| `r_C_rl*Clt8` | 149.38 | r_C_rl × [C < 8] |
| `r_C_rl*Clt16` | -223.907 | r_C_rl × [C < 16] |
| `r_C_rl*Clt32` | 159.954 | r_C_rl × [C < 32] |
| `r_C_rl*Clt64` | -38.9232 | r_C_rl × [C < 64] |
| `r_C_rl*nlt64` | -38.9232 | r_C_rl × [n < 64] |
| `r_C_rl*nge128` | -168.253 | r_C_rl × [n >= 128] |
| `r_C_rl*nge256` | 13.777 | r_C_rl × [n >= 256] |
| `r_C_rl*Lgt4` | 171.162 | r_C_rl × [L > 4] |
| `r_C_rl*Lgt16` | 60.228 | r_C_rl × [L > 16] |
| `r_C_rl*Lgt64` | 182.227 | r_C_rl × [L > 64] |
| `r_C_rl*mwlg2` | -2.15478 | r_C_rl × [mean_warp_lines > 2] |
| `r_C_rl*mwlg8` | -288.672 | r_C_rl × [mean_warp_lines > 8] |
| `r_C_rl*smISB` | -2.14615 | r_C_rl × [inner stride <= 8 bytes] |
| `nEpl_eb*const` | 0.103361 | n·epl·eb × [无（常数项）] |
| `nEpl_eb*eb1` | 0.003698 | n·epl·eb × [dtype = int8] |
| `nEpl_eb*eb2` | 0.047376 | n·epl·eb × [dtype = float16] |
| `nEpl_eb*eb4` | 0.049609 | n·epl·eb × [dtype = float32] |
| `nEpl_eb*eb8` | 0.002679 | n·epl·eb × [dtype = int64] |
| `nEpl_eb*W1` | 0.103361 | n·epl·eb × [num_warps = 1] |
| `nEpl_eb*Clt8` | -0.024911 | n·epl·eb × [C < 8] |
| `nEpl_eb*Clt16` | -0.081975 | n·epl·eb × [C < 16] |
| `nEpl_eb*Clt32` | -0.087389 | n·epl·eb × [C < 32] |
| `nEpl_eb*Clt64` | -0.025051 | n·epl·eb × [C < 64] |
| `nEpl_eb*nlt64` | -0.025051 | n·epl·eb × [n < 64] |
| `nEpl_eb*nge128` | 0.030455 | n·epl·eb × [n >= 128] |
| `nEpl_eb*nge256` | 0.043217 | n·epl·eb × [n >= 256] |
| `nEpl_eb*Lgt4` | -0.032517 | n·epl·eb × [L > 4] |
| `nEpl_eb*Lgt16` | -0.022955 | n·epl·eb × [L > 16] |
| `nEpl_eb*Lgt64` | 0.004449 | n·epl·eb × [L > 64] |
| `nEpl_eb*mwlg2` | 0.01774 | n·epl·eb × [mean_warp_lines > 2] |
| `nEpl_eb*mwlg8` | -0.070739 | n·epl·eb × [mean_warp_lines > 8] |
| `nEpl_eb*smISB` | 0.001638 | n·epl·eb × [inner stride <= 8 bytes] |
| `le_isb32xN*const` | -12.5786 | le_isb32xN × [无（常数项）] |
| `le_isb32xN*eb1` | -2.56927 | le_isb32xN × [dtype = int8] |
| `le_isb32xN*eb2` | -2.73804 | le_isb32xN × [dtype = float16] |
| `le_isb32xN*eb4` | -2.9457 | le_isb32xN × [dtype = float32] |
| `le_isb32xN*eb8` | -4.32558 | le_isb32xN × [dtype = int64] |
| `le_isb32xN*W1` | -12.5786 | le_isb32xN × [num_warps = 1] |
| `le_isb32xN*Clt8` | -17.8497 | le_isb32xN × [C < 8] |
| `le_isb32xN*Clt16` | 22.6212 | le_isb32xN × [C < 16] |
| `le_isb32xN*Clt32` | 22.1057 | le_isb32xN × [C < 32] |
| `le_isb32xN*Clt64` | -9.92094 | le_isb32xN × [C < 64] |
| `le_isb32xN*nlt64` | -9.92094 | le_isb32xN × [n < 64] |
| `le_isb32xN*nge128` | 5.54422 | le_isb32xN × [n >= 128] |
| `le_isb32xN*nge256` | -3.34755 | le_isb32xN × [n >= 256] |
| `le_isb32xN*Lgt4` | 9.95558 | le_isb32xN × [L > 4] |
| `le_isb32xN*Lgt16` | -39.1024 | le_isb32xN × [L > 16] |
| `le_isb32xN*Lgt64` | 4.11362 | le_isb32xN × [L > 64] |
| `le_isb32xN*mwlg2` | 18.618 | le_isb32xN × [mean_warp_lines > 2] |
| `le_isb32xN*smISB` | 1.04359 | le_isb32xN × [inner stride <= 8 bytes] |
| `r_T_it*const` | -8.76702 | r_T_it × [无（常数项）] |
| `r_T_it*eb1` | -1.64889 | r_T_it × [dtype = int8] |
| `r_T_it*eb2` | -1.95852 | r_T_it × [dtype = float16] |
| `r_T_it*eb4` | -2.23397 | r_T_it × [dtype = float32] |
| `r_T_it*eb8` | -2.92564 | r_T_it × [dtype = int64] |
| `r_T_it*W1` | -8.76702 | r_T_it × [num_warps = 1] |
| `r_T_it*Clt8` | 2.77776 | r_T_it × [C < 8] |
| `r_T_it*Clt16` | 4.32067 | r_T_it × [C < 16] |
| `r_T_it*Clt32` | -4.56415 | r_T_it × [C < 32] |
| `r_T_it*Clt64` | 3.85795 | r_T_it × [C < 64] |
| `r_T_it*nlt64` | 3.85795 | r_T_it × [n < 64] |
| `r_T_it*nge128` | -14.4416 | r_T_it × [n >= 128] |
| `r_T_it*nge256` | -14.1901 | r_T_it × [n >= 256] |
| `r_T_it*Lgt4` | 4.46555 | r_T_it × [L > 4] |
| `r_T_it*Lgt16` | -44.2845 | r_T_it × [L > 16] |
| `r_T_it*Lgt64` | 102.098 | r_T_it × [L > 64] |
| `r_T_it*mwlg2` | 4.02905 | r_T_it × [mean_warp_lines > 2] |
| `r_T_it*mwlg8` | 20.0597 | r_T_it × [mean_warp_lines > 8] |
| `r_T_it*smISB` | -1.11597 | r_T_it × [inner stride <= 8 bytes] |
| `le_L128xN*const` | 8.8657 | le_L128xN × [无（常数项）] |
| `le_L128xN*eb1` | 1.71041 | le_L128xN × [dtype = int8] |
| `le_L128xN*eb2` | 1.92224 | le_L128xN × [dtype = float16] |
| `le_L128xN*eb4` | 2.16364 | le_L128xN × [dtype = float32] |
| `le_L128xN*eb8` | 3.0694 | le_L128xN × [dtype = int64] |
| `le_L128xN*W1` | 8.8657 | le_L128xN × [num_warps = 1] |
| `le_L128xN*Clt8` | 0.34722 | le_L128xN × [C < 8] |
| `le_L128xN*Clt16` | 0.732947 | le_L128xN × [C < 16] |
| `le_L128xN*Clt32` | -3.70946 | le_L128xN × [C < 32] |
| `le_L128xN*Clt64` | 4.71263 | le_L128xN × [C < 64] |
| `le_L128xN*nlt64` | 4.71263 | le_L128xN × [n < 64] |
| `le_L128xN*nge128` | -3.11356 | le_L128xN × [n >= 128] |
| `le_L128xN*nge256` | 0.911461 | le_L128xN × [n >= 256] |
| `le_L128xN*Lgt4` | -11.559 | le_L128xN × [L > 4] |
| `le_L128xN*Lgt16` | 20.3951 | le_L128xN × [L > 16] |
| `le_L128xN*Lgt64` | -3.51532 | le_L128xN × [L > 64] |
| `le_L128xN*mwlg2` | -7.01896 | le_L128xN × [mean_warp_lines > 2] |
| `le_L128xN*mwlg8` | -15.8774 | le_L128xN × [mean_warp_lines > 8] |
| `le_L128xN*smISB` | 1.04359 | le_L128xN × [inner stride <= 8 bytes] |

### 3. in-sample 拟合误差

- 点数：**291**
- 项数：**217**
- **MAPE：4.24%**

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
T = 56.4635
  + 21.9914 * line_elems_max*const
  + 10.926 * line_elems_max*eb1
  + 19.0689 * line_elems_max*eb2
  + 2.81591 * line_elems_max*eb4
  + -10.8195 * line_elems_max*eb8
  + 21.9914 * line_elems_max*W1
  + -0 * line_elems_max*W2
  + -15.0005 * line_elems_max*Clt8
  + 26.0404 * line_elems_max*Clt16
  + -1.97213 * line_elems_max*Clt32
  + 1.99723 * line_elems_max*Clt64
  + -12.9301 * line_elems_max*nlt64
  + 19.402 * line_elems_max*nge128
  + -74.3975 * line_elems_max*nge256
  + 18.2778 * line_elems_max*Rgt8
  + 187.614 * line_elems_max*Rgt16
  + 24.6935 * line_elems_max*Lgt4
  + 74.0474 * line_elems_max*Lgt16
  + 33.6231 * line_elems_max*Lgt64
  + -16.255 * line_elems_max*mwlg2
  + -12.5829 * line_elems_max*mwlg8
  + -6.8266 * line_elems_max*smISB
  + 0.22273 * hepl_n32*const
  + 0.127249 * hepl_n32*eb1
  + 0.048329 * hepl_n32*eb2
  + 0.072505 * hepl_n32*eb4
  + -0.025353 * hepl_n32*eb8
  + 0.22273 * hepl_n32*W1
  + 0.255246 * hepl_n32*Clt8
  + 0.00348 * hepl_n32*Clt16
  + -0.088337 * hepl_n32*Clt32
  + -0.050241 * hepl_n32*Clt64
  + 0.037078 * hepl_n32*nlt64
  + -0.071572 * hepl_n32*nge128
  + 0.066168 * hepl_n32*nge256
  + -0.157674 * hepl_n32*Rgt8
  + -0.047663 * hepl_n32*Rgt16
  + 0.152154 * hepl_n32*Lgt4
  + -0.009374 * hepl_n32*Lgt16
  + -0.18997 * hepl_n32*Lgt64
  + -0.124131 * hepl_n32*mwlg2
  + -0.154217 * hepl_n32*mwlg8
  + -0.290046 * hepl_n32*smISB
  + 1.32472 * bsN*const
  + -20.1505 * bsN*eb1
  + -2.72524 * bsN*eb2
  + 21.2814 * bsN*eb4
  + 2.91908 * bsN*eb8
  + 1.32472 * bsN*W1
  + -27.8221 * bsN*Clt8
  + 28.8396 * bsN*Clt16
  + 1.32472 * bsN*Clt32
  + 1.32472 * bsN*Clt64
  + -17.3425 * bsN*nlt64
  + 22.6484 * bsN*nge128
  + -27.2016 * bsN*nge256
  + 0.623102 * bsN*Rgt8
  + 27.6769 * bsN*Rgt16
  + -7.24023 * bsN*Lgt4
  + 3.8605 * bsN*Lgt16
  + 2.60924 * bsN*mwlg2
  + -4.22905 * bsN*mwlg8
  + 2.84383 * bsN*smISB
  + 0.045503 * bucket_pairs_32k16*const
  + 0.00778 * bucket_pairs_32k16*eb1
  + 0.009463 * bucket_pairs_32k16*eb2
  + 0.010433 * bucket_pairs_32k16*eb4
  + 0.017826 * bucket_pairs_32k16*eb8
  + 0.045503 * bucket_pairs_32k16*W1
  + -0.015607 * bucket_pairs_32k16*Clt8
  + -0.00346 * bucket_pairs_32k16*Clt16
  + -5.1e-05 * bucket_pairs_32k16*Clt32
  + 0.000554 * bucket_pairs_32k16*Clt64
  + 0.064736 * bucket_pairs_32k16*nlt64
  + -0.037477 * bucket_pairs_32k16*nge128
  + -0.010122 * bucket_pairs_32k16*nge256
  + -0.000707 * bucket_pairs_32k16*Rgt8
  + -0.001286 * bucket_pairs_32k16*Rgt16
  + -0.07833 * bucket_pairs_32k16*Lgt4
  + 0.013501 * bucket_pairs_32k16*Lgt16
  + -0.000586 * bucket_pairs_32k16*Lgt64
  + 0.01371 * bucket_pairs_32k16*mwlg2
  + 0.009957 * bucket_pairs_32k16*mwlg8
  + 0.02456 * bucket_pairs_32k16*smISB
  + -7.02945 * ep_cap*const
  + -4.73868 * ep_cap*eb1
  + -8.42042 * ep_cap*eb2
  + -0.679175 * ep_cap*eb4
  + 6.80883 * ep_cap*eb8
  + -7.02945 * ep_cap*W1
  + -52.9572 * ep_cap*Clt8
  + 64.3358 * ep_cap*Clt16
  + 12.6373 * ep_cap*Clt32
  + -3.99245 * ep_cap*Clt64
  + 2.37565 * ep_cap*nlt64
  + 4.476 * ep_cap*nge128
  + 34.7494 * ep_cap*nge256
  + -47.0601 * ep_cap*Rgt8
  + -24.5716 * ep_cap*Rgt16
  + -11.2278 * ep_cap*Lgt4
  + 71.594 * ep_cap*Lgt16
  + 33.6231 * ep_cap*Lgt64
  + -17.1609 * ep_cap*mwlg2
  + 10.6153 * ep_cap*mwlg8
  + 5.77952 * ep_cap*smISB
  + 17.2596 * r_C_rl*const
  + -5.20244 * r_C_rl*eb1
  + -3.11854 * r_C_rl*eb2
  + 8.78067 * r_C_rl*eb4
  + 16.7999 * r_C_rl*eb8
  + 17.2596 * r_C_rl*W1
  + 63.2396 * r_C_rl*Clt8
  + -93.8508 * r_C_rl*Clt16
  + -6.70388 * r_C_rl*Clt32
  + -2.73452 * r_C_rl*Clt64
  + 14.7356 * r_C_rl*nlt64
  + -22.9129 * r_C_rl*nge128
  + 65.3546 * r_C_rl*nge256
  + 82.3485 * r_C_rl*Rgt8
  + -57.0427 * r_C_rl*Rgt16
  + -20.8682 * r_C_rl*Lgt4
  + -162.676 * r_C_rl*Lgt16
  + 33.6231 * r_C_rl*Lgt64
  + 33.4081 * r_C_rl*mwlg2
  + -12.5829 * r_C_rl*mwlg8
  + 6.25031 * r_C_rl*smISB
  + 65.9586 * ov_cshortN_L*const
  + 26.4721 * ov_cshortN_L*eb1
  + 8.21297 * ov_cshortN_L*eb2
  + 15.916 * ov_cshortN_L*eb4
  + 15.3576 * ov_cshortN_L*eb8
  + 65.9586 * ov_cshortN_L*W1
  + -14.7173 * ov_cshortN_L*Clt8
  + -70.3128 * ov_cshortN_L*Clt16
  + 65.9586 * ov_cshortN_L*Clt32
  + 65.9586 * ov_cshortN_L*Clt64
  + -48.4002 * ov_cshortN_L*nlt64
  + 125.303 * ov_cshortN_L*nge128
  + 94.306 * ov_cshortN_L*nge256
  + -1.97896 * ov_cshortN_L*Rgt8
  + -65.4164 * ov_cshortN_L*Rgt16
  + -49.7615 * ov_cshortN_L*Lgt4
  + -91.0867 * ov_cshortN_L*Lgt16
  + -73.88 * ov_cshortN_L*mwlg2
  + 14.0552 * ov_cshortN_L*smISB
  + -36.8003 * cshortN*const
  + -4.31753 * cshortN*eb1
  + -3.08489 * cshortN*eb2
  + -31.5872 * cshortN*eb4
  + 2.18931 * cshortN*eb8
  + -36.8003 * cshortN*W1
  + 38.1223 * cshortN*Clt8
  + -23.0429 * cshortN*Clt16
  + -36.8003 * cshortN*Clt32
  + -36.8003 * cshortN*Clt64
  + 52.5438 * cshortN*nlt64
  + -131.809 * cshortN*nge128
  + -27.0896 * cshortN*nge256
  + 1.2369 * cshortN*Rgt8
  + 0.349678 * cshortN*Rgt16
  + 46.3737 * cshortN*Lgt4
  + 69.3654 * cshortN*Lgt16
  + 75.0373 * cshortN*mwlg2
  + 11.1416 * cshortN*mwlg8
  + -16.719 * cshortN*smISB
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 56.4635 | 固定开销（循环发射、启动） |
| `line_elems_max*const` | 21.9914 | 单 line 内最多元素数 × [无（常数项）] |
| `line_elems_max*eb1` | 10.926 | 单 line 内最多元素数 × [dtype = int8] |
| `line_elems_max*eb2` | 19.0689 | 单 line 内最多元素数 × [dtype = float16] |
| `line_elems_max*eb4` | 2.81591 | 单 line 内最多元素数 × [dtype = float32] |
| `line_elems_max*eb8` | -10.8195 | 单 line 内最多元素数 × [dtype = int64] |
| `line_elems_max*W1` | 21.9914 | 单 line 内最多元素数 × [num_warps = 1] |
| `line_elems_max*W2` | -0 | 单 line 内最多元素数 × [num_warps = 2] |
| `line_elems_max*Clt8` | -15.0005 | 单 line 内最多元素数 × [C < 8] |
| `line_elems_max*Clt16` | 26.0404 | 单 line 内最多元素数 × [C < 16] |
| `line_elems_max*Clt32` | -1.97213 | 单 line 内最多元素数 × [C < 32] |
| `line_elems_max*Clt64` | 1.99723 | 单 line 内最多元素数 × [C < 64] |
| `line_elems_max*nlt64` | -12.9301 | 单 line 内最多元素数 × [n < 64] |
| `line_elems_max*nge128` | 19.402 | 单 line 内最多元素数 × [n >= 128] |
| `line_elems_max*nge256` | -74.3975 | 单 line 内最多元素数 × [n >= 256] |
| `line_elems_max*Rgt8` | 18.2778 | 单 line 内最多元素数 × [R > 8] |
| `line_elems_max*Rgt16` | 187.614 | 单 line 内最多元素数 × [R > 16] |
| `line_elems_max*Lgt4` | 24.6935 | 单 line 内最多元素数 × [L > 4] |
| `line_elems_max*Lgt16` | 74.0474 | 单 line 内最多元素数 × [L > 16] |
| `line_elems_max*Lgt64` | 33.6231 | 单 line 内最多元素数 × [L > 64] |
| `line_elems_max*mwlg2` | -16.255 | 单 line 内最多元素数 × [mean_warp_lines > 2] |
| `line_elems_max*mwlg8` | -12.5829 | 单 line 内最多元素数 × [mean_warp_lines > 8] |
| `line_elems_max*smISB` | -6.8266 | 单 line 内最多元素数 × [inner stride <= 8 bytes] |
| `hepl_n32*const` | 0.22273 | n · min(128/ISB, 32) × [无（常数项）] |
| `hepl_n32*eb1` | 0.127249 | n · min(128/ISB, 32) × [dtype = int8] |
| `hepl_n32*eb2` | 0.048329 | n · min(128/ISB, 32) × [dtype = float16] |
| `hepl_n32*eb4` | 0.072505 | n · min(128/ISB, 32) × [dtype = float32] |
| `hepl_n32*eb8` | -0.025353 | n · min(128/ISB, 32) × [dtype = int64] |
| `hepl_n32*W1` | 0.22273 | n · min(128/ISB, 32) × [num_warps = 1] |
| `hepl_n32*Clt8` | 0.255246 | n · min(128/ISB, 32) × [C < 8] |
| `hepl_n32*Clt16` | 0.00348 | n · min(128/ISB, 32) × [C < 16] |
| `hepl_n32*Clt32` | -0.088337 | n · min(128/ISB, 32) × [C < 32] |
| `hepl_n32*Clt64` | -0.050241 | n · min(128/ISB, 32) × [C < 64] |
| `hepl_n32*nlt64` | 0.037078 | n · min(128/ISB, 32) × [n < 64] |
| `hepl_n32*nge128` | -0.071572 | n · min(128/ISB, 32) × [n >= 128] |
| `hepl_n32*nge256` | 0.066168 | n · min(128/ISB, 32) × [n >= 256] |
| `hepl_n32*Rgt8` | -0.157674 | n · min(128/ISB, 32) × [R > 8] |
| `hepl_n32*Rgt16` | -0.047663 | n · min(128/ISB, 32) × [R > 16] |
| `hepl_n32*Lgt4` | 0.152154 | n · min(128/ISB, 32) × [L > 4] |
| `hepl_n32*Lgt16` | -0.009374 | n · min(128/ISB, 32) × [L > 16] |
| `hepl_n32*Lgt64` | -0.18997 | n · min(128/ISB, 32) × [L > 64] |
| `hepl_n32*mwlg2` | -0.124131 | n · min(128/ISB, 32) × [mean_warp_lines > 2] |
| `hepl_n32*mwlg8` | -0.154217 | n · min(128/ISB, 32) × [mean_warp_lines > 8] |
| `hepl_n32*smISB` | -0.290046 | n · min(128/ISB, 32) × [inner stride <= 8 bytes] |
| `bsN*const` | 1.32472 | [C·eb<32] · n × [无（常数项）] |
| `bsN*eb1` | -20.1505 | [C·eb<32] · n × [dtype = int8] |
| `bsN*eb2` | -2.72524 | [C·eb<32] · n × [dtype = float16] |
| `bsN*eb4` | 21.2814 | [C·eb<32] · n × [dtype = float32] |
| `bsN*eb8` | 2.91908 | [C·eb<32] · n × [dtype = int64] |
| `bsN*W1` | 1.32472 | [C·eb<32] · n × [num_warps = 1] |
| `bsN*Clt8` | -27.8221 | [C·eb<32] · n × [C < 8] |
| `bsN*Clt16` | 28.8396 | [C·eb<32] · n × [C < 16] |
| `bsN*Clt32` | 1.32472 | [C·eb<32] · n × [C < 32] |
| `bsN*Clt64` | 1.32472 | [C·eb<32] · n × [C < 64] |
| `bsN*nlt64` | -17.3425 | [C·eb<32] · n × [n < 64] |
| `bsN*nge128` | 22.6484 | [C·eb<32] · n × [n >= 128] |
| `bsN*nge256` | -27.2016 | [C·eb<32] · n × [n >= 256] |
| `bsN*Rgt8` | 0.623102 | [C·eb<32] · n × [R > 8] |
| `bsN*Rgt16` | 27.6769 | [C·eb<32] · n × [R > 16] |
| `bsN*Lgt4` | -7.24023 | [C·eb<32] · n × [L > 4] |
| `bsN*Lgt16` | 3.8605 | [C·eb<32] · n × [L > 16] |
| `bsN*mwlg2` | 2.60924 | [C·eb<32] · n × [mean_warp_lines > 2] |
| `bsN*mwlg8` | -4.22905 | [C·eb<32] · n × [mean_warp_lines > 8] |
| `bsN*smISB` | 2.84383 | [C·eb<32] · n × [inner stride <= 8 bytes] |
| `bucket_pairs_32k16*const` | 0.045503 | line 桶 (32768..16384) 对数 × [无（常数项）] |
| `bucket_pairs_32k16*eb1` | 0.00778 | line 桶 (32768..16384) 对数 × [dtype = int8] |
| `bucket_pairs_32k16*eb2` | 0.009463 | line 桶 (32768..16384) 对数 × [dtype = float16] |
| `bucket_pairs_32k16*eb4` | 0.010433 | line 桶 (32768..16384) 对数 × [dtype = float32] |
| `bucket_pairs_32k16*eb8` | 0.017826 | line 桶 (32768..16384) 对数 × [dtype = int64] |
| `bucket_pairs_32k16*W1` | 0.045503 | line 桶 (32768..16384) 对数 × [num_warps = 1] |
| `bucket_pairs_32k16*Clt8` | -0.015607 | line 桶 (32768..16384) 对数 × [C < 8] |
| `bucket_pairs_32k16*Clt16` | -0.00346 | line 桶 (32768..16384) 对数 × [C < 16] |
| `bucket_pairs_32k16*Clt32` | -5.1e-05 | line 桶 (32768..16384) 对数 × [C < 32] |
| `bucket_pairs_32k16*Clt64` | 0.000554 | line 桶 (32768..16384) 对数 × [C < 64] |
| `bucket_pairs_32k16*nlt64` | 0.064736 | line 桶 (32768..16384) 对数 × [n < 64] |
| `bucket_pairs_32k16*nge128` | -0.037477 | line 桶 (32768..16384) 对数 × [n >= 128] |
| `bucket_pairs_32k16*nge256` | -0.010122 | line 桶 (32768..16384) 对数 × [n >= 256] |
| `bucket_pairs_32k16*Rgt8` | -0.000707 | line 桶 (32768..16384) 对数 × [R > 8] |
| `bucket_pairs_32k16*Rgt16` | -0.001286 | line 桶 (32768..16384) 对数 × [R > 16] |
| `bucket_pairs_32k16*Lgt4` | -0.07833 | line 桶 (32768..16384) 对数 × [L > 4] |
| `bucket_pairs_32k16*Lgt16` | 0.013501 | line 桶 (32768..16384) 对数 × [L > 16] |
| `bucket_pairs_32k16*Lgt64` | -0.000586 | line 桶 (32768..16384) 对数 × [L > 64] |
| `bucket_pairs_32k16*mwlg2` | 0.01371 | line 桶 (32768..16384) 对数 × [mean_warp_lines > 2] |
| `bucket_pairs_32k16*mwlg8` | 0.009957 | line 桶 (32768..16384) 对数 × [mean_warp_lines > 8] |
| `bucket_pairs_32k16*smISB` | 0.02456 | line 桶 (32768..16384) 对数 × [inner stride <= 8 bytes] |
| `ep_cap*const` | -7.02945 | epl 上限 × [无（常数项）] |
| `ep_cap*eb1` | -4.73868 | epl 上限 × [dtype = int8] |
| `ep_cap*eb2` | -8.42042 | epl 上限 × [dtype = float16] |
| `ep_cap*eb4` | -0.679175 | epl 上限 × [dtype = float32] |
| `ep_cap*eb8` | 6.80883 | epl 上限 × [dtype = int64] |
| `ep_cap*W1` | -7.02945 | epl 上限 × [num_warps = 1] |
| `ep_cap*Clt8` | -52.9572 | epl 上限 × [C < 8] |
| `ep_cap*Clt16` | 64.3358 | epl 上限 × [C < 16] |
| `ep_cap*Clt32` | 12.6373 | epl 上限 × [C < 32] |
| `ep_cap*Clt64` | -3.99245 | epl 上限 × [C < 64] |
| `ep_cap*nlt64` | 2.37565 | epl 上限 × [n < 64] |
| `ep_cap*nge128` | 4.476 | epl 上限 × [n >= 128] |
| `ep_cap*nge256` | 34.7494 | epl 上限 × [n >= 256] |
| `ep_cap*Rgt8` | -47.0601 | epl 上限 × [R > 8] |
| `ep_cap*Rgt16` | -24.5716 | epl 上限 × [R > 16] |
| `ep_cap*Lgt4` | -11.2278 | epl 上限 × [L > 4] |
| `ep_cap*Lgt16` | 71.594 | epl 上限 × [L > 16] |
| `ep_cap*Lgt64` | 33.6231 | epl 上限 × [L > 64] |
| `ep_cap*mwlg2` | -17.1609 | epl 上限 × [mean_warp_lines > 2] |
| `ep_cap*mwlg8` | 10.6153 | epl 上限 × [mean_warp_lines > 8] |
| `ep_cap*smISB` | 5.77952 | epl 上限 × [inner stride <= 8 bytes] |
| `r_C_rl*const` | 17.2596 | r_C_rl × [无（常数项）] |
| `r_C_rl*eb1` | -5.20244 | r_C_rl × [dtype = int8] |
| `r_C_rl*eb2` | -3.11854 | r_C_rl × [dtype = float16] |
| `r_C_rl*eb4` | 8.78067 | r_C_rl × [dtype = float32] |
| `r_C_rl*eb8` | 16.7999 | r_C_rl × [dtype = int64] |
| `r_C_rl*W1` | 17.2596 | r_C_rl × [num_warps = 1] |
| `r_C_rl*Clt8` | 63.2396 | r_C_rl × [C < 8] |
| `r_C_rl*Clt16` | -93.8508 | r_C_rl × [C < 16] |
| `r_C_rl*Clt32` | -6.70388 | r_C_rl × [C < 32] |
| `r_C_rl*Clt64` | -2.73452 | r_C_rl × [C < 64] |
| `r_C_rl*nlt64` | 14.7356 | r_C_rl × [n < 64] |
| `r_C_rl*nge128` | -22.9129 | r_C_rl × [n >= 128] |
| `r_C_rl*nge256` | 65.3546 | r_C_rl × [n >= 256] |
| `r_C_rl*Rgt8` | 82.3485 | r_C_rl × [R > 8] |
| `r_C_rl*Rgt16` | -57.0427 | r_C_rl × [R > 16] |
| `r_C_rl*Lgt4` | -20.8682 | r_C_rl × [L > 4] |
| `r_C_rl*Lgt16` | -162.676 | r_C_rl × [L > 16] |
| `r_C_rl*Lgt64` | 33.6231 | r_C_rl × [L > 64] |
| `r_C_rl*mwlg2` | 33.4081 | r_C_rl × [mean_warp_lines > 2] |
| `r_C_rl*mwlg8` | -12.5829 | r_C_rl × [mean_warp_lines > 8] |
| `r_C_rl*smISB` | 6.25031 | r_C_rl × [inner stride <= 8 bytes] |
| `ov_cshortN_L*const` | 65.9586 | max(0, cshort · n / 32 − 触及的 128B line 数) × [无（常数项）] |
| `ov_cshortN_L*eb1` | 26.4721 | max(0, cshort · n / 32 − 触及的 128B line 数) × [dtype = int8] |
| `ov_cshortN_L*eb2` | 8.21297 | max(0, cshort · n / 32 − 触及的 128B line 数) × [dtype = float16] |
| `ov_cshortN_L*eb4` | 15.916 | max(0, cshort · n / 32 − 触及的 128B line 数) × [dtype = float32] |
| `ov_cshortN_L*eb8` | 15.3576 | max(0, cshort · n / 32 − 触及的 128B line 数) × [dtype = int64] |
| `ov_cshortN_L*W1` | 65.9586 | max(0, cshort · n / 32 − 触及的 128B line 数) × [num_warps = 1] |
| `ov_cshortN_L*Clt8` | -14.7173 | max(0, cshort · n / 32 − 触及的 128B line 数) × [C < 8] |
| `ov_cshortN_L*Clt16` | -70.3128 | max(0, cshort · n / 32 − 触及的 128B line 数) × [C < 16] |
| `ov_cshortN_L*Clt32` | 65.9586 | max(0, cshort · n / 32 − 触及的 128B line 数) × [C < 32] |
| `ov_cshortN_L*Clt64` | 65.9586 | max(0, cshort · n / 32 − 触及的 128B line 数) × [C < 64] |
| `ov_cshortN_L*nlt64` | -48.4002 | max(0, cshort · n / 32 − 触及的 128B line 数) × [n < 64] |
| `ov_cshortN_L*nge128` | 125.303 | max(0, cshort · n / 32 − 触及的 128B line 数) × [n >= 128] |
| `ov_cshortN_L*nge256` | 94.306 | max(0, cshort · n / 32 − 触及的 128B line 数) × [n >= 256] |
| `ov_cshortN_L*Rgt8` | -1.97896 | max(0, cshort · n / 32 − 触及的 128B line 数) × [R > 8] |
| `ov_cshortN_L*Rgt16` | -65.4164 | max(0, cshort · n / 32 − 触及的 128B line 数) × [R > 16] |
| `ov_cshortN_L*Lgt4` | -49.7615 | max(0, cshort · n / 32 − 触及的 128B line 数) × [L > 4] |
| `ov_cshortN_L*Lgt16` | -91.0867 | max(0, cshort · n / 32 − 触及的 128B line 数) × [L > 16] |
| `ov_cshortN_L*mwlg2` | -73.88 | max(0, cshort · n / 32 − 触及的 128B line 数) × [mean_warp_lines > 2] |
| `ov_cshortN_L*smISB` | 14.0552 | max(0, cshort · n / 32 − 触及的 128B line 数) × [inner stride <= 8 bytes] |
| `cshortN*const` | -36.8003 | cshort · n / 32 × [无（常数项）] |
| `cshortN*eb1` | -4.31753 | cshort · n / 32 × [dtype = int8] |
| `cshortN*eb2` | -3.08489 | cshort · n / 32 × [dtype = float16] |
| `cshortN*eb4` | -31.5872 | cshort · n / 32 × [dtype = float32] |
| `cshortN*eb8` | 2.18931 | cshort · n / 32 × [dtype = int64] |
| `cshortN*W1` | -36.8003 | cshort · n / 32 × [num_warps = 1] |
| `cshortN*Clt8` | 38.1223 | cshort · n / 32 × [C < 8] |
| `cshortN*Clt16` | -23.0429 | cshort · n / 32 × [C < 16] |
| `cshortN*Clt32` | -36.8003 | cshort · n / 32 × [C < 32] |
| `cshortN*Clt64` | -36.8003 | cshort · n / 32 × [C < 64] |
| `cshortN*nlt64` | 52.5438 | cshort · n / 32 × [n < 64] |
| `cshortN*nge128` | -131.809 | cshort · n / 32 × [n >= 128] |
| `cshortN*nge256` | -27.0896 | cshort · n / 32 × [n >= 256] |
| `cshortN*Rgt8` | 1.2369 | cshort · n / 32 × [R > 8] |
| `cshortN*Rgt16` | 0.349678 | cshort · n / 32 × [R > 16] |
| `cshortN*Lgt4` | 46.3737 | cshort · n / 32 × [L > 4] |
| `cshortN*Lgt16` | 69.3654 | cshort · n / 32 × [L > 16] |
| `cshortN*mwlg2` | 75.0373 | cshort · n / 32 × [mean_warp_lines > 2] |
| `cshortN*mwlg8` | 11.1416 | cshort · n / 32 × [mean_warp_lines > 8] |
| `cshortN*smISB` | -16.719 | cshort · n / 32 × [inner stride <= 8 bytes] |

### 3. in-sample 拟合误差

- 点数：**943**
- 项数：**217**
- **MAPE：5.45%**

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
T = 156.304
  + -7.93112 * minL64*const
  + -0.294187 * minL64*eb1
  + 2.164 * minL64*eb2
  + 0.069011 * minL64*eb4
  + -9.86994 * minL64*eb8
  + -0.853886 * minL64*W1
  + -2.16227 * minL64*W2
  + -0.682818 * minL64*W4
  + -1.22568 * minL64*W8
  + -0.949535 * minL64*W16
  + -0.660503 * minL64*W32
  + -1.39642 * minL64*W64
  + -0.783927 * minL64*Clt8
  + 0.065442 * minL64*Clt16
  + -0.051229 * minL64*Clt32
  + -0.257202 * minL64*Clt64
  + -0.257202 * minL64*nlt64
  + -6.7703 * minL64*nge128
  + 38.9775 * minL64*nge256
  + -0 * minL64*Rgt16
  + -10.6143 * minL64*Lgt4
  + -8.85689 * minL64*Lgt16
  + -13.4386 * minL64*Lgt64
  + -7.36019 * minL64*mwlg2
  + 29.0218 * minL64*mwlg8
  + 0.510124 * minL64*smISB
  + 2.74958 * txBytes*const
  + 0.374693 * txBytes*eb1
  + 0.783114 * txBytes*eb2
  + 0.805787 * txBytes*eb4
  + 0.785984 * txBytes*eb8
  + 3.55312 * txBytes*W1
  + -0.934377 * txBytes*W2
  + -0.713427 * txBytes*W4
  + -0.777815 * txBytes*W8
  + -0.319449 * txBytes*W16
  + 0.20663 * txBytes*W32
  + 1.7349 * txBytes*W64
  + 0.094667 * txBytes*Clt8
  + -0.302407 * txBytes*Clt16
  + 0.023176 * txBytes*Clt32
  + 0.162152 * txBytes*Clt64
  + 0.162152 * txBytes*nlt64
  + 0.008315 * txBytes*nge128
  + -0.022939 * txBytes*nge256
  + 0 * txBytes*Rgt8
  + -0 * txBytes*Rgt16
  + -3.14302 * txBytes*Lgt4
  + 0.008382 * txBytes*Lgt16
  + 0.114096 * txBytes*Lgt64
  + 0.635167 * txBytes*mwlg2
  + -0.336527 * txBytes*mwlg8
  + 1.40634 * txBytes*smISB
  + -18.4468 * r_R_mwl*const
  + -3.75132 * r_R_mwl*eb1
  + -3.92838 * r_R_mwl*eb2
  + -6.91582 * r_R_mwl*eb4
  + -3.85123 * r_R_mwl*eb8
  + 0.960558 * r_R_mwl*W1
  + 0.045129 * r_R_mwl*W2
  + -0.478646 * r_R_mwl*W4
  + -2.57165 * r_R_mwl*W8
  + 3.75144 * r_R_mwl*W16
  + -5.48914 * r_R_mwl*W32
  + -14.6644 * r_R_mwl*W64
  + 4.73796 * r_R_mwl*Clt8
  + 1.90912 * r_R_mwl*Clt16
  + -0.055708 * r_R_mwl*Clt32
  + -1.15878 * r_R_mwl*Clt64
  + -1.15878 * r_R_mwl*nlt64
  + -20.4285 * r_R_mwl*nge128
  + 8.14309 * r_R_mwl*nge256
  + 18.3851 * r_R_mwl*Lgt4
  + 162.678 * r_R_mwl*Lgt16
  + 10146.3 * r_R_mwl*Lgt64
  + 0.007752 * r_R_mwl*mwlg2
  + 71.4829 * r_R_mwl*mwlg8
  + 0.742157 * r_R_mwl*smISB
  + 4.47992 * minL32*const
  + 5.4193 * minL32*eb1
  + -0.036792 * minL32*eb2
  + 0.64866 * minL32*eb4
  + -1.55124 * minL32*eb8
  + 1.99902 * minL32*W1
  + 2.09331 * minL32*W2
  + 2.28681 * minL32*W4
  + 1.41785 * minL32*W8
  + 0.421574 * minL32*W16
  + -0.78395 * minL32*W32
  + -2.9547 * minL32*W64
  + -0.783927 * minL32*Clt8
  + 0.065442 * minL32*Clt16
  + -0.051229 * minL32*Clt32
  + -0.257202 * minL32*Clt64
  + -0.257202 * minL32*nlt64
  + 1.88701 * minL32*nge128
  + -25.2646 * minL32*nge256
  + 1.79676 * minL32*Lgt4
  + 3.55415 * minL32*Lgt16
  + -6.7193 * minL32*Lgt64
  + 5.05085 * minL32*mwlg2
  + -21.3901 * minL32*mwlg8
  + 0.510124 * minL32*smISB
  + -0.787305 * itersL*const
  + -0.000591 * itersL*eb1
  + -0.034871 * itersL*eb2
  + -0.490119 * itersL*eb4
  + -0.261724 * itersL*eb8
  + 0.766047 * itersL*W1
  + 0.275565 * itersL*W2
  + 0.070308 * itersL*W4
  + 1.41785 * itersL*W8
  + 0.421574 * itersL*W16
  + -0.78395 * itersL*W32
  + -2.9547 * itersL*W64
  + -0.783927 * itersL*Clt8
  + 0.065442 * itersL*Clt16
  + -0.051229 * itersL*Clt32
  + -0.257202 * itersL*Clt64
  + -0.257202 * itersL*nlt64
  + 0.15894 * itersL*nge128
  + 0.88877 * itersL*nge256
  + -0.269118 * itersL*Lgt4
  + -0.148613 * itersL*Lgt16
  + -0.760048 * itersL*Lgt64
  + 0.244412 * itersL*mwlg2
  + 0.242245 * itersL*mwlg8
  + 0.049335 * itersL*smISB
  + -1.04554 * LmEb*const
  + 0.106898 * LmEb*eb1
  + -0.310999 * LmEb*eb2
  + -0.356281 * LmEb*eb4
  + -0.485152 * LmEb*eb8
  + -3.87102 * LmEb*W1
  + 0.512456 * LmEb*W2
  + 0.389566 * LmEb*W4
  + 0.410339 * LmEb*W8
  + 0.383368 * LmEb*W16
  + 0.424119 * LmEb*W32
  + 0.705639 * LmEb*W64
  + 0.094667 * LmEb*Clt8
  + -0.302407 * LmEb*Clt16
  + 0.023176 * LmEb*Clt32
  + 0.162153 * LmEb*Clt64
  + 0.162153 * LmEb*nlt64
  + -0.216009 * LmEb*nge128
  + -0.082779 * LmEb*nge256
  + 3.25765 * LmEb*Lgt4
  + 0.116336 * LmEb*Lgt16
  + -1.2507 * LmEb*Lgt64
  + -0.7528 * LmEb*mwlg2
  + 0.18338 * LmEb*mwlg8
  + -1.00081 * LmEb*smISB
  + 3.684 * minL128*const
  + -2.19927 * minL128*eb1
  + -0.729713 * minL128*eb2
  + -0.420008 * minL128*eb4
  + 7.03298 * minL128*eb8
  + 0.412275 * minL128*W1
  + -0.06744 * minL128*W2
  + -0.865659 * minL128*W4
  + -1.23287 * minL128*W8
  + -0.036563 * minL128*W16
  + 1.08923 * minL128*W32
  + 4.38502 * minL128*W64
  + -0.783927 * minL128*Clt8
  + 0.065442 * minL128*Clt16
  + -0.051229 * minL128*Clt32
  + -0.257202 * minL128*Clt64
  + -0.257202 * minL128*nlt64
  + 4.84481 * minL128*nge128
  + -12.0322 * minL128*nge256
  + 1.00083 * minL128*Lgt4
  + 2.75823 * minL128*Lgt16
  + -1.82348 * minL128*Lgt64
  + 4.25493 * minL128*mwlg2
  + -5.89504 * minL128*mwlg8
  + 0.510124 * minL128*smISB
  + -0.508236 * mx_iters_mwl*const
  + 2.33121 * mx_iters_mwl*eb1
  + 2.15387 * mx_iters_mwl*eb2
  + -1.69176 * mx_iters_mwl*eb4
  + -3.30156 * mx_iters_mwl*eb8
  + -5.91919 * mx_iters_mwl*W1
  + -3.40012 * mx_iters_mwl*W2
  + -4.38544 * mx_iters_mwl*W4
  + -3.17787 * mx_iters_mwl*W8
  + -2.03272 * mx_iters_mwl*W16
  + 3.39501 * mx_iters_mwl*W32
  + 15.0121 * mx_iters_mwl*W64
  + -0.783927 * mx_iters_mwl*Clt8
  + 0.065442 * mx_iters_mwl*Clt16
  + -0.051229 * mx_iters_mwl*Clt32
  + -0.257202 * mx_iters_mwl*Clt64
  + -0.257202 * mx_iters_mwl*nlt64
  + 8.66237 * mx_iters_mwl*nge128
  + -6.95594 * mx_iters_mwl*nge256
  + 6.67817 * mx_iters_mwl*Lgt4
  + 1.6671 * mx_iters_mwl*Lgt16
  + 10.8838 * mx_iters_mwl*Lgt64
  + 2.70636 * mx_iters_mwl*mwlg2
  + -2.89895 * mx_iters_mwl*mwlg8
  + -2.13355 * mx_iters_mwl*smISB
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 156.304 | 固定开销（循环发射、启动） |
| `minL64*const` | -7.93112 | min(L,64) × [无（常数项）] |
| `minL64*eb1` | -0.294187 | min(L,64) × [dtype = int8] |
| `minL64*eb2` | 2.164 | min(L,64) × [dtype = float16] |
| `minL64*eb4` | 0.069011 | min(L,64) × [dtype = float32] |
| `minL64*eb8` | -9.86994 | min(L,64) × [dtype = int64] |
| `minL64*W1` | -0.853886 | min(L,64) × [num_warps = 1] |
| `minL64*W2` | -2.16227 | min(L,64) × [num_warps = 2] |
| `minL64*W4` | -0.682818 | min(L,64) × [num_warps = 4] |
| `minL64*W8` | -1.22568 | min(L,64) × [num_warps = 8] |
| `minL64*W16` | -0.949535 | min(L,64) × [num_warps = 16] |
| `minL64*W32` | -0.660503 | min(L,64) × [num_warps = 32] |
| `minL64*W64` | -1.39642 | min(L,64) × [num_warps = 64] |
| `minL64*Clt8` | -0.783927 | min(L,64) × [C < 8] |
| `minL64*Clt16` | 0.065442 | min(L,64) × [C < 16] |
| `minL64*Clt32` | -0.051229 | min(L,64) × [C < 32] |
| `minL64*Clt64` | -0.257202 | min(L,64) × [C < 64] |
| `minL64*nlt64` | -0.257202 | min(L,64) × [n < 64] |
| `minL64*nge128` | -6.7703 | min(L,64) × [n >= 128] |
| `minL64*nge256` | 38.9775 | min(L,64) × [n >= 256] |
| `minL64*Rgt16` | -0 | min(L,64) × [R > 16] |
| `minL64*Lgt4` | -10.6143 | min(L,64) × [L > 4] |
| `minL64*Lgt16` | -8.85689 | min(L,64) × [L > 16] |
| `minL64*Lgt64` | -13.4386 | min(L,64) × [L > 64] |
| `minL64*mwlg2` | -7.36019 | min(L,64) × [mean_warp_lines > 2] |
| `minL64*mwlg8` | 29.0218 | min(L,64) × [mean_warp_lines > 8] |
| `minL64*smISB` | 0.510124 | min(L,64) × [inner stride <= 8 bytes] |
| `txBytes*const` | 2.74958 | iters · mwl · eb × [无（常数项）] |
| `txBytes*eb1` | 0.374693 | iters · mwl · eb × [dtype = int8] |
| `txBytes*eb2` | 0.783114 | iters · mwl · eb × [dtype = float16] |
| `txBytes*eb4` | 0.805787 | iters · mwl · eb × [dtype = float32] |
| `txBytes*eb8` | 0.785984 | iters · mwl · eb × [dtype = int64] |
| `txBytes*W1` | 3.55312 | iters · mwl · eb × [num_warps = 1] |
| `txBytes*W2` | -0.934377 | iters · mwl · eb × [num_warps = 2] |
| `txBytes*W4` | -0.713427 | iters · mwl · eb × [num_warps = 4] |
| `txBytes*W8` | -0.777815 | iters · mwl · eb × [num_warps = 8] |
| `txBytes*W16` | -0.319449 | iters · mwl · eb × [num_warps = 16] |
| `txBytes*W32` | 0.20663 | iters · mwl · eb × [num_warps = 32] |
| `txBytes*W64` | 1.7349 | iters · mwl · eb × [num_warps = 64] |
| `txBytes*Clt8` | 0.094667 | iters · mwl · eb × [C < 8] |
| `txBytes*Clt16` | -0.302407 | iters · mwl · eb × [C < 16] |
| `txBytes*Clt32` | 0.023176 | iters · mwl · eb × [C < 32] |
| `txBytes*Clt64` | 0.162152 | iters · mwl · eb × [C < 64] |
| `txBytes*nlt64` | 0.162152 | iters · mwl · eb × [n < 64] |
| `txBytes*nge128` | 0.008315 | iters · mwl · eb × [n >= 128] |
| `txBytes*nge256` | -0.022939 | iters · mwl · eb × [n >= 256] |
| `txBytes*Rgt8` | 0 | iters · mwl · eb × [R > 8] |
| `txBytes*Rgt16` | -0 | iters · mwl · eb × [R > 16] |
| `txBytes*Lgt4` | -3.14302 | iters · mwl · eb × [L > 4] |
| `txBytes*Lgt16` | 0.008382 | iters · mwl · eb × [L > 16] |
| `txBytes*Lgt64` | 0.114096 | iters · mwl · eb × [L > 64] |
| `txBytes*mwlg2` | 0.635167 | iters · mwl · eb × [mean_warp_lines > 2] |
| `txBytes*mwlg8` | -0.336527 | iters · mwl · eb × [mean_warp_lines > 8] |
| `txBytes*smISB` | 1.40634 | iters · mwl · eb × [inner stride <= 8 bytes] |
| `r_R_mwl*const` | -18.4468 | r_R_mwl × [无（常数项）] |
| `r_R_mwl*eb1` | -3.75132 | r_R_mwl × [dtype = int8] |
| `r_R_mwl*eb2` | -3.92838 | r_R_mwl × [dtype = float16] |
| `r_R_mwl*eb4` | -6.91582 | r_R_mwl × [dtype = float32] |
| `r_R_mwl*eb8` | -3.85123 | r_R_mwl × [dtype = int64] |
| `r_R_mwl*W1` | 0.960558 | r_R_mwl × [num_warps = 1] |
| `r_R_mwl*W2` | 0.045129 | r_R_mwl × [num_warps = 2] |
| `r_R_mwl*W4` | -0.478646 | r_R_mwl × [num_warps = 4] |
| `r_R_mwl*W8` | -2.57165 | r_R_mwl × [num_warps = 8] |
| `r_R_mwl*W16` | 3.75144 | r_R_mwl × [num_warps = 16] |
| `r_R_mwl*W32` | -5.48914 | r_R_mwl × [num_warps = 32] |
| `r_R_mwl*W64` | -14.6644 | r_R_mwl × [num_warps = 64] |
| `r_R_mwl*Clt8` | 4.73796 | r_R_mwl × [C < 8] |
| `r_R_mwl*Clt16` | 1.90912 | r_R_mwl × [C < 16] |
| `r_R_mwl*Clt32` | -0.055708 | r_R_mwl × [C < 32] |
| `r_R_mwl*Clt64` | -1.15878 | r_R_mwl × [C < 64] |
| `r_R_mwl*nlt64` | -1.15878 | r_R_mwl × [n < 64] |
| `r_R_mwl*nge128` | -20.4285 | r_R_mwl × [n >= 128] |
| `r_R_mwl*nge256` | 8.14309 | r_R_mwl × [n >= 256] |
| `r_R_mwl*Lgt4` | 18.3851 | r_R_mwl × [L > 4] |
| `r_R_mwl*Lgt16` | 162.678 | r_R_mwl × [L > 16] |
| `r_R_mwl*Lgt64` | 10146.3 | r_R_mwl × [L > 64] |
| `r_R_mwl*mwlg2` | 0.007752 | r_R_mwl × [mean_warp_lines > 2] |
| `r_R_mwl*mwlg8` | 71.4829 | r_R_mwl × [mean_warp_lines > 8] |
| `r_R_mwl*smISB` | 0.742157 | r_R_mwl × [inner stride <= 8 bytes] |
| `minL32*const` | 4.47992 | min(L,32) × [无（常数项）] |
| `minL32*eb1` | 5.4193 | min(L,32) × [dtype = int8] |
| `minL32*eb2` | -0.036792 | min(L,32) × [dtype = float16] |
| `minL32*eb4` | 0.64866 | min(L,32) × [dtype = float32] |
| `minL32*eb8` | -1.55124 | min(L,32) × [dtype = int64] |
| `minL32*W1` | 1.99902 | min(L,32) × [num_warps = 1] |
| `minL32*W2` | 2.09331 | min(L,32) × [num_warps = 2] |
| `minL32*W4` | 2.28681 | min(L,32) × [num_warps = 4] |
| `minL32*W8` | 1.41785 | min(L,32) × [num_warps = 8] |
| `minL32*W16` | 0.421574 | min(L,32) × [num_warps = 16] |
| `minL32*W32` | -0.78395 | min(L,32) × [num_warps = 32] |
| `minL32*W64` | -2.9547 | min(L,32) × [num_warps = 64] |
| `minL32*Clt8` | -0.783927 | min(L,32) × [C < 8] |
| `minL32*Clt16` | 0.065442 | min(L,32) × [C < 16] |
| `minL32*Clt32` | -0.051229 | min(L,32) × [C < 32] |
| `minL32*Clt64` | -0.257202 | min(L,32) × [C < 64] |
| `minL32*nlt64` | -0.257202 | min(L,32) × [n < 64] |
| `minL32*nge128` | 1.88701 | min(L,32) × [n >= 128] |
| `minL32*nge256` | -25.2646 | min(L,32) × [n >= 256] |
| `minL32*Lgt4` | 1.79676 | min(L,32) × [L > 4] |
| `minL32*Lgt16` | 3.55415 | min(L,32) × [L > 16] |
| `minL32*Lgt64` | -6.7193 | min(L,32) × [L > 64] |
| `minL32*mwlg2` | 5.05085 | min(L,32) × [mean_warp_lines > 2] |
| `minL32*mwlg8` | -21.3901 | min(L,32) × [mean_warp_lines > 8] |
| `minL32*smISB` | 0.510124 | min(L,32) × [inner stride <= 8 bytes] |
| `itersL*const` | -0.787305 | iters · L × [无（常数项）] |
| `itersL*eb1` | -0.000591 | iters · L × [dtype = int8] |
| `itersL*eb2` | -0.034871 | iters · L × [dtype = float16] |
| `itersL*eb4` | -0.490119 | iters · L × [dtype = float32] |
| `itersL*eb8` | -0.261724 | iters · L × [dtype = int64] |
| `itersL*W1` | 0.766047 | iters · L × [num_warps = 1] |
| `itersL*W2` | 0.275565 | iters · L × [num_warps = 2] |
| `itersL*W4` | 0.070308 | iters · L × [num_warps = 4] |
| `itersL*W8` | 1.41785 | iters · L × [num_warps = 8] |
| `itersL*W16` | 0.421574 | iters · L × [num_warps = 16] |
| `itersL*W32` | -0.78395 | iters · L × [num_warps = 32] |
| `itersL*W64` | -2.9547 | iters · L × [num_warps = 64] |
| `itersL*Clt8` | -0.783927 | iters · L × [C < 8] |
| `itersL*Clt16` | 0.065442 | iters · L × [C < 16] |
| `itersL*Clt32` | -0.051229 | iters · L × [C < 32] |
| `itersL*Clt64` | -0.257202 | iters · L × [C < 64] |
| `itersL*nlt64` | -0.257202 | iters · L × [n < 64] |
| `itersL*nge128` | 0.15894 | iters · L × [n >= 128] |
| `itersL*nge256` | 0.88877 | iters · L × [n >= 256] |
| `itersL*Lgt4` | -0.269118 | iters · L × [L > 4] |
| `itersL*Lgt16` | -0.148613 | iters · L × [L > 16] |
| `itersL*Lgt64` | -0.760048 | iters · L × [L > 64] |
| `itersL*mwlg2` | 0.244412 | iters · L × [mean_warp_lines > 2] |
| `itersL*mwlg8` | 0.242245 | iters · L × [mean_warp_lines > 8] |
| `itersL*smISB` | 0.049335 | iters · L × [inner stride <= 8 bytes] |
| `LmEb*const` | -1.04554 | LmEb × [无（常数项）] |
| `LmEb*eb1` | 0.106898 | LmEb × [dtype = int8] |
| `LmEb*eb2` | -0.310999 | LmEb × [dtype = float16] |
| `LmEb*eb4` | -0.356281 | LmEb × [dtype = float32] |
| `LmEb*eb8` | -0.485152 | LmEb × [dtype = int64] |
| `LmEb*W1` | -3.87102 | LmEb × [num_warps = 1] |
| `LmEb*W2` | 0.512456 | LmEb × [num_warps = 2] |
| `LmEb*W4` | 0.389566 | LmEb × [num_warps = 4] |
| `LmEb*W8` | 0.410339 | LmEb × [num_warps = 8] |
| `LmEb*W16` | 0.383368 | LmEb × [num_warps = 16] |
| `LmEb*W32` | 0.424119 | LmEb × [num_warps = 32] |
| `LmEb*W64` | 0.705639 | LmEb × [num_warps = 64] |
| `LmEb*Clt8` | 0.094667 | LmEb × [C < 8] |
| `LmEb*Clt16` | -0.302407 | LmEb × [C < 16] |
| `LmEb*Clt32` | 0.023176 | LmEb × [C < 32] |
| `LmEb*Clt64` | 0.162153 | LmEb × [C < 64] |
| `LmEb*nlt64` | 0.162153 | LmEb × [n < 64] |
| `LmEb*nge128` | -0.216009 | LmEb × [n >= 128] |
| `LmEb*nge256` | -0.082779 | LmEb × [n >= 256] |
| `LmEb*Lgt4` | 3.25765 | LmEb × [L > 4] |
| `LmEb*Lgt16` | 0.116336 | LmEb × [L > 16] |
| `LmEb*Lgt64` | -1.2507 | LmEb × [L > 64] |
| `LmEb*mwlg2` | -0.7528 | LmEb × [mean_warp_lines > 2] |
| `LmEb*mwlg8` | 0.18338 | LmEb × [mean_warp_lines > 8] |
| `LmEb*smISB` | -1.00081 | LmEb × [inner stride <= 8 bytes] |
| `minL128*const` | 3.684 | min(L,128) × [无（常数项）] |
| `minL128*eb1` | -2.19927 | min(L,128) × [dtype = int8] |
| `minL128*eb2` | -0.729713 | min(L,128) × [dtype = float16] |
| `minL128*eb4` | -0.420008 | min(L,128) × [dtype = float32] |
| `minL128*eb8` | 7.03298 | min(L,128) × [dtype = int64] |
| `minL128*W1` | 0.412275 | min(L,128) × [num_warps = 1] |
| `minL128*W2` | -0.06744 | min(L,128) × [num_warps = 2] |
| `minL128*W4` | -0.865659 | min(L,128) × [num_warps = 4] |
| `minL128*W8` | -1.23287 | min(L,128) × [num_warps = 8] |
| `minL128*W16` | -0.036563 | min(L,128) × [num_warps = 16] |
| `minL128*W32` | 1.08923 | min(L,128) × [num_warps = 32] |
| `minL128*W64` | 4.38502 | min(L,128) × [num_warps = 64] |
| `minL128*Clt8` | -0.783927 | min(L,128) × [C < 8] |
| `minL128*Clt16` | 0.065442 | min(L,128) × [C < 16] |
| `minL128*Clt32` | -0.051229 | min(L,128) × [C < 32] |
| `minL128*Clt64` | -0.257202 | min(L,128) × [C < 64] |
| `minL128*nlt64` | -0.257202 | min(L,128) × [n < 64] |
| `minL128*nge128` | 4.84481 | min(L,128) × [n >= 128] |
| `minL128*nge256` | -12.0322 | min(L,128) × [n >= 256] |
| `minL128*Lgt4` | 1.00083 | min(L,128) × [L > 4] |
| `minL128*Lgt16` | 2.75823 | min(L,128) × [L > 16] |
| `minL128*Lgt64` | -1.82348 | min(L,128) × [L > 64] |
| `minL128*mwlg2` | 4.25493 | min(L,128) × [mean_warp_lines > 2] |
| `minL128*mwlg8` | -5.89504 | min(L,128) × [mean_warp_lines > 8] |
| `minL128*smISB` | 0.510124 | min(L,128) × [inner stride <= 8 bytes] |
| `mx_iters_mwl*const` | -0.508236 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [无（常数项）] |
| `mx_iters_mwl*eb1` | 2.33121 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = int8] |
| `mx_iters_mwl*eb2` | 2.15387 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = float16] |
| `mx_iters_mwl*eb4` | -1.69176 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = float32] |
| `mx_iters_mwl*eb8` | -3.30156 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = int64] |
| `mx_iters_mwl*W1` | -5.91919 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 1] |
| `mx_iters_mwl*W2` | -3.40012 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 2] |
| `mx_iters_mwl*W4` | -4.38544 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 4] |
| `mx_iters_mwl*W8` | -3.17787 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 8] |
| `mx_iters_mwl*W16` | -2.03272 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 16] |
| `mx_iters_mwl*W32` | 3.39501 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 32] |
| `mx_iters_mwl*W64` | 15.0121 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 64] |
| `mx_iters_mwl*Clt8` | -0.783927 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [C < 8] |
| `mx_iters_mwl*Clt16` | 0.065442 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [C < 16] |
| `mx_iters_mwl*Clt32` | -0.051229 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [C < 32] |
| `mx_iters_mwl*Clt64` | -0.257202 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [C < 64] |
| `mx_iters_mwl*nlt64` | -0.257202 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [n < 64] |
| `mx_iters_mwl*nge128` | 8.66237 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [n >= 128] |
| `mx_iters_mwl*nge256` | -6.95594 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [n >= 256] |
| `mx_iters_mwl*Lgt4` | 6.67817 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [L > 4] |
| `mx_iters_mwl*Lgt16` | 1.6671 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [L > 16] |
| `mx_iters_mwl*Lgt64` | 10.8838 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [L > 64] |
| `mx_iters_mwl*mwlg2` | 2.70636 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [mean_warp_lines > 2] |
| `mx_iters_mwl*mwlg8` | -2.89895 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [mean_warp_lines > 8] |
| `mx_iters_mwl*smISB` | -2.13355 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [inner stride <= 8 bytes] |

### 3. in-sample 拟合误差

- 点数：**855**
- 项数：**217**
- **MAPE：7.37%**

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
T = 115.826
  + 17.8233 * transX*const
  + 5.27762 * transX*eb1
  + 4.12767 * transX*eb2
  + 3.64199 * transX*eb4
  + 4.77606 * transX*eb8
  + -8.03919 * transX*W1
  + 4.15274 * transX*W2
  + 10.5552 * transX*W4
  + -1.62225 * transX*W8
  + -0.931568 * transX*W16
  + 2.74914 * transX*W32
  + 10.9593 * transX*W64
  + 2.03605 * transX*Clt8
  + 0.049839 * transX*Clt16
  + -1.47155 * transX*Clt32
  + -10.9528 * transX*Clt64
  + 0.672308 * transX*nlt64
  + 0.764407 * transX*nge128
  + 2.22855 * transX*nge256
  + -1.30968 * transX*Rgt8
  + -0.623409 * transX*Rgt16
  + -12.5955 * transX*Lgt4
  + 1.41802 * transX*Lgt16
  + -0.951044 * transX*Lgt64
  + 6.77401 * transX*mwlg2
  + -0.056634 * transX*mwlg8
  + 1.79623 * transX*smISB
  + -11.717 * r_L_rl*const
  + -2.51167 * r_L_rl*eb1
  + -2.4786 * r_L_rl*eb2
  + -3.89329 * r_L_rl*eb4
  + -2.83343 * r_L_rl*eb8
  + 3.89761 * r_L_rl*W1
  + -0.854335 * r_L_rl*W2
  + 1.73718 * r_L_rl*W4
  + 0.76572 * r_L_rl*W8
  + -5.27517 * r_L_rl*W16
  + -6.44563 * r_L_rl*W32
  + -5.54237 * r_L_rl*W64
  + -1.13028 * r_L_rl*Clt8
  + 3.22621 * r_L_rl*Clt16
  + 0.320816 * r_L_rl*Clt32
  + 10.6316 * r_L_rl*Clt64
  + -2.63838 * r_L_rl*nlt64
  + 2.10968 * r_L_rl*nge128
  + 1.55794 * r_L_rl*nge256
  + -0.871 * r_L_rl*Rgt8
  + 0.179953 * r_L_rl*Rgt16
  + -0.316941 * r_L_rl*Lgt4
  + 1.50133 * r_L_rl*Lgt16
  + 2.79993 * r_L_rl*Lgt64
  + 0.806513 * r_L_rl*mwlg2
  + -1.63797 * r_L_rl*mwlg8
  + 2.41345 * r_L_rl*smISB
  + -3.22901 * mx_iters_mwl*const
  + -1.22133 * mx_iters_mwl*eb1
  + 0.063738 * mx_iters_mwl*eb2
  + -0.3042 * mx_iters_mwl*eb4
  + -1.76722 * mx_iters_mwl*eb8
  + 1.54724 * mx_iters_mwl*W1
  + -4.07875 * mx_iters_mwl*W2
  + -11.8521 * mx_iters_mwl*W4
  + -1.62225 * mx_iters_mwl*W8
  + -0.931568 * mx_iters_mwl*W16
  + 2.74914 * mx_iters_mwl*W32
  + 10.9593 * mx_iters_mwl*W64
  + 1.48925 * mx_iters_mwl*Clt8
  + 0.922797 * mx_iters_mwl*Clt16
  + 0.678449 * mx_iters_mwl*Clt32
  + 1.67225 * mx_iters_mwl*Clt64
  + 0.672308 * mx_iters_mwl*nlt64
  + -1.55481 * mx_iters_mwl*nge128
  + -4.342 * mx_iters_mwl*nge256
  + 4.62244 * mx_iters_mwl*Rgt8
  + -2.39596 * mx_iters_mwl*Rgt16
  + 5.75716 * mx_iters_mwl*Lgt4
  + -4.15128 * mx_iters_mwl*Lgt16
  + 7.69268 * mx_iters_mwl*Lgt64
  + 1.96341 * mx_iters_mwl*mwlg2
  + 7.60177 * mx_iters_mwl*mwlg8
  + -1.74651 * mx_iters_mwl*smISB
  + 2.85242 * p_eb_it*const
  + 1.61972 * p_eb_it*eb1
  + 0.908493 * p_eb_it*eb2
  + 0.825231 * p_eb_it*eb4
  + -0.50102 * p_eb_it*eb8
  + -2.62752 * p_eb_it*W1
  + -2.42157 * p_eb_it*W2
  + -2.15561 * p_eb_it*W4
  + 0.259381 * p_eb_it*W8
  + 1.5412 * p_eb_it*W16
  + 1.20822 * p_eb_it*W32
  + 7.04832 * p_eb_it*W64
  + -1.61011 * p_eb_it*Clt8
  + 0.382127 * p_eb_it*Clt16
  + 0.693098 * p_eb_it*Clt32
  + 0.5255 * p_eb_it*Clt64
  + 0.372655 * p_eb_it*nlt64
  + 0.950934 * p_eb_it*nge128
  + 1.78657 * p_eb_it*nge256
  + 3.22054 * p_eb_it*Rgt8
  + 1.0448 * p_eb_it*Rgt16
  + 0.008001 * p_eb_it*Lgt4
  + 2.14328 * p_eb_it*Lgt16
  + -1.58665 * p_eb_it*Lgt64
  + -0.069428 * p_eb_it*mwlg2
  + -0.22918 * p_eb_it*mwlg8
  + -1.41931 * p_eb_it*smISB
  + -12.6422 * le_L128xL*const
  + -3.54086 * le_L128xL*eb1
  + -1.03905 * le_L128xL*eb2
  + -4.13871 * le_L128xL*eb4
  + -3.92353 * le_L128xL*eb8
  + -0.308623 * le_L128xL*W1
  + -0.125567 * le_L128xL*W2
  + -0.446023 * le_L128xL*W4
  + -0.540444 * le_L128xL*W8
  + -0.306223 * le_L128xL*W16
  + -2.36178 * le_L128xL*W32
  + -8.5535 * le_L128xL*W64
  + 0.017571 * le_L128xL*Clt8
  + -0.615285 * le_L128xL*Clt16
  + -0.939104 * le_L128xL*Clt32
  + -1.37175 * le_L128xL*Clt64
  + 0.672308 * le_L128xL*nlt64
  + 2.14882 * le_L128xL*nge128
  + -11.4136 * le_L128xL*nge256
  + 0.689596 * le_L128xL*Rgt8
  + 0.400513 * le_L128xL*Rgt16
  + 36.4958 * le_L128xL*Lgt4
  + 19.7785 * le_L128xL*Lgt16
  + 17.4987 * le_L128xL*Lgt64
  + -23.3945 * le_L128xL*mwlg2
  + -23.2606 * le_L128xL*mwlg8
  + -22.106 * le_L128xL*smISB
  + 0.787387 * itersL*const
  + 0.583751 * itersL*eb1
  + 0.461028 * itersL*eb2
  + 0.08858 * itersL*eb4
  + -0.345972 * itersL*eb8
  + -1.39338 * itersL*W1
  + -0.943729 * itersL*W2
  + -1.11261 * itersL*W4
  + 0.293086 * itersL*W8
  + 0.962201 * itersL*W16
  + 2.1923 * itersL*W32
  + 0.789518 * itersL*W64
  + -0.332851 * itersL*Clt8
  + -0.465183 * itersL*Clt16
  + 0.043981 * itersL*Clt32
  + -0.291262 * itersL*Clt64
  + 0.672308 * itersL*nlt64
  + -0.426432 * itersL*nge128
  + -0.073806 * itersL*nge256
  + -0.030457 * itersL*Rgt8
  + 0.252885 * itersL*Rgt16
  + 0.953114 * itersL*Lgt4
  + -0.154821 * itersL*Lgt16
  + 0.333231 * itersL*Lgt64
  + -0.497948 * itersL*mwlg2
  + 0.274428 * itersL*mwlg8
  + 0.203981 * itersL*smISB
  + 12.6448 * ov_L_pages*const
  + 2.92325 * ov_L_pages*eb1
  + -0.47704 * ov_L_pages*eb2
  + 4.10705 * ov_L_pages*eb4
  + 6.09153 * ov_L_pages*eb8
  + 6.07227 * ov_L_pages*W1
  + 2.92946 * ov_L_pages*W2
  + 1.42035 * ov_L_pages*W4
  + 0.270397 * ov_L_pages*W8
  + 2.18219 * ov_L_pages*W16
  + 0.014492 * ov_L_pages*W32
  + -0.244358 * ov_L_pages*W64
  + 0.177973 * ov_L_pages*Clt8
  + -2.97821 * ov_L_pages*Clt16
  + 1.02238 * ov_L_pages*Clt32
  + 6.95139 * ov_L_pages*Clt64
  + -1.26646 * ov_L_pages*nlt64
  + -2.75648 * ov_L_pages*nge128
  + 9.46875 * ov_L_pages*nge256
  + -5.33783 * ov_L_pages*Rgt8
  + 0.938299 * ov_L_pages*Rgt16
  + -29.668 * ov_L_pages*Lgt4
  + -4.94995 * ov_L_pages*Lgt16
  + 0.870471 * ov_L_pages*Lgt64
  + 16.2035 * ov_L_pages*mwlg2
  + -6.02907 * ov_L_pages*mwlg8
  + 22.8284 * ov_L_pages*smISB
  + -6.2222 * hL4*const
  + -2.30219 * hL4*eb1
  + -0.780461 * hL4*eb2
  + -0.729182 * hL4*eb4
  + -2.41036 * hL4*eb8
  + -3.85449 * hL4*W1
  + -7.60213 * hL4*W2
  + -4.22821 * hL4*W4
  + -0.852001 * hL4*W8
  + -1.21293 * hL4*W16
  + 2.28292 * hL4*W32
  + 9.24466 * hL4*W64
  + -3.1009 * hL4*Clt8
  + 2.45093 * hL4*Clt16
  + -0.00718 * hL4*Clt32
  + 4.65335 * hL4*Clt64
  + -7.31925 * hL4*nlt64
  + 1.7636 * hL4*nge128
  + 2.06516 * hL4*nge256
  + 4.89661 * hL4*Rgt8
  + -0.599671 * hL4*Rgt16
  + -6.2222 * hL4*Lgt4
  + -18.2823 * hL4*Lgt16
  + -20.9819 * hL4*Lgt64
  + 6.24981 * hL4*mwlg2
  + 29.6791 * hL4*mwlg8
  + -3.26086 * hL4*smISB
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 115.826 | 固定开销（循环发射、启动） |
| `transX*const` | 17.8233 | iters · max_warp_lines × [无（常数项）] |
| `transX*eb1` | 5.27762 | iters · max_warp_lines × [dtype = int8] |
| `transX*eb2` | 4.12767 | iters · max_warp_lines × [dtype = float16] |
| `transX*eb4` | 3.64199 | iters · max_warp_lines × [dtype = float32] |
| `transX*eb8` | 4.77606 | iters · max_warp_lines × [dtype = int64] |
| `transX*W1` | -8.03919 | iters · max_warp_lines × [num_warps = 1] |
| `transX*W2` | 4.15274 | iters · max_warp_lines × [num_warps = 2] |
| `transX*W4` | 10.5552 | iters · max_warp_lines × [num_warps = 4] |
| `transX*W8` | -1.62225 | iters · max_warp_lines × [num_warps = 8] |
| `transX*W16` | -0.931568 | iters · max_warp_lines × [num_warps = 16] |
| `transX*W32` | 2.74914 | iters · max_warp_lines × [num_warps = 32] |
| `transX*W64` | 10.9593 | iters · max_warp_lines × [num_warps = 64] |
| `transX*Clt8` | 2.03605 | iters · max_warp_lines × [C < 8] |
| `transX*Clt16` | 0.049839 | iters · max_warp_lines × [C < 16] |
| `transX*Clt32` | -1.47155 | iters · max_warp_lines × [C < 32] |
| `transX*Clt64` | -10.9528 | iters · max_warp_lines × [C < 64] |
| `transX*nlt64` | 0.672308 | iters · max_warp_lines × [n < 64] |
| `transX*nge128` | 0.764407 | iters · max_warp_lines × [n >= 128] |
| `transX*nge256` | 2.22855 | iters · max_warp_lines × [n >= 256] |
| `transX*Rgt8` | -1.30968 | iters · max_warp_lines × [R > 8] |
| `transX*Rgt16` | -0.623409 | iters · max_warp_lines × [R > 16] |
| `transX*Lgt4` | -12.5955 | iters · max_warp_lines × [L > 4] |
| `transX*Lgt16` | 1.41802 | iters · max_warp_lines × [L > 16] |
| `transX*Lgt64` | -0.951044 | iters · max_warp_lines × [L > 64] |
| `transX*mwlg2` | 6.77401 | iters · max_warp_lines × [mean_warp_lines > 2] |
| `transX*mwlg8` | -0.056634 | iters · max_warp_lines × [mean_warp_lines > 8] |
| `transX*smISB` | 1.79623 | iters · max_warp_lines × [inner stride <= 8 bytes] |
| `r_L_rl*const` | -11.717 | r_L_rl × [无（常数项）] |
| `r_L_rl*eb1` | -2.51167 | r_L_rl × [dtype = int8] |
| `r_L_rl*eb2` | -2.4786 | r_L_rl × [dtype = float16] |
| `r_L_rl*eb4` | -3.89329 | r_L_rl × [dtype = float32] |
| `r_L_rl*eb8` | -2.83343 | r_L_rl × [dtype = int64] |
| `r_L_rl*W1` | 3.89761 | r_L_rl × [num_warps = 1] |
| `r_L_rl*W2` | -0.854335 | r_L_rl × [num_warps = 2] |
| `r_L_rl*W4` | 1.73718 | r_L_rl × [num_warps = 4] |
| `r_L_rl*W8` | 0.76572 | r_L_rl × [num_warps = 8] |
| `r_L_rl*W16` | -5.27517 | r_L_rl × [num_warps = 16] |
| `r_L_rl*W32` | -6.44563 | r_L_rl × [num_warps = 32] |
| `r_L_rl*W64` | -5.54237 | r_L_rl × [num_warps = 64] |
| `r_L_rl*Clt8` | -1.13028 | r_L_rl × [C < 8] |
| `r_L_rl*Clt16` | 3.22621 | r_L_rl × [C < 16] |
| `r_L_rl*Clt32` | 0.320816 | r_L_rl × [C < 32] |
| `r_L_rl*Clt64` | 10.6316 | r_L_rl × [C < 64] |
| `r_L_rl*nlt64` | -2.63838 | r_L_rl × [n < 64] |
| `r_L_rl*nge128` | 2.10968 | r_L_rl × [n >= 128] |
| `r_L_rl*nge256` | 1.55794 | r_L_rl × [n >= 256] |
| `r_L_rl*Rgt8` | -0.871 | r_L_rl × [R > 8] |
| `r_L_rl*Rgt16` | 0.179953 | r_L_rl × [R > 16] |
| `r_L_rl*Lgt4` | -0.316941 | r_L_rl × [L > 4] |
| `r_L_rl*Lgt16` | 1.50133 | r_L_rl × [L > 16] |
| `r_L_rl*Lgt64` | 2.79993 | r_L_rl × [L > 64] |
| `r_L_rl*mwlg2` | 0.806513 | r_L_rl × [mean_warp_lines > 2] |
| `r_L_rl*mwlg8` | -1.63797 | r_L_rl × [mean_warp_lines > 8] |
| `r_L_rl*smISB` | 2.41345 | r_L_rl × [inner stride <= 8 bytes] |
| `mx_iters_mwl*const` | -3.22901 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [无（常数项）] |
| `mx_iters_mwl*eb1` | -1.22133 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = int8] |
| `mx_iters_mwl*eb2` | 0.063738 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = float16] |
| `mx_iters_mwl*eb4` | -0.3042 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = float32] |
| `mx_iters_mwl*eb8` | -1.76722 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = int64] |
| `mx_iters_mwl*W1` | 1.54724 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 1] |
| `mx_iters_mwl*W2` | -4.07875 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 2] |
| `mx_iters_mwl*W4` | -11.8521 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 4] |
| `mx_iters_mwl*W8` | -1.62225 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 8] |
| `mx_iters_mwl*W16` | -0.931568 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 16] |
| `mx_iters_mwl*W32` | 2.74914 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 32] |
| `mx_iters_mwl*W64` | 10.9593 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 64] |
| `mx_iters_mwl*Clt8` | 1.48925 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [C < 8] |
| `mx_iters_mwl*Clt16` | 0.922797 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [C < 16] |
| `mx_iters_mwl*Clt32` | 0.678449 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [C < 32] |
| `mx_iters_mwl*Clt64` | 1.67225 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [C < 64] |
| `mx_iters_mwl*nlt64` | 0.672308 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [n < 64] |
| `mx_iters_mwl*nge128` | -1.55481 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [n >= 128] |
| `mx_iters_mwl*nge256` | -4.342 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [n >= 256] |
| `mx_iters_mwl*Rgt8` | 4.62244 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [R > 8] |
| `mx_iters_mwl*Rgt16` | -2.39596 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [R > 16] |
| `mx_iters_mwl*Lgt4` | 5.75716 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [L > 4] |
| `mx_iters_mwl*Lgt16` | -4.15128 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [L > 16] |
| `mx_iters_mwl*Lgt64` | 7.69268 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [L > 64] |
| `mx_iters_mwl*mwlg2` | 1.96341 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [mean_warp_lines > 2] |
| `mx_iters_mwl*mwlg8` | 7.60177 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [mean_warp_lines > 8] |
| `mx_iters_mwl*smISB` | -1.74651 | max(ceil(n/(32·W)) = warp 轮次, mean_warp_lines：每 warp 波次触及的 line 数) × [inner stride <= 8 bytes] |
| `p_eb_it*const` | 2.85242 | p_eb_it × [无（常数项）] |
| `p_eb_it*eb1` | 1.61972 | p_eb_it × [dtype = int8] |
| `p_eb_it*eb2` | 0.908493 | p_eb_it × [dtype = float16] |
| `p_eb_it*eb4` | 0.825231 | p_eb_it × [dtype = float32] |
| `p_eb_it*eb8` | -0.50102 | p_eb_it × [dtype = int64] |
| `p_eb_it*W1` | -2.62752 | p_eb_it × [num_warps = 1] |
| `p_eb_it*W2` | -2.42157 | p_eb_it × [num_warps = 2] |
| `p_eb_it*W4` | -2.15561 | p_eb_it × [num_warps = 4] |
| `p_eb_it*W8` | 0.259381 | p_eb_it × [num_warps = 8] |
| `p_eb_it*W16` | 1.5412 | p_eb_it × [num_warps = 16] |
| `p_eb_it*W32` | 1.20822 | p_eb_it × [num_warps = 32] |
| `p_eb_it*W64` | 7.04832 | p_eb_it × [num_warps = 64] |
| `p_eb_it*Clt8` | -1.61011 | p_eb_it × [C < 8] |
| `p_eb_it*Clt16` | 0.382127 | p_eb_it × [C < 16] |
| `p_eb_it*Clt32` | 0.693098 | p_eb_it × [C < 32] |
| `p_eb_it*Clt64` | 0.5255 | p_eb_it × [C < 64] |
| `p_eb_it*nlt64` | 0.372655 | p_eb_it × [n < 64] |
| `p_eb_it*nge128` | 0.950934 | p_eb_it × [n >= 128] |
| `p_eb_it*nge256` | 1.78657 | p_eb_it × [n >= 256] |
| `p_eb_it*Rgt8` | 3.22054 | p_eb_it × [R > 8] |
| `p_eb_it*Rgt16` | 1.0448 | p_eb_it × [R > 16] |
| `p_eb_it*Lgt4` | 0.008001 | p_eb_it × [L > 4] |
| `p_eb_it*Lgt16` | 2.14328 | p_eb_it × [L > 16] |
| `p_eb_it*Lgt64` | -1.58665 | p_eb_it × [L > 64] |
| `p_eb_it*mwlg2` | -0.069428 | p_eb_it × [mean_warp_lines > 2] |
| `p_eb_it*mwlg8` | -0.22918 | p_eb_it × [mean_warp_lines > 8] |
| `p_eb_it*smISB` | -1.41931 | p_eb_it × [inner stride <= 8 bytes] |
| `le_L128xL*const` | -12.6422 | le_L128xL × [无（常数项）] |
| `le_L128xL*eb1` | -3.54086 | le_L128xL × [dtype = int8] |
| `le_L128xL*eb2` | -1.03905 | le_L128xL × [dtype = float16] |
| `le_L128xL*eb4` | -4.13871 | le_L128xL × [dtype = float32] |
| `le_L128xL*eb8` | -3.92353 | le_L128xL × [dtype = int64] |
| `le_L128xL*W1` | -0.308623 | le_L128xL × [num_warps = 1] |
| `le_L128xL*W2` | -0.125567 | le_L128xL × [num_warps = 2] |
| `le_L128xL*W4` | -0.446023 | le_L128xL × [num_warps = 4] |
| `le_L128xL*W8` | -0.540444 | le_L128xL × [num_warps = 8] |
| `le_L128xL*W16` | -0.306223 | le_L128xL × [num_warps = 16] |
| `le_L128xL*W32` | -2.36178 | le_L128xL × [num_warps = 32] |
| `le_L128xL*W64` | -8.5535 | le_L128xL × [num_warps = 64] |
| `le_L128xL*Clt8` | 0.017571 | le_L128xL × [C < 8] |
| `le_L128xL*Clt16` | -0.615285 | le_L128xL × [C < 16] |
| `le_L128xL*Clt32` | -0.939104 | le_L128xL × [C < 32] |
| `le_L128xL*Clt64` | -1.37175 | le_L128xL × [C < 64] |
| `le_L128xL*nlt64` | 0.672308 | le_L128xL × [n < 64] |
| `le_L128xL*nge128` | 2.14882 | le_L128xL × [n >= 128] |
| `le_L128xL*nge256` | -11.4136 | le_L128xL × [n >= 256] |
| `le_L128xL*Rgt8` | 0.689596 | le_L128xL × [R > 8] |
| `le_L128xL*Rgt16` | 0.400513 | le_L128xL × [R > 16] |
| `le_L128xL*Lgt4` | 36.4958 | le_L128xL × [L > 4] |
| `le_L128xL*Lgt16` | 19.7785 | le_L128xL × [L > 16] |
| `le_L128xL*Lgt64` | 17.4987 | le_L128xL × [L > 64] |
| `le_L128xL*mwlg2` | -23.3945 | le_L128xL × [mean_warp_lines > 2] |
| `le_L128xL*mwlg8` | -23.2606 | le_L128xL × [mean_warp_lines > 8] |
| `le_L128xL*smISB` | -22.106 | le_L128xL × [inner stride <= 8 bytes] |
| `itersL*const` | 0.787387 | iters · L × [无（常数项）] |
| `itersL*eb1` | 0.583751 | iters · L × [dtype = int8] |
| `itersL*eb2` | 0.461028 | iters · L × [dtype = float16] |
| `itersL*eb4` | 0.08858 | iters · L × [dtype = float32] |
| `itersL*eb8` | -0.345972 | iters · L × [dtype = int64] |
| `itersL*W1` | -1.39338 | iters · L × [num_warps = 1] |
| `itersL*W2` | -0.943729 | iters · L × [num_warps = 2] |
| `itersL*W4` | -1.11261 | iters · L × [num_warps = 4] |
| `itersL*W8` | 0.293086 | iters · L × [num_warps = 8] |
| `itersL*W16` | 0.962201 | iters · L × [num_warps = 16] |
| `itersL*W32` | 2.1923 | iters · L × [num_warps = 32] |
| `itersL*W64` | 0.789518 | iters · L × [num_warps = 64] |
| `itersL*Clt8` | -0.332851 | iters · L × [C < 8] |
| `itersL*Clt16` | -0.465183 | iters · L × [C < 16] |
| `itersL*Clt32` | 0.043981 | iters · L × [C < 32] |
| `itersL*Clt64` | -0.291262 | iters · L × [C < 64] |
| `itersL*nlt64` | 0.672308 | iters · L × [n < 64] |
| `itersL*nge128` | -0.426432 | iters · L × [n >= 128] |
| `itersL*nge256` | -0.073806 | iters · L × [n >= 256] |
| `itersL*Rgt8` | -0.030457 | iters · L × [R > 8] |
| `itersL*Rgt16` | 0.252885 | iters · L × [R > 16] |
| `itersL*Lgt4` | 0.953114 | iters · L × [L > 4] |
| `itersL*Lgt16` | -0.154821 | iters · L × [L > 16] |
| `itersL*Lgt64` | 0.333231 | iters · L × [L > 64] |
| `itersL*mwlg2` | -0.497948 | iters · L × [mean_warp_lines > 2] |
| `itersL*mwlg8` | 0.274428 | iters · L × [mean_warp_lines > 8] |
| `itersL*smISB` | 0.203981 | iters · L × [inner stride <= 8 bytes] |
| `ov_L_pages*const` | 12.6448 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [无（常数项）] |
| `ov_L_pages*eb1` | 2.92325 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [dtype = int8] |
| `ov_L_pages*eb2` | -0.47704 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [dtype = float16] |
| `ov_L_pages*eb4` | 4.10705 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [dtype = float32] |
| `ov_L_pages*eb8` | 6.09153 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [dtype = int64] |
| `ov_L_pages*W1` | 6.07227 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [num_warps = 1] |
| `ov_L_pages*W2` | 2.92946 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [num_warps = 2] |
| `ov_L_pages*W4` | 1.42035 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [num_warps = 4] |
| `ov_L_pages*W8` | 0.270397 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [num_warps = 8] |
| `ov_L_pages*W16` | 2.18219 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [num_warps = 16] |
| `ov_L_pages*W32` | 0.014492 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [num_warps = 32] |
| `ov_L_pages*W64` | -0.244358 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [num_warps = 64] |
| `ov_L_pages*Clt8` | 0.177973 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [C < 8] |
| `ov_L_pages*Clt16` | -2.97821 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [C < 16] |
| `ov_L_pages*Clt32` | 1.02238 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [C < 32] |
| `ov_L_pages*Clt64` | 6.95139 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [C < 64] |
| `ov_L_pages*nlt64` | -1.26646 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [n < 64] |
| `ov_L_pages*nge128` | -2.75648 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [n >= 128] |
| `ov_L_pages*nge256` | 9.46875 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [n >= 256] |
| `ov_L_pages*Rgt8` | -5.33783 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [R > 8] |
| `ov_L_pages*Rgt16` | 0.938299 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [R > 16] |
| `ov_L_pages*Lgt4` | -29.668 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [L > 4] |
| `ov_L_pages*Lgt16` | -4.94995 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [L > 16] |
| `ov_L_pages*Lgt64` | 0.870471 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [L > 64] |
| `ov_L_pages*mwlg2` | 16.2035 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [mean_warp_lines > 2] |
| `ov_L_pages*mwlg8` | -6.02907 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [mean_warp_lines > 8] |
| `ov_L_pages*smISB` | 22.8284 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [inner stride <= 8 bytes] |
| `hL4*const` | -6.2222 | max(0, L−4) × [无（常数项）] |
| `hL4*eb1` | -2.30219 | max(0, L−4) × [dtype = int8] |
| `hL4*eb2` | -0.780461 | max(0, L−4) × [dtype = float16] |
| `hL4*eb4` | -0.729182 | max(0, L−4) × [dtype = float32] |
| `hL4*eb8` | -2.41036 | max(0, L−4) × [dtype = int64] |
| `hL4*W1` | -3.85449 | max(0, L−4) × [num_warps = 1] |
| `hL4*W2` | -7.60213 | max(0, L−4) × [num_warps = 2] |
| `hL4*W4` | -4.22821 | max(0, L−4) × [num_warps = 4] |
| `hL4*W8` | -0.852001 | max(0, L−4) × [num_warps = 8] |
| `hL4*W16` | -1.21293 | max(0, L−4) × [num_warps = 16] |
| `hL4*W32` | 2.28292 | max(0, L−4) × [num_warps = 32] |
| `hL4*W64` | 9.24466 | max(0, L−4) × [num_warps = 64] |
| `hL4*Clt8` | -3.1009 | max(0, L−4) × [C < 8] |
| `hL4*Clt16` | 2.45093 | max(0, L−4) × [C < 16] |
| `hL4*Clt32` | -0.00718 | max(0, L−4) × [C < 32] |
| `hL4*Clt64` | 4.65335 | max(0, L−4) × [C < 64] |
| `hL4*nlt64` | -7.31925 | max(0, L−4) × [n < 64] |
| `hL4*nge128` | 1.7636 | max(0, L−4) × [n >= 128] |
| `hL4*nge256` | 2.06516 | max(0, L−4) × [n >= 256] |
| `hL4*Rgt8` | 4.89661 | max(0, L−4) × [R > 8] |
| `hL4*Rgt16` | -0.599671 | max(0, L−4) × [R > 16] |
| `hL4*Lgt4` | -6.2222 | max(0, L−4) × [L > 4] |
| `hL4*Lgt16` | -18.2823 | max(0, L−4) × [L > 16] |
| `hL4*Lgt64` | -20.9819 | max(0, L−4) × [L > 64] |
| `hL4*mwlg2` | 6.24981 | max(0, L−4) × [mean_warp_lines > 2] |
| `hL4*mwlg8` | 29.6791 | max(0, L−4) × [mean_warp_lines > 8] |
| `hL4*smISB` | -3.26086 | max(0, L−4) × [inner stride <= 8 bytes] |

### 3. in-sample 拟合误差

- 点数：**1762**
- 项数：**217**
- **MAPE：6.16%**

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
T = -11.1745
  + -0.357602 * mx_L_W*const
  + -0.091664 * mx_L_W*eb1
  + -0.043339 * mx_L_W*eb2
  + -0.136066 * mx_L_W*eb4
  + -0.086532 * mx_L_W*eb8
  + -5.22936 * mx_L_W*W1
  + 0.15343 * mx_L_W*W2
  + -0.73634 * mx_L_W*W4
  + 0.738447 * mx_L_W*W8
  + 0.696172 * mx_L_W*W16
  + 1.7757 * mx_L_W*W32
  + 2.24435 * mx_L_W*W64
  + 0.055453 * mx_L_W*Clt8
  + 0.105832 * mx_L_W*Clt16
  + -0.993069 * mx_L_W*Clt32
  + 0.492573 * mx_L_W*Clt64
  + 0.492573 * mx_L_W*nlt64
  + -3.02078 * mx_L_W*nge128
  + 3.10975 * mx_L_W*nge256
  + 0.158087 * mx_L_W*Lgt4
  + -0.123881 * mx_L_W*Lgt16
  + 1.75537 * mx_L_W*Lgt64
  + 0.067642 * mx_L_W*mwlg2
  + -0.195967 * mx_L_W*mwlg8
  + -0.038301 * mx_L_W*smISB
  + 1.02 * hLeb4*const
  + -1.25905 * hLeb4*eb1
  + 1.59801 * hLeb4*eb2
  + 0.811773 * hLeb4*eb4
  + -0.130732 * hLeb4*eb8
  + 0.024775 * hLeb4*W1
  + 0.428962 * hLeb4*W2
  + 0.243027 * hLeb4*W4
  + 0.194339 * hLeb4*W8
  + 0.469997 * hLeb4*W16
  + -0.23332 * hLeb4*W32
  + -0.107783 * hLeb4*W64
  + -0.558157 * hLeb4*Clt16
  + -0.219519 * hLeb4*Clt32
  + 0.502236 * hLeb4*Clt64
  + 0.502236 * hLeb4*nlt64
  + -0.163982 * hLeb4*nge128
  + -0.263454 * hLeb4*nge256
  + 1.02 * hLeb4*Lgt4
  + 0.522836 * hLeb4*Lgt16
  + 0.342821 * hLeb4*Lgt64
  + 0.527672 * hLeb4*mwlg2
  + 0.738456 * hLeb4*mwlg8
  + 0.492325 * hLeb4*smISB
  + 21.2141 * cmd*const
  + 8.07261 * cmd*eb1
  + 5.73907 * cmd*eb2
  + 3.23129 * cmd*eb4
  + 4.1711 * cmd*eb8
  + 3.88475 * cmd*W1
  + 2.76277 * cmd*W2
  + 2.58094 * cmd*W4
  + 0.005209 * cmd*W8
  + 7.55664 * cmd*W16
  + 2.82536 * cmd*W32
  + 1.59839 * cmd*W64
  + -0.340052 * cmd*Clt8
  + 0.035574 * cmd*Clt16
  + 2.49845 * cmd*Clt32
  + 6.13088 * cmd*Clt64
  + 6.13088 * cmd*nlt64
  + -0.274017 * cmd*nge128
  + 1.1658 * cmd*nge256
  + -6.69838 * cmd*Lgt4
  + -12.2196 * cmd*Lgt16
  + 0.417072 * cmd*Lgt64
  + 4.12694 * cmd*mwlg2
  + 1.81323 * cmd*mwlg8
  + -0.295998 * cmd*smISB
  + -7.15255 * ov_L_iters*const
  + -0.00907 * ov_L_iters*eb1
  + -0.459461 * ov_L_iters*eb2
  + -4.79246 * ov_L_iters*eb4
  + -1.89156 * ov_L_iters*eb8
  + -4.12054 * ov_L_iters*W1
  + 0.756027 * ov_L_iters*W2
  + -2.60192 * ov_L_iters*W4
  + -0.040614 * ov_L_iters*W8
  + 0.029986 * ov_L_iters*W16
  + -0.460385 * ov_L_iters*W32
  + -0.715113 * ov_L_iters*W64
  + 0.33999 * ov_L_iters*Clt8
  + -0.341364 * ov_L_iters*Clt16
  + -0.77159 * ov_L_iters*Clt32
  + 8.08025 * ov_L_iters*Clt64
  + 8.08025 * ov_L_iters*nlt64
  + 3.5042 * ov_L_iters*nge128
  + -5.33436 * ov_L_iters*nge256
  + 3.21166 * ov_L_iters*Lgt4
  + -0.052788 * ov_L_iters*Lgt16
  + -4.28951 * ov_L_iters*Lgt64
  + 2.33473 * ov_L_iters*mwlg2
  + 0.112511 * ov_L_iters*mwlg8
  + 0.569963 * ov_L_iters*smISB
  + 5.79407 * mx_L_iters*const
  + -1.27715 * mx_L_iters*eb1
  + 0.11527 * mx_L_iters*eb2
  + 4.79544 * mx_L_iters*eb4
  + 2.16051 * mx_L_iters*eb8
  + 2.91447 * mx_L_iters*W1
  + -0.23247 * mx_L_iters*W2
  + 3.70473 * mx_L_iters*W4
  + 0.269865 * mx_L_iters*W8
  + 0.165364 * mx_L_iters*W16
  + -0.365209 * mx_L_iters*W32
  + -0.662683 * mx_L_iters*W64
  + -6.2e-05 * mx_L_iters*Clt8
  + -0.30579 * mx_L_iters*Clt16
  + 1.72686 * mx_L_iters*Clt32
  + -8.12614 * mx_L_iters*Clt64
  + -8.12614 * mx_L_iters*nlt64
  + 6.58652 * mx_L_iters*nge128
  + -2.67984 * mx_L_iters*nge256
  + -2.51029 * mx_L_iters*Lgt4
  + 0.732489 * mx_L_iters*Lgt16
  + 1.75537 * mx_L_iters*Lgt64
  + -2.41466 * mx_L_iters*mwlg2
  + 0.523462 * mx_L_iters*mwlg8
  + 0.882862 * mx_L_iters*smISB
  + 0.647463 * hLW*const
  + -0.108954 * hLW*eb1
  + -0.275991 * hLW*eb2
  + 0.314831 * hLW*eb4
  + 0.717576 * hLW*eb8
  + 6.42938 * hLW*W1
  + 0.192748 * hLW*W2
  + -0.378283 * hLW*W4
  + -1.74539 * hLW*W8
  + -1.46988 * hLW*W16
  + -1.26992 * hLW*W32
  + -1.11121 * hLW*W64
  + -0.019065 * hLW*Clt8
  + -0.308497 * hLW*Clt16
  + 1.32615 * hLW*Clt32
  + -0.834377 * hLW*Clt64
  + -0.834377 * hLW*nlt64
  + 0.549353 * hLW*nge128
  + -0.433224 * hLW*nge256
  + -1.58501 * hLW*Lgt4
  + -0.792126 * hLW*Lgt16
  + -0.689857 * hLW*Lgt64
  + 1.54867 * hLW*mwlg2
  + 0.355083 * hLW*mwlg8
  + 0.193909 * hLW*smISB
  + 0.101576 * ov_L_mwl*const
  + 1.18786 * ov_L_mwl*eb1
  + 0.06497 * ov_L_mwl*eb2
  + -0.769796 * ov_L_mwl*eb4
  + -0.381461 * ov_L_mwl*eb8
  + -0.978275 * ov_L_mwl*W1
  + -0.82251 * ov_L_mwl*W2
  + -0.335484 * ov_L_mwl*W4
  + 0.122403 * ov_L_mwl*W8
  + 0.251506 * ov_L_mwl*W16
  + 1.26728 * ov_L_mwl*W32
  + 0.596659 * ov_L_mwl*W64
  + -7.46516 * ov_L_mwl*nge128
  + 4.48107 * ov_L_mwl*nge256
  + 0.686031 * ov_L_mwl*Lgt4
  + 0.105174 * ov_L_mwl*Lgt16
  + 0.788523 * ov_L_mwl*Lgt64
  + 0.712144 * ov_L_mwl*mwlg2
  + 0.331898 * ov_L_mwl*mwlg8
  + -0.610568 * ov_L_mwl*smISB
  + 0.362127 * hLW05*const
  + 1.40902 * hLW05*eb1
  + 1.35839 * hLW05*eb2
  + -1.03616 * hLW05*eb4
  + -1.36912 * hLW05*eb8
  + 0.600014 * hLW05*W1
  + -0.212811 * hLW05*W2
  + 0.364778 * hLW05*W4
  + 0.936424 * hLW05*W8
  + 0.127476 * hLW05*W16
  + -0.902074 * hLW05*W32
  + -0.55168 * hLW05*W64
  + 0.066254 * hLW05*Clt8
  + 0.74536 * hLW05*Clt16
  + -1.30833 * hLW05*Clt32
  + 0.623107 * hLW05*Clt64
  + 0.623107 * hLW05*nlt64
  + -2.17491 * hLW05*nge128
  + 1.86847 * hLW05*nge256
  + 2.6263 * hLW05*Lgt4
  + 0.398366 * hLW05*Lgt16
  + 0.532755 * hLW05*Lgt64
  + -2.10229 * hLW05*mwlg2
  + -1.24357 * hLW05*mwlg8
  + -1.35676 * hLW05*smISB
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | -11.1745 | 固定开销（循环发射、启动） |
| `mx_L_W*const` | -0.357602 | max(触及的 128B line 数, num_warps) × [无（常数项）] |
| `mx_L_W*eb1` | -0.091664 | max(触及的 128B line 数, num_warps) × [dtype = int8] |
| `mx_L_W*eb2` | -0.043339 | max(触及的 128B line 数, num_warps) × [dtype = float16] |
| `mx_L_W*eb4` | -0.136066 | max(触及的 128B line 数, num_warps) × [dtype = float32] |
| `mx_L_W*eb8` | -0.086532 | max(触及的 128B line 数, num_warps) × [dtype = int64] |
| `mx_L_W*W1` | -5.22936 | max(触及的 128B line 数, num_warps) × [num_warps = 1] |
| `mx_L_W*W2` | 0.15343 | max(触及的 128B line 数, num_warps) × [num_warps = 2] |
| `mx_L_W*W4` | -0.73634 | max(触及的 128B line 数, num_warps) × [num_warps = 4] |
| `mx_L_W*W8` | 0.738447 | max(触及的 128B line 数, num_warps) × [num_warps = 8] |
| `mx_L_W*W16` | 0.696172 | max(触及的 128B line 数, num_warps) × [num_warps = 16] |
| `mx_L_W*W32` | 1.7757 | max(触及的 128B line 数, num_warps) × [num_warps = 32] |
| `mx_L_W*W64` | 2.24435 | max(触及的 128B line 数, num_warps) × [num_warps = 64] |
| `mx_L_W*Clt8` | 0.055453 | max(触及的 128B line 数, num_warps) × [C < 8] |
| `mx_L_W*Clt16` | 0.105832 | max(触及的 128B line 数, num_warps) × [C < 16] |
| `mx_L_W*Clt32` | -0.993069 | max(触及的 128B line 数, num_warps) × [C < 32] |
| `mx_L_W*Clt64` | 0.492573 | max(触及的 128B line 数, num_warps) × [C < 64] |
| `mx_L_W*nlt64` | 0.492573 | max(触及的 128B line 数, num_warps) × [n < 64] |
| `mx_L_W*nge128` | -3.02078 | max(触及的 128B line 数, num_warps) × [n >= 128] |
| `mx_L_W*nge256` | 3.10975 | max(触及的 128B line 数, num_warps) × [n >= 256] |
| `mx_L_W*Lgt4` | 0.158087 | max(触及的 128B line 数, num_warps) × [L > 4] |
| `mx_L_W*Lgt16` | -0.123881 | max(触及的 128B line 数, num_warps) × [L > 16] |
| `mx_L_W*Lgt64` | 1.75537 | max(触及的 128B line 数, num_warps) × [L > 64] |
| `mx_L_W*mwlg2` | 0.067642 | max(触及的 128B line 数, num_warps) × [mean_warp_lines > 2] |
| `mx_L_W*mwlg8` | -0.195967 | max(触及的 128B line 数, num_warps) × [mean_warp_lines > 8] |
| `mx_L_W*smISB` | -0.038301 | max(触及的 128B line 数, num_warps) × [inner stride <= 8 bytes] |
| `hLeb4*const` | 1.02 | max(0, L/eb − 4) × [无（常数项）] |
| `hLeb4*eb1` | -1.25905 | max(0, L/eb − 4) × [dtype = int8] |
| `hLeb4*eb2` | 1.59801 | max(0, L/eb − 4) × [dtype = float16] |
| `hLeb4*eb4` | 0.811773 | max(0, L/eb − 4) × [dtype = float32] |
| `hLeb4*eb8` | -0.130732 | max(0, L/eb − 4) × [dtype = int64] |
| `hLeb4*W1` | 0.024775 | max(0, L/eb − 4) × [num_warps = 1] |
| `hLeb4*W2` | 0.428962 | max(0, L/eb − 4) × [num_warps = 2] |
| `hLeb4*W4` | 0.243027 | max(0, L/eb − 4) × [num_warps = 4] |
| `hLeb4*W8` | 0.194339 | max(0, L/eb − 4) × [num_warps = 8] |
| `hLeb4*W16` | 0.469997 | max(0, L/eb − 4) × [num_warps = 16] |
| `hLeb4*W32` | -0.23332 | max(0, L/eb − 4) × [num_warps = 32] |
| `hLeb4*W64` | -0.107783 | max(0, L/eb − 4) × [num_warps = 64] |
| `hLeb4*Clt16` | -0.558157 | max(0, L/eb − 4) × [C < 16] |
| `hLeb4*Clt32` | -0.219519 | max(0, L/eb − 4) × [C < 32] |
| `hLeb4*Clt64` | 0.502236 | max(0, L/eb − 4) × [C < 64] |
| `hLeb4*nlt64` | 0.502236 | max(0, L/eb − 4) × [n < 64] |
| `hLeb4*nge128` | -0.163982 | max(0, L/eb − 4) × [n >= 128] |
| `hLeb4*nge256` | -0.263454 | max(0, L/eb − 4) × [n >= 256] |
| `hLeb4*Lgt4` | 1.02 | max(0, L/eb − 4) × [L > 4] |
| `hLeb4*Lgt16` | 0.522836 | max(0, L/eb − 4) × [L > 16] |
| `hLeb4*Lgt64` | 0.342821 | max(0, L/eb − 4) × [L > 64] |
| `hLeb4*mwlg2` | 0.527672 | max(0, L/eb − 4) × [mean_warp_lines > 2] |
| `hLeb4*mwlg8` | 0.738456 | max(0, L/eb − 4) × [mean_warp_lines > 8] |
| `hLeb4*smISB` | 0.492325 | max(0, L/eb − 4) × [inner stride <= 8 bytes] |
| `cmd*const` | 21.2141 | ceil(L / floor(128/eb)) = MTE 命令数 × [无（常数项）] |
| `cmd*eb1` | 8.07261 | ceil(L / floor(128/eb)) = MTE 命令数 × [dtype = int8] |
| `cmd*eb2` | 5.73907 | ceil(L / floor(128/eb)) = MTE 命令数 × [dtype = float16] |
| `cmd*eb4` | 3.23129 | ceil(L / floor(128/eb)) = MTE 命令数 × [dtype = float32] |
| `cmd*eb8` | 4.1711 | ceil(L / floor(128/eb)) = MTE 命令数 × [dtype = int64] |
| `cmd*W1` | 3.88475 | ceil(L / floor(128/eb)) = MTE 命令数 × [num_warps = 1] |
| `cmd*W2` | 2.76277 | ceil(L / floor(128/eb)) = MTE 命令数 × [num_warps = 2] |
| `cmd*W4` | 2.58094 | ceil(L / floor(128/eb)) = MTE 命令数 × [num_warps = 4] |
| `cmd*W8` | 0.005209 | ceil(L / floor(128/eb)) = MTE 命令数 × [num_warps = 8] |
| `cmd*W16` | 7.55664 | ceil(L / floor(128/eb)) = MTE 命令数 × [num_warps = 16] |
| `cmd*W32` | 2.82536 | ceil(L / floor(128/eb)) = MTE 命令数 × [num_warps = 32] |
| `cmd*W64` | 1.59839 | ceil(L / floor(128/eb)) = MTE 命令数 × [num_warps = 64] |
| `cmd*Clt8` | -0.340052 | ceil(L / floor(128/eb)) = MTE 命令数 × [C < 8] |
| `cmd*Clt16` | 0.035574 | ceil(L / floor(128/eb)) = MTE 命令数 × [C < 16] |
| `cmd*Clt32` | 2.49845 | ceil(L / floor(128/eb)) = MTE 命令数 × [C < 32] |
| `cmd*Clt64` | 6.13088 | ceil(L / floor(128/eb)) = MTE 命令数 × [C < 64] |
| `cmd*nlt64` | 6.13088 | ceil(L / floor(128/eb)) = MTE 命令数 × [n < 64] |
| `cmd*nge128` | -0.274017 | ceil(L / floor(128/eb)) = MTE 命令数 × [n >= 128] |
| `cmd*nge256` | 1.1658 | ceil(L / floor(128/eb)) = MTE 命令数 × [n >= 256] |
| `cmd*Lgt4` | -6.69838 | ceil(L / floor(128/eb)) = MTE 命令数 × [L > 4] |
| `cmd*Lgt16` | -12.2196 | ceil(L / floor(128/eb)) = MTE 命令数 × [L > 16] |
| `cmd*Lgt64` | 0.417072 | ceil(L / floor(128/eb)) = MTE 命令数 × [L > 64] |
| `cmd*mwlg2` | 4.12694 | ceil(L / floor(128/eb)) = MTE 命令数 × [mean_warp_lines > 2] |
| `cmd*mwlg8` | 1.81323 | ceil(L / floor(128/eb)) = MTE 命令数 × [mean_warp_lines > 8] |
| `cmd*smISB` | -0.295998 | ceil(L / floor(128/eb)) = MTE 命令数 × [inner stride <= 8 bytes] |
| `ov_L_iters*const` | -7.15255 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [无（常数项）] |
| `ov_L_iters*eb1` | -0.00907 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [dtype = int8] |
| `ov_L_iters*eb2` | -0.459461 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [dtype = float16] |
| `ov_L_iters*eb4` | -4.79246 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [dtype = float32] |
| `ov_L_iters*eb8` | -1.89156 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [dtype = int64] |
| `ov_L_iters*W1` | -4.12054 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [num_warps = 1] |
| `ov_L_iters*W2` | 0.756027 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [num_warps = 2] |
| `ov_L_iters*W4` | -2.60192 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [num_warps = 4] |
| `ov_L_iters*W8` | -0.040614 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [num_warps = 8] |
| `ov_L_iters*W16` | 0.029986 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [num_warps = 16] |
| `ov_L_iters*W32` | -0.460385 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [num_warps = 32] |
| `ov_L_iters*W64` | -0.715113 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [num_warps = 64] |
| `ov_L_iters*Clt8` | 0.33999 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [C < 8] |
| `ov_L_iters*Clt16` | -0.341364 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [C < 16] |
| `ov_L_iters*Clt32` | -0.77159 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [C < 32] |
| `ov_L_iters*Clt64` | 8.08025 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [C < 64] |
| `ov_L_iters*nlt64` | 8.08025 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [n < 64] |
| `ov_L_iters*nge128` | 3.5042 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [n >= 128] |
| `ov_L_iters*nge256` | -5.33436 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [n >= 256] |
| `ov_L_iters*Lgt4` | 3.21166 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [L > 4] |
| `ov_L_iters*Lgt16` | -0.052788 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [L > 16] |
| `ov_L_iters*Lgt64` | -4.28951 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [L > 64] |
| `ov_L_iters*mwlg2` | 2.33473 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [mean_warp_lines > 2] |
| `ov_L_iters*mwlg8` | 0.112511 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [mean_warp_lines > 8] |
| `ov_L_iters*smISB` | 0.569963 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [inner stride <= 8 bytes] |
| `mx_L_iters*const` | 5.79407 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [无（常数项）] |
| `mx_L_iters*eb1` | -1.27715 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [dtype = int8] |
| `mx_L_iters*eb2` | 0.11527 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [dtype = float16] |
| `mx_L_iters*eb4` | 4.79544 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [dtype = float32] |
| `mx_L_iters*eb8` | 2.16051 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [dtype = int64] |
| `mx_L_iters*W1` | 2.91447 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [num_warps = 1] |
| `mx_L_iters*W2` | -0.23247 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [num_warps = 2] |
| `mx_L_iters*W4` | 3.70473 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [num_warps = 4] |
| `mx_L_iters*W8` | 0.269865 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [num_warps = 8] |
| `mx_L_iters*W16` | 0.165364 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [num_warps = 16] |
| `mx_L_iters*W32` | -0.365209 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [num_warps = 32] |
| `mx_L_iters*W64` | -0.662683 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [num_warps = 64] |
| `mx_L_iters*Clt8` | -6.2e-05 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [C < 8] |
| `mx_L_iters*Clt16` | -0.30579 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [C < 16] |
| `mx_L_iters*Clt32` | 1.72686 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [C < 32] |
| `mx_L_iters*Clt64` | -8.12614 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [C < 64] |
| `mx_L_iters*nlt64` | -8.12614 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [n < 64] |
| `mx_L_iters*nge128` | 6.58652 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [n >= 128] |
| `mx_L_iters*nge256` | -2.67984 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [n >= 256] |
| `mx_L_iters*Lgt4` | -2.51029 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [L > 4] |
| `mx_L_iters*Lgt16` | 0.732489 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [L > 16] |
| `mx_L_iters*Lgt64` | 1.75537 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [L > 64] |
| `mx_L_iters*mwlg2` | -2.41466 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [mean_warp_lines > 2] |
| `mx_L_iters*mwlg8` | 0.523462 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [mean_warp_lines > 8] |
| `mx_L_iters*smISB` | 0.882862 | max(触及的 128B line 数, ceil(n/(32·W)) = warp 轮次) × [inner stride <= 8 bytes] |
| `hLW*const` | 0.647463 | hLW × [无（常数项）] |
| `hLW*eb1` | -0.108954 | hLW × [dtype = int8] |
| `hLW*eb2` | -0.275991 | hLW × [dtype = float16] |
| `hLW*eb4` | 0.314831 | hLW × [dtype = float32] |
| `hLW*eb8` | 0.717576 | hLW × [dtype = int64] |
| `hLW*W1` | 6.42938 | hLW × [num_warps = 1] |
| `hLW*W2` | 0.192748 | hLW × [num_warps = 2] |
| `hLW*W4` | -0.378283 | hLW × [num_warps = 4] |
| `hLW*W8` | -1.74539 | hLW × [num_warps = 8] |
| `hLW*W16` | -1.46988 | hLW × [num_warps = 16] |
| `hLW*W32` | -1.26992 | hLW × [num_warps = 32] |
| `hLW*W64` | -1.11121 | hLW × [num_warps = 64] |
| `hLW*Clt8` | -0.019065 | hLW × [C < 8] |
| `hLW*Clt16` | -0.308497 | hLW × [C < 16] |
| `hLW*Clt32` | 1.32615 | hLW × [C < 32] |
| `hLW*Clt64` | -0.834377 | hLW × [C < 64] |
| `hLW*nlt64` | -0.834377 | hLW × [n < 64] |
| `hLW*nge128` | 0.549353 | hLW × [n >= 128] |
| `hLW*nge256` | -0.433224 | hLW × [n >= 256] |
| `hLW*Lgt4` | -1.58501 | hLW × [L > 4] |
| `hLW*Lgt16` | -0.792126 | hLW × [L > 16] |
| `hLW*Lgt64` | -0.689857 | hLW × [L > 64] |
| `hLW*mwlg2` | 1.54867 | hLW × [mean_warp_lines > 2] |
| `hLW*mwlg8` | 0.355083 | hLW × [mean_warp_lines > 8] |
| `hLW*smISB` | 0.193909 | hLW × [inner stride <= 8 bytes] |
| `ov_L_mwl*const` | 0.101576 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [无（常数项）] |
| `ov_L_mwl*eb1` | 1.18786 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = int8] |
| `ov_L_mwl*eb2` | 0.06497 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = float16] |
| `ov_L_mwl*eb4` | -0.769796 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = float32] |
| `ov_L_mwl*eb8` | -0.381461 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [dtype = int64] |
| `ov_L_mwl*W1` | -0.978275 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 1] |
| `ov_L_mwl*W2` | -0.82251 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 2] |
| `ov_L_mwl*W4` | -0.335484 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 4] |
| `ov_L_mwl*W8` | 0.122403 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 8] |
| `ov_L_mwl*W16` | 0.251506 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 16] |
| `ov_L_mwl*W32` | 1.26728 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 32] |
| `ov_L_mwl*W64` | 0.596659 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [num_warps = 64] |
| `ov_L_mwl*nge128` | -7.46516 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [n >= 128] |
| `ov_L_mwl*nge256` | 4.48107 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [n >= 256] |
| `ov_L_mwl*Lgt4` | 0.686031 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [L > 4] |
| `ov_L_mwl*Lgt16` | 0.105174 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [L > 16] |
| `ov_L_mwl*Lgt64` | 0.788523 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [L > 64] |
| `ov_L_mwl*mwlg2` | 0.712144 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [mean_warp_lines > 2] |
| `ov_L_mwl*mwlg8` | 0.331898 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [mean_warp_lines > 8] |
| `ov_L_mwl*smISB` | -0.610568 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [inner stride <= 8 bytes] |
| `hLW05*const` | 0.362127 | max(0, L − 0.5W) × [无（常数项）] |
| `hLW05*eb1` | 1.40902 | max(0, L − 0.5W) × [dtype = int8] |
| `hLW05*eb2` | 1.35839 | max(0, L − 0.5W) × [dtype = float16] |
| `hLW05*eb4` | -1.03616 | max(0, L − 0.5W) × [dtype = float32] |
| `hLW05*eb8` | -1.36912 | max(0, L − 0.5W) × [dtype = int64] |
| `hLW05*W1` | 0.600014 | max(0, L − 0.5W) × [num_warps = 1] |
| `hLW05*W2` | -0.212811 | max(0, L − 0.5W) × [num_warps = 2] |
| `hLW05*W4` | 0.364778 | max(0, L − 0.5W) × [num_warps = 4] |
| `hLW05*W8` | 0.936424 | max(0, L − 0.5W) × [num_warps = 8] |
| `hLW05*W16` | 0.127476 | max(0, L − 0.5W) × [num_warps = 16] |
| `hLW05*W32` | -0.902074 | max(0, L − 0.5W) × [num_warps = 32] |
| `hLW05*W64` | -0.55168 | max(0, L − 0.5W) × [num_warps = 64] |
| `hLW05*Clt8` | 0.066254 | max(0, L − 0.5W) × [C < 8] |
| `hLW05*Clt16` | 0.74536 | max(0, L − 0.5W) × [C < 16] |
| `hLW05*Clt32` | -1.30833 | max(0, L − 0.5W) × [C < 32] |
| `hLW05*Clt64` | 0.623107 | max(0, L − 0.5W) × [C < 64] |
| `hLW05*nlt64` | 0.623107 | max(0, L − 0.5W) × [n < 64] |
| `hLW05*nge128` | -2.17491 | max(0, L − 0.5W) × [n >= 128] |
| `hLW05*nge256` | 1.86847 | max(0, L − 0.5W) × [n >= 256] |
| `hLW05*Lgt4` | 2.6263 | max(0, L − 0.5W) × [L > 4] |
| `hLW05*Lgt16` | 0.398366 | max(0, L − 0.5W) × [L > 16] |
| `hLW05*Lgt64` | 0.532755 | max(0, L − 0.5W) × [L > 64] |
| `hLW05*mwlg2` | -2.10229 | max(0, L − 0.5W) × [mean_warp_lines > 2] |
| `hLW05*mwlg8` | -1.24357 | max(0, L − 0.5W) × [mean_warp_lines > 8] |
| `hLW05*smISB` | -1.35676 | max(0, L − 0.5W) × [inner stride <= 8 bytes] |

### 3. in-sample 拟合误差

- 点数：**962**
- 项数：**217**
- **MAPE：5.10%**

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
T = 16.1618
  + 0.64301 * mx_L_W*const
  + 0.164684 * mx_L_W*eb1
  + -0.296265 * mx_L_W*eb2
  + 0.343527 * mx_L_W*eb4
  + 0.431064 * mx_L_W*eb8
  + -0.159976 * mx_L_W*W1
  + 0.127501 * mx_L_W*W2
  + -0.137294 * mx_L_W*W4
  + -0.048588 * mx_L_W*W8
  + 0.060786 * mx_L_W*W16
  + 0.326966 * mx_L_W*W32
  + 0.473614 * mx_L_W*W64
  + 0.39405 * mx_L_W*Clt8
  + 0.147499 * mx_L_W*Clt16
  + -0.122745 * mx_L_W*Clt32
  + 0.093306 * mx_L_W*Clt64
  + -0.337993 * mx_L_W*nlt64
  + -0.172162 * mx_L_W*nge128
  + 0.234798 * mx_L_W*nge256
  + -0.422268 * mx_L_W*Rgt8
  + 0.683933 * mx_L_W*Rgt16
  + -0.52668 * mx_L_W*Lgt4
  + 0.448864 * mx_L_W*Lgt16
  + -6.65369 * mx_L_W*Lgt64
  + -0.334105 * mx_L_W*mwlg2
  + 0.884954 * mx_L_W*mwlg8
  + -0.148375 * mx_L_W*smISB
  + -0.006847 * bucket_pairs_32k16*const
  + -0.001327 * bucket_pairs_32k16*eb1
  + -0.000826 * bucket_pairs_32k16*eb2
  + -0.002457 * bucket_pairs_32k16*eb4
  + -0.002238 * bucket_pairs_32k16*eb8
  + -0.000276 * bucket_pairs_32k16*W1
  + -0.000915 * bucket_pairs_32k16*W2
  + -0.00105 * bucket_pairs_32k16*W4
  + -0.000764 * bucket_pairs_32k16*W8
  + -0.001272 * bucket_pairs_32k16*W16
  + -0.000263 * bucket_pairs_32k16*W32
  + -0.002307 * bucket_pairs_32k16*W64
  + -0.001073 * bucket_pairs_32k16*Clt8
  + -0.000303 * bucket_pairs_32k16*Clt16
  + 0.002024 * bucket_pairs_32k16*Clt32
  + 0.000783 * bucket_pairs_32k16*Clt64
  + -0.008557 * bucket_pairs_32k16*nlt64
  + 0.003289 * bucket_pairs_32k16*nge128
  + 0.005622 * bucket_pairs_32k16*nge256
  + 0.001905 * bucket_pairs_32k16*Rgt8
  + 0.0021 * bucket_pairs_32k16*Rgt16
  + -0.000244 * bucket_pairs_32k16*Lgt4
  + 0.000477 * bucket_pairs_32k16*Lgt16
  + -0.000925 * bucket_pairs_32k16*Lgt64
  + 0.001015 * bucket_pairs_32k16*mwlg2
  + -0.00018 * bucket_pairs_32k16*mwlg8
  + 0.001405 * bucket_pairs_32k16*smISB
  + 1.48023 * hLeb2*const
  + -6.0826 * hLeb2*eb1
  + 5.7064 * hLeb2*eb2
  + 1.22807 * hLeb2*eb4
  + 0.628367 * hLeb2*eb8
  + 0.353791 * hLeb2*W1
  + 1.02297 * hLeb2*W2
  + 0.295563 * hLeb2*W4
  + -0.385228 * hLeb2*W8
  + -0.253205 * hLeb2*W16
  + -0.218164 * hLeb2*W32
  + 0.664505 * hLeb2*W64
  + -2.19601 * hLeb2*Clt8
  + 2.33387 * hLeb2*Clt16
  + -1.4448 * hLeb2*Clt32
  + 3.86631 * hLeb2*Clt64
  + 0.256394 * hLeb2*nlt64
  + 2.97346 * hLeb2*nge128
  + -34.2644 * hLeb2*nge256
  + 3.13341 * hLeb2*Rgt8
  + -3.09078 * hLeb2*Rgt16
  + -6.86659 * hLeb2*Lgt4
  + -8.93437 * hLeb2*Lgt16
  + 23.5111 * hLeb2*Lgt64
  + -1.166 * hLeb2*mwlg2
  + -2.52467 * hLeb2*mwlg8
  + 2.23936 * hLeb2*smISB
  + 0.001721 * p_n_T*const
  + 0.000563 * p_n_T*eb1
  + 0.000658 * p_n_T*eb2
  + 0.000133 * p_n_T*eb4
  + 0.000367 * p_n_T*eb8
  + 0.009267 * p_n_T*W1
  + 0.004233 * p_n_T*W2
  + 0.000154 * p_n_T*W4
  + -0.001907 * p_n_T*W8
  + -0.002835 * p_n_T*W16
  + -0.00356 * p_n_T*W32
  + -0.00363 * p_n_T*W64
  + -0.000109 * p_n_T*Clt8
  + -1.9e-05 * p_n_T*Clt16
  + 0.00074 * p_n_T*Clt32
  + 0.00077 * p_n_T*Clt64
  + 0.000634 * p_n_T*nlt64
  + 5.2e-05 * p_n_T*nge128
  + -7.2e-05 * p_n_T*nge256
  + 0.000282 * p_n_T*Rgt8
  + -0.000576 * p_n_T*Rgt16
  + 0.000185 * p_n_T*Lgt4
  + -0.00028 * p_n_T*Lgt16
  + 0.000382 * p_n_T*Lgt64
  + 0.000243 * p_n_T*mwlg2
  + -0.00019 * p_n_T*mwlg8
  + 0.000121 * p_n_T*smISB
  + 2.48642 * hLeb4*const
  + 9.28398 * hLeb4*eb1
  + 1.57089 * hLeb4*eb2
  + -6.14667 * hLeb4*eb4
  + -2.22177 * hLeb4*eb8
  + 0.471923 * hLeb4*W1
  + 0.007588 * hLeb4*W2
  + 0.409 * hLeb4*W4
  + 1.02397 * hLeb4*W8
  + 0.407394 * hLeb4*W16
  + 0.115059 * hLeb4*W32
  + 0.051485 * hLeb4*W64
  + 0.992002 * hLeb4*Clt8
  + 0.043386 * hLeb4*Clt16
  + 1.85471 * hLeb4*Clt32
  + -1.80075 * hLeb4*Clt64
  + -1.35067 * hLeb4*nlt64
  + 0.114694 * hLeb4*nge128
  + 36.5091 * hLeb4*nge256
  + -1.14388 * hLeb4*Rgt8
  + 2.52999 * hLeb4*Rgt16
  + 2.48642 * hLeb4*Lgt4
  + 5.31829 * hLeb4*Lgt16
  + -26.2633 * hLeb4*Lgt64
  + -1.10267 * hLeb4*mwlg2
  + 3.41845 * hLeb4*mwlg8
  + -0.087702 * hLeb4*smISB
  + 0.725495 * mx_L_cshortN*const
  + 0.321092 * mx_L_cshortN*eb1
  + 0.388896 * mx_L_cshortN*eb2
  + 0.081556 * mx_L_cshortN*eb4
  + -0.066049 * mx_L_cshortN*eb8
  + 0.314359 * mx_L_cshortN*W1
  + 0.195688 * mx_L_cshortN*W2
  + 0.321342 * mx_L_cshortN*W4
  + 0.298579 * mx_L_cshortN*W8
  + 0.159596 * mx_L_cshortN*W16
  + -0.003804 * mx_L_cshortN*W32
  + -0.560265 * mx_L_cshortN*W64
  + -0.075507 * mx_L_cshortN*Clt8
  + -0.157573 * mx_L_cshortN*Clt16
  + -0.18115 * mx_L_cshortN*Clt32
  + -0.91559 * mx_L_cshortN*Clt64
  + 0.14104 * mx_L_cshortN*nlt64
  + -0.192876 * mx_L_cshortN*nge128
  + -0.371235 * mx_L_cshortN*nge256
  + 0.024579 * mx_L_cshortN*Rgt8
  + -0.264906 * mx_L_cshortN*Rgt16
  + 0.209648 * mx_L_cshortN*Lgt4
  + 0.803077 * mx_L_cshortN*Lgt16
  + 6.53741 * mx_L_cshortN*Lgt64
  + 0.033093 * mx_L_cshortN*mwlg2
  + -0.448838 * mx_L_cshortN*mwlg8
  + -0.176984 * mx_L_cshortN*smISB
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 16.1618 | 固定开销（循环发射、启动） |
| `mx_L_W*const` | 0.64301 | max(触及的 128B line 数, num_warps) × [无（常数项）] |
| `mx_L_W*eb1` | 0.164684 | max(触及的 128B line 数, num_warps) × [dtype = int8] |
| `mx_L_W*eb2` | -0.296265 | max(触及的 128B line 数, num_warps) × [dtype = float16] |
| `mx_L_W*eb4` | 0.343527 | max(触及的 128B line 数, num_warps) × [dtype = float32] |
| `mx_L_W*eb8` | 0.431064 | max(触及的 128B line 数, num_warps) × [dtype = int64] |
| `mx_L_W*W1` | -0.159976 | max(触及的 128B line 数, num_warps) × [num_warps = 1] |
| `mx_L_W*W2` | 0.127501 | max(触及的 128B line 数, num_warps) × [num_warps = 2] |
| `mx_L_W*W4` | -0.137294 | max(触及的 128B line 数, num_warps) × [num_warps = 4] |
| `mx_L_W*W8` | -0.048588 | max(触及的 128B line 数, num_warps) × [num_warps = 8] |
| `mx_L_W*W16` | 0.060786 | max(触及的 128B line 数, num_warps) × [num_warps = 16] |
| `mx_L_W*W32` | 0.326966 | max(触及的 128B line 数, num_warps) × [num_warps = 32] |
| `mx_L_W*W64` | 0.473614 | max(触及的 128B line 数, num_warps) × [num_warps = 64] |
| `mx_L_W*Clt8` | 0.39405 | max(触及的 128B line 数, num_warps) × [C < 8] |
| `mx_L_W*Clt16` | 0.147499 | max(触及的 128B line 数, num_warps) × [C < 16] |
| `mx_L_W*Clt32` | -0.122745 | max(触及的 128B line 数, num_warps) × [C < 32] |
| `mx_L_W*Clt64` | 0.093306 | max(触及的 128B line 数, num_warps) × [C < 64] |
| `mx_L_W*nlt64` | -0.337993 | max(触及的 128B line 数, num_warps) × [n < 64] |
| `mx_L_W*nge128` | -0.172162 | max(触及的 128B line 数, num_warps) × [n >= 128] |
| `mx_L_W*nge256` | 0.234798 | max(触及的 128B line 数, num_warps) × [n >= 256] |
| `mx_L_W*Rgt8` | -0.422268 | max(触及的 128B line 数, num_warps) × [R > 8] |
| `mx_L_W*Rgt16` | 0.683933 | max(触及的 128B line 数, num_warps) × [R > 16] |
| `mx_L_W*Lgt4` | -0.52668 | max(触及的 128B line 数, num_warps) × [L > 4] |
| `mx_L_W*Lgt16` | 0.448864 | max(触及的 128B line 数, num_warps) × [L > 16] |
| `mx_L_W*Lgt64` | -6.65369 | max(触及的 128B line 数, num_warps) × [L > 64] |
| `mx_L_W*mwlg2` | -0.334105 | max(触及的 128B line 数, num_warps) × [mean_warp_lines > 2] |
| `mx_L_W*mwlg8` | 0.884954 | max(触及的 128B line 数, num_warps) × [mean_warp_lines > 8] |
| `mx_L_W*smISB` | -0.148375 | max(触及的 128B line 数, num_warps) × [inner stride <= 8 bytes] |
| `bucket_pairs_32k16*const` | -0.006847 | line 桶 (32768..16384) 对数 × [无（常数项）] |
| `bucket_pairs_32k16*eb1` | -0.001327 | line 桶 (32768..16384) 对数 × [dtype = int8] |
| `bucket_pairs_32k16*eb2` | -0.000826 | line 桶 (32768..16384) 对数 × [dtype = float16] |
| `bucket_pairs_32k16*eb4` | -0.002457 | line 桶 (32768..16384) 对数 × [dtype = float32] |
| `bucket_pairs_32k16*eb8` | -0.002238 | line 桶 (32768..16384) 对数 × [dtype = int64] |
| `bucket_pairs_32k16*W1` | -0.000276 | line 桶 (32768..16384) 对数 × [num_warps = 1] |
| `bucket_pairs_32k16*W2` | -0.000915 | line 桶 (32768..16384) 对数 × [num_warps = 2] |
| `bucket_pairs_32k16*W4` | -0.00105 | line 桶 (32768..16384) 对数 × [num_warps = 4] |
| `bucket_pairs_32k16*W8` | -0.000764 | line 桶 (32768..16384) 对数 × [num_warps = 8] |
| `bucket_pairs_32k16*W16` | -0.001272 | line 桶 (32768..16384) 对数 × [num_warps = 16] |
| `bucket_pairs_32k16*W32` | -0.000263 | line 桶 (32768..16384) 对数 × [num_warps = 32] |
| `bucket_pairs_32k16*W64` | -0.002307 | line 桶 (32768..16384) 对数 × [num_warps = 64] |
| `bucket_pairs_32k16*Clt8` | -0.001073 | line 桶 (32768..16384) 对数 × [C < 8] |
| `bucket_pairs_32k16*Clt16` | -0.000303 | line 桶 (32768..16384) 对数 × [C < 16] |
| `bucket_pairs_32k16*Clt32` | 0.002024 | line 桶 (32768..16384) 对数 × [C < 32] |
| `bucket_pairs_32k16*Clt64` | 0.000783 | line 桶 (32768..16384) 对数 × [C < 64] |
| `bucket_pairs_32k16*nlt64` | -0.008557 | line 桶 (32768..16384) 对数 × [n < 64] |
| `bucket_pairs_32k16*nge128` | 0.003289 | line 桶 (32768..16384) 对数 × [n >= 128] |
| `bucket_pairs_32k16*nge256` | 0.005622 | line 桶 (32768..16384) 对数 × [n >= 256] |
| `bucket_pairs_32k16*Rgt8` | 0.001905 | line 桶 (32768..16384) 对数 × [R > 8] |
| `bucket_pairs_32k16*Rgt16` | 0.0021 | line 桶 (32768..16384) 对数 × [R > 16] |
| `bucket_pairs_32k16*Lgt4` | -0.000244 | line 桶 (32768..16384) 对数 × [L > 4] |
| `bucket_pairs_32k16*Lgt16` | 0.000477 | line 桶 (32768..16384) 对数 × [L > 16] |
| `bucket_pairs_32k16*Lgt64` | -0.000925 | line 桶 (32768..16384) 对数 × [L > 64] |
| `bucket_pairs_32k16*mwlg2` | 0.001015 | line 桶 (32768..16384) 对数 × [mean_warp_lines > 2] |
| `bucket_pairs_32k16*mwlg8` | -0.00018 | line 桶 (32768..16384) 对数 × [mean_warp_lines > 8] |
| `bucket_pairs_32k16*smISB` | 0.001405 | line 桶 (32768..16384) 对数 × [inner stride <= 8 bytes] |
| `hLeb2*const` | 1.48023 | max(0, L/eb − 2) × [无（常数项）] |
| `hLeb2*eb1` | -6.0826 | max(0, L/eb − 2) × [dtype = int8] |
| `hLeb2*eb2` | 5.7064 | max(0, L/eb − 2) × [dtype = float16] |
| `hLeb2*eb4` | 1.22807 | max(0, L/eb − 2) × [dtype = float32] |
| `hLeb2*eb8` | 0.628367 | max(0, L/eb − 2) × [dtype = int64] |
| `hLeb2*W1` | 0.353791 | max(0, L/eb − 2) × [num_warps = 1] |
| `hLeb2*W2` | 1.02297 | max(0, L/eb − 2) × [num_warps = 2] |
| `hLeb2*W4` | 0.295563 | max(0, L/eb − 2) × [num_warps = 4] |
| `hLeb2*W8` | -0.385228 | max(0, L/eb − 2) × [num_warps = 8] |
| `hLeb2*W16` | -0.253205 | max(0, L/eb − 2) × [num_warps = 16] |
| `hLeb2*W32` | -0.218164 | max(0, L/eb − 2) × [num_warps = 32] |
| `hLeb2*W64` | 0.664505 | max(0, L/eb − 2) × [num_warps = 64] |
| `hLeb2*Clt8` | -2.19601 | max(0, L/eb − 2) × [C < 8] |
| `hLeb2*Clt16` | 2.33387 | max(0, L/eb − 2) × [C < 16] |
| `hLeb2*Clt32` | -1.4448 | max(0, L/eb − 2) × [C < 32] |
| `hLeb2*Clt64` | 3.86631 | max(0, L/eb − 2) × [C < 64] |
| `hLeb2*nlt64` | 0.256394 | max(0, L/eb − 2) × [n < 64] |
| `hLeb2*nge128` | 2.97346 | max(0, L/eb − 2) × [n >= 128] |
| `hLeb2*nge256` | -34.2644 | max(0, L/eb − 2) × [n >= 256] |
| `hLeb2*Rgt8` | 3.13341 | max(0, L/eb − 2) × [R > 8] |
| `hLeb2*Rgt16` | -3.09078 | max(0, L/eb − 2) × [R > 16] |
| `hLeb2*Lgt4` | -6.86659 | max(0, L/eb − 2) × [L > 4] |
| `hLeb2*Lgt16` | -8.93437 | max(0, L/eb − 2) × [L > 16] |
| `hLeb2*Lgt64` | 23.5111 | max(0, L/eb − 2) × [L > 64] |
| `hLeb2*mwlg2` | -1.166 | max(0, L/eb − 2) × [mean_warp_lines > 2] |
| `hLeb2*mwlg8` | -2.52467 | max(0, L/eb − 2) × [mean_warp_lines > 8] |
| `hLeb2*smISB` | 2.23936 | max(0, L/eb − 2) × [inner stride <= 8 bytes] |
| `p_n_T*const` | 0.001721 | p_n_T × [无（常数项）] |
| `p_n_T*eb1` | 0.000563 | p_n_T × [dtype = int8] |
| `p_n_T*eb2` | 0.000658 | p_n_T × [dtype = float16] |
| `p_n_T*eb4` | 0.000133 | p_n_T × [dtype = float32] |
| `p_n_T*eb8` | 0.000367 | p_n_T × [dtype = int64] |
| `p_n_T*W1` | 0.009267 | p_n_T × [num_warps = 1] |
| `p_n_T*W2` | 0.004233 | p_n_T × [num_warps = 2] |
| `p_n_T*W4` | 0.000154 | p_n_T × [num_warps = 4] |
| `p_n_T*W8` | -0.001907 | p_n_T × [num_warps = 8] |
| `p_n_T*W16` | -0.002835 | p_n_T × [num_warps = 16] |
| `p_n_T*W32` | -0.00356 | p_n_T × [num_warps = 32] |
| `p_n_T*W64` | -0.00363 | p_n_T × [num_warps = 64] |
| `p_n_T*Clt8` | -0.000109 | p_n_T × [C < 8] |
| `p_n_T*Clt16` | -1.9e-05 | p_n_T × [C < 16] |
| `p_n_T*Clt32` | 0.00074 | p_n_T × [C < 32] |
| `p_n_T*Clt64` | 0.00077 | p_n_T × [C < 64] |
| `p_n_T*nlt64` | 0.000634 | p_n_T × [n < 64] |
| `p_n_T*nge128` | 5.2e-05 | p_n_T × [n >= 128] |
| `p_n_T*nge256` | -7.2e-05 | p_n_T × [n >= 256] |
| `p_n_T*Rgt8` | 0.000282 | p_n_T × [R > 8] |
| `p_n_T*Rgt16` | -0.000576 | p_n_T × [R > 16] |
| `p_n_T*Lgt4` | 0.000185 | p_n_T × [L > 4] |
| `p_n_T*Lgt16` | -0.00028 | p_n_T × [L > 16] |
| `p_n_T*Lgt64` | 0.000382 | p_n_T × [L > 64] |
| `p_n_T*mwlg2` | 0.000243 | p_n_T × [mean_warp_lines > 2] |
| `p_n_T*mwlg8` | -0.00019 | p_n_T × [mean_warp_lines > 8] |
| `p_n_T*smISB` | 0.000121 | p_n_T × [inner stride <= 8 bytes] |
| `hLeb4*const` | 2.48642 | max(0, L/eb − 4) × [无（常数项）] |
| `hLeb4*eb1` | 9.28398 | max(0, L/eb − 4) × [dtype = int8] |
| `hLeb4*eb2` | 1.57089 | max(0, L/eb − 4) × [dtype = float16] |
| `hLeb4*eb4` | -6.14667 | max(0, L/eb − 4) × [dtype = float32] |
| `hLeb4*eb8` | -2.22177 | max(0, L/eb − 4) × [dtype = int64] |
| `hLeb4*W1` | 0.471923 | max(0, L/eb − 4) × [num_warps = 1] |
| `hLeb4*W2` | 0.007588 | max(0, L/eb − 4) × [num_warps = 2] |
| `hLeb4*W4` | 0.409 | max(0, L/eb − 4) × [num_warps = 4] |
| `hLeb4*W8` | 1.02397 | max(0, L/eb − 4) × [num_warps = 8] |
| `hLeb4*W16` | 0.407394 | max(0, L/eb − 4) × [num_warps = 16] |
| `hLeb4*W32` | 0.115059 | max(0, L/eb − 4) × [num_warps = 32] |
| `hLeb4*W64` | 0.051485 | max(0, L/eb − 4) × [num_warps = 64] |
| `hLeb4*Clt8` | 0.992002 | max(0, L/eb − 4) × [C < 8] |
| `hLeb4*Clt16` | 0.043386 | max(0, L/eb − 4) × [C < 16] |
| `hLeb4*Clt32` | 1.85471 | max(0, L/eb − 4) × [C < 32] |
| `hLeb4*Clt64` | -1.80075 | max(0, L/eb − 4) × [C < 64] |
| `hLeb4*nlt64` | -1.35067 | max(0, L/eb − 4) × [n < 64] |
| `hLeb4*nge128` | 0.114694 | max(0, L/eb − 4) × [n >= 128] |
| `hLeb4*nge256` | 36.5091 | max(0, L/eb − 4) × [n >= 256] |
| `hLeb4*Rgt8` | -1.14388 | max(0, L/eb − 4) × [R > 8] |
| `hLeb4*Rgt16` | 2.52999 | max(0, L/eb − 4) × [R > 16] |
| `hLeb4*Lgt4` | 2.48642 | max(0, L/eb − 4) × [L > 4] |
| `hLeb4*Lgt16` | 5.31829 | max(0, L/eb − 4) × [L > 16] |
| `hLeb4*Lgt64` | -26.2633 | max(0, L/eb − 4) × [L > 64] |
| `hLeb4*mwlg2` | -1.10267 | max(0, L/eb − 4) × [mean_warp_lines > 2] |
| `hLeb4*mwlg8` | 3.41845 | max(0, L/eb − 4) × [mean_warp_lines > 8] |
| `hLeb4*smISB` | -0.087702 | max(0, L/eb − 4) × [inner stride <= 8 bytes] |
| `mx_L_cshortN*const` | 0.725495 | max(触及的 128B line 数, cshort · n / 32) × [无（常数项）] |
| `mx_L_cshortN*eb1` | 0.321092 | max(触及的 128B line 数, cshort · n / 32) × [dtype = int8] |
| `mx_L_cshortN*eb2` | 0.388896 | max(触及的 128B line 数, cshort · n / 32) × [dtype = float16] |
| `mx_L_cshortN*eb4` | 0.081556 | max(触及的 128B line 数, cshort · n / 32) × [dtype = float32] |
| `mx_L_cshortN*eb8` | -0.066049 | max(触及的 128B line 数, cshort · n / 32) × [dtype = int64] |
| `mx_L_cshortN*W1` | 0.314359 | max(触及的 128B line 数, cshort · n / 32) × [num_warps = 1] |
| `mx_L_cshortN*W2` | 0.195688 | max(触及的 128B line 数, cshort · n / 32) × [num_warps = 2] |
| `mx_L_cshortN*W4` | 0.321342 | max(触及的 128B line 数, cshort · n / 32) × [num_warps = 4] |
| `mx_L_cshortN*W8` | 0.298579 | max(触及的 128B line 数, cshort · n / 32) × [num_warps = 8] |
| `mx_L_cshortN*W16` | 0.159596 | max(触及的 128B line 数, cshort · n / 32) × [num_warps = 16] |
| `mx_L_cshortN*W32` | -0.003804 | max(触及的 128B line 数, cshort · n / 32) × [num_warps = 32] |
| `mx_L_cshortN*W64` | -0.560265 | max(触及的 128B line 数, cshort · n / 32) × [num_warps = 64] |
| `mx_L_cshortN*Clt8` | -0.075507 | max(触及的 128B line 数, cshort · n / 32) × [C < 8] |
| `mx_L_cshortN*Clt16` | -0.157573 | max(触及的 128B line 数, cshort · n / 32) × [C < 16] |
| `mx_L_cshortN*Clt32` | -0.18115 | max(触及的 128B line 数, cshort · n / 32) × [C < 32] |
| `mx_L_cshortN*Clt64` | -0.91559 | max(触及的 128B line 数, cshort · n / 32) × [C < 64] |
| `mx_L_cshortN*nlt64` | 0.14104 | max(触及的 128B line 数, cshort · n / 32) × [n < 64] |
| `mx_L_cshortN*nge128` | -0.192876 | max(触及的 128B line 数, cshort · n / 32) × [n >= 128] |
| `mx_L_cshortN*nge256` | -0.371235 | max(触及的 128B line 数, cshort · n / 32) × [n >= 256] |
| `mx_L_cshortN*Rgt8` | 0.024579 | max(触及的 128B line 数, cshort · n / 32) × [R > 8] |
| `mx_L_cshortN*Rgt16` | -0.264906 | max(触及的 128B line 数, cshort · n / 32) × [R > 16] |
| `mx_L_cshortN*Lgt4` | 0.209648 | max(触及的 128B line 数, cshort · n / 32) × [L > 4] |
| `mx_L_cshortN*Lgt16` | 0.803077 | max(触及的 128B line 数, cshort · n / 32) × [L > 16] |
| `mx_L_cshortN*Lgt64` | 6.53741 | max(触及的 128B line 数, cshort · n / 32) × [L > 64] |
| `mx_L_cshortN*mwlg2` | 0.033093 | max(触及的 128B line 数, cshort · n / 32) × [mean_warp_lines > 2] |
| `mx_L_cshortN*mwlg8` | -0.448838 | max(触及的 128B line 数, cshort · n / 32) × [mean_warp_lines > 8] |
| `mx_L_cshortN*smISB` | -0.176984 | max(触及的 128B line 数, cshort · n / 32) × [inner stride <= 8 bytes] |

### 3. in-sample 拟合误差

- 点数：**3416**
- 项数：**163**
- **MAPE：15.17%**

---
