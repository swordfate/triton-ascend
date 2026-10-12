# 版本 3：10 CROSSED —— 每公式 <=10 项（交叉项）

每个公式最多 10 项，但每一项可以是 `基特征 × 修饰因子` 的乘积，从完整交叉空间中贪心选取。不使用任何数据剔除。

## 汇总表

| 公式 | 场景 | 路径/操作/秩 | 点数 | 项数 | **in-sample MAPE** |
|---|---|---|---:|---:|---:|
| `simd_load_wide_r1` | SIMD load 宽分支（1D） | simd / load / rank-1 | 32 | 9 | **1.31%** |
| `simd_load_wide_r2` | SIMD load 宽分支（2D） | simd / load / rank-2 | 246 | 11 | **16.69%** |
| `simd_load_gather_r1` | SIMD load 聚集分支（1D） | simd / load / rank-1 | 284 | 11 | **18.81%** |
| `simd_load_gather_r2` | SIMD load 聚集分支（2D） | simd / load / rank-2 | 750 | 11 | **15.19%** |
| `simd_store_r1` | SIMD store（1D） | simd / store / rank-1 | 291 | 11 | **8.01%** |
| `simd_store_r2` | SIMD store（2D） | simd / store / rank-2 | 943 | 11 | **10.54%** |
| `simt_load_r1` | 纯 SIMT load（1D） | simt_only / load / rank-1 | 855 | 11 | **10.91%** |
| `simt_load_r2` | 纯 SIMT load（2D） | simt_only / load / rank-2 | 1762 | 11 | **10.06%** |
| `simt_store_r1` | 纯 SIMT store（1D） | simt_only / store / rank-1 | 962 | 11 | **15.33%** |
| `simt_store_r2` | 纯 SIMT store（2D） | simt_only / store / rank-2 | 3416 | 11 | **24.33%** |
| | | | **9541** | | **16.57%** |

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
T = 65.9123
  + -161.02 * ov_L_pages*Lgt4
  + -3.87485 * minL32*Lgt4
  + -9.95393 * hLW05*Lgt4
  + 0.776875 * rmw*
  + -3.84735 * ov_L_mwl*Lgt4
  + 0.280074 * hL2*nge128
  + 0.042311 * p_T_rs*Lgt4
  + 0.042684 * p_ISB_it*
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 65.9123 | 固定开销（循环发射、启动） |
| `ov_L_pages*Lgt4` | -161.02 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [L > 4] |
| `minL32*Lgt4` | -3.87485 | min(L,32) × [L > 4] |
| `hLW05*Lgt4` | -9.95393 | max(0, L − 0.5W) × [L > 4] |
| `rmw*` | 0.776875 | read-modify-write 比例 × [] |
| `ov_L_mwl*Lgt4` | -3.84735 | max(0, 触及的 128B line 数 − mean_warp_lines：每 warp 波次触及的 line 数) × [L > 4] |
| `hL2*nge128` | 0.280074 | max(0, L−2) × [n >= 128] |
| `p_T_rs*Lgt4` | 0.042311 | p_T_rs × [L > 4] |
| `p_ISB_it*` | 0.042684 | p_ISB_it × [] |

### 3. in-sample 拟合误差

- 点数：**32**
- 项数：**9**
- **MAPE：1.31%**

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
T = 67.5872
  + -0.284298 * comblines*Clt64
  + 13.3953 * rt_cshortN_L*Clt8
  + 0.500342 * p_R_it*e8
  + 3.59621 * rowseg*Clt64
  + -8.57739 * ov_seg_W*smISB
  + 7.14761 * rowseg*
  + 1.5354 * bsSeg*
  + -14.9503 * line_elems_min*Clt8
  + -15.3414 * r_L_rs*Clt64
  + 9.82679 * r_n_mwl*Clt8
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 67.5872 | 固定开销（循环发射、启动） |
| `comblines*Clt64` | -0.284298 | R × 每行 line 数 × [C < 64] |
| `rt_cshortN_L*Clt8` | 13.3953 | cshort · n / 32 / 触及的 128B line 数 × [C < 8] |
| `p_R_it*e8` | 0.500342 | p_R_it × [dtype = int64] |
| `rowseg*Clt64` | 3.59621 | 行段数 R（rank-2），rank-1 为 1 × [C < 64] |
| `ov_seg_W*smISB` | -8.57739 | ov_seg_W × [inner stride <= 8 bytes] |
| `rowseg*` | 7.14761 | 行段数 R（rank-2），rank-1 为 1 × [] |
| `bsSeg*` | 1.5354 | [C·eb<32] · R × [] |
| `line_elems_min*Clt8` | -14.9503 | 单 line 内最少元素数 × [C < 8] |
| `r_L_rs*Clt64` | -15.3414 | r_L_rs × [C < 64] |
| `r_n_mwl*Clt8` | 9.82679 | 元素数 = BIU 命令数 / mean_warp_lines：每 warp 波次触及的 line 数 × [C < 8] |

### 3. in-sample 拟合误差

- 点数：**246**
- 项数：**11**
- **MAPE：16.69%**

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
T = 193.333
  + 0.383126 * n2L*
  + 2.53846 * ov_n_mwl*
  + -0.122638 * nEpl1*nge128
  + -2.60206 * p_T_it*
  + -3.90671 * neplW*
  + -1.00885 * bucket_worst_2k8*
  + 1.354 * hnW80*Lgt4
  + -0.50608 * hnW40*mwlg2
  + 0.015912 * nEpl_eb*
  + -0.255615 * bucket_worst_32k16*
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 193.333 | 固定开销（循环发射、启动） |
| `n2L*` | 0.383126 | n² / L × [] |
| `ov_n_mwl*` | 2.53846 | max(0, 元素数 = BIU 命令数 − mean_warp_lines：每 warp 波次触及的 line 数) × [] |
| `nEpl1*nge128` | -0.122638 | n · floor(128/ISB) × [n >= 128] |
| `p_T_it*` | -2.60206 | p_T_it × [] |
| `neplW*` | -3.90671 | n / epl / W × [] |
| `bucket_worst_2k8*` | -1.00885 | 最差 line 桶 (2048..8192) × [] |
| `hnW80*Lgt4` | 1.354 | hnW80 × [L > 4] |
| `hnW40*mwlg2` | -0.50608 | hnW40 × [mean_warp_lines > 2] |
| `nEpl_eb*` | 0.015912 | n·epl·eb × [] |
| `bucket_worst_32k16*` | -0.255615 | 最差 line 桶 (32768..16384) × [] |

### 3. in-sample 拟合误差

- 点数：**284**
- 项数：**11**
- **MAPE：18.81%**

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
T = 110.362
  + 127.24 * mn_cshortN_L*
  + 0.188431 * n2L*
  + -0.044878 * nhe4*Clt64
  + 0.023742 * nEpl_eb*
  + -0.073647 * n2L*nge128
  + 11.0319 * mn_cshortN_L*Lgt4
  + 0.681232 * r_n_eb*
  + -19.9494 * mn_cshortN_L*smISB
  + -39.4918 * bsL*
  + -0.046555 * hepl_n*Clt64
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 110.362 | 固定开销（循环发射、启动） |
| `mn_cshortN_L*` | 127.24 | min(cshort · n / 32, 触及的 128B line 数) × [] |
| `n2L*` | 0.188431 | n² / L × [] |
| `nhe4*Clt64` | -0.044878 | n · max(0, 128/ISB − 4) × [C < 64] |
| `nEpl_eb*` | 0.023742 | n·epl·eb × [] |
| `n2L*nge128` | -0.073647 | n² / L × [n >= 128] |
| `mn_cshortN_L*Lgt4` | 11.0319 | min(cshort · n / 32, 触及的 128B line 数) × [L > 4] |
| `r_n_eb*` | 0.681232 | 元素数 = BIU 命令数 / 元素字节数 × [] |
| `mn_cshortN_L*smISB` | -19.9494 | min(cshort · n / 32, 触及的 128B line 数) × [inner stride <= 8 bytes] |
| `bsL*` | -39.4918 | [C·eb<32] · L × [] |
| `hepl_n*Clt64` | -0.046555 | n · (128/ISB) × [C < 64] |

### 3. in-sample 拟合误差

- 点数：**750**
- 项数：**11**
- **MAPE：15.19%**

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
T = 16.1073
  + 78.2348 * line_elems_max*
  + 0.457459 * n2L*
  + 2.6929 * bucket_worst_2k8*nge256
  + 0.916042 * bucket_worst_32k16*
  + 9.58329 * r_n_rl*Clt64
  + -693905 * r_it_rs*nge256
  + 139.629 * line_elems_max*nge256
  + -1.91873 * bucket_worst_2k8*nge128
  + 0.036898 * nEpl_eb*nge128
  + -11.6079 * r_n_ISB*mwlg2
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 16.1073 | 固定开销（循环发射、启动） |
| `line_elems_max*` | 78.2348 | 单 line 内最多元素数 × [] |
| `n2L*` | 0.457459 | n² / L × [] |
| `bucket_worst_2k8*nge256` | 2.6929 | 最差 line 桶 (2048..8192) × [n >= 256] |
| `bucket_worst_32k16*` | 0.916042 | 最差 line 桶 (32768..16384) × [] |
| `r_n_rl*Clt64` | 9.58329 | r_n_rl × [C < 64] |
| `r_it_rs*nge256` | -693905 | r_it_rs × [n >= 256] |
| `line_elems_max*nge256` | 139.629 | 单 line 内最多元素数 × [n >= 256] |
| `bucket_worst_2k8*nge128` | -1.91873 | 最差 line 桶 (2048..8192) × [n >= 128] |
| `nEpl_eb*nge128` | 0.036898 | n·epl·eb × [n >= 128] |
| `r_n_ISB*mwlg2` | -11.6079 | r_n_ISB × [mean_warp_lines > 2] |

### 3. in-sample 拟合误差

- 点数：**291**
- 项数：**11**
- **MAPE：8.01%**

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
T = 75.0555
  + 52.5432 * line_elems_max*
  + 0.06692 * hepl_n32*
  + 44.3481 * bsN*
  + 0.476297 * n2L*
  + 8.30638 * bsN*e1
  + -69.0896 * cshortN*
  + 42.1192 * ov_cshortN_L*
  + 2.30041 * r_n_rl*mwlg2
  + -25.6036 * r_C_rl*nge128
  + 13.7198 * r_C_mwl*
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 75.0555 | 固定开销（循环发射、启动） |
| `line_elems_max*` | 52.5432 | 单 line 内最多元素数 × [] |
| `hepl_n32*` | 0.06692 | n · min(128/ISB, 32) × [] |
| `bsN*` | 44.3481 | [C·eb<32] · n × [] |
| `n2L*` | 0.476297 | n² / L × [] |
| `bsN*e1` | 8.30638 | [C·eb<32] · n × [dtype = int8] |
| `cshortN*` | -69.0896 | cshort · n / 32 × [] |
| `ov_cshortN_L*` | 42.1192 | max(0, cshort · n / 32 − 触及的 128B line 数) × [] |
| `r_n_rl*mwlg2` | 2.30041 | r_n_rl × [mean_warp_lines > 2] |
| `r_C_rl*nge128` | -25.6036 | r_C_rl × [n >= 128] |
| `r_C_mwl*` | 13.7198 | r_C_mwl × [] |

### 3. in-sample 拟合误差

- 点数：**943**
- 项数：**11**
- **MAPE：10.54%**

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
T = 125.956
  + 2.27465 * minL64*mwlg2
  + 17.5122 * cmd*mwlg2
  + 22.7323 * iters*Lgt4
  + 0.405867 * p_W_mwl*mwlg2
  + 1.33842 * p_eb_mwl*
  + -6.78281 * minL32*Lgt4
  + 4.71939 * minL32*
  + -1.84634 * p_eb_mwl*mwlg2
  + 0.946059 * p_eb_mwl*Lgt4
  + -0.294682 * mx_L_W*mwlg2
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 125.956 | 固定开销（循环发射、启动） |
| `minL64*mwlg2` | 2.27465 | min(L,64) × [mean_warp_lines > 2] |
| `cmd*mwlg2` | 17.5122 | ceil(L / floor(128/eb)) = MTE 命令数 × [mean_warp_lines > 2] |
| `iters*Lgt4` | 22.7323 | ceil(n/(32·W)) = warp 轮次 × [L > 4] |
| `p_W_mwl*mwlg2` | 0.405867 | num_warps × mean_warp_lines：每 warp 波次触及的 line 数 × [mean_warp_lines > 2] |
| `p_eb_mwl*` | 1.33842 | 元素字节数 × mean_warp_lines：每 warp 波次触及的 line 数 × [] |
| `minL32*Lgt4` | -6.78281 | min(L,32) × [L > 4] |
| `minL32*` | 4.71939 | min(L,32) × [] |
| `p_eb_mwl*mwlg2` | -1.84634 | 元素字节数 × mean_warp_lines：每 warp 波次触及的 line 数 × [mean_warp_lines > 2] |
| `p_eb_mwl*Lgt4` | 0.946059 | 元素字节数 × mean_warp_lines：每 warp 波次触及的 line 数 × [L > 4] |
| `mx_L_W*mwlg2` | -0.294682 | max(触及的 128B line 数, num_warps) × [mean_warp_lines > 2] |

### 3. in-sample 拟合误差

- 点数：**855**
- 项数：**11**
- **MAPE：10.91%**

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
T = 119.2
  + 11.2303 * transX*
  + 1.57522 * r_L_rl*
  + -2.18945 * r_L_W*Lgt16
  + 1.90856 * p_eb_it*Lgt16
  + -8.93234 * p_it_mwl*Lgt4
  + 27.8502 * cmd*Lgt4
  + -6.46549 * linv*Clt64
  + 6.05418 * ov_L_pages*Clt64
  + 0.197137 * p_R_it*Lgt4
  + 1.10752 * minL128*Lgt4
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 119.2 | 固定开销（循环发射、启动） |
| `transX*` | 11.2303 | iters · max_warp_lines × [] |
| `r_L_rl*` | 1.57522 | r_L_rl × [] |
| `r_L_W*Lgt16` | -2.18945 | 触及的 128B line 数 / num_warps × [L > 16] |
| `p_eb_it*Lgt16` | 1.90856 | p_eb_it × [L > 16] |
| `p_it_mwl*Lgt4` | -8.93234 | p_it_mwl × [L > 4] |
| `cmd*Lgt4` | 27.8502 | ceil(L / floor(128/eb)) = MTE 命令数 × [L > 4] |
| `linv*Clt64` | -6.46549 | 1 / L × [C < 64] |
| `ov_L_pages*Clt64` | 6.05418 | max(0, 触及的 128B line 数 − span / 4096（页穿越）) × [C < 64] |
| `p_R_it*Lgt4` | 0.197137 | p_R_it × [L > 4] |
| `minL128*Lgt4` | 1.10752 | min(L,128) × [L > 4] |

### 3. in-sample 拟合误差

- 点数：**1762**
- 项数：**11**
- **MAPE：10.06%**

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
T = 19.6604
  + 1.43265 * mx_L_W*
  + 3.89707 * hLeb4*
  + 0.684181 * rmw*Lgt4
  + 0.3029 * rmwEb*Lgt16
  + -1.71599 * linv*Lgt16
  + -0.466374 * hLW05*mwlg2
  + 2.34305 * hLeb2*Lgt16
  + -0.043602 * hLW05*
  + -5.98897 * ov_L_iters*Lgt4
  + 4.85827 * ov_L_seg*Lgt4
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 19.6604 | 固定开销（循环发射、启动） |
| `mx_L_W*` | 1.43265 | max(触及的 128B line 数, num_warps) × [] |
| `hLeb4*` | 3.89707 | max(0, L/eb − 4) × [] |
| `rmw*Lgt4` | 0.684181 | read-modify-write 比例 × [L > 4] |
| `rmwEb*Lgt16` | 0.3029 | rmw·eb × [L > 16] |
| `linv*Lgt16` | -1.71599 | 1 / L × [L > 16] |
| `hLW05*mwlg2` | -0.466374 | max(0, L − 0.5W) × [mean_warp_lines > 2] |
| `hLeb2*Lgt16` | 2.34305 | max(0, L/eb − 2) × [L > 16] |
| `hLW05*` | -0.043602 | max(0, L − 0.5W) × [] |
| `ov_L_iters*Lgt4` | -5.98897 | max(0, 触及的 128B line 数 − ceil(n/(32·W)) = warp 轮次) × [L > 4] |
| `ov_L_seg*Lgt4` | 4.85827 | ov_L_seg × [L > 4] |

### 3. in-sample 拟合误差

- 点数：**962**
- 项数：**11**
- **MAPE：15.33%**

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
T = 18.7385
  + 1.04613 * mx_L_W*
  + 0.003878 * bucket_pairs_32k16*Clt32
  + 10.6224 * hLeb4*
  + 0.410505 * r_n_rl*nge256
  + -7.37983 * hLeb4*mwlg2
  + -0.497434 * minL128*Lgt4
  + 1.45346 * minLeb32*Wgt1
  + 0.029494 * p_n_it*
  + 2.57047 * hLeb4*Rgt8
  + -0.392681 * comblines*nge128
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `常数项` | 18.7385 | 固定开销（循环发射、启动） |
| `mx_L_W*` | 1.04613 | max(触及的 128B line 数, num_warps) × [] |
| `bucket_pairs_32k16*Clt32` | 0.003878 | line 桶 (32768..16384) 对数 × [C < 32] |
| `hLeb4*` | 10.6224 | max(0, L/eb − 4) × [] |
| `r_n_rl*nge256` | 0.410505 | r_n_rl × [n >= 256] |
| `hLeb4*mwlg2` | -7.37983 | max(0, L/eb − 4) × [mean_warp_lines > 2] |
| `minL128*Lgt4` | -0.497434 | min(L,128) × [L > 4] |
| `minLeb32*Wgt1` | 1.45346 | minLeb32 × [num_warps > 1] |
| `p_n_it*` | 0.029494 | p_n_it × [] |
| `hLeb4*Rgt8` | 2.57047 | max(0, L/eb − 4) × [R > 8] |
| `comblines*nge128` | -0.392681 | R × 每行 line 数 × [n >= 128] |

### 3. in-sample 拟合误差

- 点数：**3416**
- 项数：**11**
- **MAPE：24.33%**

---
