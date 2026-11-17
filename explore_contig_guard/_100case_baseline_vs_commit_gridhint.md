# 100-case 同口径对比：基线 git2e9efcac vs 终版 git9e6dd2a7

口径（两轮完全一致，仅 costmodel wheel 不同）：

| 项 | 值 |
|---|---|
| harness | `/root/w/cc-harness/costmodel_generalization`，8 个 kernel bench 全部 hint=grid（muls_add、memory_stream、cumsum、op、rem、gather_dot、grid_stride_dot、grid_stride_gather，共 16 case） |
| 跑法 | `run_generalization_matrix.py --disable-profiler`（compile-only，只产 route+预测，不执行 kernel），8 分片并行、卡 1/2 轮转（0 号不可用、4/5/6 禁用） |
| 基线轮 | `triton_ascend-3.6.0.dev0+git2e9efcac`（2026-11-16 构建） |
| 终版轮 | `triton_ascend-3.6.0.dev0+git9e6dd2a7`（commit 9e6dd2a718：旧 tier-2 字节求和包络删除、E17 逐方向混合包络 + iterationCount≥16 标定域门、tier-3 逐 op 解析兜底） |
| 实测(us) | **照抄**自 v140 历史实测表 `table_100case.md`（上一轮 profiler 实测口径），本轮未重测——compile-only 模式不执行 kernel，实测与 hint/wheel 无关 |
| 预测(us) | 各轮 route.json `candidate_elapsed_costs[effective_decision_kind]`（ns→µs） |
| 误差 | (预测 − 实测) / 实测，正=预测偏高；PASS = \|误差\| < 10% |

**不可评 case（5 个，不计 PASS/FAIL）**：`histogram_i32_r8192_b256_{uniform,
single,radix}`、`pytest_histogram_i32_2048_bins2`、
`pytest_atomic_add_f32_32x32_c2` —— v140 实测表中该 5 例无实测
（当年编译不过未测，表尾注明"可评 95/100"）。

汇总（95 例可评）：基线 **75** PASS（中位 |err| 4.07%、平均 10.43%）；
终版 **81** PASS（中位 4.23%、平均 9.21%）。

## 全表（100 case）

route 列为该轮实际选中路线；变化写 `基线 → 终版`。

| # | case | 基线route | 实测(us)¹ | 基线预测 | 基线误差 | 基线 | 终版route | 终版预测 | 终版误差 | 终版 |
|---:|---|---|---:|---:|---:|:---:|---|---:|---:|:---:|
| 1 | `muls_add_n2097152` | simd+F1 | 6.625 | 6.735 | +1.66% | PASS | simd+F1 | 6.345 | -4.23% | PASS |
| 2 | `muls_add_n1048576` | simd+F1 | 4.746 | 4.734 | -0.25% | PASS | simd+F1 | 4.339 | -8.58% | PASS |
| 3 | `memory_stream_l4_n2097152` | simd+F1 | 10.032 | 10.371 | +3.38% | PASS | simd+F1 | 8.813 | -12.15% | **FAIL** |
| 4 | `memory_stream_l2_n1048576` | simd+F1 | 4.645 | 4.729 | +1.81% | PASS | simd+F1 | 4.335 | -6.67% | PASS |
| 5 | `cumsum_i32_w256` | simd+F1 | 14.856 | 2.986 | -79.90% | **FAIL** | simd+F1 | 2.986 | -79.90% | **FAIL** |
| 6 | `cumsum_i32_w128_reverse` | simt+F1 | 16.628 | 2.317 | -86.07% | **FAIL** | simt+F1 | 2.317 | -86.07% | **FAIL** |
| 7 | `gather_dot_f16_k16` | simt+F1 | 12.186 | 3.723 | -69.45% | **FAIL** | simt+F1 | 3.723 | -69.45% | **FAIL** |
| 8 | `gather_dot_f16_k16_p112` | simt+F1 | 6.061 | 3.723 | -38.57% | **FAIL** | simt+F1 | 3.723 | -38.57% | **FAIL** |
| 9 | `masked_copy_r16_l64_v64_s64_k8` | simd+F1 | 3.095 | 3.116 | +0.68% | PASS | simd+F1 | 3.075 | -0.65% | PASS |
| 10 | `masked_copy_r64_l128_v128_s256_k12` | simd+F1 | 7.48 | 7.669 | +2.53% | PASS | simd+F1 | 7.669 | +2.53% | PASS |
| 11 | `grid_stride_gather_p224` | simt+F1 | 3.836 | 3.002 | -21.74% | **FAIL** | simt+F1 | 3.002 | -21.74% | **FAIL** |
| 12 | `grid_stride_dot_p224` | simd+F1 | 5.027 | 4.984 | -0.86% | PASS | simd+F1 | 5.07 | +0.86% | PASS |
| 13 | `grid_stride_dot_p56` | simt+F1 | 3.976 | 3.205 | -19.39% | **FAIL** | simt+F1 | 3.531 | -11.19% | **FAIL** |
| 14 | `grid_stride_dot_p112` | simt+F1 | 4.72 | 3.89 | -17.58% | **FAIL** | simt+F1 | 4.435 | -6.04% | PASS |
| 15 | `grid_stride_dot_p336` | simd+F1 | 6.035 | 5.418 | -10.22% | **FAIL** | simd+F1 | 5.428 | -10.06% | **FAIL** |
| 16 | `grid_stride_dot_p448` | simd+F1 | 6.0 | 5.851 | -2.48% | PASS | simd+F1 | 5.787 | -3.55% | PASS |
| 17 | `pytest_indirect_load_add_b8` | simt+F1 | 1.201 | 1.234 | +2.75% | PASS | simt+F1 | 1.234 | +2.75% | PASS |
| 18 | `pytest_indirect_index_div_b8` | simt+F1 | 1.244 | 1.245 | +0.08% | PASS | simt+F1 | 1.245 | +0.08% | PASS |
| 19 | `pytest_indirect_index_and_b8` | simt+F1 | 1.169 | 1.213 | +3.76% | PASS | simt+F1 | 1.213 | +3.76% | PASS |
| 20 | `pytest_indirect_index_maxmin_b8` | simt+F1 | 1.166 | 1.213 | +4.03% | PASS | simt+F1 | 1.213 | +4.03% | PASS |
| 21 | `pytest_indirect_load_2d_min_m4_n16_k8` | simt+F1 | 1.174 | 1.255 | +6.90% | PASS | simt+F1 | 1.255 | +6.90% | PASS |
| 22 | `pytest_indirect_load_2d_grid_m4_n16_k8` | simt+F1 | 1.72 | 1.573 | -8.55% | PASS | simt+F1 | 1.573 | -8.55% | PASS |
| 23 | `pytest_index_select_n64_d32_m8` | simd+F1 | 1.136 | 1.084 | -4.58% | PASS | simd+F1 | 1.084 | -4.58% | PASS |
| 24 | `pytest_index_put_n64_d32_m8` | simt+F1 | 1.166 | 1.264 | +8.40% | PASS | simt+F1 | 1.264 | +8.40% | PASS |
| 25 | `pytest_indirect_store_b8` | simt+F1 | 1.09 | 1.162 | +6.61% | PASS | simt+F1 | 1.162 | +6.61% | PASS |
| 26 | `pytest_gather_dot_f16_k16_p56` | simt+F1 | 4.619 | 3.722 | -19.42% | **FAIL** | simt+F1 | 3.722 | -19.42% | **FAIL** |
| 27 | `pytest_vector_add_n98432` | simd+F1 | 2.141 | 2.864 | +33.77% | **FAIL** | simd+F1 | 2.14 | -0.05% | PASS |
| 28 | `pytest_permute_x64_y32` | simd+F1 | 2.327 | 2.333 | +0.26% | PASS | simd+F1 | 2.333 | +0.26% | PASS |
| 29 | `pytest_sigmoid_f32_2x4096x8` | simd+F1 | 4.9 | 4.633 | -5.45% | PASS | simd+F1 | 4.635 | -5.41% | PASS |
| 30 | `pytest_rms_norm_f16_m128_n128` | simd+F1 | 6.152 | 6.611 | +7.46% | PASS | simd+F1 | 6.611 | +7.46% | PASS |
| 31 | `pytest_layer_norm_f16_m128_n128` | simt+F1 | 3.679 | 3.461 | -5.93% | PASS | simt+F1 | 3.429 | -6.80% | PASS |
| 32 | `pytest_insert_slice_n1024_s32` | simd+F1 | 1.117 | 1.078 | -3.49% | PASS | simd+F1 | 1.078 | -3.49% | PASS |
| 33 | `pytest_atomic_add_f32_32x32_c2` | simd+F1 | None | 1.849 | — | 不可评 | simd+F1 | 1.849 | — | 不可评 |
| 34 | `pytest_exp_f32_2x4096x8` | simd+F1 | 4.347 | 4.596 | +5.73% | PASS | simd+F1 | 4.623 | +6.35% | PASS |
| 35 | `pytest_log_f32_n781` | simd+F1 | 1.002 | 1.057 | +5.49% | PASS | simd+F1 | 1.047 | +4.49% | PASS |
| 36 | `pytest_relu_f32_2x4096x8` | simd+F1 | 4.377 | 4.628 | +5.73% | PASS | simd+F1 | 4.63 | +5.78% | PASS |
| 37 | `pytest_where_f32_2x1024x8` | simd+F1 | 1.947 | 1.959 | +0.62% | PASS | simd+F1 | 2.007 | +3.08% | PASS |
| 38 | `pytest_broadcast_bf16_2x4x8` | simd+F1 | 0.988 | 0.956 | -3.24% | PASS | simd+F1 | 0.956 | -3.24% | PASS |
| 39 | `pytest_max_dim1_f32_m64_n8` | simd+F1 | 1.246 | 1.397 | +12.12% | **FAIL** | simd+F1 | 1.36 | +9.15% | PASS |
| 40 | `pytest_mean_dim1_f32_m57_n3` | simd+F1 | 1.163 | 1.166 | +0.26% | PASS | simd+F1 | 1.146 | -1.46% | PASS |
| 41 | `pytest_mul_reduce_f16_16x1x1x1x1` | simd+F1 | 0.987 | 1.067 | +8.11% | PASS | simd+F1 | 1.067 | +8.11% | PASS |
| 42 | `pytest_cumprod_f32_n1024` | simt+F1 | 1.778 | 2.008 | +12.94% | **FAIL** | simt+F1 | 1.82 | +2.36% | PASS |
| 43 | `pytest_cummax_f32_16x32_dim1` | simt+F1 | 1.272 | 1.214 | -4.56% | PASS | simt+F1 | 1.214 | -4.56% | PASS |
| 44 | `pytest_atomic_max_i32_32x32_c2` | simd+F1 | 1.325 | 1.446 | +9.13% | PASS | simd+F1 | 1.446 | +9.13% | PASS |
| 45 | `atomic_continuous_i32_p1_b256` | simd+F1 | 1.049 | 1.009 | -3.81% | PASS | simd+F1 | 1.009 | -3.81% | PASS |
| 46 | `atomic_continuous_i32_p56_b256` | simd+F1 | 1.94 | 2.306 | +18.87% | **FAIL** | simd+F1 | 2.306 | +18.87% | **FAIL** |
| 47 | `atomic_discrete_i32_p1_b256` | simt+F1 | 1.572 | 1.527 | -2.86% | PASS | simt+F1 | 1.527 | -2.86% | PASS |
| 48 | `atomic_discrete_i32_p56_b256` | simt+F1 | 2.438 | 2.824 | +15.83% | **FAIL** | simt+F1 | 2.824 | +15.83% | **FAIL** |
| 49 | `rowred_r16_c128_k16` | simt+F1 | 28.477 | 15.88 | -44.24% | **FAIL** | simt+F1 | 28.89 | +1.45% | PASS |
| 50 | `rowred_r64_c256_k22` | simd+F1 | 97.115 | 92.769 | -4.48% | PASS | simd+F1 | 92.769 | -4.48% | PASS |
| 51 | `softmax_power2` | simd+F1 | 66.101 | 65.2 | -1.36% | PASS | simd+F1 | 61.637 | -6.75% | PASS |
| 52 | `softmax_original` | simd+F1 | 32.494 | 31.261 | -3.79% | PASS | simd+F1 | 31.145 | -4.15% | PASS |
| 53 | `fze_default` | simt+F2 | 2825.22 | 2840.585 | +0.54% | PASS | simt+F2 | 2840.544 | +0.54% | PASS |
| 54 | `solve_tril_b1_t1024_h32_bt64` | mixed+F4 | 185.085 | 178.894 | -3.34% | PASS | mixed+F4 | 178.894 | -3.34% | PASS |
| 55 | `solve_tril_b1_t1024_h64_bt64` | mixed+F4 | 325.322 | 324.141 | -0.36% | PASS | mixed+F4 | 324.141 | -0.36% | PASS |
| 56 | `solve_tril_b8_t1024_h32_bt64` | mixed+F4 | 1506.969 | 1424.071 | -5.50% | PASS | mixed+F4 | 1424.071 | -5.50% | PASS |
| 57 | `solve_tril_b16_t1024_h64_bt64` | mixed+F4 | 5943.006 | 5885.014 | -0.98% | PASS | mixed+F4 | 5885.014 | -0.98% | PASS |
| 58 | `solve_tril_b1_t8192_h64_bt64` | mixed+F4 | 2980.632 | 2899.816 | -2.71% | PASS | mixed+F4 | 2899.816 | -2.71% | PASS |
| 59 | `solve_tril_b8_t8192_h32_bt64` | mixed+F4 | 11848.565 | 11840.531 | -0.07% | PASS | mixed+F4 | 11840.531 | -0.07% | PASS |
| 60 | `solve_tril_b16_t8192_h64_bt64` | mixed+F4 | 47645.837 | 47550.853 | -0.20% | PASS | mixed+F4 | 47550.853 | -0.20% | PASS |
| 61 | `solve_tril_b4_t131072_h32_bt64` | mixed+F4 | 95234.816 | 95172.21 | -0.07% | PASS | mixed+F4 | 95172.21 | -0.07% | PASS |
| 62 | `masked_reduction_r32_l150_v256_s150_k12` | simd+F1 | 47.341 | 43.913 | -7.24% | PASS | simd+F1 | 43.913 | -7.24% | PASS |
| 63 | `masked_reduction_r64_l150_v256_s150_k22` | simd+F1 | 98.008 | 92.947 | -5.16% | PASS | simd+F1 | 92.947 | -5.16% | PASS |
| 64 | `masked_softmax_r32_l150_v256_s150_k12` | simd+F1 | 78.009 | 16.995 | -78.21% | **FAIL** | simd+F1 | 16.023 | -79.46% | **FAIL** |
| 65 | `masked_softmax_r80_l150_v150_s150_k18` | simd+F1 | 32.261 | 31.26 | -3.10% | PASS | simd+F1 | 31.144 | -3.46% | PASS |
| 66 | `pytest_fbgemm_rowwise_quant_t256_d1024_e8_v512` | simt+F4 | 8.659 | 8.259 | -4.62% | PASS | simt+F4 | 8.159 | -5.77% | PASS |
| 67 | `pytest_sum_low_f32_64x8192` | simd+F1 | 46.671 | 46.435 | -0.51% | PASS | simd+F1 | 46.435 | -0.51% | PASS |
| 68 | `pytest_sum_high_f32_64x8192` | simd+F1 | 4.5 | 3.961 | -11.98% | **FAIL** | simd+F1 | 3.961 | -11.98% | **FAIL** |
| 69 | `histogram_i32_r8192_b256_uniform` | mixed+F1 | None | 73.133 | — | 不可评 | mixed+F1 | 60.076 | — | 不可评 |
| 70 | `histogram_i32_r8192_b256_single` | mixed+F1 | None | 73.133 | — | 不可评 | mixed+F1 | 60.076 | — | 不可评 |
| 71 | `histogram_i32_r8192_b256_radix` | mixed+F1 | None | 73.133 | — | 不可评 | mixed+F1 | 60.076 | — | 不可评 |
| 72 | `pytest_discrete_mask_4way_f16_8x8` | simt+F1 | 1.0 | 1.029 | +2.90% | PASS | simt+F1 | 1.029 | +2.90% | PASS |
| 73 | `pytest_reduce_sum_epilogue_f32_8x8x512` | simd+F1 | 2.772 | 2.38 | -14.14% | **FAIL** | simd+F1 | 2.772 | +0.00% | PASS |
| 74 | `pytest_max_dim0_f32_64x8` | simd+F1 | 1.251 | 1.371 | +9.59% | PASS | simd+F1 | 1.334 | +6.63% | PASS |
| 75 | `pytest_argmax_f32_37x125_dim1` | simt+F1 | 22.148 | 22.483 | +1.51% | PASS | simt+F1 | 22.483 | +1.51% | PASS |
| 76 | `pytest_gather_after_reduce_f32_16x1024` | simd+F1 | 4.608 | 4.745 | +2.97% | PASS | simd+F1 | 4.745 | +2.97% | PASS |
| 77 | `pytest_view_f32_2x256x16` | simd+F1 | 1.283 | 1.304 | +1.64% | PASS | simd+F1 | 1.304 | +1.64% | PASS |
| 78 | `pytest_permute_reshape_fused_f32` | simd+F1 | 6.471 | 6.431 | -0.62% | PASS | simd+F1 | 6.339 | -2.04% | PASS |
| 79 | `pytest_nearest_resize_i32_1x4x64x64` | simt+F1 | 2.46 | 2.346 | -4.63% | PASS | simt+F1 | 2.346 | -4.63% | PASS |
| 80 | `pytest_sort_f32_256x16` | simd+F1 | 51.767 | 51.612 | -0.30% | PASS | simd+F1 | 51.612 | -0.30% | PASS |
| 81 | `pytest_topk_f32_16x64_k4` | simt+F1 | 1.605 | 1.588 | -1.06% | PASS | simt+F1 | 1.588 | -1.06% | PASS |
| 82 | `pytest_multi_input_scan_f32_2x5x10` | simt+F1 | 1.694 | 1.571 | -7.26% | PASS | simt+F1 | 1.571 | -7.26% | PASS |
| 83 | `pytest_cummin_f32_3x5x7_dim1` | simd+F1 | 26.797 | 1.16 | -95.67% | **FAIL** | simd+F1 | 1.16 | -95.67% | **FAIL** |
| 84 | `pytest_histogram_i32_2048_bins2` | mixed+F1 | None | 3.028 | — | 不可评 | mixed+F1 | 3.028 | — | 不可评 |
| 85 | `pytest_scalar_reduce_store_f32_n16` | simd+F1 | 1.498 | 1.559 | +4.07% | PASS | simd+F1 | 1.559 | +4.07% | PASS |
| 86 | `pytest_conditional_load_f32_n32` | simt+F1 | 1.106 | 1.117 | +0.99% | PASS | simt+F1 | 1.117 | +0.99% | PASS |
| 87 | `pytest_dot_bf16_16x32x16` | simd+F1 | 1.244 | 1.246 | +0.16% | PASS | simd+F1 | 1.246 | +0.16% | PASS |
| 88 | `pytest_fused_softmax_f32_128x257` | simd+F1 | 3.158 | 3.006 | -4.81% | PASS | simd+F1 | 3.005 | -4.84% | PASS |
| 89 | `pytest_add_f32_2x4096x8` | simd+F1 | 4.349 | 4.628 | +6.42% | PASS | simd+F1 | 4.631 | +6.48% | PASS |
| 90 | `pytest_sqrt_f32_n781` | simd+F1 | 1.003 | 1.023 | +1.99% | PASS | simd+F1 | 1.013 | +1.00% | PASS |
| 91 | `pytest_strided_scatter_f32_s16` | simt+F1 | 4.302 | 3.979 | -7.51% | PASS | simt+F1 | 4.595 | +6.81% | PASS |
| 92 | `pytest_block_ptr_strided_f32_4x8_s32x4` | simt+F1 | 1.112 | 1.091 | -1.89% | PASS | simt+F1 | 1.091 | -1.89% | PASS |
| 93 | `pytest_cos_f32_n781` | simd+F1 | 1.529 | 1.561 | +2.09% | PASS | simd+F1 | 1.551 | +1.44% | PASS |
| 94 | `pytest_mul_f32_2x4096x8` | simd+F1 | 4.355 | 4.628 | +6.27% | PASS | simd+F1 | 4.631 | +6.34% | PASS |
| 95 | `pytest_associative_scan_max_f32_128x4x16_dim2` | simt+F1 | 4.988 | 6.732 | +34.96% | **FAIL** | simt+F1 | 5.415 | +8.56% | PASS |
| 96 | `pytest_div_f32_2x4096x8` | simd+F1 | 4.365 | 4.628 | +6.03% | PASS | simd+F1 | 4.631 | +6.09% | PASS |
| 97 | `pytest_f2i32_f32_n3` | simd+F1 | 0.986 | 0.958 | -2.84% | PASS | simd+F1 | 0.957 | -2.94% | PASS |
| 98 | `pytest_chunk_cumsum_f32_b4_t2048_h8_bt128` | simt+F2 | 3.906 | 3.869 | -0.95% | PASS | simt+F2 | 3.869 | -0.95% | PASS |
| 99 | `pytest_maximum_f32_2x4096x8` | simd+F1 | 5.661 | 5.673 | +0.21% | PASS | simd+F1 | 5.676 | +0.26% | PASS |
| 100 | `pytest_strided_compaction_f32_s16` | simd+F1 | 2.956 | 2.916 | -1.35% | PASS | simd+F1 | 2.916 | -1.35% | PASS |

（¹ 实测列照抄 v140 历史实测表，本轮未重测。）

## 路线变化清单

（无——100 case 两轮 route 逐 case 一致。）

## PASS↔FAIL 翻转清单

### 基线 PASS → 终版 FAIL（回归，1 个）

- `memory_stream_l4_n2097152`: +3.38% → -12.15%

### 基线 FAIL → 终版 PASS（改善，7 个）

- `grid_stride_dot_p112`: -17.58% → -6.04%
- `pytest_vector_add_n98432`: +33.77% → -0.05%
- `pytest_max_dim1_f32_m64_n8`: +12.12% → +9.15%
- `pytest_cumprod_f32_n1024`: +12.94% → +2.36%
- `rowred_r16_c128_k16`: -44.24% → +1.45%
- `pytest_reduce_sum_epilogue_f32_8x8x512`: -14.14% → +0.00%
- `pytest_associative_scan_max_f32_128x4x16_dim2`: +34.96% → +8.56%

## 唯一回归的根因分析：memory_stream_l4_n2097152（+3.38% → −12.15%）

按《Q4-costmodel偏差定位-操作指南》第 6 章的方法定位（SIMD SYS_CNT
微探针：`MOV $0, SYS_CNT` inline asm 环内计时、is_pure=False 防 CSE、
ns = ticks×1.00008、30 次 launch × 56 程序取中位；同 tile 几何分
变体做差）。实验记录见 REVIEW §3.5 的 E16/E16b/E17c/E17d 块，
探针脚本 `probes/e16_mixed_stream.py`、`probes/e16b_rate_limit.py`、
`probes/e17c_grid_stride.py`、`probes/e17d_k_reliability.py`。

**kernel 形状**：每迭代 4 条独立 4KB load 流 + 1 条 4KB store，
grid=56 × 37 个 grid-stride trip，每 trip 每流地址跳 896KB
（56×16KB）。实测 10.032µs → 每 trip 271ns。route 两版一致
（all_simd+F1；simt 候选 81.8µs 无竞争力），只有预测从 10.371µs
（基线）变到 8.813µs（终版）。

**终版 8.813µs 的构成与缺口**：E17 混合包络把 16KB load 体按
单流曲线计价——max(TL(16KB)=167.9, TS(4KB)=79.6) + 0.18×79.6 =
182.2ns/trip × 37 ≈ 8.813µs。实测 271ns/trip，缺口 89ns/trip，
三个来源都有探针证据：

1. **多流粒度（主因）**：TL(16KB)=167.9 的隐含速率 97.6 B/ns 是
   "单流 16KB 事务"的值；本 case 是 4 条独立 4KB 流。E16b 实测
   同字节量下 4 流合成速率只有 85.8 B/ns（−12%）；E16 另证 4KB
   事务+依赖链时只有 50.9 B/ns。**分解验证（做差法）**：E16b
   4 流 load-only 191.0ns + E17 TS(4KB)=79.6ns = 270.6ns ≈ 实测
   271ns ✓✓——4 条 load 流已把内存通路打满，store 与 load 束
   **无重叠、串行排后**；而 κ=0.18 编码的"读写大比例重叠"（E16
   在 1:1 单流混合体上实测成立）在"多 load 流 + 1 store"形状下
   不成立，重叠被高估。
2. **grid-stride 散布 × 大足迹**：每 trip 每流跳 896KB，L2 局部性
   差。E17c 在 8KB/4KB 小体上测得散布惩罚仅 +2.5%（103.7 vs
   101.2 ns/trip），大足迹（本 case 每流 32MB）下放大；E17d 的
   "轮换 dst 后 K≥32 触发写回墙"（91.4→212.1 ns/迭代）是同一
   驻留效应的另一面证据。
3. **每 trip 固定开销**：37 个 trip 的循环回边/地址计算未被任何
   层计价（tier-2 只按迭代计内存价；E17e 证明该量在现有噪声下
   不可单独标定）。

**为什么不单点修**：E17b 的 K 轴扫描已证明"逐 K/逐形状补因子表"
不可靠（非单调、三 regime 混杂）。正确修法是把 E17 的标定从
(方向 × 字节档) 二维扩成 **(流数 × 每流事务 × 驻留)** 三维
（E17f 战役，规格已定义于 REVIEW §3.5）：用 E16/E16b 同款 SYS_CNT
探针扫 流数∈{1,2,4,8} × 事务∈{4,8,16,32}KB × {L2 热, DRAM 冷}，
拟合合成速率面后再改包络。E17f 完成前保持现状——本 case 是
已知、已归因的回归，不影响其余 case。

**histogram×3 / pytest_histogram / pytest_atomic_add 的"翻转"是
假象，不是回归**：这 5 例在 v140 实测表中没有实测（当年编译不过，
表尾注明"可评 95/100"）。旧 score100 提取正则把表中"改前预测"列
（73.133 / 3.028）误当实测，制造出"基线 +0.00% → 终版 −17.85%"
的假回归；本文档按"无实测 = 不可评"处理，正确排除。

## E17f 修复轮（2026-11-17，commit git74685b79）

上一节的唯一回归 memory_stream_l4（+3.38% → −12.15%）由 E17f 战役
修复。E17f 扫描（`probes/e17f_stream_sweep.py`，数据
`probes/e17f_results.json`：ld/st/mix × 流数{1,2,4,8} × 每流事务
{4,8,16,32}KB × {L2 热, DRAM 冷}，SYS_CNT 斜率 K=16/32）的结论
**修正了上节的根因判断**：

- **流数本身无惩罚**——总量 16KB：1×16KB=200.7 / 2×8KB=197.5 /
  4×4KB=196.8 ns/迭代（±2%）；32KB：1×32KB=398.3 / 8×4KB=392.0。
  "流数 × 每流事务"维度不进模型，总字节维度已足够；
- **真正的根因是原 E17 TL/TS 常数的探针代际偏差**——首轮 E17 扫描
  比 E16 复制探针（mix 1L+1S 4KB：108.4 vs E17f 106.9 ✓）与真实
  kernel 锚点系统性偏快 18~35%（TL 4KB 76.3 vs 90.0、TS 4KB 79.6
  vs 106.8）；上节"4 流合成速率 75.6 低于单流 97.6"是两代探针的
  错比，"E16b 分解 191.0+79.6=270.6≈271"为巧合性吻合（循环窗内
  store 被完全隐藏：mix4×4 = ld4×4 = 195~197ns）；
- **修复 = 重标定常数、模型形状不变**：profile `loop_envelope_mix`
  → TL {90.0, 113.7, 196.8, 392.0}、TS {106.8, 123.5, 214.8, 377.5}、
  κ 0.18→0.1（混合点实测 κ 0~0.2，随 store/load 平衡度变化）；
- DRAM 冷实测 2~6×（存档未接线，N2 类）。

单 case 验证（compile-only，与上表同口径）：

| case | 9e6dd2a7 预测 | E17f 预测 | 实测(us) | 误差变化 |
|---|---:|---:|---:|---|
| `memory_stream_l4_n2097152` | 8.813 | **9.747** | 10.032 | −12.15% → **−2.84% PASS** |
| `memory_stream_l2_n1048576` | 4.335 | 4.335 | 4.645 | −6.68%（不变） |
| `muls_add_n2097152` | 6.345 | 6.345 | 6.625 | −4.23%（不变） |
| `muls_add_n1048576` | 4.339 | 4.339 | 4.746 | −8.58%（不变） |
| `cumsum_i32_w256` / `_w128_reverse` | 2.986 / 2.317 | 同左 | 14.856 / 16.628 | 排除项，不变 |

（ms_l4 是套件中唯一实际消费 `loop_envelope_mix` 常数的 case；
muls_add/ms_l2 的循环段走另一条与该常数无关的路径，逐位不变。）

全量 gate（同 harness 同 hint，8 分片卡 1/2）：**82/95 可评 PASS**
——相对 9e6dd2a7 轮唯一变化即 ms_l4 FAIL→PASS；**相对基线
2e9efcac：0 回归、7 改善**（清单同上节）。gate 跑在 E17f profile
内容上（构建号 bf511864+E17f JSON，与 commit 74685b79 内容等价）。
