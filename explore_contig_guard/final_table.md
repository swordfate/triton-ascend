# 100-case 同口径对比：新基线 4902ae4a(#3126) vs 终版（行表退役）

口径（两轮完全一致，仅 costmodel wheel 不同）：

| 项 | 值 |
|---|---|
| harness | `/root/w/cc-harness/costmodel_generalization`，8 个 kernel bench 全部 hint=grid（muls_add、memory_stream、cumsum、op、rem、gather_dot、grid_stride_dot、grid_stride_gather，共 16 case） |
| 跑法 | `run_generalization_matrix.py --disable-profiler`（compile-only，只产 route+预测，不执行 kernel），8 分片并行、卡 1/2 轮转（0 号不可用、4/5/6 禁用） |
| 基线轮 | `triton_ascend-3.6.0.dev0+git0868cdae`（= 上游 4902ae4a 内容等价提交，含 #3126） |
| 终版轮 | `triton_ascend-3.6.0.dev0+git36068e34`（分支 calib/unified-set-final = 4902 内容 + 本 PR 全部连续访存标定 + 行表退役） |
| 实测(us) | **照抄**自 v140 历史实测表 `table_100case.md`（2026-10-12 profiler 实测），本轮未重测——compile-only 模式不执行 kernel，实测与 hint/wheel 无关 |
| 预测(us) | 各轮 route.json `candidate_elapsed_costs[effective_decision_kind]`（ns→µs） |
| 误差 | (预测 − 实测) / 实测，正=预测偏高；PASS = \|误差\| < 10% |

**不可评 case（5 个，不计 PASS/FAIL）**：`histogram_i32_r8192_b256_{uniform,
single,radix}`、`pytest_histogram_i32_2048_bins2`、
`pytest_atomic_add_f32_32x32_c2` —— v140 实测表中该 5 例无实测
（当年编译不过未测，表尾注明"可评 95/100"）。

汇总（95 例可评）：基线 **75** PASS（中位 |err| 4.10%、平均 10.49%）；
终版 **75** PASS（中位 4.56%、平均 10.71%）。

## 全表（100 case）

route 列为该轮实际选中路线；变化写 `基线 → 终版`。

| # | case | 基线route | 实测(us)¹ | 基线预测 | 基线误差 | 基线 | 终版route | 终版预测 | 终版误差 | 终版 |
|---:|---|---|---:|---:|---:|:---:|---|---:|---:|:---:|
| 1 | `muls_add_n2097152` | simd+F1 | 6.625 | 6.735 | +1.66% | PASS | simd+F1 | 6.282 | -5.18% | PASS |
| 2 | `muls_add_n1048576` | simd+F1 | 4.746 | 4.734 | -0.25% | PASS | simd+F1 | 4.236 | -10.75% | **FAIL** |
| 3 | `memory_stream_l4_n2097152` | simd+F1 | 10.032 | 10.371 | +3.38% | PASS | simd+F1 | 9.352 | -6.78% | PASS |
| 4 | `memory_stream_l2_n1048576` | simd+F1 | 4.645 | 4.729 | +1.81% | PASS | simd+F1 | 4.231 | -8.91% | PASS |
| 5 | `cumsum_i32_w256` | simd+F1 | 14.856 | 2.986 | -79.90% | **FAIL** | simd+F1 | 2.986 | -79.90% | **FAIL** |
| 6 | `cumsum_i32_w128_reverse` | simt+F1 | 16.628 | 2.317 | -86.07% | **FAIL** | simt+F1 | 2.317 | -86.07% | **FAIL** |
| 7 | `gather_dot_f16_k16` | simt+F1 | 12.186 | 3.723 | -69.45% | **FAIL** | simt+F1 | 3.723 | -69.45% | **FAIL** |
| 8 | `gather_dot_f16_k16_p112` | simt+F1 | 6.061 | 3.723 | -38.57% | **FAIL** | simt+F1 | 3.723 | -38.57% | **FAIL** |
| 9 | `masked_copy_r16_l64_v64_s64_k8` | simd+F1 | 3.095 | 3.116 | +0.68% | PASS | simd+F1 | 2.932 | -5.27% | PASS |
| 10 | `masked_copy_r64_l128_v128_s256_k12` | simd+F1 | 7.48 | 7.669 | +2.53% | PASS | simd+F1 | 7.669 | +2.53% | PASS |
| 11 | `grid_stride_gather_p224` | simt+F1 | 3.836 | 3.002 | -21.74% | **FAIL** | simt+F1 | 3.002 | -21.74% | **FAIL** |
| 12 | `grid_stride_dot_p224` | simd+F1 | 5.027 | 4.984 | -0.86% | PASS | simd+F1 | 4.972 | -1.09% | PASS |
| 13 | `grid_stride_dot_p56` | simt+F1 | 3.976 | 3.205 | -19.39% | **FAIL** | simt+F1 | 3.158 | -20.57% | **FAIL** |
| 14 | `grid_stride_dot_p112` | simt+F1 | 4.72 | 3.89 | -17.58% | **FAIL** | simt+F1 | 3.795 | -19.60% | **FAIL** |
| 15 | `grid_stride_dot_p336` | simd+F1 | 6.035 | 5.418 | -10.22% | **FAIL** | simd+F1 | 5.399 | -10.54% | **FAIL** |
| 16 | `grid_stride_dot_p448` | simd+F1 | 6.0 | 5.851 | -2.48% | PASS | simd+F1 | 5.826 | -2.90% | PASS |
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
| 29 | `pytest_sigmoid_f32_2x4096x8` | simd+F1 | 4.9 | 4.633 | -5.45% | PASS | simd+F1 | 4.713 | -3.82% | PASS |
| 30 | `pytest_rms_norm_f16_m128_n128` | simd+F1 | 6.152 | 6.611 | +7.46% | PASS | simd+F1 | 6.611 | +7.46% | PASS |
| 31 | `pytest_layer_norm_f16_m128_n128` | simt+F1 | 3.679 | 3.461 | -5.93% | PASS | simt+F1 | 3.429 | -6.80% | PASS |
| 32 | `pytest_insert_slice_n1024_s32` | simd+F1 | 1.117 | 1.078 | -3.49% | PASS | simd+F1 | 1.078 | -3.49% | PASS |
| 33 | `pytest_atomic_add_f32_32x32_c2` | simd+F1 | None | 1.849 | — | 不可评 | simd+F1 | 1.849 | — | 不可评 |
| 34 | `pytest_exp_f32_2x4096x8` | simd+F1 | 4.347 | 4.596 | +5.73% | PASS | simd+F1 | 4.488 | +3.24% | PASS |
| 35 | `pytest_log_f32_n781` | simd+F1 | 1.002 | 1.057 | +5.49% | PASS | simd+F1 | 1.047 | +4.49% | PASS |
| 36 | `pytest_relu_f32_2x4096x8` | simd+F1 | 4.377 | 4.628 | +5.73% | PASS | simd+F1 | 4.708 | +7.56% | PASS |
| 37 | `pytest_where_f32_2x1024x8` | simd+F1 | 1.947 | 1.959 | +0.62% | PASS | simd+F1 | 1.98 | +1.69% | PASS |
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
| 49 | `rowred_r16_c128_k16` | simt+F1 | 28.477 | 15.88 | -44.24% | **FAIL** | simt+F1 | 19.93 | -30.01% | **FAIL** |
| 50 | `rowred_r64_c256_k22` | simd+F1 | 97.115 | 92.769 | -4.48% | PASS | simd+F1 | 92.769 | -4.48% | PASS |
| 51 | `softmax_power2` | simd+F1 | 66.101 | 65.2 | -1.36% | PASS | simd+F1 | 61.637 | -6.75% | PASS |
| 52 | `softmax_original` | simd+F1 | 32.494 | 31.261 | -3.79% | PASS | simd+F1 | 31.265 | -3.78% | PASS |
| 53 | `fze_default` | simt+F2 | 2825.22 | 2840.585 | +0.54% | PASS | simt+F2 | 2840.544 | +0.54% | PASS |
| 54 | `solve_tril_b1_t1024_h32_bt64` | mixed+F4 | 185.085 | 177.505 | -4.10% | PASS | mixed+F4 | 177.505 | -4.10% | PASS |
| 55 | `solve_tril_b1_t1024_h64_bt64` | mixed+F4 | 325.322 | 321.309 | -1.23% | PASS | mixed+F4 | 321.309 | -1.23% | PASS |
| 56 | `solve_tril_b8_t1024_h32_bt64` | mixed+F4 | 1506.969 | 1413.332 | -6.21% | PASS | mixed+F4 | 1413.332 | -6.21% | PASS |
| 57 | `solve_tril_b16_t1024_h64_bt64` | mixed+F4 | 5943.006 | 5840.174 | -1.73% | PASS | mixed+F4 | 5840.174 | -1.73% | PASS |
| 58 | `solve_tril_b1_t8192_h64_bt64` | mixed+F4 | 2980.632 | 2877.396 | -3.46% | PASS | mixed+F4 | 2877.396 | -3.46% | PASS |
| 59 | `solve_tril_b8_t8192_h32_bt64` | mixed+F4 | 11848.565 | 11754.981 | -0.79% | PASS | mixed+F4 | 11754.981 | -0.79% | PASS |
| 60 | `solve_tril_b16_t8192_h64_bt64` | mixed+F4 | 47645.837 | 47192.594 | -0.95% | PASS | mixed+F4 | 47192.594 | -0.95% | PASS |
| 61 | `solve_tril_b4_t131072_h32_bt64` | mixed+F4 | 95234.816 | 94488.175 | -0.78% | PASS | mixed+F4 | 94488.175 | -0.78% | PASS |
| 62 | `masked_reduction_r32_l150_v256_s150_k12` | simd+F1 | 47.341 | 43.913 | -7.24% | PASS | simd+F1 | 43.913 | -7.24% | PASS |
| 63 | `masked_reduction_r64_l150_v256_s150_k22` | simd+F1 | 98.008 | 92.947 | -5.16% | PASS | simd+F1 | 92.947 | -5.16% | PASS |
| 64 | `masked_softmax_r32_l150_v256_s150_k12` | simd+F1 | 78.009 | 16.995 | -78.21% | **FAIL** | simd+F1 | 16.023 | -79.46% | **FAIL** |
| 65 | `masked_softmax_r80_l150_v150_s150_k18` | simd+F1 | 32.261 | 31.26 | -3.10% | PASS | simd+F1 | 31.264 | -3.09% | PASS |
| 66 | `pytest_fbgemm_rowwise_quant_t256_d1024_e8_v512` | simt+F4 | 8.659 | 8.259 | -4.62% | PASS | simt+F4 | 8.159 | -5.77% | PASS |
| 67 | `pytest_sum_low_f32_64x8192` | simd+F1 | 46.671 | 46.435 | -0.51% | PASS | simd+F1 | 46.435 | -0.51% | PASS |
| 68 | `pytest_sum_high_f32_64x8192` | simd+F1 | 4.5 | 3.961 | -11.98% | **FAIL** | simd+F1 | 3.961 | -11.98% | **FAIL** |
| 69 | `histogram_i32_r8192_b256_uniform` | mixed+F1 | None | 73.133 | — | 不可评 | mixed+F1 | 60.076 | — | 不可评 |
| 70 | `histogram_i32_r8192_b256_single` | mixed+F1 | None | 73.133 | — | 不可评 | mixed+F1 | 60.076 | — | 不可评 |
| 71 | `histogram_i32_r8192_b256_radix` | mixed+F1 | None | 73.133 | — | 不可评 | mixed+F1 | 60.076 | — | 不可评 |
| 72 | `pytest_discrete_mask_4way_f16_8x8` | simt+F1 | 1.0 | 1.029 | +2.90% | PASS | simt+F1 | 1.029 | +2.90% | PASS |
| 73 | `pytest_reduce_sum_epilogue_f32_8x8x512` | simd+F1 | 2.772 | 2.38 | -14.14% | **FAIL** | simd+F1 | 2.013 | -27.38% | **FAIL** |
| 74 | `pytest_max_dim0_f32_64x8` | simd+F1 | 1.251 | 1.371 | +9.59% | PASS | simd+F1 | 1.334 | +6.63% | PASS |
| 75 | `pytest_argmax_f32_37x125_dim1` | simt+F1 | 22.148 | 22.483 | +1.51% | PASS | simt+F1 | 22.483 | +1.51% | PASS |
| 76 | `pytest_gather_after_reduce_f32_16x1024` | simd+F1 | 4.608 | 4.745 | +2.97% | PASS | simd+F1 | 4.745 | +2.97% | PASS |
| 77 | `pytest_view_f32_2x256x16` | simd+F1 | 1.283 | 1.304 | +1.64% | PASS | simd+F1 | 1.304 | +1.64% | PASS |
| 78 | `pytest_permute_reshape_fused_f32` | simd+F1 | 6.471 | 6.431 | -0.62% | PASS | simd+F1 | 3.99 | -38.34% | **FAIL** |
| 79 | `pytest_nearest_resize_i32_1x4x64x64` | simt+F1 | 2.46 | 2.346 | -4.63% | PASS | simt+F1 | 2.346 | -4.63% | PASS |
| 80 | `pytest_sort_f32_256x16` | simd+F1 | 51.767 | 51.612 | -0.30% | PASS | simd+F1 | 51.612 | -0.30% | PASS |
| 81 | `pytest_topk_f32_16x64_k4` | simt+F1 | 1.605 | 1.588 | -1.06% | PASS | simt+F1 | 1.588 | -1.06% | PASS |
| 82 | `pytest_multi_input_scan_f32_2x5x10` | simt+F1 | 1.694 | 1.571 | -7.26% | PASS | simt+F1 | 1.571 | -7.26% | PASS |
| 83 | `pytest_cummin_f32_3x5x7_dim1` | simd+F1 | 26.797 | 1.16 | -95.67% | **FAIL** | simd+F1 | 1.16 | -95.67% | **FAIL** |
| 84 | `pytest_histogram_i32_2048_bins2` | mixed+F1 | None | 3.028 | — | 不可评 | mixed+F1 | 3.028 | — | 不可评 |
| 85 | `pytest_scalar_reduce_store_f32_n16` | simd+F1 | 1.498 | 1.559 | +4.07% | PASS | simd+F1 | 1.559 | +4.07% | PASS |
| 86 | `pytest_conditional_load_f32_n32` | simt+F1 | 1.106 | 1.117 | +0.99% | PASS | simt+F1 | 1.117 | +0.99% | PASS |
| 87 | `pytest_dot_bf16_16x32x16` | simd+F1 | 1.244 | 1.246 | +0.16% | PASS | simd+F1 | 1.246 | +0.16% | PASS |
| 88 | `pytest_fused_softmax_f32_128x257` | simd+F1 | 3.158 | 3.006 | -4.81% | PASS | simd+F1 | 2.737 | -13.33% | **FAIL** |
| 89 | `pytest_add_f32_2x4096x8` | simd+F1 | 4.349 | 4.628 | +6.42% | PASS | simd+F1 | 4.709 | +8.28% | PASS |
| 90 | `pytest_sqrt_f32_n781` | simd+F1 | 1.003 | 1.023 | +1.99% | PASS | simd+F1 | 1.013 | +1.00% | PASS |
| 91 | `pytest_strided_scatter_f32_s16` | simt+F1 | 4.302 | 3.979 | -7.51% | PASS | simt+F1 | 3.727 | -13.37% | **FAIL** |
| 92 | `pytest_block_ptr_strided_f32_4x8_s32x4` | simt+F1 | 1.112 | 1.091 | -1.89% | PASS | simt+F1 | 1.091 | -1.89% | PASS |
| 93 | `pytest_cos_f32_n781` | simd+F1 | 1.529 | 1.561 | +2.09% | PASS | simd+F1 | 1.551 | +1.44% | PASS |
| 94 | `pytest_mul_f32_2x4096x8` | simd+F1 | 4.355 | 4.628 | +6.27% | PASS | simd+F1 | 4.709 | +8.13% | PASS |
| 95 | `pytest_associative_scan_max_f32_128x4x16_dim2` | simt+F1 | 4.988 | 6.732 | +34.96% | **FAIL** | simt+F1 | 5.415 | +8.56% | PASS |
| 96 | `pytest_div_f32_2x4096x8` | simd+F1 | 4.365 | 4.628 | +6.03% | PASS | simd+F1 | 4.709 | +7.88% | PASS |
| 97 | `pytest_f2i32_f32_n3` | simd+F1 | 0.986 | 0.958 | -2.84% | PASS | simd+F1 | 0.957 | -2.94% | PASS |
| 98 | `pytest_chunk_cumsum_f32_b4_t2048_h8_bt128` | simt+F2 | 3.906 | 3.869 | -0.95% | PASS | simt+F2 | 3.869 | -0.95% | PASS |
| 99 | `pytest_maximum_f32_2x4096x8` | simd+F1 | 5.661 | 5.673 | +0.21% | PASS | simd+F1 | 5.754 | +1.64% | PASS |
| 100 | `pytest_strided_compaction_f32_s16` | simd+F1 | 2.956 | 2.916 | -1.35% | PASS | simd+F1 | 2.916 | -1.35% | PASS |

（¹ 实测列照抄 v140 历史实测表，本轮未重测。）

## 路线变化清单

（无——100 case 两轮 route 逐 case 一致。）

## PASS↔FAIL 翻转清单

### 基线 PASS → 终版 FAIL（回归，4 个）

- `muls_add_n1048576`: -0.25% → -10.75%
- `pytest_permute_reshape_fused_f32`: -0.62% → -38.34%
- `pytest_fused_softmax_f32_128x257`: -4.81% → -13.33%
- `pytest_strided_scatter_f32_s16`: -7.51% → -13.37%

### 基线 FAIL → 终版 PASS（改善，4 个）

- `pytest_vector_add_n98432`: +33.77% → -0.05%
- `pytest_max_dim1_f32_m64_n8`: +12.12% → +9.15%
- `pytest_cumprod_f32_n1024`: +12.94% → +2.36%
- `pytest_associative_scan_max_f32_128x4x16_dim2`: +34.96% → +8.56%
