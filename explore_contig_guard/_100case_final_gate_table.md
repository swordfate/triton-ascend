## 4. 100-case 终版回归（2026-11-20，准入门控后）

**终版 wheel = `triton_ascend-3.6.0.dev0+git56d9e023`**（HEAD = `56d9e02384`
= `c380f6da` + 循环域准入两项修正）：
1. `d92bc5a870` 循环域准入门控——loopfit 曲线标定域为 grid=1 热/无 mask/
   线性走址/无归约耦合的干净体（e27b_fit2）；携带 grid-stride shell、
   运行时部分掩码、归约/scan 耦合或 StridedWide 成员的循环体一概不予
   认领，交还 legacy 实测组曲线/跨步层（与基线 4902ae4a 完全同价）。
2. `56d9e02384` 关闭 simt 循环认领——simt loopfit 曲线为 drain-forced
   grid=1 标定，其全部 100-case 消费者欠预测 26-54%/trip
   （strided_scatter −26%、rowred_r16 −54%），且无任何正向贡献者。

**验收结果（95 例可评）**：终版 **79 PASS**，相对基线 4902ae4a（75 PASS）
**零新增 FAIL**（硬目标达成）；median 4.06% / p90 19.39%，均优于基线
（4.10% / 21.74%）。改善 4 例全部保持（vector_add −33.8→−0.0、max_dim1
+12.1→+9.2、cumprod +12.9→+2.4、associative_scan +35.0→+8.6，均来自
直线域/scan-stage 认领）。16 例 FAIL 全部为基线本身 FAIL 的 case
（归因见 `_100case_gt10_contig_analysis.md`：持久化 scan/softmax 结构、
gather 间接寻址、atomic 争用、归约结构——非连续访存标定所致）。

归因方法（Q4 指南 §6 stage 级做差）：5 例新增 FAIL 全部核实为循环域
kernel，且整核缺口 100% 落在被 v2 认领的单一循环 stage 上
（muls_add +28.2 ns/trip −20%、memory_stream_l4 +70.0 −31%（与 E33d
mask +11 + 跳址 +38 吻合）、fused_softmax +83.8 −23%、strided_scatter
−83.6 −26%、permute_reshape −204 −50%（32B 行 @384B 跨步成员被当作
连续计价））。loopfit 解析复刻（probes/loopfit_replica.py）与 stage
价逐位一致，证明缺口全部来自标定域外推。

| # | case | 2e9efcac route | 实测(us) | 4902预测 | 4902误差 | 4902 | 终版 route | 终版预测 | 终版误差 | 终版 |
|---:|---|---|---:|---:|---:|:---:|---|---:|---:|:---:|
| 1 | `muls_add_n2097152` | simd+F1 | 6.625 | 6.735 | +1.66% | ✅ | simd+F1 | 6.735 | +1.66% | ✅ |
| 2 | `muls_add_n1048576` | simd+F1 | 4.746 | 4.734 | -0.25% | ✅ | simd+F1 | 4.734 | -0.25% | ✅ |
| 3 | `memory_stream_l4_n2097152` | simd+F1 | 10.032 | 10.371 | +3.38% | ✅ | simd+F1 | 10.371 | +3.38% | ✅ |
| 4 | `memory_stream_l2_n1048576` | simd+F1 | 4.645 | 4.729 | +1.81% | ✅ | simd+F1 | 4.729 | +1.81% | ✅ |
| 5 | `cumsum_i32_w256` | simd+F1 | 14.856 | 2.986 | -79.90% | ❌ | simd+F1 | 2.986 | -79.90% | ❌ |
| 6 | `cumsum_i32_w128_reverse` | simt+F1 | 16.628 | 2.317 | -86.06% | ❌ | simt+F2 | 2.317 | -86.06% | ❌ |
| 7 | `gather_dot_f16_k16` | simt+F1 | 12.186 | 3.723 | -69.45% | ❌ | simt+F1 | 3.723 | -69.45% | ❌ |
| 8 | `gather_dot_f16_k16_p112` | simt+F1 | 6.061 | 3.723 | -38.58% | ❌ | simt+F1 | 3.723 | -38.58% | ❌ |
| 9 | `masked_copy_r16_l64_v64_s64_k8` | simd+F1 | 3.095 | 3.116 | +0.67% | ✅ | simd+F1 | 2.875 | -7.12% | ✅ |
| 10 | `masked_copy_r64_l128_v128_s256_k12` | simd+F1 | 7.48 | 7.669 | +2.53% | ✅ | simd+F1 | 7.669 | +2.53% | ✅ |
| 11 | `grid_stride_gather_p224` | simt+F1 | 3.836 | 3.002 | -21.74% | ❌ | simt+F1 | 3.002 | -21.74% | ❌ |
| 12 | `grid_stride_dot_p224` | simd+F1 | 5.027 | 4.984 | -0.85% | ✅ | simd+F1 | 4.984 | -0.85% | ✅ |
| 13 | `grid_stride_dot_p56` | simt+F1 | 3.976 | 3.205 | -19.39% | ❌ | simt+F1 | 3.205 | -19.39% | ❌ |
| 14 | `grid_stride_dot_p112` | simt+F1 | 4.72 | 3.890 | -17.59% | ❌ | simt+F1 | 3.890 | -17.59% | ❌ |
| 15 | `grid_stride_dot_p336` | simd+F1 | 6.035 | 5.418 | -10.23% | ❌ | simd+F1 | 5.418 | -10.23% | ❌ |
| 16 | `grid_stride_dot_p448` | simd+F1 | 6.0 | 5.851 | -2.48% | ✅ | simd+F1 | 5.851 | -2.48% | ✅ |
| 17 | `pytest_indirect_load_add_b8` | simt+F1 | 1.201 | 1.234 | +2.71% | ✅ | simt+F1 | 1.234 | +2.71% | ✅ |
| 18 | `pytest_indirect_index_div_b8` | simt+F1 | 1.244 | 1.245 | +0.05% | ✅ | simt+F1 | 1.245 | +0.05% | ✅ |
| 19 | `pytest_indirect_index_and_b8` | simt+F1 | 1.169 | 1.213 | +3.73% | ✅ | simt+F1 | 1.213 | +3.73% | ✅ |
| 20 | `pytest_indirect_index_maxmin_b8` | simt+F1 | 1.166 | 1.213 | +4.01% | ✅ | simt+F1 | 1.213 | +4.01% | ✅ |
| 21 | `pytest_indirect_load_2d_min_m4_n16_k8` | simt+F1 | 1.174 | 1.255 | +6.94% | ✅ | simt+F1 | 1.255 | +6.94% | ✅ |
| 22 | `pytest_indirect_load_2d_grid_m4_n16_k8` | simt+F1 | 1.72 | 1.573 | -8.54% | ✅ | simt+F1 | 1.573 | -8.54% | ✅ |
| 23 | `pytest_index_select_n64_d32_m8` | simd+F1 | 1.136 | 1.084 | -4.59% | ✅ | simd+F1 | 1.084 | -4.59% | ✅ |
| 24 | `pytest_index_put_n64_d32_m8` | simt+F1 | 1.166 | 1.264 | +8.39% | ✅ | simt+F1 | 1.264 | +8.39% | ✅ |
| 25 | `pytest_indirect_store_b8` | simt+F1 | 1.09 | 1.162 | +6.61% | ✅ | simt+F1 | 1.162 | +6.61% | ✅ |
| 26 | `pytest_gather_dot_f16_k16_p56` | simt+F1 | 4.619 | 3.722 | -19.41% | ❌ | simt+F1 | 3.722 | -19.41% | ❌ |
| 27 | `pytest_vector_add_n98432` | simd+F1 | 2.141 | 2.864 | +33.77% | ❌ | simd+F1 | 2.140 | -0.03% | ✅ |
| 28 | `pytest_permute_x64_y32` | simd+F1 | 2.327 | 2.333 | +0.24% | ✅ | simd+F1 | 2.333 | +0.24% | ✅ |
| 29 | `pytest_sigmoid_f32_2x4096x8` | simd+F1 | 4.9 | 4.633 | -5.46% | ✅ | simd+F1 | 4.650 | -5.10% | ✅ |
| 30 | `pytest_rms_norm_f16_m128_n128` | simd+F1 | 6.152 | 6.611 | +7.46% | ✅ | simd+F1 | 6.611 | +7.46% | ✅ |
| 31 | `pytest_layer_norm_f16_m128_n128` | simt+F1 | 3.679 | 3.461 | -5.93% | ✅ | simt+F4 | 3.429 | -6.80% | ✅ |
| 32 | `pytest_insert_slice_n1024_s32` | simd+F1 | 1.117 | 1.078 | -3.52% | ✅ | simd+F1 | 1.078 | -3.52% | ✅ |
| 33 | `pytest_atomic_add_f32_32x32_c2` | simd+F1 | None | 1.849 | — | 不可评 | simd+F1 | 1.849 | — | 不可评 |
| 34 | `pytest_exp_f32_2x4096x8` | simd+F1 | 4.347 | 4.596 | +5.73% | ✅ | simd+F1 | 4.258 | -2.04% | ✅ |
| 35 | `pytest_log_f32_n781` | simd+F1 | 1.002 | 1.057 | +5.51% | ✅ | simd+F1 | 1.047 | +4.48% | ✅ |
| 36 | `pytest_relu_f32_2x4096x8` | simd+F1 | 4.377 | 4.628 | +5.73% | ✅ | simd+F1 | 4.645 | +6.12% | ✅ |
| 37 | `pytest_where_f32_2x1024x8` | simd+F1 | 1.947 | 1.959 | +0.64% | ✅ | simd+F1 | 1.964 | +0.86% | ✅ |
| 38 | `pytest_broadcast_bf16_2x4x8` | simd+F1 | 0.988 | 0.956 | -3.23% | ✅ | simd+F1 | 0.956 | -3.23% | ✅ |
| 39 | `pytest_max_dim1_f32_m64_n8` | simd+F1 | 1.246 | 1.397 | +12.10% | ❌ | simd+F1 | 1.360 | +9.16% | ✅ |
| 40 | `pytest_mean_dim1_f32_m57_n3` | simd+F1 | 1.163 | 1.166 | +0.27% | ✅ | simd+F1 | 1.146 | -1.45% | ✅ |
| 41 | `pytest_mul_reduce_f16_16x1x1x1x1` | simd+F1 | 0.987 | 1.067 | +8.08% | ✅ | simd+F1 | 1.067 | +8.08% | ✅ |
| 42 | `pytest_cumprod_f32_n1024` | simt+F1 | 1.778 | 2.008 | +12.92% | ❌ | simt+F1 | 1.820 | +2.35% | ✅ |
| 43 | `pytest_cummax_f32_16x32_dim1` | simt+F1 | 1.272 | 1.214 | -4.55% | ✅ | simt+F1 | 1.214 | -4.55% | ✅ |
| 44 | `pytest_atomic_max_i32_32x32_c2` | simd+F1 | 1.325 | 1.446 | +9.11% | ✅ | simd+F1 | 1.446 | +9.11% | ✅ |
| 45 | `atomic_continuous_i32_p1_b256` | simd+F1 | 1.049 | 1.009 | -3.82% | ✅ | simd+F1 | 1.009 | -3.82% | ✅ |
| 46 | `atomic_continuous_i32_p56_b256` | simd+F1 | 1.94 | 2.306 | +18.84% | ❌ | simd+F1 | 2.306 | +18.84% | ❌ |
| 47 | `atomic_discrete_i32_p1_b256` | simt+F1 | 1.572 | 1.527 | -2.85% | ✅ | simt+F1 | 1.527 | -2.85% | ✅ |
| 48 | `atomic_discrete_i32_p56_b256` | simt+F1 | 2.438 | 2.824 | +15.83% | ❌ | simt+F1 | 2.824 | +15.83% | ❌ |
| 49 | `rowred_r16_c128_k16` | simt+F1 | 28.477 | 15.880 | -44.24% | ❌ | simt+F8 | 15.880 | -44.24% | ❌ |
| 50 | `rowred_r64_c256_k22` | simd+F1 | 97.115 | 92.769 | -4.48% | ✅ | simd+F1 | 92.769 | -4.48% | ✅ |
| 51 | `softmax_power2` | simd+F1 | 66.101 | 65.200 | -1.36% | ✅ | simd+F1 | 61.637 | -6.75% | ✅ |
| 52 | `softmax_original` | simd+F1 | 32.494 | 31.261 | -3.79% | ✅ | simd+F1 | 31.261 | -3.79% | ✅ |
| 53 | `fze_default` | simt+F2 | 2825.22 | 2840.585 | +0.54% | ✅ | simt+F2 | 2840.544 | +0.54% | ✅ |
| 54 | `solve_tril_b1_t1024_h32_bt64` | mixed+F4 | 185.085 | 177.505 | -4.10% | ✅ | mixed_simd_simt+F4 | 177.505 | -4.10% | ✅ |
| 55 | `solve_tril_b1_t1024_h64_bt64` | mixed+F4 | 325.322 | 321.309 | -1.23% | ✅ | mixed_simd_simt+F4 | 321.309 | -1.23% | ✅ |
| 56 | `solve_tril_b8_t1024_h32_bt64` | mixed+F4 | 1506.969 | 1413.332 | -6.21% | ✅ | mixed_simd_simt+F4 | 1413.332 | -6.21% | ✅ |
| 57 | `solve_tril_b16_t1024_h64_bt64` | mixed+F4 | 5943.006 | 5840.174 | -1.73% | ✅ | mixed_simd_simt+F4 | 5840.174 | -1.73% | ✅ |
| 58 | `solve_tril_b1_t8192_h64_bt64` | mixed+F4 | 2980.632 | 2877.396 | -3.46% | ✅ | mixed_simd_simt+F4 | 2877.396 | -3.46% | ✅ |
| 59 | `solve_tril_b8_t8192_h32_bt64` | mixed+F4 | 11848.565 | 11754.981 | -0.79% | ✅ | mixed_simd_simt+F4 | 11754.981 | -0.79% | ✅ |
| 60 | `solve_tril_b16_t8192_h64_bt64` | mixed+F4 | 47645.837 | 47192.594 | -0.95% | ✅ | mixed_simd_simt+F4 | 47192.594 | -0.95% | ✅ |
| 61 | `solve_tril_b4_t131072_h32_bt64` | mixed+F4 | 95234.816 | 94488.175 | -0.78% | ✅ | mixed_simd_simt+F4 | 94488.175 | -0.78% | ✅ |
| 62 | `masked_reduction_r32_l150_v256_s150_k12` | simd+F1 | 47.341 | 43.913 | -7.24% | ✅ | simd+F1 | 43.913 | -7.24% | ✅ |
| 63 | `masked_reduction_r64_l150_v256_s150_k22` | simd+F1 | 98.008 | 92.947 | -5.16% | ✅ | simd+F1 | 92.947 | -5.16% | ✅ |
| 64 | `masked_softmax_r32_l150_v256_s150_k12` | simd+F1 | 78.009 | 16.995 | -78.21% | ❌ | simd+F1 | 16.023 | -79.46% | ❌ |
| 65 | `masked_softmax_r80_l150_v150_s150_k18` | simd+F1 | 32.261 | 31.260 | -3.10% | ✅ | simd+F1 | 31.260 | -3.10% | ✅ |
| 66 | `pytest_fbgemm_rowwise_quant_t256_d1024_e8_v512` | simt+F4 | 8.659 | 8.259 | -4.62% | ✅ | simt+F4 | 8.159 | -5.77% | ✅ |
| 67 | `pytest_sum_low_f32_64x8192` | simd+F1 | 46.671 | 46.435 | -0.51% | ✅ | simd+F1 | 46.435 | -0.51% | ✅ |
| 68 | `pytest_sum_high_f32_64x8192` | simd+F1 | 4.5 | 3.961 | -11.98% | ❌ | simd+F1 | 3.961 | -11.98% | ❌ |
| 69 | `histogram_i32_r8192_b256_uniform` | mixed+F1 | None | 73.133 | — | 不可评 | mixed_simd_simt+F1 | 60.076 | — | 不可评 |
| 70 | `histogram_i32_r8192_b256_single` | mixed+F1 | None | 73.133 | — | 不可评 | mixed_simd_simt+F1 | 60.076 | — | 不可评 |
| 71 | `histogram_i32_r8192_b256_radix` | mixed+F1 | None | 73.133 | — | 不可评 | mixed_simd_simt+F1 | 60.076 | — | 不可评 |
| 72 | `pytest_discrete_mask_4way_f16_8x8` | simt+F1 | 1.0 | 1.029 | +2.89% | ✅ | simt+F1 | 1.029 | +2.89% | ✅ |
| 73 | `pytest_reduce_sum_epilogue_f32_8x8x512` | simd+F1 | 2.772 | 2.380 | -14.13% | ❌ | simd+F1 | 2.380 | -14.13% | ❌ |
| 74 | `pytest_max_dim0_f32_64x8` | simd+F1 | 1.251 | 1.371 | +9.60% | ✅ | simd+F1 | 1.334 | +6.63% | ✅ |
| 75 | `pytest_argmax_f32_37x125_dim1` | simt+F1 | 22.148 | 22.483 | +1.51% | ✅ | simt+F1 | 22.483 | +1.51% | ✅ |
| 76 | `pytest_gather_after_reduce_f32_16x1024` | simd+F1 | 4.608 | 4.745 | +2.97% | ✅ | simd+F1 | 4.745 | +2.97% | ✅ |
| 77 | `pytest_view_f32_2x256x16` | simd+F1 | 1.283 | 1.304 | +1.67% | ✅ | simd+F1 | 1.304 | +1.67% | ✅ |
| 78 | `pytest_permute_reshape_fused_f32` | simd+F1 | 6.471 | 6.431 | -0.62% | ✅ | simd+F1 | 6.431 | -0.62% | ✅ |
| 79 | `pytest_nearest_resize_i32_1x4x64x64` | simt+F1 | 2.46 | 2.346 | -4.62% | ✅ | simt+F4 | 2.346 | -4.63% | ✅ |
| 80 | `pytest_sort_f32_256x16` | simd+F1 | 51.767 | 51.612 | -0.30% | ✅ | simd+F1 | 51.612 | -0.30% | ✅ |
| 81 | `pytest_topk_f32_16x64_k4` | simt+F1 | 1.605 | 1.588 | -1.06% | ✅ | simt+F1 | 1.588 | -1.06% | ✅ |
| 82 | `pytest_multi_input_scan_f32_2x5x10` | simt+F1 | 1.694 | 1.571 | -7.28% | ✅ | simt+F1 | 1.571 | -7.28% | ✅ |
| 83 | `pytest_cummin_f32_3x5x7_dim1` | simd+F1 | 26.797 | 1.160 | -95.67% | ❌ | simd+F1 | 1.160 | -95.67% | ❌ |
| 84 | `pytest_histogram_i32_2048_bins2` | mixed+F1 | None | 3.028 | — | 不可评 | mixed_simd_simt+F1 | 3.028 | — | 不可评 |
| 85 | `pytest_scalar_reduce_store_f32_n16` | simd+F1 | 1.498 | 1.559 | +4.06% | ✅ | simd+F1 | 1.559 | +4.06% | ✅ |
| 86 | `pytest_conditional_load_f32_n32` | simt+F1 | 1.106 | 1.117 | +1.03% | ✅ | simt+F1 | 1.117 | +1.03% | ✅ |
| 87 | `pytest_dot_bf16_16x32x16` | simd+F1 | 1.244 | 1.246 | +0.14% | ✅ | simd+F1 | 1.246 | +0.14% | ✅ |
| 88 | `pytest_fused_softmax_f32_128x257` | simd+F1 | 3.158 | 3.006 | -4.82% | ✅ | simd+F1 | 3.006 | -4.82% | ✅ |
| 89 | `pytest_add_f32_2x4096x8` | simd+F1 | 4.349 | 4.628 | +6.42% | ✅ | simd+F1 | 4.646 | +6.82% | ✅ |
| 90 | `pytest_sqrt_f32_n781` | simd+F1 | 1.003 | 1.023 | +2.00% | ✅ | simd+F1 | 1.013 | +0.97% | ✅ |
| 91 | `pytest_strided_scatter_f32_s16` | simt+F1 | 4.302 | 3.979 | -7.50% | ✅ | simt+F1 | 3.979 | -7.50% | ✅ |
| 92 | `pytest_block_ptr_strided_f32_4x8_s32x4` | simt+F1 | 1.112 | 1.091 | -1.87% | ✅ | simt+F1 | 1.091 | -1.87% | ✅ |
| 93 | `pytest_cos_f32_n781` | simd+F1 | 1.529 | 1.561 | +2.12% | ✅ | simd+F1 | 1.551 | +1.45% | ✅ |
| 94 | `pytest_mul_f32_2x4096x8` | simd+F1 | 4.355 | 4.628 | +6.28% | ✅ | simd+F1 | 4.646 | +6.67% | ✅ |
| 95 | `pytest_associative_scan_max_f32_128x4x16_dim2` | simt+F1 | 4.988 | 6.732 | +34.97% | ❌ | simt+F1 | 5.415 | +8.55% | ✅ |
| 96 | `pytest_div_f32_2x4096x8` | simd+F1 | 4.365 | 4.628 | +6.03% | ✅ | simd+F1 | 4.646 | +6.43% | ✅ |
| 97 | `pytest_f2i32_f32_n3` | simd+F1 | 0.986 | 0.958 | -2.81% | ✅ | simd+F1 | 0.957 | -2.93% | ✅ |
| 98 | `pytest_chunk_cumsum_f32_b4_t2048_h8_bt128` | simt+F2 | 3.906 | 3.869 | -0.94% | ✅ | simt+F2 | 3.869 | -0.94% | ✅ |
| 99 | `pytest_maximum_f32_2x4096x8` | simd+F1 | 5.661 | 5.673 | +0.21% | ✅ | simd+F1 | 5.690 | +0.52% | ✅ |
| 100 | `pytest_strided_compaction_f32_s16` | simd+F1 | 2.956 | 2.916 | -1.34% | ✅ | simd+F1 | 2.916 | -1.34% | ✅ |
