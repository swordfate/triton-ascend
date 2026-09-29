[INFO] Using merged native A5 regbase pipeline
warning: warning: --enable-preload=true overrides --enablecv-pipeline-mode; forcing 'Skew'.
warning: warning: --enable-preload=true overrides --enablecv-pipeline-mode; forcing 'Skew'.
// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @strided_kernel(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, false, false, false]> : vector<7xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "aiv", parallel_mode = "simd"} {
  %c12288_i64 = arith.constant 12288 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c0_i64 = arith.constant 0 : i64
  %c16_i32 = arith.constant 16 : i32
  %c56_i32 = arith.constant 56 : i32
  %0 = arith.muli %arg4, %arg5 : i32
  %1 = arith.muli %0, %arg6 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg7 = %3 to %1 step %c56_i32  : i32 {
    %4 = hivm.hir.multi_buffer_counter -> i64
    %c2_i64 = arith.constant 2 : i64
    %5 = arith.remui %4, %c2_i64 : i64
    %6 = arith.index_cast %5 : i64 to index
    %7 = arith.index_cast %6 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %8 = arith.select %7, %c0_i64_0, %c1_i64 : i64
    %9 = hivm.hir.pointer_cast(%c4096_i64, %c12288_i64) : memref<1024xf32, #hivm.address_space<ub>>
    annotation.mark %9 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c0_i64, %c8192_i64) : memref<1024xf32, #hivm.address_space<ub>>
    annotation.mark %10 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %11 = arith.remsi %arg7, %arg4 : i32
    %12 = arith.muli %11, %c16_i32 : i32
    %13 = arith.index_cast %12 : i32 to index
    %14 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%13]
    %15 = affine.apply affine_map<()[s0] -> (s0 * 192)>()[%13]
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [16, 64], strides: [192, 3] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[192, 3], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape = memref.collapse_shape %reinterpret_cast [[0, 1]] : memref<16x64xf32, strided<[192, 3], offset: ?>, #hivm.address_space<gm>> into memref<1024xf32, strided<[3], offset: ?>, #hivm.address_space<gm>>
    annotation.mark %10 {hivm.skip_stride_align_for_vload = #hivm.skip_stride_align_for_vload} : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %8]
    hivm.hir.load ins(%collapse_shape : memref<1024xf32, strided<[3], offset: ?>, #hivm.address_space<gm>>) outs(%10 : memref<1024xf32, #hivm.address_space<ub>>) eviction_policy = <EvictFirst> core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %8]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    func.call @strided_kernel_outlined_vf_0(%10, %9) {hivm.vector_function, no_inline} : (memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %8]
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%14], sizes: [16, 64], strides: [64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_2 = memref.collapse_shape %reinterpret_cast_1 [[0, 1]] : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%9 : memref<1024xf32, #hivm.address_space<ub>>) outs(%collapse_shape_2 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %8]
    hivm.hir.set_ctrl true at ctrl[60]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
<inline asm>:2:1: warning: _mlir_ciface_abs_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_abs_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:5:1: warning: _mlir_ciface_acos_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_acos_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:8:1: warning: _mlir_ciface_acos_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_acos_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:11:1: warning: _mlir_ciface_acosh_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_acosh_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:14:1: warning: _mlir_ciface_acosh_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_acosh_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:17:1: warning: _mlir_ciface_add_rd_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_add_rd_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:20:1: warning: _mlir_ciface_add_rn_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_add_rn_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:23:1: warning: _mlir_ciface_add_ru_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_add_ru_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:26:1: warning: _mlir_ciface_add_rz_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_add_rz_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:29:1: warning: _mlir_ciface_asin_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_asin_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:32:1: warning: _mlir_ciface_asin_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_asin_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:35:1: warning: _mlir_ciface_asinh_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_asinh_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:38:1: warning: _mlir_ciface_asinh_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_asinh_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:41:1: warning: _mlir_ciface_atan_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_atan_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:44:1: warning: _mlir_ciface_atan_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_atan_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:47:1: warning: _mlir_ciface_atan2_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_atan2_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:50:1: warning: _mlir_ciface_atan2_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_atan2_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:53:1: warning: _mlir_ciface_atanh_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_atanh_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:56:1: warning: _mlir_ciface_atanh_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_atanh_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:59:1: warning: _mlir_ciface_brev_i32.vector changed binding to STB_WEAK
.weak _mlir_ciface_brev_i32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:62:1: warning: _mlir_ciface_byte_perm_i32.vector changed binding to STB_WEAK
.weak _mlir_ciface_byte_perm_i32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:65:1: warning: _mlir_ciface_cbrt_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_cbrt_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:68:1: warning: _mlir_ciface_ceil_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_ceil_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:71:1: warning: _mlir_ciface_clz_i32.vector changed binding to STB_WEAK
.weak _mlir_ciface_clz_i32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:74:1: warning: _mlir_ciface_copysign_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_copysign_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:77:1: warning: _mlir_ciface_copysign_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_copysign_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:80:1: warning: _mlir_ciface_cos_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_cos_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:83:1: warning: _mlir_ciface_cos_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_cos_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:86:1: warning: _mlir_ciface_cosh_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_cosh_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:89:1: warning: _mlir_ciface_cosh_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_cosh_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:92:1: warning: _mlir_ciface_cospi_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_cospi_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:95:1: warning: _mlir_ciface_cyl_bessel_i0_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_cyl_bessel_i0_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:98:1: warning: _mlir_ciface_cyl_bessel_i0_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_cyl_bessel_i0_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:101:1: warning: _mlir_ciface_cyl_bessel_i1_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_cyl_bessel_i1_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:104:1: warning: _mlir_ciface_div_rd_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_div_rd_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:107:1: warning: _mlir_ciface_div_rn_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_div_rn_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:110:1: warning: _mlir_ciface_div_ru_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_div_ru_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:113:1: warning: _mlir_ciface_div_rz_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_div_rz_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:116:1: warning: _mlir_ciface_div_rz_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_div_rz_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:119:1: warning: _mlir_ciface_erf_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_erf_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:122:1: warning: _mlir_ciface_erf_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_erf_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:125:1: warning: _mlir_ciface_erfc_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_erfc_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:128:1: warning: _mlir_ciface_erfcinv_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_erfcinv_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:131:1: warning: _mlir_ciface_erfcx_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_erfcx_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:134:1: warning: _mlir_ciface_erfinv_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_erfinv_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:137:1: warning: _mlir_ciface_erfinv_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_erfinv_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:140:1: warning: _mlir_ciface_exp_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_exp_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:143:1: warning: _mlir_ciface_exp10_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_exp10_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:146:1: warning: _mlir_ciface_exp2_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_exp2_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:149:1: warning: _mlir_ciface_expm1_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_expm1_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:152:1: warning: _mlir_ciface_expm1_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_expm1_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:155:1: warning: _mlir_ciface_fast_cos_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_fast_cos_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:158:1: warning: _mlir_ciface_fast_divide_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_fast_divide_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:161:1: warning: _mlir_ciface_fast_divide_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_fast_divide_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:164:1: warning: _mlir_ciface_fast_exp10_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_fast_exp10_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:167:1: warning: _mlir_ciface_fast_exp_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_fast_exp_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:170:1: warning: _mlir_ciface_fast_exp_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_fast_exp_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:173:1: warning: _mlir_ciface_fast_log_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_fast_log_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:176:1: warning: _mlir_ciface_fast_log10_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_fast_log10_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:179:1: warning: _mlir_ciface_fast_log2_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_fast_log2_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:182:1: warning: _mlir_ciface_fast_pow_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_fast_pow_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:185:1: warning: _mlir_ciface_fast_sin_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_fast_sin_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:188:1: warning: _mlir_ciface_fast_sincos_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_fast_sincos_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:191:1: warning: _mlir_ciface_fast_tan_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_fast_tan_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:194:1: warning: _mlir_ciface_fast_tanh_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_fast_tanh_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:197:1: warning: _mlir_ciface_fast_tanh_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_fast_tanh_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:200:1: warning: _mlir_ciface_fdim_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_fdim_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:203:1: warning: _mlir_ciface_ffs_i32.vector changed binding to STB_WEAK
.weak _mlir_ciface_ffs_i32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:206:1: warning: _mlir_ciface_ffs_i64.vector changed binding to STB_WEAK
.weak _mlir_ciface_ffs_i64.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:209:1: warning: _mlir_ciface_finite_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_finite_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:212:1: warning: _mlir_ciface_finite_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_finite_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:215:1: warning: _mlir_ciface_float2half_rn_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_float2half_rn_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:218:1: warning: _mlir_ciface_float2int_rd_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_float2int_rd_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:221:1: warning: _mlir_ciface_float2int_rn_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_float2int_rn_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:224:1: warning: _mlir_ciface_float2int_ru_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_float2int_ru_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:227:1: warning: _mlir_ciface_float2int_rz_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_float2int_rz_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:230:1: warning: _mlir_ciface_float2ll_rd_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_float2ll_rd_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:233:1: warning: _mlir_ciface_float2ll_rn_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_float2ll_rn_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:236:1: warning: _mlir_ciface_float2ll_ru_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_float2ll_ru_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:239:1: warning: _mlir_ciface_float2ll_rz_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_float2ll_rz_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:242:1: warning: _mlir_ciface_float2uint_rd_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_float2uint_rd_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:245:1: warning: _mlir_ciface_float2uint_rn_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_float2uint_rn_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:248:1: warning: _mlir_ciface_float2uint_ru_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_float2uint_ru_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:251:1: warning: _mlir_ciface_float2uint_rz_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_float2uint_rz_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:254:1: warning: _mlir_ciface_float2ull_rd_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_float2ull_rd_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:257:1: warning: _mlir_ciface_float2ull_rn_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_float2ull_rn_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:260:1: warning: _mlir_ciface_float2ull_ru_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_float2ull_ru_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:263:1: warning: _mlir_ciface_float2ull_rz_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_float2ull_rz_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:266:1: warning: _mlir_ciface_float_as_int_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_float_as_int_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:269:1: warning: _mlir_ciface_float_as_uint_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_float_as_uint_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:272:1: warning: _mlir_ciface_floor_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_floor_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:275:1: warning: _mlir_ciface_fma_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_fma_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:278:1: warning: _mlir_ciface_fma_rd_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_fma_rd_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:281:1: warning: _mlir_ciface_fma_rn_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_fma_rn_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:284:1: warning: _mlir_ciface_fma_ru_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_fma_ru_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:287:1: warning: _mlir_ciface_fma_rz_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_fma_rz_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:290:1: warning: _mlir_ciface_fmax_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_fmax_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:293:1: warning: _mlir_ciface_fmin_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_fmin_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:296:1: warning: _mlir_ciface_fmod_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_fmod_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:299:1: warning: _mlir_ciface_fmod_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_fmod_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:302:1: warning: _mlir_ciface_frexp_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_frexp_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:305:1: warning: _mlir_ciface_gamma_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_gamma_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:308:1: warning: _mlir_ciface_gamma_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_gamma_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:311:1: warning: _mlir_ciface_hadd_i32.vector changed binding to STB_WEAK
.weak _mlir_ciface_hadd_i32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:314:1: warning: _mlir_ciface_half2float_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_half2float_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:317:1: warning: _mlir_ciface_hypot_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_hypot_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:320:1: warning: _mlir_ciface_hypot_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_hypot_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:323:1: warning: _mlir_ciface_ilogb_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_ilogb_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:326:1: warning: _mlir_ciface_ilogb_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_ilogb_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:329:1: warning: _mlir_ciface_int2float_rd_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_int2float_rd_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:332:1: warning: _mlir_ciface_int2float_rn_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_int2float_rn_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:335:1: warning: _mlir_ciface_int2float_ru_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_int2float_ru_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:338:1: warning: _mlir_ciface_int2float_rz_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_int2float_rz_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:341:1: warning: _mlir_ciface_int_as_float_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_int_as_float_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:344:1: warning: _mlir_ciface_isinf_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_isinf_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:347:1: warning: _mlir_ciface_isinf_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_isinf_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:350:1: warning: _mlir_ciface_isnan_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_isnan_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:353:1: warning: _mlir_ciface_isnan_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_isnan_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:356:1: warning: _mlir_ciface_j0_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_j0_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:359:1: warning: _mlir_ciface_j1_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_j1_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:362:1: warning: _mlir_ciface_jn_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_jn_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:365:1: warning: _mlir_ciface_ldexp_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_ldexp_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:368:1: warning: _mlir_ciface_ldexp_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_ldexp_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:371:1: warning: _mlir_ciface_lgamma_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_lgamma_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:374:1: warning: _mlir_ciface_lgamma_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_lgamma_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:377:1: warning: _mlir_ciface_ll2float_rd_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_ll2float_rd_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:380:1: warning: _mlir_ciface_ll2float_rn_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_ll2float_rn_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:383:1: warning: _mlir_ciface_ll2float_ru_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_ll2float_ru_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:386:1: warning: _mlir_ciface_ll2float_rz_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_ll2float_rz_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:389:1: warning: _mlir_ciface_llrint_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_llrint_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:392:1: warning: _mlir_ciface_llround_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_llround_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:395:1: warning: _mlir_ciface_log_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_log_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:398:1: warning: _mlir_ciface_log10_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_log10_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:401:1: warning: _mlir_ciface_log10_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_log10_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:404:1: warning: _mlir_ciface_log1p_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_log1p_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:407:1: warning: _mlir_ciface_log1p_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_log1p_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:410:1: warning: _mlir_ciface_log2_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_log2_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:413:1: warning: _mlir_ciface_logb_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_logb_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:416:1: warning: _mlir_ciface_max_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_max_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:419:1: warning: _mlir_ciface_min_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_min_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:422:1: warning: _mlir_ciface_mul24_i32.vector changed binding to STB_WEAK
.weak _mlir_ciface_mul24_i32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:425:1: warning: _mlir_ciface_mul_rd_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_mul_rd_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:428:1: warning: _mlir_ciface_mul_rn_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_mul_rn_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:431:1: warning: _mlir_ciface_mul_ru_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_mul_ru_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:434:1: warning: _mlir_ciface_mul_rz_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_mul_rz_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:437:1: warning: _mlir_ciface_mulhi_i32.vector changed binding to STB_WEAK
.weak _mlir_ciface_mulhi_i32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:440:1: warning: _mlir_ciface_nan_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_nan_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:443:1: warning: _mlir_ciface_nearbyint_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_nearbyint_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:446:1: warning: _mlir_ciface_nearbyint_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_nearbyint_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:449:1: warning: _mlir_ciface_nextafter_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_nextafter_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:452:1: warning: _mlir_ciface_nextafter_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_nextafter_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:455:1: warning: _mlir_ciface_norm3d_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_norm3d_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:458:1: warning: _mlir_ciface_norm4d_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_norm4d_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:461:1: warning: _mlir_ciface_normcdf_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_normcdf_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:464:1: warning: _mlir_ciface_normcdfinv_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_normcdfinv_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:467:1: warning: _mlir_ciface_popc_i32.vector changed binding to STB_WEAK
.weak _mlir_ciface_popc_i32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:470:1: warning: _mlir_ciface_pow_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_pow_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:473:1: warning: _mlir_ciface_pow_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_pow_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:476:1: warning: _mlir_ciface_rcbrt_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_rcbrt_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:479:1: warning: _mlir_ciface_rcp_rd_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_rcp_rd_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:482:1: warning: _mlir_ciface_rcp_rn_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_rcp_rn_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:485:1: warning: _mlir_ciface_rcp_ru_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_rcp_ru_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:488:1: warning: _mlir_ciface_rcp_rz_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_rcp_rz_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:491:1: warning: _mlir_ciface_reciprocal_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_reciprocal_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:494:1: warning: _mlir_ciface_reciprocal_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_reciprocal_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:497:1: warning: _mlir_ciface_relu_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_relu_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:500:1: warning: _mlir_ciface_relu_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_relu_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:503:1: warning: _mlir_ciface_remainder_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_remainder_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:506:1: warning: _mlir_ciface_remquo_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_remquo_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:509:1: warning: _mlir_ciface_rhadd_i32.vector changed binding to STB_WEAK
.weak _mlir_ciface_rhadd_i32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:512:1: warning: _mlir_ciface_rhypot_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_rhypot_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:515:1: warning: _mlir_ciface_rint_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_rint_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:518:1: warning: _mlir_ciface_rint_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_rint_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:521:1: warning: _mlir_ciface_rnorm3d_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_rnorm3d_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:524:1: warning: _mlir_ciface_rnorm4d_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_rnorm4d_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:527:1: warning: _mlir_ciface_round_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_round_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:530:1: warning: _mlir_ciface_round_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_round_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:533:1: warning: _mlir_ciface_rsqrt_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_rsqrt_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:536:1: warning: _mlir_ciface_rsqrt_rn_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_rsqrt_rn_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:539:1: warning: _mlir_ciface_sad_i32.vector changed binding to STB_WEAK
.weak _mlir_ciface_sad_i32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:542:1: warning: _mlir_ciface_saturate_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_saturate_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:545:1: warning: _mlir_ciface_scalbn_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_scalbn_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:548:1: warning: _mlir_ciface_signbit_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_signbit_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:551:1: warning: _mlir_ciface_signbit_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_signbit_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:554:1: warning: _mlir_ciface_sin_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_sin_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:557:1: warning: _mlir_ciface_sin_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_sin_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:560:1: warning: _mlir_ciface_sincos_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_sincos_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:563:1: warning: _mlir_ciface_sincospi_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_sincospi_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:566:1: warning: _mlir_ciface_sinh_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_sinh_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:569:1: warning: _mlir_ciface_sinh_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_sinh_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:572:1: warning: _mlir_ciface_sinpi_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_sinpi_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:575:1: warning: _mlir_ciface_sqrt_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_sqrt_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:578:1: warning: _mlir_ciface_sqrt_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_sqrt_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:581:1: warning: _mlir_ciface_sqrt_rd_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_sqrt_rd_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:584:1: warning: _mlir_ciface_sqrt_rn_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_sqrt_rn_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:587:1: warning: _mlir_ciface_sqrt_ru_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_sqrt_ru_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:590:1: warning: _mlir_ciface_sqrt_rz_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_sqrt_rz_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:593:1: warning: _mlir_ciface_sub_rd_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_sub_rd_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:596:1: warning: _mlir_ciface_sub_rn_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_sub_rn_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:599:1: warning: _mlir_ciface_sub_ru_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_sub_ru_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:602:1: warning: _mlir_ciface_sub_rz_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_sub_rz_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:605:1: warning: _mlir_ciface_tan_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_tan_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:608:1: warning: _mlir_ciface_tan_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_tan_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:611:1: warning: _mlir_ciface_tanh_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_tanh_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:614:1: warning: _mlir_ciface_tanh_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_tanh_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:617:1: warning: _mlir_ciface_tgamma_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_tgamma_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:620:1: warning: _mlir_ciface_tgamma_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_tgamma_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:623:1: warning: _mlir_ciface_trunc_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_trunc_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:626:1: warning: _mlir_ciface_trunc_fp16.vector changed binding to STB_WEAK
.weak _mlir_ciface_trunc_fp16.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:629:1: warning: _mlir_ciface_uint2float_rd_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_uint2float_rd_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:632:1: warning: _mlir_ciface_uint2float_rn_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_uint2float_rn_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:635:1: warning: _mlir_ciface_uint2float_ru_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_uint2float_ru_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:638:1: warning: _mlir_ciface_uint2float_rz_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_uint2float_rz_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:641:1: warning: _mlir_ciface_uint_as_float_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_uint_as_float_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:644:1: warning: _mlir_ciface_ull2float_rd_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_ull2float_rd_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:647:1: warning: _mlir_ciface_ull2float_rn_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_ull2float_rn_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:650:1: warning: _mlir_ciface_ull2float_ru_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_ull2float_ru_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:653:1: warning: _mlir_ciface_ull2float_rz_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_ull2float_rz_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:656:1: warning: _mlir_ciface_umulhi_u32.vector changed binding to STB_WEAK
.weak _mlir_ciface_umulhi_u32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:659:1: warning: _mlir_ciface_y0_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_y0_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:662:1: warning: _mlir_ciface_y1_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_y1_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
<inline asm>:665:1: warning: _mlir_ciface_yn_fp32.vector changed binding to STB_WEAK
.weak _mlir_ciface_yn_fp32.vector
^
warning: cannot compile inline asm [-Winline-asm]
222 warnings generated.
