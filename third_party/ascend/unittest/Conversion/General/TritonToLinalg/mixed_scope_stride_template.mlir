// RUN: triton-opt %s --triton-to-unstructure='compile-on-910-95=true' \
// RUN:   --triton-to-linalg='compile-on-910-95=true compile-mode=simd_simt' --split-input-file \
// RUN:   | FileCheck %s

// -----
// Model-controlled mixed route: a rank1 static non-power-of-two stride=3 load
// inside an explicit local SIMT scope must take the triton_stride_load
// template path.  The function carries the cost-model execution decision but
// does not need to run the cost-model pass for this lowering contract.
// CHECK-LABEL: func.func @mixed_scope_stride3_load
// CHECK: call @triton_stride_load
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mixed_scope_stride3_load(
      %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32})
      attributes {ascend.simt_costmodel.effective = "mixed_simd_simt"} {
    %c3 = arith.constant dense<3> : tensor<256xi32>
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %stride = arith.muli %range, %c3 : tensor<256xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %stride : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %loaded = scope.scope : () -> tensor<256xf32> {
      %v = tt.load %ptrs : tensor<256x!tt.ptr<f32>>
      scope.return %v : tensor<256xf32>
    } {vector_mode = "simt"}
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %range : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %out_ptrs, %loaded : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Same local SIMT scope but with a power-of-two stride=4 access.  The backend
// must keep the structured SIMD path and must not emit the template call.
// CHECK-LABEL: func.func @mixed_scope_stride4_load_control
// CHECK-NOT: call @triton_stride_load
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mixed_scope_stride4_load_control(
      %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32})
      attributes {ascend.simt_costmodel.effective = "mixed_simd_simt"} {
    %c4 = arith.constant dense<4> : tensor<256xi32>
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %stride = arith.muli %range, %c4 : tensor<256xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %stride : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %loaded = scope.scope : () -> tensor<256xf32> {
      %v = tt.load %ptrs : tensor<256x!tt.ptr<f32>>
      scope.return %v : tensor<256xf32>
    } {vector_mode = "simt"}
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %range : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %out_ptrs, %loaded : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Store variant: the local SIMT scope owns the strided store anchor.
// CHECK-LABEL: func.func @mixed_scope_stride3_store
// CHECK: call @triton_stride_store
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mixed_scope_stride3_store(
      %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32})
      attributes {ascend.simt_costmodel.effective = "mixed_simd_simt"} {
    %c3 = arith.constant dense<3> : tensor<256xi32>
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %stride = arith.muli %range, %c3 : tensor<256xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %stride : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %value = arith.constant dense<1.000000e+00> : tensor<256xf32>
    scope.scope : () -> () {
      tt.store %ptrs, %value : tensor<256x!tt.ptr<f32>>
      scope.return
    } {vector_mode = "simt"}
    tt.return
  }
}


// -----
// Control dispatch matrix for the remaining stride classes.  Only static
// non-power-of-two stride >= 3 may enter the template; these functions must
// remain on the structured/deinterleave paths.
// CHECK-LABEL: func.func @mixed_scope_stride1_load_control
// CHECK-NOT: call @triton_stride_load
// CHECK-LABEL: func.func @mixed_scope_stride2_load_control
// CHECK-NOT: call @triton_stride_load
// CHECK-LABEL: func.func @mixed_scope_stride8_load_control
// CHECK-NOT: call @triton_stride_load
// CHECK-LABEL: func.func @mixed_scope_stride256_load_control
// CHECK-NOT: call @triton_stride_load
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mixed_scope_stride1_load_control(
      %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32})
      attributes {ascend.simt_costmodel.effective = "mixed_simd_simt"} {
    %c1 = arith.constant dense<1> : tensor<256xi32>
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %stride = arith.muli %range, %c1 : tensor<256xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %stride : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %loaded = scope.scope : () -> tensor<256xf32> {
      %v = tt.load %ptrs : tensor<256x!tt.ptr<f32>>
      scope.return %v : tensor<256xf32>
    } {vector_mode = "simt"}
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %range : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %out_ptrs, %loaded : tensor<256x!tt.ptr<f32>>
    tt.return
  }

  tt.func public @mixed_scope_stride2_load_control(
      %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32})
      attributes {ascend.simt_costmodel.effective = "mixed_simd_simt"} {
    %c2 = arith.constant dense<2> : tensor<256xi32>
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %stride = arith.muli %range, %c2 : tensor<256xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %stride : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %loaded = scope.scope : () -> tensor<256xf32> {
      %v = tt.load %ptrs : tensor<256x!tt.ptr<f32>>
      scope.return %v : tensor<256xf32>
    } {vector_mode = "simt"}
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %range : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %out_ptrs, %loaded : tensor<256x!tt.ptr<f32>>
    tt.return
  }

  tt.func public @mixed_scope_stride8_load_control(
      %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32})
      attributes {ascend.simt_costmodel.effective = "mixed_simd_simt"} {
    %c8 = arith.constant dense<8> : tensor<256xi32>
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %stride = arith.muli %range, %c8 : tensor<256xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %stride : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %loaded = scope.scope : () -> tensor<256xf32> {
      %v = tt.load %ptrs : tensor<256x!tt.ptr<f32>>
      scope.return %v : tensor<256xf32>
    } {vector_mode = "simt"}
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %range : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %out_ptrs, %loaded : tensor<256x!tt.ptr<f32>>
    tt.return
  }

  tt.func public @mixed_scope_stride256_load_control(
      %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32})
      attributes {ascend.simt_costmodel.effective = "mixed_simd_simt"} {
    %c256 = arith.constant dense<256> : tensor<256xi32>
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %stride = arith.muli %range, %c256 : tensor<256xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %stride : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %loaded = scope.scope : () -> tensor<256xf32> {
      %v = tt.load %ptrs : tensor<256x!tt.ptr<f32>>
      scope.return %v : tensor<256xf32>
    } {vector_mode = "simt"}
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %range : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %out_ptrs, %loaded : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
