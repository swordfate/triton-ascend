#loc = loc("/workspace/triton-ascend/docs/zh/examples/vecadd.py":9:0)
#loc9 = loc("x_ptr"(#loc))
#loc10 = loc("y_ptr"(#loc))
#loc11 = loc("output_ptr"(#loc))
#loc12 = loc("n_elements"(#loc))
module attributes {hacc.target = #hacc.target<"Ascend910B4-1">} {
  func.func @add_kernel(
    %arg0: memref<?xi8> loc("/workspace/triton-ascend/docs/zh/examples/vecadd.py":9:0), 
    %arg1: memref<?xi8> loc("/workspace/triton-ascend/docs/zh/examples/vecadd.py":9:0), 
    %x_ptr: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} loc("x_ptr"(#loc)), 
    %y_ptr: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} loc("y_ptr"(#loc)), 
    %output_ptr: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32} loc("output_ptr"(#loc)), 
    %n_elements: i32 {tt.divisibility = 16 : i32} loc("n_elements"(#loc)), 
    %arg6: i32 loc("/workspace/triton-ascend/docs/zh/examples/vecadd.py":9:0), 
    %arg7: i32 loc("/workspace/triton-ascend/docs/zh/examples/vecadd.py":9:0), 
    %arg8: i32 loc("/workspace/triton-ascend/docs/zh/examples/vecadd.py":9:0), 
    %arg9: i32 loc("/workspace/triton-ascend/docs/zh/examples/vecadd.py":9:0), 
    %arg10: i32 loc("/workspace/triton-ascend/docs/zh/examples/vecadd.py":9:0), 
    %arg11: i32 loc("/workspace/triton-ascend/docs/zh/examples/vecadd.py":9:0)) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %y = arith.constant 0.000000e+00 : f32 loc(#loc13)
    %x = arith.constant 1024 : index loc(#loc14)
    %block_start = arith.constant 1024 : i32 loc(#loc15)
    %block_start_0 = arith.muli %arg9, %block_start : i32 loc(#loc15)
    %block_start_1 = arith.index_cast %block_start_0 : i32 to index loc(#loc15)
    %x_2 = memref.reinterpret_cast %x_ptr to offset: [%block_start_1], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>> loc(#loc16)
    %x_3 = memref.alloc() : memref<1024xf32> loc(#loc14)
    %x_4 = arith.addi %block_start_1, %x : index loc(#loc14)
    %n_elements_5 = arith.index_cast %n_elements : i32 to index loc(#loc12)
    %x_6 = arith.maxsi %block_start_1, %n_elements_5 : index loc(#loc14)
    %x_7 = arith.minsi %x_4, %x_6 : index loc(#loc14)
    %x_8 = arith.subi %x_7, %block_start_1 : index loc(#loc14)
    %x_9 = arith.cmpi slt, %x_8, %x : index loc(#loc14)
    scf.if %x_9 {
      linalg.fill ins(%y : f32) outs(%x_3 : memref<1024xf32>) loc(#loc14)
    } {hivm.unlikely_condition} loc(#loc14)
    %x_10 = memref.subview %x_2[0] [%x_8] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>> loc(#loc14)
    %x_11 = memref.subview %x_3[0] [%x_8] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>> loc(#loc14)
    memref.copy %x_10, %x_11 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>> loc(#loc14)
    %x_12 = bufferization.to_tensor %x_3 restrict writable : memref<1024xf32> to tensor<1024xf32> loc(#loc14)
    %y_13 = memref.reinterpret_cast %y_ptr to offset: [%block_start_1], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>> loc(#loc17)
    %y_14 = memref.alloc() : memref<1024xf32> loc(#loc13)
    scf.if %x_9 {
      linalg.fill ins(%y : f32) outs(%y_14 : memref<1024xf32>) loc(#loc13)
    } {hivm.unlikely_condition} loc(#loc13)
    %y_15 = memref.subview %y_13[0] [%x_8] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>> loc(#loc13)
    %y_16 = memref.subview %y_14[0] [%x_8] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>> loc(#loc13)
    memref.copy %y_15, %y_16 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>> loc(#loc13)
    %y_17 = bufferization.to_tensor %y_14 restrict writable : memref<1024xf32> to tensor<1024xf32> loc(#loc13)
    %output = arith.addf %x_12, %y_17 : tensor<1024xf32> loc(#loc18)
    %reinterpret_cast = memref.reinterpret_cast %output_ptr to offset: [%block_start_1], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>> loc(#loc7)
    %extracted_slice = tensor.extract_slice %output[0] [%x_8] [1] : tensor<1024xf32> to tensor<?xf32> loc(#loc8)
    %subview = memref.subview %reinterpret_cast[0] [%x_8] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>> loc(#loc8)
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> () loc(#loc8)
    return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/workspace/triton-ascend/docs/zh/examples/vecadd.py":27:16)
#loc2 = loc("/workspace/triton-ascend/docs/zh/examples/vecadd.py":26:16)
#loc3 = loc("/workspace/triton-ascend/docs/zh/examples/vecadd.py":21:24)
#loc4 = loc("/workspace/triton-ascend/docs/zh/examples/vecadd.py":26:24)
#loc5 = loc("/workspace/triton-ascend/docs/zh/examples/vecadd.py":27:24)
#loc6 = loc("/workspace/triton-ascend/docs/zh/examples/vecadd.py":28:17)
#loc7 = loc("/workspace/triton-ascend/docs/zh/examples/vecadd.py":30:26)
#loc8 = loc("/workspace/triton-ascend/docs/zh/examples/vecadd.py":30:35)
#loc13 = loc("y"(#loc1))
#loc14 = loc("x"(#loc2))
#loc15 = loc("block_start"(#loc3))
#loc16 = loc("x"(#loc4))
#loc17 = loc("y"(#loc5))
#loc18 = loc("output"(#loc6))
