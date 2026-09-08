# NPUIR 全量 IR 中的 sin 展开观察

方法：`TRITON_DEBUG=1` 抓 `kernel.mlir`，再用 `bishengir-compile --mlir-print-ir-after-all --enable-tuning-mode` 打印全量 IR。

## 路径

1. TTIR：`math.sin`
2. HFusion：`hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>}`
3. 后续出现 range reduction 常量（pi 等）和 polynomial 求值。

实际 vector 展开示例：

```mlir
%7 = arith.addf %3, %cst_3          // -0.166666672
%8 = arith.mulf %7, %4
%9 = arith.addf %8, %cst_2          // 1.0
%10 = arith.mulf %9, %5
%11 = arith.mulf %10, %6
...
%14 = arith.mulf %1, %1
%15 = arith.cmpf olt, %14, %cst
%16 = arith.select %15, %1, %13
```

## 结论

- `tl.sin` 在 SIMD 路径下不是单条硬件 sin；
- 实际是 range reduction + polynomial / select；
- 因此 CCE 直接写 sin demo 很难代表真实 Triton 路径；
- 用 Triton 端到端标定是更合理的验证方式。
