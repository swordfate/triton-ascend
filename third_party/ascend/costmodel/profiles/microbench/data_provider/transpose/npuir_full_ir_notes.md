# transpose 的 NPUIR 全量 IR 观察

方法同 `63-workspace/07-npuir-full-ir-dump/README.md`。

## 复现用 kernel

```python
@triton.jit
def trans_kernel(x_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    offs_m = pm * BM + tl.arange(0, BM)
    offs_n = pn * BN + tl.arange(0, BN)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])
    y = tl.trans(x)
    tl.store(y_ptr + offs_n[:, None] * M + offs_m[None, :], y)
```

SIMD、f32、2x2 block 32x32。

## IR 路径

1. TTIR / 早期 linalg：
   ```mlir
   linalg.transpose ins(%6 : tensor<32x32xf32>) outs(%7 : tensor<32x32xf32>) permutation = [1, 0]
   ```

2. 后续 HIVM：
   ```mlir
   hivm.hir.vtranspose ins(%12 : tensor<32x32xf32>) outs(%13 : tensor<32x32xf32>) permutation = [1, 0]
   ```

3. 再经过 `hivm-fuse-transpose-into-load` 之后，这个简单 load → transpose → store
   例子里的显式 `vtranspose` 被消掉，变成带 stride 的 load/store：
   - 源侧 reinterpret 成 `strides: [1, 64]`
   - 目的侧 reinterpret 成 `strides: [64, 1]`

也就是：

> transpose 不一定会留下一个独立的 `vtranspose` 指令；在 load/store 紧邻的场景，
> 它可以被表达成 strided load/store / 访存模式的变化。

## 对 cost model 的意义

- 当前 autoscope cost model 没有 transpose 专用 workload/rate；
- transpose 如果只是作为 `tt.trans` 出现在 TTIR，现有代码会把它算进 `generic.issue`；
- 从 full IR 看，transpose 的实际成本高度依赖它是否被 `FuseTransposeIntoLoad`
  吸收、是否变成 strided load/store，或者是否保留为 `vtranspose`；
- 所以不能简单照搬 sin/cos 的 “relative to f32.add” factor，需要进一步区分场景。
