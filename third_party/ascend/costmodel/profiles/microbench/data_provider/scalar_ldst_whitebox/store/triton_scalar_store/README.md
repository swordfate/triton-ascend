# Triton scalar store 到底走不走 MTE3？

## 结论

**走 MTE3，而且和“有没有类型转换/标量计算”无关。**
即使是最简单的 `tl.store(out_ptr, 1.0)`，CAModel 里也是：

```text
TTIR:      scalar tt.store(ptr, scalar)
TTAdapter: tensor<1xf32> + linalg.fill + bufferization.materialize_in_destination -> memref<1xf32>
CAModel:   SCALAR ST_XD_XN_IMM  accessUb:1, accessDdr:0   (写 UB)
           SET_FLAG PIPE:SCALAR -> MTE3
           MTE3  MOV_SRC_TO_DST_ALIGNv2  Src:UB, Dst:OUT (写 GM)
```

三个 mode 都没有出现 `ST_XD_XN_IMM accessDdr:1`（直接 GM scalar store）。

## 测试的 3 个 mode

文件：`triton_scalar_store_direct_demo.py`；`compile_mode=simd`，grid=4（const0 为 grid=1）。

| mode | kernel body | scalar ST | MTE3 |
|---|---|---|---|
| `const` | `tl.store(out_ptr + pid, 1.0)` | `ST_XD_XN_IMM accessUb:1` | ✅ `MOV_SRC_TO_DST_ALIGNv2 Src:UB Dst:OUT` |
| `const0` | `tl.store(out_ptr, 1.0)`（grid=1，连 addptr 都没有） | `ST_XD_XN_IMM accessUb:1` | ✅ 同样 |
| `add` | `v = pid.to(tl.float32) + 1.0; tl.store(out_ptr + pid, v)`（原 demo 模式） | `ST_XD_XN_IMM accessUb:1` | ✅ 同样，且前面多出 vector 计算链 |

每个 mode 的 `instr_log.dump` 中都只有 1 条 scalar `ST_XD_XN_IMM accessUb:1`、0 条 `accessDdr:1`，
`mte3_issque.dump` 中都有 `MOV_SRC_TO_DST_ALIGNv2`。

## 为什么

问题不在算子写法，而在 lowering：

1. TTIR 里 `tt.store` 确实是标量指针 + 标量值；
2. TTAdapter 做 bufferization 时，把它变成 `tensor<1xf32>` 的
   `bufferization.materialize_in_destination` → `memref<1xf32>`；
3. 后端对 materialize/1-element tile store 的默认实现就是
   **标量写 UB → MTE3 UB→GM**，所以 MTE3 必然出现。

`const` 和 `const0` 的 TTIR 里没有任何 `sitofp` / `addf` / 向量计算，仍然走 MTE3，可以直接排除
“类型转换/计算导致 MTE3”这个猜测。

## 单次 run 观测（`core0.veccore0`，不代表稳定硬件成本）

| mode | scalar ST push→retire | MTE3 事件 | MTE3 MOV push→retire |
|---|---:|---|---:|
| const | 16342 → 16354 | MOV 16359 → 16911 | 552 |
| const0 | 9452 → 9464 | MOV 9839 → 10296 | 457 |
| add | 5176 → 5188 | vector 算完后 MOV 5745 → 6255 | 510（不含等 vector） |

这里的时间只说明“MTE3 确实在干活”，不要当成性能参数。

## 文件

```text
triton_scalar_store/
  README.md                               # 本文件
  triton_scalar_store_demo.py             # 用户原始 demo（add + loop）
  triton_scalar_store_direct_demo.py      # const / const0 / cast / add / arg 五种最小 demo
  run_triton_scalar_store_camodel.sh      # 原 demo 的 CAModel runner
  run_triton_scalar_store_direct_camodel.sh
  camodel_results/
    const/    run.log + instr_log/mte3_issque/scalar_issque/biu_bwif + ttir/ttadapter
    const0/   同上
    add/      同上
```

关键 OPPROF：

- `const` : `OPPROF_20260918145240_IRWHSISHOJOHHTGT`
- `const0`: `OPPROF_20260918145431_HOZIONQGIEHTGLAW`
- `add`   : `OPPROF_20260918145642_VMUGVLSQPGNIROEN`
- 原始 demo 的 add：`triton_scalar_store_demo/OPPROF_20260918123720_IGLWXPHGPCWLMSZN`

## 对 cost model 的含义

Triton 的 scalar store 和 CCE 的 MainScalar direct GM store 不是同一条链：

```text
Triton:   SCALAR ST_XD_XN_IMM -> UB   (execTime≈12, accessUb:1)
          MTE3  MOV_SRC_TO_DST_ALIGNv2 UB -> GM
CCE:      (理想路径) ST_XD_XN_IMM -> GM, accessDdr:1
```

所以有两点要注意：

1. costmodel 的 `ScalarStore` 不能假设走 MainScalar direct store；`main_store_*` 已删除，改为
   MTE3 白盒 `T = 20 + 450 + (K-1)*480`；
2. CAModel 里 scalar store 的 12 cycles 只覆盖写 UB，真正的 GM 写要算在 MTE3 上；seeded 6-kernel
   验证中 SIMT route 的 store 白盒预测 450 vs CAModel `SIMT_STG` 551/559（-18%~-19%）。
