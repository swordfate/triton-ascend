# CAModel SIMD sin/cos 观察

日期：2026-09-07  
机器：Ascend950PR  
方式：极小 SIMD Triton kernel + `msprof op simulator`

## 测试 kernel

- `copy_kernel`：`y = x`，作为 load/store baseline
- `add_kernel`：`y = x + 1`
- `sin_kernel`：`y = tl.sin(x)`

## simulator cycle 汇总（core0.veccore0_instr_exe.csv）

| BLOCK | copy 总 cycles | add 总 cycles | sin 总 cycles |
|---|---:|---:|---:|
| 128 | 3519 | 4822 | 7638 |
| 1024 | 3778 | 5562 | 23655 |

扣除 copy baseline 后：

| BLOCK | add compute cycles | sin compute cycles | sin/add compute ratio |
|---|---:|---:|---:|
| 128 | 1303 | 4119 | 3.16 |
| 1024 | 1784 | 19877 | 11.14 |

## 结论

- 小尺寸下 sin 额外开销不明显；
- 大尺寸下 sin 相对 add 的 compute 倍数约 11；
- 这与 Triton 端到端 slope 的约 14.8 在同一量级；
- 当前 profile SIMD factor=15 处于合理范围，但仍是低置信度。

## 指令特征

1024 元素 sin 相对 add 增加的主要指令：

- `RV_VMUL` / `RV_VMULS`
- `RV_VADD` / `RV_VADDS`
- 以及大量整数/位操作：`MOV_XD_IMM`、`RV_SMOVI`、`MOVK`、`RV_VAND`、`RV_VSHRI`

说明 sin 不是一条硬件指令，而是 range reduction + polynomial 实现。
