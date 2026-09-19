# 6-kernel scalar load/store 打分总表（forced SIMD，matched + stage union）

固定：`shape=(4,256,4,2)`, `BLOCK_X=64`, `superblock_factor=1`, `num_warps=1`；costmodel 用 SIMD implementation，CAModel 用 `compile_mode=simd`。

| kernel | mode | 类别 | costmodel matched | CAModel stage-union | 误差 | matched stages | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| padded_copy_gather | simd | direct_scalar_load | 894.0 | 883.0 | 1.2% | 2/2 |  |
| padded_copy_gather | simd | indirect_scalar_load | 450.0 | 533.0 | -15.6% | 1/1 | matched indirect stages; per-stage union |
| padded_copy_scatter | simd | direct_scalar_load | 894.0 | 1011.0 | -11.6% | 2/2 |  |
| padded_copy_scatter | simd | indirect_scalar_load | 897.0 | 930.0 | -3.5% | 2/2 | matched indirect stages; per-stage union |
| padded_copy_wgrad | simd | direct_scalar_load | 894.0 | 801.0 | 11.6% | 2/2 |  |
| padded_copy_wgrad | simd | indirect_scalar_load | 450.0 | 508.0 | -11.4% | 1/1 | matched indirect stages; per-stage union |
| padded_copy_wgrad | simd | scalar_store | 470.0 | 496.0 | -5.2% | 1/1 | CAModel MTE3 MOV |
| binned_copy_gather | simd | direct_scalar_load | 447.0 | 364.0 | 22.8% | 1/2 | unmatched stage (control-flow) |
| binned_copy_gather | simd | indirect_scalar_load | 447.0 | 528.0 | -15.3% | 1/1 | matched indirect stages; per-stage union |
| binned_copy_scatter | simd | direct_scalar_load | 447.0 | 475.0 | -5.9% | 1/2 | unmatched stage (control-flow) |
| binned_copy_scatter | simd | indirect_scalar_load | 447.0 | 425.0 | 5.2% | 1/1 | matched indirect stages; per-stage union |
| binned_copy_wgrad | simd | direct_scalar_load | 447.0 | 510.0 | -12.4% | 1/2 | unmatched stage (control-flow) |
| binned_copy_wgrad | simd | indirect_scalar_load | 447.0 | 493.0 | -9.3% | 1/1 | matched indirect stages; per-stage union |
| binned_copy_wgrad | simd | scalar_store | 470.0 | 413.0 | 13.8% | 1/1 | CAModel MTE3 MOV |
