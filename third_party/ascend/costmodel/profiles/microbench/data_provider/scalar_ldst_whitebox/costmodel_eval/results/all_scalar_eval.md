# 6-kernel scalar load/store 打分总表（当前 profile fill=480，matched + stage union）

固定：`shape=(4,256,4,2)`, `BLOCK_X=64`, `superblock_factor=1`, `num_warps=1`；当前 route 全部 `all_simt_only`，CAModel 用 `compile_mode=simt_only`。

| kernel | mode | 类别 | costmodel matched | CAModel stage-union | 误差 | matched stages | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| padded_copy_gather | simt | direct_scalar_load | 972.0 | 1113.0 | -12.7% | 2/2 |  |
| padded_copy_gather | simt | indirect_scalar_load | 486.0 | 509.0 | -4.5% | 1/1 | matched indirect stages; per-stage union |
| padded_copy_scatter | simt | direct_scalar_load | 972.0 | 1115.0 | -12.8% | 2/2 |  |
| padded_copy_scatter | simt | indirect_scalar_load | 972.0 | 973.0 | -0.1% | 2/2 | matched indirect stages; per-stage union |
| padded_copy_wgrad | simt | direct_scalar_load | 972.0 | 897.0 | 8.4% | 2/2 |  |
| padded_copy_wgrad | simt | indirect_scalar_load | 486.0 | 481.0 | 1.0% | 1/1 | matched indirect stages; per-stage union |
| padded_copy_wgrad | simt | scalar_store | 450.0 | 559.0 | -19.5% | 1/1 | CAModel SIMT_STG |
| binned_copy_gather | simt | direct_scalar_load | 486.0 | 497.0 | -2.2% | 1/2 | unmatched stage (control-flow) |
| binned_copy_gather | simt | indirect_scalar_load | 486.0 | 436.0 | 11.5% | 1/1 | matched indirect stages; per-stage union |
| binned_copy_scatter | simt | direct_scalar_load | 486.0 | 497.0 | -2.2% | 1/2 | unmatched stage (control-flow) |
| binned_copy_scatter | simt | indirect_scalar_load | 486.0 | 436.0 | 11.5% | 1/1 | matched indirect stages; per-stage union |
| binned_copy_wgrad | simt | direct_scalar_load | 486.0 | 442.0 | 10.0% | 1/2 | unmatched stage (control-flow) |
| binned_copy_wgrad | simt | indirect_scalar_load | 486.0 | 406.0 | 19.7% | 1/1 | matched indirect stages; per-stage union |
| binned_copy_wgrad | simt | scalar_store | 450.0 | 551.0 | -18.3% | 1/1 | CAModel SIMT_STG |
