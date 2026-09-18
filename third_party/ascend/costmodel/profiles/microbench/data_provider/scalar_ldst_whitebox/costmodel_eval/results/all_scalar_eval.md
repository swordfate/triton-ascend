# 6-kernel scalar load/store 打分总表（matched + stage union，2026-09-18 代码修订后）

固定：`shape=(4,256,4,2)`, `BLOCK_X=64`, `superblock_factor=1`, `num_warps=1`；costmodel 用新代码/profile（全部 route = `all_simt_only`）。

CAModel：padded kernel `seed=12`（`bin_idx>0`），binned kernel `seed=0`（`bins[0]>0`）；取 `core0.veccore0` 一个 program。
`costmodel_matched` 只累加真正执行的 stage；`camodel_stage_union` 为每个 matched stage 的 union window 之和；不列 static。

| kernel | mode | 类别 | costmodel matched | CAModel stage-union | 误差 | matched stages | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| padded_copy_gather | simt | 直接标量 load | 1060.0 | 1113.0 | -4.8% | 2/2 |  |
| padded_copy_gather | simt | 间接标量 load | 530.0 | 509.0 | 4.1% | 1/1 | matched indirect stages; per-stage union |
| padded_copy_scatter | simt | 直接标量 load | 1060.0 | 1115.0 | -4.9% | 2/2 |  |
| padded_copy_scatter | simt | 间接标量 load | 1060.0 | 973.0 | 8.9% | 2/2 | matched indirect stages; per-stage union |
| padded_copy_wgrad | simt | 直接标量 load | 1060.0 | 897.0 | 18.2% | 2/2 |  |
| padded_copy_wgrad | simt | 间接标量 load | 530.0 | 481.0 | 10.2% | 1/1 | matched indirect stages; per-stage union |
| padded_copy_wgrad | simt | 标量 store | 450.0 | 559.0 | -19.5% | 1/1 | CAModel SIMT_STG |
| binned_copy_gather | simt | 直接标量 load | 530.0 | 497.0 | 6.6% | 1/2 | unmatched stage (control-flow) |
| binned_copy_gather | simt | 间接标量 load | 530.0 | 436.0 | 21.6% | 1/1 | matched indirect stages; per-stage union |
| binned_copy_scatter | simt | 直接标量 load | 530.0 | 497.0 | 6.6% | 1/2 | unmatched stage (control-flow) |
| binned_copy_scatter | simt | 间接标量 load | 530.0 | 436.0 | 21.6% | 1/1 | matched indirect stages; per-stage union |
| binned_copy_wgrad | simt | 直接标量 load | 530.0 | 442.0 | 19.9% | 1/2 | unmatched stage (control-flow) |
| binned_copy_wgrad | simt | 间接标量 load | 530.0 | 406.0 | 30.5% | 1/1 | matched indirect stages; per-stage union |
| binned_copy_wgrad | simt | 标量 store | 450.0 | 551.0 | -18.3% | 1/1 | CAModel SIMT_STG |

MAPE：direct ~10.2%，indirect ~16.2%，store ~18.9%；全部 <50%。
