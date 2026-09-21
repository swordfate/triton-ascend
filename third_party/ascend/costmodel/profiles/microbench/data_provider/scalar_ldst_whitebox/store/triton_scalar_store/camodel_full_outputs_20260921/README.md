# Triton scalar store（MTE3）CAModel 完整结果（2026-09-21 重跑）

## 内容

本目录是 `triton_scalar_store_demo.py` 的完整 CAModel 输出：

- Triton scalar `tt.store` 在 `compile_mode=simd` 下的 MTE3 链路；
- `N_ST=1`、`GRID=4`、`num_warps=1`、`superblock_factor=1`；
- 一个 program 一条标量 store，验证 `scalar → UB staging → MTE3 MOV UB→OUT`。

## 运行命令

服务器：`ascend-950pr-63`，CANN 9.1.0，`msopprof simulator`。

```bash
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
ulimit -n 1048576

cd ~/triton_mte3_scalar_store_camodel_20260921/n_st1
N_ST=1 GRID=4 bash run_triton_scalar_store_camodel.sh
```

等价核心命令：

```bash
msopprof simulator --soc-version=Ascend950PR_9599 --core-id=0 \
  --launch-count=1 --timeout=10 --kernel-name=_triton_scalar_store_demo \
  python3 triton_scalar_store_demo.py --n-st 1 --grid 4
```

## 结果文件

```text
OPPROF_20260921143501_XGODAFDWOXMIJVTR.tar.gz
  OPPROF_20260921143501_XGODAFDWOXMIJVTR/
    dump/
    simulator/
      core0.veccore0/
      core0.veccore1/
```

解压：

```bash
tar -xzf OPPROF_20260921143501_XGODAFDWOXMIJVTR.tar.gz
```

- `run.log`：msopprof 完整 stdout/stderr；
- `triton_scalar_store_demo.py` / `run_triton_scalar_store_camodel.sh`：本次运行源码与 runner；
- `SHA256SUMS`：本目录校验和。

## 关键时间线（`core0.veccore0`，block0）

| 事件 | 时间戳 | 说明 |
|---|---|---|
| `SCALAR ST_XD_XN_IMM accessUb:1` push | 6132 | scalar 值写 UB staging |
| 同指令 retire | 6144 | `execTime=12` |
| MTE3 `WAIT_FLAG` push / release | 6153 / 6700 | 等 VEC flag |
| MTE3 `MOV_SRC_TO_DST_ALIGNv2` push | 6157 | UB→OUT 写 GM |
| MTE3 `MOV` retire | 7211 | push→retire **1054** cycle |
| 整条短链路（scalar push → MTE3 retire） | — | 7211 − 6132 = **1079** cycle |

- CAModel 内部为 1.8 GHz：1079 cycle ≈ 599.4 ns，1054 cycle ≈ 585.6 ns；
- `dump/` 中可见 `ST_XD_XN_IMM accessUb:1, accessDdr:0` 和 `MOV_SRC_TO_DST_ALIGNv2 Src:UB,Dst:OUT`，没有 `ST_XD_XN_IMM accessDdr:1`；
- `msopprof` 在进程退出阶段报 `Child process killed by signal 11`（该 demo 的已知退出行为），
  但 `Start parse dump file`、`Profiling results saved`、`Op profiling finish` 均已完成，OPPROF 包完整。

## 校验和

见 `SHA256SUMS`。其中 OPPROF 包：

```text
bc3aaf3515b345b534210d30d811004734be7c359ba0d9b91955c5de8a9a578c  OPPROF_20260921143501_XGODAFDWOXMIJVTR.tar.gz
```
