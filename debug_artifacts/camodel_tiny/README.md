# padded_gather F16/F32 CAModel tiny full output

These two tarballs contain the complete `msprof op simulator` / CAModel output
for the forced whole-kernel SuperBlock factors on a reduced but structurally
representative `_padded_copy_gather` shape.

## Configuration

- kernel: `npu_padded_copy_gather._padded_copy_gather`
- shape: `sl=8, hs=1536, ne=8, top_k=4` (grid=32)
- `BLOCK_X=256`, `num_warps=2`
- `F16`: `superblock_factor=16`, per-thread stack TLV `(9, 4, 0)`
- `F32`: `superblock_factor=32`, per-thread stack TLV `(9, 4, 48)`
- simulator: `msopprof simulator --soc-version=Ascend950PR_9599`
- output: full `OPPROF_*` directories (`dump/`, `simulator/`) plus cache/run logs

## Direct evidence

`padded_camodel_tiny_force32.tgz` contains SIMT stack spill/reload
microinstructions:

- `SIMT_STK` (store to per-thread stack): 832 occurrences on `core0.veccore0`
- `SIMT_LDK` (load from per-thread stack): 166 occurrences on `core0.veccore0`
- per warp: 13 `SIMT_STK` + 2 `SIMT_LDK` (some warps: 21 `SIMT_LDK`)

`padded_camodel_tiny_force16.tgz` contains none:

- `SIMT_STK = 0`
- `SIMT_LDK = 0`
- per-thread stack TLV = 0

## SHA256

```text
9bb145d6e277480bc0ab564fc6d5b36558909a18f644b004dd6f19fc8627250d  padded_camodel_tiny_force16.tgz
91b76084d665c3a93043fe2e4191ae3fcb219df1e5e7fc4fbcc9d5ab1fd9e35d  padded_camodel_tiny_force32.tgz
```

## How to inspect

```bash
tar -xzf padded_camodel_tiny_force32.tgz
grep -R "SIMT_STK" tiny_force32/OPPROF_*/dump/ | head
grep -R "SIMT_LDK" tiny_force32/OPPROF_*/dump/ | head

tar -xzf padded_camodel_tiny_force16.tgz
grep -R "SIMT_STK" tiny_force16/OPPROF_*/dump/ | wc -l   # expect 0
grep -R "SIMT_LDK" tiny_force16/OPPROF_*/dump/ | wc -l   # expect 0
```
