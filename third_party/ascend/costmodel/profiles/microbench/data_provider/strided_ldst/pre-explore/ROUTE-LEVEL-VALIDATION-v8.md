# Route-level validation report: v8 SIMT strided-load formula

## Status

- Executed on `ascend-950pr-63` from a clean clone of
  `feature/strided-load-store-costmodel-v4` (tip `6eb4ecd42`).
- A/B used the same C++ code for both runs and only toggled the temporary
  `simt_load_formula_version` field in
  `third_party/ascend/costmodel/profiles/simd_simt/david_v100_simd_simt_v1_v8.json`
  from 1 (v7 compatibility) to 2 (v8).
- A/B passed.  The production profile now enables v8 only; v7 fields and the
  formula switch have been deleted from C++, profile and schema.
- No board measurement was rerun and no coefficient was refit.

## Method

Report mode was exercised with `compile_mode=simd_simt`,
`auto_simt_scope_mode=report`, one fresh `TRITON_CACHE_DIR` per case, and the
formal `david_v100_simd_simt_v1.json` / temporary v8 profile.  For each case we
recorded the selected decision, all candidate totals, the strided Stage's
SIMT/SIMD `resource_system_cycles.load_per_iteration`, and the selected
SuperBlock factor.

Coverage:

- `block=8`, `stride=256`, `W=1/2/3`;
- `block=64`, `stride=32`, `W=1/2`;
- `block=128`, `stride=64`, `W=1/2`;
- `block=512`, `stride=256`, `W=1/2/3/6/12`;
- stride=3 small-tile route-boundary shapes (`block=32/64/128/256`, W=32,
  grid 8/4/2/1) plus the envelope target `block=64/grid=16`.

## Results

Units are SYS_CNT cycles.  `v7/v8 str load` is the strided Stage's SIMT
`load_per_iteration`; `all_simt total` is the all_simt_only candidate total.

| case | v7 decision | v8 decision | v7 str load | v8 str load | v7 all_simt | v8 all_simt |
|---|---|---|---:|---:|---:|---:|
| b8_s256_w1 | all_simt_only | all_simt_only | 207.8 | 244.0 | 603.4 | 639.6 |
| b8_s256_w2 | all_simd | all_simt_only | 272.2 | 241.9 | 667.8 | 637.6 |
| b8_s256_w3 | all_simd | all_simt_only | 336.5 | 239.9 | 732.2 | 635.5 |
| b64_s32_w1 | all_simd | all_simd | 495.9 | 588.6 | 905.3 | 998.0 |
| b64_s32_w2 | all_simd | all_simd | 495.9 | 559.5 | 905.3 | 968.9 |
| b128_s64_w1 | all_simd | all_simd | 495.9 | 694.4 | 924.9 | 1123.4 |
| b128_s64_w2 | all_simd | all_simd | 495.9 | 636.2 | 924.9 | 1065.2 |
| b512_s256_w1 | all_simt_only | all_simd | 495.9 | 1329.2 | 1066.9 | 1900.2 |
| b512_s256_w2 | all_simt_only | all_simd | 495.9 | 1096.4 | 1066.9 | 1667.4 |
| b512_s256_w3 | all_simt_only | all_simt_only | 495.9 | 863.6 | 1066.9 | 1434.5 |
| b512_s256_w6 | all_simt_only | all_simt_only | 495.9 | 630.7 | 1066.9 | 1201.7 |
| b512_s256_w12 | all_simt_only | all_simt_only | 495.9 | 630.7 | 1418.5 | 1201.7 |
| st_b32_s3_w32_g8 | all_simt_only | all_simt_only | 270.3 | 206.3 | 480.5 | 416.6 |
| st_b64_s3_w32_g4 | all_simt_only | all_simt_only | 270.3 | 206.3 | 480.5 | 416.6 |
| st_b128_s3_w32_g2 | all_simt_only | all_simt_only | 270.3 | 206.3 | 480.5 | 416.6 |
| st_b256_s3_w32_g1 | all_simt_only | all_simt_only | 270.3 | 206.3 | 496.3 | 432.4 |
| st_b64_s3_w32_g16 (envelope) | all_simt_only | all_simt_only | 495.9 | 510.5 | 897.5 | 912.1 |

## Acceptance assessment

1. **No route collapse.**  The legal candidate set is unchanged; mixed is
   unavailable for these shapes in both runs.  Selected-route changes are
   confined to the shapes where v7 was known to be severely miscalibrated.
2. **b8/s256:** v8 removes the v7 `dupL` overcharge for W2/W3, so the route
   moves from all_simd to all_simt; W1 stays all_simt.  This matches the
   block-8 group error reduction in the v4 fit (68.4% -> 14.2%).
3. **b512 large stride:** v8 corrects the v7 underprediction of 512-line
   single/double-warp latency.  W1/W2 move to all_simd; W3/W6/W12 remain
   all_simt.  The low-warp threshold is therefore route-insensitive above
   W=2 for this workload.
4. **Small-tile boundary:** all stride=3 shapes keep all_simt.  The envelope
   target `block=64/grid=16` keeps all_simt with 897.5 -> 912.1 (+1.6%);
   no collapse.
5. **Route-flip risk:** b8/s256 W2/W3 and b512/s256 W1/W2 are formula-driven
   changes, not measured in this task (board measurement was forbidden).
   They are accepted for enabling v8 because they are exactly the regions the
   v4 calibration was introduced to fix, but they remain the first candidates
   for a future in-domain board check.

## Post-deletion regression

After the v7 C++ fields and profile/schema entries were removed, the same 17
report cases were rebuilt with the final v8-only profile.  Comparing each
final report against the pre-deletion temporary v8 report:

```text
labels=17 max candidate diff=0.000e+00 max strided-load diff=0.000e+00
FINAL_VS_TEMP_V8_MATCH
```

This confirms that deleting the v7 compatibility path did not change v8
scoring.

## Residual risks

- Alignment / mask / negative stride / runtime stride / non-4B dtype remain
  outside the model's fact domain.
- The `max(0, 4-W)` threshold has no independent W=3/6/12 calibration point;
  route-level evidence only shows the b512/s256 route is insensitive above
  W=2.
- b512 single/double-warp large-span bank/set contention is still a known
  high-error tail.
- The profile now requires the five v8 fields; parsers or fixtures that still
  provide only a v7 profile will fail fast.  This is intentional.

## Artifacts

- Route A/B reports/logs: server `~/strided-v4-route/reports/`,
  `~/strided-v4-route/logs/`, summary
  `~/strided-v4-route/reports/route_ab_summary.csv`.
- Final-regression reports: server `~/strided-v4-route/reports_final/`,
  logs in `~/strided-v4-route/logs_final/`.
- A/B temporary profile was created during A/B and removed after the
  post-deletion regression; it is not committed.
- C++ UT: `SimdSimtCostModel` 47/47 passed, `CostModelPasses` 18/18 passed on
  the final v8-only build.
- Final profile: `david-v100-simd-simt-20261009-v30-strided-memory-v8`,
  schema 15.
