from triton.backends.ascend.empirical_superblock import (
    is_empirical_spill_penalty_enabled,
    penalty_by_factor,
)


def test_measured_penalty_entries():
    # padded_gather nw=2 F32 => 64 total warps; target F16.
    assert penalty_by_factor("_padded_copy_gather", 4096, 2) == {32: 1200.0}
    # padded_gather nw=1 F64 => 64 total warps; only the large-grid factor
    # is penalized, F32 remains available for the F16/F32 comparison.
    assert penalty_by_factor("_padded_copy_gather", 65536, 1) == {64: 1200.0}
    # binned_gather nw=2 F32 => 64 total warps; target F16.
    assert penalty_by_factor("_binned_copy_gather", 4096, 2) == {32: 800.0}


def test_unmeasured_configs_are_native():
    # For nw=1, F32 is the measured best for binned_wgrad/padded_wgrad.
    # Only F64 (64 total warps) receives the empirical penalty; F32 stays
    # untouched so the native route can keep it as the winner.
    assert penalty_by_factor("_binned_copy_wgrad", 4096, 1) == {64: 1200.0}
    assert penalty_by_factor("_padded_copy_wgrad", 4096, 1) == {64: 1500.0}
    # Unknown kernel or unknown shape falls back to the native decision.
    assert penalty_by_factor("unknown_kernel", 4096, 2) == {}
    assert penalty_by_factor("_padded_copy_gather", 4, 1) == {}


def test_penalty_factor_respects_legal_warp_limit():
    # nw=2 can never publish F64; the 64-total-warp point is F32.
    assert penalty_by_factor("_padded_copy_gather", 65536, 2) == {32: 1200.0}
    # nw=4 / F16 and nw=16 / F4 also reach 64 total warps and stay legal.
    assert penalty_by_factor("_padded_copy_gather", 65536, 4) == {16: 1200.0}
    assert penalty_by_factor("_padded_copy_gather", 65536, 16) == {4: 1200.0}
    assert 64 not in penalty_by_factor("_padded_copy_gather", 65536, 2)


def test_env_switch(monkeypatch):
    monkeypatch.delenv("TRITON_ASCEND_EMPIRICAL_SPILL_PENALTY", raising=False)
    monkeypatch.delenv("TRITON_ASCEND_EMPIRICAL_SUPERBLOCK", raising=False)
    assert is_empirical_spill_penalty_enabled()
    monkeypatch.setenv("TRITON_ASCEND_EMPIRICAL_SPILL_PENALTY", "0")
    assert not is_empirical_spill_penalty_enabled()
    assert penalty_by_factor("_padded_copy_gather", 4096, 2) == {}
    monkeypatch.setenv("TRITON_ASCEND_EMPIRICAL_SPILL_PENALTY", "1")
    assert is_empirical_spill_penalty_enabled()
