"""
test_dispatch.py — Dynamic Template Instantiation Dispatcher Validation Suite

Validates that the DISPATCH_FINGERPRINT macro in wst_bindings.cu correctly
routes runtime (J, Q) parameters to the appropriate pre-compiled
WSTEngine<HopperTag, J, Q> template instantiation.

Test categories:
  1. Positive dispatch: each supported (J, Q) pair produces a valid tensor.
  2. Negative dispatch: unsupported (J, Q) pairs raise a clear error.
  3. Cross-config isolation: different (J, Q) pairs produce distinct outputs.
"""

import numpy as np
import omni_wst_core as wst
import pytest


# ---- Fixtures ----

VALID_CONFIGS = [
    (8, 16),
    (10, 16),
    (8, 8),
]

UNSUPPORTED_CONFIGS = [
    (99, 99),
    (4, 32),
    (16, 4),
    (7, 7),
]


# ---- Positive Dispatch Tests ----

@pytest.mark.parametrize("j, q", VALID_CONFIGS)
def test_dispatch_valid_config_produces_tensor(j, q):
    """Each supported (J, Q) pair must produce a finite, non-zero tensor."""
    cfg = wst.WSTConfig(J=j, Q=q, depth=2, jtfs=False)
    signal = np.random.randn(4096).astype(np.float32)

    result = wst.fingerprint(signal, cfg)

    assert result is not None, f"fingerprint returned None for (J={j}, Q={q})"
    assert result.shape == (4096,), f"Unexpected shape {result.shape} for (J={j}, Q={q})"
    assert np.all(np.isfinite(result)), f"Non-finite values in output for (J={j}, Q={q})"


@pytest.mark.parametrize("j, q", VALID_CONFIGS)
def test_dispatch_determinism(j, q):
    """Repeated calls with identical input and config must yield identical output."""
    cfg = wst.WSTConfig(J=j, Q=q, depth=2, jtfs=False)
    signal = np.random.randn(4096).astype(np.float32)

    result_a = wst.fingerprint(signal, cfg)
    result_b = wst.fingerprint(signal, cfg)

    np.testing.assert_array_equal(
        result_a, result_b,
        err_msg=f"Non-deterministic output for (J={j}, Q={q})"
    )


@pytest.mark.parametrize("j, q", VALID_CONFIGS)
def test_dispatch_batch_mode(j, q):
    """Batch (2D) input must dispatch through the same template path."""
    cfg = wst.WSTConfig(J=j, Q=q, depth=2, jtfs=False)
    batch = np.random.randn(4, 4096).astype(np.float32)

    result = wst.fingerprint(batch, cfg)

    assert result.shape == (4, 4096), (
        f"Batch shape mismatch for (J={j}, Q={q}): got {result.shape}"
    )
    assert np.all(np.isfinite(result)), f"Non-finite batch output for (J={j}, Q={q})"


# ---- Negative Dispatch Tests ----

@pytest.mark.parametrize("j, q", UNSUPPORTED_CONFIGS)
def test_dispatch_unsupported_config_raises(j, q):
    """Unsupported (J, Q) pairs must raise a clear error — never silently
    fall back to a hardcoded template or produce garbage output."""
    cfg = wst.WSTConfig(J=j, Q=q, depth=2, jtfs=False)
    signal = np.random.randn(4096).astype(np.float32)

    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        wst.fingerprint(signal, cfg)

    # Verify the error message is diagnostic, not a generic segfault
    error_msg = str(exc_info.value).lower()
    assert "unsupported" in error_msg or "dispatch" in error_msg, (
        f"Error message for (J={j}, Q={q}) is not diagnostic: {exc_info.value}"
    )


# ---- Cross-Config Isolation Test ----

def test_dispatch_cross_config_produces_distinct_outputs():
    """Different (J, Q) pairs operating on the same input must produce
    numerically distinct scattering coefficients, proving the dispatch
    macro actually routed to different template instantiations."""
    signal = np.random.randn(4096).astype(np.float32)

    cfg_a = wst.WSTConfig(J=8, Q=16, depth=2, jtfs=False)
    cfg_b = wst.WSTConfig(J=10, Q=16, depth=2, jtfs=False)
    cfg_c = wst.WSTConfig(J=8, Q=8, depth=2, jtfs=False)

    result_a = wst.fingerprint(signal, cfg_a)
    result_b = wst.fingerprint(signal, cfg_b)
    result_c = wst.fingerprint(signal, cfg_c)

    # At least one pair must differ (they use different filter banks)
    assert not np.array_equal(result_a, result_b), (
        "J=8,Q=16 and J=10,Q=16 produced identical output — dispatch likely hardcoded"
    )
    assert not np.array_equal(result_a, result_c), (
        "J=8,Q=16 and J=8,Q=8 produced identical output — dispatch likely hardcoded"
    )
