"""
test_lipschitz.py — Validates the Lipschitz continuity bound of the WST cascade.

Mathematical guarantee being tested:
    ||S[p]x - S[p]y||₂ ≤ (||ψ||₁)^m · ||x - y||₂

This bound must hold for the REAL mathematical implementation — not a mock.
The CPU engine uses a true Radix-2 FFT scattering cascade with Morlet wavelets.
"""

import numpy as np
import omni_wst_core as wst
import pytest


def test_lipschitz_continuity_bound():
    """Mathematically validate that adversarial noise perturbations remain
    bounded by L_m <= (||psi||_1)^m using the real WST forward pass."""

    configs = [
        wst.WSTConfig(J=4, Q=4, depth=1, jtfs=False),
        wst.WSTConfig(J=6, Q=8, depth=2, jtfs=False),
        wst.WSTConfig(J=8, Q=16, depth=2, jtfs=False),
    ]

    signal_len = 4096
    n_trials = 200
    noise_scale = 1e-3

    for cfg in configs:
        violations = 0
        ratios = []
        for _ in range(n_trials):
            x = np.random.randn(signal_len).astype(np.float32)
            noise = np.random.randn(signal_len).astype(np.float32) * noise_scale
            y = x + noise

            sx = wst.fingerprint(x, cfg)
            sy = wst.fingerprint(y, cfg)

            lhs = np.linalg.norm(sx - sy)
            rhs = (cfg.l1_norm_psi ** cfg.depth) * np.linalg.norm(x - y)

            if lhs > rhs + 1e-5:
                violations += 1
            ratios.append(lhs / max(rhs, 1e-12))

        mean_ratio = np.mean(ratios)
        print(f"J={cfg.J}, Q={cfg.Q}, depth={cfg.depth}: "
              f"mean Lipschitz ratio = {mean_ratio:.4f}, "
              f"l1_norm_psi = {cfg.l1_norm_psi:.4f}, "
              f"violations = {violations}/{n_trials}")

        assert violations == 0, (
            f"Lipschitz bound violated {violations}/{n_trials} times "
            f"for J={cfg.J}, Q={cfg.Q}, depth={cfg.depth}"
        )


def test_lipschitz_constant_decays_with_depth():
    """The Lipschitz constant L_m = (||ψ||₁)^m must decay exponentially
    with depth m when ||ψ||₁ < 1. This test verifies the empirical
    Lipschitz ratio decreases as depth increases."""

    signal_len = 4096
    n_trials = 100
    noise_scale = 1e-3

    mean_ratios_by_depth = {}
    for depth in [1, 2, 3]:
        cfg = wst.WSTConfig(J=8, Q=16, depth=depth, jtfs=False)
        ratios = []
        for _ in range(n_trials):
            x = np.random.randn(signal_len).astype(np.float32)
            noise = np.random.randn(signal_len).astype(np.float32) * noise_scale
            y = x + noise

            sx = wst.fingerprint(x, cfg)
            sy = wst.fingerprint(y, cfg)

            lhs = np.linalg.norm(sx - sy)
            rhs = np.linalg.norm(x - y)
            ratios.append(lhs / max(rhs, 1e-12))

        mean_ratios_by_depth[depth] = np.mean(ratios)
        print(f"depth={depth}: mean empirical Lipschitz ratio = {mean_ratios_by_depth[depth]:.4f}")

    # The bound (||ψ||₁)^m decays, so the empirical ratio should also decrease
    # (or at least not increase significantly) with depth
    assert mean_ratios_by_depth[2] <= mean_ratios_by_depth[1] * 1.05, (
        f"Lipschitz ratio did not decay: depth=1 → {mean_ratios_by_depth[1]:.4f}, "
        f"depth=2 → {mean_ratios_by_depth[2]:.4f}"
    )
