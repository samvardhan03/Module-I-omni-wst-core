import numpy as np
import omni_wst_core as wst
import pytest

@pytest.mark.skipif(not wst.cuda_available(), reason="CUDA not available")
def test_lipschitz_continuity_bound():
    """Mathematically validate that adversarial noise perturbations remain bounded by L_m <= (||psi||_1)^m."""
    
    configs = [
        wst.WSTConfig(J=4, Q=4, depth=1, jtfs=False),
        wst.WSTConfig(J=6, Q=8, depth=2, jtfs=False),
        wst.WSTConfig(J=8, Q=16, depth=2, jtfs=False)
    ]
    
    signal_len = 4096
    n_trials = 500
    noise_scale = 1e-3
    
    for cfg in configs:
        errors = []
        for _ in range(n_trials):
            x = np.random.randn(signal_len).astype(np.float32)
            noise = np.random.randn(signal_len).astype(np.float32) * noise_scale
            y = x + noise

            sx = wst.fingerprint(x, cfg)
            sy = wst.fingerprint(y, cfg)

            lhs = np.linalg.norm(sx - sy)
            rhs = (cfg.l1_norm_psi ** cfg.depth) * np.linalg.norm(x - y)

            assert lhs <= rhs + 1e-5, f"Lipschitz violation for J={cfg.J}, Q={cfg.Q}: {lhs} > {rhs}"
            errors.append(lhs / rhs)

        print(f"Mean empirical Lipschitz ratio for J={cfg.J}, Q={cfg.Q}: {np.mean(errors):.4f}")
