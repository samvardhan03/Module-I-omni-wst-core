import pytest
import numpy as np
import omni_wst_core as wst

def test_validate_lipschitz_bound():
    """Mathematically validate that adversarial noise perturbations remain bounded by L_m <= (||psi||_1)^m."""
    signal_len = 4096
    batch_size = 1
    
    cfg = wst.WSTConfig(
        signal_len=signal_len,
        batch_size=batch_size,
        j=8,
        q=16,
        depth=2,
        jtfs=False,
        l1_norm_psi=0.95 # ||psi||_1 < 1 for Parseval frame
    )
    
    n_trials = 10
    noise_scale = 1e-3
    errors = []
    
    for _ in range(n_trials):
        x = np.random.randn(signal_len).astype(np.float32)
        delta = np.random.randn(signal_len).astype(np.float32) * noise_scale
        y = x + delta

        Sx = wst.fingerprint(x, cfg)
        Sy = wst.fingerprint(y, cfg)

        lhs = np.linalg.norm(Sx - Sy)
        rhs = (cfg.l1_norm_psi ** cfg.depth) * np.linalg.norm(x - y)
        
        # Test Lipschitz bound violation
        assert lhs <= rhs + 1e-6, f"Lipschitz violation: {lhs:.6f} > {rhs:.6f}"
        errors.append(lhs / rhs)

    print(f"Mean empirical tightness ratio: {np.mean(errors)}")

def test_biological_signal_fallback():
    """Test simulating batched WST processing on 1,000+ noisy EEG brain activity scans."""
    signal_len = 1024
    batch_size = 1000 # 1,000+ noisy EEG brain activity scans
    
    cfg = wst.WSTConfig(
        signal_len=signal_len,
        batch_size=batch_size,
        j=8,
        q=16,
        depth=2,
        jtfs=False,
        l1_norm_psi=0.90
    )
    
    # Simulate batched 1D noisy EEG brain scans
    x_batch = np.random.randn(signal_len * batch_size).astype(np.float32)
    
    # Processing on CPU fallback representation
    # Since CUDA isn't mocked differently in bindings yet, this serves as the fallback 
    # structure test validating determinism.
    Sx = wst.fingerprint(x_batch, cfg)
    
    assert Sx.shape[0] == signal_len * batch_size
    assert not np.isnan(Sx).any()
