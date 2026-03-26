"""
Test suite for Transfer Entropy implementation.
Uses pytest for proper test structure with assertions.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_transfer_entropy_positive_for_correlated_data():
    """Test that TE is positive for correlated time series."""
    from transfer_entropy_implementation import TransferEntropyCalculator
    
    np.random.seed(42)
    n = 1000
    x = np.random.randn(n)
    y = 0.8 * x[:-1] + 0.2 * np.random.randn(n-1)  # y depends on past x
    
    calc = TransferEntropyCalculator(n_bins=8, lag=1)
    te = calc.compute_transfer_entropy_joint(x, y)
    
    assert te > 0, f"TE should be positive for correlated data, got {te}"


def test_transfer_entropy_asymmetric():
    """Test that TE is asymmetric (TE(X->Y) != TE(Y->X))."""
    from transfer_entropy_implementation import TransferEntropyCalculator
    
    np.random.seed(42)
    n = 1000
    x = np.random.randn(n)
    y = 0.8 * x[:-1] + 0.2 * np.random.randn(n-1)  # y depends on past x
    
    calc = TransferEntropyCalculator(n_bins=8, lag=1)
    te_forward = calc.compute_transfer_entropy_joint(x, y)  # X -> Y
    te_reverse = calc.compute_transfer_entropy_joint(y, x)  # Y -> X
    
    assert te_forward != te_reverse, "TE should be asymmetric"
    assert te_forward > te_reverse, "TE(X->Y) should be > TE(Y->X) when X causes Y"


def test_transfer_entropy_low_for_independent_data():
    """Test that TE is low (near zero) for independent time series."""
    from transfer_entropy_implementation import TransferEntropyCalculator
    
    np.random.seed(42)
    n = 1000
    x = np.random.randn(n)
    y = np.random.randn(n)  # Independent
    
    calc = TransferEntropyCalculator(n_bins=8, lag=1)
    te = calc.compute_transfer_entropy_joint(x, y)
    
    # TE should be close to zero for independent data
    assert te < 0.1, f"TE should be low for independent data, got {te}"


def test_transfer_entropy_with_different_lags():
    """Test TE calculation with different lag values."""
    from transfer_entropy_implementation import TransferEntropyCalculator
    
    np.random.seed(42)
    n = 1000
    x = np.random.randn(n)
    y = 0.8 * x[:-1] + 0.2 * np.random.randn(n-1)
    
    calc = TransferEntropyCalculator(n_bins=8, lag=1)
    
    # Test various lags
    te_values = []
    for lag in range(1, 5):
        if len(x) > lag:
            te = calc.compute_transfer_entropy_joint(x[:n-1], y, lag=lag)
            te_values.append(te)
            assert te >= 0, f"TE should be non-negative for lag={lag}, got {te}"
    
    assert len(te_values) > 0, "Should compute TE for at least one lag"


def test_transfer_entropy_with_different_bins():
    """Test TE calculation with different bin numbers."""
    from transfer_entropy_implementation import TransferEntropyCalculator
    
    np.random.seed(42)
    n = 1000
    x = np.random.randn(n)
    y = 0.8 * x[:-1] + 0.2 * np.random.randn(n-1)
    
    # Test various bin numbers
    for n_bins in [4, 8, 10]:
        calc = TransferEntropyCalculator(n_bins=n_bins, lag=1)
        te = calc.compute_transfer_entropy_joint(x, y)
        assert te >= 0, f"TE should be non-negative with n_bins={n_bins}"


def test_causal_direction_detection():
    """Test that we can detect causal direction correctly."""
    from transfer_entropy_implementation import TransferEntropyCalculator
    
    np.random.seed(42)
    n = 1000
    
    # Case 1: X causes Y
    x1 = np.random.randn(n)
    y1 = 0.7 * x1[:-1] + 0.3 * np.random.randn(n-1)
    
    calc = TransferEntropyCalculator(n_bins=8, lag=1)
    te_xy = calc.compute_transfer_entropy_joint(x1, y1)
    te_yx = calc.compute_transfer_entropy_joint(y1, x1)
    
    assert te_xy > te_yx, "Should detect X -> Y causation"
    
    # Case 2: Y causes X (reverse)
    x2 = 0.7 * y1[:-1] + 0.3 * np.random.randn(n-1)
    
    te_x2y1 = calc.compute_transfer_entropy_joint(x2, y1)
    te_y1x2 = calc.compute_transfer_entropy_joint(y1, x2)
    
    assert te_y1x2 > te_x2y1, "Should detect Y -> X causation in reverse case"


def test_discretization_valid_bins():
    """Test that discretization produces valid bin indices."""
    from transfer_entropy_implementation import TransferEntropyCalculator
    
    np.random.seed(42)
    n = 1000
    x = np.random.randn(n)
    
    calc = TransferEntropyCalculator(n_bins=8, lag=1)
    
    # Test discretization
    x_disc = calc._discretize(x)
    
    assert x_disc.min() >= 0, "Discretized values should be >= 0"
    assert x_disc.max() < calc.n_bins, "Discretized values should be < n_bins"
    assert x_disc.shape == x.shape, "Discretized shape should match input shape"


def test_input_length_validation():
    """Test that TE handles different input lengths correctly."""
    from transfer_entropy_implementation import TransferEntropyCalculator
    
    np.random.seed(42)
    n = 1000
    x = np.random.randn(n)
    y = np.random.randn(n)
    
    calc = TransferEntropyCalculator(n_bins=8, lag=1)
    
    # Test with slightly different lengths (should handle gracefully)
    te = calc.compute_transfer_entropy_joint(x, y[:n-1])
    assert te >= 0, "TE should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
