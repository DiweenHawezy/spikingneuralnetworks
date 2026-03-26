"""
Simple test suite for Transfer Entropy implementation.
Quick verification that TE works correctly with basic scenarios.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_basic_te_computation():
    """Test basic TE computation with correlated data."""
    from transfer_entropy_implementation import TransferEntropyCalculator
    
    np.random.seed(42)
    n = 1000
    x = np.random.randn(n)
    y = 0.8 * x[:-1] + 0.2 * np.random.randn(n-1)
    
    calc = TransferEntropyCalculator(n_bins=8, lag=1)
    te = calc.compute_transfer_entropy_joint(x, y)
    
    assert te > 0, f"TE should be positive for correlated data, got {te}"
    assert te < 2.0, f"TE should be reasonable, got {te}"


def test_basic_te_asymmetry():
    """Test TE asymmetry with simple data."""
    from transfer_entropy_implementation import TransferEntropyCalculator
    
    np.random.seed(42)
    n = 500
    x = np.random.randn(n)
    y = 0.6 * x[:-1] + 0.4 * np.random.randn(n-1)
    
    calc = TransferEntropyCalculator(n_bins=8, lag=1)
    te_xy = calc.compute_transfer_entropy_joint(x, y)
    te_yx = calc.compute_transfer_entropy_joint(y, x)
    
    # Should detect directionality
    assert te_xy != te_yx, "TE should be asymmetric"


def test_independent_data():
    """Test TE with independent data."""
    from transfer_entropy_implementation import TransferEntropyCalculator
    
    np.random.seed(42)
    n = 500
    x = np.random.randn(n)
    y = np.random.randn(n)
    
    calc = TransferEntropyCalculator(n_bins=8, lag=1)
    te = calc.compute_transfer_entropy_joint(x, y)
    
    # TE should be low for independent data
    assert te < 0.15, f"TE should be low for independent data, got {te}"


def test_multiple_lags():
    """Test TE with different lag values."""
    from transfer_entropy_implementation import TransferEntropyCalculator
    
    np.random.seed(42)
    n = 500
    x = np.random.randn(n)
    y = 0.7 * x[:-1] + 0.3 * np.random.randn(n-1)
    
    for lag in [1, 2, 3]:
        calc = TransferEntropyCalculator(n_bins=8, lag=lag)
        te = calc.compute_transfer_entropy_joint(x[:n-1], y)
        assert te >= 0, f"TE should be non-negative for lag={lag}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
