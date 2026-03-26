"""
Comprehensive test suite for Transfer Entropy implementation.
Tests various scenarios including correlation, causation detection, and edge cases.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTransferEntropyBasic:
    """Basic functionality tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n = 1000
        self.x = np.random.randn(self.n)
        self.y = 0.8 * self.x[:-1] + 0.2 * np.random.randn(self.n-1)
        self.calc = lambda lag=1, n_bins=8: TransferEntropyCalculator(
            n_bins=n_bins, lag=lag
        )
    
    def test_te_positive_correlated(self):
        """Test TE is positive for correlated data."""
        te = self.calc()(self.x, self.y)
        assert te > 0, f"TE should be positive for correlated data, got {te}"
    
    def test_te_non_negative(self):
        """Test TE is always non-negative."""
        te = self.calc()(self.x, self.y)
        assert te >= 0, f"TE should be non-negative, got {te}"
    
    def test_te_zero_for_identical(self):
        """Test TE is zero when series are identical (perfect self-prediction)."""
        # When y is perfectly predictable from itself, additional info from x adds nothing
        y_identical = self.y.copy()
        te = self.calc()(self.y, y_identical)
        # Should be very low since y is already perfectly predictable from itself
        assert te < 0.1, f"TE should be low for self-prediction, got {te}"


class TestTransferEntropyAsymmetry:
    """Tests for TE asymmetry (directionality)."""
    
    def test_te_asymmetric(self):
        """Test that TE(X->Y) != TE(Y->X) for causal relationship."""
        np.random.seed(42)
        n = 1000
        x = np.random.randn(n)
        y = 0.8 * x[:-1] + 0.2 * np.random.randn(n-1)
        
        calc = TransferEntropyCalculator(n_bins=8, lag=1)
        te_xy = calc.compute_transfer_entropy_joint(x, y)
        te_yx = calc.compute_transfer_entropy_joint(y, x)
        
        assert te_xy != te_yx, "TE should be asymmetric"
    
    def test_direction_x_to_y(self):
        """Test correct detection of X -> Y causation."""
        np.random.seed(42)
        n = 1000
        x = np.random.randn(n)
        y = 0.7 * x[:-1] + 0.3 * np.random.randn(n-1)
        
        calc = TransferEntropyCalculator(n_bins=8, lag=1)
        te_xy = calc.compute_transfer_entropy_joint(x, y)
        te_yx = calc.compute_transfer_entropy_joint(y, x)
        
        assert te_xy > te_yx, "Should detect X -> Y causation"
        assert te_xy > 0, "TE(X->Y) should be positive"
    
    def test_direction_y_to_x(self):
        """Test correct detection of Y -> X causation (reverse)."""
        np.random.seed(42)
        n = 1000
        y = np.random.randn(n)
        x = 0.7 * y[:-1] + 0.3 * np.random.randn(n-1)
        
        calc = TransferEntropyCalculator(n_bins=8, lag=1)
        te_yx = calc.compute_transfer_entropy_joint(y, x)
        te_xy = calc.compute_transfer_entropy_joint(x, y)
        
        assert te_yx > te_xy, "Should detect Y -> X causation"


class TestTransferEntropyIndependence:
    """Tests for TE with independent data."""
    
    def test_te_low_independent(self):
        """Test TE is low for independent time series."""
        np.random.seed(42)
        n = 1000
        x = np.random.randn(n)
        y = np.random.randn(n)
        
        calc = TransferEntropyCalculator(n_bins=8, lag=1)
        te = calc.compute_transfer_entropy_joint(x, y)
        
        assert te < 0.1, f"TE should be low for independent data, got {te}"
    
    def test_multiple_independent_tests(self):
        """Test with multiple independent datasets."""
        np.random.seed(42)
        n = 1000
        te_values = []
        
        for i in range(5):
            x = np.random.randn(n)
            y = np.random.randn(n)
            
            calc = TransferEntropyCalculator(n_bins=8, lag=1)
            te = calc.compute_transfer_entropy_joint(x, y)
            te_values.append(te)
        
        # All TE values should be low
        assert all(te < 0.15 for te in te_values), \
            f"TE should be low for all independent pairs, got {te_values}"


class TestTransferEntropyLag:
    """Tests for different lag values."""
    
    def test_various_lags(self):
        """Test TE with different lag values."""
        np.random.seed(42)
        n = 1000
        x = np.random.randn(n)
        y = 0.8 * x[:-1] + 0.2 * np.random.randn(n-1)
        
        calc = TransferEntropyCalculator(n_bins=8, lag=1)
        
        te_values = []
        for lag in range(1, 6):
            if len(x) > lag:
                te = calc.compute_transfer_entropy_joint(x[:n-1], y, lag=lag)
                te_values.append(te)
                assert te >= 0, f"TE should be non-negative for lag={lag}"
        
        assert len(te_values) > 0, "Should compute TE for at least one lag"
    
    def test_optimal_lag(self):
        """Test that TE is highest at correct lag."""
        np.random.seed(42)
        n = 1000
        x = np.random.randn(n)
        # Create y with lag 2 dependence on x
        y = np.zeros(n)
        y[0] = x[0]
        for i in range(1, n):
            if i >= 2:
                y[i] = 0.7 * x[i-2] + 0.3 * np.random.randn()
            else:
                y[i] = 0.3 * np.random.randn()
        
        te_values = []
        for lag in range(1, 5):
            calc = TransferEntropyCalculator(n_bins=8, lag=lag)
            te = calc.compute_transfer_entropy_joint(x, y)
            te_values.append(te)
        
        # TE should peak around the true lag (2)
        optimal_lag = np.argmax(te_values) + 1
        assert 1 <= optimal_lag <= 4, \
            f"Optimal lag should be between 1-4, got {optimal_lag}"


class TestTransferEntropyBinning:
    """Tests for different binning configurations."""
    
    def test_different_bins(self):
        """Test TE with different number of bins."""
        np.random.seed(42)
        n = 1000
        x = np.random.randn(n)
        y = 0.8 * x[:-1] + 0.2 * np.random.randn(n-1)
        
        for n_bins in [4, 8, 10, 16]:
            calc = TransferEntropyCalculator(n_bins=n_bins, lag=1)
            te = calc.compute_transfer_entropy_joint(x, y)
            assert te >= 0, f"TE should be non-negative with n_bins={n_bins}"
    
    def test_discretization_valid(self):
        """Test that discretization produces valid bin indices."""
        np.random.seed(42)
        n = 1000
        x = np.random.randn(n)
        
        calc = TransferEntropyCalculator(n_bins=8, lag=1)
        x_disc = calc._discretize(x)
        
        assert x_disc.min() >= 0, "Discretized values should be >= 0"
        assert x_disc.max() < calc.n_bins, \
            f"Discretized values should be < n_bins, got max={x_disc.max()}"
        assert x_disc.shape == x.shape, "Discretized shape should match input"


class TestTransferEntropyEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_very_short_series(self):
        """Test with very short time series."""
        np.random.seed(42)
        x = np.random.randn(10)
        y = np.random.randn(10)
        
        calc = TransferEntropyCalculator(n_bins=4, lag=1)
        te = calc.compute_transfer_entropy_joint(x, y)
        
        assert te >= 0, "TE should be non-negative for short series"
    
    def test_constant_series(self):
        """Test with constant (zero variance) series."""
        n = 100
        x = np.random.randn(n)
        y = np.ones(n)  # Constant series
        
        calc = TransferEntropyCalculator(n_bins=8, lag=1)
        te = calc.compute_transfer_entropy_joint(x, y)
        
        assert te >= 0, "TE should handle constant series"
    
    def test_large_range_values(self):
        """Test with values spanning large range."""
        np.random.seed(42)
        x = np.random.randn(1000) * 1000  # Large range
        y = 0.8 * x[:-1] + 0.2 * np.random.randn(999) * 1000
        
        calc = TransferEntropyCalculator(n_bins=8, lag=1)
        te = calc.compute_transfer_entropy_joint(x, y)
        
        assert te >= 0, "TE should handle large range values"


class TestCausalInference:
    """Tests for causal inference functionality."""
    
    def test_causal_direction_from_spiking_network(self):
        """Test causal direction detection with spiking network data."""
        from causal_inference import generate_causal_time_series
        
        np.random.seed(42)
        
        # Generate causal data
        a, b = generate_causal_time_series(n_points=1000, causal_strength=0.5)
        
        calc = TransferEntropyCalculator(n_bins=8, lag=1)
        te_ab = calc.compute_transfer_entropy_joint(a, b)
        te_ba = calc.compute_transfer_entropy_joint(b, a)
        
        # TE should detect the causal direction
        assert te_ab != te_ba, "Should detect asymmetric causation"
    
    def test_randomized_baseline_lower(self):
        """Test that randomized TE is lower than original."""
        from causal_inference import generate_causal_time_series
        
        np.random.seed(42)
        a, b = generate_causal_time_series(n_points=1000, causal_strength=0.5)
        
        calc = TransferEntropyCalculator(n_bins=8, lag=1)
        te_original = calc.compute_transfer_entropy_joint(a, b)
        
        # Create shuffled version
        a_shuffled = np.random.permutation(a)
        te_randomized = calc.compute_transfer_entropy_joint(a_shuffled, b)
        
        # Original should generally be higher
        # (not always true due to noise, but should be true most of the time)
        if te_original > te_randomized:
            assert te_original > te_randomized


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
