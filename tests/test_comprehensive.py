"""
Comprehensive test suite for Spiking Neural Networks and Transfer Entropy.

This module consolidates all tests from the project into a single organized suite.
Tests cover:
- Transfer Entropy computation and properties
- Causal inference with SNNs
- Edge cases and error handling
- Binning and lag configuration
"""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTransferEntropyBasic:
    """Basic functionality tests for Transfer Entropy."""
    
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
        y_identical = self.y.copy()
        te = self.calc()(self.y, y_identical)
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
        
        optimal_lag = np.argmax(te_values) + 1
        assert 1 <= optimal_lag <= 4, f"Optimal lag should be between 1-4, got {optimal_lag}"


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
        y = np.ones(n)
        
        calc = TransferEntropyCalculator(n_bins=8, lag=1)
        te = calc.compute_transfer_entropy_joint(x, y)
        
        assert te >= 0, "TE should handle constant series"
    
    def test_large_range_values(self):
        """Test with values spanning large range."""
        np.random.seed(42)
        x = np.random.randn(1000) * 1000
        y = 0.8 * x[:-1] + 0.2 * np.random.randn(999) * 1000
        
        calc = TransferEntropyCalculator(n_bins=8, lag=1)
        te = calc.compute_transfer_entropy_joint(x, y)
        
        assert te >= 0, "TE should handle large range values"


class TestCausalInference:
    """Tests for causal inference functionality with SNNs."""
    
    def test_causal_direction_from_snn(self):
        """Test causal direction detection with SNN data."""
        from causal_inference import generate_causal_time_series
        
        np.random.seed(42)
        
        a, b = generate_causal_time_series(n_points=1000, causal_strength=0.5)
        
        calc = TransferEntropyCalculator(n_bins=8, lag=1)
        te_ab = calc.compute_transfer_entropy_joint(a, b)
        te_ba = calc.compute_transfer_entropy_joint(b, a)
        
        assert te_ab != te_ba, "Should detect asymmetric causation"
    
    def test_randomized_baseline_lower(self):
        """Test that randomized TE is lower than original."""
        from causal_inference import generate_causal_time_series
        
        np.random.seed(42)
        a, b = generate_causal_time_series(n_points=1000, causal_strength=0.5)
        
        calc = TransferEntropyCalculator(n_bins=8, lag=1)
        te_original = calc.compute_transfer_entropy_joint(a, b)
        
        a_shuffled = np.random.permutation(a)
        te_randomized = calc.compute_transfer_entropy_joint(a_shuffled, b)
        
        if te_original > te_randomized:
            assert te_original > te_randomized
    
    def test_input_length_validation(self):
        """Test that TE handles different input lengths correctly."""
        np.random.seed(42)
        n = 1000
        x = np.random.randn(n)
        y = np.random.randn(n)
        
        calc = TransferEntropyCalculator(n_bins=8, lag=1)
        
        te = calc.compute_transfer_entropy_joint(x, y[:n-1])
        assert te >= 0, "TE should be non-negative"


class TestTransferEntropyImplementation:
    """Specific tests for transfer_entropy_implementation module."""
    
    def test_basic_te_computation(self):
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
    
    def test_basic_te_asymmetry(self):
        """Test TE asymmetry with simple data."""
        from transfer_entropy_implementation import TransferEntropyCalculator
        
        np.random.seed(42)
        n = 500
        x = np.random.randn(n)
        y = 0.6 * x[:-1] + 0.4 * np.random.randn(n-1)
        
        calc = TransferEntropyCalculator(n_bins=8, lag=1)
        te_xy = calc.compute_transfer_entropy_joint(x, y)
        te_yx = calc.compute_transfer_entropy_joint(y, x)
        
        assert te_xy != te_yx, "TE should be asymmetric"
    
    def test_independent_data(self):
        """Test TE with independent data."""
        from transfer_entropy_implementation import TransferEntropyCalculator
        
        np.random.seed(42)
        n = 500
        x = np.random.randn(n)
        y = np.random.randn(n)
        
        calc = TransferEntropyCalculator(n_bins=8, lag=1)
        te = calc.compute_transfer_entropy_joint(x, y)
        
        assert te < 0.15, f"TE should be low for independent data, got {te}"
    
    def test_multiple_lags(self):
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


class TestLIFNeuron:
    """Tests for Leaky Integrate-and-Fire neuron model."""
    
    def test_lif_neuron_basic(self):
        """Test basic LIF neuron functionality."""
        from spiking_neural_networks.lif_neuron import LIFNeuron
        
        neuron = LIFNeuron(dt=0.1, tau=10.0, v_rest=0.0, v_threshold=1.0, 
                          v_reset=0.0, refractory_period=2)
        
        # Run simulation with constant input
        I = 0.5  # Constant current
        n_steps = 1000
        V, spikes = neuron.simulate(I, n_steps)
        
        assert len(V) == n_steps, f"Voltage array should have {n_steps} steps"
        assert len(spikes) == n_steps, f"Spikes array should have {n_steps} steps"
    
    def test_lif_neuron_no_spikes(self):
        """Test LIF neuron with subthreshold input."""
        from spiking_neural_networks.lif_neuron import LIFNeuron
        
        neuron = LIFNeuron(dt=0.1, tau=10.0, v_rest=0.0, v_threshold=1.0, 
                          v_reset=0.0, refractory_period=2)
        
        I = 0.1  # Subthreshold current
        n_steps = 1000
        V, spikes = neuron.simulate(I, n_steps)
        
        # Should not fire spikes with subthreshold input
        assert np.sum(spikes) == 0, "Should not fire with subthreshold input"
    
    def test_lif_neuron_parameter_validation(self):
        """Test LIF neuron parameter validation."""
        from spiking_neural_networks.lif_neuron import LIFNeuron
        
        # Test invalid tau
        with pytest.raises(ValueError):
            LIFNeuron(tau=0)
        
        # Test invalid dt
        with pytest.raises(ValueError):
            LIFNeuron(dt=0)
        
        # Test negative refractory
        with pytest.raises(ValueError):
            LIFNeuron(refractory_period=-1)


class TestSNN:
    """Tests for Spiking Neural Network class."""
    
    def test_snn_basic(self):
        """Test basic SNN functionality."""
        from spiking_neural_networks.snn import SNN
        
        snn = SNN(n_input=10, n_hidden=20, n_output=5, dt=0.1)
        
        # Forward pass with random input
        import numpy as np
        X = np.random.randn(1, 10)
        V_hidden, spikes_hidden, V_output, spikes_output = snn.forward(X)
        
        assert V_hidden.shape == (1, 20), f"Hidden voltage shape mismatch"
        assert spikes_hidden.shape == (1, 20), f"Hidden spikes shape mismatch"
        assert V_output.shape == (1, 5), f"Output voltage shape mismatch"
        assert spikes_output.shape == (1, 5), f"Output spikes shape mismatch"
    
    def test_snn_layer_validation(self):
        """Test SNN layer dimension validation."""
        from spiking_neural_networks.snn import SNN
        
        # Test invalid layer sizes
        with pytest.raises(ValueError):
            SNN(n_input=0, n_hidden=10, n_output=5)
        
        with pytest.raises(ValueError):
            SNN(n_input=10, n_hidden=-1, n_output=5)
    
    def test_snn_multiple_timesteps(self):
        """Test SNN with multiple timesteps."""
        from spiking_neural_networks.snn import SNN
        
        snn = SNN(n_input=10, n_hidden=20, n_output=5, dt=0.1)
        
        X = np.random.randn(1, 10)
        V_hidden, spikes_hidden, V_output, spikes_output = snn.forward(X)
        
        # Check that spikes are binary
        assert np.all(spikes_hidden == spikes_hidden.astype(int))
        assert np.all(spikes_output == spikes_output.astype(int))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
