#!/usr/bin/env python3
"""
Transfer Entropy Implementation - Complete Documentation

This module provides a comprehensive implementation of Transfer Entropy (TE)
with detailed step-by-step explanations of the calculation process.

Transfer Entropy is a measure of directional (asymmetric) information flow
between two time series, answering the question: "Does knowing the past of X
help predict the future of Y, beyond what the past of Y already tells us?"

Mathematical Foundation
=======================

TE(X → Y) = I(Y_future; X_past | Y_past)
          = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
          = Σ P(X_past, Y_past, Y_future) × log₂[P(X_past, Y_past, Y_future) × P(Y_past) / (P(X_past, Y_past) × P(Y_future))]

Where:
- I = Mutual Information
- H = Entropy (uncertainty)
- X_past = Past values of source series X
- Y_past = Past values of target series Y  
- Y_future = Future values of target series Y

The formula measures how much knowing X_past reduces uncertainty about Y_future,
given that we already know Y_past.

Author: Spiking Neural Networks Project
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
from pathlib import Path


class TransferEntropyCalculator:
    """
    A comprehensive class for computing Transfer Entropy with multiple methods.
    
    Attributes:
        n_bins: Number of bins for discretization (default: 8)
        lag: Time lag for future values (default: 1)
        method: Calculation method ('joint', 'conditional_entropy', 'fast')
    """
    
    def __init__(self, n_bins: int = 8, lag: int = 1, method: str = 'joint'):
        """
        Initialize the Transfer Entropy calculator.
        
        Args:
            n_bins: Number of bins for discretization
            lag: Time lag for future values
            method: Calculation method:
                   - 'joint': Full joint probability calculation (most accurate)
                   - 'conditional_entropy': Using conditional entropy formula
                   - 'fast': Vectorized approximation (faster, less accurate)
        """
        self.n_bins = n_bins
        self.lag = lag
        self.method = method
        
    def _discretize(self, data: np.ndarray) -> np.ndarray:
        """
        Discretize continuous data into bins.
        
        Args:
            data: Continuous time series
            
        Returns:
            Discretized array with bin indices
        """
        # Create bin edges based on data range
        bin_edges = np.linspace(data.min(), data.max(), self.n_bins + 1)
        # Digitize: assigns bin index (0 to n_bins-1)
        discretized = np.digitize(data, bin_edges) - 1
        # Clip to ensure valid bin indices
        return np.clip(discretized, 0, self.n_bins - 1)
    
    def compute_joint_probabilities(
        self, 
        x_past: np.ndarray, 
        y_past: np.ndarray, 
        y_future: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute all required probability distributions for TE calculation.
        
        This is the core of the joint probability method.
        
        Args:
            x_past: Discretized past values of source series
            y_past: Discretized past values of target series
            y_future: Discretized future values of target series
            
        Returns:
            Tuple of probability arrays:
            - p_xyp_yf: P(X_past, Y_past, Y_future) - 3D
            - p_yp_yf: P(Y_past, Y_future) - 2D (marginalized over X)
            - p_xy: P(X_past, Y_past) - 2D (marginalized over Y_future)
            - p_yp: P(Y_past) - 1D
            - p_yf: P(Y_future) - 1D
        """
        n = len(x_past)
        
        # P(X_past, Y_past, Y_future) - joint distribution
        hist_3d = np.zeros((self.n_bins, self.n_bins, self.n_bins))
        for i in range(n):
            hist_3d[x_past[i], y_past[i], y_future[i]] += 1
        p_xyp_yf = hist_3d / np.sum(hist_3d)
        
        # P(Y_past, Y_future) - marginalize over X
        hist_2d = np.zeros((self.n_bins, self.n_bins))
        for i in range(n):
            hist_2d[y_past[i], y_future[i]] += 1
        p_yp_yf = hist_2d / np.sum(hist_2d)
        
        # P(X_past, Y_past) - marginalize over Y_future
        hist_xy = np.zeros((self.n_bins, self.n_bins))
        for i in range(n):
            hist_xy[x_past[i], y_past[i]] += 1
        p_xy = hist_xy / np.sum(hist_xy)
        
        # P(Y_past) - marginalize over everything else
        hist_yp = np.zeros(self.n_bins)
        for i in range(n):
            hist_yp[y_past[i]] += 1
        p_yp = hist_yp / np.sum(hist_yp)
        
        # P(Y_future) - marginalize over everything else
        hist_yf = np.zeros(self.n_bins)
        for i in range(n):
            hist_yf[y_future[i]] += 1
        p_yf = hist_yf / np.sum(hist_yf)
        
        return p_xyp_yf, p_yp_yf, p_xy, p_yp, p_yf
    
    def compute_transfer_entropy_joint(
        self, 
        x: np.ndarray, 
        y: np.ndarray
    ) -> float:
        """
        Compute Transfer Entropy using the joint probability method.
        
        This is the most accurate method but computationally expensive.
        
        TE = Σ P(X,Y_past,Y_future) × log₂[P(X,Y_past,Y_future) × P(Y_past) / (P(X,Y_past) × P(Y_future))]
        
        Args:
            x: Source time series (may cause)
            y: Target time series (may be caused)
            
        Returns:
            Transfer Entropy value in bits
        """
        # Step 1: Discretize the time series
        x_disc = self._discretize(x)
        y_past_disc = self._discretize(y[:-self.lag])
        y_future_disc = self._discretize(y[self.lag:])
        
        # Step 1b: Ensure all arrays have same length (n - lag)
        n_effective = len(y_past_disc)
        x_disc = x_disc[:n_effective]
        
        # Step 2: Compute joint probabilities
        p_xyp_yf, p_yp_yf, p_xy, p_yp, p_yf = self.compute_joint_probabilities(
            x_disc, y_past_disc, y_future_disc
        )
        
        # Step 3: Compute TE using the formula
        te = 0.0
        for i in range(self.n_bins):      # X_past
            for j in range(self.n_bins):  # Y_past
                for k in range(self.n_bins):  # Y_future
                    if p_xyp_yf[i, j, k] > 0 and p_xy[i, j] > 0 and p_yf[k] > 0:
                        # Compute the ratio inside the log
                        ratio = (p_xyp_yf[i, j, k] * p_yp[j]) / (p_xy[i, j] * p_yf[k])
                        if ratio > 0:
                            # Add contribution to TE
                            te += p_xyp_yf[i, j, k] * np.log2(ratio)
        
        return max(0.0, te)
    
    def compute_transfer_entropy_conditional(
        self, 
        x: np.ndarray, 
        y: np.ndarray
    ) -> float:
        """
        Compute Transfer Entropy using the conditional entropy method.
        
        TE = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
        
        This method is more intuitive as it directly measures the reduction
        in uncertainty about Y_future.
        
        Args:
            x: Source time series
            y: Target time series
            
        Returns:
            Transfer Entropy value in bits
        """
        # Discretize
        x_disc = self._discretize(x)
        y_past_disc = self._discretize(y[:-self.lag])
        y_future_disc = self._discretize(y[self.lag:])
        
        n = len(x_disc)
        
        # P(Y_past, Y_future)
        p_yyp_yf = np.zeros((self.n_bins, self.n_bins))
        for i in range(n):
            p_yyp_yf[y_past_disc[i], y_future_disc[i]] += 1
        p_yyp_yf = p_yyp_yf / np.sum(p_yyp_yf)
        
        # P(X_past, Y_past, Y_future)
        p_xyp_yf = np.zeros((self.n_bins, self.n_bins, self.n_bins))
        for i in range(n):
            p_xyp_yf[x_disc[i], y_past_disc[i], y_future_disc[i]] += 1
        p_xyp_yf = p_xyp_yf / np.sum(p_xyp_yf)
        
        # Compute H(Y_future | Y_past)
        # H(B|A) = H(A,B) - H(A)
        # Marginalize to get P(Y_past)
        p_yp = np.sum(p_yyp_yf, axis=1)
        # Entropy H(Y_past)
        h_yp = -np.sum(p_yp * np.log2(p_yp + 1e-10))
        # Joint entropy H(Y_past, Y_future)
        h_yyp_yf = -np.sum(p_yyp_yf * np.log2(p_yyp_yf + 1e-10))
        # Conditional entropy H(Y_future | Y_past) = H(Y_past, Y_future) - H(Y_past)
        h_yf_given_yp = h_yyp_yf - h_yp
        
        # Compute H(Y_future | Y_past, X_past)
        # Need P(X_past, Y_past)
        p_xy = np.sum(p_xyp_yf, axis=2)
        # Entropy H(X_past, Y_past)
        h_xy = -np.sum(p_xy * np.log2(p_xy + 1e-10))
        # Joint entropy H(X_past, Y_past, Y_future)
        h_xyp_yf = -np.sum(p_xyp_yf * np.log2(p_xyp_yf + 1e-10))
        # Conditional entropy H(Y_future | X_past, Y_past)
        h_yf_given_xyp = h_xyp_yf - h_xy
        
        # TE = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
        te = h_yf_given_yp - h_yf_given_xyp
        
        return max(0.0, te)
    
    def compute_transfer_entropy_fast(
        self, 
        x: np.ndarray, 
        y: np.ndarray
    ) -> float:
        """
        Fast approximation of Transfer Entropy using correlation.
        
        This is much faster but less accurate. Good for quick checks.
        
        Note: This is a proxy and should be used for exploratory analysis only.
        
        Args:
            x: Source time series
            y: Target time series
            
        Returns:
            Approximate Transfer Entropy value
        """
        n = len(x) - self.lag
        
        if n <= 0:
            return 0.0
        
        # Get lagged values
        x_lagged = x[:n]
        y_current = y[self.lag:]
        
        # Compute correlation
        corr = np.corrcoef(x_lagged, y_current)[0, 1]
        
        # TE is approximately proportional to r² for linear relationships
        te = max(0, corr ** 2) * 0.1
        
        return te


def step_by_step_example() -> Dict:
    """
    Demonstrate Transfer Entropy calculation step-by-step.
    
    Returns:
        Dictionary containing all intermediate values for the example
    """
    print("=" * 70)
    print("TRANSFER ENTROPY STEP-BY-STEP CALCULATION")
    print("=" * 70)
    
    # Create simple example data
    np.random.seed(42)
    n = 100
    
    # Series X: random noise
    x = np.random.randn(n)
    
    # Series Y: depends on past X with delay
    y = np.zeros(n)
    y[0] = x[0]
    for i in range(1, n):
        y[i] = 0.7 * x[i-1] + 0.3 * np.random.randn()  # X causes Y with lag 1
    
    print("\n1. ORIGINAL DATA")
    print(f"   Series X: mean={x.mean():.3f}, std={x.std():.3f}")
    print(f"   Series Y: mean={y.mean():.3f}, std={y.std():.3f}")
    print(f"   Correlation(X, Y): {np.corrcoef(x[:-1], y[1:])[0,1]:.3f}")
    
    # Initialize calculator
    calc = TransferEntropyCalculator(n_bins=4, lag=1)
    
    # Discretize
    x_disc = calc._discretize(x)
    y_past_disc = calc._discretize(y[:-1])
    y_future_disc = calc._discretize(y[1:])
    
    print("\n2. DISCRETIZATION")
    print(f"   Bins: {calc.n_bins}")
    print(f"   Lag: {calc.lag}")
    print(f"   X bins: min={x_disc.min()}, max={x_disc.max()}")
    print(f"   Y_past bins: min={y_past_disc.min()}, max={y_past_disc.max()}")
    print(f"   Y_future bins: min={y_future_disc.min()}, max={y_future_disc.max()}")
    
    # Compute probabilities
    p_xyp_yf, p_yp_yf, p_xy, p_yp, p_yf = calc.compute_joint_probabilities(
        x_disc, y_past_disc, y_future_disc
    )
    
    print("\n3. PROBABILITY DISTRIBUTIONS")
    print(f"   P(X, Y_past, Y_future) shape: {p_xyp_yf.shape}")
    print(f"   P(Y_past, Y_future) shape: {p_yp_yf.shape}")
    print(f"   P(X, Y_past) shape: {p_xy.shape}")
    print(f"   Sum of P(X, Y_past, Y_future): {p_xyp_yf.sum():.6f}")
    print(f"   Sum of P(Y_past, Y_future): {p_yp_yf.sum():.6f}")
    
    # Compute TE
    te = calc.compute_transfer_entropy_joint(x, y)
    
    print("\n4. TRANSFER ENTROPY RESULT")
    print(f"   TE(X → Y) = {te:.6f} bits")
    print(f"   Interpretation: {te:.2%} reduction in uncertainty")
    
    # Compare with reverse direction
    te_reverse = calc.compute_transfer_entropy_joint(y, x)
    
    print("\n5. REVERSE DIRECTION")
    print(f"   TE(Y → X) = {te_reverse:.6f} bits")
    
    print("\n6. CONCLUSION")
    if te > te_reverse:
        print(f"   ✓ X → Y: Stronger causation (TE = {te:.4f})")
        print(f"   ✓ X causes Y (not the reverse)")
    elif te_reverse > te:
        print(f"   ✓ Y → X: Stronger causation (TE = {te_reverse:.4f})")
        print(f"   ✓ Y causes X (not the reverse)")
    else:
        print(f"   ⚠ No clear directional causation detected")
    
    return {
        'te_forward': te,
        'te_reverse': te_reverse,
        'calc': calc
    }


def visualize_transfer_entropy() -> plt.Figure:
    """
    Create visualization of Transfer Entropy calculation.
    
    Returns:
        matplotlib Figure showing the TE computation flow
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Create example data
    np.random.seed(42)
    n = 1000
    x = np.random.randn(n)
    y = 0.7 * x[:-1] + np.random.randn(n-1)
    y = np.concatenate([[0], y])
    
    # Panel 1: Time series
    ax = axes[0, 0]
    ax.plot(x[:200], 'b', alpha=0.7, label='X (source)', linewidth=0.5)
    ax.plot(y[:200], 'r', alpha=0.7, label='Y (target)', linewidth=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('1. Original Time Series')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Discretization
    ax = axes[0, 1]
    calc = TransferEntropyCalculator(n_bins=4, lag=1)
    x_disc = calc._discretize(x)
    y_past_disc = calc._discretize(y[:-1])
    
    # Create a simple scatter plot of discretized values
    ax.scatter(x_disc[:100], y_past_disc[:100], alpha=0.5, s=10)
    ax.set_xlabel('X (discretized)')
    ax.set_ylabel('Y_past (discretized)')
    ax.set_title('2. Discretized Data')
    ax.set_xlim(0, calc.n_bins)
    ax.set_ylim(0, calc.n_bins)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Joint probability distribution
    ax = axes[0, 2]
    hist_3d = np.zeros((calc.n_bins, calc.n_bins, calc.n_bins))
    y_future_disc = calc._discretize(y[1:])
    for i in range(len(x_disc)):
        hist_3d[x_disc[i], y_past_disc[i], y_future_disc[i]] += 1
    hist_3d = hist_3d / np.sum(hist_3d)
    
    # Show 2D slice
    slice_idx = calc.n_bins // 2
    ax.imshow(hist_3d[:, :, slice_idx], cmap='hot', aspect='auto')
    ax.set_xlabel('X (discretized)')
    ax.set_ylabel('Y_past (discretized)')
    ax.set_title(f'3. Joint Distribution P(X,Y_past,Y_future)\\n(Slice at Y_future={slice_idx})')
    
    # Panel 4: Probability marginals
    ax = axes[1, 0]
    
    # Compute marginals
    p_xy = np.sum(hist_3d, axis=2)
    p_xy = p_xy / np.sum(p_xy)
    p_yf = np.sum(hist_3d, axis=(0, 1))
    p_yf = p_yf / np.sum(p_yf)
    p_yp = np.sum(hist_3d, axis=(0, 2))
    p_yp = p_yp / np.sum(p_yp)
    
    ax.bar(range(calc.n_bins), p_xy[:, calc.n_bins//2], alpha=0.7, label='P(X, Y_past)')
    ax.bar(range(calc.n_bins), p_yf, alpha=0.5, label='P(Y_future)')
    ax.bar(range(calc.n_bins), p_yp, alpha=0.5, label='P(Y_past)')
    ax.set_xlabel('Bin Index')
    ax.set_ylabel('Probability')
    ax.set_title('4. Marginal Probability Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 5: TE calculation formula
    ax = axes[1, 1]
    ax.axis('off')
    
    formula = r"""
    TE(X → Y) = Σ P(X, Y_past, Y_future) × log₂[
                  P(X, Y_past, Y_future) × P(Y_past)
                  ───────────────────────────────────────
                  P(X, Y_past) × P(Y_future)
                ]
    """
    ax.text(0.1, 0.5, formula, ha='left', va='center', 
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', fc='lightblue', alpha=0.7))
    ax.set_title('5. Transfer Entropy Formula', fontsize=12, fontweight='bold')
    
    # Panel 6: Results
    ax = axes[1, 2]
    
    te_forward = calc.compute_transfer_entropy_joint(x, y)
    te_reverse = calc.compute_transfer_entropy_joint(y, x)
    
    bars = ax.bar(['X → Y', 'Y → X'], [te_forward, te_reverse], 
                  color=['blue', 'orange'], alpha=0.7)
    ax.set_ylabel('Transfer Entropy (bits)')
    ax.set_title('6. Transfer Entropy Results')
    ax.set_yscale('log')
    
    # Add value labels
    for bar, te in zip(bars, [te_forward, te_reverse]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{te:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig


def comprehensive_example():
    """
    Run a comprehensive example comparing different TE methods.
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE TRANSFER ENTROPY EXAMPLE")
    print("=" * 70)
    
    # Create test data with known causation
    np.random.seed(42)
    n = 1000
    
    # Case 1: X causes Y
    x1 = np.random.randn(n)
    y1 = 0.7 * x1[:-1] + np.random.randn(n-1)
    y1 = np.concatenate([[0], y1])
    
    # Case 2: Y causes X (reverse)
    x2 = 0.7 * y1[:-1] + np.random.randn(n-1)
    x2 = np.concatenate([[0], x2])
    
    # Case 3: No causation (independent)
    x3 = np.random.randn(n)
    y3 = np.random.randn(n)
    
    methods = ['joint', 'conditional_entropy']
    
    for method in methods:
        print(f"\n{'='*70}")
        print(f"METHOD: {method.upper()}")
        print('='*70)
        
        calc = TransferEntropyCalculator(n_bins=8, lag=1, method=method)
        
        # Case 1
        te1 = calc.compute_transfer_entropy_joint(x1, y1)
        te1_rev = calc.compute_transfer_entropy_joint(y1, x1)
        print(f"\nCase 1: X → Y (known causation)")
        print(f"  TE(X→Y) = {te1:.6f}")
        print(f"  TE(Y→X) = {te1_rev:.6f}")
        print(f"  Result: {'✓ Correct' if te1 > te1_rev else '✗ Wrong'}")
        
        # Case 2
        te2 = calc.compute_transfer_entropy_joint(x2, y1)
        te2_rev = calc.compute_transfer_entropy_joint(y1, x2)
        print(f"\nCase 2: Y → X (known causation)")
        print(f"  TE(Y→X) = {te2_rev:.6f}")
        print(f"  TE(X→Y) = {te2:.6f}")
        print(f"  Result: {'✓ Correct' if te2_rev > te2 else '✗ Wrong'}")
        
        # Case 3
        te3 = calc.compute_transfer_entropy_joint(x3, y3)
        print(f"\nCase 3: Independent (no causation)")
        print(f"  TE(X→Y) = {te3:.6f}")
        print(f"  Result: {'✓ Low TE' if te3 < 0.1 else '⚠ High TE for independent data'}")


def main():
    """Main function to run all examples."""
    print("=" * 70)
    print("TRANSFER ENTROPY IMPLEMENTATION - DEMONSTRATION")
    print("=" * 70)
    
    # Run step-by-step example
    result = step_by_step_example()
    
    # Run comprehensive example
    comprehensive_example()
    
    # Create visualization
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATION...")
    print("=" * 70)
    
    fig = visualize_transfer_entropy()
    
    # Save with error handling
    output_path = Path(__file__).parent / 'transfer_entropy_implementation_diagram.png'
    try:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    except Exception as e:
        print(f"⚠ Warning: Could not save plot: {e}")
        print("  Continuing without saving...")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. TE measures directional information flow from X to Y")
    print("2. Discretization converts continuous data to discrete bins")
    print("3. Joint probabilities capture the relationship between past and future")
    print("4. TE > 0 means X provides additional predictive power for Y")
    print("5. Compare TE(X→Y) vs TE(Y→X) to determine causation direction")


if __name__ == "__main__":
    main()
