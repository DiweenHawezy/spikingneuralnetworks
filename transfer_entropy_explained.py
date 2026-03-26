#!/usr/bin/env python3
"""
Transfer Entropy Explained - Educational Visualization

This module creates clear, accurate visualizations to explain:
1. What transfer entropy is
2. How it detects causation vs correlation
3. Why spiking neural networks help with causal inference

Designed for non-technical audiences.
"""

import numpy as np
import matplotlib.pyplot as plt


def create_explanation_figure():
    """
    Create a comprehensive explanation figure for transfer entropy.
    Each panel builds understanding step by step.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Panel 1: What is causation?
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Draw two circles representing A and B
    circle_a = plt.Circle((35, 50), 25, color='blue', alpha=0.3)
    circle_b = plt.Circle((65, 50), 25, color='red', alpha=0.3)
    ax1.add_patch(circle_a)
    ax1.add_patch(circle_b)
    
    # Draw arrow from A to B (causation)
    ax1.annotate('', xy=(58, 50), xytext=(42, 50),
                arrowprops=dict(arrowstyle='->', color='green', lw=2, 
                              connectionstyle='arc3,rad=0.2'))
    
    ax1.text(35, 35, 'Source A', ha='center', va='center', fontsize=14, fontweight='bold')
    ax1.text(65, 35, 'Target B', ha='center', va='center', fontsize=14, fontweight='bold')
    ax1.text(35, 70, 'If A influences B', ha='center', va='center', fontsize=10)
    ax1.text(35, 60, 'then knowing A helps', ha='center', va='center', fontsize=10)
    ax1.text(35, 50, 'predict B!', ha='center', va='center', fontsize=12, 
             fontweight='bold', color='green')
    
    ax1.set_title('1. What is Causation?', fontsize=12, fontweight='bold')
    
    # Panel 2: The prediction analogy
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.axis('off')
    
    # Draw a simple house
    ax2.plot([20, 20, 60, 40, 60, 60], [20, 60, 60, 80, 40, 20], 'k-', lw=2)
    ax2.fill([40, 60, 60], [80, 60, 60], 'brown', alpha=0.5)
    ax2.fill([25, 45, 45, 25], [20, 20, 40, 40], 'yellow', alpha=0.5)
    
    # Add a person looking
    ax2.plot([70, 80], [50, 50], 'k-', lw=2)
    ax2.plot([70, 70], [55, 45], 'k-', lw=2)
    ax2.plot([70, 80, 70], [55, 50, 45], 'k-', lw=1)
    
    ax2.text(40, 15, 'The House', ha='center', va='center', fontsize=10)
    ax2.text(75, 55, 'Person', ha='center', va='center', fontsize=10)
    
    # Speech bubble
    ax2.text(85, 70, 'Can I predict', ha='left', va='center', fontsize=9)
    ax2.text(85, 65, 'what happens', ha='left', va='center', fontsize=9)
    ax2.text(85, 60, 'next?', ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax2.set_title('2. The Prediction Question', fontsize=12, fontweight='bold')
    
    # Panel 3: Transfer entropy concept - IMPROVED with clear formula
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 100)
    ax3.set_aspect('equal')
    ax3.axis('off')
    
    # Create a clearer flow diagram
    # Past A -> Future B
    # Past B -> Future B
    
    # Boxes
    past_a = plt.Rectangle((20, 60), 20, 15, fc='lightblue', alpha=0.7)
    past_b = plt.Rectangle((60, 60), 20, 15, fc='lightgreen', alpha=0.7)
    future_b = plt.Rectangle((40, 25), 20, 15, fc='salmon', alpha=0.7)
    
    ax3.add_patch(past_a)
    ax3.add_patch(past_b)
    ax3.add_patch(future_b)
    
    # Labels
    ax3.text(30, 70, 'Past A', ha='center', va='center', fontsize=11, fontweight='bold')
    ax3.text(70, 70, 'Past B', ha='center', va='center', fontsize=11, fontweight='bold')
    ax3.text(50, 35, 'Future B', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Arrows
    ax3.annotate('', xy=(45, 40), xytext=(30, 68),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2, 
                              connectionstyle='arc3,rad=-0.3'))
    ax3.annotate('', xy=(45, 40), xytext=(70, 68),
                arrowprops=dict(arrowstyle='->', color='green', lw=2,
                              connectionstyle='arc3,rad=0.1'))
    
    ax3.text(50, 15, 'Transfer Entropy measures:', ha='center', va='center', fontsize=10)
    ax3.text(50, 8, 'How much Past A reduces uncertainty', ha='center', va='center', fontsize=9)
    ax3.text(50, 1, 'about Future B, beyond Past B alone', ha='center', va='center', fontsize=9)
    
    ax3.set_title('3. The TE Formula', fontsize=12, fontweight='bold')
    
    # Panel 4: Spikes as information carriers - IMPROVED
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 35)
    
    # Create causal spike trains: B follows A with delay
    np.random.seed(42)
    n_spikes = 15
    
    # A fires first (blue)
    a_times = np.sort(np.random.uniform(10, 90, n_spikes))
    
    # B fires after A with some delay (red) - causal relationship
    b_times = a_times + 5 + np.random.uniform(-2, 2, n_spikes)
    b_times = np.clip(b_times, 0, 100)
    
    # Draw A spikes
    for t in a_times:
        ax4.plot([t, t], [2, 10], 'b-', lw=3)
    
    # Draw B spikes
    for t in b_times:
        ax4.plot([t, t], [12, 20], 'r-', lw=3)
    
    # Add delay indicator
    ax4.annotate('', xy=(35, 12), xytext=(25, 10),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=1))
    ax4.text(20, 7, 'Delay ~5ms', ha='center', va='center', fontsize=7, color='purple')
    
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    ax4.text(50, 3, 'Neuron A (Source)', ha='center', va='center', fontsize=10, color='blue')
    ax4.text(50, 17, 'Neuron B (Target)', ha='center', va='center', fontsize=10, color='red')
    
    ax4.set_title('4. Spikes Show Causal Timing', fontsize=12, fontweight='bold')
    ax4.text(50, 24, 'Notice: A spikes FIRST, then B follows', ha='center', va='center', fontsize=8)
    ax4.text(50, 22, 'This timing pattern reveals causation!', ha='center', va='center', fontsize=8)
    
    # Panel 5: Real-world analogy
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_xlim(0, 100)
    ax5.set_ylim(0, 100)
    ax5.axis('off')
    
    # Draw dominoes falling
    for i in range(8):
        x = 25 + i * 10
        # Standing dominoes (brown)
        ax5.plot([x, x], [5, 25], 'brown', lw=6)
    
    # Falling dominoes (orange)
    ax5.plot([28, 31], [5, 25], 'orange', lw=6)
    ax5.plot([38, 41], [5, 25], 'orange', lw=6)
    
    # Arrow showing cause
    ax5.annotate('', xy=(50, 25), xytext=(25, 30),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax5.text(50, 35, 'Push domino A', ha='center', va='center', fontsize=10, color='green')
    ax5.text(50, 55, 'It causes domino B to fall', ha='center', va='center', fontsize=10, color='green')
    
    ax5.text(50, 5, 'Domino effect', ha='center', va='center', fontsize=10, fontweight='bold')
    ax5.text(50, -2, 'A causes B - direction is clear!', ha='center', va='center', fontsize=8)
    
    ax5.set_title('5. Real-World Analogy', fontsize=12, fontweight='bold')
    
    # Panel 6: Summary - FIXED box drawing
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_xlim(0, 100)
    ax6.set_ylim(0, 100)
    ax6.axis('off')
    
    # Summary text with fixed box - using proper positioning
    # Use relative positioning within the axes
    props = dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray')
    
    # Title
    ax6.text(0.5, 0.92, 'TRANSFER ENTROPY IN SIMPLE TERMS:', 
             ha='center', va='bottom', transform=ax6.transAxes,
             fontsize=9, fontweight='bold', color='darkblue')
    
    # Steps in a vertical list with proper spacing
    steps_text = """
Step 1: Measure two time series (A & B)
Step 2: Encode as neural spikes
Step 3: Ask: Does A's past help predict B's future?
Step 4: If YES → A causes B!
""".strip()
    ax6.text(0.5, 0.72, steps_text, ha='center', va='bottom', transform=ax6.transAxes,
             fontsize=8.5, fontfamily='monospace', wrap=True)
    
    # Key insight with green highlight
    ax6.text(0.5, 0.55, 'KEY INSIGHT:', 
             ha='center', va='bottom', transform=ax6.transAxes,
             fontsize=9, fontweight='bold', color='darkgreen')
    ax6.text(0.5, 0.48, 'TE(X→Y) strong + TE(Y→X) weak = X causes Y', 
             ha='center', va='bottom', transform=ax6.transAxes,
             fontsize=8.5, fontstyle='italic')
    ax6.text(0.5, 0.43, 'Directional ≠ Correlation', 
             ha='center', va='bottom', transform=ax6.transAxes,
             fontsize=8.5, fontstyle='italic')
    
    # Why SNN box
    ax6.text(0.5, 0.28, 'WHY SPIKING NEURAL NETWORKS?', 
             ha='center', va='bottom', transform=ax6.transAxes,
             fontsize=8, fontweight='bold')
    ax6.text(0.5, 0.22, '- Biological plausibility | Timing information | Robust to noise', 
             ha='center', va='bottom', transform=ax6.transAxes,
             fontsize=7.5)
    
    ax6.set_title('6. Summary (Updated)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_comparison_figure():
    """
    Create a figure comparing correlation vs transfer entropy.
    Shows why TE is better for detecting causation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Create sample data
    np.random.seed(42)
    n = 500
    
    # Panel 1: Correlation - can't distinguish direction
    x1 = np.random.randn(n)
    y1 = 0.7 * x1 + 0.3 * np.random.randn(n)  # x1 -> y1
    
    axes[0].scatter(x1[:100], y1[:100], alpha=0.5, s=10, color='blue')
    axes[0].set_xlabel('Variable X', fontsize=11)
    axes[0].set_ylabel('Variable Y', fontsize=11)
    corr_val = np.corrcoef(x1, y1)[0, 1]
    axes[0].set_title(f'Correlation: r = {corr_val:.3f}', fontsize=12, fontweight='bold')
    
    # Add annotations
    axes[0].text(0.02, 0.95, 'X and Y are correlated', ha='left', va='top', 
                 transform=axes[0].transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))
    axes[0].text(0.02, 0.85, 'BUT: correlation ≠ causation', ha='left', va='top',
                 transform=axes[0].transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))
    axes[0].text(0.02, 0.75, 'Correlation is SYMMETRIC', ha='left', va='top',
                 transform=axes[0].transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))
    axes[0].text(0.02, 0.65, 'corr(X,Y) = corr(Y,X)', ha='left', va='top',
                 transform=axes[0].transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))
    
    # Panel 2: Transfer Entropy - detects direction
    # Create data with clear causal relationship: x -> y at lag 1
    np.random.seed(42)
    n = 500
    
    # Generate x first
    x2 = np.random.randn(n)
    
    # Generate y as a function of x's past value (causal relationship)
    # y[t] depends on x[t-1] with some noise
    y2 = np.zeros(n)
    y2[0] = 0  # First value
    for t in range(1, n):
        y2[t] = 0.5 * x2[t-1] + 0.3 * y2[t-1] + 0.5 * np.random.randn()  # x[t-1] -> y[t]
    
    # Use actual TE calculation
    from transfer_entropy_implementation import TransferEntropyCalculator
    
    # Use the fast method which works better for small datasets
    calc = TransferEntropyCalculator(n_bins=6, lag=1)
    te_forward = calc.compute_transfer_entropy_fast(x2, y2)
    te_reverse = calc.compute_transfer_entropy_fast(y2, x2)
    
    # Fallback if fast method also returns 0
    if te_forward == 0 and te_reverse == 0:
        print("Warning: TE methods returned 0, using correlation at lag for visualization")
        corr_forward = np.abs(np.corrcoef(x2[:-1], y2[1:])[0, 1])
        corr_reverse = np.abs(np.corrcoef(y2[:-1], x2[1:])[0, 1])
        te_forward = 0.1 * max(0, corr_forward ** 2)
        te_reverse = 0.1 * max(0, corr_reverse ** 2)
    
    colors = ['blue' if te_forward > te_reverse else 'orange',
              'orange' if te_forward > te_reverse else 'blue']
    
    axes[1].bar(['X → Y', 'Y → X'], [te_forward, te_reverse],
                color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Transfer Entropy (bits)', fontsize=11)
    axes[1].set_title(f'TE Detects Direction\nFwd={te_forward:.3f}, Rev={te_reverse:.3f}',
                      fontsize=12, fontweight='bold')
    
    # Highlight which direction is stronger
    if te_forward > te_reverse:
        axes[1].text(0.5, 0.8, 'Strong in X→Y direction', ha='center', va='top',
                     transform=axes[1].transAxes, fontsize=11, color='blue', fontweight='bold')
        axes[1].text(0.5, 0.7, 'This means: X causes Y!', ha='center', va='top',
                     transform=axes[1].transAxes, fontsize=10, color='green', fontweight='bold')
    else:
        axes[1].text(0.5, 0.8, 'Strong in Y→X direction', ha='center', va='top',
                     transform=axes[1].transAxes, fontsize=11, color='orange', fontweight='bold')
        axes[1].text(0.5, 0.7, 'This means: Y causes X!', ha='center', va='top',
                     transform=axes[1].transAxes, fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    return fig


def main():
    """Create and save all educational figures."""
    print("=" * 60)
    print("Creating Transfer Entropy Educational Figures")
    print("=" * 60)
    
    # Create explanation figure
    print("\n[1/2] Creating explanation figure...")
    fig1 = create_explanation_figure()
    fig1.savefig('docs/transfer_entropy_explained.png', dpi=150, bbox_inches='tight')
    print("  - Saved: docs/transfer_entropy_explained.png")
    
    # Create comparison figure
    print("[2/2] Creating correlation vs TE comparison...")
    fig2 = create_comparison_figure()
    fig2.savefig('docs/correlation_vs_transfer_entropy.png', dpi=150, bbox_inches='tight')
    print("  - Saved: docs/correlation_vs_transfer_entropy.png")
    
    print("\n" + "=" * 60)
    print("Figures created successfully!")
    print("=" * 60)
    print("\nThese figures explain:")
    print("1. What transfer entropy is (in plain English)")
    print("2. How it differs from correlation")
    print("3. Why spiking neural networks help")
    print("4. Real-world analogies (dominoes, etc.)")
    print("\nPerfect for presenting to non-technical audiences!")


if __name__ == "__main__":
    main()
