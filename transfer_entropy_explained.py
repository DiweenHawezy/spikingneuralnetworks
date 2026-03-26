#!/usr/bin/env python3
"""
Transfer Entropy Explained - Educational Visualization

This module creates simple, intuitive visualizations to explain:
1. What transfer entropy is
2. How it detects causation
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
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    ax1.text(35, 35, 'Source A', ha='center', va='center', fontsize=14, fontweight='bold')
    ax1.text(65, 35, 'Target B', ha='center', va='center', fontsize=14, fontweight='bold')
    ax1.text(35, 70, 'If A influences B', ha='center', va='center', fontsize=10)
    ax1.text(35, 60, 'then knowing A helps', ha='center', va='center', fontsize=10)
    ax1.text(35, 50, 'predict B!', ha='center', va='center', fontsize=12, fontweight='bold', color='green')
    
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
    
    # Panel 3: Transfer entropy concept
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 100)
    ax3.set_aspect('equal')
    ax3.axis('off')
    
    # Draw a flow chart
    boxes = [
        ('Past A', 25, 75),
        ('Past B', 75, 75),
        ('Future B', 50, 30)
    ]
    
    for text, x, y in boxes:
        plt.annotate(text, (x, y), xytext=(0, 0),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.7),
                    ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows showing the relationship
    ax3.annotate('', xy=(50, 50), xytext=(75, 65),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.5))
    ax3.annotate('', xy=(50, 50), xytext=(25, 65),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    ax3.text(50, 20, 'Transfer Entropy asks:', ha='center', va='center', fontsize=10)
    ax3.text(50, 12, 'Does Past A help predict', ha='center', va='center', fontsize=9)
    ax3.text(50, 4, 'Future B beyond Past B alone?', ha='center', va='center', fontsize=9)
    
    ax3.set_title('3. Transfer Entropy Formula', fontsize=12, fontweight='bold')
    
    # Panel 4: Spikes as information carriers
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 100)
    
    # Draw spike trains
    for i in range(10):
        # Neuron A spikes (blue)
        t = 5 + i * 8 + np.random.randn() * 2
        if t < 95:
            ax4.plot([t, t], [5 + i, i + 8], 'b-', lw=3)
        # Neuron B spikes (red) - follows A with delay
        t_b = 10 + i * 8 + np.random.randn() * 2
        if t_b < 95:
            ax4.plot([t_b, t_b], [15 + i, i + 18], 'r-', lw=3)
    
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 35)
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    ax4.text(50, 5, 'Neuron A (Source)', ha='center', va='center', fontsize=10, color='blue')
    ax4.text(50, 22, 'Neuron B (Target)', ha='center', va='center', fontsize=10, color='red')
    
    ax4.set_title('4. Spikes Carry Information', fontsize=12, fontweight='bold')
    ax4.text(50, 30, 'When A fires, B often fires after', ha='center', va='center', fontsize=8)
    ax4.text(50, 28, 'a short delay - this pattern', ha='center', va='center', fontsize=8)
    ax4.text(50, 26, 'can be detected by transfer entropy', ha='center', va='center', fontsize=8)
    
    # Panel 5: Real-world analogy
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_xlim(0, 100)
    ax5.set_ylim(0, 100)
    ax5.axis('off')
    
    # Draw a simple domino setup
    for i in range(8):
        x = 20 + i * 10
        ax5.plot([x, x + 3], [5, 20], 'brown', lw=5)
    
    # Falling dominoes
    ax5.plot([20, 23], [5, 20], 'orange', lw=5)  # First falling
    ax5.plot([30, 33], [5, 20], 'orange', lw=5)  # Second falling
    
    ax5.text(50, 10, 'Domino effect', ha='center', va='center', fontsize=10, fontweight='bold')
    ax5.text(50, 3, 'A causes B - like dominoes falling', ha='center', va='center', fontsize=8)
    
    ax5.set_title('5. Real-World Analogy', fontsize=12, fontweight='bold')
    
    # Panel 6: Summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_xlim(0, 100)
    ax6.set_ylim(0, 100)
    ax6.axis('off')
    
    # Summary text
    summary = """
    TRANSFER ENTROPY IN SIMPLE TERMS:

    ┌─────────────────────────────────────┐
    │  1. We have two time series A & B   │
    │  2. We want to know: Does A cause B?│
    │  3. We encode them as neural spikes │
    │  4. We measure: How much does A's  │
    │     past help predict B's future?   │
    │  5. If YES → A causes B!            │
    └─────────────────────────────────────┐
    """
    ax6.text(0.5, 0.8, 'TRANSFER ENTROPY:', ha='center', va='center',
             fontsize=14, fontweight='bold', color='darkblue')
    ax6.text(0.5, 0.72, 'A Simple Measure of', ha='center', va='center',
             fontsize=12, fontweight='bold')
    ax6.text(0.5, 0.65, 'Causal Information Flow', ha='center', va='center',
             fontsize=12, fontweight='bold')
    
    ax6.text(0.5, 0.5, 'KEY INSIGHT:', ha='center', va='center',
             fontsize=11, fontweight='bold', color='green')
    ax6.text(0.5, 0.45, 'Directional ≠ Correlation', ha='center', va='center',
             fontsize=10)
    ax6.text(0.5, 0.4, 'A→B strong, B→A weak = A causes B', ha='center', va='center',
             fontsize=9)
    
    ax6.text(0.5, 0.25, 'WHY SPIKING NEURAL NETWORKS?', ha='center', va='center',
             fontsize=10, fontweight='bold')
    ax6.text(0.5, 0.20, '- Biological plausibility', ha='center', va='center', fontsize=8)
    ax6.text(0.5, 0.15, '- Handle timing information', ha='center', va='center', fontsize=8)
    ax6.text(0.5, 0.10, '- Robust to noise', ha='center', va='center', fontsize=8)
    
    ax6.set_title('6. Summary', fontsize=12, fontweight='bold')
    
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
    
    axes[0].scatter(x1[:100], y1[:100], alpha=0.5, s=10)
    axes[0].set_xlabel('Variable X')
    axes[0].set_ylabel('Variable Y')
    axes[0].set_title(f'Correlation: r = {np.corrcoef(x1, y1)[0,1]:.3f}', fontsize=12)
    axes[0].text(0.02, 0.9, 'X and Y are correlated', ha='left', va='top', transform=axes[0].transAxes,
                 bbox=dict(boxstyle='round', fc='lightyellow'))
    axes[0].text(0.02, 0.8, 'BUT: correlation ≠ causation', ha='left', va='top', transform=axes[0].transAxes,
                 bbox=dict(boxstyle='round', fc='lightyellow'))
    axes[0].text(0.02, 0.7, 'Correlation is SYMMETRIC', ha='left', va='top', transform=axes[0].transAxes,
                 bbox=dict(boxstyle='round', fc='lightyellow'))
    axes[0].text(0.02, 0.6, 'corr(X,Y) = corr(Y,X)', ha='left', va='top', transform=axes[0].transAxes,
                 bbox=dict(boxstyle='round', fc='lightyellow'))
    
    # Panel 2: Transfer Entropy - detects direction
    x2 = np.random.randn(n)
    y2 = 0.7 * x2[:n-1] + np.random.randn(n-1)  # x2 -> y2 with delay
    y2 = np.concatenate([[0], y2])  # Align shapes
    
    # Compute simple TE
    te_forward = np.corrcoef(x2[:n-1], y2[1:])[0,1] ** 2
    te_reverse = np.corrcoef(y2[:n-1], x2[1:])[0,1] ** 2
    
    axes[1].bar(['X → Y', 'Y → X'], [te_forward, te_reverse], 
                color=['blue', 'orange'], alpha=0.7)
    axes[1].set_ylabel('Transfer Entropy (proxy)')
    axes[1].set_title(f'Transfer Entropy Detects Direction\nFwd={te_forward:.3f}, Rev={te_reverse:.3f}', fontsize=12)
    axes[1].text(0.5, 0.7, 'TE is ASYMMETRIC', ha='center', va='top', transform=axes[1].transAxes,
                 bbox=dict(boxstyle='round', fc='lightgreen'))
    axes[1].text(0.5, 0.6, 'Strong TE(X→Y) +', ha='center', va='top', transform=axes[1].transAxes)
    axes[1].text(0.5, 0.55, 'Weak TE(Y→X) = X causes Y!', ha='center', va='top', transform=axes[1].transAxes)
    
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
    fig1.savefig('transfer_entropy_explained.png', dpi=150, bbox_inches='tight')
    print("  - Saved: transfer_entropy_explained.png")
    
    # Create comparison figure
    print("[2/2] Creating correlation vs TE comparison...")
    fig2 = create_comparison_figure()
    fig2.savefig('correlation_vs_transfer_entropy.png', dpi=150, bbox_inches='tight')
    print("  - Saved: correlation_vs_transfer_entropy.png")
    
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
