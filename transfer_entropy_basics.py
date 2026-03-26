#!/usr/bin/env python3
"""
Transfer Entropy Basics - Educational Tutorial

This module provides a beginner-friendly introduction to Transfer Entropy
with plain English explanations, simple examples, and visual demonstrations.

No advanced mathematics required!

Table of Contents
=================
1. What is Transfer Entropy?
2. Why not just use correlation?
3. How does it work? (Step-by-step)
4. Real-world examples
5. Common misconceptions

Author: Spiking Neural Networks Project
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# SECTION 1: WHAT IS TRANSFER ENTROPY?
# ============================================================================

def explain_what_is_te():
    """
    Explain Transfer Entropy in simple terms.
    """
    print("\n" + "=" * 70)
    print("SECTION 1: WHAT IS TRANSFER ENTROPY?")
    print("=" * 70)
    
    print("""
Transfer Entropy (TE) answers a simple question:

    "Does knowing about X's past help me predict Y's future?"

If YES → X might be causing Y
If NO  → X and Y are unrelated (or the relationship is complex)

Think of it as a "causation detector" for time series data.

Examples:
┌─────────────────────────────────────────────────────────────────┐
│ Scenario                          │ TE Tells Us                 │
├─────────────────────────────────┼─────────────────────────────┤
│ Ice cream sales & drowning      │ No causation (both caused   │
│ (summer effect)                 │ by heat, not each other)    │
├─────────────────────────────────┼─────────────────────────────┤
│ Rain → Puddles                  │ YES! Rain causes puddles    │
├─────────────────────────────────┼─────────────────────────────┤
│ Stock price A → Stock price B   │ Maybe! A influences B       │
├─────────────────────────────────┼─────────────────────────────┤
│ Brain region A → Brain region B │ YES! Neural communication   │
└─────────────────────────────────────────────────────────────────┘
""")


# ============================================================================
# SECTION 2: WHY NOT JUST USE CORRELATION?
# ============================================================================

def explain_correlation_vs_te():
    """
    Explain the difference between correlation and transfer entropy.
    """
    print("\n" + "=" * 70)
    print("SECTION 2: CORRELATION vs TRANSFER ENTROPY")
    print("=" * 70)
    
    print("""
Correlation measures: "Do X and Y move together?"
Transfer Entropy measures: "Does X cause Y?"

┌─────────────────────────────────────────────────────────────────┐
│ Problem with Correlation:                                      │
│ ───────────────────────────                                     │
│ • Symmetric: corr(X,Y) = corr(Y,X)                             │
│   → Can't tell direction!                                       │
│                                                                 │
│ • Correlation ≠ Causation                                      │
│   → Just because they're related doesn't mean one causes the  │
│     other!                                                      │
│                                                                 │
│ • No timing information                                        │
│   → Can't detect "X happens BEFORE Y"                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Advantage of Transfer Entropy:                                 │
│ ─────────────────────────────                                   │
│ • Asymmetric: TE(X→Y) ≠ TE(Y→X)                                │
│   → Can tell which way causation flows!                        │
│                                                                 │
│ • Uses timing information                                      │
│   → Checks if X's PAST predicts Y's FUTURE                     │
│                                                                 │
│ • Detects causal direction                                     │
│   → High TE(X→Y) + Low TE(Y→X) = X causes Y!                   │
└─────────────────────────────────────────────────────────────────┘

Simple Example:
───────────────
Imagine two people walking:

Person A:  •     •     •     •     •     (steps at t=0,2,4,6,8)
Person B:      •     •     •     •     (steps at t=1,3,5,7)

Correlation: "A and B both step frequently" ✓
Transfer Entropy: "A steps, then B steps 1 second later" → A influences B! ✓
""")


# ============================================================================
# SECTION 3: HOW DOES IT WORK? (STEP-BY-STEP)
# ============================================================================

def explain_te_calculation():
    """
    Explain TE calculation in simple terms.
    """
    print("\n" + "=" * 70)
    print("SECTION 3: HOW TRANSFER ENTROPY IS CALCULATED")
    print("=" * 70)
    
    print("""
Step-by-step breakdown:

STEP 1: Collect data
────────────────────
X = [1.2, 0.8, 1.5, 0.9, 1.3, 0.7, 1.1, 0.6, ...]  (source)
Y = [0.5, 0.9, 0.6, 1.0, 0.7, 1.1, 0.8, 1.2, ...]  (target)

STEP 2: Split into past and future
──────────────────────────────────
X_past = X[0:-1] = [1.2, 0.8, 1.5, 0.9, 1.3, 0.7, 1.1, ...]
Y_past = Y[0:-1] = [0.5, 0.9, 0.6, 1.0, 0.7, 1.1, 0.8, ...]
Y_fut  = Y[1:]   = [0.9, 0.6, 1.0, 0.7, 1.1, 0.8, 1.2, ...]

STEP 3: Discretize (bin the data)
─────────────────────────────────
Convert continuous values to bins:
X_past → [2, 0, 2, 1, 2, 0, 1, ...]  (bin indices)
Y_past → [1, 2, 1, 3, 1, 3, 2, ...]
Y_fut  → [2, 1, 3, 1, 3, 2, 3, ...]

STEP 4: Count patterns (compute probabilities)
──────────────────────────────────────────────
Count how often each pattern occurs:

P(X=2, Y_past=1, Y_fut=3) = 15 times out of 1000 = 0.015
P(X=2, Y_past=1)          = 50 times out of 1000 = 0.050
P(Y_fut=3)                = 200 times out of 1000 = 0.200
P(Y_past=1)               = 300 times out of 1000 = 0.300

STEP 5: Apply the formula
─────────────────────────
TE = Σ P(X, Y_past, Y_fut) × log₂[
         P(X, Y_past, Y_fut) × P(Y_past)
         ───────────────────────────────────────
         P(X, Y_past) × P(Y_fut)
     ]

In simple terms: For each pattern, check if X provides extra
information about Y_fut beyond what Y_past already tells us.

STEP 6: Sum everything up
─────────────────────────
TE = 0.123 bits

Interpretation:
• TE = 0.123 bits means knowing X reduces uncertainty about Y_fut
  by 12.3% compared to knowing Y_past alone.
• TE > 0 means X provides predictive information about Y.
""")


# ============================================================================
# SECTION 4: REAL-WORLD EXAMPLES
# ============================================================================

def create_real_world_examples():
    """
    Create visual examples of real-world TE applications.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Example 1: Weather patterns
    ax = axes[0, 0]
    np.random.seed(42)
    n = 200
    
    # Temperature causes humidity (with delay)
    temp = 20 + 5 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.randn(n) * 0.5
    humidity = 0.6 * temp[:-1] + np.random.randn(n-1) * 0.3
    humidity = np.concatenate([[20], humidity])
    
    ax.plot(temp[:100], 'b', alpha=0.7, label='Temperature')
    ax.plot(humidity[:100], 'r', alpha=0.7, label='Humidity')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Example 1: Weather (Temp → Humidity)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Example 2: Stock prices
    ax = axes[0, 1]
    # Tech stocks often move together
    sp500 = np.cumsum(np.random.randn(n) * 0.02) + 3000
    tech = 0.7 * sp500[:-1] + np.random.randn(n-1) * 0.3
    tech = np.concatenate([[3000], tech])
    
    ax.plot(sp500[:100], 'g', alpha=0.7, label='S&P 500')
    ax.plot(tech[:100], 'orange', alpha=0.7, label='Tech Stocks')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title('Example 2: Finance (Market → Tech)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Example 3: Neural activity
    ax = axes[1, 0]
    # Create spiking patterns
    t = np.linspace(0, 10, n)
    neuron_a = 1 / (1 + np.exp(-10 * (t - 5))) + np.random.randn(n) * 0.1
    neuron_b = 0.7 * neuron_a[:-1] + np.random.randn(n-1) * 0.3
    neuron_b = np.concatenate([[0], neuron_b])
    
    ax.plot(t[:100], neuron_a[:100], 'b', alpha=0.7, label='Neuron A')
    ax.plot(t[:100], neuron_b[:100], 'r', alpha=0.7, label='Neuron B')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Activity')
    ax.set_title('Example 3: Neuroscience (Neuron A → Neuron B)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Example 4: Climate (El Niño)
    ax = axes[1, 1]
    # Sea surface temperature affects rainfall
    sst = 25 + 2 * np.sin(np.linspace(0, 2*np.pi, n)) + np.random.randn(n) * 0.5
    rainfall = 0.5 * sst[:-1] + np.random.randn(n-1) * 0.4
    rainfall = np.maximum(0, np.concatenate([[0], rainfall]))
    
    ax.plot(t[:100], sst[:100], 'blue', alpha=0.7, label='Sea Temperature')
    ax.plot(t[:100], rainfall[:100], 'green', alpha=0.7, label='Rainfall')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Example 4: Climate (SST → Rainfall)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def explain_real_world_applications():
    """
    Explain real-world applications of Transfer Entropy.
    """
    print("\n" + "=" * 70)
    print("SECTION 4: REAL-WORLD APPLICATIONS")
    print("=" * 70)
    
    print("""
Transfer Entropy is used in many fields:

1. NEUROSCIENCE
   └─ Brain connectivity analysis
   └─ EEG/fMRI data interpretation
   └─ Understanding neural circuits

2. FINANCE
   └─ Stock market causation analysis
   └─ Risk propagation between assets
   └─ Trading signal detection

3. CLIMATOLOGY
   └─ El Niño forecasting
   └─ Weather pattern prediction
   └─ Climate change analysis

4. BIOLOGY
   └─ Gene regulatory networks
   └─ Protein interaction pathways
   └→ Population dynamics

5. ENGINEERING
   └─ Control systems analysis
   └─ Sensor network monitoring
   └─ Fault diagnosis

6. MEDICINE
   └─ Disease progression modeling
   └─ Drug interaction effects
   └─ Physiological signal analysis
""")


# ============================================================================
# SECTION 5: COMMON MISCONCEPTIONS
# ============================================================================

def explain_common_misconceptions():
    """
    Address common misconceptions about Transfer Entropy.
    """
    print("\n" + "=" * 70)
    print("SECTION 5: COMMON MISCONCEPTIONS")
    print("=" * 70)
    
    print("""
❌ Misconception 1: "High TE means strong causation"
   ✓ Reality: TE measures information flow, not causation strength.
              Need statistical testing to confirm significance.

❌ Misconception 2: "TE works on any data"
   ✓ Reality: TE requires:
              - Stationary data (stable statistics)
              - Sufficient data points (hundreds minimum)
              - Appropriate time resolution

❌ Misconception 3: "TE finds all causal relationships"
   ✓ Reality: TE only finds pairwise causation.
              Complex systems may have confounding variables.
              Use partial TE for more advanced analysis.

❌ Misconception 4: "TE = correlation"
   ✓ Reality: Correlation is symmetric and timing-agnostic.
              TE is asymmetric and explicitly uses timing.

❌ Misconception 5: "TE is always positive"
   ✓ Reality: TE should be ≥ 0 by definition.
              Negative values indicate computation errors.

❌ Misconception 6: "More bins = better results"
   ✓ Reality: Too many bins → sparse data → unreliable estimates.
              Too few bins → loss of information.
              4-8 bins is usually a good starting point.
""")


# ============================================================================
# QUICK START GUIDE
# ============================================================================

def quick_start():
    """
    Quick start guide for using Transfer Entropy.
    """
    print("\n" + "=" * 70)
    print("QUICK START GUIDE")
    print("=" * 70)
    
    print("""
Step 1: Install required packages
─────────────────────────────────
pip install numpy scipy matplotlib

Step 2: Import and create calculator
────────────────────────────────────
from transfer_entropy_basics import TransferEntropyCalculator

calc = TransferEntropyCalculator(n_bins=8, lag=1)

Step 3: Compute Transfer Entropy
────────────────────────────────
te = calc.compute_transfer_entropy(x, y)
print(f"TE(X→Y) = {te:.4f} bits")

Step 4: Check directionality
────────────────────────────
te_reverse = calc.compute_transfer_entropy(y, x)
print(f"TE(Y→X) = {te_reverse:.4f} bits")

Step 5: Interpret results
─────────────────────────
if te > te_reverse:
    print("X → Y: X causes Y")
elif te_reverse > te:
    print("Y → X: Y causes X")
else:
    print("No clear causal direction")

Step 6: Validate with surrogate testing
───────────────────────────────────────
# Shuffle X to break causality
x_shuffled = np.random.permutation(x)
te_randomized = calc.compute_transfer_entropy(x_shuffled, y)

# If real TE >> randomized TE, the causation is real
if te > 2 * te_randomized:
    print("Significant causal relationship!")
""")


def main():
    """Main function to run all explanations."""
    print("=" * 70)
    print("TRANSFER ENTROPY BASICS - EDUCATIONAL TUTORIAL")
    print("=" * 70)
    
    # Run explanations
    explain_what_is_te()
    explain_correlation_vs_te()
    explain_te_calculation()
    explain_real_world_applications()
    explain_common_misconceptions()
    quick_start()
    
    # Create visualization
    print("\n" + "=" * 70)
    print("CREATING VISUAL EXAMPLES...")
    print("=" * 70)
    
    fig = create_real_world_examples()
    fig.savefig('docs/transfer_entropy_real_world_examples.png', 
                dpi=150, bbox_inches='tight')
    print("✓ Saved: transfer_entropy_real_world_examples.png")
    
    print("\n" + "=" * 70)
    print("TUTORIAL COMPLETE")
    print("=" * 70)
    print("\nYou now understand:")
    print("• What Transfer Entropy is and why it matters")
    print("• How it differs from correlation")
    print("• The step-by-step calculation process")
    print("• Real-world applications across fields")
    print("• Common pitfalls to avoid")
    print("• How to use it in your own projects")


if __name__ == "__main__":
    main()
