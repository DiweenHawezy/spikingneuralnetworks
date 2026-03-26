# Spiking Neural Networks (SNN)

A Python library and framework for implementing and simulating Spiking Neural Networks.

## Project Purpose

This repository tests whether a small LLM Agent can replicate research work on spiking neural networks, including implementation of neuron models, learning rules, and causal inference methods.

## Overview

Spiking Neural Networks (SNNs) are bio-inspired neural network models that communicate via discrete electrical pulses called spikes (action potentials). Unlike traditional artificial neural networks (ANNs) that use continuous activation values, SNNs capture the temporal dynamics of biological neurons, making them more efficient for certain types of tasks, especially in neuromorphic computing and event-driven processing.

## Features

- [x] Leaky Integrate-and-Fire (LIF) neuron model
- [x] SpikingNeuralNetwork class with multiple neurons and synaptic weights
- [x] Granger causality and causal inference using Transfer Entropy
- [x] Educational materials for non-technical audiences
- [ ] Temporal learning rules (STDP, etc.)
- [ ] Izhikevich neuron model
- [ ] Neuromorphic hardware support
- [ ] Energy-efficient computation
- [ ] Event-driven processing

## Installation

Install dependencies using pip:

```bash
pip install -r pyproject.toml
```

Or manually install required packages:

```bash
pip install numpy scipy matplotlib
```

## Usage

### Single Neuron

```python
from spiking_neural_networks import LIFNeuron

# Create a Leaky Integrate-and-Fire neuron
neuron = LIFNeuron(
    v_rest=0.0,
    v_thresh=1.0,
    v_reset=-0.5,
    tau=20.0,
    dt=1.0
)

# Simulate with input current
for step in range(100):
    spike = neuron.step(input_current=0.1)
    if spike:
        print(f"Spiked at step {step}")
```

### Spiking Neural Network

```python
from spiking_neural_networks import SpikingNeuralNetwork, LIFNeuron

# Create a network with 5 neurons
network = SpikingNeuralNetwork(
    n_neurons=5,
    neuron_model=LIFNeuron,
    dt=1.0
)

# Run simulation for 1000 steps
spike_times = network.run(steps=1000)

# Get spike counts per neuron
counts = network.get_spike_counts()
```

### Causal Inference

```python
from causal_inference import run_causal_analysis

# Run causal analysis with transfer entropy
results = run_causal_analysis()

# Results include:
# - A, B: Causal time series
# - TE: Transfer entropy values
# - TE_randomized: Randomized transfer entropy (baseline)
# - plot_path: Path to causal plot
```

## Transfer Entropy (TE) - Detailed Documentation

### What is Transfer Entropy?

Transfer Entropy (TE) is a measure of **directional information flow** between two time series. It answers the fundamental question:

> "Does knowing the past of series X help predict the future of series Y, beyond what the past of Y already tells us?"

**Key properties:**
- **Asymmetric**: TE(X→Y) ≠ TE(Y→X) - can detect causal direction
- **Model-free**: No assumptions about linear relationships
- **Timing-aware**: Explicitly uses temporal information
- **Non-linear**: Can detect complex causal relationships

### Mathematical Foundation

```
TE(X → Y) = I(Y_future; X_past | Y_past)
          = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
          = Σ P(X_past, Y_past, Y_future) × log₂[P(X_past, Y_past, Y_future) × P(Y_past) / (P(X_past, Y_past) × P(Y_future))]
```

Where:
- **I** = Mutual Information
- **H** = Entropy (uncertainty)
- **X_past** = Past values of source series X
- **Y_past** = Past values of target series Y
- **Y_future** = Future values of target series Y

### Why Use Transfer Entropy?

**vs. Correlation:**
| Property | Correlation | Transfer Entropy |
|----------|-------------|------------------|
| Symmetric | Yes (corr(X,Y) = corr(Y,X)) | No (asymmetric) |
| Directionality | No | Yes (X→Y vs Y→X) |
| Timing | No | Yes (uses lag) |
| Linearity | Assumes linear | Non-linear capable |
| Causation | Cannot detect | Can detect |

### Implementation Files

1. **transfer_entropy_implementation.py** - Complete implementation with:
   - Step-by-step calculation
   - Multiple methods (joint probability, conditional entropy)
   - Detailed visualizations
   - Run: `python transfer_entropy_implementation.py`

2. **transfer_entropy_basics.py** - Educational tutorial with:
   - Plain English explanations
   - Real-world examples
   - Common misconceptions
   - Run: `python transfer_entropy_basics.py`

3. **causal_inference.py** - Applied causal analysis with:
   - Spiking neural network encoding
   - Synthetic data generation
   - Directionality testing
   - Run: `python run_causal.py`

### Quick Usage

```python
from causal_inference import run_causal_analysis

# Run causal analysis
results = run_causal_analysis()

# Or use the calculator directly
from transfer_entropy_implementation import TransferEntropyCalculator

calc = TransferEntropyCalculator(n_bins=8, lag=1)
te = calc.compute_transfer_entropy_joint(x, y)
print(f"TE(X→Y) = {te:.4f} bits")
```

### Interpretation Guide

| TE Value | Interpretation |
|----------|----------------|
| TE ≈ 0 | No causal influence |
| TE > 0 | X provides predictive information about Y |
| TE(X→Y) > TE(Y→X) | X likely causes Y |
| TE(Y→X) > TE(X→Y) | Y likely causes X |

### Real-World Applications

- **Neuroscience**: Brain connectivity, neural circuits
- **Finance**: Stock market causation, risk propagation
- **Climate**: El Niño forecasting, weather patterns
- **Biology**: Gene regulatory networks, protein pathways
- **Medicine**: Disease progression, drug interactions

### Key Takeaways

1. **TE measures information flow, not just correlation**
2. **Higher TE in one direction indicates causation direction**
3. **Discretization (binning) converts continuous data for TE calculation**
4. **Statistical validation (surrogate testing) confirms significance**
5. **TE works with spiking neural network data naturally**

## Project Structure

```
spikingneuralnetworks/
├── spiking_neural_networks/   # Main library
│   ├── __init__.py
│   ├── lif_neuron.py          # LIF neuron model
│   └── snn.py                  # SpikingNeuralNetwork class
├── causal_inference.py        # Transfer entropy causal analysis
├── run_causal.py              # Quick causal demo script
├── transfer_entropy_implementation.py  # Complete TE implementation
├── transfer_entropy_basics.py         # Educational tutorial
├── transfer_entropy_explained.py  # Educational visualizations
├── transfer_entropy_tutorial.md   # Non-technical tutorial
├── examples/
│   └── basic_snn_demo.py      # Demo script
├── pyproject.toml             # Project configuration
├── README.md                  # This file
├── causal_results.png         # Causal analysis visualization
├── transfer_entropy_implementation_diagram.png  # TE implementation diagram
├── transfer_entropy_real_world_examples.png       # Real-world examples
└── *.png                      # Generated visualizations
```

## Examples

Run the basic demo:

```bash
python examples/basic_snn_demo.py
```

Run causal inference:

```bash
python run_causal.py
```

## Testing

Run the transfer entropy tests:

```bash
python test_te.py
python test_te_correct.py
```

## License

This project is licensed under the MIT License.

## References

- [Spiking Neural Networks - Wikipedia](https://en.wikipedia.org/wiki/Spiking_neural_network)
- [Brian2 - Python simulation environment for spiking neural networks](https://brian2.readthedocs.io/)
- [PyTorch SNN](https://pytorch-snn.readthedocs.io/)
- [Transfer Entropy - Scholarpedia](http://www.scholarpedia.org/article/Transfer_entropy)

---

*Built with ❤️ for the future of neuromorphic computing*
