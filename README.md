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

## Educational Materials

Want to understand Transfer Entropy without the math? We've got you covered!

### For Non-Technical Audiences

1. **Transfer Entropy Explained** - Visual guide with diagrams:
   ```bash
   python transfer_entropy_explained.py
   ```
   Creates: `transfer_entropy_explained.png` and `correlation_vs_transfer_entropy.png`

2. **Tutorial** - Plain English explanation:
   Read `transfer_entropy_tutorial.md` for a step-by-step breakdown of:
   - What causation means
   - How transfer entropy detects it
   - Why spiking neural networks are useful
   - Real-world analogies (dominoes, etc.)

### Key Concepts Explained

- **Correlation vs. Causation**: Why they're different
- **Transfer Entropy**: A measure of directional information flow
- **Spiking Neural Networks**: How they preserve timing information
- **Directionality**: Why A→B ≠ B→A

## Project Structure

```
spikingneuralnetworks/
├── spiking_neural_networks/   # Main library
│   ├── __init__.py
│   ├── lif_neuron.py          # LIF neuron model
│   └── snn.py                  # SpikingNeuralNetwork class
├── causal_inference.py        # Transfer entropy causal analysis
├── run_causal.py              # Quick causal demo script
├── transfer_entropy_explained.py  # Educational visualizations
├── transfer_entropy_tutorial.md   # Non-technical tutorial
├── examples/
│   └── basic_snn_demo.py      # Demo script
├── test_te*.py                # Transfer entropy tests
├── pyproject.toml             # Project configuration
├── README.md                  # This file
├── causal_results.png         # Causal analysis visualization
├── transfer_entropy_explained.png  # Educational figure
└── correlation_vs_transfer_entropy.png  # Comparison figure
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
