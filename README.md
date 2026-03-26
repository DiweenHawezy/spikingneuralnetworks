# Spiking Neural Networks (SNN)

A Python library and framework for implementing and simulating Spiking Neural Networks.

## Project Purpose

This repository tests whether a small LLM Agent can replicate research work on spiking neural networks, including implementation of neuron models, learning rules, and causal inference methods.

## Overview

Spiking Neural Networks (SNNs) are bio-inspired neural network models that communicate via discrete electrical pulses called spikes. Unlike traditional ANNs that use continuous activation values, SNNs capture temporal dynamics of biological neurons, making them more efficient for neuromorphic computing and event-driven processing.

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

```bash
pip install -r pyproject.toml
```

Or install dependencies manually:
```bash
pip install numpy scipy matplotlib
```

## Quick Start

### Single Neuron

```python
from spiking_neural_networks import LIFNeuron

neuron = LIFNeuron(v_rest=0.0, v_thresh=1.0, v_reset=-0.5, tau=20.0, dt=1.0)

for step in range(100):
    spike = neuron.step(input_current=0.1)
```

### Spiking Network

```python
from spiking_neural_networks import SpikingNeuralNetwork, LIFNeuron

network = SpikingNeuralNetwork(n_neurons=5, neuron_model=LIFNeuron, dt=1.0)
spike_times = network.run(steps=1000)
counts = network.get_spike_counts()
```

### Transfer Entropy (Causal Inference)

```python
from transfer_entropy_implementation import TransferEntropyCalculator

calc = TransferEntropyCalculator(n_bins=8, lag=1)
te = calc.compute_transfer_entropy_joint(x, y)
```

## Transfer Entropy Explained

**What it measures:** Directional information flow between time series. Does knowing the past of X help predict Y's future?

**Key properties:**
- **Asymmetric:** TE(X→Y) ≠ TE(Y→X) - detects causal direction
- **Non-linear:** No assumptions about relationship type
- **Timing-aware:** Uses explicit temporal lags

**Interpretation:**
| TE Value | Meaning |
|----------|---------|
| TE ≈ 0 | No causal influence |
| TE > 0 | X provides predictive info about Y |
| TE(X→Y) > TE(Y→X) | X likely causes Y |

## Project Structure

```
spikingneuralnetworks/
├── spiking_neural_networks/   # Main library
│   ├── __init__.py
│   ├── lif_neuron.py
│   └── snn.py
├── causal_inference.py        # Transfer entropy analysis
├── transfer_entropy_implementation.py
├── transfer_entropy_basics.py
├── run_causal.py
├── examples/
│   └── basic_snn_demo.py
└── tests/
```

## Running Scripts

| Script | Description |
|--------|-------------|
| `python examples/basic_snn_demo.py` | Basic SNN simulation demo |
| `python run_causal.py` | Causal inference with TE |
| `python transfer_entropy_implementation.py` | Detailed TE implementation walkthrough |
| `python transfer_entropy_basics.py` | Educational tutorial |

## Testing

```bash
pytest tests/ -v
```

## License

MIT License

## References

- [Spiking Neural Networks - Wikipedia](https://en.wikipedia.org/wiki/Spiking_neural_network)
- [Brian2 - SNN simulation](https://brian2.readthedocs.io/)
- [Transfer Entropy - Scholarpedia](http://www.scholarpedia.org/article/Transfer_entropy)

---

*Built with ❤️ for the future of neuromorphic computing*
