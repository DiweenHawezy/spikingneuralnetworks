"""
Spiking Neural Networks Library

A Python framework for simulating spiking neural networks.
"""

__version__ = "0.1.0"

from .lif_neuron import LIFNeuron
from .snn import SpikingNeuralNetwork

__all__ = ["LIFNeuron", "SpikingNeuralNetwork"]
