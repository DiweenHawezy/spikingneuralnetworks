#!/usr/bin/env python3
"""
Basic Spiking Neural Network Demo

This example demonstrates a simple spiking neural network simulation.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spiking_neural_networks import LIFNeuron, SpikingNeuralNetwork


def demo_single_neuron():
    """Demonstrate a single LIF neuron"""
    print("\n" + "=" * 60)
    print("Demo 1: Single LIF Neuron")
    print("=" * 60)
    
    neuron = LIFNeuron(
        v_rest=0.0,
        v_thresh=1.0,
        v_reset=-0.5,
        tau=20.0,
        dt=1.0
    )
    
    print(f"\nNeuron parameters:")
    print(f"  Resting potential: {neuron.v_rest}")
    print(f"  Threshold: {neuron.v_thresh}")
    print(f"  Reset potential: {neuron.v_reset}")
    print(f"  Time constant: {neuron.tau} ms")
    
    # Apply constant input
    input_current = 0.1
    print(f"\nApplying constant input current: {input_current}")
    
    for step in range(50):
        spike = neuron.step(input_current)
        if spike:
            print(f"  Step {step}: SPKE! v={neuron.v:.3f} -> {neuron.v_reset:.3f}")
    
    print(f"\nTotal spikes: {len(neuron.spike_history)}")


def demo_small_network():
    """Demonstrate a small spiking neural network"""
    print("\n" + "=" * 60)
    print("Demo 2: Small Spiking Neural Network")
    print("=" * 60)
    
    # Create a network with 5 neurons
    network = SpikingNeuralNetwork(
        n_neurons=5,
        neuron_model=LIFNeuron,
        dt=1.0
    )
    
    print(f"\nNetwork: {network.n_neurons} neurons")
    print(f"Running for 1000 time steps...")
    
    # Run simulation
    network.run(steps=1000)
    
    spike_counts = network.get_spike_counts()
    print(f"\nSpike counts per neuron: {spike_counts}")
    print(f"Network activity: {'Active' if any(c > 0 for c in spike_counts) else 'Inactive'}")


def demo_external_input():
    """Demonstrate network with external input"""
    print("\n" + "=" * 60)
    print("Demo 3: Network with External Input")
    print("=" * 60)
    
    network = SpikingNeuralNetwork(
        n_neurons=3,
        neuron_model=LIFNeuron,
        dt=1.0
    )
    
    print("\nApplying external input to neuron 0...")
    
    # Apply input to first neuron
    for _ in range(100):
        network.add_input(0, 0.5)
    
    print(f"Neuron 0 spike count: {network.get_spike_counts()[0]}")
    print(f"Other neurons spike count: {sum(network.get_spike_counts()[1:])}")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# Spiking Neural Network - Basic Demo")
    print("#" * 60)
    
    try:
        demo_single_neuron()
        demo_small_network()
        demo_external_input()
        
        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
