"""
Spiking Neural Network Implementation

A basic SNN class that manages multiple LIF neurons and synaptic connections.
"""

import numpy as np
from .lif_neuron import LIFNeuron


class SpikingNeuralNetwork:
    """
    Simple Spiking Neural Network
    
    A basic SNN implementation with multiple neurons and synaptic weights.
    
    Parameters:
    -----------
    n_neurons : int
        Number of neurons in the network
    neuron_model : class, default=LIFNeuron
        Neuron model class to use
    dt : float, default=0.1
        Time step (ms)
    """
    
    def __init__(self, n_neurons=10, neuron_model=LIFNeuron, dt=0.1):
        self.n_neurons = n_neurons
        self.neuron_model = neuron_model
        self.dt = dt
        
        # Initialize neurons
        self.neurons = [neuron_model() for _ in range(n_neurons)]
        
        # Synaptic weights (initially random)
        self.weights = np.random.randn(n_neurons, n_neurons) * 0.1
        
        # Current state
        self.t = 0
        self.spike_times = []
    
    def step(self):
        """
        Perform one time step of the network
        
        Returns:
        --------
        spikes : numpy.ndarray
            Binary spike vector for all neurons
        """
        spikes = np.zeros(self.n_neurons)
        
        for i in range(self.n_neurons):
            # Calculate input from other neurons
            input_current = np.sum(self.weights[i] * spikes)
            
            # Step each neuron
            self.neurons[i].step(input_current)
            
            # Record spike
            if self.neurons[i].spike_history:
                spikes[i] = 1.0
        
        return spikes
    
    def run(self, steps=1000):
        """
        Run the network for a number of steps
        
        Parameters:
        -----------
        steps : int
            Number of time steps to simulate
        """
        print(f"Running SNN for {steps} steps...")
        print("-" * 50)
        
        self.spike_times = []
        
        for t in range(steps):
            spikes = self.step()
            
            # Record spikes
            if np.any(spikes):
                self.spike_times.append((t, spikes))
            
            # Print progress every 100 steps
            if (t + 1) % 100 == 0:
                print(f"Step {t+1}/{steps}, Total spikes: {len(self.spike_times)}")
        
        print("-" * 50)
        print(f"Simulation complete. Total spikes: {len(self.spike_times)}")
        return self.spike_times
    
    def add_input(self, neuron_idx, input_current):
        """Add input current to a specific neuron"""
        if 0 <= neuron_idx < self.n_neurons:
            self.neurons[neuron_idx].step(input_current)
            return True
        return False
    
    def get_spike_counts(self):
        """Get spike count for each neuron"""
        return [len(neuron.spike_history) for neuron in self.neurons]
    
    def reset(self):
        """Reset all neurons"""
        for neuron in self.neurons:
            neuron.reset()
        self.spike_times = []


if __name__ == "__main__":
    # Simple test
    network = SpikingNeuralNetwork(n_neurons=5)
    network.run(steps=1000)
    print(f"\nSpike counts per neuron: {network.get_spike_counts()}")
