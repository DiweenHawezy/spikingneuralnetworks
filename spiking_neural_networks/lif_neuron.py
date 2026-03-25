"""
Leaky Integrate-and-Fire (LIF) Neuron Implementation

The LIF neuron is one of the most common spiking neuron models.
It integrates incoming inputs and fires when the membrane potential
reaches a threshold, then resets.
"""

import numpy as np


class LIFNeuron:
    """
    Leaky Integrate-and-Fire Neuron Model
    
    Parameters:
    -----------
    v_rest : float, default=0.0
        Resting membrane potential
    v_thresh : float, default=1.0
        Firing threshold
    v_reset : float, default=0.0
        Reset potential after firing
    tau : float, default=20.0
        Membrane time constant (ms)
    dt : float, default=0.1
        Time step for integration (ms)
    """
    
    def __init__(self, v_rest=0.0, v_thresh=1.0, v_reset=0.0, tau=20.0, dt=0.1):
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.tau = tau
        self.dt = dt
        self.v = v_rest
        self.t = 0.0
        self.spike_history = []
    
    def step(self, input_current=0.0):
        """
        Perform one time step of the LIF equation
        
        Parameters:
        -----------
        input_current : float
            Current input to the neuron
            
        Returns:
        --------
        spike : bool
            Whether the neuron spiked in this step
        """
        self.t += self.dt
        
        # Leaky integration
        self.v = self.v * np.exp(-self.dt / self.tau) + input_current * self.dt
        
        # Check for spike
        spike = False
        if self.v >= self.v_thresh:
            self.v = self.v_reset
            spike = True
        
        if spike:
            self.spike_history.append(self.t)
        
        return spike
    
    def fire(self):
        """Force the neuron to fire"""
        self.v = self.v_reset
        self.t += self.dt
        self.spike_history.append(self.t)
        return True
    
    def get_state(self):
        """Return current state of the neuron"""
        return {
            'v': self.v,
            't': self.t,
            'v_rest': self.v_rest,
            'v_thresh': self.v_thresh,
            'v_reset': self.v_reset,
            'tau': self.tau,
            'dt': self.dt
        }
    
    def reset(self):
        """Reset neuron to resting state"""
        self.v = self.v_rest
        self.t = 0.0
        self.spike_history = []


if __name__ == "__main__":
    # Simple test
    neuron = LIFNeuron(v_thresh=1.0, v_reset=-0.5, tau=10.0)
    
    print("Testing LIF Neuron")
    print("=" * 50)
    
    for i in range(100):
        spike = neuron.step(input_current=0.1)
        if spike:
            print(f"Spiked at t={neuron.t:.2f}")
    
    print(f"\nTotal spikes: {len(neuron.spike_history)}")
