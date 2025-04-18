"""
Neuromorphic Computing module for cryptocurrency applications.

This module implements neuromorphic computing methods inspired by the
brain's neural architecture, offering novel approaches to temporal data
processing and pattern recognition in cryptocurrency markets.

Classes:
    SpikingNetwork: Spiking Neural Network implementation
    LiquidStateModel: Liquid State Machine implementation
    ReservoirComputing: Echo State Network for time series
    NeuroevolutionModel: Neuroevolution techniques for adaptive models
"""

import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Union, Optional, Tuple, Callable
import copy
import random
import matplotlib.pyplot as plt

class SpikingNetwork:
    """
    Spiking Neural Network implementation.
    
    Spiking Neural Networks (SNNs) process information through discrete spikes
    rather than continuous activations, potentially offering greater efficiency
    for processing temporal market data.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
               threshold: float = 1.0, refractory_period: int = 4,
               time_steps: int = 100, learning_rate: float = 0.01):
        """
        Initialize a Spiking Neural Network.
        
        Args:
            input_size: Number of input neurons
            hidden_size: Number of hidden neurons
            output_size: Number of output neurons
            threshold: Spiking threshold
            refractory_period: Refractory period in time steps
            time_steps: Number of time steps for temporal processing
            learning_rate: Learning rate for STDP
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.threshold = threshold
        self.refractory_period = refractory_period
        self.time_steps = time_steps
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.w_in_hidden = np.random.normal(0, 0.1, (input_size, hidden_size))
        self.w_hidden_out = np.random.normal(0, 0.1, (hidden_size, output_size))
        
        # Initialize neuron states
        self.reset_state()
        
    def reset_state(self) -> None:
        """
        Reset the state of all neurons.
        """
        # Membrane potentials
        self.hidden_potential = np.zeros(self.hidden_size)
        self.output_potential = np.zeros(self.output_size)
        
        # Refractory counters (time steps until neuron can fire again)
        self.hidden_refractory = np.zeros(self.hidden_size, dtype=int)
        self.output_refractory = np.zeros(self.output_size, dtype=int)
        
        # Spike histories for STDP
        self.hidden_spikes_history = np.zeros((self.time_steps, self.hidden_size))
        self.output_spikes_history = np.zeros((self.time_steps, self.output_size))
        
        # Current time step
        self.current_step = 0
        
    def _encode_input(self, x: np.ndarray) -> np.ndarray:
        """
        Convert continuous inputs to spike trains.
        
        Args:
            x: Input features (batch_size, input_size)
            
        Returns:
            Spike train encoding (time_steps, batch_size, input_size)
        """
        batch_size = x.shape[0]
        spike_trains = np.zeros((self.time_steps, batch_size, self.input_size))
        
        # Rate coding: higher values lead to more frequent spikes
        for t in range(self.time_steps):
            # Generate spikes with probability proportional to input value
            spikes = np.random.random((batch_size, self.input_size)) < x
            spike_trains[t] = spikes
            
        return spike_trains
    
    def _decode_output(self, spikes: np.ndarray) -> np.ndarray:
        """
        Convert output spike trains to continuous values.
        
        Args:
            spikes: Output spike trains (time_steps, batch_size, output_size)
            
        Returns:
            Decoded outputs (batch_size, output_size)
        """
        # Count spikes over time for rate coding
        spike_counts = np.sum(spikes, axis=0)
        
        # Normalize by time steps
        return spike_counts / self.time_steps
    
    def _leaky_integrate_and_fire(self, inputs: np.ndarray, 
                                potentials: np.ndarray, 
                                refractory: np.ndarray, 
                                weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply the leaky integrate-and-fire neuron model.
        
        Args:
            inputs: Input spikes
            potentials: Current membrane potentials
            refractory: Current refractory counters
            weights: Connection weights
            
        Returns:
            Tuple of (output_spikes, updated_potentials, updated_refractory)
        """
        # Leak term (decay of membrane potential)
        leak = 0.1 * potentials
        
        # Integrate input spikes
        input_current = np.dot(inputs, weights)
        
        # Update potential for neurons not in refractory period
        active_mask = (refractory <= 0)
        potentials[active_mask] = potentials[active_mask] - leak[active_mask] + input_current[active_mask]
        
        # Generate spikes where potential exceeds threshold
        spikes = np.zeros_like(potentials)
        spike_mask = (potentials >= self.threshold) & active_mask
        spikes[spike_mask] = 1
        
        # Reset potential and set refractory period for spiking neurons
        potentials[spike_mask] = 0
        refractory[spike_mask] = self.refractory_period
        
        # Decrement refractory counters
        refractory = np.maximum(refractory - 1, 0)
        
        return spikes, potentials, refractory
    
    def _stdp_update(self, pre_spikes: np.ndarray, post_spikes: np.ndarray, 
                   weights: np.ndarray) -> np.ndarray:
        """
        Apply Spike-Timing-Dependent Plasticity (STDP) learning rule.
        
        Args:
            pre_spikes: Presynaptic spike history
            post_spikes: Postsynaptic spike history
            weights: Current weights
            
        Returns:
            Updated weights
        """
        # Parameters for STDP
        a_plus = self.learning_rate  # Magnitude of LTP
        a_minus = -self.learning_rate * 1.2  # Magnitude of LTD (slightly stronger)
        tau = 20  # Time constant
        
        # Initialize weight changes
        dw = np.zeros_like(weights)
        
        # Loop over each pair of pre and post neurons
        for i in range(pre_spikes.shape[1]):
            for j in range(post_spikes.shape[1]):
                # Extract spike times
                pre_spike_times = np.where(pre_spikes[:, i])[0]
                post_spike_times = np.where(post_spikes[:, j])[0]
                
                # No spikes, no update
                if len(pre_spike_times) == 0 or len(post_spike_times) == 0:
                    continue
                
                # For each pre-post spike pair
                for t_pre in pre_spike_times:
                    for t_post in post_spike_times:
                        # Compute time difference
                        delta_t = t_post - t_pre
                        
                        # LTP: Post spike after pre spike
                        if delta_t > 0:
                            dw[i, j] += a_plus * np.exp(-delta_t / tau)
                        # LTD: Pre spike after post spike
                        elif delta_t < 0:
                            dw[i, j] += a_minus * np.exp(delta_t / tau)
        
        # Update weights with constraints
        weights += dw
        weights = np.clip(weights, 0, None)  # Non-negative weights
        
        return weights
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            x: Input features (batch_size, input_size)
            
        Returns:
            Network outputs (batch_size, output_size)
        """
        # Reset network state
        self.reset_state()
        
        # Encode inputs as spike trains
        input_spikes = self._encode_input(x)
        batch_size = x.shape[0]
        
        # Initialize output spike storage
        hidden_spikes = np.zeros((self.time_steps, batch_size, self.hidden_size))
        output_spikes = np.zeros((self.time_steps, batch_size, self.output_size))
        
        # Process time steps
        for t in range(self.time_steps):
            # Process hidden layer
            for b in range(batch_size):
                h_spikes, self.hidden_potential, self.hidden_refractory = self._leaky_integrate_and_fire(
                    input_spikes[t, b], 
                    self.hidden_potential,
                    self.hidden_refractory,
                    self.w_in_hidden
                )
                hidden_spikes[t, b] = h_spikes
                
                # Process output layer
                o_spikes, self.output_potential, self.output_refractory = self._leaky_integrate_and_fire(
                    h_spikes,
                    self.output_potential,
                    self.output_refractory,
                    self.w_hidden_out
                )
                output_spikes[t, b] = o_spikes
            
            # Store spike history for STDP
            self.hidden_spikes_history[t] = hidden_spikes[t, 0]  # For simplicity, only use first batch item
            self.output_spikes_history[t] = output_spikes[t, 0]
            
            self.current_step = t
        
        # Decode output spikes to continuous values
        decoded_output = self._decode_output(output_spikes)
        
        return decoded_output
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 10) -> List[float]:
        """
        Train the network using STDP.
        
        Args:
            x: Training features
            y: Target outputs
            epochs: Number of training epochs
            
        Returns:
            List of error values per epoch
        """
        errors = []
        
        for epoch in range(epochs):
            total_error = 0
            
            # Process each sample
            for i in range(len(x)):
                # Forward pass
                output = self.forward(x[i:i+1])
                
                # Compute error
                error = np.mean((output - y[i:i+1])**2)
                total_error += error
                
                # Apply STDP
                self.w_in_hidden = self._stdp_update(
                    np.stack([x[i]] * self.time_steps),
                    self.hidden_spikes_history,
                    self.w_in_hidden
                )
                
                self.w_hidden_out = self._stdp_update(
                    self.hidden_spikes_history,
                    self.output_spikes_history,
                    self.w_hidden_out
                )
            
            # Average error for epoch
            avg_error = total_error / len(x)
            errors.append(avg_error)
            
            print(f"Epoch {epoch+1}/{epochs}, Error: {avg_error:.4f}")
            
        return errors
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Generate predictions with the network.
        
        Args:
            x: Input features
            
        Returns:
            Network predictions
        """
        return self.forward(x)


class LiquidStateModel:
    """
    Liquid State Machine implementation.
    
    Liquid State Machines (LSMs) are a type of reservoir computing that
    uses a recurrent spiking neural network as a reservoir to transform
    inputs into a high-dimensional state space.
    """
    
    def __init__(self, input_size: int, reservoir_size: int, output_size: int,
               connectivity: float = 0.2, spectral_radius: float = 0.9):
        """
        Initialize a Liquid State Machine.
        
        Args:
            input_size: Number of input features
            reservoir_size: Number of neurons in the reservoir
            output_size: Number of output neurons
            connectivity: Probability of connection between reservoir neurons
            spectral_radius: Controls stability of reservoir dynamics
        """
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.connectivity = connectivity
        self.spectral_radius = spectral_radius
        
        # Initialize weights
        self._initialize_weights()
        
        # Initialize readout (trained with linear regression)
        self.readout = None
        
    def _initialize_weights(self) -> None:
        """
        Initialize network weights for the reservoir computing model.
        """
        # Input to reservoir weights (fixed)
        self.w_in = np.random.uniform(-1, 1, (self.reservoir_size, self.input_size))
        
        # Reservoir internal weights (fixed)
        # Sparse connectivity
        self.w_res = np.random.uniform(-1, 1, (self.reservoir_size, self.reservoir_size))
        mask = np.random.random((self.reservoir_size, self.reservoir_size)) < self.connectivity
        self.w_res = self.w_res * mask
        
        # Scale by spectral radius for stability
        radius = np.max(np.abs(np.linalg.eigvals(self.w_res)))
        if radius > 0:
            self.w_res = self.w_res * (self.spectral_radius / radius)
            
    def _activate(self, x: np.ndarray) -> np.ndarray:
        """
        Apply a nonlinear activation function to the input.
        
        Args:
            x: Input array
            
        Returns:
            Activated values
        """
        # Hyperbolic tangent activation (common in LSMs)
        return np.tanh(x)
    
    def _run_reservoir(self, u: np.ndarray, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Run the reservoir dynamics on input sequence.
        
        Args:
            u: Input sequence of shape (time_steps, input_size)
            initial_state: Initial reservoir state (None for zero)
            
        Returns:
            Reservoir states for each time step (time_steps, reservoir_size)
        """
        time_steps = u.shape[0]
        
        # Initialize reservoir state
        if initial_state is None:
            x = np.zeros(self.reservoir_size)
        else:
            x = initial_state.copy()
            
        # Storage for reservoir states
        states = np.zeros((time_steps, self.reservoir_size))
        
        # Run reservoir dynamics
        for t in range(time_steps):
            # Update state: leak + input + recurrent
            leak = 0.1 * x  # Leaky integration
            input_term = np.dot(self.w_in, u[t])
            recurrent_term = np.dot(self.w_res, x)
            
            x = (1 - 0.1) * x + self._activate(input_term + recurrent_term)
            states[t] = x
            
        return states
    
    def fit(self, X: np.ndarray, y: np.ndarray, washout: int = 0) -> None:
        """
        Train the readout layer using collected reservoir states.
        
        Args:
            X: Input sequences of shape (n_samples, time_steps, input_size)
            y: Target outputs of shape (n_samples, output_size)
            washout: Number of initial time steps to discard
        """
        n_samples = X.shape[0]
        time_steps = X.shape[1]
        
        # Collect reservoir states for all sequences
        all_states = []
        
        for i in range(n_samples):
            states = self._run_reservoir(X[i])
            # Remove washout period and reshape
            if washout > 0 and washout < time_steps:
                effective_states = states[washout:].reshape(-1, self.reservoir_size)
            else:
                effective_states = states.reshape(-1, self.reservoir_size)
            all_states.append(effective_states)
            
        # Concatenate states and prepare targets
        all_states = np.vstack(all_states)
        
        # Repeat targets to match number of time points (minus washout)
        effective_length = time_steps - washout if washout > 0 else time_steps
        targets = np.repeat(y, effective_length, axis=0)
        
        # Add bias term
        all_states_bias = np.hstack((all_states, np.ones((all_states.shape[0], 1))))
        
        # Train readout with ridge regression
        reg_factor = 1e-8  # Ridge regularization
        self.readout = np.linalg.solve(
            all_states_bias.T @ all_states_bias + reg_factor * np.eye(all_states_bias.shape[1]),
            all_states_bias.T @ targets
        )
        
    def predict(self, X: np.ndarray, washout: int = 0) -> np.ndarray:
        """
        Generate predictions for input sequences.
        
        Args:
            X: Input sequences of shape (n_samples, time_steps, input_size)
            washout: Number of initial time steps to discard
            
        Returns:
            Predictions of shape (n_samples, output_size)
        """
        if self.readout is None:
            raise ValueError("Model has not been trained. Call fit() first.")
            
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.output_size))
        
        for i in range(n_samples):
            # Run reservoir
            states = self._run_reservoir(X[i])
            
            # Discard washout period
            if washout > 0 and washout < states.shape[0]:
                effective_states = states[washout:]
            else:
                effective_states = states
                
            # Average states over time for a single prediction per sequence
            avg_state = np.mean(effective_states, axis=0)
            
            # Add bias term
            state_bias = np.append(avg_state, 1)
            
            # Apply readout
            predictions[i] = np.dot(state_bias, self.readout)
            
        return predictions
    
    def visualize_reservoir(self, u: np.ndarray) -> None:
        """
        Visualize reservoir activity for an input sequence.
        
        Args:
            u: Input sequence of shape (time_steps, input_size)
        """
        states = self._run_reservoir(u)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(211)
        plt.plot(u)
        plt.title('Input Sequence')
        plt.xlabel('Time Step')
        plt.ylabel('Input Value')
        
        plt.subplot(212)
        plt.imshow(states.T, aspect='auto', cmap='viridis')
        plt.title('Reservoir Activity')
        plt.xlabel('Time Step')
        plt.ylabel('Neuron Index')
        plt.colorbar(label='Activation')
        
        plt.tight_layout()
        plt.show()


class ReservoirComputing:
    """
    Echo State Network for time series processing.
    
    Echo State Networks (ESNs) are a type of reservoir computing that
    use a recurrent neural network with fixed weights to transform inputs
    into a high-dimensional state space, useful for financial time series.
    """
    
    def __init__(self, input_size: int, reservoir_size: int, output_size: int,
               spectral_radius: float = 0.9, connectivity: float = 0.1,
               input_scaling: float = 1.0, leaking_rate: float = 0.3):
        """
        Initialize an Echo State Network.
        
        Args:
            input_size: Number of input features
            reservoir_size: Number of neurons in the reservoir
            output_size: Number of output dimensions
            spectral_radius: Spectral radius of reservoir weight matrix
            connectivity: Connectivity of reservoir (sparsity)
            input_scaling: Scaling of input weights
            leaking_rate: Leakage rate for reservoir neurons
        """
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.connectivity = connectivity
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate
        
        # Initialize weights
        self._initialize_weights()
        
        # Initialize readout weights
        self.w_out = None
        
    def _initialize_weights(self) -> None:
        """
        Initialize network weights for the ESN.
        """
        # Input to reservoir weights (fixed)
        self.w_in = np.random.uniform(-self.input_scaling, self.input_scaling, 
                                    (self.reservoir_size, self.input_size + 1))  # +1 for bias
        
        # Reservoir internal weights (fixed)
        # Generate a sparse random matrix
        self.w_res = np.random.uniform(-1, 1, (self.reservoir_size, self.reservoir_size))
        mask = np.random.random((self.reservoir_size, self.reservoir_size)) < self.connectivity
        self.w_res = self.w_res * mask
        
        # Scale by spectral radius
        eigenvalues = np.linalg.eigvals(self.w_res)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        if max_eigenvalue > 0:
            self.w_res = self.w_res * (self.spectral_radius / max_eigenvalue)
            
    def _update_reservoir(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Update reservoir state.
        
        Args:
            x: Current reservoir state
            u: Input with bias term
            
        Returns:
            Updated reservoir state
        """
        # Compute new state (leaky integration)
        x_new = (1 - self.leaking_rate) * x + self.leaking_rate * np.tanh(
            np.dot(self.w_in, u) + np.dot(self.w_res, x)
        )
        return x_new
    
    def _compute_reservoir_states(self, u: np.ndarray, washout: int = 0) -> np.ndarray:
        """
        Compute reservoir states for a given input sequence.
        
        Args:
            u: Input sequence of shape (time_steps, input_size)
            washout: Number of initial time steps to discard
            
        Returns:
            Reservoir states (effective_time_steps, reservoir_size)
        """
        time_steps = u.shape[0]
        
        # Initialize reservoir state
        x = np.zeros(self.reservoir_size)
        
        # Add bias term to inputs
        u_bias = np.hstack((u, np.ones((time_steps, 1))))
        
        # Collect reservoir states
        states = np.zeros((time_steps, self.reservoir_size))
        
        # Run reservoir
        for t in range(time_steps):
            x = self._update_reservoir(x, u_bias[t])
            states[t] = x
            
        # Discard washout period
        return states[washout:] if washout > 0 else states
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
           washout: int = 0, ridge_param: float = 1e-6) -> None:
        """
        Train the readout layer using ridge regression.
        
        Args:
            X: Input sequences of shape (n_samples, time_steps, input_size)
            y: Target values of shape (n_samples, time_steps, output_size) or (n_samples, output_size)
            washout: Number of initial time steps to discard (transient)
            ridge_param: Regularization parameter for ridge regression
        """
        n_samples = X.shape[0]
        time_steps = X.shape[1]
        
        # Handle different y shapes
        if len(y.shape) == 2:  # (n_samples, output_size)
            # Repeat targets for each time step (used for sequence classification)
            y_expanded = np.repeat(y[:, np.newaxis, :], time_steps - washout, axis=1)
        else:  # (n_samples, time_steps, output_size)
            # Use targets for each time step (used for sequence prediction)
            y_expanded = y[:, washout:] if washout > 0 else y
        
        # Collect states from all sequences
        all_states = []
        all_targets = []
        
        for i in range(n_samples):
            # Compute reservoir states
            states = self._compute_reservoir_states(X[i], washout)
            
            # Collect states and targets
            all_states.append(states)
            all_targets.append(y_expanded[i])
            
        # Stack all collected data
        all_states = np.vstack(all_states)
        all_targets = np.vstack(all_targets)
        
        # Add bias term to states
        extended_states = np.hstack((all_states, np.ones((all_states.shape[0], 1))))
        
        # Train with ridge regression
        if ridge_param > 0:
            # Ridge regression: W = (X^T X + alpha I)^-1 X^T y
            self.w_out = np.dot(
                np.dot(
                    np.linalg.inv(
                        np.dot(extended_states.T, extended_states) + 
                        ridge_param * np.eye(extended_states.shape[1])
                    ),
                    extended_states.T
                ),
                all_targets
            )
        else:
            # Ordinary least squares
            self.w_out = np.dot(
                np.linalg.pinv(extended_states),
                all_targets
            )
    
    def predict(self, X: np.ndarray, washout: int = 0) -> np.ndarray:
        """
        Generate predictions for input sequences.
        
        Args:
            X: Input sequences of shape (n_samples, time_steps, input_size)
            washout: Number of initial time steps to discard
            
        Returns:
            Predictions of shape (n_samples, time_steps-washout, output_size)
        """
        if self.w_out is None:
            raise ValueError("Model has not been trained. Call fit() first.")
            
        n_samples = X.shape[0]
        time_steps = X.shape[1]
        effective_steps = time_steps - washout if washout > 0 else time_steps
        
        predictions = np.zeros((n_samples, effective_steps, self.output_size))
        
        for i in range(n_samples):
            # Compute reservoir states
            states = self._compute_reservoir_states(X[i], washout)
            
            # Add bias term
            extended_states = np.hstack((states, np.ones((states.shape[0], 1))))
            
            # Apply readout
            predictions[i] = np.dot(extended_states, self.w_out)
            
        return predictions
    
    def predict_next_step(self, X: np.ndarray, steps_ahead: int = 1) -> np.ndarray:
        """
        Predict steps ahead in the future (for forecasting).
        
        Args:
            X: Input sequences of shape (n_samples, time_steps, input_size)
            steps_ahead: Number of steps to predict into the future
            
        Returns:
            Predictions of shape (n_samples, steps_ahead, output_size)
        """
        if self.w_out is None:
            raise ValueError("Model has not been trained. Call fit() first.")
            
        n_samples = X.shape[0]
        forecasts = np.zeros((n_samples, steps_ahead, self.output_size))
        
        for i in range(n_samples):
            # Initialize with known sequence
            current_input = X[i].copy()
            
            # Initialize reservoir state
            x = np.zeros(self.reservoir_size)
            u_bias = np.hstack((current_input, np.ones((current_input.shape[0], 1))))
            
            # Run through known sequence to warm up reservoir
            for t in range(current_input.shape[0]):
                x = self._update_reservoir(x, u_bias[t])
            
            # Predict future steps
            next_input = current_input[-1].copy()
            
            for step in range(steps_ahead):
                # Add bias to current input
                next_input_bias = np.append(next_input, 1)
                
                # Update reservoir
                x = self._update_reservoir(x, next_input_bias)
                
                # Predict output
                extended_state = np.append(x, 1)  # Add bias
                prediction = np.dot(extended_state, self.w_out)
                
                # Store prediction
                forecasts[i, step] = prediction
                
                # Update input for next prediction (assuming prediction becomes next input)
                if self.output_size == self.input_size:
                    next_input = prediction
                
        return forecasts


class NeuroevolutionModel:
    """
    Neuroevolution Model implementation.
    
    Neuroevolution combines neural networks with evolutionary algorithms
    to evolve network architectures and weights, allowing adaptive optimization
    for market conditions.
    """
    
    def __init__(self, input_size: int, output_size: int,
               pop_size: int = 50, hidden_layers: List[int] = [64, 32],
               mutation_rate: float = 0.1, mutation_scale: float = 0.2):
        """
        Initialize a Neuroevolution Model.
        
        Args:
            input_size: Number of input features
            output_size: Number of output dimensions
            pop_size: Size of the population
            hidden_layers: List of hidden layer sizes
            mutation_rate: Probability of mutation
            mutation_scale: Scale of mutations
        """
        self.input_size = input_size
        self.output_size = output_size
        self.pop_size = pop_size
        self.hidden_layers = hidden_layers
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        
        # Create initial population
        self.population = self._initialize_population()
        self.fitnesses = np.zeros(pop_size)
        
        # Keep track of best network
        self.best_network = None
        self.best_fitness = -np.inf
        
    def _create_network(self) -> tf.keras.Model:
        """
        Create a neural network with specified architecture.
        
        Returns:
            Neural network model
        """
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Input(shape=(self.input_size,)))
        
        # Hidden layers
        for units in self.hidden_layers:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
            
        # Output layer
        if self.output_size > 1:
            model.add(tf.keras.layers.Dense(self.output_size, activation='softmax'))
        else:
            model.add(tf.keras.layers.Dense(1))
            
        return model
    
    def _initialize_population(self) -> List[tf.keras.Model]:
        """
        Initialize a population of neural networks.
        
        Returns:
            List of neural network models
        """
        population = []
        
        for _ in range(self.pop_size):
            # Create network
            network = self._create_network()
            population.append(network)
            
        return population
    
    def _compute_fitness(self, network: tf.keras.Model, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute fitness of a network.
        
        Args:
            network: Neural network to evaluate
            X: Input data
            y: Target data
            
        Returns:
            Fitness score (higher is better)
        """
        # Make predictions
        predictions = network.predict(X, verbose=0)
        
        # Compute mean squared error
        mse = np.mean((predictions - y)**2)
        
        # Fitness is negative MSE (higher is better)
        return -mse
    
    def _select_parents(self, num_parents: int) -> List[int]:
        """
        Select parents using tournament selection.
        
        Args:
            num_parents: Number of parents to select
            
        Returns:
            Indices of selected parents
        """
        parent_indices = []
        
        for _ in range(num_parents):
            # Select random competitors
            tournament_size = 3
            competitors = np.random.choice(self.pop_size, size=tournament_size, replace=False)
            
            # Find the best competitor
            best_idx = competitors[0]
            best_fitness = self.fitnesses[best_idx]
            
            for idx in competitors[1:]:
                if self.fitnesses[idx] > best_fitness:
                    best_idx = idx
                    best_fitness = self.fitnesses[idx]
                    
            parent_indices.append(best_idx)
            
        return parent_indices
    
    def _crossover(self, parent1: tf.keras.Model, parent2: tf.keras.Model) -> tf.keras.Model:
        """
        Perform crossover between two parent networks.
        
        Args:
            parent1: First parent network
            parent2: Second parent network
            
        Returns:
            Child network
        """
        # Create a new network
        child = self._create_network()
        
        # Get weights from parents
        weights1 = parent1.get_weights()
        weights2 = parent2.get_weights()
        
        # Perform crossover
        child_weights = []
        for w1, w2 in zip(weights1, weights2):
            # Random binary mask for each parameter
            mask = np.random.random(w1.shape) < 0.5
            
            # Mix weights
            w_child = np.where(mask, w1, w2)
            child_weights.append(w_child)
            
        # Set weights to child
        child.set_weights(child_weights)
        
        return child
    
    def _mutate(self, network: tf.keras.Model) -> tf.keras.Model:
        """
        Apply mutation to a network.
        
        Args:
            network: Network to mutate
            
        Returns:
            Mutated network
        """
        # Get weights
        weights = network.get_weights()
        
        # Apply mutation
        for i in range(len(weights)):
            # Mutation mask (only mutate some weights)
            mutation_mask = np.random.random(weights[i].shape) < self.mutation_rate
            
            # Random noise
            noise = np.random.normal(0, self.mutation_scale, weights[i].shape)
            
            # Apply noise to selected weights
            weights[i] = weights[i] + mutation_mask * noise
            
        # Set mutated weights
        network.set_weights(weights)
        
        return network
    
    def evolve(self, X: np.ndarray, y: np.ndarray, generations: int = 100) -> List[float]:
        """
        Evolve the population over multiple generations.
        
        Args:
            X: Input data
            y: Target data
            generations: Number of generations to evolve
            
        Returns:
            List of best fitness per generation
        """
        best_fitnesses = []
        
        for generation in range(generations):
            # Evaluate fitness of each network
            for i in range(self.pop_size):
                self.fitnesses[i] = self._compute_fitness(self.population[i], X, y)
                
            # Update best network
            best_idx = np.argmax(self.fitnesses)
            current_best_fitness = self.fitnesses[best_idx]
            
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_network = self.population[best_idx]
                
            # Log best fitness
            best_fitnesses.append(self.best_fitness)
            print(f"Generation {generation+1}/{generations}, Best Fitness: {self.best_fitness:.4f}")
            
            # Create next generation
            next_population = []
            
            # Elitism: Keep the best network
            next_population.append(self.population[best_idx])
            
            # Create offspring
            while len(next_population) < self.pop_size:
                # Select parents
                parent_indices = self._select_parents(2)
                
                # Crossover
                child = self._crossover(
                    self.population[parent_indices[0]],
                    self.population[parent_indices[1]]
                )
                
                # Mutation
                child = self._mutate(child)
                
                # Add to next generation
                next_population.append(child)
                
            # Replace current population
            self.population = next_population
            
        return best_fitnesses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the best network.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        if self.best_network is None:
            raise ValueError("No network has been evolved yet. Call evolve() first.")
            
        return self.best_network.predict(X, verbose=0) 