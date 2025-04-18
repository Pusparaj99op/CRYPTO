"""
Energy-based generative models for cryptocurrency market modeling.
This module implements energy-based generative models optimized
for capturing complex market dynamics and distributions.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, backend as K
import matplotlib.pyplot as plt


class EnergyModel:
    """Base class for energy-based models."""
    
    def __init__(self, energy_network=None, learning_rate=0.001, mcmc_steps=10):
        """Initialize energy-based model.
        
        Args:
            energy_network: Neural network to compute energy (if None, will be created)
            learning_rate: Learning rate for the optimizer
            mcmc_steps: Number of MCMC steps for sampling
        """
        self.energy_network = energy_network
        self.learning_rate = learning_rate
        self.mcmc_steps = mcmc_steps
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)
        
        if energy_network is None:
            self.build_energy_network()
    
    def build_energy_network(self):
        """Build the energy function network. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def energy(self, x):
        """Compute the energy of samples."""
        return self.energy_network(x)
    
    def sample_langevin_dynamics(self, initial_samples, step_size=0.1):
        """Sample from the model using Langevin dynamics.
        
        Args:
            initial_samples: Initial points for the Markov chain
            step_size: Step size for the Langevin dynamics updates
            
        Returns:
            Samples from the model distribution
        """
        samples = tf.identity(initial_samples)
        
        # Run MCMC steps
        for _ in range(self.mcmc_steps):
            with tf.GradientTape() as tape:
                tape.watch(samples)
                energy = self.energy(samples)
            
            # Compute gradient of energy w.r.t samples
            grad = tape.gradient(energy, samples)
            
            # Update samples using Langevin dynamics
            noise = tf.random.normal(shape=samples.shape, stddev=np.sqrt(step_size * 2))
            samples = samples - step_size * grad + noise
        
        return samples
    
    def train(self, X_train, epochs=100, batch_size=32, verbose=1):
        """Train the energy-based model.
        
        Args:
            X_train: Training data
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def generate_samples(self, num_samples, input_dim, init_std=1.0):
        """Generate samples from the energy-based model.
        
        Args:
            num_samples: Number of samples to generate
            input_dim: Dimension of each sample
            init_std: Standard deviation of the initial noise
            
        Returns:
            Generated samples
        """
        # Initialize with random noise
        initial_samples = tf.random.normal((num_samples,) + input_dim, stddev=init_std)
        
        # Sample using Langevin dynamics
        return self.sample_langevin_dynamics(initial_samples)


class EnergyBasedMarketModel(EnergyModel):
    """Energy-based model for cryptocurrency market modeling."""
    
    def __init__(self, input_shape, hidden_dims=[128, 256, 128], learning_rate=0.001, mcmc_steps=20):
        """Initialize energy-based market model.
        
        Args:
            input_shape: Shape of the input data
            hidden_dims: Hidden dimensions of the energy network
            learning_rate: Learning rate for the optimizer
            mcmc_steps: Number of MCMC steps for sampling
        """
        self.input_shape = input_shape
        self.hidden_dims = hidden_dims
        
        super().__init__(None, learning_rate, mcmc_steps)
    
    def build_energy_network(self):
        """Build an energy network for market data."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Flatten if input is multi-dimensional
        if len(self.input_shape) > 1:
            x = layers.Flatten()(inputs)
        else:
            x = inputs
        
        # Hidden layers
        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation='swish')(x)
        
        # Energy output (scalar)
        energy = layers.Dense(1)(x)
        
        # Create model
        self.energy_network = Model(inputs, energy, name="energy_network")
        return self.energy_network
    
    def train(self, X_train, epochs=100, batch_size=32, validation_data=None, verbose=1):
        """Train the energy-based market model using contrastive divergence.
        
        Args:
            X_train: Training data
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_data: Optional validation data
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        # Function to compute gradients for a batch
        def compute_gradients(pos_samples, neg_samples):
            with tf.GradientTape() as tape:
                # Energy of positive samples (real data)
                pos_energy = tf.reduce_mean(self.energy(pos_samples))
                
                # Energy of negative samples (generated data)
                neg_energy = tf.reduce_mean(self.energy(neg_samples))
                
                # Contrastive divergence loss: maximize difference between positive and negative energies
                loss = pos_energy - neg_energy
            
            # Compute gradients
            grads = tape.gradient(loss, self.energy_network.trainable_variables)
            return loss, grads
        
        # Convert data to TensorFlow dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(10000).batch(batch_size)
        
        if validation_data is not None:
            val_dataset = tf.data.Dataset.from_tensor_slices(validation_data).batch(batch_size)
        
        # Training history
        history = {'loss': [], 'val_loss': []}
        
        # Training loop
        for epoch in range(epochs):
            # Training
            epoch_losses = []
            for pos_batch in train_dataset:
                # Generate negative samples via MCMC
                initial_noise = tf.random.normal(shape=pos_batch.shape)
                neg_samples = self.sample_langevin_dynamics(initial_noise)
                
                # Compute loss and gradients
                loss, grads = compute_gradients(pos_batch, neg_samples)
                
                # Update model parameters
                self.optimizer.apply_gradients(zip(grads, self.energy_network.trainable_variables))
                
                epoch_losses.append(loss.numpy())
            
            avg_loss = np.mean(epoch_losses)
            history['loss'].append(avg_loss)
            
            # Validation
            if validation_data is not None:
                val_losses = []
                for val_batch in val_dataset:
                    initial_noise = tf.random.normal(shape=val_batch.shape)
                    neg_val_samples = self.sample_langevin_dynamics(initial_noise)
                    
                    val_pos_energy = tf.reduce_mean(self.energy(val_batch))
                    val_neg_energy = tf.reduce_mean(self.energy(neg_val_samples))
                    val_loss = val_pos_energy - val_neg_energy
                    
                    val_losses.append(val_loss.numpy())
                
                avg_val_loss = np.mean(val_losses)
                history['val_loss'].append(avg_val_loss)
                
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            else:
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
            
            # Visualize samples periodically
            if (epoch + 1) % (epochs // 10 or 1) == 0:
                self._visualize_samples(epoch + 1)
        
        # Plot training curve
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], label='Training Loss')
        if validation_data is not None:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Energy-Based Model Training')
        plt.legend()
        plt.savefig('energy_model_training.png')
        plt.close()
        
        return history
    
    def _visualize_samples(self, epoch):
        """Generate and visualize samples from the model."""
        n_samples = 5
        samples = self.generate_samples(n_samples, self.input_shape)
        
        # Plot the samples
        plt.figure(figsize=(15, 3 * n_samples))
        
        for i in range(n_samples):
            plt.subplot(n_samples, 1, i + 1)
            
            # For 1D data (e.g., time series)
            if len(self.input_shape) == 1:
                plt.plot(samples[i])
                plt.title(f"Sample {i+1}")
            
            # For 2D data (e.g., images or heatmaps)
            elif len(self.input_shape) == 2:
                plt.imshow(samples[i], cmap='viridis')
                plt.colorbar()
                plt.title(f"Sample {i+1}")
            
            # For 3D data (e.g., multichannel time series)
            elif len(self.input_shape) == 3 and self.input_shape[2] <= 5:
                for j in range(self.input_shape[2]):
                    plt.plot(samples[i, :, j], label=f'Channel {j+1}')
                plt.legend()
                plt.title(f"Sample {i+1}")
        
        plt.suptitle(f"Generated Samples - Epoch {epoch}")
        plt.tight_layout()
        plt.savefig(f"energy_model_samples_epoch_{epoch}.png")
        plt.close()
    
    def generate_market_data(self, n_samples=10, visualize=True):
        """Generate synthetic market data samples.
        
        Args:
            n_samples: Number of samples to generate
            visualize: Whether to visualize the generated samples
            
        Returns:
            Generated market data samples
        """
        samples = self.generate_samples(n_samples, self.input_shape)
        
        if visualize:
            self._visualize_samples(-1)  # Use -1 to indicate these are final samples
        
        return samples


class DeepEnergyModel(EnergyModel):
    """Deep energy-based model with more sophisticated architecture."""
    
    def __init__(self, input_shape, hidden_dims=[128, 256, 512, 256, 128], 
                 dropout_rate=0.1, learning_rate=0.001, mcmc_steps=25):
        """Initialize deep energy-based model.
        
        Args:
            input_shape: Shape of the input data
            hidden_dims: Hidden dimensions of the energy network
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for the optimizer
            mcmc_steps: Number of MCMC steps for sampling
        """
        self.input_shape = input_shape
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        super().__init__(None, learning_rate, mcmc_steps)
    
    def build_energy_network(self):
        """Build a deep energy network with residual connections."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Process sequence data with 1D CNN if input is a sequence
        if len(self.input_shape) > 1 and self.input_shape[0] > 1:
            # Initial convolution
            x = layers.Conv1D(64, kernel_size=3, padding='same')(inputs)
            x = layers.LeakyReLU(alpha=0.2)(x)
            x = layers.BatchNormalization()(x)
            
            # Convolutional blocks with residual connections
            for filters in [64, 128, 128]:
                res = x
                x = layers.Conv1D(filters, kernel_size=3, padding='same')(x)
                x = layers.LeakyReLU(alpha=0.2)(x)
                x = layers.BatchNormalization()(x)
                x = layers.Conv1D(filters, kernel_size=3, padding='same')(x)
                x = layers.LeakyReLU(alpha=0.2)(x)
                x = layers.BatchNormalization()(x)
                
                # Residual connection
                if res.shape[-1] != filters:
                    res = layers.Conv1D(filters, kernel_size=1, padding='same')(res)
                
                x = layers.Add()([x, res])
                x = layers.AveragePooling1D(pool_size=2)(x)
            
            # Flatten to feed into dense layers
            x = layers.Flatten()(x)
        else:
            # For non-sequential data, just flatten
            x = layers.Flatten()(inputs)
        
        # Dense layers with residual connections
        for i, dim in enumerate(self.hidden_dims):
            res = x
            x = layers.Dense(dim)(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
            x = layers.Dropout(self.dropout_rate)(x)
            
            # Add residual connection every 2 layers
            if i > 0 and i % 2 == 1:
                # Match dimensions if needed
                if res.shape[-1] != dim:
                    res = layers.Dense(dim)(res)
                
                x = layers.Add()([x, res])
        
        # Final energy output
        energy = layers.Dense(1)(x)
        
        # Create model
        self.energy_network = Model(inputs, energy, name="deep_energy_network")
        return self.energy_network
    
    def train(self, X_train, epochs=100, batch_size=32, validation_data=None, verbose=1):
        """Train the deep energy-based model using persistent contrastive divergence.
        
        Args:
            X_train: Training data
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_data: Optional validation data
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        # Initialize persistent Markov chains
        n_persistent = min(batch_size * 2, 1000)
        persistent_chains = tf.random.normal(shape=(n_persistent,) + self.input_shape)
        
        # Function to compute gradients for a batch
        def compute_gradients(pos_samples, neg_samples):
            with tf.GradientTape() as tape:
                # Energy of positive samples (real data)
                pos_energy = self.energy(pos_samples)
                
                # Energy of negative samples (generated data)
                neg_energy = self.energy(neg_samples)
                
                # Contrastive divergence loss
                # We want to minimize energy of real data and maximize energy of generated data
                loss = tf.reduce_mean(pos_energy) - tf.reduce_mean(neg_energy)
                
                # Add L2 regularization
                reg_loss = tf.reduce_sum([tf.nn.l2_loss(w) for w in self.energy_network.trainable_weights])
                loss += 1e-5 * reg_loss
            
            # Compute gradients
            grads = tape.gradient(loss, self.energy_network.trainable_variables)
            return loss, grads
        
        # Convert data to TensorFlow dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(10000).batch(batch_size)
        
        if validation_data is not None:
            val_dataset = tf.data.Dataset.from_tensor_slices(validation_data).batch(batch_size)
        
        # Training history
        history = {'loss': [], 'val_loss': []}
        
        # Training loop
        for epoch in range(epochs):
            # Training
            epoch_losses = []
            for pos_batch in train_dataset:
                # Select a random subset of persistent chains
                idx = tf.random.uniform(shape=[batch_size], maxval=n_persistent, dtype=tf.int32)
                current_chains = tf.gather(persistent_chains, idx)
                
                # Run MCMC to update the chains
                updated_chains = self.sample_langevin_dynamics(current_chains)
                
                # Update the persistent chains
                for i, idx_i in enumerate(idx):
                    persistent_chains = tf.tensor_scatter_nd_update(
                        persistent_chains, [[idx_i]], [updated_chains[i:i+1]])
                
                # Compute loss and gradients
                loss, grads = compute_gradients(pos_batch, updated_chains)
                
                # Update model parameters
                self.optimizer.apply_gradients(zip(grads, self.energy_network.trainable_variables))
                
                epoch_losses.append(loss.numpy())
            
            avg_loss = np.mean(epoch_losses)
            history['loss'].append(avg_loss)
            
            # Validation
            if validation_data is not None:
                val_losses = []
                for val_batch in val_dataset:
                    # Generate negative samples via MCMC
                    initial_noise = tf.random.normal(shape=val_batch.shape)
                    neg_val_samples = self.sample_langevin_dynamics(initial_noise)
                    
                    val_pos_energy = tf.reduce_mean(self.energy(val_batch))
                    val_neg_energy = tf.reduce_mean(self.energy(neg_val_samples))
                    val_loss = val_pos_energy - val_neg_energy
                    
                    val_losses.append(val_loss.numpy())
                
                avg_val_loss = np.mean(val_losses)
                history['val_loss'].append(avg_val_loss)
                
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            else:
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
            
            # Visualize samples periodically
            if (epoch + 1) % (epochs // 5 or 1) == 0:
                self._visualize_samples(epoch + 1, persistent_chains[:5])
        
        # Plot training curve
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], label='Training Loss')
        if validation_data is not None:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Deep Energy Model Training')
        plt.legend()
        plt.savefig('deep_energy_model_training.png')
        plt.close()
        
        return history
    
    def _visualize_samples(self, epoch, samples=None):
        """Generate and visualize samples from the model."""
        n_samples = 5
        if samples is None:
            samples = self.generate_samples(n_samples, self.input_shape)
        else:
            n_samples = min(n_samples, len(samples))
            samples = samples[:n_samples]
        
        # Plot the samples
        plt.figure(figsize=(15, 10))
        
        # For time-series data with multiple channels (e.g., OHLCV)
        if len(self.input_shape) > 1 and self.input_shape[-1] > 1:
            for i in range(n_samples):
                for j in range(min(self.input_shape[-1], 5)):  # Plot up to 5 channels
                    plt.subplot(n_samples, min(self.input_shape[-1], 5), i * min(self.input_shape[-1], 5) + j + 1)
                    plt.plot(samples[i, :, j])
                    plt.title(f"Sample {i+1}, Channel {j+1}")
                    plt.grid(True)
        # For single-channel time series
        elif len(self.input_shape) == 1 or (len(self.input_shape) > 1 and self.input_shape[-1] == 1):
            for i in range(n_samples):
                plt.subplot(n_samples, 1, i + 1)
                if len(self.input_shape) > 1:
                    plt.plot(samples[i, :, 0])
                else:
                    plt.plot(samples[i])
                plt.title(f"Sample {i+1}")
                plt.grid(True)
        
        plt.suptitle(f"Generated Samples - Epoch {epoch}")
        plt.tight_layout()
        plt.savefig(f"deep_energy_model_samples_epoch_{epoch}.png")
        plt.close()
    
    def generate_samples_with_conditioning(self, num_samples=5, conditioning=None, cond_weight=1.0):
        """Generate samples with optional conditioning on certain data properties.
        
        Args:
            num_samples: Number of samples to generate
            conditioning: Function that computes a conditioning score (lower is better)
            cond_weight: Weight of the conditioning term in the energy
            
        Returns:
            Generated samples that satisfy the conditioning
        """
        # Initialize with random noise
        samples = tf.random.normal((num_samples,) + self.input_shape)
        
        # Define energy function with conditioning
        def total_energy(x):
            # Base energy from the model
            base_energy = self.energy(x)
            
            if conditioning is not None:
                # Add conditioning term
                cond_energy = conditioning(x)
                return base_energy + cond_weight * cond_energy
            else:
                return base_energy
        
        # Run Langevin dynamics with the conditioned energy
        step_size = 0.1
        for _ in range(self.mcmc_steps):
            with tf.GradientTape() as tape:
                tape.watch(samples)
                energy = total_energy(samples)
            
            # Compute gradient of energy w.r.t samples
            grad = tape.gradient(energy, samples)
            
            # Update samples using Langevin dynamics
            noise = tf.random.normal(shape=samples.shape, stddev=np.sqrt(step_size * 2))
            samples = samples - step_size * grad + noise
        
        return samples
    
    def generate_price_trajectories(self, num_samples=5, initial_price=None, target_price=None,
                                   min_volatility=None, max_volatility=None):
        """Generate price trajectories with specific properties.
        
        Args:
            num_samples: Number of trajectories to generate
            initial_price: Starting price (if None, unconstrained)
            target_price: Target ending price (if None, unconstrained)
            min_volatility: Minimum volatility (if None, unconstrained)
            max_volatility: Maximum volatility (if None, unconstrained)
            
        Returns:
            Generated price trajectories
        """
        # Define conditioning function based on the constraints
        def conditioning(x):
            energy = 0.0
            
            # Extract price sequences (assuming first channel if multi-channel)
            if len(self.input_shape) > 1 and self.input_shape[-1] > 1:
                price_seq = x[..., 0]  # First channel is price
            else:
                price_seq = x
            
            # Condition on initial price
            if initial_price is not None:
                energy += tf.reduce_mean(tf.square(price_seq[:, 0] - initial_price))
            
            # Condition on target price
            if target_price is not None:
                energy += tf.reduce_mean(tf.square(price_seq[:, -1] - target_price))
            
            # Condition on volatility
            if min_volatility is not None or max_volatility is not None:
                # Compute returns
                returns = price_seq[:, 1:] - price_seq[:, :-1]
                vols = tf.math.reduce_std(returns, axis=1)
                
                if min_volatility is not None:
                    # Penalize if volatility is below minimum
                    energy += tf.reduce_mean(tf.maximum(0.0, min_volatility - vols))
                
                if max_volatility is not None:
                    # Penalize if volatility is above maximum
                    energy += tf.reduce_mean(tf.maximum(0.0, vols - max_volatility))
            
            return energy
        
        # Generate samples with conditioning
        samples = self.generate_samples_with_conditioning(
            num_samples=num_samples, 
            conditioning=conditioning,
            cond_weight=10.0  # Higher weight for stronger conditioning
        )
        
        # Visualize the generated trajectories
        plt.figure(figsize=(12, 8))
        
        for i in range(num_samples):
            if len(self.input_shape) > 1 and self.input_shape[-1] > 1:
                plt.plot(samples[i, :, 0], label=f"Trajectory {i+1}")
            else:
                plt.plot(samples[i], label=f"Trajectory {i+1}")
        
        # Add annotations for constraints
        title = "Generated Price Trajectories"
        if initial_price is not None:
            title += f", Initial: {initial_price:.2f}"
        if target_price is not None:
            title += f", Target: {target_price:.2f}"
        if min_volatility is not None:
            title += f", Min Vol: {min_volatility:.4f}"
        if max_volatility is not None:
            title += f", Max Vol: {max_volatility:.4f}"
        
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()
        plt.savefig("deep_energy_model_price_trajectories.png")
        plt.close()
        
        return samples 