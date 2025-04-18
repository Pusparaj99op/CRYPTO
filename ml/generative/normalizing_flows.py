"""
Normalizing flows for modeling cryptocurrency data distributions.
This module implements various normalizing flow models specifically
optimized for financial and cryptocurrency data distribution modeling.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, backend as K
import matplotlib.pyplot as plt


class CouplingLayer(layers.Layer):
    """Coupling layer implementation for normalizing flows."""
    
    def __init__(self, input_dim, hidden_dim=64, mask_type='alternate', **kwargs):
        """Initialize coupling layer.
        
        Args:
            input_dim: Dimension of the input data
            hidden_dim: Dimension of the hidden layers
            mask_type: Type of masking strategy ('alternate' or 'half')
        """
        super(CouplingLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mask_type = mask_type
        
        # Create binary mask
        if mask_type == 'alternate':
            # Alternate masking (1, 0, 1, 0, ...)
            self.mask = np.arange(input_dim) % 2
        elif mask_type == 'half':
            # Half masking (1, 1, ..., 0, 0)
            self.mask = np.concatenate([np.ones(input_dim // 2), np.zeros(input_dim - input_dim // 2)])
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
        
        self.mask = tf.constant(self.mask, dtype=tf.float32)
        self.mask_inv = 1 - self.mask
        
        # Scale and translation networks
        self.s_network = self._build_network()
        self.t_network = self._build_network()
    
    def _build_network(self):
        """Build a small neural network for scale/translation."""
        inputs = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(self.hidden_dim, activation='relu')(inputs)
        x = layers.Dense(self.hidden_dim, activation='relu')(x)
        outputs = layers.Dense(self.input_dim)(x)
        return Model(inputs, outputs)
    
    def call(self, inputs, inverse=False):
        """Forward or inverse pass through the coupling layer."""
        masked_inputs = self.mask * inputs
        
        if not inverse:
            # Forward transformation
            # x_1 remains unchanged
            # x_2 = t(x_1) + s(x_1) * x_2
            s = self.s_network(masked_inputs) * self.mask_inv
            t = self.t_network(masked_inputs) * self.mask_inv
            
            # Apply scale and translation (s is parametrized as tanh for stability)
            outputs = masked_inputs + self.mask_inv * (inputs * tf.exp(tf.tanh(s)) + t)
            
            # Log determinant of the Jacobian
            ldj = tf.reduce_sum(tf.tanh(s), axis=1)
            return outputs, ldj
        else:
            # Inverse transformation
            # x_1 remains unchanged
            # x_2 = (x_2 - t(x_1)) / s(x_1)
            s = self.s_network(masked_inputs) * self.mask_inv
            t = self.t_network(masked_inputs) * self.mask_inv
            
            # Apply inverse scale and translation
            outputs = masked_inputs + self.mask_inv * ((inputs - t) * tf.exp(-tf.tanh(s)))
            
            # Log determinant of the Jacobian (negative of forward)
            ldj = -tf.reduce_sum(tf.tanh(s), axis=1)
            return outputs, ldj


class RealNVP(Model):
    """Real-valued Non-Volume Preserving (RealNVP) flow model."""
    
    def __init__(self, input_dim, num_coupling=4, hidden_dim=64, learning_rate=0.001, name="realnvp", **kwargs):
        """Initialize RealNVP.
        
        Args:
            input_dim: Dimension of the input data
            num_coupling: Number of coupling layers to use
            hidden_dim: Dimension of the hidden layers
            learning_rate: Learning rate for the optimizer
        """
        super(RealNVP, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.num_coupling = num_coupling
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        # Build flow layers
        self.coupling_layers = []
        for i in range(num_coupling):
            # Alternate between 'alternate' and 'half' masks for better mixing
            mask_type = 'alternate' if i % 2 == 0 else 'half'
            layer = CouplingLayer(input_dim, hidden_dim, mask_type=mask_type)
            self.coupling_layers.append(layer)
        
        # Set up optimizer
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)
        
        # Define loss metric
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    
    def call(self, inputs):
        """Forward pass through the flow (data -> latent)."""
        x = inputs
        sum_ldj = 0
        
        # Apply each coupling layer
        for layer in self.coupling_layers:
            x, ldj = layer(x)
            sum_ldj += ldj
        
        return x, sum_ldj
    
    def inverse(self, latent):
        """Inverse pass through the flow (latent -> data)."""
        z = latent
        sum_ldj = 0
        
        # Apply each coupling layer in reverse
        for layer in reversed(self.coupling_layers):
            z, ldj = layer(z, inverse=True)
            sum_ldj += ldj
        
        return z, sum_ldj
    
    @tf.function
    def train_step(self, data):
        """Train the model on a batch of data."""
        with tf.GradientTape() as tape:
            # Transform data to latent space
            z, ldj = self(data)
            
            # Compute log-likelihood of the latent variables under prior
            log_likelihood_prior = -0.5 * tf.reduce_sum(z**2 + np.log(2 * np.pi), axis=1)
            
            # Total log-likelihood
            log_likelihood = log_likelihood_prior + ldj
            
            # Negative log-likelihood as loss (to maximize likelihood)
            loss = -tf.reduce_mean(log_likelihood)
        
        # Compute gradients and update model
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update loss metric
        self.loss_tracker.update_state(loss)
        
        return {"loss": self.loss_tracker.result()}
    
    @property
    def metrics(self):
        """Return list of metrics to track."""
        return [self.loss_tracker]
    
    def fit(self, x, epochs=100, batch_size=64, verbose=1, **kwargs):
        """Train the model on data."""
        return super(RealNVP, self).fit(x, x, epochs=epochs, batch_size=batch_size, verbose=verbose, **kwargs)
    
    def sample(self, num_samples=1000):
        """Generate samples from the model."""
        # Sample from prior (standard normal)
        z = tf.random.normal(shape=(num_samples, self.input_dim))
        
        # Transform from latent space to data space
        samples, _ = self.inverse(z)
        return samples.numpy()
    
    def log_prob(self, x):
        """Compute log probability of samples."""
        z, ldj = self(x)
        log_likelihood_prior = -0.5 * tf.reduce_sum(z**2 + np.log(2 * np.pi), axis=1)
        log_likelihood = log_likelihood_prior + ldj
        return log_likelihood.numpy()


class MAF(Model):
    """Masked Autoregressive Flow model for cryptocurrency data distribution modeling."""
    
    def __init__(self, input_dim, num_layers=5, hidden_dim=64, learning_rate=0.001, name="maf", **kwargs):
        """Initialize MAF model.
        
        Args:
            input_dim: Dimension of the input data
            num_layers: Number of autoregressive layers
            hidden_dim: Dimension of hidden layers
            learning_rate: Learning rate for the optimizer
        """
        super(MAF, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        # Build MAF layers
        self.maf_layers = []
        for i in range(num_layers):
            # Create MADE (Masked Autoencoder for Distribution Estimation) layers
            # with alternating degree orderings for better mixing
            reverse_ordering = (i % 2 == 0)
            layer = self._build_made_layer(reverse_ordering)
            self.maf_layers.append(layer)
        
        # Set up optimizer
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)
        
        # Define loss metric
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    
    def _build_made_layer(self, reverse_ordering=False):
        """Build a MADE layer for autoregressive transformations."""
        # A simplified MADE implementation using masked dense layers
        permutation = tf.range(self.input_dim)
        if reverse_ordering:
            permutation = tf.reverse(permutation, axis=[0])
        
        # Masks will ensure autoregressive property: output i depends only on inputs 1:i
        def build_autoregressive_masks(input_size, hidden_size, output_size, ordering):
            # This is a simplified masking strategy
            # In a full implementation, we would use more sophisticated masking
            hidden_mask = np.zeros((input_size, hidden_size))
            output_mask = np.zeros((hidden_size, 2 * output_size))  # For mean and scale
            
            for i in range(input_size):
                for j in range(hidden_size):
                    if j % input_size >= ordering[i]:
                        hidden_mask[i, j] = 1
            
            for i in range(hidden_size):
                for j in range(output_size):
                    if i % input_size >= ordering[j]:
                        output_mask[i, j] = 1
                        output_mask[i, j + output_size] = 1  # Same mask for mean and scale
            
            return hidden_mask, output_mask
        
        class MaskedDense(layers.Layer):
            def __init__(self, units, mask, activation=None, **kwargs):
                super(MaskedDense, self).__init__(**kwargs)
                self.units = units
                self.mask = tf.constant(mask, dtype=tf.float32)
                self.activation = tf.keras.activations.get(activation)
                self.kernel = None
                self.bias = None
            
            def build(self, input_shape):
                self.kernel = self.add_weight("kernel",
                                             shape=(input_shape[-1], self.units),
                                             initializer="glorot_uniform",
                                             trainable=True)
                self.bias = self.add_weight("bias",
                                           shape=(self.units,),
                                           initializer="zeros",
                                           trainable=True)
            
            def call(self, inputs):
                # Apply mask to kernel during the forward pass
                masked_kernel = self.kernel * self.mask
                outputs = tf.matmul(inputs, masked_kernel) + self.bias
                if self.activation is not None:
                    outputs = self.activation(outputs)
                return outputs
        
        # Build the MADE layer
        hidden_mask, output_mask = build_autoregressive_masks(
            self.input_dim, self.hidden_dim, self.input_dim, permutation.numpy())
        
        inputs = layers.Input(shape=(self.input_dim,))
        x = MaskedDense(self.hidden_dim, hidden_mask, activation='relu')(inputs)
        params = MaskedDense(2 * self.input_dim, output_mask)(x)
        
        # Split into mean and log_scale
        mean, log_scale = tf.split(params, 2, axis=-1)
        
        # Create the model
        model = Model(inputs, [mean, log_scale, permutation])
        return model
    
    def call(self, inputs, inverse=False):
        """Forward pass through the flow (data -> latent)."""
        if not inverse:
            # Forward transformation
            x = inputs
            sum_ldj = 0
            
            for layer in self.maf_layers:
                # Get the autoregressive parameters
                mean, log_scale, permutation = layer(x)
                
                # Apply transformation: z = (x - μ) * exp(-s)
                scale = tf.exp(tf.tanh(log_scale))  # Tanh for stability
                z = (x - mean) / scale
                
                # Compute log determinant of Jacobian
                ldj = -tf.reduce_sum(tf.tanh(log_scale), axis=1)
                sum_ldj += ldj
                
                # Permute for next layer
                x = tf.gather(z, permutation, axis=1)
            
            return x, sum_ldj
        else:
            # Inverse transformation (for sampling)
            # This requires an iterative procedure due to autoregressive nature
            z = inputs
            sum_ldj = 0
            
            for layer in reversed(self.maf_layers):
                # Inverse permutation
                _, _, permutation = layer(tf.zeros_like(z))
                inverse_permutation = tf.argsort(permutation)
                z_permuted = tf.gather(z, inverse_permutation, axis=1)
                
                # Initialize output tensor
                x = tf.zeros_like(z_permuted)
                
                # Iteratively compute x
                for i in range(self.input_dim):
                    # Get conditional parameters for this dimension
                    mean, log_scale, _ = layer(x)
                    mean_i = mean[:, i:i+1]
                    scale_i = tf.exp(tf.tanh(log_scale[:, i:i+1]))
                    
                    # Compute x_i
                    x_i = z_permuted[:, i:i+1] * scale_i + mean_i
                    
                    # Update x
                    x = tf.concat([x[:, :i], x_i, x[:, i+1:]], axis=1)
                
                # Compute log determinant of Jacobian
                _, log_scale, _ = layer(x)
                ldj = tf.reduce_sum(tf.tanh(log_scale), axis=1)
                sum_ldj += ldj
                
                z = x
            
            return z, sum_ldj
    
    @tf.function
    def train_step(self, data):
        """Train the model on a batch of data."""
        with tf.GradientTape() as tape:
            # Transform data to latent space
            z, ldj = self(data[0])  # data[0] because of the Keras convention
            
            # Compute log-likelihood of the latent variables under prior
            log_likelihood_prior = -0.5 * tf.reduce_sum(z**2 + np.log(2 * np.pi), axis=1)
            
            # Total log-likelihood
            log_likelihood = log_likelihood_prior + ldj
            
            # Negative log-likelihood as loss (to maximize likelihood)
            loss = -tf.reduce_mean(log_likelihood)
        
        # Compute gradients and update model
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update loss metric
        self.loss_tracker.update_state(loss)
        
        return {"loss": self.loss_tracker.result()}
    
    @property
    def metrics(self):
        """Return list of metrics to track."""
        return [self.loss_tracker]
    
    def sample(self, num_samples=1000):
        """Generate samples from the model."""
        # Sample from prior (standard normal)
        z = tf.random.normal(shape=(num_samples, self.input_dim))
        
        # Transform from latent space to data space
        samples, _ = self(z, inverse=True)
        return samples.numpy()
    
    def log_prob(self, x):
        """Compute log probability of samples."""
        z, ldj = self(x)
        log_likelihood_prior = -0.5 * tf.reduce_sum(z**2 + np.log(2 * np.pi), axis=1)
        log_likelihood = log_likelihood_prior + ldj
        return log_likelihood.numpy()


class Glow(Model):
    """Glow model for crypto price modeling with more expressive transformations."""
    
    def __init__(self, input_dim, num_steps=4, hidden_dim=64, learning_rate=0.001, name="glow", **kwargs):
        """Initialize Glow model.
        
        Args:
            input_dim: Dimension of the input data
            num_steps: Number of flow steps
            hidden_dim: Dimension of hidden layers
            learning_rate: Learning rate for the optimizer
        """
        super(Glow, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        # Build Glow layers (simplified version)
        self.flow_steps = []
        for _ in range(num_steps):
            # Each step consists of:
            # 1. ActNorm (normalization)
            # 2. Invertible 1x1 convolution (for 1D data, this is a permutation)
            # 3. Coupling layer
            step = {
                'actnorm': self._build_actnorm(),
                'permutation': self._build_permutation(),
                'coupling': CouplingLayer(input_dim, hidden_dim, mask_type='alternate')
            }
            self.flow_steps.append(step)
        
        # Set up optimizer
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)
        
        # Define loss metric
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    
    def _build_actnorm(self):
        """Build an ActNorm layer for normalization."""
        # ActNorm performs per-channel normalization with learnable parameters
        # We simplify it for 1D data
        class ActNorm(layers.Layer):
            def __init__(self, **kwargs):
                super(ActNorm, self).__init__(**kwargs)
            
            def build(self, input_shape):
                # Initialize logs and bias parameters
                self.logs = self.add_weight(
                    name='logs', shape=(1, input_shape[-1]),
                    initializer='zeros', trainable=True
                )
                self.bias = self.add_weight(
                    name='bias', shape=(1, input_shape[-1]),
                    initializer='zeros', trainable=True
                )
                self.initialized = tf.Variable(False, trainable=False)
                self.input_shape_value = input_shape
            
            def call(self, inputs, inverse=False):
                # Data dependent initialization
                if not self.initialized:
                    # Compute mean and std of first batch
                    mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
                    std = tf.math.reduce_std(inputs, axis=0, keepdims=True)
                    
                    # Initialize parameters to normalize first batch
                    self.bias.assign(-mean)
                    self.logs.assign(tf.math.log(1.0 / (std + 1e-6)))
                    self.initialized.assign(True)
                
                if not inverse:
                    # Forward: y = (x + bias) * exp(logs)
                    outputs = (inputs + self.bias) * tf.exp(self.logs)
                    ldj = tf.reduce_sum(self.logs) * tf.cast(tf.shape(inputs)[0], tf.float32)
                else:
                    # Inverse: x = y * exp(-logs) - bias
                    outputs = inputs * tf.exp(-self.logs) - self.bias
                    ldj = -tf.reduce_sum(self.logs) * tf.cast(tf.shape(inputs)[0], tf.float32)
                
                return outputs, ldj
        
        return ActNorm()
    
    def _build_permutation(self):
        """Build a permutation layer (invertible 1x1 conv for 1D data)."""
        # For 1D data, this is just a permutation matrix
        class Permutation(layers.Layer):
            def __init__(self, dim, **kwargs):
                super(Permutation, self).__init__(**kwargs)
                self.dim = dim
                
                # Initialize random permutation matrix
                permutation = np.random.permutation(dim)
                self.permutation = tf.Variable(permutation, trainable=False, dtype=tf.int32)
                self.inverse_permutation = tf.Variable(np.argsort(permutation), trainable=False, dtype=tf.int32)
            
            def call(self, inputs, inverse=False):
                if not inverse:
                    outputs = tf.gather(inputs, self.permutation, axis=1)
                    return outputs, 0.0  # Permutation has log_det_jacobian = 0
                else:
                    outputs = tf.gather(inputs, self.inverse_permutation, axis=1)
                    return outputs, 0.0
        
        return Permutation(self.input_dim)
    
    def call(self, inputs, inverse=False):
        """Forward or inverse pass through the flow."""
        if not inverse:
            # Forward transformation
            x = inputs
            sum_ldj = 0
            
            for step in self.flow_steps:
                # Apply actnorm
                x, ldj1 = step['actnorm'](x, inverse=False)
                sum_ldj += ldj1
                
                # Apply permutation
                x, ldj2 = step['permutation'](x, inverse=False)
                sum_ldj += ldj2
                
                # Apply coupling
                x, ldj3 = step['coupling'](x, inverse=False)
                sum_ldj += ldj3
            
            return x, sum_ldj
        else:
            # Inverse transformation
            z = inputs
            sum_ldj = 0
            
            for step in reversed(self.flow_steps):
                # Apply coupling (inverse)
                z, ldj3 = step['coupling'](z, inverse=True)
                sum_ldj += ldj3
                
                # Apply permutation (inverse)
                z, ldj2 = step['permutation'](z, inverse=True)
                sum_ldj += ldj2
                
                # Apply actnorm (inverse)
                z, ldj1 = step['actnorm'](z, inverse=True)
                sum_ldj += ldj1
            
            return z, sum_ldj
    
    @tf.function
    def train_step(self, data):
        """Train the model on a batch of data."""
        with tf.GradientTape() as tape:
            # Transform data to latent space
            z, ldj = self(data[0])  # data[0] because of the Keras convention
            
            # Compute log-likelihood of the latent variables under prior
            log_likelihood_prior = -0.5 * tf.reduce_sum(z**2 + np.log(2 * np.pi), axis=1)
            
            # Total log-likelihood
            log_likelihood = log_likelihood_prior + ldj
            
            # Negative log-likelihood as loss (to maximize likelihood)
            loss = -tf.reduce_mean(log_likelihood)
        
        # Compute gradients and update model
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update loss metric
        self.loss_tracker.update_state(loss)
        
        return {"loss": self.loss_tracker.result()}
    
    @property
    def metrics(self):
        """Return list of metrics to track."""
        return [self.loss_tracker]
    
    def fit(self, x, epochs=100, batch_size=64, verbose=1, **kwargs):
        """Train the model on data."""
        return super(Glow, self).fit(x, x, epochs=epochs, batch_size=batch_size, verbose=verbose, **kwargs)
    
    def sample(self, num_samples=1000):
        """Generate samples from the model."""
        # Sample from prior (standard normal)
        z = tf.random.normal(shape=(num_samples, self.input_dim))
        
        # Transform from latent space to data space
        samples, _ = self(z, inverse=True)
        return samples.numpy()
    
    def log_prob(self, x):
        """Compute log probability of samples."""
        z, ldj = self(x)
        log_likelihood_prior = -0.5 * tf.reduce_sum(z**2 + np.log(2 * np.pi), axis=1)
        log_likelihood = log_likelihood_prior + ldj
        return log_likelihood.numpy()
    
    def visualize_samples(self, n_samples=500, save_path=None):
        """Generate and visualize samples from the model."""
        samples = self.sample(n_samples)
        
        # Visualize the first 2 dimensions
        plt.figure(figsize=(10, 8))
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6)
        plt.title('Generated Samples from Flow Model')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
        return samples 