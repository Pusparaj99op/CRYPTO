"""
Diffusion models for cryptocurrency price evolution modeling.
This module implements diffusion-based generative models designed
for generating realistic cryptocurrency price paths and market dynamics.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import matplotlib.pyplot as plt


class DiffusionModel:
    """Base class for diffusion models."""
    
    def __init__(self, time_steps=1000, beta_schedule='linear', beta_start=1e-4, beta_end=0.02):
        """Initialize diffusion model parameters.
        
        Args:
            time_steps: Number of diffusion time steps
            beta_schedule: Schedule for noise variance ('linear' or 'cosine')
            beta_start: Starting value for noise schedule
            beta_end: Ending value for noise schedule
        """
        self.time_steps = time_steps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Initialize noise schedule
        self._init_noise_schedule()
        
        # Create or build additional model components
        self.model = None
    
    def _init_noise_schedule(self):
        """Initialize the noise schedule for the diffusion process."""
        if self.beta_schedule == 'linear':
            # Linear schedule from beta_start to beta_end
            self.betas = np.linspace(self.beta_start, self.beta_end, self.time_steps)
        elif self.beta_schedule == 'cosine':
            # Cosine schedule (often works better for images)
            steps = self.time_steps + 1
            x = np.linspace(0, self.time_steps, steps)
            alphas_cumprod = np.cos(((x / self.time_steps) + self.beta_start) / (1 + self.beta_end) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = np.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
        
        # Pre-compute values used in diffusion process
        self.alphas = 1. - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.append(1., self.alphas_cumprod[:-1])
        
        # Calculations for diffusion q(x_t | x_{t-1}) and posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = self.betas * np.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion process: q(x_t | x_0).
        
        Args:
            x_0: Initial data (clean samples)
            t: Timestep
            noise: Noise to add (if None, random noise will be used)
            
        Returns:
            The noised sample x_t
        """
        if noise is None:
            noise = tf.random.normal(shape=x_0.shape)
        
        # Extract the corresponding alpha values for timestep t
        t_idx = tf.cast(t, tf.int32)
        sqrt_alphas_cumprod_t = tf.constant(self.sqrt_alphas_cumprod, dtype=tf.float32)[t_idx]
        sqrt_one_minus_alphas_cumprod_t = tf.constant(self.sqrt_one_minus_alphas_cumprod, dtype=tf.float32)[t_idx]
        
        # Reshape for proper broadcasting
        sqrt_alphas_cumprod_t = tf.reshape(sqrt_alphas_cumprod_t, (-1, 1, 1) if len(x_0.shape) == 3 else (-1, 1))
        sqrt_one_minus_alphas_cumprod_t = tf.reshape(
            sqrt_one_minus_alphas_cumprod_t, (-1, 1, 1) if len(x_0.shape) == 3 else (-1, 1))
        
        # Apply the diffusion process
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model_output, x_t, t):
        """Single reverse diffusion step: p(x_{t-1} | x_t).
        
        Args:
            model_output: Output from the noise prediction model
            x_t: Current noisy sample at timestep t
            t: Current timestep
            
        Returns:
            The predicted sample at timestep t-1
        """
        t_idx = tf.cast(t, tf.int32)
        
        # Extract the necessary pre-computed values
        betas_t = tf.constant(self.betas, dtype=tf.float32)[t_idx]
        sqrt_one_minus_alphas_cumprod_t = tf.constant(self.sqrt_one_minus_alphas_cumprod, dtype=tf.float32)[t_idx]
        sqrt_recip_alphas_t = tf.constant(self.sqrt_recip_alphas_cumprod, dtype=tf.float32)[t_idx]
        
        # Reshape for proper broadcasting
        betas_t = tf.reshape(betas_t, (-1, 1, 1) if len(x_t.shape) == 3 else (-1, 1))
        sqrt_one_minus_alphas_cumprod_t = tf.reshape(
            sqrt_one_minus_alphas_cumprod_t, (-1, 1, 1) if len(x_t.shape) == 3 else (-1, 1))
        sqrt_recip_alphas_t = tf.reshape(sqrt_recip_alphas_t, (-1, 1, 1) if len(x_t.shape) == 3 else (-1, 1))
        
        # Compute the predicted mean
        pred_original = (x_t - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t) * sqrt_recip_alphas_t
        
        # Add noise based on the posterior variance
        posterior_variance_t = tf.constant(self.posterior_variance, dtype=tf.float32)[t_idx]
        posterior_variance_t = tf.reshape(posterior_variance_t, (-1, 1, 1) if len(x_t.shape) == 3 else (-1, 1))
        
        noise = tf.random.normal(shape=x_t.shape)
        if t_idx > 0:  # No noise for the final step
            return pred_original + tf.sqrt(posterior_variance_t) * noise
        else:
            return pred_original
    
    def p_sample_loop(self, shape, batch_size=1, verbose=True):
        """Sample from the model by iterating through the reverse diffusion process.
        
        Args:
            shape: Shape of the samples to generate
            batch_size: Number of samples to generate in parallel
            verbose: Whether to print progress
            
        Returns:
            Generated samples
        """
        # Initialize with random noise
        sample_shape = (batch_size,) + shape
        samples = tf.random.normal(sample_shape)
        
        # Iteratively denoise
        for t in reversed(range(self.time_steps)):
            if verbose and t % 100 == 0:
                print(f"Sampling timestep {t}/{self.time_steps}")
            
            # Time embedding for the model
            timesteps = tf.ones(batch_size, dtype=tf.int32) * t
            
            # Predict the noise
            predicted_noise = self.model([samples, timesteps], training=False)
            
            # Update sample using a single reverse diffusion step
            samples = self.p_sample(predicted_noise, samples, tf.constant([t], dtype=tf.int32))
        
        return samples
    
    def build_model(self):
        """Build the noise prediction model. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def train(self, X_train, epochs=100, batch_size=32, verbose=1):
        """Train the diffusion model.
        
        Args:
            X_train: Training data
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def generate_samples(self, num_samples=1, shape=None):
        """Generate samples using the trained model.
        
        Args:
            num_samples: Number of samples to generate
            shape: Shape of each sample (if None, will use training data shape)
            
        Returns:
            Generated samples
        """
        if shape is None:
            raise ValueError("Sample shape must be provided")
        
        return self.p_sample_loop(shape, batch_size=num_samples)


class PriceDiffusion(DiffusionModel):
    """Diffusion model for generating cryptocurrency price sequences."""
    
    def __init__(self, sequence_length=100, features=1, time_steps=1000, 
                 beta_schedule='linear', beta_start=1e-4, beta_end=0.02,
                 hidden_dims=[128, 256, 512], learning_rate=1e-4):
        """Initialize price diffusion model.
        
        Args:
            sequence_length: Length of the price sequence
            features: Number of features in each step (default 1 for price only)
            time_steps: Number of diffusion time steps
            beta_schedule: Schedule for noise variance
            beta_start: Starting value for noise schedule
            beta_end: Ending value for noise schedule
            hidden_dims: Hidden dimensions of the model
            learning_rate: Learning rate for the optimizer
        """
        super().__init__(time_steps, beta_schedule, beta_start, beta_end)
        self.sequence_length = sequence_length
        self.features = features
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        
        # Build the model
        self.build_model()
    
    def build_model(self):
        """Build a CNN-based model for noise prediction in price sequences."""
        # Input for noisy price sequence
        sequence_input = layers.Input(shape=(self.sequence_length, self.features))
        
        # Time embedding
        time_input = layers.Input(shape=(), dtype=tf.int32)
        time_embed_dim = self.hidden_dims[0]
        
        # Sinusoidal time embedding (similar to transformer positional encoding)
        freqs = 10000 ** -tf.range(0, time_embed_dim // 2, 1, dtype=tf.float32) / (time_embed_dim // 2)
        args = tf.cast(time_input, tf.float32)[:, None] * freqs[None]
        time_embed = tf.concat([tf.math.cos(args), tf.math.sin(args)], axis=-1)
        time_embed = layers.Dense(time_embed_dim, activation='swish')(time_embed)
        time_embed = layers.Dense(time_embed_dim, activation='swish')(time_embed)
        
        # Reshape time embedding for broadcasting
        time_embed = tf.reshape(time_embed, [-1, 1, time_embed_dim])
        time_embed = tf.tile(time_embed, [1, self.sequence_length, 1])
        
        # Combine sequence input with time embedding
        x = layers.Dense(self.hidden_dims[0])(sequence_input)
        x = layers.Concatenate()([x, time_embed])
        
        # 1D CNN layers with residual connections
        skip_connections = []
        
        for i, dim in enumerate(self.hidden_dims):
            # Conv block
            res = x
            x = layers.Conv1D(dim, kernel_size=3, padding='same')(x)
            x = layers.LayerNormalization()(x)
            x = layers.Activation('swish')(x)
            
            x = layers.Conv1D(dim, kernel_size=3, padding='same')(x)
            x = layers.LayerNormalization()(x)
            x = layers.Activation('swish')(x)
            
            # Residual connection
            if i > 0 and self.hidden_dims[i-1] != dim:
                res = layers.Conv1D(dim, kernel_size=1, padding='same')(res)
            
            x = layers.Add()([x, res])
            
            # Skip connection
            skip_connections.append(x)
            
            # Downsample if not the last layer
            if i < len(self.hidden_dims) - 1:
                x = layers.Conv1D(dim, kernel_size=3, strides=2, padding='same')(x)
                x = layers.Activation('swish')(x)
        
        # Upsample and use skip connections (U-Net style)
        for i in range(len(self.hidden_dims) - 1, 0, -1):
            dim = self.hidden_dims[i-1]
            
            # Upsample
            x = layers.Conv1DTranspose(dim, kernel_size=3, strides=2, padding='same')(x)
            x = layers.Activation('swish')(x)
            
            # Skip connection
            x = layers.Concatenate()([x, skip_connections[i-1]])
            
            # Conv block
            res = x
            x = layers.Conv1D(dim, kernel_size=3, padding='same')(x)
            x = layers.LayerNormalization()(x)
            x = layers.Activation('swish')(x)
            
            x = layers.Conv1D(dim, kernel_size=3, padding='same')(x)
            x = layers.LayerNormalization()(x)
            x = layers.Activation('swish')(x)
            
            # Residual connection
            if self.hidden_dims[i] != dim:
                res = layers.Conv1D(dim, kernel_size=1, padding='same')(res)
            
            x = layers.Add()([x, res])
        
        # Output layer
        output = layers.Conv1D(self.features, kernel_size=1)(x)
        
        # Create the model
        self.model = Model([sequence_input, time_input], output)
        
        # Compile the model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return self.model
    
    def train(self, X_train, epochs=100, batch_size=32, validation_data=None, verbose=1):
        """Train the diffusion model on price sequence data.
        
        Args:
            X_train: Training data (price sequences)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_data: Optional validation data
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        # Define a callback to track and visualize progress
        class DiffusionCallback(tf.keras.callbacks.Callback):
            def __init__(self, diffusion_model, interval=10):
                super().__init__()
                self.diffusion_model = diffusion_model
                self.interval = interval
            
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % self.interval == 0:
                    # Generate a sample
                    sample_shape = (self.diffusion_model.sequence_length, self.diffusion_model.features)
                    samples = self.diffusion_model.generate_samples(1, sample_shape)[0]
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(samples[:, 0])  # Plot the first feature (price)
                    plt.title(f"Generated Price Sequence - Epoch {epoch+1}")
                    plt.xlabel("Time Steps")
                    plt.ylabel("Price")
                    plt.savefig(f"price_diffusion_sample_epoch_{epoch+1}.png")
                    plt.close()
        
        # Create a custom training loop to handle the noise target
        @tf.function
        def train_step(x_batch):
            with tf.GradientTape() as tape:
                # Sample timesteps uniformly
                t = tf.random.uniform(
                    shape=(batch_size,), minval=0, maxval=self.time_steps, dtype=tf.int32
                )
                
                # Sample noise
                noise = tf.random.normal(shape=x_batch.shape)
                
                # Apply forward diffusion to get noisy samples
                x_noisy = self.q_sample(x_batch, t, noise)
                
                # Predict the noise
                predicted_noise = self.model([x_noisy, t], training=True)
                
                # Loss is MSE between the original noise and predicted noise
                loss = tf.reduce_mean(tf.square(noise - predicted_noise))
            
            # Compute and apply gradients
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            return loss
        
        # Training loop
        train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(10000).batch(batch_size)
        
        if validation_data is not None:
            val_dataset = tf.data.Dataset.from_tensor_slices(validation_data).batch(batch_size)
        
        history = {'loss': [], 'val_loss': []}
        callback = DiffusionCallback(self, interval=epochs // 10 if epochs > 10 else 1)
        
        for epoch in range(epochs):
            # Training
            epoch_losses = []
            for x_batch in train_dataset:
                loss = train_step(x_batch)
                epoch_losses.append(loss)
            
            avg_loss = tf.reduce_mean(epoch_losses)
            history['loss'].append(avg_loss.numpy())
            
            # Validation
            if validation_data is not None:
                val_losses = []
                for x_val in val_dataset:
                    # Sample timesteps uniformly
                    t = tf.random.uniform(
                        shape=(len(x_val),), minval=0, maxval=self.time_steps, dtype=tf.int32
                    )
                    
                    # Sample noise
                    noise = tf.random.normal(shape=x_val.shape)
                    
                    # Apply forward diffusion to get noisy samples
                    x_noisy = self.q_sample(x_val, t, noise)
                    
                    # Predict the noise
                    predicted_noise = self.model([x_noisy, t], training=False)
                    
                    # Loss is MSE between the original noise and predicted noise
                    val_loss = tf.reduce_mean(tf.square(noise - predicted_noise))
                    val_losses.append(val_loss)
                
                avg_val_loss = tf.reduce_mean(val_losses)
                history['val_loss'].append(avg_val_loss.numpy())
                
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            else:
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
            
            # Call the callback
            callback.on_epoch_end(epoch)
        
        # Plot training curve
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], label='Training Loss')
        if validation_data is not None:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Diffusion Model Training')
        plt.legend()
        plt.savefig('price_diffusion_training.png')
        plt.close()
        
        return history
    
    def generate_price_paths(self, num_paths=10, starting_price=None):
        """Generate multiple price paths from the diffusion model.
        
        Args:
            num_paths: Number of price paths to generate
            starting_price: Optional starting price (if None, random start)
            
        Returns:
            Generated price paths
        """
        # Generate samples
        sample_shape = (self.sequence_length, self.features)
        samples = self.generate_samples(num_paths, sample_shape)
        
        # If starting price is provided, adjust all paths to start from it
        if starting_price is not None:
            for i in range(num_paths):
                # Calculate the offset between the generated start and desired start
                offset = starting_price - samples[i, 0, 0]
                # Add the offset to the entire path
                samples[i, :, 0] += offset
        
        return samples


class MarketDiffusion(DiffusionModel):
    """Diffusion model for generating cryptocurrency market data including multiple indicators."""
    
    def __init__(self, sequence_length=100, features=5, time_steps=1000, 
                 beta_schedule='linear', beta_start=1e-4, beta_end=0.02,
                 hidden_dims=[128, 256, 512, 512], learning_rate=1e-4):
        """Initialize market diffusion model.
        
        Args:
            sequence_length: Length of the market data sequence
            features: Number of features (e.g., OHLCV)
            time_steps: Number of diffusion time steps
            beta_schedule: Schedule for noise variance
            beta_start: Starting value for noise schedule
            beta_end: Ending value for noise schedule
            hidden_dims: Hidden dimensions of the model
            learning_rate: Learning rate for the optimizer
        """
        super().__init__(time_steps, beta_schedule, beta_start, beta_end)
        self.sequence_length = sequence_length
        self.features = features
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        
        # Build the model
        self.build_model()
    
    def build_model(self):
        """Build a Transformer-based model for noise prediction in market data."""
        # Input for noisy market data
        sequence_input = layers.Input(shape=(self.sequence_length, self.features))
        
        # Time embedding
        time_input = layers.Input(shape=(), dtype=tf.int32)
        time_embed_dim = self.hidden_dims[0]
        
        # Sinusoidal time embedding
        freqs = 10000 ** -tf.range(0, time_embed_dim // 2, 1, dtype=tf.float32) / (time_embed_dim // 2)
        args = tf.cast(time_input, tf.float32)[:, None] * freqs[None]
        time_embed = tf.concat([tf.math.cos(args), tf.math.sin(args)], axis=-1)
        time_embed = layers.Dense(time_embed_dim, activation='swish')(time_embed)
        time_embed = layers.Dense(time_embed_dim, activation='swish')(time_embed)
        
        # Project input to hidden dimension
        x = layers.Dense(self.hidden_dims[0])(sequence_input)
        
        # Add positional encoding
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        position_embedding = layers.Embedding(
            input_dim=self.sequence_length, 
            output_dim=self.hidden_dims[0]
        )(positions)
        x = x + position_embedding
        
        # Add time embedding to each position
        time_embed = tf.reshape(time_embed, [-1, 1, time_embed_dim])
        time_embed = tf.tile(time_embed, [1, self.sequence_length, 1])
        x = x + time_embed
        
        # Transformer layers
        for dim in self.hidden_dims:
            # Multi-head self-attention
            attn_output = layers.MultiHeadAttention(
                num_heads=8, key_dim=dim // 8
            )(x, x, x)
            x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
            
            # Position-wise feed-forward network
            ffn_output = layers.Dense(dim * 4, activation='swish')(x)
            ffn_output = layers.Dense(dim)(ffn_output)
            x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        # Output layer
        output = layers.Dense(self.features)(x)
        
        # Create the model
        self.model = Model([sequence_input, time_input], output)
        
        # Compile the model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return self.model
    
    def train(self, X_train, epochs=100, batch_size=32, validation_data=None, verbose=1):
        """Train the diffusion model on market data.
        
        Args:
            X_train: Training data (market sequences)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_data: Optional validation data
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        # Define a callback to track and visualize progress
        class MarketDiffusionCallback(tf.keras.callbacks.Callback):
            def __init__(self, diffusion_model, interval=10):
                super().__init__()
                self.diffusion_model = diffusion_model
                self.interval = interval
            
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % self.interval == 0:
                    # Generate a sample
                    sample_shape = (self.diffusion_model.sequence_length, self.diffusion_model.features)
                    samples = self.diffusion_model.generate_samples(1, sample_shape)[0]
                    
                    # Plot the generated market data
                    plt.figure(figsize=(15, 10))
                    
                    # Define feature names (assuming OHLCV data)
                    feature_names = ['Open', 'High', 'Low', 'Close', 'Volume'] 
                    if self.diffusion_model.features < len(feature_names):
                        feature_names = feature_names[:self.diffusion_model.features]
                    elif self.diffusion_model.features > len(feature_names):
                        # Add generic feature names if needed
                        feature_names += [f'Feature {i+len(feature_names)}' 
                                          for i in range(self.diffusion_model.features - len(feature_names))]
                    
                    # Plot each feature
                    for i in range(min(self.diffusion_model.features, 5)):  # Plot at most 5 features
                        plt.subplot(min(self.diffusion_model.features, 5), 1, i+1)
                        plt.plot(samples[:, i])
                        plt.title(f"{feature_names[i]}")
                        plt.grid(True)
                    
                    plt.suptitle(f"Generated Market Data - Epoch {epoch+1}")
                    plt.tight_layout()
                    plt.savefig(f"market_diffusion_sample_epoch_{epoch+1}.png")
                    plt.close()
        
        # Use the same training approach as PriceDiffusion
        @tf.function
        def train_step(x_batch):
            with tf.GradientTape() as tape:
                # Sample timesteps uniformly
                t = tf.random.uniform(
                    shape=(batch_size,), minval=0, maxval=self.time_steps, dtype=tf.int32
                )
                
                # Sample noise
                noise = tf.random.normal(shape=x_batch.shape)
                
                # Apply forward diffusion to get noisy samples
                x_noisy = self.q_sample(x_batch, t, noise)
                
                # Predict the noise
                predicted_noise = self.model([x_noisy, t], training=True)
                
                # Loss is MSE between the original noise and predicted noise
                loss = tf.reduce_mean(tf.square(noise - predicted_noise))
            
            # Compute and apply gradients
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            return loss
        
        # Training loop
        train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(10000).batch(batch_size)
        
        if validation_data is not None:
            val_dataset = tf.data.Dataset.from_tensor_slices(validation_data).batch(batch_size)
        
        history = {'loss': [], 'val_loss': []}
        callback = MarketDiffusionCallback(self, interval=epochs // 10 if epochs > 10 else 1)
        
        for epoch in range(epochs):
            # Training
            epoch_losses = []
            for x_batch in train_dataset:
                loss = train_step(x_batch)
                epoch_losses.append(loss)
            
            avg_loss = tf.reduce_mean(epoch_losses)
            history['loss'].append(avg_loss.numpy())
            
            # Validation
            if validation_data is not None:
                val_losses = []
                for x_val in val_dataset:
                    # Sample timesteps uniformly
                    t = tf.random.uniform(
                        shape=(len(x_val),), minval=0, maxval=self.time_steps, dtype=tf.int32
                    )
                    
                    # Sample noise
                    noise = tf.random.normal(shape=x_val.shape)
                    
                    # Apply forward diffusion to get noisy samples
                    x_noisy = self.q_sample(x_val, t, noise)
                    
                    # Predict the noise
                    predicted_noise = self.model([x_noisy, t], training=False)
                    
                    # Loss is MSE between the original noise and predicted noise
                    val_loss = tf.reduce_mean(tf.square(noise - predicted_noise))
                    val_losses.append(val_loss)
                
                avg_val_loss = tf.reduce_mean(val_losses)
                history['val_loss'].append(avg_val_loss.numpy())
                
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            else:
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
            
            # Call the callback
            callback.on_epoch_end(epoch)
        
        # Plot training curve
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], label='Training Loss')
        if validation_data is not None:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Market Diffusion Model Training')
        plt.legend()
        plt.savefig('market_diffusion_training.png')
        plt.close()
        
        return history
    
    def generate_market_scenarios(self, num_scenarios=5, conditions=None):
        """Generate multiple market scenarios from the diffusion model.
        
        Args:
            num_scenarios: Number of scenarios to generate
            conditions: Optional conditioning (not implemented in this basic version)
            
        Returns:
            Generated market scenarios
        """
        # Generate samples
        sample_shape = (self.sequence_length, self.features)
        samples = self.generate_samples(num_scenarios, sample_shape)
        
        # Visualize the generated scenarios
        plt.figure(figsize=(15, 10))
        
        # Plot closing price for each scenario
        for i in range(num_scenarios):
            plt.subplot(num_scenarios, 1, i+1)
            plt.plot(samples[i, :, 3] if self.features > 3 else samples[i, :, 0])  # Assuming 4th feature is closing price
            plt.title(f"Scenario {i+1}")
            plt.grid(True)
        
        plt.suptitle("Generated Market Scenarios")
        plt.tight_layout()
        plt.savefig("market_diffusion_scenarios.png")
        plt.close()
        
        return samples 