"""
Variational Autoencoders for cryptocurrency applications.
This module contains VAE implementations designed for encoding,
generating and modeling cryptocurrency data distributions.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses, backend as K
import matplotlib.pyplot as plt


class CryptoVAE:
    """Base class for all cryptocurrency-focused VAEs."""
    
    def __init__(self, latent_dim=32, learning_rate=0.001):
        """Initialize base VAE.
        
        Args:
            latent_dim: The dimensionality of the latent space
            learning_rate: Learning rate for the optimizer
        """
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.encoder = None
        self.decoder = None
        self.vae = None
    
    def build_encoder(self):
        """Build the encoder model."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def build_decoder(self):
        """Build the decoder model."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def build_vae(self):
        """Build the complete VAE model."""
        raise NotImplementedError("Subclasses must implement this method")
    
    @staticmethod
    def sampling(args):
        """Sample from the latent space using reparameterization trick."""
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    @staticmethod
    def vae_loss(x, x_decoded_mean, z_mean, z_log_var, original_dim):
        """VAE loss function: reconstruction loss + KL divergence."""
        # Reconstruction loss
        reconstruction_loss = original_dim * losses.mse(x, x_decoded_mean)
        # KL divergence
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        # Total loss
        return K.mean(reconstruction_loss + kl_loss)
    
    def generate_samples(self, n_samples=1):
        """Generate samples from the latent space."""
        z_sample = np.random.normal(0, 1, size=(n_samples, self.latent_dim))
        return self.decoder.predict(z_sample)
    
    def save_models(self, path_prefix):
        """Save the encoder and decoder models."""
        self.encoder.save(f"{path_prefix}_encoder.h5")
        self.decoder.save(f"{path_prefix}_decoder.h5")
    
    def load_models(self, encoder_path, decoder_path):
        """Load saved encoder and decoder models."""
        self.encoder = tf.keras.models.load_model(encoder_path)
        self.decoder = tf.keras.models.load_model(decoder_path)
        # Rebuild the full VAE
        self.build_vae()


class PriceVAE(CryptoVAE):
    """VAE for modeling cryptocurrency price distributions."""
    
    def __init__(self, input_dim=100, latent_dim=20, learning_rate=0.001):
        """Initialize Price VAE.
        
        Args:
            input_dim: The dimensionality of the input data (sequence length)
            latent_dim: The dimensionality of the latent space
            learning_rate: Learning rate for the optimizer
        """
        super().__init__(latent_dim, learning_rate)
        self.input_dim = input_dim
        self.build_encoder()
        self.build_decoder()
        self.build_vae()
    
    def build_encoder(self):
        """Build the encoder model for price data."""
        # Input layer
        inputs = layers.Input(shape=(self.input_dim,), name='encoder_input')
        
        # Hidden layers
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        
        # Mean and variance for the latent distribution
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        # Use reparameterization trick to sample from the latent space
        z = layers.Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        
        # Create encoder model
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        return self.encoder
    
    def build_decoder(self):
        """Build the decoder model for price data."""
        # Input layer for latent space
        latent_inputs = layers.Input(shape=(self.latent_dim,), name='decoder_input')
        
        # Hidden layers
        x = layers.Dense(64, activation='relu')(latent_inputs)
        x = layers.Dense(128, activation='relu')(x)
        
        # Output layer
        outputs = layers.Dense(self.input_dim, activation='tanh', name='decoder_output')(x)
        
        # Create decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        return self.decoder
    
    def build_vae(self):
        """Build the complete VAE model."""
        # Get the encoder outputs
        inputs = self.encoder.input
        z_mean, z_log_var, z = self.encoder(inputs)
        
        # Connect the decoder to the encoder output
        outputs = self.decoder(z)
        
        # Create the VAE model
        self.vae = Model(inputs, outputs, name='price_vae')
        
        # Define the loss function
        reconstruction_loss = self.input_dim * losses.mse(inputs, outputs)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        
        # Add the loss to the model
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        
        return self.vae
    
    def train(self, X_train, epochs=100, batch_size=32, validation_data=None):
        """Train the VAE model."""
        # Create a callback to generate samples during training
        class SampleCallback(tf.keras.callbacks.Callback):
            def __init__(self, vae_model, interval=10):
                super().__init__()
                self.vae_model = vae_model
                self.interval = interval
            
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % self.interval == 0:
                    # Generate a sample
                    sample = self.vae_model.generate_samples(1)[0]
                    plt.figure(figsize=(10, 4))
                    plt.plot(sample)
                    plt.title(f"Generated Price Sample - Epoch {epoch+1}")
                    plt.xlabel("Time Steps")
                    plt.ylabel("Normalized Price")
                    plt.savefig(f"price_vae_sample_epoch_{epoch+1}.png")
                    plt.close()
        
        # Create model checkpoint callback
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            "price_vae_best.h5", monitor='val_loss' if validation_data else 'loss', 
            save_best_only=True, verbose=1
        )
        
        # Train the model
        history = self.vae.fit(
            X_train, None,  # No targets needed for VAE
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(validation_data, None) if validation_data is not None else None,
            callbacks=[SampleCallback(self, 10), checkpoint]
        )
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.plot(history.history['loss'])
        if validation_data is not None:
            plt.plot(history.history['val_loss'])
            plt.legend(['training', 'validation'])
        plt.title('VAE Loss')
        plt.xlabel('Epoch')
        plt.savefig("price_vae_training_history.png")
        plt.close()
        
        return history
    
    def encode_data(self, data):
        """Encode data into the latent space."""
        z_mean, _, _ = self.encoder.predict(data)
        return z_mean
    
    def reconstruct_data(self, data):
        """Reconstruct data from its original form."""
        return self.vae.predict(data)
    
    def interpolate(self, data1, data2, n_steps=10):
        """Interpolate between two data points in latent space."""
        # Encode both data points
        z1 = self.encode_data(np.array([data1]))[0]
        z2 = self.encode_data(np.array([data2]))[0]
        
        # Create interpolation steps in latent space
        ratios = np.linspace(0, 1, n_steps)
        interpolated = []
        
        for ratio in ratios:
            # Linear interpolation in latent space
            z_interp = z1 * (1 - ratio) + z2 * ratio
            # Decode the interpolated point
            sample = self.decoder.predict(np.array([z_interp]))[0]
            interpolated.append(sample)
        
        return np.array(interpolated)


class SequenceVAE(CryptoVAE):
    """VAE for modeling cryptocurrency sequence data with temporal structure."""
    
    def __init__(self, sequence_length=100, features=5, latent_dim=32, learning_rate=0.001):
        """Initialize Sequence VAE.
        
        Args:
            sequence_length: Length of the input sequences
            features: Number of features in each sequence step (OHLCV)
            latent_dim: The dimensionality of the latent space
            learning_rate: Learning rate for the optimizer
        """
        super().__init__(latent_dim, learning_rate)
        self.sequence_length = sequence_length
        self.features = features
        self.build_encoder()
        self.build_decoder()
        self.build_vae()
    
    def build_encoder(self):
        """Build the encoder model for sequence data using LSTM layers."""
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.features), name='encoder_input')
        
        # LSTM layers to process the sequence
        x = layers.LSTM(128, return_sequences=True)(inputs)
        x = layers.LSTM(64)(x)
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        
        # Mean and variance for the latent distribution
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        # Use reparameterization trick to sample from the latent space
        z = layers.Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        
        # Create encoder model
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='sequence_encoder')
        return self.encoder
    
    def build_decoder(self):
        """Build the decoder model for sequence data."""
        # Input layer for latent space
        latent_inputs = layers.Input(shape=(self.latent_dim,), name='decoder_input')
        
        # Dense layers to expand the latent representation
        x = layers.Dense(64, activation='relu')(latent_inputs)
        
        # Reshape to prepare for LSTM layers
        x = layers.Dense(self.sequence_length * 32, activation='relu')(x)
        x = layers.Reshape((self.sequence_length, 32))(x)
        
        # LSTM layers to generate the sequence
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.LSTM(128, return_sequences=True)(x)
        
        # Output layer
        outputs = layers.TimeDistributed(
            layers.Dense(self.features, activation='tanh'),
            name='decoder_output'
        )(x)
        
        # Create decoder model
        self.decoder = Model(latent_inputs, outputs, name='sequence_decoder')
        return self.decoder
    
    def build_vae(self):
        """Build the complete VAE model."""
        # Get the encoder outputs
        inputs = self.encoder.input
        z_mean, z_log_var, z = self.encoder(inputs)
        
        # Connect the decoder to the encoder output
        outputs = self.decoder(z)
        
        # Create the VAE model
        self.vae = Model(inputs, outputs, name='sequence_vae')
        
        # Define the loss function
        original_dim = self.sequence_length * self.features
        reconstruction_loss = original_dim * losses.mse(
            K.flatten(inputs), K.flatten(outputs)
        )
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        
        # Add the loss to the model
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        
        return self.vae
    
    def train(self, X_train, epochs=100, batch_size=32, validation_data=None):
        """Train the sequence VAE model."""
        # Create a callback to generate samples during training
        class SampleCallback(tf.keras.callbacks.Callback):
            def __init__(self, vae_model, interval=10):
                super().__init__()
                self.vae_model = vae_model
                self.interval = interval
            
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % self.interval == 0:
                    # Generate a sample
                    sample = self.vae_model.generate_samples(1)[0]
                    # Plot the first feature (closing price)
                    plt.figure(figsize=(12, 6))
                    for i in range(min(5, sample.shape[2])):
                        plt.subplot(5, 1, i+1)
                        plt.plot(sample[:, i])
                        plt.ylabel(f"Feature {i}")
                    plt.suptitle(f"Generated Sequence - Epoch {epoch+1}")
                    plt.tight_layout()
                    plt.savefig(f"sequence_vae_sample_epoch_{epoch+1}.png")
                    plt.close()
        
        # Create model checkpoint callback
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            "sequence_vae_best.h5", monitor='val_loss' if validation_data else 'loss', 
            save_best_only=True, verbose=1
        )
        
        # Train the model
        history = self.vae.fit(
            X_train, None,  # No targets needed for VAE
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(validation_data, None) if validation_data is not None else None,
            callbacks=[SampleCallback(self, 10), checkpoint]
        )
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.plot(history.history['loss'])
        if validation_data is not None:
            plt.plot(history.history['val_loss'])
            plt.legend(['training', 'validation'])
        plt.title('Sequence VAE Loss')
        plt.xlabel('Epoch')
        plt.savefig("sequence_vae_training_history.png")
        plt.close()
        
        return history
    
    def encode_sequence(self, sequence):
        """Encode a sequence into the latent space."""
        z_mean, _, _ = self.encoder.predict(sequence)
        return z_mean
    
    def reconstruct_sequence(self, sequence):
        """Reconstruct a sequence from its original form."""
        return self.vae.predict(sequence)
    
    def generate_sequence_variations(self, sequence, n_variations=5, std_dev=0.5):
        """Generate variations of a sequence by perturbing its latent representation."""
        # Encode the sequence
        z_mean = self.encode_sequence(np.array([sequence]))[0]
        
        # Generate variations by adding noise to the latent representation
        variations = []
        for _ in range(n_variations):
            # Add random noise to the latent vector
            noise = np.random.normal(0, std_dev, size=z_mean.shape)
            z_new = z_mean + noise
            
            # Decode the perturbed latent vector
            variation = self.decoder.predict(np.array([z_new]))[0]
            variations.append(variation)
        
        return np.array(variations) 