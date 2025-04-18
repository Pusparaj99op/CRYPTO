"""
Generative Adversarial Networks for cryptocurrency applications.
This module contains GAN implementations specifically designed for
generating synthetic crypto market data, price sequences, and more.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import matplotlib.pyplot as plt


class CryptoGAN:
    """Base class for all cryptocurrency-focused GANs."""
    
    def __init__(self, input_dim=100, learning_rate=0.0002, beta_1=0.5):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.generator = None
        self.discriminator = None
        self.gan = None
        
    def build_generator(self):
        """Build the generator model."""
        raise NotImplementedError("Subclasses must implement this method")
        
    def build_discriminator(self):
        """Build the discriminator model."""
        raise NotImplementedError("Subclasses must implement this method")
        
    def build_gan(self):
        """Build the GAN by connecting generator and discriminator."""
        self.discriminator.trainable = False
        gan_input = layers.Input(shape=(self.input_dim,))
        generator_output = self.generator(gan_input)
        gan_output = self.discriminator(generator_output)
        self.gan = Model(gan_input, gan_output)
        self.gan.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1)
        )
        
    def train(self, X_train, epochs=10000, batch_size=128, save_interval=1000):
        """Train the GAN model."""
        # Training logic to be implemented in subclasses
        raise NotImplementedError("Subclasses must implement this method")
    
    def generate_samples(self, num_samples):
        """Generate samples using the trained generator."""
        noise = np.random.normal(0, 1, (num_samples, self.input_dim))
        return self.generator.predict(noise)
    
    def save_models(self, path_prefix):
        """Save the generator and discriminator models."""
        self.generator.save(f"{path_prefix}_generator.h5")
        self.discriminator.save(f"{path_prefix}_discriminator.h5")
        
    def load_models(self, generator_path, discriminator_path):
        """Load saved generator and discriminator models."""
        self.generator = tf.keras.models.load_model(generator_path)
        self.discriminator = tf.keras.models.load_model(discriminator_path)
        self.build_gan()


class PriceGAN(CryptoGAN):
    """GAN for generating synthetic cryptocurrency price sequences."""
    
    def __init__(self, sequence_length=100, features=5, input_dim=100, learning_rate=0.0002):
        """
        Initialize PriceGAN.
        
        Args:
            sequence_length: Length of the price sequence to generate
            features: Number of features in each step (OHLCV)
            input_dim: Dimension of the input noise vector
            learning_rate: Learning rate for the optimizer
        """
        super().__init__(input_dim, learning_rate)
        self.sequence_length = sequence_length
        self.features = features
        self.build_generator()
        self.build_discriminator()
        self.build_gan()
        
    def build_generator(self):
        """Build a generator for creating synthetic price sequences."""
        noise_input = layers.Input(shape=(self.input_dim,))
        
        # Start with dense layers
        x = layers.Dense(256, activation='relu')(noise_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(self.sequence_length * 8, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Reshape for convolutional layers
        x = layers.Reshape((self.sequence_length, 8))(x)
        
        # Use LSTM layers for time series generation
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.LSTM(32, return_sequences=True)(x)
        
        # Output layer - generate sequence with OHLCV features
        output = layers.Dense(self.features, activation='tanh')(x)
        
        self.generator = Model(noise_input, output, name="price_generator")
        return self.generator
        
    def build_discriminator(self):
        """Build a discriminator to classify real vs fake price sequences."""
        sequence_input = layers.Input(shape=(self.sequence_length, self.features))
        
        # Conv1D layers to process the sequence
        x = layers.Conv1D(32, kernel_size=3, strides=2, padding='same')(sequence_input)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv1D(64, kernel_size=3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv1D(128, kernel_size=3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.25)(x)
        
        # LSTM layer to capture temporal patterns
        x = layers.LSTM(64)(x)
        
        # Dense layers for classification
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        self.discriminator = Model(sequence_input, output, name="price_discriminator")
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1),
            metrics=['accuracy']
        )
        return self.discriminator
    
    def train(self, X_train, epochs=10000, batch_size=128, save_interval=1000):
        """Train the GAN on price sequence data."""
        # Create arrays for real and fake labels with noise
        real = np.ones((batch_size, 1)) * 0.9  # Label smoothing
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # Train discriminator
            # Select random batch of real price sequences
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_sequences = X_train[idx]
            
            # Generate batch of fake price sequences
            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            fake_sequences = self.generator.predict(noise)
            
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_sequences, real)
            d_loss_fake = self.discriminator.train_on_batch(fake_sequences, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            g_loss = self.gan.train_on_batch(noise, real)  # Try to fool discriminator
            
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, D Loss: {d_loss[0]}, D Accuracy: {d_loss[1]*100}%, G Loss: {g_loss}")
            
            # Save models at intervals
            if epoch % save_interval == 0:
                self.save_models(f"price_gan_epoch_{epoch}")
                # Generate and visualize sample
                self._visualize_sample(epoch)
    
    def _visualize_sample(self, epoch):
        """Generate and visualize a sample price sequence."""
        sample = self.generate_samples(1)[0]
        # Assuming the first feature is the closing price
        plt.figure(figsize=(10, 6))
        plt.plot(sample[:, 0])
        plt.title(f"Generated Price Sequence - Epoch {epoch}")
        plt.xlabel("Time Steps")
        plt.ylabel("Normalized Price")
        plt.savefig(f"price_gan_sample_epoch_{epoch}.png")
        plt.close()


class MarketGAN(CryptoGAN):
    """GAN for generating full market data including price, volume, and order books."""
    
    def __init__(self, input_dim=200, learning_rate=0.0001, market_dims=(24, 10)):
        """
        Initialize MarketGAN.
        
        Args:
            input_dim: Dimension of the input noise vector
            learning_rate: Learning rate for the optimizer
            market_dims: Dimensions of the market data (time_steps, features)
        """
        super().__init__(input_dim, learning_rate)
        self.market_dims = market_dims
        self.build_generator()
        self.build_discriminator()
        self.build_gan()
        
    def build_generator(self):
        """Build a generator for creating synthetic market data."""
        noise_input = layers.Input(shape=(self.input_dim,))
        
        # Dense layers to process the noise
        x = layers.Dense(512, activation='relu')(noise_input)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Reshape for convolutional layers
        x = layers.Dense(self.market_dims[0] * self.market_dims[1] * 8)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Reshape((self.market_dims[0], self.market_dims[1], 8))(x)
        
        # Conv2D layers to generate market data
        x = layers.Conv2DTranspose(128, kernel_size=3, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Output layer
        output = layers.Conv2D(1, kernel_size=3, padding='same', activation='tanh')(x)
        output = layers.Reshape((self.market_dims[0], self.market_dims[1]))(output)
        
        self.generator = Model(noise_input, output, name="market_generator")
        return self.generator
    
    def build_discriminator(self):
        """Build a discriminator to classify real vs fake market data."""
        market_input = layers.Input(shape=(self.market_dims[0], self.market_dims[1]))
        x = layers.Reshape((self.market_dims[0], self.market_dims[1], 1))(market_input)
        
        x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(512)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        self.discriminator = Model(market_input, output, name="market_discriminator")
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1),
            metrics=['accuracy']
        )
        return self.discriminator
    
    def train(self, X_train, epochs=5000, batch_size=64, save_interval=500):
        """Train the GAN on market data."""
        # Similar to PriceGAN training but adapted for market data
        real = np.ones((batch_size, 1)) * 0.9  # Label smoothing
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_market_data = X_train[idx]
            
            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            fake_market_data = self.generator.predict(noise)
            
            d_loss_real = self.discriminator.train_on_batch(real_market_data, real)
            d_loss_fake = self.discriminator.train_on_batch(fake_market_data, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            g_loss = self.gan.train_on_batch(noise, real)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, D Loss: {d_loss[0]}, D Accuracy: {d_loss[1]*100}%, G Loss: {g_loss}")
            
            if epoch % save_interval == 0:
                self.save_models(f"market_gan_epoch_{epoch}")
                self._visualize_market_sample(epoch)
    
    def _visualize_market_sample(self, epoch):
        """Generate and visualize a sample of market data."""
        sample = self.generate_samples(1)[0]
        plt.figure(figsize=(12, 8))
        plt.imshow(sample, cmap='viridis', aspect='auto')
        plt.colorbar(label='Market Value')
        plt.title(f"Generated Market Data - Epoch {epoch}")
        plt.xlabel("Features")
        plt.ylabel("Time Steps")
        plt.savefig(f"market_gan_sample_epoch_{epoch}.png")
        plt.close() 