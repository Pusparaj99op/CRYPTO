"""
Transfer Learning module for cryptocurrency applications.

This module implements transfer learning techniques that allow leveraging
knowledge from pre-trained models and adapting them to new tasks in the
cryptocurrency domain, enabling better performance with limited data.

Classes:
    TransferModel: Base class for transfer learning models
    FeatureExtractor: Class for extracting features from pre-trained models
    DomainAdaptation: Implementation of domain adaptation techniques
    PretrainedFinanceModel: Specialized models pre-trained on financial data
"""

import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Union, Optional, Tuple, Callable
import os
import json

class TransferModel:
    """
    Base class for transfer learning models.
    
    This class provides functionality for loading pre-trained models
    and fine-tuning them for cryptocurrency-specific tasks.
    """
    
    def __init__(self, base_model: Union[str, tf.keras.Model], 
                 input_shape: Tuple, output_shape: int,
                 freeze_layers: Union[int, List[int]] = None,
                 learning_rate: float = 0.0001):
        """
        Initialize a transfer learning model.
        
        Args:
            base_model: Pre-trained model or path to model
            input_shape: Shape of the input data
            output_shape: Dimension of the output (e.g., number of classes)
            freeze_layers: Number of layers to freeze or list of layer indices
            learning_rate: Learning rate for fine-tuning
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        
        # Load the base model
        if isinstance(base_model, str):
            try:
                self.base_model = tf.keras.models.load_model(base_model)
            except Exception as e:
                # If loading fails, try loading from TensorFlow Hub
                try:
                    import tensorflow_hub as hub
                    self.base_model = hub.load(base_model)
                except:
                    raise ValueError(f"Could not load model from {base_model}: {e}")
        else:
            self.base_model = base_model
            
        # Create the full model with adaptation layers
        self.model = self._build_transfer_model()
        
        # Freeze specified layers if any
        if freeze_layers is not None:
            self._freeze_layers(freeze_layers)
            
        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy' if output_shape > 1 else 'mse',
            metrics=['accuracy'] if output_shape > 1 else ['mae']
        )
    
    def _build_transfer_model(self) -> tf.keras.Model:
        """
        Build a model using the pre-trained base model with new output layers.
        
        Returns:
            Keras model configured for transfer learning
        """
        # Get the output of the base model without the final classification layer
        if hasattr(self.base_model, 'layers'):
            # Assume it's a Keras model
            base_layers = self.base_model.layers[:-1]
            base_output = base_layers[-1].output
            input_tensor = self.base_model.input
        else:
            # Assume it's a model from TensorFlow Hub
            input_tensor = tf.keras.layers.Input(shape=self.input_shape)
            base_output = self.base_model(input_tensor)
        
        # Add task-specific layers
        x = tf.keras.layers.Flatten()(base_output)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        
        # Output layer
        if self.output_shape > 1:
            outputs = tf.keras.layers.Dense(self.output_shape, activation='softmax')(x)
        else:
            outputs = tf.keras.layers.Dense(1)(x)
        
        return tf.keras.Model(inputs=input_tensor, outputs=outputs)
    
    def _freeze_layers(self, freeze_layers: Union[int, List[int]]) -> None:
        """
        Freeze specified layers of the base model.
        
        Args:
            freeze_layers: Number of layers to freeze or list of layer indices
        """
        if isinstance(freeze_layers, int):
            # Freeze the first n layers
            for layer in self.model.layers[:freeze_layers]:
                layer.trainable = False
        else:
            # Freeze specific layers
            for idx in freeze_layers:
                if idx < len(self.model.layers):
                    self.model.layers[idx].trainable = False
    
    def fit(self, x: np.ndarray, y: np.ndarray, 
            batch_size: int = 32, epochs: int = 10,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            callbacks: List[tf.keras.callbacks.Callback] = None) -> tf.keras.callbacks.History:
        """
        Fine-tune the model on new data.
        
        Args:
            x: Training features
            y: Training labels
            batch_size: Batch size for training
            epochs: Number of epochs for training
            validation_data: Optional tuple of validation (features, labels)
            callbacks: Optional list of Keras callbacks
            
        Returns:
            Training history
        """
        return self.model.fit(
            x, y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks
        )
    
    def predict(self, x: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Generate predictions with the fine-tuned model.
        
        Args:
            x: Input features
            batch_size: Batch size for prediction
            
        Returns:
            Array of predictions
        """
        return self.model.predict(x, batch_size=batch_size)
    
    def evaluate(self, x: np.ndarray, y: np.ndarray, batch_size: int = 32) -> List[float]:
        """
        Evaluate the model performance.
        
        Args:
            x: Evaluation features
            y: Evaluation labels
            batch_size: Batch size for evaluation
            
        Returns:
            List of metric values
        """
        return self.model.evaluate(x, y, batch_size=batch_size)
    
    def save(self, path: str) -> None:
        """
        Save the fine-tuned model to disk.
        
        Args:
            path: Path where to save the model
        """
        self.model.save(path)
    

class FeatureExtractor:
    """
    Extract features from pre-trained models for use in downstream tasks.
    
    This class allows extracting intermediate representations from
    pre-trained models to be used as features for other models.
    """
    
    def __init__(self, model: Union[str, tf.keras.Model], layer_name: Optional[str] = None):
        """
        Initialize a feature extractor.
        
        Args:
            model: Pre-trained model or path to model
            layer_name: Name of layer to extract features from (None for last layer)
        """
        # Load the model
        if isinstance(model, str):
            try:
                self.model = tf.keras.models.load_model(model)
            except Exception as e:
                raise ValueError(f"Could not load model from {model}: {e}")
        else:
            self.model = model
            
        # Determine the layer to extract features from
        if layer_name is None:
            # Default to the layer before the output layer
            self.layer_name = self.model.layers[-2].name
        else:
            self.layer_name = layer_name
            
        # Create the feature extraction model
        self.feature_model = self._build_feature_model()
    
    def _build_feature_model(self) -> tf.keras.Model:
        """
        Build a model that outputs features from the specified layer.
        
        Returns:
            Keras model that outputs intermediate features
        """
        # Get the specified layer
        layer = None
        for l in self.model.layers:
            if l.name == self.layer_name:
                layer = l
                break
                
        if layer is None:
            raise ValueError(f"Layer '{self.layer_name}' not found in model")
            
        return tf.keras.Model(inputs=self.model.input, outputs=layer.output)
    
    def extract_features(self, x: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Extract features from the input data.
        
        Args:
            x: Input data
            batch_size: Batch size for feature extraction
            
        Returns:
            Array of extracted features
        """
        return self.feature_model.predict(x, batch_size=batch_size)
    
    def get_feature_shape(self) -> Tuple:
        """
        Get the shape of the extracted features.
        
        Returns:
            Shape of features (excluding batch dimension)
        """
        return self.feature_model.output_shape[1:]
    
    def save_features(self, x: np.ndarray, path: str, batch_size: int = 32) -> None:
        """
        Extract features and save them to disk.
        
        Args:
            x: Input data
            path: Path where to save the features
            batch_size: Batch size for feature extraction
        """
        features = self.extract_features(x, batch_size=batch_size)
        np.save(path, features)


class DomainAdaptation:
    """
    Implementation of domain adaptation techniques for transfer learning.
    
    This class provides methods for adapting models trained on one domain
    (e.g., stock market data) to work effectively on another domain
    (e.g., cryptocurrency market data).
    """
    
    def __init__(self, source_model: tf.keras.Model, adaptation_method: str = 'dann'):
        """
        Initialize a domain adaptation model.
        
        Args:
            source_model: Model trained on source domain
            adaptation_method: Method for domain adaptation ('dann', 'coral', or 'mmd')
        """
        self.source_model = source_model
        self.adaptation_method = adaptation_method
        self.model = None
    
    def adapt(self, source_data: np.ndarray, target_data: np.ndarray,
              source_labels: np.ndarray = None, target_labels: np.ndarray = None,
              epochs: int = 10, batch_size: int = 32) -> tf.keras.Model:
        """
        Adapt the source model to the target domain.
        
        Args:
            source_data: Data from source domain
            target_data: Data from target domain
            source_labels: Labels from source domain (if available)
            target_labels: Labels from target domain (if available)
            epochs: Number of epochs for adaptation
            batch_size: Batch size for training
            
        Returns:
            Adapted model
        """
        if self.adaptation_method == 'dann':
            self.model = self._domain_adversarial_adaptation(
                source_data, target_data, source_labels, target_labels,
                epochs, batch_size
            )
        elif self.adaptation_method == 'coral':
            self.model = self._coral_adaptation(
                source_data, target_data, source_labels, target_labels,
                epochs, batch_size
            )
        elif self.adaptation_method == 'mmd':
            self.model = self._mmd_adaptation(
                source_data, target_data, source_labels, target_labels,
                epochs, batch_size
            )
        else:
            raise ValueError(f"Unknown adaptation method: {self.adaptation_method}")
            
        return self.model
    
    def _domain_adversarial_adaptation(self, source_data: np.ndarray, target_data: np.ndarray,
                                     source_labels: np.ndarray, target_labels: np.ndarray,
                                     epochs: int, batch_size: int) -> tf.keras.Model:
        """
        Implement Domain-Adversarial Neural Network (DANN) for adaptation.
        
        Args:
            source_data: Data from source domain
            target_data: Data from target domain
            source_labels: Labels from source domain
            target_labels: Labels from target domain (can be None)
            epochs: Number of epochs for adaptation
            batch_size: Batch size for training
            
        Returns:
            Adapted model
        """
        # This is a simplified implementation of DANN
        # In practice, this would involve a custom training loop with gradient reversal
        
        # Extract feature layers from source model
        feature_extractor = self.source_model.layers[:-1]
        
        # Create input layer
        input_layer = tf.keras.layers.Input(shape=self.source_model.input_shape[1:])
        
        # Feature extractor
        features = input_layer
        for layer in feature_extractor:
            features = layer(features)
        
        # Task classifier (label predictor)
        task_output = self.source_model.layers[-1](features)
        
        # Domain classifier
        domain_features = tf.keras.layers.GradientReversalLayer()(features)  # Pseudo-code, need custom implementation
        domain_features = tf.keras.layers.Dense(128, activation='relu')(domain_features)
        domain_output = tf.keras.layers.Dense(1, activation='sigmoid')(domain_features)
        
        # Create model with two outputs
        model = tf.keras.Model(inputs=input_layer, outputs=[task_output, domain_output])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=['sparse_categorical_crossentropy', 'binary_crossentropy'],
            metrics=['accuracy']
        )
        
        # Prepare domain labels
        source_domain = np.zeros(len(source_data))
        target_domain = np.ones(len(target_data))
        
        # Train model
        # This is simplified; actual implementation would require custom training loop
        model.fit(
            [np.concatenate([source_data, target_data])],
            [np.concatenate([source_labels, np.zeros_like(target_domain)]),
             np.concatenate([source_domain, target_domain])],
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Return just the task part for inference
        inference_model = tf.keras.Model(inputs=input_layer, outputs=task_output)
        return inference_model
    
    def _coral_adaptation(self, source_data: np.ndarray, target_data: np.ndarray,
                        source_labels: np.ndarray, target_labels: np.ndarray,
                        epochs: int, batch_size: int) -> tf.keras.Model:
        """
        Implement Correlation Alignment (CORAL) for adaptation.
        
        Args:
            source_data: Data from source domain
            target_data: Data from target domain
            source_labels: Labels from source domain
            target_labels: Labels from target domain (can be None)
            epochs: Number of epochs for adaptation
            batch_size: Batch size for training
            
        Returns:
            Adapted model
        """
        # Extract features
        extractor = FeatureExtractor(self.source_model)
        source_features = extractor.extract_features(source_data)
        target_features = extractor.extract_features(target_data)
        
        # Calculate covariance matrices
        source_cov = np.cov(source_features, rowvar=False)
        target_cov = np.cov(target_features, rowvar=False)
        
        # Calculate transformation matrix
        source_cov_sqrt_inv = np.linalg.inv(scipy.linalg.sqrtm(source_cov))
        transform = np.dot(source_cov_sqrt_inv, scipy.linalg.sqrtm(np.dot(np.dot(source_cov_sqrt_inv, target_cov), source_cov_sqrt_inv)))
        
        # Create transformation layer
        class CoralLayer(tf.keras.layers.Layer):
            def __init__(self, transform_matrix, **kwargs):
                super(CoralLayer, self).__init__(**kwargs)
                self.transform_matrix = transform_matrix
                
            def call(self, inputs):
                return tf.matmul(inputs, self.transform_matrix)
        
        # Build adapted model
        input_layer = tf.keras.layers.Input(shape=self.source_model.input_shape[1:])
        features = extractor.feature_model(input_layer)
        coral_features = CoralLayer(transform)(features)
        output = self.source_model.layers[-1](coral_features)
        
        adapted_model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        # Compile model
        adapted_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='sparse_categorical_crossentropy' if isinstance(output, tf.Tensor) and output.shape[-1] > 1 else 'mse',
            metrics=['accuracy'] if isinstance(output, tf.Tensor) and output.shape[-1] > 1 else ['mae']
        )
        
        return adapted_model
    
    def _mmd_adaptation(self, source_data: np.ndarray, target_data: np.ndarray,
                      source_labels: np.ndarray, target_labels: np.ndarray,
                      epochs: int, batch_size: int) -> tf.keras.Model:
        """
        Implement Maximum Mean Discrepancy (MMD) for adaptation.
        
        Args:
            source_data: Data from source domain
            target_data: Data from target domain
            source_labels: Labels from source domain
            target_labels: Labels from target domain (can be None)
            epochs: Number of epochs for adaptation
            batch_size: Batch size for training
            
        Returns:
            Adapted model
        """
        # This is a placeholder for MMD implementation
        # Actual implementation would require custom loss functions and training loop
        
        # For now, return a simple fine-tuned model if target labels are available
        if target_labels is not None:
            model = tf.keras.models.clone_model(self.source_model)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='sparse_categorical_crossentropy' if model.output_shape[-1] > 1 else 'mse',
                metrics=['accuracy'] if model.output_shape[-1] > 1 else ['mae']
            )
            model.fit(target_data, target_labels, epochs=epochs, batch_size=batch_size)
            return model
        else:
            return self.source_model


class PretrainedFinanceModel:
    """
    Specialized models pre-trained on financial data.
    
    This class provides access to models that have been pre-trained on
    financial time series data, which can be fine-tuned for crypto tasks.
    """
    
    def __init__(self, model_type: str = 'price_prediction', market: str = 'stocks'):
        """
        Initialize a pre-trained finance model.
        
        Args:
            model_type: Type of model ('price_prediction', 'volatility', etc.)
            market: Market the model was trained on ('stocks', 'forex', etc.)
        """
        self.model_type = model_type
        self.market = market
        self.model = self._load_pretrained_model()
        
    def _load_pretrained_model(self) -> tf.keras.Model:
        """
        Load a pre-trained model for the specified task and market.
        
        Returns:
            Pre-trained Keras model
        """
        # This would typically load from a model repository
        # For now, creating dummy pre-trained models
        
        if self.model_type == 'price_prediction':
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(30, 5)),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
        elif self.model_type == 'volatility':
            model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(30, 5)),
                tf.keras.layers.MaxPooling1D(2),
                tf.keras.layers.Conv1D(128, 3, activation='relu'),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='relu')
            ])
        elif self.model_type == 'sentiment':
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(10000, 128, input_length=100),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        # Pretend the model is pre-trained
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def adapt_to_crypto(self, data: np.ndarray, labels: np.ndarray = None,
                      epochs: int = 5, batch_size: int = 32,
                      adaptation_method: str = 'fine_tune') -> tf.keras.Model:
        """
        Adapt the pre-trained finance model to cryptocurrency data.
        
        Args:
            data: Cryptocurrency data
            labels: Labels for supervised adaptation (if available)
            epochs: Number of epochs for adaptation
            batch_size: Batch size for training
            adaptation_method: Method for adaptation ('fine_tune', 'dann', 'feature_extract')
            
        Returns:
            Adapted model for cryptocurrency
        """
        if adaptation_method == 'fine_tune':
            # Simple fine-tuning if labels are available
            if labels is None:
                raise ValueError("Labels are required for fine-tuning")
                
            model = tf.keras.models.clone_model(self.model)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='mse',
                metrics=['mae']
            )
            model.fit(data, labels, epochs=epochs, batch_size=batch_size)
            return model
            
        elif adaptation_method == 'dann':
            # Domain-adversarial adaptation
            adapter = DomainAdaptation(self.model, adaptation_method='dann')
            
            # Generate synthetic source data if needed
            source_data = np.random.normal(size=(1000, *data.shape[1:]))
            source_labels = np.random.normal(size=(1000, 1)) if self.model.output_shape[-1] == 1 else \
                           np.random.randint(0, self.model.output_shape[-1], size=1000)
            
            return adapter.adapt(source_data, data, source_labels, labels, epochs, batch_size)
            
        elif adaptation_method == 'feature_extract':
            # Feature extraction and new head
            extractor = FeatureExtractor(self.model)
            features = extractor.extract_features(data)
            
            # Train a new model on extracted features
            feature_input = tf.keras.layers.Input(shape=extractor.get_feature_shape())
            x = tf.keras.layers.Dense(64, activation='relu')(feature_input)
            output = tf.keras.layers.Dense(1)(x) if self.model.output_shape[-1] == 1 else \
                    tf.keras.layers.Dense(self.model.output_shape[-1], activation='softmax')(x)
            
            head_model = tf.keras.Model(inputs=feature_input, outputs=output)
            head_model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss='mse' if self.model.output_shape[-1] == 1 else 'sparse_categorical_crossentropy',
                metrics=['mae'] if self.model.output_shape[-1] == 1 else ['accuracy']
            )
            
            if labels is not None:
                head_model.fit(features, labels, epochs=epochs, batch_size=batch_size)
            
            # Create full pipeline model
            input_layer = tf.keras.layers.Input(shape=self.model.input_shape[1:])
            features = extractor.feature_model(input_layer)
            output = head_model(features)
            
            full_model = tf.keras.Model(inputs=input_layer, outputs=output)
            return full_model
        
        else:
            raise ValueError(f"Unknown adaptation method: {adaptation_method}")
    
    def predict(self, x: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Generate predictions with the model.
        
        Args:
            x: Input data
            batch_size: Batch size for prediction
            
        Returns:
            Array of predictions
        """
        return self.model.predict(x, batch_size=batch_size)
    
    def get_model(self) -> tf.keras.Model:
        """
        Get the pre-trained model.
        
        Returns:
            Pre-trained Keras model
        """
        return self.model 