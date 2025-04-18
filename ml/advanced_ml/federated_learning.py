"""
Federated Learning module for cryptocurrency applications.

This module implements federated learning techniques that allow training ML models
across multiple decentralized devices or servers holding local data samples without
exchanging them, thus addressing privacy concerns while enabling collaborative learning.

Classes:
    FederatedModel: Base class for federated learning models
    FederatedClient: Client implementation for federated learning
    FederatedServer: Server orchestrating federated learning
    FederatedAggregator: Component handling model aggregation
"""

import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Callable, Union, Optional, Tuple
import copy
import json
import os
import pickle

class FederatedModel:
    """
    Base class for models that can be trained in a federated manner.
    
    This class provides the foundation for creating models that can participate
    in federated learning processes, supporting various model architectures.
    """
    
    def __init__(self, model_architecture: Union[tf.keras.Model, Dict[str, Any]], 
                 input_shape: Tuple, output_shape: int,
                 learning_rate: float = 0.001):
        """
        Initialize a federated model.
        
        Args:
            model_architecture: Either a Keras model or a dictionary describing the architecture
            input_shape: Shape of the input data
            output_shape: Dimension of the output (e.g., number of classes)
            learning_rate: Learning rate for optimization
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        
        if isinstance(model_architecture, tf.keras.Model):
            self.model = model_architecture
        else:
            self.model = self._build_model(model_architecture)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy' if output_shape > 1 else 'mse',
            metrics=['accuracy'] if output_shape > 1 else ['mae']
        )
    
    def _build_model(self, architecture: Dict[str, Any]) -> tf.keras.Model:
        """
        Build a model from architecture description.
        
        Args:
            architecture: Dictionary describing the model architecture
            
        Returns:
            Compiled Keras model
        """
        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs
        
        # Process the layers based on the architecture dictionary
        for layer in architecture.get('layers', []):
            layer_type = layer.get('type')
            layer_params = layer.get('params', {})
            
            if layer_type == 'Dense':
                x = tf.keras.layers.Dense(**layer_params)(x)
            elif layer_type == 'LSTM':
                x = tf.keras.layers.LSTM(**layer_params)(x)
            elif layer_type == 'Conv1D':
                x = tf.keras.layers.Conv1D(**layer_params)(x)
            elif layer_type == 'Dropout':
                x = tf.keras.layers.Dropout(**layer_params)(x)
            elif layer_type == 'BatchNormalization':
                x = tf.keras.layers.BatchNormalization(**layer_params)(x)
            # Add other layer types as needed
        
        # Output layer
        if self.output_shape > 1:
            outputs = tf.keras.layers.Dense(self.output_shape, activation='softmax')(x)
        else:
            outputs = tf.keras.layers.Dense(1)(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def get_weights(self) -> List[np.ndarray]:
        """
        Get the current model weights.
        
        Returns:
            List of numpy arrays containing model weights
        """
        return self.model.get_weights()
    
    def set_weights(self, weights: List[np.ndarray]) -> None:
        """
        Set new weights to the model.
        
        Args:
            weights: List of numpy arrays containing model weights
        """
        self.model.set_weights(weights)
    
    def train(self, x: np.ndarray, y: np.ndarray, 
              batch_size: int = 32, epochs: int = 1,
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, List[float]]:
        """
        Train the model on local data.
        
        Args:
            x: Training features
            y: Training labels
            batch_size: Batch size for training
            epochs: Number of epochs to train
            validation_data: Optional tuple of validation (features, labels)
            
        Returns:
            Dictionary containing training history
        """
        history = self.model.fit(
            x, y, 
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            verbose=0
        )
        return history.history
    
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
        return self.model.evaluate(x, y, batch_size=batch_size, verbose=0)
    
    def predict(self, x: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Generate predictions with the model.
        
        Args:
            x: Input features
            batch_size: Batch size for prediction
            
        Returns:
            Array of predictions
        """
        return self.model.predict(x, batch_size=batch_size, verbose=0)
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path where to save the model
        """
        self.model.save(path)
    
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path from where to load the model
        """
        self.model = tf.keras.models.load_model(path)


class FederatedClient:
    """
    Client implementation for federated learning.
    
    This class represents a client in a federated learning system,
    which trains a local model on its private data and shares only
    model updates with the server.
    """
    
    def __init__(self, model: FederatedModel, client_id: str):
        """
        Initialize a federated client.
        
        Args:
            model: The federated model instance
            client_id: Unique identifier for this client
        """
        self.model = model
        self.client_id = client_id
        self.local_data = None
        self.local_labels = None
        
    def set_data(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Set local private data for this client.
        
        Args:
            x: Features data
            y: Labels data
        """
        self.local_data = x
        self.local_labels = y
        
    def train_local_model(self, epochs: int = 1, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the local model on private data.
        
        Args:
            epochs: Number of local training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary containing training metrics and updated weights
        """
        if self.local_data is None or self.local_labels is None:
            raise ValueError("Local data not set. Call set_data() first.")
        
        # Get initial weights
        initial_weights = copy.deepcopy(self.model.get_weights())
        
        # Train local model
        training_history = self.model.train(
            self.local_data, self.local_labels,
            batch_size=batch_size,
            epochs=epochs
        )
        
        # Calculate weight updates (difference from initial weights)
        final_weights = self.model.get_weights()
        weight_updates = [fw - iw for fw, iw in zip(final_weights, initial_weights)]
        
        return {
            "client_id": self.client_id,
            "training_metrics": training_history,
            "weight_updates": weight_updates,
            "data_samples": len(self.local_data)
        }
    
    def update_local_model(self, global_weights: List[np.ndarray]) -> None:
        """
        Update local model with global weights.
        
        Args:
            global_weights: Global model weights from the server
        """
        self.model.set_weights(global_weights)
    
    def evaluate_local_model(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate local model on private data.
        
        Args:
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.local_data is None or self.local_labels is None:
            raise ValueError("Local data not set. Call set_data() first.")
        
        metrics = self.model.evaluate(self.local_data, self.local_labels, batch_size=batch_size)
        
        # Match metrics with their names
        metric_names = self.model.model.metrics_names
        return {name: value for name, value in zip(metric_names, metrics)}


class FederatedServer:
    """
    Server orchestrating federated learning.
    
    This class represents the central server in a federated learning system,
    responsible for coordinating clients and aggregating model updates.
    """
    
    def __init__(self, global_model: FederatedModel, aggregator: 'FederatedAggregator'):
        """
        Initialize a federated server.
        
        Args:
            global_model: The global model to be trained
            aggregator: Aggregation method for client updates
        """
        self.global_model = global_model
        self.aggregator = aggregator
        self.clients = {}
        self.round_history = []
        
    def register_client(self, client: FederatedClient) -> None:
        """
        Register a client with the server.
        
        Args:
            client: The client to register
        """
        self.clients[client.client_id] = client
        
    def select_clients(self, num_clients: int = None, client_fraction: float = None) -> List[str]:
        """
        Select a subset of clients for the current round.
        
        Args:
            num_clients: Number of clients to select
            client_fraction: Fraction of clients to select
            
        Returns:
            List of selected client IDs
        """
        all_clients = list(self.clients.keys())
        
        if num_clients:
            num_to_select = min(num_clients, len(all_clients))
        elif client_fraction:
            num_to_select = max(1, int(client_fraction * len(all_clients)))
        else:
            num_to_select = len(all_clients)
            
        # Random selection without replacement
        selected_clients = np.random.choice(all_clients, num_to_select, replace=False)
        return selected_clients.tolist()
    
    def train_round(self, client_ids: List[str], local_epochs: int = 1, 
                    batch_size: int = 32) -> Dict[str, Any]:
        """
        Conduct a single round of federated learning.
        
        Args:
            client_ids: List of client IDs to participate in this round
            local_epochs: Number of local epochs for client training
            batch_size: Batch size for client training
            
        Returns:
            Dictionary containing round information and metrics
        """
        # Distribute global model to selected clients
        global_weights = self.global_model.get_weights()
        for client_id in client_ids:
            self.clients[client_id].update_local_model(global_weights)
        
        # Collect client updates
        client_updates = []
        for client_id in client_ids:
            update = self.clients[client_id].train_local_model(
                epochs=local_epochs,
                batch_size=batch_size
            )
            client_updates.append(update)
        
        # Aggregate updates
        aggregated_weights = self.aggregator.aggregate(global_weights, client_updates)
        
        # Update global model
        self.global_model.set_weights(aggregated_weights)
        
        # Evaluate global model performance (if test data available)
        # This would typically be done on a validation set held by the server
        
        # Record round information
        round_info = {
            "round_number": len(self.round_history) + 1,
            "num_clients": len(client_ids),
            "client_ids": client_ids,
            "client_updates": [
                {"client_id": update["client_id"], 
                 "samples": update["data_samples"],
                 "metrics": update["training_metrics"]} 
                for update in client_updates
            ]
        }
        
        self.round_history.append(round_info)
        return round_info
    
    def get_global_model(self) -> FederatedModel:
        """
        Get the current global model.
        
        Returns:
            The current global model
        """
        return self.global_model
    
    def save_global_model(self, path: str) -> None:
        """
        Save the global model to disk.
        
        Args:
            path: Path where to save the model
        """
        self.global_model.save(path)
    
    def load_global_model(self, path: str) -> None:
        """
        Load the global model from disk.
        
        Args:
            path: Path from where to load the model
        """
        self.global_model.load(path)
    
    def save_round_history(self, path: str) -> None:
        """
        Save the training history to disk.
        
        Args:
            path: Path where to save the history
        """
        with open(path, 'w') as f:
            json.dump(self.round_history, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


class FederatedAggregator:
    """
    Component handling model aggregation in federated learning.
    
    This class implements different strategies for aggregating model updates
    from multiple clients in a federated learning system.
    """
    
    def __init__(self, aggregation_method: str = 'fedavg'):
        """
        Initialize an aggregator with a specific method.
        
        Args:
            aggregation_method: Method to use for aggregation ('fedavg', 'weighted', etc.)
        """
        self.aggregation_method = aggregation_method
        self.aggregation_functions = {
            'fedavg': self._federated_averaging,
            'weighted': self._weighted_averaging,
            'median': self._median_aggregation
        }
        
    def aggregate(self, global_weights: List[np.ndarray], 
                 client_updates: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Aggregate updates from multiple clients.
        
        Args:
            global_weights: Current global model weights
            client_updates: List of client update dictionaries
            
        Returns:
            Aggregated model weights
        """
        if self.aggregation_method not in self.aggregation_functions:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
            
        return self.aggregation_functions[self.aggregation_method](global_weights, client_updates)
    
    def _federated_averaging(self, global_weights: List[np.ndarray], 
                           client_updates: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Implement FedAvg aggregation algorithm.
        
        Args:
            global_weights: Current global model weights
            client_updates: List of client update dictionaries
            
        Returns:
            Aggregated model weights
        """
        # Extract weight updates and sample counts
        weight_updates = []
        sample_counts = []
        
        for update in client_updates:
            weight_updates.append(update["weight_updates"])
            sample_counts.append(update["data_samples"])
            
        # Calculate weighted average of updates
        total_samples = sum(sample_counts)
        weighted_updates = []
        
        for i in range(len(global_weights)):
            layer_updates = np.zeros_like(global_weights[i])
            
            for j in range(len(weight_updates)):
                weight = sample_counts[j] / total_samples
                layer_updates += weight * weight_updates[j][i]
                
            weighted_updates.append(layer_updates)
            
        # Apply updates to global model
        new_weights = [gw + wu for gw, wu in zip(global_weights, weighted_updates)]
        
        return new_weights
    
    def _weighted_averaging(self, global_weights: List[np.ndarray], 
                          client_updates: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Implement weighted averaging with custom weights.
        
        This method allows for non-uniform weighting of client updates,
        potentially based on factors other than just data sample count.
        
        Args:
            global_weights: Current global model weights
            client_updates: List of client update dictionaries
            
        Returns:
            Aggregated model weights
        """
        # This implementation could include additional weighting factors
        # For now, using sample count as in FedAvg
        return self._federated_averaging(global_weights, client_updates)
    
    def _median_aggregation(self, global_weights: List[np.ndarray], 
                          client_updates: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Implement element-wise median aggregation.
        
        This method is more robust against outliers or poisoning attacks.
        
        Args:
            global_weights: Current global model weights
            client_updates: List of client update dictionaries
            
        Returns:
            Aggregated model weights
        """
        # Extract weight updates
        all_weights = []
        
        # First, apply all updates to get final weights from each client
        for update in client_updates:
            client_final_weights = [gw + wu for gw, wu in zip(global_weights, update["weight_updates"])]
            all_weights.append(client_final_weights)
            
        # Compute element-wise median
        median_weights = []
        for i in range(len(global_weights)):
            # Stack same layer from all clients
            stacked = np.stack([cw[i] for cw in all_weights], axis=0)
            # Compute median along client dimension
            median_layer = np.median(stacked, axis=0)
            median_weights.append(median_layer)
            
        return median_weights 