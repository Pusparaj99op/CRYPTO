"""
Continual Learning module for cryptocurrency applications.

This module implements continual learning techniques that allow models to
learn new tasks/patterns incrementally without forgetting previously learned
knowledge, which is essential for adapting to changing market conditions.

Classes:
    EWCModel: Elastic Weight Consolidation implementation
    ReplayBuffer: Experience replay for continual learning
    ProgressiveNets: Progressive neural networks approach
    DynamicArchitecture: Dynamic architecture adaptation
"""

import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Union, Optional, Tuple, Callable
import copy
import random
from collections import deque

class EWCModel:
    """
    Elastic Weight Consolidation (EWC) implementation.
    
    EWC prevents catastrophic forgetting by adding a regularization term
    that discourages updates to parameters that were important for 
    previously learned tasks.
    """
    
    def __init__(self, model: tf.keras.Model, ewc_lambda: float = 0.1):
        """
        Initialize an EWC model.
        
        Args:
            model: Base model to be protected against forgetting
            ewc_lambda: Regularization strength for EWC penalty
        """
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher_information = None
        self.optimal_weights = None
        
    def _compute_fisher_information(self, dataset: tf.data.Dataset) -> Dict[str, np.ndarray]:
        """
        Compute Fisher Information Matrix for model parameters.
        
        Args:
            dataset: Dataset to compute Fisher information from
            
        Returns:
            Dictionary mapping parameter names to Fisher information
        """
        # Initialize Fisher information matrices for each parameter
        fisher_information = {}
        for var in self.model.trainable_variables:
            fisher_information[var.name] = tf.zeros_like(var)
            
        # Compute Fisher information using gradients
        batch_count = 0
        for x_batch, y_batch in dataset:
            batch_count += 1
            
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = self.model(x_batch, training=True)
                
                # Compute log-likelihood (for classification)
                if y_batch.shape[-1] > 1:  # One-hot encoded labels
                    log_likelihood = tf.reduce_sum(
                        tf.nn.log_softmax(predictions, axis=1) * y_batch, axis=1
                    )
                else:  # Regression
                    # Use normal distribution log-likelihood
                    log_likelihood = -0.5 * tf.square(predictions - y_batch)
                    
            # Compute gradients of log-likelihood
            gradients = tape.gradient(log_likelihood, self.model.trainable_variables)
            
            # Accumulate squared gradients in Fisher information matrices
            for i, var in enumerate(self.model.trainable_variables):
                if gradients[i] is not None:
                    fisher_information[var.name] += tf.square(gradients[i])
        
        # Average over batches
        for var_name in fisher_information:
            fisher_information[var_name] /= batch_count
            
        return fisher_information
    
    def consolidate_task(self, dataset: tf.data.Dataset) -> None:
        """
        Consolidate current task to protect it from forgetting.
        
        Args:
            dataset: Dataset for the current task
        """
        # Compute Fisher information matrix
        self.fisher_information = self._compute_fisher_information(dataset)
        
        # Store current model weights as optimal for this task
        self.optimal_weights = {}
        for var in self.model.trainable_variables:
            self.optimal_weights[var.name] = var.numpy().copy()
            
    def ewc_loss(self, model_variables: List[tf.Variable]) -> tf.Tensor:
        """
        Compute the EWC regularization loss.
        
        Args:
            model_variables: Current model trainable variables
            
        Returns:
            EWC penalty term
        """
        if self.fisher_information is None or self.optimal_weights is None:
            return tf.constant(0.0)  # No penalty if no previous task
            
        # Compute EWC penalty
        penalty = 0
        for i, var in enumerate(model_variables):
            if var.name in self.fisher_information and var.name in self.optimal_weights:
                # Penalize changes to important parameters
                _penalty = tf.reduce_sum(
                    self.fisher_information[var.name] * 
                    tf.square(var - self.optimal_weights[var.name])
                )
                penalty += _penalty
                
        return 0.5 * self.ewc_lambda * penalty
    
    def train_step(self, x: tf.Tensor, y: tf.Tensor, 
                 optimizer: tf.keras.optimizers.Optimizer,
                 loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]) -> float:
        """
        Perform one training step with EWC regularization.
        
        Args:
            x: Input data
            y: Target data
            optimizer: Optimizer instance
            loss_fn: Loss function
            
        Returns:
            Total loss value (task loss + EWC penalty)
        """
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(x, training=True)
            
            # Task loss
            task_loss = loss_fn(y, predictions)
            
            # EWC penalty
            ewc_penalty = self.ewc_loss(self.model.trainable_variables)
            
            # Total loss
            total_loss = task_loss + ewc_penalty
            
        # Compute gradients and update model
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return total_loss
    
    def fit(self, x: np.ndarray, y: np.ndarray, 
           batch_size: int = 32, epochs: int = 10,
           optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
           validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, List[float]]:
        """
        Train the model with EWC regularization.
        
        Args:
            x: Training features
            y: Training targets
            batch_size: Batch size for training
            epochs: Number of epochs to train
            optimizer: Optimizer instance (creates Adam if None)
            validation_data: Optional tuple of validation (x, y)
            
        Returns:
            Dictionary containing training history
        """
        # Create optimizer if not provided
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam()
            
        # Determine loss function based on task type
        if len(y.shape) > 1 and y.shape[1] > 1:  # Classification with one-hot encoding
            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        else:  # Regression
            loss_fn = tf.keras.losses.MeanSquaredError()
            
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.batch(batch_size)
        
        # Training history
        history = {
            'loss': [],
            'val_loss': [] if validation_data is not None else None
        }
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            for x_batch, y_batch in dataset:
                batch_loss = self.train_step(x_batch, y_batch, optimizer, loss_fn)
                epoch_loss += batch_loss
                batch_count += 1
                
            # Average loss for epoch
            epoch_loss /= batch_count
            history['loss'].append(epoch_loss.numpy())
            
            # Validation if provided
            if validation_data is not None:
                val_x, val_y = validation_data
                val_pred = self.model(val_x)
                val_loss = loss_fn(val_y, val_pred)
                history['val_loss'].append(val_loss.numpy())
                
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
                
        return history
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Generate predictions with the model.
        
        Args:
            x: Input features
            
        Returns:
            Model predictions
        """
        return self.model.predict(x)


class ReplayBuffer:
    """
    Experience replay buffer for continual learning.
    
    This class implements a replay buffer that stores examples from previous
    tasks and periodically replays them during training to prevent forgetting.
    """
    
    def __init__(self, capacity: int = 10000, strategy: str = 'random'):
        """
        Initialize a replay buffer.
        
        Args:
            capacity: Maximum number of examples to store
            strategy: Strategy for managing buffer ('random', 'reservoir', 'balanced')
        """
        self.capacity = capacity
        self.strategy = strategy
        self.buffer = []
        self.task_indices = {}  # Keeps track of examples per task
        self.total_seen = 0  # Total number of examples seen (for reservoir sampling)
        
    def add_example(self, x: np.ndarray, y: np.ndarray, task_id: int) -> None:
        """
        Add a single example to the buffer.
        
        Args:
            x: Input features
            y: Target value
            task_id: ID of the task this example belongs to
        """
        self.total_seen += 1
        
        # Initialize task entry if not exists
        if task_id not in self.task_indices:
            self.task_indices[task_id] = []
            
        if self.strategy == 'random':
            # Simple random replacement when buffer is full
            if len(self.buffer) < self.capacity:
                self.buffer.append((x, y, task_id))
                self.task_indices[task_id].append(len(self.buffer) - 1)
            else:
                # Replace a random example
                idx_to_replace = random.randint(0, self.capacity - 1)
                old_task_id = self.buffer[idx_to_replace][2]
                
                # Update task indices
                self.task_indices[old_task_id].remove(idx_to_replace)
                self.buffer[idx_to_replace] = (x, y, task_id)
                self.task_indices[task_id].append(idx_to_replace)
                
        elif self.strategy == 'reservoir':
            # Reservoir sampling (each example has equal probability to be in buffer)
            if len(self.buffer) < self.capacity:
                self.buffer.append((x, y, task_id))
                self.task_indices[task_id].append(len(self.buffer) - 1)
            else:
                # Replace with probability capacity/total_seen
                if random.random() < self.capacity / self.total_seen:
                    idx_to_replace = random.randint(0, self.capacity - 1)
                    old_task_id = self.buffer[idx_to_replace][2]
                    
                    # Update task indices
                    self.task_indices[old_task_id].remove(idx_to_replace)
                    self.buffer[idx_to_replace] = (x, y, task_id)
                    self.task_indices[task_id].append(idx_to_replace)
                    
        elif self.strategy == 'balanced':
            # Keep a balanced number of examples per task
            if len(self.buffer) < self.capacity:
                self.buffer.append((x, y, task_id))
                self.task_indices[task_id].append(len(self.buffer) - 1)
            else:
                # Find task with most examples
                max_task = max(self.task_indices.items(), key=lambda x: len(x[1]))
                max_task_id, max_indices = max_task
                
                # If current task has fewer examples than max, replace one from max
                if len(self.task_indices[task_id]) < len(max_indices):
                    # Replace a random example from the task with most examples
                    idx_in_task = random.randint(0, len(max_indices) - 1)
                    idx_to_replace = max_indices[idx_in_task]
                    
                    # Update task indices
                    self.task_indices[max_task_id].remove(idx_to_replace)
                    self.buffer[idx_to_replace] = (x, y, task_id)
                    self.task_indices[task_id].append(idx_to_replace)
        else:
            raise ValueError(f"Unknown replay buffer strategy: {self.strategy}")
            
    def add_dataset(self, x: np.ndarray, y: np.ndarray, task_id: int, 
                   sample_percentage: float = 0.1) -> None:
        """
        Add examples from a dataset to the buffer.
        
        Args:
            x: Input features array
            y: Target values array
            task_id: ID of the task this dataset belongs to
            sample_percentage: Percentage of dataset to store (0.0-1.0)
        """
        # Determine number of examples to store
        n_examples = x.shape[0]
        n_to_store = max(1, int(n_examples * sample_percentage))
        
        # Randomly select indices to store
        indices = np.random.choice(n_examples, size=n_to_store, replace=False)
        
        # Add selected examples to buffer
        for idx in indices:
            self.add_example(x[idx], y[idx], task_id)
            
    def get_batch(self, batch_size: int = 32, task_id: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a batch of examples from the buffer.
        
        Args:
            batch_size: Number of examples to retrieve
            task_id: If provided, only get examples from this task
            
        Returns:
            Tuple of (x_batch, y_batch) arrays
        """
        if not self.buffer:
            raise ValueError("Cannot get batch from empty buffer")
            
        if task_id is not None and (task_id not in self.task_indices or not self.task_indices[task_id]):
            raise ValueError(f"No examples stored for task {task_id}")
            
        # Select indices to retrieve
        if task_id is not None:
            # Get examples only from specified task
            indices = np.random.choice(
                self.task_indices[task_id], 
                size=min(batch_size, len(self.task_indices[task_id])), 
                replace=False
            )
        else:
            # Get examples from all tasks
            indices = np.random.choice(
                len(self.buffer), 
                size=min(batch_size, len(self.buffer)), 
                replace=False
            )
            
        # Extract examples
        batch_x = []
        batch_y = []
        
        for idx in indices:
            x, y, _ = self.buffer[idx]
            batch_x.append(x)
            batch_y.append(y)
            
        return np.array(batch_x), np.array(batch_y)
    
    def get_all_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all examples stored in the buffer.
        
        Returns:
            Tuple of (x_array, y_array, task_ids_array)
        """
        if not self.buffer:
            return np.array([]), np.array([]), np.array([])
            
        x_data = []
        y_data = []
        task_ids = []
        
        for x, y, task_id in self.buffer:
            x_data.append(x)
            y_data.append(y)
            task_ids.append(task_id)
            
        return np.array(x_data), np.array(y_data), np.array(task_ids)
    
    def __len__(self) -> int:
        """
        Get the number of examples in the buffer.
        
        Returns:
            Number of examples stored
        """
        return len(self.buffer)


class ProgressiveNets:
    """
    Progressive Neural Networks implementation.
    
    This approach prevents forgetting by creating a new network for each
    task, while allowing forward transfer of knowledge through lateral
    connections from previously learned networks.
    """
    
    def __init__(self, input_shape: Tuple, 
               hidden_layers: List[int] = [128, 64],
               output_shapes: List[int] = [],
               activation: str = 'relu'):
        """
        Initialize Progressive Neural Networks.
        
        Args:
            input_shape: Shape of input data (without batch dimension)
            hidden_layers: List of hidden layer sizes
            output_shapes: List of output sizes for each task (can add later)
            activation: Activation function to use in hidden layers
        """
        self.input_shape = input_shape
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.columns = []  # List of column networks
        
        # Add initial tasks if provided
        for output_shape in output_shapes:
            self.add_task(output_shape)
            
    def _create_lateral_connections(self, h_prev: List[tf.Tensor], h_new: tf.Tensor, 
                                  target_shape: Tuple) -> tf.Tensor:
        """
        Create lateral connections from previous columns to new column.
        
        Args:
            h_prev: List of hidden activations from previous columns
            h_new: Current column's activation
            target_shape: Shape of the target activation
            
        Returns:
            Combined activation including lateral connections
        """
        if not h_prev:
            return h_new  # No lateral connections for first column
            
        lateral_outputs = [h_new]
        
        # Add lateral connection from each previous column
        for i, h in enumerate(h_prev):
            # Adapter to match shapes
            adapter = tf.keras.layers.Dense(
                target_shape[-1],
                activation=self.activation,
                name=f"lateral_{i}_to_{len(self.columns)}"
            )(h)
            
            lateral_outputs.append(adapter)
            
        # Combine all inputs if there are lateral connections
        if len(lateral_outputs) > 1:
            return tf.keras.layers.Add()(lateral_outputs)
        else:
            return h_new
            
    def add_task(self, output_shape: int) -> tf.keras.Model:
        """
        Add a new task by creating a new column network.
        
        Args:
            output_shape: Size of output for this task
            
        Returns:
            Model for the new task
        """
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        
        # Keep track of hidden layer activations from previous columns
        prev_hidden_activations = []
        for col in self.columns:
            # Extract intermediate activations
            col_hidden = []
            for i in range(len(self.hidden_layers)):
                hidden_layer_name = f"hidden_{i}"
                for layer in col.layers:
                    if layer.name == hidden_layer_name:
                        # Create model to get this layer's output
                        activation_model = tf.keras.Model(
                            inputs=col.inputs,
                            outputs=layer.output
                        )
                        col_hidden.append(activation_model)
                        break
            prev_hidden_activations.append(col_hidden)
            
        # Create new column
        x = inputs
        hidden_outputs = []
        
        # Hidden layers with lateral connections
        for i, units in enumerate(self.hidden_layers):
            # Direct connection in current column
            h_new = tf.keras.layers.Dense(
                units, 
                activation=self.activation, 
                name=f"hidden_{i}"
            )(x)
            
            # Get activations from previous columns at this layer
            h_prev = []
            for col_idx, col_hidden in enumerate(prev_hidden_activations):
                if i < len(col_hidden):
                    h_prev.append(col_hidden[i](inputs))
                    
            # Add lateral connections
            h_combined = self._create_lateral_connections(
                h_prev, h_new, h_new.shape
            )
            
            x = h_combined
            hidden_outputs.append(x)
            
        # Output layer (no lateral connections)
        if output_shape > 1:
            outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
        else:
            outputs = tf.keras.layers.Dense(1)(x)
            
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Freeze previous columns
        for col in self.columns:
            for layer in col.layers:
                layer.trainable = False
                
        # Add new column to the list
        self.columns.append(model)
        
        return model
    
    def get_model(self, task_id: int) -> tf.keras.Model:
        """
        Get model for a specific task.
        
        Args:
            task_id: ID of the task (column index)
            
        Returns:
            Model for the specified task
        """
        if task_id < 0 or task_id >= len(self.columns):
            raise ValueError(f"Invalid task ID: {task_id}. Only {len(self.columns)} tasks available.")
            
        return self.columns[task_id]
    
    def predict(self, x: np.ndarray, task_id: int) -> np.ndarray:
        """
        Generate predictions for a specific task.
        
        Args:
            x: Input features
            task_id: ID of the task to use for prediction
            
        Returns:
            Model predictions
        """
        model = self.get_model(task_id)
        return model.predict(x)


class DynamicArchitecture:
    """
    Dynamic architecture for continual learning.
    
    This approach prevents forgetting by dynamically expanding
    the architecture (adding units or layers) when needed to learn
    new tasks without interfering with previous knowledge.
    """
    
    def __init__(self, base_model: tf.keras.Model, growth_rate: float = 0.2,
               similarity_threshold: float = 0.5):
        """
        Initialize dynamic architecture.
        
        Args:
            base_model: Initial model architecture
            growth_rate: Rate at which to grow network when needed
            similarity_threshold: Threshold for determining when to expand
        """
        self.base_model = base_model
        self.growth_rate = growth_rate
        self.similarity_threshold = similarity_threshold
        self.task_models = {0: base_model}  # Maps task_id to model
        self.current_task = 0
        
    def _compute_task_similarity(self, x1: np.ndarray, y1: np.ndarray,
                               x2: np.ndarray, y2: np.ndarray) -> float:
        """
        Compute similarity between two tasks.
        
        Args:
            x1, y1: Data for first task
            x2, y2: Data for second task
            
        Returns:
            Similarity score between 0 and 1
        """
        # Extract features using current model
        features1 = self._extract_features(self.base_model, x1)
        features2 = self._extract_features(self.base_model, x2)
        
        # Compute correlation between feature activations
        corr = np.corrcoef(features1.mean(axis=0), features2.mean(axis=0))[0, 1]
        
        # Normalize to [0, 1]
        similarity = (corr + 1) / 2
        
        return similarity
    
    def _extract_features(self, model: tf.keras.Model, x: np.ndarray) -> np.ndarray:
        """
        Extract features from the second-to-last layer of a model.
        
        Args:
            model: Model to extract features from
            x: Input data
            
        Returns:
            Feature activations
        """
        # Create feature extractor model
        if hasattr(model, 'layers') and len(model.layers) > 1:
            feature_layer = model.layers[-2].output
            extractor = tf.keras.Model(inputs=model.inputs, outputs=feature_layer)
            return extractor.predict(x)
        else:
            # Fallback if model structure doesn't allow easy extraction
            return x
    
    def _expand_network(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Expand network capacity by adding units to layers.
        
        Args:
            model: Model to expand
            
        Returns:
            Expanded model
        """
        # Create a new model with increased capacity
        input_shape = model.input_shape[1:]
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        x = inputs
        for i, layer in enumerate(model.layers[1:-1]):  # Skip input and output layers
            if isinstance(layer, tf.keras.layers.Dense):
                # Increase units by growth_rate
                new_units = int(layer.units * (1 + self.growth_rate))
                
                # Copy weights from old layer to new layer (for existing units)
                old_weights = layer.get_weights()
                old_kernel = old_weights[0]
                old_bias = old_weights[1]
                
                # Create new layer with increased capacity
                x = tf.keras.layers.Dense(
                    new_units,
                    activation=layer.activation,
                    name=layer.name
                )(x)
                
                # Initialize new layer with old weights
                new_layer = x.op.inputs[0].op._layer
                new_kernel = np.zeros((old_kernel.shape[0], new_units))
                new_bias = np.zeros(new_units)
                
                # Copy old weights
                new_kernel[:, :old_kernel.shape[1]] = old_kernel
                new_bias[:old_bias.shape[0]] = old_bias
                
                # Initialize new units with small random values
                new_kernel[:, old_kernel.shape[1]:] = np.random.normal(
                    0, 0.05, (old_kernel.shape[0], new_units - old_kernel.shape[1])
                )
                new_bias[old_bias.shape[0]:] = np.random.normal(
                    0, 0.05, (new_units - old_bias.shape[0])
                )
                
                # Set weights to the new layer
                new_layer.set_weights([new_kernel, new_bias])
            else:
                # For other layer types, just apply them as is
                x = layer(x)
                
        # Output layer (keep same size)
        output_layer = model.layers[-1]
        if isinstance(output_layer, tf.keras.layers.Dense):
            outputs = tf.keras.layers.Dense(
                output_layer.units,
                activation=output_layer.activation
            )(x)
        else:
            outputs = output_layer(x)
            
        # Create expanded model
        expanded_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Copy output layer weights
        if isinstance(output_layer, tf.keras.layers.Dense):
            output_weights = output_layer.get_weights()
            expanded_output_layer = expanded_model.layers[-1]
            
            # Adjust weights for potentially larger input size
            old_kernel = output_weights[0]
            old_bias = output_weights[1]
            new_kernel = np.zeros((expanded_output_layer.input_shape[1], old_kernel.shape[1]))
            
            # Copy shared dimensions
            min_rows = min(old_kernel.shape[0], new_kernel.shape[0])
            new_kernel[:min_rows, :] = old_kernel[:min_rows, :]
            
            # Set adjusted weights
            expanded_output_layer.set_weights([new_kernel, old_bias])
            
        return expanded_model
    
    def adapt_to_task(self, x: np.ndarray, y: np.ndarray, 
                    task_id: int, epochs: int = 10, batch_size: int = 32) -> tf.keras.Model:
        """
        Adapt the architecture to a new task.
        
        Args:
            x: Training features
            y: Training targets
            task_id: ID of the new task
            epochs: Number of epochs to train
            batch_size: Batch size for training
            
        Returns:
            Model adapted for the new task
        """
        self.current_task = task_id
        
        # If we already have a model for this task, return it
        if task_id in self.task_models:
            return self.task_models[task_id]
            
        # Find closest task
        closest_task = 0
        closest_similarity = 0
        
        for tid, model in self.task_models.items():
            # Need some data from the previous task
            # This is simplified - in practice, you'd need to store samples
            # from previous tasks or use a replay buffer
            if hasattr(self, 'task_data') and tid in self.task_data:
                prev_x, prev_y = self.task_data[tid]
                similarity = self._compute_task_similarity(prev_x, prev_y, x, y)
                
                if similarity > closest_similarity:
                    closest_similarity = similarity
                    closest_task = tid
        
        # Get model from closest task as starting point
        base_task_model = copy.deepcopy(self.task_models[closest_task])
        
        # Determine if we need to expand the network
        if closest_similarity < self.similarity_threshold:
            # Tasks are different enough to warrant expansion
            base_task_model = self._expand_network(base_task_model)
            
        # Fine-tune the model for the new task
        if len(y.shape) > 1 and y.shape[1] > 1:
            # Classification with one-hot encoding
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        else:
            # Regression
            loss = 'mse'
            metrics = ['mae']
            
        base_task_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=loss,
            metrics=metrics
        )
        
        base_task_model.fit(
            x, y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Store the model for this task
        self.task_models[task_id] = base_task_model
        
        # Store some data for task similarity computation
        if not hasattr(self, 'task_data'):
            self.task_data = {}
            
        # Store a subset of the data (to save memory)
        indices = np.random.choice(len(x), min(100, len(x)), replace=False)
        self.task_data[task_id] = (x[indices], y[indices])
        
        return base_task_model
    
    def predict(self, x: np.ndarray, task_id: Optional[int] = None) -> np.ndarray:
        """
        Generate predictions for a specific task.
        
        Args:
            x: Input features
            task_id: ID of the task to use (uses current_task if None)
            
        Returns:
            Model predictions
        """
        if task_id is None:
            task_id = self.current_task
            
        if task_id not in self.task_models:
            raise ValueError(f"No model available for task {task_id}")
            
        return self.task_models[task_id].predict(x)
    
    def get_model(self, task_id: Optional[int] = None) -> tf.keras.Model:
        """
        Get model for a specific task.
        
        Args:
            task_id: ID of the task (uses current_task if None)
            
        Returns:
            Model for the specified task
        """
        if task_id is None:
            task_id = self.current_task
            
        if task_id not in self.task_models:
            raise ValueError(f"No model available for task {task_id}")
            
        return self.task_models[task_id] 