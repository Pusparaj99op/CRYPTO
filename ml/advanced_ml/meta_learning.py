"""
Meta-Learning module for cryptocurrency applications.

This module implements meta-learning techniques that enable models to learn
how to learn, adapting quickly to new tasks with minimal data, which is
particularly useful for financial markets where conditions change rapidly.

Classes:
    MAMLLearner: Model-Agnostic Meta-Learning implementation
    PrototypicalNetwork: Few-shot learning with prototypical networks
    RelationNetwork: Few-shot learning with relation networks
    MetaOptimizer: Meta-optimizer for learning optimization strategies
"""

import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Union, Optional, Tuple, Callable
import copy
import random

class MAMLLearner:
    """
    Model-Agnostic Meta-Learning (MAML) implementation.
    
    MAML learns an initialization for model parameters that can be quickly
    adapted to new tasks with just a few examples and gradient steps.
    """
    
    def __init__(self, model_fn: Callable[[], tf.keras.Model],
                 inner_lr: float = 0.01, meta_lr: float = 0.001,
                 inner_steps: int = 1, task_batch_size: int = 5):
        """
        Initialize a MAML learner.
        
        Args:
            model_fn: Function that returns a new model instance
            inner_lr: Learning rate for task-specific adaptation
            meta_lr: Learning rate for meta-update
            inner_steps: Number of gradient steps for task adaptation
            task_batch_size: Number of tasks to sample per meta-update
        """
        self.model_fn = model_fn
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.task_batch_size = task_batch_size
        
        # Create model and optimizer
        self.model = model_fn()
        self.meta_optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr)
        
    def _compute_task_loss(self, model: tf.keras.Model, batch: Tuple[np.ndarray, np.ndarray]) -> tf.Tensor:
        """
        Compute loss for a specific task.
        
        Args:
            model: Model to compute loss for
            batch: Tuple of (x, y) data
            
        Returns:
            Loss value as a tensor
        """
        x, y = batch
        predictions = model(x, training=True)
        
        if y.shape[-1] > 1:  # One-hot encoded labels (classification)
            loss = tf.keras.losses.categorical_crossentropy(y, predictions)
        else:  # Regression
            loss = tf.keras.losses.mean_squared_error(y, predictions)
            
        return tf.reduce_mean(loss)
    
    def _inner_loop(self, model: tf.keras.Model, support_batch: Tuple[np.ndarray, np.ndarray]) -> Dict[str, tf.Variable]:
        """
        Perform inner loop adaptation for a task.
        
        Args:
            model: Starting model (meta-initialization)
            support_batch: Support set data (x, y) for adaptation
            
        Returns:
            Dictionary of adapted parameters
        """
        adapted_variables = {}
        variables = model.trainable_variables
        
        # Store the initial parameters
        for variable in variables:
            adapted_variables[variable.name] = tf.Variable(variable)
            
        # Perform adaptation steps
        for _ in range(self.inner_steps):
            with tf.GradientTape() as tape:
                loss = self._compute_task_loss(model, support_batch)
                
            gradients = tape.gradient(loss, variables)
            
            # Update each variable manually
            for var, grad in zip(variables, gradients):
                if grad is not None:
                    adapted_variables[var.name].assign_sub(self.inner_lr * grad)
                    var.assign(adapted_variables[var.name])
                    
        return adapted_variables
    
    def meta_train_step(self, tasks: List[Dict[str, Tuple[np.ndarray, np.ndarray]]]) -> float:
        """
        Perform one meta-training step.
        
        Args:
            tasks: List of tasks, each with support and query batches
            
        Returns:
            Meta-loss value
        """
        meta_loss = 0.0
        
        with tf.GradientTape() as meta_tape:
            for task in tasks:
                support_batch = task['support']
                query_batch = task['query']
                
                # Clone model to avoid inplace updates
                task_model = tf.keras.models.clone_model(self.model)
                task_model.set_weights(self.model.get_weights())
                
                # Adapt model to task
                adapted_variables = self._inner_loop(task_model, support_batch)
                
                # Apply adapted parameters temporarily
                original_vars = {}
                for var in self.model.trainable_variables:
                    original_vars[var.name] = var.value()
                    var.assign(adapted_variables[var.name])
                
                # Compute loss on query set with adapted parameters
                query_loss = self._compute_task_loss(self.model, query_batch)
                meta_loss += query_loss
                
                # Restore original parameters
                for var in self.model.trainable_variables:
                    var.assign(original_vars[var.name])
                
            # Average meta-loss
            meta_loss /= len(tasks)
            
        # Meta-update
        meta_gradients = meta_tape.gradient(meta_loss, self.model.trainable_variables)
        self.meta_optimizer.apply_gradients(zip(meta_gradients, self.model.trainable_variables))
        
        return meta_loss.numpy()
    
    def meta_train(self, task_generator: Callable[[int], List[Dict[str, Tuple[np.ndarray, np.ndarray]]]],
                 epochs: int, tasks_per_epoch: int) -> List[float]:
        """
        Train the model using MAML.
        
        Args:
            task_generator: Function that generates tasks for meta-learning
            epochs: Number of meta-training epochs
            tasks_per_epoch: Number of tasks to sample per epoch
            
        Returns:
            List of meta-loss values per epoch
        """
        meta_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for i in range(tasks_per_epoch):
                tasks = task_generator(self.task_batch_size)
                loss = self.meta_train_step(tasks)
                epoch_loss += loss
                
            epoch_loss /= tasks_per_epoch
            meta_losses.append(epoch_loss)
            
            # Log progress
            print(f"Epoch {epoch+1}/{epochs}, Meta-Loss: {epoch_loss:.4f}")
            
        return meta_losses
    
    def adapt_to_task(self, support_batch: Tuple[np.ndarray, np.ndarray], 
                     steps: int = None) -> tf.keras.Model:
        """
        Adapt the meta-trained model to a new task.
        
        Args:
            support_batch: Support set data (x, y) for adaptation
            steps: Number of adaptation steps (uses self.inner_steps if None)
            
        Returns:
            Adapted model for the new task
        """
        steps = steps if steps is not None else self.inner_steps
        
        # Clone model to avoid modifying the original
        adapted_model = tf.keras.models.clone_model(self.model)
        adapted_model.set_weights(self.model.get_weights())
        
        # Configure for training
        x, y = support_batch
        if y.shape[-1] > 1:  # Classification
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
        else:  # Regression
            loss_fn = tf.keras.losses.MeanSquaredError()
            
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.inner_lr)
        
        # Perform adaptation
        for _ in range(steps):
            with tf.GradientTape() as tape:
                predictions = adapted_model(x, training=True)
                loss = loss_fn(y, predictions)
                
            grads = tape.gradient(loss, adapted_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, adapted_model.trainable_variables))
            
        return adapted_model


class PrototypicalNetwork:
    """
    Few-shot learning with Prototypical Networks.
    
    Prototypical Networks learn an embedding space where classes can be
    represented by a single prototype (mean of support examples), allowing
    for effective few-shot classification.
    """
    
    def __init__(self, input_shape: Tuple, embedding_dim: int = 64):
        """
        Initialize a Prototypical Network.
        
        Args:
            input_shape: Shape of input data (without batch dimension)
            embedding_dim: Dimension of the embedding space
        """
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.model = self._build_embedding_model()
        self.optimizer = tf.keras.optimizers.Adam()
        
    def _build_embedding_model(self) -> tf.keras.Model:
        """
        Build the embedding network.
        
        Returns:
            Keras model for embedding inputs
        """
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        
        # For image inputs
        if len(self.input_shape) == 3:  # (height, width, channels)
            x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D(2)(x)
            
            x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D(2)(x)
            
            x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D(2)(x)
            
            x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D(2)(x)
            
            x = tf.keras.layers.Flatten()(x)
            
        # For time series or 1D data
        elif len(self.input_shape) == 2:  # (time_steps, features)
            x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
            x = tf.keras.layers.LSTM(64)(x)
            
        # For vector inputs
        else:
            x = tf.keras.layers.Dense(128, activation='relu')(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            
        # Final embedding layer
        outputs = tf.keras.layers.Dense(self.embedding_dim)(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def compute_prototypes(self, support_images: tf.Tensor, support_labels: tf.Tensor) -> tf.Tensor:
        """
        Compute class prototypes from support examples.
        
        Args:
            support_images: Support set images [n_support, height, width, channels]
            support_labels: Support set one-hot labels [n_support, n_classes]
            
        Returns:
            Class prototypes [n_classes, embedding_dim]
        """
        embeddings = self.model(support_images)
        n_classes = support_labels.shape[1]
        
        prototypes = []
        for i in range(n_classes):
            # Get embeddings for this class
            class_mask = support_labels[:, i]
            class_embeddings = embeddings * tf.expand_dims(class_mask, -1)
            
            # Compute mean, but avoid division by zero
            class_size = tf.maximum(tf.reduce_sum(class_mask), 1.0)
            prototype = tf.reduce_sum(class_embeddings, axis=0) / class_size
            prototypes.append(prototype)
            
        return tf.stack(prototypes)
    
    def compute_distances(self, query_embeddings: tf.Tensor, prototypes: tf.Tensor) -> tf.Tensor:
        """
        Compute distances between query embeddings and class prototypes.
        
        Args:
            query_embeddings: Embedded query examples [n_query, embedding_dim]
            prototypes: Class prototypes [n_classes, embedding_dim]
            
        Returns:
            Negative squared Euclidean distances [n_query, n_classes]
        """
        n_queries = tf.shape(query_embeddings)[0]
        n_classes = tf.shape(prototypes)[0]
        
        # Reshape to enable broadcasting
        queries = tf.expand_dims(query_embeddings, 1)  # [n_query, 1, embedding_dim]
        protos = tf.expand_dims(prototypes, 0)  # [1, n_classes, embedding_dim]
        
        # Compute negative squared Euclidean distance
        return -tf.reduce_sum(tf.square(queries - protos), axis=2)
    
    def train_step(self, support_batch: Tuple[tf.Tensor, tf.Tensor], 
                  query_batch: Tuple[tf.Tensor, tf.Tensor]) -> float:
        """
        Perform one training step of Prototypical Networks.
        
        Args:
            support_batch: Support set data (x, y)
            query_batch: Query set data (x, y)
            
        Returns:
            Loss value
        """
        support_x, support_y = support_batch
        query_x, query_y = query_batch
        
        with tf.GradientTape() as tape:
            # Compute prototypes from support set
            prototypes = self.compute_prototypes(support_x, support_y)
            
            # Embed query examples
            query_embeddings = self.model(query_x)
            
            # Compute class probabilities
            logits = self.compute_distances(query_embeddings, prototypes)
            log_p_y = tf.nn.log_softmax(logits, axis=-1)
            
            # Compute loss
            loss = -tf.reduce_mean(tf.reduce_sum(query_y * log_p_y, axis=1))
            
        # Update model
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss.numpy()
    
    def train(self, episode_generator: Callable[[], Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]],
             episodes: int, log_interval: int = 100) -> List[float]:
        """
        Train the Prototypical Network.
        
        Args:
            episode_generator: Function that generates episodes (support and query sets)
            episodes: Number of episodes to train for
            log_interval: How often to log progress
            
        Returns:
            List of loss values
        """
        losses = []
        
        for episode in range(episodes):
            support_batch, query_batch = episode_generator()
            loss = self.train_step(support_batch, query_batch)
            losses.append(loss)
            
            if (episode + 1) % log_interval == 0:
                print(f"Episode {episode+1}/{episodes}, Loss: {loss:.4f}")
                
        return losses
    
    def predict(self, support_batch: Tuple[tf.Tensor, tf.Tensor], 
               query_x: tf.Tensor) -> tf.Tensor:
        """
        Predict classes for query examples.
        
        Args:
            support_batch: Support set data (x, y)
            query_x: Query examples to classify
            
        Returns:
            Class probabilities for query examples
        """
        support_x, support_y = support_batch
        
        # Compute prototypes from support set
        prototypes = self.compute_prototypes(support_x, support_y)
        
        # Embed query examples
        query_embeddings = self.model(query_x)
        
        # Compute class probabilities
        logits = self.compute_distances(query_embeddings, prototypes)
        return tf.nn.softmax(logits, axis=-1)


class RelationNetwork:
    """
    Few-shot learning with Relation Networks.
    
    Relation Networks learn to compare query examples with support examples
    using a learnable relation module, allowing for more complex comparison
    than simple distance metrics.
    """
    
    def __init__(self, input_shape: Tuple, embedding_dim: int = 64, relation_dim: int = 8):
        """
        Initialize a Relation Network.
        
        Args:
            input_shape: Shape of input data (without batch dimension)
            embedding_dim: Dimension of the embedding space
            relation_dim: Dimension of relation module's hidden layer
        """
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.relation_dim = relation_dim
        
        self.embedding_model = self._build_embedding_model()
        self.relation_model = self._build_relation_model()
        self.optimizer = tf.keras.optimizers.Adam()
        
    def _build_embedding_model(self) -> tf.keras.Model:
        """
        Build the embedding network.
        
        Returns:
            Keras model for embedding inputs
        """
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        
        # Similar architecture as PrototypicalNetwork but with fewer parameters
        # as the relation module will do additional processing
        
        # For time series or 1D data (common in finance)
        if len(self.input_shape) == 2:  # (time_steps, features)
            x = tf.keras.layers.LSTM(32, return_sequences=True)(inputs)
            x = tf.keras.layers.LSTM(32)(x)
            
        # For vector inputs
        else:
            x = tf.keras.layers.Dense(64, activation='relu')(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            
        # Final embedding layer
        outputs = tf.keras.layers.Dense(self.embedding_dim)(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def _build_relation_model(self) -> tf.keras.Model:
        """
        Build the relation module.
        
        Returns:
            Keras model for computing relation scores
        """
        # Input is the concatenation of two embeddings
        inputs = tf.keras.layers.Input(shape=(2 * self.embedding_dim,))
        
        x = tf.keras.layers.Dense(self.relation_dim, activation='relu')(inputs)
        x = tf.keras.layers.Dense(self.relation_dim, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def compute_class_embeddings(self, support_x: tf.Tensor, support_y: tf.Tensor) -> Tuple[tf.Tensor, List[int]]:
        """
        Compute embeddings for each class in the support set.
        
        Args:
            support_x: Support set examples
            support_y: Support set one-hot labels
            
        Returns:
            Tuple of (class_embeddings, class_indices)
        """
        support_embeddings = self.embedding_model(support_x)
        n_classes = support_y.shape[1]
        
        class_embeddings = []
        class_indices = []
        
        for i in range(n_classes):
            # Get indices of examples for this class
            indices = tf.where(support_y[:, i] > 0)[:, 0]
            
            if tf.size(indices) > 0:
                # Get embeddings for this class
                embeddings = tf.gather(support_embeddings, indices)
                class_embeddings.append(embeddings)
                class_indices.append(i)
                
        return class_embeddings, class_indices
    
    def compute_relations(self, query_embeddings: tf.Tensor, 
                         class_embeddings: List[tf.Tensor]) -> tf.Tensor:
        """
        Compute relation scores between query examples and class examples.
        
        Args:
            query_embeddings: Embedded query examples [n_query, embedding_dim]
            class_embeddings: List of embedded support examples per class
            
        Returns:
            Relation scores [n_query, n_classes]
        """
        n_queries = tf.shape(query_embeddings)[0]
        n_classes = len(class_embeddings)
        
        all_relations = []
        
        for query_idx in range(n_queries):
            query_emb = query_embeddings[query_idx]
            query_relations = []
            
            for class_idx in range(n_classes):
                class_emb = class_embeddings[class_idx]
                n_examples = tf.shape(class_emb)[0]
                
                # Repeat query embedding for each class example
                repeated_query = tf.repeat(tf.expand_dims(query_emb, 0), n_examples, axis=0)
                
                # Concatenate query and support embeddings
                pairs = tf.concat([repeated_query, class_emb], axis=1)
                
                # Compute relation scores for all pairs
                relations = self.relation_model(pairs)
                
                # Average relations for this class
                query_relations.append(tf.reduce_mean(relations))
                
            all_relations.append(tf.stack(query_relations))
            
        return tf.stack(all_relations)
    
    def train_step(self, support_batch: Tuple[tf.Tensor, tf.Tensor], 
                  query_batch: Tuple[tf.Tensor, tf.Tensor]) -> float:
        """
        Perform one training step of Relation Network.
        
        Args:
            support_batch: Support set data (x, y)
            query_batch: Query set data (x, y)
            
        Returns:
            Loss value
        """
        support_x, support_y = support_batch
        query_x, query_y = query_batch
        
        with tf.GradientTape() as tape:
            # Embed support and query examples
            query_embeddings = self.embedding_model(query_x)
            
            # Get class embeddings
            class_embeddings, class_indices = self.compute_class_embeddings(support_x, support_y)
            
            # Compute relation scores
            relations = self.compute_relations(query_embeddings, class_embeddings)
            
            # Convert query labels to match class indices
            query_labels = tf.gather(query_y, class_indices, axis=1)
            
            # MSE loss between relation scores and ground truth
            loss = tf.reduce_mean(tf.square(relations - query_labels))
            
        # Update models
        variables = self.embedding_model.trainable_variables + self.relation_model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        return loss.numpy()
    
    def train(self, episode_generator: Callable[[], Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]],
             episodes: int, log_interval: int = 100) -> List[float]:
        """
        Train the Relation Network.
        
        Args:
            episode_generator: Function that generates episodes (support and query sets)
            episodes: Number of episodes to train for
            log_interval: How often to log progress
            
        Returns:
            List of loss values
        """
        losses = []
        
        for episode in range(episodes):
            support_batch, query_batch = episode_generator()
            loss = self.train_step(support_batch, query_batch)
            losses.append(loss)
            
            if (episode + 1) % log_interval == 0:
                print(f"Episode {episode+1}/{episodes}, Loss: {loss:.4f}")
                
        return losses
    
    def predict(self, support_batch: Tuple[tf.Tensor, tf.Tensor], 
               query_x: tf.Tensor) -> Tuple[tf.Tensor, List[int]]:
        """
        Predict classes for query examples.
        
        Args:
            support_batch: Support set data (x, y)
            query_x: Query examples to classify
            
        Returns:
            Tuple of (relation scores, class indices)
        """
        support_x, support_y = support_batch
        
        # Embed query examples
        query_embeddings = self.embedding_model(query_x)
        
        # Get class embeddings
        class_embeddings, class_indices = self.compute_class_embeddings(support_x, support_y)
        
        # Compute relation scores
        relations = self.compute_relations(query_embeddings, class_embeddings)
        
        return relations, class_indices


class MetaOptimizer:
    """
    Meta-optimizer for learning optimization strategies.
    
    This class implements a meta-optimizer that learns how to optimize models,
    adapting the optimization strategy to specific tasks or problems.
    """
    
    def __init__(self, optimizer_model: Optional[tf.keras.Model] = None,
                lstm_units: int = 20, meta_lr: float = 0.001):
        """
        Initialize a meta-optimizer.
        
        Args:
            optimizer_model: Pre-built optimizer model (LSTM-based)
            lstm_units: Number of LSTM units if building a new optimizer model
            meta_lr: Learning rate for meta-optimizer
        """
        self.lstm_units = lstm_units
        self.meta_lr = meta_lr
        
        if optimizer_model is None:
            self.optimizer_model = self._build_optimizer_model()
        else:
            self.optimizer_model = optimizer_model
            
        self.meta_optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr)
        
    def _build_optimizer_model(self) -> tf.keras.Model:
        """
        Build the optimizer model (LSTM-based).
        
        Returns:
            Keras model for generating parameter updates
        """
        # Inputs: gradient, parameter value, learning rate
        inputs = tf.keras.layers.Input(shape=(3,))
        
        # LSTM cell for maintaining optimizer state
        lstm = tf.keras.layers.LSTM(self.lstm_units, return_state=True)
        
        # Initial states (will be maintained between steps)
        h_state = tf.keras.layers.Input(shape=(self.lstm_units,))
        c_state = tf.keras.layers.Input(shape=(self.lstm_units,))
        
        # Generate update from current inputs and previous state
        outputs, h_new, c_new = lstm(tf.expand_dims(inputs, 1), initial_state=[h_state, c_state])
        
        # Final update scaling
        update = tf.keras.layers.Dense(1, activation='tanh')(outputs)
        
        return tf.keras.Model(inputs=[inputs, h_state, c_state], outputs=[update, h_new, c_new])
    
    def _initialize_optimizer_state(self, n_params: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Initialize the optimizer's internal state.
        
        Args:
            n_params: Number of parameters to optimize
            
        Returns:
            Tuple of (h_state, c_state) initial states
        """
        h_state = tf.zeros((n_params, self.lstm_units))
        c_state = tf.zeros((n_params, self.lstm_units))
        return h_state, c_state
    
    def _apply_update(self, params: List[tf.Variable], gradients: List[tf.Tensor], 
                    learning_rate: float = 0.01) -> Tuple[List[tf.Tensor], tf.Tensor, tf.Tensor]:
        """
        Apply meta-optimizer to generate parameter updates.
        
        Args:
            params: List of model parameters
            gradients: List of parameter gradients
            learning_rate: Base learning rate
            
        Returns:
            Tuple of (updated_params, h_state, c_state)
        """
        # Initialize optimizer state if needed
        n_params = sum(tf.size(p) for p in params)
        h_state, c_state = self._initialize_optimizer_state(n_params)
        
        updated_params = []
        all_updates = []
        flat_idx = 0
        
        # Process each parameter tensor
        for param, gradient in zip(params, gradients):
            if gradient is None:
                updated_params.append(param)
                continue
                
            # Flatten parameter and gradient
            flat_param = tf.reshape(param, [-1])
            flat_grad = tf.reshape(gradient, [-1])
            param_size = tf.size(flat_param)
            
            # Prepare inputs for each parameter element
            inputs_list = []
            for i in range(param_size):
                # Input: [gradient, parameter value, learning rate]
                inputs_list.append([flat_grad[i].numpy(), flat_param[i].numpy(), learning_rate])
                
            inputs_array = tf.convert_to_tensor(inputs_list, dtype=tf.float32)
            
            # Get relevant state slices
            h_slice = h_state[flat_idx:flat_idx+param_size]
            c_slice = c_state[flat_idx:flat_idx+param_size]
            
            # Generate updates and new states
            updates, h_new, c_new = self.optimizer_model([inputs_array, h_slice, c_slice])
            
            # Apply updates
            flat_updated = flat_param - tf.squeeze(updates)
            updated_param = tf.reshape(flat_updated, param.shape)
            updated_params.append(updated_param)
            
            # Store updates and update states
            all_updates.append(updates)
            h_state = tf.tensor_scatter_nd_update(
                h_state, 
                tf.range(flat_idx, flat_idx+param_size)[:, tf.newaxis], 
                h_new
            )
            c_state = tf.tensor_scatter_nd_update(
                c_state, 
                tf.range(flat_idx, flat_idx+param_size)[:, tf.newaxis], 
                c_new
            )
            
            flat_idx += param_size
            
        return updated_params, h_state, c_state
    
    def meta_train_step(self, model: tf.keras.Model, train_batch: Tuple[tf.Tensor, tf.Tensor],
                      val_batch: Tuple[tf.Tensor, tf.Tensor], unroll_steps: int = 5) -> float:
        """
        Perform one meta-training step.
        
        Args:
            model: Model to optimize
            train_batch: Training data (x, y) for inner optimization
            val_batch: Validation data (x, y) for meta-update
            unroll_steps: Number of optimization steps to unroll
            
        Returns:
            Meta-loss value
        """
        train_x, train_y = train_batch
        val_x, val_y = val_batch
        
        with tf.GradientTape() as meta_tape:
            # Clone model to avoid modifying the original
            clone_model = tf.keras.models.clone_model(model)
            clone_model.set_weights(model.get_weights())
            
            # Forward pass on validation data (pre-optimization baseline)
            val_pred_before = clone_model(val_x)
            
            # Compute loss function
            if val_y.shape[-1] > 1:  # One-hot encoded labels (classification)
                loss_fn = tf.keras.losses.CategoricalCrossentropy()
            else:  # Regression
                loss_fn = tf.keras.losses.MeanSquaredError()
                
            # Initial validation loss
            val_loss_before = loss_fn(val_y, val_pred_before)
            
            # Unroll optimization for several steps
            for _ in range(unroll_steps):
                with tf.GradientTape() as train_tape:
                    train_pred = clone_model(train_x)
                    train_loss = loss_fn(train_y, train_pred)
                    
                # Compute gradients for model parameters
                gradients = train_tape.gradient(train_loss, clone_model.trainable_variables)
                
                # Apply meta-optimizer to update parameters
                updated_params, _, _ = self._apply_update(
                    clone_model.trainable_variables, 
                    gradients,
                    learning_rate=0.01  # Base learning rate
                )
                
                # Update model weights
                for i, param in enumerate(clone_model.trainable_variables):
                    param.assign(updated_params[i])
            
            # Forward pass on validation data (post-optimization)
            val_pred_after = clone_model(val_x)
            val_loss_after = loss_fn(val_y, val_pred_after)
            
            # Meta-loss is the validation loss after optimization
            meta_loss = val_loss_after
            
        # Compute gradients for meta-optimizer
        meta_gradients = meta_tape.gradient(meta_loss, self.optimizer_model.trainable_variables)
        
        # Apply gradients to meta-optimizer
        self.meta_optimizer.apply_gradients(zip(meta_gradients, self.optimizer_model.trainable_variables))
        
        # Return improvement in validation loss
        improvement = val_loss_before - val_loss_after
        return improvement.numpy()
    
    def meta_train(self, model: tf.keras.Model, 
                 task_generator: Callable[[], Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]],
                 episodes: int, unroll_steps: int = 5, log_interval: int = 10) -> List[float]:
        """
        Train the meta-optimizer.
        
        Args:
            model: Base model architecture to optimize
            task_generator: Function that generates tasks (train and val batches)
            episodes: Number of episodes to train for
            unroll_steps: Number of optimization steps to unroll
            log_interval: How often to log progress
            
        Returns:
            List of improvement values
        """
        improvements = []
        
        for episode in range(episodes):
            train_batch, val_batch = task_generator()
            improvement = self.meta_train_step(model, train_batch, val_batch, unroll_steps)
            improvements.append(improvement)
            
            if (episode + 1) % log_interval == 0:
                print(f"Episode {episode+1}/{episodes}, Improvement: {improvement:.4f}")
                
        return improvements
    
    def optimize(self, model: tf.keras.Model, data_batch: Tuple[tf.Tensor, tf.Tensor],
               steps: int = 10) -> tf.keras.Model:
        """
        Optimize a model using the learned optimization strategy.
        
        Args:
            model: Model to optimize
            data_batch: Data (x, y) for optimization
            steps: Number of optimization steps
            
        Returns:
            Optimized model
        """
        x, y = data_batch
        
        # Clone model to avoid modifying the original
        optimized_model = tf.keras.models.clone_model(model)
        optimized_model.set_weights(model.get_weights())
        
        # Compute loss function
        if y.shape[-1] > 1:  # One-hot encoded labels (classification)
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
        else:  # Regression
            loss_fn = tf.keras.losses.MeanSquaredError()
            
        # Initialize optimizer state
        h_state, c_state = self._initialize_optimizer_state(
            sum(tf.size(p) for p in optimized_model.trainable_variables)
        )
        
        # Perform optimization steps
        for step in range(steps):
            with tf.GradientTape() as tape:
                pred = optimized_model(x)
                loss = loss_fn(y, pred)
                
            # Compute gradients
            gradients = tape.gradient(loss, optimized_model.trainable_variables)
            
            # Apply meta-optimizer to update parameters
            updated_params, h_state, c_state = self._apply_update(
                optimized_model.trainable_variables, 
                gradients,
                learning_rate=0.01  # Base learning rate
            )
            
            # Update model weights
            for i, param in enumerate(optimized_model.trainable_variables):
                param.assign(updated_params[i])
                
            if (step + 1) % 5 == 0:
                print(f"Optimization step {step+1}/{steps}, Loss: {loss:.4f}")
                
        return optimized_model 