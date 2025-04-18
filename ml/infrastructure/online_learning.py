import os
import logging
import time
import json
import numpy as np
import pandas as pd
import threading
import queue
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
import pickle
import torch
from collections import deque

logger = logging.getLogger(__name__)

class DataBuffer:
    """
    Buffer for storing recent data for online learning.
    Supports different buffering strategies.
    """
    
    def __init__(self, 
                max_size: int = 10000,
                strategy: str = "fifo"):
        """
        Initialize the data buffer.
        
        Args:
            max_size: Maximum buffer size
            strategy: Buffering strategy ('fifo', 'reservoir', 'importance')
        """
        self.max_size = max_size
        self.strategy = strategy
        
        # Initialize buffer
        if strategy == "fifo" or strategy == "importance":
            self.buffer = deque(maxlen=max_size)
            self.importance_scores = deque(maxlen=max_size) if strategy == "importance" else None
        elif strategy == "reservoir":
            self.buffer = []
            self.count = 0
        else:
            raise ValueError(f"Unknown buffer strategy: {strategy}")
            
        self.total_added = 0
        
    def add(self, item: Any, importance: Optional[float] = None):
        """
        Add an item to the buffer.
        
        Args:
            item: Item to add
            importance: Importance score for the item (used with 'importance' strategy)
        """
        self.total_added += 1
        
        if self.strategy == "fifo":
            self.buffer.append(item)
        elif self.strategy == "importance":
            if importance is None:
                importance = 1.0  # Default importance
            
            if len(self.buffer) < self.max_size:
                # Buffer not full yet, just append
                self.buffer.append(item)
                self.importance_scores.append(importance)
            else:
                # Buffer full, check if this item is more important than the least important
                min_idx = np.argmin(self.importance_scores)
                min_score = self.importance_scores[min_idx]
                
                if importance > min_score:
                    # Replace least important item
                    self.buffer[min_idx] = item
                    self.importance_scores[min_idx] = importance
        elif self.strategy == "reservoir":
            # Reservoir sampling
            if len(self.buffer) < self.max_size:
                # Buffer not full yet, just append
                self.buffer.append(item)
            else:
                # Randomly decide whether to replace an existing item
                import random
                self.count += 1
                j = random.randint(0, self.count)
                if j < self.max_size:
                    self.buffer[j] = item
    
    def get_all(self) -> List[Any]:
        """
        Get all items in the buffer.
        
        Returns:
            List of all items
        """
        return list(self.buffer)
    
    def sample(self, n: int, weighted: bool = False) -> List[Any]:
        """
        Sample n items from the buffer.
        
        Args:
            n: Number of items to sample
            weighted: Whether to use importance weighting (only for 'importance' strategy)
            
        Returns:
            List of sampled items
        """
        if len(self.buffer) == 0:
            return []
            
        n = min(n, len(self.buffer))
        
        if weighted and self.strategy == "importance":
            # Weighted sampling based on importance scores
            weights = np.array(self.importance_scores)
            weights = weights / weights.sum()  # Normalize
            indices = np.random.choice(len(self.buffer), size=n, replace=False, p=weights)
            return [self.buffer[i] for i in indices]
        else:
            # Random sampling
            import random
            return random.sample(list(self.buffer), n)
    
    def clear(self):
        """Clear the buffer."""
        if self.strategy == "fifo" or self.strategy == "importance":
            self.buffer.clear()
            if self.importance_scores:
                self.importance_scores.clear()
        else:  # reservoir
            self.buffer = []
            self.count = 0
            
        self.total_added = 0
        
    def __len__(self) -> int:
        """Get the number of items in the buffer."""
        return len(self.buffer)


class OnlineLearner:
    """
    Base class for online learning algorithms.
    """
    
    def __init__(self, model: Any):
        """
        Initialize the online learner.
        
        Args:
            model: ML model instance
        """
        self.model = model
        self.is_pytorch = isinstance(model, torch.nn.Module)
        
        # Initialize training stats
        self.train_count = 0
        self.last_trained = None
        self.train_history = []
    
    def learn(self, X: Any, y: Any) -> Dict[str, Any]:
        """
        Update the model with new data.
        
        Args:
            X: Features
            y: Labels or targets
            
        Returns:
            Dictionary with training metrics
        """
        raise NotImplementedError("Subclasses must implement learn()")
    
    def predict(self, X: Any) -> Any:
        """
        Make predictions with the current model.
        
        Args:
            X: Features
            
        Returns:
            Model predictions
        """
        if self.is_pytorch:
            # PyTorch model
            self.model.eval()
            with torch.no_grad():
                if not isinstance(X, torch.Tensor):
                    X = torch.tensor(X, dtype=torch.float32)
                return self.model(X).cpu().numpy()
        else:
            # Scikit-learn like model
            return self.model.predict(X)


class SGDOnlineLearner(OnlineLearner):
    """
    Online learner using stochastic gradient descent for incremental updates.
    """
    
    def __init__(self, 
                model: Any,
                optimizer: Optional[Any] = None,
                loss_fn: Optional[Any] = None,
                batch_size: int = 32,
                learning_rate: float = 0.001):
        """
        Initialize the SGD online learner.
        
        Args:
            model: ML model instance
            optimizer: Optimizer instance (for PyTorch models)
            loss_fn: Loss function (for PyTorch models)
            batch_size: Batch size for updates
            learning_rate: Learning rate for optimization
        """
        super().__init__(model)
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        if self.is_pytorch:
            # PyTorch setup
            if optimizer is None:
                self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            else:
                self.optimizer = optimizer
                
            if loss_fn is None:
                # Default to MSE loss
                self.loss_fn = torch.nn.MSELoss()
            else:
                self.loss_fn = loss_fn
        else:
            # Scikit-learn like models
            # Check if the model has a partial_fit method for online learning
            if not hasattr(self.model, 'partial_fit'):
                raise ValueError("Model must have a partial_fit method for online learning")
    
    def learn(self, X: Any, y: Any) -> Dict[str, Any]:
        """
        Update the model with new data using SGD.
        
        Args:
            X: Features
            y: Labels or targets
            
        Returns:
            Dictionary with training metrics
        """
        start_time = time.time()
        
        if self.is_pytorch:
            # PyTorch model update
            self.model.train()
            
            # Convert inputs to tensors if needed
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float32)
                
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            y_pred = self.model(X)
            
            # Compute loss
            loss = self.loss_fn(y_pred, y)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Compute metrics
            metrics = {
                "loss": loss.item(),
            }
            
        else:
            # Scikit-learn model update
            self.model.partial_fit(X, y)
            
            # Try to compute loss if model has a score method
            if hasattr(self.model, 'score'):
                score = self.model.score(X, y)
                metrics = {"score": score}
            else:
                metrics = {}
                
        # Update training stats
        self.train_count += 1
        self.last_trained = datetime.now()
        
        duration = time.time() - start_time
        metrics["duration"] = duration
        
        self.train_history.append({
            "timestamp": self.last_trained.isoformat(),
            "samples": len(X) if hasattr(X, "__len__") else 1,
            "metrics": metrics
        })
        
        return metrics


class StatelessOnlineLearner(OnlineLearner):
    """
    Online learner for stateless algorithms (e.g., naive Bayes, decision trees).
    It maintains its own data buffer and rebuilds the model with accumulated data.
    """
    
    def __init__(self, 
                model_factory: Callable[[], Any],
                max_buffer_size: int = 10000,
                buffer_strategy: str = "fifo"):
        """
        Initialize the stateless online learner.
        
        Args:
            model_factory: Function that creates a new model instance
            max_buffer_size: Maximum buffer size
            buffer_strategy: Buffering strategy
        """
        # Create initial model
        model = model_factory()
        super().__init__(model)
        
        self.model_factory = model_factory
        
        # Create data buffer
        self.buffer = DataBuffer(max_size=max_buffer_size, strategy=buffer_strategy)
        
    def learn(self, X: Any, y: Any) -> Dict[str, Any]:
        """
        Update the model with new data by rebuilding on accumulated data.
        
        Args:
            X: Features
            y: Labels or targets
            
        Returns:
            Dictionary with training metrics
        """
        start_time = time.time()
        
        # Add data to buffer
        for i in range(len(X)):
            self.buffer.add((X[i], y[i]))
        
        # Get all data from buffer
        buffer_data = self.buffer.get_all()
        
        if not buffer_data:
            return {"error": "No data in buffer"}
        
        # Split data back into X and y
        X_buffer, y_buffer = zip(*buffer_data)
        X_buffer = np.array(X_buffer)
        y_buffer = np.array(y_buffer)
        
        # Rebuild model with all data
        self.model = self.model_factory()
        self.model.fit(X_buffer, y_buffer)
        
        # Try to compute score if available
        if hasattr(self.model, 'score'):
            score = self.model.score(X_buffer, y_buffer)
            metrics = {"score": score}
        else:
            metrics = {}
            
        # Update training stats
        self.train_count += 1
        self.last_trained = datetime.now()
        
        duration = time.time() - start_time
        metrics["duration"] = duration
        metrics["buffer_size"] = len(self.buffer)
        
        self.train_history.append({
            "timestamp": self.last_trained.isoformat(),
            "samples": len(X),
            "metrics": metrics
        })
        
        return metrics


class OnlineLearningPipeline:
    """
    Pipeline for continuous online learning.
    """
    
    def __init__(self, 
                learner: OnlineLearner,
                data_buffer: Optional[DataBuffer] = None,
                update_interval: int = 100,
                validate_fn: Optional[Callable[[Any, Any, Any], Dict[str, float]]] = None,
                save_path: Optional[str] = None):
        """
        Initialize the online learning pipeline.
        
        Args:
            learner: Online learner instance
            data_buffer: Optional data buffer (if not provided by the learner)
            update_interval: How many samples to accumulate before updating
            validate_fn: Optional function to validate the model after updates
            save_path: Optional path to save models
        """
        self.learner = learner
        self.data_buffer = data_buffer or getattr(learner, 'buffer', DataBuffer())
        self.update_interval = update_interval
        self.validate_fn = validate_fn
        self.save_path = save_path
        
        # Create save directory if needed
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            
        # Initialize tracking variables
        self.pending_count = 0
        self.update_count = 0
        self.last_update = None
        self.last_validation = None
        self.validation_history = []
        self.running = False
        self.processing_thread = None
        self.data_queue = queue.Queue()
    
    def start(self):
        """Start the online learning pipeline."""
        if self.running:
            return
            
        self.running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Online learning pipeline started")
        
    def stop(self):
        """Stop the online learning pipeline."""
        self.running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
            self.processing_thread = None
            
        logger.info("Online learning pipeline stopped")
    
    def add_data(self, X: Any, y: Any):
        """
        Add data to the pipeline for processing.
        
        Args:
            X: Features
            y: Labels or targets
        """
        self.data_queue.put((X, y))
        self.pending_count += len(X) if hasattr(X, "__len__") else 1
    
    def _processing_loop(self):
        """Background thread for processing data and updating the model."""
        batch_X, batch_y = [], []
        
        while self.running:
            try:
                # Get data from queue with timeout
                X, y = self.data_queue.get(timeout=0.1)
                
                # Add to batch
                if isinstance(X, (list, np.ndarray, torch.Tensor)):
                    # Batch of samples
                    batch_X.extend(X)
                    batch_y.extend(y)
                else:
                    # Single sample
                    batch_X.append(X)
                    batch_y.append(y)
                
                # Check if it's time to update
                if len(batch_X) >= self.update_interval:
                    self._update_model(batch_X, batch_y)
                    batch_X, batch_y = [], []
                
            except queue.Empty:
                # No data in queue, check if we have pending samples to process
                if batch_X:
                    self._update_model(batch_X, batch_y)
                    batch_X, batch_y = [], []
                    
                # Sleep briefly to prevent CPU spinning
                time.sleep(0.01)
    
    def _update_model(self, X: List[Any], y: List[Any]):
        """
        Update the model with a batch of data.
        
        Args:
            X: Batch of features
            y: Batch of labels or targets
        """
        # Convert lists to appropriate format
        if isinstance(X[0], (list, np.ndarray)):
            X = np.array(X)
        if isinstance(y[0], (list, np.ndarray)):
            y = np.array(y)
            
        # Add data to buffer
        for i in range(len(X)):
            self.data_buffer.add((X[i], y[i]))
        
        # Update the model
        metrics = self.learner.learn(X, y)
        
        # Update tracking variables
        self.update_count += 1
        self.last_update = datetime.now()
        self.pending_count -= len(X)
        
        # Run validation if provided
        if self.validate_fn:
            validation_metrics = self._validate_model()
            if validation_metrics:
                metrics.update(validation_metrics)
        
        # Save model if needed
        if self.save_path:
            self._save_model()
            
        # Log update
        logger.info(f"Model updated (#{self.update_count}): {metrics}")
    
    def _validate_model(self) -> Dict[str, float]:
        """
        Validate the current model.
        
        Returns:
            Dictionary of validation metrics
        """
        validation_metrics = self.validate_fn(self.learner.model, self.learner.predict)
        
        self.validation_history.append({
            "timestamp": datetime.now().isoformat(),
            "update": self.update_count,
            "metrics": validation_metrics
        })
        
        self.last_validation = datetime.now()
        
        # Prepend "val_" to metric names for clarity
        return {f"val_{k}": v for k, v in validation_metrics.items()}
    
    def _save_model(self):
        """Save the current model."""
        filename = f"model_{self.update_count:04d}_{int(time.time())}.pkl"
        filepath = os.path.join(self.save_path, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.learner.model, f)
                
            # Save a "latest" pointer
            with open(os.path.join(self.save_path, "latest_model.txt"), 'w') as f:
                f.write(filename)
                
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the pipeline.
        
        Returns:
            Dictionary with status information
        """
        return {
            "running": self.running,
            "update_count": self.update_count,
            "pending_count": self.pending_count,
            "total_samples_processed": self.learner.train_count,
            "buffer_size": len(self.data_buffer),
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "last_validation": self.last_validation.isoformat() if self.last_validation else None,
            "latest_validation": self.validation_history[-1] if self.validation_history else None
        }


class ConceptDriftDetector:
    """
    Detector for concept drift in data streams.
    """
    
    def __init__(self, 
                window_size: int = 100,
                detection_method: str = "adwin",
                significance_level: float = 0.05):
        """
        Initialize the concept drift detector.
        
        Args:
            window_size: Window size for drift detection
            detection_method: Detection method ('adwin', 'page_hinkley', 'ddm')
            significance_level: Statistical significance level
        """
        self.window_size = window_size
        self.detection_method = detection_method
        self.significance_level = significance_level
        
        # Initialize detection algorithm
        if detection_method == "adwin":
            try:
                from skmultiflow.drift_detection import ADWIN
                self.detector = ADWIN(delta=significance_level)
            except ImportError:
                raise ImportError("scikit-multiflow is required for ADWIN detector")
        elif detection_method == "page_hinkley":
            try:
                from skmultiflow.drift_detection import PageHinkley
                self.detector = PageHinkley(min_instances=window_size, delta=0.005, 
                                           threshold=significance_level * 100, alpha=0.9999)
            except ImportError:
                raise ImportError("scikit-multiflow is required for PageHinkley detector")
        elif detection_method == "ddm":
            try:
                from skmultiflow.drift_detection import DDM
                self.detector = DDM(warning_level=significance_level*2, out_control_level=significance_level)
            except ImportError:
                raise ImportError("scikit-multiflow is required for DDM detector")
        else:
            raise ValueError(f"Unknown drift detection method: {detection_method}")
            
        # Track drift events
        self.drift_detected = False
        self.warning_detected = False
        self.drift_count = 0
        self.warning_count = 0
        self.last_drift = None
        self.drift_history = []
        
    def update(self, error: float) -> bool:
        """
        Update the detector with a new error measurement.
        
        Args:
            error: Error measurement (e.g., prediction error or loss)
            
        Returns:
            True if drift detected, False otherwise
        """
        # Reset flags
        self.drift_detected = False
        self.warning_detected = False
        
        # Update detector
        try:
            self.detector.add_element(error)
            
            # Check for drift or warning
            if hasattr(self.detector, 'detected_change'):
                self.drift_detected = self.detector.detected_change()
            elif hasattr(self.detector, 'detected_warning_zone'):
                self.warning_detected = self.detector.detected_warning_zone()
                self.drift_detected = self.detector.detected_change()
        except Exception as e:
            logger.error(f"Error updating drift detector: {str(e)}")
            return False
        
        # Record drift event
        if self.drift_detected:
            self.drift_count += 1
            self.last_drift = datetime.now()
            
            self.drift_history.append({
                "timestamp": self.last_drift.isoformat(),
                "error": error
            })
            
            logger.warning(f"Concept drift detected (#{self.drift_count})")
        
        # Record warning
        if self.warning_detected:
            self.warning_count += 1
            logger.info(f"Concept drift warning (#{self.warning_count})")
            
        return self.drift_detected
    
    def reset(self):
        """Reset the detector."""
        if self.detection_method == "adwin":
            try:
                from skmultiflow.drift_detection import ADWIN
                self.detector = ADWIN(delta=self.significance_level)
            except ImportError:
                raise ImportError("scikit-multiflow is required for ADWIN detector")
        elif self.detection_method == "page_hinkley":
            try:
                from skmultiflow.drift_detection import PageHinkley
                self.detector = PageHinkley(min_instances=self.window_size, delta=0.005, 
                                           threshold=self.significance_level * 100, alpha=0.9999)
            except ImportError:
                raise ImportError("scikit-multiflow is required for PageHinkley detector")
        elif self.detection_method == "ddm":
            try:
                from skmultiflow.drift_detection import DDM
                self.detector = DDM(warning_level=self.significance_level*2, out_control_level=self.significance_level)
            except ImportError:
                raise ImportError("scikit-multiflow is required for DDM detector")
                
        logger.info("Drift detector reset")
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the detector.
        
        Returns:
            Dictionary with status information
        """
        return {
            "drift_detected": self.drift_detected,
            "warning_detected": self.warning_detected,
            "drift_count": self.drift_count,
            "warning_count": self.warning_count,
            "last_drift": self.last_drift.isoformat() if self.last_drift else None,
            "detection_method": self.detection_method
        } 