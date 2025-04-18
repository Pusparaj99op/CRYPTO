import os
import logging
import time
import json
import threading
import queue
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import numpy as np
import torch
from datetime import datetime
import uuid
import http.server
import socketserver
import pickle
import base64
from io import BytesIO
import traceback

logger = logging.getLogger(__name__)

class ModelPredictor:
    """
    Base class for model serving that handles prediction requests.
    """
    
    def __init__(self, model: Any, preprocessor: Optional[Callable] = None, postprocessor: Optional[Callable] = None):
        """
        Initialize the model predictor.
        
        Args:
            model: ML model instance
            preprocessor: Optional function to preprocess input data
            postprocessor: Optional function to postprocess model output
        """
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        
    def predict(self, data: Any) -> Any:
        """
        Make a prediction with the model.
        
        Args:
            data: Input data
            
        Returns:
            Model predictions
        """
        # Preprocess if needed
        if self.preprocessor is not None:
            data = self.preprocessor(data)
            
        # Make prediction
        start_time = time.time()
        
        if isinstance(self.model, torch.nn.Module):
            # PyTorch model
            with torch.no_grad():
                if not isinstance(data, torch.Tensor):
                    # Convert to tensor if not already
                    data = torch.tensor(data, dtype=torch.float32)
                predictions = self.model(data)
                
                # Convert tensor to numpy for consistency
                if isinstance(predictions, torch.Tensor):
                    predictions = predictions.cpu().numpy()
        else:
            # Assume scikit-learn like model with predict method
            predictions = self.model.predict(data)
        
        latency = time.time() - start_time
        
        # Postprocess if needed
        if self.postprocessor is not None:
            predictions = self.postprocessor(predictions)
            
        return {
            "predictions": predictions,
            "latency": latency
        }
    
    def predict_proba(self, data: Any) -> Any:
        """
        Make a probability prediction with the model, if supported.
        
        Args:
            data: Input data
            
        Returns:
            Model probability predictions
        """
        # Preprocess if needed
        if self.preprocessor is not None:
            data = self.preprocessor(data)
            
        # Make prediction
        start_time = time.time()
        
        if isinstance(self.model, torch.nn.Module):
            # PyTorch model - add softmax for classification
            with torch.no_grad():
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data, dtype=torch.float32)
                logits = self.model(data)
                proba = torch.nn.functional.softmax(logits, dim=1)
                
                # Convert tensor to numpy for consistency
                if isinstance(proba, torch.Tensor):
                    proba = proba.cpu().numpy()
        else:
            # Assume scikit-learn like model with predict_proba method
            try:
                proba = self.model.predict_proba(data)
            except (AttributeError, NotImplementedError):
                raise NotImplementedError("Model does not support probability predictions")
        
        latency = time.time() - start_time
        
        # Postprocess if needed
        if self.postprocessor is not None:
            proba = self.postprocessor(proba)
            
        return {
            "probabilities": proba,
            "latency": latency
        }
        

class PredictionService:
    """
    Service that handles prediction requests with monitoring and batching.
    """
    
    def __init__(self, 
                predictor: ModelPredictor,
                batch_size: Optional[int] = None,
                max_batch_wait_time: float = 0.1,
                logger: Optional[logging.Logger] = None):
        """
        Initialize the prediction service.
        
        Args:
            predictor: ModelPredictor instance
            batch_size: Optional batch size for batching requests
            max_batch_wait_time: Maximum time to wait for batch to fill
            logger: Optional logger instance
        """
        self.predictor = predictor
        self.batch_size = batch_size
        self.max_batch_wait_time = max_batch_wait_time
        self.logger = logger or logging.getLogger(__name__)
        
        self.request_queue = queue.Queue()
        self.running = False
        self.batch_thread = None
        self.metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "total_latency": 0,
            "requests_per_minute": {},
            "latency_history": []
        }
        
    def start(self):
        """Start the prediction service."""
        if self.running:
            return
            
        self.running = True
        
        if self.batch_size is not None:
            # Start batch processing thread
            self.batch_thread = threading.Thread(target=self._batch_processor)
            self.batch_thread.daemon = True
            self.batch_thread.start()
            
        self.logger.info("Prediction service started")
            
    def stop(self):
        """Stop the prediction service."""
        self.running = False
        
        if self.batch_thread:
            self.batch_thread.join(timeout=5.0)
            self.batch_thread = None
            
        self.logger.info("Prediction service stopped")
    
    def predict(self, data: Any) -> Any:
        """
        Make a prediction with batching support.
        
        Args:
            data: Input data
            
        Returns:
            Model predictions
        """
        self.metrics["total_requests"] += 1
        
        timestamp = datetime.now()
        minute_key = timestamp.strftime("%Y-%m-%d %H:%M")
        self.metrics["requests_per_minute"][minute_key] = \
            self.metrics["requests_per_minute"].get(minute_key, 0) + 1
        
        if self.batch_size is None:
            # No batching, make prediction directly
            try:
                result = self.predictor.predict(data)
                latency = result["latency"]
                self.metrics["total_latency"] += latency
                self.metrics["latency_history"].append(latency)
                return result
            except Exception as e:
                self.metrics["total_errors"] += 1
                self.logger.error(f"Prediction error: {str(e)}")
                raise
        else:
            # Use batching via queue
            response_queue = queue.Queue()
            request_id = str(uuid.uuid4())
            
            self.request_queue.put((request_id, data, response_queue))
            
            # Wait for result
            try:
                result = response_queue.get(timeout=10.0)  # 10 second timeout
                if isinstance(result, Exception):
                    self.metrics["total_errors"] += 1
                    raise result
                return result
            except queue.Empty:
                self.metrics["total_errors"] += 1
                raise TimeoutError("Prediction request timed out")
    
    def _batch_processor(self):
        """Background thread for batch processing."""
        while self.running:
            batch_requests = []
            batch_response_queues = {}
            
            # Try to fill a batch
            start_time = time.time()
            while len(batch_requests) < self.batch_size and time.time() - start_time < self.max_batch_wait_time:
                try:
                    request_id, data, response_queue = self.request_queue.get(timeout=self.max_batch_wait_time)
                    batch_requests.append(data)
                    batch_response_queues[len(batch_requests) - 1] = (request_id, response_queue)
                except queue.Empty:
                    break
            
            if not batch_requests:
                # No requests, sleep briefly
                time.sleep(0.01)
                continue
                
            # Process the batch
            try:
                if isinstance(batch_requests[0], np.ndarray):
                    # Numpy arrays
                    batch_data = np.vstack(batch_requests)
                elif isinstance(batch_requests[0], torch.Tensor):
                    # PyTorch tensors
                    batch_data = torch.vstack(batch_requests)
                else:
                    # Just use a list
                    batch_data = batch_requests
                    
                # Make batch prediction
                batch_result = self.predictor.predict(batch_data)
                
                # Distribute results
                for i, (request_id, response_queue) in batch_response_queues.items():
                    result = {
                        "predictions": batch_result["predictions"][i],
                        "latency": batch_result["latency"]
                    }
                    response_queue.put(result)
                    
                # Update metrics
                latency = batch_result["latency"]
                self.metrics["total_latency"] += latency
                self.metrics["latency_history"].append(latency)
                
            except Exception as e:
                # Handle errors
                self.logger.error(f"Batch prediction error: {str(e)}")
                for _, response_queue in batch_response_queues.values():
                    response_queue.put(e)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get service metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = self.metrics.copy()
        
        # Calculate derived metrics
        if metrics["total_requests"] > 0:
            metrics["avg_latency"] = metrics["total_latency"] / metrics["total_requests"]
        else:
            metrics["avg_latency"] = 0
            
        if metrics["latency_history"]:
            metrics["latency_p50"] = np.percentile(metrics["latency_history"], 50)
            metrics["latency_p95"] = np.percentile(metrics["latency_history"], 95)
            metrics["latency_p99"] = np.percentile(metrics["latency_history"], 99)
        
        return metrics


class ModelServer:
    """
    HTTP server for model serving.
    """
    
    def __init__(self, 
                prediction_service: PredictionService,
                host: str = "localhost",
                port: int = 8000):
        """
        Initialize the model server.
        
        Args:
            prediction_service: PredictionService instance
            host: Host address to bind
            port: Port to bind
        """
        self.prediction_service = prediction_service
        self.host = host
        self.port = port
        self.httpd = None
        
        # Create request handler
        self._create_handler()
        
    def _create_handler(self):
        """Create HTTP request handler class."""
        prediction_service = self.prediction_service
        
        class PredictionHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/predict":
                    self._handle_predict()
                elif self.path == "/metrics":
                    self._handle_metrics()
                else:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b"Not found")
                    
            def _handle_predict(self):
                # Get content length
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                try:
                    # Parse request data
                    request = json.loads(post_data.decode('utf-8'))
                    
                    if "data" not in request:
                        raise ValueError("Missing 'data' field in request")
                        
                    # Handle different data formats
                    data = request["data"]
                    
                    if "format" in request:
                        if request["format"] == "numpy":
                            # Base64 encoded numpy array
                            data = pickle.loads(base64.b64decode(data))
                        elif request["format"] == "tensor":
                            # Serialized torch tensor
                            buffer = BytesIO(base64.b64decode(data))
                            data = torch.load(buffer)
                    
                    # Make prediction
                    if request.get("probability", False) and hasattr(prediction_service.predictor, "predict_proba"):
                        result = prediction_service.predictor.predict_proba(data)
                    else:
                        result = prediction_service.predict(data)
                        
                    # Convert numpy arrays to lists for JSON serialization
                    if "predictions" in result and isinstance(result["predictions"], np.ndarray):
                        result["predictions"] = result["predictions"].tolist()
                    if "probabilities" in result and isinstance(result["probabilities"], np.ndarray):
                        result["probabilities"] = result["probabilities"].tolist()
                        
                    # Send response
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode('utf-8'))
                    
                except Exception as e:
                    # Handle errors
                    error_message = str(e)
                    stack_trace = traceback.format_exc()
                    
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "error": error_message,
                        "stack_trace": stack_trace
                    }).encode('utf-8'))
            
            def _handle_metrics(self):
                """Handle metrics request."""
                metrics = prediction_service.get_metrics()
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(metrics).encode('utf-8'))
                
            def log_message(self, format, *args):
                """Override to use our logger."""
                prediction_service.logger.info("%s - %s", self.client_address[0], format % args)
        
        self.handler_class = PredictionHandler
        
    def start(self):
        """Start the model server."""
        # Start prediction service
        self.prediction_service.start()
        
        # Create and start HTTP server
        self.httpd = socketserver.ThreadingTCPServer((self.host, self.port), self.handler_class)
        server_thread = threading.Thread(target=self.httpd.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        logger.info(f"Model server started at http://{self.host}:{self.port}")
        
        return server_thread
        
    def stop(self):
        """Stop the model server."""
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
            
        # Stop prediction service
        self.prediction_service.stop()
        
        logger.info("Model server stopped")


class ModelClient:
    """
    Client for interacting with ModelServer.
    """
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        """
        Initialize the model client.
        
        Args:
            server_url: URL of the model server
        """
        self.server_url = server_url.rstrip('/')
        
    def predict(self, 
               data: Any, 
               probability: bool = False, 
               timeout: float = 10.0) -> Dict[str, Any]:
        """
        Make a prediction request.
        
        Args:
            data: Input data
            probability: Whether to get probability predictions
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary with prediction results
        """
        import requests
        
        # Prepare request data
        request_data = {"data": data, "probability": probability}
        
        # Handle numpy arrays and torch tensors
        if isinstance(data, np.ndarray):
            buffer = BytesIO()
            pickle.dump(data, buffer)
            request_data["data"] = base64.b64encode(buffer.getvalue()).decode('utf-8')
            request_data["format"] = "numpy"
        elif isinstance(data, torch.Tensor):
            buffer = BytesIO()
            torch.save(data, buffer)
            request_data["data"] = base64.b64encode(buffer.getvalue()).decode('utf-8')
            request_data["format"] = "tensor"
        
        # Make request
        response = requests.post(
            f"{self.server_url}/predict",
            json=request_data,
            timeout=timeout
        )
        
        if response.status_code != 200:
            # Handle error
            error_data = response.json()
            raise RuntimeError(f"Prediction failed: {error_data['error']}")
            
        return response.json()
    
    def get_metrics(self, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Get server metrics.
        
        Args:
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary of metrics
        """
        import requests
        
        response = requests.post(
            f"{self.server_url}/metrics",
            timeout=timeout
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get metrics: {response.text}")
            
        return response.json() 