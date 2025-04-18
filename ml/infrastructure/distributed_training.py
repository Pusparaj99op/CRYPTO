import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from torch.nn.parallel import DistributedDataParallel as DDP
import socket
import time

logger = logging.getLogger(__name__)

class DistributedTrainer:
    """
    Distributed training implementation for PyTorch models.
    Supports multi-GPU, multi-node training with various distributed strategies.
    """
    
    def __init__(self, 
                num_nodes: int = 1, 
                gpus_per_node: int = 1,
                backend: str = "nccl",
                master_addr: str = "localhost",
                master_port: str = "12355"):
        """
        Initialize the distributed trainer.
        
        Args:
            num_nodes: Number of compute nodes
            gpus_per_node: Number of GPUs per node
            backend: PyTorch distributed backend (nccl, gloo, mpi)
            master_addr: Master node address
            master_port: Master node port
        """
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.world_size = num_nodes * gpus_per_node
        self.backend = backend
        self.master_addr = master_addr
        self.master_port = master_port
        
        # Check if CUDA is available
        self.cuda_available = torch.cuda.is_available()
        if self.gpus_per_node > 0 and not self.cuda_available:
            logger.warning("CUDA is not available but gpus_per_node > 0. Falling back to CPU.")
            self.gpus_per_node = 0
            self.world_size = num_nodes
    
    def _setup_distributed(self, rank: int, world_size: int):
        """
        Set up the distributed environment.
        
        Args:
            rank: Process rank
            world_size: Total number of processes
        """
        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = self.master_port
        
        # Initialize the process group
        dist.init_process_group(backend=self.backend, rank=rank, world_size=world_size)
        
        # Set device for this process
        if self.cuda_available:
            local_rank = rank % self.gpus_per_node
            torch.cuda.set_device(local_rank)
            
        logger.info(f"Initialized process group with rank {rank}/{world_size-1}")
    
    def _cleanup_distributed(self):
        """Clean up the distributed environment."""
        if dist.is_initialized():
            dist.destroy_process_group()
    
    def _train_process(self, 
                      rank: int, 
                      world_size: int, 
                      model_fn: Callable[[], torch.nn.Module],
                      optimizer_fn: Callable[[torch.nn.Module], torch.optim.Optimizer],
                      train_fn: Callable[[torch.nn.Module, torch.optim.Optimizer, int, int], Dict[str, Any]],
                      node_id: int = 0):
        """
        Function to run in each distributed process.
        
        Args:
            rank: Global process rank
            world_size: Total number of processes
            model_fn: Function that returns the model
            optimizer_fn: Function that returns the optimizer
            train_fn: Function that performs the training
            node_id: ID of the current node
        """
        # Calculate the global rank
        global_rank = node_id * self.gpus_per_node + rank
        
        # Set up the distributed environment
        self._setup_distributed(global_rank, world_size)
        
        try:
            # Create model
            model = model_fn()
            
            # Move model to device
            if self.cuda_available:
                model = model.cuda()
            
            # Wrap model with DDP
            if world_size > 1:
                model = DDP(model, device_ids=[rank % self.gpus_per_node] if self.cuda_available else None)
            
            # Create optimizer
            optimizer = optimizer_fn(model)
            
            # Train the model
            results = train_fn(model, optimizer, global_rank, world_size)
            
            return results
        finally:
            # Clean up
            self._cleanup_distributed()
    
    def train(self, 
             model_fn: Callable[[], torch.nn.Module],
             optimizer_fn: Callable[[torch.nn.Module], torch.optim.Optimizer],
             train_fn: Callable[[torch.nn.Module, torch.optim.Optimizer, int, int], Dict[str, Any]],
             node_id: int = 0) -> Dict[str, Any]:
        """
        Train a model in a distributed setting.
        
        Args:
            model_fn: Function that returns the model
            optimizer_fn: Function that returns the optimizer
            train_fn: Function that performs the training
            node_id: ID of the current node
            
        Returns:
            Dictionary of training results
        """
        if self.world_size == 1:
            # Single process case
            return self._train_process(0, 1, model_fn, optimizer_fn, train_fn, node_id)
        
        # Multi-process case
        if self.cuda_available:
            # Use torch.multiprocessing for CUDA
            torch_mp.spawn(
                self._train_process,
                args=(self.world_size, model_fn, optimizer_fn, train_fn, node_id),
                nprocs=self.gpus_per_node,
                join=True
            )
        else:
            # Use standard multiprocessing for CPU
            processes = []
            for rank in range(self.gpus_per_node):
                p = mp.Process(
                    target=self._train_process,
                    args=(rank, self.world_size, model_fn, optimizer_fn, train_fn, node_id)
                )
                p.start()
                processes.append(p)
                
            for p in processes:
                p.join()
        
        return {"status": "completed"}


class DataParallelTrainer:
    """
    Data parallel training implementation for CPU and single-node multi-GPU setups.
    Simpler alternative to DistributedTrainer for single-node setups.
    """
    
    def __init__(self, num_workers: int = None, device: str = "auto"):
        """
        Initialize the data parallel trainer.
        
        Args:
            num_workers: Number of workers (defaults to CPU count)
            device: Device to use ("cpu", "cuda", or "auto")
        """
        self.num_workers = num_workers or mp.cpu_count()
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is not available, falling back to CPU.")
            self.device = "cpu"
            
        # Determine GPU count if using CUDA
        self.gpu_count = torch.cuda.device_count() if self.device == "cuda" else 0
        
        logger.info(f"Initialized DataParallelTrainer with {self.num_workers} workers on {self.device}")
        if self.device == "cuda":
            logger.info(f"Using {self.gpu_count} GPUs")
    
    def _train_batch(self, 
                    model: torch.nn.Module, 
                    batch_data: Any, 
                    device: torch.device) -> Dict[str, Any]:
        """
        Process a single batch of data.
        
        Args:
            model: The model to train
            batch_data: Batch of data to process
            device: Device to use
            
        Returns:
            Dictionary of batch results
        """
        # This is a placeholder for batch training logic
        # In a real implementation, this would process the batch and return metrics
        
        # Move data to device
        if isinstance(batch_data, torch.Tensor):
            batch_data = batch_data.to(device)
        elif isinstance(batch_data, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in batch_data):
            batch_data = [x.to(device) for x in batch_data]
        
        # Forward pass
        outputs = model(batch_data)
        
        # Compute loss and other metrics
        # ...
        
        return {"loss": outputs.mean().item()}
    
    def _gpu_worker(self, 
                  worker_id: int, 
                  model: torch.nn.Module,
                  batches: List[Any],
                  results_queue: mp.Queue):
        """
        Worker function for GPU parallel processing.
        
        Args:
            worker_id: ID of the worker
            model: Model to train
            batches: List of batches to process
            results_queue: Queue to put results in
        """
        device = torch.device(f"cuda:{worker_id % self.gpu_count}")
        
        # Move model to device
        model = model.to(device)
        
        # Process batches
        results = []
        for batch in batches:
            batch_result = self._train_batch(model, batch, device)
            results.append(batch_result)
            
        # Put results in queue
        results_queue.put(results)
    
    def _cpu_worker(self, 
                   worker_id: int, 
                   model_fn: Callable[[], torch.nn.Module],
                   batches: List[Any],
                   results_queue: mp.Queue):
        """
        Worker function for CPU parallel processing.
        
        Args:
            worker_id: ID of the worker
            model_fn: Function to create the model
            batches: List of batches to process
            results_queue: Queue to put results in
        """
        # Create model
        model = model_fn()
        device = torch.device("cpu")
        
        # Process batches
        results = []
        for batch in batches:
            batch_result = self._train_batch(model, batch, device)
            results.append(batch_result)
            
        # Put results in queue
        results_queue.put(results)
    
    def train(self, 
             model_fn: Callable[[], torch.nn.Module],
             data_loader: Any,
             batch_size: int = 32,
             epochs: int = 1) -> Dict[str, Any]:
        """
        Train a model using data parallelism.
        
        Args:
            model_fn: Function that returns the model
            data_loader: Data loader or dataset
            batch_size: Batch size
            epochs: Number of epochs
            
        Returns:
            Dictionary of training results
        """
        # Create model for master process
        model = model_fn()
        
        if self.device == "cuda" and self.gpu_count > 1:
            # Multi-GPU setup
            model = torch.nn.DataParallel(model)
        
        # Move model to device
        device = torch.device(self.device)
        model = model.to(device)
        
        # Training loop
        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            
            epoch_results = []
            
            # Process batches
            for batch in data_loader:
                batch_result = self._train_batch(model, batch, device)
                epoch_results.append(batch_result)
            
            # Compute epoch metrics
            epoch_loss = np.mean([r["loss"] for r in epoch_results])
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        return {"status": "completed", "epochs": epochs}


class ParameterServerTrainer:
    """
    Parameter server training implementation.
    Uses a centralized server to aggregate model updates from workers.
    """
    
    def __init__(self, 
                num_workers: int = None,
                aggregation_method: str = "average"):
        """
        Initialize the parameter server trainer.
        
        Args:
            num_workers: Number of worker processes
            aggregation_method: Method to aggregate worker updates
        """
        self.num_workers = num_workers or mp.cpu_count()
        self.aggregation_method = aggregation_method
        
        logger.info(f"Initialized ParameterServerTrainer with {self.num_workers} workers")
    
    def _parameter_server(self, 
                         model: torch.nn.Module,
                         update_queue: mp.Queue,
                         param_queue: mp.Queue,
                         control_queue: mp.Queue):
        """
        Parameter server process.
        
        Args:
            model: Master model
            update_queue: Queue for receiving updates from workers
            param_queue: Queue for sending parameters to workers
            control_queue: Queue for control signals
        """
        logger.info("Parameter server started")
        
        # Get initial model parameters
        parameters = {name: param.data.cpu().numpy() for name, param in model.named_parameters()}
        
        # Send initial parameters to workers
        for _ in range(self.num_workers):
            param_queue.put(parameters)
        
        # Main loop
        active_workers = self.num_workers
        while active_workers > 0:
            # Get update from a worker
            message = update_queue.get()
            
            if message["type"] == "update":
                # Process parameter update
                worker_id = message["worker_id"]
                updates = message["parameters"]
                
                # Apply update to master model
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if self.aggregation_method == "average":
                            # Average the updates
                            update = updates[name]
                            param.data = param.data * 0.9 + torch.tensor(update, device=param.device) * 0.1
                        # Add more aggregation methods as needed
                
                # Send updated parameters back to the worker
                parameters = {name: param.data.cpu().numpy() for name, param in model.named_parameters()}
                param_queue.put(parameters)
                
            elif message["type"] == "done":
                # Worker is done
                active_workers -= 1
        
        # Signal all workers to stop
        for _ in range(self.num_workers):
            control_queue.put({"type": "stop"})
            
        logger.info("Parameter server finished")
    
    def _worker(self, 
               worker_id: int,
               model_fn: Callable[[], torch.nn.Module],
               data_loader: Any,
               update_queue: mp.Queue,
               param_queue: mp.Queue,
               control_queue: mp.Queue):
        """
        Worker process.
        
        Args:
            worker_id: ID of the worker
            model_fn: Function to create the model
            data_loader: Data loader for this worker
            update_queue: Queue for sending updates to the server
            param_queue: Queue for receiving parameters from the server
            control_queue: Queue for control signals
        """
        logger.info(f"Worker {worker_id} started")
        
        # Create local model
        model = model_fn()
        device = torch.device("cpu")  # Workers use CPU in this implementation
        model = model.to(device)
        
        # Get initial parameters from server
        parameters = param_queue.get()
        
        # Set initial parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data = torch.tensor(parameters[name], device=device)
        
        # Training loop
        for batch in data_loader:
            # Check for control messages
            if not control_queue.empty():
                message = control_queue.get()
                if message["type"] == "stop":
                    break
            
            # Process batch
            if isinstance(batch, torch.Tensor):
                batch = batch.to(device)
            elif isinstance(batch, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in batch):
                batch = [x.to(device) for x in batch]
            
            # Forward and backward pass
            # ...
            
            # Send parameter update to server
            parameters = {name: param.data.cpu().numpy() for name, param in model.named_parameters()}
            update_queue.put({
                "type": "update",
                "worker_id": worker_id,
                "parameters": parameters
            })
            
            # Get updated parameters from server
            parameters = param_queue.get()
            
            # Update local model
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.data = torch.tensor(parameters[name], device=device)
        
        # Signal that this worker is done
        update_queue.put({"type": "done", "worker_id": worker_id})
        logger.info(f"Worker {worker_id} finished")
    
    def train(self, 
             model_fn: Callable[[], torch.nn.Module],
             data_loader: Any,
             epochs: int = 1) -> Dict[str, Any]:
        """
        Train a model using the parameter server architecture.
        
        Args:
            model_fn: Function that returns the model
            data_loader: Data loader or dataset
            epochs: Number of epochs
            
        Returns:
            Dictionary of training results
        """
        # Create master model
        master_model = model_fn()
        
        # Create queues for communication
        update_queue = mp.Queue()  # Workers -> Server
        param_queue = mp.Queue()   # Server -> Workers
        control_queue = mp.Queue() # Control signals
        
        # Start parameter server process
        server_process = mp.Process(
            target=self._parameter_server,
            args=(master_model, update_queue, param_queue, control_queue)
        )
        server_process.start()
        
        # Start worker processes
        worker_processes = []
        for worker_id in range(self.num_workers):
            p = mp.Process(
                target=self._worker,
                args=(worker_id, model_fn, data_loader, update_queue, param_queue, control_queue)
            )
            p.start()
            worker_processes.append(p)
        
        # Wait for all processes to finish
        for p in worker_processes:
            p.join()
            
        server_process.join()
        
        return {"status": "completed", "epochs": epochs} 