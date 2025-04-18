import os
import logging
import numpy as np
import time
import random
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import json

# Optional imports for specialized optimization techniques
try:
    from skopt import Optimizer
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)

class HyperparameterSpace:
    """
    Defines a space of hyperparameters for optimization.
    """
    
    def __init__(self):
        """Initialize an empty hyperparameter space."""
        self.params = {}
        
    def add_real(self, name: str, low: float, high: float, log_scale: bool = False):
        """
        Add a real-valued hyperparameter.
        
        Args:
            name: Name of the hyperparameter
            low: Lower bound
            high: Upper bound
            log_scale: Whether to sample in log scale
        """
        self.params[name] = {
            'type': 'real',
            'low': low,
            'high': high,
            'log_scale': log_scale
        }
        return self
    
    def add_integer(self, name: str, low: int, high: int):
        """
        Add an integer-valued hyperparameter.
        
        Args:
            name: Name of the hyperparameter
            low: Lower bound (inclusive)
            high: Upper bound (inclusive)
        """
        self.params[name] = {
            'type': 'integer',
            'low': low,
            'high': high
        }
        return self
    
    def add_categorical(self, name: str, choices: List[Any]):
        """
        Add a categorical hyperparameter.
        
        Args:
            name: Name of the hyperparameter
            choices: List of possible values
        """
        self.params[name] = {
            'type': 'categorical',
            'choices': choices
        }
        return self
    
    def sample(self) -> Dict[str, Any]:
        """
        Randomly sample a point from the hyperparameter space.
        
        Returns:
            Dictionary of sampled hyperparameter values
        """
        result = {}
        for name, param in self.params.items():
            if param['type'] == 'real':
                if param['log_scale']:
                    # Log-uniform sampling
                    log_low = np.log(param['low'])
                    log_high = np.log(param['high'])
                    value = np.exp(random.uniform(log_low, log_high))
                else:
                    # Uniform sampling
                    value = random.uniform(param['low'], param['high'])
                result[name] = value
            elif param['type'] == 'integer':
                result[name] = random.randint(param['low'], param['high'])
            elif param['type'] == 'categorical':
                result[name] = random.choice(param['choices'])
        return result
    
    def to_skopt_space(self):
        """
        Convert to scikit-optimize space format.
        
        Returns:
            List of skopt space dimensions
        """
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is not installed. Install it with: pip install scikit-optimize")
            
        space = []
        for name, param in self.params.items():
            if param['type'] == 'real':
                space.append(Real(param['low'], param['high'], 
                                  name=name, prior='log-uniform' if param['log_scale'] else 'uniform'))
            elif param['type'] == 'integer':
                space.append(Integer(param['low'], param['high'], name=name))
            elif param['type'] == 'categorical':
                space.append(Categorical(param['choices'], name=name))
        return space
    
    def to_optuna_space(self, trial: 'optuna.Trial') -> Dict[str, Any]:
        """
        Sample from the hyperparameter space using an Optuna trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of sampled hyperparameter values
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Install it with: pip install optuna")
            
        result = {}
        for name, param in self.params.items():
            if param['type'] == 'real':
                if param['log_scale']:
                    result[name] = trial.suggest_loguniform(name, param['low'], param['high'])
                else:
                    result[name] = trial.suggest_uniform(name, param['low'], param['high'])
            elif param['type'] == 'integer':
                result[name] = trial.suggest_int(name, param['low'], param['high'])
            elif param['type'] == 'categorical':
                result[name] = trial.suggest_categorical(name, param['choices'])
        return result


class RandomSearch:
    """
    Random hyperparameter search implementation.
    """
    
    def __init__(self, 
                param_space: HyperparameterSpace,
                n_trials: int = 10,
                random_state: Optional[int] = None):
        """
        Initialize random search.
        
        Args:
            param_space: Hyperparameter space to search
            n_trials: Number of random trials
            random_state: Optional random seed for reproducibility
        """
        self.param_space = param_space
        self.n_trials = n_trials
        
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
    
    def _evaluate_trial(self, 
                       trial_id: int, 
                       objective_fn: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        """
        Evaluate a single trial.
        
        Args:
            trial_id: ID of the trial
            objective_fn: Function that computes the objective value
            
        Returns:
            Dictionary with trial results
        """
        # Sample parameters
        params = self.param_space.sample()
        
        # Evaluate objective
        start_time = time.time()
        try:
            obj_value = objective_fn(params)
            status = "completed"
        except Exception as e:
            obj_value = float('inf')  # Worst possible value
            status = f"failed: {str(e)}"
            logger.warning(f"Trial {trial_id} failed: {str(e)}")
        
        duration = time.time() - start_time
        
        return {
            "trial_id": trial_id,
            "params": params,
            "objective": obj_value,
            "status": status,
            "duration": duration
        }
    
    def optimize(self, 
                objective_fn: Callable[[Dict[str, Any]], float],
                n_jobs: int = 1) -> Dict[str, Any]:
        """
        Run the random search optimization.
        
        Args:
            objective_fn: Function that computes the objective value
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        all_trials = []
        
        if n_jobs > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = [executor.submit(self._evaluate_trial, i, objective_fn) 
                          for i in range(self.n_trials)]
                
                for future in futures:
                    trial = future.result()
                    all_trials.append(trial)
        else:
            # Sequential execution
            for i in range(self.n_trials):
                trial = self._evaluate_trial(i, objective_fn)
                all_trials.append(trial)
                logger.info(f"Trial {i+1}/{self.n_trials}, Objective: {trial['objective']:.6f}")
        
        # Sort trials by objective value (assuming lower is better)
        all_trials.sort(key=lambda x: x["objective"])
        
        # Extract best parameters and value
        best_trial = all_trials[0]
        best_params = best_trial["params"]
        best_value = best_trial["objective"]
        
        total_time = time.time() - start_time
        
        results = {
            "best_params": best_params,
            "best_value": best_value,
            "all_trials": all_trials,
            "total_time": total_time,
            "n_trials": self.n_trials
        }
        
        logger.info(f"Random search completed in {total_time:.2f}s")
        logger.info(f"Best value: {best_value:.6f}")
        logger.info(f"Best parameters: {best_params}")
        
        return results


class GridSearch:
    """
    Grid search implementation for hyperparameter optimization.
    """
    
    def __init__(self, param_grid: Dict[str, List[Any]]):
        """
        Initialize grid search.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
        """
        self.param_grid = param_grid
        
        # Calculate total number of combinations
        self.n_combinations = 1
        for values in param_grid.values():
            self.n_combinations *= len(values)
            
        logger.info(f"Grid search initialized with {self.n_combinations} combinations")
    
    def _generate_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate all combinations of parameters.
        
        Returns:
            List of parameter dictionaries
        """
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        combinations = []
        
        # Helper function for recursive combination generation
        def _generate_recursive(curr_params, depth):
            if depth == len(param_names):
                combinations.append(curr_params.copy())
                return
                
            param_name = param_names[depth]
            for value in param_values[depth]:
                curr_params[param_name] = value
                _generate_recursive(curr_params, depth + 1)
        
        _generate_recursive({}, 0)
        return combinations
    
    def optimize(self, 
                objective_fn: Callable[[Dict[str, Any]], float],
                n_jobs: int = 1) -> Dict[str, Any]:
        """
        Run the grid search optimization.
        
        Args:
            objective_fn: Function that computes the objective value
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        # Generate all combinations
        combinations = self._generate_combinations()
        
        all_trials = []
        
        def _evaluate_trial(trial_id, params):
            """Evaluate a single parameter combination"""
            start_time = time.time()
            try:
                obj_value = objective_fn(params)
                status = "completed"
            except Exception as e:
                obj_value = float('inf')
                status = f"failed: {str(e)}"
                logger.warning(f"Trial {trial_id} failed: {str(e)}")
            
            duration = time.time() - start_time
            
            return {
                "trial_id": trial_id,
                "params": params,
                "objective": obj_value,
                "status": status,
                "duration": duration
            }
        
        if n_jobs > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = [executor.submit(_evaluate_trial, i, params) 
                          for i, params in enumerate(combinations)]
                
                for future in futures:
                    trial = future.result()
                    all_trials.append(trial)
        else:
            # Sequential execution
            for i, params in enumerate(combinations):
                trial = _evaluate_trial(i, params)
                all_trials.append(trial)
                logger.info(f"Trial {i+1}/{len(combinations)}, Objective: {trial['objective']:.6f}")
        
        # Sort trials by objective value (assuming lower is better)
        all_trials.sort(key=lambda x: x["objective"])
        
        # Extract best parameters and value
        best_trial = all_trials[0]
        best_params = best_trial["params"]
        best_value = best_trial["objective"]
        
        total_time = time.time() - start_time
        
        results = {
            "best_params": best_params,
            "best_value": best_value,
            "all_trials": all_trials,
            "total_time": total_time,
            "n_trials": len(combinations)
        }
        
        logger.info(f"Grid search completed in {total_time:.2f}s")
        logger.info(f"Best value: {best_value:.6f}")
        logger.info(f"Best parameters: {best_params}")
        
        return results


class BayesianOptimization:
    """
    Bayesian optimization implementation using scikit-optimize.
    """
    
    def __init__(self, 
                param_space: HyperparameterSpace,
                n_initial_points: int = 10,
                n_trials: int = 50,
                random_state: Optional[int] = None):
        """
        Initialize Bayesian optimization.
        
        Args:
            param_space: Hyperparameter space to search
            n_initial_points: Number of initial random points
            n_trials: Total number of trials
            random_state: Optional random seed for reproducibility
        """
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is not installed. Install it with: pip install scikit-optimize")
            
        self.param_space = param_space
        self.n_initial_points = n_initial_points
        self.n_trials = n_trials
        self.random_state = random_state
        
        # Convert to skopt space
        self.skopt_space = param_space.to_skopt_space()
        
        # Initialize optimizer
        self.optimizer = Optimizer(
            dimensions=self.skopt_space,
            random_state=random_state,
            base_estimator="GP",  # Gaussian Process
            n_initial_points=n_initial_points,
            acq_func="EI"  # Expected Improvement
        )
        
        # Store parameter names
        self.param_names = [dim.name for dim in self.skopt_space]
    
    def _params_list_to_dict(self, params_list: List[Any]) -> Dict[str, Any]:
        """Convert list of parameters to dictionary"""
        return {name: value for name, value in zip(self.param_names, params_list)}
    
    def _params_dict_to_list(self, params_dict: Dict[str, Any]) -> List[Any]:
        """Convert dictionary of parameters to list"""
        return [params_dict[name] for name in self.param_names]
    
    def optimize(self, 
                objective_fn: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        """
        Run the Bayesian optimization.
        
        Args:
            objective_fn: Function that computes the objective value
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        all_trials = []
        
        def _objective_wrapper(params_list):
            """Wrapper for the objective function to convert list to dict"""
            params_dict = self._params_list_to_dict(params_list)
            return objective_fn(params_dict)
        
        # Run optimization
        for i in range(self.n_trials):
            # Ask for next point
            params_list = self.optimizer.ask()
            params_dict = self._params_list_to_dict(params_list)
            
            # Evaluate objective
            trial_start_time = time.time()
            try:
                obj_value = objective_fn(params_dict)
                status = "completed"
            except Exception as e:
                obj_value = float('inf')
                status = f"failed: {str(e)}"
                logger.warning(f"Trial {i+1} failed: {str(e)}")
            
            # Tell optimizer the result
            self.optimizer.tell(params_list, obj_value)
            
            duration = time.time() - trial_start_time
            
            # Record trial
            trial = {
                "trial_id": i,
                "params": params_dict,
                "objective": obj_value,
                "status": status,
                "duration": duration
            }
            all_trials.append(trial)
            
            logger.info(f"Trial {i+1}/{self.n_trials}, Objective: {obj_value:.6f}")
        
        # Get best parameters
        best_params_list = self.optimizer.Xi[np.argmin(self.optimizer.yi)]
        best_params = self._params_list_to_dict(best_params_list)
        best_value = np.min(self.optimizer.yi)
        
        total_time = time.time() - start_time
        
        results = {
            "best_params": best_params,
            "best_value": best_value,
            "all_trials": all_trials,
            "total_time": total_time,
            "n_trials": self.n_trials
        }
        
        logger.info(f"Bayesian optimization completed in {total_time:.2f}s")
        logger.info(f"Best value: {best_value:.6f}")
        logger.info(f"Best parameters: {best_params}")
        
        return results


class OptunaOptimization:
    """
    Hyperparameter optimization using Optuna.
    """
    
    def __init__(self, 
                param_space: HyperparameterSpace,
                n_trials: int = 100,
                study_name: Optional[str] = None,
                storage: Optional[str] = None,
                random_state: Optional[int] = None):
        """
        Initialize Optuna optimization.
        
        Args:
            param_space: Hyperparameter space to search
            n_trials: Number of trials
            study_name: Optional name for the study
            storage: Optional storage URL for the study
            random_state: Optional random seed for reproducibility
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Install it with: pip install optuna")
            
        self.param_space = param_space
        self.n_trials = n_trials
        self.study_name = study_name or f"study_{int(time.time())}"
        self.storage = storage
        self.random_state = random_state
    
    def optimize(self, 
                objective_fn: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        """
        Run the Optuna optimization.
        
        Args:
            objective_fn: Function that computes the objective value
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        # Create a study object
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=sampler,
            direction="minimize",
            load_if_exists=True
        )
        
        # Define the objective function for Optuna
        def _optuna_objective(trial):
            params = self.param_space.to_optuna_space(trial)
            try:
                return objective_fn(params)
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return float('inf')
        
        # Run optimization
        study.optimize(_optuna_objective, n_trials=self.n_trials)
        
        # Get best parameters and value
        best_params = study.best_params
        best_value = study.best_value
        
        # Prepare trial information
        all_trials = []
        for i, trial in enumerate(study.trials):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                status = "completed"
            elif trial.state == optuna.trial.TrialState.PRUNED:
                status = "pruned"
            elif trial.state == optuna.trial.TrialState.FAIL:
                status = "failed"
            else:
                status = str(trial.state)
            
            all_trials.append({
                "trial_id": i,
                "params": trial.params,
                "objective": trial.value if trial.value is not None else float('inf'),
                "status": status,
                "duration": trial.duration.total_seconds() if trial.duration else None
            })
        
        total_time = time.time() - start_time
        
        results = {
            "best_params": best_params,
            "best_value": best_value,
            "all_trials": all_trials,
            "total_time": total_time,
            "n_trials": self.n_trials,
            "study_name": self.study_name
        }
        
        logger.info(f"Optuna optimization completed in {total_time:.2f}s")
        logger.info(f"Best value: {best_value:.6f}")
        logger.info(f"Best parameters: {best_params}")
        
        return results


class HyperparameterOptimizationResult:
    """
    Class for storing and analyzing hyperparameter optimization results.
    """
    
    def __init__(self, results: Dict[str, Any]):
        """
        Initialize with optimization results.
        
        Args:
            results: Dictionary of optimization results
        """
        self.results = results
        self.best_params = results["best_params"]
        self.best_value = results["best_value"]
        self.all_trials = results["all_trials"]
        self.total_time = results["total_time"]
        self.n_trials = results["n_trials"]
    
    def save(self, filepath: str):
        """
        Save results to a JSON file.
        
        Args:
            filepath: Path to save the file
        """
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'HyperparameterOptimizationResult':
        """
        Load results from a JSON file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            HyperparameterOptimizationResult object
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
        return cls(results)
    
    def plot_objective_history(self, ax=None, show_best=True):
        """
        Plot the history of objective values.
        
        Args:
            ax: Optional matplotlib axis
            show_best: Whether to highlight the best value
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract trial data
        trial_ids = [t["trial_id"] for t in self.all_trials if t["status"] == "completed"]
        objectives = [t["objective"] for t in self.all_trials if t["status"] == "completed"]
        
        # Plot values
        ax.plot(trial_ids, objectives, 'o-', alpha=0.6)
        
        # Plot running minimum if showing best
        if show_best and objectives:
            running_min = np.minimum.accumulate(objectives)
            ax.step(trial_ids, running_min, 'r-', where='post', 
                   label=f'Best value: {self.best_value:.6f}')
            ax.legend()
        
        ax.set_xlabel('Trial')
        ax.set_ylabel('Objective Value')
        ax.set_title('Hyperparameter Optimization Progress')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_parallel_coordinates(self, ax=None, top_n=10):
        """
        Plot parallel coordinates of the top_n trials.
        
        Args:
            ax: Optional matplotlib axis
            top_n: Number of top trials to include
            
        Returns:
            Matplotlib axis
        """
        try:
            import pandas as pd
            from pandas.plotting import parallel_coordinates
        except ImportError:
            logger.warning("pandas is required for parallel coordinates plot")
            return None
        
        # Get completed trials
        completed_trials = [t for t in self.all_trials if t["status"] == "completed"]
        
        # Sort by objective value
        completed_trials.sort(key=lambda x: x["objective"])
        
        # Take top_n trials
        top_trials = completed_trials[:min(top_n, len(completed_trials))]
        
        # Convert to DataFrame
        data = []
        for i, trial in enumerate(top_trials):
            row = {"trial": i, "objective": trial["objective"]}
            row.update(trial["params"])
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        parallel_coordinates(df, 'trial', ax=ax)
        ax.set_title(f'Parallel Coordinates Plot of Top {len(top_trials)} Trials')
        
        return ax 