import os
import json
import time
import datetime
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import uuid

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """
    Experiment tracking system for ML experiments.
    Tracks metrics, parameters, artifacts, and results.
    """
    
    def __init__(self, base_dir: str = "./experiments"):
        """
        Initialize the experiment tracker.
        
        Args:
            base_dir: Base directory for storing experiment data
        """
        self.base_dir = base_dir
        self._ensure_dir_exists()
        
    def _ensure_dir_exists(self):
        """Create base directory if it doesn't exist"""
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir, exist_ok=True)
            logger.info(f"Created experiment directory at {self.base_dir}")
    
    def create_experiment(self, 
                         name: str, 
                         description: Optional[str] = None,
                         tags: Optional[List[str]] = None) -> str:
        """
        Create a new experiment.
        
        Args:
            name: Name of the experiment
            description: Optional description
            tags: Optional list of tags
            
        Returns:
            experiment_id: Unique ID for the experiment
        """
        # Generate a unique ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        experiment_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Create experiment directory
        experiment_dir = os.path.join(self.base_dir, experiment_id)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(os.path.join(experiment_dir, "artifacts"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "plots"), exist_ok=True)
        
        # Create metadata file
        metadata = {
            "id": experiment_id,
            "name": name,
            "description": description,
            "tags": tags or [],
            "created_at": datetime.datetime.now().isoformat(),
            "status": "created",
            "completed_at": None,
            "parameters": {}
        }
        
        with open(os.path.join(experiment_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Created experiment '{name}' with ID {experiment_id}")
        return experiment_id
    
    def log_parameters(self, experiment_id: str, parameters: Dict[str, Any]):
        """
        Log parameters for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            parameters: Dictionary of parameters
        """
        experiment_dir = os.path.join(self.base_dir, experiment_id)
        if not os.path.exists(experiment_dir):
            raise ValueError(f"Experiment with ID {experiment_id} not found")
            
        # Update metadata
        metadata_file = os.path.join(experiment_dir, "metadata.json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            
        metadata["parameters"] = parameters
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Logged parameters for experiment {experiment_id}")
    
    def log_metric(self, 
                  experiment_id: str, 
                  metric_name: str, 
                  value: Union[float, int], 
                  step: Optional[int] = None):
        """
        Log a metric value for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            metric_name: Name of the metric
            value: Metric value
            step: Optional step number
        """
        experiment_dir = os.path.join(self.base_dir, experiment_id)
        if not os.path.exists(experiment_dir):
            raise ValueError(f"Experiment with ID {experiment_id} not found")
            
        metrics_dir = os.path.join(experiment_dir, "metrics")
        metric_file = os.path.join(metrics_dir, f"{metric_name}.csv")
        
        # Create or append to metric file
        timestamp = time.time()
        
        if not os.path.exists(metric_file):
            # Create new file with header
            with open(metric_file, "w") as f:
                f.write("timestamp,step,value\n")
                
        # Append metric value
        with open(metric_file, "a") as f:
            f.write(f"{timestamp},{step or 'null'},{value}\n")
            
        logger.debug(f"Logged metric {metric_name}={value} for experiment {experiment_id}")
    
    def log_artifact(self, 
                    experiment_id: str, 
                    artifact_path: str, 
                    artifact_name: Optional[str] = None):
        """
        Log an artifact for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            artifact_path: Path to the artifact file
            artifact_name: Optional name to save the artifact as
        """
        experiment_dir = os.path.join(self.base_dir, experiment_id)
        if not os.path.exists(experiment_dir):
            raise ValueError(f"Experiment with ID {experiment_id} not found")
            
        artifacts_dir = os.path.join(experiment_dir, "artifacts")
        
        # Use original filename if not specified
        if artifact_name is None:
            artifact_name = os.path.basename(artifact_path)
            
        # Copy the artifact
        import shutil
        shutil.copy2(artifact_path, os.path.join(artifacts_dir, artifact_name))
        
        logger.info(f"Logged artifact {artifact_name} for experiment {experiment_id}")
    
    def log_plot(self, 
                experiment_id: str,
                figure: plt.Figure,
                plot_name: str):
        """
        Log a matplotlib figure for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            figure: Matplotlib figure object
            plot_name: Name to save the plot as
        """
        experiment_dir = os.path.join(self.base_dir, experiment_id)
        if not os.path.exists(experiment_dir):
            raise ValueError(f"Experiment with ID {experiment_id} not found")
            
        plots_dir = os.path.join(experiment_dir, "plots")
        
        # Ensure plot name has proper extension
        if not plot_name.endswith(('.png', '.jpg', '.jpeg', '.pdf')):
            plot_name += '.png'
            
        # Save the figure
        figure.savefig(os.path.join(plots_dir, plot_name))
        
        logger.info(f"Logged plot {plot_name} for experiment {experiment_id}")
    
    def complete_experiment(self, 
                           experiment_id: str, 
                           status: str = "completed", 
                           final_metrics: Optional[Dict[str, float]] = None):
        """
        Mark an experiment as completed.
        
        Args:
            experiment_id: ID of the experiment
            status: Final status (completed, failed, etc.)
            final_metrics: Optional dictionary of final metrics
        """
        experiment_dir = os.path.join(self.base_dir, experiment_id)
        if not os.path.exists(experiment_dir):
            raise ValueError(f"Experiment with ID {experiment_id} not found")
            
        # Update metadata
        metadata_file = os.path.join(experiment_dir, "metadata.json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            
        metadata["status"] = status
        metadata["completed_at"] = datetime.datetime.now().isoformat()
        
        if final_metrics:
            metadata["final_metrics"] = final_metrics
            
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Marked experiment {experiment_id} as {status}")
    
    def get_experiment_metadata(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get metadata for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Dictionary of experiment metadata
        """
        experiment_dir = os.path.join(self.base_dir, experiment_id)
        if not os.path.exists(experiment_dir):
            raise ValueError(f"Experiment with ID {experiment_id} not found")
            
        metadata_file = os.path.join(experiment_dir, "metadata.json")
        with open(metadata_file, "r") as f:
            return json.load(f)
    
    def get_experiment_metrics(self, 
                              experiment_id: str, 
                              metric_name: Optional[str] = None) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Get metrics for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            metric_name: Optional name of a specific metric
            
        Returns:
            DataFrame of metrics or dict of DataFrames if metric_name is None
        """
        experiment_dir = os.path.join(self.base_dir, experiment_id)
        if not os.path.exists(experiment_dir):
            raise ValueError(f"Experiment with ID {experiment_id} not found")
            
        metrics_dir = os.path.join(experiment_dir, "metrics")
        
        if metric_name:
            # Return specific metric
            metric_file = os.path.join(metrics_dir, f"{metric_name}.csv")
            if not os.path.exists(metric_file):
                raise ValueError(f"Metric {metric_name} not found for experiment {experiment_id}")
                
            return pd.read_csv(metric_file)
        else:
            # Return all metrics
            metrics = {}
            for file in os.listdir(metrics_dir):
                if file.endswith(".csv"):
                    metric_name = file[:-4]  # Remove .csv extension
                    metrics[metric_name] = pd.read_csv(os.path.join(metrics_dir, file))
                    
            return metrics
    
    def list_experiments(self, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all experiments, optionally filtered by tag.
        
        Args:
            tag: Optional tag to filter by
            
        Returns:
            List of experiment metadata dictionaries
        """
        if not os.path.exists(self.base_dir):
            return []
            
        experiments = []
        for exp_id in os.listdir(self.base_dir):
            metadata_file = os.path.join(self.base_dir, exp_id, "metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    
                if tag is None or tag in metadata.get("tags", []):
                    experiments.append(metadata)
                    
        return experiments
    
    def compare_experiments(self, 
                           experiment_ids: List[str], 
                           metric_name: str) -> pd.DataFrame:
        """
        Compare a specific metric across multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metric_name: Name of the metric to compare
            
        Returns:
            DataFrame with metric values for each experiment
        """
        result = {}
        
        for exp_id in experiment_ids:
            try:
                metadata = self.get_experiment_metadata(exp_id)
                exp_name = metadata.get("name", exp_id)
                
                metrics_df = self.get_experiment_metrics(exp_id, metric_name)
                result[exp_name] = metrics_df["value"].values
                
            except ValueError as e:
                logger.warning(f"Could not get metrics for experiment {exp_id}: {str(e)}")
                
        return pd.DataFrame(result)
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment and all its data.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            True if deletion was successful
        """
        experiment_dir = os.path.join(self.base_dir, experiment_id)
        if not os.path.exists(experiment_dir):
            logger.warning(f"Experiment {experiment_id} not found, nothing to delete")
            return False
            
        import shutil
        shutil.rmtree(experiment_dir)
        logger.info(f"Deleted experiment {experiment_id}")
        return True 