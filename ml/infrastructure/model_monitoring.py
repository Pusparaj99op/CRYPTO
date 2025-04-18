import os
import logging
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
import threading
import matplotlib.pyplot as plt
from collections import deque
import pickle
import warnings
from scipy import stats

logger = logging.getLogger(__name__)

class MetricTracker:
    """
    Tracks model metrics over time, with statistical analysis capabilities.
    """
    
    def __init__(self, 
                window_size: int = 1000,
                alert_threshold: float = 2.0):
        """
        Initialize the metric tracker.
        
        Args:
            window_size: Maximum number of values to store in the window
            alert_threshold: Z-score threshold for anomaly detection
        """
        self.metrics = {}
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        
    def add_metric(self, 
                  name: str, 
                  value: float, 
                  timestamp: Optional[datetime] = None):
        """
        Add a metric value to the tracker.
        
        Args:
            name: Name of the metric
            value: Metric value
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        if name not in self.metrics:
            self.metrics[name] = {
                "values": deque(maxlen=self.window_size),
                "timestamps": deque(maxlen=self.window_size),
                "alerts": []
            }
            
        self.metrics[name]["values"].append(value)
        self.metrics[name]["timestamps"].append(timestamp)
        
        # Check for anomalies if we have enough data
        if len(self.metrics[name]["values"]) >= 30:
            self._check_anomaly(name, value, timestamp)
            
    def _check_anomaly(self, 
                      name: str, 
                      value: float, 
                      timestamp: datetime):
        """
        Check if a metric value is anomalous.
        
        Args:
            name: Name of the metric
            value: Metric value
            timestamp: Timestamp for the value
        """
        values = list(self.metrics[name]["values"])
        
        # Calculate z-score
        mean = np.mean(values[:-1])  # Exclude the latest value
        std = np.std(values[:-1])
        
        if std == 0:
            return  # Avoid division by zero
            
        z_score = abs((value - mean) / std)
        
        # Check if anomalous
        if z_score > self.alert_threshold:
            alert = {
                "timestamp": timestamp,
                "value": value,
                "z_score": z_score,
                "mean": mean,
                "std": std
            }
            
            self.metrics[name]["alerts"].append(alert)
            logger.warning(f"Metric anomaly detected for {name}: value={value:.4f}, z-score={z_score:.2f}")
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all metrics data.
        
        Returns:
            Dictionary of metrics data
        """
        result = {}
        
        for name, data in self.metrics.items():
            values = list(data["values"])
            
            if not values:
                continue
                
            timestamps = [t.isoformat() for t in data["timestamps"]]
            
            stats_dict = {
                "current": values[-1],
                "mean": np.mean(values),
                "median": np.median(values),
                "min": np.min(values),
                "max": np.max(values),
                "std": np.std(values),
                "values": values,
                "timestamps": timestamps,
                "alerts": data["alerts"]
            }
            
            result[name] = stats_dict
            
        return result
    
    def get_metric(self, name: str) -> Dict[str, Any]:
        """
        Get data for a specific metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            Dictionary of metric data
        """
        if name not in self.metrics:
            return {}
            
        data = self.metrics[name]
        values = list(data["values"])
        
        if not values:
            return {}
            
        timestamps = [t.isoformat() for t in data["timestamps"]]
        
        return {
            "current": values[-1],
            "mean": np.mean(values),
            "median": np.median(values),
            "min": np.min(values),
            "max": np.max(values),
            "std": np.std(values),
            "values": values,
            "timestamps": timestamps,
            "alerts": data["alerts"]
        }
    
    def plot_metric(self, 
                   name: str, 
                   figsize: Tuple[int, int] = (10, 6),
                   show_alerts: bool = True) -> plt.Figure:
        """
        Plot a metric's value over time.
        
        Args:
            name: Name of the metric
            figsize: Figure size
            show_alerts: Whether to highlight alert points
            
        Returns:
            Matplotlib figure
        """
        if name not in self.metrics:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"No data for metric '{name}'", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
            
        data = self.metrics[name]
        values = list(data["values"])
        timestamps = list(data["timestamps"])
        
        if not values:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"No data points for metric '{name}'", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot metric values
        ax.plot(timestamps, values, 'b-', label=name)
        
        # Plot alerts if enabled
        if show_alerts and data["alerts"]:
            alert_times = [a["timestamp"] for a in data["alerts"]]
            alert_values = [a["value"] for a in data["alerts"]]
            ax.plot(alert_times, alert_values, 'ro', label='Alerts')
            
        ax.set_title(f"Metric: {name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.autofmt_xdate()  # Rotate date labels
        
        return fig
        

class ModelMonitor:
    """
    Monitors model performance and drift.
    """
    
    def __init__(self, 
                window_size: int = 1000,
                alert_threshold: float = 2.0,
                feature_drift_method: str = "ks",
                save_dir: Optional[str] = None):
        """
        Initialize the model monitor.
        
        Args:
            window_size: Window size for metrics
            alert_threshold: Threshold for alerts
            feature_drift_method: Method for feature drift detection ('ks' or 'chi2')
            save_dir: Directory to save monitoring reports
        """
        self.metric_tracker = MetricTracker(
            window_size=window_size,
            alert_threshold=alert_threshold
        )
        
        self.feature_drift_method = feature_drift_method
        self.save_dir = save_dir
        
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        # Store reference distributions
        self.reference_features = {}
        self.reference_targets = None
        self.prediction_history = deque(maxlen=window_size)
        
        # Initialize drift statistics
        self.drift_statistics = {
            "feature_drift": {},
            "target_drift": {},
            "prediction_drift": {}
        }
    
    def set_reference_data(self, 
                          features: pd.DataFrame, 
                          targets: Optional[pd.Series] = None):
        """
        Set reference data for drift detection.
        
        Args:
            features: Reference feature distribution
            targets: Reference target distribution
        """
        # Store column-wise reference distributions
        for col in features.columns:
            self.reference_features[col] = features[col].values
            
        if targets is not None:
            self.reference_targets = targets.values
            
        logger.info(f"Reference data set with {len(features)} samples, {len(features.columns)} features")
    
    def log_prediction(self, 
                      features: pd.DataFrame, 
                      prediction: Any, 
                      true_value: Optional[Any] = None,
                      metrics: Optional[Dict[str, float]] = None):
        """
        Log a prediction event.
        
        Args:
            features: Feature values
            prediction: Model prediction
            true_value: Optional ground truth
            metrics: Optional performance metrics
        """
        timestamp = datetime.now()
        
        # Store prediction details
        prediction_info = {
            "timestamp": timestamp,
            "prediction": prediction,
            "true_value": true_value,
            "features": features
        }
        
        self.prediction_history.append(prediction_info)
        
        # Log metrics if provided
        if metrics:
            for name, value in metrics.items():
                self.metric_tracker.add_metric(name, value, timestamp)
                
        # Check for feature drift
        self._check_feature_drift(features)
        
        # Check for prediction drift if we have enough data
        if len(self.prediction_history) >= 30:
            self._check_prediction_drift()
            
        # Check for target drift if ground truth is provided
        if true_value is not None and self.reference_targets is not None:
            self._check_target_drift(true_value)
    
    def _check_feature_drift(self, features: pd.DataFrame):
        """
        Check for drift in feature distributions.
        
        Args:
            features: Current feature values
        """
        if not self.reference_features:
            return  # No reference data available
            
        for col in features.columns:
            if col not in self.reference_features:
                continue
                
            current_value = features[col].values
            reference = self.reference_features[col]
            
            # Skip if only one value
            if len(current_value) < 2:
                continue
                
            # Perform statistical test
            if self.feature_drift_method == "ks":
                # Kolmogorov-Smirnov test for continuous features
                try:
                    stat, p_value = stats.ks_2samp(reference, current_value)
                    
                    drift_info = {
                        "timestamp": datetime.now(),
                        "feature": col,
                        "statistic": stat,
                        "p_value": p_value,
                        "drift_detected": p_value < 0.05
                    }
                    
                    if p_value < 0.05:
                        logger.warning(f"Feature drift detected for {col}: p-value={p_value:.4f}")
                    
                    self.drift_statistics["feature_drift"][col] = drift_info
                    
                except Exception as e:
                    logger.error(f"Error computing KS test for {col}: {str(e)}")
                    
            elif self.feature_drift_method == "chi2":
                # Chi-squared test for categorical features
                try:
                    # Convert to categorical if needed
                    ref_counts = pd.Series(reference).value_counts()
                    curr_counts = pd.Series(current_value).value_counts()
                    
                    # Ensure same categories
                    all_cats = set(ref_counts.index) | set(curr_counts.index)
                    ref_counts = ref_counts.reindex(all_cats, fill_value=0)
                    curr_counts = curr_counts.reindex(all_cats, fill_value=0)
                    
                    stat, p_value = stats.chisquare(curr_counts, ref_counts)
                    
                    drift_info = {
                        "timestamp": datetime.now(),
                        "feature": col,
                        "statistic": stat,
                        "p_value": p_value,
                        "drift_detected": p_value < 0.05
                    }
                    
                    if p_value < 0.05:
                        logger.warning(f"Feature drift detected for {col}: p-value={p_value:.4f}")
                    
                    self.drift_statistics["feature_drift"][col] = drift_info
                    
                except Exception as e:
                    logger.error(f"Error computing Chi-squared test for {col}: {str(e)}")
    
    def _check_prediction_drift(self):
        """Check for drift in model predictions."""
        if len(self.prediction_history) < 30:
            return  # Not enough data
            
        # Get first half as reference, second half as current
        n = len(self.prediction_history)
        half = n // 2
        
        reference_preds = [p["prediction"] for p in list(self.prediction_history)[:half]]
        current_preds = [p["prediction"] for p in list(self.prediction_history)[half:]]
        
        # Convert to arrays
        reference_preds = np.array(reference_preds)
        current_preds = np.array(current_preds)
        
        # Perform KS test
        try:
            stat, p_value = stats.ks_2samp(reference_preds, current_preds)
            
            drift_info = {
                "timestamp": datetime.now(),
                "statistic": stat,
                "p_value": p_value,
                "drift_detected": p_value < 0.05
            }
            
            if p_value < 0.05:
                logger.warning(f"Prediction drift detected: p-value={p_value:.4f}")
            
            self.drift_statistics["prediction_drift"] = drift_info
            
        except Exception as e:
            logger.error(f"Error computing prediction drift: {str(e)}")
    
    def _check_target_drift(self, true_value: Any):
        """
        Check for drift in target distribution.
        
        Args:
            true_value: Current ground truth value
        """
        if self.reference_targets is None or len(self.reference_targets) < 30:
            return  # No reference data available
            
        # Append to recent targets
        recent_targets = [p["true_value"] for p in self.prediction_history 
                         if p["true_value"] is not None]
        
        if len(recent_targets) < 30:
            return  # Not enough data
            
        recent_targets = np.array(recent_targets)
        
        # Perform KS test
        try:
            stat, p_value = stats.ks_2samp(self.reference_targets, recent_targets)
            
            drift_info = {
                "timestamp": datetime.now(),
                "statistic": stat,
                "p_value": p_value,
                "drift_detected": p_value < 0.05
            }
            
            if p_value < 0.05:
                logger.warning(f"Target drift detected: p-value={p_value:.4f}")
            
            self.drift_statistics["target_drift"] = drift_info
            
        except Exception as e:
            logger.error(f"Error computing target drift: {str(e)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all monitoring metrics.
        
        Returns:
            Dictionary of monitoring metrics
        """
        metrics = self.metric_tracker.get_metrics()
        
        monitoring_data = {
            "metrics": metrics,
            "drift_statistics": self.drift_statistics,
            "prediction_count": len(self.prediction_history),
            "last_updated": datetime.now().isoformat()
        }
        
        return monitoring_data
    
    def get_alerts(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all alerts.
        
        Returns:
            Dictionary of alerts by metric
        """
        metrics = self.metric_tracker.get_metrics()
        
        alerts = {}
        for name, data in metrics.items():
            if "alerts" in data and data["alerts"]:
                alerts[name] = data["alerts"]
                
        drift_alerts = {}
        
        # Add feature drift alerts
        for feature, data in self.drift_statistics["feature_drift"].items():
            if data.get("drift_detected", False):
                if "feature_drift" not in drift_alerts:
                    drift_alerts["feature_drift"] = []
                drift_alerts["feature_drift"].append(data)
                
        # Add prediction drift alerts
        if self.drift_statistics["prediction_drift"].get("drift_detected", False):
            drift_alerts["prediction_drift"] = [self.drift_statistics["prediction_drift"]]
            
        # Add target drift alerts
        if self.drift_statistics["target_drift"].get("drift_detected", False):
            drift_alerts["target_drift"] = [self.drift_statistics["target_drift"]]
            
        alerts["drift"] = drift_alerts
        
        return alerts
    
    def generate_report(self, 
                       report_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a monitoring report.
        
        Args:
            report_path: Optional path to save report
            
        Returns:
            Dictionary with report data
        """
        metrics = self.metric_tracker.get_metrics()
        alerts = self.get_alerts()
        
        report = {
            "metrics": metrics,
            "alerts": alerts,
            "drift_statistics": self.drift_statistics,
            "prediction_count": len(self.prediction_history),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save report if path provided
        if report_path:
            try:
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Monitoring report saved to {report_path}")
            except Exception as e:
                logger.error(f"Error saving report: {str(e)}")
                
        # Or save to default directory if set
        elif self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(self.save_dir, f"monitoring_report_{timestamp}.json")
            
            try:
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Monitoring report saved to {report_path}")
            except Exception as e:
                logger.error(f"Error saving report: {str(e)}")
        
        return report


class PerformanceMonitor:
    """
    Specialized monitor for model performance metrics.
    """
    
    def __init__(self, 
                metrics: List[str],
                window_size: int = 1000,
                alert_threshold: float = 2.0,
                baseline_performance: Optional[Dict[str, float]] = None):
        """
        Initialize the performance monitor.
        
        Args:
            metrics: List of metrics to track
            window_size: Window size for metrics
            alert_threshold: Threshold for alerts
            baseline_performance: Optional baseline metrics
        """
        self.metrics = metrics
        self.tracker = MetricTracker(
            window_size=window_size,
            alert_threshold=alert_threshold
        )
        self.baseline_performance = baseline_performance or {}
        
        # Initialize performance history
        self.performance_history = deque(maxlen=window_size)
        self.current_performance = {}
        
    def log_metric(self, 
                  name: str, 
                  value: float, 
                  timestamp: Optional[datetime] = None):
        """
        Log a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            timestamp: Optional timestamp
        """
        if name not in self.metrics:
            # Add to metrics if not already tracked
            self.metrics.append(name)
            
        self.tracker.add_metric(name, value, timestamp)
        
        # Update current performance
        self.current_performance[name] = value
        
        # Record in history
        history_entry = {
            "timestamp": timestamp or datetime.now(),
            "metrics": {name: value}
        }
        
        self.performance_history.append(history_entry)
    
    def log_metrics(self, 
                   metrics: Dict[str, float], 
                   timestamp: Optional[datetime] = None):
        """
        Log multiple performance metrics.
        
        Args:
            metrics: Dictionary of metrics
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        for name, value in metrics.items():
            self.log_metric(name, value, timestamp)
    
    def get_performance(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        performance = {}
        
        for metric in self.metrics:
            data = self.tracker.get_metric(metric)
            if data:
                performance[metric] = data
                
                # Add comparison to baseline if available
                if metric in self.baseline_performance:
                    baseline = self.baseline_performance[metric]
                    current = data["current"]
                    delta = current - baseline
                    delta_pct = (delta / baseline) * 100 if baseline != 0 else 0
                    
                    data["baseline"] = baseline
                    data["delta"] = delta
                    data["delta_pct"] = delta_pct
        
        return performance
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """
        Get performance alerts.
        
        Returns:
            List of performance alerts
        """
        alerts = []
        
        metrics_data = self.tracker.get_metrics()
        for name, data in metrics_data.items():
            if "alerts" in data and data["alerts"]:
                for alert in data["alerts"]:
                    alert_info = {
                        "metric": name,
                        "timestamp": alert["timestamp"],
                        "value": alert["value"],
                        "z_score": alert["z_score"],
                        "mean": alert["mean"]
                    }
                    alerts.append(alert_info)
        
        return alerts
    
    def set_baseline(self, metrics: Dict[str, float]):
        """
        Set baseline performance metrics.
        
        Args:
            metrics: Dictionary of baseline metrics
        """
        self.baseline_performance.update(metrics)
    
    def plot_metric_trend(self, 
                         metric: str, 
                         figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot trend for a specific metric.
        
        Args:
            metric: Metric name
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        return self.tracker.plot_metric(metric, figsize)
    
    def plot_performance_comparison(self, 
                                   figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot comparison of current performance vs baseline.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        performance = self.get_performance()
        
        # Filter metrics with baseline
        metrics_with_baseline = [m for m in self.metrics 
                                if m in performance and "baseline" in performance[m]]
        
        if not metrics_with_baseline:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No baseline metrics available for comparison", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        fig, ax = plt.subplots(figsize=figsize)
        
        labels = metrics_with_baseline
        current_values = [performance[m]["current"] for m in metrics_with_baseline]
        baseline_values = [performance[m]["baseline"] for m in metrics_with_baseline]
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, current_values, width, label='Current')
        ax.bar(x + width/2, baseline_values, width, label='Baseline')
        
        ax.set_ylabel('Value')
        ax.set_title('Performance Metrics: Current vs Baseline')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        # Add value labels
        for i, v in enumerate(current_values):
            ax.text(i - width/2, v + 0.01, f"{v:.3f}", ha='center')
            
        for i, v in enumerate(baseline_values):
            ax.text(i + width/2, v + 0.01, f"{v:.3f}", ha='center')
        
        fig.tight_layout()
        
        return fig 