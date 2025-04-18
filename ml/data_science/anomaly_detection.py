"""
Anomaly Detection Module for Cryptocurrency Trading

This module provides tools for detecting anomalies and outliers in
cryptocurrency market data using various statistical and machine learning methods.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional, Callable
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class AnomalyDetector:
    """Class for detecting anomalies in cryptocurrency market data."""
    
    def __init__(self):
        """Initialize the anomaly detector."""
        self.models = {}
        self.anomaly_scores = {}
        self.anomaly_indices = {}
        self.threshold_values = {}
    
    def detect_isolation_forest(self, X: pd.DataFrame, contamination: float = 0.05, 
                              n_estimators: int = 100, max_features: float = 1.0,
                              standardize: bool = True) -> pd.Series:
        """
        Detect anomalies using Isolation Forest algorithm.
        
        Args:
            X: Feature DataFrame
            contamination: Expected proportion of outliers
            n_estimators: Number of base estimators
            max_features: Number of features to draw for each tree
            standardize: Whether to standardize data
            
        Returns:
            Series of anomaly labels (1: normal, -1: anomaly)
        """
        data = X.copy()
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            self.models['scaler'] = scaler
        else:
            data_scaled = data.values
        
        # Train Isolation Forest
        model = IsolationForest(
            n_estimators=n_estimators,
            max_features=max_features,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        model.fit(data_scaled)
        
        # Get anomaly predictions
        labels = model.predict(data_scaled)
        scores = model.decision_function(data_scaled)
        
        # Store results
        self.models['isolation_forest'] = model
        self.anomaly_scores['isolation_forest'] = pd.Series(scores, index=data.index)
        self.anomaly_indices['isolation_forest'] = data.index[labels == -1]
        
        # Return labels as Series
        return pd.Series(labels, index=data.index, name='isolation_forest')
    
    def detect_local_outlier_factor(self, X: pd.DataFrame, contamination: float = 0.05, 
                                  n_neighbors: int = 20, standardize: bool = True) -> pd.Series:
        """
        Detect anomalies using Local Outlier Factor.
        
        Args:
            X: Feature DataFrame
            contamination: Expected proportion of outliers
            n_neighbors: Number of neighbors to consider
            standardize: Whether to standardize data
            
        Returns:
            Series of anomaly labels (1: normal, -1: anomaly)
        """
        data = X.copy()
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            self.models['scaler'] = scaler
        else:
            data_scaled = data.values
        
        # Train LOF
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            n_jobs=-1
        )
        
        # LOF has no predict method, fit_predict directly
        labels = model.fit_predict(data_scaled)
        
        # For LOF, the negative outlier factor is the anomaly score
        # We need to access the private attribute
        scores = -model.negative_outlier_factor_
        
        # Store results
        self.models['lof'] = model
        self.anomaly_scores['lof'] = pd.Series(scores, index=data.index)
        self.anomaly_indices['lof'] = data.index[labels == -1]
        
        # Return labels as Series
        return pd.Series(labels, index=data.index, name='lof')
    
    def detect_elliptic_envelope(self, X: pd.DataFrame, contamination: float = 0.05, 
                               standardize: bool = True) -> pd.Series:
        """
        Detect anomalies using Robust Covariance (Elliptic Envelope).
        
        Args:
            X: Feature DataFrame
            contamination: Expected proportion of outliers
            standardize: Whether to standardize data
            
        Returns:
            Series of anomaly labels (1: normal, -1: anomaly)
        """
        data = X.copy()
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            self.models['scaler'] = scaler
        else:
            data_scaled = data.values
        
        # Train Elliptic Envelope
        model = EllipticEnvelope(
            contamination=contamination,
            random_state=42
        )
        model.fit(data_scaled)
        
        # Get anomaly predictions
        labels = model.predict(data_scaled)
        scores = model.decision_function(data_scaled)
        
        # Store results
        self.models['elliptic_envelope'] = model
        self.anomaly_scores['elliptic_envelope'] = pd.Series(scores, index=data.index)
        self.anomaly_indices['elliptic_envelope'] = data.index[labels == -1]
        
        # Return labels as Series
        return pd.Series(labels, index=data.index, name='elliptic_envelope')
    
    def detect_one_class_svm(self, X: pd.DataFrame, nu: float = 0.05, 
                          kernel: str = 'rbf', gamma: Union[str, float] = 'scale',
                          standardize: bool = True) -> pd.Series:
        """
        Detect anomalies using One-Class SVM.
        
        Args:
            X: Feature DataFrame
            nu: An upper bound on the fraction of training errors (anomalies)
            kernel: Kernel type
            gamma: Kernel coefficient
            standardize: Whether to standardize data
            
        Returns:
            Series of anomaly labels (1: normal, -1: anomaly)
        """
        data = X.copy()
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            self.models['scaler'] = scaler
        else:
            data_scaled = data.values
        
        # Train One-Class SVM
        model = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma
        )
        model.fit(data_scaled)
        
        # Get anomaly predictions
        labels = model.predict(data_scaled)
        scores = model.decision_function(data_scaled)
        
        # Store results
        self.models['one_class_svm'] = model
        self.anomaly_scores['one_class_svm'] = pd.Series(scores, index=data.index)
        self.anomaly_indices['one_class_svm'] = data.index[labels == -1]
        
        # Return labels as Series
        return pd.Series(labels, index=data.index, name='one_class_svm')
    
    def detect_zscore_outliers(self, X: pd.DataFrame, threshold: float = 3.0, 
                            method: str = 'mean') -> pd.DataFrame:
        """
        Detect outliers using Z-score method.
        
        Args:
            X: Feature DataFrame
            threshold: Z-score threshold for outlier detection
            method: 'mean' for standard Z-score, 'median' for modified Z-score
            
        Returns:
            DataFrame with boolean mask of outliers
        """
        data = X.copy()
        
        # Calculate Z-scores
        if method == 'mean':
            # Standard Z-score
            z_scores = pd.DataFrame(
                stats.zscore(data, nan_policy='omit'),
                index=data.index,
                columns=data.columns
            )
        elif method == 'median':
            # Modified Z-score based on MAD
            median = data.median()
            mad = (data - median).abs().median() * 1.4826  # Scaling factor for normal distribution
            z_scores = (data - median) / mad
        else:
            raise ValueError("Method must be 'mean' or 'median'")
        
        # Identify outliers
        outliers = z_scores.abs() > threshold
        
        # Store results
        self.anomaly_scores[f'zscore_{method}'] = z_scores
        self.threshold_values[f'zscore_{method}'] = threshold
        
        # Get indices of rows with at least one outlier
        outlier_indices = outliers.any(axis=1)
        self.anomaly_indices[f'zscore_{method}'] = data.index[outlier_indices]
        
        return outliers
    
    def detect_iqr_outliers(self, X: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers using Interquartile Range method.
        
        Args:
            X: Feature DataFrame
            k: Multiplier for IQR
            
        Returns:
            DataFrame with boolean mask of outliers
        """
        data = X.copy()
        
        # Calculate Q1, Q3, and IQR for each feature
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR
        
        # Identify outliers
        outliers_low = data < lower_bound
        outliers_high = data > upper_bound
        outliers = outliers_low | outliers_high
        
        # Store results
        self.threshold_values['iqr_lower'] = lower_bound
        self.threshold_values['iqr_upper'] = upper_bound
        
        # Get indices of rows with at least one outlier
        outlier_indices = outliers.any(axis=1)
        self.anomaly_indices['iqr'] = data.index[outlier_indices]
        
        return outliers
    
    def detect_mahalanobis_outliers(self, X: pd.DataFrame, threshold: float = None) -> pd.Series:
        """
        Detect outliers using Mahalanobis distance.
        
        Args:
            X: Feature DataFrame
            threshold: Chi-square threshold (default: based on 0.975 quantile)
            
        Returns:
            Series of Mahalanobis distances
        """
        data = X.copy()
        
        # Calculate mean and covariance
        mean = data.mean()
        cov = data.cov()
        
        # Calculate Mahalanobis distance
        inv_cov = np.linalg.inv(cov)
        distances = []
        
        for i, row in data.iterrows():
            x_minus_mean = row - mean
            distance = np.sqrt(x_minus_mean.dot(inv_cov).dot(x_minus_mean))
            distances.append(distance)
        
        # Create Series of distances
        mahalanobis_dist = pd.Series(distances, index=data.index, name='mahalanobis_distance')
        
        # Determine threshold if not provided
        if threshold is None:
            # Use chi-square distribution with p=0.975 and df=number of features
            threshold = stats.chi2.ppf(0.975, df=data.shape[1])
        
        # Store results
        self.anomaly_scores['mahalanobis'] = mahalanobis_dist
        self.threshold_values['mahalanobis'] = threshold
        self.anomaly_indices['mahalanobis'] = data.index[mahalanobis_dist > threshold]
        
        return mahalanobis_dist
    
    def detect_price_anomalies(self, price_series: pd.Series, window_size: int = 20, 
                            n_std: float = 3.0, method: str = 'rolling') -> pd.Series:
        """
        Detect anomalies in price time series data.
        
        Args:
            price_series: Series of prices
            window_size: Size of rolling window
            n_std: Number of standard deviations for threshold
            method: 'rolling' for rolling statistics or 'ewm' for exponential weighted
            
        Returns:
            Series of boolean anomaly indicators
        """
        prices = price_series.copy()
        
        if method == 'rolling':
            # Calculate rolling statistics
            rolling_mean = prices.rolling(window=window_size).mean()
            rolling_std = prices.rolling(window=window_size).std()
            
            # Calculate upper and lower bounds
            upper_bound = rolling_mean + n_std * rolling_std
            lower_bound = rolling_mean - n_std * rolling_std
            
        elif method == 'ewm':
            # Calculate exponential weighted statistics
            ewm_mean = prices.ewm(span=window_size).mean()
            ewm_std = prices.ewm(span=window_size).std()
            
            # Calculate upper and lower bounds
            upper_bound = ewm_mean + n_std * ewm_std
            lower_bound = ewm_mean - n_std * ewm_std
            
        else:
            raise ValueError("Method must be 'rolling' or 'ewm'")
        
        # Identify anomalies
        anomalies = (prices > upper_bound) | (prices < lower_bound)
        
        # Store results
        method_name = f'price_{method}'
        self.threshold_values[f'{method_name}_upper'] = upper_bound
        self.threshold_values[f'{method_name}_lower'] = lower_bound
        self.anomaly_indices[method_name] = prices.index[anomalies]
        
        return anomalies
    
    def detect_volume_anomalies(self, volume_series: pd.Series, window_size: int = 20, 
                              threshold_factor: float = 5.0) -> pd.Series:
        """
        Detect anomalies in trading volume.
        
        Args:
            volume_series: Series of trading volumes
            window_size: Size of rolling window
            threshold_factor: Multiplier for median volume
            
        Returns:
            Series of boolean anomaly indicators
        """
        volume = volume_series.copy()
        
        # Calculate rolling median
        rolling_median = volume.rolling(window=window_size).median()
        
        # Calculate upper bound
        upper_bound = rolling_median * threshold_factor
        
        # Identify anomalies (unusually high volume)
        anomalies = volume > upper_bound
        
        # Store results
        self.threshold_values['volume_upper'] = upper_bound
        self.anomaly_indices['volume'] = volume.index[anomalies]
        
        return anomalies
    
    def detect_volatility_anomalies(self, price_series: pd.Series, window_size: int = 20, 
                                 vol_window: int = 5, n_std: float = 3.0) -> pd.Series:
        """
        Detect anomalous volatility periods.
        
        Args:
            price_series: Series of prices
            window_size: Size of rolling window for volatility baseline
            vol_window: Window for calculating volatility
            n_std: Number of standard deviations for threshold
            
        Returns:
            Series of boolean anomaly indicators
        """
        prices = price_series.copy()
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=vol_window).std() * np.sqrt(252)  # Annualized
        
        # Calculate rolling statistics for volatility
        rolling_mean = volatility.rolling(window=window_size).mean()
        rolling_std = volatility.rolling(window=window_size).std()
        
        # Calculate upper bound
        upper_bound = rolling_mean + n_std * rolling_std
        
        # Identify anomalies (unusually high volatility)
        anomalies = volatility > upper_bound
        
        # Store results
        self.anomaly_scores['volatility'] = volatility
        self.threshold_values['volatility_upper'] = upper_bound
        self.anomaly_indices['volatility'] = volatility.index[anomalies]
        
        return anomalies
    
    def detect_ensemble_anomalies(self, X: pd.DataFrame, methods: List[str] = None, 
                               voting_threshold: float = 0.5) -> pd.Series:
        """
        Detect anomalies using ensemble of multiple methods.
        
        Args:
            X: Feature DataFrame
            methods: List of methods to use in ensemble
            voting_threshold: Proportion of methods required to flag anomaly
            
        Returns:
            Series of boolean anomaly indicators
        """
        # Set default methods if none provided
        if methods is None:
            methods = ['isolation_forest', 'lof', 'one_class_svm']
        
        # Apply each method if not already applied
        results = {}
        for method in methods:
            if method == 'isolation_forest' and 'isolation_forest' not in self.anomaly_indices:
                self.detect_isolation_forest(X)
            elif method == 'lof' and 'lof' not in self.anomaly_indices:
                self.detect_local_outlier_factor(X)
            elif method == 'one_class_svm' and 'one_class_svm' not in self.anomaly_indices:
                self.detect_one_class_svm(X)
            elif method == 'mahalanobis' and 'mahalanobis' not in self.anomaly_indices:
                self.detect_mahalanobis_outliers(X)
            elif method == 'zscore_mean' and 'zscore_mean' not in self.anomaly_indices:
                self.detect_zscore_outliers(X, method='mean')
            elif method == 'zscore_median' and 'zscore_median' not in self.anomaly_indices:
                self.detect_zscore_outliers(X, method='median')
            elif method == 'iqr' and 'iqr' not in self.anomaly_indices:
                self.detect_iqr_outliers(X)
            elif method == 'elliptic_envelope' and 'elliptic_envelope' not in self.anomaly_indices:
                self.detect_elliptic_envelope(X)
        
        # Create votes DataFrame
        votes = pd.DataFrame(index=X.index, columns=methods)
        
        # Fill with votes (1 for anomaly, 0 for normal)
        for method in methods:
            votes[method] = X.index.isin(self.anomaly_indices[method]).astype(int)
        
        # Calculate vote proportion
        vote_proportion = votes.mean(axis=1)
        
        # Identify ensemble anomalies
        ensemble_anomalies = vote_proportion >= voting_threshold
        
        # Store results
        self.anomaly_scores['ensemble'] = vote_proportion
        self.threshold_values['ensemble'] = voting_threshold
        self.anomaly_indices['ensemble'] = X.index[ensemble_anomalies]
        
        return ensemble_anomalies
    
    def plot_anomalies(self, data: pd.Series, method: str, figsize: Tuple = (14, 7), 
                     title: str = None, plot_thresholds: bool = True):
        """
        Plot time series with highlighted anomalies.
        
        Args:
            data: Time series to plot
            method: Anomaly detection method
            figsize: Figure size
            title: Plot title
            plot_thresholds: Whether to plot threshold bounds
            
        Returns:
            matplotlib figure
        """
        if method not in self.anomaly_indices:
            raise ValueError(f"Method '{method}' not found in anomaly indices")
        
        # Get anomaly indices
        anomaly_idx = self.anomaly_indices[method]
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot original data
        plt.plot(data, label='Original data', alpha=0.7)
        
        # Plot anomalies
        plt.scatter(anomaly_idx, data.loc[anomaly_idx], color='r', marker='o', s=50, label='Anomalies')
        
        # Plot threshold bounds if requested and available
        if plot_thresholds:
            if f'{method}_upper' in self.threshold_values:
                plt.plot(self.threshold_values[f'{method}_upper'], 'g--', alpha=0.5, label='Upper bound')
            if f'{method}_lower' in self.threshold_values:
                plt.plot(self.threshold_values[f'{method}_lower'], 'g--', alpha=0.5, label='Lower bound')
        
        # Set title
        if title is None:
            title = f'Anomaly Detection using {method}'
        plt.title(title)
        
        # Add legend and grid
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_anomaly_scores(self, method: str, threshold: float = None,
                         figsize: Tuple = (14, 7), title: str = None):
        """
        Plot anomaly scores with threshold.
        
        Args:
            method: Anomaly detection method
            threshold: Anomaly threshold to plot
            figsize: Figure size
            title: Plot title
            
        Returns:
            matplotlib figure
        """
        if method not in self.anomaly_scores:
            raise ValueError(f"Method '{method}' not found in anomaly scores")
        
        # Get anomaly scores
        scores = self.anomaly_scores[method]
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot scores
        plt.plot(scores, label='Anomaly scores', alpha=0.7)
        
        # Plot threshold if requested
        if threshold is None and method in self.threshold_values:
            threshold = self.threshold_values[method]
            
        if threshold is not None:
            plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label='Threshold')
        
        # Set title
        if title is None:
            title = f'Anomaly Scores for {method}'
        plt.title(title)
        
        # Add legend and grid
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_anomaly_heatmap(self, X: pd.DataFrame, method: str, figsize: Tuple = (12, 10)):
        """
        Plot heatmap of anomalies across features.
        
        Args:
            X: Feature DataFrame
            method: Anomaly detection method (only for methods that return outliers per feature)
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        # Check if method returns feature-wise outliers
        if method not in ['zscore_mean', 'zscore_median', 'iqr']:
            raise ValueError(f"Method '{method}' does not support feature-wise outlier detection")
        
        # Detect outliers if not already done
        if method == 'zscore_mean':
            outliers = self.detect_zscore_outliers(X, method='mean')
        elif method == 'zscore_median':
            outliers = self.detect_zscore_outliers(X, method='median')
        elif method == 'iqr':
            outliers = self.detect_iqr_outliers(X)
        
        # Get only rows with at least one outlier
        outlier_mask = outliers.any(axis=1)
        outliers_subset = outliers[outlier_mask]
        
        # If no outliers found
        if outliers_subset.empty:
            print("No outliers found")
            return None
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(outliers_subset.T, cmap='YlOrRd', cbar_kws={'label': 'Is Outlier'})
        
        # Set title and labels
        plt.title(f'Outlier Detection Heatmap ({method})')
        plt.xlabel('Observations')
        plt.ylabel('Features')
        plt.tight_layout()
        
        return plt.gcf()


# Utility functions

def detect_changes(series: pd.Series, window_size: int = 20, threshold: float = 3.0, 
                 method: str = 'z_score') -> pd.Series:
    """
    Detect significant changes in time series.
    
    Args:
        series: Time series data
        window_size: Window size for baseline calculation
        threshold: Detection threshold
        method: Detection method ('z_score', 'percent_change', 'cusum')
        
    Returns:
        Series of boolean change indicators
    """
    data = series.copy()
    
    if method == 'z_score':
        # Use rolling mean and std to calculate z-scores
        rolling_mean = data.rolling(window=window_size).mean()
        rolling_std = data.rolling(window=window_size).std()
        z_scores = (data - rolling_mean) / rolling_std
        
        # Detect changes
        changes = z_scores.abs() > threshold
        
    elif method == 'percent_change':
        # Calculate percent change relative to previous window
        previous_values = data.shift(1).rolling(window=window_size).mean()
        pct_change = (data - previous_values) / previous_values
        
        # Detect changes
        changes = pct_change.abs() > threshold
        
    elif method == 'cusum':
        # Cumulative sum based change detection
        changes = pd.Series(False, index=data.index)
        
        # Placeholder for CUSUM calculation
        mean = data.iloc[:window_size].mean()
        std = data.iloc[:window_size].std()
        
        # Calculate upper and lower cumulative sums
        s_pos = 0
        s_neg = 0
        
        for i in range(window_size, len(data)):
            x = data.iloc[i]
            s_pos = max(0, s_pos + (x - (mean + 0.5*std)))
            s_neg = min(0, s_neg + (x - (mean - 0.5*std)))
            
            if s_pos > threshold*std or s_neg < -threshold*std:
                changes.iloc[i] = True
                # Reset CUSUM after detecting change
                s_pos = 0
                s_neg = 0
                mean = data.iloc[max(0, i-window_size):i].mean()
                std = data.iloc[max(0, i-window_size):i].std()
    
    else:
        raise ValueError("Method must be 'z_score', 'percent_change', or 'cusum'")
    
    return changes

def detect_seasonality_anomalies(time_series: pd.Series, period: int, 
                              n_std: float = 3.0) -> pd.Series:
    """
    Detect anomalies considering seasonality patterns.
    
    Args:
        time_series: Time series data
        period: Seasonality period (e.g., 24 for hourly with daily seasonality)
        n_std: Number of standard deviations for threshold
        
    Returns:
        Series of boolean anomaly indicators
    """
    data = time_series.copy()
    
    # Create seasonal groups
    seasonal_groups = data.groupby(data.index.map(lambda x: x % period))
    
    # Calculate seasonal statistics
    seasonal_mean = seasonal_groups.transform('mean')
    seasonal_std = seasonal_groups.transform('std')
    
    # Calculate upper and lower bounds
    upper_bound = seasonal_mean + n_std * seasonal_std
    lower_bound = seasonal_mean - n_std * seasonal_std
    
    # Identify anomalies
    anomalies = (data > upper_bound) | (data < lower_bound)
    
    return anomalies

def detect_event_based_anomalies(time_series: pd.Series, events: List[Tuple[str, str]],
                              window_size: int = 5, n_std: float = 3.0) -> pd.Series:
    """
    Detect anomalies around specific events.
    
    Args:
        time_series: Time series data
        events: List of event periods as (start_date, end_date) tuples
        window_size: Size of window around events
        n_std: Number of standard deviations for threshold
        
    Returns:
        Series of boolean anomaly indicators
    """
    data = time_series.copy()
    anomalies = pd.Series(False, index=data.index)
    
    # Convert events to datetime if string
    event_periods = []
    for start, end in events:
        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)
        event_periods.append((start, end))
    
    # Get non-event data for baseline statistics
    mask = pd.Series(True, index=data.index)
    for start, end in event_periods:
        event_mask = (data.index >= start) & (data.index <= end)
        # Also exclude window before and after
        window_before = pd.Timedelta(days=window_size)
        window_after = pd.Timedelta(days=window_size)
        extended_mask = (data.index >= (start - window_before)) & (data.index <= (end + window_after))
        mask = mask & ~extended_mask
    
    # Calculate baseline statistics
    baseline_data = data[mask]
    baseline_mean = baseline_data.mean()
    baseline_std = baseline_data.std()
    
    # Calculate thresholds
    upper_bound = baseline_mean + n_std * baseline_std
    lower_bound = baseline_mean - n_std * baseline_std
    
    # Check for anomalies only during events
    for start, end in event_periods:
        event_data = data[(data.index >= start) & (data.index <= end)]
        event_anomalies = (event_data > upper_bound) | (event_data < lower_bound)
        anomalies.loc[event_anomalies.index] = event_anomalies
    
    return anomalies 