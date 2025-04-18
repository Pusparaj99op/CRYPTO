"""
Time Series Analysis Module

This module provides functions for analyzing and decomposing time series data
in cryptocurrency markets.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Tuple
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.statespace.tools import cfa_simulation
from scipy import signal

def decompose_time_series(data: Union[np.ndarray, pd.Series],
                         period: int = None,
                         model: str = 'additive') -> Dict[str, np.ndarray]:
    """
    Decompose time series into trend, seasonal, and residual components.
    
    Args:
        data: Time series data
        period: Seasonal period (if None, will be inferred)
        model: Decomposition model ('additive' or 'multiplicative')
        
    Returns:
        Dictionary containing decomposed components
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    decomposition = seasonal_decompose(data, period=period, model=model)
    
    return {
        'observed': decomposition.observed,
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid
    }

def detect_seasonality(data: Union[np.ndarray, pd.Series],
                      max_lag: int = 50) -> Dict[str, float]:
    """
    Detect seasonality in time series data.
    
    Args:
        data: Time series data
        max_lag: Maximum lag to consider
        
    Returns:
        Dictionary containing seasonality detection results
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    # Calculate autocorrelation
    acf_values = acf(data, nlags=max_lag)
    
    # Find significant peaks
    peaks, _ = signal.find_peaks(np.abs(acf_values))
    
    # Calculate periodogram
    f, Pxx = signal.periodogram(data)
    
    return {
        'autocorrelation': acf_values,
        'significant_peaks': peaks,
        'periodogram_frequencies': f,
        'periodogram_power': Pxx
    }

def detect_trend(data: Union[np.ndarray, pd.Series],
                window: int = 20) -> Dict[str, np.ndarray]:
    """
    Detect trend in time series data.
    
    Args:
        data: Time series data
        window: Rolling window size
        
    Returns:
        Dictionary containing trend detection results
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    # Calculate moving averages
    ma = data.rolling(window=window).mean()
    ema = data.ewm(span=window).mean()
    
    # Calculate trend strength
    trend_strength = np.abs(data - ma) / ma
    
    return {
        'moving_average': ma,
        'exponential_moving_average': ema,
        'trend_strength': trend_strength
    }

def calculate_autocorrelation(data: Union[np.ndarray, pd.Series],
                            nlags: int = 50) -> Dict[str, np.ndarray]:
    """
    Calculate autocorrelation and partial autocorrelation.
    
    Args:
        data: Time series data
        nlags: Number of lags to calculate
        
    Returns:
        Dictionary containing autocorrelation results
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    # Calculate ACF and PACF
    acf_values = acf(data, nlags=nlags)
    pacf_values = pacf(data, nlags=nlags)
    
    return {
        'autocorrelation': acf_values,
        'partial_autocorrelation': pacf_values
    }

def perform_stationarity_test(data: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    
    Args:
        data: Time series data
        
    Returns:
        Dictionary containing stationarity test results
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    # Perform ADF test
    adf_result = adfuller(data)
    
    return {
        'adf_statistic': adf_result[0],
        'p_value': adf_result[1],
        'critical_values': adf_result[4],
        'n_lags': adf_result[2],
        'n_observations': adf_result[3]
    } 