"""
Ichimoku Cloud Analysis Module

This module provides functions for calculating and analyzing Ichimoku Cloud
components for cryptocurrency price data.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict, Optional


def tenkan_sen(high: Union[pd.Series, np.ndarray], 
               low: Union[pd.Series, np.ndarray], 
               period: int = 9) -> np.ndarray:
    """
    Calculate Tenkan-sen (Conversion Line)
    
    Parameters:
    -----------
    high : Union[pd.Series, np.ndarray]
        High price data
    low : Union[pd.Series, np.ndarray]
        Low price data
    period : int
        Number of periods to consider (default: 9)
        
    Returns:
    --------
    np.ndarray
        Tenkan-sen values
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    
    tenkan = np.zeros(len(high) - period + 1)
    
    for i in range(len(tenkan)):
        highest_high = np.max(high[i:i+period])
        lowest_low = np.min(low[i:i+period])
        tenkan[i] = (highest_high + lowest_low) / 2
    
    return tenkan


def kijun_sen(high: Union[pd.Series, np.ndarray], 
              low: Union[pd.Series, np.ndarray], 
              period: int = 26) -> np.ndarray:
    """
    Calculate Kijun-sen (Base Line)
    
    Parameters:
    -----------
    high : Union[pd.Series, np.ndarray]
        High price data
    low : Union[pd.Series, np.ndarray]
        Low price data
    period : int
        Number of periods to consider (default: 26)
        
    Returns:
    --------
    np.ndarray
        Kijun-sen values
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    
    kijun = np.zeros(len(high) - period + 1)
    
    for i in range(len(kijun)):
        highest_high = np.max(high[i:i+period])
        lowest_low = np.min(low[i:i+period])
        kijun[i] = (highest_high + lowest_low) / 2
    
    return kijun


def senkou_span_a(tenkan: np.ndarray, 
                  kijun: np.ndarray, 
                  shift_period: int = 26) -> np.ndarray:
    """
    Calculate Senkou Span A (Leading Span A)
    
    Parameters:
    -----------
    tenkan : np.ndarray
        Tenkan-sen values
    kijun : np.ndarray
        Kijun-sen values
    shift_period : int
        Number of periods to shift forward (default: 26)
        
    Returns:
    --------
    np.ndarray
        Senkou Span A values
    """
    # Find minimum length
    min_len = min(len(tenkan), len(kijun))
    tenkan = tenkan[-min_len:]
    kijun = kijun[-min_len:]
    
    # Calculate average of Tenkan-sen and Kijun-sen
    span_a = (tenkan + kijun) / 2
    
    # Append shift_period NaN values to represent future projection
    projection = np.full(shift_period, np.nan)
    return np.concatenate([span_a, projection])


def senkou_span_b(high: Union[pd.Series, np.ndarray], 
                  low: Union[pd.Series, np.ndarray], 
                  period: int = 52, 
                  shift_period: int = 26) -> np.ndarray:
    """
    Calculate Senkou Span B (Leading Span B)
    
    Parameters:
    -----------
    high : Union[pd.Series, np.ndarray]
        High price data
    low : Union[pd.Series, np.ndarray]
        Low price data
    period : int
        Number of periods to consider (default: 52)
    shift_period : int
        Number of periods to shift forward (default: 26)
        
    Returns:
    --------
    np.ndarray
        Senkou Span B values
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    
    span_b = np.zeros(len(high) - period + 1)
    
    for i in range(len(span_b)):
        highest_high = np.max(high[i:i+period])
        lowest_low = np.min(low[i:i+period])
        span_b[i] = (highest_high + lowest_low) / 2
    
    # Append shift_period NaN values to represent future projection
    projection = np.full(shift_period, np.nan)
    return np.concatenate([span_b, projection])


def chikou_span(close: Union[pd.Series, np.ndarray], 
                shift_period: int = 26) -> np.ndarray:
    """
    Calculate Chikou Span (Lagging Span)
    
    Parameters:
    -----------
    close : Union[pd.Series, np.ndarray]
        Close price data
    shift_period : int
        Number of periods to shift backward (default: 26)
        
    Returns:
    --------
    np.ndarray
        Chikou Span values
    """
    if isinstance(close, pd.Series):
        close = close.values
    
    # Prepend shift_period NaN values to represent historical shift
    padding = np.full(shift_period, np.nan)
    return np.concatenate([padding, close])


def ichimoku_cloud(high: Union[pd.Series, np.ndarray], 
                   low: Union[pd.Series, np.ndarray], 
                   close: Union[pd.Series, np.ndarray], 
                   tenkan_period: int = 9, 
                   kijun_period: int = 26, 
                   senkou_b_period: int = 52, 
                   displacement: int = 26) -> Dict[str, np.ndarray]:
    """
    Calculate all components of the Ichimoku Cloud
    
    Parameters:
    -----------
    high : Union[pd.Series, np.ndarray]
        High price data
    low : Union[pd.Series, np.ndarray]
        Low price data
    close : Union[pd.Series, np.ndarray]
        Close price data
    tenkan_period : int
        Period for Tenkan-sen calculation (default: 9)
    kijun_period : int
        Period for Kijun-sen calculation (default: 26)
    senkou_b_period : int
        Period for Senkou Span B calculation (default: 52)
    displacement : int
        Displacement period for cloud components (default: 26)
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing all Ichimoku components
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values
    
    # Calculate components
    tenkan = tenkan_sen(high, low, tenkan_period)
    kijun = kijun_sen(high, low, kijun_period)
    span_a = senkou_span_a(tenkan, kijun, displacement)
    span_b = senkou_span_b(high, low, senkou_b_period, displacement)
    chikou = chikou_span(close, displacement)
    
    return {
        'tenkan_sen': tenkan,
        'kijun_sen': kijun,
        'senkou_span_a': span_a,
        'senkou_span_b': span_b,
        'chikou_span': chikou
    }


def is_bullish_cloud(senkou_a: np.ndarray, senkou_b: np.ndarray) -> np.ndarray:
    """
    Determine if the cloud is bullish (Senkou Span A > Senkou Span B)
    
    Parameters:
    -----------
    senkou_a : np.ndarray
        Senkou Span A values
    senkou_b : np.ndarray
        Senkou Span B values
        
    Returns:
    --------
    np.ndarray
        Boolean array indicating bullish cloud
    """
    min_len = min(len(senkou_a), len(senkou_b))
    return senkou_a[-min_len:] > senkou_b[-min_len:]


def identify_tk_cross(tenkan: np.ndarray, kijun: np.ndarray) -> np.ndarray:
    """
    Identify Tenkan/Kijun (TK) crossovers
    
    Parameters:
    -----------
    tenkan : np.ndarray
        Tenkan-sen values
    kijun : np.ndarray
        Kijun-sen values
        
    Returns:
    --------
    np.ndarray
        Array with values: 1 (bullish cross), -1 (bearish cross), 0 (no cross)
    """
    min_len = min(len(tenkan), len(kijun))
    tenkan = tenkan[-min_len:]
    kijun = kijun[-min_len:]
    
    # Initialize results array
    cross = np.zeros(min_len)
    
    # Check for crossovers (starting from index 1)
    for i in range(1, min_len):
        if tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]:
            cross[i] = 1  # Bullish crossover
        elif tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]:
            cross[i] = -1  # Bearish crossover
    
    return cross


def price_relative_to_cloud(close: np.ndarray, 
                           span_a: np.ndarray, 
                           span_b: np.ndarray) -> np.ndarray:
    """
    Determine the position of price relative to the cloud
    
    Parameters:
    -----------
    close : np.ndarray
        Close price data
    span_a : np.ndarray
        Senkou Span A values
    span_b : np.ndarray
        Senkou Span B values
        
    Returns:
    --------
    np.ndarray
        Array with values: 
        2 (above cloud), 
        1 (in cloud), 
        0 (below cloud)
    """
    # Align lengths
    min_len = min(len(close), len(span_a), len(span_b))
    close = close[-min_len:]
    span_a = span_a[-min_len:]
    span_b = span_b[-min_len:]
    
    # Calculate cloud top and bottom
    cloud_top = np.maximum(span_a, span_b)
    cloud_bottom = np.minimum(span_a, span_b)
    
    # Determine position
    position = np.zeros(min_len)
    
    for i in range(min_len):
        if close[i] > cloud_top[i]:
            position[i] = 2  # Above cloud
        elif close[i] < cloud_bottom[i]:
            position[i] = 0  # Below cloud
        else:
            position[i] = 1  # In cloud
    
    return position


def ichimoku_signals(high: Union[pd.Series, np.ndarray],
                     low: Union[pd.Series, np.ndarray],
                     close: Union[pd.Series, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Generate trading signals based on Ichimoku Cloud analysis
    
    Parameters:
    -----------
    high : Union[pd.Series, np.ndarray]
        High price data
    low : Union[pd.Series, np.ndarray]
        Low price data
    close : Union[pd.Series, np.ndarray]
        Close price data
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing trading signals
    """
    # Calculate Ichimoku components
    components = ichimoku_cloud(high, low, close)
    
    # Extract components
    tenkan = components['tenkan_sen']
    kijun = components['kijun_sen']
    span_a = components['senkou_span_a']
    span_b = components['senkou_span_b']
    
    # Generate signals
    tk_cross = identify_tk_cross(tenkan, kijun)
    
    # Check if close price is above/below cloud
    if isinstance(close, pd.Series):
        close_values = close.values
    else:
        close_values = close
        
    price_position = price_relative_to_cloud(close_values, span_a, span_b)
    
    # Determine cloud direction
    cloud_bullish = is_bullish_cloud(span_a, span_b)
    
    # Combine signals to generate overall signal
    # 2: Strong buy, 1: Buy, 0: Neutral, -1: Sell, -2: Strong sell
    overall_signal = np.zeros(len(tk_cross))
    
    for i in range(len(overall_signal)):
        # TK Cross signal
        tk_signal = tk_cross[i]
        
        # Price position signal
        pos_signal = price_position[i] - 1  # Convert to -1, 0, 1
        
        # Cloud direction
        cloud_signal = 1 if cloud_bullish[i] else -1
        
        # Combine signals
        overall_signal[i] = tk_signal + pos_signal + cloud_signal
        
        # Clamp values
        overall_signal[i] = max(-2, min(2, overall_signal[i]))
    
    return {
        'tk_cross': tk_cross,
        'price_position': price_position,
        'cloud_direction': cloud_bullish,
        'overall_signal': overall_signal
    }
