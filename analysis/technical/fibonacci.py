"""
Fibonacci Technical Analysis Module

This module provides functions for using Fibonacci sequences and ratios
for technical analysis in cryptocurrency trading.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Dict, Optional
from scipy import signal


def _find_swing_points(data: Union[pd.Series, np.ndarray], 
                     min_distance: int = 10, 
                     prominence: float = 0.03) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find significant swing high and swing low points in price data
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    min_distance : int
        Minimum number of samples between swing points
    prominence : float
        Minimum prominence of swing points relative to surrounding data
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Arrays containing swing high and swing low indices
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    # Find swing highs (local maxima)
    highs, _ = signal.find_peaks(data, distance=min_distance, prominence=prominence * np.mean(data))
    
    # Find swing lows (local minima by finding peaks in inverted data)
    lows, _ = signal.find_peaks(-data, distance=min_distance, prominence=prominence * np.mean(data))
    
    return highs, lows


def fibonacci_retracement(data: Union[pd.Series, np.ndarray], 
                         is_uptrend: bool = True,
                         start_idx: Optional[int] = None,
                         end_idx: Optional[int] = None,
                         levels: List[float] = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]) -> Dict[float, float]:
    """
    Calculate Fibonacci retracement levels for a given price swing
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    is_uptrend : bool
        True if calculating retracement for an uptrend, False for downtrend
    start_idx : Optional[int]
        Index of the start point (swing low for uptrend, swing high for downtrend)
        If None, automatically detects a significant swing point
    end_idx : Optional[int]
        Index of the end point (swing high for uptrend, swing low for downtrend)
        If None, automatically detects a significant swing point
    levels : List[float]
        Fibonacci levels to calculate (0.0 = start, 1.0 = end)
        
    Returns:
    --------
    Dict[float, float]
        Dictionary mapping Fibonacci levels to price values
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    # Automatically detect significant swing points if not provided
    if start_idx is None or end_idx is None:
        highs, lows = _find_swing_points(data)
        
        if is_uptrend:
            if start_idx is None and len(lows) > 0:
                # For uptrend, start from the lowest low
                lowest_low_idx = lows[np.argmin(data[lows])]
                start_idx = lowest_low_idx
            
            if end_idx is None and len(highs) > 0:
                # For uptrend, end at the highest high after the start point
                if start_idx:
                    valid_highs = highs[highs > start_idx]
                    if len(valid_highs) > 0:
                        highest_high_idx = valid_highs[np.argmax(data[valid_highs])]
                        end_idx = highest_high_idx
                    else:
                        end_idx = len(data) - 1
                else:
                    highest_high_idx = highs[np.argmax(data[highs])]
                    end_idx = highest_high_idx
        else:
            if start_idx is None and len(highs) > 0:
                # For downtrend, start from the highest high
                highest_high_idx = highs[np.argmax(data[highs])]
                start_idx = highest_high_idx
            
            if end_idx is None and len(lows) > 0:
                # For downtrend, end at the lowest low after the start point
                if start_idx:
                    valid_lows = lows[lows > start_idx]
                    if len(valid_lows) > 0:
                        lowest_low_idx = valid_lows[np.argmin(data[valid_lows])]
                        end_idx = lowest_low_idx
                    else:
                        end_idx = len(data) - 1
                else:
                    lowest_low_idx = lows[np.argmin(data[lows])]
                    end_idx = lowest_low_idx
    
    if start_idx is None or end_idx is None:
        raise ValueError("Could not automatically detect swing points. Please provide start_idx and end_idx.")
    
    # Get price values at the start and end points
    start_price = data[start_idx]
    end_price = data[end_idx]
    
    # Calculate the price range
    price_range = end_price - start_price
    
    # Calculate Fibonacci retracement levels
    retracement_levels = {}
    for level in levels:
        retracement_levels[level] = start_price + (price_range * level)
    
    return retracement_levels


def fibonacci_extension(data: Union[pd.Series, np.ndarray],
                       is_uptrend: bool = True,
                       point_a_idx: Optional[int] = None,
                       point_b_idx: Optional[int] = None,
                       point_c_idx: Optional[int] = None,
                       levels: List[float] = [0.0, 0.618, 1.0, 1.618, 2.618, 3.618, 4.236]) -> Dict[float, float]:
    """
    Calculate Fibonacci extension levels for a three-point pattern (ABC)
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    is_uptrend : bool
        True if calculating extension for an uptrend, False for downtrend
    point_a_idx : Optional[int]
        Index of point A (typically a significant swing low for uptrend, high for downtrend)
    point_b_idx : Optional[int]
        Index of point B (typically a significant swing high for uptrend, low for downtrend)
    point_c_idx : Optional[int]
        Index of point C (typically a retracement level, e.g., 0.618 of AB)
    levels : List[float]
        Fibonacci extension levels to calculate
        
    Returns:
    --------
    Dict[float, float]
        Dictionary mapping Fibonacci levels to price values
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    # Automatically detect significant swing points if not provided
    if point_a_idx is None or point_b_idx is None or point_c_idx is None:
        highs, lows = _find_swing_points(data)
        
        if is_uptrend:
            # In uptrend: A=low, B=high, C=higher low
            if point_a_idx is None and len(lows) > 0:
                point_a_idx = lows[0]  # First significant low
            
            if point_b_idx is None and len(highs) > 0 and point_a_idx is not None:
                valid_highs = highs[highs > point_a_idx]
                if len(valid_highs) > 0:
                    point_b_idx = valid_highs[0]  # First high after A
            
            if point_c_idx is None and len(lows) > 0 and point_b_idx is not None:
                valid_lows = lows[lows > point_b_idx]
                if len(valid_lows) > 0:
                    point_c_idx = valid_lows[0]  # First low after B
        else:
            # In downtrend: A=high, B=low, C=lower high
            if point_a_idx is None and len(highs) > 0:
                point_a_idx = highs[0]  # First significant high
            
            if point_b_idx is None and len(lows) > 0 and point_a_idx is not None:
                valid_lows = lows[lows > point_a_idx]
                if len(valid_lows) > 0:
                    point_b_idx = valid_lows[0]  # First low after A
            
            if point_c_idx is None and len(highs) > 0 and point_b_idx is not None:
                valid_highs = highs[highs > point_b_idx]
                if len(valid_highs) > 0:
                    point_c_idx = valid_highs[0]  # First high after B
    
    if point_a_idx is None or point_b_idx is None or point_c_idx is None:
        raise ValueError("Could not automatically detect required points. Please provide point_a_idx, point_b_idx, and point_c_idx.")
    
    # Get price values at the three points
    price_a = data[point_a_idx]
    price_b = data[point_b_idx]
    price_c = data[point_c_idx]
    
    # Calculate the price range of the first swing (AB)
    ab_range = price_b - price_a
    
    # Calculate Fibonacci extension levels from point C
    extension_levels = {}
    for level in levels:
        extension_levels[level] = price_c + (ab_range * level)
    
    return extension_levels


def fibonacci_projection(data: Union[pd.Series, np.ndarray],
                        point_a_idx: int, 
                        point_b_idx: int,
                        point_c_idx: int,
                        levels: List[float] = [0.0, 0.382, 0.618, 1.0, 1.618, 2.618, 3.618, 4.236]) -> Dict[float, float]:
    """
    Calculate Fibonacci projection levels for a three-point pattern (ABC),
    projecting a CD move similar to AB
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    point_a_idx : int
        Index of point A
    point_b_idx : int
        Index of point B
    point_c_idx : int
        Index of point C (where the projection starts)
    levels : List[float]
        Fibonacci levels to calculate projections
        
    Returns:
    --------
    Dict[float, float]
        Dictionary mapping Fibonacci levels to price values
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    # Get price values at the three points
    price_a = data[point_a_idx]
    price_b = data[point_b_idx]
    price_c = data[point_c_idx]
    
    # Calculate the price range and direction of the AB leg
    ab_range = price_b - price_a
    
    # Calculate Fibonacci projection levels from point C
    projection_levels = {}
    for level in levels:
        projection_levels[level] = price_c + (ab_range * level)
    
    return projection_levels


def fibonacci_time_zones(data: Union[pd.Series, np.ndarray], 
                        start_idx: int,
                        fibonacci_sequence: List[int] = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]) -> List[int]:
    """
    Calculate Fibonacci time zones (future time points based on Fibonacci sequence)
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    start_idx : int
        Starting index for Fibonacci time zones
    fibonacci_sequence : List[int]
        Sequence of Fibonacci numbers to use for time projections
        
    Returns:
    --------
    List[int]
        List of indices representing Fibonacci time zones
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    time_zones = []
    
    for fib in fibonacci_sequence:
        time_idx = start_idx + fib
        if time_idx < len(data):
            time_zones.append(time_idx)
        else:
            break
    
    return time_zones


def fibonacci_fan(data: Union[pd.Series, np.ndarray],
                 start_idx: int,
                 end_idx: int,
                 projection_length: int = 50,
                 levels: List[float] = [0.382, 0.5, 0.618]) -> Dict[float, List[Tuple[int, float]]]:
    """
    Calculate Fibonacci fan levels (diagonal support/resistance lines)
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    start_idx : int
        Starting index for the trend
    end_idx : int
        Ending index for the trend
    projection_length : int
        Number of periods to project the fan lines
    levels : List[float]
        Fibonacci levels to use for the fan lines
        
    Returns:
    --------
    Dict[float, List[Tuple[int, float]]]
        Dictionary mapping Fibonacci levels to lists of (index, price) points
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    # Get price values at the start and end points
    start_price = data[start_idx]
    end_price = data[end_idx]
    
    # Calculate the price range and time range
    price_range = end_price - start_price
    time_range = end_idx - start_idx
    
    # Calculate the slope of the full trend line
    slope = price_range / time_range
    
    # Maximum projection index
    max_idx = min(len(data), end_idx + projection_length)
    
    # Calculate fan lines
    fan_lines = {}
    for level in levels:
        # Calculate the fan line's slope (adjusted by Fibonacci level)
        fan_slope = slope * level
        
        # Calculate fan line points from the start point
        fan_line = []
        for i in range(start_idx, max_idx):
            time_delta = i - start_idx
            price = start_price + (fan_slope * time_delta)
            fan_line.append((i, price))
        
        fan_lines[level] = fan_line
    
    return fan_lines


def fibonacci_circles(data: Union[pd.Series, np.ndarray],
                     center_idx: int,
                     radius_idx: int,
                     levels: List[float] = [0.382, 0.5, 0.618, 1.0, 1.618, 2.618]) -> Dict[float, float]:
    """
    Calculate Fibonacci circles (concentric circles at Fibonacci distances)
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    center_idx : int
        Index of the center point of the circles
    radius_idx : int
        Index that defines the radius (distance from center to this point)
    levels : List[float]
        Fibonacci levels to scale the base radius
        
    Returns:
    --------
    Dict[float, float]
        Dictionary mapping Fibonacci levels to radius values
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    # Get the center price
    center_price = data[center_idx]
    
    # Calculate the base radius
    radius_price = data[radius_idx]
    base_radius = abs(radius_price - center_price)
    
    # Calculate circle radii at different Fibonacci levels
    circle_radii = {}
    for level in levels:
        circle_radii[level] = base_radius * level
    
    return circle_radii


# Helper functions for visualization and analysis
def _detect_swings(data: Union[pd.Series, np.ndarray], lookback_period: int = 50) -> Dict[str, np.ndarray]:
    """
    Detect swing highs and lows for potential Fibonacci analysis
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    lookback_period : int
        Period to look back for significant swings
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing 'uptrends' and 'downtrends' - arrays of (start_idx, end_idx) for each trend
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    # First find all significant swing points
    highs, lows = _find_swing_points(data)
    
    uptrends = []
    downtrends = []
    
    # Detect major uptrends (from significant low to significant high)
    for low_idx in lows:
        # Find the next significant high after this low
        future_highs = highs[highs > low_idx]
        if len(future_highs) > 0:
            high_idx = future_highs[0]
            price_change = (data[high_idx] - data[low_idx]) / data[low_idx]
            
            # Only consider significant price moves
            if price_change > 0.05:  # 5% or more
                uptrends.append((low_idx, high_idx))
    
    # Detect major downtrends (from significant high to significant low)
    for high_idx in highs:
        # Find the next significant low after this high
        future_lows = lows[lows > high_idx]
        if len(future_lows) > 0:
            low_idx = future_lows[0]
            price_change = (data[low_idx] - data[high_idx]) / data[high_idx]
            
            # Only consider significant price moves
            if price_change < -0.05:  # -5% or more
                downtrends.append((high_idx, low_idx))
    
    return {
        'uptrends': np.array(uptrends),
        'downtrends': np.array(downtrends)
    }


def _analyze_fibonacci_patterns(ohlcv_df: pd.DataFrame, lookback_period: int = 100) -> Dict[str, any]:
    """
    Full Fibonacci analysis of price data
    
    Parameters:
    -----------
    ohlcv_df : pd.DataFrame
        DataFrame with OHLCV data
    lookback_period : int
        Period to look back for pattern analysis
        
    Returns:
    --------
    Dict[str, any]
        Dictionary containing various Fibonacci analyses
    """
    close_prices = ohlcv_df['close'].values
    
    # Detect swings for Fibonacci analysis
    swings = _detect_swings(close_prices[-lookback_period:])
    
    results = {}
    
    # Analyze most recent uptrend for retracement
    if len(swings['uptrends']) > 0:
        most_recent_uptrend = swings['uptrends'][-1]
        start_idx, end_idx = most_recent_uptrend
        
        # Adjust indices to match the original dataframe
        data_offset = len(close_prices) - lookback_period
        start_idx += data_offset
        end_idx += data_offset
        
        results['retracement_up'] = fibonacci_retracement(close_prices, 
                                                          is_uptrend=True,
                                                          start_idx=start_idx,
                                                          end_idx=end_idx)
    
    # Analyze most recent downtrend for retracement
    if len(swings['downtrends']) > 0:
        most_recent_downtrend = swings['downtrends'][-1]
        start_idx, end_idx = most_recent_downtrend
        
        # Adjust indices to match the original dataframe
        data_offset = len(close_prices) - lookback_period
        start_idx += data_offset
        end_idx += data_offset
        
        results['retracement_down'] = fibonacci_retracement(close_prices, 
                                                            is_uptrend=False,
                                                            start_idx=start_idx,
                                                            end_idx=end_idx)
    
    # If we have a clear ABC pattern in the recent data, calculate extensions
    if len(swings['uptrends']) > 0 and len(swings['downtrends']) > 0:
        # Check if we have an uptrend followed by a downtrend (potential ABC pattern)
        if swings['uptrends'][-1][1] < swings['downtrends'][-1][0]:
            point_a_idx = swings['uptrends'][-1][0] + data_offset
            point_b_idx = swings['uptrends'][-1][1] + data_offset
            point_c_idx = swings['downtrends'][-1][1] + data_offset
            
            results['extension_up'] = fibonacci_extension(close_prices,
                                                         is_uptrend=True,
                                                         point_a_idx=point_a_idx,
                                                         point_b_idx=point_b_idx,
                                                         point_c_idx=point_c_idx)
    
    return results
