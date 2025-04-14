"""
Divergence Detection Module

This module provides functions for detecting divergences between price action
and technical indicators, which can signal potential trend reversals.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Dict, Optional


def find_swing_points(data: Union[pd.Series, np.ndarray], 
                      window: int = 5, 
                      threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find swing highs and lows in price or indicator data
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price or indicator data
    window : int
        Window size for detecting swing points (default: 5)
    threshold : float
        Minimum percentage change to qualify as a swing point (default: 0.0)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Arrays containing indices of swing highs and swing lows
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    data_len = len(data)
    swing_highs = []
    swing_lows = []
    
    # Need at least 2*window+1 data points
    if data_len < 2 * window + 1:
        return np.array(swing_highs), np.array(swing_lows)
    
    for i in range(window, data_len - window):
        # Check if current point is higher than all points in window
        is_high = True
        for j in range(i - window, i + window + 1):
            if j != i and data[j] >= data[i]:
                is_high = False
                break
        
        # Check if current point is lower than all points in window
        is_low = True
        for j in range(i - window, i + window + 1):
            if j != i and data[j] <= data[i]:
                is_low = False
                break
        
        # Check threshold if specified
        if threshold > 0:
            min_change = abs(data[i] * threshold)
            if is_high:
                is_high = all(data[i] - data[j] >= min_change for j in range(i-window, i)) and \
                          all(data[i] - data[j] >= min_change for j in range(i+1, i+window+1))
            if is_low:
                is_low = all(data[j] - data[i] >= min_change for j in range(i-window, i)) and \
                         all(data[j] - data[i] >= min_change for j in range(i+1, i+window+1))
        
        if is_high:
            swing_highs.append(i)
        if is_low:
            swing_lows.append(i)
    
    return np.array(swing_highs), np.array(swing_lows)


def detect_regular_divergence(price: Union[pd.Series, np.ndarray], 
                              indicator: Union[pd.Series, np.ndarray], 
                              window: int = 5,
                              threshold: float = 0.0,
                              lookback: int = 10) -> Tuple[List[int], List[int]]:
    """
    Detect regular (classical) divergences between price and indicator
    
    Parameters:
    -----------
    price : Union[pd.Series, np.ndarray]
        Price data
    indicator : Union[pd.Series, np.ndarray]
        Technical indicator data
    window : int
        Window size for detecting swing points (default: 5)
    threshold : float
        Minimum percentage change to qualify as a swing point (default: 0.0)
    lookback : int
        Number of swing points to look back for divergence (default: 10)
        
    Returns:
    --------
    Tuple[List[int], List[int]]
        Lists of indices where bullish and bearish divergences occur
    """
    if isinstance(price, pd.Series):
        price = price.values
    if isinstance(indicator, pd.Series):
        indicator = indicator.values
    
    # Find swing points for both price and indicator
    price_highs, price_lows = find_swing_points(price, window, threshold)
    ind_highs, ind_lows = find_swing_points(indicator, window, threshold)
    
    bullish_div = []  # Price makes lower lows but indicator makes higher lows
    bearish_div = []  # Price makes higher highs but indicator makes lower highs
    
    # Detect bullish divergences (compare lows)
    for i in range(1, min(lookback, len(price_lows))):
        current_low_idx = price_lows[-i]
        prev_low_idx = price_lows[-i-1] if i < len(price_lows)-1 else None
        
        if prev_low_idx is None:
            continue
        
        # Find corresponding indicator lows (closest in time)
        matching_ind_lows = [idx for idx in ind_lows if abs(idx - current_low_idx) <= window]
        prev_matching_ind_lows = [idx for idx in ind_lows if abs(idx - prev_low_idx) <= window]
        
        if not matching_ind_lows or not prev_matching_ind_lows:
            continue
        
        current_ind_low_idx = matching_ind_lows[0]
        prev_ind_low_idx = prev_matching_ind_lows[0]
        
        # Check for bullish divergence: price lower low but indicator higher low
        if price[current_low_idx] < price[prev_low_idx] and indicator[current_ind_low_idx] > indicator[prev_ind_low_idx]:
            bullish_div.append(current_low_idx)
    
    # Detect bearish divergences (compare highs)
    for i in range(1, min(lookback, len(price_highs))):
        current_high_idx = price_highs[-i]
        prev_high_idx = price_highs[-i-1] if i < len(price_highs)-1 else None
        
        if prev_high_idx is None:
            continue
        
        # Find corresponding indicator highs (closest in time)
        matching_ind_highs = [idx for idx in ind_highs if abs(idx - current_high_idx) <= window]
        prev_matching_ind_highs = [idx for idx in ind_highs if abs(idx - prev_high_idx) <= window]
        
        if not matching_ind_highs or not prev_matching_ind_highs:
            continue
        
        current_ind_high_idx = matching_ind_highs[0]
        prev_ind_high_idx = prev_matching_ind_highs[0]
        
        # Check for bearish divergence: price higher high but indicator lower high
        if price[current_high_idx] > price[prev_high_idx] and indicator[current_ind_high_idx] < indicator[prev_ind_high_idx]:
            bearish_div.append(current_high_idx)
    
    return bullish_div, bearish_div


def detect_hidden_divergence(price: Union[pd.Series, np.ndarray], 
                             indicator: Union[pd.Series, np.ndarray], 
                             window: int = 5,
                             threshold: float = 0.0,
                             lookback: int = 10) -> Tuple[List[int], List[int]]:
    """
    Detect hidden divergences between price and indicator
    
    Parameters:
    -----------
    price : Union[pd.Series, np.ndarray]
        Price data
    indicator : Union[pd.Series, np.ndarray]
        Technical indicator data
    window : int
        Window size for detecting swing points (default: 5)
    threshold : float
        Minimum percentage change to qualify as a swing point (default: 0.0)
    lookback : int
        Number of swing points to look back for divergence (default: 10)
        
    Returns:
    --------
    Tuple[List[int], List[int]]
        Lists of indices where hidden bullish and bearish divergences occur
    """
    if isinstance(price, pd.Series):
        price = price.values
    if isinstance(indicator, pd.Series):
        indicator = indicator.values
    
    # Find swing points for both price and indicator
    price_highs, price_lows = find_swing_points(price, window, threshold)
    ind_highs, ind_lows = find_swing_points(indicator, window, threshold)
    
    hidden_bullish_div = []  # Price makes higher lows but indicator makes lower lows
    hidden_bearish_div = []  # Price makes lower highs but indicator makes higher highs
    
    # Detect hidden bullish divergences (compare lows)
    for i in range(1, min(lookback, len(price_lows))):
        current_low_idx = price_lows[-i]
        prev_low_idx = price_lows[-i-1] if i < len(price_lows)-1 else None
        
        if prev_low_idx is None:
            continue
        
        # Find corresponding indicator lows (closest in time)
        matching_ind_lows = [idx for idx in ind_lows if abs(idx - current_low_idx) <= window]
        prev_matching_ind_lows = [idx for idx in ind_lows if abs(idx - prev_low_idx) <= window]
        
        if not matching_ind_lows or not prev_matching_ind_lows:
            continue
        
        current_ind_low_idx = matching_ind_lows[0]
        prev_ind_low_idx = prev_matching_ind_lows[0]
        
        # Check for hidden bullish divergence: price higher low but indicator lower low
        if price[current_low_idx] > price[prev_low_idx] and indicator[current_ind_low_idx] < indicator[prev_ind_low_idx]:
            hidden_bullish_div.append(current_low_idx)
    
    # Detect hidden bearish divergences (compare highs)
    for i in range(1, min(lookback, len(price_highs))):
        current_high_idx = price_highs[-i]
        prev_high_idx = price_highs[-i-1] if i < len(price_highs)-1 else None
        
        if prev_high_idx is None:
            continue
        
        # Find corresponding indicator highs (closest in time)
        matching_ind_highs = [idx for idx in ind_highs if abs(idx - current_high_idx) <= window]
        prev_matching_ind_highs = [idx for idx in ind_highs if abs(idx - prev_high_idx) <= window]
        
        if not matching_ind_highs or not prev_matching_ind_highs:
            continue
        
        current_ind_high_idx = matching_ind_highs[0]
        prev_ind_high_idx = prev_matching_ind_highs[0]
        
        # Check for hidden bearish divergence: price lower high but indicator higher high
        if price[current_high_idx] < price[prev_high_idx] and indicator[current_ind_high_idx] > indicator[prev_ind_high_idx]:
            hidden_bearish_div.append(current_high_idx)
    
    return hidden_bullish_div, hidden_bearish_div


def detect_all_divergences(price: Union[pd.Series, np.ndarray], 
                           indicator: Union[pd.Series, np.ndarray],
                           window: int = 5,
                           threshold: float = 0.0,
                           lookback: int = 10) -> Dict[str, List[int]]:
    """
    Detect all types of divergences between price and indicator
    
    Parameters:
    -----------
    price : Union[pd.Series, np.ndarray]
        Price data
    indicator : Union[pd.Series, np.ndarray]
        Technical indicator data
    window : int
        Window size for detecting swing points (default: 5)
    threshold : float
        Minimum percentage change to qualify as a swing point (default: 0.0)
    lookback : int
        Number of swing points to look back for divergence (default: 10)
        
    Returns:
    --------
    Dict[str, List[int]]
        Dictionary containing indices for all types of divergences
    """
    # Get regular divergences
    bullish_div, bearish_div = detect_regular_divergence(
        price, indicator, window, threshold, lookback
    )
    
    # Get hidden divergences
    hidden_bullish_div, hidden_bearish_div = detect_hidden_divergence(
        price, indicator, window, threshold, lookback
    )
    
    return {
        'regular_bullish': bullish_div,
        'regular_bearish': bearish_div,
        'hidden_bullish': hidden_bullish_div,
        'hidden_bearish': hidden_bearish_div
    }


def divergence_strength(price: Union[pd.Series, np.ndarray], 
                        indicator: Union[pd.Series, np.ndarray],
                        divergence_type: str,
                        divergence_idx: int,
                        lookback: int = 2) -> float:
    """
    Calculate the strength of a detected divergence
    
    Parameters:
    -----------
    price : Union[pd.Series, np.ndarray]
        Price data
    indicator : Union[pd.Series, np.ndarray]
        Technical indicator data
    divergence_type : str
        Type of divergence ('regular_bullish', 'regular_bearish', 
                           'hidden_bullish', 'hidden_bearish')
    divergence_idx : int
        Index of the divergence point
    lookback : int
        Number of swing points to look back for strength calculation (default: 2)
        
    Returns:
    --------
    float
        Strength score of the divergence (0.0 to 1.0)
    """
    if isinstance(price, pd.Series):
        price = price.values
    if isinstance(indicator, pd.Series):
        indicator = indicator.values
    
    # Find swing points for reference
    window = 5  # Default window
    price_highs, price_lows = find_swing_points(price, window)
    
    # Identify previous swing points
    if 'bullish' in divergence_type:
        # For bullish divergences, we look at lows
        swing_indices = price_lows
    else:
        # For bearish divergences, we look at highs
        swing_indices = price_highs
    
    # Find the closest swing point to the divergence_idx
    closest_idx = min(swing_indices, key=lambda x: abs(x - divergence_idx))
    
    # Find the previous swing point
    prev_idx = None
    for idx in swing_indices:
        if idx < closest_idx:
            if prev_idx is None or idx > prev_idx:
                prev_idx = idx
    
    if prev_idx is None:
        return 0.0  # Can't calculate strength without a previous point
    
    # Calculate price and indicator percentage changes
    price_change = abs((price[closest_idx] - price[prev_idx]) / price[prev_idx])
    ind_change = abs((indicator[closest_idx] - indicator[prev_idx]) / indicator[prev_idx])
    
    # Calculate divergence strength based on the difference in changes
    if 'regular_bullish' == divergence_type:
        # Price makes lower lows, indicator makes higher lows
        if price[closest_idx] < price[prev_idx] and indicator[closest_idx] > indicator[prev_idx]:
            strength = min(1.0, (ind_change + price_change) / 2)
        else:
            return 0.0  # Not a valid regular bullish divergence
    
    elif 'regular_bearish' == divergence_type:
        # Price makes higher highs, indicator makes lower highs
        if price[closest_idx] > price[prev_idx] and indicator[closest_idx] < indicator[prev_idx]:
            strength = min(1.0, (ind_change + price_change) / 2)
        else:
            return 0.0  # Not a valid regular bearish divergence
    
    elif 'hidden_bullish' == divergence_type:
        # Price makes higher lows, indicator makes lower lows
        if price[closest_idx] > price[prev_idx] and indicator[closest_idx] < indicator[prev_idx]:
            strength = min(1.0, (ind_change + price_change) / 2)
        else:
            return 0.0  # Not a valid hidden bullish divergence
    
    elif 'hidden_bearish' == divergence_type:
        # Price makes lower highs, indicator makes higher highs
        if price[closest_idx] < price[prev_idx] and indicator[closest_idx] > indicator[prev_idx]:
            strength = min(1.0, (ind_change + price_change) / 2)
        else:
            return 0.0  # Not a valid hidden bearish divergence
    
    else:
        return 0.0  # Unknown divergence type
    
    return strength


def multi_timeframe_divergence(price_data: Dict[str, Union[pd.Series, np.ndarray]],
                              indicator_data: Dict[str, Union[pd.Series, np.ndarray]],
                              timeframes: List[str]) -> Dict[str, Dict[str, List[int]]]:
    """
    Detect divergences across multiple timeframes
    
    Parameters:
    -----------
    price_data : Dict[str, Union[pd.Series, np.ndarray]]
        Dictionary of price data for each timeframe
    indicator_data : Dict[str, Union[pd.Series, np.ndarray]]
        Dictionary of indicator data for each timeframe
    timeframes : List[str]
        List of timeframe keys to analyze
        
    Returns:
    --------
    Dict[str, Dict[str, List[int]]]
        Dictionary of divergence results for each timeframe
    """
    results = {}
    
    for tf in timeframes:
        if tf in price_data and tf in indicator_data:
            results[tf] = detect_all_divergences(price_data[tf], indicator_data[tf])
    
    return results


def divergence_confirmation(price: Union[pd.Series, np.ndarray], 
                           primary_indicator: Union[pd.Series, np.ndarray],
                           secondary_indicator: Union[pd.Series, np.ndarray],
                           window: int = 5) -> Dict[str, List[int]]:
    """
    Find divergences confirmed by two indicators
    
    Parameters:
    -----------
    price : Union[pd.Series, np.ndarray]
        Price data
    primary_indicator : Union[pd.Series, np.ndarray]
        Primary technical indicator data
    secondary_indicator : Union[pd.Series, np.ndarray]
        Secondary technical indicator data for confirmation
    window : int
        Window size for detecting swing points (default: 5)
        
    Returns:
    --------
    Dict[str, List[int]]
        Dictionary of confirmed divergences
    """
    # Detect divergences with primary indicator
    primary_divs = detect_all_divergences(price, primary_indicator, window)
    
    # Detect divergences with secondary indicator
    secondary_divs = detect_all_divergences(price, secondary_indicator, window)
    
    # Find confirmed divergences (present in both indicators within a small window)
    confirmed = {
        'regular_bullish': [],
        'regular_bearish': [],
        'hidden_bullish': [],
        'hidden_bearish': []
    }
    
    for div_type in confirmed.keys():
        for primary_idx in primary_divs[div_type]:
            # Check if there's a matching divergence in secondary indicator
            for secondary_idx in secondary_divs[div_type]:
                if abs(primary_idx - secondary_idx) <= window:
                    confirmed[div_type].append(primary_idx)
                    break
    
    return confirmed
