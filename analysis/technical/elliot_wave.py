"""
Elliott Wave Analysis Module

This module provides functions for identifying and analyzing Elliott Wave patterns
in cryptocurrency price data.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Dict, Optional
from scipy import signal
from scipy.stats import linregress


def find_swing_points(data: Union[pd.Series, np.ndarray], 
                     min_distance: int = 5, 
                     prominence: float = 0.03) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find significant swing high and low points in price data
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data (typically close prices)
    min_distance : int
        Minimum number of samples between swing points
    prominence : float
        Minimum prominence of swing points relative to surrounding data
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Arrays containing peak and valley indices
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    # Find peaks (local maxima)
    peak_indices, _ = signal.find_peaks(data, distance=min_distance, prominence=prominence * np.mean(data))
    
    # Find valleys (local minima by finding peaks in inverted data)
    valley_indices, _ = signal.find_peaks(-data, distance=min_distance, prominence=prominence * np.mean(data))
    
    return peak_indices, valley_indices


def identify_impulse_wave(data: Union[pd.Series, np.ndarray],
                         min_distance: int = 5,
                         prominence: float = 0.03) -> List[Dict[str, any]]:
    """
    Identify potential Elliott Wave impulse patterns (5-wave structure)
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    min_distance : int
        Minimum number of samples between swing points
    prominence : float
        Minimum prominence for peak detection
        
    Returns:
    --------
    List[Dict[str, any]]
        List of dictionaries containing pattern information
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    # Find peaks and valleys
    peaks, valleys = find_swing_points(data, min_distance, prominence)
    
    # We need at least 3 peaks and 3 valleys for a 5-wave structure
    if len(peaks) < 3 or len(valleys) < 3:
        return []
    
    impulse_patterns = []
    
    # Iterate through potential wave patterns
    # Impulse wave pattern: valley(0) - peak(1) - valley(2) - peak(3) - valley(4) - peak(5)
    for i in range(len(valleys) - 2):
        # Need at least 3 peaks after this valley
        if i + 2 >= len(peaks):
            continue
        
        # Define wave points
        wave0_idx = valleys[i]  # Starting point
        wave1_idx = peaks[i]    # Wave 1 peak
        wave2_idx = valleys[i+1]  # Wave 2 valley
        wave3_idx = peaks[i+1]    # Wave 3 peak
        wave4_idx = valleys[i+2]  # Wave 4 valley
        wave5_idx = peaks[i+2]    # Wave 5 peak
        
        # Check if points are in sequence
        if not (wave0_idx < wave1_idx < wave2_idx < wave3_idx < wave4_idx < wave5_idx):
            continue
        
        # Get prices at each point
        wave0_price = data[wave0_idx]
        wave1_price = data[wave1_idx]
        wave2_price = data[wave2_idx]
        wave3_price = data[wave3_idx]
        wave4_price = data[wave4_idx]
        wave5_price = data[wave5_idx]
        
        # Calculate wave measurements
        wave1_height = wave1_price - wave0_price
        wave2_retracement = (wave1_price - wave2_price) / wave1_height
        wave3_height = wave3_price - wave2_price
        wave4_retracement = (wave3_price - wave4_price) / wave3_height
        wave5_height = wave5_price - wave4_price
        
        # Check Elliott Wave rules
        
        # Rule 1: Wave 2 cannot retrace more than 100% of Wave 1
        if wave2_retracement > 1.0:
            continue
        
        # Rule 2: Wave 3 must be extended (longer than Wave 1 or Wave 5)
        if not (wave3_height > wave1_height or wave3_height > wave5_height):
            continue
        
        # Rule 3: Wave 4 cannot overlap Wave 1 (price should be higher)
        if wave4_price < wave1_price:
            continue
        
        # Add pattern to results
        impulse_patterns.append({
            'type': 'impulse_wave',
            'start_idx': wave0_idx,
            'end_idx': wave5_idx,
            'wave_points': {
                '0': wave0_idx,
                '1': wave1_idx,
                '2': wave2_idx,
                '3': wave3_idx,
                '4': wave4_idx,
                '5': wave5_idx
            },
            'wave_heights': {
                '1': wave1_height,
                '3': wave3_height,
                '5': wave5_height
            },
            'retracements': {
                '2': wave2_retracement,
                '4': wave4_retracement
            }
        })
    
    return impulse_patterns


def identify_corrective_wave(data: Union[pd.Series, np.ndarray],
                           min_distance: int = 5,
                           prominence: float = 0.03) -> List[Dict[str, any]]:
    """
    Identify potential Elliott Wave corrective patterns (ABC structure)
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    min_distance : int
        Minimum number of samples between swing points
    prominence : float
        Minimum prominence for peak detection
        
    Returns:
    --------
    List[Dict[str, any]]
        List of dictionaries containing pattern information
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    # Find peaks and valleys
    peaks, valleys = find_swing_points(data, min_distance, prominence)
    
    corrective_patterns = []
    
    # Look for bearish ABC patterns (high-low-high)
    for i in range(len(peaks) - 1):
        # Find a valley between these peaks
        valid_valleys = [v for v in valleys if peaks[i] < v < peaks[i+1]]
        
        if len(valid_valleys) == 0:
            continue
        
        # Use the deepest valley
        wave_b_idx = valid_valleys[np.argmin(data[valid_valleys])]
        
        # Define wave points
        wave_a_idx = peaks[i]
        wave_c_idx = peaks[i+1]
        
        # Get prices
        wave_a_price = data[wave_a_idx]
        wave_b_price = data[wave_b_idx]
        wave_c_price = data[wave_c_idx]
        
        # Check if it's a valid pattern
        if wave_b_price >= min(wave_a_price, wave_c_price):
            continue  # B wave must be lower than both A and C
        
        # Add pattern to results
        corrective_patterns.append({
            'type': 'corrective_wave_bear',
            'start_idx': wave_a_idx,
            'end_idx': wave_c_idx,
            'wave_points': {
                'A': wave_a_idx,
                'B': wave_b_idx,
                'C': wave_c_idx
            },
            'wave_prices': {
                'A': wave_a_price,
                'B': wave_b_price,
                'C': wave_c_price
            }
        })
    
    # Look for bullish ABC patterns (low-high-low)
    for i in range(len(valleys) - 1):
        # Find a peak between these valleys
        valid_peaks = [p for p in peaks if valleys[i] < p < valleys[i+1]]
        
        if len(valid_peaks) == 0:
            continue
        
        # Use the highest peak
        wave_b_idx = valid_peaks[np.argmax(data[valid_peaks])]
        
        # Define wave points
        wave_a_idx = valleys[i]
        wave_c_idx = valleys[i+1]
        
        # Get prices
        wave_a_price = data[wave_a_idx]
        wave_b_price = data[wave_b_idx]
        wave_c_price = data[wave_c_idx]
        
        # Check if it's a valid pattern
        if wave_b_price <= max(wave_a_price, wave_c_price):
            continue  # B wave must be higher than both A and C
        
        # Add pattern to results
        corrective_patterns.append({
            'type': 'corrective_wave_bull',
            'start_idx': wave_a_idx,
            'end_idx': wave_c_idx,
            'wave_points': {
                'A': wave_a_idx,
                'B': wave_b_idx,
                'C': wave_c_idx
            },
            'wave_prices': {
                'A': wave_a_price,
                'B': wave_b_price,
                'C': wave_c_price
            }
        })
    
    return corrective_patterns


def wave_degree(data: Union[pd.Series, np.ndarray], 
               window_sizes: List[int] = [20, 50, 100, 200]) -> Dict[str, Dict]:
    """
    Identify Elliott Waves at different degrees (timescales)
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    window_sizes : List[int]
        List of window sizes to analyze (smaller to larger)
        
    Returns:
    --------
    Dict[str, Dict]
        Dictionary mapping wave degrees to wave patterns
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    # Define wave degrees corresponding to window sizes
    degrees = ['minuette', 'minute', 'minor', 'intermediate']
    
    # Ensure we have the same number of degrees as window sizes
    degrees = degrees[:len(window_sizes)]
    
    result = {}
    
    for degree, window_size in zip(degrees, window_sizes):
        # Only analyze if we have enough data
        if len(data) >= window_size:
            # Adjust prominence based on window size
            prominence = 0.03 * (1 + 0.2 * window_sizes.index(window_size))
            min_dist = max(5, window_size // 10)
            
            # Analyze relevant data
            analysis_data = data[-window_size:] if len(data) > window_size else data
            
            # Identify waves
            impulse_waves = identify_impulse_wave(analysis_data, 
                                                min_distance=min_dist, 
                                                prominence=prominence)
            
            corrective_waves = identify_corrective_wave(analysis_data, 
                                                      min_distance=min_dist, 
                                                      prominence=prominence)
            
            result[degree] = {
                'impulse': impulse_waves,
                'corrective': corrective_waves
            }
    
    return result


def wave_channels(data: Union[pd.Series, np.ndarray], 
                 wave_pattern: Dict[str, any]) -> Dict[str, Tuple[float, float]]:
    """
    Calculate Elliott Wave channels (trendlines connecting key points)
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    wave_pattern : Dict[str, any]
        Wave pattern dictionary
        
    Returns:
    --------
    Dict[str, Tuple[float, float]]
        Dictionary of channel (slope, intercept) pairs
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    channels = {}
    
    if wave_pattern['type'] == 'impulse_wave':
        # Get wave points
        points = wave_pattern['wave_points']
        
        # Calculate 0-2-4 channel (connecting waves 0, 2, and 4)
        if all(k in points for k in ['0', '2', '4']):
            x = np.array([points['0'], points['2'], points['4']])
            y = np.array([data[points['0']], data[points['2']], data[points['4']]])
            slope, intercept, _, _, _ = linregress(x, y)
            channels['base_channel'] = (slope, intercept)
    
    elif 'corrective_wave' in wave_pattern['type']:
        # Get wave points
        points = wave_pattern['wave_points']
        
        # Calculate A-C channel (connecting waves A and C)
        if all(k in points for k in ['A', 'C']):
            x = np.array([points['A'], points['C']])
            y = np.array([data[points['A']], data[points['C']]])
            slope, intercept, _, _, _ = linregress(x, y)
            channels['corrective_channel'] = (slope, intercept)
    
    return channels


def elliott_oscillator(data: Union[pd.Series, np.ndarray], 
                      fast_period: int = 5, 
                      slow_period: int = 35) -> np.ndarray:
    """
    Calculate Elliott Oscillator (difference between fast and slow moving averages)
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    fast_period : int
        Period for the fast moving average
    slow_period : int
        Period for the slow moving average
        
    Returns:
    --------
    np.ndarray
        Elliott Oscillator values
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    # Calculate fast and slow moving averages
    fast_ma = np.zeros(len(data) - fast_period + 1)
    slow_ma = np.zeros(len(data) - slow_period + 1)
    
    for i in range(len(fast_ma)):
        fast_ma[i] = np.mean(data[i:i+fast_period])
    
    for i in range(len(slow_ma)):
        slow_ma[i] = np.mean(data[i:i+slow_period])
    
    # Align arrays to the same length
    min_len = min(len(fast_ma), len(slow_ma))
    fast_ma = fast_ma[-min_len:]
    slow_ma = slow_ma[-min_len:]
    
    # Calculate the oscillator (difference between fast and slow MAs)
    oscillator = fast_ma - slow_ma
    
    return oscillator


# Helper function for analysis
def _analyze_elliott_waves(ohlcv_df: pd.DataFrame, lookback_period: int = 200) -> Dict[str, any]:
    """
    Complete Elliott Wave analysis of price data
    
    Parameters:
    -----------
    ohlcv_df : pd.DataFrame
        DataFrame with OHLCV data
    lookback_period : int
        Period to look back for analysis
        
    Returns:
    --------
    Dict[str, any]
        Dictionary containing Elliott Wave analysis results
    """
    close_prices = ohlcv_df['close'].values
    
    # Use the most recent data for analysis
    if len(close_prices) > lookback_period:
        analysis_data = close_prices[-lookback_period:]
    else:
        analysis_data = close_prices
    
    results = {}
    
    # Find impulse waves
    results['impulse_waves'] = identify_impulse_wave(analysis_data)
    
    # Find corrective waves
    results['corrective_waves'] = identify_corrective_wave(analysis_data)
    
    # Analyze waves at different degrees
    results['wave_degrees'] = wave_degree(analysis_data)
    
    # Calculate Elliott Oscillator
    results['elliott_oscillator'] = elliott_oscillator(analysis_data)
    
    # For the most recent impulse wave, calculate extended relationships
    if results['impulse_waves']:
        most_recent_impulse = results['impulse_waves'][-1]
        results['current_impulse_channels'] = wave_channels(analysis_data, most_recent_impulse)
        results['current_impulse_relationships'] = wave_relationships(most_recent_impulse)
    
    # For the most recent corrective wave, calculate extended relationships
    if results['corrective_waves']:
        most_recent_corrective = results['corrective_waves'][-1]
        results['current_corrective_channels'] = wave_channels(analysis_data, most_recent_corrective)
        results['current_corrective_relationships'] = wave_relationships(most_recent_corrective)
    
    return results
