"""
Chart Pattern Recognition Module

This module provides functions for recognizing various chart patterns 
in cryptocurrency price data, such as head and shoulders, triangles, etc.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Dict, Optional
from scipy import signal
from scipy.stats import linregress


def find_peaks(data: Union[pd.Series, np.ndarray], 
              min_distance: int = 5, 
              prominence: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find peaks and valleys in price data
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    min_distance : int
        Minimum number of samples between peaks
    prominence : float
        Minimum prominence of peaks relative to surrounding data
        
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


def head_and_shoulders(data: Union[pd.Series, np.ndarray],
                       window_size: int = 20,
                       peak_prominence: float = 0.03) -> List[Dict[str, any]]:
    """
    Detect head and shoulders patterns
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data (typically close prices)
    window_size : int
        Size of the sliding window to look for patterns
    peak_prominence : float
        Minimum prominence for peak detection
        
    Returns:
    --------
    List[Dict[str, any]]
        List of dictionaries containing pattern information:
        {
            'type': 'head_and_shoulders' or 'inverse_head_and_shoulders',
            'start_idx': starting index,
            'end_idx': ending index,
            'left_shoulder_idx': left shoulder index,
            'head_idx': head index,
            'right_shoulder_idx': right shoulder index,
            'neckline': (slope, intercept) of neckline
        }
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    patterns = []
    peaks, valleys = find_peaks(data, min_distance=window_size // 5, prominence=peak_prominence)
    
    # Loop through potential pattern combinations
    for i in range(len(peaks) - 2):
        # Check regular head and shoulders
        if len(valleys) >= 2 and i < len(peaks) - 2:
            # Identify potential head and shoulders components
            left_shoulder_idx = peaks[i]
            potential_head_idx = peaks[i + 1]
            right_shoulder_idx = peaks[i + 2]
            
            # Find valleys between shoulders and head
            left_valley = None
            right_valley = None
            
            for v in valleys:
                if left_shoulder_idx < v < potential_head_idx:
                    left_valley = v
                elif potential_head_idx < v < right_shoulder_idx:
                    right_valley = v
            
            if left_valley is not None and right_valley is not None:
                left_shoulder_height = data[left_shoulder_idx]
                head_height = data[potential_head_idx]
                right_shoulder_height = data[right_shoulder_idx]
                
                # Check if the pattern matches head and shoulders criteria
                if (head_height > left_shoulder_height and 
                    head_height > right_shoulder_height and
                    abs(left_shoulder_height - right_shoulder_height) < 0.1 * head_height):
                    
                    # Calculate neckline (connecting the two valleys)
                    x = np.array([left_valley, right_valley])
                    y = data[x]
                    slope, intercept, _, _, _ = linregress(x, y)
                    
                    patterns.append({
                        'type': 'head_and_shoulders',
                        'start_idx': left_shoulder_idx,
                        'end_idx': right_shoulder_idx,
                        'left_shoulder_idx': left_shoulder_idx,
                        'head_idx': potential_head_idx,
                        'right_shoulder_idx': right_shoulder_idx,
                        'neckline': (slope, intercept)
                    })
    
    # Detect inverse head and shoulders (look for pattern in inverted data)
    inv_peaks, inv_valleys = find_peaks(-data, min_distance=window_size // 5, prominence=peak_prominence)
    
    for i in range(len(inv_peaks) - 2):
        if len(inv_valleys) >= 2 and i < len(inv_peaks) - 2:
            # Identify potential inverse head and shoulders components
            left_shoulder_idx = inv_peaks[i]
            potential_head_idx = inv_peaks[i + 1]
            right_shoulder_idx = inv_peaks[i + 2]
            
            # Find peaks between shoulders and head
            left_peak = None
            right_peak = None
            
            for p in inv_valleys:  # These are actually peaks in the original data
                if left_shoulder_idx < p < potential_head_idx:
                    left_peak = p
                elif potential_head_idx < p < right_shoulder_idx:
                    right_peak = p
            
            if left_peak is not None and right_peak is not None:
                left_shoulder_depth = -data[left_shoulder_idx]
                head_depth = -data[potential_head_idx]
                right_shoulder_depth = -data[right_shoulder_idx]
                
                # Check if the pattern matches inverse head and shoulders criteria
                if (head_depth > left_shoulder_depth and 
                    head_depth > right_shoulder_depth and
                    abs(left_shoulder_depth - right_shoulder_depth) < 0.1 * head_depth):
                    
                    # Calculate neckline
                    x = np.array([left_peak, right_peak])
                    y = data[x]
                    slope, intercept, _, _, _ = linregress(x, y)
                    
                    patterns.append({
                        'type': 'inverse_head_and_shoulders',
                        'start_idx': left_shoulder_idx,
                        'end_idx': right_shoulder_idx,
                        'left_shoulder_idx': left_shoulder_idx,
                        'head_idx': potential_head_idx,
                        'right_shoulder_idx': right_shoulder_idx,
                        'neckline': (slope, intercept)
                    })
    
    return patterns


def double_top_bottom(data: Union[pd.Series, np.ndarray],
                     window_size: int = 20,
                     peak_prominence: float = 0.03,
                     similarity_threshold: float = 0.03) -> List[Dict[str, any]]:
    """
    Detect double top and double bottom patterns
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    window_size : int
        Size of the sliding window to look for patterns
    peak_prominence : float
        Minimum prominence for peak detection
    similarity_threshold : float
        Maximum allowed difference between tops/bottoms as a percentage
        
    Returns:
    --------
    List[Dict[str, any]]
        List of dictionaries containing pattern information:
        {
            'type': 'double_top' or 'double_bottom',
            'start_idx': starting index,
            'end_idx': ending index,
            'first_point_idx': first top/bottom index,
            'second_point_idx': second top/bottom index,
            'valley_idx': valley index (for double top) or peak index (for double bottom)
        }
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    patterns = []
    peaks, valleys = find_peaks(data, min_distance=window_size // 5, prominence=peak_prominence)
    
    # Detect double tops
    for i in range(len(peaks) - 1):
        peak1_idx = peaks[i]
        peak2_idx = peaks[i + 1]
        peak1_value = data[peak1_idx]
        peak2_value = data[peak2_idx]
        
        # Check if peaks are close enough in value
        if abs(peak1_value - peak2_value) <= similarity_threshold * peak1_value:
            # Find valley between peaks
            valley_between = None
            valley_value = float('inf')
            
            for v in valleys:
                if peak1_idx < v < peak2_idx:
                    if data[v] < valley_value:
                        valley_value = data[v]
                        valley_between = v
            
            # Check if there's a significant valley between peaks
            if valley_between is not None and valley_value < min(peak1_value, peak2_value) * 0.97:
                patterns.append({
                    'type': 'double_top',
                    'start_idx': peak1_idx,
                    'end_idx': peak2_idx,
                    'first_point_idx': peak1_idx,
                    'second_point_idx': peak2_idx,
                    'valley_idx': valley_between
                })
    
    # Detect double bottoms
    for i in range(len(valleys) - 1):
        valley1_idx = valleys[i]
        valley2_idx = valleys[i + 1]
        valley1_value = data[valley1_idx]
        valley2_value = data[valley2_idx]
        
        # Check if valleys are close enough in value
        if abs(valley1_value - valley2_value) <= similarity_threshold * valley1_value:
            # Find peak between valleys
            peak_between = None
            peak_value = float('-inf')
            
            for p in peaks:
                if valley1_idx < p < valley2_idx:
                    if data[p] > peak_value:
                        peak_value = data[p]
                        peak_between = p
            
            # Check if there's a significant peak between valleys
            if peak_between is not None and peak_value > max(valley1_value, valley2_value) * 1.03:
                patterns.append({
                    'type': 'double_bottom',
                    'start_idx': valley1_idx,
                    'end_idx': valley2_idx,
                    'first_point_idx': valley1_idx,
                    'second_point_idx': valley2_idx,
                    'peak_idx': peak_between
                })
    
    return patterns


def triangle_patterns(data: Union[pd.Series, np.ndarray],
                     window_size: int = 50,
                     min_points: int = 5) -> List[Dict[str, any]]:
    """
    Detect triangle patterns (ascending, descending, symmetric)
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    window_size : int
        Size of the window to look for patterns
    min_points : int
        Minimum number of points required to form reliable trendlines
        
    Returns:
    --------
    List[Dict[str, any]]
        List of dictionaries containing pattern information:
        {
            'type': 'ascending_triangle', 'descending_triangle', or 'symmetric_triangle',
            'start_idx': starting index,
            'end_idx': ending index,
            'upper_trendline': (slope, intercept) of upper trendline,
            'lower_trendline': (slope, intercept) of lower trendline
        }
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    patterns = []
    
    # Sliding window approach
    for start_idx in range(0, len(data) - window_size, window_size // 2):
        end_idx = start_idx + window_size
        window_data = data[start_idx:end_idx]
        
        # Find peaks and valleys in the window
        peaks, valleys = find_peaks(window_data, min_distance=3, prominence=0.01)
        
        # Adjust indices to be relative to original data
        peaks = peaks + start_idx
        valleys = valleys + start_idx
        
        if len(peaks) >= min_points // 2 and len(valleys) >= min_points // 2:
            # Get peak and valley points for trendline calculation
            peak_points = [(i, data[i]) for i in peaks]
            valley_points = [(i, data[i]) for i in valleys]
            
            # Calculate upper trendline (using peaks)
            upper_x = np.array([p[0] for p in peak_points])
            upper_y = np.array([p[1] for p in peak_points])
            if len(upper_x) >= 2:
                upper_slope, upper_intercept, _, _, _ = linregress(upper_x, upper_y)
            else:
                continue
            
            # Calculate lower trendline (using valleys)
            lower_x = np.array([v[0] for v in valley_points])
            lower_y = np.array([v[1] for v in valley_points])
            if len(lower_x) >= 2:
                lower_slope, lower_intercept, _, _, _ = linregress(lower_x, lower_y)
            else:
                continue
            
            # Determine triangle type
            pattern_type = None
            
            if abs(upper_slope) < 0.0001 and lower_slope > 0.0001:
                pattern_type = 'ascending_triangle'
            elif upper_slope < -0.0001 and abs(lower_slope) < 0.0001:
                pattern_type = 'descending_triangle'
            elif upper_slope < -0.0001 and lower_slope > 0.0001:
                pattern_type = 'symmetric_triangle'
            
            if pattern_type:
                # Calculate pattern fit quality (r-squared)
                upper_r_squared = np.corrcoef(upper_x, upper_y)[0, 1] ** 2
                lower_r_squared = np.corrcoef(lower_x, lower_y)[0, 1] ** 2
                
                # Only add high-quality patterns
                if upper_r_squared > 0.6 and lower_r_squared > 0.6:
                    patterns.append({
                        'type': pattern_type,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'upper_trendline': (upper_slope, upper_intercept),
                        'lower_trendline': (lower_slope, lower_intercept),
                        'quality': (upper_r_squared + lower_r_squared) / 2
                    })
    
    return patterns


def rectangle_patterns(data: Union[pd.Series, np.ndarray],
                      window_size: int = 50,
                      min_points: int = 4,
                      similarity_threshold: float = 0.03) -> List[Dict[str, any]]:
    """
    Detect rectangle patterns (support and resistance channels)
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    window_size : int
        Size of the window to look for patterns
    min_points : int
        Minimum number of points required to identify the pattern
    similarity_threshold : float
        Maximum allowed difference between support/resistance levels
        
    Returns:
    --------
    List[Dict[str, any]]
        List of dictionaries containing pattern information:
        {
            'type': 'rectangle',
            'start_idx': starting index,
            'end_idx': ending index,
            'resistance_level': upper boundary,
            'support_level': lower boundary
        }
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    patterns = []
    
    # Sliding window approach
    for start_idx in range(0, len(data) - window_size, window_size // 2):
        end_idx = start_idx + window_size
        window_data = data[start_idx:end_idx]
        
        # Find peaks and valleys in the window
        peaks, valleys = find_peaks(window_data, min_distance=3, prominence=0.01)
        
        if len(peaks) >= min_points // 2 and len(valleys) >= min_points // 2:
            # Get peak and valley values
            peak_values = window_data[peaks]
            valley_values = window_data[valleys]
            
            # Cluster resistance levels
            resistance_clusters = []
            for p in peak_values:
                added_to_cluster = False
                for i, cluster in enumerate(resistance_clusters):
                    if abs(p - np.mean(cluster)) / np.mean(cluster) <= similarity_threshold:
                        resistance_clusters[i].append(p)
                        added_to_cluster = True
                        break
                
                if not added_to_cluster:
                    resistance_clusters.append([p])
            
            # Cluster support levels
            support_clusters = []
            for v in valley_values:
                added_to_cluster = False
                for i, cluster in enumerate(support_clusters):
                    if abs(v - np.mean(cluster)) / np.mean(cluster) <= similarity_threshold:
                        support_clusters[i].append(v)
                        added_to_cluster = True
                        break
                
                if not added_to_cluster:
                    support_clusters.append([v])
            
            # Find the largest clusters
            resistance_clusters = sorted(resistance_clusters, key=lambda x: len(x), reverse=True)
            support_clusters = sorted(support_clusters, key=lambda x: len(x), reverse=True)
            
            if (len(resistance_clusters) > 0 and len(support_clusters) > 0 and
                len(resistance_clusters[0]) >= min_points // 2 and 
                len(support_clusters[0]) >= min_points // 2):
                
                resistance_level = np.mean(resistance_clusters[0])
                support_level = np.mean(support_clusters[0])
                
                # Check if the channel is significant (> 2% difference)
                if (resistance_level - support_level) / support_level >= 0.02:
                    patterns.append({
                        'type': 'rectangle',
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'resistance_level': resistance_level,
                        'support_level': support_level
                    })
    
    return patterns


def wedge_patterns(data: Union[pd.Series, np.ndarray],
                  window_size: int = 50,
                  min_points: int = 5) -> List[Dict[str, any]]:
    """
    Detect wedge patterns (rising and falling wedges)
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    window_size : int
        Size of the window to look for patterns
    min_points : int
        Minimum number of points required for trendlines
        
    Returns:
    --------
    List[Dict[str, any]]
        List of dictionaries containing pattern information:
        {
            'type': 'rising_wedge' or 'falling_wedge',
            'start_idx': starting index,
            'end_idx': ending index,
            'upper_trendline': (slope, intercept) of upper trendline,
            'lower_trendline': (slope, intercept) of lower trendline
        }
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    patterns = []
    
    # Sliding window approach
    for start_idx in range(0, len(data) - window_size, window_size // 2):
        end_idx = start_idx + window_size
        window_data = data[start_idx:end_idx]
        
        # Find peaks and valleys in the window
        peaks, valleys = find_peaks(window_data, min_distance=3, prominence=0.01)
        
        # Adjust indices to be relative to original data
        peaks = peaks + start_idx
        valleys = valleys + start_idx
        
        if len(peaks) >= min_points // 2 and len(valleys) >= min_points // 2:
            # Get peak and valley points for trendline calculation
            peak_points = [(i, data[i]) for i in peaks]
            valley_points = [(i, data[i]) for i in valleys]
            
            # Calculate upper trendline (using peaks)
            upper_x = np.array([p[0] for p in peak_points])
            upper_y = np.array([p[1] for p in peak_points])
            if len(upper_x) >= 2:
                upper_slope, upper_intercept, _, _, _ = linregress(upper_x, upper_y)
            else:
                continue
            
            # Calculate lower trendline (using valleys)
            lower_x = np.array([v[0] for v in valley_points])
            lower_y = np.array([v[1] for v in valley_points])
            if len(lower_x) >= 2:
                lower_slope, lower_intercept, _, _, _ = linregress(lower_x, lower_y)
            else:
                continue
            
            # Determine wedge type
            pattern_type = None
            
            # Rising wedge: both trendlines slope upward, lower line steeper
            if upper_slope > 0 and lower_slope > 0 and lower_slope > upper_slope:
                pattern_type = 'rising_wedge'
            # Falling wedge: both trendlines slope downward, upper line steeper
            elif upper_slope < 0 and lower_slope < 0 and upper_slope < lower_slope:
                pattern_type = 'falling_wedge'
            
            if pattern_type:
                # Calculate pattern fit quality (r-squared)
                upper_r_squared = np.corrcoef(upper_x, upper_y)[0, 1] ** 2
                lower_r_squared = np.corrcoef(lower_x, lower_y)[0, 1] ** 2
                
                # Only add high-quality patterns
                if upper_r_squared > 0.6 and lower_r_squared > 0.6:
                    patterns.append({
                        'type': pattern_type,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'upper_trendline': (upper_slope, upper_intercept),
                        'lower_trendline': (lower_slope, lower_intercept),
                        'quality': (upper_r_squared + lower_r_squared) / 2
                    })
    
    return patterns


def flag_pennant_patterns(data: Union[pd.Series, np.ndarray],
                         window_size: int = 50,
                         pole_window: int = 20) -> List[Dict[str, any]]:
    """
    Detect flag and pennant patterns
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    window_size : int
        Size of the window to look for the flag/pennant part
    pole_window : int
        Size of the window to look for the pole part
        
    Returns:
    --------
    List[Dict[str, any]]
        List of dictionaries containing pattern information:
        {
            'type': 'bull_flag', 'bear_flag', 'bull_pennant', or 'bear_pennant',
            'start_idx': starting index,
            'end_idx': ending index,
            'pole_start_idx': pole start index,
            'pole_end_idx': pole end index,
            'flag_trendlines': [(upper_slope, upper_intercept), (lower_slope, lower_intercept)]
        }
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    patterns = []
    
    # Look for flags/pennants
    for i in range(pole_window, len(data) - window_size):
        # Check for pole (strong trend)
        pole_data = data[i-pole_window:i]
        pole_change = (pole_data[-1] - pole_data[0]) / pole_data[0]
        
        # If there's a strong trend, look for a flag/pennant after it
        if abs(pole_change) >= 0.05:  # At least 5% price movement for a pole
            flag_data = data[i:i+window_size]
            peaks, valleys = find_peaks(flag_data, min_distance=3, prominence=0.01)
            
            if len(peaks) >= 2 and len(valleys) >= 2:
                # Get peak and valley points for trendline calculation
                peak_points = [(j+i, data[j+i]) for j in peaks]
                valley_points = [(j+i, data[j+i]) for j in valleys]
                
                # Calculate upper trendline (using peaks)
                upper_x = np.array([p[0] for p in peak_points])
                upper_y = np.array([p[1] for p in peak_points])
                upper_slope, upper_intercept, _, _, _ = linregress(upper_x, upper_y)
                
                # Calculate lower trendline (using valleys)
                lower_x = np.array([v[0] for v in valley_points])
                lower_y = np.array([v[1] for v in valley_points])
                lower_slope, lower_intercept, _, _, _ = linregress(lower_x, lower_y)
                
                # Check for bull flag (downward channel after uptrend)
                if pole_change > 0.05 and upper_slope < 0 and lower_slope < 0:
                    pattern_type = 'bull_flag'
                    if abs(upper_slope - lower_slope) < 0.001:  # Parallel lines
                        patterns.append({
                            'type': pattern_type,
                            'start_idx': i,
                            'end_idx': i + window_size,
                            'pole_start_idx': i - pole_window,
                            'pole_end_idx': i,
                            'flag_trendlines': [(upper_slope, upper_intercept), 
                                              (lower_slope, lower_intercept)]
                        })
                
                # Check for bear flag (upward channel after downtrend)
                elif pole_change < -0.05 and upper_slope > 0 and lower_slope > 0:
                    pattern_type = 'bear_flag'
                    if abs(upper_slope - lower_slope) < 0.001:  # Parallel lines
                        patterns.append({
                            'type': pattern_type,
                            'start_idx': i,
                            'end_idx': i + window_size,
                            'pole_start_idx': i - pole_window,
                            'pole_end_idx': i,
                            'flag_trendlines': [(upper_slope, upper_intercept), 
                                              (lower_slope, lower_intercept)]
                        })
                
                # Check for bull pennant (converging lines after uptrend)
                elif pole_change > 0.05 and upper_slope < 0 and lower_slope > 0:
                    pattern_type = 'bull_pennant'
                    patterns.append({
                        'type': pattern_type,
                        'start_idx': i,
                        'end_idx': i + window_size,
                        'pole_start_idx': i - pole_window,
                        'pole_end_idx': i,
                        'flag_trendlines': [(upper_slope, upper_intercept), 
                                          (lower_slope, lower_intercept)]
                    })
                
                # Check for bear pennant (converging lines after downtrend)
                elif pole_change < -0.05 and upper_slope < 0 and lower_slope > 0:
                    pattern_type = 'bear_pennant'
                    patterns.append({
                        'type': pattern_type,
                        'start_idx': i,
                        'end_idx': i + window_size,
                        'pole_start_idx': i - pole_window,
                        'pole_end_idx': i,
                        'flag_trendlines': [(upper_slope, upper_intercept), 
                                          (lower_slope, lower_intercept)]
                    })
    
    return patterns


def support_resistance(data: Union[pd.Series, np.ndarray],
                      window_size: int = 100,
                      num_levels: int = 3,
                      tolerance: float = 0.02) -> Dict[str, List[float]]:
    """
    Identify support and resistance levels
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    window_size : int
        Size of the window to analyze
    num_levels : int
        Number of support/resistance levels to return
    tolerance : float
        Price range (as percentage) to consider as the same level
        
    Returns:
    --------
    Dict[str, List[float]]
        Dictionary containing 'support' and 'resistance' lists with price levels
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    # Use the most recent data for analysis
    if len(data) > window_size:
        analysis_data = data[-window_size:]
    else:
        analysis_data = data
    
    # Find all peaks and valleys
    peaks, valleys = find_peaks(analysis_data, min_distance=5, prominence=0.01)
    
    # Extract peak and valley prices
    peak_prices = analysis_data[peaks]
    valley_prices = analysis_data[valleys]
    
    # Cluster resistance levels
    resistance_clusters = []
    for price in peak_prices:
        added_to_cluster = False
        for i, cluster in enumerate(resistance_clusters):
            if abs(price - np.mean(cluster)) / np.mean(cluster) <= tolerance:
                resistance_clusters[i].append(price)
                added_to_cluster = True
                break
        
        if not added_to_cluster:
            resistance_clusters.append([price])
    
    # Cluster support levels
    support_clusters = []
    for price in valley_prices:
        added_to_cluster = False
        for i, cluster in enumerate(support_clusters):
            if abs(price - np.mean(cluster)) / np.mean(cluster) <= tolerance:
                support_clusters[i].append(price)
                added_to_cluster = True
                break
        
        if not added_to_cluster:
            support_clusters.append([price])
    
    # Calculate average price for each cluster
    resistance_levels = [np.mean(cluster) for cluster in resistance_clusters 
                        if len(cluster) >= 2]  # Require at least 2 points
    support_levels = [np.mean(cluster) for cluster in support_clusters 
                     if len(cluster) >= 2]  # Require at least 2 points
    
    # Sort levels by strength (cluster size)
    resistance_strength = [(level, len(cluster)) for level, cluster 
                          in zip([np.mean(c) for c in resistance_clusters], resistance_clusters)
                          if len(cluster) >= 2]
    support_strength = [(level, len(cluster)) for level, cluster 
                       in zip([np.mean(c) for c in support_clusters], support_clusters)
                       if len(cluster) >= 2]
    
    resistance_strength.sort(key=lambda x: x[1], reverse=True)
    support_strength.sort(key=lambda x: x[1], reverse=True)
    
    # Get the strongest levels
    top_resistance = [level for level, _ in resistance_strength[:num_levels]]
    top_support = [level for level, _ in support_strength[:num_levels]]
    
    return {
        'resistance': top_resistance,
        'support': top_support
    }


# Helper function for testing
def _analyze_chart_patterns(ohlcv_df: pd.DataFrame, window_size: int = 100) -> Dict[str, List]:
    """
    Analyze chart patterns in a price dataframe
    
    Parameters:
    -----------
    ohlcv_df : pd.DataFrame
        DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
    window_size : int
        Size of the window to analyze
        
    Returns:
    --------
    Dict[str, List]
        Dictionary containing various identified patterns
    """
    # Use the close prices for analysis
    close_prices = ohlcv_df['close'].values
    high_prices = ohlcv_df['high'].values
    low_prices = ohlcv_df['low'].values
    
    patterns = {}
    
    # Get various patterns
    patterns['head_and_shoulders'] = head_and_shoulders(close_prices, window_size=window_size)
    patterns['double_patterns'] = double_top_bottom(close_prices, window_size=window_size)
    patterns['triangles'] = triangle_patterns(close_prices, window_size=window_size)
    patterns['rectangles'] = rectangle_patterns(close_prices, window_size=window_size)
    patterns['wedges'] = wedge_patterns(close_prices, window_size=window_size)
    patterns['flags_pennants'] = flag_pennant_patterns(close_prices, window_size=window_size)
    patterns['support_resistance'] = support_resistance(close_prices, window_size=window_size)
    
    return patterns
