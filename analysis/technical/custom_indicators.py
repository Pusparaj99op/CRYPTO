"""
Custom Technical Indicators Module

This module provides implementations of various custom and specialized technical indicators
that are not typically found in standard libraries but are valuable for cryptocurrency trading.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Dict, Optional
from scipy import stats


def supertrend(high: Union[pd.Series, np.ndarray],
              low: Union[pd.Series, np.ndarray],
              close: Union[pd.Series, np.ndarray],
              period: int = 10,
              multiplier: float = 3.0) -> Dict[str, np.ndarray]:
    """
    Calculate the SuperTrend indicator
    
    Parameters:
    -----------
    high : Union[pd.Series, np.ndarray]
        High prices
    low : Union[pd.Series, np.ndarray]
        Low prices
    close : Union[pd.Series, np.ndarray]
        Closing prices
    period : int
        Lookback period for ATR calculation
    multiplier : float
        Multiplier for ATR
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing 'supertrend', 'trend' (1 for up, -1 for down), and 'direction_changes'
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values
    
    # Calculate ATR
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
    
    # Calculate ATR with smoothing
    atr = np.zeros(len(close))
    atr[period] = np.mean(tr[:period])
    for i in range(period + 1, len(close)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i-1]) / period
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize SuperTrend and trend direction
    supertrend = np.zeros(len(close))
    trend = np.zeros(len(close))
    
    # Set initial values
    supertrend[period] = upper_band[period]
    trend[period] = -1  # Start with downtrend
    
    # Calculate SuperTrend
    for i in range(period + 1, len(close)):
        # Determine current trend
        if close[i-1] <= supertrend[i-1]:
            # Previous trend was down
            if lower_band[i] > supertrend[i-1]:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = supertrend[i-1]
                
            # Update trend based on price action
            if close[i] > supertrend[i]:
                trend[i] = 1  # Trend changes to up
            else:
                trend[i] = -1  # Trend remains down
        else:
            # Previous trend was up
            if upper_band[i] < supertrend[i-1]:
                supertrend[i] = upper_band[i]
            else:
                supertrend[i] = supertrend[i-1]
                
            # Update trend based on price action
            if close[i] < supertrend[i]:
                trend[i] = -1  # Trend changes to down
            else:
                trend[i] = 1  # Trend remains up
    
    # Detect direction changes
    direction_changes = np.zeros(len(trend))
    for i in range(period + 1, len(trend)):
        if trend[i] != trend[i-1]:
            direction_changes[i] = trend[i]
    
    return {
        'supertrend': supertrend,
        'trend': trend,
        'direction_changes': direction_changes
    }


def donchian_channel(high: Union[pd.Series, np.ndarray],
                    low: Union[pd.Series, np.ndarray],
                    period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Donchian Channels
    
    Parameters:
    -----------
    high : Union[pd.Series, np.ndarray]
        High prices
    low : Union[pd.Series, np.ndarray]
        Low prices
    period : int
        Lookback period for channel calculation
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Upper band, middle band, and lower band
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    
    upper_band = np.zeros(len(high))
    lower_band = np.zeros(len(low))
    
    # Calculate bands
    for i in range(period - 1, len(high)):
        upper_band[i] = np.max(high[i-period+1:i+1])
        lower_band[i] = np.min(low[i-period+1:i+1])
    
    # Calculate middle band
    middle_band = (upper_band + lower_band) / 2
    
    return upper_band, middle_band, lower_band


def keltner_channel(high: Union[pd.Series, np.ndarray],
                   low: Union[pd.Series, np.ndarray],
                   close: Union[pd.Series, np.ndarray],
                   ema_period: int = 20,
                   atr_period: int = 10,
                   multiplier: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Keltner Channels
    
    Parameters:
    -----------
    high : Union[pd.Series, np.ndarray]
        High prices
    low : Union[pd.Series, np.ndarray]
        Low prices
    close : Union[pd.Series, np.ndarray]
        Closing prices
    ema_period : int
        Period for EMA calculation
    atr_period : int
        Period for ATR calculation
    multiplier : float
        Multiplier for ATR
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Upper band, middle band, and lower band
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values
    
    # Calculate EMA of typical price
    tp = (high + low + close) / 3
    
    # Calculate EMA
    ema = np.zeros(len(close))
    ema[ema_period-1] = np.mean(tp[:ema_period])
    k = 2 / (ema_period + 1)
    for i in range(ema_period, len(close)):
        ema[i] = tp[i] * k + ema[i-1] * (1 - k)
    
    # Calculate ATR
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
    
    atr = np.zeros(len(close))
    atr[atr_period] = np.mean(tr[:atr_period])
    for i in range(atr_period + 1, len(close)):
        atr[i] = (atr[i-1] * (atr_period - 1) + tr[i-1]) / atr_period
    
    # Calculate bands
    upper_band = ema + multiplier * atr
    lower_band = ema - multiplier * atr
    middle_band = ema
    
    return upper_band, middle_band, lower_band


def pivot_points(high: Union[pd.Series, np.ndarray],
                low: Union[pd.Series, np.ndarray],
                close: Union[pd.Series, np.ndarray],
                method: str = 'standard') -> Dict[str, float]:
    """
    Calculate pivot points for the next period based on previous period data
    
    Parameters:
    -----------
    high : Union[pd.Series, np.ndarray]
        High prices of previous period
    low : Union[pd.Series, np.ndarray]
        Low prices of previous period
    close : Union[pd.Series, np.ndarray]
        Closing prices of previous period
    method : str
        Pivot point calculation method ('standard', 'fibonacci', 'woodie', 'camarilla', 'demark')
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing pivot point and support/resistance levels
    """
    if isinstance(high, pd.Series):
        high = high.values[-1]  # Use last value
    else:
        high = high[-1]
        
    if isinstance(low, pd.Series):
        low = low.values[-1]  # Use last value
    else:
        low = low[-1]
        
    if isinstance(close, pd.Series):
        close = close.values[-1]  # Use last value
    else:
        close = close[-1]
    
    result = {}
    
    if method == 'standard':
        # Calculate standard pivot points
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
        result = {
            'pivot': pivot,
            'r1': r1, 's1': s1,
            'r2': r2, 's2': s2,
            'r3': r3, 's3': s3
        }
    
    elif method == 'fibonacci':
        # Calculate Fibonacci pivot points
        pivot = (high + low + close) / 3
        r1 = pivot + 0.382 * (high - low)
        s1 = pivot - 0.382 * (high - low)
        r2 = pivot + 0.618 * (high - low)
        s2 = pivot - 0.618 * (high - low)
        r3 = pivot + 1.0 * (high - low)
        s3 = pivot - 1.0 * (high - low)
        
        result = {
            'pivot': pivot,
            'r1': r1, 's1': s1,
            'r2': r2, 's2': s2,
            'r3': r3, 's3': s3
        }
    
    elif method == 'woodie':
        # Calculate Woodie pivot points
        pivot = (high + low + 2 * close) / 4
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        
        result = {
            'pivot': pivot,
            'r1': r1, 's1': s1,
            'r2': r2, 's2': s2
        }
    
    elif method == 'camarilla':
        # Calculate Camarilla pivot points
        pivot = (high + low + close) / 3
        r1 = close + (high - low) * 1.1 / 12
        s1 = close - (high - low) * 1.1 / 12
        r2 = close + (high - low) * 1.1 / 6
        s2 = close - (high - low) * 1.1 / 6
        r3 = close + (high - low) * 1.1 / 4
        s3 = close - (high - low) * 1.1 / 4
        r4 = close + (high - low) * 1.1 / 2
        s4 = close - (high - low) * 1.1 / 2
        
        result = {
            'pivot': pivot,
            'r1': r1, 's1': s1,
            'r2': r2, 's2': s2,
            'r3': r3, 's3': s3,
            'r4': r4, 's4': s4
        }
    
    elif method == 'demark':
        # Calculate DeMark pivot points
        if close < open:
            x = high + 2 * low + close
        elif close > open:
            x = 2 * high + low + close
        else:
            x = high + low + 2 * close
        
        pivot = x / 4
        r1 = x / 2 - low
        s1 = x / 2 - high
        
        result = {
            'pivot': pivot,
            'r1': r1, 's1': s1
        }
    
    return result


def volume_profile(price: Union[pd.Series, np.ndarray],
                  volume: Union[pd.Series, np.ndarray],
                  n_bins: int = 10) -> Dict[str, np.ndarray]:
    """
    Calculate Volume Profile (Volume by Price)
    
    Parameters:
    -----------
    price : Union[pd.Series, np.ndarray]
        Price data
    volume : Union[pd.Series, np.ndarray]
        Volume data
    n_bins : int
        Number of price bins to create
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing 'price_levels', 'volumes', and 'poc' (Point of Control)
    """
    if isinstance(price, pd.Series):
        price = price.values
    if isinstance(volume, pd.Series):
        volume = volume.values
    
    # Create bins for price range
    min_price = np.min(price)
    max_price = np.max(price)
    
    # Check if price range is too small
    if abs(max_price - min_price) < 1e-5:
        bins = np.linspace(min_price - 0.5, max_price + 0.5, n_bins + 1)
    else:
        bins = np.linspace(min_price, max_price, n_bins + 1)
    
    # Calculate volume for each price bin
    volumes, bin_edges = np.histogram(price, bins=bins, weights=volume)
    
    # Calculate price level for each bin (middle of bin)
    price_levels = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find Point of Control (price level with highest volume)
    poc_idx = np.argmax(volumes)
    poc = price_levels[poc_idx]
    
    return {
        'price_levels': price_levels,
        'volumes': volumes,
        'poc': poc
    }


def market_facilitation_index(high: Union[pd.Series, np.ndarray],
                             low: Union[pd.Series, np.ndarray],
                             volume: Union[pd.Series, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate Bill Williams' Market Facilitation Index (MFI)
    
    Parameters:
    -----------
    high : Union[pd.Series, np.ndarray]
        High prices
    low : Union[pd.Series, np.ndarray]
        Low prices
    volume : Union[pd.Series, np.ndarray]
        Volume data
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing 'mfi' and 'zones' values
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(volume, pd.Series):
        volume = volume.values
    
    # Calculate price range
    price_range = high - low
    
    # Calculate MFI
    mfi = price_range / volume
    
    # Handle division by zero
    mfi = np.where(volume == 0, 0, mfi)
    
    # Identify zones (colors) based on changes
    delta_mfi = np.diff(mfi, prepend=mfi[0])
    delta_volume = np.diff(volume, prepend=volume[0])
    
    zones = np.zeros(len(mfi), dtype=int)
    
    # Green (1): MFI up, volume up
    mask_green = (delta_mfi > 0) & (delta_volume > 0)
    zones[mask_green] = 1
    
    # Blue (2): MFI up, volume down
    mask_blue = (delta_mfi > 0) & (delta_volume <= 0)
    zones[mask_blue] = 2
    
    # Pink (3): MFI down, volume up
    mask_pink = (delta_mfi <= 0) & (delta_volume > 0)
    zones[mask_pink] = 3
    
    # Brown (4): MFI down, volume down
    mask_brown = (delta_mfi <= 0) & (delta_volume <= 0)
    zones[mask_brown] = 4
    
    return {
        'mfi': mfi,
        'zones': zones
    }


def elder_force_index(close: Union[pd.Series, np.ndarray],
                     volume: Union[pd.Series, np.ndarray],
                     period: int = 13) -> np.ndarray:
    """
    Calculate Elder's Force Index
    
    Parameters:
    -----------
    close : Union[pd.Series, np.ndarray]
        Closing prices
    volume : Union[pd.Series, np.ndarray]
        Volume data
    period : int
        EMA smoothing period
        
    Returns:
    --------
    np.ndarray
        Force Index values
    """
    if isinstance(close, pd.Series):
        close = close.values
    if isinstance(volume, pd.Series):
        volume = volume.values
    
    # Calculate price changes
    price_changes = np.diff(close, prepend=close[0])
    
    # Calculate raw force index
    raw_force = price_changes * volume
    
    # Calculate EMA smoothed force index
    force_index = np.zeros(len(close))
    
    # Calculate first EMA value
    force_index[period-1] = np.mean(raw_force[:period])
    
    # Calculate EMA of force index
    k = 2 / (period + 1)
    for i in range(period, len(close)):
        force_index[i] = raw_force[i] * k + force_index[i-1] * (1 - k)
    
    return force_index


def guppy_multiple_moving_average(close: Union[pd.Series, np.ndarray],
                                short_periods: List[int] = [3, 5, 8, 10, 12, 15],
                                long_periods: List[int] = [30, 35, 40, 45, 50, 60]) -> Dict[str, np.ndarray]:
    """
    Calculate Guppy Multiple Moving Average (GMMA)
    
    Parameters:
    -----------
    close : Union[pd.Series, np.ndarray]
        Closing prices
    short_periods : List[int]
        Periods for short-term EMAs
    long_periods : List[int]
        Periods for long-term EMAs
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing short_emas and long_emas
    """
    if isinstance(close, pd.Series):
        close = close.values
    
    # Calculate short-term EMAs
    short_emas = {}
    for period in short_periods:
        ema = np.zeros(len(close))
        ema[period-1] = np.mean(close[:period])
        k = 2 / (period + 1)
        for i in range(period, len(close)):
            ema[i] = close[i] * k + ema[i-1] * (1 - k)
        short_emas[f'ema_{period}'] = ema
    
    # Calculate long-term EMAs
    long_emas = {}
    for period in long_periods:
        ema = np.zeros(len(close))
        ema[period-1] = np.mean(close[:period])
        k = 2 / (period + 1)
        for i in range(period, len(close)):
            ema[i] = close[i] * k + ema[i-1] * (1 - k)
        long_emas[f'ema_{period}'] = ema
    
    # Combine results
    return {
        'short_emas': short_emas,
        'long_emas': long_emas
    }


def hull_moving_average(data: Union[pd.Series, np.ndarray], period: int = 16) -> np.ndarray:
    """
    Calculate Hull Moving Average
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    period : int
        HMA period
        
    Returns:
    --------
    np.ndarray
        Hull Moving Average values
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    # Calculate weighted moving averages
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # Calculate WMA with period
    wma_period = np.zeros(len(data) - period + 1)
    for i in range(len(wma_period)):
        weights = np.arange(1, period + 1)
        wma_period[i] = np.sum(data[i:i+period] * weights) / np.sum(weights)
    
    # Calculate WMA with half period
    wma_half_period = np.zeros(len(data) - half_period + 1)
    for i in range(len(wma_half_period)):
        weights = np.arange(1, half_period + 1)
        wma_half_period[i] = np.sum(data[i:i+half_period] * weights) / np.sum(weights)
    
    # Calculate 2 * WMA(half period) - WMA(full period)
    raw_hma = 2 * wma_half_period[:len(wma_period)] - wma_period
    
    # Calculate WMA of raw_hma with sqrt(period)
    hma = np.zeros(len(raw_hma) - sqrt_period + 1)
    for i in range(len(hma)):
        weights = np.arange(1, sqrt_period + 1)
        hma[i] = np.sum(raw_hma[i:i+sqrt_period] * weights) / np.sum(weights)
    
    # Pad with NaNs to match original data length
    result = np.full(len(data), np.nan)
    result[-(len(hma)):] = hma
    
    return result


def mesa_adaptive_moving_average(data: Union[pd.Series, np.ndarray], 
                                fast_period: int = 12,
                                slow_period: int = 26,
                                price_offset: int = 0) -> Dict[str, np.ndarray]:
    """
    Calculate Mesa Adaptive Moving Average (MAMA)
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data
    fast_period : int
        Fast EMA period
    slow_period : int
        Slow EMA period
    price_offset : int
        Number of periods to offset price data
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing 'mama' and 'fama' values
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    # Initialize output arrays
    mama = np.zeros(len(data))
    fama = np.zeros(len(data))
    
    # Initialization
    smooth = np.zeros(len(data))
    phase = np.zeros(len(data))
    detrender = np.zeros(len(data))
    i1 = np.zeros(len(data))
    q1 = np.zeros(len(data))
    jI = np.zeros(len(data))
    jQ = np.zeros(len(data))
    
    # MAMA calculations
    for i in range(max(slow_period, fast_period), len(data)):
        # Smooth price
        smooth[i] = (4 * data[i] + 3 * data[i-1] + 2 * data[i-2] + data[i-3]) / 10
        
        # Detrend smoothed price
        detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i-2] - 0.5769 * smooth[i-4] - 0.0962 * smooth[i-6]) * (0.075 * phase[i-1] + 0.54)
        
        # In-phase and quadrature components
        q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i-2] - 0.5769 * detrender[i-4] - 0.0962 * detrender[i-6]) * (0.075 * phase[i-1] + 0.54)
        i1[i] = detrender[i-3]
        
        # Jitter components for added stability
        jI[i] = (0.0962 * i1[i] + 0.5769 * i1[i-2] - 0.5769 * i1[i-4] - 0.0962 * i1[i-6]) * (0.075 * phase[i-1] + 0.54)
        jQ[i] = (0.0962 * q1[i] + 0.5769 * q1[i-2] - 0.5769 * q1[i-4] - 0.0962 * q1[i-6]) * (0.075 * phase[i-1] + 0.54)
        
        # Calculate phase
        phase[i] = np.arctan2(jQ[i], jI[i])
        
        # Calculate adaptive alpha factor
        alpha = 0.0
        delta_phase = 0.0
        
        if i > 0:
            delta_phase = phase[i-1] - phase[i]
            if delta_phase < 0:
                delta_phase = -delta_phase
                
        # Fast alpha calculation (0.5 max)
        alpha = fast_period / (delta_phase + 1e-8)
        if alpha < 0.05:
            alpha = 0.05
        elif alpha > 0.5:
            alpha = 0.5
        
        # Calculate MAMA and FAMA
        if i > 0:
            mama[i] = alpha * data[i] + (1 - alpha) * mama[i-1]
            fama[i] = 0.5 * alpha * mama[i] + (1 - 0.5 * alpha) * fama[i-1]
    
    return {
        'mama': mama,
        'fama': fama
    }
