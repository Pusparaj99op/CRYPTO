"""
Oscillator-based Technical Indicators Module

This module provides implementations of various oscillator-based indicators 
for cryptocurrency technical analysis.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Dict, Optional

# Import some useful functions from indicators module
from .indicators import sma, ema


def rsi(data: Union[pd.Series, np.ndarray], period: int = 14, 
        smoothing_type: str = 'sma') -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI)
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data (typically close prices)
    period : int
        RSI period
    smoothing_type : str
        Type of moving average to use ('sma' or 'ema')
        
    Returns:
    --------
    np.ndarray
        RSI values (between 0 and 100)
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    if len(data) < period + 1:
        raise ValueError(f"Data length ({len(data)}) must be >= period+1 ({period+1})")
        
    # Calculate price changes
    deltas = np.diff(data)
    
    # Create arrays for gains and losses
    gains = np.zeros_like(deltas)
    losses = np.zeros_like(deltas)
    
    # Separate gains and losses
    gains[deltas > 0] = deltas[deltas > 0]
    losses[deltas < 0] = -deltas[deltas < 0]
    
    # Calculate average gains and losses
    if smoothing_type.lower() == 'sma':
        avg_gains = np.zeros(len(deltas) - period + 1)
        avg_losses = np.zeros(len(deltas) - period + 1)
        
        # Calculate initial averages
        avg_gains[0] = np.mean(gains[:period])
        avg_losses[0] = np.mean(losses[:period])
        
        # Calculate subsequent averages
        for i in range(1, len(avg_gains)):
            avg_gains[i] = ((avg_gains[i-1] * (period - 1)) + gains[i+period-1]) / period
            avg_losses[i] = ((avg_losses[i-1] * (period - 1)) + losses[i+period-1]) / period
            
    elif smoothing_type.lower() == 'ema':
        avg_gains = ema(gains, period)
        avg_losses = ema(losses, period)
        
    else:
        raise ValueError(f"Unknown smoothing_type: {smoothing_type}, use 'sma' or 'ema'")
    
    # Calculate RS and RSI, handling division by zero
    rs = np.zeros_like(avg_gains)
    nonzero_idx = avg_losses != 0
    rs[nonzero_idx] = avg_gains[nonzero_idx] / avg_losses[nonzero_idx]
    rs[~nonzero_idx] = 100.0  # If avg_loss is zero, set RSI to 100
    
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values


def stochastic(high: Union[pd.Series, np.ndarray], 
              low: Union[pd.Series, np.ndarray], 
              close: Union[pd.Series, np.ndarray], 
              k_period: int = 14, 
              d_period: int = 3,
              slowing: int = 3) -> Dict[str, np.ndarray]:
    """
    Calculate Stochastic Oscillator (%K and %D)
    
    Parameters:
    -----------
    high : Union[pd.Series, np.ndarray]
        High prices
    low : Union[pd.Series, np.ndarray]
        Low prices
    close : Union[pd.Series, np.ndarray]
        Close prices
    k_period : int
        Period for %K line
    d_period : int
        Period for %D line (moving average of %K)
    slowing : int
        Slowing period (moving average of raw %K)
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing 'k' and 'd' values
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values
    
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("High, low, and close arrays must have the same length")
    
    if len(high) < k_period:
        raise ValueError(f"Data length ({len(high)}) must be >= k_period ({k_period})")
    
    # Calculate highest high and lowest low for the period
    highest_high = np.zeros(len(high) - k_period + 1)
    lowest_low = np.zeros(len(high) - k_period + 1)
    
    for i in range(len(highest_high)):
        highest_high[i] = np.max(high[i:i+k_period])
        lowest_low[i] = np.min(low[i:i+k_period])
    
    # Calculate raw %K (without slowing)
    raw_k = np.zeros_like(highest_high)
    denominator = highest_high - lowest_low
    nonzero_idx = denominator != 0
    
    # Latest close for each window
    latest_close = close[k_period-1:][:len(raw_k)]
    
    raw_k[nonzero_idx] = 100 * (latest_close[nonzero_idx] - lowest_low[nonzero_idx]) / denominator[nonzero_idx]
    
    # Apply slowing to raw %K if specified
    if slowing > 1:
        k = np.zeros(len(raw_k) - slowing + 1)
        for i in range(len(k)):
            k[i] = np.mean(raw_k[i:i+slowing])
    else:
        k = raw_k
    
    # Calculate %D (moving average of %K)
    d = np.zeros(len(k) - d_period + 1)
    for i in range(len(d)):
        d[i] = np.mean(k[i:i+d_period])
    
    # Trim %K to match %D length
    k = k[-len(d):]
    
    return {
        'k': k,
        'd': d
    }


def cci(high: Union[pd.Series, np.ndarray], 
       low: Union[pd.Series, np.ndarray], 
       close: Union[pd.Series, np.ndarray], 
       period: int = 20,
       constant: float = 0.015) -> np.ndarray:
    """
    Calculate Commodity Channel Index (CCI)
    
    Parameters:
    -----------
    high : Union[pd.Series, np.ndarray]
        High prices
    low : Union[pd.Series, np.ndarray]
        Low prices
    close : Union[pd.Series, np.ndarray]
        Close prices
    period : int
        CCI period
    constant : float
        Scaling constant (typically 0.015)
        
    Returns:
    --------
    np.ndarray
        CCI values
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values
    
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("High, low, and close arrays must have the same length")
    
    if len(high) < period:
        raise ValueError(f"Data length ({len(high)}) must be >= period ({period})")
    
    # Calculate typical price
    typical_price = (high + low + close) / 3
    
    # Calculate SMA of typical price
    tp_sma = np.zeros(len(typical_price) - period + 1)
    
    for i in range(len(tp_sma)):
        tp_sma[i] = np.mean(typical_price[i:i+period])
    
    # Calculate mean deviation
    mean_deviation = np.zeros_like(tp_sma)
    
    for i in range(len(mean_deviation)):
        mean_deviation[i] = np.mean(np.abs(typical_price[i:i+period] - tp_sma[i]))
    
    # Calculate CCI
    cci_values = np.zeros_like(tp_sma)
    
    # Latest typical price for each window
    latest_tp = typical_price[period-1:][:len(cci_values)]
    
    # Handle division by zero
    nonzero_idx = mean_deviation != 0
    cci_values[nonzero_idx] = (latest_tp[nonzero_idx] - tp_sma[nonzero_idx]) / (constant * mean_deviation[nonzero_idx])
    
    return cci_values


def williams_r(high: Union[pd.Series, np.ndarray], 
              low: Union[pd.Series, np.ndarray], 
              close: Union[pd.Series, np.ndarray], 
              period: int = 14) -> np.ndarray:
    """
    Calculate Williams %R
    
    Parameters:
    -----------
    high : Union[pd.Series, np.ndarray]
        High prices
    low : Union[pd.Series, np.ndarray]
        Low prices
    close : Union[pd.Series, np.ndarray]
        Close prices
    period : int
        Lookback period
        
    Returns:
    --------
    np.ndarray
        Williams %R values (between -100 and 0)
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values
    
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("High, low, and close arrays must have the same length")
    
    if len(high) < period:
        raise ValueError(f"Data length ({len(high)}) must be >= period ({period})")
    
    # Calculate highest high and lowest low for the period
    highest_high = np.zeros(len(high) - period + 1)
    lowest_low = np.zeros(len(high) - period + 1)
    
    for i in range(len(highest_high)):
        highest_high[i] = np.max(high[i:i+period])
        lowest_low[i] = np.min(low[i:i+period])
    
    # Latest close for each window
    latest_close = close[period-1:][:len(highest_high)]
    
    # Calculate Williams %R
    williams_r_values = np.zeros_like(highest_high)
    
    # Handle division by zero
    denom = highest_high - lowest_low
    nonzero_idx = denom != 0
    
    williams_r_values[nonzero_idx] = -100 * (highest_high[nonzero_idx] - latest_close[nonzero_idx]) / denom[nonzero_idx]
    
    return williams_r_values


def ultimate_oscillator(high: Union[pd.Series, np.ndarray], 
                       low: Union[pd.Series, np.ndarray], 
                       close: Union[pd.Series, np.ndarray], 
                       period1: int = 7, 
                       period2: int = 14, 
                       period3: int = 28,
                       weights: List[float] = [4, 2, 1]) -> np.ndarray:
    """
    Calculate Ultimate Oscillator
    
    Parameters:
    -----------
    high : Union[pd.Series, np.ndarray]
        High prices
    low : Union[pd.Series, np.ndarray]
        Low prices
    close : Union[pd.Series, np.ndarray]
        Close prices
    period1 : int
        First (shortest) period
    period2 : int
        Second (medium) period
    period3 : int
        Third (longest) period
    weights : List[float]
        Weights for each period [w1, w2, w3]
        
    Returns:
    --------
    np.ndarray
        Ultimate Oscillator values (between 0 and 100)
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values
    
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("High, low, and close arrays must have the same length")
    
    if len(high) <= period3:
        raise ValueError(f"Data length ({len(high)}) must be > longest period ({period3})")
    
    if len(weights) != 3:
        raise ValueError("Weights must contain exactly 3 values")
    
    # Calculate buying pressure (BP) and true range (TR)
    bp = np.zeros(len(close) - 1)
    tr = np.zeros(len(close) - 1)
    
    prev_close = close[:-1]
    curr_close = close[1:]
    curr_high = high[1:]
    curr_low = low[1:]
    
    # Buying pressure = Close - Minimum(Low, PreviousClose)
    bp = curr_close - np.minimum(curr_low, prev_close)
    
    # True range = Maximum(High, PreviousClose) - Minimum(Low, PreviousClose)
    tr = np.maximum(curr_high, prev_close) - np.minimum(curr_low, prev_close)
    
    # Calculate average values for each period
    period1_len = len(bp) - period1 + 1
    period2_len = len(bp) - period2 + 1
    period3_len = len(bp) - period3 + 1
    
    min_len = min(period1_len, period2_len, period3_len)
    result_len = min_len
    
    # Initialize arrays for sum of BP and TR for each period
    sum_bp1 = np.zeros(result_len)
    sum_tr1 = np.zeros(result_len)
    
    sum_bp2 = np.zeros(result_len)
    sum_tr2 = np.zeros(result_len)
    
    sum_bp3 = np.zeros(result_len)
    sum_tr3 = np.zeros(result_len)
    
    # Calculate sums for each period
    for i in range(result_len):
        idx1 = i + len(bp) - result_len - period1 + 1
        idx2 = i + len(bp) - result_len - period2 + 1
        idx3 = i + len(bp) - result_len - period3 + 1
        
        sum_bp1[i] = np.sum(bp[idx1:idx1+period1])
        sum_tr1[i] = np.sum(tr[idx1:idx1+period1])
        
        sum_bp2[i] = np.sum(bp[idx2:idx2+period2])
        sum_tr2[i] = np.sum(tr[idx2:idx2+period2])
        
        sum_bp3[i] = np.sum(bp[idx3:idx3+period3])
        sum_tr3[i] = np.sum(tr[idx3:idx3+period3])
    
    # Calculate averages for each period
    avg1 = np.zeros_like(sum_bp1)
    avg2 = np.zeros_like(sum_bp2)
    avg3 = np.zeros_like(sum_bp3)
    
    # Handle division by zero
    nonzero_idx1 = sum_tr1 != 0
    nonzero_idx2 = sum_tr2 != 0
    nonzero_idx3 = sum_tr3 != 0
    
    avg1[nonzero_idx1] = sum_bp1[nonzero_idx1] / sum_tr1[nonzero_idx1]
    avg2[nonzero_idx2] = sum_bp2[nonzero_idx2] / sum_tr2[nonzero_idx2]
    avg3[nonzero_idx3] = sum_bp3[nonzero_idx3] / sum_tr3[nonzero_idx3]
    
    # Calculate Ultimate Oscillator
    uo = 100 * (weights[0] * avg1 + weights[1] * avg2 + weights[2] * avg3) / np.sum(weights)
    
    return uo


def awesome_oscillator(high: Union[pd.Series, np.ndarray], 
                      low: Union[pd.Series, np.ndarray], 
                      fast_period: int = 5, 
                      slow_period: int = 34) -> np.ndarray:
    """
    Calculate Awesome Oscillator
    
    Parameters:
    -----------
    high : Union[pd.Series, np.ndarray]
        High prices
    low : Union[pd.Series, np.ndarray]
        Low prices
    fast_period : int
        Period for the fast SMA
    slow_period : int
        Period for the slow SMA
        
    Returns:
    --------
    np.ndarray
        Awesome Oscillator values
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    
    if len(high) != len(low):
        raise ValueError("High and low arrays must have the same length")
    
    if len(high) <= slow_period:
        raise ValueError(f"Data length ({len(high)}) must be > slow_period ({slow_period})")
    
    # Calculate median price
    median_price = (high + low) / 2
    
    # Calculate fast and slow SMAs
    fast_sma = sma(median_price, fast_period)
    slow_sma = sma(median_price, slow_period)
    
    # Adjust the lengths to match
    min_len = min(len(fast_sma), len(slow_sma))
    
    # Calculate Awesome Oscillator
    ao = fast_sma[-min_len:] - slow_sma[-min_len:]
    
    return ao


def momentum(data: Union[pd.Series, np.ndarray], period: int = 14) -> np.ndarray:
    """
    Calculate the Momentum oscillator
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data (typically close prices)
    period : int
        Momentum period
        
    Returns:
    --------
    np.ndarray
        Momentum values
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    if len(data) <= period:
        raise ValueError(f"Data length ({len(data)}) must be > period ({period})")
    
    # Calculate momentum
    momentum_values = data[period:] - data[:-period]
    
    return momentum_values


def rate_of_change(data: Union[pd.Series, np.ndarray], period: int = 14) -> np.ndarray:
    """
    Calculate Rate of Change (ROC)
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data (typically close prices)
    period : int
        ROC period
        
    Returns:
    --------
    np.ndarray
        ROC values (percentage change)
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    if len(data) <= period:
        raise ValueError(f"Data length ({len(data)}) must be > period ({period})")
    
    # Calculate ROC
    roc_values = np.zeros(len(data) - period)
    
    # Handle division by zero
    reference_prices = data[:-period]
    nonzero_idx = reference_prices != 0
    
    roc_values[nonzero_idx] = 100 * (data[period:][nonzero_idx] - reference_prices[nonzero_idx]) / reference_prices[nonzero_idx]
    
    return roc_values


def tsi(data: Union[pd.Series, np.ndarray], 
        long_period: int = 25, 
        short_period: int = 13,
        signal_period: int = 7) -> Dict[str, np.ndarray]:
    """
    Calculate True Strength Index (TSI)
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Price data (typically close prices)
    long_period : int
        Long EMA period
    short_period : int
        Short EMA period
    signal_period : int
        Signal line EMA period
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing 'tsi' and 'signal' values
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    if len(data) <= long_period + short_period:
        raise ValueError(f"Data length ({len(data)}) must be > long_period + short_period ({long_period + short_period})")
    
    # Calculate momentum (price changes)
    momentum_values = np.diff(data)
    
    # Calculate first smoothing (EMA of momentum)
    momentum_ema_long = ema(momentum_values, long_period)
    
    # Calculate second smoothing (EMA of first smoothing)
    double_smoothed_momentum = ema(momentum_ema_long, short_period)
    
    # Calculate absolute momentum
    abs_momentum = np.abs(momentum_values)
    
    # Calculate first smoothing of absolute momentum
    abs_momentum_ema_long = ema(abs_momentum, long_period)
    
    # Calculate second smoothing of absolute momentum
    double_smoothed_abs_momentum = ema(abs_momentum_ema_long, short_period)
    
    # Adjust the lengths to match
    min_len = min(len(double_smoothed_momentum), len(double_smoothed_abs_momentum))
    
    # Calculate TSI
    tsi_values = np.zeros(min_len)
    
    # Handle division by zero
    nonzero_idx = double_smoothed_abs_momentum[-min_len:] != 0
    tsi_values[nonzero_idx] = 100 * (double_smoothed_momentum[-min_len:][nonzero_idx] / 
                                    double_smoothed_abs_momentum[-min_len:][nonzero_idx])
    
    # Calculate signal line
    signal_line = ema(tsi_values, signal_period)
    
    # Adjust TSI length to match signal line length
    tsi_values = tsi_values[-len(signal_line):]
    
    return {
        'tsi': tsi_values,
        'signal': signal_line
    }


# Helper function for testing
def _calculate_all_oscillators(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all oscillators and add them to the dataframe
    
    Parameters:
    -----------
    ohlcv_df : pd.DataFrame
        DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with all oscillators added
    """
    df = ohlcv_df.copy()
    
    # Add RSI
    rsi_vals = rsi(df['close'])
    df['rsi'] = pd.Series(rsi_vals, index=df.index[-len(rsi_vals):])
    
    # Add Stochastic Oscillator
    stoch = stochastic(df['high'], df['low'], df['close'])
    df['stoch_k'] = pd.Series(stoch['k'], index=df.index[-len(stoch['k']):])
    df['stoch_d'] = pd.Series(stoch['d'], index=df.index[-len(stoch['d']):])
    
    # Add CCI
    cci_vals = cci(df['high'], df['low'], df['close'])
    df['cci'] = pd.Series(cci_vals, index=df.index[-len(cci_vals):])
    
    # Add Williams %R
    williams_vals = williams_r(df['high'], df['low'], df['close'])
    df['williams_r'] = pd.Series(williams_vals, index=df.index[-len(williams_vals):])
    
    # Add Awesome Oscillator
    ao_vals = awesome_oscillator(df['high'], df['low'])
    df['ao'] = pd.Series(ao_vals, index=df.index[-len(ao_vals):])
    
    # Add Momentum
    mom_vals = momentum(df['close'])
    df['momentum'] = pd.Series(mom_vals, index=df.index[-len(mom_vals):])
    
    # Add Rate of Change
    roc_vals = rate_of_change(df['close'])
    df['roc'] = pd.Series(roc_vals, index=df.index[-len(roc_vals):])
    
    return df
