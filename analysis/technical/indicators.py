"""
Standard Technical Indicators

This module provides implementations of standard technical indicators
used in financial market analysis and algorithmic trading systems.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional, Dict, List
import talib


def moving_average(data: np.ndarray, window: int, ma_type: str = 'simple') -> np.ndarray:
    """
    Calculate various types of moving averages
    
    Parameters:
    -----------
    data : np.ndarray
        Price or other data to compute moving average on
    window : int
        Window size for the moving average calculation
    ma_type : str
        Type of moving average: 'simple', 'exponential', 'weighted', or 'triangular'
        
    Returns:
    --------
    np.ndarray
        Moving average values
    """
    if len(data) < window:
        return np.full_like(data, np.nan)
    
    result = np.full_like(data, np.nan)
    
    if ma_type.lower() == 'simple':
        # Simple Moving Average (SMA)
        for i in range(window-1, len(data)):
            result[i] = np.mean(data[i-window+1:i+1])
            
    elif ma_type.lower() == 'exponential':
        # Exponential Moving Average (EMA)
        alpha = 2 / (window + 1)
        result[window-1] = np.mean(data[:window])
        for i in range(window, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            
    elif ma_type.lower() == 'weighted':
        # Weighted Moving Average (WMA)
        weights = np.arange(1, window + 1)
        sum_weights = np.sum(weights)
        
        for i in range(window-1, len(data)):
            result[i] = np.sum(data[i-window+1:i+1] * weights) / sum_weights
            
    elif ma_type.lower() == 'triangular':
        # Triangular Moving Average (TMA)
        # First compute SMA
        sma = np.full_like(data, np.nan)
        for i in range(window-1, len(data)):
            sma[i] = np.mean(data[i-window+1:i+1])
        
        # Then compute SMA of SMA
        for i in range(2*window-2, len(data)):
            result[i] = np.mean(sma[i-window+1:i+1])
            
    else:
        raise ValueError(f"Unsupported moving average type: {ma_type}")
    
    return result


def bollinger_bands(data: np.ndarray, window: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands®
    
    Parameters:
    -----------
    data : np.ndarray
        Price data, typically close prices
    window : int
        Window size for moving average calculation
    num_std : float
        Number of standard deviations for the bands
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Middle band (SMA), upper band, and lower band
    """
    # Calculate middle band (SMA)
    middle_band = moving_average(data, window, 'simple')
    
    # Calculate standard deviation
    rolling_std = np.full_like(data, np.nan)
    for i in range(window-1, len(data)):
        rolling_std[i] = np.std(data[i-window+1:i+1], ddof=1)
    
    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    
    return middle_band, upper_band, lower_band


def macd(data: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Moving Average Convergence Divergence (MACD)
    
    Parameters:
    -----------
    data : np.ndarray
        Price data, typically close prices
    fast_period : int
        Period for the fast EMA
    slow_period : int
        Period for the slow EMA
    signal_period : int
        Period for the signal line
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        MACD line, signal line, and histogram
    """
    # Calculate fast and slow EMAs
    fast_ema = moving_average(data, fast_period, 'exponential')
    slow_ema = moving_average(data, slow_period, 'exponential')
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = moving_average(macd_line, signal_period, 'exponential')
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def rsi(data: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI)
    
    Parameters:
    -----------
    data : np.ndarray
        Price data, typically close prices
    window : int
        Lookback period
        
    Returns:
    --------
    np.ndarray
        RSI values ranging from 0 to 100
    """
    # Calculate price changes
    delta = np.zeros_like(data)
    delta[1:] = data[1:] - data[:-1]
    
    # Separate gains and losses
    gains = np.copy(delta)
    losses = np.copy(delta)
    
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = np.abs(losses)
    
    # Initialize RSI array
    rsi_values = np.full_like(data, np.nan)
    
    # Calculate initial averages
    avg_gain = np.mean(gains[1:window+1])
    avg_loss = np.mean(losses[1:window+1])
    
    # Initial RSI value
    if avg_loss == 0:
        rsi_values[window] = 100
    else:
        rs = avg_gain / avg_loss
        rsi_values[window] = 100 - (100 / (1 + rs))
    
    # Calculate rest of RSI values
    for i in range(window+1, len(data)):
        avg_gain = ((avg_gain * (window - 1)) + gains[i]) / window
        avg_loss = ((avg_loss * (window - 1)) + losses[i]) / window
        
        if avg_loss == 0:
            rsi_values[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100 - (100 / (1 + rs))
    
    return rsi_values


def average_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Calculate Average True Range (ATR)
    
    Parameters:
    -----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
    window : int
        Lookback period
        
    Returns:
    --------
    np.ndarray
        ATR values
    """
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("High, low, and close arrays must be the same length")
    
    # Calculate true range
    tr = np.zeros_like(high)
    
    # First value is just the high-low range
    tr[0] = high[0] - low[0]
    
    # Rest of the values consider previous close
    for i in range(1, len(high)):
        tr[i] = max(
            high[i] - low[i],                  # Current high-low range
            abs(high[i] - close[i-1]),         # Current high to previous close
            abs(low[i] - close[i-1])           # Current low to previous close
        )
    
    # Calculate ATR
    atr = np.full_like(high, np.nan)
    
    # Initial ATR value (simple average of first window periods)
    atr[window-1] = np.mean(tr[:window])
    
    # Calculate rest of ATR values using smoothing
    for i in range(window, len(high)):
        atr[i] = ((atr[i-1] * (window - 1)) + tr[i]) / window
    
    return atr


def stochastic_oscillator(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                          k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Stochastic Oscillator
    
    Parameters:
    -----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
    k_period : int
        %K period
    d_period : int
        %D period (moving average of %K)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        %K and %D values
    """
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("High, low, and close arrays must be the same length")
    
    # Initialize arrays
    k = np.full_like(close, np.nan)
    d = np.full_like(close, np.nan)
    
    # Calculate %K
    for i in range(k_period-1, len(close)):
        window_low = np.min(low[i-k_period+1:i+1])
        window_high = np.max(high[i-k_period+1:i+1])
        
        if window_high == window_low:
            k[i] = 50  # Avoid division by zero
        else:
            k[i] = 100 * (close[i] - window_low) / (window_high - window_low)
    
    # Calculate %D (moving average of %K)
    for i in range(k_period+d_period-2, len(close)):
        d[i] = np.mean(k[i-d_period+1:i+1])
    
    return k, d


def on_balance_volume(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Calculate On Balance Volume (OBV)
    
    Parameters:
    -----------
    close : np.ndarray
        Close prices
    volume : np.ndarray
        Volume data
        
    Returns:
    --------
    np.ndarray
        OBV values
    """
    if len(close) != len(volume):
        raise ValueError("Close and volume arrays must be the same length")
    
    obv = np.zeros_like(close)
    
    # First value is just the first volume
    obv[0] = volume[0]
    
    # Calculate OBV for the rest of the values
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            # Price up - add volume
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            # Price down - subtract volume
            obv[i] = obv[i-1] - volume[i]
        else:
            # Price unchanged - keep OBV the same
            obv[i] = obv[i-1]
    
    return obv


def keltner_channels(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                    ema_period: int = 20, atr_period: int = 10, 
                    multiplier: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Keltner Channels
    
    Parameters:
    -----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
    ema_period : int
        Period for the EMA (middle line)
    atr_period : int
        Period for the ATR calculation
    multiplier : float
        Multiplier for the ATR
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Middle line, upper band, and lower band
    """
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("High, low, and close arrays must be the same length")
    
    # Calculate middle line (EMA of close)
    middle_line = moving_average(close, ema_period, 'exponential')
    
    # Calculate ATR
    atr_values = average_true_range(high, low, close, atr_period)
    
    # Calculate upper and lower bands
    upper_band = middle_line + (atr_values * multiplier)
    lower_band = middle_line - (atr_values * multiplier)
    
    return middle_line, upper_band, lower_band


def parabolic_sar(high: np.ndarray, low: np.ndarray, 
                 af_start: float = 0.02, af_increment: float = 0.02, 
                 af_max: float = 0.2) -> np.ndarray:
    """
    Calculate Parabolic SAR (Stop and Reverse)
    
    Parameters:
    -----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    af_start : float
        Starting acceleration factor
    af_increment : float
        Acceleration factor increment
    af_max : float
        Maximum acceleration factor
        
    Returns:
    --------
    np.ndarray
        PSAR values
    """
    if len(high) != len(low):
        raise ValueError("High and low arrays must be the same length")
    
    # Initialize outputs
    psar = np.full_like(high, np.nan)
    
    if len(high) < 2:
        return psar
    
    # Initialize variables
    uptrend = True  # Start with an uptrend assumption
    ep = high[0]    # Extreme point
    psar[0] = low[0]  # Start with first low in an uptrend
    af = af_start    # Current acceleration factor
    
    # Calculate PSAR values
    for i in range(1, len(high)):
        # Previous PSAR value
        prev_psar = psar[i-1]
        
        # Calculate current PSAR
        psar[i] = prev_psar + af * (ep - prev_psar)
        
        # Check if we need to switch trend
        if uptrend:
            # In uptrend, PSAR is below price
            if psar[i] > low[i]:
                uptrend = False
                psar[i] = ep  # Switch PSAR to the extreme point
                ep = low[i]   # New extreme point is current low
                af = af_start
            else:
                # Still in uptrend
                if high[i] > ep:
                    ep = high[i]  # Update extreme point
                    af = min(af + af_increment, af_max)  # Increase AF
        else:
            # In downtrend, PSAR is above price
            if psar[i] < high[i]:
                uptrend = True
                psar[i] = ep  # Switch PSAR to the extreme point
                ep = high[i]  # New extreme point is current high
                af = af_start
            else:
                # Still in downtrend
                if low[i] < ep:
                    ep = low[i]  # Update extreme point
                    af = min(af + af_increment, af_max)  # Increase AF
        
        # Ensure PSAR is below/above price in respective trends
        if uptrend:
            psar[i] = min(psar[i], low[i-1], low[i] if i > 1 else low[i])
        else:
            psar[i] = max(psar[i], high[i-1], high[i] if i > 1 else high[i])
    
    return psar


def ichimoku_cloud(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  tenkan_period: int = 9, kijun_period: int = 26,
                  senkou_span_b_period: int = 52, displacement: int = 26) -> Dict[str, np.ndarray]:
    """
    Calculate Ichimoku Cloud components
    
    Parameters:
    -----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
    tenkan_period : int
        Tenkan-sen (Conversion Line) period
    kijun_period : int
        Kijun-sen (Base Line) period
    senkou_span_b_period : int
        Senkou Span B (Leading Span B) period
    displacement : int
        Displacement for Senkou Span A and B (Kumo/Cloud)
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary with Ichimoku components:
        - tenkan_sen: Conversion Line
        - kijun_sen: Base Line
        - senkou_span_a: Leading Span A
        - senkou_span_b: Leading Span B
        - chikou_span: Lagging Span
    """
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("High, low, and close arrays must be the same length")
    
    # Initialize arrays
    tenkan_sen = np.full_like(close, np.nan)
    kijun_sen = np.full_like(close, np.nan)
    senkou_span_a = np.full_like(close, np.nan)
    senkou_span_b = np.full_like(close, np.nan)
    chikou_span = np.full_like(close, np.nan)
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past tenkan_period
    for i in range(tenkan_period-1, len(close)):
        highest_high = np.max(high[i-tenkan_period+1:i+1])
        lowest_low = np.min(low[i-tenkan_period+1:i+1])
        tenkan_sen[i] = (highest_high + lowest_low) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past kijun_period
    for i in range(kijun_period-1, len(close)):
        highest_high = np.max(high[i-kijun_period+1:i+1])
        lowest_low = np.min(low[i-kijun_period+1:i+1])
        kijun_sen[i] = (highest_high + lowest_low) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 displaced forward
    for i in range(kijun_period-1, len(close)):
        senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the past senkou_span_b_period
    for i in range(senkou_span_b_period-1, len(close)):
        highest_high = np.max(high[i-senkou_span_b_period+1:i+1])
        lowest_low = np.min(low[i-senkou_span_b_period+1:i+1])
        senkou_span_b[i] = (highest_high + lowest_low) / 2
    
    # Calculate Chikou Span (Lagging Span): Current close displaced backward
    for i in range(len(close)):
        if i + displacement < len(close):
            chikou_span[i+displacement] = close[i]
    
    # Displace Senkou Spans forward
    senkou_span_a_displaced = np.full_like(close, np.nan)
    senkou_span_b_displaced = np.full_like(close, np.nan)
    
    for i in range(len(close)):
        if i + displacement < len(close):
            senkou_span_a_displaced[i+displacement] = senkou_span_a[i]
            senkou_span_b_displaced[i+displacement] = senkou_span_b[i]
    
    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a_displaced,
        'senkou_span_b': senkou_span_b_displaced,
        'chikou_span': chikou_span
    }


def volume_weighted_average_price(high: np.ndarray, low: np.ndarray, 
                                 close: np.ndarray, volume: np.ndarray, 
                                 window: Optional[int] = None) -> np.ndarray:
    """
    Calculate Volume Weighted Average Price (VWAP)
    
    Parameters:
    -----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
    volume : np.ndarray
        Volume data
    window : Optional[int]
        Rolling window for VWAP calculation. If None, calculates from the beginning
        
    Returns:
    --------
    np.ndarray
        VWAP values
    """
    if len(high) != len(low) or len(high) != len(close) or len(high) != len(volume):
        raise ValueError("Price and volume arrays must be the same length")
    
    # Calculate typical price: (high + low + close) / 3
    typical_price = (high + low + close) / 3
    
    # Calculate price * volume
    price_volume = typical_price * volume
    
    # Initialize VWAP array
    vwap = np.full_like(close, np.nan)
    
    if window is None:
        # Calculate cumulative VWAP from the beginning
        cum_price_volume = np.cumsum(price_volume)
        cum_volume = np.cumsum(volume)
        
        # Avoid division by zero
        valid_indices = cum_volume > 0
        vwap[valid_indices] = cum_price_volume[valid_indices] / cum_volume[valid_indices]
    else:
        # Calculate rolling VWAP
        for i in range(window-1, len(close)):
            sum_price_volume = np.sum(price_volume[i-window+1:i+1])
            sum_volume = np.sum(volume[i-window+1:i+1])
            
            if sum_volume > 0:
                vwap[i] = sum_price_volume / sum_volume
    
    return vwap


def money_flow_index(high: np.ndarray, low: np.ndarray, 
                    close: np.ndarray, volume: np.ndarray, 
                    period: int = 14) -> np.ndarray:
    """
    Calculate Money Flow Index (MFI)
    
    Parameters:
    -----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
    volume : np.ndarray
        Volume data
    period : int
        Lookback period
        
    Returns:
    --------
    np.ndarray
        MFI values ranging from 0 to 100
    """
    if len(high) != len(low) or len(high) != len(close) or len(high) != len(volume):
        raise ValueError("Price and volume arrays must be the same length")
    
    # Calculate typical price: (high + low + close) / 3
    typical_price = (high + low + close) / 3
    
    # Calculate raw money flow
    raw_money_flow = typical_price * volume
    
    # Initialize arrays
    money_flow_positive = np.zeros_like(typical_price)
    money_flow_negative = np.zeros_like(typical_price)
    
    # Separate positive and negative money flow
    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]:
            money_flow_positive[i] = raw_money_flow[i]
            money_flow_negative[i] = 0
        elif typical_price[i] < typical_price[i-1]:
            money_flow_positive[i] = 0
            money_flow_negative[i] = raw_money_flow[i]
        else:
            money_flow_positive[i] = 0
            money_flow_negative[i] = 0
    
    # Calculate MFI
    mfi = np.full_like(close, np.nan)
    
    for i in range(period, len(close)):
        positive_sum = np.sum(money_flow_positive[i-period+1:i+1])
        negative_sum = np.sum(money_flow_negative[i-period+1:i+1])
        
        if negative_sum == 0:
            mfi[i] = 100  # Avoid division by zero
        else:
            money_ratio = positive_sum / negative_sum
            mfi[i] = 100 - (100 / (1 + money_ratio))
    
    return mfi


def williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Williams %R
    
    Parameters:
    -----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
    period : int
        Lookback period
        
    Returns:
    --------
    np.ndarray
        Williams %R values ranging from -100 to 0
    """
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("High, low, and close arrays must be the same length")
    
    # Initialize Williams %R array
    williams_r_values = np.full_like(close, np.nan)
    
    # Calculate Williams %R
    for i in range(period-1, len(close)):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            williams_r_values[i] = -50  # Avoid division by zero
        else:
            williams_r_values[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return williams_r_values


def rate_of_change(data: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Rate of Change (ROC)
    
    Parameters:
    -----------
    data : np.ndarray
        Price data
    period : int
        Lookback period
        
    Returns:
    --------
    np.ndarray
        ROC values as percentages
    """
    # Initialize ROC array
    roc = np.full_like(data, np.nan)
    
    # Calculate ROC
    for i in range(period, len(data)):
        roc[i] = ((data[i] - data[i-period]) / data[i-period]) * 100
    
    return roc


def accumulation_distribution_line(high: np.ndarray, low: np.ndarray, 
                                  close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Calculate Accumulation/Distribution Line
    
    Parameters:
    -----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
    volume : np.ndarray
        Volume data
        
    Returns:
    --------
    np.ndarray
        Accumulation/Distribution Line values
    """
    if len(high) != len(low) or len(high) != len(close) or len(high) != len(volume):
        raise ValueError("Price and volume arrays must be the same length")
    
    # Calculate Money Flow Multiplier
    mfm = np.zeros_like(close)
    for i in range(len(close)):
        if high[i] == low[i]:
            mfm[i] = 0  # Avoid division by zero
        else:
            mfm[i] = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
    
    # Calculate Money Flow Volume
    mfv = mfm * volume
    
    # Calculate ADL
    adl = np.zeros_like(close)
    adl[0] = mfv[0]
    
    for i in range(1, len(close)):
        adl[i] = adl[i-1] + mfv[i]
    
    return adl


def commodity_channel_index(high: np.ndarray, low: np.ndarray, 
                           close: np.ndarray, period: int = 20, 
                           factor: float = 0.015) -> np.ndarray:
    """
    Calculate Commodity Channel Index (CCI)
    
    Parameters:
    -----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
    period : int
        Lookback period
    factor : float
        Scaling factor, typically 0.015
        
    Returns:
    --------
    np.ndarray
        CCI values
    """
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("High, low, and close arrays must be the same length")
    
    # Calculate typical price: (high + low + close) / 3
    typical_price = (high + low + close) / 3
    
    # Initialize arrays
    cci = np.full_like(close, np.nan)
    sma_tp = np.full_like(close, np.nan)
    mad = np.full_like(close, np.nan)
    
    # Calculate SMA of typical price
    for i in range(period-1, len(typical_price)):
        sma_tp[i] = np.mean(typical_price[i-period+1:i+1])
    
    # Calculate Mean Absolute Deviation (MAD)
    for i in range(period-1, len(typical_price)):
        mad[i] = np.mean(np.abs(typical_price[i-period+1:i+1] - sma_tp[i]))
    
    # Calculate CCI
    for i in range(period-1, len(typical_price)):
        if mad[i] == 0:
            cci[i] = 0  # Avoid division by zero
        else:
            cci[i] = (typical_price[i] - sma_tp[i]) / (factor * mad[i])
    
    return cci


def standard_deviation(data: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Calculate Rolling Standard Deviation
    
    Parameters:
    -----------
    data : np.ndarray
        Price or other data
    window : int
        Window size for calculation
        
    Returns:
    --------
    np.ndarray
        Standard deviation values
    """
    # Initialize standard deviation array
    std_dev = np.full_like(data, np.nan)
    
    # Calculate rolling standard deviation
    for i in range(window-1, len(data)):
        std_dev[i] = np.std(data[i-window+1:i+1], ddof=1)
    
    return std_dev


def fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """
    Calculate Fibonacci retracement levels
    
    Parameters:
    -----------
    high : float
        Highest price in the trend
    low : float
        Lowest price in the trend
        
    Returns:
    --------
    Dict[str, float]
        Dictionary with Fibonacci retracement levels
    """
    diff = high - low
    
    levels = {
        "0.0": low,
        "0.236": low + 0.236 * diff,
        "0.382": low + 0.382 * diff,
        "0.5": low + 0.5 * diff,
        "0.618": low + 0.618 * diff,
        "0.786": low + 0.786 * diff,
        "1.0": high,
        "1.272": high + 0.272 * diff,
        "1.618": high + 0.618 * diff,
        "2.0": high + diff,
        "2.618": high + 1.618 * diff
    }
    
    return levels
