"""
Feature Engineering Module for Cryptocurrency Trading

This module provides advanced feature creation and engineering tools
specifically designed for cryptocurrency market data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Callable
from scipy import stats


class FeatureEngineer:
    """Class for creating advanced trading features from time series data."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.features = {}
        
    def create_technical_features(self, df: pd.DataFrame, 
                                  price_col: str = 'close',
                                  volume_col: Optional[str] = 'volume') -> pd.DataFrame:
        """
        Create common technical indicators as features.
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for price data
            volume_col: Column name for volume data
            
        Returns:
            DataFrame with added technical features
        """
        result = df.copy()
        
        # Moving averages
        for window in [7, 14, 21, 50, 200]:
            result[f'ma_{window}'] = result[price_col].rolling(window=window).mean()
            
        # Relative position
        for window in [7, 14, 21]:
            result[f'rp_{window}'] = (result[price_col] - result[price_col].rolling(window).min()) / \
                               (result[price_col].rolling(window).max() - result[price_col].rolling(window).min())
        
        # Volatility features
        for window in [7, 14, 21]:
            result[f'volatility_{window}'] = result[price_col].rolling(window).std() / result[price_col].rolling(window).mean()
        
        # Volume-based features
        if volume_col in df.columns:
            # Volume moving averages
            for window in [7, 14, 21]:
                result[f'volume_ma_{window}'] = result[volume_col].rolling(window=window).mean()
            
            # Volume relative to moving average
            result['volume_rel_ma'] = result[volume_col] / result[volume_col].rolling(window=7).mean()
            
            # On-balance volume
            result['obv'] = (np.sign(result[price_col].diff()) * result[volume_col]).cumsum()
        
        # Return features
        for window in [1, 3, 7, 14, 21]:
            result[f'return_{window}d'] = result[price_col].pct_change(window)
        
        return result
    
    def create_cyclic_features(self, df: pd.DataFrame, datetime_col: str = 'timestamp') -> pd.DataFrame:
        """
        Create cyclic features from datetime components.
        
        Args:
            df: DataFrame with datetime index or column
            datetime_col: Name of datetime column if not using index
            
        Returns:
            DataFrame with added cyclic features
        """
        result = df.copy()
        
        # Ensure datetime format
        if datetime_col in result.columns:
            dt = pd.to_datetime(result[datetime_col])
        else:
            dt = pd.to_datetime(result.index)
        
        # Hour of day - sin/cos encoding for cyclical nature
        result['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
        result['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
        
        # Day of week - sin/cos encoding
        result['day_of_week_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        result['day_of_week_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
        
        # Month - sin/cos encoding
        result['month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
        result['month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)
        
        return result
    
    def create_relationship_features(self, df: pd.DataFrame, related_assets: Dict[str, pd.DataFrame], 
                                    base_col: str = 'close') -> pd.DataFrame:
        """
        Create features based on relationships with other assets.
        
        Args:
            df: Main DataFrame
            related_assets: Dictionary of DataFrames for related assets
            base_col: Column name to use for relationships
            
        Returns:
            DataFrame with added relationship features
        """
        result = df.copy()
        
        for asset_name, asset_df in related_assets.items():
            # Ensure same index
            asset_data = asset_df.reindex(result.index, method='nearest')
            
            # Correlations (rolling windows)
            for window in [7, 14, 30]:
                result[f'corr_{asset_name}_{window}d'] = result[base_col].rolling(window).corr(asset_data[base_col])
            
            # Relative strength
            result[f'rel_strength_{asset_name}'] = result[base_col] / asset_data[base_col]
            
            # Divergence features
            for window in [7, 14]:
                result[f'divergence_{asset_name}_{window}d'] = (
                    result[base_col].pct_change(window) - asset_data[base_col].pct_change(window)
                )
        
        return result
    
    def create_statistical_features(self, df: pd.DataFrame, col: str = 'close') -> pd.DataFrame:
        """
        Create statistical features from a time series.
        
        Args:
            df: DataFrame with time series data
            col: Column to create statistical features from
            
        Returns:
            DataFrame with added statistical features
        """
        result = df.copy()
        
        # Statistical moments
        for window in [14, 30]:
            result[f'skew_{window}d'] = result[col].rolling(window).skew()
            result[f'kurtosis_{window}d'] = result[col].rolling(window).kurt()
        
        # Z-scores
        for window in [14, 30]:
            rolling_mean = result[col].rolling(window=window).mean()
            rolling_std = result[col].rolling(window=window).std()
            result[f'zscore_{window}d'] = (result[col] - rolling_mean) / rolling_std
        
        # Quantile features
        for window in [14, 30]:
            for q in [0.25, 0.5, 0.75]:
                result[f'quantile_{int(q*100)}_{window}d'] = result[col].rolling(window).quantile(q)
        
        return result
    
    def create_blockchain_features(self, df: pd.DataFrame, 
                                  blockchain_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from blockchain data for cryptocurrency analysis.
        
        Args:
            df: Market DataFrame
            blockchain_data: DataFrame with blockchain metrics
            
        Returns:
            DataFrame with added blockchain features
        """
        # Ensure both DataFrames have the same index
        result = df.copy()
        blockchain_aligned = blockchain_data.reindex(result.index, method='nearest')
        
        # Add blockchain metrics directly
        for col in blockchain_aligned.columns:
            result[f'blockchain_{col}'] = blockchain_aligned[col]
        
        # Create additional derived features
        if 'active_addresses' in blockchain_aligned and 'transaction_count' in blockchain_aligned:
            # Transactions per active address
            result['tx_per_address'] = (
                blockchain_aligned['transaction_count'] / blockchain_aligned['active_addresses']
            )
        
        if 'hash_rate' in blockchain_aligned:
            # Hash rate change
            result['hash_rate_change'] = blockchain_aligned['hash_rate'].pct_change(7)
        
        if 'difficulty' in blockchain_aligned:
            # Difficulty adjusted return (returns divided by change in mining difficulty)
            if 'close' in result:
                price_returns = result['close'].pct_change(7)
                difficulty_change = blockchain_aligned['difficulty'].pct_change(7)
                result['difficulty_adjusted_return'] = price_returns / (difficulty_change + 1e-10)
        
        return result
    
    def create_custom_feature(self, df: pd.DataFrame, 
                            feature_func: Callable[[pd.DataFrame], pd.Series],
                            feature_name: str) -> pd.DataFrame:
        """
        Create a custom feature using a provided function.
        
        Args:
            df: DataFrame to add feature to
            feature_func: Function that takes a DataFrame and returns a Series
            feature_name: Name for the new feature
            
        Returns:
            DataFrame with added custom feature
        """
        result = df.copy()
        result[feature_name] = feature_func(df)
        return result


# Utility functions for feature creation

def create_fractional_differentiation(series: pd.Series, d: float = 0.5, window: int = 100) -> pd.Series:
    """
    Apply fractional differentiation to a time series.
    
    Args:
        series: Time series to differentiate
        d: Differentiation order (between 0 and 1)
        window: Window size for approximation
        
    Returns:
        Fractionally differentiated series
    """
    # Create weights for the fractional differentiation
    weights = [1]
    for k in range(1, window):
        weight = weights[-1] * (k - 1 - d) / k
        weights.append(weight)
    
    weights = np.array(weights[::-1])
    
    # Apply weights to the series
    return series.rolling(window=window).apply(
        lambda x: np.sum(weights * x) if len(x) == window else np.nan
    )

def create_energy_features(ohlc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on price movement energy.
    
    Args:
        ohlc_df: DataFrame with OHLC data
        
    Returns:
        DataFrame with energy features
    """
    df = ohlc_df.copy()
    
    # Calculate price range and volume-weighted price range
    df['price_range'] = df['high'] - df['low']
    if 'volume' in df.columns:
        df['volume_weighted_range'] = df['price_range'] * df['volume']
    
    # Create kinetic energy feature (proportional to square of velocity)
    df['price_velocity'] = df['close'].diff()
    df['price_acceleration'] = df['price_velocity'].diff()
    df['kinetic_energy'] = df['price_velocity'] ** 2
    
    # Create potential energy feature
    max_20d = df['close'].rolling(20).max()
    df['potential_energy'] = max_20d - df['close']
    
    return df

def create_market_regime_features(ohlc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for detecting market regimes.
    
    Args:
        ohlc_df: DataFrame with OHLC data
        
    Returns:
        DataFrame with market regime features
    """
    df = ohlc_df.copy()
    
    # Volatility regimes
    for window in [14, 30]:
        vol = df['close'].pct_change().rolling(window).std()
        df[f'volatility_regime_{window}d'] = pd.qcut(vol, 4, labels=False, duplicates='drop')
    
    # Trend strength using ADX-like calculation
    for window in [14]:
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff(-1)
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            )
        )
        
        plus_di = 100 * plus_dm.rolling(window).sum() / tr.rolling(window).sum()
        minus_di = 100 * minus_dm.rolling(window).sum() / tr.rolling(window).sum()
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df[f'adx_{window}d'] = dx.rolling(window).mean()
        
    return df 