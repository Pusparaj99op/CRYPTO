import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class LeverageManager:
    """
    LeverageManager is responsible for determining and adjusting the 
    appropriate leverage for trading positions based on market conditions,
    volatility, and risk parameters.
    """
    
    def __init__(
        self,
        max_leverage: float = 10.0,
        default_leverage: float = 3.0,
        vol_scaling_enabled: bool = True,
        vol_window: int = 20,
        vol_leverage_ratio: float = 0.5,
        min_leverage: float = 1.0,
        market_stress_threshold: float = 0.25,
        adapt_to_exchange_limits: bool = True,
        exchange_leverage_limits: Dict[str, float] = None,
        vol_metric: str = "atr",  # "atr" or "stddev"
        bankruptcy_protection: bool = True,
        bankruptcy_buffer: float = 0.15,
    ):
        """
        Initialize the LeverageManager with configuration parameters.
        
        Args:
            max_leverage: Maximum allowed leverage
            default_leverage: Default leverage to use when no specific calculation is done
            vol_scaling_enabled: Whether to adjust leverage based on volatility
            vol_window: Lookback window for volatility calculations
            vol_leverage_ratio: Ratio to apply to volatility when scaling leverage
            min_leverage: Minimum leverage allowed
            market_stress_threshold: Threshold for market stress indicator to reduce leverage
            adapt_to_exchange_limits: Whether to respect exchange-specific leverage limits
            exchange_leverage_limits: Dictionary mapping exchange names to their max leverage
            vol_metric: Volatility metric to use ("atr" or "stddev")
            bankruptcy_protection: Whether to implement bankruptcy protection measures
            bankruptcy_buffer: Additional buffer for bankruptcy protection
        """
        self.max_leverage = max_leverage
        self.default_leverage = default_leverage
        self.vol_scaling_enabled = vol_scaling_enabled
        self.vol_window = vol_window
        self.vol_leverage_ratio = vol_leverage_ratio
        self.min_leverage = min_leverage
        self.market_stress_threshold = market_stress_threshold
        self.adapt_to_exchange_limits = adapt_to_exchange_limits
        self.exchange_leverage_limits = exchange_leverage_limits or {"default": 100.0}
        self.vol_metric = vol_metric
        self.bankruptcy_protection = bankruptcy_protection
        self.bankruptcy_buffer = bankruptcy_buffer
        
        self.current_leverage = default_leverage
        self.historical_volatility: Dict[str, float] = {}
        self.market_stress_indicators: Dict[str, float] = {}
    
    def update_price_history(self, symbol: str, price_data: pd.DataFrame) -> None:
        """
        Update price history for volatility calculations.
        
        Args:
            symbol: Trading pair symbol
            price_data: DataFrame containing OHLCV data
        """
        if self.vol_metric == "atr":
            self.historical_volatility[symbol] = self._calculate_atr(price_data, self.vol_window)
        else:
            self.historical_volatility[symbol] = self._calculate_stddev(price_data, self.vol_window)
    
    def _calculate_atr(self, price_data: pd.DataFrame, window: int) -> float:
        """
        Calculate Average True Range (ATR) as volatility metric.
        
        Args:
            price_data: DataFrame with OHLCV data
            window: Lookback period for ATR calculation
            
        Returns:
            float: ATR value
        """
        if len(price_data) < 2:
            return 0.0
            
        highs = price_data['high'].values
        lows = price_data['low'].values
        closes = price_data['close'].values
        
        # Previous closes (shifted by 1)
        prev_closes = np.roll(closes, 1)
        prev_closes[0] = closes[0]
        
        tr1 = np.abs(highs - lows)
        tr2 = np.abs(highs - prev_closes)
        tr3 = np.abs(lows - prev_closes)
        
        true_ranges = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(true_ranges[-window:]) if len(true_ranges) >= window else np.mean(true_ranges)
        
        return atr
    
    def _calculate_stddev(self, price_data: pd.DataFrame, window: int) -> float:
        """
        Calculate standard deviation of returns as volatility metric.
        
        Args:
            price_data: DataFrame with OHLCV data
            window: Lookback period for volatility calculation
            
        Returns:
            float: Standard deviation value
        """
        if len(price_data) < 2:
            return 0.0
            
        close_prices = price_data['close'].values
        returns = np.diff(np.log(close_prices))
        
        if len(returns) >= window:
            stddev = np.std(returns[-window:]) * np.sqrt(252)  # Annualized
        else:
            stddev = np.std(returns) * np.sqrt(252)  # Annualized
            
        return stddev
    
    def update_market_stress(self, symbol: str, stress_level: float) -> None:
        """
        Update market stress indicator for a symbol.
        
        Args:
            symbol: Trading pair symbol
            stress_level: Market stress indicator (0-1 scale)
        """
        self.market_stress_indicators[symbol] = max(0.0, min(1.0, stress_level))
    
    def calculate_leverage(
        self, 
        symbol: str, 
        entry_price: float, 
        stop_loss: Optional[float] = None,
        exchange: str = "default"
    ) -> float:
        """
        Calculate appropriate leverage based on market conditions and risk parameters.
        
        Args:
            symbol: Trading pair symbol
            entry_price: Entry price for the position
            stop_loss: Stop loss price (if None, volatility-based leverage is used)
            exchange: Exchange name to apply leverage limits
            
        Returns:
            float: Calculated leverage
        """
        # Start with default leverage
        leverage = self.default_leverage
        
        # Adjust based on volatility if enabled
        if self.vol_scaling_enabled and symbol in self.historical_volatility:
            vol = self.historical_volatility[symbol]
            if vol > 0:
                vol_adjusted_leverage = 1.0 / (vol * self.vol_leverage_ratio)
                leverage = min(leverage, vol_adjusted_leverage)
        
        # Adjust based on stop loss if provided
        if stop_loss is not None and stop_loss > 0 and entry_price > 0:
            risk_percent = abs(entry_price - stop_loss) / entry_price
            if risk_percent > 0:
                stop_loss_leverage = 1.0 / risk_percent
                
                if self.bankruptcy_protection:
                    # Add safety buffer to prevent liquidation
                    buffer_factor = 1.0 - self.bankruptcy_buffer
                    stop_loss_leverage *= buffer_factor
                    
                leverage = min(leverage, stop_loss_leverage)
        
        # Reduce leverage during high market stress
        if symbol in self.market_stress_indicators:
            stress = self.market_stress_indicators[symbol]
            if stress > self.market_stress_threshold:
                stress_factor = 1.0 - (stress - self.market_stress_threshold)
                leverage *= max(0.1, stress_factor)
        
        # Apply exchange-specific leverage limits if enabled
        if self.adapt_to_exchange_limits and exchange in self.exchange_leverage_limits:
            exchange_limit = self.exchange_leverage_limits[exchange]
            leverage = min(leverage, exchange_limit)
        
        # Apply general limits
        leverage = max(self.min_leverage, min(self.max_leverage, leverage))
        
        # Round to 2 decimal places for practical application
        leverage = round(leverage, 2)
        
        self.current_leverage = leverage
        logger.info(f"Calculated leverage for {symbol}: {leverage}")
        return leverage
    
    def get_current_leverage(self) -> float:
        """
        Get the current leverage setting.
        
        Returns:
            float: Current leverage
        """
        return self.current_leverage
    
    def adjust_for_correlation(
        self, 
        base_leverage: float, 
        correlation_matrix: pd.DataFrame, 
        current_positions: Dict[str, float]
    ) -> float:
        """
        Adjust leverage based on correlation with existing positions.
        
        Args:
            base_leverage: Initial calculated leverage
            correlation_matrix: DataFrame of correlations between assets
            current_positions: Dictionary of current position sizes by symbol
            
        Returns:
            float: Adjusted leverage accounting for portfolio correlation
        """
        if not current_positions or len(current_positions) == 0:
            return base_leverage
            
        # Calculate weighted average correlation
        total_exposure = sum(abs(size) for size in current_positions.values())
        if total_exposure == 0:
            return base_leverage
            
        weighted_correlation = 0.0
        for symbol, size in current_positions.items():
            if symbol in correlation_matrix.columns:
                weight = abs(size) / total_exposure
                for other_symbol, other_size in current_positions.items():
                    if other_symbol in correlation_matrix.index:
                        corr = correlation_matrix.loc[symbol, other_symbol]
                        other_weight = abs(other_size) / total_exposure
                        weighted_correlation += corr * weight * other_weight
        
        # Adjust leverage based on correlation (higher correlation = lower leverage)
        if weighted_correlation > 0.5:
            correlation_factor = 1.0 - (weighted_correlation - 0.5)
            adjusted_leverage = base_leverage * max(0.5, correlation_factor)
            return max(self.min_leverage, min(self.max_leverage, adjusted_leverage))
        
        return base_leverage 