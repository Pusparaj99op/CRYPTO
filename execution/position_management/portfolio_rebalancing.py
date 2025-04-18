import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
import datetime

logger = logging.getLogger(__name__)

class PortfolioRebalancer:
    """
    PortfolioRebalancer handles portfolio rebalancing operations including threshold-based,
    calendar-based, and drift-based rebalancing strategies for crypto assets.
    """
    
    def __init__(
        self,
        target_weights: Dict[str, float] = None,
        rebalance_threshold: float = 0.05,
        rebalance_frequency: str = "weekly",  # "daily", "weekly", "monthly", "quarterly"
        min_rebalance_interval: int = 24,  # hours
        drift_calculation: str = "absolute",  # "absolute" or "relative"
        tax_efficiency: bool = False,
        trading_cost_model: Optional[Callable] = None,
        volatility_targeting: bool = False,
        vol_target: float = 0.15,  # 15% annualized volatility target
        vol_lookback: int = 20,  # days
        rebalance_blacklist: List[str] = None,
        allow_shorting: bool = False,
        max_turnover: float = 1.0,  # 100% turnover limit
    ):
        """
        Initialize the PortfolioRebalancer with configuration parameters.
        
        Args:
            target_weights: Dictionary mapping asset symbols to target portfolio weights
            rebalance_threshold: Threshold for drift-based rebalancing (as decimal)
            rebalance_frequency: Calendar frequency for systematic rebalancing
            min_rebalance_interval: Minimum hours between rebalancing operations
            drift_calculation: Method to calculate drift ("absolute" or "relative")
            tax_efficiency: Whether to optimize rebalancing for tax efficiency
            trading_cost_model: Optional function to calculate trading costs
            volatility_targeting: Whether to adjust weights based on volatility targeting
            vol_target: Annualized volatility target for the portfolio
            vol_lookback: Lookback period for volatility calculations
            rebalance_blacklist: List of assets that should not be rebalanced
            allow_shorting: Whether to allow negative weights (short positions)
            max_turnover: Maximum allowed portfolio turnover per rebalance operation
        """
        self.target_weights = target_weights or {}
        self.rebalance_threshold = rebalance_threshold
        self.rebalance_frequency = rebalance_frequency
        self.min_rebalance_interval = min_rebalance_interval
        self.drift_calculation = drift_calculation
        self.tax_efficiency = tax_efficiency
        self.trading_cost_model = trading_cost_model
        self.volatility_targeting = volatility_targeting
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.rebalance_blacklist = rebalance_blacklist or []
        self.allow_shorting = allow_shorting
        self.max_turnover = max_turnover
        
        self.last_rebalance_time = None
        self.current_weights = {}
        self.historical_weights = []
        self.volatility_estimates = {}
        
    def set_target_weights(self, target_weights: Dict[str, float]) -> None:
        """
        Set target portfolio weights.
        
        Args:
            target_weights: Dictionary mapping asset symbols to target weights
        """
        # Validate that weights sum to approximately 1.0
        weight_sum = sum(target_weights.values())
        if not (0.99 <= weight_sum <= 1.01):
            logger.warning(f"Target weights sum to {weight_sum}, not 1.0. Normalizing.")
            target_weights = {k: v / weight_sum for k, v in target_weights.items()}
        
        self.target_weights = target_weights
        logger.info(f"Set target weights: {target_weights}")
    
    def update_current_weights(self, current_positions: Dict[str, float], current_prices: Dict[str, float]) -> None:
        """
        Update current portfolio weights based on positions and prices.
        
        Args:
            current_positions: Dictionary of current position sizes (in units) by symbol
            current_prices: Dictionary of current prices by symbol
        """
        # Calculate position values
        position_values = {}
        for symbol, units in current_positions.items():
            if symbol in current_prices and current_prices[symbol] > 0:
                position_values[symbol] = units * current_prices[symbol]
            else:
                logger.warning(f"Missing price for {symbol}, cannot calculate position value")
        
        # Calculate total portfolio value
        total_value = sum(position_values.values())
        
        if total_value > 0:
            # Calculate current weights
            self.current_weights = {symbol: value / total_value for symbol, value in position_values.items()}
            
            # Store weights history
            timestamp = datetime.datetime.now()
            self.historical_weights.append({
                "timestamp": timestamp,
                "weights": self.current_weights.copy()
            })
        else:
            logger.warning("Total portfolio value is zero or negative, cannot calculate weights")
    
    def calculate_drift(self) -> Dict[str, float]:
        """
        Calculate the drift between current and target weights.
        
        Returns:
            Dict[str, float]: Dictionary of drift values for each asset
        """
        drift = {}
        
        # Check all symbols from both current and target weights
        all_symbols = set(list(self.current_weights.keys()) + list(self.target_weights.keys()))
        
        for symbol in all_symbols:
            current = self.current_weights.get(symbol, 0.0)
            target = self.target_weights.get(symbol, 0.0)
            
            if self.drift_calculation == "absolute":
                # Absolute difference between current and target
                drift[symbol] = current - target
            else:  # relative
                # Relative difference as percentage of target weight
                if target != 0:
                    drift[symbol] = (current - target) / target
                else:
                    drift[symbol] = float('inf') if current > 0 else 0.0
        
        return drift
    
    def check_rebalance_needed(self) -> Tuple[bool, Dict[str, float]]:
        """
        Check if portfolio rebalancing is needed based on drift thresholds 
        and time constraints.
        
        Returns:
            Tuple[bool, Dict[str, float]]: (rebalance_needed, drift_values)
        """
        # Check minimum time interval
        if self.last_rebalance_time is not None:
            hours_since_last = (datetime.datetime.now() - self.last_rebalance_time).total_seconds() / 3600
            if hours_since_last < self.min_rebalance_interval:
                return False, {}
        
        # Calculate current drift
        drift = self.calculate_drift()
        
        # Check if any assets exceed threshold
        for symbol, drift_value in drift.items():
            if symbol in self.rebalance_blacklist:
                continue
                
            if abs(drift_value) > self.rebalance_threshold:
                return True, drift
        
        # Check calendar-based rebalancing
        if self.should_calendar_rebalance():
            return True, drift
        
        return False, drift
    
    def should_calendar_rebalance(self) -> bool:
        """
        Check if calendar-based rebalancing is due.
        
        Returns:
            bool: True if calendar rebalance is needed
        """
        if self.last_rebalance_time is None:
            return True
            
        now = datetime.datetime.now()
        last = self.last_rebalance_time
        
        if self.rebalance_frequency == "daily":
            return now.date() > last.date()
        elif self.rebalance_frequency == "weekly":
            # Start new week (Monday)
            return now.date().weekday() < last.date().weekday()
        elif self.rebalance_frequency == "monthly":
            return now.month != last.month or now.year != last.year
        elif self.rebalance_frequency == "quarterly":
            last_quarter = (last.month - 1) // 3
            current_quarter = (now.month - 1) // 3
            return current_quarter != last_quarter or now.year != last.year
        
        return False
    
    def update_volatility_estimates(self, price_history: Dict[str, pd.DataFrame]) -> None:
        """
        Update volatility estimates for assets.
        
        Args:
            price_history: Dictionary mapping symbols to price history DataFrames
        """
        for symbol, df in price_history.items():
            if 'close' in df.columns and len(df) > 1:
                # Calculate daily returns
                returns = df['close'].pct_change().dropna()
                
                if len(returns) >= self.vol_lookback:
                    # Calculate annualized volatility
                    daily_vol = returns.tail(self.vol_lookback).std()
                    annualized_vol = daily_vol * np.sqrt(252)  # Annualize using trading days
                    self.volatility_estimates[symbol] = annualized_vol
    
    def get_volatility_adjusted_weights(self) -> Dict[str, float]:
        """
        Calculate volatility-adjusted target weights.
        
        Returns:
            Dict[str, float]: Volatility-adjusted weights
        """
        if not self.volatility_targeting or not self.volatility_estimates:
            return self.target_weights.copy()
        
        # Calculate inverse volatility weights
        inverse_vol = {}
        for symbol, target_weight in self.target_weights.items():
            if symbol in self.volatility_estimates and self.volatility_estimates[symbol] > 0:
                inverse_vol[symbol] = target_weight / self.volatility_estimates[symbol]
            else:
                inverse_vol[symbol] = target_weight
        
        # Normalize to sum to 1.0
        total_inverse_vol = sum(inverse_vol.values())
        if total_inverse_vol > 0:
            vol_adjusted_weights = {k: v / total_inverse_vol for k, v in inverse_vol.items()}
            
            # Scale to match target volatility
            portfolio_vol = self.estimate_portfolio_volatility(vol_adjusted_weights)
            if portfolio_vol > 0:
                scaling_factor = self.vol_target / portfolio_vol
                vol_adjusted_weights = {k: min(v * scaling_factor, 1.0) for k, v in vol_adjusted_weights.items()}
                
                # Re-normalize if necessary
                weight_sum = sum(vol_adjusted_weights.values())
                if weight_sum > 0 and abs(weight_sum - 1.0) > 0.01:
                    vol_adjusted_weights = {k: v / weight_sum for k, v in vol_adjusted_weights.items()}
            
            return vol_adjusted_weights
        
        return self.target_weights.copy()
    
    def estimate_portfolio_volatility(self, weights: Dict[str, float]) -> float:
        """
        Estimate portfolio volatility based on asset weights and volatilities.
        
        Args:
            weights: Asset allocation weights
            
        Returns:
            float: Estimated portfolio volatility (annualized)
        """
        # Simple estimation using weighted average volatility (assumes no correlation)
        weighted_vol = 0.0
        for symbol, weight in weights.items():
            if symbol in self.volatility_estimates:
                weighted_vol += weight * self.volatility_estimates[symbol]
        
        return weighted_vol
    
    def calculate_rebalance_trades(self, current_positions: Dict[str, float], 
                                  current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate trades needed to rebalance the portfolio.
        
        Args:
            current_positions: Current position sizes in units
            current_prices: Current prices for assets
            
        Returns:
            Dict[str, float]: Required trades in units (positive for buy, negative for sell)
        """
        # Update current weights first
        self.update_current_weights(current_positions, current_prices)
        
        # Check if rebalancing is needed
        rebalance_needed, drift = self.check_rebalance_needed()
        if not rebalance_needed:
            return {}
        
        # Get target weights, potentially adjusted for volatility
        target_weights = self.get_volatility_adjusted_weights() if self.volatility_targeting else self.target_weights
        
        # Calculate total portfolio value
        total_value = sum(pos * current_prices.get(sym, 0) for sym, pos in current_positions.items())
        
        # Calculate target position values
        target_values = {symbol: target_weights.get(symbol, 0.0) * total_value 
                         for symbol in set(list(current_positions.keys()) + list(target_weights.keys()))}
                         
        # Calculate current position values
        current_values = {symbol: current_positions.get(symbol, 0.0) * current_prices.get(symbol, 0.0) 
                         for symbol in set(list(current_positions.keys()) + list(target_weights.keys()))}
        
        # Calculate value differences
        value_differences = {symbol: target_values.get(symbol, 0.0) - current_values.get(symbol, 0.0) 
                            for symbol in set(list(target_values.keys()) + list(current_values.keys()))}
        
        # Check if shorting is allowed
        if not self.allow_shorting:
            for symbol, value in target_values.items():
                if value < 0:
                    target_values[symbol] = 0.0
        
        # Convert value differences to unit differences
        trades = {}
        for symbol, value_diff in value_differences.items():
            if symbol in current_prices and current_prices[symbol] > 0:
                trades[symbol] = value_diff / current_prices[symbol]
            else:
                trades[symbol] = 0.0
                
        # Apply blacklist
        for symbol in self.rebalance_blacklist:
            if symbol in trades:
                trades[symbol] = 0.0
        
        # Apply turnover limit
        if self.max_turnover < 1.0:
            total_turnover = sum(abs(trade_units) * current_prices.get(symbol, 0.0) 
                               for symbol, trade_units in trades.items()) / total_value
            
            if total_turnover > self.max_turnover:
                scaling_factor = self.max_turnover / total_turnover
                trades = {symbol: units * scaling_factor for symbol, units in trades.items()}
        
        # Update last rebalance time
        self.last_rebalance_time = datetime.datetime.now()
        
        # Log rebalance event
        logger.info(f"Portfolio rebalancing initiated: {trades}")
        
        return trades
    
    def get_rebalance_stats(self) -> Dict[str, Union[str, float, datetime.datetime]]:
        """
        Get statistics about rebalancing operations.
        
        Returns:
            Dict containing rebalancing statistics
        """
        return {
            "last_rebalance_time": self.last_rebalance_time,
            "rebalance_frequency": self.rebalance_frequency,
            "drift_threshold": self.rebalance_threshold,
            "current_weights": self.current_weights,
            "target_weights": self.target_weights,
            "blacklisted_assets": self.rebalance_blacklist,
            "volatility_targeting": self.volatility_targeting,
            "vol_target": self.vol_target if self.volatility_targeting else None,
        } 