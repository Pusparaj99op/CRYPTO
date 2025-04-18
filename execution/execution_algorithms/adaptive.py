import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable
from enum import Enum

class MarketCondition(Enum):
    """Enum for different market conditions"""
    NORMAL = 0
    TRENDING_UP = 1
    TRENDING_DOWN = 2
    HIGH_VOLATILITY = 3
    LOW_VOLATILITY = 4
    HIGH_VOLUME = 5
    LOW_VOLUME = 6

class AdaptiveExecution:
    """
    Adaptive execution algorithm.
    
    Dynamically adjusts execution strategy based on real-time market conditions,
    switching between different execution styles (TWAP, VWAP, etc.) as needed.
    """
    
    def __init__(
        self, 
        symbol: str,
        total_quantity: float,
        start_time: datetime,
        end_time: datetime,
        side: str,
        price_limit: Optional[float] = None,
        aggression_level: float = 0.5,
        min_execution_rate: float = 0.01,
        max_execution_rate: float = 0.5,
        volatility_threshold: float = 0.02,
        momentum_threshold: float = 0.01,
        volume_lookback: int = 10,
        price_lookback: int = 20
    ):
        """
        Initialize Adaptive execution algorithm.
        
        Args:
            symbol: Trading pair symbol
            total_quantity: Total quantity to execute
            start_time: Algorithm start time
            end_time: Algorithm end time
            side: Order side ('buy' or 'sell')
            price_limit: Optional limit price for the orders
            aggression_level: Base aggression level (0.0-1.0)
            min_execution_rate: Minimum execution rate per period
            max_execution_rate: Maximum execution rate per period
            volatility_threshold: Threshold for detecting high volatility
            momentum_threshold: Threshold for detecting price momentum
            volume_lookback: Periods to look back for volume analysis
            price_lookback: Periods to look back for price analysis
        """
        self.symbol = symbol
        self.total_quantity = total_quantity
        self.start_time = start_time
        self.end_time = end_time
        self.side = side.lower()
        self.price_limit = price_limit
        self.aggression_level = max(0.0, min(1.0, aggression_level))
        self.min_execution_rate = min_execution_rate
        self.max_execution_rate = max_execution_rate
        self.volatility_threshold = volatility_threshold
        self.momentum_threshold = momentum_threshold
        self.volume_lookback = max(5, volume_lookback)
        self.price_lookback = max(10, price_lookback)
        
        # Market data history
        self.price_history = []
        self.volume_history = []
        self.time_history = []
        
        # Execution state
        self.is_running = False
        self.executed_quantity = 0.0
        self.remaining_quantity = total_quantity
        self.execution_history = []
        self.current_market_condition = MarketCondition.NORMAL
        self.execution_rate = 0.0
        
    def start(self):
        """Start the Adaptive execution"""
        if self.is_running:
            return
            
        self.is_running = True
        self.remaining_quantity = self.total_quantity
        self.executed_quantity = 0.0
        self.execution_history = []
        self.price_history = []
        self.volume_history = []
        self.time_history = []
        self.current_market_condition = MarketCondition.NORMAL
        self.execution_rate = self._calculate_base_execution_rate()
    
    def stop(self):
        """Stop the Adaptive execution"""
        self.is_running = False
    
    def _calculate_base_execution_rate(self) -> float:
        """
        Calculate base execution rate based on time and quantity.
        
        Returns:
            Base execution rate as fraction of total quantity
        """
        # Calculate total trading time in seconds
        total_seconds = (self.end_time - self.start_time).total_seconds()
        
        # Default to 5-minute intervals for base rate calculation
        interval_seconds = 300  # 5 minutes
        num_intervals = max(1, total_seconds / interval_seconds)
        
        # Base rate distributes evenly across intervals
        base_rate = 1.0 / num_intervals
        
        # Apply aggression factor (higher = more front-loaded)
        if self.aggression_level > 0.5:
            # More aggressive - higher initial rate
            base_rate *= (1.0 + (self.aggression_level - 0.5) * 2)
        elif self.aggression_level < 0.5:
            # Less aggressive - lower initial rate
            base_rate *= (self.aggression_level * 2)
            
        # Ensure within allowed range
        base_rate = max(self.min_execution_rate, min(self.max_execution_rate, base_rate))
        
        return base_rate
    
    def update_market_data(self, current_time: datetime, current_price: float, 
                          current_volume: float) -> None:
        """
        Update market data history.
        
        Args:
            current_time: Current market time
            current_price: Current market price
            current_volume: Current market volume
        """
        if not self.is_running:
            return
            
        self.time_history.append(current_time)
        self.price_history.append(current_price)
        self.volume_history.append(current_volume)
        
        # Keep history within lookback window
        if len(self.price_history) > self.price_lookback:
            self.time_history = self.time_history[-self.price_lookback:]
            self.price_history = self.price_history[-self.price_lookback:]
            
        if len(self.volume_history) > self.volume_lookback:
            self.volume_history = self.volume_history[-self.volume_lookback:]
        
        # Update market condition and execution rate
        self._update_market_condition()
        self._adjust_execution_rate()
    
    def _calculate_volatility(self) -> float:
        """
        Calculate recent price volatility.
        
        Returns:
            Price volatility as a percentage
        """
        if len(self.price_history) < 5:
            return 0.0
            
        # Calculate returns
        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1]
        
        # Annualized volatility based on available returns
        volatility = np.std(returns)
        
        return volatility
    
    def _calculate_momentum(self) -> float:
        """
        Calculate recent price momentum.
        
        Returns:
            Price momentum as a percentage
        """
        if len(self.price_history) < 5:
            return 0.0
            
        # Use linear regression slope as momentum indicator
        prices = np.array(self.price_history)
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        
        # Normalize by average price
        avg_price = np.mean(prices)
        momentum = slope / avg_price if avg_price > 0 else 0
        
        return momentum
    
    def _calculate_volume_profile(self) -> Tuple[float, float]:
        """
        Calculate volume profile metrics.
        
        Returns:
            Tuple of (volume_trend, volume_deviation)
        """
        if len(self.volume_history) < 5:
            return (0.0, 0.0)
            
        volumes = np.array(self.volume_history)
        
        # Volume trend (slope of volume)
        x = np.arange(len(volumes))
        volume_trend, _ = np.polyfit(x, volumes, 1)
        
        # Volume deviation from average
        avg_volume = np.mean(volumes)
        recent_volume = volumes[-1]
        volume_deviation = (recent_volume / avg_volume) - 1.0 if avg_volume > 0 else 0
        
        return (volume_trend, volume_deviation)
    
    def _update_market_condition(self) -> None:
        """Update the current market condition based on indicators"""
        if len(self.price_history) < 5 or len(self.volume_history) < 5:
            return
            
        # Calculate key indicators
        volatility = self._calculate_volatility()
        momentum = self._calculate_momentum() 
        volume_trend, volume_deviation = self._calculate_volume_profile()
        
        # Determine market condition
        if volatility > self.volatility_threshold:
            self.current_market_condition = MarketCondition.HIGH_VOLATILITY
        elif volatility < self.volatility_threshold * 0.3:
            self.current_market_condition = MarketCondition.LOW_VOLATILITY
        elif momentum > self.momentum_threshold:
            self.current_market_condition = MarketCondition.TRENDING_UP
        elif momentum < -self.momentum_threshold:
            self.current_market_condition = MarketCondition.TRENDING_DOWN
        elif volume_deviation > 0.5:  # Volume 50% above average
            self.current_market_condition = MarketCondition.HIGH_VOLUME
        elif volume_deviation < -0.3:  # Volume 30% below average
            self.current_market_condition = MarketCondition.LOW_VOLUME
        else:
            self.current_market_condition = MarketCondition.NORMAL
    
    def _adjust_execution_rate(self) -> None:
        """Adjust execution rate based on current market condition"""
        # Start with base execution rate
        base_rate = self._calculate_base_execution_rate()
        adjusted_rate = base_rate
        
        # Adjust based on market condition
        if self.current_market_condition == MarketCondition.HIGH_VOLATILITY:
            # In high volatility, be more aggressive if favorable, less if unfavorable
            momentum = self._calculate_momentum()
            momentum_direction = 1 if self.side == 'buy' else -1
            if momentum * momentum_direction > 0:  # Favorable momentum
                adjusted_rate = base_rate * 1.5
            else:  # Unfavorable momentum
                adjusted_rate = base_rate * 0.5
                
        elif self.current_market_condition == MarketCondition.LOW_VOLATILITY:
            # In low volatility, execute more evenly
            adjusted_rate = base_rate * 0.8
            
        elif self.current_market_condition == MarketCondition.TRENDING_UP:
            # In trending up market
            if self.side == 'buy':
                # For buys in rising market, potentially accelerate to catch trend
                adjusted_rate = base_rate * 1.3
            else:
                # For sells in rising market, potentially accelerate to get better prices
                adjusted_rate = base_rate * 1.2
                
        elif self.current_market_condition == MarketCondition.TRENDING_DOWN:
            # In trending down market
            if self.side == 'buy':
                # For buys in falling market, potentially slow down to get better prices
                adjusted_rate = base_rate * 0.8
            else:
                # For sells in falling market, potentially accelerate to avoid worse prices
                adjusted_rate = base_rate * 1.5
                
        elif self.current_market_condition == MarketCondition.HIGH_VOLUME:
            # In high volume market, increase participation
            adjusted_rate = base_rate * 1.3
            
        elif self.current_market_condition == MarketCondition.LOW_VOLUME:
            # In low volume market, reduce participation
            adjusted_rate = base_rate * 0.7
            
        # Ensure rate is within allowed range
        self.execution_rate = max(self.min_execution_rate, min(self.max_execution_rate, adjusted_rate))
    
    def get_target_quantity(self, current_time: datetime) -> float:
        """
        Calculate the target quantity to execute for the current time period.
        
        Args:
            current_time: Current market time
            
        Returns:
            Target quantity to execute
        """
        if not self.is_running or current_time < self.start_time or current_time > self.end_time:
            return 0.0
            
        # If we're near the end time, accelerate execution
        time_remaining = (self.end_time - current_time).total_seconds()
        total_time = (self.end_time - self.start_time).total_seconds()
        time_progress = 1.0 - (time_remaining / total_time) if total_time > 0 else 1.0
        
        # Accelerate in the final 20% of the execution window
        end_acceleration = 1.0
        if time_progress > 0.8:
            end_acceleration = 1.0 + (time_progress - 0.8) * 5  # Up to 2x in the final moments
            
        # Calculate target quantity based on execution rate and urgency
        target_quantity = self.remaining_quantity * self.execution_rate * end_acceleration
        
        # Ensure we don't exceed remaining quantity
        target_quantity = min(target_quantity, self.remaining_quantity)
        
        return target_quantity
    
    def execute_slice(self, current_time: datetime, current_price: float, 
                     current_volume: float) -> Dict:
        """
        Execute a slice of the order based on current conditions.
        
        Args:
            current_time: Current market time
            current_price: Current market price
            current_volume: Current market volume
            
        Returns:
            Execution details as a dictionary
        """
        if not self.is_running or self.remaining_quantity <= 0:
            return {"executed": False, "quantity": 0, "price": 0}
            
        # Update market data and condition
        self.update_market_data(current_time, current_price, current_volume)
        
        # Get target quantity
        target_quantity = self.get_target_quantity(current_time)
        
        # Check if we should execute
        if target_quantity <= 0:
            return {"executed": False, "quantity": 0, "price": 0}
            
        # Check price limit if specified
        if self.price_limit:
            if (self.side == 'buy' and current_price > self.price_limit) or \
               (self.side == 'sell' and current_price < self.price_limit):
                return {"executed": False, "quantity": 0, "price": 0}
        
        # Execute the slice
        self.executed_quantity += target_quantity
        self.remaining_quantity -= target_quantity
        
        execution_record = {
            "time": current_time,
            "quantity": target_quantity,
            "price": current_price,
            "side": self.side,
            "executed": True,
            "market_condition": self.current_market_condition.name,
            "execution_rate": self.execution_rate
        }
        
        self.execution_history.append(execution_record)
        
        return execution_record
    
    def get_execution_summary(self) -> Dict:
        """
        Get summary of execution performance.
        
        Returns:
            Dictionary with execution summary
        """
        if not self.execution_history:
            return {
                "status": "not_started" if not self.is_running else "running",
                "executed_quantity": 0,
                "remaining_quantity": self.total_quantity,
                "vwap_achieved": None,
                "completion_percentage": 0.0,
                "current_market_condition": self.current_market_condition.name
            }
            
        # Calculate VWAP achieved
        vwap_numerator = sum(record["price"] * record["quantity"] for record in self.execution_history)
        vwap_denominator = sum(record["quantity"] for record in self.execution_history)
        vwap_achieved = vwap_numerator / vwap_denominator if vwap_denominator > 0 else None
        
        # Count occurrences of each market condition
        market_conditions = {}
        for condition in MarketCondition:
            market_conditions[condition.name] = sum(
                1 for record in self.execution_history 
                if record["market_condition"] == condition.name
            )
            
        # Calculate average execution rate
        avg_execution_rate = sum(record["execution_rate"] for record in self.execution_history) / len(self.execution_history)
        
        completion_percentage = (self.executed_quantity / self.total_quantity) * 100 if self.total_quantity > 0 else 0
        
        return {
            "status": "completed" if self.remaining_quantity <= 0 else "running",
            "executed_quantity": self.executed_quantity,
            "remaining_quantity": self.remaining_quantity,
            "vwap_achieved": vwap_achieved,
            "completion_percentage": completion_percentage,
            "current_market_condition": self.current_market_condition.name,
            "market_condition_counts": market_conditions,
            "avg_execution_rate": avg_execution_rate,
            "current_execution_rate": self.execution_rate
        } 