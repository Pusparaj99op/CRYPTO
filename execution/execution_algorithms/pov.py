import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable

class POV:
    """
    Percentage of Volume (POV) execution algorithm.
    
    Executes an order by participating in the market at a specified percentage
    of the market volume, adapting to changing market conditions.
    """
    
    def __init__(
        self, 
        symbol: str,
        total_quantity: float,
        start_time: datetime,
        end_time: datetime,
        side: str,
        target_participation_rate: float = 0.1,
        min_participation_rate: Optional[float] = None,
        max_participation_rate: Optional[float] = None,
        price_limit: Optional[float] = None,
        adaptive_rate: bool = False,
        volume_trend_lookback: int = 5
    ):
        """
        Initialize POV execution algorithm.
        
        Args:
            symbol: Trading pair symbol
            total_quantity: Total quantity to execute
            start_time: Algorithm start time
            end_time: Algorithm end time
            side: Order side ('buy' or 'sell')
            target_participation_rate: Target participation as fraction of market volume
            min_participation_rate: Optional minimum participation rate
            max_participation_rate: Optional maximum participation rate
            price_limit: Optional limit price for the orders
            adaptive_rate: Adjust participation rate based on market conditions
            volume_trend_lookback: Number of periods to consider for volume trend
        """
        self.symbol = symbol
        self.total_quantity = total_quantity
        self.start_time = start_time
        self.end_time = end_time
        self.side = side.lower()
        
        # Validate and set participation rates
        self.target_participation_rate = max(0.0, min(1.0, target_participation_rate))
        self.min_participation_rate = min_participation_rate or max(0.01, self.target_participation_rate / 2)
        self.max_participation_rate = max_participation_rate or min(0.5, self.target_participation_rate * 2)
        
        self.price_limit = price_limit
        self.adaptive_rate = adaptive_rate
        self.volume_trend_lookback = max(2, volume_trend_lookback)
        
        # Execution state
        self.is_running = False
        self.executed_quantity = 0.0
        self.remaining_quantity = total_quantity
        self.execution_history = []
        self.volume_history = []
        self.current_participation_rate = self.target_participation_rate
        
    def start(self):
        """Start the POV execution"""
        if self.is_running:
            return
            
        self.is_running = True
        self.remaining_quantity = self.total_quantity
        self.executed_quantity = 0.0
        self.execution_history = []
        self.volume_history = []
        self.current_participation_rate = self.target_participation_rate
    
    def stop(self):
        """Stop the POV execution"""
        self.is_running = False
    
    def _adjust_participation_rate(self, market_volume: float) -> float:
        """
        Adjust participation rate based on market conditions.
        
        Args:
            market_volume: Current market volume
            
        Returns:
            Adjusted participation rate
        """
        if not self.adaptive_rate or len(self.volume_history) < self.volume_trend_lookback:
            return self.target_participation_rate
            
        # Calculate volume trend
        recent_volumes = self.volume_history[-self.volume_trend_lookback:]
        volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
        
        # Calculate volume volatility (coefficient of variation)
        volume_mean = np.mean(recent_volumes)
        volume_std = np.std(recent_volumes)
        volume_cv = volume_std / volume_mean if volume_mean > 0 else 0
        
        # Adjust participation rate based on trend and volatility
        adjusted_rate = self.target_participation_rate
        
        # Increase rate if volume is trending up (positive slope)
        if volume_trend > 0:
            trend_factor = min(1.5, 1.0 + (volume_trend / volume_mean) * 5)
            adjusted_rate *= trend_factor
            
        # Decrease rate if volume is trending down (negative slope)
        elif volume_trend < 0:
            trend_factor = max(0.5, 1.0 + (volume_trend / volume_mean) * 5)
            adjusted_rate *= trend_factor
            
        # Adjust for volatility - reduce participation in highly volatile markets
        if volume_cv > 0.2:  # Significant volatility
            volatility_factor = max(0.5, 1.0 - (volume_cv - 0.2))
            adjusted_rate *= volatility_factor
            
        # Ensure we stay within allowed participation range
        adjusted_rate = max(self.min_participation_rate, min(self.max_participation_rate, adjusted_rate))
        
        return adjusted_rate
    
    def update_market_volume(self, current_time: datetime, market_volume: float):
        """
        Update market volume history.
        
        Args:
            current_time: Current market time
            market_volume: Current market volume
        """
        if self.is_running and current_time >= self.start_time and current_time <= self.end_time:
            self.volume_history.append(market_volume)
            
            # Update current participation rate if using adaptive mode
            if self.adaptive_rate:
                self.current_participation_rate = self._adjust_participation_rate(market_volume)
    
    def get_target_quantity(self, market_volume: float) -> float:
        """
        Calculate the target quantity to execute based on market volume.
        
        Args:
            market_volume: Current market volume
            
        Returns:
            Target quantity to execute
        """
        if not self.is_running:
            return 0.0
            
        # Calculate target quantity based on participation rate
        target_quantity = market_volume * self.current_participation_rate
        
        # Ensure we don't exceed remaining quantity
        target_quantity = min(target_quantity, self.remaining_quantity)
        
        return target_quantity
    
    def execute_slice(self, current_time: datetime, market_volume: float, 
                     current_price: float) -> Dict:
        """
        Execute a slice of the order based on POV algorithm.
        
        Args:
            current_time: Current market time
            market_volume: Current market volume
            current_price: Current market price
            
        Returns:
            Execution details as a dictionary
        """
        if not self.is_running or self.remaining_quantity <= 0:
            return {"executed": False, "quantity": 0, "price": 0}
            
        if current_time < self.start_time or current_time > self.end_time:
            return {"executed": False, "quantity": 0, "price": 0}
            
        # Update market volume history and participation rate
        self.update_market_volume(current_time, market_volume)
        
        # Calculate target quantity
        target_quantity = self.get_target_quantity(market_volume)
        
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
            "market_volume": market_volume,
            "participation_rate": self.current_participation_rate
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
                "avg_participation_rate": self.target_participation_rate
            }
            
        # Calculate VWAP achieved
        vwap_numerator = sum(record["price"] * record["quantity"] for record in self.execution_history)
        vwap_denominator = sum(record["quantity"] for record in self.execution_history)
        vwap_achieved = vwap_numerator / vwap_denominator if vwap_denominator > 0 else None
        
        # Calculate average participation rate
        avg_participation = sum(record["participation_rate"] for record in self.execution_history) / len(self.execution_history)
        
        # Calculate total market volume we've participated in
        total_market_volume = sum(record["market_volume"] for record in self.execution_history)
        actual_participation = self.executed_quantity / total_market_volume if total_market_volume > 0 else 0
        
        completion_percentage = (self.executed_quantity / self.total_quantity) * 100 if self.total_quantity > 0 else 0
        
        return {
            "status": "completed" if self.remaining_quantity <= 0 else "running",
            "executed_quantity": self.executed_quantity,
            "remaining_quantity": self.remaining_quantity,
            "vwap_achieved": vwap_achieved,
            "completion_percentage": completion_percentage,
            "avg_participation_rate": avg_participation,
            "actual_participation_rate": actual_participation,
            "total_market_volume": total_market_volume,
            "current_participation_rate": self.current_participation_rate
        } 