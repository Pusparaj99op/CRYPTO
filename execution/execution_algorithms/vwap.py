import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

class VWAP:
    """
    Volume Weighted Average Price (VWAP) execution algorithm.
    
    Executes an order over time, targeting the VWAP price by distributing
    the order volume according to expected volume profile.
    """
    
    def __init__(
        self, 
        symbol: str,
        total_quantity: float,
        start_time: datetime,
        end_time: datetime,
        side: str,
        volume_profile: Optional[Dict[datetime, float]] = None,
        min_participation_rate: float = 0.01,
        max_participation_rate: float = 0.1,
        price_limit: Optional[float] = None
    ):
        """
        Initialize VWAP execution algorithm.
        
        Args:
            symbol: Trading pair symbol
            total_quantity: Total quantity to execute
            start_time: Algorithm start time
            end_time: Algorithm end time
            side: Order side ('buy' or 'sell')
            volume_profile: Optional custom volume profile as {time: volume_percent}
            min_participation_rate: Minimum market participation rate
            max_participation_rate: Maximum market participation rate
            price_limit: Optional limit price for the orders
        """
        self.symbol = symbol
        self.total_quantity = total_quantity
        self.start_time = start_time
        self.end_time = end_time
        self.side = side.lower()
        self.min_participation_rate = min_participation_rate
        self.max_participation_rate = max_participation_rate
        self.price_limit = price_limit
        
        # Use provided volume profile or generate default
        self.volume_profile = volume_profile or self._generate_default_volume_profile()
        
        # Execution state
        self.is_running = False
        self.executed_quantity = 0.0
        self.remaining_quantity = total_quantity
        self.execution_history = []
        
    def _generate_default_volume_profile(self) -> Dict[datetime, float]:
        """
        Generate a default volume profile based on typical intraday patterns.
        
        Returns:
            Dictionary mapping time buckets to expected volume percentages
        """
        # Calculate time intervals (default 30 minute buckets)
        interval_minutes = 30
        intervals = int((self.end_time - self.start_time).total_seconds() / 60 / interval_minutes)
        intervals = max(intervals, 1)  # At least one interval
        
        # U-shaped volume profile (higher at start and end)
        weights = []
        for i in range(intervals):
            # More volume at start and end, less in the middle
            position = i / (intervals - 1) if intervals > 1 else 0.5
            # U-shape function: y = a*(x-0.5)^2 + b
            weight = 1.5 * (position - 0.5)**2 + 0.5
            weights.append(weight)
            
        # Normalize weights to sum to 1.0
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Create time buckets with normalized volume weights
        volume_profile = {}
        for i in range(intervals):
            bucket_time = self.start_time + timedelta(minutes=i * interval_minutes)
            volume_profile[bucket_time] = normalized_weights[i]
            
        return volume_profile
        
    def start(self):
        """Start the VWAP execution"""
        if self.is_running:
            return
            
        self.is_running = True
        self.remaining_quantity = self.total_quantity
        self.executed_quantity = 0.0
        self.execution_history = []
    
    def stop(self):
        """Stop the VWAP execution"""
        self.is_running = False
    
    def get_target_quantity(self, current_time: datetime, market_volume: float) -> float:
        """
        Calculate the target quantity to execute for the current time period.
        
        Args:
            current_time: Current market time
            market_volume: Current market volume for this period
            
        Returns:
            Target quantity to execute
        """
        if not self.is_running or current_time < self.start_time or current_time > self.end_time:
            return 0.0
            
        # Find the appropriate time bucket
        active_bucket = None
        for bucket_time in sorted(self.volume_profile.keys()):
            if bucket_time <= current_time:
                active_bucket = bucket_time
            else:
                break
                
        if active_bucket is None:
            return 0.0
            
        # Calculate target quantity based on volume profile
        profile_percentage = self.volume_profile[active_bucket]
        target_quantity = self.total_quantity * profile_percentage
        
        # Apply participation rate constraints
        min_quantity = market_volume * self.min_participation_rate
        max_quantity = market_volume * self.max_participation_rate
        
        # Bound target quantity by participation constraints
        target_quantity = max(min(target_quantity, max_quantity), min_quantity)
        
        # Ensure we don't exceed remaining quantity
        target_quantity = min(target_quantity, self.remaining_quantity)
        
        return target_quantity
        
    def execute_slice(self, current_time: datetime, market_volume: float, 
                     current_price: float) -> Dict:
        """
        Execute a slice of the order based on VWAP algorithm.
        
        Args:
            current_time: Current market time
            market_volume: Current market volume for this time period
            current_price: Current market price
            
        Returns:
            Execution details as a dictionary
        """
        if not self.is_running or self.remaining_quantity <= 0:
            return {"executed": False, "quantity": 0, "price": 0}
            
        target_quantity = self.get_target_quantity(current_time, market_volume)
        
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
            "executed": True
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
                "completion_percentage": 0.0
            }
            
        # Calculate VWAP achieved
        vwap_numerator = sum(record["price"] * record["quantity"] for record in self.execution_history)
        vwap_denominator = sum(record["quantity"] for record in self.execution_history)
        vwap_achieved = vwap_numerator / vwap_denominator if vwap_denominator > 0 else None
        
        completion_percentage = (self.executed_quantity / self.total_quantity) * 100 if self.total_quantity > 0 else 0
        
        return {
            "status": "completed" if self.remaining_quantity <= 0 else "running",
            "executed_quantity": self.executed_quantity,
            "remaining_quantity": self.remaining_quantity,
            "vwap_achieved": vwap_achieved,
            "completion_percentage": completion_percentage,
            "execution_count": len(self.execution_history)
        } 