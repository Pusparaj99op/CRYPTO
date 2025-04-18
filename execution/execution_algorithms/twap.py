import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

class TWAP:
    """
    Time Weighted Average Price (TWAP) execution algorithm.
    
    Executes an order by dividing it into equal-sized slices spread evenly 
    over a specified time period, aiming to achieve the average price over
    that period.
    """
    
    def __init__(
        self, 
        symbol: str,
        total_quantity: float,
        start_time: datetime,
        end_time: datetime,
        side: str,
        num_slices: int = 10,
        random_variance: float = 0.0,
        price_limit: Optional[float] = None
    ):
        """
        Initialize TWAP execution algorithm.
        
        Args:
            symbol: Trading pair symbol
            total_quantity: Total quantity to execute
            start_time: Algorithm start time
            end_time: Algorithm end time
            side: Order side ('buy' or 'sell')
            num_slices: Number of equal-sized order slices
            random_variance: Add randomness to slice sizes (0.0-1.0)
            price_limit: Optional limit price for the orders
        """
        self.symbol = symbol
        self.total_quantity = total_quantity
        self.start_time = start_time
        self.end_time = end_time
        self.side = side.lower()
        self.num_slices = max(1, num_slices)
        self.random_variance = max(0.0, min(1.0, random_variance))
        self.price_limit = price_limit
        
        # Calculate slice sizes and times
        self._calculate_execution_schedule()
        
        # Execution state
        self.is_running = False
        self.executed_quantity = 0.0
        self.remaining_quantity = total_quantity
        self.current_slice = 0
        self.execution_history = []
        
    def _calculate_execution_schedule(self):
        """Calculate the execution schedule with slices and times"""
        # Calculate base slice size
        base_slice_size = self.total_quantity / self.num_slices
        
        # Calculate time interval between slices
        total_seconds = (self.end_time - self.start_time).total_seconds()
        interval_seconds = total_seconds / self.num_slices
        
        # Generate slice schedule
        self.execution_schedule = []
        remaining_qty = self.total_quantity
        
        for i in range(self.num_slices):
            # Add randomness to slice size if specified
            if self.random_variance > 0 and i < self.num_slices - 1:
                # Random factor between (1-variance) and (1+variance)
                random_factor = 1.0 + (np.random.random() * 2 - 1) * self.random_variance
                slice_size = base_slice_size * random_factor
                
                # Ensure we don't exceed total quantity
                if slice_size > remaining_qty:
                    slice_size = remaining_qty
                    
                remaining_qty -= slice_size
            else:
                # Last slice or no randomness - use remaining quantity
                slice_size = remaining_qty if i == self.num_slices - 1 else base_slice_size
                remaining_qty -= slice_size
            
            # Calculate target execution time for this slice
            slice_time = self.start_time + timedelta(seconds=i * interval_seconds)
            
            self.execution_schedule.append({
                "slice_number": i + 1,
                "target_time": slice_time,
                "quantity": slice_size
            })
            
    def start(self):
        """Start the TWAP execution"""
        if self.is_running:
            return
            
        self.is_running = True
        self.remaining_quantity = self.total_quantity
        self.executed_quantity = 0.0
        self.current_slice = 0
        self.execution_history = []
        
        # Recalculate execution schedule
        self._calculate_execution_schedule()
    
    def stop(self):
        """Stop the TWAP execution"""
        self.is_running = False
    
    def get_next_slice(self, current_time: datetime) -> Optional[Dict]:
        """
        Get the next slice to execute based on current time.
        
        Args:
            current_time: Current market time
            
        Returns:
            Next slice details or None if no slice is ready
        """
        if not self.is_running or self.current_slice >= self.num_slices:
            return None
            
        next_slice = self.execution_schedule[self.current_slice]
        
        # Check if it's time to execute this slice
        if current_time >= next_slice["target_time"]:
            return next_slice
            
        return None
        
    def execute_slice(self, current_time: datetime, current_price: float) -> Dict:
        """
        Execute the next slice if it's time.
        
        Args:
            current_time: Current market time
            current_price: Current market price
            
        Returns:
            Execution details as a dictionary
        """
        if not self.is_running or self.remaining_quantity <= 0:
            return {"executed": False, "quantity": 0, "price": 0}
            
        next_slice = self.get_next_slice(current_time)
        
        if next_slice is None:
            return {"executed": False, "quantity": 0, "price": 0}
            
        # Check price limit if specified
        if self.price_limit:
            if (self.side == 'buy' and current_price > self.price_limit) or \
               (self.side == 'sell' and current_price < self.price_limit):
                return {"executed": False, "quantity": 0, "price": 0}
        
        # Execute the slice
        quantity = next_slice["quantity"]
        self.executed_quantity += quantity
        self.remaining_quantity -= quantity
        self.current_slice += 1
        
        execution_record = {
            "time": current_time,
            "quantity": quantity,
            "price": current_price,
            "side": self.side,
            "executed": True,
            "slice_number": next_slice["slice_number"],
            "target_time": next_slice["target_time"],
            "time_deviation": (current_time - next_slice["target_time"]).total_seconds()
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
                "twap_achieved": None,
                "completion_percentage": 0.0,
                "slices_executed": 0,
                "slices_remaining": self.num_slices
            }
            
        # Calculate TWAP achieved
        twap_numerator = sum(record["price"] * record["quantity"] for record in self.execution_history)
        twap_denominator = sum(record["quantity"] for record in self.execution_history)
        twap_achieved = twap_numerator / twap_denominator if twap_denominator > 0 else None
        
        # Calculate average time deviation (from target execution times)
        time_deviations = [abs(record["time_deviation"]) for record in self.execution_history]
        avg_time_deviation = sum(time_deviations) / len(time_deviations) if time_deviations else 0
        
        completion_percentage = (self.executed_quantity / self.total_quantity) * 100 if self.total_quantity > 0 else 0
        
        return {
            "status": "completed" if self.remaining_quantity <= 0 else "running",
            "executed_quantity": self.executed_quantity,
            "remaining_quantity": self.remaining_quantity,
            "twap_achieved": twap_achieved,
            "completion_percentage": completion_percentage,
            "slices_executed": self.current_slice,
            "slices_remaining": self.num_slices - self.current_slice,
            "avg_time_deviation_seconds": avg_time_deviation
        } 