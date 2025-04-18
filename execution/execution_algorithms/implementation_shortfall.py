import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

class ImplementationShortfall:
    """
    Implementation Shortfall execution algorithm.
    
    An algorithm that balances market impact and timing risk by adapting
    execution speed based on price movements and volatility. It aims to 
    minimize the difference between the decision price and the final
    execution price.
    """
    
    def __init__(
        self, 
        symbol: str,
        total_quantity: float,
        start_time: datetime,
        end_time: datetime,
        side: str,
        decision_price: float,
        urgency: float = 0.5,
        market_impact_model: Optional[callable] = None,
        price_momentum_threshold: float = 0.002,
        volatility_threshold: float = 0.05
    ):
        """
        Initialize Implementation Shortfall algorithm.
        
        Args:
            symbol: Trading pair symbol
            total_quantity: Total quantity to execute
            start_time: Algorithm start time
            end_time: Algorithm end time
            side: Order side ('buy' or 'sell')
            decision_price: Initial price when trading decision was made
            urgency: Urgency parameter (0.0-1.0) where higher values front-load execution
            market_impact_model: Optional custom market impact model function
            price_momentum_threshold: Threshold for price momentum to adjust trading speed
            volatility_threshold: Volatility threshold to adjust trading speed
        """
        self.symbol = symbol
        self.total_quantity = total_quantity
        self.start_time = start_time
        self.end_time = end_time
        self.side = side.lower()
        self.decision_price = decision_price
        self.urgency = max(0.0, min(1.0, urgency))
        self.price_momentum_threshold = price_momentum_threshold
        self.volatility_threshold = volatility_threshold
        
        # Default square-root market impact model if none provided
        self.market_impact_model = market_impact_model or self._default_market_impact_model
        
        # Calculate execution schedule
        self._generate_initial_schedule()
        
        # Execution state
        self.is_running = False
        self.executed_quantity = 0.0
        self.remaining_quantity = total_quantity
        self.execution_history = []
        self.price_history = []
        
    def _default_market_impact_model(self, quantity: float, price: float) -> float:
        """
        Default market impact model based on square-root formula.
        
        Args:
            quantity: Order quantity
            price: Current price
            
        Returns:
            Estimated market impact in price units
        """
        # Simple square-root model: impact = k * sigma * sqrt(quantity/ADV)
        # Using approximation with fixed params
        k = 0.1  # Market impact factor
        sigma = price * 0.01  # Approximated volatility (1% of price)
        adv = self.total_quantity * 10  # Approximated average daily volume
        
        impact = k * sigma * np.sqrt(quantity / adv) if adv > 0 else 0
        return impact
        
    def _generate_initial_schedule(self):
        """Generate the initial trading schedule based on urgency parameter"""
        # Calculate time intervals (default 15 minute buckets)
        interval_minutes = 15
        total_minutes = (self.end_time - self.start_time).total_seconds() / 60
        intervals = max(1, int(total_minutes / interval_minutes))
        
        # Generate schedule based on urgency parameter
        # Higher urgency = more front-loaded execution
        self.execution_schedule = []
        remaining_qty = self.total_quantity
        total_intervals = intervals
        
        for i in range(intervals):
            # Relative position in schedule (0 to 1)
            t = i / (intervals - 1) if intervals > 1 else 0.5
            
            # Calculate trading rate based on urgency
            # For high urgency (near 1.0), we trade more at the beginning
            # For low urgency (near 0.0), we trade more uniformly
            if self.urgency > 0.5:
                # Front-loaded schedule (exponential decay)
                rate_factor = np.exp(-4 * self.urgency * t)
            elif self.urgency < 0.5:
                # Back-loaded schedule (exponential growth)
                rate_factor = np.exp(4 * (1 - self.urgency) * (t - 1))
            else:
                # Neutral schedule (uniform)
                rate_factor = 1.0
                
            # Normalize rate factor
            rate_factor = rate_factor / sum([np.exp(-4 * self.urgency * (j / (intervals - 1))) 
                                           if intervals > 1 else 1.0 
                                           for j in range(intervals)])
            
            # Calculate quantity for this interval
            interval_qty = self.total_quantity * rate_factor
            
            # Ensure we don't exceed total quantity due to rounding
            if i == intervals - 1:
                interval_qty = remaining_qty
            else:
                remaining_qty -= interval_qty
                
            # Calculate target time for this interval
            interval_time = self.start_time + timedelta(minutes=i * interval_minutes)
            
            self.execution_schedule.append({
                "interval": i + 1,
                "target_time": interval_time,
                "target_quantity": interval_qty,
                "executed": False
            })
    
    def _estimate_shortfall(self, current_price: float) -> float:
        """
        Estimate the current implementation shortfall.
        
        Args:
            current_price: Current market price
            
        Returns:
            Estimated implementation shortfall in price units
        """
        if self.side == 'buy':
            return current_price - self.decision_price
        else:  # sell
            return self.decision_price - current_price
    
    def _calculate_price_momentum(self) -> float:
        """
        Calculate recent price momentum.
        
        Returns:
            Price momentum as a percentage
        """
        if len(self.price_history) < 5:
            return 0.0
            
        # Use last 5 price points
        recent_prices = self.price_history[-5:]
        
        # Simple linear regression slope
        x = np.arange(len(recent_prices))
        y = np.array(recent_prices)
        slope, _ = np.polyfit(x, y, 1)
        
        # Normalize by average price
        avg_price = np.mean(recent_prices)
        momentum = slope / avg_price if avg_price > 0 else 0
        
        return momentum
    
    def _calculate_volatility(self) -> float:
        """
        Calculate recent price volatility.
        
        Returns:
            Volatility as a percentage of price
        """
        if len(self.price_history) < 10:
            return 0.0
            
        # Use recent price points
        recent_prices = self.price_history[-10:]
        
        # Calculate returns
        returns = np.diff(recent_prices) / recent_prices[:-1]
        
        # Volatility as standard deviation of returns
        volatility = np.std(returns)
        
        return volatility
        
    def start(self):
        """Start the Implementation Shortfall execution"""
        if self.is_running:
            return
            
        self.is_running = True
        self.remaining_quantity = self.total_quantity
        self.executed_quantity = 0.0
        self.execution_history = []
        self.price_history = []
        
        # Reset execution schedule
        self._generate_initial_schedule()
    
    def stop(self):
        """Stop the Implementation Shortfall execution"""
        self.is_running = False
    
    def update_market_conditions(self, current_time: datetime, current_price: float):
        """
        Update market conditions data.
        
        Args:
            current_time: Current market time
            current_price: Current market price
        """
        if not self.is_running:
            return
            
        self.price_history.append(current_price)
        
        # Potentially adjust execution schedule based on price movements
        self._adjust_schedule(current_time, current_price)
    
    def _adjust_schedule(self, current_time: datetime, current_price: float):
        """
        Adjust execution schedule based on market conditions.
        
        Args:
            current_time: Current market time
            current_price: Current market price
        """
        if len(self.price_history) < 5:
            return
            
        # Calculate price momentum and volatility
        momentum = self._calculate_price_momentum()
        volatility = self._calculate_volatility()
        
        # Calculate shortfall
        shortfall = self._estimate_shortfall(current_price)
        
        # Adjust execution based on market conditions
        # For buys: accelerate if prices are rising, slow down if falling
        # For sells: accelerate if prices are falling, slow down if rising
        momentum_signal = momentum * (1 if self.side == 'buy' else -1)
        
        # Adjust incomplete intervals in the schedule
        remaining_intervals = [i for i, interval in enumerate(self.execution_schedule) 
                             if not interval["executed"] and interval["target_time"] > current_time]
        
        if not remaining_intervals:
            return
            
        # Base adjustment on momentum, volatility and shortfall
        adjustment_factor = 1.0
        
        # Momentum adjustment
        if abs(momentum) > self.price_momentum_threshold:
            # Favorable momentum - accelerate execution
            if momentum_signal > 0:
                adjustment_factor *= (1.0 + min(2.0, momentum_signal * 10))
            # Adverse momentum - potentially slow down
            else:
                adjustment_factor *= max(0.5, 1.0 + momentum_signal * 5)
        
        # Volatility adjustment - accelerate in high volatility
        if volatility > self.volatility_threshold:
            vol_factor = 1.0 + min(1.0, (volatility - self.volatility_threshold) * 5)
            adjustment_factor *= vol_factor
        
        # Shortfall adjustment - react to unfavorable shortfall
        if shortfall > 0:  # Unfavorable shortfall
            shortfall_factor = 1.0 - min(0.3, shortfall / self.decision_price)
            adjustment_factor *= shortfall_factor
            
        # Apply adjustment to remaining intervals
        total_remaining_qty = sum(self.execution_schedule[i]["target_quantity"] for i in remaining_intervals)
        
        if total_remaining_qty > 0:
            # Redistribute quantities based on adjustment factor
            for i in remaining_intervals:
                original_qty = self.execution_schedule[i]["target_quantity"]
                proportion = original_qty / total_remaining_qty
                
                # Earlier intervals get more aggressive adjustment
                position_in_remaining = remaining_intervals.index(i) / len(remaining_intervals) if len(remaining_intervals) > 1 else 0.5
                interval_adjustment = adjustment_factor * (1.0 - 0.5 * position_in_remaining)
                
                # Update target quantity
                self.execution_schedule[i]["target_quantity"] = proportion * total_remaining_qty * interval_adjustment
            
            # Normalize to ensure total quantity remains the same
            adjusted_total = sum(self.execution_schedule[i]["target_quantity"] for i in remaining_intervals)
            scaling_factor = total_remaining_qty / adjusted_total if adjusted_total > 0 else 1.0
            
            for i in remaining_intervals:
                self.execution_schedule[i]["target_quantity"] *= scaling_factor
    
    def get_target_quantity(self, current_time: datetime) -> float:
        """
        Get the target quantity to execute at the current time.
        
        Args:
            current_time: Current market time
            
        Returns:
            Target quantity to execute
        """
        if not self.is_running or current_time < self.start_time:
            return 0.0
            
        # If past end time, return all remaining quantity
        if current_time > self.end_time:
            return self.remaining_quantity
            
        # Find the appropriate interval
        for interval in self.execution_schedule:
            if not interval["executed"] and current_time >= interval["target_time"]:
                return min(interval["target_quantity"], self.remaining_quantity)
                
        return 0.0
    
    def execute_slice(self, current_time: datetime, current_price: float) -> Dict:
        """
        Execute a slice of the order based on the algorithm.
        
        Args:
            current_time: Current market time
            current_price: Current market price
            
        Returns:
            Execution details as a dictionary
        """
        if not self.is_running or self.remaining_quantity <= 0:
            return {"executed": False, "quantity": 0, "price": 0}
            
        # Update market conditions data
        self.update_market_conditions(current_time, current_price)
        
        # Get target quantity for current time
        target_quantity = self.get_target_quantity(current_time)
        
        # Check if we should execute
        if target_quantity <= 0:
            return {"executed": False, "quantity": 0, "price": 0}
        
        # Execute the slice
        self.executed_quantity += target_quantity
        self.remaining_quantity -= target_quantity
        
        # Mark the interval as executed
        for interval in self.execution_schedule:
            if not interval["executed"] and current_time >= interval["target_time"]:
                interval["executed"] = True
                break
        
        # Calculate market impact
        market_impact = self.market_impact_model(target_quantity, current_price)
        
        # Calculate shortfall for this execution
        executed_shortfall = self._estimate_shortfall(current_price)
        
        execution_record = {
            "time": current_time,
            "quantity": target_quantity,
            "price": current_price,
            "side": self.side,
            "executed": True,
            "shortfall": executed_shortfall,
            "market_impact": market_impact,
            "decision_price": self.decision_price
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
                "avg_shortfall": 0.0,
                "completion_percentage": 0.0
            }
            
        # Calculate VWAP achieved
        vwap_numerator = sum(record["price"] * record["quantity"] for record in self.execution_history)
        vwap_denominator = sum(record["quantity"] for record in self.execution_history)
        vwap_achieved = vwap_numerator / vwap_denominator if vwap_denominator > 0 else None
        
        # Calculate average shortfall and market impact
        total_shortfall = sum(record["shortfall"] * record["quantity"] for record in self.execution_history)
        avg_shortfall = total_shortfall / self.executed_quantity if self.executed_quantity > 0 else 0
        
        total_impact = sum(record["market_impact"] * record["quantity"] for record in self.execution_history)
        avg_impact = total_impact / self.executed_quantity if self.executed_quantity > 0 else 0
        
        # Calculate overall implementation shortfall
        overall_shortfall = (vwap_achieved - self.decision_price) if self.side == 'buy' else (self.decision_price - vwap_achieved) if vwap_achieved else 0
        
        completion_percentage = (self.executed_quantity / self.total_quantity) * 100 if self.total_quantity > 0 else 0
        
        return {
            "status": "completed" if self.remaining_quantity <= 0 else "running",
            "executed_quantity": self.executed_quantity,
            "remaining_quantity": self.remaining_quantity,
            "vwap_achieved": vwap_achieved,
            "decision_price": self.decision_price,
            "overall_shortfall": overall_shortfall,
            "avg_shortfall": avg_shortfall,
            "avg_market_impact": avg_impact,
            "completion_percentage": completion_percentage
        } 