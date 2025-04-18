import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

class AlmgrenChriss:
    """
    Almgren-Chriss optimal execution algorithm.
    
    Implements the Almgren-Chriss model for optimal execution, which balances
    market impact costs and price risk using a mean-variance optimization
    framework.
    """
    
    def __init__(
        self, 
        symbol: str,
        total_quantity: float,
        start_time: datetime,
        end_time: datetime,
        side: str,
        market_impact_factor: float = 0.1,
        volatility: Optional[float] = None,
        risk_aversion: float = 1.0,
        fixed_cost_per_trade: float = 0.0,
        temporary_impact_factor: Optional[float] = None,
        permanent_impact_factor: Optional[float] = None
    ):
        """
        Initialize Almgren-Chriss optimal execution algorithm.
        
        Args:
            symbol: Trading pair symbol
            total_quantity: Total quantity to execute
            start_time: Algorithm start time
            end_time: Algorithm end time
            side: Order side ('buy' or 'sell')
            market_impact_factor: Market impact factor (eta)
            volatility: Asset volatility per time unit (if None, will be estimated)
            risk_aversion: Risk aversion parameter (lambda)
            fixed_cost_per_trade: Fixed cost per trade
            temporary_impact_factor: Temporary impact factor (epsilon)
            permanent_impact_factor: Permanent impact factor (gamma)
        """
        self.symbol = symbol
        self.total_quantity = total_quantity
        self.start_time = start_time
        self.end_time = end_time
        self.side = side.lower()
        self.market_impact_factor = market_impact_factor
        self.volatility = volatility
        self.risk_aversion = max(0.0, risk_aversion)
        self.fixed_cost_per_trade = fixed_cost_per_trade
        
        # Initialize impact factors or use defaults
        self.temporary_impact_factor = temporary_impact_factor or (0.5 * market_impact_factor)
        self.permanent_impact_factor = permanent_impact_factor or (0.5 * market_impact_factor)
        
        # Calculate time units and trading schedule
        self._setup_time_parameters()
        
        # Initialize execution state
        self.is_running = False
        self.executed_quantity = 0.0
        self.remaining_quantity = total_quantity
        self.execution_history = []
        self.price_history = []
        
        # Calculate optimal trajectory
        self._calculate_optimal_trajectory()
        
    def _setup_time_parameters(self):
        """Set up time-related parameters"""
        # Calculate trading periods (default to 10)
        total_seconds = (self.end_time - self.start_time).total_seconds()
        self.trading_periods = 10  # Default number of periods
        
        # Calculate time per period
        self.seconds_per_period = total_seconds / self.trading_periods
        
        # Create schedule points
        self.schedule_times = []
        for i in range(self.trading_periods + 1):  # +1 for start point
            period_time = self.start_time + timedelta(seconds=i * self.seconds_per_period)
            self.schedule_times.append(period_time)
            
    def _calculate_optimal_trajectory(self):
        """Calculate the optimal trading trajectory based on Almgren-Chriss model"""
        # Initialize trajectory with total quantity
        self.trajectory = [self.total_quantity]
        self.trading_rates = []
        
        # If volatility is not provided, use default
        if self.volatility is None:
            self.volatility = 0.01  # 1% default volatility per time unit
            
        # Almgren-Chriss optimization parameters
        eta = self.market_impact_factor  # Market impact parameter
        lambda_factor = self.risk_aversion  # Risk aversion
        sigma = self.volatility  # Volatility per period
        T = self.trading_periods  # Number of periods
        
        # Calculate trading intensity kappa
        kappa = np.sqrt(lambda_factor * sigma**2 / eta)
        
        # Calculate hyperbolic terms
        tau = T / 2  # Midpoint of trading horizon
        alpha = kappa * tau
        sinh_term = np.sinh(2 * alpha)
        
        # Calculate trading rates for each period
        remaining_shares = self.total_quantity
        
        for t in range(T):
            # Calculate normalized time
            normalized_t = t / T
            
            # Calculate Almgren-Chriss optimal execution rate
            if alpha > 1e-8:  # Non-zero risk aversion
                numerator = np.sinh(kappa * (T - t))
                rate = remaining_shares * (2 * kappa * np.sinh(kappa)) / sinh_term
            else:  # Risk-neutral case (uniform execution)
                rate = remaining_shares / (T - t) if T > t else remaining_shares
                
            # Ensure we don't exceed remaining quantity
            rate = min(rate, remaining_shares)
            
            self.trading_rates.append(rate)
            remaining_shares -= rate
            self.trajectory.append(remaining_shares)
            
        # Ensure final quantity is zero (adjust for numerical precision)
        self.trajectory[-1] = 0.0
        
    def start(self):
        """Start the Almgren-Chriss execution"""
        if self.is_running:
            return
            
        self.is_running = True
        self.remaining_quantity = self.total_quantity
        self.executed_quantity = 0.0
        self.execution_history = []
        self.price_history = []
        
        # Reset trajectory calculation
        self._calculate_optimal_trajectory()
    
    def stop(self):
        """Stop the Almgren-Chriss execution"""
        self.is_running = False
    
    def update_market_data(self, current_time: datetime, current_price: float):
        """
        Update market data.
        
        Args:
            current_time: Current market time
            current_price: Current market price
        """
        if not self.is_running:
            return
            
        self.price_history.append((current_time, current_price))
        
        # Potentially update volatility estimate
        if len(self.price_history) >= 3 and self.volatility is None:
            self._update_volatility_estimate()
    
    def _update_volatility_estimate(self):
        """Update volatility estimate based on observed prices"""
        # Extract prices
        prices = [price for _, price in self.price_history]
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate volatility per period
        vol = np.std(returns)
        
        # Update only if we have meaningful data
        if not np.isnan(vol) and vol > 0:
            # Scale to match our time periods
            self.volatility = vol * np.sqrt(self.seconds_per_period / 3600)  # Assuming returns are hourly
            
            # Recalculate trajectory
            self._calculate_optimal_trajectory()
    
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
            
        # Find the appropriate time period
        for i, time_point in enumerate(self.schedule_times[:-1]):
            next_time = self.schedule_times[i+1]
            
            if current_time >= time_point and current_time < next_time:
                # We're in period i
                return self.trading_rates[i]
                
        return 0.0
    
    def _estimate_market_impact(self, quantity: float, current_price: float) -> Tuple[float, float]:
        """
        Estimate market impact based on Almgren-Chriss model.
        
        Args:
            quantity: Trade quantity
            current_price: Current price
            
        Returns:
            Tuple of (temporary_impact, permanent_impact)
        """
        # Simple square-root model for temporary impact
        temporary_impact = self.temporary_impact_factor * current_price * np.sqrt(quantity / self.total_quantity)
        
        # Linear model for permanent impact
        permanent_impact = self.permanent_impact_factor * current_price * (quantity / self.total_quantity)
        
        return (temporary_impact, permanent_impact)
    
    def execute_slice(self, current_time: datetime, current_price: float) -> Dict:
        """
        Execute a slice of the order based on the Almgren-Chriss trajectory.
        
        Args:
            current_time: Current market time
            current_price: Current market price
            
        Returns:
            Execution details as a dictionary
        """
        if not self.is_running or self.remaining_quantity <= 0:
            return {"executed": False, "quantity": 0, "price": 0}
            
        # Update market data
        self.update_market_data(current_time, current_price)
        
        # Get target quantity for current time
        target_quantity = self.get_target_quantity(current_time)
        
        # Check if we should execute
        if target_quantity <= 0:
            return {"executed": False, "quantity": 0, "price": 0}
            
        # Ensure we don't exceed remaining quantity
        target_quantity = min(target_quantity, self.remaining_quantity)
        
        # Calculate market impact
        temporary_impact, permanent_impact = self._estimate_market_impact(target_quantity, current_price)
        
        # Effective execution price including impact
        if self.side == 'buy':
            effective_price = current_price + temporary_impact
        else:  # sell
            effective_price = current_price - temporary_impact
            
        # Execute the slice
        self.executed_quantity += target_quantity
        self.remaining_quantity -= target_quantity
        
        # Find current period
        current_period = 0
        for i, time_point in enumerate(self.schedule_times[:-1]):
            if current_time >= time_point and current_time < self.schedule_times[i+1]:
                current_period = i
                break
                
        execution_record = {
            "time": current_time,
            "quantity": target_quantity,
            "price": current_price,
            "effective_price": effective_price,
            "side": self.side,
            "executed": True,
            "temporary_impact": temporary_impact,
            "permanent_impact": permanent_impact,
            "period": current_period
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
                "trajectory_adherence": 0.0
            }
            
        # Calculate VWAP achieved
        vwap_numerator = sum(record["price"] * record["quantity"] for record in self.execution_history)
        vwap_denominator = sum(record["quantity"] for record in self.execution_history)
        vwap_achieved = vwap_numerator / vwap_denominator if vwap_denominator > 0 else None
        
        # Calculate effective VWAP (including impact)
        effective_vwap_numerator = sum(record["effective_price"] * record["quantity"] for record in self.execution_history)
        effective_vwap = effective_vwap_numerator / vwap_denominator if vwap_denominator > 0 else None
        
        # Calculate average impacts
        avg_temp_impact = sum(record["temporary_impact"] for record in self.execution_history) / len(self.execution_history)
        avg_perm_impact = sum(record["permanent_impact"] for record in self.execution_history) / len(self.execution_history)
        
        # Calculate trajectory adherence (how closely we followed the plan)
        # Group executions by period
        period_quantities = {}
        for record in self.execution_history:
            period = record["period"]
            period_quantities[period] = period_quantities.get(period, 0) + record["quantity"]
            
        # Compare with planned trajectory
        adherence_score = 0.0
        num_periods_with_execution = len(period_quantities)
        
        if num_periods_with_execution > 0:
            for period, executed_qty in period_quantities.items():
                if period < len(self.trading_rates):
                    planned_qty = self.trading_rates[period]
                    period_adherence = 1.0 - min(1.0, abs(executed_qty - planned_qty) / max(1e-6, planned_qty))
                    adherence_score += period_adherence
                    
            adherence_score /= num_periods_with_execution
            
        completion_percentage = (self.executed_quantity / self.total_quantity) * 100 if self.total_quantity > 0 else 0
        
        return {
            "status": "completed" if self.remaining_quantity <= 0 else "running",
            "executed_quantity": self.executed_quantity,
            "remaining_quantity": self.remaining_quantity,
            "vwap_achieved": vwap_achieved,
            "effective_vwap": effective_vwap,
            "completion_percentage": completion_percentage,
            "avg_temporary_impact": avg_temp_impact,
            "avg_permanent_impact": avg_perm_impact,
            "trajectory_adherence": adherence_score,
            "volatility_estimate": self.volatility
        } 