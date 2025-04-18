import numpy as np
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Callable

class SizingMethod(Enum):
    """Enum for different position sizing methods"""
    FIXED = 0              # Fixed size in asset units
    PERCENTAGE = 1         # Percentage of capital
    VOLATILITY_BASED = 2   # Volatility-adjusted position sizing
    KELLY = 3              # Kelly criterion
    RISK_PARITY = 4        # Risk parity
    CUSTOM = 5             # Custom sizing method

class PositionSizer:
    """
    Position sizing algorithms for crypto trading.
    
    Handles various position sizing strategies including fixed size,
    percentage-based, volatility-based, Kelly criterion, and risk-parity.
    """

    def __init__(
        self,
        account_balance: float,
        max_capital_risk: float = 0.02,
        sizing_method: SizingMethod = SizingMethod.PERCENTAGE,
        default_size_percentage: float = 0.1,
        max_position_size: Optional[float] = None,
        custom_sizing_function: Optional[Callable] = None,
        lookback_periods: int = 20,
        target_portfolio_volatility: float = 0.15,
        max_leverage: float = 3.0,
        position_limits: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        Initialize the position sizer.
        
        Args:
            account_balance: Total account balance in base currency
            max_capital_risk: Maximum capital to risk per trade as fraction
            sizing_method: Method to use for position sizing
            default_size_percentage: Default position size as percentage of capital
            max_position_size: Maximum position size in base currency
            custom_sizing_function: Custom function for position sizing
            lookback_periods: Number of periods to use for volatility calculation
            target_portfolio_volatility: Target portfolio volatility (annualized)
            max_leverage: Maximum allowed leverage
            position_limits: Optional limits per symbol {symbol: {"min": min_amount, "max": max_amount}}
        """
        self.account_balance = account_balance
        self.max_capital_risk = max(0.001, min(0.5, max_capital_risk))
        self.sizing_method = sizing_method
        self.default_size_percentage = max(0.01, min(1.0, default_size_percentage))
        self.max_position_size = max_position_size or (account_balance * 0.5)
        self.custom_sizing_function = custom_sizing_function
        self.lookback_periods = max(5, lookback_periods)
        self.target_portfolio_volatility = target_portfolio_volatility
        self.max_leverage = max_leverage
        self.position_limits = position_limits or {}
        
        # Historical data for volatility calculations
        self.price_history = {}  # {symbol: [prices]}
        self.volatility_estimates = {}  # {symbol: volatility}
        self.win_loss_history = []  # For Kelly calculations
        
    def update_account_balance(self, new_balance: float):
        """
        Update the account balance.
        
        Args:
            new_balance: New account balance value
        """
        if new_balance <= 0:
            raise ValueError("Account balance must be positive")
        self.account_balance = new_balance
        
    def update_price_history(self, symbol: str, price: float):
        """
        Update price history for a symbol.
        
        Args:
            symbol: Trading pair symbol
            price: Current price
        """
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(price)
        
        # Keep history within lookback window
        if len(self.price_history[symbol]) > self.lookback_periods:
            self.price_history[symbol] = self.price_history[symbol][-self.lookback_periods:]
            
        # Update volatility estimate
        self._update_volatility_estimate(symbol)
        
    def _update_volatility_estimate(self, symbol: str):
        """
        Update volatility estimate for a symbol.
        
        Args:
            symbol: Trading pair symbol
        """
        prices = self.price_history.get(symbol, [])
        if len(prices) < 2:
            self.volatility_estimates[symbol] = 0.0
            return
            
        # Calculate log returns
        prices = np.array(prices)
        log_returns = np.diff(np.log(prices))
        
        # Calculate volatility (standard deviation of returns)
        volatility = np.std(log_returns)
        
        # Annualize volatility (assuming daily data with 365 trading days)
        annualized_vol = volatility * np.sqrt(365)
        
        self.volatility_estimates[symbol] = annualized_vol
        
    def update_trade_result(self, profit_loss: float):
        """
        Update trade result history for Kelly criterion.
        
        Args:
            profit_loss: Profit/loss from the trade
        """
        self.win_loss_history.append(profit_loss)
        
        # Keep reasonable history length
        if len(self.win_loss_history) > 100:
            self.win_loss_history = self.win_loss_history[-100:]
            
    def _calculate_kelly_fraction(self) -> float:
        """
        Calculate Kelly optimal fraction.
        
        Returns:
            Kelly fraction between 0 and 1
        """
        if not self.win_loss_history:
            return 0.25  # Default value
            
        # Count wins and calculate win probability
        wins = sum(1 for pl in self.win_loss_history if pl > 0)
        win_probability = wins / len(self.win_loss_history)
        
        # Calculate average win and loss
        if wins > 0:
            avg_win = sum(pl for pl in self.win_loss_history if pl > 0) / wins
        else:
            avg_win = 0
            
        losses = len(self.win_loss_history) - wins
        if losses > 0:
            avg_loss = abs(sum(pl for pl in self.win_loss_history if pl < 0) / losses)
        else:
            avg_loss = 1  # Avoid division by zero
            
        # Kelly formula: f* = (p * b - (1 - p)) / b
        # where p is win probability, b is win/loss ratio
        if avg_loss == 0:
            return 0.5  # Default when no losses
            
        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio
        
        # Limit Kelly fraction to reasonable bounds
        kelly_fraction = max(0.0, min(0.5, kelly_fraction))  # Half-Kelly for safety
        
        return kelly_fraction
        
    def get_position_size(
        self, 
        symbol: str, 
        entry_price: float, 
        stop_loss_price: Optional[float] = None, 
        risk_reward_ratio: Optional[float] = None,
        override_method: Optional[SizingMethod] = None
    ) -> Dict:
        """
        Calculate position size based on the selected method.
        
        Args:
            symbol: Trading pair symbol
            entry_price: Entry price
            stop_loss_price: Optional stop loss price
            risk_reward_ratio: Optional risk-reward ratio
            override_method: Optional override for sizing method
            
        Returns:
            Dictionary with position size details
        """
        method = override_method or self.sizing_method
        
        # Calculate position size based on method
        if method == SizingMethod.FIXED:
            size = self._calculate_fixed_size(symbol)
        elif method == SizingMethod.PERCENTAGE:
            size = self._calculate_percentage_size(symbol)
        elif method == SizingMethod.VOLATILITY_BASED:
            size = self._calculate_volatility_size(symbol, entry_price)
        elif method == SizingMethod.KELLY:
            size = self._calculate_kelly_size(symbol)
        elif method == SizingMethod.RISK_PARITY:
            size = self._calculate_risk_parity_size(symbol)
        elif method == SizingMethod.CUSTOM and self.custom_sizing_function:
            size = self.custom_sizing_function(
                symbol=symbol,
                entry_price=entry_price,
                account_balance=self.account_balance,
                volatility=self.volatility_estimates.get(symbol, 0.0),
                stop_loss_price=stop_loss_price
            )
        else:
            # Default to percentage method
            size = self._calculate_percentage_size(symbol)
        
        # Calculate risk if stop loss is provided
        risk_amount = 0.0
        risk_percentage = 0.0
        
        if stop_loss_price:
            # Calculate risk based on stop loss
            price_risk = abs(entry_price - stop_loss_price) / entry_price
            risk_amount = size * price_risk
            risk_percentage = risk_amount / self.account_balance
            
            # Adjust size if risk is too high
            if risk_percentage > self.max_capital_risk:
                adjustment_factor = self.max_capital_risk / risk_percentage
                size *= adjustment_factor
                risk_amount = size * price_risk
                risk_percentage = risk_amount / self.account_balance
                
        # Apply position limits if specified
        if symbol in self.position_limits:
            limits = self.position_limits[symbol]
            if "min" in limits and size < limits["min"]:
                size = limits["min"]
            if "max" in limits and size > limits["max"]:
                size = limits["max"]
                
        # Ensure we don't exceed max position size
        size = min(size, self.max_position_size)
        
        # Calculate position value and leverage
        position_value = size * entry_price
        leverage = position_value / (self.account_balance * self.default_size_percentage)
        
        # Ensure leverage doesn't exceed maximum
        if leverage > self.max_leverage:
            adjustment_factor = self.max_leverage / leverage
            size *= adjustment_factor
            position_value = size * entry_price
            leverage = position_value / (self.account_balance * self.default_size_percentage)
            
        # Calculate units based on entry price
        units = size / entry_price if entry_price > 0 else 0
        
        return {
            "position_size_base": size,  # Size in base currency
            "position_size_units": units,  # Size in asset units
            "entry_price": entry_price,
            "position_value": position_value,
            "risk_amount": risk_amount,
            "risk_percentage": risk_percentage,
            "leverage": leverage,
            "sizing_method": method.name
        }
        
    def _calculate_fixed_size(self, symbol: str) -> float:
        """
        Calculate fixed position size.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Position size in base currency
        """
        # Use default percentage for fixed calculation
        position_size = self.account_balance * self.default_size_percentage
        return position_size
        
    def _calculate_percentage_size(self, symbol: str) -> float:
        """
        Calculate percentage-based position size.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Position size in base currency
        """
        position_size = self.account_balance * self.default_size_percentage
        return position_size
        
    def _calculate_volatility_size(self, symbol: str, entry_price: float) -> float:
        """
        Calculate volatility-based position size.
        
        Args:
            symbol: Trading pair symbol
            entry_price: Entry price
            
        Returns:
            Position size in base currency
        """
        volatility = self.volatility_estimates.get(symbol, 0.0)
        if volatility <= 0:
            return self._calculate_percentage_size(symbol)
            
        # Calculate target volatility contribution
        target_vol_contribution = self.target_portfolio_volatility * self.default_size_percentage
        
        # Position size that contributes target volatility
        position_size = (target_vol_contribution / volatility) * self.account_balance
        
        return position_size
        
    def _calculate_kelly_size(self, symbol: str) -> float:
        """
        Calculate Kelly criterion position size.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Position size in base currency
        """
        kelly_fraction = self._calculate_kelly_fraction()
        position_size = self.account_balance * kelly_fraction
        return position_size
        
    def _calculate_risk_parity_size(self, symbol: str) -> float:
        """
        Calculate risk parity position size.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Position size in base currency
        """
        volatility = self.volatility_estimates.get(symbol, 0.0)
        if volatility <= 0:
            return self._calculate_percentage_size(symbol)
            
        # Using a simplified risk parity approach
        # Position size inversely proportional to volatility
        position_size = (self.account_balance * self.default_size_percentage) / max(0.01, volatility / self.target_portfolio_volatility)
        
        return position_size
        
    def get_portfolio_allocation(self, symbols: List[str], target_portfolio_value: Optional[float] = None) -> Dict[str, Dict]:
        """
        Calculate portfolio allocation across multiple assets.
        
        Args:
            symbols: List of trading pair symbols
            target_portfolio_value: Optional target portfolio value
            
        Returns:
            Dictionary of position sizes per symbol
        """
        portfolio_value = target_portfolio_value or self.account_balance
        
        if self.sizing_method == SizingMethod.RISK_PARITY:
            # Risk parity allocation
            return self._risk_parity_allocation(symbols, portfolio_value)
        else:
            # Default equal-weighted allocation
            return self._equal_weighted_allocation(symbols, portfolio_value)
            
    def _equal_weighted_allocation(self, symbols: List[str], portfolio_value: float) -> Dict[str, Dict]:
        """
        Calculate equal-weighted portfolio allocation.
        
        Args:
            symbols: List of trading pair symbols
            portfolio_value: Target portfolio value
            
        Returns:
            Dictionary of position sizes per symbol
        """
        num_assets = len(symbols)
        if num_assets == 0:
            return {}
            
        allocation = {}
        equal_weight = 1.0 / num_assets
        
        for symbol in symbols:
            allocation[symbol] = {
                "weight": equal_weight,
                "allocation": portfolio_value * equal_weight,
                "allocation_percentage": equal_weight * 100,
                "method": "equal_weight"
            }
            
        return allocation
        
    def _risk_parity_allocation(self, symbols: List[str], portfolio_value: float) -> Dict[str, Dict]:
        """
        Calculate risk parity portfolio allocation.
        
        Args:
            symbols: List of trading pair symbols
            portfolio_value: Target portfolio value
            
        Returns:
            Dictionary of position sizes per symbol
        """
        # Get volatilities
        volatilities = {}
        for symbol in symbols:
            vol = self.volatility_estimates.get(symbol, 0.0)
            if vol <= 0:
                # Use default volatility if not available
                vol = 0.3  # Default annual volatility
            volatilities[symbol] = vol
            
        # Calculate inverse volatility weights
        inverse_vols = {s: 1.0 / max(0.01, v) for s, v in volatilities.items()}
        total_inverse_vol = sum(inverse_vols.values())
        
        # Normalize weights
        weights = {s: v / total_inverse_vol for s, v in inverse_vols.items()}
        
        # Calculate allocations
        allocation = {}
        for symbol in symbols:
            weight = weights.get(symbol, 0.0)
            allocation[symbol] = {
                "weight": weight,
                "allocation": portfolio_value * weight,
                "allocation_percentage": weight * 100,
                "method": "risk_parity",
                "volatility": volatilities.get(symbol, 0.0)
            }
            
        return allocation 