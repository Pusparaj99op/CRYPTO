import logging
import math
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OrderSizeConfig:
    # Base risk settings
    risk_per_trade: float = 0.01  # 1% of capital per trade
    max_risk_per_trade: float = 0.03  # 3% maximum risk per trade
    position_sizing_method: str = "risk_based"  # Options: risk_based, volatility_adjusted, kelly
    
    # Volatility settings
    volatility_lookback: int = 14  # Days to look back for volatility calc
    volatility_multiplier: float = 1.5  # Multiplier for volatility-based sizing
    
    # Kelly criterion settings
    kelly_fraction: float = 0.5  # Fractional Kelly (conservative)
    
    # Scaling settings
    enable_scaling: bool = False  # Whether to use scaled entries
    scale_levels: int = 3  # Number of scaling levels
    scale_factor: float = 0.7  # Reduction factor for each level
    
    # Limits
    min_order_size: float = 0.0  # Minimum allowed order size
    max_order_size: float = float('inf')  # Maximum allowed order size


class OrderSizeCalculator:
    """
    Calculates appropriate order sizes based on account balance, 
    market conditions, and risk parameters.
    """
    
    def __init__(
        self,
        config: OrderSizeConfig = None,
        market_info: Dict[str, Dict[str, Any]] = None
    ):
        self.config = config or OrderSizeConfig()
        self.market_info = market_info or {}
        
        # Historical data for calculations
        self.win_rate_history = {}  # Per-market win rates
        self.position_sizes = {}  # Track previously calculated sizes
        
    def update_market_info(self, market: str, info: Dict[str, Any]) -> None:
        """Update market specific information."""
        if market not in self.market_info:
            self.market_info[market] = {}
            
        self.market_info[market].update(info)
        
    def update_win_rate(self, market: str, win_rate: float, trades_count: int) -> None:
        """Update the win rate for a specific market."""
        self.win_rate_history[market] = {
            'win_rate': win_rate,
            'trades_count': trades_count,
            'last_updated': True  # Can be timestamp in real implementation
        }
        
    def calculate_order_size(
        self,
        market: str,
        account_balance: float,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        market_volatility: Optional[float] = None,
        leverage: float = 1.0,
        expected_win_rate: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate the appropriate order size based on configured
        position sizing method and risk parameters.
        
        Args:
            market: Market symbol
            account_balance: Current account balance
            entry_price: Planned entry price
            stop_loss_price: Planned stop loss price (optional)
            take_profit_price: Planned take profit price (optional)
            market_volatility: Market volatility, normalized (optional)
            leverage: Leverage multiplier
            expected_win_rate: Expected win rate for this trade (optional)
            
        Returns:
            Dictionary with calculated order details
        """
        # Get market-specific info
        market_data = self.market_info.get(market, {})
        min_order_size = market_data.get('min_order_size', self.config.min_order_size)
        max_order_size = market_data.get('max_order_size', self.config.max_order_size)
        
        # Calculate risk based size if method is risk_based or we need stop loss for other methods
        risk_based_size = None
        if stop_loss_price:
            risk_per_trade = self.config.risk_per_trade
            
            # Calculate risk amount in account currency
            risk_amount = account_balance * risk_per_trade
            
            # Calculate stop loss distance as percentage
            stop_distance_percent = abs(entry_price - stop_loss_price) / entry_price
            
            # Adjust for leverage
            effective_stop_distance = stop_distance_percent / leverage
            
            # Calculate position size based on risk
            if effective_stop_distance > 0:
                risk_based_size = risk_amount / effective_stop_distance
            else:
                logger.warning(f"Stop distance is zero for {market}, using fallback sizing")
                risk_based_size = account_balance * self.config.risk_per_trade * leverage
        else:
            # Fallback if no stop loss provided
            risk_based_size = account_balance * self.config.risk_per_trade * leverage
        
        # Choose sizing method based on configuration
        calculated_size = 0
        
        if self.config.position_sizing_method == "risk_based":
            calculated_size = risk_based_size
            
        elif self.config.position_sizing_method == "volatility_adjusted":
            # Use provided volatility or get from market info
            vol = market_volatility or market_data.get('volatility', 0.1)  # Default 10% volatility
            
            # Adjust position size inversely with volatility
            # Higher volatility = smaller position size
            volatility_factor = self.config.volatility_multiplier / (1 + vol)
            
            if risk_based_size:
                calculated_size = risk_based_size * volatility_factor
            else:
                calculated_size = account_balance * self.config.risk_per_trade * leverage * volatility_factor
                
        elif self.config.position_sizing_method == "kelly":
            # Get win rate and payoff ratio
            win_rate = expected_win_rate
            
            # If no expected win rate provided, use historical data or default
            if win_rate is None:
                if market in self.win_rate_history:
                    win_rate = self.win_rate_history[market]['win_rate']
                else:
                    win_rate = 0.5  # Default 50% win rate
            
            # Calculate payoff ratio (R:R) if stop loss and take profit provided
            payoff_ratio = 1.0
            if stop_loss_price and take_profit_price:
                potential_loss = abs(entry_price - stop_loss_price)
                potential_gain = abs(take_profit_price - entry_price)
                if potential_loss > 0:
                    payoff_ratio = potential_gain / potential_loss
            
            # Kelly formula: f* = (p * b - (1 - p)) / b
            # where p = win rate, b = payoff ratio (odds)
            kelly_percentage = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio
            
            # Apply fractional Kelly for safety
            kelly_percentage = kelly_percentage * self.config.kelly_fraction
            
            # Limit to positive values only
            kelly_percentage = max(0, kelly_percentage)
            
            # Calculate the position size
            calculated_size = account_balance * kelly_percentage * leverage
        
        # Apply minimum and maximum constraints
        calculated_size = max(min_order_size, min(calculated_size, max_order_size))
        calculated_size = min(calculated_size, account_balance * self.config.max_risk_per_trade * leverage)
        
        # Convert to market units (e.g. BTC) if needed
        position_units = calculated_size / entry_price if entry_price > 0 else 0
        
        # Calculate scaling levels if enabled
        scaling_levels = []
        if self.config.enable_scaling and self.config.scale_levels > 1:
            base_size = calculated_size / sum(self.config.scale_factor ** i for i in range(self.config.scale_levels))
            
            for i in range(self.config.scale_levels):
                level_size = base_size * (self.config.scale_factor ** i)
                scaling_levels.append({
                    'level': i + 1,
                    'size': level_size,
                    'units': level_size / entry_price if entry_price > 0 else 0
                })
        
        # Store the result
        self.position_sizes[market] = calculated_size
        
        # Return comprehensive result
        result = {
            'market': market,
            'order_size': calculated_size,
            'position_units': position_units,
            'method_used': self.config.position_sizing_method,
            'risk_percentage': self.config.risk_per_trade,
            'leverage': leverage,
            'scaling_levels': scaling_levels if self.config.enable_scaling else []
        }
        
        if stop_loss_price:
            result['stop_loss_price'] = stop_loss_price
            result['risk_per_unit'] = abs(entry_price - stop_loss_price)
            
        if take_profit_price:
            result['take_profit_price'] = take_profit_price
            
        return result
    
    def calculate_position_adjustment(
        self,
        market: str,
        current_position_size: float,
        current_price: float,
        target_risk_percentage: Optional[float] = None,
        account_balance: Optional[float] = None,
        new_stop_loss: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate size adjustment for an existing position.
        
        Args:
            market: Market symbol
            current_position_size: Current position size in base currency
            current_price: Current market price
            target_risk_percentage: Target risk % (optional)
            account_balance: Current account balance (optional)
            new_stop_loss: New stop loss price (optional)
            
        Returns:
            Dictionary with adjustment details
        """
        if target_risk_percentage is None:
            target_risk_percentage = self.config.risk_per_trade
            
        # Get latest market info
        market_data = self.market_info.get(market, {})
        
        # Determine the optimal position size
        optimal_size = 0
        
        if account_balance and new_stop_loss:
            # Risk-based adjustment
            risk_amount = account_balance * target_risk_percentage
            stop_distance = abs(current_price - new_stop_loss) / current_price
            
            if stop_distance > 0:
                optimal_size = risk_amount / stop_distance
            else:
                optimal_size = current_position_size  # Maintain current size
        elif account_balance:
            # Simple percentage-based adjustment
            optimal_size = account_balance * target_risk_percentage
        else:
            # No change if insufficient data
            optimal_size = current_position_size
            
        # Calculate the difference (positive = increase, negative = decrease)
        size_difference = optimal_size - current_position_size
        adjustment_percentage = (size_difference / current_position_size) * 100 if current_position_size > 0 else 0
        
        # Determine action
        action = 'INCREASE' if size_difference > 0 else 'DECREASE' if size_difference < 0 else 'MAINTAIN'
        
        return {
            'market': market,
            'current_size': current_position_size,
            'optimal_size': optimal_size,
            'adjustment': size_difference,
            'adjustment_percentage': adjustment_percentage,
            'action': action,
            'units_adjustment': size_difference / current_price if current_price > 0 else 0
        }
    
    def calculate_multi_position_sizes(
        self,
        markets: List[str],
        account_balance: float,
        equal_weighting: bool = False,
        custom_weights: Dict[str, float] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate position sizes for multiple markets simultaneously,
        distributing risk across all positions.
        
        Args:
            markets: List of market symbols
            account_balance: Current account balance
            equal_weighting: Whether to weight all markets equally
            custom_weights: Custom weight dictionary (market -> weight factor)
            
        Returns:
            Dictionary mapping markets to their position details
        """
        results = {}
        num_markets = len(markets)
        
        if num_markets == 0:
            return results
            
        # Determine allocation method
        if equal_weighting:
            # Divide risk equally among all markets
            risk_per_market = self.config.risk_per_trade / num_markets
            weights = {market: 1.0 for market in markets}
        elif custom_weights:
            # Use custom weights, normalizing them
            total_weight = sum(custom_weights.get(market, 1.0) for market in markets)
            if total_weight > 0:
                weights = {
                    market: custom_weights.get(market, 1.0) / total_weight 
                    for market in markets
                }
                risk_per_market = {
                    market: self.config.risk_per_trade * weights[market]
                    for market in markets
                }
            else:
                # Fallback to equal weighting
                risk_per_market = self.config.risk_per_trade / num_markets
                weights = {market: 1.0 for market in markets}
        else:
            # Use market-specific factors (like volatility) to allocate risk
            volatilities = {}
            for market in markets:
                market_data = self.market_info.get(market, {})
                vol = market_data.get('volatility', 0.1)  # Default 10% volatility
                volatilities[market] = vol
                
            # Inverse volatility weighting: less volatile markets get higher allocation
            # Formula: weight = (1/vol) / sum(1/vol)
            total_inverse_vol = sum(1.0 / vol for vol in volatilities.values() if vol > 0)
            
            if total_inverse_vol > 0:
                weights = {
                    market: (1.0 / volatilities[market]) / total_inverse_vol if volatilities[market] > 0 else 0
                    for market in markets
                }
                risk_per_market = {
                    market: self.config.risk_per_trade * weights[market]
                    for market in markets
                }
            else:
                # Fallback to equal weighting
                risk_per_market = self.config.risk_per_trade / num_markets
                weights = {market: 1.0 for market in markets}
        
        # Calculate individual sizes
        for market in markets:
            market_data = self.market_info.get(market, {})
            
            # Get market parameters for calculation
            entry_price = market_data.get('current_price', 1.0)
            stop_loss = market_data.get('stop_loss', None)
            take_profit = market_data.get('take_profit', None)
            leverage = market_data.get('leverage', 1.0)
            volatility = market_data.get('volatility', None)
            
            # Use custom risk percentage if using custom weights
            market_risk = risk_per_market[market] if isinstance(risk_per_market, dict) else risk_per_market
            
            # Save original risk setting
            original_risk = self.config.risk_per_trade
            
            # Temporarily set risk to market-specific risk
            self.config.risk_per_trade = market_risk
            
            # Calculate size
            size_result = self.calculate_order_size(
                market=market,
                account_balance=account_balance,
                entry_price=entry_price,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
                market_volatility=volatility,
                leverage=leverage
            )
            
            # Add weight information
            size_result['weight'] = weights.get(market, 0)
            size_result['relative_risk'] = market_risk / original_risk if original_risk > 0 else 0
            
            results[market] = size_result
            
            # Restore original risk setting
            self.config.risk_per_trade = original_risk
            
        return results
    
    def convert_to_scaled_orders(
        self, 
        base_order: Dict[str, Any],
        price_levels: List[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert a single order into multiple scaled orders at different price levels.
        
        Args:
            base_order: Base order details
            price_levels: List of price levels for scaling (optional)
            
        Returns:
            List of scaled orders
        """
        if not self.config.enable_scaling or self.config.scale_levels <= 1:
            return [base_order]
            
        market = base_order['market']
        entry_price = base_order.get('entry_price', self.market_info.get(market, {}).get('current_price', 0))
        total_size = base_order['order_size']
        
        # If price levels not provided, generate them
        if not price_levels or len(price_levels) < self.config.scale_levels:
            # Get market data for price level generation
            market_data = self.market_info.get(market, {})
            volatility = market_data.get('volatility', 0.1)
            
            # Generate levels based on volatility and direction
            direction = base_order.get('direction', 'buy')
            price_factor = 0.05 * volatility  # 5% of volatility
            
            price_levels = []
            for i in range(self.config.scale_levels):
                if direction.lower() == 'buy':
                    # For buys, scale down
                    level_price = entry_price * (1 - (i * price_factor))
                else:
                    # For sells, scale up
                    level_price = entry_price * (1 + (i * price_factor))
                price_levels.append(level_price)
                
        # Generate scaled orders
        scaled_orders = []
        base_size = total_size / sum(self.config.scale_factor ** i for i in range(self.config.scale_levels))
        
        for i in range(min(self.config.scale_levels, len(price_levels))):
            level_size = base_size * (self.config.scale_factor ** i)
            level_price = price_levels[i]
            
            # Create order for this level
            scaled_order = dict(base_order)
            scaled_order['order_size'] = level_size
            scaled_order['entry_price'] = level_price
            scaled_order['scale_level'] = i + 1
            
            # Recalculate position units
            if level_price > 0:
                scaled_order['position_units'] = level_size / level_price
                
            scaled_orders.append(scaled_order)
            
        return scaled_orders 