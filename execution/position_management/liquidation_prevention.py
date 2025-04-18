import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from common.types import Exchange, Market, Timeframe, Side
from config.config_manager import ConfigManager
from data.market_data import MarketData
from execution.order_management.order_manager import OrderManager
from execution.position_management.leverage import LeverageManager


class PreventionStrategy(Enum):
    """Defines different strategies for liquidation prevention."""
    STOP_LOSS = "stop_loss"  # Simple stop loss orders
    DYNAMIC_DELEVERAGING = "dynamic_deleveraging"  # Reduce leverage as position approaches liquidation
    AUTO_HEDGE = "auto_hedge"  # Automatically hedge on another venue
    PARTIAL_CLOSE = "partial_close"  # Gradually reduce position size
    COLLATERAL_MANAGEMENT = "collateral_management"  # Automatically add collateral


@dataclass
class LiquidationConfig:
    """Configuration for liquidation prevention."""
    # Safety buffer percentage above liquidation price
    safety_buffer_pct: float = 0.05
    
    # Threshold percentage from liquidation to start preventive actions
    action_threshold_pct: float = 0.15
    
    # How much to reduce leverage when approaching liquidation
    deleveraging_factor: float = 0.5
    
    # For partial close strategy, percent of position to close at each step
    partial_close_pct: List[float] = None
    
    # Thresholds for each partial close step as percentage from liquidation
    partial_close_thresholds: List[float] = None
    
    # Maximum percentage of portfolio to use for adding collateral
    max_additional_collateral_pct: float = 0.1
    
    # How frequently to check for liquidation risk (seconds)
    check_interval_seconds: int = 60
    
    # Whether to enable automatic preventive actions
    auto_prevention_enabled: bool = True
    
    # The default strategy to use
    default_strategy: PreventionStrategy = PreventionStrategy.DYNAMIC_DELEVERAGING
    
    def __post_init__(self):
        """Set defaults for any empty fields."""
        if self.partial_close_pct is None:
            self.partial_close_pct = [0.25, 0.5, 0.75]
            
        if self.partial_close_thresholds is None:
            self.partial_close_thresholds = [0.15, 0.1, 0.05]


class LiquidationPrevention:
    """
    Monitors positions for liquidation risk and takes preventive actions
    
    This module:
    1. Continuously monitors all positions to evaluate liquidation risk
    2. Executes predefined strategies to prevent liquidation
    3. Maintains position health metrics and history
    4. Provides alerts when positions are at liquidation risk
    """
    
    def __init__(
        self, 
        config_manager: ConfigManager, 
        market_data: MarketData,
        order_manager: OrderManager,
        leverage_manager: LeverageManager
    ):
        """
        Initialize the LiquidationPrevention service.
        
        Args:
            config_manager: Configuration manager
            market_data: Market data provider
            order_manager: For executing orders
            leverage_manager: For adjusting leverage
        """
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        self.market_data = market_data
        self.order_manager = order_manager
        self.leverage_manager = leverage_manager
        
        # Load configuration
        self.config = self._load_config()
        
        # Position health metrics: {(exchange, market): health_metrics}
        self.position_health: Dict[Tuple[Exchange, Market], Dict] = {}
        
        # Liquidation prevention history
        self.prevention_history: List[Dict] = []
        
        # Active prevention actions
        self.active_preventions: Dict[Tuple[Exchange, Market], Dict] = {}
        
        # Last check timestamps
        self.last_check: Dict[Tuple[Exchange, Market], int] = {}
        
    def _load_config(self) -> LiquidationConfig:
        """Load liquidation prevention configuration."""
        config = self.config_manager.get_config("liquidation_prevention")
        
        return LiquidationConfig(
            safety_buffer_pct=float(config.get("safety_buffer_pct", 0.05)),
            action_threshold_pct=float(config.get("action_threshold_pct", 0.15)),
            deleveraging_factor=float(config.get("deleveraging_factor", 0.5)),
            partial_close_pct=[float(x) for x in config.get("partial_close_pct", [0.25, 0.5, 0.75])],
            partial_close_thresholds=[float(x) for x in config.get("partial_close_thresholds", [0.15, 0.1, 0.05])],
            max_additional_collateral_pct=float(config.get("max_additional_collateral_pct", 0.1)),
            check_interval_seconds=int(config.get("check_interval_seconds", 60)),
            auto_prevention_enabled=bool(config.get("auto_prevention_enabled", True)),
            default_strategy=PreventionStrategy(config.get("default_strategy", "dynamic_deleveraging"))
        )
    
    def check_liquidation_risk(
        self, 
        exchange: Exchange, 
        market: Market, 
        position: Dict,
        force_check: bool = False
    ) -> Dict:
        """
        Check liquidation risk for a specific position.
        
        Args:
            exchange: The exchange
            market: The market
            position: Position details including size, entry price, leverage
            force_check: Whether to force check regardless of interval
            
        Returns:
            Risk assessment including health metrics
        """
        current_time = int(time.time())
        
        # Check if we should do the assessment based on interval
        if not force_check and (exchange, market) in self.last_check:
            last_check_time = self.last_check.get((exchange, market), 0)
            if current_time - last_check_time < self.config.check_interval_seconds:
                # Return cached assessment if available
                if (exchange, market) in self.position_health:
                    return self.position_health[(exchange, market)]
        
        self.last_check[(exchange, market)] = current_time
        
        try:
            # Get latest market data
            last_price = self.market_data.get_last_price(exchange, market)
            
            # Extract position details
            position_size = position.get("size", 0)
            entry_price = position.get("entry_price", last_price)
            leverage = position.get("leverage", 1.0)
            side = position.get("side", Side.BUY if position_size > 0 else Side.SELL)
            liquidation_price = position.get("liquidation_price")
            
            # If no liquidation price provided, estimate it
            if not liquidation_price:
                liquidation_price = self._estimate_liquidation_price(
                    entry_price, leverage, side, 
                    maintenance_margin=position.get("maintenance_margin", 0.005)
                )
            
            # Calculate distance to liquidation
            if side == Side.BUY:  # Long position
                price_diff = last_price - liquidation_price
                distance_pct = price_diff / last_price if last_price > 0 else 0
            else:  # Short position
                price_diff = liquidation_price - last_price
                distance_pct = price_diff / last_price if last_price > 0 else 0
            
            # Unrealized PnL percentage
            if side == Side.BUY:  # Long position
                unrealized_pnl_pct = (last_price - entry_price) / entry_price
            else:  # Short position
                unrealized_pnl_pct = (entry_price - last_price) / entry_price
            
            # Calculate risk level (0-100, higher is riskier)
            # Based on how close we are to liquidation relative to action threshold
            risk_level = min(100, max(0, 100 - (distance_pct / self.config.action_threshold_pct * 100)))
            
            # Health score (0-100, higher is healthier)
            health_score = max(0, min(100, 100 - risk_level))
            
            # Determine if action is needed
            action_needed = distance_pct < self.config.action_threshold_pct
            
            # Create health metrics
            health_metrics = {
                "timestamp": current_time,
                "exchange": exchange.value,
                "market": market.value,
                "last_price": last_price,
                "position_size": position_size,
                "entry_price": entry_price,
                "leverage": leverage,
                "side": side.value,
                "liquidation_price": liquidation_price,
                "distance_to_liquidation": price_diff,
                "distance_pct": distance_pct,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "risk_level": risk_level,
                "health_score": health_score,
                "action_needed": action_needed,
                "action_threshold": self.config.action_threshold_pct
            }
            
            # Update position health
            self.position_health[(exchange, market)] = health_metrics
            
            # Log if risk is high
            if risk_level > 75:
                self.logger.warning(
                    f"HIGH LIQUIDATION RISK: {exchange.value}.{market.value} position "
                    f"health at {health_score:.1f}/100. {distance_pct*100:.2f}% from liquidation."
                )
            elif risk_level > 50:
                self.logger.info(
                    f"Elevated liquidation risk: {exchange.value}.{market.value} position "
                    f"health at {health_score:.1f}/100. {distance_pct*100:.2f}% from liquidation."
                )
            
            # Take automatic action if needed and enabled
            if action_needed and self.config.auto_prevention_enabled:
                self._take_preventive_action(exchange, market, health_metrics)
            
            return health_metrics
            
        except Exception as e:
            self.logger.error(f"Error checking liquidation risk for {exchange.value}.{market.value}: {e}")
            return {
                "timestamp": current_time,
                "exchange": exchange.value,
                "market": market.value,
                "error": str(e),
                "health_score": 0,  # Assume worst case when error occurs
                "action_needed": True
            }
    
    def _estimate_liquidation_price(
        self, 
        entry_price: float, 
        leverage: float, 
        side: Side,
        maintenance_margin: float = 0.005
    ) -> float:
        """
        Estimate the liquidation price based on position details.
        
        Args:
            entry_price: Position entry price
            leverage: Position leverage
            side: Position side (BUY/SELL)
            maintenance_margin: Maintenance margin requirement (default 0.5%)
            
        Returns:
            Estimated liquidation price
        """
        # Basic liquidation price formula
        # For longs: entry_price * (1 - (1 / leverage) + maintenance_margin)
        # For shorts: entry_price * (1 + (1 / leverage) - maintenance_margin)
        
        if side == Side.BUY:  # Long position
            liquidation_price = entry_price * (1 - (1 / leverage) + maintenance_margin)
        else:  # Short position
            liquidation_price = entry_price * (1 + (1 / leverage) - maintenance_margin)
            
        return liquidation_price
    
    def _take_preventive_action(self, exchange: Exchange, market: Market, health_metrics: Dict) -> Dict:
        """
        Take preventive action based on position health.
        
        Args:
            exchange: The exchange
            market: The market
            health_metrics: Position health metrics
            
        Returns:
            Result of the action taken
        """
        # Get position details from health metrics
        position_size = health_metrics.get("position_size", 0)
        side = Side(health_metrics.get("side", "BUY"))
        leverage = health_metrics.get("leverage", 1.0)
        risk_level = health_metrics.get("risk_level", 0)
        distance_pct = health_metrics.get("distance_pct", 1.0)
        
        # Skip if no position or very small risk
        if position_size == 0 or risk_level < 10:
            return {"action": "none", "reason": "no_significant_risk"}
            
        # Select strategy - use the default or override based on risk level
        strategy = self.config.default_strategy
        
        # For extreme risk, override with most aggressive strategy
        if risk_level > 90:
            strategy = PreventionStrategy.PARTIAL_CLOSE
            
        # Get existing prevention action if any
        existing_action = self.active_preventions.get((exchange, market), {})
        
        # Record action details
        action_details = {
            "timestamp": int(time.time()),
            "exchange": exchange.value,
            "market": market.value,
            "risk_level": risk_level,
            "distance_pct": distance_pct,
            "strategy": strategy.value,
            "position_size_before": position_size,
            "leverage_before": leverage
        }
        
        result = {}
        
        try:
            # Execute strategy
            if strategy == PreventionStrategy.STOP_LOSS:
                result = self._execute_stop_loss(exchange, market, health_metrics)
                
            elif strategy == PreventionStrategy.DYNAMIC_DELEVERAGING:
                result = self._execute_deleveraging(exchange, market, health_metrics)
                
            elif strategy == PreventionStrategy.AUTO_HEDGE:
                result = self._execute_auto_hedge(exchange, market, health_metrics)
                
            elif strategy == PreventionStrategy.PARTIAL_CLOSE:
                result = self._execute_partial_close(exchange, market, health_metrics)
                
            elif strategy == PreventionStrategy.COLLATERAL_MANAGEMENT:
                result = self._execute_add_collateral(exchange, market, health_metrics)
            
            # Update action details with result
            action_details.update(result)
            
            # Log the action
            self.logger.info(
                f"Liquidation prevention action for {exchange.value}.{market.value}: "
                f"{strategy.value} - {result.get('status', 'unknown')}"
            )
            
            # Update active prevention
            self.active_preventions[(exchange, market)] = action_details
            
            # Add to history
            self.prevention_history.append(action_details)
            
            return action_details
            
        except Exception as e:
            error_details = {
                "status": "error",
                "error": str(e)
            }
            
            action_details.update(error_details)
            self.prevention_history.append(action_details)
            
            self.logger.error(
                f"Error executing liquidation prevention for {exchange.value}.{market.value}: {e}"
            )
            
            return action_details
    
    def _execute_stop_loss(self, exchange: Exchange, market: Market, health_metrics: Dict) -> Dict:
        """
        Execute stop loss strategy by placing a stop loss order.
        
        Args:
            exchange: The exchange
            market: The market
            health_metrics: Position health metrics
            
        Returns:
            Result of the stop loss placement
        """
        position_size = health_metrics.get("position_size", 0)
        side = Side(health_metrics.get("side", "BUY"))
        liquidation_price = health_metrics.get("liquidation_price", 0)
        last_price = health_metrics.get("last_price", 0)
        
        # Calculate stop price with safety buffer
        if side == Side.BUY:  # Long position
            buffer_amount = (last_price - liquidation_price) * self.config.safety_buffer_pct
            stop_price = liquidation_price + buffer_amount
        else:  # Short position
            buffer_amount = (liquidation_price - last_price) * self.config.safety_buffer_pct
            stop_price = liquidation_price - buffer_amount
        
        # Place stop loss order
        stop_side = Side.SELL if side == Side.BUY else Side.BUY
        
        try:
            order_result = self.order_manager.create_order(
                exchange=exchange,
                market=market,
                side=stop_side,
                size=abs(position_size),
                order_type="STOP_MARKET",
                reduce_only=True,
                stop_price=stop_price
            )
            
            return {
                "status": "success",
                "action": "stop_loss",
                "stop_price": stop_price,
                "order_id": order_result.get("order_id")
            }
            
        except Exception as e:
            return {
                "status": "error",
                "action": "stop_loss",
                "error": str(e)
            }
    
    def _execute_deleveraging(self, exchange: Exchange, market: Market, health_metrics: Dict) -> Dict:
        """
        Execute dynamic deleveraging strategy by reducing position leverage.
        
        Args:
            exchange: The exchange
            market: The market
            health_metrics: Position health metrics
            
        Returns:
            Result of the deleveraging
        """
        current_leverage = health_metrics.get("leverage", 1.0)
        risk_level = health_metrics.get("risk_level", 0)
        
        # Skip if leverage is already low
        if current_leverage <= 1.5:
            return {
                "status": "skipped",
                "action": "deleveraging",
                "reason": "leverage_already_low",
                "current_leverage": current_leverage
            }
        
        # Calculate new leverage based on risk level
        # Higher risk = lower leverage
        leverage_reduction = self.config.deleveraging_factor * (risk_level / 100)
        new_leverage = max(1.0, current_leverage * (1 - leverage_reduction))
        
        try:
            # Use leverage manager to update leverage
            adjustment_result = self.leverage_manager.set_leverage(
                exchange=exchange,
                market=market,
                leverage=new_leverage
            )
            
            return {
                "status": "success",
                "action": "deleveraging",
                "old_leverage": current_leverage,
                "new_leverage": new_leverage,
                "leverage_reduction_pct": leverage_reduction * 100
            }
            
        except Exception as e:
            return {
                "status": "error",
                "action": "deleveraging",
                "error": str(e)
            }
    
    def _execute_auto_hedge(self, exchange: Exchange, market: Market, health_metrics: Dict) -> Dict:
        """
        Execute auto-hedge strategy by opening an opposite position on another venue.
        
        Args:
            exchange: The exchange
            market: The market
            health_metrics: Position health metrics
            
        Returns:
            Result of the hedge
        """
        position_size = health_metrics.get("position_size", 0)
        side = Side(health_metrics.get("side", "BUY"))
        
        # Find best alternative exchange for hedge
        hedge_exchange = self._find_hedge_exchange(exchange, market)
        
        if not hedge_exchange:
            return {
                "status": "skipped",
                "action": "auto_hedge",
                "reason": "no_suitable_exchange"
            }
        
        # Calculate hedge size (typically 50-100% of position)
        hedge_size = abs(position_size) * 0.5  # Hedge 50% by default
        hedge_side = Side.SELL if side == Side.BUY else Side.BUY
        
        try:
            # Place hedge order
            order_result = self.order_manager.create_order(
                exchange=hedge_exchange,
                market=market,
                side=hedge_side,
                size=hedge_size,
                order_type="MARKET"
            )
            
            return {
                "status": "success",
                "action": "auto_hedge",
                "original_exchange": exchange.value,
                "hedge_exchange": hedge_exchange.value,
                "hedge_size": hedge_size,
                "hedge_side": hedge_side.value,
                "order_id": order_result.get("order_id")
            }
            
        except Exception as e:
            return {
                "status": "error",
                "action": "auto_hedge",
                "error": str(e)
            }
    
    def _find_hedge_exchange(self, current_exchange: Exchange, market: Market) -> Optional[Exchange]:
        """
        Find the best exchange for hedging the current position.
        
        Args:
            current_exchange: Current exchange with the position
            market: The market
            
        Returns:
            Best exchange for hedging, or None if none found
        """
        # In a real implementation, this would consider:
        # - Available exchanges with the same market
        # - Liquidity on each exchange
        # - Transaction costs
        # - Funding rates
        
        # Simple implementation - find any other exchange with the market
        markets_config = self.config_manager.get_config("markets")
        
        for exchange_str in markets_config.get("exchanges", []):
            exchange = Exchange(exchange_str)
            
            if exchange == current_exchange:
                continue
                
            # Check if market exists on this exchange
            markets = markets_config.get("markets", {}).get(exchange_str, [])
            
            if market.value in markets:
                return exchange
                
        return None
    
    def _execute_partial_close(self, exchange: Exchange, market: Market, health_metrics: Dict) -> Dict:
        """
        Execute partial close strategy by reducing position size in steps.
        
        Args:
            exchange: The exchange
            market: The market
            health_metrics: Position health metrics
            
        Returns:
            Result of the partial close
        """
        position_size = health_metrics.get("position_size", 0)
        side = Side(health_metrics.get("side", "BUY"))
        distance_pct = health_metrics.get("distance_pct", 1.0)
        
        # Determine which threshold we've crossed
        threshold_index = None
        for i, threshold in enumerate(self.config.partial_close_thresholds):
            if distance_pct < threshold:
                threshold_index = i
                break
                
        if threshold_index is None:
            return {
                "status": "skipped",
                "action": "partial_close",
                "reason": "no_threshold_crossed"
            }
        
        # Get the percentage to close at this threshold
        close_pct = self.config.partial_close_pct[threshold_index]
        
        # Check if we already did this step
        existing_action = self.active_preventions.get((exchange, market), {})
        if existing_action.get("action") == "partial_close" and existing_action.get("threshold_index") == threshold_index:
            return {
                "status": "skipped",
                "action": "partial_close",
                "reason": "already_executed_this_step",
                "threshold_index": threshold_index
            }
        
        # Calculate size to close
        close_size = abs(position_size) * close_pct
        close_side = Side.SELL if side == Side.BUY else Side.BUY
        
        try:
            # Place order to reduce position
            order_result = self.order_manager.create_order(
                exchange=exchange,
                market=market,
                side=close_side,
                size=close_size,
                order_type="MARKET",
                reduce_only=True
            )
            
            return {
                "status": "success",
                "action": "partial_close",
                "threshold_index": threshold_index,
                "close_pct": close_pct * 100,
                "close_size": close_size,
                "remaining_size": abs(position_size) - close_size,
                "order_id": order_result.get("order_id")
            }
            
        except Exception as e:
            return {
                "status": "error",
                "action": "partial_close",
                "error": str(e)
            }
    
    def _execute_add_collateral(self, exchange: Exchange, market: Market, health_metrics: Dict) -> Dict:
        """
        Execute collateral management strategy by adding collateral to the position.
        
        Args:
            exchange: The exchange
            market: The market
            health_metrics: Position health metrics
            
        Returns:
            Result of adding collateral
        """
        # This is a simplified implementation
        # In reality, this would interact with the wallet management system
        
        position_size = health_metrics.get("position_size", 0)
        last_price = health_metrics.get("last_price", 0)
        distance_pct = health_metrics.get("distance_pct", 1.0)
        
        # Skip if we're not close enough to liquidation
        if distance_pct > self.config.action_threshold_pct * 0.5:
            return {
                "status": "skipped",
                "action": "add_collateral",
                "reason": "not_close_enough_to_liquidation"
            }
        
        # Calculate position value
        position_value = abs(position_size) * last_price
        
        # Calculate how much collateral to add
        # The closer to liquidation, the more we add
        proximity_factor = 1 - (distance_pct / self.config.action_threshold_pct)
        collateral_pct = self.config.max_additional_collateral_pct * proximity_factor
        additional_collateral = position_value * collateral_pct
        
        # In a real implementation, this would transfer funds to the account
        # or add collateral to the specific position
        
        # Simulate adding collateral
        try:
            # This would be replaced with actual API call to add collateral
            success = True
            
            if success:
                return {
                    "status": "success",
                    "action": "add_collateral",
                    "additional_collateral": additional_collateral,
                    "collateral_pct": collateral_pct * 100
                }
            else:
                return {
                    "status": "error",
                    "action": "add_collateral",
                    "error": "Failed to add collateral"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "action": "add_collateral",
                "error": str(e)
            }
    
    def get_health_summary(self) -> Dict:
        """
        Get a summary of the health of all positions.
        
        Returns:
            Summary of position health
        """
        summary = {
            "timestamp": int(time.time()),
            "positions": len(self.position_health),
            "positions_at_risk": 0,
            "average_health": 0,
            "critical_positions": []
        }
        
        # If no positions, return empty summary
        if not self.position_health:
            summary["average_health"] = 100  # Assume perfect health if no positions
            return summary
            
        total_health = 0
        critical_count = 0
        
        for (exchange, market), health in self.position_health.items():
            health_score = health.get("health_score", 0)
            total_health += health_score
            
            if health.get("action_needed", False):
                summary["positions_at_risk"] += 1
                
            if health_score < 50:  # Critical threshold
                critical_count += 1
                summary["critical_positions"].append({
                    "exchange": exchange.value,
                    "market": market.value,
                    "health_score": health_score,
                    "distance_pct": health.get("distance_pct", 0) * 100
                })
        
        # Calculate average health
        summary["average_health"] = total_health / len(self.position_health)
        summary["critical_positions_count"] = critical_count
        
        return summary
    
    def create_safe_stops(self, exchange: Exchange, market: Market, position: Dict) -> Dict:
        """
        Create safe stop loss orders for a position.
        
        Args:
            exchange: The exchange
            market: The market
            position: Position details
            
        Returns:
            Results of stop loss placement
        """
        # First check liquidation risk to get up-to-date metrics
        health_metrics = self.check_liquidation_risk(exchange, market, position, force_check=True)
        
        # Execute stop loss strategy
        return self._execute_stop_loss(exchange, market, health_metrics)
    
    def cancel_prevention_actions(self, exchange: Exchange, market: Market) -> Dict:
        """
        Cancel active prevention actions for a position.
        
        Args:
            exchange: The exchange
            market: The market
            
        Returns:
            Results of cancellation
        """
        # Check if there are active preventions
        active_prevention = self.active_preventions.get((exchange, market))
        
        if not active_prevention:
            return {
                "status": "success",
                "action": "cancel_prevention",
                "message": "No active prevention actions to cancel"
            }
            
        action_type = active_prevention.get("action")
        
        # Handle different types of preventions
        if action_type == "stop_loss":
            # Cancel stop loss order
            order_id = active_prevention.get("order_id")
            if order_id:
                try:
                    cancel_result = self.order_manager.cancel_order(
                        exchange=exchange,
                        market=market,
                        order_id=order_id
                    )
                    
                    # Remove from active preventions
                    del self.active_preventions[(exchange, market)]
                    
                    return {
                        "status": "success",
                        "action": "cancel_prevention",
                        "prevention_type": action_type,
                        "order_id": order_id
                    }
                except Exception as e:
                    return {
                        "status": "error",
                        "action": "cancel_prevention",
                        "error": str(e)
                    }
        
        # For other prevention types, just clear them
        del self.active_preventions[(exchange, market)]
        
        return {
            "status": "success",
            "action": "cancel_prevention",
            "prevention_type": action_type
        }
    
    def estimate_safe_leverage(self, exchange: Exchange, market: Market, desired_buffer_pct: float = 0.2) -> float:
        """
        Estimate a safe leverage level based on historical volatility.
        
        Args:
            exchange: The exchange
            market: The market
            desired_buffer_pct: Desired safety buffer as percentage of price
            
        Returns:
            Recommended safe leverage
        """
        try:
            # Get historical volatility data
            volatility_data = self._get_historical_volatility(exchange, market)
            
            if not volatility_data:
                return 1.0  # Default to 1x if no data
                
            # Get maximum daily volatility (as decimal)
            max_daily_volatility = volatility_data.get("max_daily_volatility", 0.05)
            
            # Use a more conservative measure (1.5x max volatility)
            expected_max_move = max_daily_volatility * 1.5
            
            # Calculate safe leverage based on desired buffer
            # For a long position: 1 / (expected_max_move / desired_buffer_pct)
            safe_leverage = desired_buffer_pct / expected_max_move
            
            # Cap at reasonable limits
            safe_leverage = min(20.0, max(1.0, safe_leverage))
            
            return round(safe_leverage, 1)  # Round to 1 decimal place
            
        except Exception as e:
            self.logger.error(f"Error estimating safe leverage for {exchange.value}.{market.value}: {e}")
            return 1.0  # Default to 1x on error
    
    def _get_historical_volatility(self, exchange: Exchange, market: Market) -> Dict:
        """
        Get historical volatility data for a market.
        
        Args:
            exchange: The exchange
            market: The market
            
        Returns:
            Dictionary with volatility metrics
        """
        try:
            # Get daily OHLCV data for the past 30 days
            ohlcv = self.market_data.get_candles(
                exchange=exchange,
                market=market,
                timeframe=Timeframe.DAY_1,
                limit=30
            )
            
            if not ohlcv or len(ohlcv) < 7:
                return {
                    "max_daily_volatility": 0.05,  # Default 5% daily volatility if no data
                    "avg_daily_volatility": 0.03
                }
            
            # Calculate daily returns
            closes = [candle[4] for candle in ohlcv]
            daily_returns = [
                (closes[i] - closes[i-1]) / closes[i-1] 
                for i in range(1, len(closes))
            ]
            
            # Calculate volatility metrics
            daily_volatility = np.std(daily_returns)
            max_daily_move = max(abs(np.max(daily_returns)), abs(np.min(daily_returns)))
            
            return {
                "max_daily_volatility": max_daily_move,
                "avg_daily_volatility": daily_volatility,
                "rolling_volatility": np.std(daily_returns[-7:])  # 7-day volatility
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility for {exchange.value}.{market.value}: {e}")
            return {
                "max_daily_volatility": 0.05,  # Default to 5% 
                "avg_daily_volatility": 0.03
            } 