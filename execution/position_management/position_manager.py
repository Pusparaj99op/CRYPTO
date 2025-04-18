import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
import time
from dataclasses import dataclass

from .liquidation_prevention import LiquidationManager, LiquidationConfig
from .exposure_management import ExposureManager, ExposureConfig
from .order_size_calculator import OrderSizeCalculator, OrderSizeConfig

logger = logging.getLogger(__name__)

@dataclass
class PositionManagerConfig:
    """Configuration for the PositionManager"""
    # General settings
    check_interval: float = 60.0  # Seconds between position checks
    emergency_check_interval: float = 10.0  # Faster checks during emergencies
    max_leverage: float = 10.0  # Maximum allowed leverage
    auto_deleverage: bool = True  # Automatically reduce leverage when needed
    
    # Risk limits
    max_drawdown_percent: float = 20.0  # Maximum allowed drawdown
    max_open_positions: int = 15  # Maximum number of concurrent positions
    
    # Component configs
    liquidation_config: LiquidationConfig = None
    exposure_config: ExposureConfig = None
    order_size_config: OrderSizeConfig = None


class PositionManager:
    """
    Comprehensive position manager that integrates liquidation prevention,
    exposure management, and order sizing.
    
    This class coordinates all aspects of position management to ensure
    that trading positions adhere to risk parameters and don't endanger
    the account.
    """
    
    def __init__(
        self,
        config: PositionManagerConfig = None,
        execute_trade_callback: Optional[Callable] = None,
        update_order_callback: Optional[Callable] = None,
        notification_callback: Optional[Callable] = None
    ):
        self.config = config or PositionManagerConfig()
        
        # Initialize component managers
        self.liquidation_manager = LiquidationManager(
            config=self.config.liquidation_config or LiquidationConfig()
        )
        
        self.exposure_manager = ExposureManager(
            config=self.config.exposure_config or ExposureConfig()
        )
        
        self.order_size_calculator = OrderSizeCalculator(
            config=self.config.order_size_config or OrderSizeConfig()
        )
        
        # Callback functions
        self.execute_trade = execute_trade_callback
        self.update_order = update_order_callback
        self.send_notification = notification_callback
        
        # State tracking
        self.positions = {}  # Current positions
        self.account_balance = 0
        self.margin_used = 0
        self.available_margin = 0
        self.current_prices = {}
        self.is_emergency_mode = False
        self.last_check_time = 0
        self.position_history = []
        self.active_orders = {}
        self.market_data = {}
        
    def update_market_data(self, market: str, data: Dict[str, Any]) -> None:
        """Update market metadata for a specific market"""
        if market not in self.market_data:
            self.market_data[market] = {}
            
        self.market_data[market].update(data)
        
        # Propagate relevant data to subcomponents
        if 'volatility' in data:
            self.exposure_manager.update_market_volatility(market, data['volatility'])
            self.order_size_calculator.update_market_info(market, {'volatility': data['volatility']})
    
    def update_account_info(
        self,
        balance: float,
        used_margin: float = None,
        available_margin: float = None
    ) -> None:
        """Update account balance and margin information"""
        self.account_balance = balance
        
        if used_margin is not None:
            self.margin_used = used_margin
            
        if available_margin is not None:
            self.available_margin = available_margin
        else:
            # Estimate if not provided
            self.available_margin = max(0, self.account_balance - self.margin_used)
    
    def update_position(self, market: str, position_data: Dict[str, Any]) -> None:
        """
        Update information about a specific position
        
        Args:
            market: Market symbol
            position_data: Position details including size, entry price, etc.
        """
        self.positions[market] = position_data
        
        # Calculate total margin used if not directly provided
        if 'margin_used' in position_data:
            self._recalculate_total_margin()
            
        # Update position history
        self.position_history.append({
            'timestamp': time.time(),
            'market': market,
            'data': dict(position_data)  # Copy to avoid reference issues
        })
        
        # Limit history size
        if len(self.position_history) > 1000:
            self.position_history = self.position_history[-1000:]
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update current prices for multiple markets
        
        Args:
            prices: Dictionary mapping market symbols to current prices
        """
        self.current_prices.update(prices)
    
    def check_positions(self) -> Dict[str, Any]:
        """
        Perform a comprehensive check of all positions and take
        preventive actions if needed.
        
        Returns:
            Dictionary with check results
        """
        current_time = time.time()
        
        # Throttle checks
        interval = self.config.emergency_check_interval if self.is_emergency_mode else self.config.check_interval
        if current_time - self.last_check_time < interval:
            return {"status": "SKIPPED", "reason": "Check interval not elapsed"}
        
        self.last_check_time = current_time
        
        results = {
            "liquidation_check": None,
            "exposure_check": None,
            "issues_detected": False,
            "actions_taken": [],
            "status": "OK"
        }
        
        # Skip if no positions
        if not self.positions:
            return results
        
        # 1. Check liquidation risks
        liq_check = self._check_liquidation_risks()
        results["liquidation_check"] = liq_check
        
        if liq_check["has_risks"]:
            results["issues_detected"] = True
            results["status"] = "WARNING" if liq_check["highest_risk"] < 0.8 else "DANGER"
            
            # Take preventive actions for liquidation risks
            actions = self._handle_liquidation_risks(liq_check)
            results["actions_taken"].extend(actions)
        
        # 2. Check exposure limits
        exp_check = self._check_exposure_limits()
        results["exposure_check"] = exp_check
        
        if not exp_check["is_compliant"]:
            results["issues_detected"] = True
            results["status"] = "WARNING" if results["status"] == "OK" else results["status"]
            
            # Take actions to address exposure issues
            actions = self._handle_exposure_issues(exp_check)
            results["actions_taken"].extend(actions)
        
        # 3. Set emergency mode if needed
        self.is_emergency_mode = results["status"] == "DANGER"
        
        # 4. Send notifications if needed
        if results["issues_detected"] and self.send_notification:
            notification = {
                "type": results["status"],
                "issues": {
                    "liquidation": liq_check["has_risks"],
                    "exposure": not exp_check["is_compliant"]
                },
                "actions_taken": results["actions_taken"]
            }
            self.send_notification(notification)
        
        return results
    
    def _check_liquidation_risks(self) -> Dict[str, Any]:
        """Check for liquidation risks across all positions"""
        # Update liquidation manager with current data
        self.liquidation_manager.update_position_info(
            positions=self.positions,
            balance=self.account_balance,
            current_prices=self.current_prices
        )
        
        # Check risks
        risk_assessment = self.liquidation_manager.check_liquidation_risk()
        
        # Calculate highest risk
        max_risk = 0
        for market, data in risk_assessment.items():
            if data['risk_factor'] > max_risk:
                max_risk = data['risk_factor']
        
        return {
            "has_risks": max_risk > 0.5,  # Threshold for danger
            "risk_assessment": risk_assessment,
            "highest_risk": max_risk
        }
    
    def _handle_liquidation_risks(self, risk_check: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Take actions to address liquidation risks"""
        actions_taken = []
        
        # Get recommended actions
        recommendations = self.liquidation_manager.get_recommended_actions(
            risk_check["risk_assessment"]
        )
        
        # Execute preventive actions if callback is available
        if self.execute_trade and recommendations:
            for action in recommendations:
                # Execute action
                result = self.liquidation_manager.execute_preventive_actions(
                    action, self.execute_trade
                )
                
                if result["success"]:
                    actions_taken.append({
                        "type": "LIQUIDATION_PREVENTION",
                        "action": action["action"],
                        "market": action["market"],
                        "result": "SUCCESS"
                    })
                else:
                    actions_taken.append({
                        "type": "LIQUIDATION_PREVENTION",
                        "action": action["action"],
                        "market": action["market"],
                        "result": "FAILED",
                        "reason": result.get("error", "Unknown error")
                    })
        
        return actions_taken
    
    def _check_exposure_limits(self) -> Dict[str, Any]:
        """Check if current positions exceed exposure limits"""
        # Calculate current exposure
        self.exposure_manager.calculate_exposure(
            positions=self.positions,
            account_value=self.account_balance
        )
        
        # Check if limits are exceeded
        return self.exposure_manager.check_exposure_limits()
    
    def _handle_exposure_issues(self, exposure_check: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Take actions to address exposure issues"""
        actions_taken = []
        
        # Get rebalance suggestions
        suggestions = self.exposure_manager.suggest_exposure_rebalance()
        
        # Execute rebalance actions if callback is available
        if self.execute_trade and suggestions:
            for suggestion in suggestions:
                action_type = suggestion["action"]
                
                if action_type == "REDUCE_POSITION":
                    # Calculate reduction amount
                    market = suggestion["market"]
                    reduction_amount = suggestion["target_reduction"] * self.account_balance
                    
                    # Execute reduction
                    if market in self.positions:
                        current_position = self.positions[market]
                        current_size = current_position.get("size", 0)
                        
                        # Calculate percentage to reduce
                        reduction_pct = min(1.0, reduction_amount / current_size) if current_size > 0 else 0
                        
                        if reduction_pct > 0:
                            result = self.execute_trade({
                                "market": market,
                                "action": "reduce_position",
                                "reduce_by": reduction_pct,
                                "reason": "Exposure management"
                            })
                            
                            actions_taken.append({
                                "type": "EXPOSURE_MANAGEMENT",
                                "action": "REDUCE_POSITION",
                                "market": market,
                                "reduction_pct": reduction_pct,
                                "result": "SUCCESS" if result.get("success") else "FAILED"
                            })
                
                elif action_type == "HEDGE_EXPOSURE":
                    # Implement hedging logic
                    # This would create offsetting positions
                    markets = suggestion.get("markets", [])
                    
                    for market in markets:
                        # Check if we can determine an appropriate hedge
                        # This is a simplified approach; real hedging is more complex
                        if market in self.positions:
                            position = self.positions[market]
                            side = position.get("side", "").lower()
                            
                            # Create opposite position in a correlated asset
                            hedge_market = self._find_hedge_instrument(market)
                            
                            if hedge_market:
                                # Calculate hedge size based on correlation
                                hedge_size = self._calculate_hedge_size(market, hedge_market)
                                
                                result = self.execute_trade({
                                    "market": hedge_market,
                                    "action": "open_position",
                                    "side": "sell" if side == "buy" else "buy",
                                    "size": hedge_size,
                                    "reason": "Hedge exposure"
                                })
                                
                                actions_taken.append({
                                    "type": "EXPOSURE_MANAGEMENT",
                                    "action": "HEDGE_POSITION",
                                    "source_market": market,
                                    "hedge_market": hedge_market,
                                    "size": hedge_size,
                                    "result": "SUCCESS" if result.get("success") else "FAILED"
                                })
        
        return actions_taken
    
    def _recalculate_total_margin(self) -> None:
        """Recalculate total margin used by all positions"""
        total_margin = 0
        for market, position in self.positions.items():
            margin = position.get('margin_used', 0)
            total_margin += margin
        
        self.margin_used = total_margin
        self.available_margin = max(0, self.account_balance - self.margin_used)
    
    def _find_hedge_instrument(self, market: str) -> Optional[str]:
        """Find a suitable instrument to hedge the given market"""
        # This would use correlation data to find a suitable hedge
        # For simplicity, we're returning a placeholder
        # In a real implementation, this would use the correlation matrix
        if market in self.exposure_manager.correlation_matrix:
            # Find most negatively correlated instrument
            correlations = []
            for key, value in self.exposure_manager.correlation_matrix.items():
                if market in key:
                    other = key[0] if key[1] == market else key[1]
                    correlations.append((other, value))
            
            if correlations:
                # Sort by correlation (ascending for negative correlation)
                correlations.sort(key=lambda x: x[1])
                return correlations[0][0]
        
        return None
    
    def _calculate_hedge_size(self, source_market: str, hedge_market: str) -> float:
        """Calculate appropriate size for a hedging position"""
        if source_market not in self.positions:
            return 0.0
            
        source_position = self.positions[source_market]
        source_size = source_position.get('size', 0)
        source_price = self.current_prices.get(source_market, 0)
        hedge_price = self.current_prices.get(hedge_market, 0)
        
        # Get correlation between markets
        correlation = 0.5  # Default medium correlation
        key = (source_market, hedge_market) if source_market < hedge_market else (hedge_market, source_market)
        if key in self.exposure_manager.correlation_matrix:
            correlation = abs(self.exposure_manager.correlation_matrix[key])
        
        # Calculate notional value of source position
        source_notional = source_size * source_price
        
        # Calculate hedge size based on correlation and price
        # Higher correlation = smaller hedge needed
        hedge_factor = 1.0 - correlation
        hedge_notional = source_notional * hedge_factor
        
        # Convert to actual size in hedge market
        if hedge_price > 0:
            return hedge_notional / hedge_price
        else:
            return 0.0
    
    def calculate_order_size(
        self,
        market: str,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate appropriate size for a new order
        
        Args:
            market: Market symbol
            entry_price: Planned entry price
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            
        Returns:
            Dictionary with order size details
        """
        # First check exposure limits
        exposure_metrics = self.exposure_manager.get_exposure_metrics()
        
        # Get market specific info
        market_info = self.market_data.get(market, {})
        volatility = market_info.get('volatility', 0.1)
        leverage = market_info.get('leverage', 1.0)
        
        # Calculate base size
        size_result = self.order_size_calculator.calculate_order_size(
            market=market,
            account_balance=self.account_balance,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            market_volatility=volatility,
            leverage=leverage
        )
        
        # Adjust size based on exposure limits
        if exposure_metrics['total_exposure'] >= self.exposure_manager.config.max_total_exposure * 0.9:
            # Approaching total exposure limit, reduce size
            reduction_factor = 0.5
            size_result['order_size'] *= reduction_factor
            size_result['position_units'] *= reduction_factor
            size_result['exposure_limited'] = True
            
        # Ensure leverage limits are respected
        if leverage > self.config.max_leverage:
            # Reduce size to match leverage limit
            leverage_factor = self.config.max_leverage / leverage
            size_result['order_size'] *= leverage_factor
            size_result['position_units'] *= leverage_factor
            size_result['leverage_limited'] = True
            size_result['leverage'] = self.config.max_leverage
            
        # Add market info
        size_result['market_volatility'] = volatility
        
        return size_result
    
    def get_position_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of all positions and risks
        
        Returns:
            Dictionary with position summary details
        """
        # Get liquidation stats
        liquidation_stats = self.liquidation_manager.get_liquidation_stats()
        
        # Get exposure metrics
        exposure_metrics = self.exposure_manager.get_exposure_metrics()
        
        # Calculate portfolio stats
        total_pnl = 0
        total_value = 0
        
        for market, position in self.positions.items():
            pnl = position.get('unrealized_pnl', 0)
            total_pnl += pnl
            
            size = position.get('size', 0)
            price = self.current_prices.get(market, 0)
            value = size * price
            total_value += value
        
        # Calculate overall metrics
        account_leverage = total_value / self.account_balance if self.account_balance > 0 else 0
        
        return {
            "account_balance": self.account_balance,
            "margin_used": self.margin_used,
            "available_margin": self.available_margin,
            "total_position_value": total_value,
            "total_pnl": total_pnl,
            "account_leverage": account_leverage,
            "position_count": len(self.positions),
            "liquidation_status": {
                "active_warnings": liquidation_stats.get("active_warnings", 0),
                "emergency_actions": liquidation_stats.get("emergency_actions", 0),
                "highest_risk_market": liquidation_stats.get("highest_risk_market", None),
                "highest_risk_factor": liquidation_stats.get("highest_risk_factor", 0)
            },
            "exposure_status": {
                "total_exposure": exposure_metrics.get("total_exposure", 0),
                "direction": exposure_metrics.get("direction", "NEUTRAL"),
                "compliance_status": exposure_metrics.get("compliance_status", "UNKNOWN"),
                "highest_single_exposure": exposure_metrics.get("highest_single_exposure", 0)
            },
            "emergency_mode": self.is_emergency_mode
        } 