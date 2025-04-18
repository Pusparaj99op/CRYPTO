#!/usr/bin/env python
"""
Example script demonstrating how to use the position management system.
"""

import logging
import time
import random
from typing import Dict, Any

from execution.position_management import (
    PositionManager,
    PositionManagerConfig,
    LiquidationConfig,
    ExposureConfig,
    OrderSizeConfig
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample data for demonstration
SAMPLE_MARKETS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT"]
SAMPLE_CORRELATION_MATRIX = {
    ("BTC-USDT", "ETH-USDT"): 0.85,
    ("BTC-USDT", "SOL-USDT"): 0.75,
    ("BTC-USDT", "BNB-USDT"): 0.65,
    ("BTC-USDT", "XRP-USDT"): 0.45,
    ("ETH-USDT", "SOL-USDT"): 0.82,
    ("ETH-USDT", "BNB-USDT"): 0.60,
    ("ETH-USDT", "XRP-USDT"): 0.40,
    ("SOL-USDT", "BNB-USDT"): 0.55,
    ("SOL-USDT", "XRP-USDT"): 0.35,
    ("BNB-USDT", "XRP-USDT"): 0.50,
}
SAMPLE_MARKET_CATEGORIES = {
    "BTC-USDT": "major",
    "ETH-USDT": "major",
    "SOL-USDT": "altcoin",
    "BNB-USDT": "exchange",
    "XRP-USDT": "payment",
}


def mock_execute_trade(trade_params: Dict[str, Any]) -> Dict[str, Any]:
    """Mock function to simulate executing a trade."""
    logger.info(f"EXECUTING TRADE: {trade_params}")
    # In a real implementation, this would interact with an exchange API
    return {"success": True, "order_id": "mock-order-" + str(int(time.time()))}


def mock_send_notification(notification: Dict[str, Any]) -> None:
    """Mock function to simulate sending notifications."""
    logger.info(f"NOTIFICATION: {notification}")
    # In a real implementation, this would send an email, SMS, etc.


def generate_mock_position(market: str, price: float) -> Dict[str, Any]:
    """Generate a mock position for testing."""
    side = random.choice(["buy", "sell"])
    leverage = random.choice([1, 2, 3, 5, 10])
    size = random.uniform(100, 1000)
    entry_price = price * random.uniform(0.95, 1.05)
    
    return {
        "market": market,
        "side": side,
        "size": size,
        "entry_price": entry_price,
        "current_price": price,
        "leverage": leverage,
        "margin_used": size / leverage,
        "unrealized_pnl": size * (price - entry_price) * (1 if side == "buy" else -1),
        "liquidation_price": entry_price * (0.9 if side == "buy" else 1.1)
    }


def main():
    # Create custom configurations
    liquidation_config = LiquidationConfig(
        safety_margin=0.2,  # 20% buffer before liquidation
        auto_deleverage=True,
        emergency_threshold=0.8
    )
    
    exposure_config = ExposureConfig(
        max_total_exposure=1.0,  # Max 100% of capital
        max_single_market_exposure=0.25,  # Max 25% in single market
        correlation_threshold=0.7
    )
    
    order_size_config = OrderSizeConfig(
        risk_per_trade=0.01,  # 1% risk per trade
        position_sizing_method="risk_based"
    )
    
    position_config = PositionManagerConfig(
        check_interval=30.0,  # Check every 30 seconds
        max_leverage=10.0,
        liquidation_config=liquidation_config,
        exposure_config=exposure_config,
        order_size_config=order_size_config
    )
    
    # Create position manager with callbacks
    position_manager = PositionManager(
        config=position_config,
        execute_trade_callback=mock_execute_trade,
        notification_callback=mock_send_notification
    )
    
    # Set up correlation matrix
    position_manager.exposure_manager.correlation_matrix = SAMPLE_CORRELATION_MATRIX
    position_manager.exposure_manager.market_categories = SAMPLE_MARKET_CATEGORIES
    
    # Initialize account balance
    account_balance = 10000.0
    position_manager.update_account_info(account_balance)
    
    # Generate random current prices
    current_prices = {
        "BTC-USDT": random.uniform(25000, 30000),
        "ETH-USDT": random.uniform(1500, 2000),
        "SOL-USDT": random.uniform(50, 100),
        "BNB-USDT": random.uniform(200, 300),
        "XRP-USDT": random.uniform(0.5, 0.7)
    }
    position_manager.update_prices(current_prices)
    
    # Update market volatility data
    for market in SAMPLE_MARKETS:
        volatility = random.uniform(0.05, 0.30)  # 5-30% volatility
        position_manager.update_market_data(market, {
            "volatility": volatility,
            "current_price": current_prices[market]
        })
    
    # Create some sample positions
    for market in ["BTC-USDT", "ETH-USDT", "SOL-USDT"]:
        position = generate_mock_position(market, current_prices[market])
        position_manager.update_position(market, position)
    
    # Run a position check
    logger.info("Running initial position check")
    check_results = position_manager.check_positions()
    logger.info(f"Check results: {check_results}")
    
    # Calculate a new order size
    market = "BNB-USDT"
    entry_price = current_prices[market]
    stop_loss = entry_price * 0.95  # 5% stop loss
    take_profit = entry_price * 1.15  # 15% take profit
    
    logger.info(f"Calculating order size for {market}")
    size_result = position_manager.calculate_order_size(
        market=market,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit
    )
    logger.info(f"Recommended order size: {size_result}")
    
    # Simulate a liquidation risk scenario
    logger.info("Simulating liquidation risk")
    # Update BTC price to create risk
    btc_price = current_prices["BTC-USDT"] * 0.85  # 15% drop
    position_manager.update_prices({"BTC-USDT": btc_price})
    
    # Get updated position summary
    summary = position_manager.get_position_summary()
    logger.info(f"Position summary: {summary}")
    
    # Run check again to see preventive actions
    check_results = position_manager.check_positions()
    logger.info(f"Updated check results: {check_results}")


if __name__ == "__main__":
    main() 