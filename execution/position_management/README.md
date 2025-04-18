# Position Management System

This package provides a comprehensive solution for managing trading positions in cryptocurrency markets, with a focus on risk management, liquidation prevention, and optimal position sizing.

## Components

### PositionManager

The main class that integrates all position management functionality. It orchestrates:
- Liquidation risk monitoring and prevention
- Exposure management across markets
- Position sizing and order calculation
- Automatic remedial actions when risk thresholds are breached

### LiquidationManager

Specialized in monitoring and preventing liquidation events:
- Continuously monitors margin ratios and liquidation risk
- Estimates time to liquidation based on price movements
- Recommends and executes preventive actions like adding margin, reducing leverage, or closing positions
- Runs stress tests to identify potential liquidation scenarios

### ExposureManager

Manages trading exposure across markets:
- Enforces maximum exposure limits (total and per-market)
- Monitors correlated asset exposure
- Ensures proper diversification
- Suggests rebalancing actions when limits are exceeded

### OrderSizeCalculator

Calculates appropriate position sizes based on risk parameters:
- Supports multiple sizing methods: risk-based, volatility-adjusted, and Kelly criterion
- Accounts for leverage and market volatility
- Provides scaling functionality for gradual position building
- Calculates position adjustments for existing positions

## Usage

Basic example:

```python
from execution.position_management import (
    PositionManager, 
    PositionManagerConfig,
    LiquidationConfig
)

# Create configuration
config = PositionManagerConfig(
    check_interval=60.0,  # Check positions every 60 seconds
    max_leverage=10.0,
    liquidation_config=LiquidationConfig(
        safety_margin=0.2,  # 20% buffer before liquidation
        auto_deleverage=True
    )
)

# Create position manager with callbacks
def execute_trade(trade_params):
    # Implement actual trading logic here
    pass

def send_notification(notification):
    # Implement notification logic here
    pass

position_manager = PositionManager(
    config=config,
    execute_trade_callback=execute_trade,
    notification_callback=send_notification
)

# Update account information
position_manager.update_account_info(balance=10000.0)

# Update position information
position_manager.update_position("BTC-USDT", {
    "size": 1000.0,
    "entry_price": 28000.0,
    "leverage": 5.0,
    "side": "buy"
})

# Update current prices
position_manager.update_prices({
    "BTC-USDT": 27500.0
})

# Run position checks
check_results = position_manager.check_positions()

# Calculate order size for a new trade
size_result = position_manager.calculate_order_size(
    market="ETH-USDT",
    entry_price=1800.0,
    stop_loss=1700.0
)
```

See the `examples/position_management_example.py` file for a more detailed example.

## Features

- **Auto-hedging**: Automatically creates hedge positions when exposure exceeds thresholds
- **Dynamic position sizing**: Adjusts position sizes based on market volatility
- **Stress testing**: Tests portfolio against extreme market scenarios
- **Multi-market exposure management**: Monitors correlated market exposures
- **Scaling strategies**: Supports gradual position building with multiple entry levels
- **Kelly criterion**: Optimal position sizing based on win rate and risk/reward
- **Emergency actions**: Automatically executes preventive actions when liquidation risk is high

## Configuration

Each component has its own configuration class that allows customization of risk parameters:

- `PositionManagerConfig`: Main configuration for the position manager
- `LiquidationConfig`: Liquidation prevention parameters
- `ExposureConfig`: Exposure management parameters
- `OrderSizeConfig`: Position sizing parameters

These configurations can be adjusted to match your risk tolerance and trading style. 