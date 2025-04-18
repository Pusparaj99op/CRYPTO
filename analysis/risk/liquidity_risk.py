"""
Liquidity Risk Modeling

This module implements liquidity risk assessment tools for analyzing market
liquidity and its impact on trading and portfolio management.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional
from scipy import stats

def calculate_liquidity_metrics(orderbook_data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive liquidity metrics from orderbook data.
    
    Args:
        orderbook_data: DataFrame containing orderbook data with columns:
                       ['price', 'size', 'side'] (bid/ask)
        
    Returns:
        Dict[str, float]: Dictionary of liquidity metrics
    """
    # Separate bids and asks
    bids = orderbook_data[orderbook_data['side'] == 'bid'].sort_values('price', ascending=False)
    asks = orderbook_data[orderbook_data['side'] == 'ask'].sort_values('price', ascending=True)
    
    # Calculate basic metrics
    spread = asks['price'].iloc[0] - bids['price'].iloc[0]
    mid_price = (asks['price'].iloc[0] + bids['price'].iloc[0]) / 2
    relative_spread = spread / mid_price
    
    # Calculate depth metrics
    bid_depth = bids['size'].sum()
    ask_depth = asks['size'].sum()
    total_depth = bid_depth + ask_depth
    
    # Calculate weighted average prices
    bid_vwap = np.sum(bids['price'] * bids['size']) / bid_depth
    ask_vwap = np.sum(asks['price'] * asks['size']) / ask_depth
    
    # Calculate volume imbalance
    volume_imbalance = (bid_depth - ask_depth) / total_depth
    
    # Calculate price impact
    price_impact = (ask_vwap - bid_vwap) / mid_price
    
    return {
        'Spread': spread,
        'Relative_Spread': relative_spread,
        'Bid_Depth': bid_depth,
        'Ask_Depth': ask_depth,
        'Total_Depth': total_depth,
        'Bid_VWAP': bid_vwap,
        'Ask_VWAP': ask_vwap,
        'Volume_Imbalance': volume_imbalance,
        'Price_Impact': price_impact
    }

def assess_market_impact(trades: pd.DataFrame,
                        orderbook_snapshots: List[pd.DataFrame],
                        trade_size: float) -> Dict[str, float]:
    """
    Assess market impact of a trade.
    
    Args:
        trades: DataFrame of historical trades
        orderbook_snapshots: List of orderbook snapshots
        trade_size: Size of the trade to assess
        
    Returns:
        Dict[str, float]: Market impact metrics
    """
    # Calculate average trade size
    avg_trade_size = trades['size'].mean()
    
    # Calculate relative trade size
    relative_size = trade_size / avg_trade_size
    
    # Calculate historical price impact
    price_changes = []
    for i in range(1, len(trades)):
        price_change = (trades['price'].iloc[i] - trades['price'].iloc[i-1]) / trades['price'].iloc[i-1]
        price_changes.append(price_change)
    
    # Calculate impact coefficient
    impact_coef = np.mean(np.abs(price_changes)) / avg_trade_size
    
    # Estimate market impact
    estimated_impact = impact_coef * trade_size
    
    # Calculate liquidation cost
    liquidation_cost = calculate_liquidation_cost(orderbook_snapshots, trade_size)
    
    return {
        'Relative_Size': relative_size,
        'Impact_Coefficient': impact_coef,
        'Estimated_Impact': estimated_impact,
        'Liquidation_Cost': liquidation_cost
    }

def calculate_liquidation_cost(orderbook_snapshots: List[pd.DataFrame],
                             position_size: float) -> float:
    """
    Calculate liquidation cost for a given position size.
    
    Args:
        orderbook_snapshots: List of orderbook snapshots
        position_size: Size of position to liquidate
        
    Returns:
        float: Estimated liquidation cost
    """
    total_cost = 0.0
    remaining_size = position_size
    
    for snapshot in orderbook_snapshots:
        # Get asks sorted by price
        asks = snapshot[snapshot['side'] == 'ask'].sort_values('price')
        
        for _, order in asks.iterrows():
            if remaining_size <= 0:
                break
                
            # Calculate size to take from this order
            size_to_take = min(remaining_size, order['size'])
            cost = size_to_take * order['price']
            total_cost += cost
            remaining_size -= size_to_take
    
    # Calculate average price
    avg_price = total_cost / position_size
    
    # Get best bid price
    best_bid = max(snapshot[snapshot['side'] == 'bid']['price'])
    
    # Calculate cost relative to best bid
    relative_cost = (avg_price - best_bid) / best_bid
    
    return relative_cost

def calculate_liquidity_adjusted_var(returns: Union[np.ndarray, pd.Series],
                                   orderbook_data: pd.DataFrame,
                                   confidence_level: float = 0.95) -> float:
    """
    Calculate liquidity-adjusted Value at Risk.
    
    Args:
        returns: Array or Series of returns
        orderbook_data: Orderbook data for liquidity assessment
        confidence_level: Confidence level for VaR calculation
        
    Returns:
        float: Liquidity-adjusted VaR
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # Calculate regular VaR
    var = np.percentile(returns, (1 - confidence_level) * 100)
    
    # Calculate liquidity metrics
    liquidity_metrics = calculate_liquidity_metrics(orderbook_data)
    
    # Adjust VaR for liquidity
    spread_impact = liquidity_metrics['Relative_Spread'] / 2
    depth_impact = 1 / (1 + liquidity_metrics['Total_Depth'])
    
    liquidity_adjusted_var = var * (1 + spread_impact + depth_impact)
    
    return liquidity_adjusted_var

def plot_liquidity_profile(orderbook_data: pd.DataFrame,
                         ax=None) -> None:
    """
    Plot liquidity profile from orderbook data.
    
    Args:
        orderbook_data: DataFrame containing orderbook data
        ax: Matplotlib axis object (if None, will create new figure)
    """
    import matplotlib.pyplot as plt
    
    # Separate bids and asks
    bids = orderbook_data[orderbook_data['side'] == 'bid'].sort_values('price', ascending=False)
    asks = orderbook_data[orderbook_data['side'] == 'ask'].sort_values('price', ascending=True)
    
    # Calculate cumulative depth
    bids['cumulative_depth'] = bids['size'].cumsum()
    asks['cumulative_depth'] = asks['size'].cumsum()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot liquidity profile
    ax.plot(bids['cumulative_depth'], bids['price'], 'g-', label='Bids')
    ax.plot(asks['cumulative_depth'], asks['price'], 'r-', label='Asks')
    
    ax.set_xlabel('Cumulative Depth')
    ax.set_ylabel('Price')
    ax.set_title('Liquidity Profile')
    ax.legend()
    ax.grid(True)
    
    if ax is None:
        plt.show() 