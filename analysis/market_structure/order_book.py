import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Order:
    """Represents a single order in the order book."""
    order_id: str
    price: float
    quantity: float
    side: str  # 'buy' or 'sell'
    timestamp: float
    order_type: str  # 'limit', 'market', 'stop'
    trader_id: Optional[str] = None

class OrderBook:
    """
    Implementation of an order book for modeling market microstructure.
    Includes order matching, price formation, and order book dynamics.
    """
    
    def __init__(self):
        """Initialize the order book."""
        self.bids = defaultdict(list)  # price -> [orders]
        self.asks = defaultdict(list)  # price -> [orders]
        self.order_map = {}  # order_id -> order
        self.trades = []
        self.price_history = []
        self.volume_history = []
        self.spread_history = []
        
    def add_order(self, order: Order) -> List[Dict[str, Any]]:
        """
        Add an order to the order book and match if possible.
        
        Args:
            order (Order): The order to add
            
        Returns:
            List[Dict]: List of trades executed
        """
        self.order_map[order.order_id] = order
        
        if order.side == 'buy':
            self._match_buy_order(order)
        else:
            self._match_sell_order(order)
            
        self._update_history()
        return self.trades[-1] if self.trades else []
    
    def _match_buy_order(self, order: Order) -> None:
        """Match a buy order against existing sell orders."""
        executed_trades = []
        remaining_quantity = order.quantity
        
        # Sort asks by price (ascending)
        ask_prices = sorted(self.asks.keys())
        
        for price in ask_prices:
            if price > order.price and order.order_type == 'limit':
                break
                
            orders_at_price = self.asks[price]
            for ask_order in orders_at_price[:]:
                if remaining_quantity <= 0:
                    break
                    
                trade_quantity = min(remaining_quantity, ask_order.quantity)
                trade = {
                    'price': price,
                    'quantity': trade_quantity,
                    'buyer': order.trader_id,
                    'seller': ask_order.trader_id,
                    'timestamp': order.timestamp
                }
                executed_trades.append(trade)
                
                # Update quantities
                remaining_quantity -= trade_quantity
                ask_order.quantity -= trade_quantity
                
                if ask_order.quantity <= 0:
                    orders_at_price.remove(ask_order)
                    del self.order_map[ask_order.order_id]
                    
        if remaining_quantity > 0 and order.order_type == 'limit':
            self.bids[order.price].append(order)
            
        if executed_trades:
            self.trades.extend(executed_trades)
    
    def _match_sell_order(self, order: Order) -> None:
        """Match a sell order against existing buy orders."""
        executed_trades = []
        remaining_quantity = order.quantity
        
        # Sort bids by price (descending)
        bid_prices = sorted(self.bids.keys(), reverse=True)
        
        for price in bid_prices:
            if price < order.price and order.order_type == 'limit':
                break
                
            orders_at_price = self.bids[price]
            for bid_order in orders_at_price[:]:
                if remaining_quantity <= 0:
                    break
                    
                trade_quantity = min(remaining_quantity, bid_order.quantity)
                trade = {
                    'price': price,
                    'quantity': trade_quantity,
                    'buyer': bid_order.trader_id,
                    'seller': order.trader_id,
                    'timestamp': order.timestamp
                }
                executed_trades.append(trade)
                
                # Update quantities
                remaining_quantity -= trade_quantity
                bid_order.quantity -= trade_quantity
                
                if bid_order.quantity <= 0:
                    orders_at_price.remove(bid_order)
                    del self.order_map[bid_order.order_id]
                    
        if remaining_quantity > 0 and order.order_type == 'limit':
            self.asks[order.price].append(order)
            
        if executed_trades:
            self.trades.extend(executed_trades)
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id (str): ID of the order to cancel
            
        Returns:
            bool: True if order was cancelled, False otherwise
        """
        if order_id not in self.order_map:
            return False
            
        order = self.order_map[order_id]
        if order.side == 'buy':
            orders = self.bids[order.price]
        else:
            orders = self.asks[order.price]
            
        for i, o in enumerate(orders):
            if o.order_id == order_id:
                orders.pop(i)
                del self.order_map[order_id]
                return True
                
        return False
    
    def get_best_bid(self) -> Optional[float]:
        """Get the best (highest) bid price."""
        return max(self.bids.keys()) if self.bids else None
    
    def get_best_ask(self) -> Optional[float]:
        """Get the best (lowest) ask price."""
        return min(self.asks.keys()) if self.asks else None
    
    def get_spread(self) -> Optional[float]:
        """Get the current bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return best_ask - best_bid
        return None
    
    def get_mid_price(self) -> Optional[float]:
        """Get the current mid price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        return None
    
    def get_order_book_state(self) -> Dict[str, Any]:
        """
        Get the current state of the order book.
        
        Returns:
            Dict: Order book state including price levels and volumes
        """
        return {
            'bids': {price: sum(o.quantity for o in orders) 
                    for price, orders in self.bids.items()},
            'asks': {price: sum(o.quantity for o in orders) 
                    for price, orders in self.asks.items()},
            'best_bid': self.get_best_bid(),
            'best_ask': self.get_best_ask(),
            'spread': self.get_spread(),
            'mid_price': self.get_mid_price()
        }
    
    def _update_history(self) -> None:
        """Update price and volume history."""
        mid_price = self.get_mid_price()
        if mid_price:
            self.price_history.append(mid_price)
            
        total_volume = sum(trade['quantity'] for trade in self.trades[-1:])
        self.volume_history.append(total_volume)
        
        spread = self.get_spread()
        if spread:
            self.spread_history.append(spread)
    
    def plot_order_book(self, depth: int = 10) -> plt.Figure:
        """
        Plot the current order book state.
        
        Args:
            depth (int): Number of price levels to show
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Get top bids and asks
        top_bids = sorted(self.bids.items(), reverse=True)[:depth]
        top_asks = sorted(self.asks.items())[:depth]
        
        # Plot bids
        bid_prices = [price for price, _ in top_bids]
        bid_volumes = [sum(o.quantity for o in orders) 
                      for _, orders in top_bids]
        ax1.barh(bid_prices, bid_volumes, color='green', alpha=0.6)
        
        # Plot asks
        ask_prices = [price for price, _ in top_asks]
        ask_volumes = [sum(o.quantity for o in orders) 
                      for _, orders in top_asks]
        ax1.barh(ask_prices, ask_volumes, color='red', alpha=0.6)
        
        ax1.set_xlabel('Volume')
        ax1.set_ylabel('Price')
        ax1.set_title('Order Book')
        ax1.grid(True, alpha=0.3)
        
        # Plot price history
        if self.price_history:
            ax2.plot(self.price_history, label='Mid Price')
            ax2.plot(self.spread_history, label='Spread', alpha=0.6)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Price')
            ax2.set_title('Price History')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def analyze_order_book(self) -> Dict[str, Any]:
        """
        Analyze order book dynamics and statistics.
        
        Returns:
            Dict: Analysis results
        """
        if not self.price_history:
            return {}
            
        price_history = np.array(self.price_history)
        volume_history = np.array(self.volume_history)
        spread_history = np.array(self.spread_history)
        
        return {
            'price_stats': {
                'mean': np.mean(price_history),
                'std': np.std(price_history),
                'min': np.min(price_history),
                'max': np.max(price_history)
            },
            'volume_stats': {
                'total': np.sum(volume_history),
                'mean': np.mean(volume_history),
                'std': np.std(volume_history)
            },
            'spread_stats': {
                'mean': np.mean(spread_history),
                'std': np.std(spread_history),
                'min': np.min(spread_history),
                'max': np.max(spread_history)
            },
            'order_book_imbalance': self._calculate_imbalance(),
            'price_impact': self._calculate_price_impact()
        }
    
    def _calculate_imbalance(self) -> float:
        """Calculate order book imbalance."""
        bid_volume = sum(sum(o.quantity for o in orders) 
                        for orders in self.bids.values())
        ask_volume = sum(sum(o.quantity for o in orders) 
                        for orders in self.asks.values())
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0
            
        return (bid_volume - ask_volume) / total_volume
    
    def _calculate_price_impact(self) -> Dict[str, float]:
        """Calculate price impact of hypothetical trades."""
        impacts = {}
        for size in [0.1, 0.5, 1.0]:  # Trade sizes as fraction of average volume
            # Calculate impact of buying
            buy_impact = self._simulate_trade_impact(size, 'buy')
            # Calculate impact of selling
            sell_impact = self._simulate_trade_impact(size, 'sell')
            
            impacts[f'buy_impact_{size}'] = buy_impact
            impacts[f'sell_impact_{size}'] = sell_impact
            
        return impacts
    
    def _simulate_trade_impact(self, size: float, side: str) -> float:
        """
        Simulate the price impact of a trade.
        
        Args:
            size (float): Trade size as fraction of average volume
            side (str): 'buy' or 'sell'
            
        Returns:
            float: Price impact in percentage
        """
        avg_volume = np.mean(self.volume_history) if self.volume_history else 1
        trade_size = size * avg_volume
        
        if side == 'buy':
            orders = sorted(self.asks.items())
        else:
            orders = sorted(self.bids.items(), reverse=True)
            
        remaining = trade_size
        weighted_price = 0
        
        for price, order_list in orders:
            volume_at_price = sum(o.quantity for o in order_list)
            if remaining <= 0:
                break
                
            volume_traded = min(remaining, volume_at_price)
            weighted_price += price * volume_traded
            remaining -= volume_traded
            
        if trade_size > 0:
            avg_execution_price = weighted_price / trade_size
            mid_price = self.get_mid_price() or 0
            return (avg_execution_price - mid_price) / mid_price * 100
            
        return 0 