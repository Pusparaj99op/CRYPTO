import numpy as np
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple, Any

class HedgeType(Enum):
    """Enum for different hedging strategies"""
    DELTA_HEDGE = 0          # Delta hedging
    CROSS_ASSET = 1          # Cross-asset hedging
    OPTIONS = 2              # Options-based hedging
    FUTURES = 3              # Futures-based hedging
    CORRELATION = 4          # Correlation-based hedging
    CONDITIONAL = 5          # Conditional hedging strategies

class HedgingStrategy:
    """
    Hedging strategies for crypto trading.
    
    Implements various hedging approaches to reduce directional risk,
    volatility exposure, and manage tail risk in crypto portfolios.
    """
    
    def __init__(
        self,
        portfolio: Dict[str, Dict[str, Any]],
        hedge_ratio: float = 1.0,
        hedge_type: HedgeType = HedgeType.FUTURES,
        rebalance_threshold: float = 0.1,
        correlation_lookback: int = 30,
        max_hedge_cost: float = 0.01,
        dynamic_hedge: bool = True,
        hedge_instruments: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize the hedging strategy.
        
        Args:
            portfolio: Current portfolio {symbol: {"position": position_size, "price": current_price}}
            hedge_ratio: Ratio of position to hedge (0.0-1.0)
            hedge_type: Type of hedging strategy to use
            rebalance_threshold: Threshold to trigger hedge rebalancing
            correlation_lookback: Periods for correlation calculation
            max_hedge_cost: Maximum cost as a fraction of portfolio value
            dynamic_hedge: Whether to dynamically adjust hedges based on market conditions
            hedge_instruments: Available hedge instruments {symbol: [hedge_instruments]}
        """
        self.portfolio = portfolio
        self.hedge_ratio = max(0.0, min(1.0, hedge_ratio))
        self.hedge_type = hedge_type
        self.rebalance_threshold = max(0.01, rebalance_threshold)
        self.correlation_lookback = max(10, correlation_lookback)
        self.max_hedge_cost = max(0.001, max_hedge_cost)
        self.dynamic_hedge = dynamic_hedge
        self.hedge_instruments = hedge_instruments or {}
        
        # Data for calculations
        self.price_history = {}  # {symbol: [prices]}
        self.correlation_matrix = {}  # {symbol1: {symbol2: correlation}}
        self.active_hedges = {}  # {position_id: hedge_details}
        self.hedge_performance = {}  # {position_id: performance_metrics}
        
    def update_portfolio(self, portfolio: Dict[str, Dict[str, Any]]):
        """
        Update the current portfolio.
        
        Args:
            portfolio: Updated portfolio information
        """
        self.portfolio = portfolio
        
    def update_price_history(self, symbol: str, price: float, timestamp: datetime):
        """
        Update price history for correlation calculations.
        
        Args:
            symbol: Trading pair symbol
            price: Current price
            timestamp: Timestamp for the price
        """
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            
        self.price_history[symbol].append((timestamp, price))
        
        # Keep history within lookback window
        if len(self.price_history[symbol]) > self.correlation_lookback:
            self.price_history[symbol] = self.price_history[symbol][-self.correlation_lookback:]
            
    def _update_correlations(self):
        """Update correlation matrix between assets"""
        symbols = list(self.price_history.keys())
        
        for i, symbol1 in enumerate(symbols):
            if symbol1 not in self.correlation_matrix:
                self.correlation_matrix[symbol1] = {}
                
            # Get prices for symbol1
            prices1 = [p[1] for p in sorted(self.price_history[symbol1], key=lambda x: x[0])]
            
            for j, symbol2 in enumerate(symbols[i:], i):
                if j == i:  # Self correlation is always 1.0
                    self.correlation_matrix[symbol1][symbol2] = 1.0
                    continue
                    
                # Get prices for symbol2
                prices2 = [p[1] for p in sorted(self.price_history[symbol2], key=lambda x: x[0])]
                
                # Ensure we have matching data points
                min_length = min(len(prices1), len(prices2))
                if min_length < 2:
                    # Not enough data for correlation
                    self.correlation_matrix[symbol1][symbol2] = 0.0
                    continue
                    
                # Calculate returns
                returns1 = np.diff(np.log(prices1[:min_length]))
                returns2 = np.diff(np.log(prices2[:min_length]))
                
                # Calculate correlation
                if len(returns1) > 1 and len(returns2) > 1:
                    correlation = np.corrcoef(returns1, returns2)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0.0
                else:
                    correlation = 0.0
                    
                self.correlation_matrix[symbol1][symbol2] = correlation
                
                # Also store the reverse lookup
                if symbol2 not in self.correlation_matrix:
                    self.correlation_matrix[symbol2] = {}
                self.correlation_matrix[symbol2][symbol1] = correlation
                
    def _find_best_hedge_instrument(self, symbol: str) -> Tuple[str, float]:
        """
        Find the best hedging instrument for a symbol based on correlation.
        
        Args:
            symbol: Symbol to find hedge for
            
        Returns:
            Tuple of (hedge_symbol, correlation)
        """
        # Update correlations
        self._update_correlations()
        
        best_hedge = None
        best_correlation = 0.0
        
        # First check specific hedge instruments if defined
        if symbol in self.hedge_instruments and self.hedge_instruments[symbol]:
            available_hedges = self.hedge_instruments[symbol]
            
            for hedge_symbol in available_hedges:
                if hedge_symbol in self.correlation_matrix.get(symbol, {}):
                    correlation = abs(self.correlation_matrix[symbol].get(hedge_symbol, 0.0))
                    
                    if correlation > best_correlation:
                        best_correlation = correlation
                        best_hedge = hedge_symbol
        
        # If no specific hedge or none found, look for the most correlated asset
        if best_hedge is None:
            for hedge_symbol, correlation in self.correlation_matrix.get(symbol, {}).items():
                if hedge_symbol != symbol:  # Don't hedge with the same asset
                    correlation = abs(correlation)
                    
                    if correlation > best_correlation:
                        best_correlation = correlation
                        best_hedge = hedge_symbol
        
        # Default to a Bitcoin hedge if nothing found
        if best_hedge is None:
            best_hedge = "BTCUSDT"
            best_correlation = 0.5  # Assumed default correlation
            
        return best_hedge, best_correlation
    
    def calculate_hedge_position(self, symbol: str, position_size: float, position_price: float) -> Dict:
        """
        Calculate the appropriate hedge position.
        
        Args:
            symbol: Symbol to hedge
            position_size: Size of the position to hedge
            position_price: Current price of the position
            
        Returns:
            Dictionary with hedge details
        """
        # Find best hedge instrument
        hedge_symbol, correlation = self._find_best_hedge_instrument(symbol)
        
        # Adjust hedge ratio based on correlation
        effective_hedge_ratio = self.hedge_ratio
        if self.dynamic_hedge:
            # Adjust hedge ratio based on correlation
            # Higher correlation = more effective hedge = can use lower ratio
            if correlation > 0.8:
                effective_hedge_ratio *= 0.8
            elif correlation < 0.4:
                effective_hedge_ratio *= 1.2
                
        # Position value
        position_value = position_size * position_price
        
        # Get current price of hedge instrument
        hedge_prices = self.price_history.get(hedge_symbol, [])
        if not hedge_prices:
            return {
                "can_hedge": False,
                "reason": "No price data for hedge instrument"
            }
            
        hedge_price = hedge_prices[-1][1]
        
        # Calculate hedge size based on strategy type
        if self.hedge_type == HedgeType.FUTURES:
            # For futures, we want to create a position of opposite sign
            # with the appropriate value
            hedge_size = -(position_value * effective_hedge_ratio) / hedge_price
            hedge_type = "futures"
            
        elif self.hedge_type == HedgeType.DELTA_HEDGE:
            # Delta hedging - adjust for correlation
            hedge_size = -(position_value * effective_hedge_ratio * correlation) / hedge_price
            hedge_type = "delta"
            
        elif self.hedge_type == HedgeType.CORRELATION:
            # Correlation-based hedge sizing
            # For inversely correlated assets, we need less hedge
            hedge_size = -(position_value * effective_hedge_ratio) / (hedge_price * max(0.2, abs(correlation)))
            hedge_type = "correlation"
            
        elif self.hedge_type == HedgeType.OPTIONS:
            # Options-based hedging (simplified)
            # For options, we're buying protection, so size is based on notional value
            option_premium = hedge_price * 0.05  # Simplified option premium estimate
            contracts = position_value * effective_hedge_ratio / (hedge_price * 100)  # Standard 100 multiplier
            hedge_size = contracts
            hedge_type = "options"
            hedge_price = option_premium
            
        else:
            # Default to futures-style hedging
            hedge_size = -(position_value * effective_hedge_ratio) / hedge_price
            hedge_type = "default"
            
        # Estimate hedge cost
        if hedge_type == "options":
            # For options, cost is the premium
            hedge_cost = abs(hedge_size * hedge_price)
        else:
            # For futures, estimate cost based on spreads and fees
            hedge_cost = abs(hedge_size * hedge_price * 0.001)  # Simplified cost estimate
            
        # Check if hedge cost is acceptable
        hedge_cost_percentage = hedge_cost / position_value
        if hedge_cost_percentage > self.max_hedge_cost:
            return {
                "can_hedge": False,
                "reason": f"Hedge cost ({hedge_cost_percentage:.2%}) exceeds maximum ({self.max_hedge_cost:.2%})"
            }
            
        # Prepare hedge details
        hedge_details = {
            "can_hedge": True,
            "original_symbol": symbol,
            "original_position": position_size,
            "original_price": position_price,
            "original_value": position_value,
            "hedge_symbol": hedge_symbol,
            "hedge_size": hedge_size,
            "hedge_price": hedge_price,
            "hedge_value": abs(hedge_size * hedge_price),
            "hedge_ratio": effective_hedge_ratio,
            "correlation": correlation,
            "hedge_type": hedge_type,
            "hedge_cost": hedge_cost,
            "hedge_cost_percentage": hedge_cost_percentage
        }
        
        return hedge_details
        
    def create_hedge(self, symbol: str, position_id: str, **kwargs) -> Dict:
        """
        Create a hedge for a specific position.
        
        Args:
            symbol: Symbol to hedge
            position_id: Unique identifier for the position
            **kwargs: Additional parameters
            
        Returns:
            Hedge details
        """
        # Get position details from portfolio
        if symbol not in self.portfolio:
            return {
                "success": False,
                "message": f"Symbol {symbol} not found in portfolio"
            }
            
        position_details = self.portfolio[symbol]
        position_size = position_details.get("position", 0.0)
        position_price = position_details.get("price", 0.0)
        
        # Check if we already have a hedge for this position
        if position_id in self.active_hedges:
            return {
                "success": False,
                "message": f"Hedge already exists for position {position_id}"
            }
            
        # Calculate hedge position
        hedge_details = self.calculate_hedge_position(symbol, position_size, position_price)
        
        if not hedge_details.get("can_hedge", False):
            return {
                "success": False,
                "message": f"Cannot create hedge: {hedge_details.get('reason', 'Unknown reason')}"
            }
            
        # Add timestamp and ID
        hedge_details["timestamp"] = datetime.now()
        hedge_details["position_id"] = position_id
        
        # Store the active hedge
        self.active_hedges[position_id] = hedge_details
        
        return {
            "success": True,
            "hedge": hedge_details
        }
        
    def update_hedge(self, position_id: str, position_size: float = None, position_price: float = None) -> Dict:
        """
        Update an existing hedge position.
        
        Args:
            position_id: Identifier for the position
            position_size: New position size (optional)
            position_price: New position price (optional)
            
        Returns:
            Updated hedge details
        """
        # Check if hedge exists
        if position_id not in self.active_hedges:
            return {
                "success": False,
                "message": f"No active hedge found for position {position_id}"
            }
            
        # Get current hedge details
        hedge = self.active_hedges[position_id]
        
        # Update position details if provided
        if position_size is not None:
            hedge["original_position"] = position_size
            
        if position_price is not None:
            hedge["original_price"] = position_price
            
        # Recalculate position value
        position_value = hedge["original_position"] * hedge["original_price"]
        hedge["original_value"] = position_value
        
        # Get current price of hedge instrument
        symbol = hedge["original_symbol"]
        hedge_symbol = hedge["hedge_symbol"]
        
        # Find best hedge instrument (might have changed)
        new_hedge_symbol, correlation = self._find_best_hedge_instrument(symbol)
        
        # Check if hedge instrument changed
        if new_hedge_symbol != hedge_symbol:
            # Close old hedge and create new one
            self.close_hedge(position_id)
            return self.create_hedge(symbol, position_id)
            
        # Update correlation
        hedge["correlation"] = correlation
        
        # Get current hedge price
        hedge_prices = self.price_history.get(hedge_symbol, [])
        if not hedge_prices:
            return {
                "success": False,
                "message": "No price data for hedge instrument"
            }
            
        hedge_price = hedge_prices[-1][1]
        hedge["hedge_price"] = hedge_price
        
        # Check if rebalance is needed
        current_hedge_value = abs(hedge["hedge_size"] * hedge_price)
        target_hedge_value = abs(position_value * hedge["hedge_ratio"])
        hedge_deviation = abs(current_hedge_value - target_hedge_value) / target_hedge_value
        
        if hedge_deviation > self.rebalance_threshold:
            # Recalculate hedge size
            if hedge["hedge_type"] == "futures" or hedge["hedge_type"] == "default":
                hedge["hedge_size"] = -(position_value * hedge["hedge_ratio"]) / hedge_price
            elif hedge["hedge_type"] == "delta":
                hedge["hedge_size"] = -(position_value * hedge["hedge_ratio"] * correlation) / hedge_price
            elif hedge["hedge_type"] == "correlation":
                hedge["hedge_size"] = -(position_value * hedge["hedge_ratio"]) / (hedge_price * max(0.2, abs(correlation)))
            elif hedge["hedge_type"] == "options":
                option_premium = hedge_price * 0.05
                contracts = position_value * hedge["hedge_ratio"] / (hedge_price * 100)
                hedge["hedge_size"] = contracts
                hedge["hedge_price"] = option_premium
                
            # Update hedge value
            hedge["hedge_value"] = abs(hedge["hedge_size"] * hedge_price)
            
            # Update timestamp
            hedge["timestamp"] = datetime.now()
            hedge["rebalanced"] = True
            
        return {
            "success": True,
            "hedge": hedge,
            "rebalanced": hedge_deviation > self.rebalance_threshold
        }
        
    def close_hedge(self, position_id: str) -> Dict:
        """
        Close an active hedge.
        
        Args:
            position_id: Identifier for the position
            
        Returns:
            Closure details
        """
        # Check if hedge exists
        if position_id not in self.active_hedges:
            return {
                "success": False,
                "message": f"No active hedge found for position {position_id}"
            }
            
        # Get hedge details
        hedge = self.active_hedges[position_id]
        
        # Get current price of hedge instrument
        hedge_symbol = hedge["hedge_symbol"]
        hedge_prices = self.price_history.get(hedge_symbol, [])
        
        if hedge_prices:
            current_price = hedge_prices[-1][1]
            entry_price = hedge["hedge_price"]
            
            # Calculate P&L
            if hedge["hedge_type"] == "options":
                # For options, P&L is limited to the premium paid
                pnl = -hedge["hedge_cost"]  # Simplified - assume option expires worthless
            else:
                # For futures, calculate P&L based on price difference
                pnl = hedge["hedge_size"] * (current_price - entry_price)
                
            # Store hedge performance
            self.hedge_performance[position_id] = {
                "entry_time": hedge["timestamp"],
                "exit_time": datetime.now(),
                "duration": (datetime.now() - hedge["timestamp"]).total_seconds() / 3600,  # hours
                "original_symbol": hedge["original_symbol"],
                "hedge_symbol": hedge["hedge_symbol"],
                "hedge_type": hedge["hedge_type"],
                "entry_price": entry_price,
                "exit_price": current_price,
                "size": hedge["hedge_size"],
                "pnl": pnl,
                "original_value": hedge["original_value"],
                "hedge_cost": hedge["hedge_cost"],
                "cost_percentage": hedge["hedge_cost_percentage"],
                "correlation": hedge["correlation"]
            }
        
        # Remove from active hedges
        del self.active_hedges[position_id]
        
        return {
            "success": True,
            "message": f"Hedge for position {position_id} closed successfully"
        }
        
    def get_hedge_summary(self) -> Dict:
        """
        Get summary of all active hedges.
        
        Returns:
            Summary of active hedges
        """
        if not self.active_hedges:
            return {
                "active_hedges": 0,
                "total_hedge_value": 0.0,
                "total_hedge_cost": 0.0,
                "hedges": []
            }
            
        # Calculate summary statistics
        total_hedge_value = sum(hedge["hedge_value"] for hedge in self.active_hedges.values())
        total_hedge_cost = sum(hedge["hedge_cost"] for hedge in self.active_hedges.values())
        
        # Calculate average correlation
        avg_correlation = sum(hedge["correlation"] for hedge in self.active_hedges.values()) / len(self.active_hedges)
        
        # Calculate current P&L of hedges
        current_pnl = 0.0
        for hedge in self.active_hedges.values():
            hedge_symbol = hedge["hedge_symbol"]
            hedge_prices = self.price_history.get(hedge_symbol, [])
            
            if hedge_prices:
                current_price = hedge_prices[-1][1]
                entry_price = hedge["hedge_price"]
                
                if hedge["hedge_type"] != "options":
                    # For non-options, calculate P&L based on price difference
                    position_pnl = hedge["hedge_size"] * (current_price - entry_price)
                    current_pnl += position_pnl
        
        return {
            "active_hedges": len(self.active_hedges),
            "total_hedge_value": total_hedge_value,
            "total_hedge_cost": total_hedge_cost,
            "average_correlation": avg_correlation,
            "current_pnl": current_pnl,
            "hedges": list(self.active_hedges.values())
        }
        
    def get_hedge_performance(self) -> Dict:
        """
        Get performance metrics of closed hedges.
        
        Returns:
            Performance summary of closed hedges
        """
        if not self.hedge_performance:
            return {
                "closed_hedges": 0,
                "total_pnl": 0.0,
                "average_duration": 0.0,
                "total_cost": 0.0,
                "performance": []
            }
            
        # Calculate summary statistics
        total_pnl = sum(perf["pnl"] for perf in self.hedge_performance.values())
        total_cost = sum(perf["hedge_cost"] for perf in self.hedge_performance.values())
        avg_duration = sum(perf["duration"] for perf in self.hedge_performance.values()) / len(self.hedge_performance)
        
        return {
            "closed_hedges": len(self.hedge_performance),
            "total_pnl": total_pnl,
            "average_duration": avg_duration,
            "total_cost": total_cost,
            "performance": list(self.hedge_performance.values())
        } 