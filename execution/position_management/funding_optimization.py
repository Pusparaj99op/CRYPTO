import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from common.types import Exchange, Market, Timeframe, Side
from config.config_manager import ConfigManager
from data.market_data import MarketData
from execution.order_management.order_manager import OrderManager


class FundingStrategy(Enum):
    """Defines different strategies for optimizing funding rates."""
    PASSIVE = "passive"  # Monitor funding rates but don't actively manage positions
    ACTIVE = "active"  # Actively swap between venues based on funding differentials
    AGGRESSIVE = "aggressive"  # Actively trade funding rates as a primary strategy
    HEDGED = "hedged"  # Maintain market-neutral positions across venues to capture funding


@dataclass
class FundingThresholds:
    """Thresholds for funding rate actions."""
    flip_threshold: float  # Absolute funding rate to trigger a position flip
    venue_switch_threshold: float  # Differential to trigger venue switch
    min_expected_profit: float  # Minimum expected profit to take action
    min_funding_advantage: float  # Minimum funding advantage for venue switch


class FundingOptimizer:
    """
    Optimizes perpetual futures positions based on funding rates.
    
    Funding rates on perpetual futures can be a significant cost or source of profit.
    This module:
    1. Monitors funding rates across exchanges
    2. Calculates optimal position placement based on funding differentials
    3. Initiates position transfers between venues when advantageous
    4. Implements funding rate arbitrage when conditions are favorable
    """
    
    def __init__(
        self, 
        config_manager: ConfigManager, 
        market_data: MarketData,
        order_manager: OrderManager
    ):
        """
        Initialize the FundingOptimizer.
        
        Args:
            config_manager: Configuration manager
            market_data: Market data provider for funding rates
            order_manager: For executing transfers and trades
        """
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        self.market_data = market_data
        self.order_manager = order_manager
        
        # Load configuration
        self.config = self._load_config()
        
        # Funding rate history: {(exchange, market): [{timestamp, rate},...]}
        self.funding_history: Dict[Tuple[Exchange, Market], List[Dict]] = {}
        
        # Funding predictions: {(exchange, market): predicted_rate}
        self.funding_predictions: Dict[Tuple[Exchange, Market], float] = {}
        
        # Next funding timestamps: {(exchange, market): next_funding_time}
        self.next_funding: Dict[Tuple[Exchange, Market], int] = {}
        
    def _load_config(self) -> Dict:
        """Load funding optimization configuration."""
        config = self.config_manager.get_config("funding_optimization")
        
        self.strategy = FundingStrategy(config.get("strategy", "passive"))
        
        self.thresholds = FundingThresholds(
            flip_threshold=float(config.get("flip_threshold", 0.0075)),  # 0.75% per 8h
            venue_switch_threshold=float(config.get("venue_switch_threshold", 0.0025)),  # 0.25% differential
            min_expected_profit=float(config.get("min_expected_profit", 0.001)),  # 0.1% minimum expected profit
            min_funding_advantage=float(config.get("min_funding_advantage", 0.002))  # 0.2% minimum advantage
        )
        
        self.max_position_split = int(config.get("max_position_split", 3))
        self.funding_history_days = int(config.get("funding_history_days", 7))
        
        return config
    
    def update_funding_rates(self) -> None:
        """Update current funding rates and history for all monitored markets."""
        markets_config = self.config_manager.get_config("markets")
        
        for exchange_str, markets in markets_config.get("monitored_markets", {}).items():
            exchange = Exchange(exchange_str)
            
            for market_str in markets:
                market = Market(market_str)
                
                try:
                    # Get current funding rate
                    funding_data = self.market_data.get_funding_rate(exchange, market)
                    
                    if funding_data:
                        # Update funding history
                        if (exchange, market) not in self.funding_history:
                            self.funding_history[(exchange, market)] = []
                            
                        self.funding_history[(exchange, market)].append({
                            "timestamp": int(time.time()),
                            "rate": funding_data["rate"],
                            "next_funding_time": funding_data.get("next_funding_time")
                        })
                        
                        # Keep history limited to configured days
                        cutoff_time = int(time.time()) - (self.funding_history_days * 86400)
                        self.funding_history[(exchange, market)] = [
                            entry for entry in self.funding_history[(exchange, market)]
                            if entry["timestamp"] > cutoff_time
                        ]
                        
                        # Update next funding time
                        if funding_data.get("next_funding_time"):
                            self.next_funding[(exchange, market)] = funding_data["next_funding_time"]
                        
                        self.logger.info(
                            f"Updated funding rate for {exchange.value}.{market.value}: "
                            f"{funding_data['rate'] * 100:.4f}%"
                        )
                except Exception as e:
                    self.logger.error(f"Error updating funding rate for {exchange.value}.{market.value}: {e}")
    
    def predict_funding_rates(self) -> Dict[Tuple[Exchange, Market], float]:
        """
        Predict future funding rates based on historical patterns.
        
        Returns:
            Dictionary of predicted funding rates by (exchange, market)
        """
        predictions = {}
        
        for (exchange, market), history in self.funding_history.items():
            if len(history) < 6:  # Need enough history for meaningful prediction
                continue
                
            # Convert to dataframe for analysis
            df = pd.DataFrame(history)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            df.set_index("datetime", inplace=True)
            
            # Simple prediction based on recent average and trend
            recent_avg = df["rate"].iloc[-6:].mean()
            
            # Calculate trend direction
            if len(df) >= 12:
                trend = df["rate"].iloc[-6:].mean() - df["rate"].iloc[-12:-6].mean()
            else:
                trend = 0
            
            # Apply trend factor to recent average
            prediction = recent_avg + (trend * 0.5)
            
            predictions[(exchange, market)] = prediction
            
        self.funding_predictions = predictions
        return predictions
    
    def get_funding_advantage(
        self,
        market: Market,
        current_exchange: Exchange,
        alternative_exchange: Exchange
    ) -> float:
        """
        Calculate the funding rate advantage of one exchange over another.
        
        Args:
            market: The market to compare
            current_exchange: Current exchange
            alternative_exchange: Alternative exchange to compare against
            
        Returns:
            Funding rate advantage (positive means alternative is better)
        """
        # Get latest funding rates
        current_rate = self._get_latest_funding_rate(current_exchange, market)
        alt_rate = self._get_latest_funding_rate(alternative_exchange, market)
        
        if current_rate is None or alt_rate is None:
            return 0.0
            
        # For long positions:
        # - Negative funding rates are beneficial (we receive payment)
        # - Positive funding rates are costly (we pay funding)
        # So a lower (more negative) rate is better for longs
        
        # Return the advantage of switching (positive means switch is beneficial)
        return current_rate - alt_rate
    
    def _get_latest_funding_rate(self, exchange: Exchange, market: Market) -> Optional[float]:
        """Get the most recent funding rate for an exchange-market pair."""
        history = self.funding_history.get((exchange, market), [])
        
        if not history:
            return None
            
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)[0]["rate"]
    
    def get_funding_schedule(self, exchange: Exchange, market: Market) -> Dict:
        """
        Get the funding schedule information for a market.
        
        Args:
            exchange: The exchange
            market: The market
            
        Returns:
            Dictionary with next funding time and interval
        """
        # Get exchange-specific funding information
        next_time = self.next_funding.get((exchange, market))
        
        # Get interval (usually exchange-specific)
        # Common intervals: 8h for most exchanges, 1h for some
        exchange_intervals = {
            Exchange.BINANCE: 8,  # 8 hours
            Exchange.BYBIT: 8,    # 8 hours
            Exchange.OKEX: 8,     # 8 hours
            Exchange.DERIBIT: 8,  # 8 hours
            Exchange.FTX: 1,      # 1 hour
        }
        
        interval_hours = exchange_intervals.get(exchange, 8)
        
        return {
            "next_funding_time": next_time,
            "interval_hours": interval_hours,
            "seconds_until_next": next_time - int(time.time()) if next_time else None
        }
    
    def should_flip_position(self, exchange: Exchange, market: Market, current_side: Side) -> bool:
        """
        Determine if a position should be flipped due to funding rates.
        
        Args:
            exchange: The exchange
            market: The market
            current_side: Current position side
            
        Returns:
            True if position should be flipped, False otherwise
        """
        if self.strategy == FundingStrategy.PASSIVE:
            return False
            
        # Get latest funding rate
        latest_rate = self._get_latest_funding_rate(exchange, market)
        
        if latest_rate is None:
            return False
            
        # Absolute value of the rate
        abs_rate = abs(latest_rate)
        
        # High enough rate to consider action
        if abs_rate < self.thresholds.flip_threshold:
            return False
            
        # For long positions, positive rates are unfavorable
        # For short positions, negative rates are unfavorable
        if (current_side == Side.BUY and latest_rate > 0) or (current_side == Side.SELL and latest_rate < 0):
            # Condition for flipping: rate is unfavorable and above threshold
            return True
            
        return False
    
    def get_optimal_funding_venue(self, market: Market, side: Side) -> Exchange:
        """
        Find the exchange with the most favorable funding rate for a position.
        
        Args:
            market: The market to trade
            side: The position side
            
        Returns:
            Exchange with best funding rate
        """
        best_exchange = None
        best_rate = float('inf') if side == Side.BUY else float('-inf')
        
        for (exchange, mkt), history in self.funding_history.items():
            if mkt != market or not history:
                continue
                
            latest_rate = sorted(history, key=lambda x: x["timestamp"], reverse=True)[0]["rate"]
            
            # For long positions, we want the lowest (most negative) rate
            # For short positions, we want the highest (most positive) rate
            if side == Side.BUY:
                if latest_rate < best_rate:
                    best_rate = latest_rate
                    best_exchange = exchange
            else:  # Side.SELL
                if latest_rate > best_rate:
                    best_rate = latest_rate
                    best_exchange = exchange
        
        return best_exchange if best_exchange else list(Exchange)[0]  # Return first exchange as fallback
    
    def should_transfer_position(
        self,
        market: Market,
        current_exchange: Exchange,
        position_size: float,
        side: Side
    ) -> Tuple[bool, Optional[Exchange]]:
        """
        Determine if a position should be transferred to another exchange
        for better funding rates.
        
        Args:
            market: The market
            current_exchange: Current exchange
            position_size: Size of the position
            side: Position side
            
        Returns:
            (should_transfer, target_exchange)
        """
        if self.strategy == FundingStrategy.PASSIVE:
            return False, None
            
        # Get optimal exchange for this market and side
        optimal_exchange = self.get_optimal_funding_venue(market, side)
        
        if optimal_exchange == current_exchange:
            return False, None
            
        # Calculate funding advantage
        advantage = self.get_funding_advantage(market, current_exchange, optimal_exchange)
        
        # Adjust advantage sign based on position side
        if side == Side.SELL:
            advantage = -advantage
            
        # Check if advantage exceeds threshold
        if advantage < self.thresholds.min_funding_advantage:
            return False, None
            
        # Calculate expected profit from switch
        # Assuming one funding interval and accounting for transfer costs
        transfer_cost = self._estimate_transfer_cost(market, position_size, current_exchange, optimal_exchange)
        funding_savings = position_size * advantage
        
        expected_profit = funding_savings - transfer_cost
        
        if expected_profit < self.thresholds.min_expected_profit * position_size:
            return False, None
            
        return True, optimal_exchange
    
    def _estimate_transfer_cost(
        self,
        market: Market,
        position_size: float,
        from_exchange: Exchange,
        to_exchange: Exchange
    ) -> float:
        """
        Estimate the cost of transferring a position between exchanges.
        
        Args:
            market: The market
            position_size: Size of the position
            from_exchange: Source exchange
            to_exchange: Destination exchange
            
        Returns:
            Estimated cost in quote currency
        """
        # Get fee rates
        from_fee = self._get_taker_fee(from_exchange)
        to_fee = self._get_taker_fee(to_exchange)
        
        # Get current price
        try:
            price = self.market_data.get_last_price(from_exchange, market)
        except:
            try:
                price = self.market_data.get_last_price(to_exchange, market)
            except:
                # Fallback to a safe estimate
                price = 1.0
        
        # Calculate total transfer cost (close position + open new position)
        transfer_cost = position_size * price * (from_fee + to_fee)
        
        # Add estimated slippage (0.05% as a conservative estimate)
        slippage_cost = position_size * price * 0.0005
        
        return transfer_cost + slippage_cost
    
    def _get_taker_fee(self, exchange: Exchange) -> float:
        """Get the taker fee rate for an exchange."""
        # These would be retrieved from the actual fee structure
        # This is a simplified implementation with typical values
        default_fees = {
            Exchange.BINANCE: 0.0004,  # 0.04% 
            Exchange.BYBIT: 0.0006,    # 0.06%
            Exchange.OKEX: 0.0005,     # 0.05%
            Exchange.DERIBIT: 0.0005,  # 0.05%
            Exchange.FTX: 0.0007,      # 0.07%
            # Add more exchanges as needed
        }
        
        return default_fees.get(exchange, 0.001)  # Default to 0.1% if unknown
    
    def execute_funding_arbitrage(self) -> Dict:
        """
        Execute a funding rate arbitrage strategy when conditions are favorable.
        This involves opening opposing positions on different exchanges with significant
        funding rate differentials.
        
        Returns:
            Results of arbitrage attempts
        """
        if self.strategy not in [FundingStrategy.AGGRESSIVE, FundingStrategy.HEDGED]:
            return {"status": "skipped", "reason": "strategy_not_enabled"}
            
        results = {"attempts": [], "successful": 0, "failed": 0}
        
        # Get all markets with funding data on multiple exchanges
        markets_with_data = {}
        for (exchange, market) in self.funding_history.keys():
            if market not in markets_with_data:
                markets_with_data[market] = []
            markets_with_data[market].append(exchange)
            
        # Filter to markets available on at least 2 exchanges
        arbitrage_candidates = {
            market: exchanges for market, exchanges in markets_with_data.items()
            if len(exchanges) >= 2
        }
        
        for market, exchanges in arbitrage_candidates.items():
            # Find the exchange pair with the largest funding differential
            best_long = None
            best_short = None
            max_diff = 0
            
            for i, ex1 in enumerate(exchanges):
                for ex2 in exchanges[i+1:]:
                    rate1 = self._get_latest_funding_rate(ex1, market)
                    rate2 = self._get_latest_funding_rate(ex2, market)
                    
                    if rate1 is None or rate2 is None:
                        continue
                        
                    diff = abs(rate1 - rate2)
                    
                    if diff > max_diff:
                        max_diff = diff
                        # Assign long and short positions based on rates
                        if rate1 < rate2:  # Lower rate is better for longs
                            best_long = ex1
                            best_short = ex2
                        else:
                            best_long = ex2
                            best_short = ex1
            
            # Check if differential exceeds threshold
            if max_diff < self.thresholds.venue_switch_threshold * 2:  # Double the threshold for arbitrage
                continue
                
            # Calculate position size based on funding advantage and risk settings
            # This would be more sophisticated in a real implementation
            position_size = self._calculate_arbitrage_position_size(market, max_diff)
            
            if position_size <= 0:
                continue
                
            # Attempt to execute the arbitrage
            try:
                # Open long position on exchange with better funding for longs
                long_order = self.order_manager.create_order(
                    exchange=best_long,
                    market=market,
                    side=Side.BUY,
                    size=position_size,
                    order_type="MARKET",
                    reduce_only=False
                )
                
                # Open short position on exchange with better funding for shorts
                short_order = self.order_manager.create_order(
                    exchange=best_short,
                    market=market,
                    side=Side.SELL,
                    size=position_size,
                    order_type="MARKET",
                    reduce_only=False
                )
                
                if long_order["status"] == "FILLED" and short_order["status"] == "FILLED":
                    self.logger.info(
                        f"Executed funding arbitrage for {market.value}: "
                        f"Long on {best_long.value}, Short on {best_short.value}, "
                        f"Size: {position_size}, Funding diff: {max_diff*100:.4f}%"
                    )
                    
                    results["attempts"].append({
                        "market": market.value,
                        "long_exchange": best_long.value,
                        "short_exchange": best_short.value,
                        "size": position_size,
                        "funding_differential": max_diff,
                        "status": "success"
                    })
                    
                    results["successful"] += 1
                else:
                    # Handle partial fills or failures
                    self.logger.warning(
                        f"Partial funding arbitrage execution for {market.value}: "
                        f"Long order status: {long_order.get('status')}, "
                        f"Short order status: {short_order.get('status')}"
                    )
                    
                    results["attempts"].append({
                        "market": market.value,
                        "status": "partial",
                        "long_status": long_order.get("status"),
                        "short_status": short_order.get("status")
                    })
                    
                    results["failed"] += 1
                    
            except Exception as e:
                self.logger.error(f"Error executing funding arbitrage for {market.value}: {e}")
                
                results["attempts"].append({
                    "market": market.value,
                    "status": "failed",
                    "error": str(e)
                })
                
                results["failed"] += 1
                
        return results
    
    def _calculate_arbitrage_position_size(self, market: Market, funding_diff: float) -> float:
        """
        Calculate appropriate position size for funding arbitrage.
        
        Args:
            market: The market
            funding_diff: Funding rate differential
            
        Returns:
            Position size in base currency
        """
        # This would be expanded in a real implementation
        # to consider account balance, risk limits, etc.
        
        # Simple calculation based on funding differential
        # Higher differential = larger position size
        
        # Get risk settings
        risk_config = self.config_manager.get_config("risk")
        max_funding_arb_allocation = float(risk_config.get("max_funding_arbitrage_allocation", 0.05))
        
        # Get account balance
        account_value = 10000.0  # This would come from account_manager in real implementation
        
        # Scale position based on funding differential
        # 0.01 (1%) differential is considered very large
        scale_factor = min(funding_diff / 0.01, 1.0)
        
        # Calculate position size
        max_position = account_value * max_funding_arb_allocation
        position_size = max_position * scale_factor
        
        # Ensure minimum size
        min_position_size = 0.001  # This would be market-specific in real implementation
        if position_size < min_position_size:
            return 0
            
        return position_size
        
    def get_funding_stats(self, days: int = 7) -> Dict:
        """
        Get summary statistics of funding rates.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary of funding statistics
        """
        stats = {}
        cutoff_time = int(time.time()) - (days * 86400)
        
        for (exchange, market), history in self.funding_history.items():
            # Filter to requested time period
            filtered_history = [entry for entry in history if entry["timestamp"] > cutoff_time]
            
            if not filtered_history:
                continue
                
            rates = [entry["rate"] for entry in filtered_history]
            
            stats[(exchange.value, market.value)] = {
                "mean": np.mean(rates),
                "median": np.median(rates),
                "min": min(rates),
                "max": max(rates),
                "std": np.std(rates),
                "samples": len(rates),
                "annualized_cost_long": np.mean(rates) * (365 * 24 / 8),  # Assuming 8h funding
                "annualized_cost_short": -np.mean(rates) * (365 * 24 / 8)
            }
            
        return stats 