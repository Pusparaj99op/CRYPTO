"""
Pre-trade analysis module for assessing trading opportunities.
Evaluates market conditions, opportunity costs, and risk before trade execution.
"""
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class PreTradeAnalyzer:
    """Analyzer for pre-trade opportunity assessment."""
    
    def __init__(self, 
                 market_impact_threshold: float = 0.05,
                 liquidity_threshold: float = 0.10,
                 volatility_threshold: float = 1.5):
        """
        Initialize the pre-trade analyzer.
        
        Args:
            market_impact_threshold: Maximum acceptable market impact as percentage
            liquidity_threshold: Minimum required liquidity ratio
            volatility_threshold: Maximum acceptable volatility multiplier
        """
        self.market_impact_threshold = market_impact_threshold
        self.liquidity_threshold = liquidity_threshold
        self.volatility_threshold = volatility_threshold
    
    def analyze_opportunity(self, 
                           symbol: str,
                           trade_size: float,
                           side: str,
                           market_data: Dict[str, Any],
                           order_book: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a trading opportunity pre-execution.
        
        Args:
            symbol: Trading symbol
            trade_size: Size of intended trade
            side: Trade side ('buy' or 'sell')
            market_data: Market data dictionary
            order_book: Order book data
            
        Returns:
            Assessment results with execution recommendations
        """
        logger.info(f"Analyzing pre-trade opportunity for {symbol}, size: {trade_size}, side: {side}")
        
        results = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'trade_size': trade_size,
            'side': side,
            'recommendation': 'unknown',
            'confidence': 0.0,
            'metrics': {},
            'risks': {},
            'execution_params': {}
        }
        
        # Analyze market impact
        impact_estimate = self._estimate_market_impact(symbol, trade_size, side, order_book)
        results['metrics']['expected_market_impact'] = impact_estimate
        
        # Analyze liquidity
        liquidity_ratio = self._analyze_liquidity(symbol, trade_size, market_data, order_book)
        results['metrics']['liquidity_ratio'] = liquidity_ratio
        
        # Analyze volatility
        volatility = self._analyze_volatility(symbol, market_data)
        results['metrics']['current_volatility'] = volatility
        
        # Analyze spread costs
        spread_cost = self._analyze_spread_cost(symbol, trade_size, order_book)
        results['metrics']['spread_cost'] = spread_cost
        
        # Analyze timing
        timing_score = self._analyze_timing(symbol, side, market_data)
        results['metrics']['timing_score'] = timing_score
        
        # Calculate risk metrics
        results['risks'] = self._calculate_risk_metrics(symbol, trade_size, side, market_data)
        
        # Generate execution parameters
        results['execution_params'] = self._recommend_execution_params(
            symbol, trade_size, side, impact_estimate, liquidity_ratio, volatility)
        
        # Make recommendation
        recommendation, confidence = self._make_recommendation(
            impact_estimate, liquidity_ratio, volatility, spread_cost, timing_score, results['risks'])
        
        results['recommendation'] = recommendation
        results['confidence'] = confidence
        
        logger.info(f"Pre-trade analysis for {symbol}: {recommendation} (confidence: {confidence:.2f})")
        return results
    
    def _estimate_market_impact(self, 
                               symbol: str, 
                               trade_size: float, 
                               side: str, 
                               order_book: Dict[str, Any]) -> float:
        """
        Estimate the market impact of a trade.
        
        Args:
            symbol: Trading symbol
            trade_size: Size of intended trade
            side: Trade side
            order_book: Order book data
            
        Returns:
            Estimated market impact as percentage
        """
        # Extract relevant order book data
        if side.lower() == 'buy':
            book_side = order_book.get('asks', [])
        else:
            book_side = order_book.get('bids', [])
            
        if not book_side:
            return 0.5  # Default if no order book data
            
        # Simple implementation - can be enhanced with square-root impact model
        total_liquidity = sum(level[1] for level in book_side[:10])
        
        if total_liquidity == 0:
            return 1.0
            
        # Calculate impact as a percentage
        impact = min(1.0, (trade_size / total_liquidity) * 0.5)
        
        return impact
    
    def _analyze_liquidity(self, 
                          symbol: str, 
                          trade_size: float, 
                          market_data: Dict[str, Any],
                          order_book: Dict[str, Any]) -> float:
        """
        Analyze market liquidity relative to trade size.
        
        Args:
            symbol: Trading symbol
            trade_size: Size of intended trade
            market_data: Market data dictionary
            order_book: Order book data
            
        Returns:
            Liquidity ratio (higher is better)
        """
        # Extract trading volume
        volume = market_data.get('volume_24h', 0)
        
        if volume == 0:
            return 0.0
            
        # Calculate liquidity ratio
        liquidity_ratio = volume / trade_size if trade_size > 0 else float('inf')
        
        # Normalize for reasonable values
        normalized_ratio = min(1.0, liquidity_ratio / 1000)
        
        return normalized_ratio
    
    def _analyze_volatility(self, 
                           symbol: str, 
                           market_data: Dict[str, Any]) -> float:
        """
        Analyze current market volatility.
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary
            
        Returns:
            Volatility measure
        """
        # Extract volatility or calculate from price data
        if 'volatility' in market_data:
            return market_data['volatility']
            
        # Calculate from recent price history if available
        if 'recent_prices' in market_data and len(market_data['recent_prices']) > 1:
            prices = market_data['recent_prices']
            returns = np.diff(np.log(prices))
            return np.std(returns) * np.sqrt(252 * 24 * 4)  # Annualized, assuming 15-min data
            
        return 0.2  # Default volatility assumption
    
    def _analyze_spread_cost(self, 
                            symbol: str, 
                            trade_size: float, 
                            order_book: Dict[str, Any]) -> float:
        """
        Analyze the spread cost impact.
        
        Args:
            symbol: Trading symbol
            trade_size: Size of intended trade
            order_book: Order book data
            
        Returns:
            Spread cost as percentage
        """
        # Get best bid and ask
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return 0.01  # Default 1% if no data
            
        if not order_book['bids'] or not order_book['asks']:
            return 0.01
            
        best_bid = order_book['bids'][0][0] if order_book['bids'] else 0
        best_ask = order_book['asks'][0][0] if order_book['asks'] else 0
        
        if best_bid == 0 or best_ask == 0:
            return 0.01
            
        # Calculate spread as percentage
        mid_price = (best_bid + best_ask) / 2
        spread_pct = (best_ask - best_bid) / mid_price
        
        return spread_pct
    
    def _analyze_timing(self, 
                       symbol: str, 
                       side: str, 
                       market_data: Dict[str, Any]) -> float:
        """
        Analyze the timing quality for the trade.
        
        Args:
            symbol: Trading symbol
            side: Trade side
            market_data: Market data dictionary
            
        Returns:
            Timing score (0.0 to 1.0, higher is better)
        """
        # Simple implementation - to be enhanced with trend analysis
        # For now, return a random score between 0.3 and 0.8
        return 0.3 + (0.5 * np.random.random())
    
    def _calculate_risk_metrics(self, 
                               symbol: str, 
                               trade_size: float, 
                               side: str, 
                               market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate various risk metrics for the trade.
        
        Args:
            symbol: Trading symbol
            trade_size: Size of intended trade
            side: Trade side
            market_data: Market data dictionary
            
        Returns:
            Dictionary of risk metrics
        """
        # Extract or estimate volatility
        volatility = self._analyze_volatility(symbol, market_data)
        
        # Current price
        current_price = market_data.get('price', 1.0)
        
        # Calculate notional value
        notional_value = trade_size * current_price
        
        # Calculate VaR (simple approximation)
        var_95 = notional_value * volatility * 1.65 / np.sqrt(252)
        
        # Calculate expected shortfall (simple approximation)
        es_95 = notional_value * volatility * 2.06 / np.sqrt(252)
        
        # Return risk metrics
        return {
            'value_at_risk_95': var_95,
            'expected_shortfall_95': es_95,
            'notional_value': notional_value,
            'price_volatility': volatility
        }
    
    def _recommend_execution_params(self, 
                                   symbol: str, 
                                   trade_size: float, 
                                   side: str, 
                                   impact: float, 
                                   liquidity: float, 
                                   volatility: float) -> Dict[str, Any]:
        """
        Recommend execution parameters based on analysis.
        
        Args:
            symbol: Trading symbol
            trade_size: Size of intended trade
            side: Trade side
            impact: Estimated market impact
            liquidity: Liquidity ratio
            volatility: Volatility measure
            
        Returns:
            Dictionary of recommended execution parameters
        """
        # Determine execution algorithm based on size and liquidity
        if trade_size > 1000 or impact > 0.02:
            algo = 'TWAP'
            duration_minutes = min(240, max(15, int(impact * 1000)))
        elif volatility > 0.4:
            algo = 'Adaptive'
            duration_minutes = 30
        else:
            algo = 'Market'
            duration_minutes = 0
            
        # Determine order slicing
        if algo != 'Market':
            num_slices = max(1, min(8, int(impact * 100)))
        else:
            num_slices = 1
            
        # Recommend limit price buffer
        limit_buffer = max(0.001, min(0.01, volatility * 0.1))
        
        return {
            'recommended_algo': algo,
            'duration_minutes': duration_minutes,
            'num_slices': num_slices,
            'limit_buffer': limit_buffer
        }
    
    def _make_recommendation(self, 
                            impact: float, 
                            liquidity: float, 
                            volatility: float, 
                            spread_cost: float, 
                            timing_score: float,
                            risks: Dict[str, float]) -> Tuple[str, float]:
        """
        Make overall recommendation based on analysis.
        
        Args:
            impact: Estimated market impact
            liquidity: Liquidity ratio
            volatility: Volatility measure
            spread_cost: Spread cost
            timing_score: Timing quality score
            risks: Risk metrics dictionary
            
        Returns:
            Tuple of (recommendation, confidence)
        """
        # Default to proceed
        recommendation = 'proceed'
        
        # Calculate weighted score
        score = (
            (1.0 - impact) * 0.3 +
            liquidity * 0.2 +
            (1.0 - min(1.0, volatility / self.volatility_threshold)) * 0.15 +
            (1.0 - spread_cost) * 0.15 +
            timing_score * 0.2
        )
        
        # Decision logic
        if impact > self.market_impact_threshold:
            recommendation = 'defer'
            confidence = 0.7
        elif liquidity < self.liquidity_threshold:
            recommendation = 'reduce_size'
            confidence = 0.8
        elif volatility > self.volatility_threshold:
            recommendation = 'defer' if timing_score < 0.6 else 'proceed_with_caution'
            confidence = 0.6
        else:
            # Normal case - confidence based on overall score
            confidence = min(0.95, max(0.5, score))
            if score < 0.4:
                recommendation = 'defer'
            elif score < 0.6:
                recommendation = 'proceed_with_caution'
            else:
                recommendation = 'proceed'
                
        return recommendation, confidence 