import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta

class FearGreedIndex:
    def __init__(self):
        """Initialize the Fear & Greed Index calculator."""
        self.indicators = {
            'volatility': 0.25,
            'market_momentum': 0.25,
            'social_media': 0.15,
            'dominance': 0.15,
            'trends': 0.20
        }
        self.historical_data = []
        
    def calculate_volatility_score(self, price_data: pd.DataFrame, window: int = 30) -> float:
        """
        Calculate volatility score based on price movements.
        
        Args:
            price_data: DataFrame with price data
            window: Lookback window in days
            
        Returns:
            Volatility score between 0 and 100
        """
        returns = price_data['close'].pct_change()
        volatility = returns.rolling(window=window).std()
        
        # Normalize to 0-100 scale
        max_vol = volatility.max()
        min_vol = volatility.min()
        score = 100 * (volatility - min_vol) / (max_vol - min_vol)
        
        return score.iloc[-1]
        
    def calculate_momentum_score(self, price_data: pd.DataFrame) -> float:
        """
        Calculate market momentum score.
        
        Args:
            price_data: DataFrame with price data
            
        Returns:
            Momentum score between 0 and 100
        """
        # Calculate various momentum indicators
        rsi = self._calculate_rsi(price_data['close'])
        macd = self._calculate_macd(price_data['close'])
        
        # Combine indicators
        momentum_score = (rsi + macd) / 2
        return momentum_score
        
    def calculate_social_score(self, social_data: pd.DataFrame) -> float:
        """
        Calculate social media sentiment score.
        
        Args:
            social_data: DataFrame with social media sentiment data
            
        Returns:
            Social score between 0 and 100
        """
        if social_data.empty:
            return 50.0
            
        # Calculate weighted average sentiment
        weighted_sentiment = np.average(
            social_data['sentiment'],
            weights=social_data['volume']
        )
        
        # Convert to 0-100 scale
        return 50 + (weighted_sentiment * 50)
        
    def calculate_dominance_score(self, dominance_data: pd.DataFrame) -> float:
        """
        Calculate Bitcoin dominance score.
        
        Args:
            dominance_data: DataFrame with Bitcoin dominance data
            
        Returns:
            Dominance score between 0 and 100
        """
        if dominance_data.empty:
            return 50.0
            
        # Calculate deviation from mean
        mean_dominance = dominance_data['dominance'].mean()
        current_dominance = dominance_data['dominance'].iloc[-1]
        
        # Convert to 0-100 scale
        score = 50 + ((current_dominance - mean_dominance) / mean_dominance * 50)
        return np.clip(score, 0, 100)
        
    def calculate_trends_score(self, trends_data: pd.DataFrame) -> float:
        """
        Calculate search trends score.
        
        Args:
            trends_data: DataFrame with search trends data
            
        Returns:
            Trends score between 0 and 100
        """
        if trends_data.empty:
            return 50.0
            
        # Calculate relative search volume
        max_volume = trends_data['volume'].max()
        current_volume = trends_data['volume'].iloc[-1]
        
        # Convert to 0-100 scale
        return (current_volume / max_volume) * 100
        
    def calculate_index(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate the Fear & Greed Index.
        
        Args:
            data: Dictionary containing all required data sources
            
        Returns:
            Dictionary with index score and component scores
        """
        scores = {}
        
        # Calculate individual component scores
        scores['volatility'] = self.calculate_volatility_score(data['price'])
        scores['momentum'] = self.calculate_momentum_score(data['price'])
        scores['social'] = self.calculate_social_score(data['social'])
        scores['dominance'] = self.calculate_dominance_score(data['dominance'])
        scores['trends'] = self.calculate_trends_score(data['trends'])
        
        # Calculate weighted average
        total_weight = sum(self.indicators.values())
        index_score = sum(
            scores[component] * weight
            for component, weight in self.indicators.items()
        ) / total_weight
        
        # Determine sentiment category
        if index_score >= 75:
            sentiment = 'Extreme Greed'
        elif index_score >= 60:
            sentiment = 'Greed'
        elif index_score >= 40:
            sentiment = 'Neutral'
        elif index_score >= 25:
            sentiment = 'Fear'
        else:
            sentiment = 'Extreme Fear'
            
        return {
            'index_score': index_score,
            'sentiment': sentiment,
            'component_scores': scores,
            'timestamp': datetime.now()
        }
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
        
    def _calculate_macd(self, prices: pd.Series) -> float:
        """Calculate MACD signal."""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # Convert to 0-100 scale
        return 50 + (macd.iloc[-1] - signal.iloc[-1]) * 10
