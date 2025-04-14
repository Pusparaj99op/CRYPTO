import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict

class CrowdPsychologyAnalyzer:
    def __init__(self):
        """Initialize the crowd psychology analyzer."""
        self.psychology_metrics = {
            'fear_greed': 0.3,
            'social_sentiment': 0.2,
            'market_structure': 0.2,
            'trading_activity': 0.15,
            'news_sentiment': 0.15
        }
        self.historical_data = []
        self.market_phases = []
        
    def process_market_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Process market psychology data.
        
        Args:
            data: Dictionary containing various market psychology indicators
            
        Returns:
            DataFrame with processed psychology data
        """
        results = []
        
        # Extract and normalize metrics
        fear_greed = data.get('fear_greed', 50)
        social_sentiment = data.get('social_sentiment', 0)
        market_structure = data.get('market_structure', 0)
        trading_activity = data.get('trading_activity', 0)
        news_sentiment = data.get('news_sentiment', 0)
        
        # Calculate composite psychology score
        total_weight = sum(self.psychology_metrics.values())
        psychology_score = (
            fear_greed * self.psychology_metrics['fear_greed'] +
            (social_sentiment + 1) * 50 * self.psychology_metrics['social_sentiment'] +
            (market_structure + 1) * 50 * self.psychology_metrics['market_structure'] +
            trading_activity * self.psychology_metrics['trading_activity'] +
            (news_sentiment + 1) * 50 * self.psychology_metrics['news_sentiment']
        ) / total_weight
        
        results.append({
            'timestamp': data.get('timestamp', datetime.now()),
            'psychology_score': psychology_score,
            'fear_greed': fear_greed,
            'social_sentiment': social_sentiment,
            'market_structure': market_structure,
            'trading_activity': trading_activity,
            'news_sentiment': news_sentiment
        })
        
        # Update historical data
        self.historical_data.extend(results)
        
        return pd.DataFrame(results)
        
    def identify_market_phase(self, df: pd.DataFrame) -> str:
        """
        Identify current market psychology phase.
        
        Args:
            df: DataFrame with psychology data
            
        Returns:
            String identifying market phase
        """
        if df.empty:
            return 'neutral'
            
        current_score = df['psychology_score'].iloc[-1]
        
        if current_score >= 75:
            return 'euphoria'
        elif current_score >= 60:
            return 'optimism'
        elif current_score >= 40:
            return 'neutral'
        elif current_score >= 25:
            return 'anxiety'
        else:
            return 'panic'
            
    def calculate_market_cycle(self, window: str = '30D') -> Dict[str, Any]:
        """
        Calculate market psychology cycle.
        
        Args:
            window: Time window for analysis
            
        Returns:
            Dictionary with cycle analysis
        """
        if not self.historical_data:
            return {'phase': 'neutral', 'trend': 'stable'}
            
        # Convert to DataFrame
        df = pd.DataFrame(self.historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample data
        resampled = df.resample(window)['psychology_score'].mean()
        
        # Calculate trend
        if len(resampled) > 1:
            trend = np.polyfit(range(len(resampled)), resampled, 1)[0]
        else:
            trend = 0
            
        # Determine cycle phase
        current_phase = self.identify_market_phase(df)
        
        # Determine trend direction
        if trend > 0.1:
            trend_direction = 'improving'
        elif trend < -0.1:
            trend_direction = 'deteriorating'
        else:
            trend_direction = 'stable'
            
        return {
            'phase': current_phase,
            'trend': trend_direction,
            'score': df['psychology_score'].iloc[-1],
            'momentum': trend
        }
        
    def detect_psychology_shifts(self, threshold: float = 0.2) -> List[Dict[str, Any]]:
        """
        Detect significant shifts in market psychology.
        
        Args:
            threshold: Minimum change to consider significant
            
        Returns:
            List of detected psychology shifts
        """
        shifts = []
        
        if not self.historical_data:
            return shifts
            
        # Convert to DataFrame
        df = pd.DataFrame(self.historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate rolling psychology score
        df['rolling_score'] = df['psychology_score'].rolling(window=5).mean()
        
        # Find significant changes
        for i in range(1, len(df)):
            change = abs(df['rolling_score'].iloc[i] - df['rolling_score'].iloc[i-1])
            if change > threshold:
                shifts.append({
                    'timestamp': df.index[i],
                    'change': change,
                    'direction': 'positive' if df['rolling_score'].iloc[i] > df['rolling_score'].iloc[i-1] else 'negative',
                    'from_phase': self.identify_market_phase(df.iloc[:i]),
                    'to_phase': self.identify_market_phase(df.iloc[:i+1])
                })
                
        return shifts
        
    def analyze_herd_behavior(self, window: str = '7D') -> Dict[str, Any]:
        """
        Analyze herd behavior in the market.
        
        Args:
            window: Time window for analysis
            
        Returns:
            Dictionary with herd behavior analysis
        """
        if not self.historical_data:
            return {'herd_strength': 0, 'direction': 'neutral'}
            
        # Convert to DataFrame
        df = pd.DataFrame(self.historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate metrics
        resampled = df.resample(window).agg({
            'psychology_score': 'mean',
            'trading_activity': 'mean',
            'social_sentiment': 'mean'
        })
        
        # Calculate herd behavior indicators
        score_volatility = resampled['psychology_score'].std()
        activity_correlation = resampled['trading_activity'].corr(resampled['psychology_score'])
        sentiment_correlation = resampled['social_sentiment'].corr(resampled['psychology_score'])
        
        # Calculate herd strength
        herd_strength = (
            (1 - score_volatility) * 0.4 +
            abs(activity_correlation) * 0.3 +
            abs(sentiment_correlation) * 0.3
        ) * 100
        
        # Determine herd direction
        if resampled['psychology_score'].iloc[-1] > 60:
            direction = 'bullish'
        elif resampled['psychology_score'].iloc[-1] < 40:
            direction = 'bearish'
        else:
            direction = 'neutral'
            
        return {
            'herd_strength': herd_strength,
            'direction': direction,
            'score_volatility': score_volatility,
            'activity_correlation': activity_correlation,
            'sentiment_correlation': sentiment_correlation
        }
        
    def predict_market_turning_points(self, window: str = '14D') -> List[Dict[str, Any]]:
        """
        Predict potential market turning points based on psychology.
        
        Args:
            window: Time window for analysis
            
        Returns:
            List of predicted turning points
        """
        predictions = []
        
        if not self.historical_data:
            return predictions
            
        # Convert to DataFrame
        df = pd.DataFrame(self.historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate indicators
        df['score_ma'] = df['psychology_score'].rolling(window=window).mean()
        df['score_std'] = df['psychology_score'].rolling(window=window).std()
        df['z_score'] = (df['psychology_score'] - df['score_ma']) / df['score_std']
        
        # Find potential turning points
        for i in range(window, len(df)):
            if abs(df['z_score'].iloc[i]) > 2:  # Extreme psychology
                predictions.append({
                    'timestamp': df.index[i],
                    'type': 'top' if df['z_score'].iloc[i] > 0 else 'bottom',
                    'confidence': min(abs(df['z_score'].iloc[i]) / 3, 1),
                    'psychology_score': df['psychology_score'].iloc[i],
                    'z_score': df['z_score'].iloc[i]
                })
                
        return predictions
