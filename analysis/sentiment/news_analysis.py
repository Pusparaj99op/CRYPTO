import pandas as pd
import numpy as np
from transformers import pipeline
from typing import List, Dict, Any
from datetime import datetime

class NewsAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the news analyzer with a sentiment analysis model.
        
        Args:
            model_name: Name of the pre-trained model to use for sentiment analysis
        """
        self.sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)
        self.news_data = []
        
    def analyze_news(self, news_items: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Analyze a list of news items and return sentiment scores.
        
        Args:
            news_items: List of dictionaries containing news data with 'title' and 'content' keys
            
        Returns:
            DataFrame containing sentiment analysis results
        """
        results = []
        
        for item in news_items:
            # Combine title and content for analysis
            text = f"{item.get('title', '')} {item.get('content', '')}"
            
            # Get sentiment score
            sentiment_result = self.sentiment_analyzer(text)[0]
            
            # Convert sentiment to numerical score (-1 to 1)
            score = 1 if sentiment_result['label'] == 'POSITIVE' else -1
            score *= sentiment_result['score']
            
            results.append({
                'timestamp': item.get('timestamp', datetime.now()),
                'source': item.get('source', 'unknown'),
                'title': item.get('title', ''),
                'sentiment_score': score,
                'confidence': sentiment_result['score']
            })
            
        return pd.DataFrame(results)
    
    def aggregate_sentiment(self, df: pd.DataFrame, window: str = '1D') -> pd.DataFrame:
        """
        Aggregate sentiment scores over time.
        
        Args:
            df: DataFrame with sentiment scores
            window: Time window for aggregation (e.g., '1D', '1H')
            
        Returns:
            DataFrame with aggregated sentiment scores
        """
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate weighted average sentiment
        df['weighted_sentiment'] = df['sentiment_score'] * df['confidence']
        
        # Aggregate by time window
        aggregated = df.resample(window).agg({
            'sentiment_score': 'mean',
            'weighted_sentiment': 'mean',
            'confidence': 'mean'
        })
        
        return aggregated
    
    def get_sentiment_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate sentiment trend metrics.
        
        Args:
            df: DataFrame with sentiment scores
            
        Returns:
            Dictionary containing trend metrics
        """
        if len(df) < 2:
            return {'trend': 'neutral', 'strength': 0}
            
        # Calculate trend
        recent_scores = df['sentiment_score'].tail(5)
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        # Determine trend direction and strength
        if trend > 0.1:
            direction = 'positive'
        elif trend < -0.1:
            direction = 'negative'
        else:
            direction = 'neutral'
            
        return {
            'trend': direction,
            'strength': abs(trend),
            'current_sentiment': df['sentiment_score'].iloc[-1]
        }
