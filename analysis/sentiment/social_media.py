import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta
from textblob import TextBlob
import re

class SocialMediaAnalyzer:
    def __init__(self):
        """Initialize the social media analyzer."""
        self.platform_weights = {
            'twitter': 1.0,
            'reddit': 0.8,
            'telegram': 0.7,
            'discord': 0.6
        }
        self.sentiment_data = []
        
    def clean_text(self, text: str) -> str:
        """
        Clean social media text for analysis.
        
        Args:
            text: Raw social media text
            
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
        
    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of a text using TextBlob.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score between -1 and 1
        """
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
        
    def process_social_posts(self, posts: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process social media posts and calculate sentiment scores.
        
        Args:
            posts: List of social media posts with platform, text, and timestamp
            
        Returns:
            DataFrame with sentiment analysis results
        """
        results = []
        
        for post in posts:
            platform = post.get('platform', 'unknown')
            text = self.clean_text(post.get('text', ''))
            
            if not text:
                continue
                
            sentiment_score = self.analyze_sentiment(text)
            platform_weight = self.platform_weights.get(platform, 0.5)
            
            results.append({
                'timestamp': post.get('timestamp', datetime.now()),
                'platform': platform,
                'text': text,
                'raw_sentiment': sentiment_score,
                'weighted_sentiment': sentiment_score * platform_weight,
                'engagement': post.get('engagement', 0)
            })
            
        return pd.DataFrame(results)
        
    def calculate_platform_sentiment(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate average sentiment per platform.
        
        Args:
            df: DataFrame with sentiment data
            
        Returns:
            Dictionary with platform sentiment scores
        """
        platform_sentiment = {}
        
        for platform in df['platform'].unique():
            platform_data = df[df['platform'] == platform]
            weighted_avg = np.average(
                platform_data['raw_sentiment'],
                weights=platform_data['engagement']
            )
            platform_sentiment[platform] = weighted_avg
            
        return platform_sentiment
        
    def get_social_momentum(self, df: pd.DataFrame, window: str = '1H') -> float:
        """
        Calculate social media momentum based on recent sentiment changes.
        
        Args:
            df: DataFrame with sentiment data
            window: Time window for momentum calculation
            
        Returns:
            Momentum score between -1 and 1
        """
        if len(df) < 2:
            return 0.0
            
        # Convert timestamp to datetime if needed
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate weighted sentiment over time
        df['weighted_sentiment'] = df['weighted_sentiment'] * df['engagement']
        df.set_index('timestamp', inplace=True)
        
        # Resample and calculate momentum
        resampled = df.resample(window)['weighted_sentiment'].mean()
        momentum = resampled.diff().mean()
        
        # Normalize momentum to [-1, 1]
        return np.clip(momentum, -1, 1)
        
    def detect_sentiment_shifts(self, df: pd.DataFrame, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Detect significant shifts in social media sentiment.
        
        Args:
            df: DataFrame with sentiment data
            threshold: Minimum change in sentiment to consider significant
            
        Returns:
            List of detected sentiment shifts
        """
        shifts = []
        
        # Calculate rolling sentiment
        df['rolling_sentiment'] = df['weighted_sentiment'].rolling(window=5).mean()
        
        # Find significant changes
        for i in range(1, len(df)):
            change = abs(df['rolling_sentiment'].iloc[i] - df['rolling_sentiment'].iloc[i-1])
            if change > threshold:
                shifts.append({
                    'timestamp': df.index[i],
                    'change': change,
                    'direction': 'positive' if df['rolling_sentiment'].iloc[i] > df['rolling_sentiment'].iloc[i-1] else 'negative'
                })
                
        return shifts
