import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import re
from textblob import TextBlob
from collections import defaultdict

class ForumAnalyzer:
    def __init__(self):
        """Initialize the forum analyzer."""
        self.forum_data = []
        self.topic_sentiment = defaultdict(list)
        self.user_sentiment = defaultdict(list)
        
    def clean_text(self, text: str) -> str:
        """
        Clean forum text for analysis.
        
        Args:
            text: Raw forum text
            
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.lower()
        
    def extract_topics(self, text: str) -> List[str]:
        """
        Extract relevant topics from forum text.
        
        Args:
            text: Forum text to analyze
            
        Returns:
            List of identified topics
        """
        # Common crypto topics
        topics = [
            'bitcoin', 'ethereum', 'defi', 'nft', 'staking',
            'mining', 'trading', 'wallet', 'exchange', 'scam',
            'regulation', 'adoption', 'technology', 'price',
            'market', 'bullish', 'bearish'
        ]
        
        found_topics = []
        for topic in topics:
            if topic in text.lower():
                found_topics.append(topic)
                
        return found_topics
        
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of forum text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        analysis = TextBlob(text)
        
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity
        }
        
    def process_forum_posts(self, posts: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process forum posts and extract sentiment data.
        
        Args:
            posts: List of forum posts with text and metadata
            
        Returns:
            DataFrame with analysis results
        """
        results = []
        
        for post in posts:
            text = self.clean_text(post.get('text', ''))
            if not text:
                continue
                
            # Extract topics and sentiment
            topics = self.extract_topics(text)
            sentiment = self.analyze_sentiment(text)
            
            results.append({
                'timestamp': post.get('timestamp', datetime.now()),
                'user_id': post.get('user_id', ''),
                'forum': post.get('forum', ''),
                'topics': topics,
                'polarity': sentiment['polarity'],
                'subjectivity': sentiment['subjectivity'],
                'upvotes': post.get('upvotes', 0),
                'downvotes': post.get('downvotes', 0)
            })
            
            # Update topic and user sentiment tracking
            for topic in topics:
                self.topic_sentiment[topic].append({
                    'timestamp': results[-1]['timestamp'],
                    'polarity': sentiment['polarity'],
                    'subjectivity': sentiment['subjectivity']
                })
                
            self.user_sentiment[results[-1]['user_id']].append({
                'timestamp': results[-1]['timestamp'],
                'polarity': sentiment['polarity'],
                'subjectivity': sentiment['subjectivity']
            })
            
        return pd.DataFrame(results)
        
    def get_topic_sentiment(self, topic: str, window: str = '1D') -> Dict[str, Any]:
        """
        Get sentiment analysis for a specific topic.
        
        Args:
            topic: Topic to analyze
            window: Time window for aggregation
            
        Returns:
            Dictionary with topic sentiment analysis
        """
        if topic not in self.topic_sentiment:
            return {'sentiment': 'neutral', 'strength': 0}
            
        # Convert to DataFrame
        df = pd.DataFrame(self.topic_sentiment[topic])
        if df.empty:
            return {'sentiment': 'neutral', 'strength': 0}
            
        # Convert timestamp and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample and calculate average sentiment
        resampled = df.resample(window)['polarity'].mean()
        
        # Calculate trend
        if len(resampled) > 1:
            trend = np.polyfit(range(len(resampled)), resampled, 1)[0]
        else:
            trend = 0
            
        # Determine sentiment
        current_sentiment = resampled.iloc[-1]
        if current_sentiment > 0.1:
            sentiment = 'positive'
        elif current_sentiment < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        return {
            'sentiment': sentiment,
            'strength': abs(current_sentiment),
            'trend': trend,
            'sample_size': len(df)
        }
        
    def get_user_sentiment_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Get sentiment profile for a specific user.
        
        Args:
            user_id: User ID to analyze
            
        Returns:
            Dictionary with user sentiment profile
        """
        if user_id not in self.user_sentiment:
            return {'sentiment': 'neutral', 'consistency': 0}
            
        # Convert to DataFrame
        df = pd.DataFrame(self.user_sentiment[user_id])
        if df.empty:
            return {'sentiment': 'neutral', 'consistency': 0}
            
        # Calculate average sentiment
        avg_polarity = df['polarity'].mean()
        avg_subjectivity = df['subjectivity'].mean()
        
        # Calculate sentiment consistency
        consistency = 1 - df['polarity'].std()
        
        # Determine sentiment type
        if avg_polarity > 0.1:
            sentiment = 'bullish'
        elif avg_polarity < -0.1:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
            
        return {
            'sentiment': sentiment,
            'strength': abs(avg_polarity),
            'subjectivity': avg_subjectivity,
            'consistency': consistency,
            'post_count': len(df)
        }
        
    def detect_sentiment_shifts(self, window: str = '1H') -> List[Dict[str, Any]]:
        """
        Detect significant shifts in forum sentiment.
        
        Args:
            window: Time window for analysis
            
        Returns:
            List of detected sentiment shifts
        """
        shifts = []
        
        # Aggregate all posts by time window
        df = pd.DataFrame(self.forum_data)
        if df.empty:
            return shifts
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate rolling sentiment
        rolling_sentiment = df['polarity'].rolling(window=window).mean()
        
        # Find significant changes
        for i in range(1, len(rolling_sentiment)):
            change = abs(rolling_sentiment.iloc[i] - rolling_sentiment.iloc[i-1])
            if change > 0.3:  # Threshold for significant change
                shifts.append({
                    'timestamp': rolling_sentiment.index[i],
                    'change': change,
                    'direction': 'positive' if rolling_sentiment.iloc[i] > rolling_sentiment.iloc[i-1] else 'negative',
                    'top_topics': self._get_top_topics(rolling_sentiment.index[i], window)
                })
                
        return shifts
        
    def _get_top_topics(self, timestamp: datetime, window: str) -> List[str]:
        """
        Get top topics discussed around a specific timestamp.
        
        Args:
            timestamp: Reference timestamp
            window: Time window to consider
            
        Returns:
            List of top topics
        """
        # Filter posts within the time window
        window_start = timestamp - pd.Timedelta(window)
        relevant_posts = [
            post for post in self.forum_data
            if window_start <= post['timestamp'] <= timestamp
        ]
        
        # Count topic occurrences
        topic_counts = defaultdict(int)
        for post in relevant_posts:
            for topic in post.get('topics', []):
                topic_counts[topic] += 1
                
        # Return top 5 topics
        return sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
