import pandas as pd
import numpy as np
from typing import List, Dict, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict

class InfluencerTracker:
    def __init__(self):
        """Initialize the influencer tracker."""
        self.influencers = {}  # Dictionary to store influencer data
        self.tracked_metrics = {
            'followers': 0.3,
            'engagement': 0.3,
            'sentiment': 0.2,
            'accuracy': 0.2
        }
        self.historical_performance = defaultdict(list)
        
    def add_influencer(self, influencer_id: str, data: Dict[str, Any]):
        """
        Add or update an influencer's data.
        
        Args:
            influencer_id: Unique identifier for the influencer
            data: Dictionary containing influencer metrics
        """
        self.influencers[influencer_id] = {
            'name': data.get('name', ''),
            'platform': data.get('platform', ''),
            'followers': data.get('followers', 0),
            'engagement_rate': data.get('engagement_rate', 0),
            'sentiment_score': data.get('sentiment_score', 0),
            'prediction_accuracy': data.get('prediction_accuracy', 0.5),
            'last_updated': datetime.now()
        }
        
    def calculate_influence_score(self, influencer_id: str) -> float:
        """
        Calculate overall influence score for an influencer.
        
        Args:
            influencer_id: ID of the influencer
            
        Returns:
            Influence score between 0 and 100
        """
        if influencer_id not in self.influencers:
            return 0.0
            
        influencer = self.influencers[influencer_id]
        
        # Normalize metrics to 0-100 scale
        followers_score = min(influencer['followers'] / 1000000, 100)  # Cap at 1M followers
        engagement_score = influencer['engagement_rate'] * 100
        sentiment_score = (influencer['sentiment_score'] + 1) * 50  # Convert -1 to 1 to 0 to 100
        accuracy_score = influencer['prediction_accuracy'] * 100
        
        # Calculate weighted average
        total_weight = sum(self.tracked_metrics.values())
        influence_score = (
            followers_score * self.tracked_metrics['followers'] +
            engagement_score * self.tracked_metrics['engagement'] +
            sentiment_score * self.tracked_metrics['sentiment'] +
            accuracy_score * self.tracked_metrics['accuracy']
        ) / total_weight
        
        return influence_score
        
    def track_prediction(self, influencer_id: str, prediction: Dict[str, Any], actual_outcome: float):
        """
        Track and update an influencer's prediction accuracy.
        
        Args:
            influencer_id: ID of the influencer
            prediction: Dictionary containing prediction details
            actual_outcome: Actual market outcome
        """
        if influencer_id not in self.influencers:
            return
            
        # Calculate prediction accuracy
        predicted_direction = prediction.get('direction', 'neutral')
        predicted_magnitude = prediction.get('magnitude', 0)
        
        # Determine if prediction was correct
        if predicted_direction == 'up' and actual_outcome > 0:
            accuracy = 1.0
        elif predicted_direction == 'down' and actual_outcome < 0:
            accuracy = 1.0
        else:
            accuracy = 0.0
            
        # Update historical performance
        self.historical_performance[influencer_id].append({
            'timestamp': datetime.now(),
            'prediction': prediction,
            'actual_outcome': actual_outcome,
            'accuracy': accuracy
        })
        
        # Update influencer's accuracy score
        recent_performance = self.historical_performance[influencer_id][-10:]  # Last 10 predictions
        if recent_performance:
            new_accuracy = sum(p['accuracy'] for p in recent_performance) / len(recent_performance)
            self.influencers[influencer_id]['prediction_accuracy'] = new_accuracy
            
    def get_top_influencers(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get top n influencers by influence score.
        
        Args:
            n: Number of top influencers to return
            
        Returns:
            List of top influencer data
        """
        influencer_scores = [
            {
                'id': influencer_id,
                'name': data['name'],
                'platform': data['platform'],
                'influence_score': self.calculate_influence_score(influencer_id),
                'followers': data['followers'],
                'engagement_rate': data['engagement_rate'],
                'prediction_accuracy': data['prediction_accuracy']
            }
            for influencer_id, data in self.influencers.items()
        ]
        
        # Sort by influence score
        top_influencers = sorted(
            influencer_scores,
            key=lambda x: x['influence_score'],
            reverse=True
        )[:n]
        
        return top_influencers
        
    def analyze_sentiment_trend(self, influencer_id: str, window: int = 7) -> Dict[str, Any]:
        """
        Analyze sentiment trend for an influencer.
        
        Args:
            influencer_id: ID of the influencer
            window: Lookback window in days
            
        Returns:
            Dictionary with sentiment trend analysis
        """
        if influencer_id not in self.historical_performance:
            return {'trend': 'neutral', 'strength': 0}
            
        # Get recent predictions
        recent_predictions = [
            p for p in self.historical_performance[influencer_id]
            if (datetime.now() - p['timestamp']).days <= window
        ]
        
        if not recent_predictions:
            return {'trend': 'neutral', 'strength': 0}
            
        # Calculate sentiment trend
        sentiments = [1 if p['prediction']['direction'] == 'up' else -1 for p in recent_predictions]
        trend = np.polyfit(range(len(sentiments)), sentiments, 1)[0]
        
        # Determine trend direction and strength
        if trend > 0.1:
            direction = 'bullish'
        elif trend < -0.1:
            direction = 'bearish'
        else:
            direction = 'neutral'
            
        return {
            'trend': direction,
            'strength': abs(trend),
            'sample_size': len(recent_predictions)
        }
        
    def get_influencer_network(self) -> Dict[str, Set[str]]:
        """
        Analyze influencer network and relationships.
        
        Returns:
            Dictionary mapping influencer IDs to sets of related influencers
        """
        network = defaultdict(set)
        
        # This is a placeholder for actual network analysis
        # In a real implementation, this would analyze interactions,
        # mentions, and relationships between influencers
        
        return network
