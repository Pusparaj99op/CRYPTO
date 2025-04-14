import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict

class SearchTrendsAnalyzer:
    def __init__(self):
        """Initialize the search trends analyzer."""
        self.search_data = defaultdict(list)
        self.related_terms = defaultdict(list)
        self.trend_metrics = {
            'volume': 0.4,
            'growth': 0.3,
            'correlation': 0.3
        }
        
    def process_search_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process search trend data.
        
        Args:
            data: List of search trend data points
            
        Returns:
            DataFrame with processed search data
        """
        results = []
        
        for item in data:
            term = item.get('term', '')
            if not term:
                continue
                
            results.append({
                'timestamp': item.get('timestamp', datetime.now()),
                'term': term,
                'volume': item.get('volume', 0),
                'related_terms': item.get('related_terms', []),
                'region': item.get('region', 'global')
            })
            
            # Update search data tracking
            self.search_data[term].append({
                'timestamp': results[-1]['timestamp'],
                'volume': results[-1]['volume'],
                'region': results[-1]['region']
            })
            
            # Update related terms tracking
            for related in results[-1]['related_terms']:
                self.related_terms[term].append({
                    'term': related,
                    'timestamp': results[-1]['timestamp']
                })
                
        return pd.DataFrame(results)
        
    def calculate_trend_score(self, term: str, window: str = '7D') -> float:
        """
        Calculate trend score for a search term.
        
        Args:
            term: Search term to analyze
            window: Time window for analysis
            
        Returns:
            Trend score between 0 and 100
        """
        if term not in self.search_data:
            return 0.0
            
        # Convert to DataFrame
        df = pd.DataFrame(self.search_data[term])
        if df.empty:
            return 0.0
            
        # Convert timestamp and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate metrics
        current_volume = df['volume'].iloc[-1]
        max_volume = df['volume'].max()
        volume_score = (current_volume / max_volume) * 100 if max_volume > 0 else 0
        
        # Calculate growth rate
        resampled = df.resample(window)['volume'].mean()
        if len(resampled) > 1:
            growth = (resampled.iloc[-1] - resampled.iloc[-2]) / resampled.iloc[-2] * 100
            growth_score = min(max(growth, 0), 100)
        else:
            growth_score = 0
            
        # Calculate correlation with price (if available)
        correlation_score = self._calculate_correlation_score(term, window)
        
        # Calculate weighted average score
        total_weight = sum(self.trend_metrics.values())
        trend_score = (
            volume_score * self.trend_metrics['volume'] +
            growth_score * self.trend_metrics['growth'] +
            correlation_score * self.trend_metrics['correlation']
        ) / total_weight
        
        return trend_score
        
    def get_related_terms(self, term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get related search terms for a given term.
        
        Args:
            term: Search term to analyze
            limit: Maximum number of related terms to return
            
        Returns:
            List of related terms with scores
        """
        if term not in self.related_terms:
            return []
            
        # Count term occurrences
        term_counts = defaultdict(int)
        for related in self.related_terms[term]:
            term_counts[related['term']] += 1
            
        # Calculate scores
        total_occurrences = sum(term_counts.values())
        related_terms = []
        
        for related_term, count in term_counts.items():
            score = (count / total_occurrences) * 100
            related_terms.append({
                'term': related_term,
                'score': score,
                'occurrences': count
            })
            
        # Sort by score and limit results
        return sorted(related_terms, key=lambda x: x['score'], reverse=True)[:limit]
        
    def detect_trending_terms(self, window: str = '1D', min_volume: int = 100) -> List[Dict[str, Any]]:
        """
        Detect trending search terms.
        
        Args:
            window: Time window for analysis
            min_volume: Minimum search volume threshold
            
        Returns:
            List of trending terms with metrics
        """
        trending_terms = []
        
        for term, data in self.search_data.items():
            df = pd.DataFrame(data)
            if df.empty:
                continue
                
            # Convert timestamp and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Calculate metrics
            current_volume = df['volume'].iloc[-1]
            if current_volume < min_volume:
                continue
                
            # Calculate growth rate
            resampled = df.resample(window)['volume'].mean()
            if len(resampled) > 1:
                growth = (resampled.iloc[-1] - resampled.iloc[-2]) / resampled.iloc[-2] * 100
            else:
                growth = 0
                
            # Calculate trend score
            trend_score = self.calculate_trend_score(term, window)
            
            trending_terms.append({
                'term': term,
                'current_volume': current_volume,
                'growth_rate': growth,
                'trend_score': trend_score,
                'regions': df['region'].unique().tolist()
            })
            
        # Sort by trend score
        return sorted(trending_terms, key=lambda x: x['trend_score'], reverse=True)
        
    def _calculate_correlation_score(self, term: str, window: str) -> float:
        """
        Calculate correlation score between search volume and price.
        
        Args:
            term: Search term to analyze
            window: Time window for analysis
            
        Returns:
            Correlation score between 0 and 100
        """
        # This is a placeholder for actual correlation calculation
        # In a real implementation, this would compare search volume
        # with price data to determine correlation strength
        
        return 50.0  # Default neutral score
        
    def get_regional_trends(self, region: str = 'global') -> Dict[str, Any]:
        """
        Get search trend analysis for a specific region.
        
        Args:
            region: Region to analyze
            
        Returns:
            Dictionary with regional trend analysis
        """
        regional_data = []
        
        for term, data in self.search_data.items():
            df = pd.DataFrame(data)
            if df.empty:
                continue
                
            # Filter by region
            regional_df = df[df['region'] == region]
            if regional_df.empty:
                continue
                
            # Calculate metrics
            current_volume = regional_df['volume'].iloc[-1]
            avg_volume = regional_df['volume'].mean()
            
            regional_data.append({
                'term': term,
                'current_volume': current_volume,
                'relative_volume': current_volume / avg_volume if avg_volume > 0 else 0,
                'trend_score': self.calculate_trend_score(term)
            })
            
        # Sort by trend score
        regional_data.sort(key=lambda x: x['trend_score'], reverse=True)
        
        return {
            'region': region,
            'top_terms': regional_data[:10],
            'total_volume': sum(item['current_volume'] for item in regional_data)
        }
