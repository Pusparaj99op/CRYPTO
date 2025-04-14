import pandas as pd
import numpy as np
import re
from typing import List, Dict, Union, Optional, Tuple
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')


class CryptoSentimentAnalyzer:
    """
    Advanced sentiment analysis for cryptocurrency text data
    """
    
    def __init__(self, 
                 model_name: str = 'finiteautomata/bertweet-base-sentiment-analysis',
                 use_gpu: bool = torch.cuda.is_available(),
                 crypto_lexicon: Optional[Dict[str, float]] = None):
        """
        Initialize the sentiment analyzer
        
        Args:
            model_name (str): Name of the pre-trained model to use
            use_gpu (bool): Whether to use GPU acceleration
            crypto_lexicon (Dict): Custom cryptocurrency sentiment lexicon
        """
        self.sid = SentimentIntensityAnalyzer()
        self.transformer_loaded = False
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Load transformer model if available
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            device = 0 if self.use_gpu else -1
            self.sentiment_pipeline = pipeline("sentiment-analysis", 
                                               model=model, 
                                               tokenizer=tokenizer,
                                               device=device)
            self.transformer_loaded = True
        except Exception as e:
            print(f"Transformer model could not be loaded: {e}")
            print("Falling back to VADER sentiment analysis")
        
        # Add custom crypto terms to VADER lexicon
        if crypto_lexicon:
            self.sid.lexicon.update(crypto_lexicon)
        else:
            # Default crypto lexicon
            default_crypto_lexicon = {
                # Bullish terms
                'hodl': 2.0,
                'mooning': 3.0, 
                'bullish': 2.5,
                'lambo': 2.0,
                'fomo': 1.0,
                'staking': 1.0,
                'yield': 1.5,
                'accumulate': 1.0,
                'stake': 1.0,
                'consensus': 0.5,
                'mainnet': 1.0,
                'protocol': 0.5,
                'defi': 0.5,
                'nft': 0.5,
                
                # Bearish terms
                'rekt': -2.5,
                'rug': -3.0,
                'rugpull': -3.0,
                'shitcoin': -2.5,
                'dump': -2.0,
                'dumping': -2.0,
                'bearish': -2.5,
                'ponzi': -3.0,
                'scam': -3.0,
                'scamcoin': -3.0,
                'shilling': -1.5,
                'fud': -2.0,
                'selloff': -2.0,
                'correction': -1.5,
                'liquidated': -2.0,
                'delist': -2.5
            }
            self.sid.lexicon.update(default_crypto_lexicon)
    
    def preprocess_text(self, text: str, 
                        remove_urls: bool = True,
                        remove_mentions: bool = True,
                        remove_hashtags: bool = False,
                        lowercase: bool = True,
                        remove_punctuation: bool = True,
                        remove_stopwords: bool = False) -> str:
        """
        Preprocess text for sentiment analysis
        
        Args:
            text (str): Input text
            remove_urls (bool): Whether to remove URLs
            remove_mentions (bool): Whether to remove mentions (@username)
            remove_hashtags (bool): Whether to remove hashtags (#tag)
            lowercase (bool): Whether to convert text to lowercase
            remove_punctuation (bool): Whether to remove punctuation
            remove_stopwords (bool): Whether to remove stopwords
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        if lowercase:
            text = text.lower()
        
        # Remove URLs
        if remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove mentions
        if remove_mentions:
            text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        if remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        
        # Remove punctuation
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Remove stopwords
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            tokens = word_tokenize(text)
            text = ' '.join([word for word in tokens if word not in stop_words])
        
        return text
    
    def analyze_with_vader(self, text: str, preprocess: bool = True) -> Dict[str, Union[float, str]]:
        """
        Analyze sentiment using VADER
        
        Args:
            text (str): Input text
            preprocess (bool): Whether to preprocess the text
            
        Returns:
            Dict: Sentiment analysis results
        """
        if preprocess:
            text = self.preprocess_text(text)
        
        if not text:
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 1.0,
                'neg': 0.0,
                'sentiment': 'neutral'
            }
        
        # Get VADER scores
        scores = self.sid.polarity_scores(text)
        
        # Determine sentiment label
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        scores['sentiment'] = sentiment
        
        return scores
    
    def analyze_with_transformer(self, text: str, preprocess: bool = True) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment using transformer model
        
        Args:
            text (str): Input text
            preprocess (bool): Whether to preprocess the text
            
        Returns:
            Dict: Sentiment analysis results
        """
        if not self.transformer_loaded:
            return self.analyze_with_vader(text, preprocess)
        
        if preprocess:
            text = self.preprocess_text(text, remove_stopwords=False, remove_punctuation=False)
        
        if not text:
            return {
                'label': 'neutral',
                'score': 0.5
            }
        
        # Analyze with transformer
        try:
            result = self.sentiment_pipeline(text)[0]
            return result
        except Exception as e:
            print(f"Error analyzing with transformer: {e}")
            # Fall back to VADER
            return self.analyze_with_vader(text, False)
    
    def analyze_sentiment(self, text: str, 
                          method: str = 'auto', 
                          preprocess: bool = True) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment using the specified method
        
        Args:
            text (str): Input text
            method (str): Method to use ('vader', 'transformer', or 'auto')
            preprocess (bool): Whether to preprocess the text
            
        Returns:
            Dict: Sentiment analysis results
        """
        if method == 'vader' or (method == 'auto' and not self.transformer_loaded):
            return self.analyze_with_vader(text, preprocess)
        elif method == 'transformer' or method == 'auto':
            return self.analyze_with_transformer(text, preprocess)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def analyze_batch(self, 
                      texts: List[str], 
                      method: str = 'auto', 
                      preprocess: bool = True, 
                      batch_size: int = 16) -> List[Dict[str, Union[str, float]]]:
        """
        Analyze sentiment for a batch of texts
        
        Args:
            texts (List[str]): List of texts to analyze
            method (str): Method to use ('vader', 'transformer', or 'auto')
            preprocess (bool): Whether to preprocess the texts
            batch_size (int): Batch size for transformer analysis
            
        Returns:
            List[Dict]: List of sentiment analysis results
        """
        results = []
        
        if method == 'vader' or (method == 'auto' and not self.transformer_loaded):
            for text in texts:
                results.append(self.analyze_with_vader(text, preprocess))
        elif method == 'transformer' or method == 'auto':
            # Preprocess if needed
            if preprocess:
                processed_texts = [self.preprocess_text(text, remove_stopwords=False, remove_punctuation=False) for text in texts]
            else:
                processed_texts = texts
            
            # Process in batches
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i+batch_size]
                try:
                    batch_results = self.sentiment_pipeline(batch)
                    results.extend(batch_results)
                except Exception as e:
                    print(f"Error analyzing batch with transformer: {e}")
                    # Fall back to VADER for this batch
                    for text in batch:
                        results.append(self.analyze_with_vader(text, False))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return results
    
    def analyze_dataframe(self, 
                          df: pd.DataFrame, 
                          text_column: str, 
                          method: str = 'auto',
                          preprocess: bool = True,
                          batch_size: int = 16) -> pd.DataFrame:
        """
        Analyze sentiment for texts in a DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame containing text data
            text_column (str): Name of the column containing text
            method (str): Method to use ('vader', 'transformer', or 'auto')
            preprocess (bool): Whether to preprocess the texts
            batch_size (int): Batch size for transformer analysis
            
        Returns:
            pd.DataFrame: DataFrame with sentiment analysis results added as columns
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        # Get texts from DataFrame
        texts = df[text_column].tolist()
        
        # Analyze sentiment
        results = self.analyze_batch(texts, method, preprocess, batch_size)
        
        # Create result DataFrame
        if method == 'vader' or (method == 'auto' and not self.transformer_loaded):
            result_df = pd.DataFrame(results)
            df_with_sentiment = pd.concat([df.reset_index(drop=True), 
                                          result_df[['compound', 'pos', 'neu', 'neg', 'sentiment']]], 
                                          axis=1)
        else:
            result_df = pd.DataFrame(results)
            df_with_sentiment = pd.concat([df.reset_index(drop=True), 
                                          result_df[['label', 'score']].rename(
                                              columns={'label': 'sentiment', 'score': 'confidence'})], 
                                          axis=1)
        
        return df_with_sentiment
    
    def extract_influential_terms(self, 
                                  text: str, 
                                  preprocess: bool = True,
                                  method: str = 'lexicon',
                                  top_n: int = 10) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        """
        Extract terms that influence sentiment the most
        
        Args:
            text (str): Input text
            preprocess (bool): Whether to preprocess the text
            method (str): Method to use ('lexicon' or 'tfidf')
            top_n (int): Number of top terms to return
            
        Returns:
            Dict: Dictionary with positive and negative influential terms
        """
        if preprocess:
            text = self.preprocess_text(text, remove_stopwords=False)
        
        if not text:
            return {'positive': [], 'negative': []}
        
        if method == 'lexicon':
            # Tokenize
            tokens = word_tokenize(text.lower())
            
            # Score each token
            token_scores = []
            for token in tokens:
                if token in self.sid.lexicon:
                    token_scores.append({
                        'term': token,
                        'score': self.sid.lexicon[token]
                    })
            
            # Sort by absolute score
            token_scores.sort(key=lambda x: abs(x['score']), reverse=True)
            
            # Split into positive and negative
            positive = [term for term in token_scores if term['score'] > 0][:top_n]
            negative = [term for term in token_scores if term['score'] < 0][:top_n]
            
            return {
                'positive': positive,
                'negative': negative
            }
        
        elif method == 'tfidf':
            # This would require a trained TF-IDF model with sentiment labels
            # Simplified implementation:
            # For now, return lexicon-based results
            return self.extract_influential_terms(text, False, 'lexicon', top_n)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def analyze_sentiment_trend(self, 
                               df: pd.DataFrame, 
                               text_column: str,
                               date_column: str,
                               method: str = 'auto',
                               preprocess: bool = True,
                               batch_size: int = 16,
                               freq: str = 'D',
                               window_size: int = 7) -> pd.DataFrame:
        """
        Analyze sentiment trend over time
        
        Args:
            df (pd.DataFrame): DataFrame containing text data
            text_column (str): Name of the column containing text
            date_column (str): Name of the column containing dates
            method (str): Method to use ('vader', 'transformer', or 'auto')
            preprocess (bool): Whether to preprocess the texts
            batch_size (int): Batch size for transformer analysis
            freq (str): Frequency for grouping ('D' = daily, 'W' = weekly, etc.)
            window_size (int): Window size for rolling average
            
        Returns:
            pd.DataFrame: DataFrame with sentiment trend data
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        if date_column not in df.columns:
            raise ValueError(f"Column '{date_column}' not found in DataFrame")
        
        # Ensure date column is datetime
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Analyze sentiment
        df_with_sentiment = self.analyze_dataframe(df, text_column, method, preprocess, batch_size)
        
        # Group by date
        if method == 'vader' or (method == 'auto' and not self.transformer_loaded):
            sentiment_col = 'compound'
        else:
            # For transformer models, convert label to numeric
            df_with_sentiment['sentiment_value'] = df_with_sentiment['sentiment'].map(
                {'positive': 1, 'neutral': 0, 'negative': -1})
            sentiment_col = 'sentiment_value'
            
        # Get sentiment by date
        sentiment_by_date = df_with_sentiment.groupby(
            pd.Grouper(key=date_column, freq=freq)
        ).agg({
            sentiment_col: ['mean', 'std', 'count'],
            'sentiment': lambda x: (x == 'positive').mean()
        })
        
        # Flatten columns
        sentiment_by_date.columns = ['sentiment_mean', 'sentiment_std', 'count', 'positive_ratio']
        
        # Add rolling metrics
        sentiment_by_date['sentiment_rolling_mean'] = sentiment_by_date['sentiment_mean'].rolling(window=window_size).mean()
        sentiment_by_date['positive_ratio_rolling'] = sentiment_by_date['positive_ratio'].rolling(window=window_size).mean()
        
        return sentiment_by_date
    
    def plot_sentiment_trend(self, 
                           trend_data: pd.DataFrame,
                           plot_volume: bool = True,
                           figsize: Tuple[int, int] = (12, 6),
                           title: str = 'Sentiment Trend Over Time') -> plt.Figure:
        """
        Plot sentiment trend
        
        Args:
            trend_data (pd.DataFrame): DataFrame with sentiment trend data from analyze_sentiment_trend
            plot_volume (bool): Whether to plot volume (count)
            figsize (Tuple[int, int]): Figure size
            title (str): Plot title
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Plot sentiment trend
        ax1.plot(trend_data.index, trend_data['sentiment_mean'], 'b-', label='Sentiment')
        ax1.plot(trend_data.index, trend_data['sentiment_rolling_mean'], 'r-', label='Rolling Average')
        ax1.fill_between(trend_data.index, 
                         trend_data['sentiment_mean'] - trend_data['sentiment_std'],
                         trend_data['sentiment_mean'] + trend_data['sentiment_std'],
                         alpha=0.2)
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sentiment', color='b')
        ax1.tick_params('y', colors='b')
        
        # Add volume if requested
        if plot_volume and 'count' in trend_data.columns:
            ax2 = ax1.twinx()
            ax2.bar(trend_data.index, trend_data['count'], alpha=0.3, color='g', label='Volume')
            ax2.set_ylabel('Volume', color='g')
            ax2.tick_params('y', colors='g')
        
        # Add positive ratio on the same axis
        ax1.plot(trend_data.index, trend_data['positive_ratio'], 'g--', label='Positive Ratio')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        if plot_volume and 'count' in trend_data.columns:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax1.legend(loc='upper left')
        
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        return fig
    
    def compare_entity_sentiment(self,
                               df: pd.DataFrame,
                               text_column: str,
                               entity_column: str,
                               method: str = 'auto',
                               preprocess: bool = True,
                               batch_size: int = 16,
                               min_count: int = 5) -> pd.DataFrame:
        """
        Compare sentiment across different entities
        
        Args:
            df (pd.DataFrame): DataFrame containing text data
            text_column (str): Name of the column containing text
            entity_column (str): Name of the column containing entity labels
            method (str): Method to use ('vader', 'transformer', or 'auto')
            preprocess (bool): Whether to preprocess the texts
            batch_size (int): Batch size for transformer analysis
            min_count (int): Minimum number of mentions to include entity
            
        Returns:
            pd.DataFrame: DataFrame with entity sentiment comparison
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        if entity_column not in df.columns:
            raise ValueError(f"Column '{entity_column}' not found in DataFrame")
        
        # Analyze sentiment
        df_with_sentiment = self.analyze_dataframe(df, text_column, method, preprocess, batch_size)
        
        # Group by entity
        if method == 'vader' or (method == 'auto' and not self.transformer_loaded):
            sentiment_col = 'compound'
        else:
            # For transformer models, convert label to numeric
            df_with_sentiment['sentiment_value'] = df_with_sentiment['sentiment'].map(
                {'positive': 1, 'neutral': 0, 'negative': -1})
            sentiment_col = 'sentiment_value'
            
        # Get sentiment by entity
        entity_sentiment = df_with_sentiment.groupby(entity_column).agg({
            sentiment_col: ['mean', 'std', 'count'],
            'sentiment': lambda x: (x == 'positive').mean()
        })
        
        # Flatten columns
        entity_sentiment.columns = ['sentiment_mean', 'sentiment_std', 'count', 'positive_ratio']
        
        # Filter by minimum count
        entity_sentiment = entity_sentiment[entity_sentiment['count'] >= min_count]
        
        # Sort by sentiment score
        entity_sentiment = entity_sentiment.sort_values('sentiment_mean', ascending=False)
        
        return entity_sentiment 