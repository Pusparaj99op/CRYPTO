import re
import nltk
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
from collections import Counter, defaultdict
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import networkx as nx
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import logging

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)

class CryptoChatSummarizer:
    """
    Class for summarizing crypto-related chats, conversations, and discussions
    """
    
    def __init__(self, 
                 model_name: str = "facebook/bart-large-cnn",
                 custom_tokens: Optional[List[str]] = None,
                 device: Optional[str] = None):
        """
        Initialize the chat summarizer
        
        Args:
            model_name (str): Name of the pre-trained model to use for summarization
            custom_tokens (List[str], optional): Custom tokens to add to tokenizer
            device (str, optional): Device to use for computation ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.custom_tokens = custom_tokens or []
        
        # Add crypto-specific terms to custom tokens if not provided
        crypto_terms = ["Bitcoin", "BTC", "Ethereum", "ETH", "crypto", "blockchain", 
                       "altcoin", "token", "wallet", "exchange", "DeFi", "NFT",
                       "mining", "stake", "yield", "liquidity", "APY", "APR"]
        
        for term in crypto_terms:
            if term not in self.custom_tokens:
                self.custom_tokens.append(term)
                
        # Initialize tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Add custom tokens if any
            if self.custom_tokens:
                num_added_tokens = self.tokenizer.add_tokens(self.custom_tokens)
                self.model.resize_token_embeddings(len(self.tokenizer))
                logger.info(f"Added {num_added_tokens} custom tokens to tokenizer")
            
            # Create summarization pipeline
            self.summarizer = pipeline(
                "summarization", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" else -1
            )
            logger.info(f"Summarization model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading summarization model: {e}")
            # Fallback to extractive summarization
            self.summarizer = None
            logger.warning("Falling back to extractive summarization methods")
            
        # Set up stopwords
        self.stopwords = set(stopwords.words('english'))
        self.stopwords.update(['im', 'u', 'ur', 'r', 'n', 'ill', 'id', 'dont', 'ive'])
    
    def clean_chat_text(self, text: str) -> str:
        """
        Clean and preprocess chat text
        
        Args:
            text (str): Chat text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove user mentions (e.g., @username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags symbol but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep periods, question marks, and exclamation points
        text = re.sub(r'[^\w\s.?!]', '', text)
        
        return text
    
    def preprocess_chat_messages(self, 
                               messages: List[Dict[str, Any]],
                               message_key: str = 'text',
                               username_key: Optional[str] = 'username',
                               timestamp_key: Optional[str] = 'timestamp',
                               min_message_length: int = 10) -> List[Dict[str, Any]]:
        """
        Preprocess a list of chat messages
        
        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries
            message_key (str): Key for message text in the dictionary
            username_key (str, optional): Key for username in the dictionary
            timestamp_key (str, optional): Key for timestamp in the dictionary
            min_message_length (int): Minimum length of message to include
            
        Returns:
            List[Dict[str, Any]]: Preprocessed message dictionaries
        """
        processed_messages = []
        
        for msg in messages:
            if message_key not in msg:
                continue
                
            text = msg[message_key]
            
            # Skip empty or very short messages
            if not text or len(text) < min_message_length:
                continue
                
            # Clean text
            cleaned_text = self.clean_chat_text(text)
            
            # Create processed message
            processed_msg = {message_key: cleaned_text}
            
            # Include username if available
            if username_key and username_key in msg:
                processed_msg[username_key] = msg[username_key]
                
            # Include timestamp if available
            if timestamp_key and timestamp_key in msg:
                processed_msg[timestamp_key] = msg[timestamp_key]
                
            processed_messages.append(processed_msg)
            
        return processed_messages
    
    def group_messages_by_conversation(self,
                                     messages: List[Dict[str, Any]],
                                     message_key: str = 'text',
                                     timestamp_key: Optional[str] = 'timestamp',
                                     max_time_gap: int = 3600) -> List[List[Dict[str, Any]]]:
        """
        Group messages into conversations based on time gaps
        
        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries
            message_key (str): Key for message text in the dictionary
            timestamp_key (str, optional): Key for timestamp in the dictionary
            max_time_gap (int): Maximum time gap in seconds to consider part of same conversation
            
        Returns:
            List[List[Dict[str, Any]]]: List of conversations, each containing message dictionaries
        """
        if not messages:
            return []
            
        # Sort messages by timestamp if available
        if timestamp_key:
            try:
                sorted_messages = sorted(messages, key=lambda x: x[timestamp_key])
            except (KeyError, TypeError):
                logger.warning("Could not sort messages by timestamp, using original order")
                sorted_messages = messages
        else:
            sorted_messages = messages
            
        conversations = []
        current_conversation = [sorted_messages[0]]
        
        for i in range(1, len(sorted_messages)):
            current_msg = sorted_messages[i]
            prev_msg = sorted_messages[i-1]
            
            # Check if messages should be part of the same conversation
            if timestamp_key and timestamp_key in current_msg and timestamp_key in prev_msg:
                time_diff = (current_msg[timestamp_key] - prev_msg[timestamp_key]).total_seconds() \
                    if isinstance(prev_msg[timestamp_key], datetime.datetime) \
                    else current_msg[timestamp_key] - prev_msg[timestamp_key]
                
                if time_diff > max_time_gap:
                    # Start a new conversation
                    conversations.append(current_conversation)
                    current_conversation = [current_msg]
                else:
                    # Add to current conversation
                    current_conversation.append(current_msg)
            else:
                # If we can't use timestamps, just add to current conversation
                current_conversation.append(current_msg)
                
        # Add the last conversation
        if current_conversation:
            conversations.append(current_conversation)
            
        return conversations
    
    def conversation_to_text(self,
                          conversation: List[Dict[str, Any]],
                          message_key: str = 'text',
                          username_key: Optional[str] = 'username',
                          include_usernames: bool = True) -> str:
        """
        Convert a conversation to a single text for summarization
        
        Args:
            conversation (List[Dict[str, Any]]): List of message dictionaries in a conversation
            message_key (str): Key for message text in the dictionary
            username_key (str, optional): Key for username in the dictionary
            include_usernames (bool): Whether to include usernames in the concatenated text
            
        Returns:
            str: Concatenated conversation text
        """
        text_parts = []
        
        for msg in conversation:
            if message_key not in msg:
                continue
                
            if include_usernames and username_key and username_key in msg:
                text_parts.append(f"{msg[username_key]}: {msg[message_key]}")
            else:
                text_parts.append(msg[message_key])
                
        return " ".join(text_parts)
    
    def extract_key_sentences(self, text: str, top_n: int = 3) -> List[str]:
        """
        Extract key sentences from text using TextRank-like algorithm
        
        Args:
            text (str): Input text
            top_n (int): Number of key sentences to extract
            
        Returns:
            List[str]: List of extracted key sentences
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        # Return if there are fewer sentences than requested
        if len(sentences) <= top_n:
            return sentences
            
        # Clean sentences
        clean_sentences = [re.sub(r'[^\w\s]', '', s.lower()) for s in sentences]
        
        # Create sentence vectors using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            sentence_vectors = vectorizer.fit_transform(clean_sentences)
        except ValueError:
            # If vectorization fails, return first top_n sentences
            return sentences[:top_n]
            
        # Calculate sentence similarity matrix
        similarity_matrix = (sentence_vectors * sentence_vectors.T).toarray()
        
        # Create graph and apply PageRank
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        # Get top N sentences
        ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
        top_sentences = [s for _, i, s in ranked_sentences[:top_n]]
        
        # Sort by original order
        top_sentence_indices = [i for _, i, _ in ranked_sentences[:top_n]]
        ordered_sentences = [(i, s) for i, s in zip(top_sentence_indices, top_sentences)]
        ordered_sentences.sort(key=lambda x: x[0])
        
        return [s for _, s in ordered_sentences]
    
    def extract_key_phrases(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract key phrases from text
        
        Args:
            text (str): Input text
            top_n (int): Number of key phrases to extract
            
        Returns:
            List[Tuple[str, float]]: List of key phrases with their scores
        """
        # Tokenize text
        words = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic words
        filtered_words = [w for w in words if w.isalpha() and w not in self.stopwords]
        
        # Extract bigrams and trigrams
        bigrams = list(nltk.bigrams(filtered_words))
        trigrams = list(nltk.trigrams(filtered_words))
        
        # Convert to strings
        bigram_phrases = [' '.join(bg) for bg in bigrams]
        trigram_phrases = [' '.join(tg) for tg in trigrams]
        
        # Combine unigrams, bigrams, and trigrams
        all_phrases = filtered_words + bigram_phrases + trigram_phrases
        
        # Count phrase frequencies
        phrase_counts = Counter(all_phrases)
        
        # Get top phrases
        top_phrases = phrase_counts.most_common(top_n * 2)  # Get more than needed initially
        
        # Filter out subphrases if they have lower counts
        filtered_phrases = []
        for phrase, count in top_phrases:
            is_subphrase = False
            for other_phrase, other_count in top_phrases:
                if phrase != other_phrase and phrase in other_phrase and count <= other_count:
                    is_subphrase = True
                    break
            if not is_subphrase:
                filtered_phrases.append((phrase, count))
        
        # Calculate scores (normalized frequencies)
        total_count = sum(count for _, count in filtered_phrases)
        scored_phrases = [(phrase, count / total_count) for phrase, count in filtered_phrases]
        
        return sorted(scored_phrases, key=lambda x: x[1], reverse=True)[:top_n]
    
    def summarize_text(self, 
                     text: str, 
                     max_length: int = 130, 
                     min_length: int = 30) -> str:
        """
        Summarize text using the loaded model
        
        Args:
            text (str): Text to summarize
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary
            
        Returns:
            str: Generated summary
        """
        # Check if text is too short for summarization
        if len(text.split()) < min_length:
            return text
            
        if self.summarizer:
            # Use transformer model for summarization
            try:
                summary = self.summarizer(
                    text, 
                    max_length=max_length, 
                    min_length=min_length, 
                    do_sample=False
                )
                return summary[0]['summary_text']
            except Exception as e:
                logger.error(f"Error in model summarization: {e}")
                # Fall back to extractive summarization if model fails
                pass
                
        # Extractive summarization as fallback
        key_sentences = self.extract_key_sentences(text, top_n=3)
        return " ".join(key_sentences)
    
    def summarize_conversation(self,
                             conversation: List[Dict[str, Any]],
                             message_key: str = 'text',
                             username_key: Optional[str] = 'username',
                             include_usernames: bool = True,
                             max_length: int = 130,
                             min_length: int = 30) -> Dict[str, Any]:
        """
        Summarize a conversation
        
        Args:
            conversation (List[Dict[str, Any]]): List of message dictionaries in a conversation
            message_key (str): Key for message text in the dictionary
            username_key (str, optional): Key for username in the dictionary
            include_usernames (bool): Whether to include usernames in the text
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary
            
        Returns:
            Dict[str, Any]: Summary information, including text summary, key phrases, participant info
        """
        # Convert conversation to text
        conversation_text = self.conversation_to_text(
            conversation, 
            message_key=message_key,
            username_key=username_key,
            include_usernames=include_usernames
        )
        
        # Generate summary
        summary_text = self.summarize_text(
            conversation_text,
            max_length=max_length,
            min_length=min_length
        )
        
        # Extract key phrases
        key_phrases = self.extract_key_phrases(conversation_text, top_n=5)
        
        # Extract participant information
        participants = set()
        message_counts = defaultdict(int)
        
        if username_key:
            for msg in conversation:
                if username_key in msg:
                    username = msg[username_key]
                    participants.add(username)
                    message_counts[username] += 1
        
        # Get start and end times if available
        timestamp_key = 'timestamp'
        start_time = min([msg.get(timestamp_key) for msg in conversation if timestamp_key in msg], default=None)
        end_time = max([msg.get(timestamp_key) for msg in conversation if timestamp_key in msg], default=None)
        
        # Prepare result
        result = {
            'summary': summary_text,
            'key_phrases': key_phrases,
            'message_count': len(conversation),
            'participants': list(participants),
            'participant_message_counts': dict(message_counts),
            'start_time': start_time,
            'end_time': end_time
        }
        
        return result
    
    def summarize_multiple_conversations(self,
                                       conversations: List[List[Dict[str, Any]]],
                                       message_key: str = 'text',
                                       username_key: Optional[str] = 'username',
                                       include_usernames: bool = True,
                                       max_length: int = 130,
                                       min_length: int = 30) -> List[Dict[str, Any]]:
        """
        Summarize multiple conversations
        
        Args:
            conversations (List[List[Dict[str, Any]]]): List of conversations
            message_key (str): Key for message text in the dictionary
            username_key (str, optional): Key for username in the dictionary
            include_usernames (bool): Whether to include usernames in the text
            max_length (int): Maximum length of each summary
            min_length (int): Minimum length of each summary
            
        Returns:
            List[Dict[str, Any]]: List of summaries for each conversation
        """
        summaries = []
        
        for conversation in conversations:
            summary = self.summarize_conversation(
                conversation,
                message_key=message_key,
                username_key=username_key,
                include_usernames=include_usernames,
                max_length=max_length,
                min_length=min_length
            )
            summaries.append(summary)
            
        return summaries
    
    def create_topic_wordcloud(self, 
                             text: str,
                             title: str = "Topic Word Cloud",
                             width: int = 800, 
                             height: int = 400) -> plt.Figure:
        """
        Create a word cloud from text
        
        Args:
            text (str): Text to create word cloud from
            title (str): Title for the word cloud
            width (int): Width of the word cloud
            height (int): Height of the word cloud
            
        Returns:
            plt.Figure: Matplotlib figure containing the word cloud
        """
        # Generate word cloud
        wordcloud = WordCloud(
            width=width, 
            height=height,
            background_color='white',
            stopwords=self.stopwords,
            collocations=False,
            max_words=100
        ).generate(text)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        
        return fig
    
    def analyze_conversation_trend(self,
                                 conversations: List[List[Dict[str, Any]]],
                                 message_key: str = 'text',
                                 timestamp_key: str = 'timestamp',
                                 time_format: str = '%Y-%m-%d') -> Tuple[plt.Figure, pd.DataFrame]:
        """
        Analyze conversation trends over time
        
        Args:
            conversations (List[List[Dict[str, Any]]]): List of conversations
            message_key (str): Key for message text
            timestamp_key (str): Key for timestamp
            time_format (str): Time format for grouping and display
            
        Returns:
            Tuple[plt.Figure, pd.DataFrame]: Matplotlib figure and DataFrame with conversation data
        """
        # Check if we have timestamps
        has_timestamps = all(
            timestamp_key in conversation[0] for conversation in conversations if conversation
        )
        
        if not has_timestamps:
            raise ValueError(f"Timestamp key '{timestamp_key}' not found in conversations")
            
        # Extract data for each conversation
        conversation_data = []
        
        for i, conversation in enumerate(conversations):
            if not conversation:
                continue
                
            # Get conversation stats
            start_time = min(msg.get(timestamp_key) for msg in conversation if timestamp_key in msg)
            message_count = len(conversation)
            avg_message_length = np.mean([len(msg.get(message_key, '')) for msg in conversation])
            
            # Extract summary and key phrases
            summary = self.summarize_conversation(conversation, message_key, timestamp_key)
            
            conversation_data.append({
                'conversation_id': i,
                'date': start_time.strftime(time_format) if isinstance(start_time, datetime.datetime) else start_time,
                'timestamp': start_time,
                'message_count': message_count,
                'avg_message_length': avg_message_length,
                'summary': summary.get('summary', ''),
                'key_phrases': summary.get('key_phrases', [])
            })
            
        # Create DataFrame
        df = pd.DataFrame(conversation_data)
        
        # Group by date and count conversations
        if 'date' in df.columns:
            daily_counts = df.groupby('date').size().reset_index(name='conversation_count')
            daily_message_counts = df.groupby('date')['message_count'].sum().reset_index()
            
            # Merge the data
            trend_data = pd.merge(daily_counts, daily_message_counts, on='date')
            
            # Create plot
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot conversation count
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Conversation Count', color='tab:blue')
            ax1.plot(trend_data['date'], trend_data['conversation_count'], 'o-', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            
            # Create second y-axis
            ax2 = ax1.twinx()
            ax2.set_ylabel('Total Messages', color='tab:red')
            ax2.plot(trend_data['date'], trend_data['message_count'], 'o-', color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            
            # Rotate date labels
            plt.xticks(rotation=45)
            
            fig.tight_layout()
            plt.title('Conversation Trends Over Time')
            
            return fig, df
        else:
            # If we don't have dates, just return the DataFrame
            return None, df
    
    def extract_crypto_mentions(self, 
                              text: str, 
                              crypto_list: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Extract cryptocurrency mentions from text
        
        Args:
            text (str): Text to analyze
            crypto_list (List[str], optional): List of cryptocurrencies to look for
            
        Returns:
            Dict[str, int]: Dictionary with cryptocurrency mentions and counts
        """
        if crypto_list is None:
            # Default list of common cryptocurrencies
            crypto_list = [
                "Bitcoin", "BTC", "Ethereum", "ETH", "Binance Coin", "BNB",
                "Cardano", "ADA", "Solana", "SOL", "XRP", "Polkadot", "DOT",
                "Dogecoin", "DOGE", "Avalanche", "AVAX", "Shiba Inu", "SHIB",
                "USDT", "Tether", "USD Coin", "USDC", "Terra", "LUNA"
            ]
            
        # Create regex pattern for whole word matching
        pattern = r'\b(' + '|'.join(re.escape(crypto) for crypto in crypto_list) + r')\b'
        
        # Find all matches
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        # Count occurrences
        counts = {}
        for match in matches:
            match_upper = match.upper()
            # Standardize some tokens
            if match_upper == "BITCOIN":
                match_upper = "BTC"
            elif match_upper == "ETHEREUM":
                match_upper = "ETH"
                
            if match_upper in counts:
                counts[match_upper] += 1
            else:
                counts[match_upper] = 1
                
        return counts
    
    def analyze_crypto_mentions(self,
                              conversations: List[List[Dict[str, Any]]],
                              message_key: str = 'text',
                              crypto_list: Optional[List[str]] = None) -> Tuple[plt.Figure, Dict[str, int]]:
        """
        Analyze cryptocurrency mentions across all conversations
        
        Args:
            conversations (List[List[Dict[str, Any]]]): List of conversations
            message_key (str): Key for message text
            crypto_list (List[str], optional): List of cryptocurrencies to look for
            
        Returns:
            Tuple[plt.Figure, Dict[str, int]]: Matplotlib figure and dictionary with crypto mentions
        """
        # Combine all messages
        all_text = ""
        for conversation in conversations:
            for msg in conversation:
                if message_key in msg:
                    all_text += " " + msg[message_key]
                    
        # Extract crypto mentions
        crypto_mentions = self.extract_crypto_mentions(all_text, crypto_list)
        
        # Sort by count
        sorted_mentions = dict(sorted(crypto_mentions.items(), key=lambda x: x[1], reverse=True))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if sorted_mentions:
            cryptos = list(sorted_mentions.keys())
            counts = list(sorted_mentions.values())
            
            # Use only top 10 for visibility
            if len(cryptos) > 10:
                cryptos = cryptos[:10]
                counts = counts[:10]
                
            bars = ax.bar(cryptos, counts)
            
            # Add count labels on top of bars
            for i, bar in enumerate(bars):
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.1,
                    str(counts[i]),
                    ha='center'
                )
                
            ax.set_xlabel('Cryptocurrency')
            ax.set_ylabel('Mention Count')
            ax.set_title('Cryptocurrency Mentions in Conversations')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
        return fig, sorted_mentions
    
    def save_summaries_to_dataframe(self,
                                  summaries: List[Dict[str, Any]],
                                  output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Save conversation summaries to a DataFrame and optionally to a file
        
        Args:
            summaries (List[Dict[str, Any]]): List of conversation summaries
            output_file (str, optional): Path to save DataFrame (CSV or Excel)
            
        Returns:
            pd.DataFrame: DataFrame with conversation summaries
        """
        # Create DataFrame
        df = pd.DataFrame()
        
        if not summaries:
            return df
            
        # Extract common fields
        common_fields = ['summary', 'message_count', 'start_time', 'end_time']
        for field in common_fields:
            if all(field in summary for summary in summaries):
                df[field] = [summary[field] for summary in summaries]
                
        # Extract key phrases
        if all('key_phrases' in summary for summary in summaries):
            df['key_phrases'] = [
                ', '.join([phrase for phrase, _ in summary['key_phrases'][:3]])
                for summary in summaries
            ]
            
            df['top_phrase'] = [
                summary['key_phrases'][0][0] if summary['key_phrases'] else ''
                for summary in summaries
            ]
            
        # Extract participant info
        if all('participants' in summary for summary in summaries):
            df['participant_count'] = [len(summary['participants']) for summary in summaries]
            df['participants'] = [', '.join(summary['participants'][:5]) for summary in summaries]
            
        # Calculate duration if timestamps exist
        if 'start_time' in df.columns and 'end_time' in df.columns:
            # Check if timestamps are datetime objects
            if isinstance(df['start_time'].iloc[0], datetime.datetime):
                df['duration_minutes'] = [(end - start).total_seconds() / 60 
                                       for start, end in zip(df['start_time'], df['end_time'])]
            
        # Save to file if specified
        if output_file:
            if output_file.endswith('.csv'):
                df.to_csv(output_file, index=False)
            elif output_file.endswith('.xlsx'):
                df.to_excel(output_file, index=False)
            else:
                df.to_csv(output_file + '.csv', index=False)
                
        return df
    
    def cluster_conversations_by_topic(self,
                                    conversations: List[List[Dict[str, Any]]],
                                    message_key: str = 'text',
                                    n_clusters: int = 5) -> Dict[str, Any]:
        """
        Cluster conversations by topic
        
        Args:
            conversations (List[List[Dict[str, Any]]]): List of conversations
            message_key (str): Key for message text
            n_clusters (int): Number of clusters to create
            
        Returns:
            Dict[str, Any]: Clustering results with cluster assignments and representative texts
        """
        from sklearn.cluster import KMeans
        
        # Convert conversations to text
        conversation_texts = []
        for conversation in conversations:
            text = self.conversation_to_text(conversation, message_key=message_key)
            conversation_texts.append(text)
            
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            min_df=2,
            max_df=0.7
        )
        
        # Check if we have enough conversations
        if len(conversation_texts) < n_clusters:
            n_clusters = len(conversation_texts)
            
        if n_clusters < 2:
            return {
                'cluster_assignments': [0] * len(conversation_texts),
                'cluster_sizes': [len(conversation_texts)],
                'representative_conversations': conversation_texts[:1] if conversation_texts else [],
                'cluster_summaries': ['All conversations'] if conversation_texts else []
            }
            
        # Create vectors
        try:
            tfidf_matrix = vectorizer.fit_transform(conversation_texts)
            
            # Apply clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_assignments = kmeans.fit_predict(tfidf_matrix)
            
            # Get cluster sizes
            cluster_sizes = [0] * n_clusters
            for cluster_id in cluster_assignments:
                cluster_sizes[cluster_id] += 1
                
            # Find representative conversation for each cluster
            representative_conversations = []
            cluster_summaries = []
            
            for cluster_id in range(n_clusters):
                # Get indices of conversations in this cluster
                indices = [i for i, c in enumerate(cluster_assignments) if c == cluster_id]
                
                if not indices:
                    representative_conversations.append("")
                    cluster_summaries.append(f"Cluster {cluster_id} (empty)")
                    continue
                    
                # Find the conversation closest to cluster center
                cluster_center = kmeans.cluster_centers_[cluster_id]
                distances = []
                
                for idx in indices:
                    distance = np.linalg.norm(tfidf_matrix[idx].toarray() - cluster_center)
                    distances.append((idx, distance))
                    
                # Sort by distance to center
                distances.sort(key=lambda x: x[1])
                representative_idx = distances[0][0]
                
                # Add representative conversation
                representative_conversations.append(conversation_texts[representative_idx])
                
                # Summarize the cluster
                summary = self.summarize_text(conversation_texts[representative_idx])
                cluster_summaries.append(f"Cluster {cluster_id} ({cluster_sizes[cluster_id]} conversations): {summary}")
                
            return {
                'cluster_assignments': cluster_assignments.tolist(),
                'cluster_sizes': cluster_sizes,
                'representative_conversations': representative_conversations,
                'cluster_summaries': cluster_summaries
            }
            
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            return {
                'error': str(e),
                'cluster_assignments': [0] * len(conversation_texts),
                'cluster_sizes': [len(conversation_texts)],
                'representative_conversations': [],
                'cluster_summaries': []
            }
    
    def create_summary_report(self,
                           conversations: List[List[Dict[str, Any]]],
                           message_key: str = 'text',
                           username_key: Optional[str] = 'username',
                           timestamp_key: Optional[str] = 'timestamp',
                           output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a comprehensive summary report of all conversations
        
        Args:
            conversations (List[List[Dict[str, Any]]]): List of conversations
            message_key (str): Key for message text
            username_key (str, optional): Key for username
            timestamp_key (str, optional): Key for timestamp
            output_file (str, optional): Path to save report data (JSON)
            
        Returns:
            Dict[str, Any]: Summary report data
        """
        import json
        
        # Start with basic stats
        total_conversations = len(conversations)
        total_messages = sum(len(conv) for conv in conversations)
        
        # Process all conversations
        summaries = self.summarize_multiple_conversations(
            conversations,
            message_key=message_key,
            username_key=username_key
        )
        
        # Extract all participants
        all_participants = set()
        for summary in summaries:
            if 'participants' in summary:
                all_participants.update(summary['participants'])
                
        # Count messages per participant
        participant_message_counts = defaultdict(int)
        for summary in summaries:
            if 'participant_message_counts' in summary:
                for participant, count in summary['participant_message_counts'].items():
                    participant_message_counts[participant] += count
                    
        # Get top participants by message count
        top_participants = sorted(
            participant_message_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        # Extract all key phrases
        all_key_phrases = []
        for summary in summaries:
            if 'key_phrases' in summary:
                all_key_phrases.extend(summary['key_phrases'])
                
        # Count phrase occurrences
        phrase_counts = defaultdict(float)
        for phrase, score in all_key_phrases:
            phrase_counts[phrase] += score
            
        # Get top phrases
        top_phrases = sorted(
            phrase_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20]
        
        # Analyze crypto mentions
        _, crypto_mentions = self.analyze_crypto_mentions(
            conversations,
            message_key=message_key
        )
        
        # Cluster conversations
        clustering_results = self.cluster_conversations_by_topic(
            conversations,
            message_key=message_key,
            n_clusters=min(5, total_conversations)
        )
        
        # Create report data
        report = {
            'total_conversations': total_conversations,
            'total_messages': total_messages,
            'total_participants': len(all_participants),
            'top_participants': top_participants,
            'top_phrases': top_phrases,
            'crypto_mentions': crypto_mentions,
            'clustering_results': {
                'cluster_sizes': clustering_results.get('cluster_sizes', []),
                'cluster_summaries': clustering_results.get('cluster_summaries', [])
            },
            'conversation_summaries': summaries
        }
        
        # Save report if output file specified
        if output_file:
            try:
                # Convert datetime objects to strings for JSON serialization
                def json_serial(obj):
                    if isinstance(obj, datetime.datetime):
                        return obj.isoformat()
                    raise TypeError(f"Type {type(obj)} not serializable")
                
                with open(output_file, 'w') as f:
                    json.dump(report, f, default=json_serial, indent=2)
            except Exception as e:
                logger.error(f"Error saving report: {e}")
        
        return report 