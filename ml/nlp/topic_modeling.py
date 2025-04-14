import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from gensim import corpora, models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import pyLDAvis
import pyLDAvis.gensim_models
import pyLDAvis.sklearn

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class CryptoTopicModeling:
    """
    Topic modeling for cryptocurrency-related text data
    """
    
    def __init__(self, 
                additional_stopwords: Optional[List[str]] = None,
                crypto_specific_terms: Optional[List[str]] = None):
        """
        Initialize the topic modeling system
        
        Args:
            additional_stopwords (List[str], optional): Additional stopwords to exclude from analysis
            crypto_specific_terms (List[str], optional): Crypto-specific terms to include in analysis
        """
        # Standard stopwords
        self.stopwords = set(stopwords.words('english'))
        
        # Add additional stopwords if provided
        if additional_stopwords:
            self.stopwords.update(additional_stopwords)
        
        # Default additional crypto stopwords (terms that are too common in crypto discussions)
        default_crypto_stopwords = [
            'crypto', 'cryptocurrency', 'bitcoin', 'btc', 'eth', 'ethereum',
            'coin', 'token', 'blockchain', 'market', 'price', 'trading',
            'buy', 'sell', 'hodl', 'exchange', 'wallet', 'address', 'transaction'
        ]
        self.stopwords.update(default_crypto_stopwords)
        
        # Remove crypto-specific terms from stopwords if provided
        if crypto_specific_terms:
            self.stopwords = self.stopwords - set(t.lower() for t in crypto_specific_terms)
        
        # Initialize NLTK lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Pre-trained models
        self.vectorizer = None
        self.lda_model = None
        self.nmf_model = None
        self.dictionary = None
        self.corpus = None
        self.gensim_lda = None
    
    def preprocess_text(self, 
                       text: str,
                       lowercase: bool = True,
                       remove_punctuation: bool = True,
                       remove_urls: bool = True,
                       remove_numbers: bool = True,
                       lemmatize: bool = True,
                       min_word_length: int = 3) -> List[str]:
        """
        Preprocess text for topic modeling
        
        Args:
            text (str): Input text
            lowercase (bool): Whether to convert to lowercase
            remove_punctuation (bool): Whether to remove punctuation
            remove_urls (bool): Whether to remove URLs
            remove_numbers (bool): Whether to remove numbers
            lemmatize (bool): Whether to lemmatize words
            min_word_length (int): Minimum word length to keep
            
        Returns:
            List[str]: Preprocessed token list
        """
        if not isinstance(text, str):
            return []
        
        # Remove URLs
        if remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Lowercase
        if lowercase:
            text = text.lower()
        
        # Remove punctuation
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Remove numbers
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [
            word for word in tokens 
            if word not in self.stopwords and len(word) >= min_word_length
        ]
        
        # Lemmatize
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return tokens
    
    def preprocess_texts(self, 
                        texts: List[str],
                        lowercase: bool = True,
                        remove_punctuation: bool = True,
                        remove_urls: bool = True,
                        remove_numbers: bool = True,
                        lemmatize: bool = True,
                        min_word_length: int = 3) -> List[List[str]]:
        """
        Preprocess multiple texts for topic modeling
        
        Args:
            texts (List[str]): List of input texts
            lowercase (bool): Whether to convert to lowercase
            remove_punctuation (bool): Whether to remove punctuation
            remove_urls (bool): Whether to remove URLs
            remove_numbers (bool): Whether to remove numbers
            lemmatize (bool): Whether to lemmatize words
            min_word_length (int): Minimum word length to keep
            
        Returns:
            List[List[str]]: List of preprocessed token lists
        """
        processed_texts = []
        for text in texts:
            tokens = self.preprocess_text(
                text,
                lowercase=lowercase,
                remove_punctuation=remove_punctuation,
                remove_urls=remove_urls,
                remove_numbers=remove_numbers,
                lemmatize=lemmatize,
                min_word_length=min_word_length
            )
            processed_texts.append(tokens)
        
        return processed_texts
    
    def create_document_term_matrix(self, 
                                  processed_texts: List[List[str]],
                                  min_df: float = 0.01,
                                  max_df: float = 0.95,
                                  ngram_range: Tuple[int, int] = (1, 1),
                                  use_tfidf: bool = False) -> Union[CountVectorizer, TfidfVectorizer]:
        """
        Create document-term matrix from processed texts
        
        Args:
            processed_texts (List[List[str]]): List of preprocessed token lists
            min_df (float): Minimum document frequency for terms
            max_df (float): Maximum document frequency for terms
            ngram_range (Tuple[int, int]): N-gram range to consider
            use_tfidf (bool): Whether to use TF-IDF instead of raw counts
            
        Returns:
            Union[CountVectorizer, TfidfVectorizer]: Trained vectorizer
        """
        # Convert token lists to strings
        doc_strings = [' '.join(tokens) for tokens in processed_texts]
        
        # Create vectorizer
        if use_tfidf:
            vectorizer = TfidfVectorizer(
                min_df=min_df,
                max_df=max_df,
                ngram_range=ngram_range,
                stop_words=None  # Already removed in preprocessing
            )
        else:
            vectorizer = CountVectorizer(
                min_df=min_df,
                max_df=max_df,
                ngram_range=ngram_range,
                stop_words=None  # Already removed in preprocessing
            )
        
        # Fit vectorizer
        dtm = vectorizer.fit_transform(doc_strings)
        
        # Store for later use
        self.vectorizer = vectorizer
        
        return vectorizer, dtm
    
    def create_gensim_corpus(self, processed_texts: List[List[str]]) -> Tuple[corpora.Dictionary, List[List[Tuple[int, int]]]]:
        """
        Create Gensim dictionary and corpus from processed texts
        
        Args:
            processed_texts (List[List[str]]): List of preprocessed token lists
            
        Returns:
            Tuple: (Gensim dictionary, Gensim corpus)
        """
        # Create dictionary
        dictionary = corpora.Dictionary(processed_texts)
        
        # Filter extremes (uncommon and very common words)
        dictionary.filter_extremes(no_below=2, no_above=0.9)
        
        # Create corpus
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        
        # Store for later use
        self.dictionary = dictionary
        self.corpus = corpus
        
        return dictionary, corpus
    
    def train_lda_model(self, 
                      dtm,
                      n_topics: int = 5,
                      max_iter: int = 10,
                      learning_method: str = 'online',
                      random_state: int = 42) -> LatentDirichletAllocation:
        """
        Train LDA topic model using scikit-learn
        
        Args:
            dtm: Document-term matrix from create_document_term_matrix
            n_topics (int): Number of topics to extract
            max_iter (int): Maximum number of iterations
            learning_method (str): Learning method ('online' or 'batch')
            random_state (int): Random seed
            
        Returns:
            LatentDirichletAllocation: Trained LDA model
        """
        # Create and train LDA model
        lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=max_iter,
            learning_method=learning_method,
            random_state=random_state
        )
        lda_model.fit(dtm)
        
        # Store for later use
        self.lda_model = lda_model
        
        return lda_model
    
    def train_nmf_model(self, 
                      dtm,
                      n_topics: int = 5,
                      max_iter: int = 200,
                      random_state: int = 42) -> NMF:
        """
        Train NMF topic model using scikit-learn
        
        Args:
            dtm: Document-term matrix from create_document_term_matrix
            n_topics (int): Number of topics to extract
            max_iter (int): Maximum number of iterations
            random_state (int): Random seed
            
        Returns:
            NMF: Trained NMF model
        """
        # Create and train NMF model
        nmf_model = NMF(
            n_components=n_topics,
            max_iter=max_iter,
            random_state=random_state
        )
        nmf_model.fit(dtm)
        
        # Store for later use
        self.nmf_model = nmf_model
        
        return nmf_model
    
    def train_gensim_lda(self, 
                       corpus,
                       dictionary,
                       n_topics: int = 5,
                       passes: int = 10,
                       alpha: str = 'auto',
                       eta: str = 'auto',
                       random_state: int = 42) -> models.LdaModel:
        """
        Train LDA topic model using Gensim
        
        Args:
            corpus: Corpus from create_gensim_corpus
            dictionary: Dictionary from create_gensim_corpus
            n_topics (int): Number of topics to extract
            passes (int): Number of passes through the corpus
            alpha (str): Prior for document-topic distribution ('auto' or a specific value)
            eta (str): Prior for topic-word distribution ('auto' or a specific value)
            random_state (int): Random seed
            
        Returns:
            models.LdaModel: Trained LDA model
        """
        # Create and train LDA model
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=n_topics,
            passes=passes,
            alpha=alpha,
            eta=eta,
            random_state=random_state
        )
        
        # Store for later use
        self.gensim_lda = lda_model
        
        return lda_model
    
    def get_topics_sklearn(self, 
                         model,
                         n_top_words: int = 10,
                         vectorizer=None) -> List[List[str]]:
        """
        Get top words for each topic from a scikit-learn topic model
        
        Args:
            model: Trained scikit-learn topic model (LDA or NMF)
            n_top_words (int): Number of top words per topic
            vectorizer: Vectorizer used to create the document-term matrix
            
        Returns:
            List[List[str]]: List of topics, where each topic is a list of top words
        """
        if vectorizer is None:
            vectorizer = self.vectorizer
        
        if vectorizer is None:
            raise ValueError("Vectorizer not provided and not found in instance")
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top words for each topic
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            # Sort words by importance
            top_indices = topic.argsort()[:-n_top_words-1:-1]
            top_words = [feature_names[i] for i in top_indices]
            topics.append(top_words)
        
        return topics
    
    def get_topics_gensim(self, 
                        model,
                        n_top_words: int = 10) -> List[List[str]]:
        """
        Get top words for each topic from a Gensim topic model
        
        Args:
            model: Trained Gensim LDA model
            n_top_words (int): Number of top words per topic
            
        Returns:
            List[List[str]]: List of topics, where each topic is a list of top words
        """
        # Get top words for each topic
        topics = []
        for topic_idx in range(model.num_topics):
            # Get top words with probabilities
            top_words = model.show_topic(topic_idx, n_top_words)
            # Extract just the words
            words = [word for word, _ in top_words]
            topics.append(words)
        
        return topics
    
    def assign_topics_sklearn(self, 
                            model,
                            dtm) -> List[int]:
        """
        Assign topics to documents using a scikit-learn topic model
        
        Args:
            model: Trained scikit-learn topic model (LDA or NMF)
            dtm: Document-term matrix
            
        Returns:
            List[int]: List of dominant topic indices for each document
        """
        # Get topic distribution for each document
        doc_topic_dist = model.transform(dtm)
        
        # Get dominant topic for each document
        dominant_topics = doc_topic_dist.argmax(axis=1).tolist()
        
        return dominant_topics
    
    def assign_topics_gensim(self, 
                           model,
                           corpus) -> List[int]:
        """
        Assign topics to documents using a Gensim topic model
        
        Args:
            model: Trained Gensim LDA model
            corpus: Gensim corpus
            
        Returns:
            List[int]: List of dominant topic indices for each document
        """
        # Get topic distribution for each document
        dominant_topics = []
        for doc in corpus:
            topic_dist = model.get_document_topics(doc)
            # Find dominant topic
            dominant_topic = max(topic_dist, key=lambda x: x[1])[0] if topic_dist else -1
            dominant_topics.append(dominant_topic)
        
        return dominant_topics
    
    def visualize_topics_pyldavis(self, 
                                model,
                                dtm=None,
                                corpus=None,
                                dictionary=None,
                                vectorizer=None,
                                use_gensim: bool = False) -> Any:
        """
        Create interactive pyLDAvis visualization of topics
        
        Args:
            model: Trained topic model (scikit-learn or Gensim)
            dtm: Document-term matrix (for scikit-learn)
            corpus: Gensim corpus (for Gensim)
            dictionary: Gensim dictionary (for Gensim)
            vectorizer: Vectorizer (for scikit-learn)
            use_gensim (bool): Whether to use Gensim visualization
            
        Returns:
            Any: pyLDAvis visualization
        """
        if use_gensim:
            if corpus is None:
                corpus = self.corpus
            if dictionary is None:
                dictionary = self.dictionary
            
            if corpus is None or dictionary is None:
                raise ValueError("Corpus and dictionary required for Gensim visualization")
            
            # Create visualization
            vis = pyLDAvis.gensim_models.prepare(
                model, corpus, dictionary,
                sort_topics=False
            )
        else:
            if dtm is None or vectorizer is None:
                raise ValueError("DTM and vectorizer required for scikit-learn visualization")
            
            # Create visualization
            vis = pyLDAvis.sklearn.prepare(
                model, dtm, vectorizer,
                sort_topics=False
            )
        
        return vis
    
    def plot_topics_wordcloud(self, 
                            topics: List[List[str]],
                            n_cols: int = 3,
                            figsize: Tuple[int, int] = (15, 12),
                            title_prefix: str = "Topic") -> plt.Figure:
        """
        Plot topics as word clouds
        
        Args:
            topics (List[List[str]]): List of topics (each a list of words)
            n_cols (int): Number of columns in the plot grid
            figsize (Tuple[int, int]): Figure size
            title_prefix (str): Prefix for subplot titles
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        n_topics = len(topics)
        n_rows = (n_topics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Flatten axes if there's more than one row
        if n_rows > 1:
            axes = axes.flatten()
        
        # Create word cloud for each topic
        for i, topic_words in enumerate(topics):
            if i < len(axes):
                ax = axes[i] if n_topics > 1 else axes
                
                # Join words for word cloud
                text = ' '.join(topic_words)
                
                # Create word cloud
                wordcloud = WordCloud(
                    background_color='white',
                    max_words=100,
                    contour_width=3,
                    contour_color='steelblue'
                ).generate(text)
                
                # Plot
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f"{title_prefix} {i+1}")
                ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_topics, len(axes) if isinstance(axes, np.ndarray) else 1):
            if n_topics > 1:
                axes[i].axis('off')
        
        plt.tight_layout()
        
        return fig
    
    def plot_topic_distribution(self, 
                              dominant_topics: List[int],
                              n_topics: int,
                              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot topic distribution
        
        Args:
            dominant_topics (List[int]): List of dominant topic indices for each document
            n_topics (int): Total number of topics
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Count documents per topic
        topic_counts = Counter(dominant_topics)
        
        # Create DataFrame for plotting
        topic_df = pd.DataFrame({
            'Topic': list(range(n_topics)),
            'Count': [topic_counts.get(i, 0) for i in range(n_topics)]
        })
        
        # Sort by count
        topic_df = topic_df.sort_values('Count', ascending=False)
        
        # Calculate percentage
        total_docs = len(dominant_topics)
        topic_df['Percentage'] = topic_df['Count'] / total_docs * 100
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x='Topic', y='Count', data=topic_df, ax=ax)
        
        # Add percentage labels on top of bars
        for i, row in topic_df.iterrows():
            ax.text(
                i, row['Count'] + 1,
                f"{row['Percentage']:.1f}%",
                ha='center'
            )
        
        ax.set_title('Document Distribution Across Topics')
        ax.set_xlabel('Topic')
        ax.set_ylabel('Number of Documents')
        
        plt.tight_layout()
        
        return fig
    
    def analyze_topics_over_time(self, 
                               dominant_topics: List[int],
                               dates: List,
                               n_topics: int,
                               freq: str = 'M') -> pd.DataFrame:
        """
        Analyze topic distribution over time
        
        Args:
            dominant_topics (List[int]): List of dominant topic indices for each document
            dates (List): List of dates for each document
            n_topics (int): Total number of topics
            freq (str): Frequency for time grouping ('D' = daily, 'W' = weekly, 'M' = monthly, etc.)
            
        Returns:
            pd.DataFrame: DataFrame with topic counts over time
        """
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'topic': dominant_topics
        })
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Group by date and topic
        topic_time = df.groupby([pd.Grouper(key='date', freq=freq), 'topic']).size().unstack(fill_value=0)
        
        # Rename columns
        topic_time.columns = [f'Topic {i}' for i in topic_time.columns]
        
        # Calculate percentage
        topic_time_pct = topic_time.div(topic_time.sum(axis=1), axis=0) * 100
        
        return topic_time, topic_time_pct
    
    def plot_topics_over_time(self, 
                            topic_time: pd.DataFrame,
                            plot_type: str = 'area',
                            figsize: Tuple[int, int] = (12, 6),
                            is_percentage: bool = False) -> plt.Figure:
        """
        Plot topic distribution over time
        
        Args:
            topic_time (pd.DataFrame): DataFrame from analyze_topics_over_time
            plot_type (str): Type of plot ('area', 'line', or 'bar')
            figsize (Tuple[int, int]): Figure size
            is_percentage (bool): Whether data is in percentages
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if plot_type == 'area':
            topic_time.plot.area(ax=ax, stacked=True)
        elif plot_type == 'line':
            topic_time.plot.line(ax=ax)
        elif plot_type == 'bar':
            topic_time.plot.bar(ax=ax, stacked=True)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        ax.set_title('Topic Evolution Over Time')
        ax.set_xlabel('Date')
        
        if is_percentage:
            ax.set_ylabel('Percentage of Documents')
            ax.set_ylim(0, 100)
        else:
            ax.set_ylabel('Number of Documents')
        
        ax.legend(title='Topics')
        
        plt.tight_layout()
        
        return fig
    
    def find_representative_documents(self, 
                                    model,
                                    texts: List[str],
                                    dtm=None,
                                    corpus=None,
                                    n_docs: int = 3,
                                    use_gensim: bool = False) -> Dict[int, List[Tuple[int, str]]]:
        """
        Find representative documents for each topic
        
        Args:
            model: Trained topic model (scikit-learn or Gensim)
            texts (List[str]): Original text documents
            dtm: Document-term matrix (for scikit-learn)
            corpus: Gensim corpus (for Gensim)
            n_docs (int): Number of representative documents per topic
            use_gensim (bool): Whether to use Gensim model
            
        Returns:
            Dict[int, List[Tuple[int, str]]]: Dictionary mapping topic indices to lists of (document index, document text) tuples
        """
        if use_gensim:
            if corpus is None:
                corpus = self.corpus
            
            if corpus is None:
                raise ValueError("Corpus required for Gensim model")
            
            # Get topic distribution for each document
            doc_topics = [model.get_document_topics(doc) for doc in corpus]
            
            # Organize by topic
            topic_docs = defaultdict(list)
            for i, doc_topic in enumerate(doc_topics):
                for topic_idx, prob in doc_topic:
                    topic_docs[topic_idx].append((i, prob))
            
            # Get top documents for each topic
            representative_docs = {}
            for topic_idx, docs in topic_docs.items():
                # Sort by probability
                docs.sort(key=lambda x: x[1], reverse=True)
                # Get top n documents
                top_docs = [(doc_idx, texts[doc_idx]) for doc_idx, _ in docs[:n_docs]]
                representative_docs[topic_idx] = top_docs
        
        else:
            if dtm is None:
                raise ValueError("DTM required for scikit-learn model")
            
            # Get topic distribution for each document
            doc_topic_dist = model.transform(dtm)
            
            # Get top documents for each topic
            representative_docs = {}
            for topic_idx in range(doc_topic_dist.shape[1]):
                # Get probability of this topic for each document
                topic_probs = doc_topic_dist[:, topic_idx]
                # Get indices of top n documents
                top_indices = topic_probs.argsort()[-n_docs:][::-1]
                # Get documents
                top_docs = [(int(idx), texts[idx]) for idx in top_indices]
                representative_docs[topic_idx] = top_docs
        
        return representative_docs
    
    def run_complete_analysis(self, 
                            texts: List[str],
                            n_topics: int = 5,
                            method: str = 'lda',
                            min_df: float = 0.01,
                            max_df: float = 0.95) -> Dict[str, Any]:
        """
        Run complete topic modeling analysis
        
        Args:
            texts (List[str]): List of text documents
            n_topics (int): Number of topics to extract
            method (str): Topic modeling method ('lda', 'nmf', or 'gensim_lda')
            min_df (float): Minimum document frequency for terms
            max_df (float): Maximum document frequency for terms
            
        Returns:
            Dict[str, Any]: Dictionary with analysis results
        """
        # Preprocess texts
        processed_texts = self.preprocess_texts(texts)
        
        if method == 'gensim_lda':
            # Create Gensim corpus and dictionary
            dictionary, corpus = self.create_gensim_corpus(processed_texts)
            
            # Train model
            model = self.train_gensim_lda(corpus, dictionary, n_topics=n_topics)
            
            # Get topics
            topics = self.get_topics_gensim(model, n_top_words=10)
            
            # Assign topics
            dominant_topics = self.assign_topics_gensim(model, corpus)
            
            # Find representative documents
            representative_docs = self.find_representative_documents(
                model, texts, corpus=corpus, n_docs=3, use_gensim=True
            )
            
            # Prepare results
            results = {
                'model': model,
                'topics': topics,
                'dominant_topics': dominant_topics,
                'corpus': corpus,
                'dictionary': dictionary,
                'representative_docs': representative_docs,
                'method': 'gensim_lda'
            }
        
        else:
            # Create document-term matrix
            vectorizer, dtm = self.create_document_term_matrix(
                processed_texts,
                min_df=min_df,
                max_df=max_df,
                use_tfidf=(method == 'nmf')
            )
            
            # Train model
            if method == 'lda':
                model = self.train_lda_model(dtm, n_topics=n_topics)
            elif method == 'nmf':
                model = self.train_nmf_model(dtm, n_topics=n_topics)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Get topics
            topics = self.get_topics_sklearn(model, n_top_words=10, vectorizer=vectorizer)
            
            # Assign topics
            dominant_topics = self.assign_topics_sklearn(model, dtm)
            
            # Find representative documents
            representative_docs = self.find_representative_documents(
                model, texts, dtm=dtm, n_docs=3, use_gensim=False
            )
            
            # Prepare results
            results = {
                'model': model,
                'topics': topics,
                'dominant_topics': dominant_topics,
                'dtm': dtm,
                'vectorizer': vectorizer,
                'representative_docs': representative_docs,
                'method': method
            }
        
        return results
    
    def apply_to_new_data(self, 
                         new_texts: List[str],
                         model=None,
                         vectorizer=None,
                         dictionary=None,
                         method: str = 'lda') -> List[int]:
        """
        Apply trained topic model to new data
        
        Args:
            new_texts (List[str]): List of new text documents
            model: Trained topic model
            vectorizer: Trained vectorizer (for scikit-learn models)
            dictionary: Trained dictionary (for Gensim models)
            method (str): Topic modeling method ('lda', 'nmf', or 'gensim_lda')
            
        Returns:
            List[int]: List of dominant topic indices for new documents
        """
        if model is None:
            if method == 'lda':
                model = self.lda_model
            elif method == 'nmf':
                model = self.nmf_model
            elif method == 'gensim_lda':
                model = self.gensim_lda
            else:
                raise ValueError(f"Unknown method: {method}")
        
        if model is None:
            raise ValueError("No model provided and no model found in instance")
        
        # Preprocess new texts
        processed_texts = self.preprocess_texts(new_texts)
        
        if method == 'gensim_lda':
            if dictionary is None:
                dictionary = self.dictionary
            
            if dictionary is None:
                raise ValueError("Dictionary required for Gensim model")
            
            # Convert to corpus
            corpus = [dictionary.doc2bow(text) for text in processed_texts]
            
            # Assign topics
            dominant_topics = self.assign_topics_gensim(model, corpus)
        
        else:
            if vectorizer is None:
                vectorizer = self.vectorizer
            
            if vectorizer is None:
                raise ValueError("Vectorizer required for scikit-learn model")
            
            # Convert to document-term matrix
            doc_strings = [' '.join(tokens) for tokens in processed_texts]
            dtm = vectorizer.transform(doc_strings)
            
            # Assign topics
            dominant_topics = self.assign_topics_sklearn(model, dtm)
        
        return dominant_topics 