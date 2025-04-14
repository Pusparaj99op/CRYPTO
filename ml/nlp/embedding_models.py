import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
import re
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import umap
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

class CryptoTextEmbeddings:
    """
    Text embedding techniques for cryptocurrency-related text analysis
    """
    
    def __init__(self, 
                model_name: str = 'distilbert-base-nli-mean-tokens',
                device: str = None,
                cache_dir: Optional[str] = None):
        """
        Initialize the text embeddings model
        
        Args:
            model_name (str): Name of the pre-trained model to use
            device (str): Device to use for computation ('cpu', 'cuda', 'cuda:0', etc.)
            cache_dir (str, optional): Directory to cache models
        """
        self.model_name = model_name
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        try:
            self.model = SentenceTransformer(model_name, device=self.device, cache_folder=cache_dir)
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            print("Falling back to direct AutoModel loading")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
            self.model.to(self.device)
    
    def generate_embeddings(self, 
                           texts: List[str], 
                           batch_size: int = 32, 
                           show_progress_bar: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Batch size for processing
            show_progress_bar (bool): Whether to show a progress bar
            
        Returns:
            np.ndarray: Array of embeddings with shape (len(texts), embedding_dim)
        """
        if isinstance(self.model, SentenceTransformer):
            # Generate embeddings using SentenceTransformer
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True
            )
        else:
            # Generate embeddings using manual implementation with AutoModel
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                # Tokenize
                encoded_input = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                
                # Compute token embeddings
                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                    
                    # Mean Pooling - Take average of all tokens
                    attention_mask = encoded_input['attention_mask']
                    token_embeddings = model_output[0]
                    
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    
                    # Convert to numpy and append to results
                    embeddings.append(batch_embeddings.cpu().numpy())
            
            # Concatenate all batch embeddings
            embeddings = np.vstack(embeddings)
        
        return embeddings
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Replace URLs with placeholder
        text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
        
        # Replace emojis with placeholder
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F700-\U0001F77F"  # alchemical symbols
            u"\U0001F780-\U0001F7FF"  # Geometric Shapes
            u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  # Chess Symbols
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            u"\U00002702-\U000027B0"  # Dingbats
            u"\U000024C2-\U0001F251" 
            "]+", flags=re.UNICODE
        )
        text = emoji_pattern.sub(r'[EMOJI]', text)
        
        return text
    
    def save_embeddings(self, embeddings: np.ndarray, file_path: str) -> None:
        """
        Save embeddings to a file
        
        Args:
            embeddings (np.ndarray): Embeddings array
            file_path (str): Path to save the embeddings
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Save embeddings
        np.save(file_path, embeddings)
    
    def load_embeddings(self, file_path: str) -> np.ndarray:
        """
        Load embeddings from a file
        
        Args:
            file_path (str): Path to the embeddings file
            
        Returns:
            np.ndarray: Loaded embeddings
        """
        return np.load(file_path)
    
    def reduce_dimensions(self, 
                         embeddings: np.ndarray, 
                         method: str = 'pca', 
                         n_components: int = 2,
                         random_state: int = 42) -> np.ndarray:
        """
        Reduce dimensionality of embeddings for visualization
        
        Args:
            embeddings (np.ndarray): Embeddings array
            method (str): Dimensionality reduction method ('pca', 'tsne', or 'umap')
            n_components (int): Number of components in the reduced space
            random_state (int): Random seed
            
        Returns:
            np.ndarray: Reduced embeddings
        """
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=random_state)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=random_state)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        return reduced_embeddings
    
    def plot_embeddings(self, 
                       reduced_embeddings: np.ndarray, 
                       labels: Optional[List[Union[str, int]]] = None,
                       figsize: Tuple[int, int] = (12, 8),
                       title: str = 'Text Embeddings Visualization',
                       alpha: float = 0.7,
                       s: int = 50,
                       colormap: str = 'tab10') -> plt.Figure:
        """
        Plot reduced embeddings
        
        Args:
            reduced_embeddings (np.ndarray): Reduced embeddings with shape (n_samples, 2)
            labels (List[Union[str, int]], optional): Labels for each point
            figsize (Tuple[int, int]): Figure size
            title (str): Plot title
            alpha (float): Alpha value for points
            s (int): Point size
            colormap (str): Colormap for labels
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if labels is not None:
            # Convert labels to numeric if they are strings
            if isinstance(labels[0], str):
                unique_labels = sorted(set(labels))
                label_to_id = {label: i for i, label in enumerate(unique_labels)}
                numeric_labels = [label_to_id[label] for label in labels]
            else:
                numeric_labels = labels
                unique_labels = sorted(set(numeric_labels))
            
            # Create scatter plot with colored points
            scatter = ax.scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                c=numeric_labels,
                cmap=colormap,
                alpha=alpha,
                s=s
            )
            
            # Add legend
            if len(unique_labels) <= 10:  # Only add legend if there aren't too many labels
                legend1 = ax.legend(
                    *scatter.legend_elements(),
                    loc="upper right",
                    title="Labels"
                )
                ax.add_artist(legend1)
        else:
            # Create scatter plot without labels
            ax.scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                alpha=alpha,
                s=s
            )
        
        ax.set_title(title)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def find_similar_texts(self, 
                          query_text: str, 
                          texts: List[str], 
                          embeddings: Optional[np.ndarray] = None,
                          top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Find similar texts to a query text
        
        Args:
            query_text (str): Query text
            texts (List[str]): List of texts to search in
            embeddings (np.ndarray, optional): Pre-computed embeddings for texts
            top_k (int): Number of top similar texts to return
            
        Returns:
            List[Tuple[int, float, str]]: List of (index, similarity, text) tuples
        """
        # Preprocess query text
        query_text = self.preprocess_text(query_text)
        
        # Generate query embedding
        query_embedding = self.generate_embeddings([query_text], show_progress_bar=False)[0]
        
        # Generate embeddings for texts if not provided
        if embeddings is None:
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            embeddings = self.generate_embeddings(processed_texts)
        
        # Calculate similarities
        similarities = []
        for i, embedding in enumerate(embeddings):
            similarity = 1 - cosine(query_embedding, embedding)
            similarities.append((i, similarity, texts[i]))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k similar texts
        return similarities[:top_k]
    
    def create_embeddings_dataframe(self, 
                                  texts: List[str], 
                                  metadata: Optional[Dict[str, List]] = None,
                                  batch_size: int = 32) -> pd.DataFrame:
        """
        Create a DataFrame with texts, embeddings, and optional metadata
        
        Args:
            texts (List[str]): List of texts to embed
            metadata (Dict[str, List], optional): Dictionary with metadata columns
            batch_size (int): Batch size for processing
            
        Returns:
            pd.DataFrame: DataFrame with texts, embeddings, and metadata
        """
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(processed_texts, batch_size=batch_size)
        
        # Create dataframe
        df = pd.DataFrame({'text': texts, 'processed_text': processed_texts})
        
        # Add embeddings columns
        for i in range(embeddings.shape[1]):
            df[f'embedding_{i}'] = embeddings[:, i]
        
        # Add metadata if provided
        if metadata is not None:
            for col_name, col_data in metadata.items():
                if len(col_data) == len(texts):
                    df[col_name] = col_data
        
        return df
    
    def cluster_embeddings(self, 
                         embeddings: np.ndarray, 
                         method: str = 'kmeans',
                         n_clusters: int = 5,
                         random_state: int = 42) -> np.ndarray:
        """
        Cluster embeddings
        
        Args:
            embeddings (np.ndarray): Embeddings array
            method (str): Clustering method ('kmeans', 'agglomerative', or 'dbscan')
            n_clusters (int): Number of clusters (for 'kmeans' and 'agglomerative')
            random_state (int): Random seed
            
        Returns:
            np.ndarray: Cluster labels
        """
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
        elif method == 'agglomerative':
            from sklearn.cluster import AgglomerativeClustering
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        elif method == 'dbscan':
            from sklearn.cluster import DBSCAN
            clusterer = DBSCAN(min_samples=5)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Fit and get labels
        cluster_labels = clusterer.fit_predict(embeddings)
        
        return cluster_labels
    
    def calculate_sentence_similarity_matrix(self, 
                                          texts: List[str], 
                                          embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate similarity matrix for a list of texts
        
        Args:
            texts (List[str]): List of texts
            embeddings (np.ndarray, optional): Pre-computed embeddings for texts
            
        Returns:
            np.ndarray: Similarity matrix
        """
        # Generate embeddings if not provided
        if embeddings is None:
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            embeddings = self.generate_embeddings(processed_texts)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        return similarity_matrix
    
    def plot_similarity_heatmap(self, 
                              similarity_matrix: np.ndarray,
                              labels: Optional[List[str]] = None,
                              figsize: Tuple[int, int] = (12, 10),
                              cmap: str = 'YlGnBu') -> plt.Figure:
        """
        Plot similarity matrix as a heatmap
        
        Args:
            similarity_matrix (np.ndarray): Similarity matrix
            labels (List[str], optional): Labels for each text
            figsize (Tuple[int, int]): Figure size
            cmap (str): Colormap
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            similarity_matrix,
            annot=False,
            cmap=cmap,
            xticklabels=labels if labels and len(labels) < 30 else False,
            yticklabels=labels if labels and len(labels) < 30 else False,
            ax=ax
        )
        
        plt.title('Text Similarity Matrix')
        plt.tight_layout()
        
        return fig
    
    def get_embedding_statistics(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics for embeddings
        
        Args:
            embeddings (np.ndarray): Embeddings array
            
        Returns:
            Dict[str, float]: Dictionary with embedding statistics
        """
        # Calculate basic statistics
        mean_values = np.mean(embeddings, axis=0)
        std_values = np.std(embeddings, axis=0)
        min_values = np.min(embeddings, axis=0)
        max_values = np.max(embeddings, axis=0)
        
        # Calculate norms
        norms = np.linalg.norm(embeddings, axis=1)
        
        # Calculate average cosine similarity
        n_samples = embeddings.shape[0]
        total_similarity = 0.0
        similarity_samples = min(n_samples, 1000)  # Limit computation for large datasets
        
        if n_samples <= 1:
            avg_similarity = 0.0
        else:
            # Sample pairs to compute average similarity
            import random
            pairs = []
            for _ in range(similarity_samples):
                i, j = random.sample(range(n_samples), 2)
                sim = 1 - cosine(embeddings[i], embeddings[j])
                total_similarity += sim
            
            avg_similarity = total_similarity / similarity_samples
        
        # Prepare statistics
        stats = {
            'n_samples': n_samples,
            'embedding_dim': embeddings.shape[1],
            'mean_norm': float(np.mean(norms)),
            'std_norm': float(np.std(norms)),
            'min_norm': float(np.min(norms)),
            'max_norm': float(np.max(norms)),
            'avg_similarity': float(avg_similarity),
            'mean_values': mean_values,
            'std_values': std_values,
            'min_values': min_values,
            'max_values': max_values
        }
        
        return stats
    
    def semantic_search(self, 
                      query: str,
                      texts: List[str],
                      embeddings: Optional[np.ndarray] = None,
                      top_k: int = 5,
                      threshold: Optional[float] = None) -> List[Dict[str, Union[int, float, str]]]:
        """
        Perform semantic search to find texts similar to a query
        
        Args:
            query (str): Query text
            texts (List[str]): Corpus of texts to search in
            embeddings (np.ndarray, optional): Pre-computed embeddings for texts
            top_k (int): Number of top results to return
            threshold (float, optional): Minimum similarity threshold
            
        Returns:
            List[Dict]: List of search results
        """
        # Get similar texts
        similar_texts = self.find_similar_texts(query, texts, embeddings, top_k=len(texts))
        
        # Filter by threshold if provided
        if threshold is not None:
            similar_texts = [item for item in similar_texts if item[1] >= threshold]
        
        # Limit to top_k
        similar_texts = similar_texts[:top_k]
        
        # Format results
        results = [
            {
                'index': idx,
                'similarity': float(sim),
                'text': text
            }
            for idx, sim, text in similar_texts
        ]
        
        return results
    
    def create_text_clusters(self, 
                           texts: List[str],
                           n_clusters: int = 5,
                           cluster_method: str = 'kmeans',
                           dim_reduction_method: str = 'umap',
                           random_state: int = 42) -> Dict[str, Any]:
        """
        Create clusters from text embeddings
        
        Args:
            texts (List[str]): List of texts to cluster
            n_clusters (int): Number of clusters
            cluster_method (str): Clustering method
            dim_reduction_method (str): Dimensionality reduction method
            random_state (int): Random seed
            
        Returns:
            Dict: Dictionary with clustering results
        """
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(processed_texts)
        
        # Cluster embeddings
        cluster_labels = self.cluster_embeddings(
            embeddings,
            method=cluster_method,
            n_clusters=n_clusters,
            random_state=random_state
        )
        
        # Reduce dimensions for visualization
        reduced_embeddings = self.reduce_dimensions(
            embeddings,
            method=dim_reduction_method,
            n_components=2,
            random_state=random_state
        )
        
        # Calculate cluster centers
        cluster_centers = {}
        for cluster_id in range(n_clusters):
            if cluster_method == 'dbscan' and cluster_id == -1:
                # Skip noise points in DBSCAN
                continue
                
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                cluster_centers[cluster_id] = np.mean(embeddings[cluster_indices], axis=0)
        
        # Create a DataFrame with results
        result_df = pd.DataFrame({
            'text': texts,
            'processed_text': processed_texts,
            'cluster': cluster_labels,
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1]
        })
        
        # Calculate representative texts for each cluster
        representative_texts = {}
        for cluster_id in range(n_clusters):
            if cluster_method == 'dbscan' and cluster_id == -1:
                # Skip noise points in DBSCAN
                continue
                
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                cluster_embeddings = embeddings[cluster_indices]
                cluster_center = cluster_centers[cluster_id]
                
                # Calculate distance to center
                distances = []
                for i, idx in enumerate(cluster_indices):
                    dist = cosine(cluster_embeddings[i], cluster_center)
                    distances.append((idx, dist))
                
                # Sort by distance
                distances.sort(key=lambda x: x[1])
                
                # Get closest texts
                closest_indices = [idx for idx, _ in distances[:5]]
                closest_texts = [texts[idx] for idx in closest_indices]
                
                representative_texts[cluster_id] = closest_texts
        
        # Prepare results
        results = {
            'dataframe': result_df,
            'embeddings': embeddings,
            'reduced_embeddings': reduced_embeddings,
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers,
            'representative_texts': representative_texts
        }
        
        return results 