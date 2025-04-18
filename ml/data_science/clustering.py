"""
Clustering Module for Cryptocurrency Trading

This module provides tools for identifying market regimes and patterns
through various clustering techniques applied to cryptocurrency data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from tslearn.clustering import TimeSeriesKMeans, KShape

class MarketRegimeClusterer:
    """Class for identifying market regimes through clustering."""
    
    def __init__(self):
        """Initialize the market regime clusterer."""
        self.models = {}
        self.clusters = {}
        self.scaled_data = {}
        self.metrics = {}
    
    def cluster_kmeans(self, X: pd.DataFrame, n_clusters: int = 3, 
                      standardize: bool = True) -> pd.Series:
        """
        Cluster data using K-means algorithm.
        
        Args:
            X: Feature DataFrame
            n_clusters: Number of clusters
            standardize: Whether to standardize data
            
        Returns:
            Series of cluster labels
        """
        data = X.copy()
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            self.scaled_data['kmeans'] = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)
        else:
            data_scaled = data.values
        
        # Apply K-means
        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        labels = model.fit_predict(data_scaled)
        
        # Store results
        self.models['kmeans'] = model
        clusters = pd.Series(labels, index=data.index, name='kmeans_cluster')
        self.clusters['kmeans'] = clusters
        
        # Calculate metrics
        if n_clusters > 1 and n_clusters < len(data) - 1:
            self.metrics['kmeans'] = {
                'inertia': model.inertia_,
                'silhouette': silhouette_score(data_scaled, labels),
                'calinski_harabasz': calinski_harabasz_score(data_scaled, labels)
            }
        
        return clusters
    
    def cluster_gmm(self, X: pd.DataFrame, n_components: int = 3, 
                   covariance_type: str = 'full', standardize: bool = True) -> pd.Series:
        """
        Cluster data using Gaussian Mixture Model.
        
        Args:
            X: Feature DataFrame
            n_components: Number of mixture components
            covariance_type: Type of covariance parameters
            standardize: Whether to standardize data
            
        Returns:
            Series of cluster labels
        """
        data = X.copy()
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            self.scaled_data['gmm'] = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)
        else:
            data_scaled = data.values
        
        # Apply GMM
        model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=42
        )
        model.fit(data_scaled)
        labels = model.predict(data_scaled)
        
        # Store results
        self.models['gmm'] = model
        clusters = pd.Series(labels, index=data.index, name='gmm_cluster')
        self.clusters['gmm'] = clusters
        
        # Calculate metrics
        if n_components > 1:
            self.metrics['gmm'] = {
                'bic': model.bic(data_scaled),
                'aic': model.aic(data_scaled)
            }
            
            if n_components < len(data) - 1:
                self.metrics['gmm'].update({
                    'silhouette': silhouette_score(data_scaled, labels),
                    'calinski_harabasz': calinski_harabasz_score(data_scaled, labels)
                })
        
        return clusters
    
    def cluster_agglomerative(self, X: pd.DataFrame, n_clusters: int = 3, 
                           linkage: str = 'ward', affinity: str = 'euclidean',
                           standardize: bool = True) -> pd.Series:
        """
        Cluster data using Agglomerative Hierarchical Clustering.
        
        Args:
            X: Feature DataFrame
            n_clusters: Number of clusters
            linkage: Linkage criterion
            affinity: Distance metric
            standardize: Whether to standardize data
            
        Returns:
            Series of cluster labels
        """
        data = X.copy()
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            self.scaled_data['agglomerative'] = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)
        else:
            data_scaled = data.values
        
        # Apply Agglomerative Clustering
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            affinity=affinity
        )
        labels = model.fit_predict(data_scaled)
        
        # Store results
        self.models['agglomerative'] = model
        clusters = pd.Series(labels, index=data.index, name='agglomerative_cluster')
        self.clusters['agglomerative'] = clusters
        
        # Calculate metrics
        if n_clusters > 1 and n_clusters < len(data) - 1:
            self.metrics['agglomerative'] = {
                'silhouette': silhouette_score(data_scaled, labels),
                'calinski_harabasz': calinski_harabasz_score(data_scaled, labels)
            }
        
        return clusters
    
    def cluster_dbscan(self, X: pd.DataFrame, eps: float = 0.5, min_samples: int = 5,
                     standardize: bool = True) -> pd.Series:
        """
        Cluster data using DBSCAN.
        
        Args:
            X: Feature DataFrame
            eps: Maximum distance between samples
            min_samples: Minimum number of samples in a neighborhood
            standardize: Whether to standardize data
            
        Returns:
            Series of cluster labels
        """
        data = X.copy()
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            self.scaled_data['dbscan'] = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)
        else:
            data_scaled = data.values
        
        # Apply DBSCAN
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )
        labels = model.fit_predict(data_scaled)
        
        # Store results
        self.models['dbscan'] = model
        clusters = pd.Series(labels, index=data.index, name='dbscan_cluster')
        self.clusters['dbscan'] = clusters
        
        # Calculate metrics if more than one cluster found
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1:
            # Filter out noise points for metric calculation
            mask = labels != -1
            if sum(mask) > n_clusters:  # Ensure enough points for calculation
                self.metrics['dbscan'] = {
                    'silhouette': silhouette_score(data_scaled[mask], labels[mask]),
                    'calinski_harabasz': calinski_harabasz_score(data_scaled[mask], labels[mask]),
                    'n_clusters': n_clusters,
                    'n_noise': sum(labels == -1)
                }
        
        return clusters
    
    def cluster_timeseries(self, X: pd.DataFrame, n_clusters: int = 3, 
                         method: str = 'kmeans', metric: str = 'dtw',
                         standardize: bool = True) -> pd.Series:
        """
        Cluster time series data using specialized algorithms.
        
        Args:
            X: Feature DataFrame with time series
            n_clusters: Number of clusters
            method: Clustering method ('kmeans', 'kshape')
            metric: Distance metric for kmeans ('euclidean', 'dtw', 'softdtw')
            standardize: Whether to standardize data
            
        Returns:
            Series of cluster labels
        """
        data = X.copy()
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            self.scaled_data['timeseries'] = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)
        else:
            data_scaled = data.values
        
        # Apply time series clustering
        if method == 'kmeans':
            model = TimeSeriesKMeans(
                n_clusters=n_clusters,
                metric=metric,
                random_state=42
            )
        elif method == 'kshape':
            # KShape requires z-normalization
            model = KShape(
                n_clusters=n_clusters,
                random_state=42
            )
        else:
            raise ValueError("Method must be 'kmeans' or 'kshape'")
        
        # Reshape for tslearn if needed
        data_reshaped = data_scaled.reshape(data_scaled.shape[0], data_scaled.shape[1], 1)
        labels = model.fit_predict(data_reshaped)
        
        # Store results
        method_name = f'timeseries_{method}'
        self.models[method_name] = model
        clusters = pd.Series(labels, index=data.index, name=f'{method_name}_cluster')
        self.clusters[method_name] = clusters
        
        return clusters
    
    def find_optimal_k(self, X: pd.DataFrame, max_k: int = 10, 
                     method: str = 'kmeans', standardize: bool = True) -> Dict:
        """
        Find optimal number of clusters using multiple methods.
        
        Args:
            X: Feature DataFrame
            max_k: Maximum number of clusters to consider
            method: Clustering method
            standardize: Whether to standardize data
            
        Returns:
            Dictionary with evaluation metrics
        """
        data = X.copy()
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
        else:
            data_scaled = data.values
        
        # Initialize results
        results = {
            'k': list(range(2, max_k + 1)),
            'silhouette': [],
            'calinski_harabasz': []
        }
        
        if method == 'kmeans':
            results['inertia'] = []
        elif method == 'gmm':
            results['bic'] = []
            results['aic'] = []
        
        # Evaluate different k values
        for k in range(2, max_k + 1):
            if method == 'kmeans':
                # K-means
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(data_scaled)
                results['inertia'].append(model.inertia_)
                
            elif method == 'gmm':
                # Gaussian Mixture Model
                model = GaussianMixture(n_components=k, random_state=42)
                model.fit(data_scaled)
                labels = model.predict(data_scaled)
                results['bic'].append(model.bic(data_scaled))
                results['aic'].append(model.aic(data_scaled))
                
            elif method == 'agglomerative':
                # Agglomerative Clustering
                model = AgglomerativeClustering(n_clusters=k)
                labels = model.fit_predict(data_scaled)
                
            else:
                raise ValueError(f"Method '{method}' not supported for optimal k search")
            
            # Calculate common metrics
            results['silhouette'].append(silhouette_score(data_scaled, labels))
            results['calinski_harabasz'].append(calinski_harabasz_score(data_scaled, labels))
        
        return results
    
    def plot_clusters_2d(self, X: pd.DataFrame, method: str = 'kmeans', 
                      dims: List[int] = [0, 1], figsize: Tuple = (10, 8)):
        """
        Plot clusters in 2D space.
        
        Args:
            X: Original feature DataFrame
            method: Clustering method
            dims: Indices of dimensions to plot
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        if method not in self.clusters:
            raise ValueError(f"Method '{method}' not found in clusters")
        
        clusters = self.clusters[method]
        
        # Get scaled data if available
        if method in self.scaled_data:
            data = self.scaled_data[method]
        else:
            data = X
        
        # Check dimensions
        if max(dims) >= data.shape[1]:
            raise ValueError(f"Dimension index {max(dims)} out of range for data with {data.shape[1]} dimensions")
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Get unique clusters
        unique_clusters = sorted(clusters.unique())
        
        # Create colormap
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        
        # Plot points
        for i, cluster in enumerate(unique_clusters):
            mask = clusters == cluster
            plt.scatter(
                data.iloc[mask, dims[0]], 
                data.iloc[mask, dims[1]], 
                c=[colors[i]], 
                label=f'Cluster {cluster}',
                alpha=0.7
            )
        
        # Add labels and legend
        plt.xlabel(data.columns[dims[0]])
        plt.ylabel(data.columns[dims[1]])
        plt.title(f'Clusters using {method}')
        plt.legend()
        plt.grid(alpha=0.3)
        
        return plt.gcf()
    
    def plot_cluster_dendrogram(self, X: pd.DataFrame, standardize: bool = True, 
                             method: str = 'ward', figsize: Tuple = (12, 8)):
        """
        Plot hierarchical clustering dendrogram.
        
        Args:
            X: Feature DataFrame
            standardize: Whether to standardize data
            method: Linkage method
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        data = X.copy()
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
        else:
            data_scaled = data.values
        
        # Calculate linkage
        Z = linkage(data_scaled, method=method)
        
        # Create plot
        plt.figure(figsize=figsize)
        dendrogram(Z)
        plt.title(f'Hierarchical Clustering Dendrogram ({method})')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_optimal_k(self, results: Dict, figsize: Tuple = (12, 8)):
        """
        Plot metrics for optimal k selection.
        
        Args:
            results: Results from find_optimal_k
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        plt.figure(figsize=figsize)
        
        # Plot metrics in subplots
        n_metrics = len(results) - 1  # Subtract 'k' from count
        
        for i, (metric, values) in enumerate(results.items()):
            if metric == 'k':
                continue
                
            plt.subplot(n_metrics, 1, i)
            
            # Plot metric
            plt.plot(results['k'], values, 'o-')
            
            # Handle specific metrics
            if metric == 'inertia':
                # Find elbow point
                from kneed import KneeLocator
                try:
                    kneedle = KneeLocator(
                        results['k'], values, 
                        curve='convex', direction='decreasing'
                    )
                    elbow = kneedle.elbow
                    plt.axvline(x=elbow, color='r', linestyle='--', 
                               label=f'Elbow point: {elbow}')
                    plt.legend()
                except:
                    pass
            
            plt.title(f'{metric.capitalize()} by number of clusters')
            plt.xlabel('Number of clusters')
            plt.ylabel(metric.capitalize())
            plt.grid(alpha=0.3)
        
        plt.tight_layout()
        return plt.gcf()
    
    def analyze_clusters(self, X: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Analyze clusters characteristics.
        
        Args:
            X: Original feature DataFrame
            method: Clustering method
            
        Returns:
            DataFrame with cluster statistics
        """
        if method not in self.clusters:
            raise ValueError(f"Method '{method}' not found in clusters")
        
        clusters = self.clusters[method]
        
        # Initialize results DataFrame
        stats = pd.DataFrame(index=sorted(clusters.unique()))
        
        # Calculate statistics for each feature
        for col in X.columns:
            # Group by cluster
            grouped = X[col].groupby(clusters)
            
            # Calculate statistics
            stats[f'{col}_mean'] = grouped.mean()
            stats[f'{col}_std'] = grouped.std()
            stats[f'{col}_min'] = grouped.min()
            stats[f'{col}_max'] = grouped.max()
        
        # Add count of samples in each cluster
        stats['sample_count'] = clusters.groupby(clusters).count()
        stats['sample_percent'] = 100 * stats['sample_count'] / len(clusters)
        
        return stats

class PatternClusterer:
    """Class for identifying patterns in price time series."""
    
    def __init__(self, window_size: int = 10):
        """
        Initialize the pattern clusterer.
        
        Args:
            window_size: Size of pattern window
        """
        self.window_size = window_size
        self.models = {}
        self.patterns = {}
        self.cluster_maps = {}
    
    def extract_subsequences(self, time_series: pd.Series, 
                           step: int = 1, normalize: bool = True) -> np.ndarray:
        """
        Extract subsequences from time series.
        
        Args:
            time_series: Original time series
            step: Step size for sliding window
            normalize: Whether to normalize subsequences
            
        Returns:
            Array of subsequences
        """
        # Convert to numpy array
        data = time_series.values
        
        # Get subsequences
        n_subsequences = (len(data) - self.window_size) // step + 1
        subsequences = np.zeros((n_subsequences, self.window_size))
        
        for i in range(n_subsequences):
            start_idx = i * step
            end_idx = start_idx + self.window_size
            subsequence = data[start_idx:end_idx]
            
            # Normalize if requested
            if normalize:
                subsequence = (subsequence - np.mean(subsequence)) / np.std(subsequence)
                
            subsequences[i] = subsequence
        
        return subsequences
    
    def cluster_patterns(self, time_series: pd.Series, n_clusters: int = 5, 
                       step: int = 1, normalize: bool = True,
                       method: str = 'kmeans') -> Dict:
        """
        Cluster patterns in time series.
        
        Args:
            time_series: Original time series
            n_clusters: Number of pattern clusters
            step: Step size for sliding window
            normalize: Whether to normalize subsequences
            method: Clustering method
            
        Returns:
            Dictionary with cluster results
        """
        # Extract subsequences
        subsequences = self.extract_subsequences(
            time_series, step=step, normalize=normalize
        )
        
        # Get original indices
        original_indices = np.array([
            range(i * step, i * step + self.window_size)
            for i in range(subsequences.shape[0])
        ])
        
        # Apply clustering
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'kshape':
            from tslearn.clustering import KShape
            # Reshape for KShape
            subsequences_reshaped = subsequences.reshape(
                subsequences.shape[0], subsequences.shape[1], 1
            )
            model = KShape(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(subsequences_reshaped)
            
            # Get cluster centers
            cluster_centers = model.cluster_centers_.reshape(
                model.cluster_centers_.shape[0], model.cluster_centers_.shape[1]
            )
            
            # Store results
            self.models[method] = model
            self.patterns[method] = {
                'subsequences': subsequences,
                'labels': labels,
                'centers': cluster_centers,
                'indices': original_indices
            }
            
            # Create cluster map for original time series
            cluster_map = np.zeros(len(time_series)) - 1  # Initialize with -1 (no cluster)
            for i, idx_range in enumerate(original_indices):
                for idx in idx_range:
                    if idx < len(cluster_map):
                        # If multiple patterns overlap, keep the latest assignment
                        cluster_map[idx] = labels[i]
            
            self.cluster_maps[method] = pd.Series(
                cluster_map, index=time_series.index, name=f'pattern_{method}'
            )
            
            return {
                'subsequences': subsequences,
                'labels': labels,
                'centers': cluster_centers,
                'indices': original_indices,
                'cluster_map': self.cluster_maps[method]
            }
            
        else:
            raise ValueError(f"Method '{method}' not supported")
        
        # Fit model for non-KShape methods
        labels = model.fit_predict(subsequences)
        
        # Get cluster centers
        if hasattr(model, 'cluster_centers_'):
            cluster_centers = model.cluster_centers_
        else:
            # Calculate centers manually
            cluster_centers = np.array([
                subsequences[labels == i].mean(axis=0)
                for i in range(n_clusters)
            ])
        
        # Store results
        self.models[method] = model
        self.patterns[method] = {
            'subsequences': subsequences,
            'labels': labels,
            'centers': cluster_centers,
            'indices': original_indices
        }
        
        # Create cluster map for original time series
        cluster_map = np.zeros(len(time_series)) - 1  # Initialize with -1 (no cluster)
        for i, idx_range in enumerate(original_indices):
            for idx in idx_range:
                if idx < len(cluster_map):
                    # If multiple patterns overlap, keep the latest assignment
                    cluster_map[idx] = labels[i]
        
        self.cluster_maps[method] = pd.Series(
            cluster_map, index=time_series.index, name=f'pattern_{method}'
        )
        
        return {
            'subsequences': subsequences,
            'labels': labels,
            'centers': cluster_centers,
            'indices': original_indices,
            'cluster_map': self.cluster_maps[method]
        }
    
    def plot_pattern_clusters(self, time_series: pd.Series, method: str = 'kmeans',
                           figsize: Tuple = (15, 10)):
        """
        Plot clustered patterns.
        
        Args:
            time_series: Original time series
            method: Clustering method
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        if method not in self.patterns:
            raise ValueError(f"Method '{method}' not found in patterns")
        
        patterns = self.patterns[method]
        centers = patterns['centers']
        n_clusters = centers.shape[0]
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot original time series
        plt.subplot(n_clusters + 1, 1, 1)
        plt.plot(time_series)
        plt.title('Original Time Series')
        plt.grid(alpha=0.3)
        
        # Plot pattern clusters
        for i in range(n_clusters):
            plt.subplot(n_clusters + 1, 1, i + 2)
            
            # Get indices for this cluster
            mask = patterns['labels'] == i
            cluster_indices = patterns['indices'][mask]
            
            # Plot center pattern
            x = np.arange(self.window_size)
            plt.plot(x, centers[i], 'r-', linewidth=2, label='Cluster Center')
            
            # Plot some examples from this cluster
            n_examples = min(10, sum(mask))
            example_indices = np.where(mask)[0][:n_examples]
            
            for idx in example_indices:
                plt.plot(x, patterns['subsequences'][idx], 'k-', alpha=0.2)
            
            plt.title(f'Cluster {i} (n={sum(mask)})')
            plt.grid(alpha=0.3)
            
            if i == 0:
                plt.legend()
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_cluster_map(self, time_series: pd.Series, method: str = 'kmeans',
                      figsize: Tuple = (15, 8)):
        """
        Plot time series with pattern cluster mapping.
        
        Args:
            time_series: Original time series
            method: Clustering method
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        if method not in self.cluster_maps:
            raise ValueError(f"Method '{method}' not found in cluster maps")
        
        cluster_map = self.cluster_maps[method]
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot original time series
        plt.subplot(2, 1, 1)
        plt.plot(time_series)
        plt.title('Original Time Series')
        plt.grid(alpha=0.3)
        
        # Plot cluster map
        plt.subplot(2, 1, 2)
        
        # Get unique clusters
        clusters = sorted(set(cluster_map.dropna().unique()))
        
        # Plot each cluster with a different color
        for cluster in clusters:
            if cluster >= 0:  # Skip unassigned (-1)
                mask = cluster_map == cluster
                plt.scatter(
                    cluster_map.index[mask], 
                    np.zeros(sum(mask)) + cluster,
                    label=f'Cluster {int(cluster)}' if cluster in [0, 1] else None,  # Limit legend
                    alpha=0.7
                )
        
        plt.yticks(clusters)
        plt.title('Pattern Clusters')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        return plt.gcf()
    
    def find_similar_patterns(self, time_series: pd.Series, query_pattern: np.ndarray,
                            method: str = 'kmeans', top_n: int = 5,
                            normalize: bool = True) -> List[Dict]:
        """
        Find patterns similar to a query pattern.
        
        Args:
            time_series: Original time series
            query_pattern: Pattern to search for
            method: Clustering method
            top_n: Number of top matches to return
            normalize: Whether to normalize patterns
            
        Returns:
            List of dictionaries with match information
        """
        if method not in self.patterns:
            raise ValueError(f"Method '{method}' not found in patterns")
        
        patterns = self.patterns[method]
        subsequences = patterns['subsequences']
        
        # Normalize query pattern if requested
        if normalize:
            query_pattern = (query_pattern - np.mean(query_pattern)) / np.std(query_pattern)
        
        # Calculate distances to query pattern
        from scipy.spatial.distance import euclidean
        distances = np.array([
            euclidean(query_pattern, subseq) for subseq in subsequences
        ])
        
        # Get top matches
        top_indices = np.argsort(distances)[:top_n]
        
        # Prepare results
        matches = []
        for idx in top_indices:
            match_info = {
                'distance': distances[idx],
                'subsequence': subsequences[idx],
                'start_idx': patterns['indices'][idx][0],
                'end_idx': patterns['indices'][idx][-1],
                'cluster': patterns['labels'][idx]
            }
            matches.append(match_info)
        
        return matches


# Utility functions

def find_similar_periods(time_series: pd.Series, window_size: int = 20, 
                       top_n: int = 3, method: str = 'euclidean') -> List[Dict]:
    """
    Find periods in a time series that are similar to the most recent period.
    
    Args:
        time_series: Time series data
        window_size: Size of the comparison window
        top_n: Number of top matches to return
        method: Distance metric ('euclidean', 'dtw')
        
    Returns:
        List of dictionaries with similar period information
    """
    # Get the most recent window as the query
    query = time_series[-window_size:].values
    
    # Normalize query
    query_norm = (query - np.mean(query)) / np.std(query)
    
    # Extract all possible windows
    n_windows = len(time_series) - window_size + 1
    distances = []
    
    for i in range(n_windows - 1):  # Exclude the query window
        # Extract window
        window = time_series[i:i+window_size].values
        
        # Normalize window
        window_norm = (window - np.mean(window)) / np.std(window)
        
        # Calculate distance
        if method == 'euclidean':
            from scipy.spatial.distance import euclidean
            dist = euclidean(query_norm, window_norm)
        elif method == 'dtw':
            from scipy.spatial.distance import euclidean
            from tslearn.metrics import dtw
            dist = dtw(query_norm.reshape(-1, 1), window_norm.reshape(-1, 1))
        else:
            raise ValueError(f"Method '{method}' not supported")
        
        distances.append((i, dist))
    
    # Sort by distance
    sorted_distances = sorted(distances, key=lambda x: x[1])
    
    # Get top matches
    matches = []
    for i, (start_idx, dist) in enumerate(sorted_distances[:top_n]):
        match_info = {
            'rank': i + 1,
            'distance': dist,
            'start_idx': start_idx,
            'end_idx': start_idx + window_size - 1,
            'period': time_series.iloc[start_idx:start_idx+window_size]
        }
        matches.append(match_info)
    
    return matches

def detect_regime_changes(time_series: pd.Series, feature_windows: List[int] = [5, 10, 20, 50],
                        n_regimes: int = 3, standardize: bool = True) -> pd.Series:
    """
    Detect regime changes in a time series.
    
    Args:
        time_series: Time series data
        feature_windows: Window sizes for feature extraction
        n_regimes: Number of regimes to identify
        standardize: Whether to standardize features
        
    Returns:
        Series with regime labels
    """
    # Create features DataFrame
    features = pd.DataFrame(index=time_series.index)
    
    # Extract features
    for w in feature_windows:
        # Add rolling statistics
        features[f'mean_{w}'] = time_series.rolling(window=w).mean()
        features[f'std_{w}'] = time_series.rolling(window=w).std()
        features[f'skew_{w}'] = time_series.rolling(window=w).skew()
        
        # Add returns
        features[f'return_{w}'] = time_series.pct_change(w)
        
        # Add rolling min/max
        features[f'min_{w}'] = time_series.rolling(window=w).min()
        features[f'max_{w}'] = time_series.rolling(window=w).max()
    
    # Drop NaN values
    features = features.dropna()
    
    # Apply clustering
    clusterer = MarketRegimeClusterer()
    regimes = clusterer.cluster_kmeans(
        features, n_clusters=n_regimes, standardize=standardize
    )
    
    return regimes 