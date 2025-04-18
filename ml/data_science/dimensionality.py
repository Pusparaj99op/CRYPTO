"""
Dimensionality Reduction Module for Cryptocurrency Trading

This module provides dimensionality reduction techniques to transform
high-dimensional feature spaces into lower dimensions while preserving
important characteristics of the data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional
from sklearn.decomposition import (
    PCA, KernelPCA, SparsePCA, FastICA, FactorAnalysis, 
    LatentDirichletAllocation, NMF
)
from sklearn.manifold import (
    TSNE, LocallyLinearEmbedding, Isomap, SpectralEmbedding, MDS
)
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
import umap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


class DimensionalityReducer:
    """Class for reducing dimensionality of cryptocurrency trading features."""
    
    def __init__(self):
        """Initialize the dimensionality reducer."""
        self.fitted_models = {}
        self.transformed_data = {}
        self.explained_variance = {}
        self.feature_weights = {}
    
    def fit_pca(self, X: pd.DataFrame, n_components: Optional[int] = None, 
               whiten: bool = False, standardize: bool = True) -> pd.DataFrame:
        """
        Apply Principal Component Analysis (PCA).
        
        Args:
            X: Feature DataFrame
            n_components: Number of components to keep
            whiten: Apply whitening to decorrelate components
            standardize: Standardize features before PCA
            
        Returns:
            DataFrame with PCA-transformed features
        """
        data = X.copy()
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            self.fitted_models['scaler'] = scaler
        else:
            data_scaled = data.values
        
        # Determine number of components
        if n_components is None:
            n_components = min(data.shape[0], data.shape[1])
        
        # Fit PCA
        pca = PCA(n_components=n_components, whiten=whiten, random_state=42)
        transformed = pca.fit_transform(data_scaled)
        
        # Store model and results
        self.fitted_models['pca'] = pca
        
        # Calculate explained variance
        explained_var = pca.explained_variance_ratio_
        self.explained_variance['pca'] = explained_var
        
        # Feature loadings/weights
        feature_weights = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=data.columns
        )
        self.feature_weights['pca'] = feature_weights
        
        # Create and return DataFrame
        transformed_df = pd.DataFrame(
            transformed, 
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=data.index
        )
        self.transformed_data['pca'] = transformed_df
        
        return transformed_df
    
    def fit_kernel_pca(self, X: pd.DataFrame, n_components: int = 2, 
                      kernel: str = 'rbf', gamma: float = None, 
                      standardize: bool = True) -> pd.DataFrame:
        """
        Apply Kernel PCA for nonlinear dimensionality reduction.
        
        Args:
            X: Feature DataFrame
            n_components: Number of components to keep
            kernel: Kernel type ('rbf', 'poly', 'sigmoid', 'cosine')
            gamma: Kernel coefficient
            standardize: Standardize features before KPCA
            
        Returns:
            DataFrame with KPCA-transformed features
        """
        data = X.copy()
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            self.fitted_models['scaler'] = scaler
        else:
            data_scaled = data.values
        
        # Fit Kernel PCA
        kpca = KernelPCA(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            random_state=42
        )
        transformed = kpca.fit_transform(data_scaled)
        
        # Store model and results
        self.fitted_models['kpca'] = kpca
        
        # Create and return DataFrame
        transformed_df = pd.DataFrame(
            transformed, 
            columns=[f'KPCA{i+1}' for i in range(n_components)],
            index=data.index
        )
        self.transformed_data['kpca'] = transformed_df
        
        return transformed_df
    
    def fit_tsne(self, X: pd.DataFrame, n_components: int = 2, 
                perplexity: float = 30.0, learning_rate: float = 200.0,
                early_exaggeration: float = 12.0, 
                standardize: bool = True) -> pd.DataFrame:
        """
        Apply t-SNE for visualization of high-dimensional data.
        
        Args:
            X: Feature DataFrame
            n_components: Number of dimensions (usually 2 or 3)
            perplexity: Related to number of nearest neighbors
            learning_rate: Learning rate for optimization
            early_exaggeration: Early exaggeration factor
            standardize: Standardize features before t-SNE
            
        Returns:
            DataFrame with t-SNE-transformed features
        """
        data = X.copy()
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            self.fitted_models['scaler'] = scaler
        else:
            data_scaled = data.values
        
        # Fit t-SNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            early_exaggeration=early_exaggeration,
            random_state=42
        )
        transformed = tsne.fit_transform(data_scaled)
        
        # Store model and results
        self.fitted_models['tsne'] = tsne
        
        # Create and return DataFrame
        transformed_df = pd.DataFrame(
            transformed, 
            columns=[f'TSNE{i+1}' for i in range(n_components)],
            index=data.index
        )
        self.transformed_data['tsne'] = transformed_df
        
        return transformed_df
    
    def fit_umap(self, X: pd.DataFrame, n_components: int = 2, 
               n_neighbors: int = 15, min_dist: float = 0.1,
               metric: str = 'euclidean', standardize: bool = True) -> pd.DataFrame:
        """
        Apply UMAP for visualization and dimensionality reduction.
        
        Args:
            X: Feature DataFrame
            n_components: Number of dimensions
            n_neighbors: Number of neighbors for local approximation
            min_dist: Minimum distance between points
            metric: Distance metric
            standardize: Standardize features before UMAP
            
        Returns:
            DataFrame with UMAP-transformed features
        """
        data = X.copy()
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            self.fitted_models['scaler'] = scaler
        else:
            data_scaled = data.values
        
        # Fit UMAP
        mapper = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42
        )
        transformed = mapper.fit_transform(data_scaled)
        
        # Store model and results
        self.fitted_models['umap'] = mapper
        
        # Create and return DataFrame
        transformed_df = pd.DataFrame(
            transformed, 
            columns=[f'UMAP{i+1}' for i in range(n_components)],
            index=data.index
        )
        self.transformed_data['umap'] = transformed_df
        
        return transformed_df
    
    def fit_ica(self, X: pd.DataFrame, n_components: int, 
               algorithm: str = 'parallel', standardize: bool = True) -> pd.DataFrame:
        """
        Apply Independent Component Analysis (ICA).
        
        Args:
            X: Feature DataFrame
            n_components: Number of independent components
            algorithm: ICA algorithm ('parallel', 'deflation')
            standardize: Standardize features before ICA
            
        Returns:
            DataFrame with ICA-transformed features
        """
        data = X.copy()
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            self.fitted_models['scaler'] = scaler
        else:
            data_scaled = data.values
        
        # Fit ICA
        ica = FastICA(
            n_components=n_components,
            algorithm=algorithm,
            random_state=42
        )
        transformed = ica.fit_transform(data_scaled)
        
        # Store model and results
        self.fitted_models['ica'] = ica
        
        # Feature mixing matrix
        mixing_matrix = pd.DataFrame(
            ica.mixing_,
            columns=[f'IC{i+1}' for i in range(n_components)],
            index=data.columns
        )
        self.feature_weights['ica'] = mixing_matrix
        
        # Create and return DataFrame
        transformed_df = pd.DataFrame(
            transformed, 
            columns=[f'IC{i+1}' for i in range(n_components)],
            index=data.index
        )
        self.transformed_data['ica'] = transformed_df
        
        return transformed_df
    
    def fit_factor_analysis(self, X: pd.DataFrame, n_components: int, 
                           rotation: str = 'varimax', 
                           standardize: bool = True) -> pd.DataFrame:
        """
        Apply Factor Analysis.
        
        Args:
            X: Feature DataFrame
            n_components: Number of factors
            rotation: Factor rotation method
            standardize: Standardize features before analysis
            
        Returns:
            DataFrame with factor-transformed features
        """
        data = X.copy()
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            self.fitted_models['scaler'] = scaler
        else:
            data_scaled = data.values
        
        # Fit Factor Analysis
        fa = FactorAnalysis(
            n_components=n_components,
            random_state=42
        )
        transformed = fa.fit_transform(data_scaled)
        
        # Apply rotation if requested
        if rotation == 'varimax':
            from scipy.stats import special_ortho_group
            from scipy.linalg import svd
            
            # Varimax rotation
            components = fa.components_
            n_features = components.shape[1]
            
            # Kaiser normalization
            h = np.sqrt(np.sum(components**2, axis=0))
            A = components / h
            
            # Varimax rotation
            A_rotated = self._varimax_rotation(A)
            
            # De-normalize
            components_rotated = A_rotated * h
            fa.components_ = components_rotated
            
            # Recalculate transformed data
            transformed = data_scaled @ fa.components_.T
        
        # Store model and results
        self.fitted_models['factor_analysis'] = fa
        
        # Store factor loadings
        loadings = pd.DataFrame(
            fa.components_.T,
            columns=[f'Factor{i+1}' for i in range(n_components)],
            index=data.columns
        )
        self.feature_weights['factor_analysis'] = loadings
        
        # Create and return DataFrame
        transformed_df = pd.DataFrame(
            transformed, 
            columns=[f'Factor{i+1}' for i in range(n_components)],
            index=data.index
        )
        self.transformed_data['factor_analysis'] = transformed_df
        
        return transformed_df
    
    def fit_nmf(self, X: pd.DataFrame, n_components: int, 
              init: str = 'nndsvd', standardize: bool = False) -> pd.DataFrame:
        """
        Apply Non-negative Matrix Factorization.
        
        Args:
            X: Feature DataFrame (must be non-negative)
            n_components: Number of components
            init: Initialization method
            standardize: Standardize features before NMF
            
        Returns:
            DataFrame with NMF-transformed features
        """
        data = X.copy()
        
        # Ensure non-negative data
        if (data.values < 0).any():
            raise ValueError("NMF requires non-negative data")
        
        # Standardize if requested (maintaining non-negativity)
        if standardize:
            # Min-max scaling to [0, 1]
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)
            self.fitted_models['scaler'] = scaler
        else:
            data_scaled = data.values
        
        # Fit NMF
        nmf = NMF(
            n_components=n_components,
            init=init,
            random_state=42
        )
        transformed = nmf.fit_transform(data_scaled)
        
        # Store model and results
        self.fitted_models['nmf'] = nmf
        
        # Store components
        components = pd.DataFrame(
            nmf.components_,
            columns=data.columns,
            index=[f'Component{i+1}' for i in range(n_components)]
        )
        self.feature_weights['nmf'] = components
        
        # Create and return DataFrame
        transformed_df = pd.DataFrame(
            transformed, 
            columns=[f'NMF{i+1}' for i in range(n_components)],
            index=data.index
        )
        self.transformed_data['nmf'] = transformed_df
        
        return transformed_df
    
    def fit_isomap(self, X: pd.DataFrame, n_components: int = 2, 
                 n_neighbors: int = 5, standardize: bool = True) -> pd.DataFrame:
        """
        Apply Isomap embedding.
        
        Args:
            X: Feature DataFrame
            n_components: Number of dimensions
            n_neighbors: Number of neighbors to consider
            standardize: Standardize features before Isomap
            
        Returns:
            DataFrame with Isomap-transformed features
        """
        data = X.copy()
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            self.fitted_models['scaler'] = scaler
        else:
            data_scaled = data.values
        
        # Fit Isomap
        isomap = Isomap(
            n_components=n_components,
            n_neighbors=n_neighbors
        )
        transformed = isomap.fit_transform(data_scaled)
        
        # Store model and results
        self.fitted_models['isomap'] = isomap
        
        # Create and return DataFrame
        transformed_df = pd.DataFrame(
            transformed, 
            columns=[f'Isomap{i+1}' for i in range(n_components)],
            index=data.index
        )
        self.transformed_data['isomap'] = transformed_df
        
        return transformed_df
    
    def fit_lle(self, X: pd.DataFrame, n_components: int = 2, 
              n_neighbors: int = 5, method: str = 'standard',
              standardize: bool = True) -> pd.DataFrame:
        """
        Apply Locally Linear Embedding.
        
        Args:
            X: Feature DataFrame
            n_components: Number of dimensions
            n_neighbors: Number of neighbors to consider
            method: LLE method ('standard', 'modified', 'hessian', 'ltsa')
            standardize: Standardize features before LLE
            
        Returns:
            DataFrame with LLE-transformed features
        """
        data = X.copy()
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            self.fitted_models['scaler'] = scaler
        else:
            data_scaled = data.values
        
        # Fit LLE
        lle = LocallyLinearEmbedding(
            n_components=n_components,
            n_neighbors=n_neighbors,
            method=method,
            random_state=42
        )
        transformed = lle.fit_transform(data_scaled)
        
        # Store model and results
        self.fitted_models['lle'] = lle
        
        # Create and return DataFrame
        transformed_df = pd.DataFrame(
            transformed, 
            columns=[f'LLE{i+1}' for i in range(n_components)],
            index=data.index
        )
        self.transformed_data['lle'] = transformed_df
        
        return transformed_df
    
    def fit_random_projection(self, X: pd.DataFrame, n_components: int,
                            eps: float = 0.1, standardize: bool = True) -> pd.DataFrame:
        """
        Apply Random Projection.
        
        Args:
            X: Feature DataFrame
            n_components: Number of dimensions
            eps: Error tolerance
            standardize: Standardize features before projection
            
        Returns:
            DataFrame with randomly projected features
        """
        data = X.copy()
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            self.fitted_models['scaler'] = scaler
        else:
            data_scaled = data.values
        
        # Fit Random Projection
        rp = GaussianRandomProjection(
            n_components=n_components,
            eps=eps,
            random_state=42
        )
        transformed = rp.fit_transform(data_scaled)
        
        # Store model and results
        self.fitted_models['random_projection'] = rp
        
        # Create and return DataFrame
        transformed_df = pd.DataFrame(
            transformed, 
            columns=[f'RP{i+1}' for i in range(n_components)],
            index=data.index
        )
        self.transformed_data['random_projection'] = transformed_df
        
        return transformed_df
    
    def plot_explained_variance(self, method: str = 'pca', cumulative: bool = True, figsize: Tuple = (10, 6)):
        """
        Plot explained variance for methods that provide it.
        
        Args:
            method: Method to plot variance for
            cumulative: Whether to plot cumulative variance
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        if method not in self.explained_variance:
            raise ValueError(f"Method '{method}' doesn't have stored explained variance")
        
        # Get explained variance
        exp_var = self.explained_variance[method]
        n_components = len(exp_var)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot individual variance
        plt.bar(range(1, n_components + 1), exp_var, alpha=0.5, 
                label='Individual explained variance')
        
        # Plot cumulative variance if requested
        if cumulative:
            plt.step(range(1, n_components + 1), np.cumsum(exp_var), 
                    where='mid', label='Cumulative explained variance')
        
        # Add reference lines
        plt.axhline(y=0.9, linewidth=1, color='r', linestyle='--', 
                   label='90% explained variance')
        plt.axhline(y=0.95, linewidth=1, color='g', linestyle='--', 
                   label='95% explained variance')
        
        # Add labels and legend
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance ratio')
        plt.title(f'Explained variance ({method.upper()})')
        plt.legend(loc='best')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_2d_projection(self, labels: Optional[pd.Series] = None,
                         method: str = 'pca', figsize: Tuple = (10, 8),
                         interactive: bool = False):
        """
        Plot 2D projection of data.
        
        Args:
            labels: Optional series of labels for coloring
            method: Dimensionality reduction method to plot
            figsize: Figure size
            interactive: Whether to use plotly for interactive plot
            
        Returns:
            matplotlib figure or plotly figure
        """
        if method not in self.transformed_data:
            raise ValueError(f"Method '{method}' not found in transformed data")
        
        # Get transformed data
        transformed = self.transformed_data[method]
        
        # Ensure we have at least 2 components
        if transformed.shape[1] < 2:
            raise ValueError(f"Need at least 2 components, got {transformed.shape[1]}")
        
        # Get first two components
        x = transformed.iloc[:, 0]
        y = transformed.iloc[:, 1]
        
        # Create plot
        if interactive:
            # Interactive plotly plot
            if labels is not None:
                fig = px.scatter(
                    x=x, y=y, color=labels, 
                    labels={'x': transformed.columns[0], 'y': transformed.columns[1]},
                    title=f'2D Projection using {method.upper()}'
                )
            else:
                fig = px.scatter(
                    x=x, y=y,
                    labels={'x': transformed.columns[0], 'y': transformed.columns[1]},
                    title=f'2D Projection using {method.upper()}'
                )
            return fig
        else:
            # Static matplotlib plot
            plt.figure(figsize=figsize)
            
            if labels is not None:
                # Plot with color by labels
                scatter = plt.scatter(x, y, c=labels, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, label='Labels')
            else:
                plt.scatter(x, y, alpha=0.7)
                
            plt.xlabel(transformed.columns[0])
            plt.ylabel(transformed.columns[1])
            plt.title(f'2D Projection using {method.upper()}')
            plt.tight_layout()
            
            return plt.gcf()
    
    def plot_feature_weights(self, method: str = 'pca', n_components: int = 2, 
                           n_features: int = 10, figsize: Tuple = (12, 10)):
        """
        Plot feature weights/loadings for a method.
        
        Args:
            method: Dimensionality reduction method
            n_components: Number of components to plot
            n_features: Number of top features to show
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        if method not in self.feature_weights:
            raise ValueError(f"Feature weights not available for method '{method}'")
        
        # Get feature weights
        weights = self.feature_weights[method]
        
        # Limit to n_components
        n_components = min(n_components, weights.shape[1])
        weights = weights.iloc[:, :n_components]
        
        # For each component, get top features by absolute weight
        top_features = {}
        for col in weights.columns:
            abs_weights = weights[col].abs()
            top_features[col] = abs_weights.nlargest(n_features).index.tolist()
        
        # Create a new DataFrame with just the top features
        unique_features = list(set(feature for features in top_features.values() for feature in features))
        weights_subset = weights.loc[unique_features]
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Heatmap of weights
        plt.imshow(weights_subset, cmap='coolwarm', aspect='auto')
        
        # Add labels
        plt.colorbar(label='Weight/Loading')
        plt.xticks(range(n_components), weights.columns[:n_components], rotation=45)
        plt.yticks(range(len(unique_features)), unique_features)
        plt.xlabel('Components')
        plt.ylabel('Features')
        plt.title(f'Feature Weights/Loadings for {method.upper()}')
        plt.tight_layout()
        
        return plt.gcf()
    
    def _varimax_rotation(self, loadings, max_iter=100, tol=1e-6):
        """Helper method for varimax rotation."""
        n_rows, n_cols = loadings.shape
        rotation_matrix = np.eye(n_cols)
        
        d = 0
        for _ in range(max_iter):
            old_d = d
            
            # Step 1: Standardize columns (rows for components matrix)
            B = loadings @ rotation_matrix
            
            # Step 2: Project onto "Varimax Function"
            z = B**3 - (1.0/n_rows) * np.sum(B**2, axis=0) * B
            
            # Step 3: Calculate SVD
            u, s, v = np.linalg.svd(loadings.T @ z)
            
            # Step 4: Update rotation matrix
            rotation_matrix = u @ v
            
            # Calculate variance to check for convergence
            d = np.sum(s)
            
            # Check convergence
            if old_d != 0 and d/old_d < 1 + tol:
                break
                
        # Return rotated loadings
        return loadings @ rotation_matrix
    
    def transform_new_data(self, X: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Transform new data using fitted model.
        
        Args:
            X: New data to transform
            method: Method to use for transformation
            
        Returns:
            Transformed DataFrame
        """
        if method not in self.fitted_models:
            raise ValueError(f"Method '{method}' not found in fitted models")
        
        # Get model
        model = self.fitted_models[method]
        
        # Standardize if scaler is available
        if 'scaler' in self.fitted_models:
            X_scaled = self.fitted_models['scaler'].transform(X)
        else:
            X_scaled = X.values
        
        # Transform data
        transformed = model.transform(X_scaled)
        
        # Create column names based on method
        if method == 'pca':
            cols = [f'PC{i+1}' for i in range(transformed.shape[1])]
        elif method == 'kpca':
            cols = [f'KPCA{i+1}' for i in range(transformed.shape[1])]
        elif method == 'ica':
            cols = [f'IC{i+1}' for i in range(transformed.shape[1])]
        elif method == 'nmf':
            cols = [f'NMF{i+1}' for i in range(transformed.shape[1])]
        elif method == 'factor_analysis':
            cols = [f'Factor{i+1}' for i in range(transformed.shape[1])]
        else:
            cols = [f'{method.upper()}{i+1}' for i in range(transformed.shape[1])]
        
        # Create and return DataFrame
        transformed_df = pd.DataFrame(
            transformed, 
            columns=cols,
            index=X.index
        )
        
        return transformed_df


# Utility functions

def find_optimal_components(X: pd.DataFrame, max_components: int = 20, 
                           method: str = 'pca', threshold: float = 0.95) -> int:
    """
    Find optimal number of components based on explained variance.
    
    Args:
        X: Feature DataFrame
        max_components: Maximum number of components to consider
        method: Dimensionality reduction method
        threshold: Explained variance threshold
        
    Returns:
        Optimal number of components
    """
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if method == 'pca':
        model = PCA(n_components=min(max_components, X.shape[1]))
        model.fit(X_scaled)
        explained_var = model.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        # Find first component that exceeds threshold
        optimal_components = np.argmax(cumulative_var >= threshold) + 1
        
    elif method == 'factor_analysis':
        # For factor analysis, use BIC to find optimal components
        from sklearn.decomposition import FactorAnalysis
        
        bic = []
        n_components_range = range(1, min(max_components, X.shape[1]) + 1)
        
        for n in n_components_range:
            fa = FactorAnalysis(n_components=n)
            fa.fit(X_scaled)
            
            # Calculate BIC approximation
            n_samples, n_features = X_scaled.shape
            ll = fa.score(X_scaled)
            bic.append(-2 * ll + np.log(n_samples) * (n_features * n))
        
        # Find components with minimum BIC
        optimal_components = n_components_range[np.argmin(bic)]
        
    else:
        raise ValueError(f"Method '{method}' not supported for optimal component selection")
    
    return optimal_components

def time_series_dimensionality_reduction(df: pd.DataFrame, window_size: int = 10, 
                                        n_components: int = 2, standardize: bool = True):
    """
    Apply dimensionality reduction to rolling windows of time series.
    
    Args:
        df: Time series DataFrame
        window_size: Size of rolling window
        n_components: Number of components to extract
        standardize: Whether to standardize data
        
    Returns:
        DataFrame with time series of principal components
    """
    # Initialize results
    results = pd.DataFrame(index=df.index)
    
    # Loop through windows
    for i in range(window_size, len(df) + 1):
        # Get window
        window = df.iloc[i-window_size:i]
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            window_scaled = scaler.fit_transform(window)
        else:
            window_scaled = window.values
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca.fit(window_scaled)
        
        # Store explained variance for last point in window
        for j in range(n_components):
            results.loc[window.index[-1], f'PC{j+1}_var'] = pca.explained_variance_ratio_[j]
        
        # Store loadings for first component (most important direction)
        for j, feature in enumerate(df.columns):
            results.loc[window.index[-1], f'{feature}_loading'] = pca.components_[0, j]
    
    return results.dropna()

def plot_3d_projection(X_transformed: pd.DataFrame, labels: Optional[pd.Series] = None,
                     title: str = '3D Projection', interactive: bool = True):
    """
    Create a 3D visualization of transformed data.
    
    Args:
        X_transformed: Transformed data (at least 3 dimensions)
        labels: Optional labels for coloring points
        title: Plot title
        interactive: Whether to use plotly for interactive 3D plot
        
    Returns:
        Plotly figure or Matplotlib figure
    """
    if X_transformed.shape[1] < 3:
        raise ValueError(f"Need at least 3 dimensions, got {X_transformed.shape[1]}")
    
    # Get first three components
    x = X_transformed.iloc[:, 0]
    y = X_transformed.iloc[:, 1]
    z = X_transformed.iloc[:, 2]
    
    if interactive:
        # Create interactive plotly plot
        if labels is not None:
            fig = px.scatter_3d(
                x=x, y=y, z=z, color=labels,
                labels={
                    'x': X_transformed.columns[0],
                    'y': X_transformed.columns[1],
                    'z': X_transformed.columns[2]
                },
                title=title
            )
        else:
            fig = px.scatter_3d(
                x=x, y=y, z=z,
                labels={
                    'x': X_transformed.columns[0],
                    'y': X_transformed.columns[1],
                    'z': X_transformed.columns[2]
                },
                title=title
            )
        
        # Improve layout
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
        
        return fig
    
    else:
        # Create static matplotlib plot
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if labels is not None:
            scatter = ax.scatter(x, y, z, c=labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Labels')
        else:
            ax.scatter(x, y, z, alpha=0.7)
        
        ax.set_xlabel(X_transformed.columns[0])
        ax.set_ylabel(X_transformed.columns[1])
        ax.set_zlabel(X_transformed.columns[2])
        ax.set_title(title)
        
        return fig 