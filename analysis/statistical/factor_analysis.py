"""
Factor Analysis Module

This module provides functions for performing factor analysis and principal
component analysis on cryptocurrency data.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy import stats

def perform_pca(data: pd.DataFrame,
               n_components: int = None,
               explained_variance: float = 0.95) -> Dict[str, Union[np.ndarray, float]]:
    """
    Perform Principal Component Analysis.
    
    Args:
        data: DataFrame with features
        n_components: Number of components to keep
        explained_variance: Minimum explained variance ratio
        
    Returns:
        Dictionary containing PCA results
    """
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled_data)
    
    # Calculate explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    return {
        'components': components,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'loadings': pca.components_,
        'eigenvalues': pca.explained_variance_
    }

def perform_factor_analysis(data: pd.DataFrame,
                          n_factors: int = None,
                          rotation: str = 'varimax') -> Dict[str, Union[np.ndarray, float]]:
    """
    Perform Factor Analysis.
    
    Args:
        data: DataFrame with features
        n_factors: Number of factors to extract
        rotation: Rotation method ('varimax', 'promax', 'quartimax')
        
    Returns:
        Dictionary containing factor analysis results
    """
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Perform Factor Analysis
    fa = FactorAnalysis(n_components=n_factors, rotation=rotation)
    factors = fa.fit_transform(scaled_data)
    
    return {
        'factors': factors,
        'loadings': fa.components_,
        'noise_variance': fa.noise_variance_,
        'log_likelihood': fa.score(scaled_data)
    }

def calculate_factor_loadings(data: pd.DataFrame,
                            factors: np.ndarray) -> pd.DataFrame:
    """
    Calculate factor loadings matrix.
    
    Args:
        data: Original feature data
        factors: Extracted factors
        
    Returns:
        DataFrame with factor loadings
    """
    # Calculate correlation between factors and original variables
    loadings = np.corrcoef(data.T, factors.T)[:data.shape[1], data.shape[1]:]
    
    return pd.DataFrame(
        loadings,
        index=data.columns,
        columns=[f'Factor_{i+1}' for i in range(factors.shape[1])]
    )

def perform_factor_rotation(loadings: np.ndarray,
                          method: str = 'varimax') -> np.ndarray:
    """
    Perform rotation on factor loadings.
    
    Args:
        loadings: Factor loadings matrix
        method: Rotation method ('varimax', 'promax', 'quartimax')
        
    Returns:
        Rotated factor loadings
    """
    if method == 'varimax':
        # Implement varimax rotation
        k = loadings.shape[1]
        n = loadings.shape[0]
        
        # Normalize loadings
        normalized_loadings = loadings / np.sqrt(np.sum(loadings**2, axis=0))
        
        # Initialize rotation matrix
        rotation_matrix = np.eye(k)
        
        # Iterative rotation
        for _ in range(100):
            rotated_loadings = np.dot(normalized_loadings, rotation_matrix)
            new_rotation = np.dot(
                normalized_loadings.T,
                rotated_loadings**3 - np.dot(rotated_loadings, np.diag(np.sum(rotated_loadings**2, axis=0))) / n
            )
            u, s, v = np.linalg.svd(new_rotation)
            rotation_matrix = np.dot(u, v)
        
        return np.dot(loadings, rotation_matrix)
    
    elif method == 'promax':
        # Implement promax rotation
        # (Implementation would be similar but with power transformation)
        pass
    
    return loadings

def calculate_factor_scores(data: pd.DataFrame,
                          loadings: np.ndarray) -> np.ndarray:
    """
    Calculate factor scores for observations.
    
    Args:
        data: Original feature data
        loadings: Factor loadings matrix
        
    Returns:
        Factor scores
    """
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Calculate factor scores using regression method
    factor_scores = np.dot(
        np.dot(scaled_data, loadings),
        np.linalg.inv(np.dot(loadings.T, loadings))
    )
    
    return factor_scores 