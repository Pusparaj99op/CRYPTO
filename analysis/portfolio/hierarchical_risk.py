"""
Hierarchical Risk Parity implementation for cryptocurrency portfolios.
This module provides functions to implement Hierarchical Risk Parity (HRP),
a portfolio optimization technique introduced by Marcos Lopez de Prado that uses
hierarchical clustering to build a diversified portfolio without the need for
inverting a covariance matrix, making it more robust for cryptocurrency markets.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

def get_correlation_distance(corr):
    """
    Convert a correlation matrix to a distance matrix.
    
    Parameters:
    -----------
    corr : pd.DataFrame
        Correlation matrix
        
    Returns:
    --------
    np.array
        Distance matrix
    """
    # Ensure the correlation matrix is valid (values between -1 and 1)
    corr = np.clip(corr, -1.0, 1.0)
    
    # Convert correlation to distance: distance = sqrt(0.5*(1-correlation))
    dist = np.sqrt(0.5 * (1 - corr))
    return dist

def get_quasi_diag(link):
    """
    Quasi-diagonalization using the dendrogram structure.
    
    Parameters:
    -----------
    link : np.array
        Linkage matrix from hierarchical clustering
        
    Returns:
    --------
    list
        List of indices in quasi-diagonal form
    """
    # Initialize with empty list
    link = link.astype(int)
    sorted_idx = []
    num_items = link[-1, 3]  # number of original items
    
    # Recursive function to get the order of indices
    def get_quasi_diag_recursive(node_id):
        if node_id < num_items:
            return [node_id]  # Terminal node
        else:
            # Recursively get left and right children
            left = int(link[node_id - num_items, 0])
            right = int(link[node_id - num_items, 1])
            return get_quasi_diag_recursive(left) + get_quasi_diag_recursive(right)
    
    # Get recursion started
    sorted_idx = get_quasi_diag_recursive(2 * num_items - 2)
    return sorted_idx

def get_cluster_var(cov, cluster_items):
    """
    Compute the variance of a cluster based on its constituents.
    
    Parameters:
    -----------
    cov : pd.DataFrame
        Covariance matrix
    cluster_items : list
        List of indices belonging to the cluster
        
    Returns:
    --------
    float
        Cluster variance
    """
    # Get the sub-covariance matrix for this cluster
    cov_slice = cov.iloc[cluster_items, cluster_items]
    
    # Calculate the variance
    cluster_var = 0
    for i in range(len(cluster_items)):
        for j in range(len(cluster_items)):
            cluster_var += cov_slice.iloc[i, j]
    
    return cluster_var

def get_rec_bipart(cov, sorted_idx):
    """
    Compute weights using recursive bisection based on the clustering.
    
    Parameters:
    -----------
    cov : pd.DataFrame
        Covariance matrix
    sorted_idx : list
        List of indices in quasi-diagonal form
        
    Returns:
    --------
    pd.Series
        Optimal weights for each asset
    """
    # Initialize weights with ones
    weights = pd.Series(1, index=cov.index)
    
    # Recursive function to assign weights
    def bipart_weights(curr_idx, curr_weights):
        if len(curr_idx) == 1:
            return
        
        # Divide into two clusters
        mid = len(curr_idx) // 2
        left_idx = curr_idx[:mid]
        right_idx = curr_idx[mid:]
        
        # Get left and right sub-cluster variances
        left_var = get_cluster_var(cov, left_idx)
        right_var = get_cluster_var(cov, right_idx)
        
        # Calculate alpha (weight scalar) based on inverse variance
        left_alpha = 1 - left_var / (left_var + right_var)
        right_alpha = 1 - left_alpha
        
        # Adjust weights based on alpha
        curr_weights[left_idx] *= left_alpha
        curr_weights[right_idx] *= right_alpha
        
        # Recursively apply to sub-clusters
        bipart_weights(left_idx, curr_weights)
        bipart_weights(right_idx, curr_weights)
    
    # Start recursion
    bipart_weights(sorted_idx, weights)
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    return weights

def hrp_portfolio(returns):
    """
    Construct a portfolio using the Hierarchical Risk Parity algorithm.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
        
    Returns:
    --------
    dict
        Dictionary with optimal weights and portfolio metrics
    """
    # Calculate correlation and covariance matrices
    corr = returns.corr()
    cov = returns.cov()
    
    # Convert correlation to distance
    dist = get_correlation_distance(corr)
    
    # Perform hierarchical clustering
    link = linkage(squareform(dist), method='single')
    
    # Quasi-diagonalize the covariance matrix
    sorted_idx = get_quasi_diag(link)
    
    # Map sorted_idx to original asset names
    original_assets = returns.columns.tolist()
    sorted_assets = [original_assets[i] for i in sorted_idx]
    
    # Recursive bisection
    weights = get_rec_bipart(cov.loc[sorted_assets, sorted_assets], list(range(len(sorted_assets))))
    
    # Convert weights series to use original asset names
    weights.index = sorted_assets
    
    # Reorder weights to match the original order of assets
    weights = weights.reindex(original_assets)
    
    # Calculate portfolio metrics
    weights_array = weights.values
    expected_return = np.sum(returns.mean() * weights_array) * 252  # Annualized
    expected_volatility = np.sqrt(np.dot(weights_array, np.dot(cov, weights_array))) * np.sqrt(252)  # Annualized
    
    return {
        'weights': weights,
        'expected_return': expected_return,
        'expected_volatility': expected_volatility
    }

def plot_dendogram(returns, figsize=(12, 8)):
    """
    Plot the dendrogram of the hierarchical clustering of assets.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    figsize : tuple, optional
        Figure size, by default (12, 8)
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object of the dendrogram
    """
    # Calculate correlation matrix
    corr = returns.corr()
    
    # Convert correlation to distance
    dist = get_correlation_distance(corr)
    
    # Perform hierarchical clustering
    link = linkage(squareform(dist), method='single')
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the dendrogram
    dendrogram(link, labels=returns.columns, ax=ax)
    
    # Add title and labels
    plt.title('Hierarchical Clustering of Assets', fontsize=16)
    plt.xlabel('Assets', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.xticks(rotation=90)
    
    plt.tight_layout()
    return fig

def cluster_assets(returns, n_clusters=None):
    """
    Cluster assets based on their return correlation structure.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    n_clusters : int, optional
        Number of clusters to form, by default None (auto-determined)
        
    Returns:
    --------
    dict
        Dictionary mapping cluster number to list of assets in that cluster
    """
    # Calculate correlation matrix
    corr = returns.corr()
    
    # Convert correlation to distance
    dist = get_correlation_distance(corr)
    
    # Perform hierarchical clustering
    link = linkage(squareform(dist), method='single')
    
    # If n_clusters is not specified, try to determine automatically
    if n_clusters is None:
        # A simple heuristic: number of clusters = sqrt(number of assets)
        n_clusters = max(2, int(np.sqrt(len(returns.columns))))
    
    # Cut the tree to get clusters
    clusters = cut_tree(link, n_clusters=n_clusters).flatten()
    
    # Create dictionary mapping cluster number to list of assets
    cluster_dict = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(returns.columns[i])
    
    return cluster_dict

def multi_level_hrp(returns, top_level_clusters=None):
    """
    Implement a multi-level HRP approach that first allocates across clusters,
    then within each cluster.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    top_level_clusters : int, optional
        Number of top-level clusters, by default None (auto-determined)
        
    Returns:
    --------
    dict
        Dictionary with optimal weights and portfolio metrics
    """
    # Cluster the assets
    clusters = cluster_assets(returns, n_clusters=top_level_clusters)
    
    # Initialize dictionaries for cluster portfolios
    cluster_weights = {}
    cluster_returns = {}
    cluster_volatilities = {}
    
    # For each cluster, calculate HRP weights
    for cluster_id, assets in clusters.items():
        # Skip if cluster has only one asset
        if len(assets) == 1:
            cluster_weights[cluster_id] = pd.Series(1.0, index=assets)
            cluster_returns[cluster_id] = returns[assets].mean() * 252  # Annualized
            cluster_volatilities[cluster_id] = returns[assets].std() * np.sqrt(252)  # Annualized
        else:
            # Calculate HRP for this cluster
            hrp_result = hrp_portfolio(returns[assets])
            cluster_weights[cluster_id] = hrp_result['weights']
            cluster_returns[cluster_id] = hrp_result['expected_return']
            cluster_volatilities[cluster_id] = hrp_result['expected_volatility']
    
    # Create a synthetic return series for each cluster
    cluster_return_series = pd.DataFrame(index=returns.index)
    for cluster_id, assets in clusters.items():
        # Weight the returns in this cluster according to the HRP weights
        weights = cluster_weights[cluster_id]
        weighted_returns = returns[assets].dot(weights)
        cluster_return_series[f'Cluster_{cluster_id}'] = weighted_returns
    
    # Run HRP on the clusters
    top_level_hrp = hrp_portfolio(cluster_return_series)
    top_level_weights = top_level_hrp['weights']
    
    # Combine cluster weights with top-level weights to get final asset weights
    final_weights = pd.Series(0, index=returns.columns)
    
    for cluster_id, assets in clusters.items():
        # Get the weight for this cluster
        cluster_weight = top_level_weights[f'Cluster_{cluster_id}']
        
        # Get the weights within this cluster
        inner_weights = cluster_weights[cluster_id]
        
        # Scale the inner weights by the cluster weight
        for asset in assets:
            final_weights[asset] = inner_weights[asset] * cluster_weight
    
    # Calculate final portfolio metrics
    weights_array = final_weights.values
    cov = returns.cov()
    expected_return = np.sum(returns.mean() * weights_array) * 252  # Annualized
    expected_volatility = np.sqrt(np.dot(weights_array, np.dot(cov, weights_array))) * np.sqrt(252)  # Annualized
    
    return {
        'weights': final_weights,
        'expected_return': expected_return,
        'expected_volatility': expected_volatility,
        'clusters': clusters
    } 