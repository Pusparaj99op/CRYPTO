import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import Tuple, List, Dict, Union, Optional, Any
import scipy.stats as stats
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
import community as community_louvain  # python-louvain package


class NetworkAnalysis:
    """
    Implementation of network analysis methods for cryptocurrency market structure.
    Includes correlation networks, minimum spanning trees, community detection,
    centrality measures and market structural analysis.
    """
    
    def __init__(self):
        """Initialize the NetworkAnalysis class."""
        pass
    
    def correlation_network(self, returns_df: pd.DataFrame, 
                           threshold: float = 0.5,
                           absolute: bool = False) -> nx.Graph:
        """
        Create a correlation network from asset returns.
        
        Args:
            returns_df (pd.DataFrame): DataFrame with asset returns (rows: time, columns: assets)
            threshold (float): Correlation threshold for edge creation
            absolute (bool): Whether to use absolute correlation values
            
        Returns:
            nx.Graph: Network graph where nodes are assets and edges represent correlations
        """
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Create network
        G = nx.Graph()
        
        # Add nodes (assets)
        for asset in returns_df.columns:
            G.add_node(asset)
        
        # Add edges based on correlations
        for i, asset_i in enumerate(returns_df.columns):
            for j, asset_j in enumerate(returns_df.columns):
                if i < j:  # Avoid duplicate edges and self-loops
                    corr = corr_matrix.loc[asset_i, asset_j]
                    
                    # Apply absolute if requested
                    if absolute:
                        corr_value = abs(corr)
                    else:
                        corr_value = corr
                    
                    # Add edge if correlation exceeds threshold
                    if corr_value > threshold:
                        G.add_edge(asset_i, asset_j, weight=corr_value, raw_corr=corr)
        
        return G
    
    def minimum_spanning_tree(self, returns_df: pd.DataFrame) -> nx.Graph:
        """
        Create a minimum spanning tree from asset returns.
        
        Args:
            returns_df (pd.DataFrame): DataFrame with asset returns (rows: time, columns: assets)
            
        Returns:
            nx.Graph: Minimum spanning tree
        """
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Transform correlations to distances
        # Using d = sqrt(2(1-ρ)) as distance metric
        distance_matrix = np.sqrt(2 * (1 - corr_matrix))
        
        # Create complete graph
        G = nx.Graph()
        
        # Add nodes (assets)
        for asset in returns_df.columns:
            G.add_node(asset)
        
        # Add edges with distances
        for i, asset_i in enumerate(returns_df.columns):
            for j, asset_j in enumerate(returns_df.columns):
                if i < j:  # Avoid duplicate edges and self-loops
                    distance = distance_matrix.loc[asset_i, asset_j]
                    correlation = corr_matrix.loc[asset_i, asset_j]
                    G.add_edge(asset_i, asset_j, weight=distance, correlation=correlation)
        
        # Extract minimum spanning tree
        mst = nx.minimum_spanning_tree(G, weight='weight')
        
        return mst
    
    def planar_maximally_filtered_graph(self, returns_df: pd.DataFrame) -> nx.Graph:
        """
        Create a Planar Maximally Filtered Graph (PMFG) from asset returns.
        
        The PMFG is a graph that contains the MST and additional edges,
        while maintaining planarity.
        
        Args:
            returns_df (pd.DataFrame): DataFrame with asset returns (rows: time, columns: assets)
            
        Returns:
            nx.Graph: Planar Maximally Filtered Graph
        """
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Create network structure for PMFG
        G_pmfg = nx.Graph()
        
        # Add nodes (assets)
        for asset in returns_df.columns:
            G_pmfg.add_node(asset)
        
        # Sort edges by correlation (descending)
        edges = []
        for i, asset_i in enumerate(returns_df.columns):
            for j, asset_j in enumerate(returns_df.columns):
                if i < j:  # Avoid duplicate edges and self-loops
                    correlation = corr_matrix.loc[asset_i, asset_j]
                    edges.append((asset_i, asset_j, correlation))
        
        edges.sort(key=lambda x: x[2], reverse=True)
        
        # Add edges while preserving planarity
        for asset_i, asset_j, correlation in edges:
            # Try to add edge
            G_pmfg.add_edge(asset_i, asset_j, weight=correlation)
            
            # Check if graph is still planar
            if not nx.check_planarity(G_pmfg)[0]:
                # If not, remove the edge
                G_pmfg.remove_edge(asset_i, asset_j)
        
        return G_pmfg
    
    def hierarchical_structure(self, returns_df: pd.DataFrame, method: str = 'single',
                             threshold: float = 0.5) -> Dict[str, Any]:
        """
        Extract hierarchical structure from asset returns using clustering.
        
        Args:
            returns_df (pd.DataFrame): DataFrame with asset returns (rows: time, columns: assets)
            method (str): Linkage method ('single', 'complete', 'average', 'ward')
            threshold (float): Distance threshold for cluster formation
            
        Returns:
            Dict: Dictionary with hierarchical clustering results
        """
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Transform correlations to distances
        distance_matrix = np.sqrt(2 * (1 - corr_matrix))
        
        # Perform hierarchical clustering
        Z = linkage(distance_matrix.values[np.triu_indices(len(distance_matrix), k=1)], 
                   method=method)
        
        # Form clusters
        clusters = fcluster(Z, threshold, criterion='distance')
        
        # Map assets to clusters
        asset_clusters = dict(zip(returns_df.columns, clusters))
        
        # Group assets by cluster
        cluster_assets = {}
        for asset, cluster in asset_clusters.items():
            if cluster not in cluster_assets:
                cluster_assets[cluster] = []
            cluster_assets[cluster].append(asset)
        
        # Calculate intra-cluster and inter-cluster correlations
        intra_corr = {}
        for cluster, assets in cluster_assets.items():
            if len(assets) > 1:
                # Get the submatrix for this cluster
                sub_matrix = corr_matrix.loc[assets, assets]
                # Calculate average correlation (excluding self-correlations)
                intra_corr[cluster] = (sub_matrix.sum().sum() - len(assets)) / (len(assets) * (len(assets) - 1))
            else:
                intra_corr[cluster] = 1.0  # Single-asset cluster
        
        # Calculate inter-cluster correlations
        inter_corr = {}
        for c1 in cluster_assets:
            for c2 in cluster_assets:
                if c1 < c2:
                    assets1 = cluster_assets[c1]
                    assets2 = cluster_assets[c2]
                    # Calculate average correlation between clusters
                    sub_matrix = corr_matrix.loc[assets1, assets2]
                    inter_corr[(c1, c2)] = sub_matrix.values.mean()
        
        return {
            'linkage': Z,
            'clusters': clusters,
            'asset_clusters': asset_clusters,
            'cluster_assets': cluster_assets,
            'intra_cluster_corr': intra_corr,
            'inter_cluster_corr': inter_corr
        }
    
    def community_detection(self, G: nx.Graph, method: str = 'louvain') -> Dict[str, Any]:
        """
        Detect communities in a network.
        
        Args:
            G (nx.Graph): Network graph
            method (str): Community detection method ('louvain', 'girvan_newman', 'label_prop')
            
        Returns:
            Dict: Dictionary with community detection results
        """
        communities = {}
        
        if method == 'louvain':
            # Apply Louvain method
            partition = community_louvain.best_partition(G)
            
            # Group nodes by community
            for node, comm_id in partition.items():
                if comm_id not in communities:
                    communities[comm_id] = []
                communities[comm_id].append(node)
                
            # Calculate modularity
            modularity = community_louvain.modularity(partition, G)
            
        elif method == 'girvan_newman':
            # Apply Girvan-Newman method
            comp = nx.community.girvan_newman(G)
            
            # Take first level of communities
            communities_list = next(comp)
            
            # Convert to dictionary
            for i, comm in enumerate(communities_list):
                communities[i] = list(comm)
                
            # Calculate modularity
            comm_sets = [set(comm) for comm in communities.values()]
            modularity = nx.community.quality.modularity(G, comm_sets)
            
        elif method == 'label_prop':
            # Apply Label Propagation
            communities_list = nx.community.label_propagation_communities(G)
            
            # Convert to dictionary
            for i, comm in enumerate(communities_list):
                communities[i] = list(comm)
                
            # Calculate modularity
            comm_sets = [set(comm) for comm in communities.values()]
            modularity = nx.community.quality.modularity(G, comm_sets)
            
        else:
            raise ValueError(f"Unknown community detection method: {method}")
        
        return {
            'communities': communities,
            'modularity': modularity
        }
    
    def calculate_centrality(self, G: nx.Graph) -> Dict[str, Dict[str, float]]:
        """
        Calculate centrality measures for nodes in a network.
        
        Args:
            G (nx.Graph): Network graph
            
        Returns:
            Dict: Dictionary with centrality measures for each node
        """
        # Calculate various centrality measures
        degree_cent = nx.degree_centrality(G)
        closeness_cent = nx.closeness_centrality(G)
        betweenness_cent = nx.betweenness_centrality(G)
        eigenvector_cent = nx.eigenvector_centrality_numpy(G)
        
        # Compile results
        centrality = {}
        for node in G.nodes():
            centrality[node] = {
                'degree': degree_cent[node],
                'closeness': closeness_cent[node],
                'betweenness': betweenness_cent[node],
                'eigenvector': eigenvector_cent[node]
            }
        
        # Calculate a composite centrality score (average of normalized measures)
        for node in centrality:
            centrality[node]['composite'] = np.mean([
                centrality[node]['degree'],
                centrality[node]['closeness'],
                centrality[node]['betweenness'],
                centrality[node]['eigenvector']
            ])
        
        return centrality
    
    def identify_influential_nodes(self, G: nx.Graph, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Identify the most influential nodes in a network.
        
        Args:
            G (nx.Graph): Network graph
            top_n (int): Number of top influential nodes to return
            
        Returns:
            List[Tuple[str, float]]: List of (node, influence_score) pairs
        """
        # Calculate centrality measures
        centrality = self.calculate_centrality(G)
        
        # Sort nodes by composite centrality
        influential_nodes = sorted(
            [(node, cent['composite']) for node, cent in centrality.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return influential_nodes[:top_n]
    
    def network_metrics(self, G: nx.Graph) -> Dict[str, float]:
        """
        Calculate global network metrics.
        
        Args:
            G (nx.Graph): Network graph
            
        Returns:
            Dict: Dictionary with network metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['num_nodes'] = G.number_of_nodes()
        metrics['num_edges'] = G.number_of_edges()
        metrics['density'] = nx.density(G)
        
        # Average degree
        degrees = dict(G.degree())
        metrics['avg_degree'] = np.mean(list(degrees.values()))
        
        # Clustering coefficient
        metrics['clustering'] = nx.average_clustering(G)
        
        # Average path length (if the graph is connected)
        if nx.is_connected(G):
            metrics['avg_path_length'] = nx.average_shortest_path_length(G)
        else:
            # Use the largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            largest_subgraph = G.subgraph(largest_cc)
            metrics['avg_path_length'] = nx.average_shortest_path_length(largest_subgraph)
            metrics['largest_cc_size'] = len(largest_cc)
            metrics['largest_cc_fraction'] = len(largest_cc) / metrics['num_nodes']
        
        # Diameter (if the graph is connected)
        if nx.is_connected(G):
            metrics['diameter'] = nx.diameter(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            largest_subgraph = G.subgraph(largest_cc)
            metrics['diameter'] = nx.diameter(largest_subgraph)
        
        # Assortativity (degree correlation)
        metrics['assortativity'] = nx.degree_assortativity_coefficient(G)
        
        return metrics
    
    def plot_network(self, G: nx.Graph, 
                    node_color: Optional[List] = None,
                    node_size: Optional[List] = None,
                    title: str = "Market Network",
                    layout: str = 'spring',
                    show_labels: bool = True) -> plt.Figure:
        """
        Plot a network graph.
        
        Args:
            G (nx.Graph): Network graph
            node_color (List, optional): List of node colors or values for colormap
            node_size (List, optional): List of node sizes
            title (str): Plot title
            layout (str): Layout algorithm ('spring', 'circular', 'kamada_kawai', 'spectral')
            show_labels (bool): Whether to show node labels
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Determine layout
        if layout == 'spring':
            pos = nx.spring_layout(G, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Use centrality for node size if not provided
        if node_size is None:
            centrality = nx.degree_centrality(G)
            node_size = [5000 * centrality[node] for node in G.nodes()]
        
        # Use communities for node color if not provided
        if node_color is None:
            # Detect communities
            communities = self.community_detection(G)
            
            # Map nodes to community IDs
            node_to_comm = {}
            for comm_id, nodes in communities['communities'].items():
                for node in nodes:
                    node_to_comm[node] = comm_id
                    
            node_color = [node_to_comm.get(node, 0) for node in G.nodes()]
        
        # Get edge weights if available
        if 'weight' in G.edges[list(G.edges())[0]]:
            edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
        else:
            edge_weights = [1 for _ in G.edges()]
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, 
                              cmap=plt.cm.viridis, alpha=0.8, ax=ax)
        
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.3, ax=ax)
        
        if show_labels:
            nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax)
        
        # Add title and remove axis
        plt.title(title, fontsize=16)
        plt.axis('off')
        
        plt.tight_layout()
        
        return fig
    
    def market_structure_analysis(self, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform a comprehensive market structure analysis.
        
        Args:
            returns_df (pd.DataFrame): DataFrame with asset returns (rows: time, columns: assets)
            
        Returns:
            Dict: Dictionary with market structure analysis results
        """
        # Create correlation network (with absolute correlations)
        corr_net = self.correlation_network(returns_df, threshold=0.3, absolute=True)
        
        # Calculate network metrics
        net_metrics = self.network_metrics(corr_net)
        
        # Create minimum spanning tree
        mst = self.minimum_spanning_tree(returns_df)
        
        # Detect communities in the correlation network
        communities = self.community_detection(corr_net)
        
        # Identify influential assets
        influential = self.identify_influential_nodes(corr_net, top_n=5)
        
        # Calculate centrality measures for all assets
        centrality = self.calculate_centrality(corr_net)
        
        # Extract hierarchical structure
        hierarchy = self.hierarchical_structure(returns_df)
        
        # Calculate average correlation
        avg_corr = returns_df.corr().values[np.triu_indices_from(returns_df.corr(), k=1)].mean()
        
        # Compile results
        results = {
            'correlation_network': corr_net,
            'mst': mst,
            'network_metrics': net_metrics,
            'communities': communities,
            'influential_assets': influential,
            'centrality': centrality,
            'hierarchy': hierarchy,
            'avg_correlation': avg_corr
        }
        
        # Generate summary insights
        insights = []
        
        # Network density insight
        if net_metrics['density'] > 0.7:
            insights.append("Market shows high interconnectedness, suggesting potential for widespread contagion")
        elif net_metrics['density'] < 0.3:
            insights.append("Market shows low overall correlation, indicating good diversification potential")
        
        # Community structure insight
        if communities['modularity'] > 0.5:
            insights.append(f"Strong community structure detected ({len(communities['communities'])} distinct groups)")
        elif communities['modularity'] < 0.2:
            insights.append("Weak community structure, market behaves more as a single entity")
        
        # Central assets insight
        if influential:
            central_assets = [asset for asset, _ in influential[:3]]
            insights.append(f"Key influential assets: {', '.join(central_assets)}")
        
        results['insights'] = insights
        
        return results
    
    def plot_market_backbone(self, returns_df: pd.DataFrame, title: str = "Market Backbone Structure") -> plt.Figure:
        """
        Plot the market backbone structure using minimum spanning tree.
        
        Args:
            returns_df (pd.DataFrame): DataFrame with asset returns (rows: time, columns: assets)
            title (str): Plot title
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Create minimum spanning tree
        mst = self.minimum_spanning_tree(returns_df)
        
        # Calculate centrality for node sizing
        centrality = nx.betweenness_centrality(mst)
        node_size = [10000 * centrality[node] for node in mst.nodes()]
        
        # Create community coloring
        communities = self.community_detection(mst)
        
        # Map nodes to community IDs
        node_to_comm = {}
        for comm_id, nodes in communities['communities'].items():
            for node in nodes:
                node_to_comm[node] = comm_id
                
        node_color = [node_to_comm.get(node, 0) for node in mst.nodes()]
        
        # Create edge weights based on correlation strength
        edge_weights = []
        for u, v in mst.edges():
            if 'correlation' in mst[u][v]:
                weight = abs(mst[u][v]['correlation']) * 5  # Scale for visibility
            else:
                weight = 1
            edge_weights.append(weight)
        
        # Plot the MST
        fig = plt.figure(figsize=(14, 12))
        pos = nx.spring_layout(mst, seed=42, k=0.5)
        
        nx.draw_networkx_nodes(mst, pos, node_size=node_size, node_color=node_color, 
                              cmap=plt.cm.tab20, alpha=0.8)
        
        nx.draw_networkx_edges(mst, pos, width=edge_weights, alpha=0.7, 
                              edge_color="gray")
        
        nx.draw_networkx_labels(mst, pos, font_size=10, font_family='sans-serif')
        
        plt.title(title, fontsize=16)
        plt.axis('off')
        
        # Add legend for communities
        comm_colors = plt.cm.tab20([node_to_comm.get(node, 0) for node in mst.nodes()])
        unique_comms = sorted(set(node_to_comm.values()))
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=plt.cm.tab20(i), 
                                     markersize=10, label=f'Group {i+1}')
                          for i in unique_comms]
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        return fig


def create_correlation_distance_matrix(returns_df: pd.DataFrame, 
                                     method: str = 'pearson') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create correlation and distance matrices from asset returns.
    
    Args:
        returns_df (pd.DataFrame): DataFrame with asset returns (rows: time, columns: assets)
        method (str): Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Correlation and distance matrices
    """
    # Calculate correlation matrix
    if method == 'pearson':
        corr_matrix = returns_df.corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = returns_df.corr(method='spearman')
    elif method == 'kendall':
        corr_matrix = returns_df.corr(method='kendall')
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    # Transform correlations to distances
    # Using d = sqrt(2(1-ρ)) as distance metric
    distance_matrix = np.sqrt(2 * (1 - corr_matrix))
    
    return corr_matrix, distance_matrix 