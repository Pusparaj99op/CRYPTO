"""
Causal Discovery Module for Cryptocurrency Trading

This module provides tools for discovering and analyzing causal relationships
in cryptocurrency market data using various causal inference techniques.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import networkx as nx
from scipy import stats

# Optional imports for advanced causal discovery
try:
    import causallearn
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ScoreBased.GES import ges
    CAUSALLEARN_AVAILABLE = True
except ImportError:
    CAUSALLEARN_AVAILABLE = False

try:
    import causalnex
    from causalnex.structure import StructureModel
    from causalnex.structure.notears import from_pandas
    CAUSALNEX_AVAILABLE = True
except ImportError:
    CAUSALNEX_AVAILABLE = False

class GrangerCausality:
    """Class for Granger causality analysis in time series data."""
    
    def __init__(self, max_lag: int = 5, test: str = 'ssr_chi2test'):
        """
        Initialize Granger causality analyzer.
        
        Args:
            max_lag: Maximum number of lags to test
            test: Statistical test ('ssr_chi2test', 'ssr_ftest', 'ssr_chi2test', 'lrtest')
        """
        self.max_lag = max_lag
        self.test = test
        self.results = {}
        self.summary = pd.DataFrame()
    
    def is_stationary(self, series: pd.Series, significance: float = 0.05) -> bool:
        """
        Check if a time series is stationary using Augmented Dickey-Fuller test.
        
        Args:
            series: Time series to check
            significance: Significance level
            
        Returns:
            Boolean indicating stationarity
        """
        result = adfuller(series.dropna())
        return result[1] < significance
    
    def make_stationary(self, series: pd.Series) -> pd.Series:
        """
        Transform a series to make it stationary (difference if needed).
        
        Args:
            series: Original time series
            
        Returns:
            Stationary time series
        """
        # Check if already stationary
        if self.is_stationary(series):
            return series
        
        # Try first difference
        diff1 = series.diff().dropna()
        if self.is_stationary(diff1):
            return diff1
        
        # Try second difference
        diff2 = diff1.diff().dropna()
        if self.is_stationary(diff2):
            return diff2
        
        # If still not stationary, return first difference anyway
        return diff1
    
    def test_causality(self, data: pd.DataFrame, target_col: str = None) -> Dict:
        """
        Test Granger causality between all variables or toward a target variable.
        
        Args:
            data: DataFrame with time series variables
            target_col: Target column name (if None, test all pairs)
            
        Returns:
            Dictionary with test results
        """
        # Make all series stationary
        stationary_data = pd.DataFrame()
        for col in data.columns:
            stationary_data[col] = self.make_stationary(data[col])
        
        stationary_data = stationary_data.dropna()
        results = {}
        
        if target_col is not None:
            # Test causality toward the target
            if target_col not in stationary_data.columns:
                raise ValueError(f"Target column {target_col} not found in data")
            
            for col in stationary_data.columns:
                if col != target_col:
                    col_pair = (col, target_col)
                    test_result = grangercausalitytests(
                        stationary_data[[col, target_col]], 
                        maxlag=self.max_lag, 
                        verbose=False
                    )
                    results[col_pair] = test_result
        else:
            # Test all pairwise combinations
            for i, col1 in enumerate(stationary_data.columns):
                for col2 in stationary_data.columns[i+1:]:
                    # Test causality in both directions
                    pair1 = (col1, col2)
                    test_result1 = grangercausalitytests(
                        stationary_data[[col1, col2]], 
                        maxlag=self.max_lag, 
                        verbose=False
                    )
                    results[pair1] = test_result1
                    
                    pair2 = (col2, col1)
                    test_result2 = grangercausalitytests(
                        stationary_data[[col2, col1]], 
                        maxlag=self.max_lag, 
                        verbose=False
                    )
                    results[pair2] = test_result2
        
        self.results = results
        return results
    
    def summarize_results(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Summarize Granger causality test results.
        
        Args:
            alpha: Significance level
            
        Returns:
            DataFrame with summarized results
        """
        if not self.results:
            raise ValueError("No results to summarize. Run test_causality first.")
        
        summary_data = []
        
        for pair, result in self.results.items():
            cause, effect = pair
            
            for lag, values in result.items():
                # Get p-values for the selected test
                p_value = values[0][self.test][1]
                
                # Check if significant
                is_significant = p_value < alpha
                
                summary_data.append({
                    'Cause': cause,
                    'Effect': effect,
                    'Lag': lag,
                    'Test': self.test,
                    'p-value': p_value,
                    'Significant': is_significant
                })
        
        summary_df = pd.DataFrame(summary_data)
        self.summary = summary_df
        return summary_df
    
    def get_strongest_lag(self) -> pd.DataFrame:
        """
        Get the lag with the strongest causal effect for each pair.
        
        Returns:
            DataFrame with strongest lag for each causal relationship
        """
        if self.summary.empty:
            self.summarize_results()
        
        # Get strongest (lowest p-value) lag for each cause-effect pair
        strongest = (
            self.summary
            .sort_values('p-value')
            .groupby(['Cause', 'Effect'])
            .first()
            .reset_index()
        )
        
        return strongest[strongest['Significant']]
    
    def plot_causal_graph(self, figsize: Tuple = (10, 8)):
        """
        Plot Granger causality graph.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure with causality graph
        """
        strongest = self.get_strongest_lag()
        
        if strongest.empty:
            print("No significant causal relationships found.")
            return None
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add edges with weights based on p-values
        for _, row in strongest.iterrows():
            G.add_edge(
                row['Cause'], 
                row['Effect'], 
                weight=1-row['p-value'],
                lag=row['Lag']
            )
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Define node positions using spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_size=700, 
            node_color='lightblue'
        )
        
        # Draw edges with width based on weight
        edges = G.edges(data=True)
        edge_widths = [2 + 3 * edge[2]['weight'] for edge in edges]
        
        nx.draw_networkx_edges(
            G, pos, 
            width=edge_widths,
            arrowsize=20, 
            edge_color='gray'
        )
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=12)
        
        # Add edge labels (lag info)
        edge_labels = {(e[0], e[1]): f"Lag: {e[2]['lag']}" for e in edges}
        nx.draw_networkx_edge_labels(
            G, pos, 
            edge_labels=edge_labels, 
            font_size=10
        )
        
        plt.title("Granger Causality Network")
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()


class ConstraintBasedCausal:
    """Class for constraint-based causal discovery."""
    
    def __init__(self, method: str = 'pc'):
        """
        Initialize constraint-based causal discovery.
        
        Args:
            method: Method to use ('pc' for PC algorithm)
        """
        if not CAUSALLEARN_AVAILABLE:
            raise ImportError(
                "causallearn package is required for constraint-based causal discovery. "
                "Install it with: pip install causallearn"
            )
        
        self.method = method
        self.graph = None
        self.results = None
        self.feature_names = None
    
    def discover_causal_structure(self, data: pd.DataFrame, alpha: float = 0.05) -> nx.DiGraph:
        """
        Discover causal structure using constraint-based methods.
        
        Args:
            data: DataFrame with variables
            alpha: Significance level for independence tests
            
        Returns:
            NetworkX DiGraph representation of causal graph
        """
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Save feature names
        self.feature_names = data.columns.tolist()
        
        # Apply PC algorithm
        if self.method == 'pc':
            # Run PC algorithm
            self.results = pc(scaled_data, alpha=alpha, verbose=False)
            
            # Extract causal graph
            causal_graph = self.results.G
            
            # Convert to NetworkX graph
            G = nx.DiGraph()
            
            # Add nodes
            for i, name in enumerate(self.feature_names):
                G.add_node(name)
            
            # Add edges based on causal graph
            for i in range(len(causal_graph)):
                for j in range(len(causal_graph)):
                    if causal_graph[i, j] == 1 and causal_graph[j, i] == 0:
                        # Directed edge i -> j
                        G.add_edge(self.feature_names[i], self.feature_names[j])
                    elif causal_graph[i, j] == 1 and causal_graph[j, i] == 1:
                        # Undirected edge (potential causal but unresolved direction)
                        G.add_edge(self.feature_names[i], self.feature_names[j], undirected=True)
            
            self.graph = G
            return G
        
        else:
            raise ValueError(f"Method {self.method} not supported")
    
    def plot_causal_graph(self, figsize: Tuple = (10, 8)):
        """
        Plot the discovered causal graph.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure with causality graph
        """
        if self.graph is None:
            raise ValueError("No graph to plot. Run discover_causal_structure first.")
        
        G = self.graph
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Define node positions using spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Find directed and undirected edges
        directed_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('undirected', False)]
        undirected_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('undirected', False)]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_size=700, 
            node_color='lightblue'
        )
        
        # Draw directed edges
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=directed_edges,
            arrowsize=20, 
            edge_color='blue'
        )
        
        # Draw undirected edges
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=undirected_edges,
            style='dashed',
            edge_color='gray',
            arrowsize=0  # No arrowhead for undirected edges
        )
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=12)
        
        plt.title(f"Causal Graph ({self.method.upper()} Algorithm)")
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()


class ScoreBasedCausal:
    """Class for score-based causal discovery."""
    
    def __init__(self, method: str = 'notears'):
        """
        Initialize score-based causal discovery.
        
        Args:
            method: Method to use ('notears', 'ges')
        """
        self.method = method
        self.graph = None
        self.sm = None  # Structure model for NOTEARS
        
        if method == 'notears' and not CAUSALNEX_AVAILABLE:
            raise ImportError(
                "causalnex package is required for NOTEARS. "
                "Install it with: pip install causalnex"
            )
        
        if method == 'ges' and not CAUSALLEARN_AVAILABLE:
            raise ImportError(
                "causallearn package is required for GES. "
                "Install it with: pip install causallearn"
            )
    
    def discover_causal_structure(self, data: pd.DataFrame, 
                               max_iter: int = 100, 
                               threshold: float = 0.2) -> Union[nx.DiGraph, StructureModel]:
        """
        Discover causal structure using score-based methods.
        
        Args:
            data: DataFrame with variables
            max_iter: Maximum iterations for optimization
            threshold: Threshold for edge inclusion
            
        Returns:
            NetworkX DiGraph or StructureModel representation of causal graph
        """
        if self.method == 'notears':
            # Apply NOTEARS algorithm from causalnex
            sm = from_pandas(data, tabu_edges=None, max_iter=max_iter)
            
            # Apply threshold
            sm = sm.threshold(threshold)
            
            self.sm = sm
            self.graph = sm
            return sm
        
        elif self.method == 'ges':
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Apply GES algorithm from causallearn
            results = ges(scaled_data)
            
            # Convert to NetworkX graph
            G = nx.DiGraph()
            
            # Add nodes
            for i, name in enumerate(data.columns):
                G.add_node(name)
            
            # Extract the adjacency matrix
            causal_graph = results.graph.graph
            
            # Add edges based on causal graph
            for i in range(len(causal_graph)):
                for j in range(len(causal_graph)):
                    if causal_graph[i, j] == 1:
                        G.add_edge(data.columns[i], data.columns[j])
            
            self.graph = G
            return G
        
        else:
            raise ValueError(f"Method {self.method} not supported")
    
    def plot_causal_graph(self, figsize: Tuple = (10, 8)):
        """
        Plot the discovered causal graph.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure with causality graph
        """
        if self.graph is None:
            raise ValueError("No graph to plot. Run discover_causal_structure first.")
        
        plt.figure(figsize=figsize)
        
        if self.method == 'notears':
            # Plot using built-in causalnex visualizer
            from causalnex.plots import plot_structure
            
            ax = plt.gca()
            plot_structure(
                self.sm,
                graph_attributes={"rankdir": "LR"},
                ax=ax
            )
            plt.title("NOTEARS Causal Graph")
            
        else:  # Assume nx.DiGraph
            G = self.graph
            
            # Define node positions using spring layout
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos, 
                node_size=700, 
                node_color='lightblue'
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                G, pos, 
                arrowsize=20, 
                edge_color='blue'
            )
            
            # Add labels
            nx.draw_networkx_labels(G, pos, font_size=12)
            
            plt.title(f"Causal Graph ({self.method.upper()} Algorithm)")
            plt.axis('off')
        
        plt.tight_layout()
        return plt.gcf()


class CryptoCausalAnalyzer:
    """Specialized class for causal analysis in cryptocurrency markets."""
    
    def __init__(self):
        """Initialize the cryptocurrency causal analyzer."""
        self.granger = GrangerCausality(max_lag=5)
        self.graphs = {}
        self.results = {}
    
    def analyze_market_drivers(self, price_data: pd.DataFrame, 
                            external_factors: pd.DataFrame,
                            price_col: str = 'close') -> Dict:
        """
        Analyze what drives cryptocurrency prices using Granger causality.
        
        Args:
            price_data: Price time series data
            external_factors: External factors data
            price_col: Column name for price
            
        Returns:
            Dictionary with analysis results
        """
        # Ensure indices match
        ext_factors = external_factors.copy()
        prices = price_data.copy()
        
        common_idx = ext_factors.index.intersection(prices.index)
        ext_factors = ext_factors.loc[common_idx]
        prices = prices.loc[common_idx]
        
        # Combine data
        combined_data = pd.concat([prices[price_col], ext_factors], axis=1)
        combined_data.columns = [price_col] + list(ext_factors.columns)
        
        # Run Granger causality tests with price as target
        result = self.granger.test_causality(combined_data, target_col=price_col)
        summary = self.granger.summarize_results()
        strongest = self.granger.get_strongest_lag()
        
        # Store results
        self.results['market_drivers'] = {
            'raw_results': result,
            'summary': summary,
            'strongest': strongest
        }
        
        return self.results['market_drivers']
    
    def analyze_inter_market_effects(self, crypto_prices: pd.DataFrame, 
                                 price_col: str = 'close') -> Dict:
        """
        Analyze causal relationships between different cryptocurrencies.
        
        Args:
            crypto_prices: DataFrame with cryptocurrency prices
            price_col: Column name for price
            
        Returns:
            Dictionary with analysis results
        """
        # Extract price columns for different cryptocurrencies
        if isinstance(crypto_prices, pd.DataFrame) and len(crypto_prices.columns) == 1:
            raise ValueError("Need prices for multiple cryptocurrencies to analyze inter-market effects")
        
        # If multi-level columns, extract price_col for each symbol
        if isinstance(crypto_prices.columns, pd.MultiIndex):
            symbols = crypto_prices.columns.levels[0]
            prices_df = pd.DataFrame({
                symbol: crypto_prices[symbol][price_col] for symbol in symbols
            })
        else:
            # Assume directly passed dataframe with price series
            prices_df = crypto_prices.copy()
        
        # Run Granger causality tests between all pairs
        result = self.granger.test_causality(prices_df)
        summary = self.granger.summarize_results()
        strongest = self.granger.get_strongest_lag()
        
        # Plot causal graph
        causal_graph = self.granger.plot_causal_graph()
        
        # Store results
        self.results['inter_market'] = {
            'raw_results': result,
            'summary': summary,
            'strongest': strongest,
            'graph': causal_graph
        }
        
        return self.results['inter_market']
    
    def discover_factor_structure(self, data: pd.DataFrame, 
                              method: str = 'notears',
                              threshold: float = 0.2) -> nx.DiGraph:
        """
        Discover causal structure among market factors.
        
        Args:
            data: DataFrame with market factors
            method: Causal discovery method 
            threshold: Threshold for edge inclusion
            
        Returns:
            NetworkX DiGraph with causal structure
        """
        # Choose appropriate causal discovery method
        if method in ['pc', 'fci']:
            discoverer = ConstraintBasedCausal(method=method)
        elif method in ['notears', 'ges']:
            discoverer = ScoreBasedCausal(method=method)
        else:
            raise ValueError(f"Method {method} not supported")
        
        # Discover causal structure
        graph = discoverer.discover_causal_structure(
            data, 
            threshold=threshold if method == 'notears' else None
        )
        
        # Plot the graph
        causal_graph_plot = discoverer.plot_causal_graph()
        
        # Store results
        self.graphs[method] = graph
        self.results[f'factor_structure_{method}'] = {
            'graph': graph,
            'plot': causal_graph_plot
        }
        
        return graph
    
    def analyze_lead_lag_relationships(self, data: pd.DataFrame, 
                                   max_lag: int = 10,
                                   target_col: str = None) -> pd.DataFrame:
        """
        Analyze lead-lag relationships in cryptocurrency data.
        
        Args:
            data: Time series data
            max_lag: Maximum lag to consider
            target_col: Target column (if None, analyze all pairs)
            
        Returns:
            DataFrame with lead-lag relationships
        """
        # Create cross-correlation analyzer
        lead_lag_df = pd.DataFrame()
        
        if target_col is not None:
            # Analyze against target
            target_series = data[target_col]
            
            for col in data.columns:
                if col != target_col:
                    # Compute cross-correlation
                    cross_corr = pd.Series(
                        [data[col].shift(lag).corr(target_series) 
                         for lag in range(-max_lag, max_lag+1)],
                        index=range(-max_lag, max_lag+1)
                    )
                    
                    # Find max correlation and corresponding lag
                    max_corr = cross_corr.abs().max()
                    max_lag_val = cross_corr.abs().idxmax()
                    
                    # Determine lead/lag relationship
                    relationship = "leads" if max_lag_val < 0 else "lags" if max_lag_val > 0 else "concurrent"
                    
                    # Add to results
                    lead_lag_df = lead_lag_df.append({
                        'variable': col,
                        'correlation': cross_corr[max_lag_val],
                        'abs_correlation': abs(cross_corr[max_lag_val]),
                        'lag': max_lag_val,
                        'relationship': relationship
                    }, ignore_index=True)
        
        else:
            # Analyze all pairs
            for i, col1 in enumerate(data.columns):
                for col2 in data.columns[i+1:]:
                    # Compute cross-correlation
                    cross_corr = pd.Series(
                        [data[col1].shift(lag).corr(data[col2]) 
                         for lag in range(-max_lag, max_lag+1)],
                        index=range(-max_lag, max_lag+1)
                    )
                    
                    # Find max correlation and corresponding lag
                    max_corr = cross_corr.abs().max()
                    max_lag_val = cross_corr.abs().idxmax()
                    
                    # Determine lead/lag relationship
                    if max_lag_val < 0:
                        # col1 leads col2
                        lead_var, lag_var = col1, col2
                        relationship = "leads"
                        effect_lag = abs(max_lag_val)
                    elif max_lag_val > 0:
                        # col2 leads col1
                        lead_var, lag_var = col2, col1
                        relationship = "leads" 
                        effect_lag = max_lag_val
                    else:
                        # concurrent
                        lead_var, lag_var = col1, col2
                        relationship = "concurrent"
                        effect_lag = 0
                    
                    # Add to results
                    lead_lag_df = lead_lag_df.append({
                        'leading_var': lead_var,
                        'lagging_var': lag_var,
                        'correlation': cross_corr[max_lag_val],
                        'abs_correlation': abs(cross_corr[max_lag_val]),
                        'lag': effect_lag,
                        'relationship': relationship
                    }, ignore_index=True)
        
        # Store results
        self.results['lead_lag'] = lead_lag_df
        
        return lead_lag_df
    
    def plot_lead_lag_heatmap(self, data: pd.DataFrame, max_lag: int = 10, figsize: Tuple = (12, 10)):
        """
        Plot lead-lag relationships as a heatmap.
        
        Args:
            data: Time series data
            max_lag: Maximum lag to consider
            figsize: Figure size
            
        Returns:
            Matplotlib figure with lead-lag heatmap
        """
        columns = data.columns
        n_cols = len(columns)
        
        # Create matrix to store lead-lag values
        lead_lag_matrix = np.zeros((n_cols, n_cols))
        corr_matrix = np.zeros((n_cols, n_cols))
        
        # Calculate lead-lag relationships
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i != j:
                    # Compute cross-correlation
                    cross_corr = [data[col1].shift(lag).corr(data[col2]) 
                                 for lag in range(-max_lag, max_lag+1)]
                    
                    # Find max correlation and corresponding lag
                    max_corr_idx = np.argmax(np.abs(cross_corr))
                    max_lag_val = max_corr_idx - max_lag  # Convert to actual lag
                    max_corr = cross_corr[max_corr_idx]
                    
                    # Store values
                    lead_lag_matrix[i, j] = max_lag_val
                    corr_matrix[i, j] = max_corr
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot lead-lag heatmap
        plt.subplot(1, 2, 1)
        sns.heatmap(
            lead_lag_matrix, 
            annot=True, 
            fmt=".0f", 
            cmap="coolwarm",
            xticklabels=columns,
            yticklabels=columns,
            center=0
        )
        plt.title("Lead-Lag Relationships (Lag in Periods)")
        plt.xlabel("Effect Variable")
        plt.ylabel("Cause Variable")
        
        # Plot correlation heatmap
        plt.subplot(1, 2, 2)
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap="viridis",
            xticklabels=columns,
            yticklabels=columns,
            vmin=-1, 
            vmax=1
        )
        plt.title("Max Cross-Correlation")
        plt.xlabel("Effect Variable")
        plt.ylabel("Cause Variable")
        
        plt.tight_layout()
        return plt.gcf()


# Utility functions for causal analysis

def causal_impact_analysis(data: pd.DataFrame, 
                         intervention_time: pd.Timestamp,
                         target: str, 
                         covariates: List[str],
                         pre_period_days: int = 30,
                         post_period_days: int = 10) -> Dict:
    """
    Perform causal impact analysis for an intervention.
    
    Args:
        data: Time series data
        intervention_time: Time of intervention
        target: Target variable to measure impact
        covariates: Predictor variables
        pre_period_days: Number of days for pre-intervention period
        post_period_days: Number of days for post-intervention period
        
    Returns:
        Dictionary with analysis results
    """
    try:
        from causalimpact import CausalImpact
    except ImportError:
        raise ImportError(
            "causalimpact package is required for causal impact analysis. "
            "Install it with: pip install causalimpact"
        )
    
    # Define pre and post periods
    pre_period = [
        (intervention_time - pd.Timedelta(days=pre_period_days)).strftime('%Y-%m-%d'),
        intervention_time.strftime('%Y-%m-%d')
    ]
    
    post_period = [
        intervention_time.strftime('%Y-%m-%d'),
        (intervention_time + pd.Timedelta(days=post_period_days)).strftime('%Y-%m-%d')
    ]
    
    # Prepare data for CausalImpact
    impact_data = pd.concat([data[target], data[covariates]], axis=1)
    
    # Run causal impact analysis
    ci = CausalImpact(impact_data, pre_period, post_period)
    
    # Get summary
    summary = ci.summary()
    report = ci.summary(output='report')
    
    # Plot results
    plot = ci.plot()
    
    return {
        'model': ci,
        'summary': summary,
        'report': report,
        'plot': plot,
        'data': impact_data
    }

def create_causal_diagram(variables: List[str], 
                        edges: List[Tuple[str, str]],
                        weights: List[float] = None,
                        figsize: Tuple = (10, 8)):
    """
    Create a causal diagram from expert knowledge.
    
    Args:
        variables: List of variable names
        edges: List of directed edges as (cause, effect) tuples
        weights: Optional list of edge weights
        figsize: Figure size
        
    Returns:
        Matplotlib figure with causal diagram
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for var in variables:
        G.add_node(var)
    
    # Add edges
    if weights is not None:
        for (cause, effect), weight in zip(edges, weights):
            G.add_edge(cause, effect, weight=weight)
    else:
        for cause, effect in edges:
            G.add_edge(cause, effect)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Define node positions using hierarchical layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=700, 
        node_color='lightblue'
    )
    
    # Draw edges with width based on weight if provided
    if weights is not None:
        edges = G.edges(data=True)
        edge_widths = [1 + 2 * edge[2].get('weight', 1) for edge in edges]
        
        nx.draw_networkx_edges(
            G, pos, 
            width=edge_widths,
            arrowsize=20, 
            edge_color='blue'
        )
    else:
        nx.draw_networkx_edges(
            G, pos, 
            arrowsize=20, 
            edge_color='blue'
        )
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    plt.title("Causal Diagram")
    plt.axis('off')
    plt.tight_layout()
    
    return plt.gcf()

def analyze_causal_effect(data: pd.DataFrame, 
                         cause: str, 
                         effect: str,
                         adjustment_set: List[str] = None) -> Dict:
    """
    Analyze causal effect using simple covariate adjustment.
    
    Args:
        data: DataFrame with variables
        cause: Cause variable
        effect: Effect variable
        adjustment_set: Variables to adjust for (confounders)
        
    Returns:
        Dictionary with analysis results
    """
    import statsmodels.formula.api as smf
    
    results = {}
    
    # Unadjusted effect (potentially biased due to confounding)
    formula_unadjusted = f"{effect} ~ {cause}"
    model_unadjusted = smf.ols(formula_unadjusted, data=data).fit()
    results['unadjusted'] = {
        'model': model_unadjusted,
        'effect': model_unadjusted.params[cause],
        'p_value': model_unadjusted.pvalues[cause],
        'confidence_interval': model_unadjusted.conf_int().loc[cause].tolist()
    }
    
    # Adjusted effect (accounting for confounders)
    if adjustment_set:
        adjustment_terms = " + ".join(adjustment_set)
        formula_adjusted = f"{effect} ~ {cause} + {adjustment_terms}"
        model_adjusted = smf.ols(formula_adjusted, data=data).fit()
        results['adjusted'] = {
            'model': model_adjusted,
            'effect': model_adjusted.params[cause],
            'p_value': model_adjusted.pvalues[cause],
            'confidence_interval': model_adjusted.conf_int().loc[cause].tolist(),
            'adjustment_set': adjustment_set
        }
    
    return results 