import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Union, Optional, Any
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
from sklearn.cluster import KMeans
import networkx as nx
from sklearn import metrics


class ComplexityMeasures:
    """
    Implementation of complexity measures for cryptocurrency market analysis.
    Includes entropy measures, complexity indices, network complexity, and
    information-theoretic approaches.
    """
    
    def __init__(self):
        """Initialize the ComplexityMeasures class."""
        pass
    
    def sample_entropy(self, time_series: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Calculate the sample entropy of a time series.
        
        Sample entropy quantifies the irregularity or unpredictability of a time series.
        Higher values indicate greater complexity.
        
        Args:
            time_series (np.ndarray): Input time series
            m (int): Embedding dimension
            r (float): Tolerance (typically 0.1 to 0.25 of the standard deviation)
            
        Returns:
            float: Sample entropy value
        """
        # Normalize time series
        time_series = np.array(time_series)
        if np.std(time_series) == 0:
            return 0.0
            
        time_series = (time_series - np.mean(time_series)) / np.std(time_series)
        r = r * np.std(time_series)
        
        n = len(time_series)
        
        # Create templates of length m and m+1
        xm = np.array([time_series[i:i+m] for i in range(n-m)])
        xm_plus_1 = np.array([time_series[i:i+m+1] for i in range(n-m-1)])
        
        # Calculate template matches
        B = 0  # Counter for matches of length m
        A = 0  # Counter for matches of length m+1
        
        for i in range(len(xm)):
            # Calculate distances for templates of length m
            distances_m = np.max(np.abs(xm - xm[i]), axis=1)
            # Count matches (excluding self-match)
            B += np.sum((distances_m < r) & (np.arange(len(xm)) != i))
            
            if i < len(xm_plus_1):
                # Calculate distances for templates of length m+1
                distances_m_plus_1 = np.max(np.abs(xm_plus_1 - xm_plus_1[i]), axis=1)
                # Count matches (excluding self-match)
                A += np.sum((distances_m_plus_1 < r) & (np.arange(len(xm_plus_1)) != i))
        
        # Calculate sample entropy
        if B == 0 or A == 0:
            return np.inf
        else:
            return -np.log(A / B)
    
    def approximate_entropy(self, time_series: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Calculate the approximate entropy of a time series.
        
        Approximate entropy quantifies the unpredictability of fluctuations in a time series.
        
        Args:
            time_series (np.ndarray): Input time series
            m (int): Embedding dimension
            r (float): Tolerance (typically 0.1 to 0.25 of the standard deviation)
            
        Returns:
            float: Approximate entropy value
        """
        # Normalize time series
        time_series = np.array(time_series)
        if np.std(time_series) == 0:
            return 0.0
            
        time_series = (time_series - np.mean(time_series)) / np.std(time_series)
        r = r * np.std(time_series)
        
        n = len(time_series)
        
        # Calculate phi(m)
        def phi(m_val):
            xm = np.array([time_series[i:i+m_val] for i in range(n-m_val+1)])
            count = np.zeros(n-m_val+1)
            
            for i in range(n-m_val+1):
                # Calculate distances for templates of length m
                distances = np.max(np.abs(xm - xm[i]), axis=1)
                # Count matches (including self-match)
                count[i] = np.sum(distances < r) / (n-m_val+1)
            
            # Return logarithmic sum
            return np.sum(np.log(count)) / (n-m_val+1)
        
        # Calculate ApEn = phi(m) - phi(m+1)
        return phi(m) - phi(m+1)
    
    def permutation_entropy(self, time_series: np.ndarray, order: int = 3, delay: int = 1) -> float:
        """
        Calculate the permutation entropy of a time series.
        
        Permutation entropy quantifies the diversity of ordinal patterns in a time series.
        
        Args:
            time_series (np.ndarray): Input time series
            order (int): Order of permutation entropy
            delay (int): Delay between elements
            
        Returns:
            float: Permutation entropy value (normalized to [0, 1])
        """
        time_series = np.array(time_series)
        n = len(time_series)
        
        if order > n:
            raise ValueError("Order is too large for time series length")
        
        # Create permutations
        permutations = {}
        for i in range(n - (order - 1) * delay):
            # Extract pattern
            pattern = time_series[i:i+order*delay:delay]
            
            # Get permutation
            perm = tuple(np.argsort(pattern))
            
            # Count frequency
            permutations[perm] = permutations.get(perm, 0) + 1
        
        # Calculate normalized permutation entropy
        c = len(permutations)
        total = sum(permutations.values())
        p = np.array(list(permutations.values())) / total
        h = -np.sum(p * np.log(p))
        h_max = np.log(np.math.factorial(order))
        
        # Return normalized entropy
        return h / h_max if h_max > 0 else 0
    
    def lempel_ziv_complexity(self, time_series: np.ndarray, binary: bool = True, 
                             threshold: Optional[float] = None) -> float:
        """
        Calculate the Lempel-Ziv complexity of a time series.
        
        Lempel-Ziv complexity measures the number of distinct patterns in a sequence.
        
        Args:
            time_series (np.ndarray): Input time series
            binary (bool): Whether to binarize the time series
            threshold (float, optional): Threshold for binarization (default: median)
            
        Returns:
            float: Lempel-Ziv complexity (normalized)
        """
        time_series = np.array(time_series)
        
        # Binarize the time series if requested
        if binary:
            if threshold is None:
                threshold = np.median(time_series)
            binary_series = (time_series > threshold).astype(int)
        else:
            # Convert to string representation
            binary_series = time_series.astype(int)
        
        # Convert to string
        s = ''.join(map(str, binary_series))
        
        # Calculate complexity
        i, c = 0, 1
        vocabulary = {s[0]}
        
        while i + c <= len(s):
            if s[i:i+c] in vocabulary:
                c += 1
            else:
                vocabulary.add(s[i:i+c])
                i += c
                c = 1
        
        # Normalize by the theoretical upper bound (for binary sequences)
        n = len(s)
        b = 1.0 if binary else min(max(binary_series) + 1, 10)  # Base of representation
        upper_bound = n / np.log2(n) * np.log2(b) if n > 0 else 1
        
        return len(vocabulary) / upper_bound if upper_bound > 0 else 0
    
    def hurst_complexity(self, time_series: np.ndarray, max_lag: int = 20) -> Dict[str, float]:
        """
        Calculate complexity measures based on Hurst exponent.
        
        Args:
            time_series (np.ndarray): Input time series
            max_lag (int): Maximum lag to consider
            
        Returns:
            Dict: Dictionary with Hurst exponent and derived complexity measures
        """
        # Calculate lags ranging from 2 to max_lag
        lags = range(2, max_lag + 1)
        
        # Calculate variance of the differences
        tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
        
        # Calculate the slope of the log-log plot -> Hurst exponent
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = reg[0]
        
        # Calculate complexity measures
        complexity = 1.0 - abs(hurst - 0.5) * 2  # Maximum complexity at H=0.5
        antipersistence = max(0, 0.5 - hurst) * 2  # Scale to [0, 1]
        persistence = max(0, hurst - 0.5) * 2  # Scale to [0, 1]
        
        return {
            'hurst': hurst,
            'complexity': complexity,
            'antipersistence': antipersistence,
            'persistence': persistence
        }
    
    def fisher_information(self, time_series: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Fisher information of a time series.
        
        Fisher information measures the amount of information a sample provides about an
        unknown parameter.
        
        Args:
            time_series (np.ndarray): Input time series
            bins (int): Number of bins for probability estimation
            
        Returns:
            float: Fisher information measure
        """
        # Calculate histogram (probability density estimation)
        hist, bin_edges = np.histogram(time_series, bins=bins, density=True)
        
        # Add small constant to avoid division by zero
        hist = hist + 1e-10
        
        # Calculate discrete Fisher information
        fisher_info = 0
        for i in range(len(hist) - 1):
            p1, p2 = hist[i], hist[i+1]
            fisher_info += ((p2 - p1) ** 2) / p1
        
        return fisher_info / (len(hist) - 1)  # Normalize by number of transitions
    
    def network_complexity(self, time_series: np.ndarray, n_clusters: int = 5, 
                          embedding_dim: int = 3, embedding_delay: int = 1) -> Dict[str, float]:
        """
        Calculate complexity measures based on network representation of time series.
        
        Args:
            time_series (np.ndarray): Input time series
            n_clusters (int): Number of clusters for state discretization
            embedding_dim (int): Embedding dimension
            embedding_delay (int): Embedding delay
            
        Returns:
            Dict: Dictionary with network complexity measures
        """
        # Create phase space reconstruction
        n = len(time_series)
        if embedding_dim * embedding_delay > n:
            embedding_dim = 2
            embedding_delay = 1
            
        n_points = n - (embedding_dim - 1) * embedding_delay
        phase_space = np.zeros((n_points, embedding_dim))
        
        for i in range(n_points):
            for j in range(embedding_dim):
                phase_space[i, j] = time_series[i + j * embedding_delay]
        
        # Discretize the phase space using clustering
        if n_points < n_clusters:
            n_clusters = max(2, n_points // 2)
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        states = kmeans.fit_predict(phase_space)
        
        # Create transition network
        transitions = {}
        for i in range(len(states) - 1):
            source = states[i]
            target = states[i + 1]
            
            if (source, target) in transitions:
                transitions[(source, target)] += 1
            else:
                transitions[(source, target)] = 1
        
        # Create a directed graph
        G = nx.DiGraph()
        for i in range(n_clusters):
            G.add_node(i)
            
        for (source, target), weight in transitions.items():
            G.add_edge(source, target, weight=weight)
        
        # Calculate network measures
        results = {}
        
        # Number of edges (transition types)
        results['num_edges'] = G.number_of_edges()
        
        # Network density
        results['density'] = nx.density(G)
        
        # Average path length (if graph is connected, otherwise infinity)
        if nx.is_strongly_connected(G):
            results['avg_path_length'] = nx.average_shortest_path_length(G)
        else:
            # Use the largest strongly connected component
            largest_scc = max(nx.strongly_connected_components(G), key=len)
            scc = G.subgraph(largest_scc)
            results['avg_path_length'] = nx.average_shortest_path_length(scc) if scc.number_of_edges() > 0 else 0
        
        # Average clustering coefficient
        results['clustering'] = nx.average_clustering(G)
        
        # Graph entropy (based on degree distribution)
        degrees = dict(G.degree())
        degree_values = np.array(list(degrees.values()))
        if sum(degree_values) > 0:
            probs = degree_values / sum(degree_values)
            probs = probs[probs > 0]  # Remove zeros
            results['entropy'] = -np.sum(probs * np.log(probs))
        else:
            results['entropy'] = 0
        
        return results
    
    def multiscale_entropy(self, time_series: np.ndarray, max_scale: int = 10, 
                          m: int = 2, r: float = 0.2) -> np.ndarray:
        """
        Calculate multiscale entropy of a time series.
        
        Multiscale entropy quantifies the complexity of a time series across different scales.
        
        Args:
            time_series (np.ndarray): Input time series
            max_scale (int): Maximum scale factor
            m (int): Embedding dimension for sample entropy
            r (float): Tolerance for sample entropy
            
        Returns:
            np.ndarray: Multiscale entropy values for each scale
        """
        # Initialize results array
        mse = np.zeros(max_scale)
        
        # Adjust max_scale if the time series is too short
        n = len(time_series)
        max_scale = min(max_scale, n // 10)
        
        # Calculate sample entropy for each scale
        for scale in range(1, max_scale + 1):
            # Perform coarse-graining
            if scale == 1:
                coarse_grained = time_series
            else:
                # Average consecutive values
                remainder = n % scale
                if remainder > 0:
                    padded = time_series[:-remainder]
                else:
                    padded = time_series
                reshaped = padded.reshape((len(padded) // scale, scale))
                coarse_grained = np.mean(reshaped, axis=1)
            
            # Calculate sample entropy at this scale
            mse[scale - 1] = self.sample_entropy(coarse_grained, m, r)
            
            # Handle infinity values
            if np.isinf(mse[scale - 1]):
                mse[scale - 1] = np.nan
        
        return mse
    
    def complexity_index(self, time_series: np.ndarray) -> Dict[str, float]:
        """
        Calculate a comprehensive complexity index for a time series.
        
        Args:
            time_series (np.ndarray): Input time series
            
        Returns:
            Dict: Dictionary with various complexity measures
        """
        # Calculate various complexity measures
        results = {}
        
        # Basic entropy measures
        try:
            results['sample_entropy'] = self.sample_entropy(time_series)
        except:
            results['sample_entropy'] = np.nan
            
        try:
            results['perm_entropy'] = self.permutation_entropy(time_series)
        except:
            results['perm_entropy'] = np.nan
            
        try:
            results['lempel_ziv'] = self.lempel_ziv_complexity(time_series)
        except:
            results['lempel_ziv'] = np.nan
        
        # Statistical complexity
        hurst_results = self.hurst_complexity(time_series)
        results.update(hurst_results)
        
        # Fisher information
        try:
            results['fisher_info'] = self.fisher_information(time_series)
        except:
            results['fisher_info'] = np.nan
        
        # Autocorrelation structures
        try:
            acf_values = acf(time_series, nlags=min(20, len(time_series) // 4))
            # Decay rate of autocorrelation
            acf_abs = np.abs(acf_values[1:])  # Skip lag 0
            if len(acf_abs) > 1:
                results['acf_decay'] = np.polyfit(np.arange(len(acf_abs)), np.log(acf_abs + 1e-10), 1)[0]
            else:
                results['acf_decay'] = 0
        except:
            results['acf_decay'] = np.nan
        
        # Composite complexity index (normalized to [0, 1])
        try:
            # Use key complexity indicators
            indicators = [
                results['sample_entropy'] / 2.5,  # Normalize sample entropy
                results['perm_entropy'],  # Already normalized
                results['lempel_ziv'],    # Already normalized
                hurst_results['complexity'],  # Already normalized
                min(1.0, results['fisher_info'] / 10.0)  # Normalize Fisher information
            ]
            
            # Remove NaN values
            valid_indicators = [i for i in indicators if not np.isnan(i)]
            
            if valid_indicators:
                results['complexity_index'] = np.mean(valid_indicators)
            else:
                results['complexity_index'] = np.nan
        except:
            results['complexity_index'] = np.nan
        
        return results
    
    def plot_complexity_profile(self, time_series: np.ndarray, 
                               window_size: int = 100, overlap: int = 50) -> plt.Figure:
        """
        Plot complexity measures over time using a sliding window.
        
        Args:
            time_series (np.ndarray): Input time series
            window_size (int): Size of the sliding window
            overlap (int): Overlap between consecutive windows
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Adjust parameters if the time series is too short
        n = len(time_series)
        window_size = min(window_size, n // 2)
        overlap = min(overlap, window_size // 2)
        
        # Calculate number of windows
        step = window_size - overlap
        n_windows = (n - window_size) // step + 1
        
        # Initialize arrays for storing results
        times = np.zeros(n_windows)
        sample_entropy = np.zeros(n_windows)
        perm_entropy = np.zeros(n_windows)
        lempel_ziv = np.zeros(n_windows)
        hurst = np.zeros(n_windows)
        
        # Calculate complexity measures for each window
        for i in range(n_windows):
            start = i * step
            end = start + window_size
            window = time_series[start:end]
            
            times[i] = start + window_size // 2  # Middle of the window
            
            try:
                sample_entropy[i] = self.sample_entropy(window)
            except:
                sample_entropy[i] = np.nan
                
            try:
                perm_entropy[i] = self.permutation_entropy(window)
            except:
                perm_entropy[i] = np.nan
                
            try:
                lempel_ziv[i] = self.lempel_ziv_complexity(window)
            except:
                lempel_ziv[i] = np.nan
                
            try:
                hurst[i] = self.hurst_complexity(window)['hurst']
            except:
                hurst[i] = np.nan
        
        # Create figure
        fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
        
        # Plot original time series
        axes[0].plot(np.arange(n), time_series)
        axes[0].set_ylabel('Value')
        axes[0].set_title('Original Time Series')
        axes[0].grid(True, alpha=0.3)
        
        # Plot complexity measures
        axes[1].plot(times, sample_entropy, 'o-')
        axes[1].set_ylabel('Sample Entropy')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(times, perm_entropy, 'o-')
        axes[2].set_ylabel('Permutation\nEntropy')
        axes[2].grid(True, alpha=0.3)
        
        axes[3].plot(times, lempel_ziv, 'o-')
        axes[3].set_ylabel('Lempel-Ziv\nComplexity')
        axes[3].grid(True, alpha=0.3)
        
        axes[4].plot(times, hurst, 'o-')
        axes[4].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        axes[4].set_ylabel('Hurst\nExponent')
        axes[4].set_xlabel('Time')
        axes[4].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def analyze_price_data(self, prices: pd.Series, window_sizes: Optional[List[int]] = None) -> Dict:
        """
        Perform comprehensive complexity analysis on price data.
        
        Args:
            prices (pd.Series): Price time series
            window_sizes (List[int], optional): Window sizes for analysis
            
        Returns:
            Dict: Dictionary with analysis results
        """
        # Convert to numpy array and take returns
        price_array = prices.values
        returns = np.diff(np.log(price_array))
        
        # Set default window sizes if not provided
        if window_sizes is None:
            window_sizes = [30, 60, 90, 180]
            window_sizes = [w for w in window_sizes if w < len(returns)]
        
        # Initialize results dictionary
        results = {
            'price_complexity': self.complexity_index(price_array),
            'returns_complexity': self.complexity_index(returns),
            'windows': {}
        }
        
        # Calculate complexity for different window sizes
        for window in window_sizes:
            if window >= len(returns):
                continue
                
            window_results = []
            
            # Calculate complexity for rolling windows
            step = max(1, window // 10)  # Use a step to reduce computation
            for i in range(0, len(returns) - window + 1, step):
                window_returns = returns[i:i+window]
                
                window_results.append({
                    'start_idx': i,
                    'end_idx': i + window,
                    'complexity': self.complexity_index(window_returns)['complexity_index'],
                    'sample_entropy': self.sample_entropy(window_returns),
                    'hurst': self.hurst_complexity(window_returns)['hurst']
                })
            
            results['windows'][window] = window_results
        
        # Add interpretation based on returns complexity
        rc = results['returns_complexity']
        
        if np.isnan(rc['complexity_index']):
            results['interpretation'] = "Unable to calculate complexity index"
        elif rc['complexity_index'] > 0.7:
            results['interpretation'] = "High complexity - Market likely exhibits complex dynamics, potentially chaotic behavior"
        elif rc['complexity_index'] > 0.5:
            results['interpretation'] = "Moderate complexity - Market has some predictable patterns mixed with randomness"
        else:
            results['interpretation'] = "Low complexity - Market may be more predictable or trend-dominated"
        
        return results


def multifractal_detrended_fluctuation_analysis(time_series: np.ndarray, 
                                              q_order: np.ndarray = np.arange(-5, 6, 1),
                                              scales: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Perform Multifractal Detrended Fluctuation Analysis (MFDFA) on a time series.
    
    Args:
        time_series (np.ndarray): Input time series
        q_order (np.ndarray): Array of q-orders for multifractal analysis
        scales (np.ndarray, optional): Array of scales for fluctuation computation
        
    Returns:
        Dict: Dictionary with MFDFA results (scales, fluctuations, Hq, tau, alpha, f_alpha)
    """
    # Ensure the time series is a numpy array
    time_series = np.array(time_series)
    n = len(time_series)
    
    # Define scales if not provided
    if scales is None:
        scales = np.logspace(1, np.log10(n/4), 20).astype(int)
        scales = scales[scales > 1]
    
    # Calculate the profile (cumulative sum)
    profile = np.cumsum(time_series - np.mean(time_series))
    
    # Initialize results
    fluctuations = np.zeros((len(scales), len(q_order)))
    
    # Calculate fluctuations for each scale and q-order
    for i, scale in enumerate(scales):
        # Skip scales that are too large
        if scale >= n/4:
            continue
            
        # Number of segments
        n_segments = int(n / scale)
        
        # Calculate fluctuations for forward and backward segments
        fluct_segments = np.zeros(2 * n_segments)
        
        for j in range(n_segments):
            # Forward segment
            segment = profile[j*scale:(j+1)*scale]
            # Fit polynomial (linear trend)
            x = np.arange(scale)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            # Calculate fluctuation (standard deviation)
            fluct_segments[j] = np.sqrt(np.mean((segment - trend) ** 2))
            
            # Backward segment
            segment = profile[n - (j+1)*scale:n - j*scale]
            # Fit polynomial (linear trend)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            # Calculate fluctuation
            fluct_segments[j + n_segments] = np.sqrt(np.mean((segment - trend) ** 2))
        
        # Calculate q-order fluctuations
        for k, q in enumerate(q_order):
            if q == 0:
                # For q=0, use logarithmic average
                fluctuations[i, k] = np.exp(0.5 * np.mean(np.log(fluct_segments ** 2)))
            else:
                # For q≠0, use q-order average
                fluctuations[i, k] = (np.mean(fluct_segments ** q)) ** (1/q)
    
    # Calculate scaling exponents (Hq) using linear regression
    Hq = np.zeros_like(q_order, dtype=float)
    
    for k, q in enumerate(q_order):
        log_scales = np.log(scales)
        log_fluct = np.log(fluctuations[:, k])
        
        # Use only valid values for regression
        valid = ~np.isnan(log_fluct) & ~np.isinf(log_fluct) & (fluctuations[:, k] > 0)
        if np.sum(valid) > 1:
            Hq[k] = np.polyfit(log_scales[valid], log_fluct[valid], 1)[0]
        else:
            Hq[k] = np.nan
    
    # Calculate mass exponent tau(q) = q*Hq - 1
    tau = q_order * Hq - 1
    
    # Calculate singularity spectrum
    alpha = np.gradient(tau, q_order)
    f_alpha = q_order * alpha - tau
    
    # Return results
    return {
        'scales': scales,
        'fluctuations': fluctuations,
        'Hq': Hq,
        'tau': tau,
        'alpha': alpha,
        'f_alpha': f_alpha
    } 