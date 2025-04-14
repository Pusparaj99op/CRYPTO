import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, List, Dict, Union, Optional
import statsmodels.tsa.stattools as ts
from numpy.random import RandomState
from sklearn.linear_model import LinearRegression


class FractalAnalysis:
    """
    Implementation of fractal market analysis methods for cryptocurrency data.
    Includes Hurst exponent calculation, Detrended Fluctuation Analysis (DFA),
    fractal dimension estimation, and R/S analysis.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the FractalAnalysis class.
        
        Args:
            random_state (int, optional): Seed for random number generation
        """
        self.random_state = RandomState(random_state)
    
    def hurst_exponent(self, time_series: np.ndarray, max_lag: int = 20) -> float:
        """
        Calculate the Hurst exponent of a time series.
        
        The Hurst exponent measures the long-term memory of a time series:
        - H < 0.5: Time series is mean-reverting
        - H = 0.5: Time series is random walk
        - H > 0.5: Time series is trending
        
        Args:
            time_series (np.ndarray): Input time series
            max_lag (int): Maximum lag to consider
            
        Returns:
            float: Hurst exponent
        """
        # Calculate lags ranging from 2 to max_lag
        lags = range(2, max_lag)
        
        # Calculate variance of the differences
        tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
        
        # Calculate the slope of the log-log plot -> Hurst exponent
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        
        return reg[0]
    
    def detrended_fluctuation_analysis(self, time_series: np.ndarray, 
                                        scales: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Perform Detrended Fluctuation Analysis (DFA) on time series.
        
        DFA is a method for determining the statistical self-affinity of a signal.
        
        Args:
            time_series (np.ndarray): Input time series
            scales (np.ndarray, optional): Array of scales to measure fluctuations
            
        Returns:
            Tuple: (alpha, scales, fluctuations)
                - alpha: scaling exponent
                - scales: scales used for calculation
                - fluctuations: corresponding fluctuations
        """
        # Ensure the time series is a numpy array
        time_series = np.asarray(time_series)
        
        # Calculate the cumulative sum of the time series
        y = np.cumsum(time_series - np.mean(time_series))
        
        # Define the scales (window sizes) at which to measure fluctuations
        if scales is None:
            scales = np.logspace(1, np.log10(len(time_series) / 4), 20).astype(int)
        
        # Calculate fluctuations for each scale
        fluctuations = np.zeros(len(scales))
        
        for i, scale in enumerate(scales):
            # Skip scales that are too large
            if scale >= len(time_series) / 4:
                continue
                
            # Calculate fluctuation at this scale
            fluctuations[i] = self._calculate_fluctuation(y, scale)
        
        # Remove zero values
        scales = scales[fluctuations > 0]
        fluctuations = fluctuations[fluctuations > 0]
        
        # Calculate the scaling exponent (alpha) using linear regression
        reg = np.polyfit(np.log10(scales), np.log10(fluctuations), 1)
        
        return reg[0], scales, fluctuations
    
    def _calculate_fluctuation(self, y: np.ndarray, scale: int) -> float:
        """
        Calculate fluctuation for a specific scale in DFA.
        
        Args:
            y (np.ndarray): Cumulative sum of time series
            scale (int): Scale (window size)
            
        Returns:
            float: Fluctuation value
        """
        # Number of windows
        n_windows = len(y) // scale
        
        # Calculate fluctuation across all windows
        fluctuation = 0
        
        for i in range(n_windows):
            # Extract segment
            segment = y[i * scale:(i + 1) * scale]
            
            # Calculate local trend (using linear regression)
            x = np.arange(scale)
            coef = np.polyfit(x, segment, 1)
            trend = np.polyval(coef, x)
            
            # Calculate fluctuation
            fluctuation += np.mean((segment - trend) ** 2)
        
        # Return square root of the mean fluctuation
        return np.sqrt(fluctuation / n_windows) if n_windows > 0 else 0
    
    def rescaled_range(self, time_series: np.ndarray, min_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate rescaled range (R/S) analysis for a time series.
        
        Args:
            time_series (np.ndarray): Input time series
            min_samples (int): Minimum number of samples per window
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (log(n), log(R/S)) pairs
        """
        # Ensure the time series is a numpy array
        time_series = np.asarray(time_series)
        
        # Calculate the possible divisions
        n_samples = len(time_series)
        divisions = []
        
        # Get divisors of length
        for i in range(min_samples, n_samples // 2):
            if n_samples % i == 0:
                divisions.append(i)
        
        # Add the entire series
        divisions.append(n_samples)
        
        # Calculate R/S for each division
        log_rs = []
        log_n = []
        
        for n in divisions:
            n_intervals = n_samples // n
            
            # Calculate R/S for each interval and average
            rs_values = []
            
            for i in range(n_intervals):
                segment = time_series[i * n:(i + 1) * n]
                rs = self._calculate_rs(segment)
                rs_values.append(rs)
            
            # Calculate average R/S
            avg_rs = np.mean(rs_values)
            
            log_rs.append(np.log10(avg_rs))
            log_n.append(np.log10(n))
        
        return np.array(log_n), np.array(log_rs)
    
    def _calculate_rs(self, segment: np.ndarray) -> float:
        """
        Calculate the R/S statistic for a segment.
        
        Args:
            segment (np.ndarray): Segment of time series
            
        Returns:
            float: R/S statistic
        """
        # Calculate mean and standard deviation
        mean = np.mean(segment)
        std = np.std(segment)
        
        if std == 0:
            return 1.0  # Avoid division by zero
        
        # Calculate cumulative deviation
        cumulative = np.cumsum(segment - mean)
        
        # Calculate R (range)
        r = np.max(cumulative) - np.min(cumulative)
        
        # Calculate R/S
        rs = r / std if std > 0 else 0
        
        return rs
    
    def fractal_dimension(self, time_series: np.ndarray, eps_range: Optional[np.ndarray] = None) -> float:
        """
        Calculate the fractal dimension of a time series using the box-counting method.
        
        Args:
            time_series (np.ndarray): Input time series
            eps_range (np.ndarray, optional): Range of box sizes to use
            
        Returns:
            float: Estimated fractal dimension
        """
        # Normalize time series to [0, 1]
        min_val = np.min(time_series)
        max_val = np.max(time_series)
        norm_ts = (time_series - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(time_series)
        
        # Define range of box sizes if not provided
        if eps_range is None:
            eps_range = np.logspace(-3, 0, 20)
        
        # Calculate box counts
        counts = []
        
        for eps in eps_range:
            count = self._box_count(norm_ts, eps)
            counts.append(count)
        
        # Calculate fractal dimension as the slope of log(count) vs log(1/eps)
        log_eps = np.log(1 / eps_range)
        log_count = np.log(counts)
        
        # Use only valid values for regression
        valid = ~np.isnan(log_count) & ~np.isinf(log_count) & (counts > 0)
        if sum(valid) > 1:
            reg = np.polyfit(log_eps[valid], log_count[valid], 1)
            return reg[0]
        else:
            return np.nan
    
    def _box_count(self, norm_ts: np.ndarray, eps: float) -> int:
        """
        Count the number of boxes needed to cover the time series.
        
        Args:
            norm_ts (np.ndarray): Normalized time series
            eps (float): Box size
            
        Returns:
            int: Number of boxes
        """
        # Create time axis
        time_axis = np.linspace(0, 1, len(norm_ts))
        
        # Initialize count
        count = 0
        visited = set()
        
        # Count boxes
        for i in range(len(norm_ts)):
            # Calculate box indices
            box_x = int(time_axis[i] / eps)
            box_y = int(norm_ts[i] / eps)
            
            # Create unique box identifier
            box_id = (box_x, box_y)
            
            # Count unique boxes
            if box_id not in visited:
                visited.add(box_id)
                count += 1
        
        return count
    
    def multi_fractal_spectrum(self, time_series: np.ndarray, q_range: np.ndarray = np.arange(-5, 6)) -> Dict[str, np.ndarray]:
        """
        Calculate the multifractal spectrum of a time series.
        
        Args:
            time_series (np.ndarray): Input time series
            q_range (np.ndarray): Range of q values to calculate
            
        Returns:
            Dict: Dictionary containing multifractal spectrum parameters:
                - q: q values
                - tau: mass exponent
                - h: generalized Hurst exponent
                - D: multifractal spectrum
        """
        # Ensure the time series is a numpy array
        time_series = np.asarray(time_series)
        
        # Define window sizes (scales)
        n = len(time_series)
        scales = np.logspace(1, np.log10(n/4), 10).astype(int)
        scales = scales[scales > 1]  # Ensure scales are at least 2
        
        # Initialize arrays for results
        tau = np.zeros_like(q_range, dtype=float)
        
        # Iterate over each q
        for i, q in enumerate(q_range):
            # Calculate partition function for different scales
            log_scales = np.log(scales)
            log_fq = np.zeros_like(scales, dtype=float)
            
            for j, scale in enumerate(scales):
                log_fq[j] = np.log(self._partition_function(time_series, scale, q))
            
            # Calculate mass exponent tau(q)
            reg = np.polyfit(log_scales, log_fq, 1)
            tau[i] = reg[0]
        
        # Calculate generalized Hurst exponent
        h = np.zeros_like(q_range, dtype=float)
        for i, q in enumerate(q_range):
            h[i] = tau[i] / q if q != 0 else 0
        
        # Calculate multifractal spectrum
        alpha = np.gradient(tau, q_range)
        D = q_range * alpha - tau
        
        return {
            'q': q_range,
            'tau': tau,
            'h': h,
            'alpha': alpha,
            'D': D
        }
    
    def _partition_function(self, time_series: np.ndarray, scale: int, q: float) -> float:
        """
        Calculate the partition function for multifractal analysis.
        
        Args:
            time_series (np.ndarray): Input time series
            scale (int): Scale (window size)
            q (float): Moment order
            
        Returns:
            float: Partition function value
        """
        # Calculate the cumulative sum
        y = np.cumsum(time_series - np.mean(time_series))
        
        # Number of windows
        n_windows = len(y) // scale
        
        # Calculate fluctuation for each window
        fluctuations = np.zeros(n_windows)
        
        for i in range(n_windows):
            # Extract segment
            segment = y[i * scale:(i + 1) * scale]
            
            # Calculate local trend (using linear regression)
            x = np.arange(scale)
            coef = np.polyfit(x, segment, 1)
            trend = np.polyval(coef, x)
            
            # Calculate fluctuation
            fluctuations[i] = np.sqrt(np.mean((segment - trend) ** 2))
        
        # Calculate partition function
        if q == 0:
            return n_windows
        else:
            fq = np.sum(fluctuations ** q)
            return fq
    
    def is_mean_reverting(self, time_series: np.ndarray, threshold: float = 0.45) -> bool:
        """
        Test if a time series is mean-reverting based on the Hurst exponent.
        
        Args:
            time_series (np.ndarray): Input time series
            threshold (float): Threshold for the Hurst exponent (default 0.45)
            
        Returns:
            bool: True if the time series is mean-reverting, False otherwise
        """
        h = self.hurst_exponent(time_series)
        return h < threshold
    
    def self_similarity_test(self, time_series: np.ndarray, min_scale: int = 10, 
                             max_scale: int = 100, n_scales: int = 10) -> Tuple[bool, float, float]:
        """
        Test for self-similarity in a time series.
        
        Args:
            time_series (np.ndarray): Input time series
            min_scale (int): Minimum scale for testing
            max_scale (int): Maximum scale for testing
            n_scales (int): Number of scales to test
            
        Returns:
            Tuple: (is_self_similar, p_value, r_squared)
                - is_self_similar: whether the time series is self-similar
                - p_value: p-value of the test
                - r_squared: r-squared of the log-log regression
        """
        # Define scales
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales).astype(int)
        scales = np.unique(scales)  # Remove duplicates
        
        # Calculate standard deviation at each scale
        std_devs = np.zeros_like(scales, dtype=float)
        
        for i, scale in enumerate(scales):
            # Downsample the time series
            downsampled = time_series[::scale]
            std_devs[i] = np.std(downsampled)
        
        # Perform log-log regression
        log_scales = np.log10(scales)
        log_std_devs = np.log10(std_devs)
        
        # Calculate regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_scales, log_std_devs)
        
        # Calculate R-squared
        r_squared = r_value ** 2
        
        # Check for self-similarity (strong linear relationship in log-log plot)
        is_self_similar = r_squared > 0.9 and p_value < 0.05
        
        return is_self_similar, p_value, r_squared
    
    def plot_hurst_analysis(self, time_series: np.ndarray, title: str = "Hurst Exponent Analysis") -> plt.Figure:
        """
        Plot Hurst exponent analysis.
        
        Args:
            time_series (np.ndarray): Input time series
            title (str): Plot title
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Calculate Hurst exponent
        h = self.hurst_exponent(time_series)
        
        # Calculate lags ranging from 2 to 20
        lags = range(2, 21)
        
        # Calculate variance of the differences
        tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot log-log relationship
        ax.loglog(lags, tau, 'o-', label=f'Data (H = {h:.3f})')
        
        # Plot regression line
        log_lags = np.log(lags)
        log_tau = np.log(tau)
        coef = np.polyfit(log_lags, log_tau, 1)
        reg_line = np.exp(coef[1]) * np.array(lags) ** coef[0]
        ax.loglog(lags, reg_line, 'r--', label=f'Regression (slope = {coef[0]:.3f})')
        
        # Add random walk reference line (H = 0.5)
        rw_line = reg_line[0] * (np.array(lags) / lags[0]) ** 0.5
        ax.loglog(lags, rw_line, 'g--', label='Random Walk (H = 0.5)')
        
        # Add labels and title
        ax.set_xlabel('Lag')
        ax.set_ylabel('Variance')
        ax.set_title(f'{title} (H = {h:.3f})')
        ax.legend()
        
        # Add interpretation
        if h < 0.45:
            interpretation = "Mean-reverting behavior detected"
        elif h > 0.55:
            interpretation = "Trend-following behavior detected"
        else:
            interpretation = "Random walk behavior detected"
            
        ax.text(0.05, 0.05, interpretation, transform=ax.transAxes, 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        return fig
    
    def plot_multifractal_spectrum(self, time_series: np.ndarray, 
                                   q_range: np.ndarray = np.arange(-5, 6, 1),
                                   title: str = "Multifractal Spectrum") -> plt.Figure:
        """
        Plot multifractal spectrum.
        
        Args:
            time_series (np.ndarray): Input time series
            q_range (np.ndarray): Range of q values
            title (str): Plot title
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Calculate multifractal spectrum
        spectrum = self.multi_fractal_spectrum(time_series, q_range)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot tau(q)
        axes[0, 0].plot(spectrum['q'], spectrum['tau'], 'o-')
        axes[0, 0].set_xlabel('q')
        axes[0, 0].set_ylabel('τ(q)')
        axes[0, 0].set_title('Mass Exponent τ(q)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot h(q)
        axes[0, 1].plot(spectrum['q'], spectrum['h'], 'o-')
        axes[0, 1].set_xlabel('q')
        axes[0, 1].set_ylabel('h(q)')
        axes[0, 1].set_title('Generalized Hurst Exponent h(q)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot multifractal spectrum f(α)
        axes[1, 0].plot(spectrum['alpha'], spectrum['D'], 'o-')
        axes[1, 0].set_xlabel('α')
        axes[1, 0].set_ylabel('f(α)')
        axes[1, 0].set_title('Multifractal Spectrum f(α)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot original time series
        t = np.arange(len(time_series))
        axes[1, 1].plot(t, time_series)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Original Time Series')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Calculate spectrum width and interpretation
        alpha_min = np.min(spectrum['alpha'])
        alpha_max = np.max(spectrum['alpha'])
        spectrum_width = alpha_max - alpha_min
        
        # Add interpretation text
        if spectrum_width > 0.5:
            interpretation = "Strong multifractality detected"
        elif spectrum_width > 0.2:
            interpretation = "Moderate multifractality detected"
        else:
            interpretation = "Weak or no multifractality detected"
            
        fig.suptitle(f"{title}\n{interpretation} (Width = {spectrum_width:.3f})", fontsize=14)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def analyze_price_data(self, prices: pd.Series, window_sizes: Optional[List[int]] = None) -> Dict:
        """
        Perform comprehensive fractal analysis on price data.
        
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
            window_sizes = [30, 60, 90, 180, 360]
            window_sizes = [w for w in window_sizes if w < len(returns)]
        
        # Initialize results dictionary
        results = {
            'hurst_full': self.hurst_exponent(returns),
            'dfa_alpha': self.detrended_fluctuation_analysis(returns)[0],
            'fractal_dim': self.fractal_dimension(price_array),
            'windows': {}
        }
        
        # Calculate metrics for different window sizes
        for window in window_sizes:
            if window >= len(returns):
                continue
                
            window_results = []
            
            # Calculate metrics for rolling windows
            for i in range(len(returns) - window + 1):
                window_returns = returns[i:i+window]
                
                window_results.append({
                    'start_idx': i,
                    'end_idx': i + window,
                    'hurst': self.hurst_exponent(window_returns),
                    'dfa_alpha': self.detrended_fluctuation_analysis(window_returns)[0]
                })
            
            results['windows'][window] = window_results
        
        # Add interpretation
        h = results['hurst_full']
        if h < 0.45:
            results['interpretation'] = "Mean-reverting behavior detected in the full series"
        elif h > 0.55:
            results['interpretation'] = "Trend-following behavior detected in the full series"
        else:
            results['interpretation'] = "Random walk behavior detected in the full series"
        
        return results

def generate_multifractal_brownian_motion(n_points: int = 1000, H: float = 0.7, 
                                         n_levels: int = 5, random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate a multifractal Brownian motion time series.
    
    Args:
        n_points (int): Number of points to generate
        H (float): Hurst parameter
        n_levels (int): Number of cascade levels
        random_state (int, optional): Random seed
        
    Returns:
        np.ndarray: Multifractal Brownian motion time series
    """
    # Set random state
    rng = np.random.RandomState(random_state)
    
    # Generate standard Brownian motion
    dt = 1.0 / n_points
    B = np.zeros(n_points)
    
    for i in range(1, n_points):
        B[i] = B[i-1] + np.sqrt(dt) * rng.normal(0, 1)
    
    # Generate multifractal weights
    weights = np.ones(n_points)
    
    for level in range(n_levels):
        segment_size = n_points // (2 ** level)
        n_segments = 2 ** level
        
        for i in range(n_segments):
            start = i * segment_size
            end = (i + 1) * segment_size
            
            # Generate random multiplier
            m = rng.uniform(0.5, 1.5)
            weights[start:end] *= m
    
    # Apply weights to the Brownian motion
    mBm = np.zeros(n_points)
    
    for i in range(1, n_points):
        # Fractional integration
        increments = B[i] - B[i-1]
        mBm[i] = mBm[i-1] + weights[i] * increments * (i ** (H - 0.5))
    
    return mBm 