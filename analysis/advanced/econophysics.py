import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Union, Optional, Any
from scipy import stats
from scipy.optimize import curve_fit


class Econophysics:
    """
    Implementation of econophysics models and methods for cryptocurrency market analysis.
    Includes power law analysis, scaling laws, random matrix theory, and other
    physics-inspired approaches to financial markets.
    """
    
    def __init__(self):
        """Initialize the Econophysics class."""
        pass
    
    def fit_power_law(self, data: np.ndarray, xmin: Optional[float] = None) -> Dict[str, Any]:
        """
        Fit a power law distribution to data.
        
        Args:
            data (np.ndarray): Input data
            xmin (float, optional): Minimum value for fitting
            
        Returns:
            Dict: Power law fit results
        """
        # Ensure data is a numpy array and sorted
        data = np.sort(np.asarray(data))
        
        # Determine xmin if not provided
        if xmin is None:
            # Use simple approach - take 10% of the data range
            xmin = np.min(data) + 0.1 * (np.max(data) - np.min(data))
        
        # Filter data above xmin
        filtered_data = data[data >= xmin]
        
        if len(filtered_data) < 20:
            return {
                'alpha': np.nan,
                'xmin': xmin,
                'error': 'Too few data points',
                'ks_statistic': np.nan,
                'ks_p_value': np.nan
            }
        
        # Estimate power law exponent using maximum likelihood
        n = len(filtered_data)
        alpha = 1 + n / np.sum(np.log(filtered_data / xmin))
        
        # Calculate standard error of alpha
        alpha_error = (alpha - 1) / np.sqrt(n)
        
        # Perform Kolmogorov-Smirnov test to assess goodness of fit
        # Generate CDF for fitted power law
        def power_law_cdf(x, alpha, xmin):
            return 1 - (x / xmin) ** (1 - alpha)
        
        # Calculate empirical CDF
        empirical_cdf = np.arange(1, n + 1) / n
        
        # Calculate theoretical CDF
        theoretical_cdf = power_law_cdf(filtered_data, alpha, xmin)
        
        # Calculate KS statistic
        ks_statistic = np.max(np.abs(empirical_cdf - theoretical_cdf))
        
        # Calculate p-value using Monte Carlo simulation
        # This is a simplified version; a more accurate approach would involve
        # multiple simulations and comparison of KS statistics
        ks_p_value = np.exp(-2 * n * ks_statistic**2)
        
        return {
            'alpha': alpha,
            'alpha_error': alpha_error,
            'xmin': xmin,
            'n_tail': n,
            'ks_statistic': ks_statistic,
            'ks_p_value': ks_p_value
        }
    
    def tail_distribution(self, data: np.ndarray, n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the cumulative distribution function of data.
        
        Args:
            data (np.ndarray): Input data
            n_bins (int): Number of bins for histogram
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: x, P(X>x) values
        """
        # Sort data in ascending order
        sorted_data = np.sort(np.asarray(data))
        
        # Calculate CCDF: P(X>x)
        ccdf = np.arange(len(sorted_data), 0, -1) / len(sorted_data)
        
        return sorted_data, ccdf
    
    def random_matrix_theory(self, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply Random Matrix Theory to analyze correlations.
        
        Args:
            returns_df (pd.DataFrame): DataFrame with asset returns (rows: time, columns: assets)
            
        Returns:
            Dict: Dictionary with RMT analysis results
        """
        # Extract dimensions
        T, N = returns_df.shape  # T: time length, N: number of assets
        
        # Normalize returns
        normalized_returns = (returns_df - returns_df.mean()) / returns_df.std()
        
        # Calculate correlation matrix
        C = normalized_returns.corr().values
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Calculate theoretical bounds of eigenvalues based on RMT
        Q = T / N
        lambda_min = 1 + 1/Q - 2 * np.sqrt(1/Q)
        lambda_max = 1 + 1/Q + 2 * np.sqrt(1/Q)
        
        # Count significant eigenvalues (larger than lambda_max)
        significant_eigenvalues = eigenvalues[eigenvalues > lambda_max]
        num_significant = len(significant_eigenvalues)
        
        # Calculate participation ratio to measure eigenvector localization
        participation_ratio = []
        for i in range(len(eigenvalues)):
            v = eigenvectors[:, i]
            pr = 1 / np.sum(v**4)
            participation_ratio.append(pr)
        
        # Calculate cleaned correlation matrix (filtering out noise)
        C_cleaned = np.zeros_like(C)
        for i in range(num_significant):
            lambda_i = eigenvalues[i]
            v_i = eigenvectors[:, i].reshape(-1, 1)
            C_cleaned += lambda_i * (v_i @ v_i.T)
        
        # Add minimal diagonal to ensure positive-definiteness
        np.fill_diagonal(C_cleaned, 1)
        
        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'lambda_min': lambda_min,
            'lambda_max': lambda_max,
            'num_significant': num_significant,
            'participation_ratio': participation_ratio,
            'cleaned_correlation': C_cleaned
        }
    
    def fractal_dimension(self, time_series: np.ndarray, max_lag: int = 20) -> float:
        """
        Estimate the fractal dimension of a time series.
        
        Args:
            time_series (np.ndarray): Input time series
            max_lag (int): Maximum lag for analysis
            
        Returns:
            float: Estimated fractal dimension
        """
        # Ensure time series is a numpy array
        time_series = np.asarray(time_series)
        
        # Calculate lags
        lags = range(1, max_lag + 1)
        
        # Calculate variance for each lag
        variances = []
        for lag in lags:
            # Calculate variance of the time series differences at this lag
            diff = time_series[lag:] - time_series[:-lag]
            variance = np.var(diff)
            variances.append(variance)
        
        # Convert to numpy arrays
        lags_array = np.array(lags)
        variances_array = np.array(variances)
        
        # Perform log-log regression to estimate Hurst exponent
        log_lags = np.log(lags_array)
        log_variances = np.log(variances_array)
        
        slope, _, _, _, _ = stats.linregress(log_lags, log_variances)
        
        # Calculate Hurst exponent
        hurst = slope / 2
        
        # Calculate fractal dimension
        fractal_dim = 2 - hurst
        
        return fractal_dim
    
    def volatility_clustering(self, returns: np.ndarray, max_lag: int = 100) -> Dict[str, Any]:
        """
        Analyze volatility clustering in returns.
        
        Args:
            returns (np.ndarray): Return time series
            max_lag (int): Maximum lag for autocorrelation
            
        Returns:
            Dict: Dictionary with volatility clustering measures
        """
        # Ensure returns is a numpy array
        returns = np.asarray(returns)
        
        # Calculate absolute returns (proxy for volatility)
        abs_returns = np.abs(returns)
        
        # Calculate squared returns
        squared_returns = returns**2
        
        # Calculate autocorrelation of absolute returns
        abs_acf = []
        for lag in range(1, max_lag + 1):
            if lag >= len(abs_returns):
                break
            # Calculate autocorrelation at this lag
            corr = np.corrcoef(abs_returns[lag:], abs_returns[:-lag])[0, 1]
            abs_acf.append(corr)
        
        # Calculate autocorrelation of squared returns
        squared_acf = []
        for lag in range(1, max_lag + 1):
            if lag >= len(squared_returns):
                break
            # Calculate autocorrelation at this lag
            corr = np.corrcoef(squared_returns[lag:], squared_returns[:-lag])[0, 1]
            squared_acf.append(corr)
        
        # Fit a power law decay to autocorrelation
        lags = np.arange(1, len(abs_acf) + 1)
        
        # Define power law decay function
        def power_law(x, a, b):
            return a * x**(-b)
        
        try:
            # Fit the function to the absolute returns ACF
            abs_params, _ = curve_fit(power_law, lags, abs_acf, maxfev=5000)
            abs_decay_exponent = abs_params[1]
        except:
            abs_decay_exponent = np.nan
        
        try:
            # Fit the function to the squared returns ACF
            squared_params, _ = curve_fit(power_law, lags, squared_acf, maxfev=5000)
            squared_decay_exponent = squared_params[1]
        except:
            squared_decay_exponent = np.nan
        
        # Calculate volatility clustering index (persistence of volatility)
        clustering_index = np.sum(abs_acf[:20]) / 20 if len(abs_acf) >= 20 else np.nan
        
        return {
            'abs_returns_acf': abs_acf,
            'squared_returns_acf': squared_acf,
            'abs_decay_exponent': abs_decay_exponent,
            'squared_decay_exponent': squared_decay_exponent,
            'clustering_index': clustering_index
        }
    
    def scaling_analysis(self, time_series: np.ndarray, 
                        q_values: Optional[np.ndarray] = None,
                        max_lag: int = 100) -> Dict[str, Any]:
        """
        Perform multifractal scaling analysis on a time series.
        
        Args:
            time_series (np.ndarray): Input time series
            q_values (np.ndarray, optional): Array of q values for scaling analysis
            max_lag (int): Maximum lag for analysis
            
        Returns:
            Dict: Dictionary with scaling analysis results
        """
        # Ensure time series is a numpy array
        time_series = np.asarray(time_series)
        
        # Set default q values if not provided
        if q_values is None:
            q_values = np.arange(0.5, 5.1, 0.5)
        
        # Calculate lags
        lags = np.logspace(0, np.log10(max_lag), 20).astype(int)
        lags = np.unique(lags)
        
        # Initialize arrays for scaling exponents
        fluctuations = np.zeros((len(lags), len(q_values)))
        
        # Calculate fluctuations for each lag and q value
        for i, lag in enumerate(lags):
            if lag >= len(time_series) / 2:
                continue
                
            # Calculate segments
            n_segments = len(time_series) // lag
            segments = np.array([time_series[j*lag:(j+1)*lag] for j in range(n_segments)])
            
            # Calculate fluctuations for each segment
            segment_fluct = np.std(segments, axis=1)
            
            # Calculate q-order fluctuations
            for j, q in enumerate(q_values):
                if q == 0:
                    # For q=0, use logarithmic average
                    fluctuations[i, j] = np.exp(0.5 * np.mean(np.log(segment_fluct**2)))
                else:
                    # For q≠0, use q-order average
                    fluctuations[i, j] = np.mean(segment_fluct**q)**(1/q)
        
        # Calculate scaling exponents for each q value
        scaling_exponents = np.zeros(len(q_values))
        r_squared = np.zeros(len(q_values))
        
        for j, q in enumerate(q_values):
            # Log-log regression
            valid = fluctuations[:, j] > 0
            if np.sum(valid) > 1:
                log_lags = np.log(lags[valid])
                log_fluct = np.log(fluctuations[valid, j])
                
                slope, intercept, r_value, _, _ = stats.linregress(log_lags, log_fluct)
                scaling_exponents[j] = slope
                r_squared[j] = r_value**2
            else:
                scaling_exponents[j] = np.nan
                r_squared[j] = np.nan
        
        # Calculate multifractal spectrum
        h_q = scaling_exponents
        tau_q = q_values * h_q - 1
        
        # Calculate singularity spectrum (f(alpha))
        alpha = np.gradient(tau_q, q_values)
        f_alpha = q_values * alpha - tau_q
        
        return {
            'q_values': q_values,
            'h_q': h_q,
            'tau_q': tau_q,
            'alpha': alpha,
            'f_alpha': f_alpha,
            'r_squared': r_squared
        }
    
    def entropy_analysis(self, time_series: np.ndarray, 
                        bins: int = 50, 
                        time_windows: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Calculate various entropy measures for a time series.
        
        Args:
            time_series (np.ndarray): Input time series
            bins (int): Number of bins for probability estimation
            time_windows (List[int], optional): List of time windows for entropy calculation
            
        Returns:
            Dict: Dictionary with entropy measures
        """
        # Ensure time series is a numpy array
        time_series = np.asarray(time_series)
        
        # Calculate discrete probability distribution
        hist, bin_edges = np.histogram(time_series, bins=bins, density=True)
        
        # Add small constant to avoid log(0)
        hist = hist + 1e-10
        hist = hist / np.sum(hist)  # Re-normalize
        
        # Calculate Shannon entropy
        shannon_entropy = -np.sum(hist * np.log(hist))
        
        # Calculate relative entropy (KL divergence) compared to normal distribution
        # Generate normal distribution with same mean and std
        mean = np.mean(time_series)
        std = np.std(time_series)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        normal_pdf = stats.norm.pdf(bin_centers, mean, std)
        normal_pdf = normal_pdf / np.sum(normal_pdf)  # Normalize
        
        relative_entropy = np.sum(hist * np.log(hist / normal_pdf))
        
        # Calculate time-dependent entropy if time windows are provided
        time_entropy = {}
        if time_windows is not None:
            for window in time_windows:
                if window >= len(time_series):
                    continue
                    
                # Calculate entropy in sliding windows
                entropies = []
                for i in range(0, len(time_series) - window, window//2):
                    segment = time_series[i:i+window]
                    hist, _ = np.histogram(segment, bins=bins, density=True)
                    hist = hist + 1e-10
                    hist = hist / np.sum(hist)
                    segment_entropy = -np.sum(hist * np.log(hist))
                    entropies.append(segment_entropy)
                
                # Calculate entropy variability
                if entropies:
                    time_entropy[window] = {
                        'mean': np.mean(entropies),
                        'std': np.std(entropies),
                        'trend': np.polyfit(np.arange(len(entropies)), entropies, 1)[0]
                    }
        
        return {
            'shannon_entropy': shannon_entropy,
            'relative_entropy': relative_entropy,
            'time_entropy': time_entropy
        }
    
    def zipf_analysis(self, ranks: np.ndarray) -> Dict[str, Any]:
        """
        Perform Zipf's law analysis on rank data.
        
        Args:
            ranks (np.ndarray): Rank data (e.g., trading volumes, market caps)
            
        Returns:
            Dict: Dictionary with Zipf analysis results
        """
        # Ensure ranks is a numpy array
        ranks = np.asarray(ranks)
        
        # Sort ranks in descending order
        sorted_ranks = -np.sort(-ranks)
        
        # Create rank array
        rank_array = np.arange(1, len(sorted_ranks) + 1)
        
        # Take log of both arrays
        log_ranks = np.log(sorted_ranks)
        log_indices = np.log(rank_array)
        
        # Perform linear regression on log-log data
        mask = np.isfinite(log_ranks) & np.isfinite(log_indices)
        if np.sum(mask) < 2:
            return {
                'zipf_exponent': np.nan,
                'r_squared': np.nan,
                'error': 'Not enough valid data points'
            }
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_indices[mask], log_ranks[mask]
        )
        
        # Zipf's exponent is the negative of the slope
        zipf_exponent = -slope
        
        # Calculate R-squared
        r_squared = r_value**2
        
        return {
            'zipf_exponent': zipf_exponent,
            'intercept': intercept,
            'r_squared': r_squared,
            'p_value': p_value,
            'std_err': std_err
        }
    
    def calculate_drawdowns(self, time_series: np.ndarray) -> Dict[str, Any]:
        """
        Calculate drawdowns and analyze their statistics.
        
        Args:
            time_series (np.ndarray): Price or index time series
            
        Returns:
            Dict: Dictionary with drawdown analysis results
        """
        # Ensure time series is a numpy array
        time_series = np.asarray(time_series)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(time_series)
        
        # Calculate drawdowns as percentage of previous maximum
        drawdowns = (time_series - running_max) / running_max
        
        # Identify distinct drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_idx = 0
        max_depth = 0
        
        for i in range(len(drawdowns)):
            # Check for new drawdown start
            if not in_drawdown and drawdowns[i] < 0:
                in_drawdown = True
                start_idx = i
                max_depth = drawdowns[i]
            
            # Check for continuing drawdown
            elif in_drawdown:
                # Update maximum depth if deeper
                if drawdowns[i] < max_depth:
                    max_depth = drawdowns[i]
                
                # Check for drawdown end
                if drawdowns[i] == 0:
                    in_drawdown = False
                    # Record drawdown info if significant
                    if max_depth < -0.005:  # Only record drawdowns > 0.5%
                        drawdown_periods.append({
                            'start': start_idx,
                            'end': i,
                            'duration': i - start_idx + 1,
                            'depth': max_depth
                        })
        
        # If still in drawdown at the end of the time series
        if in_drawdown:
            drawdown_periods.append({
                'start': start_idx,
                'end': len(drawdowns) - 1,
                'duration': len(drawdowns) - start_idx,
                'depth': max_depth
            })
        
        # Extract drawdown depths and durations
        depths = [d['depth'] for d in drawdown_periods]
        durations = [d['duration'] for d in drawdown_periods]
        
        # Calculate summary statistics
        if depths:
            max_drawdown = min(depths)
            mean_drawdown = np.mean(depths)
            median_drawdown = np.median(depths)
            
            max_duration = max(durations)
            mean_duration = np.mean(durations)
            median_duration = np.median(durations)
            
            # Fit power law to drawdown depths
            if len(depths) > 5:
                power_law_fit = self.fit_power_law(np.abs(depths))
            else:
                power_law_fit = {'alpha': np.nan, 'xmin': np.nan}
        else:
            max_drawdown = 0
            mean_drawdown = 0
            median_drawdown = 0
            max_duration = 0
            mean_duration = 0
            median_duration = 0
            power_law_fit = {'alpha': np.nan, 'xmin': np.nan}
        
        return {
            'drawdowns': drawdowns,
            'drawdown_periods': drawdown_periods,
            'max_drawdown': max_drawdown,
            'mean_drawdown': mean_drawdown,
            'median_drawdown': median_drawdown,
            'max_duration': max_duration,
            'mean_duration': mean_duration,
            'median_duration': median_duration,
            'power_law_exponent': power_law_fit['alpha']
        }
    
    def plot_scaling_laws(self, results: Dict[str, Any], title: str = "Scaling Analysis") -> plt.Figure:
        """
        Plot results of scaling analysis.
        
        Args:
            results (Dict): Results from scaling_analysis function
            title (str): Plot title
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot scaling exponents h(q)
        axes[0, 0].plot(results['q_values'], results['h_q'], 'o-')
        axes[0, 0].set_xlabel('q')
        axes[0, 0].set_ylabel('h(q)')
        axes[0, 0].set_title('Generalized Hurst Exponent')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add reference line for h=0.5 (random walk)
        axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        
        # Plot mass exponent τ(q)
        axes[0, 1].plot(results['q_values'], results['tau_q'], 'o-')
        axes[0, 1].set_xlabel('q')
        axes[0, 1].set_ylabel('τ(q)')
        axes[0, 1].set_title('Mass Exponent τ(q)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot multifractal spectrum f(α)
        axes[1, 0].plot(results['alpha'], results['f_alpha'], 'o-')
        axes[1, 0].set_xlabel('α')
        axes[1, 0].set_ylabel('f(α)')
        axes[1, 0].set_title('Multifractal Spectrum')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot R² values
        axes[1, 1].plot(results['q_values'], results['r_squared'], 'o-')
        axes[1, 1].set_xlabel('q')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].set_title('Goodness of Fit (R²)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add interpretation
        # Calculate the width of the singularity spectrum
        alpha_min = np.min(results['alpha'])
        alpha_max = np.max(results['alpha'])
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
    
    def analyze_price_data(self, price_data: pd.Series) -> Dict[str, Any]:
        """
        Perform comprehensive econophysics analysis on price data.
        
        Args:
            price_data (pd.Series): Price time series
            
        Returns:
            Dict: Dictionary with analysis results
        """
        # Calculate log returns
        returns = np.diff(np.log(price_data.values))
        
        # Calculate absolute returns
        abs_returns = np.abs(returns)
        
        # Basic statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Fit power law to absolute returns
        power_law_results = self.fit_power_law(abs_returns)
        
        # Analyze volatility clustering
        vol_clustering = self.volatility_clustering(returns)
        
        # Perform scaling analysis
        scaling_results = self.scaling_analysis(returns)
        
        # Calculate drawdowns
        drawdown_results = self.calculate_drawdowns(price_data.values)
        
        # Calculate entropy measures
        entropy_results = self.entropy_analysis(returns, time_windows=[30, 60, 90])
        
        # Calculate fractal dimension
        fractal_dim = self.fractal_dimension(price_data.values)
        
        # Compile results
        results = {
            'basic_stats': {
                'mean_return': mean_return,
                'std_return': std_return,
                'skewness': skewness,
                'kurtosis': kurtosis
            },
            'power_law': power_law_results,
            'volatility_clustering': vol_clustering,
            'scaling': scaling_results,
            'drawdowns': drawdown_results,
            'entropy': entropy_results,
            'fractal_dimension': fractal_dim
        }
        
        # Add interpretation
        insights = []
        
        # Power law insights
        if not np.isnan(power_law_results['alpha']):
            if power_law_results['alpha'] < 3:
                insights.append(f"Heavy-tailed returns distribution (α={power_law_results['alpha']:.2f}) indicating higher risk of extreme events")
            else:
                insights.append(f"Returns distribution shows moderate tail behavior (α={power_law_results['alpha']:.2f})")
        
        # Volatility clustering insights
        clustering_index = vol_clustering['clustering_index']
        if not np.isnan(clustering_index):
            if clustering_index > 0.3:
                insights.append(f"Strong volatility clustering detected (index={clustering_index:.2f})")
            elif clustering_index > 0.1:
                insights.append(f"Moderate volatility clustering (index={clustering_index:.2f})")
        
        # Multifractal insights
        spectrum_width = np.max(scaling_results['alpha']) - np.min(scaling_results['alpha'])
        if spectrum_width > 0.5:
            insights.append(f"Strong multifractality detected (width={spectrum_width:.2f}), indicating complex price dynamics")
        elif spectrum_width > 0.2:
            insights.append(f"Moderate multifractality (width={spectrum_width:.2f})")
        
        # Entropy insights
        if 30 in entropy_results['time_entropy']:
            entropy_trend = entropy_results['time_entropy'][30]['trend']
            if entropy_trend > 0.01:
                insights.append("Increasing market complexity over time")
            elif entropy_trend < -0.01:
                insights.append("Decreasing market complexity over time")
        
        # Fractal dimension insights
        if fractal_dim > 1.5:
            insights.append(f"High fractal dimension ({fractal_dim:.2f}) indicating rougher, more complex price movements")
        elif fractal_dim < 1.3:
            insights.append(f"Low fractal dimension ({fractal_dim:.2f}) suggesting smoother price trends")
        
        results['insights'] = insights
        
        return results


def truncated_power_law(x: np.ndarray, alpha: float, lambda_: float, x_min: float = 1.0) -> np.ndarray:
    """
    Calculate the truncated power law distribution.
    
    Args:
        x (np.ndarray): Input values
        alpha (float): Power law exponent
        lambda_ (float): Exponential cutoff parameter
        x_min (float): Minimum value
        
    Returns:
        np.ndarray: Probability density values
    """
    # Normalize x by x_min
    x_norm = x / x_min
    
    # Calculate unnormalized PDF
    pdf_unnorm = x_norm**(-alpha) * np.exp(-lambda_ * x_norm)
    
    # Normalization is complex; this is an approximation
    # For exact normalization, numerical integration would be needed
    norm_const = x_min * (alpha - 1)
    
    return pdf_unnorm / norm_const 