import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import scipy.stats as stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class MarketEfficiency:
    """
    A class for analyzing market efficiency using various statistical measures.
    
    Market efficiency refers to how well prices reflect all available information.
    Efficient markets should exhibit random walks and lack predictable patterns.
    
    This class implements methods to test various forms of market efficiency:
    - Weak form: Current prices reflect all historical price information
    - Semi-strong form: Prices adjust quickly to new public information
    - Strong form: Prices reflect all information (public and private)
    """
    
    def __init__(self, price_data: Optional[pd.DataFrame] = None):
        """
        Initialize the MarketEfficiency analyzer.
        
        Parameters:
        -----------
        price_data : pd.DataFrame, optional
            DataFrame containing price time series with columns:
            - timestamp: datetime index
            - price: price series to analyze
            - returns: returns (can be calculated if not provided)
        """
        self.price_data = price_data
        self.results = {}
        
        if price_data is not None:
            self.prepare_data()
    
    def set_data(self, price_data: pd.DataFrame) -> None:
        """Set or update the price data and prepare it for analysis."""
        self.price_data = price_data
        self.prepare_data()
    
    def prepare_data(self) -> None:
        """
        Prepare the data for efficiency analysis by calculating returns if needed,
        and ensuring the data is properly formatted.
        """
        if self.price_data is None:
            logger.error("No price data available for analysis")
            return
        
        # Create a copy to avoid modifying the original
        data = self.price_data.copy()
        
        # Check if timestamp is in the DataFrame or is the index
        if 'timestamp' in data.columns:
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Set timestamp as index if it's not already
            data.set_index('timestamp', inplace=True)
        elif not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("No timestamp column or datetime index found. Using integer indexing.")
        
        # Ensure we have a price column
        if 'price' not in data.columns:
            logger.error("Price data must contain a 'price' column")
            return
        
        # Sort by index (timestamp)
        data = data.sort_index()
        
        # Calculate returns if not provided
        if 'returns' not in data.columns:
            data['returns'] = data['price'].pct_change()
            data['log_returns'] = np.log(data['price'] / data['price'].shift(1))
        
        # Remove NaN values from returns calculation
        data.dropna(subset=['returns'], inplace=True)
        
        self.processed_data = data
    
    def run_random_walk_test(self, test_type: str = 'adf') -> Dict[str, float]:
        """
        Perform statistical tests to determine if the price series follows a random walk.
        
        Parameters:
        -----------
        test_type : str
            Type of test to perform:
            - 'adf': Augmented Dickey-Fuller test (null hypothesis: unit root exists)
            - 'kpss': KPSS test (null hypothesis: series is stationary)
            - 'all': Run both tests
            
        Returns:
        --------
        Dict[str, float]
            Test statistics and p-values
        """
        if not hasattr(self, 'processed_data'):
            logger.error("No processed data available. Call prepare_data() first.")
            return {}
        
        results = {}
        
        # Augmented Dickey-Fuller test
        if test_type in ['adf', 'all']:
            adf_result = adfuller(self.processed_data['price'])
            results['adf_statistic'] = adf_result[0]
            results['adf_pvalue'] = adf_result[1]
            results['adf_lags'] = adf_result[2]
            results['adf_nobs'] = adf_result[3]
            results['adf_critical_values'] = adf_result[4]
            
            # Interpret ADF result
            if results['adf_pvalue'] <= 0.05:
                results['adf_interpretation'] = "Reject null hypothesis - Series is stationary (not a random walk)"
            else:
                results['adf_interpretation'] = "Fail to reject null hypothesis - Series has a unit root (consistent with a random walk)"
        
        # KPSS test
        if test_type in ['kpss', 'all']:
            kpss_result = kpss(self.processed_data['price'])
            results['kpss_statistic'] = kpss_result[0]
            results['kpss_pvalue'] = kpss_result[1]
            results['kpss_lags'] = kpss_result[2]
            results['kpss_critical_values'] = kpss_result[3]
            
            # Interpret KPSS result
            if results['kpss_pvalue'] <= 0.05:
                results['kpss_interpretation'] = "Reject null hypothesis - Series is not stationary (consistent with a random walk)"
            else:
                results['kpss_interpretation'] = "Fail to reject null hypothesis - Series is stationary (not a random walk)"
        
        self.results.update(results)
        return results
    
    def variance_ratio_test(self, lags: List[int] = [2, 4, 8, 16]) -> Dict[str, float]:
        """
        Perform the variance ratio test to check if returns follow a random walk.
        
        The variance ratio test compares the variance of k-period returns with
        k times the variance of 1-period returns. Under a random walk, the ratio
        should be close to 1.
        
        Parameters:
        -----------
        lags : List[int]
            List of lag periods to test
            
        Returns:
        --------
        Dict[str, float]
            Variance ratios for each lag period
        """
        if not hasattr(self, 'processed_data'):
            logger.error("No processed data available. Call prepare_data() first.")
            return {}
        
        # Get log returns
        if 'log_returns' not in self.processed_data.columns:
            self.processed_data['log_returns'] = np.log(self.processed_data['price'] / self.processed_data['price'].shift(1))
            self.processed_data.dropna(subset=['log_returns'], inplace=True)
        
        log_returns = self.processed_data['log_returns']
        
        # Calculate variance ratios
        var_ratios = {}
        var_1 = np.var(log_returns)
        
        for lag in lags:
            # Calculate k-period returns
            k_returns = log_returns.rolling(window=lag).sum()
            # Calculate variance of k-period returns
            var_k = np.var(k_returns.dropna())
            # Calculate variance ratio
            ratio = var_k / (lag * var_1)
            var_ratios[f'vr_{lag}'] = ratio
        
        # Calculate standard errors and test statistics (simplified)
        n = len(log_returns)
        for lag in lags:
            se = np.sqrt(2 * (2 * lag - 1) * (lag - 1) / (3 * lag * n))
            test_stat = (var_ratios[f'vr_{lag}'] - 1) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
            
            var_ratios[f'vr_{lag}_se'] = se
            var_ratios[f'vr_{lag}_stat'] = test_stat
            var_ratios[f'vr_{lag}_pvalue'] = p_value
            
            # Interpret result
            if p_value <= 0.05:
                var_ratios[f'vr_{lag}_interpretation'] = "Reject null hypothesis - Returns do not follow a random walk"
            else:
                var_ratios[f'vr_{lag}_interpretation'] = "Fail to reject null hypothesis - Consistent with a random walk"
        
        self.results.update(var_ratios)
        return var_ratios
    
    def runs_test(self) -> Dict[str, float]:
        """
        Perform the runs test for randomness in returns.
        
        The runs test checks if the sequence of positive and negative returns
        is random, which would be expected in an efficient market.
        
        Returns:
        --------
        Dict[str, float]
            Runs test statistics and p-value
        """
        if not hasattr(self, 'processed_data'):
            logger.error("No processed data available. Call prepare_data() first.")
            return {}
        
        # Get returns
        returns = self.processed_data['returns']
        
        # Convert returns to binary sequence (1 for positive, 0 for negative)
        binary_returns = (returns > 0).astype(int)
        
        # Count runs
        runs = 1
        for i in range(1, len(binary_returns)):
            if binary_returns.iloc[i] != binary_returns.iloc[i-1]:
                runs += 1
        
        # Count number of positive and negative returns
        n1 = sum(binary_returns)
        n2 = len(binary_returns) - n1
        
        # Calculate expected number of runs and standard deviation
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1)))
        
        # Calculate Z-statistic
        z_stat = (runs - expected_runs) / std_runs
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        runs_results = {
            'runs': runs,
            'expected_runs': expected_runs,
            'std_runs': std_runs,
            'z_stat': z_stat,
            'p_value': p_value
        }
        
        # Interpret result
        if p_value <= 0.05:
            runs_results['interpretation'] = "Reject null hypothesis - Returns sequence is not random"
        else:
            runs_results['interpretation'] = "Fail to reject null hypothesis - Returns sequence appears random"
        
        self.results.update(runs_results)
        return runs_results
    
    def autocorrelation_test(self, lags: int = 20) -> pd.DataFrame:
        """
        Calculate autocorrelation and partial autocorrelation of returns.
        
        In an efficient market, returns should not be autocorrelated.
        
        Parameters:
        -----------
        lags : int
            Number of lags to calculate
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with autocorrelation and partial autocorrelation coefficients
        """
        if not hasattr(self, 'processed_data'):
            logger.error("No processed data available. Call prepare_data() first.")
            return pd.DataFrame()
        
        # Get returns
        returns = self.processed_data['returns']
        
        # Calculate autocorrelation
        acf = [returns.autocorr(lag=i) for i in range(1, lags+1)]
        
        # Calculate partial autocorrelation
        pacf = sm.tsa.stattools.pacf(returns.dropna(), nlags=lags)[1:]
        
        # Calculate standard error (1/sqrt(n))
        n = len(returns)
        se = 1 / np.sqrt(n)
        
        # Calculate 95% confidence intervals
        ci_upper = 1.96 * se
        ci_lower = -1.96 * se
        
        # Count significant autocorrelations
        sig_acf = sum(abs(x) > ci_upper for x in acf)
        sig_pacf = sum(abs(x) > ci_upper for x in pacf)
        
        # Create DataFrame
        ac_df = pd.DataFrame({
            'lag': range(1, lags+1),
            'acf': acf,
            'pacf': pacf,
            'ci_upper': ci_upper,
            'ci_lower': ci_lower,
            'acf_significant': [abs(x) > ci_upper for x in acf],
            'pacf_significant': [abs(x) > ci_upper for x in pacf]
        })
        
        # Add results
        self.results['sig_autocorrelations'] = sig_acf
        self.results['sig_partial_autocorrelations'] = sig_pacf
        self.results['autocorrelation_interpretation'] = (
            "Evidence against market efficiency" if sig_acf > 0.05 * lags else
            "Consistent with market efficiency"
        )
        
        return ac_df
    
    def hurst_exponent(self, min_lag: int = 2, max_lag: int = 20) -> float:
        """
        Calculate the Hurst exponent to measure long-term memory in the price series.
        
        H = 0.5: Random walk (efficient market)
        H > 0.5: Trend-reinforcing series (inefficient market)
        H < 0.5: Mean-reverting series (inefficient market)
        
        Parameters:
        -----------
        min_lag : int
            Minimum lag for calculation
        max_lag : int
            Maximum lag for calculation
            
        Returns:
        --------
        float
            Hurst exponent
        """
        if not hasattr(self, 'processed_data'):
            logger.error("No processed data available. Call prepare_data() first.")
            return np.nan
        
        # Get log returns
        if 'log_returns' not in self.processed_data.columns:
            self.processed_data['log_returns'] = np.log(self.processed_data['price'] / self.processed_data['price'].shift(1))
            self.processed_data.dropna(subset=['log_returns'], inplace=True)
        
        # Get price series
        prices = self.processed_data['price']
        
        # Calculate lags and ranges
        lags = range(min_lag, max_lag)
        tau = [np.std(np.subtract(prices[lag:].values, prices[:-lag].values)) for lag in lags]
        
        # Calculate Hurst exponent using linear regression on log-log plot
        m = np.polyfit(np.log10(lags), np.log10(tau), 1)
        hurst = m[0]
        
        # Store results
        self.results['hurst_exponent'] = hurst
        
        # Interpret result
        if 0.45 <= hurst <= 0.55:
            self.results['hurst_interpretation'] = "Random walk - Consistent with market efficiency"
        elif hurst > 0.55:
            self.results['hurst_interpretation'] = "Persistent behavior (trend-reinforcing) - Inconsistent with market efficiency"
        else:  # hurst < 0.45
            self.results['hurst_interpretation'] = "Anti-persistent behavior (mean-reverting) - Inconsistent with market efficiency"
        
        return hurst
    
    def lo_mackinlay_variance_ratio(self, lags: List[int] = [2, 4, 8, 16]) -> Dict[str, float]:
        """
        Perform the Lo-MacKinlay (1988) variance ratio test.
        
        A more robust version of the variance ratio test that accounts for
        heteroskedasticity in the returns.
        
        Parameters:
        -----------
        lags : List[int]
            List of lag periods to test
            
        Returns:
        --------
        Dict[str, float]
            Test statistics and p-values for each lag
        """
        if not hasattr(self, 'processed_data'):
            logger.error("No processed data available. Call prepare_data() first.")
            return {}
        
        # Get log returns
        if 'log_returns' not in self.processed_data.columns:
            self.processed_data['log_returns'] = np.log(self.processed_data['price'] / self.processed_data['price'].shift(1))
            self.processed_data.dropna(subset=['log_returns'], inplace=True)
        
        log_returns = self.processed_data['log_returns'].dropna()
        
        results = {}
        n = len(log_returns)
        
        for k in lags:
            # Calculate variance ratio
            var1 = np.var(log_returns)
            vark = np.var(log_returns.rolling(window=k).sum().dropna()) / k
            vr = vark / var1
            
            # Homoskedastic test statistic
            phi1 = 2 * (2*k - 1) * (k - 1) / (3 * k * n)
            z1 = (vr - 1) / np.sqrt(phi1)
            p1 = 2 * (1 - stats.norm.cdf(abs(z1)))
            
            # Heteroskedastic test statistic (simplified implementation)
            delta = sum([(log_returns.iloc[t] - log_returns.mean()) * 
                         (log_returns.iloc[t-j] - log_returns.mean()) / 
                         (n * var1) for t in range(j, n) for j in range(1, k)])
            phi2 = sum([4 * (1 - j/k)**2 * delta**2 for j in range(1, k)])
            z2 = (vr - 1) / np.sqrt(phi2)
            p2 = 2 * (1 - stats.norm.cdf(abs(z2)))
            
            results[f'vr_{k}'] = vr
            results[f'z1_{k}'] = z1
            results[f'p1_{k}'] = p1
            results[f'z2_{k}'] = z2
            results[f'p2_{k}'] = p2
            
            # Interpret results
            results[f'homosk_interp_{k}'] = (
                "Reject random walk" if p1 <= 0.05 else
                "Consistent with random walk"
            )
            results[f'heterosk_interp_{k}'] = (
                "Reject random walk" if p2 <= 0.05 else
                "Consistent with random walk"
            )
        
        self.results.update(results)
        return results
    
    def test_event_response(self, event_dates: List[pd.Timestamp], 
                           window_pre: int = 5, window_post: int = 10) -> pd.DataFrame:
        """
        Test how quickly prices respond to events (semi-strong form efficiency).
        
        Parameters:
        -----------
        event_dates : List[pd.Timestamp]
            List of event dates to analyze
        window_pre : int
            Number of periods before the event to include
        window_post : int
            Number of periods after the event to include
            
        Returns:
        --------
        pd.DataFrame
            Event study results with abnormal returns
        """
        if not hasattr(self, 'processed_data'):
            logger.error("No processed data available. Call prepare_data() first.")
            return pd.DataFrame()
        
        # Check if we have a valid index
        if not isinstance(self.processed_data.index, pd.DatetimeIndex):
            logger.error("Price data must have a datetime index for event studies")
            return pd.DataFrame()
        
        # Get returns
        returns = self.processed_data['returns']
        
        # Create event windows
        event_windows = []
        
        for event_date in event_dates:
            # Find the closest index to the event date
            try:
                event_idx = self.processed_data.index.get_indexer([event_date], method='nearest')[0]
                
                # Create window around the event
                start_idx = max(0, event_idx - window_pre)
                end_idx = min(len(self.processed_data) - 1, event_idx + window_post)
                
                window_data = self.processed_data.iloc[start_idx:end_idx + 1].copy()
                
                # Add relative day column (0 = event day)
                event_day_idx = window_data.index.get_indexer([self.processed_data.index[event_idx]], method='nearest')[0]
                window_data['rel_day'] = np.arange(-event_day_idx, len(window_data) - event_day_idx)
                
                # Add event identifier
                window_data['event_id'] = event_date
                
                event_windows.append(window_data)
            except Exception as e:
                logger.warning(f"Could not process event date {event_date}: {e}")
        
        if not event_windows:
            logger.error("No valid event windows found")
            return pd.DataFrame()
        
        # Combine all event windows
        event_data = pd.concat(event_windows)
        
        # Calculate average returns by relative day
        avg_returns = event_data.groupby('rel_day')['returns'].mean()
        std_returns = event_data.groupby('rel_day')['returns'].std()
        counts = event_data.groupby('rel_day')['returns'].count()
        
        # Calculate t-statistics
        t_stats = avg_returns / (std_returns / np.sqrt(counts))
        p_values = 2 * (1 - stats.t.cdf(abs(t_stats), counts - 1))
        
        # Calculate cumulative abnormal returns
        car = avg_returns.cumsum()
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'avg_return': avg_returns,
            'std_return': std_returns,
            'count': counts,
            't_stat': t_stats,
            'p_value': p_values,
            'significant': p_values < 0.05,
            'car': car
        })
        
        # Store key results
        self.results['event_response_speed'] = results_df.loc[results_df.index > 0, 'significant'].sum()
        self.results['event_study_interpretation'] = (
            "Slow price adjustment - Inefficient market" if self.results['event_response_speed'] > 1 else
            "Rapid price adjustment - Efficient market"
        )
        
        return results_df
    
    def calculate_market_efficiency_ratio(self) -> float:
        """
        Calculate the Market Efficiency Ratio (MER).
        
        MER = Absolute price change / Path length
        
        Values close to 1 indicate an efficient market.
        
        Returns:
        --------
        float
            Market Efficiency Ratio
        """
        if not hasattr(self, 'processed_data'):
            logger.error("No processed data available. Call prepare_data() first.")
            return np.nan
        
        # Get price series
        prices = self.processed_data['price']
        
        # Calculate absolute price change
        abs_change = abs(prices.iloc[-1] - prices.iloc[0])
        
        # Calculate path length (sum of absolute price changes)
        path_length = np.sum(abs(prices.diff().dropna()))
        
        # Calculate MER
        mer = abs_change / path_length if path_length > 0 else np.nan
        
        # Store results
        self.results['market_efficiency_ratio'] = mer
        
        # Interpret result
        if mer >= 0.9:
            self.results['mer_interpretation'] = "High efficiency - Price moves directly to its final value"
        elif mer >= 0.5:
            self.results['mer_interpretation'] = "Moderate efficiency - Some price noise but generally efficient"
        else:
            self.results['mer_interpretation'] = "Low efficiency - Price path is noisy and inefficient"
        
        return mer
    
    def calculate_all_metrics(self) -> Dict:
        """
        Calculate all market efficiency metrics.
        
        Returns:
        --------
        Dict
            Dictionary containing all efficiency metrics
        """
        # Run all tests
        self.run_random_walk_test(test_type='all')
        self.variance_ratio_test()
        self.runs_test()
        self.autocorrelation_test()
        self.hurst_exponent()
        self.calculate_market_efficiency_ratio()
        
        return self.results
    
    def generate_efficiency_report(self) -> pd.DataFrame:
        """
        Generate a comprehensive report on market efficiency.
        
        Returns:
        --------
        pd.DataFrame
            Report with test results and interpretations
        """
        # Calculate all metrics if not already done
        if not self.results:
            self.calculate_all_metrics()
        
        # Create report dataframe
        report_data = []
        
        # Add random walk tests
        if 'adf_pvalue' in self.results:
            report_data.append({
                'Test': 'Augmented Dickey-Fuller',
                'Statistic': self.results['adf_statistic'],
                'P-Value': self.results['adf_pvalue'],
                'Interpretation': self.results['adf_interpretation'],
                'Category': 'Random Walk Tests'
            })
        
        if 'kpss_pvalue' in self.results:
            report_data.append({
                'Test': 'KPSS',
                'Statistic': self.results['kpss_statistic'],
                'P-Value': self.results['kpss_pvalue'],
                'Interpretation': self.results['kpss_interpretation'],
                'Category': 'Random Walk Tests'
            })
        
        # Add variance ratio tests
        for k in [2, 4, 8, 16]:
            key = f'vr_{k}'
            if key in self.results:
                report_data.append({
                    'Test': f'Variance Ratio (lag={k})',
                    'Statistic': self.results[key],
                    'P-Value': self.results.get(f'vr_{k}_pvalue', np.nan),
                    'Interpretation': self.results.get(f'vr_{k}_interpretation', ''),
                    'Category': 'Variance Ratio Tests'
                })
        
        # Add runs test
        if 'p_value' in self.results:
            report_data.append({
                'Test': 'Runs Test',
                'Statistic': self.results['z_stat'],
                'P-Value': self.results['p_value'],
                'Interpretation': self.results['interpretation'],
                'Category': 'Randomness Tests'
            })
        
        # Add autocorrelation test
        if 'sig_autocorrelations' in self.results:
            report_data.append({
                'Test': 'Autocorrelation',
                'Statistic': f"{self.results['sig_autocorrelations']} significant lags",
                'P-Value': np.nan,
                'Interpretation': self.results['autocorrelation_interpretation'],
                'Category': 'Autocorrelation Tests'
            })
        
        # Add Hurst exponent
        if 'hurst_exponent' in self.results:
            report_data.append({
                'Test': 'Hurst Exponent',
                'Statistic': self.results['hurst_exponent'],
                'P-Value': np.nan,
                'Interpretation': self.results['hurst_interpretation'],
                'Category': 'Long Memory Tests'
            })
        
        # Add market efficiency ratio
        if 'market_efficiency_ratio' in self.results:
            report_data.append({
                'Test': 'Market Efficiency Ratio',
                'Statistic': self.results['market_efficiency_ratio'],
                'P-Value': np.nan,
                'Interpretation': self.results['mer_interpretation'],
                'Category': 'Efficiency Ratios'
            })
        
        # Create DataFrame
        report = pd.DataFrame(report_data)
        
        # Add overall assessment
        efficient_count = sum(1 for row in report_data if "efficient" in row['Interpretation'].lower() or "random" in row['Interpretation'].lower())
        inefficient_count = len(report_data) - efficient_count
        
        overall = "Likely Efficient" if efficient_count > inefficient_count else "Likely Inefficient"
        confidence = abs(efficient_count - inefficient_count) / len(report_data)
        
        self.results['overall_assessment'] = overall
        self.results['assessment_confidence'] = confidence
        
        return report
    
    def plot_efficiency_tests(self, save_path: Optional[str] = None) -> None:
        """
        Create visualization of key efficiency tests.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the generated plots
        """
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Price series and random walk tests
        plt.subplot(2, 2, 1)
        plt.title('Price Series')
        plt.plot(self.processed_data['price'])
        plt.xlabel('Time')
        plt.ylabel('Price')
        
        # Add annotations for random walk tests
        if 'adf_pvalue' in self.results and 'kpss_pvalue' in self.results:
            adf_result = "Random Walk" if self.results['adf_pvalue'] > 0.05 else "Not Random Walk"
            kpss_result = "Random Walk" if self.results['kpss_pvalue'] <= 0.05 else "Not Random Walk"
            plt.annotate(f"ADF Test: {adf_result}\nKPSS Test: {kpss_result}", 
                         xy=(0.05, 0.05), xycoords='axes fraction')
        
        # Plot 2: Autocorrelation function
        ac_data = self.autocorrelation_test(lags=20)
        
        if not ac_data.empty:
            plt.subplot(2, 2, 2)
            plt.title('Return Autocorrelation')
            plt.bar(ac_data['lag'], ac_data['acf'])
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axhline(y=ac_data['ci_upper'].iloc[0], color='r', linestyle='--', alpha=0.5)
            plt.axhline(y=ac_data['ci_lower'].iloc[0], color='r', linestyle='--', alpha=0.5)
            plt.xlabel('Lag')
            plt.ylabel('Autocorrelation')
        
        # Plot 3: Variance ratios
        if any(f'vr_{k}' in self.results for k in [2, 4, 8, 16]):
            plt.subplot(2, 2, 3)
            plt.title('Variance Ratios')
            
            lags = [k for k in [2, 4, 8, 16] if f'vr_{k}' in self.results]
            vrs = [self.results[f'vr_{k}'] for k in lags]
            
            plt.bar(lags, vrs)
            plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
            plt.xlabel('Lag')
            plt.ylabel('Variance Ratio')
            plt.annotate("Random Walk = 1.0", xy=(0.05, 0.95), xycoords='axes fraction')
        
        # Plot 4: Cumulative returns for efficiency assessment
        if 'market_efficiency_ratio' in self.results:
            plt.subplot(2, 2, 4)
            plt.title('Market Efficiency Assessment')
            
            # Plot cumulative returns
            cum_returns = (1 + self.processed_data['returns']).cumprod()
            plt.plot(cum_returns / cum_returns.iloc[0], label='Actual Path')
            
            # Plot direct path (straight line)
            plt.plot([cum_returns.index[0], cum_returns.index[-1]], 
                    [1, cum_returns.iloc[-1] / cum_returns.iloc[0]], 
                    'r--', label='Efficient Path')
            
            plt.xlabel('Time')
            plt.ylabel('Cumulative Return (Normalized)')
            plt.legend()
            
            # Add MER annotation
            mer = self.results['market_efficiency_ratio']
            plt.annotate(f"Market Efficiency Ratio: {mer:.3f}\n{self.results['mer_interpretation']}", 
                         xy=(0.05, 0.05), xycoords='axes fraction')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

# Example usage
if __name__ == "__main__":
    # Create sample price data
    np.random.seed(42)
    n_days = 1000
    
    # Generate timestamps
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Generate random walk price series
    random_walk = np.zeros(n_days)
    random_walk[0] = 100
    for i in range(1, n_days):
        random_walk[i] = random_walk[i-1] + np.random.normal(0, 1)
    
    # Create price DataFrame
    df_random_walk = pd.DataFrame({
        'timestamp': dates,
        'price': random_walk
    })
    
    # Create a trending price series (inefficient)
    trend = np.zeros(n_days)
    trend[0] = 100
    for i in range(1, n_days):
        # Add positive autocorrelation
        trend[i] = trend[i-1] + 0.05 + 0.3 * (trend[i-1] - trend[i-2]) + np.random.normal(0, 0.5) if i > 1 else trend[i-1] + np.random.normal(0, 1)
    
    # Create trending DataFrame
    df_trend = pd.DataFrame({
        'timestamp': dates,
        'price': trend
    })
    
    # Test both series
    print("Testing Random Walk Series")
    efficiency_random = MarketEfficiency(df_random_walk)
    random_report = efficiency_random.generate_efficiency_report()
    print("\nRandom Walk Series Efficiency Report:")
    print(random_report)
    
    print("\n\nTesting Trending Series")
    efficiency_trend = MarketEfficiency(df_trend)
    trend_report = efficiency_trend.generate_efficiency_report()
    print("\nTrending Series Efficiency Report:")
    print(trend_report)
    
    # Plot results
    efficiency_random.plot_efficiency_tests()
    efficiency_trend.plot_efficiency_tests() 