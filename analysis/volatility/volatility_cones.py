import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy.stats import norm

class VolatilityCones:
    """Implementation of volatility cone analysis for forecasting volatility ranges."""
    
    def __init__(self):
        """Initialize the volatility cones analyzer."""
        self.cone_data = {}
        self.historical_vols = {}
        self.parkinson_vols = {}
        
    def calculate_historical_vol(self, returns: pd.Series, 
                              windows: List[int] = None) -> pd.DataFrame:
        """
        Calculate historical volatility for multiple time windows.
        
        Args:
            returns: Time series of returns
            windows: List of window sizes in days
            
        Returns:
            DataFrame with historical volatility estimates
        """
        if windows is None:
            # Default windows: 5, 10, 21, 42, 63 (1 week to 3 months)
            windows = [5, 10, 21, 42, 63]
            
        # Calculate rolling standard deviation for each window
        vol_data = pd.DataFrame(index=returns.index)
        
        for window in windows:
            vol_name = f"vol_{window}d"
            # Annualized volatility
            vol_data[vol_name] = returns.rolling(window=window).std() * np.sqrt(252)
            
        # Store historical volatilities
        self.historical_vols = vol_data
        
        return vol_data
        
    def calculate_parkinson_vol(self, high_low_data: pd.DataFrame, 
                             windows: List[int] = None) -> pd.DataFrame:
        """
        Calculate Parkinson volatility using high-low data.
        
        Args:
            high_low_data: DataFrame with high and low prices
            windows: List of window sizes in days
            
        Returns:
            DataFrame with Parkinson volatility estimates
        """
        if windows is None:
            # Default windows: 5, 10, 21, 42, 63 (1 week to 3 months)
            windows = [5, 10, 21, 42, 63]
            
        # Ensure required columns exist
        required_cols = ['high', 'low']
        missing_cols = [col for col in required_cols if col not in high_low_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Calculate Parkinson volatility
        parkinson_factor = 1.0 / (4.0 * np.log(2.0))
        log_hl_ratio = np.log(high_low_data['high'] / high_low_data['low'])
        daily_vol = np.sqrt(parkinson_factor * (log_hl_ratio ** 2))
        
        # Calculate rolling volatility for each window
        vol_data = pd.DataFrame(index=high_low_data.index)
        
        for window in windows:
            vol_name = f"parkinson_vol_{window}d"
            # Annualized volatility
            vol_data[vol_name] = (daily_vol
                               .rolling(window=window)
                               .mean() 
                               * np.sqrt(252))
            
        # Store Parkinson volatilities
        self.parkinson_vols = vol_data
        
        return vol_data
        
    def build_volatility_cone(self, volatility_df: pd.DataFrame, 
                           windows: List[int] = None, 
                           lookback_period: int = 252,
                           quantiles: List[float] = None) -> Dict[str, pd.DataFrame]:
        """
        Build volatility cone using historical volatility data.
        
        Args:
            volatility_df: DataFrame with volatility estimates
            windows: List of window sizes in days
            lookback_period: Historical period to consider
            quantiles: List of quantiles to compute
            
        Returns:
            Dictionary with volatility cone data
        """
        if windows is None:
            # Get windows from column names
            windows = []
            for col in volatility_df.columns:
                if col.startswith('vol_'):
                    try:
                        window = int(col.split('_')[1].replace('d', ''))
                        windows.append(window)
                    except (IndexError, ValueError):
                        pass
                elif col.startswith('parkinson_vol_'):
                    try:
                        window = int(col.split('_')[2].replace('d', ''))
                        windows.append(window)
                    except (IndexError, ValueError):
                        pass
                        
            windows = sorted(list(set(windows)))
            
        if not windows:
            raise ValueError("No valid window sizes found")
            
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
            
        # Initialize cone data
        cone_data = {}
        
        # Get the last lookback_period days of data
        vol_recent = volatility_df.iloc[-lookback_period:]
        
        # Calculate statistics for each window
        for window in windows:
            vol_col = f"vol_{window}d"
            parkinson_vol_col = f"parkinson_vol_{window}d"
            
            # Select the appropriate column
            if vol_col in vol_recent.columns:
                window_data = vol_recent[vol_col].dropna()
            elif parkinson_vol_col in vol_recent.columns:
                window_data = vol_recent[parkinson_vol_col].dropna()
            else:
                # Skip this window if data not available
                continue
                
            # Calculate statistics
            stats = {}
            stats['mean'] = window_data.mean()
            stats['median'] = window_data.median()
            stats['min'] = window_data.min()
            stats['max'] = window_data.max()
            stats['current'] = window_data.iloc[-1]
            
            # Calculate quantiles
            for quantile in quantiles:
                stats[f'q{int(quantile*100)}'] = window_data.quantile(quantile)
                
            # Store stats for this window
            cone_data[window] = stats
            
        # Convert to DataFrame for easy plotting
        cone_df = pd.DataFrame(cone_data)
        
        # Store cone data
        self.cone_data = {
            'stats': cone_df,
            'windows': windows,
            'quantiles': quantiles,
            'lookback_period': lookback_period
        }
        
        return {'stats': cone_df, 'raw_data': vol_recent}
        
    def forecast_vol_range(self, confidence: float = 0.95, 
                        forecast_windows: List[int] = None) -> pd.DataFrame:
        """
        Forecast volatility range for different time windows.
        
        Args:
            confidence: Confidence level for the forecast
            forecast_windows: List of forecast windows in days
            
        Returns:
            DataFrame with volatility range forecasts
        """
        if not self.cone_data:
            return pd.DataFrame()
            
        stats_df = self.cone_data['stats']
        
        if forecast_windows is None:
            # Use the same windows as in the cone
            forecast_windows = self.cone_data['windows']
            
        # Calculate z-score for the confidence interval
        z_score = norm.ppf((1 + confidence) / 2)
        
        # Initialize forecast data
        forecast_data = []
        
        for window in forecast_windows:
            if window not in stats_df.columns:
                continue
                
            # Get statistics for this window
            mean_vol = stats_df.loc['mean', window]
            current_vol = stats_df.loc['current', window]
            
            # Calculate standard error using cone data
            if 'raw_data' in self.cone_data:
                vol_col = f"vol_{window}d"
                parkinson_vol_col = f"parkinson_vol_{window}d"
                
                if vol_col in self.cone_data['raw_data'].columns:
                    std_vol = self.cone_data['raw_data'][vol_col].std()
                elif parkinson_vol_col in self.cone_data['raw_data'].columns:
                    std_vol = self.cone_data['raw_data'][parkinson_vol_col].std()
                else:
                    # Use a heuristic if raw data not available
                    std_vol = mean_vol * 0.2  # 20% of mean as an approximation
            else:
                # Use a heuristic if raw data not available
                std_vol = mean_vol * 0.2
                
            # Calculate forecast range
            lower_bound = max(0, mean_vol - z_score * std_vol)
            upper_bound = mean_vol + z_score * std_vol
            
            # Add to forecast data
            forecast_data.append({
                'window': window,
                'mean_forecast': mean_vol,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'current_vol': current_vol,
                'confidence': confidence,
                'z_score': z_score
            })
            
        return pd.DataFrame(forecast_data)
        
    def calculate_term_structure(self, current_date: pd.Timestamp = None) -> pd.DataFrame:
        """
        Calculate volatility term structure from cone data.
        
        Args:
            current_date: Date for term structure calculation
            
        Returns:
            DataFrame with term structure data
        """
        if not self.cone_data:
            return pd.DataFrame()
            
        stats_df = self.cone_data['stats']
        windows = self.cone_data['windows']
        
        # Prepare term structure data
        term_data = []
        
        for window in windows:
            if window not in stats_df.columns:
                continue
                
            # Get current volatility
            current_vol = stats_df.loc['current', window]
            
            # Get quantile values
            quantiles = {}
            for q in self.cone_data['quantiles']:
                q_key = f'q{int(q*100)}'
                if q_key in stats_df.index:
                    quantiles[q_key] = stats_df.loc[q_key, window]
                    
            # Add to term structure data
            term_data.append({
                'window': window,
                'current_vol': current_vol,
                'mean_vol': stats_df.loc['mean', window],
                'median_vol': stats_df.loc['median', window],
                'min_vol': stats_df.loc['min', window],
                'max_vol': stats_df.loc['max', window],
                **quantiles
            })
            
        return pd.DataFrame(term_data)
        
    def calculate_vol_percentile(self, current_vol: float, window: int) -> float:
        """
        Calculate the percentile of the current volatility.
        
        Args:
            current_vol: Current volatility value
            window: Window size in days
            
        Returns:
            Percentile value (0-100)
        """
        if not self.cone_data or 'raw_data' not in self.cone_data:
            return np.nan
            
        vol_col = f"vol_{window}d"
        parkinson_vol_col = f"parkinson_vol_{window}d"
        
        # Get historical volatility data for the window
        if vol_col in self.cone_data['raw_data'].columns:
            hist_vol = self.cone_data['raw_data'][vol_col].dropna()
        elif parkinson_vol_col in self.cone_data['raw_data'].columns:
            hist_vol = self.cone_data['raw_data'][parkinson_vol_col].dropna()
        else:
            return np.nan
            
        # Calculate percentile
        percentile = 100 * (hist_vol < current_vol).mean()
        
        return percentile
        
    def plot_volatility_cone(self, include_current: bool = True, 
                          include_forecast: bool = False, 
                          confidence: float = 0.95) -> None:
        """
        Plot volatility cone with current volatility and optional forecast.
        
        Args:
            include_current: Whether to include current volatility
            include_forecast: Whether to include volatility forecast
            confidence: Confidence level for forecast
            
        Returns:
            None (displays plot)
        """
        if not self.cone_data:
            print("No volatility cone data available")
            return
            
        stats_df = self.cone_data['stats']
        windows = self.cone_data['windows']
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot quantile lines
        for q in self.cone_data['quantiles']:
            q_key = f'q{int(q*100)}'
            if q_key in stats_df.index:
                plt.plot(windows, stats_df.loc[q_key], 
                       label=f"{int(q*100)}th Percentile",
                       linestyle='--')
                
        # Plot min/max
        plt.plot(windows, stats_df.loc['min'], 'b-', label='Min', alpha=0.6)
        plt.plot(windows, stats_df.loc['max'], 'r-', label='Max', alpha=0.6)
        
        # Plot mean and median
        plt.plot(windows, stats_df.loc['mean'], 'k-', label='Mean', linewidth=2)
        plt.plot(windows, stats_df.loc['median'], 'g-', label='Median', linewidth=2)
        
        # Plot current volatility
        if include_current:
            plt.plot(windows, stats_df.loc['current'], 'ro-', label='Current', linewidth=2)
            
        # Add forecast if requested
        if include_forecast:
            forecast_df = self.forecast_vol_range(confidence)
            if not forecast_df.empty:
                # Plot forecast range
                windows = forecast_df['window'].tolist()
                plt.fill_between(windows, 
                               forecast_df['lower_bound'], 
                               forecast_df['upper_bound'],
                               color='gray', alpha=0.3, 
                               label=f"{int(confidence*100)}% Forecast Range")
                
        plt.xlabel('Window Size (Days)')
        plt.ylabel('Annualized Volatility')
        plt.title('Volatility Cone Analysis')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def plot_term_structure(self, quantiles: List[float] = None) -> None:
        """
        Plot volatility term structure.
        
        Args:
            quantiles: List of quantiles to include
            
        Returns:
            None (displays plot)
        """
        term_df = self.calculate_term_structure()
        
        if term_df.empty:
            print("No term structure data available")
            return
            
        if quantiles is None:
            quantiles = [0.05, 0.5, 0.95]
            
        plt.figure(figsize=(10, 6))
        
        # Plot current volatility
        plt.plot(term_df['window'], term_df['current_vol'], 'ro-', 
               label='Current', linewidth=2)
        
        # Plot mean volatility
        plt.plot(term_df['window'], term_df['mean_vol'], 'k-', 
               label='Historical Mean', linewidth=2)
        
        # Plot selected quantiles
        for q in quantiles:
            q_key = f'q{int(q*100)}'
            if q_key in term_df.columns:
                plt.plot(term_df['window'], term_df[q_key], '--', 
                       label=f"{int(q*100)}th Percentile", alpha=0.6)
                
        plt.xlabel('Window Size (Days)')
        plt.ylabel('Annualized Volatility')
        plt.title('Volatility Term Structure')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def plot_historical_vol(self, windows: List[int] = None, 
                         lookback: int = 252) -> None:
        """
        Plot historical volatility for different windows.
        
        Args:
            windows: List of window sizes to plot
            lookback: Number of days to look back
            
        Returns:
            None (displays plot)
        """
        if self.historical_vols.empty:
            print("No historical volatility data available")
            return
            
        if windows is None:
            # Get all available windows
            windows = []
            for col in self.historical_vols.columns:
                if col.startswith('vol_'):
                    try:
                        window = int(col.split('_')[1].replace('d', ''))
                        windows.append(window)
                    except (IndexError, ValueError):
                        pass
                        
            windows = sorted(list(set(windows)))
            
        if not windows:
            print("No valid window sizes found")
            return
            
        # Get recent data
        recent_vols = self.historical_vols.iloc[-lookback:]
        
        plt.figure(figsize=(12, 6))
        
        for window in windows:
            vol_col = f"vol_{window}d"
            if vol_col in recent_vols.columns:
                plt.plot(recent_vols.index, recent_vols[vol_col], 
                       label=f"{window}-day Vol")
                
        plt.xlabel('Date')
        plt.ylabel('Annualized Volatility')
        plt.title('Historical Volatility')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def plot_vol_percentiles(self, current_date: pd.Timestamp = None) -> None:
        """
        Plot current volatility percentiles for different windows.
        
        Args:
            current_date: Date for percentile calculation
            
        Returns:
            None (displays plot)
        """
        if not self.cone_data:
            print("No volatility cone data available")
            return
            
        windows = self.cone_data['windows']
        stats_df = self.cone_data['stats']
        
        # Calculate percentiles
        percentiles = []
        for window in windows:
            current_vol = stats_df.loc['current', window]
            percentile = self.calculate_vol_percentile(current_vol, window)
            percentiles.append(percentile)
            
        plt.figure(figsize=(10, 6))
        plt.bar(windows, percentiles, alpha=0.7)
        
        # Add reference lines
        plt.axhline(y=50, color='k', linestyle='--', alpha=0.5, label='Median (50th)')
        plt.axhline(y=25, color='g', linestyle='--', alpha=0.5, label='25th Percentile')
        plt.axhline(y=75, color='r', linestyle='--', alpha=0.5, label='75th Percentile')
        
        plt.xlabel('Window Size (Days)')
        plt.ylabel('Current Volatility Percentile')
        plt.title('Volatility Percentile Analysis')
        plt.ylim(0, 100)
        plt.grid(True, axis='y')
        plt.legend()
        plt.tight_layout()
        plt.show() 