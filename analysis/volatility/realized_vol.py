import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy.stats import norm

class RealizedVolatility:
    """Implementation of realized volatility estimators and analysis."""
    
    def __init__(self):
        """Initialize the realized volatility calculator."""
        self.results = {}
        self.high_frequency_data = None
        
    def calculate_standard_deviation(self, returns: pd.Series, 
                                  window: int = 21, 
                                  annualize: bool = True) -> pd.Series:
        """
        Calculate realized volatility using standard deviation of returns.
        
        Args:
            returns: Time series of returns
            window: Rolling window size
            annualize: Whether to annualize the volatility
            
        Returns:
            Series with realized volatility
        """
        if returns.empty:
            return pd.Series()
            
        # Calculate rolling standard deviation
        realized_vol = returns.rolling(window=window).std()
        
        # Annualize if requested
        if annualize:
            # Assuming daily returns by default
            trading_days = 252
            realized_vol = realized_vol * np.sqrt(trading_days)
            
        # Store result
        method_name = f"std_{window}d"
        self.results[method_name] = realized_vol
        
        return realized_vol
        
    def calculate_parkinson_volatility(self, high_low_data: pd.DataFrame, 
                                    window: int = 21, 
                                    annualize: bool = True) -> pd.Series:
        """
        Calculate Parkinson volatility using high-low data.
        
        Args:
            high_low_data: DataFrame with high and low prices
            window: Rolling window size
            annualize: Whether to annualize the volatility
            
        Returns:
            Series with Parkinson volatility
        """
        if high_low_data.empty:
            return pd.Series()
            
        # Ensure required columns exist
        required_cols = ['high', 'low']
        missing_cols = [col for col in required_cols if col not in high_low_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Calculate Parkinson volatility (assumed high/low are in log scale)
        # If not in log scale, uncomment the log transformation below
        # log_high = np.log(high_low_data['high'])
        # log_low = np.log(high_low_data['low'])
        # log_ratio = log_high - log_low
        
        # Using the provided high/low directly
        log_ratio = np.log(high_low_data['high'] / high_low_data['low'])
        
        # Parkinson estimator
        parkinson_factor = 1.0 / (4.0 * np.log(2.0))
        daily_variance = parkinson_factor * (log_ratio ** 2)
        daily_vol = np.sqrt(daily_variance)
        
        # Calculate rolling volatility
        realized_vol = daily_vol.rolling(window=window).mean()
        
        # Annualize if requested
        if annualize:
            trading_days = 252
            realized_vol = realized_vol * np.sqrt(trading_days)
            
        # Store result
        method_name = f"parkinson_{window}d"
        self.results[method_name] = realized_vol
        
        return realized_vol
        
    def calculate_garman_klass_volatility(self, ohlc_data: pd.DataFrame, 
                                      window: int = 21, 
                                      annualize: bool = True) -> pd.Series:
        """
        Calculate Garman-Klass volatility using OHLC data.
        
        Args:
            ohlc_data: DataFrame with open, high, low, close prices
            window: Rolling window size
            annualize: Whether to annualize the volatility
            
        Returns:
            Series with Garman-Klass volatility
        """
        if ohlc_data.empty:
            return pd.Series()
            
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in ohlc_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Calculate log values
        log_high_low = np.log(ohlc_data['high'] / ohlc_data['low'])
        log_close_open = np.log(ohlc_data['close'] / ohlc_data['open'])
        
        # Garman-Klass estimator
        daily_variance = 0.5 * (log_high_low ** 2) - (2 * np.log(2) - 1) * (log_close_open ** 2)
        daily_vol = np.sqrt(daily_variance)
        
        # Calculate rolling volatility
        realized_vol = daily_vol.rolling(window=window).mean()
        
        # Annualize if requested
        if annualize:
            trading_days = 252
            realized_vol = realized_vol * np.sqrt(trading_days)
            
        # Store result
        method_name = f"garman_klass_{window}d"
        self.results[method_name] = realized_vol
        
        return realized_vol
        
    def calculate_rogers_satchell_volatility(self, ohlc_data: pd.DataFrame, 
                                         window: int = 21, 
                                         annualize: bool = True) -> pd.Series:
        """
        Calculate Rogers-Satchell volatility using OHLC data.
        
        Args:
            ohlc_data: DataFrame with open, high, low, close prices
            window: Rolling window size
            annualize: Whether to annualize the volatility
            
        Returns:
            Series with Rogers-Satchell volatility
        """
        if ohlc_data.empty:
            return pd.Series()
            
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in ohlc_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Calculate log returns
        log_ho = np.log(ohlc_data['high'] / ohlc_data['open'])
        log_lo = np.log(ohlc_data['low'] / ohlc_data['open'])
        log_co = np.log(ohlc_data['close'] / ohlc_data['open'])
        
        # Rogers-Satchell estimator
        daily_variance = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        daily_vol = np.sqrt(daily_variance)
        
        # Calculate rolling volatility
        realized_vol = daily_vol.rolling(window=window).mean()
        
        # Annualize if requested
        if annualize:
            trading_days = 252
            realized_vol = realized_vol * np.sqrt(trading_days)
            
        # Store result
        method_name = f"rogers_satchell_{window}d"
        self.results[method_name] = realized_vol
        
        return realized_vol
        
    def calculate_yang_zhang_volatility(self, ohlc_data: pd.DataFrame, 
                                     window: int = 21, 
                                     annualize: bool = True,
                                     k: float = 0.34) -> pd.Series:
        """
        Calculate Yang-Zhang volatility using OHLC data.
        
        Args:
            ohlc_data: DataFrame with open, high, low, close prices
            window: Rolling window size
            annualize: Whether to annualize the volatility
            k: Weighting parameter (default: 0.34)
            
        Returns:
            Series with Yang-Zhang volatility
        """
        if ohlc_data.empty:
            return pd.Series()
            
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in ohlc_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Calculate overnight volatility (close to next day's open)
        ohlc_data['prev_close'] = ohlc_data['close'].shift(1)
        log_co = np.log(ohlc_data['open'] / ohlc_data['prev_close'])
        overnight_var = log_co.rolling(window=window).var()
        
        # Calculate open-close volatility
        log_oc = np.log(ohlc_data['close'] / ohlc_data['open'])
        open_close_var = log_oc.rolling(window=window).var()
        
        # Calculate Rogers-Satchell volatility
        log_ho = np.log(ohlc_data['high'] / ohlc_data['open'])
        log_lo = np.log(ohlc_data['low'] / ohlc_data['open'])
        log_hc = np.log(ohlc_data['high'] / ohlc_data['close'])
        log_lc = np.log(ohlc_data['low'] / ohlc_data['close'])
        
        rs_var = log_ho * log_hc + log_lo * log_lc
        rs_var = rs_var.rolling(window=window).mean()
        
        # Yang-Zhang estimator
        daily_variance = overnight_var + k * open_close_var + (1 - k) * rs_var
        daily_vol = np.sqrt(daily_variance)
        
        # Annualize if requested
        if annualize:
            trading_days = 252
            daily_vol = daily_vol * np.sqrt(trading_days)
            
        # Store result
        method_name = f"yang_zhang_{window}d"
        self.results[method_name] = daily_vol
        
        return daily_vol
        
    def calculate_realized_range_volatility(self, high_low_data: pd.DataFrame, 
                                        window: int = 21, 
                                        annualize: bool = True,
                                        method: str = 'parkinson') -> pd.Series:
        """
        Calculate realized range-based volatility using various methods.
        
        Args:
            high_low_data: DataFrame with high and low prices
            window: Rolling window size
            annualize: Whether to annualize the volatility
            method: Range estimator method ('parkinson', 'garman_klass', 'rogers_satchell')
            
        Returns:
            Series with realized range volatility
        """
        if method == 'parkinson':
            return self.calculate_parkinson_volatility(high_low_data, window, annualize)
        elif method == 'garman_klass':
            return self.calculate_garman_klass_volatility(high_low_data, window, annualize)
        elif method == 'rogers_satchell':
            return self.calculate_rogers_satchell_volatility(high_low_data, window, annualize)
        else:
            raise ValueError(f"Unknown method: {method}")
            
    def calculate_realized_variance(self, returns: pd.Series, 
                                window: int = 21, 
                                annualize: bool = True) -> pd.Series:
        """
        Calculate realized variance using squared returns.
        
        Args:
            returns: Time series of returns
            window: Rolling window size
            annualize: Whether to annualize the variance
            
        Returns:
            Series with realized variance
        """
        if returns.empty:
            return pd.Series()
            
        # Calculate squared returns
        squared_returns = returns ** 2
        
        # Calculate rolling sum
        realized_var = squared_returns.rolling(window=window).sum()
        
        # Annualize if requested
        if annualize:
            trading_days = 252
            realized_var = realized_var * trading_days / window
            
        # Store result
        method_name = f"variance_{window}d"
        self.results[method_name] = realized_var
        
        return realized_var
        
    def calculate_bipower_variation(self, returns: pd.Series, 
                                window: int = 21, 
                                annualize: bool = True) -> pd.Series:
        """
        Calculate realized bipower variation (robust to jumps).
        
        Args:
            returns: Time series of returns
            window: Rolling window size
            annualize: Whether to annualize the result
            
        Returns:
            Series with bipower variation
        """
        if returns.empty or len(returns) <= window:
            return pd.Series()
            
        # Calculate absolute returns
        abs_returns = returns.abs()
        
        # Calculate product of consecutive absolute returns
        abs_product = abs_returns * abs_returns.shift(1)
        
        # Constant for scaling
        pi_over_2 = np.pi / 2
        scale_factor = pi_over_2
        
        # Calculate rolling sum of products
        bipower_var = abs_product.rolling(window=window).sum() * scale_factor
        
        # Annualize if requested
        if annualize:
            trading_days = 252
            bipower_var = bipower_var * trading_days / window
            
        # Calculate bipower volatility (square root of variation)
        bipower_vol = np.sqrt(bipower_var)
        
        # Store result
        method_name = f"bipower_{window}d"
        self.results[method_name] = bipower_vol
        
        return bipower_vol
        
    def calculate_jump_component(self, returns: pd.Series, 
                              window: int = 21, 
                              annualize: bool = True) -> pd.Series:
        """
        Calculate the jump component of realized volatility.
        
        Args:
            returns: Time series of returns
            window: Rolling window size
            annualize: Whether to annualize the result
            
        Returns:
            Series with jump component
        """
        if returns.empty or len(returns) <= window:
            return pd.Series()
            
        # Calculate realized variance
        rv = self.calculate_realized_variance(returns, window, False)
        
        # Calculate bipower variation
        bv = self.calculate_bipower_variation(returns, window, False) ** 2
        
        # Calculate jump component
        jump_component = (rv - bv).clip(lower=0)
        
        # Annualize if requested
        if annualize:
            trading_days = 252
            jump_component = jump_component * trading_days / window
            
        # Store result
        method_name = f"jump_{window}d"
        self.results[method_name] = jump_component
        
        return jump_component
        
    def calculate_high_frequency_volatility(self, high_freq_returns: pd.DataFrame, 
                                        sampling_freq: str = '5min', 
                                        window_days: int = 5) -> pd.Series:
        """
        Calculate realized volatility using high-frequency data.
        
        Args:
            high_freq_returns: DataFrame with high-frequency returns
            sampling_freq: Sampling frequency for returns
            window_days: Number of days for rolling window
            
        Returns:
            Series with high-frequency volatility
        """
        if high_freq_returns.empty:
            return pd.Series()
            
        # Store high-frequency data
        self.high_frequency_data = high_freq_returns
        
        # Resample to specified frequency if needed
        if sampling_freq is not None:
            resampled_returns = high_freq_returns.resample(sampling_freq).last()
        else:
            resampled_returns = high_freq_returns
            
        # Calculate squared returns
        squared_returns = resampled_returns ** 2
        
        # Calculate daily realized variance
        daily_rv = squared_returns.resample('D').sum()
        
        # Calculate rolling window realized volatility
        window = window_days  # Window in days
        realized_vol = np.sqrt(daily_rv.rolling(window=window).mean() * 252)
        
        # Store result
        method_name = f"high_freq_{sampling_freq}_{window_days}d"
        self.results[method_name] = realized_vol
        
        return realized_vol
        
    def detect_volatility_jumps(self, realized_vol: pd.Series, 
                             threshold: float = 2.0, 
                             window: int = 21) -> pd.DataFrame:
        """
        Detect jumps in realized volatility.
        
        Args:
            realized_vol: Series with realized volatility
            threshold: Z-score threshold for jump detection
            window: Window for calculating volatility baseline
            
        Returns:
            DataFrame with jump indicators
        """
        if realized_vol.empty:
            return pd.DataFrame()
            
        # Calculate rolling mean and std
        rolling_mean = realized_vol.rolling(window=window).mean()
        rolling_std = realized_vol.rolling(window=window).std()
        
        # Calculate z-score
        z_score = (realized_vol - rolling_mean) / rolling_std
        
        # Detect jumps
        jumps = (z_score.abs() > threshold)
        
        # Create results DataFrame
        jump_df = pd.DataFrame({
            'realized_vol': realized_vol,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'z_score': z_score,
            'is_jump': jumps
        })
        
        return jump_df
        
    def forecast_realized_volatility(self, realized_vol: pd.Series, 
                                 horizon: int = 5, 
                                 method: str = 'har') -> pd.Series:
        """
        Forecast realized volatility using HAR or other models.
        
        Args:
            realized_vol: Series with realized volatility
            horizon: Forecast horizon
            method: Forecasting method ('har', 'arma', 'ewma')
            
        Returns:
            Series with volatility forecasts
        """
        if realized_vol.empty:
            return pd.Series()
            
        if method == 'har':
            # Heterogeneous Autoregressive (HAR) model
            return self._forecast_har(realized_vol, horizon)
        elif method == 'arma':
            # ARMA model
            return self._forecast_arma(realized_vol, horizon)
        elif method == 'ewma':
            # Exponentially Weighted Moving Average
            return self._forecast_ewma(realized_vol, horizon)
        else:
            raise ValueError(f"Unknown method: {method}")
            
    def plot_realized_volatility(self, method_keys: List[str] = None, 
                              lookback: int = 252) -> None:
        """
        Plot realized volatility for different methods.
        
        Args:
            method_keys: List of method keys to plot
            lookback: Number of days to look back for plotting
            
        Returns:
            None (displays plot)
        """
        if not self.results:
            print("No realized volatility results available")
            return
            
        if method_keys is None:
            # Use all available methods
            method_keys = list(self.results.keys())
            
        plt.figure(figsize=(12, 6))
        
        for method in method_keys:
            if method in self.results:
                vol_series = self.results[method]
                if lookback is not None and lookback < len(vol_series):
                    vol_series = vol_series.iloc[-lookback:]
                    
                plt.plot(vol_series.index, vol_series.values, label=method)
                
        plt.xlabel('Date')
        plt.ylabel('Realized Volatility')
        plt.title('Realized Volatility Comparison')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def plot_volatility_jumps(self, jump_df: pd.DataFrame, lookback: int = 252) -> None:
        """
        Plot realized volatility with detected jumps.
        
        Args:
            jump_df: DataFrame with jump indicators
            lookback: Number of days to look back for plotting
            
        Returns:
            None (displays plot)
        """
        if jump_df.empty:
            print("No jump detection results available")
            return
            
        # Limit to lookback period if specified
        if lookback is not None and lookback < len(jump_df):
            jump_df = jump_df.iloc[-lookback:]
            
        plt.figure(figsize=(12, 6))
        
        # Plot realized volatility
        plt.plot(jump_df.index, jump_df['realized_vol'], 'b-', label='Realized Vol')
        
        # Plot mean and confidence bands
        plt.plot(jump_df.index, jump_df['rolling_mean'], 'k--', label='Rolling Mean')
        plt.fill_between(
            jump_df.index,
            jump_df['rolling_mean'] - 2 * jump_df['rolling_std'],
            jump_df['rolling_mean'] + 2 * jump_df['rolling_std'],
            color='gray', alpha=0.2, label='±2σ Band'
        )
        
        # Highlight jumps
        jump_idx = jump_df[jump_df['is_jump']].index
        jump_values = jump_df.loc[jump_idx, 'realized_vol']
        plt.scatter(jump_idx, jump_values, color='red', s=50, label='Volatility Jump')
        
        plt.xlabel('Date')
        plt.ylabel('Realized Volatility')
        plt.title('Realized Volatility with Detected Jumps')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def _forecast_har(self, realized_vol: pd.Series, horizon: int) -> pd.Series:
        """Forecast using Heterogeneous Autoregressive (HAR) model."""
        # Create lagged features
        vol_df = pd.DataFrame({'vol': realized_vol})
        
        # Daily, weekly, and monthly components
        vol_df['vol_d'] = vol_df['vol'].shift(1)
        vol_df['vol_w'] = vol_df['vol'].rolling(5).mean().shift(1)
        vol_df['vol_m'] = vol_df['vol'].rolling(22).mean().shift(1)
        
        # Drop NaN rows
        vol_df = vol_df.dropna()
        
        if vol_df.empty:
            return pd.Series()
            
        # Fit HAR model
        X = vol_df[['vol_d', 'vol_w', 'vol_m']]
        y = vol_df['vol']
        
        # Simple OLS regression
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        
        # Generate forecasts
        last_obs = vol_df.iloc[-1]
        forecasts = []
        
        # Initial values
        vol_d = last_obs['vol']
        vol_w = last_obs['vol_w']
        vol_m = last_obs['vol_m']
        
        # Generate forecasts iteratively
        for h in range(horizon):
            # Update predictors
            new_vol_d = vol_d
            
            # Update weekly component (assume 5-day weeks)
            if h < 4:
                # Partial update of the week
                new_vol_w = (vol_w * 5 - vol_df['vol'].iloc[-(5-h)] + vol_d) / 5
            else:
                # Complete replacement of the week
                new_vol_w = vol_df['vol'].iloc[-5:].append(pd.Series(forecasts)).mean()
                
            # Update monthly component (assume 22-day months)
            if h < 21:
                # Partial update of the month
                new_vol_m = (vol_m * 22 - vol_df['vol'].iloc[-(22-h)] + vol_d) / 22
            else:
                # Complete replacement of the month
                new_vol_m = vol_df['vol'].iloc[-22:].append(pd.Series(forecasts)).mean()
                
            # Generate forecast
            X_new = np.array([1, new_vol_d, new_vol_w, new_vol_m])
            forecast = model.predict(X_new)[0]
            forecasts.append(forecast)
            
            # Update for next iteration
            vol_d = forecast
            vol_w = new_vol_w
            vol_m = new_vol_m
            
        # Create forecast series
        last_date = realized_vol.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
        forecast_series = pd.Series(forecasts, index=forecast_dates)
        
        return forecast_series
        
    def _forecast_arma(self, realized_vol: pd.Series, horizon: int) -> pd.Series:
        """Forecast using ARMA model."""
        # Import required libraries
        from statsmodels.tsa.arima.model import ARIMA
        
        # Fit ARIMA model (p=1, d=0, q=1 for ARMA(1,1))
        model = ARIMA(realized_vol, order=(1, 0, 1))
        result = model.fit()
        
        # Generate forecasts
        forecasts = result.forecast(steps=horizon)
        
        return forecasts
        
    def _forecast_ewma(self, realized_vol: pd.Series, horizon: int, lambda_param: float = 0.94) -> pd.Series:
        """Forecast using Exponentially Weighted Moving Average."""
        # Calculate the EWMA forecast
        last_vol = realized_vol.iloc[-1]
        
        # For EWMA, all future forecasts are the same
        forecasts = [last_vol] * horizon
        
        # Create forecast series
        last_date = realized_vol.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
        forecast_series = pd.Series(forecasts, index=forecast_dates)
        
        return forecast_series 