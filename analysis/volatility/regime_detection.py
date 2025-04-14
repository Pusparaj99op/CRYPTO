import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
from scipy.signal import find_peaks
from scipy.stats import norm
from hmmlearn import hmm

class VolatilityRegimeDetector:
    """Implementation of volatility regime detection methods."""
    
    def __init__(self):
        """Initialize the volatility regime detector."""
        self.models = {}
        self.regimes = {}
        self.metrics = {}
        
    def detect_regimes(self, volatility: pd.Series, method: str = 'hmm', 
                     n_regimes: int = 2, **kwargs) -> pd.DataFrame:
        """
        Detect volatility regimes in time series.
        
        Args:
            volatility: Time series of volatility
            method: Method for regime detection ('hmm', 'kmeans', 'gmm', 'threshold')
            n_regimes: Number of regimes to detect
            **kwargs: Additional parameters for specific methods
            
        Returns:
            DataFrame with detected regimes
        """
        vol_values = volatility.values.reshape(-1, 1)
        
        if method == 'hmm':
            model, regimes = self._detect_hmm(vol_values, n_regimes, **kwargs)
        elif method == 'kmeans':
            model, regimes = self._detect_kmeans(vol_values, n_regimes, **kwargs)
        elif method == 'gmm':
            model, regimes = self._detect_gmm(vol_values, n_regimes, **kwargs)
        elif method == 'threshold':
            thresholds = kwargs.get('thresholds', None)
            if thresholds is None:
                # Auto-detect thresholds
                thresholds = self._auto_threshold(vol_values, n_regimes)
            model, regimes = self._detect_threshold(vol_values, thresholds, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Store model and results
        model_name = f"{method}_{n_regimes}"
        self.models[model_name] = model
        
        # Create result DataFrame
        regime_df = pd.DataFrame({
            'date': volatility.index,
            'volatility': volatility.values,
            'regime': regimes
        })
        
        # Calculate some regime stats
        regime_stats = {}
        for regime in range(n_regimes):
            regime_data = regime_df[regime_df['regime'] == regime]['volatility']
            regime_stats[regime] = {
                'mean': regime_data.mean(),
                'std': regime_data.std(),
                'min': regime_data.min(),
                'max': regime_data.max(),
                'count': len(regime_data)
            }
            
        self.regimes[model_name] = {
            'data': regime_df,
            'stats': regime_stats
        }
        
        return regime_df
        
    def forecast_regime(self, volatility: pd.Series, model_name: str, 
                      horizon: int = 5) -> Dict[str, Any]:
        """
        Forecast volatility regime for future periods.
        
        Args:
            volatility: Historical volatility time series
            model_name: Name of the regime detection model
            horizon: Forecast horizon
            
        Returns:
            Dictionary with regime forecast probabilities
        """
        if model_name not in self.models:
            return {'error': 'Model not found'}
            
        model = self.models[model_name]
        method = model_name.split('_')[0]
        
        vol_values = volatility.values.reshape(-1, 1)
        
        if method == 'hmm':
            # For HMM, forecast the hidden state probabilities
            return self._forecast_hmm(model, vol_values, horizon)
        elif method == 'threshold':
            # For threshold model, use AR model to forecast volatility and then determine regime
            return self._forecast_threshold(model, vol_values, horizon)
        elif method == 'kmeans' or method == 'gmm':
            # For clustering models, forecast is less meaningful but we can try with AR
            return self._forecast_clustering(model, vol_values, horizon, method)
        else:
            return {'error': 'Forecasting not implemented for this method'}
        
    def detect_regime_change(self, volatility: pd.Series, window: int = 20, 
                          method: str = 'zscore', threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect points where volatility regime changes.
        
        Args:
            volatility: Time series of volatility
            window: Window size for change detection
            method: Change detection method ('zscore', 'cusum', 'ewma')
            threshold: Threshold for change detection
            
        Returns:
            DataFrame with change points
        """
        if volatility.empty:
            return pd.DataFrame()
            
        if method == 'zscore':
            return self._detect_zscore_change(volatility, window, threshold)
        elif method == 'cusum':
            return self._detect_cusum_change(volatility, window, threshold)
        elif method == 'ewma':
            return self._detect_ewma_change(volatility, window, threshold)
        else:
            raise ValueError(f"Unknown method: {method}")
        
    def analyze_regime_persistence(self, model_name: str) -> Dict[str, Any]:
        """
        Analyze persistence of volatility regimes.
        
        Args:
            model_name: Name of the regime detection model
            
        Returns:
            Dictionary with regime persistence statistics
        """
        if model_name not in self.regimes:
            return {'error': 'Model results not found'}
            
        regime_data = self.regimes[model_name]['data']
        
        # Count regime durations
        durations = []
        current_regime = regime_data['regime'].iloc[0]
        current_duration = 1
        
        for i in range(1, len(regime_data)):
            if regime_data['regime'].iloc[i] == current_regime:
                current_duration += 1
            else:
                durations.append({
                    'regime': current_regime,
                    'duration': current_duration,
                    'start_idx': i - current_duration,
                    'end_idx': i - 1
                })
                current_regime = regime_data['regime'].iloc[i]
                current_duration = 1
                
        # Add last regime
        if current_duration > 0:
            durations.append({
                'regime': current_regime,
                'duration': current_duration,
                'start_idx': len(regime_data) - current_duration,
                'end_idx': len(regime_data) - 1
            })
            
        # Calculate persistence metrics
        persistence = {}
        unique_regimes = regime_data['regime'].unique()
        
        for regime in unique_regimes:
            regime_durations = [d['duration'] for d in durations if d['regime'] == regime]
            
            if regime_durations:
                persistence[int(regime)] = {
                    'mean_duration': np.mean(regime_durations),
                    'median_duration': np.median(regime_durations),
                    'max_duration': np.max(regime_durations),
                    'min_duration': np.min(regime_durations),
                    'count': len(regime_durations)
                }
            else:
                persistence[int(regime)] = {
                    'mean_duration': 0,
                    'median_duration': 0,
                    'max_duration': 0,
                    'min_duration': 0,
                    'count': 0
                }
                
        return {
            'persistence': persistence,
            'transitions': self._calculate_transition_matrix(regime_data['regime'])
        }
        
    def evaluate_regimes(self, volatility: pd.Series, returns: pd.Series, 
                      model_name: str) -> Dict[str, Any]:
        """
        Evaluate regime detection performance using returns.
        
        Args:
            volatility: Volatility time series
            returns: Returns time series
            model_name: Name of the regime detection model
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model_name not in self.regimes:
            return {'error': 'Model results not found'}
            
        regime_data = self.regimes[model_name]['data'].copy()
        
        # Align returns with volatility
        common_idx = regime_data.index.intersection(returns.index)
        aligned_returns = returns.loc[common_idx]
        regime_data = regime_data.loc[common_idx]
        
        unique_regimes = regime_data['regime'].unique()
        metrics = {}
        
        for regime in unique_regimes:
            regime_returns = aligned_returns[regime_data['regime'] == regime]
            
            if not regime_returns.empty:
                metrics[int(regime)] = {
                    'mean_return': regime_returns.mean(),
                    'std_return': regime_returns.std(),
                    'sharpe': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                    'skew': regime_returns.skew(),
                    'kurtosis': regime_returns.kurtosis(),
                    'var_95': np.percentile(regime_returns, 5),
                    'count': len(regime_returns)
                }
            else:
                metrics[int(regime)] = {
                    'mean_return': 0,
                    'std_return': 0,
                    'sharpe': 0,
                    'skew': 0,
                    'kurtosis': 0,
                    'var_95': 0,
                    'count': 0
                }
                
        # Store metrics
        self.metrics[model_name] = metrics
        
        return metrics
        
    def plot_regimes(self, model_name: str, volatility: pd.Series = None, 
                  returns: pd.Series = None) -> None:
        """
        Plot detected volatility regimes.
        
        Args:
            model_name: Name of the regime detection model
            volatility: Optional volatility time series
            returns: Optional returns time series
            
        Returns:
            None (displays plot)
        """
        if model_name not in self.regimes:
            print(f"Model {model_name} not found")
            return
            
        regime_data = self.regimes[model_name]['data']
        regime_stats = self.regimes[model_name]['stats']
        
        num_regimes = len(regime_stats)
        colors = plt.cm.viridis(np.linspace(0, 1, num_regimes))
        
        fig, axs = plt.subplots(3 if returns is not None else 2, 1, 
                             figsize=(12, 12 if returns is not None else 8),
                             sharex=True)
        
        # Plot 1: Volatility with regime colors
        ax = axs[0]
        for regime in range(num_regimes):
            mask = regime_data['regime'] == regime
            ax.scatter(regime_data.index[mask], regime_data['volatility'][mask],
                     label=f"Regime {regime}", color=colors[regime], s=25, alpha=0.6)
        
        # If additional volatility data provided, plot it too
        if volatility is not None and isinstance(volatility, pd.Series):
            all_idx = regime_data.index.union(volatility.index)
            full_vol = pd.Series(index=all_idx, dtype=float)
            full_vol.loc[volatility.index] = volatility
            
            # Plot full volatility as a line
            ax.plot(full_vol.index, full_vol.values, color='gray', alpha=0.3, 
                  label='Full Volatility')
        
        ax.set_title('Volatility Regimes')
        ax.set_ylabel('Volatility')
        ax.legend()
        ax.grid(True)
        
        # Plot 2: Regime distribution
        ax = axs[1]
        for regime in range(num_regimes):
            mask = regime_data['regime'] == regime
            regime_vol = regime_data['volatility'][mask]
            
            if len(regime_vol) > 0:
                ax.hist(regime_vol, bins=20, alpha=0.5, color=colors[regime], 
                      label=f"Regime {regime}: μ={regime_stats[regime]['mean']:.4f}")
        
        ax.set_title('Volatility Distribution by Regime')
        ax.set_xlabel('Volatility')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True)
        
        # Plot 3: Returns by regime (if provided)
        if returns is not None:
            ax = axs[2]
            common_idx = regime_data.index.intersection(returns.index)
            aligned_returns = returns.loc[common_idx]
            aligned_regime_data = regime_data.loc[common_idx]
            
            for regime in range(num_regimes):
                mask = aligned_regime_data['regime'] == regime
                regime_returns = aligned_returns[mask]
                
                if len(regime_returns) > 0:
                    ax.hist(regime_returns, bins=20, alpha=0.5, color=colors[regime], 
                          label=f"Regime {regime}: μ={regime_returns.mean():.4f}")
            
            ax.set_title('Returns Distribution by Regime')
            ax.set_xlabel('Returns')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def _detect_hmm(self, volatility: np.ndarray, n_regimes: int, **kwargs) -> Tuple[Any, np.ndarray]:
        """Detect regimes using Hidden Markov Model."""
        # Set default parameters
        n_iter = kwargs.get('n_iter', 1000)
        random_state = kwargs.get('random_state', 42)
        
        # Initialize and fit HMM
        model = hmm.GaussianHMM(n_components=n_regimes, 
                             covariance_type="diag", 
                             n_iter=n_iter,
                             random_state=random_state)
        model.fit(volatility)
        
        # Predict hidden states
        hidden_states = model.predict(volatility)
        
        # Reorder states by volatility level (lowest to highest)
        means = model.means_.flatten()
        order = np.argsort(means)
        mapping = {old: new for new, old in enumerate(order)}
        reordered_states = np.array([mapping[state] for state in hidden_states])
        
        return model, reordered_states
        
    def _detect_kmeans(self, volatility: np.ndarray, n_regimes: int, **kwargs) -> Tuple[Any, np.ndarray]:
        """Detect regimes using K-means clustering."""
        # Set default parameters
        random_state = kwargs.get('random_state', 42)
        
        # Initialize and fit K-means
        model = KMeans(n_clusters=n_regimes, random_state=random_state)
        clusters = model.fit_predict(volatility)
        
        # Reorder clusters by volatility level (lowest to highest)
        centers = [model.cluster_centers_[i][0] for i in range(n_regimes)]
        order = np.argsort(centers)
        mapping = {old: new for new, old in enumerate(order)}
        reordered_clusters = np.array([mapping[cluster] for cluster in clusters])
        
        return model, reordered_clusters
        
    def _detect_gmm(self, volatility: np.ndarray, n_regimes: int, **kwargs) -> Tuple[Any, np.ndarray]:
        """Detect regimes using Gaussian Mixture Model."""
        # Set default parameters
        random_state = kwargs.get('random_state', 42)
        
        # Initialize and fit GMM
        model = GaussianMixture(n_components=n_regimes, random_state=random_state)
        model.fit(volatility)
        clusters = model.predict(volatility)
        
        # Reorder clusters by volatility level (lowest to highest)
        centers = [model.means_[i][0] for i in range(n_regimes)]
        order = np.argsort(centers)
        mapping = {old: new for new, old in enumerate(order)}
        reordered_clusters = np.array([mapping[cluster] for cluster in clusters])
        
        return model, reordered_clusters
        
    def _detect_threshold(self, volatility: np.ndarray, thresholds: List[float], **kwargs) -> Tuple[Dict, np.ndarray]:
        """Detect regimes using fixed thresholds."""
        # Create a simple threshold model
        model = {'thresholds': thresholds}
        
        # Classify each point
        vol_flat = volatility.flatten()
        regimes = np.zeros(len(vol_flat), dtype=int)
        
        # Apply thresholds
        for i, threshold in enumerate(thresholds):
            regimes[vol_flat >= threshold] = i + 1
            
        return model, regimes
        
    def _auto_threshold(self, volatility: np.ndarray, n_regimes: int) -> List[float]:
        """Automatically determine thresholds for regime classification."""
        vol_flat = volatility.flatten()
        
        if n_regimes <= 1:
            return []
            
        # Use percentiles for thresholds
        percentiles = np.linspace(0, 100, n_regimes + 1)[1:-1]
        thresholds = [np.percentile(vol_flat, p) for p in percentiles]
        
        return thresholds
        
    def _forecast_hmm(self, model: Any, volatility: np.ndarray, horizon: int) -> Dict[str, Any]:
        """Forecast regime using HMM."""
        # Get transition matrix and current state
        trans_mat = model.transmat_
        current_state = model.predict(volatility)[-1]
        
        # Initialize forecasted state probabilities
        state_probs = np.zeros((horizon, model.n_components))
        state_probs[0, current_state] = 1.0
        
        # Forecast state probabilities
        for t in range(1, horizon):
            state_probs[t] = np.dot(state_probs[t-1], trans_mat)
            
        # Create forecast dict
        forecast = {
            'horizon': horizon,
            'state_probabilities': state_probs,
            'most_likely_states': np.argmax(state_probs, axis=1)
        }
        
        return forecast
        
    def _forecast_threshold(self, model: Dict, volatility: np.ndarray, horizon: int) -> Dict[str, Any]:
        """Forecast regime using threshold model and AR for volatility."""
        # Fit AR model to volatility
        vol_flat = volatility.flatten()
        ar_model = AutoReg(vol_flat, lags=min(5, len(vol_flat) // 10)).fit()
        
        # Generate volatility forecasts
        vol_forecast = ar_model.forecast(steps=horizon)
        
        # Apply thresholds to determine regimes
        thresholds = model['thresholds']
        regime_forecast = np.zeros(horizon, dtype=int)
        
        for i, threshold in enumerate(thresholds):
            regime_forecast[vol_forecast >= threshold] = i + 1
            
        # Create forecast dict
        forecast = {
            'horizon': horizon,
            'volatility_forecast': vol_forecast,
            'regime_forecast': regime_forecast
        }
        
        return forecast
        
    def _forecast_clustering(self, model: Any, volatility: np.ndarray, horizon: int, method: str) -> Dict[str, Any]:
        """Forecast regime using clustering models and AR for volatility."""
        # Fit AR model to volatility
        vol_flat = volatility.flatten()
        ar_model = AutoReg(vol_flat, lags=min(5, len(vol_flat) // 10)).fit()
        
        # Generate volatility forecasts
        vol_forecast = ar_model.forecast(steps=horizon)
        vol_forecast_2d = vol_forecast.reshape(-1, 1)
        
        # Predict cluster for forecasted volatility
        if method == 'kmeans':
            regime_forecast = model.predict(vol_forecast_2d)
        elif method == 'gmm':
            regime_forecast = model.predict(vol_forecast_2d)
        else:
            regime_forecast = np.zeros(horizon)
            
        # Create forecast dict
        forecast = {
            'horizon': horizon,
            'volatility_forecast': vol_forecast,
            'regime_forecast': regime_forecast
        }
        
        return forecast
        
    def _detect_zscore_change(self, volatility: pd.Series, window: int, threshold: float) -> pd.DataFrame:
        """Detect regime changes using Z-score method."""
        # Calculate rolling mean and std
        rolling_mean = volatility.rolling(window=window).mean()
        rolling_std = volatility.rolling(window=window).std()
        
        # Calculate Z-score
        z_scores = (volatility - rolling_mean) / rolling_std
        
        # Detect change points
        change_points = []
        for i in range(window, len(volatility)):
            if abs(z_scores.iloc[i]) > threshold:
                change_points.append({
                    'date': volatility.index[i],
                    'volatility': volatility.iloc[i],
                    'z_score': z_scores.iloc[i]
                })
                
        return pd.DataFrame(change_points)
        
    def _detect_cusum_change(self, volatility: pd.Series, window: int, threshold: float) -> pd.DataFrame:
        """Detect regime changes using CUSUM method."""
        # Calculate rolling mean as reference
        rolling_mean = volatility.rolling(window=window).mean()
        
        # Initialize CUSUM statistics
        S_pos = np.zeros(len(volatility))
        S_neg = np.zeros(len(volatility))
        
        # Calculate CUSUM
        for i in range(window, len(volatility)):
            # Get deviation from mean
            deviation = volatility.iloc[i] - rolling_mean.iloc[i]
            
            # Update CUSUM statistics
            S_pos[i] = max(0, S_pos[i-1] + deviation)
            S_neg[i] = max(0, S_neg[i-1] - deviation)
            
        # Detect change points
        change_points = []
        for i in range(window, len(volatility)):
            if S_pos[i] > threshold * rolling_mean.iloc[i] or S_neg[i] > threshold * rolling_mean.iloc[i]:
                change_points.append({
                    'date': volatility.index[i],
                    'volatility': volatility.iloc[i],
                    'S_pos': S_pos[i],
                    'S_neg': S_neg[i]
                })
                
                # Reset CUSUM after detection
                S_pos[i] = 0
                S_neg[i] = 0
                
        return pd.DataFrame(change_points)
        
    def _detect_ewma_change(self, volatility: pd.Series, window: int, threshold: float) -> pd.DataFrame:
        """Detect regime changes using EWMA method."""
        # Calculate EWMA
        alpha = 2 / (window + 1)
        ewma = volatility.ewm(alpha=alpha).mean()
        
        # Calculate residuals
        residuals = volatility - ewma
        
        # Calculate rolling std of residuals
        rolling_std = residuals.rolling(window=window).std()
        
        # Detect change points when residual > threshold * std
        change_points = []
        for i in range(window, len(volatility)):
            if abs(residuals.iloc[i]) > threshold * rolling_std.iloc[i]:
                change_points.append({
                    'date': volatility.index[i],
                    'volatility': volatility.iloc[i],
                    'residual': residuals.iloc[i],
                    'std_threshold': threshold * rolling_std.iloc[i]
                })
                
        return pd.DataFrame(change_points)
        
    def _calculate_transition_matrix(self, regimes: pd.Series) -> np.ndarray:
        """Calculate regime transition probability matrix."""
        n_regimes = len(np.unique(regimes))
        trans_mat = np.zeros((n_regimes, n_regimes))
        
        # Count transitions
        for i in range(len(regimes) - 1):
            from_regime = regimes.iloc[i]
            to_regime = regimes.iloc[i + 1]
            trans_mat[int(from_regime), int(to_regime)] += 1
            
        # Convert to probabilities
        for i in range(n_regimes):
            row_sum = trans_mat[i].sum()
            if row_sum > 0:
                trans_mat[i] /= row_sum
                
        return trans_mat 