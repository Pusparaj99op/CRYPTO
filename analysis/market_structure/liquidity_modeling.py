import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

class LiquidityModeling:
    """
    Implementation of liquidity modeling and analysis tools.
    Includes measures of market depth, resilience, and transaction costs.
    """
    
    def __init__(self):
        """Initialize the LiquidityModeling class."""
        pass
    
    def calculate_amihud_illiquidity(self, 
                                   price_data: pd.DataFrame,
                                   volume_data: pd.DataFrame,
                                   window: int = 20) -> pd.Series:
        """
        Calculate Amihud illiquidity measure.
        
        Args:
            price_data (pd.DataFrame): DataFrame with price data
            volume_data (pd.DataFrame): DataFrame with volume data
            window (int): Rolling window size
            
        Returns:
            pd.Series: Amihud illiquidity measure
        """
        returns = price_data['close'].pct_change()
        volume = volume_data['volume']
        
        # Calculate daily illiquidity
        illiquidity = np.abs(returns) / volume
        
        # Calculate rolling average
        rolling_illiquidity = illiquidity.rolling(window=window).mean()
        
        return rolling_illiquidity
    
    def calculate_kyle_lambda(self,
                            trades: pd.DataFrame,
                            price_data: pd.DataFrame,
                            window: int = 30) -> pd.Series:
        """
        Calculate Kyle's lambda (price impact coefficient).
        
        Args:
            trades (pd.DataFrame): DataFrame with trade data
            price_data (pd.DataFrame): DataFrame with price data
            window (int): Rolling window size
            
        Returns:
            pd.Series: Kyle's lambda series
        """
        # Calculate price changes and order flow
        price_changes = price_data['close'].pct_change()
        order_flow = trades['volume'] * np.sign(trades['price'] - trades['mid_price'])
        
        # Calculate rolling lambda
        lambda_series = []
        for i in range(len(price_changes) - window):
            window_changes = price_changes.iloc[i:i+window]
            window_flow = order_flow.iloc[i:i+window]
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(window_flow.values.reshape(-1, 1), window_changes.values)
            lambda_series.append(model.coef_[0])
        
        return pd.Series(lambda_series, index=price_changes.index[window:])
    
    def calculate_roll_spread(self,
                            price_data: pd.DataFrame,
                            window: int = 20) -> pd.Series:
        """
        Calculate Roll's effective spread estimator.
        
        Args:
            price_data (pd.DataFrame): DataFrame with price data
            window (int): Rolling window size
            
        Returns:
            pd.Series: Roll's spread estimator
        """
        # Calculate price changes
        price_changes = price_data['close'].diff()
        
        # Calculate covariance of consecutive price changes
        cov_changes = price_changes.rolling(window=2).cov(price_changes.shift(1))
        
        # Calculate spread
        spread = 2 * np.sqrt(-cov_changes)
        
        # Calculate rolling average
        rolling_spread = spread.rolling(window=window).mean()
        
        return rolling_spread
    
    def calculate_volume_imbalance(self,
                                 trades: pd.DataFrame,
                                 window: int = 30) -> pd.Series:
        """
        Calculate volume imbalance as a liquidity measure.
        
        Args:
            trades (pd.DataFrame): DataFrame with trade data
            window (int): Rolling window size
            
        Returns:
            pd.Series: Volume imbalance series
        """
        # Calculate buy and sell volumes
        buy_volume = trades[trades['side'] == 'buy']['volume']
        sell_volume = trades[trades['side'] == 'sell']['volume']
        
        # Calculate imbalance
        imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
        
        # Calculate rolling average
        rolling_imbalance = imbalance.rolling(window=window).mean()
        
        return rolling_imbalance
    
    def calculate_liquidity_metrics(self,
                                  trades: pd.DataFrame,
                                  price_data: pd.DataFrame,
                                  volume_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive liquidity metrics.
        
        Args:
            trades (pd.DataFrame): DataFrame with trade data
            price_data (pd.DataFrame): DataFrame with price data
            volume_data (pd.DataFrame): DataFrame with volume data
            
        Returns:
            Dict: Dictionary of liquidity metrics
        """
        # Calculate various liquidity measures
        amihud = self.calculate_amihud_illiquidity(price_data, volume_data)
        kyle_lambda = self.calculate_kyle_lambda(trades, price_data)
        roll_spread = self.calculate_roll_spread(price_data)
        volume_imbalance = self.calculate_volume_imbalance(trades)
        
        # Calculate statistics
        metrics = {
            'amihud_illiquidity': {
                'mean': amihud.mean(),
                'std': amihud.std(),
                'min': amihud.min(),
                'max': amihud.max()
            },
            'kyle_lambda': {
                'mean': kyle_lambda.mean(),
                'std': kyle_lambda.std(),
                'min': kyle_lambda.min(),
                'max': kyle_lambda.max()
            },
            'roll_spread': {
                'mean': roll_spread.mean(),
                'std': roll_spread.std(),
                'min': roll_spread.min(),
                'max': roll_spread.max()
            },
            'volume_imbalance': {
                'mean': volume_imbalance.mean(),
                'std': volume_imbalance.std(),
                'min': volume_imbalance.min(),
                'max': volume_imbalance.max()
            }
        }
        
        # Calculate correlations between measures
        correlations = pd.DataFrame({
            'amihud': amihud,
            'kyle_lambda': kyle_lambda,
            'roll_spread': roll_spread,
            'volume_imbalance': volume_imbalance
        }).corr()
        
        metrics['correlations'] = correlations
        
        return metrics
    
    def estimate_liquidity_regimes(self,
                                 trades: pd.DataFrame,
                                 price_data: pd.DataFrame,
                                 volume_data: pd.DataFrame,
                                 n_regimes: int = 3) -> Dict[str, Any]:
        """
        Estimate liquidity regimes using clustering.
        
        Args:
            trades (pd.DataFrame): DataFrame with trade data
            price_data (pd.DataFrame): DataFrame with price data
            volume_data (pd.DataFrame): DataFrame with volume data
            n_regimes (int): Number of liquidity regimes
            
        Returns:
            Dict: Liquidity regime analysis results
        """
        # Calculate liquidity measures
        amihud = self.calculate_amihud_illiquidity(price_data, volume_data)
        kyle_lambda = self.calculate_kyle_lambda(trades, price_data)
        roll_spread = self.calculate_roll_spread(price_data)
        volume_imbalance = self.calculate_volume_imbalance(trades)
        
        # Combine measures into features
        features = pd.DataFrame({
            'amihud': amihud,
            'kyle_lambda': kyle_lambda,
            'roll_spread': roll_spread,
            'volume_imbalance': volume_imbalance
        }).dropna()
        
        # Normalize features
        normalized_features = (features - features.mean()) / features.std()
        
        # Perform k-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        regimes = kmeans.fit_predict(normalized_features)
        
        # Calculate regime statistics
        regime_stats = {}
        for i in range(n_regimes):
            regime_data = features[regimes == i]
            regime_stats[f'regime_{i}'] = {
                'size': len(regime_data),
                'amihud_mean': regime_data['amihud'].mean(),
                'kyle_lambda_mean': regime_data['kyle_lambda'].mean(),
                'roll_spread_mean': regime_data['roll_spread'].mean(),
                'volume_imbalance_mean': regime_data['volume_imbalance'].mean()
            }
        
        return {
            'regimes': regimes,
            'regime_stats': regime_stats,
            'features': features,
            'cluster_centers': kmeans.cluster_centers_
        }
    
    def plot_liquidity_analysis(self,
                              trades: pd.DataFrame,
                              price_data: pd.DataFrame,
                              volume_data: pd.DataFrame) -> plt.Figure:
        """
        Plot liquidity analysis results.
        
        Args:
            trades (pd.DataFrame): DataFrame with trade data
            price_data (pd.DataFrame): DataFrame with price data
            volume_data (pd.DataFrame): DataFrame with volume data
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Calculate liquidity measures
        amihud = self.calculate_amihud_illiquidity(price_data, volume_data)
        kyle_lambda = self.calculate_kyle_lambda(trades, price_data)
        roll_spread = self.calculate_roll_spread(price_data)
        volume_imbalance = self.calculate_volume_imbalance(trades)
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot Amihud illiquidity
        ax1.plot(amihud, label='Amihud Illiquidity')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Illiquidity')
        ax1.set_title('Amihud Illiquidity Measure')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot Kyle's lambda
        ax2.plot(kyle_lambda, label="Kyle's Lambda")
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Lambda')
        ax2.set_title("Kyle's Lambda")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot Roll's spread
        ax3.plot(roll_spread, label="Roll's Spread")
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Spread')
        ax3.set_title("Roll's Effective Spread")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot volume imbalance
        ax4.plot(volume_imbalance, label='Volume Imbalance')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Imbalance')
        ax4.set_title('Volume Imbalance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def analyze_liquidity_risk(self,
                             trades: pd.DataFrame,
                             price_data: pd.DataFrame,
                             volume_data: pd.DataFrame,
                             confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Analyze liquidity risk using various measures.
        
        Args:
            trades (pd.DataFrame): DataFrame with trade data
            price_data (pd.DataFrame): DataFrame with price data
            volume_data (pd.DataFrame): DataFrame with volume data
            confidence_level (float): Confidence level for risk measures
            
        Returns:
            Dict: Liquidity risk analysis results
        """
        # Calculate liquidity measures
        amihud = self.calculate_amihud_illiquidity(price_data, volume_data)
        kyle_lambda = self.calculate_kyle_lambda(trades, price_data)
        roll_spread = self.calculate_roll_spread(price_data)
        volume_imbalance = self.calculate_volume_imbalance(trades)
        
        # Calculate Value at Risk for each measure
        var_amihud = np.percentile(amihud, (1 - confidence_level) * 100)
        var_kyle = np.percentile(kyle_lambda, (1 - confidence_level) * 100)
        var_spread = np.percentile(roll_spread, (1 - confidence_level) * 100)
        var_imbalance = np.percentile(volume_imbalance, (1 - confidence_level) * 100)
        
        # Calculate Expected Shortfall
        es_amihud = amihud[amihud <= var_amihud].mean()
        es_kyle = kyle_lambda[kyle_lambda <= var_kyle].mean()
        es_spread = roll_spread[roll_spread <= var_spread].mean()
        es_imbalance = volume_imbalance[volume_imbalance <= var_imbalance].mean()
        
        return {
            'value_at_risk': {
                'amihud': var_amihud,
                'kyle_lambda': var_kyle,
                'roll_spread': var_spread,
                'volume_imbalance': var_imbalance
            },
            'expected_shortfall': {
                'amihud': es_amihud,
                'kyle_lambda': es_kyle,
                'roll_spread': es_spread,
                'volume_imbalance': es_imbalance
            },
            'confidence_level': confidence_level
        } 