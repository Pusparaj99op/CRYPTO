import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class MarketImpact:
    """
    Implementation of market impact models for estimating the effect of trades
    on market prices. Includes various impact models and estimation methods.
    """
    
    def __init__(self):
        """Initialize the MarketImpact class."""
        pass
    
    def square_root_model(self, volume: float, 
                         daily_volume: float,
                         volatility: float,
                         alpha: float = 0.5) -> float:
        """
        Calculate market impact using the square root model.
        
        Args:
            volume (float): Trade volume
            daily_volume (float): Average daily volume
            volatility (float): Asset volatility
            alpha (float): Impact parameter
            
        Returns:
            float: Estimated market impact in basis points
        """
        relative_volume = volume / daily_volume
        impact = alpha * volatility * np.sqrt(relative_volume)
        return impact * 10000  # Convert to basis points
    
    def linear_model(self, volume: float,
                    daily_volume: float,
                    volatility: float,
                    beta: float = 0.1) -> float:
        """
        Calculate market impact using the linear model.
        
        Args:
            volume (float): Trade volume
            daily_volume (float): Average daily volume
            volatility (float): Asset volatility
            beta (float): Impact parameter
            
        Returns:
            float: Estimated market impact in basis points
        """
        relative_volume = volume / daily_volume
        impact = beta * volatility * relative_volume
        return impact * 10000  # Convert to basis points
    
    def power_law_model(self, volume: float,
                       daily_volume: float,
                       volatility: float,
                       gamma: float = 0.6) -> float:
        """
        Calculate market impact using the power law model.
        
        Args:
            volume (float): Trade volume
            daily_volume (float): Average daily volume
            volatility (float): Asset volatility
            gamma (float): Power law exponent
            
        Returns:
            float: Estimated market impact in basis points
        """
        relative_volume = volume / daily_volume
        impact = volatility * relative_volume ** gamma
        return impact * 10000  # Convert to basis points
    
    def estimate_impact_parameters(self, 
                                 trades: pd.DataFrame,
                                 price_data: pd.DataFrame,
                                 model_type: str = 'square_root') -> Dict[str, float]:
        """
        Estimate market impact parameters from historical data.
        
        Args:
            trades (pd.DataFrame): DataFrame with trade data
            price_data (pd.DataFrame): DataFrame with price data
            model_type (str): Type of model to fit ('square_root', 'linear', 'power_law')
            
        Returns:
            Dict: Estimated parameters and model statistics
        """
        # Calculate relative volumes and returns
        daily_volumes = trades.groupby(trades.index.date)['volume'].sum()
        relative_volumes = trades['volume'] / daily_volumes[trades.index.date].values
        
        # Calculate price impact
        returns = price_data['close'].pct_change()
        impact = returns.shift(-1)  # Next period return as impact
        
        # Prepare data for regression
        X = relative_volumes.values.reshape(-1, 1)
        y = impact.values
        
        if model_type == 'square_root':
            X = np.sqrt(X)
        elif model_type == 'power_law':
            X = np.log(X)
            y = np.log(np.abs(y))
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate statistics
        y_pred = model.predict(X)
        r2 = model.score(X, y)
        
        return {
            'coefficient': model.coef_[0],
            'intercept': model.intercept_,
            'r_squared': r2,
            'model_type': model_type
        }
    
    def estimate_temporary_impact(self,
                                trades: pd.DataFrame,
                                price_data: pd.DataFrame,
                                window: int = 30) -> Dict[str, Any]:
        """
        Estimate temporary market impact using a rolling window approach.
        
        Args:
            trades (pd.DataFrame): DataFrame with trade data
            price_data (pd.DataFrame): DataFrame with price data
            window (int): Rolling window size
            
        Returns:
            Dict: Temporary impact estimates and statistics
        """
        # Calculate price changes and trade volumes
        price_changes = price_data['close'].pct_change()
        volumes = trades['volume']
        
        # Calculate rolling impact
        impact = []
        for i in range(len(price_changes) - window):
            window_changes = price_changes.iloc[i:i+window]
            window_volumes = volumes.iloc[i:i+window]
            
            # Calculate impact as correlation between volume and price change
            correlation = np.corrcoef(window_volumes, window_changes)[0,1]
            impact.append(correlation)
        
        impact = np.array(impact)
        
        return {
            'mean_impact': np.mean(impact),
            'std_impact': np.std(impact),
            'max_impact': np.max(impact),
            'min_impact': np.min(impact),
            'impact_series': impact
        }
    
    def estimate_permanent_impact(self,
                                trades: pd.DataFrame,
                                price_data: pd.DataFrame,
                                decay_rate: float = 0.1) -> Dict[str, Any]:
        """
        Estimate permanent market impact with exponential decay.
        
        Args:
            trades (pd.DataFrame): DataFrame with trade data
            price_data (pd.DataFrame): DataFrame with price data
            decay_rate (float): Impact decay rate
            
        Returns:
            Dict: Permanent impact estimates and statistics
        """
        # Calculate price changes and trade volumes
        price_changes = price_data['close'].pct_change()
        volumes = trades['volume']
        
        # Calculate cumulative impact with decay
        impact = np.zeros(len(price_changes))
        for i in range(1, len(impact)):
            impact[i] = impact[i-1] * (1 - decay_rate) + volumes[i] * price_changes[i]
        
        return {
            'mean_impact': np.mean(impact),
            'std_impact': np.std(impact),
            'max_impact': np.max(impact),
            'min_impact': np.min(impact),
            'impact_series': impact,
            'decay_rate': decay_rate
        }
    
    def plot_impact_analysis(self,
                           trades: pd.DataFrame,
                           price_data: pd.DataFrame,
                           window: int = 30) -> plt.Figure:
        """
        Plot market impact analysis results.
        
        Args:
            trades (pd.DataFrame): DataFrame with trade data
            price_data (pd.DataFrame): DataFrame with price data
            window (int): Rolling window size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Calculate temporary and permanent impact
        temp_impact = self.estimate_temporary_impact(trades, price_data, window)
        perm_impact = self.estimate_permanent_impact(trades, price_data)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot temporary impact
        ax1.plot(temp_impact['impact_series'], label='Temporary Impact')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Impact')
        ax1.set_title('Temporary Market Impact')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot permanent impact
        ax2.plot(perm_impact['impact_series'], label='Permanent Impact')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Impact')
        ax2.set_title('Permanent Market Impact')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def analyze_market_impact(self,
                            trades: pd.DataFrame,
                            price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive market impact analysis.
        
        Args:
            trades (pd.DataFrame): DataFrame with trade data
            price_data (pd.DataFrame): DataFrame with price data
            
        Returns:
            Dict: Comprehensive market impact analysis results
        """
        # Estimate parameters for different models
        square_root_params = self.estimate_impact_parameters(trades, price_data, 'square_root')
        linear_params = self.estimate_impact_parameters(trades, price_data, 'linear')
        power_law_params = self.estimate_impact_parameters(trades, price_data, 'power_law')
        
        # Estimate temporary and permanent impact
        temp_impact = self.estimate_temporary_impact(trades, price_data)
        perm_impact = self.estimate_permanent_impact(trades, price_data)
        
        return {
            'model_parameters': {
                'square_root': square_root_params,
                'linear': linear_params,
                'power_law': power_law_params
            },
            'temporary_impact': temp_impact,
            'permanent_impact': perm_impact,
            'impact_ratio': {
                'temporary_to_permanent': temp_impact['mean_impact'] / perm_impact['mean_impact']
            }
        } 