import pandas as pd
import numpy as np
import arch
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from arch.univariate import ConstantMean, GARCH, Normal, StudentsT

class GARCHModels:
    """Implementation of GARCH family models for volatility estimation."""
    
    def __init__(self):
        """Initialize the GARCH models."""
        self.models = {}
        self.results = {}
        self.forecasts = {}
        
    def fit_model(self, returns: pd.Series, model_type: str = 'GARCH', p: int = 1, q: int = 1, 
                  distribution: str = 'normal', mean: str = 'constant', vol_scale: float = 100.0) -> Dict[str, Any]:
        """
        Fit a GARCH model to return series.
        
        Args:
            returns: Time series of returns
            model_type: Type of GARCH model ('GARCH', 'EGARCH', 'FIGARCH', 'GJR-GARCH')
            p: GARCH lag order
            q: ARCH lag order
            distribution: Error distribution ('normal', 'studentst', 'skewstudent')
            mean: Mean model specification
            vol_scale: Scale factor for volatility
            
        Returns:
            Dictionary containing model summary and statistics
        """
        # Define model name
        model_name = f"{model_type}({p},{q})_{distribution}"
        
        # Set up mean model
        if mean == 'constant':
            mean_model = ConstantMean(returns)
        else:
            mean_model = ConstantMean(returns)  # Default to constant mean
            
        # Select distribution
        if distribution == 'normal':
            dist = Normal()
        elif distribution == 'studentst':
            dist = StudentsT()
        else:
            dist = Normal()  # Default to normal
            
        # Set up volatility model
        if model_type == 'GARCH':
            vol_model = GARCH(p=p, q=q, rescale=vol_scale)
        elif model_type == 'EGARCH':
            vol_model = arch.univariate.EGARCH(p=p, q=q)
        elif model_type == 'FIGARCH':
            vol_model = arch.univariate.FIGARCH(p=p, q=q)
        elif model_type == 'GJR-GARCH':
            vol_model = arch.univariate.GARCH(p=p, o=1, q=q, rescale=vol_scale)
        else:
            vol_model = GARCH(p=p, q=q, rescale=vol_scale)  # Default to GARCH
            
        # Create model
        model = mean_model
        model.volatility = vol_model
        model.distribution = dist
        
        # Fit model
        result = model.fit(disp='off')
        
        # Store model and result
        self.models[model_name] = model
        self.results[model_name] = result
        
        # Return summary
        return {
            'model_name': model_name,
            'aic': result.aic,
            'bic': result.bic,
            'log_likelihood': result.loglikelihood,
            'parameters': result.params.to_dict(),
            'convergence': result.convergence_flag
        }
        
    def compare_models(self, returns: pd.Series, model_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple GARCH model specifications.
        
        Args:
            returns: Time series of returns
            model_configs: List of model configuration dictionaries
            
        Returns:
            DataFrame with model comparison metrics
        """
        comparison_results = []
        
        for config in model_configs:
            model_type = config.get('model_type', 'GARCH')
            p = config.get('p', 1)
            q = config.get('q', 1)
            distribution = config.get('distribution', 'normal')
            
            result = self.fit_model(
                returns=returns,
                model_type=model_type,
                p=p,
                q=q,
                distribution=distribution
            )
            
            comparison_results.append({
                'model_name': result['model_name'],
                'aic': result['aic'],
                'bic': result['bic'],
                'log_likelihood': result['log_likelihood']
            })
            
        return pd.DataFrame(comparison_results)
        
    def forecast_volatility(self, model_name: str, horizon: int = 10, simulations: int = 1000) -> Dict[str, Any]:
        """
        Forecast volatility using a fitted GARCH model.
        
        Args:
            model_name: Name of the fitted model to use
            horizon: Forecast horizon in days
            simulations: Number of simulations for forecast
            
        Returns:
            Dictionary containing forecast results
        """
        if model_name not in self.results:
            return {'error': 'Model not found'}
            
        # Get model result
        result = self.results[model_name]
        
        # Generate forecast
        forecast = result.forecast(horizon=horizon, method='simulation', simulations=simulations)
        
        # Extract variance forecast
        variance_forecast = forecast.variance.iloc[-horizon:].mean(axis=1)
        volatility_forecast = np.sqrt(variance_forecast)
        
        # Store forecast
        self.forecasts[model_name] = {
            'variance': variance_forecast,
            'volatility': volatility_forecast,
            'forecast_date': datetime.now(),
            'horizon': horizon
        }
        
        return {
            'volatility_forecast': volatility_forecast.to_dict(),
            'horizon': horizon,
            'model_name': model_name
        }
        
    def rolling_volatility(self, returns: pd.Series, model_type: str = 'GARCH', 
                         p: int = 1, q: int = 1, window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling GARCH volatility estimates.
        
        Args:
            returns: Time series of returns
            model_type: Type of GARCH model
            p: GARCH lag order
            q: ARCH lag order
            window: Rolling window size in days
            
        Returns:
            DataFrame with rolling volatility estimates
        """
        # Ensure returns are properly sorted
        returns = returns.sort_index()
        
        # Initialize result arrays
        dates = returns.index[window-1:]
        vol_estimates = np.zeros(len(dates))
        
        # Loop through each window
        for i in range(len(dates)):
            window_returns = returns.iloc[i:i+window]
            
            try:
                # Fit model on window
                model_name = f"{model_type}({p},{q})_rolling"
                result = self.fit_model(
                    returns=window_returns,
                    model_type=model_type,
                    p=p,
                    q=q
                )
                
                # Get volatility estimate (annualized)
                model_result = self.results[result['model_name']]
                conditional_vol = model_result.conditional_volatility
                vol_estimates[i] = conditional_vol.iloc[-1] * np.sqrt(252)
                
            except Exception as e:
                # Handle fitting errors
                vol_estimates[i] = np.nan
                
        # Create result DataFrame
        rolling_vol = pd.DataFrame({
            'date': dates,
            'rolling_volatility': vol_estimates
        })
        rolling_vol.set_index('date', inplace=True)
        
        return rolling_vol
        
    def volatility_impulse_response(self, model_name: str, shock_size: float = 0.05, 
                                  steps: int = 50) -> pd.DataFrame:
        """
        Calculate volatility impulse response to a market shock.
        
        Args:
            model_name: Name of the fitted model to use
            shock_size: Size of return shock (e.g., 0.05 for 5%)
            steps: Number of steps for response calculation
            
        Returns:
            DataFrame with impulse response values
        """
        if model_name not in self.results:
            return pd.DataFrame()
            
        # Get model parameters
        result = self.results[model_name]
        params = result.params
        
        # Simplified IRF calculation for GARCH(1,1)
        if 'GARCH(1,1)' in model_name:
            omega = params['omega']
            alpha = params['alpha[1]']
            beta = params['beta[1]']
            
            # Initialize arrays
            response = np.zeros(steps + 1)
            
            # Initial shock
            response[0] = omega + alpha * shock_size**2
            
            # Calculate propagation
            for i in range(1, steps + 1):
                response[i] = omega + beta * response[i-1]
                
            # Create result DataFrame
            response_df = pd.DataFrame({
                'step': range(steps + 1),
                'volatility_response': np.sqrt(response)
            })
            
            return response_df
            
        else:
            # For other models, use simulation approach
            # This is a simplified placeholder
            return pd.DataFrame({'step': range(steps + 1), 'volatility_response': np.nan})
        
    def plot_volatility_forecast(self, model_name: str, returns: pd.Series = None, 
                              confidence_level: float = 0.95) -> None:
        """
        Plot volatility forecast with confidence intervals.
        
        Args:
            model_name: Name of the fitted model to use
            returns: Historical return series for comparison
            confidence_level: Confidence level for intervals
            
        Returns:
            None (displays plot)
        """
        if model_name not in self.forecasts:
            print(f"Forecast for {model_name} not found")
            return
            
        forecast = self.forecasts[model_name]
        volatility = forecast['volatility']
        
        plt.figure(figsize=(12, 6))
        
        # Plot historical volatility if returns are provided
        if returns is not None:
            hist_vol = returns.rolling(window=20).std() * np.sqrt(252)
            plt.plot(hist_vol.index[-60:], hist_vol.values[-60:], label='Historical Volatility (20-day)')
            
        # Plot forecast
        forecast_dates = pd.date_range(
            start=volatility.index[0], 
            periods=len(volatility)
        )
        
        plt.plot(forecast_dates, volatility, label='GARCH Forecast', color='red')
        
        # Simplified confidence intervals (just for visualization)
        z_score = 1.96  # 95% confidence
        plt.fill_between(
            forecast_dates,
            volatility - z_score * volatility/2,
            volatility + z_score * volatility/2,
            color='red', alpha=0.2, label=f'{int(confidence_level*100)}% Confidence Interval'
        )
        
        plt.title(f'Volatility Forecast - {model_name}')
        plt.xlabel('Date')
        plt.ylabel('Annualized Volatility')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show() 