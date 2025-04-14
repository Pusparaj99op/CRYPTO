import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pymc3 as pm
import arviz as az
import theano.tensor as tt

class StochasticVolatility:
    """Implementation of stochastic volatility models for volatility estimation."""
    
    def __init__(self):
        """Initialize the stochastic volatility models."""
        self.models = {}
        self.traces = {}
        self.forecasts = {}
        
    def fit_basic_model(self, returns: pd.Series, samples: int = 2000, 
                      tune: int = 1000, chains: int = 2) -> Dict[str, Any]:
        """
        Fit a basic stochastic volatility model using Bayesian inference.
        
        Args:
            returns: Time series of returns
            samples: Number of posterior samples
            tune: Number of tuning samples
            chains: Number of MCMC chains
            
        Returns:
            Dictionary containing model summary
        """
        model_name = "basic_stochastic_vol"
        
        # Prepare data
        returns_data = returns.values
        
        # Create model
        with pm.Model() as model:
            # Specify priors for unknown model parameters
            nu = pm.Exponential('nu', 1/10, testval=5)  # Degrees of freedom
            sigma = pm.Exponential('sigma', 0.1)  # Volatility of volatility
            
            # Latent volatility follows an AR(1) process
            h_init = pm.Normal('h_init', 0, 10)
            h = pm.GaussianRandomWalk('h', sigma=sigma, shape=len(returns_data), init=h_init)
            
            # Specify return process
            returns_obs = pm.StudentT('returns_obs', nu=nu, 
                                    lam=tt.exp(-2*h), 
                                    observed=returns_data)
            
            # Fit model - NUTS sampling
            trace = pm.sample(samples, tune=tune, chains=chains, return_inferencedata=True)
            
        # Store model and trace
        self.models[model_name] = model
        self.traces[model_name] = trace
        
        # Calculate some summary statistics
        summary = az.summary(trace, var_names=['nu', 'sigma'])
        
        return {
            'model_name': model_name,
            'summary': summary.to_dict(),
            'num_observations': len(returns_data)
        }
        
    def fit_leverage_model(self, returns: pd.Series, samples: int = 2000, 
                         tune: int = 1000, chains: int = 2) -> Dict[str, Any]:
        """
        Fit a stochastic volatility model with leverage effect.
        
        Args:
            returns: Time series of returns
            samples: Number of posterior samples
            tune: Number of tuning samples
            chains: Number of MCMC chains
            
        Returns:
            Dictionary containing model summary
        """
        model_name = "leverage_stochastic_vol"
        
        # Prepare data
        returns_data = returns.values
        
        # Create model
        with pm.Model() as model:
            # Priors for unknown model parameters
            nu = pm.Exponential('nu', 1/10, testval=5)
            sigma = pm.Exponential('sigma', 0.1)
            
            # Leverage effect parameter (correlation between returns and vol changes)
            rho = pm.Uniform('rho', -1, 1, testval=-0.3)
            
            # Latent volatility
            h_init = pm.Normal('h_init', 0, 10)
            h_innovations = pm.Normal('h_innovations', 0, 1, shape=len(returns_data)-1)
            
            # Specify AR(1) process for log volatility
            h = tt.zeros(len(returns_data))
            h = tt.set_subtensor(h[0], h_init)
            
            for t in range(1, len(returns_data)):
                # Leverage effect: correlation between return and volatility innovation
                mu_t = rho * returns_data[t-1]
                h = tt.set_subtensor(h[t], h[t-1] + mu_t + sigma * h_innovations[t-1])
            
            # Specify return process
            returns_obs = pm.StudentT('returns_obs', nu=nu, 
                                    lam=tt.exp(-2*h), 
                                    observed=returns_data)
            
            # Fit model
            trace = pm.sample(samples, tune=tune, chains=chains, return_inferencedata=True)
            
        # Store model and trace
        self.models[model_name] = model
        self.traces[model_name] = trace
        
        # Calculate summary statistics
        summary = az.summary(trace, var_names=['nu', 'sigma', 'rho'])
        
        return {
            'model_name': model_name,
            'summary': summary.to_dict(),
            'num_observations': len(returns_data)
        }
        
    def fit_jump_diffusion_model(self, returns: pd.Series, samples: int = 2000, 
                              tune: int = 1000, chains: int = 2) -> Dict[str, Any]:
        """
        Fit a stochastic volatility model with jumps.
        
        Args:
            returns: Time series of returns
            samples: Number of posterior samples
            tune: Number of tuning samples
            chains: Number of MCMC chains
            
        Returns:
            Dictionary containing model summary
        """
        model_name = "jump_stochastic_vol"
        
        # Prepare data
        returns_data = returns.values
        
        # Create model
        with pm.Model() as model:
            # Volatility process parameters
            sigma_v = pm.Exponential('sigma_v', 0.1)
            h_init = pm.Normal('h_init', 0, 10)
            h = pm.GaussianRandomWalk('h', sigma=sigma_v, shape=len(returns_data), init=h_init)
            
            # Jump process parameters
            jump_prob = pm.Beta('jump_prob', 2, 40)  # Prior for jump probability
            jump_mean = pm.Normal('jump_mean', 0, 1)  # Prior for jump mean
            jump_sigma = pm.Exponential('jump_sigma', 1)  # Prior for jump volatility
            
            # Jump occurrences
            jumps = pm.Bernoulli('jumps', p=jump_prob, shape=len(returns_data))
            
            # Jump sizes
            jump_sizes = pm.Normal('jump_sizes', mu=jump_mean, sigma=jump_sigma, 
                                 shape=len(returns_data))
            
            # Return process with jumps
            mu = 0  # Assuming zero mean for simplicity
            returns_obs = pm.Normal('returns_obs', 
                                  mu=mu + jumps * jump_sizes,
                                  sigma=tt.exp(h/2),
                                  observed=returns_data)
            
            # Fit model
            trace = pm.sample(samples, tune=tune, chains=chains, return_inferencedata=True)
            
        # Store model and trace
        self.models[model_name] = model
        self.traces[model_name] = trace
        
        # Calculate summary statistics
        summary = az.summary(trace, var_names=['sigma_v', 'jump_prob', 'jump_mean', 'jump_sigma'])
        
        return {
            'model_name': model_name,
            'summary': summary.to_dict(),
            'num_observations': len(returns_data)
        }
        
    def extract_volatility(self, model_name: str) -> pd.Series:
        """
        Extract estimated volatility from a fitted stochastic volatility model.
        
        Args:
            model_name: Name of the fitted model
            
        Returns:
            Series with estimated volatility
        """
        if model_name not in self.traces:
            return pd.Series()
            
        trace = self.traces[model_name]
        
        # Extract volatility samples
        h_samples = trace.posterior['h'].values
        
        # Calculate mean volatility across samples
        h_mean = h_samples.mean(axis=(0, 1))
        volatility = np.exp(h_mean / 2)
        
        # Convert to pandas Series
        if model_name in self.models:
            index = np.arange(len(volatility))
            vol_series = pd.Series(volatility, index=index)
            return vol_series
        else:
            return pd.Series(volatility)
        
    def forecast_volatility(self, model_name: str, returns: pd.Series, 
                          horizon: int = 10, samples: int = 1000) -> Dict[str, Any]:
        """
        Forecast volatility using a fitted stochastic volatility model.
        
        Args:
            model_name: Name of the fitted model
            returns: Historical return series
            horizon: Forecast horizon in days
            samples: Number of forecast samples
            
        Returns:
            Dictionary containing forecast results
        """
        if model_name not in self.traces:
            return {'error': 'Model not found'}
            
        trace = self.traces[model_name]
        
        # Extract parameters
        if model_name == 'basic_stochastic_vol':
            # Get last volatility state and volatility of volatility
            h_last = trace.posterior['h'].values[:, :, -1].flatten()
            sigma = trace.posterior['sigma'].values.flatten()
            
            # Simple AR(1) forecast simulation
            h_forecast = np.zeros((len(h_last), horizon))
            h_forecast[:, 0] = h_last
            
            for t in range(1, horizon):
                for i in range(len(h_last)):
                    h_forecast[i, t] = h_forecast[i, t-1] + np.random.normal(0, sigma[i])
            
            # Convert to volatility
            vol_forecast = np.exp(h_forecast / 2)
            
            # Calculate statistics
            mean_forecast = vol_forecast.mean(axis=0)
            lower = np.percentile(vol_forecast, 5, axis=0)
            upper = np.percentile(vol_forecast, 95, axis=0)
            
            # Create forecast DataFrame
            forecast_dates = pd.date_range(
                start=returns.index[-1] + pd.Timedelta(days=1),
                periods=horizon
            )
            forecast_df = pd.DataFrame({
                'mean': mean_forecast,
                'lower_5': lower,
                'upper_95': upper
            }, index=forecast_dates)
            
            # Store forecast
            self.forecasts[model_name] = forecast_df
            
            return {
                'forecast': forecast_df.to_dict(),
                'horizon': horizon,
                'model_name': model_name
            }
        else:
            # More complex models would need model-specific forecasting logic
            return {'error': 'Forecasting not implemented for this model type'}
        
    def detect_jumps(self, model_name: str, threshold: float = 0.5) -> pd.DataFrame:
        """
        Detect jumps in returns based on a jump diffusion model.
        
        Args:
            model_name: Name of the fitted model (must be jump diffusion)
            threshold: Probability threshold for jump detection
            
        Returns:
            DataFrame with jump probabilities and occurrences
        """
        if model_name != 'jump_stochastic_vol' or model_name not in self.traces:
            return pd.DataFrame()
            
        trace = self.traces[model_name]
        
        # Extract jump probabilities
        jumps = trace.posterior['jumps'].values
        jump_sizes = trace.posterior['jump_sizes'].values
        
        # Calculate posterior probability of jump at each time point
        jump_probs = jumps.mean(axis=(0, 1))
        mean_jump_sizes = jump_sizes.mean(axis=(0, 1))
        
        # Create results DataFrame
        jumps_df = pd.DataFrame({
            'jump_probability': jump_probs,
            'jump_size': mean_jump_sizes,
            'is_jump': jump_probs > threshold
        })
        
        return jumps_df
        
    def plot_volatility(self, model_name: str, returns: pd.Series = None, 
                      ci_level: float = 0.9) -> None:
        """
        Plot estimated volatility with confidence intervals.
        
        Args:
            model_name: Name of the fitted model
            returns: Original return series for comparison
            ci_level: Confidence interval level
            
        Returns:
            None (displays plot)
        """
        if model_name not in self.traces:
            print(f"Model {model_name} not found")
            return
            
        trace = self.traces[model_name]
        
        # Extract volatility samples
        h_samples = trace.posterior['h'].values
        
        # Calculate statistics
        h_mean = h_samples.mean(axis=(0, 1))
        h_lower = np.percentile(h_samples, 100 * (1 - ci_level) / 2, axis=(0, 1))
        h_upper = np.percentile(h_samples, 100 * (1 + ci_level) / 2, axis=(0, 1))
        
        # Convert to volatility
        vol_mean = np.exp(h_mean / 2)
        vol_lower = np.exp(h_lower / 2)
        vol_upper = np.exp(h_upper / 2)
        
        # Create dates
        if returns is not None:
            dates = returns.index
        else:
            dates = pd.date_range(start='2000-01-01', periods=len(vol_mean))
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        plt.plot(dates, vol_mean, label='Estimated Volatility')
        plt.fill_between(dates, vol_lower, vol_upper, alpha=0.2, 
                        label=f'{int(ci_level*100)}% Confidence Interval')
        
        if returns is not None:
            abs_returns = np.abs(returns)
            plt.plot(dates, abs_returns, 'k.', alpha=0.3, label='|Returns|')
        
        plt.title(f'Stochastic Volatility - {model_name}')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def plot_parameter_posterior(self, model_name: str, parameter: str) -> None:
        """
        Plot posterior distribution of a model parameter.
        
        Args:
            model_name: Name of the fitted model
            parameter: Name of the parameter to plot
            
        Returns:
            None (displays plot)
        """
        if model_name not in self.traces:
            print(f"Model {model_name} not found")
            return
            
        trace = self.traces[model_name]
        
        # Plot posterior
        az.plot_posterior(trace, var_names=[parameter], hdi_prob=0.95)
        plt.title(f'Posterior Distribution - {parameter}')
        plt.tight_layout()
        plt.show() 