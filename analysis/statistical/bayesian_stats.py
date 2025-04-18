"""
Bayesian Statistics Module

This module provides functions for performing Bayesian statistical analysis
on cryptocurrency data.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple
import pymc3 as pm
import arviz as az
from scipy import stats

def perform_bayesian_inference(data: Union[np.ndarray, pd.Series],
                             prior_params: Dict[str, float],
                             likelihood: str = 'normal',
                             n_samples: int = 2000,
                             n_tune: int = 1000) -> Dict[str, Union[float, np.ndarray]]:
    """
    Perform Bayesian inference on data.
    
    Args:
        data: Input data
        prior_params: Parameters for prior distribution
        likelihood: Likelihood distribution ('normal', 'student_t', 'laplace')
        n_samples: Number of posterior samples
        n_tune: Number of tuning samples
        
    Returns:
        Dictionary containing inference results
    """
    with pm.Model() as model:
        # Define priors
        if likelihood == 'normal':
            mu = pm.Normal('mu', mu=prior_params.get('mu_mu', 0),
                          sigma=prior_params.get('mu_sigma', 10))
            sigma = pm.HalfNormal('sigma', sigma=prior_params.get('sigma_sigma', 10))
            likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=data)
        elif likelihood == 'student_t':
            mu = pm.Normal('mu', mu=prior_params.get('mu_mu', 0),
                          sigma=prior_params.get('mu_sigma', 10))
            sigma = pm.HalfNormal('sigma', sigma=prior_params.get('sigma_sigma', 10))
            nu = pm.Exponential('nu', lam=prior_params.get('nu_lam', 1))
            likelihood = pm.StudentT('likelihood', mu=mu, sigma=sigma, nu=nu, observed=data)
        
        # Sample from posterior
        trace = pm.sample(n_samples, tune=n_tune, return_inferencedata=True)
    
    return {
        'trace': trace,
        'summary': az.summary(trace),
        'posterior_predictive': pm.sample_posterior_predictive(trace, model=model)
    }

def calculate_posterior_distribution(trace: az.InferenceData,
                                   param: str) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate posterior distribution for a parameter.
    
    Args:
        trace: Inference data from Bayesian model
        param: Parameter name
        
    Returns:
        Dictionary containing posterior distribution information
    """
    posterior = trace.posterior[param].values.flatten()
    
    return {
        'mean': np.mean(posterior),
        'median': np.median(posterior),
        'std': np.std(posterior),
        'hdi_95': az.hdi(posterior, hdi_prob=0.95),
        'hdi_99': az.hdi(posterior, hdi_prob=0.99),
        'samples': posterior
    }

def perform_mcmc_sampling(model: pm.Model,
                        n_samples: int = 2000,
                        n_tune: int = 1000,
                        n_chains: int = 4) -> Dict[str, Union[float, np.ndarray]]:
    """
    Perform MCMC sampling from a PyMC3 model.
    
    Args:
        model: PyMC3 model
        n_samples: Number of samples per chain
        n_tune: Number of tuning samples
        n_chains: Number of chains
        
    Returns:
        Dictionary containing MCMC results
    """
    with model:
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            return_inferencedata=True
        )
    
    return {
        'trace': trace,
        'summary': az.summary(trace),
        'r_hat': az.rhat(trace),
        'effective_sample_size': az.ess(trace)
    }

def calculate_bayesian_credible_intervals(trace: az.InferenceData,
                                        hdi_prob: float = 0.95) -> Dict[str, np.ndarray]:
    """
    Calculate Bayesian credible intervals.
    
    Args:
        trace: Inference data
        hdi_prob: HDI probability level
        
    Returns:
        Dictionary containing credible intervals
    """
    intervals = {}
    for var in trace.posterior.data_vars:
        intervals[var] = az.hdi(trace.posterior[var], hdi_prob=hdi_prob)
    
    return intervals

def perform_bayesian_model_comparison(models: List[pm.Model],
                                    data: Union[np.ndarray, pd.Series],
                                    n_samples: int = 2000) -> Dict[str, Union[float, np.ndarray]]:
    """
    Perform Bayesian model comparison using WAIC and LOO.
    
    Args:
        models: List of PyMC3 models
        data: Observed data
        n_samples: Number of samples
        
    Returns:
        Dictionary containing model comparison results
    """
    traces = []
    for model in models:
        with model:
            trace = pm.sample(n_samples, return_inferencedata=True)
            traces.append(trace)
    
    # Calculate WAIC and LOO
    waic = az.waic(traces)
    loo = az.loo(traces)
    
    return {
        'waic': waic,
        'loo': loo,
        'model_weights': az.compare(traces)
    } 