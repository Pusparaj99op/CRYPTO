"""
Copula Models for Dependency Modeling

This module implements various copula models for modeling dependencies between
financial assets, including Gaussian, Student-t, and Archimedean copulas.
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from typing import Union, Tuple, List, Optional
from scipy.stats import norm, t

class GaussianCopula:
    """Gaussian Copula implementation."""
    
    def __init__(self, correlation_matrix: np.ndarray):
        """
        Initialize Gaussian Copula.
        
        Args:
            correlation_matrix: Correlation matrix between assets
        """
        self.correlation_matrix = correlation_matrix
        self.dim = correlation_matrix.shape[0]
        
    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> None:
        """
        Fit Gaussian Copula to data.
        
        Args:
            data: Array or DataFrame of asset returns
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Calculate empirical correlation matrix
        self.correlation_matrix = np.corrcoef(data.T)
        
    def cdf(self, u: np.ndarray) -> float:
        """
        Calculate copula CDF.
        
        Args:
            u: Array of uniform marginals
            
        Returns:
            float: Copula CDF value
        """
        # Transform to normal space
        z = norm.ppf(u)
        
        # Calculate multivariate normal CDF
        return stats.multivariate_normal.cdf(z, cov=self.correlation_matrix)
    
    def pdf(self, u: np.ndarray) -> float:
        """
        Calculate copula PDF.
        
        Args:
            u: Array of uniform marginals
            
        Returns:
            float: Copula PDF value
        """
        # Transform to normal space
        z = norm.ppf(u)
        
        # Calculate multivariate normal PDF
        return stats.multivariate_normal.pdf(z, cov=self.correlation_matrix)
    
    def simulate(self, n_samples: int) -> np.ndarray:
        """
        Simulate from Gaussian Copula.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            np.ndarray: Simulated uniform marginals
        """
        # Generate correlated normal variables
        z = np.random.multivariate_normal(
            np.zeros(self.dim),
            self.correlation_matrix,
            n_samples
        )
        
        # Transform to uniform marginals
        u = norm.cdf(z)
        
        return u

class StudentTCopula:
    """Student-t Copula implementation."""
    
    def __init__(self, correlation_matrix: np.ndarray, df: float):
        """
        Initialize Student-t Copula.
        
        Args:
            correlation_matrix: Correlation matrix between assets
            df: Degrees of freedom
        """
        self.correlation_matrix = correlation_matrix
        self.df = df
        self.dim = correlation_matrix.shape[0]
        
    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> None:
        """
        Fit Student-t Copula to data.
        
        Args:
            data: Array or DataFrame of asset returns
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Calculate empirical correlation matrix
        self.correlation_matrix = np.corrcoef(data.T)
        
        # Estimate degrees of freedom using MLE
        def neg_log_likelihood(df):
            return -np.sum(t.logpdf(data, df=df))
        
        result = optimize.minimize(neg_log_likelihood, 10.0)
        self.df = result.x[0]
        
    def cdf(self, u: np.ndarray) -> float:
        """
        Calculate copula CDF.
        
        Args:
            u: Array of uniform marginals
            
        Returns:
            float: Copula CDF value
        """
        # Transform to t space
        t_values = t.ppf(u, df=self.df)
        
        # Calculate multivariate t CDF
        return stats.multivariate_t.cdf(t_values, df=self.df, shape=self.correlation_matrix)
    
    def pdf(self, u: np.ndarray) -> float:
        """
        Calculate copula PDF.
        
        Args:
            u: Array of uniform marginals
            
        Returns:
            float: Copula PDF value
        """
        # Transform to t space
        t_values = t.ppf(u, df=self.df)
        
        # Calculate multivariate t PDF
        return stats.multivariate_t.pdf(t_values, df=self.df, shape=self.correlation_matrix)
    
    def simulate(self, n_samples: int) -> np.ndarray:
        """
        Simulate from Student-t Copula.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            np.ndarray: Simulated uniform marginals
        """
        # Generate correlated t variables
        t_values = stats.multivariate_t.rvs(
            df=self.df,
            shape=self.correlation_matrix,
            size=n_samples
        )
        
        # Transform to uniform marginals
        u = t.cdf(t_values, df=self.df)
        
        return u

def fit_copula(data: Union[np.ndarray, pd.DataFrame],
              copula_type: str = 'gaussian') -> Union[GaussianCopula, StudentTCopula]:
    """
    Fit copula to data.
    
    Args:
        data: Array or DataFrame of asset returns
        copula_type: Type of copula ('gaussian' or 'student-t')
        
    Returns:
        Union[GaussianCopula, StudentTCopula]: Fitted copula object
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # Calculate correlation matrix
    correlation_matrix = np.corrcoef(data.T)
    
    if copula_type.lower() == 'gaussian':
        copula = GaussianCopula(correlation_matrix)
    elif copula_type.lower() == 'student-t':
        copula = StudentTCopula(correlation_matrix, df=10.0)
    else:
        raise ValueError(f"Unsupported copula type: {copula_type}")
    
    copula.fit(data)
    return copula

def simulate_copula(copula: Union[GaussianCopula, StudentTCopula],
                   n_samples: int) -> np.ndarray:
    """
    Simulate from fitted copula.
    
    Args:
        copula: Fitted copula object
        n_samples: Number of samples to generate
        
    Returns:
        np.ndarray: Simulated uniform marginals
    """
    return copula.simulate(n_samples) 