"""
Volatility smile and skew modeling module.

This module provides tools for analyzing and modeling the volatility smile and skew
in options markets, which are important phenomena in option pricing.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt

class VolatilitySmile:
    """
    Class for modeling and analyzing the volatility smile and skew in options markets.
    """
    
    def __init__(self):
        """Initialize the VolatilitySmile class."""
        pass
    
    def fit_smile(self,
                 strikes: np.ndarray,
                 implied_vols: np.ndarray,
                 model: str = 'quadratic',
                 moneyness: bool = True,
                 S: Optional[float] = None) -> Dict:
        """
        Fit a volatility smile model to the data.
        
        Parameters:
        -----------
        strikes : np.ndarray
            Array of strike prices
        implied_vols : np.ndarray
            Array of implied volatilities
        model : str
            Model type ('quadratic', 'cubic', 'svi')
        moneyness : bool
            Whether to use moneyness (K/S) instead of strikes
        S : float, optional
            Current spot price (required if moneyness=True)
            
        Returns:
        --------
        Dict
            Dictionary containing model parameters and fit statistics
        """
        if moneyness and S is None:
            raise ValueError("Spot price S must be provided when using moneyness")
            
        if moneyness:
            x = strikes / S
        else:
            x = strikes
            
        if model == 'quadratic':
            def quadratic(x, a, b, c):
                return a * x**2 + b * x + c
            popt, pcov = curve_fit(quadratic, x, implied_vols)
            params = {'a': popt[0], 'b': popt[1], 'c': popt[2]}
            
        elif model == 'cubic':
            def cubic(x, a, b, c, d):
                return a * x**3 + b * x**2 + c * x + d
            popt, pcov = curve_fit(cubic, x, implied_vols)
            params = {'a': popt[0], 'b': popt[1], 'c': popt[2], 'd': popt[3]}
            
        elif model == 'svi':
            def svi(x, a, b, rho, m, sigma):
                return a + b * (rho * (x - m) + np.sqrt((x - m)**2 + sigma**2))
            popt, pcov = curve_fit(svi, x, implied_vols)
            params = {'a': popt[0], 'b': popt[1], 'rho': popt[2], 'm': popt[3], 'sigma': popt[4]}
            
        else:
            raise ValueError(f"Unknown model type: {model}")
            
        # Calculate fit statistics
        predicted = self.predict_smile(x, params, model, moneyness)
        residuals = implied_vols - predicted
        mse = np.mean(residuals**2)
        r2 = 1 - np.sum(residuals**2) / np.sum((implied_vols - np.mean(implied_vols))**2)
        
        return {
            'params': params,
            'model': model,
            'mse': mse,
            'r2': r2,
            'moneyness': moneyness
        }
    
    def predict_smile(self,
                     x: Union[float, np.ndarray],
                     params: Dict,
                     model: str,
                     moneyness: bool = True) -> Union[float, np.ndarray]:
        """
        Predict implied volatility using the fitted model.
        
        Parameters:
        -----------
        x : float or np.ndarray
            Strike prices or moneyness values
        params : Dict
            Model parameters
        model : str
            Model type
        moneyness : bool
            Whether x represents moneyness
            
        Returns:
        --------
        float or np.ndarray
            Predicted implied volatilities
        """
        if model == 'quadratic':
            return params['a'] * x**2 + params['b'] * x + params['c']
        elif model == 'cubic':
            return params['a'] * x**3 + params['b'] * x**2 + params['c'] * x + params['d']
        elif model == 'svi':
            return params['a'] + params['b'] * (params['rho'] * (x - params['m']) + 
                                              np.sqrt((x - params['m'])**2 + params['sigma']**2))
        else:
            raise ValueError(f"Unknown model type: {model}")
    
    def plot_smile(self,
                  strikes: np.ndarray,
                  implied_vols: np.ndarray,
                  fit_params: Optional[Dict] = None,
                  S: Optional[float] = None,
                  title: Optional[str] = None) -> None:
        """
        Plot the volatility smile with optional fitted curve.
        
        Parameters:
        -----------
        strikes : np.ndarray
            Array of strike prices
        implied_vols : np.ndarray
            Array of implied volatilities
        fit_params : Dict, optional
            Parameters from fitted model
        S : float, optional
            Current spot price
        title : str, optional
            Plot title
        """
        plt.figure(figsize=(10, 6))
        
        if S is not None:
            moneyness = strikes / S
            plt.plot(moneyness, implied_vols, 'bo', label='Market Data')
            plt.xlabel('Moneyness (K/S)')
        else:
            plt.plot(strikes, implied_vols, 'bo', label='Market Data')
            plt.xlabel('Strike Price')
            
        if fit_params is not None:
            model = fit_params['model']
            moneyness = fit_params['moneyness']
            
            if moneyness and S is not None:
                x = np.linspace(min(strikes/S), max(strikes/S), 100)
            else:
                x = np.linspace(min(strikes), max(strikes), 100)
                
            y = self.predict_smile(x, fit_params['params'], model, moneyness)
            plt.plot(x, y, 'r-', label=f'Fitted {model} curve')
            
        plt.ylabel('Implied Volatility')
        if title:
            plt.title(title)
        else:
            plt.title('Volatility Smile')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def calculate_skew(self,
                      strikes: np.ndarray,
                      implied_vols: np.ndarray,
                      S: float) -> float:
        """
        Calculate the volatility skew.
        
        Parameters:
        -----------
        strikes : np.ndarray
            Array of strike prices
        implied_vols : np.ndarray
            Array of implied volatilities
        S : float
            Current spot price
            
        Returns:
        --------
        float
            Skew measure (typically difference between OTM put and ATM implied vol)
        """
        # Find ATM strike
        atm_idx = np.argmin(np.abs(strikes - S))
        atm_vol = implied_vols[atm_idx]
        
        # Find OTM put strike (closest to 0.9 * S)
        otm_put_idx = np.argmin(np.abs(strikes - 0.9 * S))
        otm_put_vol = implied_vols[otm_put_idx]
        
        return otm_put_vol - atm_vol
    
    def analyze_smile_dynamics(self,
                             surface_data: pd.DataFrame,
                             S: float) -> pd.DataFrame:
        """
        Analyze the dynamics of the volatility smile over time.
        
        Parameters:
        -----------
        surface_data : pd.DataFrame
            DataFrame containing strike prices, maturities, and implied vols
        S : float
            Current spot price
            
        Returns:
        --------
        pd.DataFrame
            Analysis results including skew and smile curvature
        """
        results = []
        
        for maturity in surface_data['maturity'].unique():
            mask = surface_data['maturity'] == maturity
            strikes = surface_data.loc[mask, 'strike'].values
            vols = surface_data.loc[mask, 'implied_vol'].values
            
            # Calculate skew
            skew = self.calculate_skew(strikes, vols, S)
            
            # Fit smile
            fit = self.fit_smile(strikes, vols, model='quadratic', moneyness=True, S=S)
            
            results.append({
                'maturity': maturity,
                'skew': skew,
                'curvature': fit['params']['a'],
                'r2': fit['r2']
            })
            
        return pd.DataFrame(results) 