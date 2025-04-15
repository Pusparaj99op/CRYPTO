"""
Monte Carlo option pricing module.

This module implements Monte Carlo simulation methods for pricing various types of options,
including path-dependent options and options with complex payoffs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
from scipy.stats import norm

class MonteCarloPricing:
    """
    Class for Monte Carlo simulation of option prices.
    """
    
    def __init__(self, risk_free_rate: float = 0.0, volatility: float = 0.0):
        """
        Initialize the MonteCarloPricing class.
        
        Parameters:
        -----------
        risk_free_rate : float
            Risk-free interest rate
        volatility : float
            Volatility of the underlying asset
        """
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
    
    def simulate_paths(self,
                      S0: float,
                      T: float,
                      steps: int,
                      paths: int,
                      r: Optional[float] = None,
                      sigma: Optional[float] = None,
                      seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate geometric Brownian motion paths.
        
        Parameters:
        -----------
        S0 : float
            Initial price of the underlying asset
        T : float
            Time to expiration in years
        steps : int
            Number of time steps
        paths : int
            Number of paths to simulate
        r : float, optional
            Risk-free rate (if None, use instance value)
        sigma : float, optional
            Volatility (if None, use instance value)
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray
            Array of simulated paths (paths x steps)
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
            
        if seed is not None:
            np.random.seed(seed)
            
        dt = T / steps
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        # Generate random numbers
        Z = np.random.normal(0, 1, (paths, steps))
        
        # Calculate returns
        returns = drift + diffusion * Z
        
        # Calculate paths
        paths = np.zeros((paths, steps + 1))
        paths[:, 0] = S0
        
        for t in range(1, steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(returns[:, t-1])
            
        return paths
    
    def european_option_price(self,
                            S0: float,
                            K: float,
                            T: float,
                            option_type: str = 'call',
                            steps: int = 252,
                            paths: int = 10000,
                            r: Optional[float] = None,
                            sigma: Optional[float] = None,
                            seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Price a European option using Monte Carlo simulation.
        
        Parameters:
        -----------
        S0 : float
            Initial price of the underlying asset
        K : float
            Strike price
        T : float
            Time to expiration in years
        option_type : str
            'call' or 'put'
        steps : int
            Number of time steps
        paths : int
            Number of paths to simulate
        r : float, optional
            Risk-free rate (if None, use instance value)
        sigma : float, optional
            Volatility (if None, use instance value)
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        Tuple[float, float]
            Option price and standard error
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
            
        # Simulate paths
        paths = self.simulate_paths(S0, T, steps, paths, r, sigma, seed)
        ST = paths[:, -1]
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - ST, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
            
        # Calculate price and standard error
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(paths)
        
        return price, std_error
    
    def asian_option_price(self,
                         S0: float,
                         K: float,
                         T: float,
                         option_type: str = 'call',
                         average_type: str = 'arithmetic',
                         steps: int = 252,
                         paths: int = 10000,
                         r: Optional[float] = None,
                         sigma: Optional[float] = None,
                         seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Price an Asian option using Monte Carlo simulation.
        
        Parameters:
        -----------
        S0 : float
            Initial price of the underlying asset
        K : float
            Strike price
        T : float
            Time to expiration in years
        option_type : str
            'call' or 'put'
        average_type : str
            'arithmetic' or 'geometric'
        steps : int
            Number of time steps
        paths : int
            Number of paths to simulate
        r : float, optional
            Risk-free rate (if None, use instance value)
        sigma : float, optional
            Volatility (if None, use instance value)
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        Tuple[float, float]
            Option price and standard error
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
            
        # Simulate paths
        paths = self.simulate_paths(S0, T, steps, paths, r, sigma, seed)
        
        # Calculate averages
        if average_type.lower() == 'arithmetic':
            averages = np.mean(paths[:, 1:], axis=1)
        elif average_type.lower() == 'geometric':
            averages = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
        else:
            raise ValueError("average_type must be 'arithmetic' or 'geometric'")
            
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(averages - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - averages, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
            
        # Calculate price and standard error
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(paths)
        
        return price, std_error
    
    def barrier_option_price(self,
                           S0: float,
                           K: float,
                           T: float,
                           barrier: float,
                           option_type: str = 'call',
                           barrier_type: str = 'up-and-out',
                           rebate: float = 0.0,
                           steps: int = 252,
                           paths: int = 10000,
                           r: Optional[float] = None,
                           sigma: Optional[float] = None,
                           seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Price a barrier option using Monte Carlo simulation.
        
        Parameters:
        -----------
        S0 : float
            Initial price of the underlying asset
        K : float
            Strike price
        T : float
            Time to expiration in years
        barrier : float
            Barrier level
        option_type : str
            'call' or 'put'
        barrier_type : str
            'up-and-out', 'up-and-in', 'down-and-out', or 'down-and-in'
        rebate : float
            Rebate paid if barrier is hit
        steps : int
            Number of time steps
        paths : int
            Number of paths to simulate
        r : float, optional
            Risk-free rate (if None, use instance value)
        sigma : float, optional
            Volatility (if None, use instance value)
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        Tuple[float, float]
            Option price and standard error
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
            
        # Simulate paths
        paths = self.simulate_paths(S0, T, steps, paths, r, sigma, seed)
        
        # Check barrier conditions
        if barrier_type == 'up-and-out':
            barrier_hit = np.any(paths >= barrier, axis=1)
            payoffs = np.where(barrier_hit, rebate, np.maximum(paths[:, -1] - K, 0))
        elif barrier_type == 'up-and-in':
            barrier_hit = np.any(paths >= barrier, axis=1)
            payoffs = np.where(barrier_hit, np.maximum(paths[:, -1] - K, 0), rebate)
        elif barrier_type == 'down-and-out':
            barrier_hit = np.any(paths <= barrier, axis=1)
            payoffs = np.where(barrier_hit, rebate, np.maximum(K - paths[:, -1], 0))
        elif barrier_type == 'down-and-in':
            barrier_hit = np.any(paths <= barrier, axis=1)
            payoffs = np.where(barrier_hit, np.maximum(K - paths[:, -1], 0), rebate)
        else:
            raise ValueError("Invalid barrier_type")
            
        # Calculate price and standard error
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(paths)
        
        return price, std_error
    
    def lookback_option_price(self,
                            S0: float,
                            T: float,
                            option_type: str = 'call',
                            floating_strike: bool = True,
                            K: Optional[float] = None,
                            steps: int = 252,
                            paths: int = 10000,
                            r: Optional[float] = None,
                            sigma: Optional[float] = None,
                            seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Price a lookback option using Monte Carlo simulation.
        
        Parameters:
        -----------
        S0 : float
            Initial price of the underlying asset
        T : float
            Time to expiration in years
        option_type : str
            'call' or 'put'
        floating_strike : bool
            Whether the strike is floating
        K : float, optional
            Fixed strike price (if floating_strike=False)
        steps : int
            Number of time steps
        paths : int
            Number of paths to simulate
        r : float, optional
            Risk-free rate (if None, use instance value)
        sigma : float, optional
            Volatility (if None, use instance value)
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        Tuple[float, float]
            Option price and standard error
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
            
        if not floating_strike and K is None:
            raise ValueError("K must be provided when floating_strike=False")
            
        # Simulate paths
        paths = self.simulate_paths(S0, T, steps, paths, r, sigma, seed)
        
        # Calculate maximum and minimum prices
        max_prices = np.max(paths, axis=1)
        min_prices = np.min(paths, axis=1)
        final_prices = paths[:, -1]
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            if floating_strike:
                payoffs = final_prices - min_prices
            else:
                payoffs = np.maximum(max_prices - K, 0)
        elif option_type.lower() == 'put':
            if floating_strike:
                payoffs = max_prices - final_prices
            else:
                payoffs = np.maximum(K - min_prices, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
            
        # Calculate price and standard error
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(paths)
        
        return price, std_error
    
    def plot_simulation_results(self,
                              paths: np.ndarray,
                              title: Optional[str] = None) -> None:
        """
        Plot the simulated paths.
        
        Parameters:
        -----------
        paths : np.ndarray
            Array of simulated paths
        title : str, optional
            Plot title
        """
        plt.figure(figsize=(10, 6))
        
        # Plot a subset of paths for clarity
        num_paths_to_plot = min(100, paths.shape[0])
        for i in range(num_paths_to_plot):
            plt.plot(paths[i], alpha=0.1)
            
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        if title:
            plt.title(title)
        else:
            plt.title('Monte Carlo Simulation Paths')
        plt.grid(True)
        plt.show()
    
    def convergence_analysis(self,
                           S0: float,
                           K: float,
                           T: float,
                           option_type: str = 'call',
                           min_paths: int = 1000,
                           max_paths: int = 100000,
                           step: int = 1000,
                           r: Optional[float] = None,
                           sigma: Optional[float] = None,
                           seed: Optional[int] = None) -> pd.DataFrame:
        """
        Analyze the convergence of Monte Carlo prices with increasing number of paths.
        
        Parameters:
        -----------
        S0 : float
            Initial price of the underlying asset
        K : float
            Strike price
        T : float
            Time to expiration in years
        option_type : str
            'call' or 'put'
        min_paths : int
            Minimum number of paths
        max_paths : int
            Maximum number of paths
        step : int
            Step size for number of paths
        r : float, optional
            Risk-free rate (if None, use instance value)
        sigma : float, optional
            Volatility (if None, use instance value)
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        pd.DataFrame
            Results of convergence analysis
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
            
        results = []
        
        for n_paths in range(min_paths, max_paths + 1, step):
            price, std_error = self.european_option_price(
                S0, K, T, option_type, paths=n_paths, r=r, sigma=sigma, seed=seed
            )
            results.append({
                'n_paths': n_paths,
                'price': price,
                'std_error': std_error,
                'confidence_interval_lower': price - 1.96 * std_error,
                'confidence_interval_upper': price + 1.96 * std_error
            })
            
        return pd.DataFrame(results)
    
    def plot_convergence(self,
                        convergence_data: pd.DataFrame,
                        true_price: Optional[float] = None) -> None:
        """
        Plot the convergence of Monte Carlo prices.
        
        Parameters:
        -----------
        convergence_data : pd.DataFrame
            Results from convergence_analysis
        true_price : float, optional
            True price for comparison
        """
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(convergence_data['n_paths'], convergence_data['price'], 'b-', label='Monte Carlo Price')
        if true_price is not None:
            plt.axhline(y=true_price, color='r', linestyle='--', label='True Price')
        plt.fill_between(convergence_data['n_paths'],
                        convergence_data['confidence_interval_lower'],
                        convergence_data['confidence_interval_upper'],
                        alpha=0.2)
        plt.xlabel('Number of Paths')
        plt.ylabel('Option Price')
        plt.title('Price Convergence')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(convergence_data['n_paths'], convergence_data['std_error'], 'g-')
        plt.xlabel('Number of Paths')
        plt.ylabel('Standard Error')
        plt.title('Standard Error')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show() 