"""
Options trading strategies module.

This module implements various option trading strategies including spreads,
straddles, strangles, butterflies, and other complex option combinations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from .black_scholes import BlackScholes

class OptionsStrategies:
    """
    Class for implementing and analyzing option trading strategies.
    """
    
    def __init__(self, risk_free_rate: float = 0.0, volatility: float = 0.0):
        """
        Initialize the OptionsStrategies class.
        
        Parameters:
        -----------
        risk_free_rate : float
            Risk-free interest rate
        volatility : float
            Volatility of the underlying asset
        """
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.bs = BlackScholes(risk_free_rate, volatility)
    
    def bull_call_spread(self,
                        S: float,
                        K1: float,
                        K2: float,
                        T: float,
                        r: Optional[float] = None,
                        sigma: Optional[float] = None) -> Dict:
        """
        Calculate bull call spread strategy parameters.
        
        Parameters:
        -----------
        S : float
            Current price of the underlying asset
        K1 : float
            Lower strike price (buy call)
        K2 : float
            Higher strike price (sell call)
        T : float
            Time to expiration in years
        r : float, optional
            Risk-free rate (if None, use instance value)
        sigma : float, optional
            Volatility (if None, use instance value)
            
        Returns:
        --------
        Dict
            Strategy parameters including cost, max profit, and break-even
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
            
        # Calculate option prices
        long_call = self.bs.price(S, K1, T, 'call', r, sigma)
        short_call = self.bs.price(S, K2, T, 'call', r, sigma)
        
        # Calculate strategy metrics
        cost = long_call - short_call
        max_profit = K2 - K1 - cost
        break_even = K1 + cost
        
        return {
            'cost': cost,
            'max_profit': max_profit,
            'break_even': break_even,
            'max_loss': cost,
            'long_call_price': long_call,
            'short_call_price': short_call
        }
    
    def bear_put_spread(self,
                       S: float,
                       K1: float,
                       K2: float,
                       T: float,
                       r: Optional[float] = None,
                       sigma: Optional[float] = None) -> Dict:
        """
        Calculate bear put spread strategy parameters.
        
        Parameters:
        -----------
        S : float
            Current price of the underlying asset
        K1 : float
            Higher strike price (buy put)
        K2 : float
            Lower strike price (sell put)
        T : float
            Time to expiration in years
        r : float, optional
            Risk-free rate (if None, use instance value)
        sigma : float, optional
            Volatility (if None, use instance value)
            
        Returns:
        --------
        Dict
            Strategy parameters including cost, max profit, and break-even
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
            
        # Calculate option prices
        long_put = self.bs.price(S, K1, T, 'put', r, sigma)
        short_put = self.bs.price(S, K2, T, 'put', r, sigma)
        
        # Calculate strategy metrics
        cost = long_put - short_put
        max_profit = K1 - K2 - cost
        break_even = K1 - cost
        
        return {
            'cost': cost,
            'max_profit': max_profit,
            'break_even': break_even,
            'max_loss': cost,
            'long_put_price': long_put,
            'short_put_price': short_put
        }
    
    def straddle(self,
                S: float,
                K: float,
                T: float,
                r: Optional[float] = None,
                sigma: Optional[float] = None) -> Dict:
        """
        Calculate long straddle strategy parameters.
        
        Parameters:
        -----------
        S : float
            Current price of the underlying asset
        K : float
            Strike price
        T : float
            Time to expiration in years
        r : float, optional
            Risk-free rate (if None, use instance value)
        sigma : float, optional
            Volatility (if None, use instance value)
            
        Returns:
        --------
        Dict
            Strategy parameters including cost and break-evens
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
            
        # Calculate option prices
        call_price = self.bs.price(S, K, T, 'call', r, sigma)
        put_price = self.bs.price(S, K, T, 'put', r, sigma)
        
        # Calculate strategy metrics
        cost = call_price + put_price
        upper_break_even = K + cost
        lower_break_even = K - cost
        
        return {
            'cost': cost,
            'upper_break_even': upper_break_even,
            'lower_break_even': lower_break_even,
            'max_loss': cost,
            'call_price': call_price,
            'put_price': put_price
        }
    
    def strangle(self,
                S: float,
                K1: float,
                K2: float,
                T: float,
                r: Optional[float] = None,
                sigma: Optional[float] = None) -> Dict:
        """
        Calculate long strangle strategy parameters.
        
        Parameters:
        -----------
        S : float
            Current price of the underlying asset
        K1 : float
            Lower strike price (put)
        K2 : float
            Higher strike price (call)
        T : float
            Time to expiration in years
        r : float, optional
            Risk-free rate (if None, use instance value)
        sigma : float, optional
            Volatility (if None, use instance value)
            
        Returns:
        --------
        Dict
            Strategy parameters including cost and break-evens
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
            
        # Calculate option prices
        call_price = self.bs.price(S, K2, T, 'call', r, sigma)
        put_price = self.bs.price(S, K1, T, 'put', r, sigma)
        
        # Calculate strategy metrics
        cost = call_price + put_price
        upper_break_even = K2 + cost
        lower_break_even = K1 - cost
        
        return {
            'cost': cost,
            'upper_break_even': upper_break_even,
            'lower_break_even': lower_break_even,
            'max_loss': cost,
            'call_price': call_price,
            'put_price': put_price
        }
    
    def butterfly(self,
                 S: float,
                 K1: float,
                 K2: float,
                 K3: float,
                 T: float,
                 r: Optional[float] = None,
                 sigma: Optional[float] = None) -> Dict:
        """
        Calculate butterfly spread strategy parameters.
        
        Parameters:
        -----------
        S : float
            Current price of the underlying asset
        K1 : float
            Lower strike price
        K2 : float
            Middle strike price
        K3 : float
            Higher strike price
        T : float
            Time to expiration in years
        r : float, optional
            Risk-free rate (if None, use instance value)
        sigma : float, optional
            Volatility (if None, use instance value)
            
        Returns:
        --------
        Dict
            Strategy parameters including cost, max profit, and break-evens
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
            
        # Calculate option prices
        call1 = self.bs.price(S, K1, T, 'call', r, sigma)
        call2 = self.bs.price(S, K2, T, 'call', r, sigma)
        call3 = self.bs.price(S, K3, T, 'call', r, sigma)
        
        # Calculate strategy metrics
        cost = call1 - 2 * call2 + call3
        max_profit = K2 - K1 - cost
        upper_break_even = K2 + max_profit
        lower_break_even = K2 - max_profit
        
        return {
            'cost': cost,
            'max_profit': max_profit,
            'upper_break_even': upper_break_even,
            'lower_break_even': lower_break_even,
            'max_loss': cost,
            'call1_price': call1,
            'call2_price': call2,
            'call3_price': call3
        }
    
    def plot_strategy_payoff(self,
                           strategy: str,
                           params: Dict,
                           S_range: Tuple[float, float],
                           num_points: int = 100) -> None:
        """
        Plot the payoff diagram for a strategy.
        
        Parameters:
        -----------
        strategy : str
            Strategy name ('bull_call', 'bear_put', 'straddle', 'strangle', 'butterfly')
        params : Dict
            Strategy parameters
        S_range : Tuple[float, float]
            Range of underlying prices to plot
        num_points : int
            Number of points to calculate
        """
        S_values = np.linspace(S_range[0], S_range[1], num_points)
        payoffs = np.zeros_like(S_values)
        
        if strategy == 'bull_call':
            K1, K2 = params['K1'], params['K2']
            cost = params['cost']
            for i, S in enumerate(S_values):
                if S <= K1:
                    payoffs[i] = -cost
                elif S <= K2:
                    payoffs[i] = S - K1 - cost
                else:
                    payoffs[i] = K2 - K1 - cost
                    
        elif strategy == 'bear_put':
            K1, K2 = params['K1'], params['K2']
            cost = params['cost']
            for i, S in enumerate(S_values):
                if S >= K1:
                    payoffs[i] = -cost
                elif S >= K2:
                    payoffs[i] = K1 - S - cost
                else:
                    payoffs[i] = K1 - K2 - cost
                    
        elif strategy == 'straddle':
            K = params['K']
            cost = params['cost']
            for i, S in enumerate(S_values):
                if S <= K:
                    payoffs[i] = K - S - cost
                else:
                    payoffs[i] = S - K - cost
                    
        elif strategy == 'strangle':
            K1, K2 = params['K1'], params['K2']
            cost = params['cost']
            for i, S in enumerate(S_values):
                if S <= K1:
                    payoffs[i] = K1 - S - cost
                elif S >= K2:
                    payoffs[i] = S - K2 - cost
                else:
                    payoffs[i] = -cost
                    
        elif strategy == 'butterfly':
            K1, K2, K3 = params['K1'], params['K2'], params['K3']
            cost = params['cost']
            for i, S in enumerate(S_values):
                if S <= K1:
                    payoffs[i] = -cost
                elif S <= K2:
                    payoffs[i] = S - K1 - cost
                elif S <= K3:
                    payoffs[i] = K3 - S - cost
                else:
                    payoffs[i] = -cost
                    
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        plt.figure(figsize=(10, 6))
        plt.plot(S_values, payoffs, 'b-', label='Payoff')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Underlying Price')
        plt.ylabel('Payoff')
        plt.title(f'{strategy.replace("_", " ").title()} Strategy Payoff')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def analyze_strategy_risk(self,
                            strategy: str,
                            params: Dict,
                            S: float,
                            T: float,
                            r: Optional[float] = None,
                            sigma: Optional[float] = None) -> Dict:
        """
        Analyze the risk metrics of a strategy.
        
        Parameters:
        -----------
        strategy : str
            Strategy name
        params : Dict
            Strategy parameters
        S : float
            Current price of the underlying asset
        T : float
            Time to expiration in years
        r : float, optional
            Risk-free rate (if None, use instance value)
        sigma : float, optional
            Volatility (if None, use instance value)
            
        Returns:
        --------
        Dict
            Risk metrics including Greeks and other measures
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
            
        # Calculate individual option Greeks
        greeks = {}
        
        if strategy == 'bull_call':
            K1, K2 = params['K1'], params['K2']
            greeks['delta'] = (self.bs.greeks(S, K1, T, 'call', r, sigma)['delta'] -
                             self.bs.greeks(S, K2, T, 'call', r, sigma)['delta'])
            greeks['gamma'] = (self.bs.greeks(S, K1, T, 'call', r, sigma)['gamma'] -
                             self.bs.greeks(S, K2, T, 'call', r, sigma)['gamma'])
            greeks['theta'] = (self.bs.greeks(S, K1, T, 'call', r, sigma)['theta'] -
                             self.bs.greeks(S, K2, T, 'call', r, sigma)['theta'])
            greeks['vega'] = (self.bs.greeks(S, K1, T, 'call', r, sigma)['vega'] -
                            self.bs.greeks(S, K2, T, 'call', r, sigma)['vega'])
            
        elif strategy == 'bear_put':
            K1, K2 = params['K1'], params['K2']
            greeks['delta'] = (self.bs.greeks(S, K1, T, 'put', r, sigma)['delta'] -
                             self.bs.greeks(S, K2, T, 'put', r, sigma)['delta'])
            greeks['gamma'] = (self.bs.greeks(S, K1, T, 'put', r, sigma)['gamma'] -
                             self.bs.greeks(S, K2, T, 'put', r, sigma)['gamma'])
            greeks['theta'] = (self.bs.greeks(S, K1, T, 'put', r, sigma)['theta'] -
                             self.bs.greeks(S, K2, T, 'put', r, sigma)['theta'])
            greeks['vega'] = (self.bs.greeks(S, K1, T, 'put', r, sigma)['vega'] -
                            self.bs.greeks(S, K2, T, 'put', r, sigma)['vega'])
            
        elif strategy == 'straddle':
            K = params['K']
            call_greeks = self.bs.greeks(S, K, T, 'call', r, sigma)
            put_greeks = self.bs.greeks(S, K, T, 'put', r, sigma)
            greeks['delta'] = call_greeks['delta'] + put_greeks['delta']
            greeks['gamma'] = call_greeks['gamma'] + put_greeks['gamma']
            greeks['theta'] = call_greeks['theta'] + put_greeks['theta']
            greeks['vega'] = call_greeks['vega'] + put_greeks['vega']
            
        elif strategy == 'strangle':
            K1, K2 = params['K1'], params['K2']
            call_greeks = self.bs.greeks(S, K2, T, 'call', r, sigma)
            put_greeks = self.bs.greeks(S, K1, T, 'put', r, sigma)
            greeks['delta'] = call_greeks['delta'] + put_greeks['delta']
            greeks['gamma'] = call_greeks['gamma'] + put_greeks['gamma']
            greeks['theta'] = call_greeks['theta'] + put_greeks['theta']
            greeks['vega'] = call_greeks['vega'] + put_greeks['vega']
            
        elif strategy == 'butterfly':
            K1, K2, K3 = params['K1'], params['K2'], params['K3']
            call1_greeks = self.bs.greeks(S, K1, T, 'call', r, sigma)
            call2_greeks = self.bs.greeks(S, K2, T, 'call', r, sigma)
            call3_greeks = self.bs.greeks(S, K3, T, 'call', r, sigma)
            greeks['delta'] = call1_greeks['delta'] - 2 * call2_greeks['delta'] + call3_greeks['delta']
            greeks['gamma'] = call1_greeks['gamma'] - 2 * call2_greeks['gamma'] + call3_greeks['gamma']
            greeks['theta'] = call1_greeks['theta'] - 2 * call2_greeks['theta'] + call3_greeks['theta']
            greeks['vega'] = call1_greeks['vega'] - 2 * call2_greeks['vega'] + call3_greeks['vega']
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        return greeks 