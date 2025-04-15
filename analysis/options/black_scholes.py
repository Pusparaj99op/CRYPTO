import numpy as np
from scipy.stats import norm
from typing import Dict, Union, Tuple, Optional
import matplotlib.pyplot as plt

class BlackScholes:
    """
    Black-Scholes-Merton option pricing model implementation.
    
    This class implements the Black-Scholes model for European option pricing,
    with methods for calculating option prices, Greeks, and visualizing
    pricing relationships.
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.0, 
                 volatility: float = 0.0,
                 dividend_yield: float = 0.0):
        """
        Initialize the Black-Scholes model.
        
        Parameters:
        -----------
        risk_free_rate : float
            Annual risk-free interest rate (decimal)
        volatility : float
            Annual volatility (decimal)
        dividend_yield : float
            Annual dividend yield (decimal)
        """
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.dividend_yield = dividend_yield
    
    def price(self, 
              S: float, 
              K: float, 
              T: float, 
              option_type: str = 'call', 
              r: Optional[float] = None, 
              sigma: Optional[float] = None, 
              q: Optional[float] = None) -> float:
        """
        Calculate the Black-Scholes option price.
        
        Parameters:
        -----------
        S : float
            Current price of the underlying asset
        K : float
            Strike price of the option
        T : float
            Time to expiration in years
        option_type : str
            'call' for Call option, 'put' for Put option
        r : float, optional
            Risk-free interest rate (if None, use the instance value)
        sigma : float, optional
            Volatility (if None, use the instance value)
        q : float, optional
            Dividend yield (if None, use the instance value)
            
        Returns:
        --------
        float
            Black-Scholes option price
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
        if q is None:
            q = self.dividend_yield
        
        # Validate input parameters
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            raise ValueError("S, K, T, and sigma must be positive")
        
        # Handle very small time to expiration
        if T < 1e-10:
            if option_type.lower() == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate option price
        if option_type.lower() == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type.lower() == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        return price
    
    def _calculate_d1_d2(self, 
                         S: float, 
                         K: float, 
                         T: float, 
                         r: float, 
                         sigma: float, 
                         q: float) -> Tuple[float, float]:
        """
        Calculate d1 and d2 parameters for the Black-Scholes model.
        
        Parameters:
        -----------
        S : float
            Current price of the underlying asset
        K : float
            Strike price of the option
        T : float
            Time to expiration in years
        r : float
            Risk-free interest rate
        sigma : float
            Volatility
        q : float
            Dividend yield
            
        Returns:
        --------
        Tuple[float, float]
            d1 and d2 values
        """
        if T < 1e-10:
            # For very small T, use approximation
            d1 = np.inf if S > K else -np.inf
            d2 = np.inf if S > K else -np.inf
            return d1, d2
            
        d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    def greeks(self, 
               S: float, 
               K: float, 
               T: float, 
               option_type: str = 'call', 
               r: Optional[float] = None, 
               sigma: Optional[float] = None, 
               q: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate option Greeks.
        
        Parameters:
        -----------
        Same as price() method
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing Delta, Gamma, Theta, Vega, and Rho
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
        if q is None:
            q = self.dividend_yield
        
        # Validate input parameters
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            raise ValueError("S, K, T, and sigma must be positive")
        
        # Handle very small time to expiration
        if T < 1e-10:
            result = {
                'delta': 1.0 if S > K and option_type.lower() == 'call' else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
            return result
        
        # Calculate d1 and d2
        d1, d2 = self._calculate_d1_d2(S, K, T, r, sigma, q)
        
        # Common terms
        sqrt_t = np.sqrt(T)
        nd1 = norm.pdf(d1)
        exp_qt = np.exp(-q * T)
        exp_rt = np.exp(-r * T)
        
        # Calculate Greeks based on option type
        if option_type.lower() == 'call':
            delta = exp_qt * norm.cdf(d1)
            theta = -(S * sigma * exp_qt * nd1) / (2 * sqrt_t) - r * K * exp_rt * norm.cdf(d2) + q * S * exp_qt * norm.cdf(d1)
            rho = K * T * exp_rt * norm.cdf(d2)
        elif option_type.lower() == 'put':
            delta = exp_qt * (norm.cdf(d1) - 1)
            theta = -(S * sigma * exp_qt * nd1) / (2 * sqrt_t) + r * K * exp_rt * norm.cdf(-d2) - q * S * exp_qt * norm.cdf(-d1)
            rho = -K * T * exp_rt * norm.cdf(-d2)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Greeks that are the same for both calls and puts
        gamma = (exp_qt * nd1) / (S * sigma * sqrt_t)
        vega = S * exp_qt * nd1 * sqrt_t
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365.0,  # Convert to daily theta
            'vega': vega / 100.0,    # Convert to 1% volatility change
            'rho': rho / 100.0       # Convert to 1% rate change
        }
    
    def implied_volatility(self, 
                          price: float, 
                          S: float, 
                          K: float, 
                          T: float, 
                          option_type: str = 'call', 
                          r: Optional[float] = None, 
                          q: Optional[float] = None,
                          max_iterations: int = 100, 
                          precision: float = 1e-8) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Parameters:
        -----------
        price : float
            Market price of the option
        S, K, T, option_type, r, q : same as price() method
        max_iterations : int
            Maximum number of iterations for solver
        precision : float
            Desired precision for implied volatility
            
        Returns:
        --------
        float
            Implied volatility
        """
        if r is None:
            r = self.risk_free_rate
        if q is None:
            q = self.dividend_yield
        
        # Validate inputs
        if price <= 0 or S <= 0 or K <= 0 or T <= 0:
            raise ValueError("price, S, K, and T must be positive")
        
        # Initial guess for implied volatility
        # Start with 30% as a typical starting point
        sigma = 0.3
        
        for i in range(max_iterations):
            # Calculate option price and vega with current sigma
            option_price = self.price(S, K, T, option_type, r, sigma, q)
            option_vega = self.greeks(S, K, T, option_type, r, sigma, q)['vega'] * 100  # Adjust back from 1% to full vega
            
            # Calculate price difference
            price_diff = option_price - price
            
            # Check if precision is met
            if abs(price_diff) < precision:
                return sigma
            
            # Prevent division by zero or very small vega
            if abs(option_vega) < 1e-10:
                sigma = sigma * 1.5  # Adjust guess and try again
                continue
            
            # Update volatility estimate using Newton-Raphson step
            sigma = sigma - price_diff / option_vega
            
            # Enforce bounds to keep solver stable
            sigma = max(0.001, min(sigma, 5.0))
        
        # If we reach here, we didn't converge
        raise RuntimeError(f"Implied volatility calculation did not converge after {max_iterations} iterations")
    
    def plot_price_curve(self,
                        S_range: Optional[Tuple[float, float]] = None,
                        K: float = 100.0,
                        T: float = 1.0,
                        r: Optional[float] = None,
                        sigma: Optional[float] = None,
                        q: Optional[float] = None,
                        num_points: int = 100) -> None:
        """
        Plot option price against underlying price.
        
        Parameters:
        -----------
        S_range : Tuple[float, float], optional
            Range of underlying prices to plot (min, max)
        K, T, r, sigma, q : Same as price() method
        num_points : int
            Number of points to calculate for the curve
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
        if q is None:
            q = self.dividend_yield
        
        # Default range if not provided
        if S_range is None:
            S_range = (K * 0.5, K * 1.5)
        
        # Generate price points
        S_values = np.linspace(S_range[0], S_range[1], num_points)
        call_prices = [self.price(S, K, T, 'call', r, sigma, q) for S in S_values]
        put_prices = [self.price(S, K, T, 'put', r, sigma, q) for S in S_values]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(S_values, call_prices, 'b-', label='Call Option')
        plt.plot(S_values, put_prices, 'r-', label='Put Option')
        
        # Add reference lines
        plt.axvline(x=K, color='gray', linestyle='--', alpha=0.5, label='Strike Price')
        
        # Format plot
        plt.xlabel('Underlying Price')
        plt.ylabel('Option Price')
        title = f'Black-Scholes Option Prices (K={K}, T={T:.2f}, σ={sigma:.2f}, r={r:.2f})'
        if q > 0:
            title += f', q={q:.2f}'
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_volatility_impact(self,
                              S: float = 100.0,
                              K: float = 100.0,
                              T: float = 1.0,
                              r: Optional[float] = None,
                              q: Optional[float] = None,
                              sigma_range: Tuple[float, float] = (0.01, 1.0),
                              num_points: int = 100) -> None:
        """
        Plot the impact of volatility on option prices.
        
        Parameters:
        -----------
        S, K, T, r, q : Same as price() method
        sigma_range : Tuple[float, float]
            Range of volatility values to plot (min, max)
        num_points : int
            Number of points to calculate for the curve
        """
        if r is None:
            r = self.risk_free_rate
        if q is None:
            q = self.dividend_yield
        
        # Generate volatility points
        sigma_values = np.linspace(sigma_range[0], sigma_range[1], num_points)
        call_prices = [self.price(S, K, T, 'call', r, sigma, q) for sigma in sigma_values]
        put_prices = [self.price(S, K, T, 'put', r, sigma, q) for sigma in sigma_values]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(sigma_values, call_prices, 'b-', label='Call Option')
        plt.plot(sigma_values, put_prices, 'r-', label='Put Option')
        
        # Format plot
        plt.xlabel('Volatility (σ)')
        plt.ylabel('Option Price')
        plt.title(f'Impact of Volatility on Option Prices (S={S}, K={K}, T={T:.2f}, r={r:.2f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def put_call_parity(self,
                       S: float,
                       K: float,
                       T: float,
                       r: Optional[float] = None,
                       q: Optional[float] = None) -> Dict[str, float]:
        """
        Check put-call parity relationship.
        
        Parameters:
        -----------
        S, K, T, r, q : Same as price() method
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with call price, put price, and parity values
        """
        if r is None:
            r = self.risk_free_rate
        if q is None:
            q = self.dividend_yield
        
        # Calculate call and put prices
        call_price = self.price(S, K, T, 'call', r, self.volatility, q)
        put_price = self.price(S, K, T, 'put', r, self.volatility, q)
        
        # Calculate parity components
        lhs = call_price + K * np.exp(-r * T)  # Left-hand side
        rhs = put_price + S * np.exp(-q * T)   # Right-hand side
        
        return {
            'call_price': call_price,
            'put_price': put_price,
            'lhs': lhs,
            'rhs': rhs,
            'difference': lhs - rhs
        }

# Example usage
if __name__ == "__main__":
    # Create a Black-Scholes model with market parameters
    bs = BlackScholes(risk_free_rate=0.05, volatility=0.2, dividend_yield=0.01)
    
    # Price a call option
    S = 100       # Underlying price
    K = 100       # Strike price
    T = 1.0       # Time to expiration in years
    
    call_price = bs.price(S, K, T, 'call')
    put_price = bs.price(S, K, T, 'put')
    
    print(f"Call option price: ${call_price:.2f}")
    print(f"Put option price: ${put_price:.2f}")
    
    # Calculate option Greeks
    call_greeks = bs.greeks(S, K, T, 'call')
    print("\nCall Option Greeks:")
    for greek, value in call_greeks.items():
        print(f"{greek.capitalize()}: {value:.6f}")
    
    # Check put-call parity
    parity = bs.put_call_parity(S, K, T)
    print("\nPut-Call Parity Check:")
    print(f"LHS (C + Ke^(-rT)): {parity['lhs']:.6f}")
    print(f"RHS (P + Se^(-qT)): {parity['rhs']:.6f}")
    print(f"Difference: {parity['difference']:.10f}")
    
    # Demonstrate implied volatility calculation
    market_price = 10.5
    implied_vol = bs.implied_volatility(market_price, S, K, T, 'call')
    print(f"\nImplied Volatility for market price ${market_price}: {implied_vol:.2%}")
    
    # Visual demonstrations can be uncommented if plotting is desired
    # bs.plot_price_curve(K=K, T=T)
    # bs.plot_volatility_impact(S=S, K=K, T=T) 