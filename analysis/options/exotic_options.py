import numpy as np
from scipy.stats import norm
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt

class ExoticOptions:
    """
    A class for pricing various types of exotic options.
    
    This class implements pricing methods for exotic options using both
    analytical formulas (where available) and Monte Carlo simulation.
    
    Supported option types include:
    - Barrier options (up-and-out, down-and-out, up-and-in, down-and-in)
    - Asian options (arithmetic and geometric average)
    - Binary (digital) options
    - Lookback options (fixed and floating strike)
    - Rainbow options (multi-asset options)
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.0, 
                 volatility: float = 0.0,
                 dividend_yield: float = 0.0):
        """
        Initialize the ExoticOptions pricer.
        
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
    
    def barrier_option_price(self, 
                           S: float, 
                           K: float, 
                           T: float, 
                           barrier: float, 
                           option_type: str = 'call', 
                           barrier_type: str = 'up-and-out', 
                           rebate: float = 0.0, 
                           r: Optional[float] = None, 
                           sigma: Optional[float] = None, 
                           q: Optional[float] = None) -> float:
        """
        Calculate the price of a barrier option using analytical formulas.
        
        Parameters:
        -----------
        S : float
            Current price of the underlying asset
        K : float
            Strike price of the option
        T : float
            Time to expiration in years
        barrier : float
            Barrier level
        option_type : str
            'call' for Call option, 'put' for Put option
        barrier_type : str
            'up-and-out', 'down-and-out', 'up-and-in', or 'down-and-in'
        rebate : float
            Rebate paid if the option expires worthless due to barrier
        r : float, optional
            Risk-free interest rate (if None, use the instance value)
        sigma : float, optional
            Volatility (if None, use the instance value)
        q : float, optional
            Dividend yield (if None, use the instance value)
            
        Returns:
        --------
        float
            Barrier option price
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
        if q is None:
            q = self.dividend_yield
        
        # Verify barrier type
        valid_barrier_types = ['up-and-out', 'down-and-out', 'up-and-in', 'down-and-in']
        if barrier_type not in valid_barrier_types:
            raise ValueError(f"Invalid barrier type. Must be one of: {valid_barrier_types}")
        
        # Verify option type
        option_type = option_type.lower()
        if option_type not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")
        
        # Define variables for the formulas
        is_call = option_type == 'call'
        
        # Check if barrier has already been hit
        if (barrier_type == 'up-and-out' and S >= barrier) or \
           (barrier_type == 'down-and-out' and S <= barrier):
            return rebate * np.exp(-r * T)
        
        if (barrier_type == 'up-and-in' and S >= barrier) or \
           (barrier_type == 'down-and-in' and S <= barrier):
            # In-barrier already triggered, price as vanilla option
            return self._vanilla_price(S, K, T, option_type, r, sigma, q)
        
        # Calculate parameters
        mu = r - q - 0.5 * sigma**2
        lambda_param = np.sqrt(mu**2 + 2 * r * sigma**2) / sigma**2
        x1 = np.log(S / K) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)
        x2 = np.log(S / barrier) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)
        y1 = np.log(barrier**2 / (S * K)) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)
        y2 = np.log(barrier / S) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)
        
        # Adjust signs for put options
        if not is_call:
            x1 = -x1
            x2 = -x2
            y1 = -y1
            y2 = -y2
        
        # Define barrier option price based on type
        if barrier_type == 'down-and-out':
            if is_call:
                if barrier <= K:
                    # Down-and-out call with barrier below strike
                    term1 = S * np.exp(-q * T) * norm.cdf(x1)
                    term2 = K * np.exp(-r * T) * norm.cdf(x1 - sigma * np.sqrt(T))
                    term3 = S * np.exp(-q * T) * (barrier / S)**(2 * lambda_param) * norm.cdf(y1)
                    term4 = K * np.exp(-r * T) * (barrier / S)**(2 * lambda_param - 2) * norm.cdf(y1 - sigma * np.sqrt(T))
                    price = term1 - term2 - term3 + term4
                else:
                    # Down-and-out call with barrier above strike
                    h1 = S * np.exp(-q * T) * norm.cdf(x2)
                    h2 = K * np.exp(-r * T) * norm.cdf(x2 - sigma * np.sqrt(T))
                    h3 = S * np.exp(-q * T) * (barrier / S)**(2 * lambda_param) * norm.cdf(y2)
                    h4 = K * np.exp(-r * T) * (barrier / S)**(2 * lambda_param - 2) * norm.cdf(y2 - sigma * np.sqrt(T))
                    price = h1 - h2 - h3 + h4
            else:
                # Down-and-out put
                h1 = K * np.exp(-r * T) * norm.cdf(-x1 + sigma * np.sqrt(T))
                h2 = S * np.exp(-q * T) * norm.cdf(-x1)
                h3 = K * np.exp(-r * T) * (barrier / S)**(2 * lambda_param - 2) * norm.cdf(-y1 + sigma * np.sqrt(T))
                h4 = S * np.exp(-q * T) * (barrier / S)**(2 * lambda_param) * norm.cdf(-y1)
                price = h1 - h2 - h3 + h4
                
        elif barrier_type == 'up-and-out':
            if is_call:
                # Up-and-out call
                h1 = S * np.exp(-q * T) * norm.cdf(x1)
                h2 = K * np.exp(-r * T) * norm.cdf(x1 - sigma * np.sqrt(T))
                h3 = S * np.exp(-q * T) * (barrier / S)**(2 * lambda_param) * norm.cdf(y1)
                h4 = K * np.exp(-r * T) * (barrier / S)**(2 * lambda_param - 2) * norm.cdf(y1 - sigma * np.sqrt(T))
                price = h1 - h2 - h3 + h4
            else:
                # Up-and-out put
                if barrier >= K:
                    h1 = K * np.exp(-r * T) * norm.cdf(-x1 + sigma * np.sqrt(T))
                    h2 = S * np.exp(-q * T) * norm.cdf(-x1)
                    h3 = K * np.exp(-r * T) * (barrier / S)**(2 * lambda_param - 2) * norm.cdf(-y1 + sigma * np.sqrt(T))
                    h4 = S * np.exp(-q * T) * (barrier / S)**(2 * lambda_param) * norm.cdf(-y1)
                    price = h1 - h2 - h3 + h4
                else:
                    price = 0  # Simplified for this implementation
        
        elif barrier_type == 'down-and-in':
            # Use the in-out parity: price(in) = price(vanilla) - price(out)
            vanilla_price = self._vanilla_price(S, K, T, option_type, r, sigma, q)
            out_price = self.barrier_option_price(S, K, T, barrier, option_type, 'down-and-out', 0, r, sigma, q)
            price = vanilla_price - out_price
            
        elif barrier_type == 'up-and-in':
            # Use the in-out parity: price(in) = price(vanilla) - price(out)
            vanilla_price = self._vanilla_price(S, K, T, option_type, r, sigma, q)
            out_price = self.barrier_option_price(S, K, T, barrier, option_type, 'up-and-out', 0, r, sigma, q)
            price = vanilla_price - out_price
        
        # Add rebate
        if rebate > 0:
            if barrier_type == 'up-and-out':
                rebate_price = rebate * np.exp(-r * T) * (S / barrier)**(2 * lambda_param) * norm.cdf(y2)
            elif barrier_type == 'down-and-out':
                rebate_price = rebate * np.exp(-r * T) * (S / barrier)**(2 * lambda_param) * norm.cdf(-y2)
            else:
                rebate_price = 0  # Simplified for in-barriers
            
            price += rebate_price
        
        return max(0, price)
    
    def _vanilla_price(self, 
                     S: float, 
                     K: float, 
                     T: float, 
                     option_type: str, 
                     r: float, 
                     sigma: float, 
                     q: float) -> float:
        """
        Calculate vanilla option price using Black-Scholes.
        
        Parameters as in barrier_option_price.
        """
        if T <= 0:
            # At expiration
            if option_type == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        return price
    
    def binary_option_price(self, 
                          S: float, 
                          K: float, 
                          T: float, 
                          option_type: str = 'call', 
                          payoff: float = 1.0, 
                          r: Optional[float] = None, 
                          sigma: Optional[float] = None, 
                          q: Optional[float] = None) -> float:
        """
        Calculate the price of a binary (digital) option.
        
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
        payoff : float
            Fixed payoff amount if option expires in-the-money
        r, sigma, q : Same as other methods
            
        Returns:
        --------
        float
            Binary option price
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
        if q is None:
            q = self.dividend_yield
        
        option_type = option_type.lower()
        if option_type not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")
        
        if T <= 0:
            # At expiration
            if option_type == 'call':
                return payoff if S > K else 0
            else:
                return payoff if S < K else 0
        
        d2 = (np.log(S / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        if option_type == 'call':
            price = payoff * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = payoff * np.exp(-r * T) * norm.cdf(-d2)
        
        return price
    
    def asian_option_price(self, 
                         S: float, 
                         K: float, 
                         T: float, 
                         steps: int, 
                         paths: int = 10000, 
                         option_type: str = 'call', 
                         average_type: str = 'arithmetic', 
                         r: Optional[float] = None, 
                         sigma: Optional[float] = None, 
                         q: Optional[float] = None, 
                         seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Calculate the price of an Asian option using Monte Carlo simulation.
        
        Parameters:
        -----------
        S : float
            Current price of the underlying asset
        K : float
            Strike price of the option
        T : float
            Time to expiration in years
        steps : int
            Number of time steps in the simulation
        paths : int
            Number of simulation paths
        option_type : str
            'call' for Call option, 'put' for Put option
        average_type : str
            'arithmetic' or 'geometric' average
        r, sigma, q : Same as other methods
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        Tuple[float, float]
            (Option price, Standard error)
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
        if q is None:
            q = self.dividend_yield
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        option_type = option_type.lower()
        if option_type not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")
        
        average_type = average_type.lower()
        if average_type not in ['arithmetic', 'geometric']:
            raise ValueError("Average type must be 'arithmetic' or 'geometric'")
        
        # Special case for geometric Asian options (analytical solution)
        if average_type == 'geometric' and False:  # Disabled for simplicity
            # TODO: Implement analytical solution for geometric Asian options
            pass
        
        # Monte Carlo simulation
        dt = T / steps
        drift = (r - q - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)
        
        # Initialize arrays for path simulation
        prices = np.zeros((paths, steps + 1))
        prices[:, 0] = S
        
        # Generate random paths
        Z = np.random.standard_normal((paths, steps))
        
        # Simulate price paths
        for t in range(1, steps + 1):
            prices[:, t] = prices[:, t-1] * np.exp(drift + vol * Z[:, t-1])
        
        # Calculate average price for each path
        if average_type == 'arithmetic':
            avg_prices = np.mean(prices, axis=1)
        else:  # geometric
            avg_prices = np.exp(np.mean(np.log(prices), axis=1))
        
        # Calculate payoffs
        if option_type == 'call':
            payoffs = np.maximum(0, avg_prices - K)
        else:
            payoffs = np.maximum(0, K - avg_prices)
        
        # Discount payoffs
        discounted_payoffs = np.exp(-r * T) * payoffs
        
        # Calculate option price and standard error
        option_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(paths)
        
        return option_price, std_error
    
    def lookback_option_price(self, 
                            S: float, 
                            T: float, 
                            steps: int, 
                            paths: int = 10000, 
                            option_type: str = 'call', 
                            floating_strike: bool = True, 
                            K: Optional[float] = None, 
                            r: Optional[float] = None, 
                            sigma: Optional[float] = None, 
                            q: Optional[float] = None, 
                            seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Calculate the price of a lookback option using Monte Carlo simulation.
        
        Parameters:
        -----------
        S : float
            Current price of the underlying asset
        T : float
            Time to expiration in years
        steps : int
            Number of time steps in the simulation
        paths : int
            Number of simulation paths
        option_type : str
            'call' for Call option, 'put' for Put option
        floating_strike : bool
            If True, use floating strike lookback option, else fixed strike
        K : float, optional
            Strike price for fixed strike lookback option (required if floating_strike=False)
        r, sigma, q, seed : Same as other methods
            
        Returns:
        --------
        Tuple[float, float]
            (Option price, Standard error)
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
        if q is None:
            q = self.dividend_yield
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        option_type = option_type.lower()
        if option_type not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")
        
        # For fixed strike lookback, need a strike price
        if not floating_strike and K is None:
            raise ValueError("Strike price K must be provided for fixed strike lookback options")
        
        # Monte Carlo simulation
        dt = T / steps
        drift = (r - q - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)
        
        # Initialize arrays for path simulation
        prices = np.zeros((paths, steps + 1))
        prices[:, 0] = S
        
        # Generate random paths
        Z = np.random.standard_normal((paths, steps))
        
        # Simulate price paths
        for t in range(1, steps + 1):
            prices[:, t] = prices[:, t-1] * np.exp(drift + vol * Z[:, t-1])
        
        # Calculate payoffs based on option type
        if floating_strike:
            if option_type == 'call':
                # Call payoff = S_T - min(S_t)
                payoffs = prices[:, -1] - np.min(prices, axis=1)
            else:
                # Put payoff = max(S_t) - S_T
                payoffs = np.max(prices, axis=1) - prices[:, -1]
        else:
            if option_type == 'call':
                # Call payoff = max(0, max(S_t) - K)
                payoffs = np.maximum(0, np.max(prices, axis=1) - K)
            else:
                # Put payoff = max(0, K - min(S_t))
                payoffs = np.maximum(0, K - np.min(prices, axis=1))
        
        # Discount payoffs
        discounted_payoffs = np.exp(-r * T) * payoffs
        
        # Calculate option price and standard error
        option_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(paths)
        
        return option_price, std_error
    
    def rainbow_option_price(self, 
                           S: List[float], 
                           T: float, 
                           corr_matrix: np.ndarray, 
                           steps: int, 
                           paths: int = 10000, 
                           option_type: str = 'call_on_max', 
                           K: float = 100.0, 
                           r: Optional[float] = None, 
                           sigma: Optional[List[float]] = None, 
                           q: Optional[List[float]] = None, 
                           seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Calculate the price of a rainbow option (multi-asset option) using Monte Carlo.
        
        Parameters:
        -----------
        S : List[float]
            List of current prices for each underlying asset
        T : float
            Time to expiration in years
        corr_matrix : np.ndarray
            Correlation matrix between assets
        steps : int
            Number of time steps in the simulation
        paths : int
            Number of simulation paths
        option_type : str
            'call_on_max', 'put_on_min', 'call_on_best', 'put_on_worst', etc.
        K : float
            Strike price
        r : float, optional
            Risk-free interest rate
        sigma : List[float], optional
            List of volatilities for each asset
        q : List[float], optional
            List of dividend yields for each asset
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        Tuple[float, float]
            (Option price, Standard error)
        """
        if r is None:
            r = self.risk_free_rate
        
        n_assets = len(S)
        
        # Handle single values for sigma and q
        if sigma is None:
            sigma = [self.volatility] * n_assets
        elif isinstance(sigma, (int, float)):
            sigma = [sigma] * n_assets
            
        if q is None:
            q = [self.dividend_yield] * n_assets
        elif isinstance(q, (int, float)):
            q = [q] * n_assets
        
        # Validate inputs
        if len(sigma) != n_assets or len(q) != n_assets:
            raise ValueError("Length of sigma and q must match number of assets")
        
        if corr_matrix.shape != (n_assets, n_assets):
            raise ValueError("Correlation matrix dimensions must match number of assets")
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Create the Cholesky decomposition of the correlation matrix
        try:
            chol = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            # If correlation matrix is not positive definite
            # Use nearest positive definite matrix
            # (simplified for this implementation)
            min_eig = np.min(np.linalg.eigvalsh(corr_matrix))
            if min_eig < 0:
                corr_matrix = corr_matrix + (-min_eig + 1e-6) * np.eye(n_assets)
            chol = np.linalg.cholesky(corr_matrix)
        
        # Monte Carlo simulation
        dt = T / steps
        
        # Initialize arrays for path simulation
        prices = np.zeros((paths, n_assets, steps + 1))
        for i in range(n_assets):
            prices[:, i, 0] = S[i]
        
        # Generate correlated random paths
        for t in range(1, steps + 1):
            # Generate independent random numbers
            Z = np.random.standard_normal((paths, n_assets))
            
            # Transform to correlated random numbers
            corr_Z = np.dot(Z, chol.T)
            
            # Update prices
            for i in range(n_assets):
                drift = (r - q[i] - 0.5 * sigma[i]**2) * dt
                vol = sigma[i] * np.sqrt(dt)
                prices[:, i, t] = prices[:, i, t-1] * np.exp(drift + vol * corr_Z[:, i])
        
        # Calculate payoffs based on option type
        if option_type == 'call_on_max':
            # Call on the maximum of multiple assets
            max_prices = np.max(prices[:, :, -1], axis=1)
            payoffs = np.maximum(0, max_prices - K)
            
        elif option_type == 'put_on_min':
            # Put on the minimum of multiple assets
            min_prices = np.min(prices[:, :, -1], axis=1)
            payoffs = np.maximum(0, K - min_prices)
            
        elif option_type == 'call_on_best':
            # Call on the best-performing asset
            returns = prices[:, :, -1] / np.array(S)
            best_returns = np.max(returns, axis=1)
            payoffs = np.maximum(0, (best_returns - 1) * K)
            
        elif option_type == 'put_on_worst':
            # Put on the worst-performing asset
            returns = prices[:, :, -1] / np.array(S)
            worst_returns = np.min(returns, axis=1)
            payoffs = np.maximum(0, (1 - worst_returns) * K)
            
        elif option_type == 'basket_call':
            # Call on a weighted average of assets
            # Simplified with equal weights
            avg_prices = np.mean(prices[:, :, -1], axis=1)
            payoffs = np.maximum(0, avg_prices - K)
            
        elif option_type == 'basket_put':
            # Put on a weighted average of assets
            # Simplified with equal weights
            avg_prices = np.mean(prices[:, :, -1], axis=1)
            payoffs = np.maximum(0, K - avg_prices)
            
        else:
            raise ValueError(f"Unsupported option type: {option_type}")
        
        # Discount payoffs
        discounted_payoffs = np.exp(-r * T) * payoffs
        
        # Calculate option price and standard error
        option_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(paths)
        
        return option_price, std_error
    
    def plot_exotic_option_comparison(self, 
                                     S_range: List[float], 
                                     K: float, 
                                     T: float, 
                                     option_types: List[str], 
                                     r: Optional[float] = None, 
                                     sigma: Optional[float] = None, 
                                     q: Optional[float] = None) -> None:
        """
        Plot comparison of different option types across a range of underlying prices.
        
        Parameters:
        -----------
        S_range : List[float]
            Range of underlying prices to plot
        K : float
            Strike price
        T : float
            Time to expiration
        option_types : List[str]
            List of option types to compare
        r, sigma, q : Same as other methods
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
        if q is None:
            q = self.dividend_yield
        
        # Initialize dictionary to store prices
        prices = {option_type: [] for option_type in option_types}
        
        # Calculate option prices for each S
        for S in S_range:
            for option_type in option_types:
                if option_type == 'vanilla_call':
                    price = self._vanilla_price(S, K, T, 'call', r, sigma, q)
                    prices[option_type].append(price)
                    
                elif option_type == 'vanilla_put':
                    price = self._vanilla_price(S, K, T, 'put', r, sigma, q)
                    prices[option_type].append(price)
                    
                elif option_type in ['binary_call', 'binary_put']:
                    type_parts = option_type.split('_')
                    price = self.binary_option_price(S, K, T, type_parts[1], 1.0, r, sigma, q)
                    prices[option_type].append(price)
                    
                elif option_type in ['up_and_out_call', 'down_and_out_call',
                                    'up_and_in_call', 'down_and_in_call',
                                    'up_and_out_put', 'down_and_out_put',
                                    'up_and_in_put', 'down_and_in_put']:
                    type_parts = option_type.split('_')
                    barrier_type = f"{type_parts[0]}-and-{type_parts[2]}"
                    opt_type = type_parts[3]
                    
                    # Set barrier level based on type
                    if type_parts[0] == 'up':
                        barrier = max(K * 1.2, S * 1.1)
                    else:  # down
                        barrier = min(K * 0.8, S * 0.9)
                    
                    price = self.barrier_option_price(S, K, T, barrier, opt_type, barrier_type, 0, r, sigma, q)
                    prices[option_type].append(price)
                    
                elif option_type in ['asian_call', 'asian_put', 'lookback_call', 'lookback_put']:
                    # Monte Carlo methods: simplified for plotting by using a small number of paths
                    type_parts = option_type.split('_')
                    opt_type = type_parts[1]
                    
                    if type_parts[0] == 'asian':
                        price, _ = self.asian_option_price(S, K, T, 50, 1000, opt_type, 'arithmetic', r, sigma, q)
                    else:  # lookback
                        price, _ = self.lookback_option_price(S, T, 50, 1000, opt_type, True, K, r, sigma, q)
                        
                    prices[option_type].append(price)
                    
                else:
                    prices[option_type].append(0)  # Unsupported type
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        for option_type, price_values in prices.items():
            plt.plot(S_range, price_values, label=option_type.replace('_', ' ').title())
        
        # Add reference line for strike price
        plt.axvline(x=K, color='gray', linestyle='--', alpha=0.5, label=f'Strike (K={K})')
        
        # Add labels and title
        plt.xlabel('Underlying Price')
        plt.ylabel('Option Price')
        plt.title(f'Comparison of Exotic Option Prices (T={T:.2f}, σ={sigma:.2f}, r={r:.2f})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create exotic options pricer
    exotic_pricer = ExoticOptions(risk_free_rate=0.05, volatility=0.2, dividend_yield=0.01)
    
    # Example parameters
    S = 100         # Current stock price
    K = 100         # Strike price
    T = 1.0         # Time to expiration (years)
    
    # Barrier option example
    barrier = 120   # Barrier level
    barrier_price = exotic_pricer.barrier_option_price(S, K, T, barrier, 'call', 'up-and-out')
    print(f"Up-and-out call option price: ${barrier_price:.4f}")
    
    # Binary option example
    binary_price = exotic_pricer.binary_option_price(S, K, T, 'call', 1.0)
    print(f"Binary call option price (payoff=1): ${binary_price:.4f}")
    
    # Asian option example (Monte Carlo)
    asian_price, asian_se = exotic_pricer.asian_option_price(S, K, T, 100, 10000, 'call')
    print(f"Asian call option price: ${asian_price:.4f} ± ${asian_se:.4f}")
    
    # Lookback option example (Monte Carlo)
    lookback_price, lookback_se = exotic_pricer.lookback_option_price(S, T, 100, 10000, 'call', True)
    print(f"Floating strike lookback call option price: ${lookback_price:.4f} ± ${lookback_se:.4f}")
    
    # Rainbow option example (Monte Carlo)
    S_multiple = [100, 105, 95]  # Prices of three assets
    corr_matrix = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.2],
        [0.3, 0.2, 1.0]
    ])
    sigma_multiple = [0.2, 0.25, 0.15]  # Volatilities
    
    rainbow_price, rainbow_se = exotic_pricer.rainbow_option_price(
        S_multiple, T, corr_matrix, 100, 10000, 'call_on_max', K
    )
    print(f"Rainbow call-on-max option price: ${rainbow_price:.4f} ± ${rainbow_se:.4f}")
    
    # Uncomment to run visualization
    # S_range = np.linspace(80, 120, 41)
    # option_types = ['vanilla_call', 'binary_call', 'up_and_out_call', 'down_and_in_call', 'asian_call']
    # exotic_pricer.plot_exotic_option_comparison(S_range, K, T, option_types) 