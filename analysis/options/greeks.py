import numpy as np
from scipy.stats import norm
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
import pandas as pd

class OptionGreeks:
    """
    A comprehensive class for calculating option Greeks and risk measures.
    
    Option Greeks are risk measures that describe the sensitivity of option prices
    to different factors such as underlying price, volatility, time to expiration,
    interest rates, and dividend yield.
    
    This class implements calculations for all standard Greeks (delta, gamma, theta,
    vega, rho) as well as higher-order and cross Greeks (charm, vanna, vomma, etc.).
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.0, 
                 volatility: float = 0.0,
                 dividend_yield: float = 0.0):
        """
        Initialize the OptionGreeks calculator.
        
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
    
    def _d1_d2(self, 
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
        if T <= 0 or sigma <= 0:
            raise ValueError("Time to expiration and volatility must be positive")
        
        d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    def calculate_greeks(self, 
                         S: float, 
                         K: float, 
                         T: float, 
                         option_type: str = 'call', 
                         r: Optional[float] = None, 
                         sigma: Optional[float] = None, 
                         q: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate standard option Greeks.
        
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
        Dict[str, float]
            Dictionary containing all standard Greeks
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
        if q is None:
            q = self.dividend_yield
        
        # Handle special case for zero or very small time to expiration
        if T <= 1e-10:
            result = {}
            # For very small T, option behaves like its intrinsic value
            if option_type.lower() == 'call':
                result['delta'] = 1.0 if S > K else 0.0
            else:  # Put
                result['delta'] = -1.0 if S < K else 0.0
            
            result['gamma'] = 0.0
            result['theta'] = 0.0
            result['vega'] = 0.0
            result['rho'] = 0.0
            result['charm'] = 0.0
            result['vanna'] = 0.0
            result['vomma'] = 0.0
            result['veta'] = 0.0
            result['speed'] = 0.0
            result['zomma'] = 0.0
            result['color'] = 0.0
            result['ultima'] = 0.0
            
            return result
        
        # Calculate d1 and d2
        d1, d2 = self._d1_d2(S, K, T, r, sigma, q)
        
        # Common terms
        sqrt_t = np.sqrt(T)
        nd1 = norm.pdf(d1)  # Standard normal PDF at d1
        nd2 = norm.pdf(d2)  # Standard normal PDF at d2
        cdf_d1 = norm.cdf(d1)  # Standard normal CDF at d1
        cdf_neg_d1 = norm.cdf(-d1)  # Standard normal CDF at -d1
        cdf_d2 = norm.cdf(d2)  # Standard normal CDF at d2
        cdf_neg_d2 = norm.cdf(-d2)  # Standard normal CDF at -d2
        
        exp_qt = np.exp(-q * T)
        exp_rt = np.exp(-r * T)
        
        # Initialize result dictionary
        result = {}
        
        # Delta - Sensitivity to change in underlying price
        if option_type.lower() == 'call':
            result['delta'] = exp_qt * cdf_d1
        else:  # Put
            result['delta'] = exp_qt * (cdf_d1 - 1)
        
        # Gamma - Sensitivity of delta to change in underlying price
        # Same for both calls and puts
        result['gamma'] = (exp_qt * nd1) / (S * sigma * sqrt_t)
        
        # Theta - Sensitivity to change in time to expiration
        # Convert to daily theta (252 trading days per year)
        days_per_year = 365.0
        if option_type.lower() == 'call':
            theta = -(S * sigma * exp_qt * nd1) / (2 * sqrt_t) - r * K * exp_rt * cdf_d2 + q * S * exp_qt * cdf_d1
        else:  # Put
            theta = -(S * sigma * exp_qt * nd1) / (2 * sqrt_t) + r * K * exp_rt * cdf_neg_d2 - q * S * exp_qt * cdf_neg_d1
        
        result['theta'] = theta / days_per_year
        
        # Vega - Sensitivity to change in volatility
        # Typically expressed per 1% change in volatility
        result['vega'] = S * exp_qt * nd1 * sqrt_t / 100
        
        # Rho - Sensitivity to change in interest rate
        # Typically expressed per 1% change in rate
        if option_type.lower() == 'call':
            result['rho'] = K * T * exp_rt * cdf_d2 / 100
        else:  # Put
            result['rho'] = -K * T * exp_rt * cdf_neg_d2 / 100
        
        # Higher order and cross Greeks
        
        # Charm (Delta decay) - Rate of change of delta with respect to time
        if option_type.lower() == 'call':
            charm = q * exp_qt * cdf_d1 - exp_qt * nd1 * (2 * (r - q) - d2 * sigma * sqrt_t) / (2 * T * sigma * sqrt_t)
        else:  # Put
            charm = -q * exp_qt * cdf_neg_d1 - exp_qt * nd1 * (2 * (r - q) - d2 * sigma * sqrt_t) / (2 * T * sigma * sqrt_t)
        
        result['charm'] = charm / days_per_year
        
        # Vanna - Sensitivity of delta to change in volatility
        # (or sensitivity of vega to change in underlying price)
        result['vanna'] = -exp_qt * nd1 * d2 / sigma / 100
        
        # Vomma (Volga) - Sensitivity of vega to change in volatility
        result['vomma'] = S * exp_qt * nd1 * sqrt_t * d1 * d2 / 100**2
        
        # Veta - Sensitivity of vega to change in time to expiration
        veta = -S * exp_qt * nd1 * sqrt_t * (q + (r - q) * d1 / (sigma * sqrt_t) - (1 + d1 * d2) / (2 * T))
        result['veta'] = veta / (100 * days_per_year)
        
        # Speed - Sensitivity of gamma to change in underlying price
        result['speed'] = -exp_qt * nd1 * (d1 / (S * sigma * sqrt_t) + 1) / (S**2 * sigma * sqrt_t)
        
        # Zomma - Sensitivity of gamma to change in volatility
        result['zomma'] = exp_qt * nd1 * (d1 * d2 - 1) / (S * sigma**2 * sqrt_t) / 100
        
        # Color - Sensitivity of gamma to change in time to expiration
        color_term = 2 * (r - q) * d1 + (2 * q + 1) * sigma * sqrt_t
        color = -exp_qt * nd1 * (color_term / (2 * T * S * sigma * sqrt_t))
        result['color'] = color / days_per_year
        
        # Ultima - Sensitivity of vomma to change in volatility
        result['ultima'] = S * exp_qt * nd1 * sqrt_t * d1 * d2 * (1 + d1 * d2) / (sigma**2) / 100**3
        
        return result
    
    def calculate_all_greeks_series(self,
                                    S: float,
                                    K: float,
                                    T_range: Union[List[float], np.ndarray],
                                    option_type: str = 'call',
                                    r: Optional[float] = None,
                                    sigma: Optional[float] = None,
                                    q: Optional[float] = None) -> pd.DataFrame:
        """
        Calculate all Greeks over a range of time to expiration values.
        
        Parameters:
        -----------
        S : float
            Current price of the underlying asset
        K : float
            Strike price of the option
        T_range : List[float] or np.ndarray
            Range of time to expiration values in years
        option_type, r, sigma, q : Same as calculate_greeks method
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with T values as index and Greeks as columns
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
        if q is None:
            q = self.dividend_yield
        
        # Create a list to hold results
        results = []
        
        # Calculate Greeks for each T value
        for T in T_range:
            # Get Greeks for this T
            greeks = self.calculate_greeks(S, K, T, option_type, r, sigma, q)
            
            # Add T value to the dictionary
            greeks['time_to_expiry'] = T
            
            # Append to results
            results.append(greeks)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Set time_to_expiry as index
        df.set_index('time_to_expiry', inplace=True)
        
        return df
    
    def plot_greeks_vs_price(self,
                            K: float,
                            T: float,
                            price_range: Optional[Tuple[float, float]] = None,
                            option_type: str = 'call',
                            r: Optional[float] = None,
                            sigma: Optional[float] = None,
                            q: Optional[float] = None,
                            num_points: int = 100,
                            plot_greeks: List[str] = None) -> None:
        """
        Plot option Greeks against underlying price.
        
        Parameters:
        -----------
        K : float
            Strike price of the option
        T : float
            Time to expiration in years
        price_range : Tuple[float, float], optional
            Range of underlying prices to plot (min, max)
        option_type, r, sigma, q : Same as calculate_greeks method
        num_points : int
            Number of points to calculate for the curve
        plot_greeks : List[str], optional
            List of Greeks to plot. If None, plots delta, gamma, theta, vega, rho
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
        if q is None:
            q = self.dividend_yield
        
        # Set default Greeks to plot if not specified
        if plot_greeks is None:
            plot_greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
        
        # Default price range if not provided
        if price_range is None:
            price_range = (K * 0.6, K * 1.4)
        
        # Generate price points
        S_values = np.linspace(price_range[0], price_range[1], num_points)
        
        # Calculate Greeks for each price point
        data = pd.DataFrame(index=S_values)
        
        for S in S_values:
            greeks = self.calculate_greeks(S, K, T, option_type, r, sigma, q)
            for greek, value in greeks.items():
                if greek in plot_greeks:
                    data.loc[S, greek] = value
        
        # Create plot
        fig, axes = plt.subplots(len(plot_greeks), 1, figsize=(10, 3 * len(plot_greeks)), sharex=True)
        
        # If only one Greek is plotted, axes is not a list
        if len(plot_greeks) == 1:
            axes = [axes]
        
        # Plot each Greek
        for i, greek in enumerate(plot_greeks):
            ax = axes[i]
            ax.plot(S_values, data[greek])
            ax.set_ylabel(greek.capitalize())
            ax.set_title(f"{greek.capitalize()} vs. Underlying Price")
            ax.grid(True, alpha=0.3)
            
            # Add strike price reference line
            ax.axvline(x=K, color='r', linestyle='--', alpha=0.5)
        
        # Set common x-axis label
        axes[-1].set_xlabel('Underlying Price')
        
        # Add overall title
        option_type_str = "Call" if option_type.lower() == 'call' else "Put"
        plt.suptitle(f"{option_type_str} Option Greeks (K={K}, T={T:.2f}, σ={sigma:.2f}, r={r:.2f})", fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()
    
    def plot_greeks_surface(self,
                           K: float,
                           S_range: Tuple[float, float],
                           T_range: Tuple[float, float],
                           option_type: str = 'call',
                           greek: str = 'delta',
                           r: Optional[float] = None,
                           sigma: Optional[float] = None,
                           q: Optional[float] = None,
                           num_points: int = 20) -> None:
        """
        Plot a 3D surface of a specific Greek as a function of
        underlying price and time to expiration.
        
        Parameters:
        -----------
        K : float
            Strike price of the option
        S_range : Tuple[float, float]
            Range of underlying prices (min, max)
        T_range : Tuple[float, float]
            Range of time to expiration in years (min, max)
        option_type : str
            'call' for Call option, 'put' for Put option
        greek : str
            The Greek to plot (e.g., 'delta', 'gamma', etc.)
        r, sigma, q : Same as calculate_greeks method
        num_points : int
            Number of points to calculate in each dimension
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
        if q is None:
            q = self.dividend_yield
        
        # Generate meshgrid of S and T values
        S_values = np.linspace(S_range[0], S_range[1], num_points)
        T_values = np.linspace(T_range[0], T_range[1], num_points)
        S_mesh, T_mesh = np.meshgrid(S_values, T_values)
        
        # Calculate the Greek values for each (S, T) pair
        Z = np.zeros_like(S_mesh)
        
        for i in range(num_points):
            for j in range(num_points):
                S = S_mesh[i, j]
                T = T_mesh[i, j]
                try:
                    greeks = self.calculate_greeks(S, K, T, option_type, r, sigma, q)
                    Z[i, j] = greeks[greek]
                except:
                    Z[i, j] = np.nan
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create the surface plot
        surface = ax.plot_surface(S_mesh, T_mesh, Z, cmap='viridis', alpha=0.8)
        
        # Add color bar
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
        
        # Add labels and title
        ax.set_xlabel('Underlying Price')
        ax.set_ylabel('Time to Expiration (years)')
        ax.set_zlabel(f'{greek.capitalize()}')
        
        option_type_str = "Call" if option_type.lower() == 'call' else "Put"
        ax.set_title(f"{greek.capitalize()} Surface for {option_type_str} Option (K={K})")
        
        plt.tight_layout()
        plt.show()
    
    def risk_dashboard(self,
                       positions: List[Dict],
                       plot: bool = True) -> pd.DataFrame:
        """
        Calculate aggregate Greeks for a portfolio of options positions.
        
        Parameters:
        -----------
        positions : List[Dict]
            List of dictionaries, each containing:
            - 'S': underlying price
            - 'K': strike price
            - 'T': time to expiration
            - 'type': 'call' or 'put'
            - 'quantity': number of contracts (negative for short positions)
            - 'r', 'sigma', 'q': optional overrides for individual positions
        plot : bool
            Whether to create a visualization of portfolio Greeks
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with portfolio risk metrics
        """
        if not positions:
            raise ValueError("No positions provided")
        
        # Calculate Greeks for each position
        position_greeks = []
        
        for pos in positions:
            # Extract position details
            S = pos['S']
            K = pos['K']
            T = pos['T']
            option_type = pos['type']
            quantity = pos['quantity']
            
            # Extract overrides if provided
            r = pos.get('r', self.risk_free_rate)
            sigma = pos.get('sigma', self.volatility)
            q = pos.get('q', self.dividend_yield)
            
            # Calculate Greeks
            greeks = self.calculate_greeks(S, K, T, option_type, r, sigma, q)
            
            # Multiply by quantity
            for greek in greeks:
                greeks[greek] *= quantity
            
            # Add position details
            greeks['S'] = S
            greeks['K'] = K
            greeks['T'] = T
            greeks['type'] = option_type
            greeks['quantity'] = quantity
            
            position_greeks.append(greeks)
        
        # Convert to DataFrame
        position_df = pd.DataFrame(position_greeks)
        
        # Calculate aggregate Greeks
        agg_greeks = position_df.drop(['S', 'K', 'T', 'type', 'quantity'], axis=1).sum()
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Greek': agg_greeks.index,
            'Value': agg_greeks.values,
            'Description': [
                'Change in portfolio value for a $1 change in underlying',
                'Rate of change of delta with respect to underlying price',
                'Change in portfolio value per day passing',
                'Change in portfolio value for a 1% change in volatility',
                'Change in portfolio value for a 1% change in interest rates',
                'Rate of change of delta with respect to time',
                'Change in delta for a 1% change in volatility',
                'Change in vega for a 1% change in volatility',
                'Change in vega with respect to time',
                'Rate of change of gamma with respect to underlying price',
                'Rate of change of gamma with respect to volatility',
                'Rate of change of gamma with respect to time',
                'Rate of change of vomma with respect to volatility'
            ]
        })
        
        # Visualize if requested
        if plot:
            # Focus on the main Greeks
            main_greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
            main_values = [agg_greeks[g] for g in main_greeks]
            
            # Create bar chart
            plt.figure(figsize=(12, 6))
            bars = plt.bar(main_greeks, main_values)
            
            # Color code bars (positive green, negative red)
            for i, bar in enumerate(bars):
                bar.set_color('green' if main_values[i] >= 0 else 'red')
            
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.title('Portfolio Risk Exposure')
            plt.ylabel('Aggregate Exposure')
            plt.grid(True, axis='y', alpha=0.3)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., 
                         height + (0.01 if height >= 0 else -0.01),
                         f'{height:.4f}',
                         ha='center', va='bottom' if height >= 0 else 'top')
            
            plt.tight_layout()
            plt.show()
        
        return results

    def delta_hedging_simulation(self,
                                S0: float,
                                K: float,
                                T: float,
                                option_type: str = 'call',
                                r: Optional[float] = None,
                                sigma: Optional[float] = None,
                                q: Optional[float] = None,
                                paths: int = 1,
                                steps: int = 252,
                                plot: bool = True) -> Dict:
        """
        Simulate delta hedging of an option position.
        
        Parameters:
        -----------
        S0 : float
            Initial underlying price
        K : float
            Strike price
        T : float
            Time to expiration in years
        option_type, r, sigma, q : Same as calculate_greeks method
        paths : int
            Number of price paths to simulate
        steps : int
            Number of time steps in simulation (e.g., 252 trading days)
        plot : bool
            Whether to plot the simulation results
            
        Returns:
        --------
        Dict
            Dictionary with simulation results
        """
        if r is None:
            r = self.risk_free_rate
        if sigma is None:
            sigma = self.volatility
        if q is None:
            q = self.dividend_yield
        
        # Time grid
        dt = T / steps
        times = np.linspace(0, T, steps + 1)
        
        # Price process
        from scipy.stats import norm
        
        # Initialize storage for results
        portfolio_values = np.zeros((paths, steps + 1))
        stock_positions = np.zeros((paths, steps + 1))
        option_prices = np.zeros((paths, steps + 1))
        stock_prices = np.zeros((paths, steps + 1))
        cash_positions = np.zeros((paths, steps + 1))
        
        # Simulate each path
        for path in range(paths):
            # Generate random price path
            Z = np.random.standard_normal(steps)
            S = np.zeros(steps + 1)
            S[0] = S0
            
            for t in range(1, steps + 1):
                S[t] = S[t-1] * np.exp((r - q - 0.5 * sigma**2) * dt + 
                                      sigma * np.sqrt(dt) * Z[t-1])
            
            stock_prices[path] = S
            
            # Initial option price and delta
            # For time = 0
            time_to_expiry = T
            greeks = self.calculate_greeks(S[0], K, time_to_expiry, option_type, r, sigma, q)
            
            # Calculate option price using Black-Scholes formula
            d1, d2 = self._d1_d2(S[0], K, time_to_expiry, r, sigma, q)
            if option_type.lower() == 'call':
                option_price = S[0] * np.exp(-q * time_to_expiry) * norm.cdf(d1) - \
                              K * np.exp(-r * time_to_expiry) * norm.cdf(d2)
            else:  # Put
                option_price = K * np.exp(-r * time_to_expiry) * norm.cdf(-d2) - \
                              S[0] * np.exp(-q * time_to_expiry) * norm.cdf(-d1)
            
            option_prices[path, 0] = option_price
            delta = greeks['delta']
            
            # Initialize positions (short 1 option, long delta shares)
            stock_position = delta
            cash_position = -option_price + delta * S[0]
            
            stock_positions[path, 0] = stock_position
            cash_positions[path, 0] = cash_position
            portfolio_values[path, 0] = cash_position - stock_position * S[0] + option_price
            
            # Simulate delta hedging over time
            for t in range(1, steps + 1):
                time_to_expiry = T - times[t]
                if time_to_expiry <= 0:
                    # At expiration
                    if option_type.lower() == 'call':
                        option_price = max(0, S[t] - K)
                    else:  # Put
                        option_price = max(0, K - S[t])
                    
                    # Liquidate position
                    cash_position += stock_position * S[t]
                    stock_position = 0
                else:
                    # Calculate new option price and delta
                    greeks = self.calculate_greeks(S[t], K, time_to_expiry, option_type, r, sigma, q)
                    
                    # Calculate option price
                    d1, d2 = self._d1_d2(S[t], K, time_to_expiry, r, sigma, q)
                    if option_type.lower() == 'call':
                        option_price = S[t] * np.exp(-q * time_to_expiry) * norm.cdf(d1) - \
                                      K * np.exp(-r * time_to_expiry) * norm.cdf(d2)
                    else:  # Put
                        option_price = K * np.exp(-r * time_to_expiry) * norm.cdf(-d2) - \
                                      S[t] * np.exp(-q * time_to_expiry) * norm.cdf(-d1)
                    
                    new_delta = greeks['delta']
                    
                    # Update positions
                    cash_position = cash_position * np.exp(r * dt) - (new_delta - stock_position) * S[t]
                    stock_position = new_delta
                
                option_prices[path, t] = option_price
                stock_positions[path, t] = stock_position
                cash_positions[path, t] = cash_position
                portfolio_values[path, t] = cash_position + stock_position * S[t] - option_price
        
        # Calculate statistics
        final_pnl = portfolio_values[:, -1]
        mean_pnl = np.mean(final_pnl)
        std_pnl = np.std(final_pnl)
        
        results = {
            'mean_pnl': mean_pnl,
            'std_pnl': std_pnl,
            'sharpe': mean_pnl / std_pnl if std_pnl > 0 else 0,
            'times': times,
            'stock_prices': stock_prices,
            'option_prices': option_prices,
            'stock_positions': stock_positions,
            'cash_positions': cash_positions,
            'portfolio_values': portfolio_values
        }
        
        # Plot if requested
        if plot and paths == 1:  # Only plot for single path simulation
            # Create a figure with subplots
            fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
            
            # Plot stock price
            axes[0].plot(times, stock_prices[0], 'b-')
            axes[0].set_ylabel('Stock Price')
            axes[0].set_title('Stock Price Path')
            axes[0].grid(True, alpha=0.3)
            
            # Plot option price
            axes[1].plot(times, option_prices[0], 'r-')
            axes[1].set_ylabel('Option Price')
            axes[1].set_title('Option Price')
            axes[1].grid(True, alpha=0.3)
            
            # Plot delta (stock position)
            axes[2].plot(times, stock_positions[0], 'g-')
            axes[2].set_ylabel('Delta')
            axes[2].set_title('Stock Position (Delta)')
            axes[2].grid(True, alpha=0.3)
            
            # Plot portfolio value
            axes[3].plot(times, portfolio_values[0], 'm-')
            axes[3].axhline(y=0, color='k', linestyle='--')
            axes[3].set_xlabel('Time (years)')
            axes[3].set_ylabel('P&L')
            axes[3].set_title(f'Hedging P&L (Final: ${portfolio_values[0, -1]:.2f})')
            axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize the Greeks calculator
    greeks_calc = OptionGreeks(risk_free_rate=0.05, volatility=0.2, dividend_yield=0.01)
    
    # Calculate Greeks for a call option
    S = 100       # Current stock price
    K = 100       # Strike price
    T = 1.0       # Time to expiration (years)
    
    # Get all Greeks
    greek_values = greeks_calc.calculate_greeks(S, K, T, 'call')
    
    print("Option Greeks for Call Option:")
    for greek, value in greek_values.items():
        print(f"{greek.capitalize()}: {value:.6f}")
    
    # Example of calculating Greeks for different time points
    times = np.linspace(0.1, 1.0, 10)
    greeks_series = greeks_calc.calculate_all_greeks_series(S, K, times, 'call')
    
    print("\nGreeks vs. Time to Expiration:")
    print(greeks_series[['delta', 'gamma', 'theta', 'vega', 'rho']])
    
    # Demo of portfolio risk calculation
    portfolio = [
        {'S': 100, 'K': 100, 'T': 1.0, 'type': 'call', 'quantity': 1},
        {'S': 100, 'K': 95, 'T': 1.0, 'type': 'put', 'quantity': -2},
        {'S': 100, 'K': 105, 'T': 0.5, 'type': 'call', 'quantity': -1}
    ]
    
    portfolio_risk = greeks_calc.risk_dashboard(portfolio, plot=False)
    
    print("\nPortfolio Risk Metrics:")
    print(portfolio_risk[['Greek', 'Value']])
    
    # Visual plots are commented out for non-interactive environments
    # greeks_calc.plot_greeks_vs_price(K, T, option_type='call')
    # greeks_calc.plot_greeks_surface(K, (80, 120), (0.1, 1.0), option_type='call', greek='delta')
    # greeks_calc.delta_hedging_simulation(S, K, T, option_type='call') 