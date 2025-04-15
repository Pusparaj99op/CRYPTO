import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Union, Optional, Callable
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class ImpliedVolatility:
    """
    A class for calculating implied volatility and modeling volatility surfaces.
    
    The implied volatility is the volatility parameter that makes the Black-Scholes
    price equal to the market price. This class provides methods to calculate implied
    volatility for individual options and to construct and visualize volatility surfaces.
    """
    
    def __init__(self, risk_free_rate: float = 0.0, dividend_yield: float = 0.0):
        """
        Initialize the ImpliedVolatility calculator.
        
        Parameters:
        -----------
        risk_free_rate : float
            Annual risk-free interest rate (decimal)
        dividend_yield : float
            Annual dividend yield (decimal)
        """
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
    
    def black_scholes_price(self, 
                           S: float, 
                           K: float, 
                           T: float, 
                           sigma: float, 
                           option_type: str = 'call', 
                           r: Optional[float] = None, 
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
        sigma : float
            Volatility
        option_type : str
            'call' for Call option, 'put' for Put option
        r : float, optional
            Risk-free interest rate (if None, use the instance value)
        q : float, optional
            Dividend yield (if None, use the instance value)
            
        Returns:
        --------
        float
            Black-Scholes option price
        """
        if r is None:
            r = self.risk_free_rate
        if q is None:
            q = self.dividend_yield
        
        # Handle very small time to expiration
        if T <= 1e-10:
            if option_type.lower() == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate option price
        if option_type.lower() == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        return price
    
    def _objective_function(self, 
                          sigma: float, 
                          S: float, 
                          K: float, 
                          T: float, 
                          market_price: float, 
                          option_type: str, 
                          r: float, 
                          q: float) -> float:
        """
        Objective function for implied volatility calculation.
        
        Calculates the squared difference between model price and market price.
        
        Parameters:
        -----------
        sigma : float
            Volatility parameter to optimize
        S, K, T, option_type, r, q : Same as black_scholes_price method
        market_price : float
            Observed market price of the option
            
        Returns:
        --------
        float
            Squared difference between model price and market price
        """
        model_price = self.black_scholes_price(S, K, T, sigma, option_type, r, q)
        return (model_price - market_price)**2
    
    def calc_implied_volatility(self, 
                              market_price: float, 
                              S: float, 
                              K: float, 
                              T: float, 
                              option_type: str = 'call', 
                              r: Optional[float] = None, 
                              q: Optional[float] = None,
                              initial_guess: float = 0.2,
                              method: str = 'BFGS') -> float:
        """
        Calculate implied volatility using numerical optimization.
        
        Parameters:
        -----------
        market_price : float
            Market price of the option
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
        q : float, optional
            Dividend yield (if None, use the instance value)
        initial_guess : float
            Initial guess for volatility
        method : str
            Optimization method to use
            
        Returns:
        --------
        float
            Implied volatility
        """
        if r is None:
            r = self.risk_free_rate
        if q is None:
            q = self.dividend_yield
        
        # Intrinsic value checks
        if T <= 1e-10:
            return 0.0  # No time value at expiration
        
        if option_type.lower() == 'call':
            intrinsic = max(0, S - K)
        else:
            intrinsic = max(0, K - S)
        
        if abs(market_price - intrinsic) <= 1e-10:
            return 0.0  # Priced at intrinsic value
        
        # Bracket the search space for volatility with reasonable bounds
        bounds = [(0.001, 5.0)]
        
        # Perform optimization
        result = minimize(
            self._objective_function,
            initial_guess,
            args=(S, K, T, market_price, option_type, r, q),
            method=method,
            bounds=bounds
        )
        
        if not result.success:
            raise RuntimeError(f"Implied volatility calculation failed: {result.message}")
        
        return result.x[0]
    
    def calc_volatility_surface(self, 
                              market_data: pd.DataFrame, 
                              S: float, 
                              r: Optional[float] = None, 
                              q: Optional[float] = None) -> pd.DataFrame:
        """
        Calculate implied volatility surface from market data.
        
        Parameters:
        -----------
        market_data : pd.DataFrame
            DataFrame with columns:
            - strike: Strike price
            - expiry: Time to expiration in years
            - call_price: Call option price
            - put_price: Put option price (optional, but either call or put required)
        S : float
            Current price of the underlying asset
        r : float, optional
            Risk-free interest rate (if None, use the instance value)
        q : float, optional
            Dividend yield (if None, use the instance value)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns for strike, expiry, and implied volatility
        """
        if r is None:
            r = self.risk_free_rate
        if q is None:
            q = self.dividend_yield
        
        required_cols = ['strike', 'expiry']
        if not all(col in market_data.columns for col in required_cols):
            raise ValueError(f"Market data must contain columns: {required_cols}")
        
        if 'call_price' not in market_data.columns and 'put_price' not in market_data.columns:
            raise ValueError("Market data must contain either 'call_price' or 'put_price' column")
        
        # Create a copy to avoid modifying the original
        data = market_data.copy()
        
        # Calculate implied volatility for each option
        iv_values = []
        
        for _, row in data.iterrows():
            K = row['strike']
            T = row['expiry']
            
            # Calculate IV from call price if available, otherwise use put price
            if 'call_price' in data.columns and not pd.isna(row['call_price']):
                try:
                    iv = self.calc_implied_volatility(row['call_price'], S, K, T, 'call', r, q)
                    iv_values.append(iv)
                except:
                    iv_values.append(np.nan)
            elif 'put_price' in data.columns and not pd.isna(row['put_price']):
                try:
                    iv = self.calc_implied_volatility(row['put_price'], S, K, T, 'put', r, q)
                    iv_values.append(iv)
                except:
                    iv_values.append(np.nan)
            else:
                iv_values.append(np.nan)
        
        # Add IV to the DataFrame
        data['implied_volatility'] = iv_values
        
        # Filter out invalid IV values
        data = data[~np.isnan(data['implied_volatility'])]
        
        return data
    
    def fit_volatility_surface(self, 
                              surface_data: pd.DataFrame, 
                              model: str = 'svi',
                              moneyness: bool = True) -> Dict:
        """
        Fit a parametric model to the volatility surface.
        
        Parameters:
        -----------
        surface_data : pd.DataFrame
            DataFrame with columns for strike, expiry, and implied_volatility
        model : str
            Model to fit ('svi' for Stochastic Volatility Inspired model,
            'cubic' for cubic spline, 'quadratic' for quadratic polynomial)
        moneyness : bool
            If True, use moneyness (K/S) instead of absolute strikes
            
        Returns:
        --------
        Dict
            Dictionary with fitted model parameters
        """
        required_cols = ['strike', 'expiry', 'implied_volatility']
        if not all(col in surface_data.columns for col in required_cols):
            raise ValueError(f"Surface data must contain columns: {required_cols}")
        
        if model.lower() not in ['svi', 'cubic', 'quadratic']:
            raise ValueError("Model must be one of: 'svi', 'cubic', 'quadratic'")
        
        # Create a copy to avoid modifying the original
        data = surface_data.copy()
        
        # Use moneyness (K/S) if specified
        if 'S' in data.columns and moneyness:
            data['moneyness'] = data['strike'] / data['S']
            x_col = 'moneyness'
        else:
            x_col = 'strike'
        
        # Group by expiry
        grouped = data.groupby('expiry')
        
        # Initialize results
        fit_results = {}
        
        for expiry, group in grouped:
            # Sort by strike/moneyness
            group = group.sort_values(x_col)
            
            x = group[x_col].values
            y = group['implied_volatility'].values
            
            if model.lower() == 'svi':
                # SVI model: a + b * (rho * (x - m) + sqrt((x - m)^2 + sigma^2))
                # Initialize parameters [a, b, rho, m, sigma]
                initial_guess = [np.min(y), 0.1, 0.0, np.mean(x), 0.1]
                
                # Define SVI objective function
                def svi_objective(params, x, y):
                    a, b, rho, m, sigma = params
                    if abs(rho) > 1 or b < 0 or sigma <= 0:
                        return 1e10  # penalty for invalid parameters
                    y_pred = a + b * (rho * (x - m) + np.sqrt((x - m)**2 + sigma**2))
                    return np.sum((y - y_pred)**2)
                
                # Fit SVI model
                bounds = [
                    (0, None),      # a: minimum volatility >= 0
                    (0, None),      # b: slope >= 0
                    (-1, 1),        # rho: correlation between -1 and 1
                    (None, None),   # m: location parameter (unrestricted)
                    (1e-6, None)    # sigma: standard deviation > 0
                ]
                
                result = minimize(
                    svi_objective,
                    initial_guess,
                    args=(x, y),
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                # Store parameters
                fit_results[expiry] = {
                    'model': 'svi',
                    'params': result.x,
                    'param_names': ['a', 'b', 'rho', 'm', 'sigma'],
                    'success': result.success,
                    'x_col': x_col
                }
                
            elif model.lower() == 'cubic':
                # Fit cubic spline
                from scipy.interpolate import CubicSpline
                cs = CubicSpline(x, y)
                
                # Store spline object
                fit_results[expiry] = {
                    'model': 'cubic',
                    'spline': cs,
                    'x_col': x_col
                }
                
            elif model.lower() == 'quadratic':
                # Fit quadratic polynomial
                coeffs = np.polyfit(x, y, 2)
                
                # Store coefficients
                fit_results[expiry] = {
                    'model': 'quadratic',
                    'coeffs': coeffs,
                    'x_col': x_col
                }
        
        return fit_results
    
    def predict_volatility(self, 
                          fit_results: Dict, 
                          K: Union[float, np.ndarray], 
                          T: float, 
                          S: Optional[float] = None) -> Union[float, np.ndarray]:
        """
        Predict implied volatility using a fitted model.
        
        Parameters:
        -----------
        fit_results : Dict
            Fitted model parameters from fit_volatility_surface
        K : float or np.ndarray
            Strike price(s)
        T : float
            Time to expiration in years
        S : float, optional
            Current price of the underlying asset (required if model uses moneyness)
            
        Returns:
        --------
        float or np.ndarray
            Predicted implied volatility
        """
        # Check if we have the exact expiry in our fit results
        if T not in fit_results:
            # Find nearest expiry
            expiries = np.array(list(fit_results.keys()))
            idx = np.abs(expiries - T).argmin()
            nearest_T = expiries[idx]
            model_params = fit_results[nearest_T]
        else:
            model_params = fit_results[T]
        
        # Convert to moneyness if needed
        if model_params['x_col'] == 'moneyness':
            if S is None:
                raise ValueError("S is required when model uses moneyness")
            x = K / S
        else:
            x = K
        
        # Make prediction based on model type
        if model_params['model'] == 'svi':
            a, b, rho, m, sigma = model_params['params']
            return a + b * (rho * (x - m) + np.sqrt((x - m)**2 + sigma**2))
        
        elif model_params['model'] == 'cubic':
            return model_params['spline'](x)
        
        elif model_params['model'] == 'quadratic':
            coeffs = model_params['coeffs']
            return coeffs[0] * x**2 + coeffs[1] * x + coeffs[2]
    
    def plot_volatility_surface(self, 
                               surface_data: pd.DataFrame, 
                               fit_results: Optional[Dict] = None,
                               plot_type: str = '3d',
                               moneyness: bool = True) -> None:
        """
        Plot volatility surface from market data and fitted model.
        
        Parameters:
        -----------
        surface_data : pd.DataFrame
            DataFrame with columns for strike, expiry, and implied_volatility
        fit_results : Dict, optional
            Fitted model parameters from fit_volatility_surface
        plot_type : str
            Plot type ('3d', 'contour', or 'smile')
        moneyness : bool
            If True, use moneyness (K/S) instead of absolute strikes
        """
        required_cols = ['strike', 'expiry', 'implied_volatility']
        if not all(col in surface_data.columns for col in required_cols):
            raise ValueError(f"Surface data must contain columns: {required_cols}")
        
        # Create a copy to avoid modifying the original
        data = surface_data.copy()
        
        # Use moneyness (K/S) if specified
        if 'S' in data.columns and moneyness:
            data['moneyness'] = data['strike'] / data['S']
            x_col = 'moneyness'
            x_label = 'Moneyness (K/S)'
        else:
            x_col = 'strike'
            x_label = 'Strike Price'
        
        if plot_type == '3d':
            # 3D surface plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot market data points
            scatter = ax.scatter(
                data[x_col], 
                data['expiry'],
                data['implied_volatility'],
                c=data['implied_volatility'],
                cmap='viridis',
                s=50,
                alpha=0.7
            )
            
            # Plot fitted surface if provided
            if fit_results is not None:
                # Create mesh grid for surface
                x_unique = np.linspace(data[x_col].min(), data[x_col].max(), 50)
                t_unique = np.sort(data['expiry'].unique())
                X, T = np.meshgrid(x_unique, t_unique)
                Z = np.zeros_like(X)
                
                # Calculate surface values
                for i, t in enumerate(t_unique):
                    for j, k in enumerate(x_unique):
                        S_val = None if x_col == 'strike' else 1.0  # dummy value for moneyness
                        Z[i, j] = self.predict_volatility(fit_results, k, t, S_val)
                
                # Plot surface
                surf = ax.plot_surface(
                    X, T, Z, 
                    cmap='viridis',
                    alpha=0.7,
                    linewidth=0,
                    antialiased=True
                )
            
            # Add labels and title
            ax.set_xlabel(x_label)
            ax.set_ylabel('Time to Expiry (years)')
            ax.set_zlabel('Implied Volatility')
            plt.title('Implied Volatility Surface')
            
            # Add colorbar
            fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
            
        elif plot_type == 'contour':
            # Contour plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create pivot table
            pivot = data.pivot_table(
                values='implied_volatility',
                index='expiry',
                columns=x_col,
                aggfunc='mean'
            )
            
            # Create contour plot
            contour = ax.contourf(
                pivot.columns,
                pivot.index,
                pivot.values,
                20,
                cmap='viridis'
            )
            
            # Add labels and title
            ax.set_xlabel(x_label)
            ax.set_ylabel('Time to Expiry (years)')
            plt.title('Implied Volatility Surface (Contour)')
            
            # Add colorbar
            fig.colorbar(contour, ax=ax)
            
        elif plot_type == 'smile':
            # Volatility smile plots for each expiry
            grouped = data.groupby('expiry')
            
            # Determine grid layout
            n_expiries = len(grouped)
            cols = min(3, n_expiries)
            rows = (n_expiries + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1 or cols == 1:
                axes = axes.reshape(rows, cols)
            
            # Plot each expiry
            for i, (expiry, group) in enumerate(grouped):
                row, col = i // cols, i % cols
                ax = axes[row, col]
                
                # Sort by strike/moneyness
                group = group.sort_values(x_col)
                
                # Plot market data
                ax.plot(
                    group[x_col], 
                    group['implied_volatility'],
                    'o',
                    markersize=6,
                    label='Market IV'
                )
                
                # Plot fitted curve if provided
                if fit_results is not None and expiry in fit_results:
                    x_fit = np.linspace(group[x_col].min(), group[x_col].max(), 100)
                    S_val = None if x_col == 'strike' else 1.0  # dummy value for moneyness
                    y_fit = self.predict_volatility(fit_results, x_fit, expiry, S_val)
                    
                    ax.plot(
                        x_fit,
                        y_fit,
                        '-',
                        linewidth=2,
                        label='Fitted IV'
                    )
                
                # Add labels and title
                ax.set_xlabel(x_label)
                ax.set_ylabel('Implied Volatility')
                ax.set_title(f'T = {expiry:.4f}')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            # Remove empty subplots
            for i in range(i + 1, rows * cols):
                row, col = i // cols, i % cols
                fig.delaxes(axes[row, col])
            
            plt.tight_layout()
            plt.suptitle('Volatility Smiles by Expiry', fontsize=16)
            plt.subplots_adjust(top=0.92)
        
        else:
            raise ValueError("Plot type must be one of: '3d', 'contour', 'smile'")
        
        plt.tight_layout()
        plt.show()
    
    def term_structure(self, 
                      surface_data: pd.DataFrame, 
                      K: Optional[float] = None,
                      moneyness: Optional[float] = None) -> pd.DataFrame:
        """
        Calculate volatility term structure for a specific strike or moneyness.
        
        Parameters:
        -----------
        surface_data : pd.DataFrame
            DataFrame with columns for strike, expiry, and implied_volatility
        K : float, optional
            Strike price
        moneyness : float, optional
            Moneyness level (K/S)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with term structure data
        """
        required_cols = ['strike', 'expiry', 'implied_volatility']
        if not all(col in surface_data.columns for col in required_cols):
            raise ValueError(f"Surface data must contain columns: {required_cols}")
        
        if K is None and moneyness is None:
            raise ValueError("Either K or moneyness must be provided")
        
        # Create a copy to avoid modifying the original
        data = surface_data.copy()
        
        # Filter data based on strike or moneyness
        if moneyness is not None:
            if 'S' not in data.columns:
                raise ValueError("Surface data must contain 'S' column when using moneyness")
            
            data['moneyness'] = data['strike'] / data['S']
            
            # Group by expiry and find closest moneyness level
            result = []
            for expiry, group in data.groupby('expiry'):
                idx = (group['moneyness'] - moneyness).abs().idxmin()
                result.append(group.loc[idx])
            
            term_data = pd.DataFrame(result)
            term_data = term_data.sort_values('expiry')
            
        else:  # Use strike
            # Group by expiry and find closest strike
            result = []
            for expiry, group in data.groupby('expiry'):
                idx = (group['strike'] - K).abs().idxmin()
                result.append(group.loc[idx])
            
            term_data = pd.DataFrame(result)
            term_data = term_data.sort_values('expiry')
        
        return term_data
    
    def plot_term_structure(self, 
                           term_data: pd.DataFrame,
                           title: Optional[str] = None) -> None:
        """
        Plot volatility term structure.
        
        Parameters:
        -----------
        term_data : pd.DataFrame
            DataFrame with term structure data from term_structure method
        title : str, optional
            Plot title
        """
        required_cols = ['expiry', 'implied_volatility']
        if not all(col in term_data.columns for col in required_cols):
            raise ValueError(f"Term data must contain columns: {required_cols}")
        
        plt.figure(figsize=(10, 6))
        
        # Plot term structure
        plt.plot(
            term_data['expiry'],
            term_data['implied_volatility'],
            'o-',
            markersize=8,
            linewidth=2
        )
        
        # Horizontal line at ATM volatility
        atm_idx = abs(term_data.get('moneyness', term_data.get('strike')) - 1.0).idxmin()
        atm_vol = term_data.loc[atm_idx, 'implied_volatility']
        plt.axhline(y=atm_vol, color='red', linestyle='--', alpha=0.5, label=f'ATM Vol: {atm_vol:.1%}')
        
        # Add labels and title
        plt.xlabel('Time to Expiry (years)')
        plt.ylabel('Implied Volatility')
        
        if title is None:
            if 'moneyness' in term_data.columns:
                moneyness = term_data['moneyness'].mean()
                title = f'Volatility Term Structure (Moneyness = {moneyness:.2f})'
            else:
                strike = term_data['strike'].mean()
                title = f'Volatility Term Structure (Strike = {strike:.2f})'
        
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create implied volatility calculator
    iv_calc = ImpliedVolatility(risk_free_rate=0.03, dividend_yield=0.01)
    
    # Calculate implied volatility for a single option
    S = 100          # Current stock price
    K = 100          # Strike price
    T = 1.0          # Time to expiration (years)
    market_price = 10.5  # Market price of call option
    
    iv = iv_calc.calc_implied_volatility(market_price, S, K, T, 'call')
    print(f"Implied Volatility: {iv:.2%}")
    
    # Create sample data for volatility surface
    strikes = np.linspace(80, 120, 9)
    expiries = np.array([0.25, 0.5, 0.75, 1.0])
    
    surface_data = []
    
    for K in strikes:
        for T in expiries:
            # Generate simulated market data with a volatility smile pattern
            atm_vol = 0.2
            otm_skew = 0.02 * abs(K/S - 1) * np.sign(K - S)  # skew component (higher for OTM options)
            term_effect = 0.05 * np.sqrt(T)  # term structure effect
            iv = atm_vol + otm_skew + term_effect
            
            # Calculate option price using this volatility
            price = iv_calc.black_scholes_price(S, K, T, iv, 'call')
            
            surface_data.append({
                'strike': K,
                'expiry': T,
                'call_price': price,
                'S': S,
                'implied_volatility': iv  # we know the true IV here
            })
    
    # Convert to DataFrame
    surface_df = pd.DataFrame(surface_data)
    
    # In a real application, we would calculate IV from market prices:
    # surface_df = iv_calc.calc_volatility_surface(surface_df, S)
    
    # Fit volatility surface model
    # fit_results = iv_calc.fit_volatility_surface(surface_df, model='svi', moneyness=True)
    
    # Print sample of surface data
    print("\nImplied Volatility Surface Sample:")
    print(surface_df.head())
    
    # Visual plots are commented out for non-interactive environments
    # iv_calc.plot_volatility_surface(surface_df, fit_results, plot_type='3d')
    # iv_calc.plot_volatility_surface(surface_df, fit_results, plot_type='smile')
    
    # term_data = iv_calc.term_structure(surface_df, moneyness=1.0)
    # iv_calc.plot_term_structure(term_data) 