import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import griddata, interp2d, Rbf
import scipy.stats as stats

class ImpliedVolatility:
    """Implementation of implied volatility modeling and analysis."""
    
    def __init__(self):
        """Initialize the implied volatility models."""
        self.volatility_surfaces = {}
        self.models = {}
        self.skew_data = {}
        
    def calculate_implied_vol(self, option_price: float, S: float, K: float, T: float, 
                            r: float, q: float = 0.0, option_type: str = 'call', 
                            precision: float = 0.00001, max_iterations: int = 100) -> float:
        """
        Calculate implied volatility using the Newton-Raphson method.
        
        Args:
            option_price: Market price of the option
            S: Current price of the underlying asset
            K: Strike price
            T: Time to expiration in years
            r: Risk-free interest rate
            q: Dividend yield
            option_type: Option type ('call' or 'put')
            precision: Desired precision
            max_iterations: Maximum number of iterations
            
        Returns:
            Implied volatility
        """
        # Initial volatility estimate
        vol = 0.3
        
        for i in range(max_iterations):
            # Calculate option price with current volatility
            if option_type.lower() == 'call':
                price = self._black_scholes_call(S, K, T, r, q, vol)
                vega = self._black_scholes_vega(S, K, T, r, q, vol)
            else:
                price = self._black_scholes_put(S, K, T, r, q, vol)
                vega = self._black_scholes_vega(S, K, T, r, q, vol)
                
            # Calculate difference
            diff = option_price - price
            
            # Check if precision is reached
            if abs(diff) < precision:
                return vol
                
            # Update volatility estimate
            vol = vol + diff / vega
            
            # Ensure volatility stays within reasonable bounds
            if vol <= 0:
                vol = 0.001
            elif vol > 5:
                return np.nan  # Return NaN for unreasonable values
                
        # Return best estimate if max iterations reached
        return vol
        
    def build_volatility_surface(self, option_chain: pd.DataFrame, 
                               surface_date: str = None) -> pd.DataFrame:
        """
        Build implied volatility surface from option chain data.
        
        Args:
            option_chain: DataFrame with option data
            surface_date: Date identifier for the volatility surface
            
        Returns:
            DataFrame with calculated implied volatilities
        """
        if surface_date is None:
            surface_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            
        # Ensure required columns exist
        required_cols = ['strike', 'expiration', 'price', 'option_type']
        missing_cols = [col for col in required_cols if col not in option_chain.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Get current price and risk-free rate
        S = option_chain['underlying_price'].iloc[0] if 'underlying_price' in option_chain.columns else 100
        r = option_chain['risk_free_rate'].iloc[0] if 'risk_free_rate' in option_chain.columns else 0.01
        q = option_chain['dividend_yield'].iloc[0] if 'dividend_yield' in option_chain.columns else 0
        
        # Convert expiration to years to maturity
        if 'time_to_expiry' not in option_chain.columns:
            if pd.api.types.is_datetime64_any_dtype(option_chain['expiration']):
                today = pd.Timestamp.now().floor('D')
                option_chain['time_to_expiry'] = (option_chain['expiration'] - today).dt.days / 365
            else:
                # Assume expiration is already in years
                option_chain['time_to_expiry'] = option_chain['expiration']
        
        # Calculate implied volatility for each option
        implied_vols = []
        
        for _, row in option_chain.iterrows():
            K = row['strike']
            T = row['time_to_expiry']
            price = row['price']
            option_type = row['option_type'].lower()
            
            try:
                iv = self.calculate_implied_vol(price, S, K, T, r, q, option_type)
                
                if not np.isnan(iv):
                    implied_vols.append({
                        'strike': K,
                        'time_to_expiry': T,
                        'implied_vol': iv,
                        'option_type': option_type,
                        'moneyness': K / S
                    })
            except Exception as e:
                # Skip options that fail to converge
                pass
                
        # Create DataFrame
        iv_surface = pd.DataFrame(implied_vols)
        
        # Store surface
        self.volatility_surfaces[surface_date] = iv_surface
        
        return iv_surface
        
    def fit_surface_model(self, surface_date: str, 
                       model_type: str = 'svi', 
                       params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Fit volatility surface model.
        
        Args:
            surface_date: Date identifier for the volatility surface
            model_type: Type of model to fit ('svi', 'quadratic', 'rbf')
            params: Additional parameters for the model
            
        Returns:
            Dictionary with model parameters and statistics
        """
        if surface_date not in self.volatility_surfaces:
            return {'error': 'Volatility surface not found'}
            
        iv_surface = self.volatility_surfaces[surface_date]
        
        if model_type.lower() == 'svi':
            result = self._fit_svi_model(iv_surface)
        elif model_type.lower() == 'quadratic':
            result = self._fit_quadratic_model(iv_surface)
        elif model_type.lower() == 'rbf':
            result = self._fit_rbf_model(iv_surface)
        else:
            return {'error': 'Unsupported model type'}
            
        # Store model
        self.models[surface_date] = {
            'type': model_type,
            'parameters': result
        }
        
        return result
        
    def interpolate_volatility(self, surface_date: str, strike: float, 
                            time_to_expiry: float) -> float:
        """
        Interpolate implied volatility at a specific strike and maturity.
        
        Args:
            surface_date: Date identifier for the volatility surface
            strike: Option strike price
            time_to_expiry: Time to expiry in years
            
        Returns:
            Interpolated implied volatility
        """
        if surface_date not in self.volatility_surfaces:
            return np.nan
            
        iv_surface = self.volatility_surfaces[surface_date]
        
        # If the exact point exists, return it
        exact_match = iv_surface[(iv_surface['strike'] == strike) & 
                              (iv_surface['time_to_expiry'] == time_to_expiry)]
        if not exact_match.empty:
            return exact_match['implied_vol'].iloc[0]
            
        # Get model if it exists
        if surface_date in self.models:
            model = self.models[surface_date]
            
            # Use model-specific interpolation
            if model['type'] == 'rbf':
                rbf_model = model['parameters']['model']
                S = iv_surface['strike'].iloc[0] / iv_surface['moneyness'].iloc[0]  # Extract underlying price
                moneyness = strike / S
                return float(rbf_model(np.array([[moneyness, time_to_expiry]])))
            elif model['type'] == 'quadratic':
                params = model['parameters']['parameters']
                S = iv_surface['strike'].iloc[0] / iv_surface['moneyness'].iloc[0]
                moneyness = strike / S
                a, b, c, d, e, f = params
                return float(a + b*moneyness + c*time_to_expiry + d*moneyness*moneyness + 
                             e*time_to_expiry*time_to_expiry + f*moneyness*time_to_expiry)
                
        # Fallback to basic grid interpolation
        points = iv_surface[['strike', 'time_to_expiry']].values
        values = iv_surface['implied_vol'].values
        
        try:
            # Use griddata for interpolation
            interpolated_vol = griddata(points, values, np.array([[strike, time_to_expiry]]), 
                                    method='linear')[0]
            
            # If interpolation fails (returns NaN), try nearest method
            if np.isnan(interpolated_vol):
                interpolated_vol = griddata(points, values, np.array([[strike, time_to_expiry]]), 
                                        method='nearest')[0]
                                        
            return float(interpolated_vol)
        except Exception as e:
            return np.nan
            
    def calculate_volatility_skew(self, surface_date: str, 
                               expiry: float = None) -> pd.DataFrame:
        """
        Calculate volatility skew for a specific expiry.
        
        Args:
            surface_date: Date identifier for the volatility surface
            expiry: Time to expiry in years (if None, use first available)
            
        Returns:
            DataFrame with volatility skew data
        """
        if surface_date not in self.volatility_surfaces:
            return pd.DataFrame()
            
        iv_surface = self.volatility_surfaces[surface_date]
        
        # Filter by expiry
        if expiry is None:
            expiry = iv_surface['time_to_expiry'].min()
            
        expiry_data = iv_surface[np.isclose(iv_surface['time_to_expiry'], expiry)]
        
        if expiry_data.empty:
            return pd.DataFrame()
            
        # Sort by moneyness or strike
        skew_data = expiry_data.sort_values('moneyness')
        
        # Calculate skew metrics
        S = skew_data['strike'].iloc[0] / skew_data['moneyness'].iloc[0]  # Extract underlying price
        atm_idx = np.abs(skew_data['moneyness'] - 1.0).argmin()
        atm_vol = skew_data['implied_vol'].iloc[atm_idx]
        
        skew_data['vol_difference'] = skew_data['implied_vol'] - atm_vol
        
        # Store skew data
        self.skew_data[f"{surface_date}_{expiry}"] = skew_data
        
        return skew_data
        
    def calculate_term_structure(self, surface_date: str, 
                              moneyness: float = 1.0) -> pd.DataFrame:
        """
        Calculate volatility term structure for a specific moneyness level.
        
        Args:
            surface_date: Date identifier for the volatility surface
            moneyness: Moneyness level (default is at-the-money)
            
        Returns:
            DataFrame with term structure data
        """
        if surface_date not in self.volatility_surfaces:
            return pd.DataFrame()
            
        iv_surface = self.volatility_surfaces[surface_date]
        
        # Find the closest moneyness level for each expiry
        expirations = iv_surface['time_to_expiry'].unique()
        term_data = []
        
        for expiry in sorted(expirations):
            expiry_data = iv_surface[np.isclose(iv_surface['time_to_expiry'], expiry)]
            money_idx = np.abs(expiry_data['moneyness'] - moneyness).argmin()
            closest_vol = expiry_data['implied_vol'].iloc[money_idx]
            closest_moneyness = expiry_data['moneyness'].iloc[money_idx]
            
            term_data.append({
                'time_to_expiry': expiry,
                'implied_vol': closest_vol,
                'moneyness': closest_moneyness
            })
            
        return pd.DataFrame(term_data)
        
    def extract_risk_neutral_density(self, surface_date: str, 
                                  expiry: float, 
                                  num_points: int = 50) -> pd.DataFrame:
        """
        Extract risk-neutral probability density from volatility smile.
        
        Args:
            surface_date: Date identifier for the volatility surface
            expiry: Time to expiry in years
            num_points: Number of points for density calculation
            
        Returns:
            DataFrame with risk-neutral density data
        """
        if surface_date not in self.volatility_surfaces:
            return pd.DataFrame()
            
        iv_surface = self.volatility_surfaces[surface_date]
        
        # Filter by expiry
        expiry_data = iv_surface[np.isclose(iv_surface['time_to_expiry'], expiry)]
        
        if expiry_data.empty:
            return pd.DataFrame()
            
        # Get underlying price
        S = expiry_data['strike'].iloc[0] / expiry_data['moneyness'].iloc[0]  # Extract underlying price
        
        # Generate a range of strikes
        min_strike = expiry_data['strike'].min() * 0.9
        max_strike = expiry_data['strike'].max() * 1.1
        strikes = np.linspace(min_strike, max_strike, num_points)
        
        # Create volatility spline or use model to interpolate
        implied_vols = []
        for K in strikes:
            iv = self.interpolate_volatility(surface_date, K, expiry)
            implied_vols.append(iv)
            
        implied_vols = np.array(implied_vols)
        
        # Calculate first and second derivatives (finite differences)
        dvol_dk = np.zeros_like(implied_vols)
        d2vol_dk2 = np.zeros_like(implied_vols)
        
        # First derivative
        dvol_dk[1:-1] = (implied_vols[2:] - implied_vols[:-2]) / (strikes[2:] - strikes[:-2])
        dvol_dk[0] = (implied_vols[1] - implied_vols[0]) / (strikes[1] - strikes[0])
        dvol_dk[-1] = (implied_vols[-1] - implied_vols[-2]) / (strikes[-1] - strikes[-2])
        
        # Second derivative
        for i in range(1, len(strikes)-1):
            d2vol_dk2[i] = (implied_vols[i+1] - 2*implied_vols[i] + implied_vols[i-1]) / ((strikes[i+1] - strikes[i-1])/2)**2
            
        # Risk-neutral density calculation (Breeden-Litzenberger formula)
        r = 0.01  # Assume risk-free rate
        q = 0.0   # Assume no dividends
        
        density = []
        for i, K in enumerate(strikes):
            d1 = (np.log(S/K) + (r - q + 0.5*implied_vols[i]**2)*expiry) / (implied_vols[i]*np.sqrt(expiry))
            d2 = d1 - implied_vols[i]*np.sqrt(expiry)
            
            term1 = np.exp(-r*expiry) / (K*implied_vols[i]*np.sqrt(2*np.pi*expiry)) * np.exp(-d2**2/2)
            term2 = 1 + K*np.sqrt(expiry) * (d1*d2 / implied_vols[i])
            term3 = K*K*expiry * dvol_dk[i] * d1 / implied_vols[i]
            term4 = 0.25 * K*K*expiry*np.sqrt(expiry) * d1*d2 * dvol_dk[i]**2 / implied_vols[i]**2
            term5 = 0.5 * K*K*expiry * d2vol_dk2[i]
            
            # Simplified density function (excluding some higher-order terms)
            rnd = term1 * (term2 + term3 + term4 + term5)
            density.append(max(rnd, 0))  # Ensure non-negative density
            
        # Create result DataFrame
        rnd_df = pd.DataFrame({
            'strike': strikes,
            'implied_vol': implied_vols,
            'density': density
        })
        
        return rnd_df
            
    def plot_volatility_surface(self, surface_date: str, 
                             view_type: str = '3d') -> None:
        """
        Plot volatility surface.
        
        Args:
            surface_date: Date identifier for the volatility surface
            view_type: Type of visualization ('3d', 'heatmap', or 'contour')
            
        Returns:
            None (displays plot)
        """
        if surface_date not in self.volatility_surfaces:
            print(f"Surface {surface_date} not found")
            return
            
        iv_surface = self.volatility_surfaces[surface_date]
        
        # Extract data
        X = iv_surface['moneyness'].values
        Y = iv_surface['time_to_expiry'].values
        Z = iv_surface['implied_vol'].values
        
        if view_type == '3d':
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create scatter plot
            scat = ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=50)
            
            # Label axes
            ax.set_xlabel('Moneyness (K/S)')
            ax.set_ylabel('Time to Expiry (Years)')
            ax.set_zlabel('Implied Volatility')
            
            # Add colorbar
            fig.colorbar(scat, ax=ax, label='Implied Volatility')
            
        elif view_type == 'heatmap':
            # Create grid for heatmap
            unique_moneyness = sorted(iv_surface['moneyness'].unique())
            unique_expiry = sorted(iv_surface['time_to_expiry'].unique())
            
            grid_X, grid_Y = np.meshgrid(unique_moneyness, unique_expiry)
            grid_Z = np.zeros_like(grid_X)
            
            # Fill grid with interpolated values
            for i, expiry in enumerate(unique_expiry):
                for j, moneyness in enumerate(unique_moneyness):
                    matching = iv_surface[(iv_surface['moneyness'] == moneyness) & 
                                      (iv_surface['time_to_expiry'] == expiry)]
                    if not matching.empty:
                        grid_Z[i, j] = matching['implied_vol'].iloc[0]
                    else:
                        # Find closest point
                        idx = ((iv_surface['moneyness'] - moneyness)**2 + 
                              (iv_surface['time_to_expiry'] - expiry)**2).argmin()
                        grid_Z[i, j] = iv_surface['implied_vol'].iloc[idx]
            
            plt.figure(figsize=(12, 8))
            im = plt.pcolormesh(grid_X, grid_Y, grid_Z, cmap='viridis', shading='auto')
            plt.colorbar(im, label='Implied Volatility')
            plt.xlabel('Moneyness (K/S)')
            plt.ylabel('Time to Expiry (Years)')
            
        elif view_type == 'contour':
            # Create grid for contour plot
            unique_moneyness = np.linspace(min(X), max(X), 50)
            unique_expiry = np.linspace(min(Y), max(Y), 50)
            
            grid_X, grid_Y = np.meshgrid(unique_moneyness, unique_expiry)
            grid_Z = griddata((X, Y), Z, (grid_X, grid_Y), method='cubic')
            
            plt.figure(figsize=(12, 8))
            contour = plt.contourf(grid_X, grid_Y, grid_Z, 20, cmap='viridis')
            plt.colorbar(contour, label='Implied Volatility')
            plt.xlabel('Moneyness (K/S)')
            plt.ylabel('Time to Expiry (Years)')
            plt.contour(grid_X, grid_Y, grid_Z, 10, colors='white', linewidths=0.5, alpha=0.7)
            
        plt.title(f'Implied Volatility Surface - {surface_date}')
        plt.tight_layout()
        plt.show()
        
    def plot_volatility_skew(self, surface_date: str, expiry: float = None) -> None:
        """
        Plot volatility skew for a specific expiry.
        
        Args:
            surface_date: Date identifier for the volatility surface
            expiry: Time to expiry in years (if None, use first available)
            
        Returns:
            None (displays plot)
        """
        skew_data = self.calculate_volatility_skew(surface_date, expiry)
        
        if skew_data.empty:
            print("No skew data available")
            return
            
        expiry_val = skew_data['time_to_expiry'].iloc[0]
        
        plt.figure(figsize=(10, 6))
        plt.plot(skew_data['moneyness'], skew_data['implied_vol'], 'o-', linewidth=2)
        plt.axhline(y=skew_data['implied_vol'].mean(), color='r', linestyle='--', 
                  label=f'Mean: {skew_data["implied_vol"].mean():.2f}')
        
        plt.title(f'Volatility Skew - Expiry: {expiry_val:.2f} years')
        plt.xlabel('Moneyness (K/S)')
        plt.ylabel('Implied Volatility')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def plot_term_structure(self, surface_date: str, moneyness: float = 1.0) -> None:
        """
        Plot volatility term structure for a specific moneyness level.
        
        Args:
            surface_date: Date identifier for the volatility surface
            moneyness: Moneyness level (default is at-the-money)
            
        Returns:
            None (displays plot)
        """
        term_data = self.calculate_term_structure(surface_date, moneyness)
        
        if term_data.empty:
            print("No term structure data available")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(term_data['time_to_expiry'], term_data['implied_vol'], 'o-', linewidth=2)
        
        plt.title(f'Volatility Term Structure - Moneyness: {moneyness:.2f}')
        plt.xlabel('Time to Expiry (Years)')
        plt.ylabel('Implied Volatility')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def _black_scholes_call(self, S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """Calculate Black-Scholes call option price."""
        d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        
    def _black_scholes_put(self, S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """Calculate Black-Scholes put option price."""
        d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)
        
    def _black_scholes_vega(self, S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """Calculate Black-Scholes vega (sensitivity to volatility)."""
        d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.exp(-q * T) * np.sqrt(T) * stats.norm.pdf(d1)
        
    def _fit_svi_model(self, iv_surface: pd.DataFrame) -> Dict[str, Any]:
        """Fit SVI (Stochastic Volatility Inspired) model to IV surface."""
        # Group by expiry
        expirations = iv_surface['time_to_expiry'].unique()
        svi_params = {}
        
        # SVI parameterization: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
        # where k is log-moneyness
        
        for expiry in expirations:
            expiry_data = iv_surface[np.isclose(iv_surface['time_to_expiry'], expiry)]
            
            if len(expiry_data) >= 5:  # Need enough data points
                # Log-moneyness
                log_moneyness = np.log(expiry_data['moneyness'])
                var = expiry_data['implied_vol']**2  # Total variance
                
                # Initial parameter guess
                initial_params = [0.1, 0.1, 0, 0, 0.1]  # [a, b, rho, m, sigma]
                
                # Define objective function
                def svi_objective(params):
                    a, b, rho, m, sigma = params
                    # Ensure constraints
                    if abs(rho) >= 1 or b < 0 or sigma <= 0:
                        return 1e10
                    
                    svi_var = a + b * (rho * (log_moneyness - m) + 
                                     np.sqrt((log_moneyness - m)**2 + sigma**2))
                    return np.sum((var - svi_var)**2)
                
                # Optimize
                try:
                    result = minimize(svi_objective, initial_params, method='Nelder-Mead')
                    svi_params[expiry] = result.x
                except Exception as e:
                    print(f"Optimization failed for expiry {expiry}: {str(e)}")
                    continue
        
        return {
            'svi_parameters': svi_params,
            'model_type': 'svi'
        }
        
    def _fit_quadratic_model(self, iv_surface: pd.DataFrame) -> Dict[str, Any]:
        """Fit quadratic model to IV surface."""
        # z = a + b*x + c*y + d*x^2 + e*y^2 + f*x*y
        X = iv_surface['moneyness'].values
        Y = iv_surface['time_to_expiry'].values
        Z = iv_surface['implied_vol'].values
        
        # Design matrix
        A = np.column_stack([
            np.ones_like(X),
            X,
            Y,
            X**2,
            Y**2,
            X*Y
        ])
        
        # Linear least squares solution
        params, residuals, _, _ = np.linalg.lstsq(A, Z, rcond=None)
        
        # Calculate R-squared
        z_mean = np.mean(Z)
        ss_tot = np.sum((Z - z_mean)**2)
        ss_res = residuals[0] if len(residuals) > 0 else np.sum((Z - np.dot(A, params))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'parameters': params,
            'r_squared': r_squared,
            'model_type': 'quadratic'
        }
        
    def _fit_rbf_model(self, iv_surface: pd.DataFrame) -> Dict[str, Any]:
        """Fit Radial Basis Function (RBF) model to IV surface."""
        X = iv_surface['moneyness'].values
        Y = iv_surface['time_to_expiry'].values
        Z = iv_surface['implied_vol'].values
        
        # Combine input data
        points = np.column_stack([X, Y])
        
        # Fit RBF
        rbf = Rbf(X, Y, Z, function='multiquadric', epsilon=1)
        
        # Evaluate on a grid to measure performance
        grid_x = np.linspace(min(X), max(X), 20)
        grid_y = np.linspace(min(Y), max(Y), 20)
        grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
        grid_Z_true = griddata((X, Y), Z, (grid_X, grid_Y), method='linear')
        grid_Z_pred = rbf(grid_X, grid_Y)
        
        # Calculate error metrics where we have true values
        mask = ~np.isnan(grid_Z_true)
        mse = np.mean((grid_Z_true[mask] - grid_Z_pred[mask])**2)
        rmse = np.sqrt(mse)
        
        return {
            'model': rbf,
            'rmse': rmse,
            'model_type': 'rbf'
        } 