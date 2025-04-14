import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize

class VolatilityTermStructure:
    """Implementation of volatility term structure analysis and modeling."""
    
    def __init__(self):
        """Initialize the volatility term structure analyzer."""
        self.term_structures = {}
        self.models = {}
        self.forecasts = {}
        
    def build_term_structure(self, volatilities: Dict[int, float], 
                           reference_date: str = None) -> pd.DataFrame:
        """
        Build a volatility term structure from observed volatilities.
        
        Args:
            volatilities: Dictionary mapping tenors (in days) to volatility values
            reference_date: Reference date for the term structure
            
        Returns:
            DataFrame with term structure data
        """
        if not volatilities:
            return pd.DataFrame()
            
        # Convert dictionary to dataframe
        tenors = sorted(list(volatilities.keys()))
        vols = [volatilities[tenor] for tenor in tenors]
        
        term_structure = pd.DataFrame({
            'tenor': tenors,
            'volatility': vols
        })
        
        # Add reference date if provided
        if reference_date:
            term_structure['reference_date'] = reference_date
            
        # Store the term structure
        structure_id = reference_date if reference_date else 'latest'
        self.term_structures[structure_id] = term_structure
        
        return term_structure
        
    def from_option_chain(self, option_chain: pd.DataFrame, 
                        underlying_price: float,
                        risk_free_rates: Dict[int, float] = None,
                        reference_date: str = None) -> pd.DataFrame:
        """
        Extract volatility term structure from option chain data.
        
        Args:
            option_chain: DataFrame with option chain data
            underlying_price: Current price of the underlying asset
            risk_free_rates: Dictionary mapping tenors to risk-free rates
            reference_date: Reference date for the term structure
            
        Returns:
            DataFrame with term structure data
        """
        if option_chain.empty:
            return pd.DataFrame()
            
        # Validate that required columns exist
        required_cols = ['expiry_days', 'strike', 'option_type', 'price']
        missing_cols = [col for col in required_cols if col not in option_chain.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Default risk-free rate if not provided
        if risk_free_rates is None:
            risk_free_rates = {tenor: 0.02 for tenor in option_chain['expiry_days'].unique()}
            
        # Group by expiration
        grouped = option_chain.groupby('expiry_days')
        
        # Extract ATM volatility for each tenor
        volatilities = {}
        
        for tenor, group in grouped:
            # Find ATM options (closest to underlying price)
            group['moneyness'] = np.abs(group['strike'] / underlying_price - 1.0)
            atm_options = group.nsmallest(2, 'moneyness')
            
            if len(atm_options) > 0:
                # Calculate implied volatility
                if 'implied_vol' in atm_options.columns:
                    # Use pre-calculated implied vol if available
                    atm_vol = atm_options['implied_vol'].mean()
                else:
                    # For simplicity, assuming implied vol is calculated elsewhere
                    # In practice, you would use a model to calculate it here
                    atm_vol = np.nan
                    
                if not np.isnan(atm_vol):
                    volatilities[int(tenor)] = atm_vol
                    
        # Build term structure
        return self.build_term_structure(volatilities, reference_date)
        
    def fit_curve(self, term_structure_id: str = 'latest', 
               model_type: str = 'cubic_spline',
               tenor_range: List[int] = None) -> Dict[str, Any]:
        """
        Fit a curve to the volatility term structure.
        
        Args:
            term_structure_id: Identifier for the term structure to use
            model_type: Type of curve fitting model
            tenor_range: Range of tenors to evaluate the fitted curve
            
        Returns:
            Dictionary with fitted curve data
        """
        if term_structure_id not in self.term_structures:
            return {'error': 'Term structure not found'}
            
        term_structure = self.term_structures[term_structure_id]
        
        if term_structure.empty or len(term_structure) < 2:
            return {'error': 'Insufficient data points for curve fitting'}
            
        # Get tenors and volatilities
        tenors = term_structure['tenor'].values
        vols = term_structure['volatility'].values
        
        if model_type == 'cubic_spline':
            # Fit cubic spline
            cs = CubicSpline(tenors, vols)
            model = {'type': 'cubic_spline', 'model': cs}
            
        elif model_type == 'nelson_siegel':
            # Fit Nelson-Siegel model
            params = self._fit_nelson_siegel(tenors, vols)
            model = {'type': 'nelson_siegel', 'params': params}
            
        elif model_type == 'svi':
            # Fit SVI (Stochastic Volatility Inspired) model
            params = self._fit_svi(tenors, vols)
            model = {'type': 'svi', 'params': params}
            
        else:
            # Default to linear interpolation
            interp = interp1d(tenors, vols, bounds_error=False, fill_value='extrapolate')
            model = {'type': 'linear', 'model': interp}
            
        # Store the model
        self.models[term_structure_id] = model
        
        # Evaluate the model across tenor range if provided
        if tenor_range is None:
            # Default: evaluate at 100 points between min and max tenor
            tenor_range = np.linspace(min(tenors), max(tenors), 100)
            
        # Generate fitted values
        fitted_vols = self._evaluate_model(model, tenor_range)
        
        # Create result dataframe
        fitted_curve = pd.DataFrame({
            'tenor': tenor_range,
            'fitted_volatility': fitted_vols
        })
        
        return {
            'term_structure_id': term_structure_id,
            'model_type': model_type,
            'fitted_curve': fitted_curve,
            'model': model
        }
        
    def forecast_term_structure(self, current_term_structure_id: str,
                             target_date: str,
                             method: str = 'parallel_shift',
                             shift_params: Dict[str, float] = None) -> pd.DataFrame:
        """
        Forecast volatility term structure for a future date.
        
        Args:
            current_term_structure_id: Identifier for the current term structure
            target_date: Target date for forecasting
            method: Forecasting method to use
            shift_params: Parameters for the shift model
            
        Returns:
            DataFrame with forecasted term structure
        """
        if current_term_structure_id not in self.term_structures:
            return pd.DataFrame()
            
        current_ts = self.term_structures[current_term_structure_id]
        
        if current_ts.empty:
            return pd.DataFrame()
            
        # Apply forecast method
        if method == 'parallel_shift':
            # Simple parallel shift of the term structure
            if shift_params is None:
                shift_params = {'shift': 0.0}
                
            shift = shift_params.get('shift', 0.0)
            forecast_ts = current_ts.copy()
            forecast_ts['volatility'] = forecast_ts['volatility'] + shift
            
        elif method == 'mean_reversion':
            # Mean reversion model
            if shift_params is None:
                shift_params = {'target_level': 0.2, 'speed': 0.1}
                
            target = shift_params.get('target_level', 0.2)
            speed = shift_params.get('speed', 0.1)
            
            forecast_ts = current_ts.copy()
            forecast_ts['volatility'] = target + (forecast_ts['volatility'] - target) * np.exp(-speed)
            
        else:
            # Default to no change
            forecast_ts = current_ts.copy()
            
        # Store the forecast
        self.forecasts[target_date] = forecast_ts
        
        return forecast_ts
        
    def calculate_term_premium(self, term_structure_id: str = 'latest', 
                            reference_tenor: int = 30) -> pd.DataFrame:
        """
        Calculate volatility term premium relative to a reference tenor.
        
        Args:
            term_structure_id: Identifier for the term structure to use
            reference_tenor: Reference tenor for premium calculation
            
        Returns:
            DataFrame with term premium data
        """
        if term_structure_id not in self.term_structures:
            return pd.DataFrame()
            
        term_structure = self.term_structures[term_structure_id]
        
        if term_structure.empty:
            return pd.DataFrame()
            
        # Find reference volatility
        ref_vol = None
        if reference_tenor in term_structure['tenor'].values:
            ref_vol = term_structure.loc[term_structure['tenor'] == reference_tenor, 'volatility'].iloc[0]
        else:
            # Interpolate if reference tenor not directly available
            if term_structure_id in self.models:
                model = self.models[term_structure_id]
                ref_vol = self._evaluate_model(model, [reference_tenor])[0]
            else:
                # Fit a simple model to get the reference value
                interp = interp1d(term_structure['tenor'].values, term_structure['volatility'].values,
                                bounds_error=False, fill_value='extrapolate')
                ref_vol = interp(reference_tenor)
                
        # Calculate term premium
        result = term_structure.copy()
        result['reference_tenor'] = reference_tenor
        result['reference_vol'] = ref_vol
        result['term_premium'] = result['volatility'] - ref_vol
        result['relative_premium'] = result['volatility'] / ref_vol - 1.0
        
        return result
        
    def extract_forward_volatility(self, term_structure_id: str = 'latest', 
                               start_tenor: int = 30, 
                               end_tenor: int = 90) -> float:
        """
        Extract implied forward volatility between two tenors.
        
        Args:
            term_structure_id: Identifier for the term structure to use
            start_tenor: Starting tenor for forward volatility
            end_tenor: Ending tenor for forward volatility
            
        Returns:
            Forward volatility value
        """
        if term_structure_id not in self.term_structures:
            return np.nan
            
        term_structure = self.term_structures[term_structure_id]
        
        if term_structure.empty:
            return np.nan
            
        # Get volatilities for the tenors
        start_vol = None
        end_vol = None
        
        # Check if exact tenors exist in the data
        if start_tenor in term_structure['tenor'].values:
            start_vol = term_structure.loc[term_structure['tenor'] == start_tenor, 'volatility'].iloc[0]
        
        if end_tenor in term_structure['tenor'].values:
            end_vol = term_structure.loc[term_structure['tenor'] == end_tenor, 'volatility'].iloc[0]
        
        # If not, use fitted model to interpolate
        if start_vol is None or end_vol is None:
            if term_structure_id in self.models:
                model = self.models[term_structure_id]
                tenors = [start_tenor, end_tenor]
                vols = self._evaluate_model(model, tenors)
                start_vol, end_vol = vols
            else:
                # Fit a simple model to get the values
                interp = interp1d(term_structure['tenor'].values, term_structure['volatility'].values,
                                bounds_error=False, fill_value='extrapolate')
                start_vol = interp(start_tenor)
                end_vol = interp(end_tenor)
                
        # Calculate forward volatility
        # Using the formula: σf²(T2-T1) = σ2²T2 - σ1²T1
        forward_variance = (end_vol**2 * end_tenor - start_vol**2 * start_tenor) / (end_tenor - start_tenor)
        forward_vol = np.sqrt(max(0, forward_variance))
        
        return forward_vol
        
    def compare_term_structures(self, ts_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple volatility term structures.
        
        Args:
            ts_ids: List of term structure identifiers to compare
            
        Returns:
            DataFrame with comparative term structure data
        """
        if not ts_ids or not all(ts_id in self.term_structures for ts_id in ts_ids):
            return pd.DataFrame()
            
        # Find common tenors across all term structures
        common_tenors = set()
        for ts_id in ts_ids:
            common_tenors.update(self.term_structures[ts_id]['tenor'].values)
            
        common_tenors = sorted(list(common_tenors))
        
        # Create comparison dataframe
        comparison_data = {'tenor': common_tenors}
        
        for ts_id in ts_ids:
            term_structure = self.term_structures[ts_id]
            
            # If model exists, use it to interpolate at all tenors
            if ts_id in self.models:
                model = self.models[ts_id]
                vols = self._evaluate_model(model, common_tenors)
                comparison_data[f'vol_{ts_id}'] = vols
            else:
                # Use simple interpolation
                interp = interp1d(term_structure['tenor'].values, term_structure['volatility'].values,
                               bounds_error=False, fill_value='extrapolate')
                comparison_data[f'vol_{ts_id}'] = [interp(tenor) for tenor in common_tenors]
                
        return pd.DataFrame(comparison_data)
        
    def plot_term_structure(self, term_structure_id: str = 'latest', 
                         show_fitted: bool = True) -> None:
        """
        Plot volatility term structure.
        
        Args:
            term_structure_id: Identifier for the term structure to plot
            show_fitted: Whether to show the fitted curve
            
        Returns:
            None (displays plot)
        """
        if term_structure_id not in self.term_structures:
            print(f"Term structure {term_structure_id} not found")
            return
            
        term_structure = self.term_structures[term_structure_id]
        
        plt.figure(figsize=(10, 6))
        
        # Plot observed volatilities
        plt.scatter(term_structure['tenor'], term_structure['volatility'], 
                  color='blue', label='Observed', s=50, zorder=5)
        
        # Plot fitted curve if available and requested
        if show_fitted and term_structure_id in self.models:
            model = self.models[term_structure_id]
            
            # Generate smooth curve
            tenors = np.linspace(min(term_structure['tenor']), 
                              max(term_structure['tenor']), 
                              100)
            vols = self._evaluate_model(model, tenors)
            
            plt.plot(tenors, vols, 'r-', label=f"Fitted ({model['type']})", linewidth=2)
            
        plt.xlabel('Tenor (Days)')
        plt.ylabel('Volatility')
        plt.title(f'Volatility Term Structure - {term_structure_id}')
        plt.grid(True)
        plt.legend()
        
        # Add x-axis ticks at key tenors
        key_tenors = [1, 7, 30, 90, 180, 365]
        key_tenors = [t for t in key_tenors if t <= max(term_structure['tenor'])]
        plt.xticks(key_tenors)
        
        plt.tight_layout()
        plt.show()
        
    def plot_term_premium(self, term_structure_id: str = 'latest', 
                       reference_tenor: int = 30) -> None:
        """
        Plot volatility term premium.
        
        Args:
            term_structure_id: Identifier for the term structure to use
            reference_tenor: Reference tenor for premium calculation
            
        Returns:
            None (displays plot)
        """
        premium_data = self.calculate_term_premium(term_structure_id, reference_tenor)
        
        if premium_data.empty:
            print(f"Term structure {term_structure_id} not found or premium calculation failed")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Create subplot for absolute premium
        plt.subplot(2, 1, 1)
        plt.plot(premium_data['tenor'], premium_data['term_premium'], 'b-o')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)
        plt.xlabel('Tenor (Days)')
        plt.ylabel('Absolute Premium')
        plt.title(f'Volatility Term Premium (Reference: {reference_tenor} days)')
        plt.grid(True)
        
        # Create subplot for relative premium
        plt.subplot(2, 1, 2)
        plt.plot(premium_data['tenor'], premium_data['relative_premium'] * 100, 'g-o')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)
        plt.xlabel('Tenor (Days)')
        plt.ylabel('Relative Premium (%)')
        plt.title('Relative Volatility Term Premium')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def plot_multiple_term_structures(self, ts_ids: List[str]) -> None:
        """
        Plot multiple volatility term structures for comparison.
        
        Args:
            ts_ids: List of term structure identifiers to compare
            
        Returns:
            None (displays plot)
        """
        comparison_data = self.compare_term_structures(ts_ids)
        
        if comparison_data.empty:
            print("No valid term structures for comparison")
            return
            
        plt.figure(figsize=(12, 6))
        
        for ts_id in ts_ids:
            plt.plot(comparison_data['tenor'], comparison_data[f'vol_{ts_id}'], 
                   '-o', label=ts_id)
            
        plt.xlabel('Tenor (Days)')
        plt.ylabel('Volatility')
        plt.title('Volatility Term Structure Comparison')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def _fit_nelson_siegel(self, tenors: np.ndarray, vols: np.ndarray) -> Dict[str, float]:
        """Fit Nelson-Siegel model to term structure data."""
        # Initial parameter guess
        initial_params = [0.05, 0.05, 0.05, 1.0]
        
        # Define objective function
        def objective(params):
            beta0, beta1, beta2, tau = params
            fitted = beta0 + beta1 * (1 - np.exp(-tenors/tau)) / (tenors/tau) + \
                     beta2 * ((1 - np.exp(-tenors/tau)) / (tenors/tau) - np.exp(-tenors/tau))
            return np.sum((vols - fitted) ** 2)
            
        # Constraints to ensure parameters are positive
        bounds = [(0, None), (None, None), (None, None), (0.1, None)]
        
        # Minimize the objective function
        result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
        
        # Extract optimized parameters
        beta0, beta1, beta2, tau = result.x
        
        return {
            'beta0': beta0,
            'beta1': beta1,
            'beta2': beta2,
            'tau': tau
        }
        
    def _fit_svi(self, tenors: np.ndarray, vols: np.ndarray) -> Dict[str, float]:
        """Fit SVI (Stochastic Volatility Inspired) model to term structure data."""
        # Convert tenors to years for typical SVI parameterization
        tenors_years = tenors / 365.0
        
        # Initial parameter guess (a, b, rho, m, sigma)
        initial_params = [0.04, 0.04, -0.5, 0.0, 0.1]
        
        # Define objective function
        def objective(params):
            a, b, rho, m, sigma = params
            fitted = a + b * (rho * (tenors_years - m) + np.sqrt((tenors_years - m)**2 + sigma**2))
            return np.sum((vols - fitted) ** 2)
            
        # Constraints to ensure valid parameters
        bounds = [(0, None), (0, None), (-1, 1), (None, None), (0.001, None)]
        
        # Minimize the objective function
        result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
        
        # Extract optimized parameters
        a, b, rho, m, sigma = result.x
        
        return {
            'a': a,
            'b': b,
            'rho': rho,
            'm': m,
            'sigma': sigma
        }
        
    def _evaluate_model(self, model: Dict[str, Any], tenors: List[int]) -> np.ndarray:
        """Evaluate fitted model at specified tenors."""
        model_type = model['type']
        
        if model_type == 'cubic_spline' or model_type == 'linear':
            # Direct evaluation of spline or linear interpolation
            return model['model'](tenors)
            
        elif model_type == 'nelson_siegel':
            # Evaluate Nelson-Siegel model
            params = model['params']
            beta0 = params['beta0']
            beta1 = params['beta1']
            beta2 = params['beta2']
            tau = params['tau']
            
            tenors_array = np.array(tenors)
            term = tenors_array / tau
            exp_term = np.exp(-term)
            
            return beta0 + beta1 * (1 - exp_term) / term + \
                   beta2 * ((1 - exp_term) / term - exp_term)
                   
        elif model_type == 'svi':
            # Evaluate SVI model
            params = model['params']
            a = params['a']
            b = params['b']
            rho = params['rho']
            m = params['m']
            sigma = params['sigma']
            
            tenors_years = np.array(tenors) / 365.0
            return a + b * (rho * (tenors_years - m) + np.sqrt((tenors_years - m)**2 + sigma**2))
            
        else:
            # Default to returning the input tenors (no transformation)
            return np.array(tenors) 