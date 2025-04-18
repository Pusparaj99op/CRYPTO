"""
Regression Models Module

This module provides functions for performing various regression analyses
on cryptocurrency data.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

def fit_linear_regression(X: pd.DataFrame,
                        y: pd.Series,
                        add_constant: bool = True) -> Dict[str, Union[float, np.ndarray]]:
    """
    Fit linear regression model.
    
    Args:
        X: Feature matrix
        y: Target variable
        add_constant: Whether to add constant term
        
    Returns:
        Dictionary containing regression results
    """
    if add_constant:
        X = sm.add_constant(X)
    
    model = sm.OLS(y, X)
    results = model.fit()
    
    return {
        'coefficients': results.params,
        'standard_errors': results.bse,
        't_values': results.tvalues,
        'p_values': results.pvalues,
        'r_squared': results.rsquared,
        'adjusted_r_squared': results.rsquared_adj,
        'f_statistic': results.fvalue,
        'f_p_value': results.f_pvalue,
        'residuals': results.resid
    }

def fit_logistic_regression(X: pd.DataFrame,
                          y: pd.Series,
                          add_constant: bool = True) -> Dict[str, Union[float, np.ndarray]]:
    """
    Fit logistic regression model.
    
    Args:
        X: Feature matrix
        y: Binary target variable
        add_constant: Whether to add constant term
        
    Returns:
        Dictionary containing regression results
    """
    if add_constant:
        X = sm.add_constant(X)
    
    model = sm.Logit(y, X)
    results = model.fit()
    
    return {
        'coefficients': results.params,
        'standard_errors': results.bse,
        'z_values': results.tvalues,
        'p_values': results.pvalues,
        'log_likelihood': results.llf,
        'pseudo_r_squared': results.prsquared,
        'aic': results.aic,
        'bic': results.bic
    }

def fit_quantile_regression(X: pd.DataFrame,
                          y: pd.Series,
                          quantile: float = 0.5,
                          add_constant: bool = True) -> Dict[str, Union[float, np.ndarray]]:
    """
    Fit quantile regression model.
    
    Args:
        X: Feature matrix
        y: Target variable
        quantile: Quantile to estimate
        add_constant: Whether to add constant term
        
    Returns:
        Dictionary containing regression results
    """
    if add_constant:
        X = sm.add_constant(X)
    
    model = QuantReg(y, X)
    results = model.fit(q=quantile)
    
    return {
        'coefficients': results.params,
        'standard_errors': results.bse,
        't_values': results.tvalues,
        'p_values': results.pvalues,
        'quantile': quantile,
        'residuals': results.resid
    }

def perform_stepwise_regression(X: pd.DataFrame,
                              y: pd.Series,
                              direction: str = 'both',
                              criterion: str = 'aic') -> Dict[str, Union[float, np.ndarray]]:
    """
    Perform stepwise regression for feature selection.
    
    Args:
        X: Feature matrix
        y: Target variable
        direction: Stepwise direction ('forward', 'backward', 'both')
        criterion: Selection criterion ('aic', 'bic', 'r2')
        
    Returns:
        Dictionary containing regression results
    """
    def calculate_criterion(model, X, y):
        if criterion == 'aic':
            return model.aic
        elif criterion == 'bic':
            return model.bic
        else:
            return model.rsquared
    
    # Initialize
    included = []
    best_criterion = float('inf') if criterion in ['aic', 'bic'] else float('-inf')
    best_model = None
    
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        
        # Forward step
        if direction in ['forward', 'both']:
            for candidate in excluded:
                current = included + [candidate]
                model = sm.OLS(y, sm.add_constant(X[current])).fit()
                current_criterion = calculate_criterion(model, X[current], y)
                
                if (criterion in ['aic', 'bic'] and current_criterion < best_criterion) or \
                   (criterion == 'r2' and current_criterion > best_criterion):
                    best_criterion = current_criterion
                    best_model = model
                    included = current
                    changed = True
        
        # Backward step
        if direction in ['backward', 'both'] and included:
            for candidate in included:
                current = [x for x in included if x != candidate]
                if not current:
                    continue
                model = sm.OLS(y, sm.add_constant(X[current])).fit()
                current_criterion = calculate_criterion(model, X[current], y)
                
                if (criterion in ['aic', 'bic'] and current_criterion < best_criterion) or \
                   (criterion == 'r2' and current_criterion > best_criterion):
                    best_criterion = current_criterion
                    best_model = model
                    included = current
                    changed = True
        
        if not changed:
            break
    
    return {
        'selected_features': included,
        'coefficients': best_model.params,
        'standard_errors': best_model.bse,
        't_values': best_model.tvalues,
        'p_values': best_model.pvalues,
        'r_squared': best_model.rsquared,
        'adjusted_r_squared': best_model.rsquared_adj,
        'aic': best_model.aic,
        'bic': best_model.bic
    }

def calculate_regression_metrics(y_true: pd.Series,
                               y_pred: pd.Series) -> Dict[str, float]:
    """
    Calculate regression model performance metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary containing performance metrics
    """
    return {
        'r2_score': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': np.mean(np.abs(y_true - y_pred)),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    } 