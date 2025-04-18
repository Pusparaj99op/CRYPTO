"""
Inferential Statistics Module

This module provides functions for performing statistical inference and hypothesis testing
on cryptocurrency data.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple
from scipy import stats
from statsmodels.stats.power import tt_ind_solve_power
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

def perform_hypothesis_test(data1: Union[np.ndarray, pd.Series],
                          data2: Union[np.ndarray, pd.Series] = None,
                          test_type: str = 't-test',
                          alternative: str = 'two-sided') -> Dict[str, float]:
    """
    Perform hypothesis testing on the data.
    
    Args:
        data1: First sample data
        data2: Second sample data (for two-sample tests)
        test_type: Type of test ('t-test', 'z-test', 'wilcoxon', 'mann-whitney')
        alternative: Alternative hypothesis ('two-sided', 'less', 'greater')
        
    Returns:
        Dictionary containing test results
    """
    if isinstance(data1, pd.Series):
        data1 = data1.values
    if data2 is not None and isinstance(data2, pd.Series):
        data2 = data2.values
    
    if test_type == 't-test':
        if data2 is None:
            # One-sample t-test
            t_stat, p_val = stats.ttest_1samp(data1, 0)
        else:
            # Two-sample t-test
            t_stat, p_val = stats.ttest_ind(data1, data2)
    elif test_type == 'z-test':
        if data2 is None:
            # One-sample z-test
            z_stat = (np.mean(data1) - 0) / (np.std(data1) / np.sqrt(len(data1)))
            p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            # Two-sample z-test
            z_stat = (np.mean(data1) - np.mean(data2)) / np.sqrt(
                np.var(data1)/len(data1) + np.var(data2)/len(data2)
            )
            p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    elif test_type == 'wilcoxon':
        if data2 is None:
            # One-sample Wilcoxon signed-rank test
            stat, p_val = stats.wilcoxon(data1)
        else:
            # Two-sample Wilcoxon rank-sum test
            stat, p_val = stats.mannwhitneyu(data1, data2)
    
    return {
        'test_statistic': t_stat if test_type in ['t-test', 'z-test'] else stat,
        'p_value': p_val,
        'test_type': test_type,
        'alternative': alternative
    }

def calculate_confidence_intervals(data: Union[np.ndarray, pd.Series],
                                 confidence_level: float = 0.95) -> Dict[str, float]:
    """
    Calculate confidence intervals for the data.
    
    Args:
        data: Input data
        confidence_level: Confidence level (default: 0.95)
        
    Returns:
        Dictionary containing confidence intervals
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    
    # Calculate critical value
    critical_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
    
    # Calculate margin of error
    margin_of_error = critical_value * (std / np.sqrt(n))
    
    return {
        'mean': mean,
        'lower_bound': mean - margin_of_error,
        'upper_bound': mean + margin_of_error,
        'confidence_level': confidence_level
    }

def perform_anova(data: Dict[str, Union[np.ndarray, pd.Series]]) -> Dict[str, float]:
    """
    Perform Analysis of Variance (ANOVA) on multiple groups of data.
    
    Args:
        data: Dictionary of group names and their corresponding data
        
    Returns:
        Dictionary containing ANOVA results
    """
    # Convert data to DataFrame format
    df_list = []
    for group, values in data.items():
        if isinstance(values, pd.Series):
            values = values.values
        group_df = pd.DataFrame({
            'value': values,
            'group': group
        })
        df_list.append(group_df)
    
    df = pd.concat(df_list, ignore_index=True)
    
    # Perform ANOVA
    model = ols('value ~ C(group)', data=df).fit()
    anova_results = anova_lm(model)
    
    return {
        'f_statistic': anova_results['F'][0],
        'p_value': anova_results['PR(>F)'][0],
        'df_between': anova_results['df'][0],
        'df_within': anova_results['df'][1]
    }

def perform_chi_square_test(observed: np.ndarray,
                          expected: np.ndarray = None) -> Dict[str, float]:
    """
    Perform Chi-square test of independence or goodness-of-fit.
    
    Args:
        observed: Observed frequencies
        expected: Expected frequencies (for goodness-of-fit test)
        
    Returns:
        Dictionary containing test results
    """
    if expected is None:
        # Chi-square test of independence
        chi2_stat, p_val, dof, expected = stats.chi2_contingency(observed)
    else:
        # Chi-square goodness-of-fit test
        chi2_stat, p_val = stats.chisquare(observed, expected)
        dof = len(observed) - 1
    
    return {
        'chi2_statistic': chi2_stat,
        'p_value': p_val,
        'degrees_of_freedom': dof,
        'expected_frequencies': expected
    }

def calculate_power_analysis(effect_size: float,
                           alpha: float = 0.05,
                           power: float = 0.8,
                           ratio: float = 1.0) -> Dict[str, float]:
    """
    Calculate required sample size for a given power level.
    
    Args:
        effect_size: Expected effect size
        alpha: Significance level (default: 0.05)
        power: Desired power level (default: 0.8)
        ratio: Ratio of sample sizes (default: 1.0)
        
    Returns:
        Dictionary containing power analysis results
    """
    # Calculate required sample size
    n = tt_ind_solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=ratio
    )
    
    return {
        'required_sample_size': n,
        'effect_size': effect_size,
        'alpha': alpha,
        'power': power,
        'ratio': ratio
    } 