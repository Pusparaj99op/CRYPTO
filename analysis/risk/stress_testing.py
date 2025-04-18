"""
Stress Testing Framework

This module implements a comprehensive stress testing framework for assessing
portfolio risk under extreme market conditions.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional
from datetime import datetime, timedelta

class StressScenario:
    """Class representing a stress testing scenario."""
    
    def __init__(self, name: str, description: str, severity: float):
        """
        Initialize stress scenario.
        
        Args:
            name: Scenario name
            description: Scenario description
            severity: Severity level (0-1)
        """
        self.name = name
        self.description = description
        self.severity = severity
        self.asset_shocks = {}
        
    def add_asset_shock(self, asset: str, shock: float) -> None:
        """
        Add asset price shock to scenario.
        
        Args:
            asset: Asset identifier
            shock: Price shock (percentage change)
        """
        self.asset_shocks[asset] = shock
        
    def get_shock(self, asset: str) -> float:
        """
        Get shock for specific asset.
        
        Args:
            asset: Asset identifier
            
        Returns:
            float: Price shock
        """
        return self.asset_shocks.get(asset, 0.0)

def generate_historical_scenarios(data: pd.DataFrame,
                               window: int = 252,
                               n_scenarios: int = 5) -> List[StressScenario]:
    """
    Generate stress scenarios from historical data.
    
    Args:
        data: DataFrame of historical returns
        window: Lookback window in days
        n_scenarios: Number of scenarios to generate
        
    Returns:
        List[StressScenario]: List of historical stress scenarios
    """
    scenarios = []
    
    # Calculate rolling returns
    rolling_returns = data.rolling(window).sum()
    
    # Find worst periods
    worst_periods = rolling_returns.min().sort_values().head(n_scenarios)
    
    for i, (asset, min_return) in enumerate(worst_periods.items()):
        date = rolling_returns[asset].idxmin()
        scenario = StressScenario(
            name=f"Historical_{i+1}",
            description=f"Worst {window}-day period for {asset} on {date}",
            severity=1.0 - (i / n_scenarios)
        )
        
        # Add shocks for all assets
        for col in data.columns:
            shock = rolling_returns.loc[date, col]
            scenario.add_asset_shock(col, shock)
            
        scenarios.append(scenario)
    
    return scenarios

def generate_hypothetical_scenarios() -> List[StressScenario]:
    """
    Generate hypothetical stress scenarios.
    
    Returns:
        List[StressScenario]: List of hypothetical stress scenarios
    """
    scenarios = [
        StressScenario(
            name="Market_Crash",
            description="Global market crash similar to 2008",
            severity=0.9
        ),
        StressScenario(
            name="Liquidity_Crisis",
            description="Severe liquidity crisis",
            severity=0.8
        ),
        StressScenario(
            name="Volatility_Spike",
            description="Extreme volatility spike",
            severity=0.7
        ),
        StressScenario(
            name="Correlation_Breakdown",
            description="Breakdown of normal correlations",
            severity=0.6
        )
    ]
    
    return scenarios

def perform_stress_test(portfolio: Dict[str, float],
                       scenarios: List[StressScenario],
                       current_prices: Dict[str, float]) -> pd.DataFrame:
    """
    Perform stress test on portfolio.
    
    Args:
        portfolio: Dictionary of asset positions
        scenarios: List of stress scenarios
        current_prices: Dictionary of current asset prices
        
    Returns:
        pd.DataFrame: Stress test results
    """
    results = []
    
    for scenario in scenarios:
        scenario_result = {
            'Scenario': scenario.name,
            'Description': scenario.description,
            'Severity': scenario.severity
        }
        
        # Calculate portfolio value under stress
        stressed_value = 0.0
        for asset, position in portfolio.items():
            shock = scenario.get_shock(asset)
            stressed_price = current_prices[asset] * (1 + shock)
            stressed_value += position * stressed_price
            
        scenario_result['Stressed_Value'] = stressed_value
        scenario_result['Loss'] = sum(position * current_prices[asset] 
                                    for asset, position in portfolio.items()) - stressed_value
        
        results.append(scenario_result)
    
    return pd.DataFrame(results)

def generate_stress_scenarios(data: pd.DataFrame,
                            n_historical: int = 5,
                            include_hypothetical: bool = True) -> List[StressScenario]:
    """
    Generate comprehensive set of stress scenarios.
    
    Args:
        data: DataFrame of historical returns
        n_historical: Number of historical scenarios to generate
        include_hypothetical: Whether to include hypothetical scenarios
        
    Returns:
        List[StressScenario]: Combined list of stress scenarios
    """
    scenarios = generate_historical_scenarios(data, n_scenarios=n_historical)
    
    if include_hypothetical:
        scenarios.extend(generate_hypothetical_scenarios())
    
    return scenarios

def analyze_stress_results(results: pd.DataFrame,
                         portfolio_value: float) -> Dict[str, float]:
    """
    Analyze stress test results.
    
    Args:
        results: DataFrame of stress test results
        portfolio_value: Current portfolio value
        
    Returns:
        Dict[str, float]: Analysis metrics
    """
    analysis = {
        'Max_Loss': results['Loss'].max(),
        'Max_Loss_Pct': results['Loss'].max() / portfolio_value * 100,
        'Avg_Loss': results['Loss'].mean(),
        'Avg_Loss_Pct': results['Loss'].mean() / portfolio_value * 100,
        'Worst_Scenario': results.loc[results['Loss'].idxmax(), 'Scenario'],
        'Severity_Weighted_Loss': (results['Loss'] * results['Severity']).sum() / results['Severity'].sum()
    }
    
    return analysis

def plot_stress_results(results: pd.DataFrame,
                      portfolio_value: float,
                      ax=None) -> None:
    """
    Plot stress test results.
    
    Args:
        results: DataFrame of stress test results
        portfolio_value: Current portfolio value
        ax: Matplotlib axis object (if None, will create new figure)
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort results by loss
    results = results.sort_values('Loss', ascending=False)
    
    # Plot losses
    ax.bar(results['Scenario'], results['Loss'] / portfolio_value * 100)
    ax.set_ylabel('Loss (%)')
    ax.set_title('Stress Test Results')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)
    
    if ax is None:
        plt.tight_layout()
        plt.show() 