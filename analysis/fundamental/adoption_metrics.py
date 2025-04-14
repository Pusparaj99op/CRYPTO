import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats

def analyze_user_growth(address_data, period_days=90):
    """
    Analyze user growth based on address data
    
    Args:
        address_data (pd.DataFrame): DataFrame with date, active_addresses, and new_addresses columns
        period_days (int): Period to analyze in days
        
    Returns:
        dict: User growth metrics
    """
    # Ensure data is sorted by date
    address_data = address_data.sort_values('date')
    
    # Calculate date range
    end_date = address_data['date'].max()
    start_date = end_date - timedelta(days=period_days)
    
    # Filter data for period
    period_data = address_data[address_data['date'] >= start_date]
    
    if period_data.empty:
        return {}
    
    # Calculate basic metrics
    total_new_addresses = period_data['new_addresses'].sum() if 'new_addresses' in period_data.columns else None
    total_active_addresses = period_data['active_addresses'].mean() if 'active_addresses' in period_data.columns else None
    
    # Calculate growth rates
    metrics = {
        'period_days': period_days,
        'total_new_addresses': total_new_addresses,
        'avg_active_addresses': total_active_addresses
    }
    
    # Calculate growth rates if we have enough data
    if len(period_data) > 1:
        # Active addresses growth
        if 'active_addresses' in period_data.columns:
            first_active = period_data['active_addresses'].iloc[0]
            last_active = period_data['active_addresses'].iloc[-1]
            
            if first_active > 0:
                active_growth_rate = (last_active / first_active) - 1
                metrics['active_addresses_growth_rate'] = active_growth_rate
                
                # Annualized growth rate
                days_elapsed = (period_data['date'].iloc[-1] - period_data['date'].iloc[0]).days
                if days_elapsed > 0:
                    annualized_growth = ((1 + active_growth_rate) ** (365 / days_elapsed)) - 1
                    metrics['annualized_active_growth'] = annualized_growth
        
        # New addresses trend
        if 'new_addresses' in period_data.columns:
            # Split into first half and second half
            mid_point = len(period_data) // 2
            first_half_new = period_data['new_addresses'].iloc[:mid_point].sum()
            second_half_new = period_data['new_addresses'].iloc[mid_point:].sum()
            
            if first_half_new > 0:
                new_address_trend = (second_half_new / first_half_new) - 1
                metrics['new_address_trend'] = new_address_trend
    
    # Calculate retention metrics if available
    if 'returning_addresses' in period_data.columns and 'active_addresses' in period_data.columns:
        retention_rates = []
        
        for i in range(1, len(period_data)):
            if period_data['active_addresses'].iloc[i-1] > 0:
                retention_rate = period_data['returning_addresses'].iloc[i] / period_data['active_addresses'].iloc[i-1]
                retention_rates.append(retention_rate)
        
        if retention_rates:
            metrics['avg_retention_rate'] = sum(retention_rates) / len(retention_rates)
    
    return metrics

def analyze_transaction_activity(tx_data, period_days=90):
    """
    Analyze transaction activity
    
    Args:
        tx_data (pd.DataFrame): DataFrame with date, tx_count, tx_volume columns
        period_days (int): Period to analyze in days
        
    Returns:
        dict: Transaction activity metrics
    """
    # Ensure data is sorted by date
    tx_data = tx_data.sort_values('date')
    
    # Calculate date range
    end_date = tx_data['date'].max()
    start_date = end_date - timedelta(days=period_days)
    
    # Filter data for period
    period_data = tx_data[tx_data['date'] >= start_date]
    
    if period_data.empty:
        return {}
    
    # Calculate metrics
    metrics = {
        'period_days': period_days,
        'total_tx_count': period_data['tx_count'].sum(),
        'avg_daily_tx': period_data['tx_count'].mean(),
        'max_daily_tx': period_data['tx_count'].max()
    }
    
    # Add volume metrics if available
    if 'tx_volume' in period_data.columns:
        metrics.update({
            'total_tx_volume': period_data['tx_volume'].sum(),
            'avg_daily_volume': period_data['tx_volume'].mean(),
            'max_daily_volume': period_data['tx_volume'].max()
        })
        
        # Calculate value per transaction
        metrics['avg_value_per_tx'] = period_data['tx_volume'].sum() / period_data['tx_count'].sum()
    
    # Calculate growth metrics
    if len(period_data) > 1:
        # Transaction count growth
        first_tx_count = period_data['tx_count'].iloc[0]
        last_tx_count = period_data['tx_count'].iloc[-1]
        
        if first_tx_count > 0:
            tx_count_growth = (last_tx_count / first_tx_count) - 1
            metrics['tx_count_growth'] = tx_count_growth
        
        # Transaction volume growth if available
        if 'tx_volume' in period_data.columns:
            first_volume = period_data['tx_volume'].iloc[0]
            last_volume = period_data['tx_volume'].iloc[-1]
            
            if first_volume > 0:
                volume_growth = (last_volume / first_volume) - 1
                metrics['tx_volume_growth'] = volume_growth
    
    return metrics

def analyze_merchant_adoption(merchant_data, period_days=90):
    """
    Analyze merchant adoption
    
    Args:
        merchant_data (pd.DataFrame): DataFrame with merchant adoption data
        period_days (int): Period to analyze in days
        
    Returns:
        dict: Merchant adoption metrics
    """
    # Ensure data is sorted by date
    merchant_data = merchant_data.sort_values('date')
    
    # Calculate date range
    end_date = merchant_data['date'].max()
    start_date = end_date - timedelta(days=period_days)
    
    # Filter data for period
    period_data = merchant_data[merchant_data['date'] >= start_date]
    
    if period_data.empty:
        return {}
    
    # Get current numbers
    current_merchants = period_data['merchant_count'].iloc[-1]
    
    # Calculate growth
    if len(period_data) > 1:
        initial_merchants = period_data['merchant_count'].iloc[0]
        if initial_merchants > 0:
            merchant_growth = (current_merchants / initial_merchants) - 1
        else:
            merchant_growth = None
    else:
        merchant_growth = None
    
    # Industry breakdown if available
    industry_breakdown = {}
    if 'industry' in merchant_data.columns:
        industry_counts = period_data.groupby('industry')['merchant_count'].last()
        for industry, count in industry_counts.items():
            industry_breakdown[industry] = {
                'count': count,
                'percentage': (count / current_merchants) * 100 if current_merchants > 0 else 0
            }
    
    return {
        'current_merchant_count': current_merchants,
        'merchant_growth_rate': merchant_growth,
        'industry_breakdown': industry_breakdown,
        'period_days': period_days
    }

def analyze_ecosystem_growth(ecosystem_data, period_days=90):
    """
    Analyze ecosystem growth including dApps, services, etc.
    
    Args:
        ecosystem_data (pd.DataFrame): DataFrame with ecosystem data
        period_days (int): Period to analyze in days
        
    Returns:
        dict: Ecosystem growth metrics
    """
    # Ensure data is sorted by date
    ecosystem_data = ecosystem_data.sort_values('date')
    
    # Calculate date range
    end_date = ecosystem_data['date'].max()
    start_date = end_date - timedelta(days=period_days)
    
    # Filter data for period
    period_data = ecosystem_data[ecosystem_data['date'] >= start_date]
    
    if period_data.empty:
        return {}
    
    # Get latest counts
    current_data = period_data.iloc[-1]
    
    metrics = {
        'total_projects': current_data.get('total_projects', None),
        'active_dapps': current_data.get('active_dapps', None),
        'development_activity': current_data.get('development_activity', None),
        'period_days': period_days
    }
    
    # Calculate growth rates
    if len(period_data) > 1:
        initial_data = period_data.iloc[0]
        
        # Projects growth
        if 'total_projects' in current_data and 'total_projects' in initial_data:
            if initial_data['total_projects'] > 0:
                projects_growth = (current_data['total_projects'] / initial_data['total_projects']) - 1
                metrics['projects_growth_rate'] = projects_growth
        
        # dApps growth
        if 'active_dapps' in current_data and 'active_dapps' in initial_data:
            if initial_data['active_dapps'] > 0:
                dapps_growth = (current_data['active_dapps'] / initial_data['active_dapps']) - 1
                metrics['dapps_growth_rate'] = dapps_growth
    
    # Category breakdown if available
    category_breakdown = {}
    if 'category' in ecosystem_data.columns and 'project_count' in ecosystem_data.columns:
        category_counts = period_data.groupby('category')['project_count'].last()
        for category, count in category_counts.items():
            category_breakdown[category] = {
                'count': count,
                'percentage': (count / metrics['total_projects']) * 100 if metrics.get('total_projects', 0) > 0 else 0
            }
        
        metrics['category_breakdown'] = category_breakdown
    
    return metrics

def calculate_adoption_score(metrics, weights=None):
    """
    Calculate overall adoption score based on multiple metrics
    
    Args:
        metrics (dict): Dictionary of adoption metrics
        weights (dict, optional): Custom weights for each metric
        
    Returns:
        float: Adoption score (0-100)
    """
    if not metrics:
        return 0
    
    # Default weights
    default_weights = {
        'active_addresses': 0.25,
        'transaction_count': 0.20,
        'transaction_growth': 0.15,
        'merchant_count': 0.15,
        'ecosystem_projects': 0.25
    }
    
    # Use custom weights if provided
    if weights is None:
        weights = default_weights
    
    # Define scoring functions for each metric type
    scoring_functions = {
        'active_addresses': lambda x: min(100, max(0, (x / 100000) * 100)),
        'transaction_count': lambda x: min(100, max(0, (x / 1000000) * 100)),
        'transaction_growth': lambda x: min(100, max(0, (x + 0.1) * 500)),
        'merchant_count': lambda x: min(100, max(0, (x / 10000) * 100)),
        'ecosystem_projects': lambda x: min(100, max(0, (x / 1000) * 100))
    }
    
    # Map metrics to scoring categories
    metric_mapping = {
        'avg_active_addresses': 'active_addresses',
        'total_tx_count': 'transaction_count',
        'tx_count_growth': 'transaction_growth',
        'current_merchant_count': 'merchant_count',
        'total_projects': 'ecosystem_projects'
    }
    
    # Calculate scores
    scores = {}
    for metric_name, metric_value in metrics.items():
        if metric_name in metric_mapping and metric_value is not None:
            category = metric_mapping[metric_name]
            if category in scoring_functions:
                scores[category] = scoring_functions[category](metric_value)
    
    # Calculate weighted score
    if not scores:
        return 0
    
    total_weight = sum(weights.get(category, 0) for category in scores.keys())
    if total_weight == 0:
        return 0
    
    weighted_score = sum(scores[category] * weights.get(category, 0) for category in scores.keys()) / total_weight
    
    return weighted_score

def compare_adoption_metrics(target_metrics, competitor_metrics):
    """
    Compare adoption metrics against competitors
    
    Args:
        target_metrics (dict): Metrics for the target token
        competitor_metrics (list): List of metrics for competitor tokens
        
    Returns:
        dict: Comparative analysis
    """
    if not target_metrics or not competitor_metrics:
        return {}
    
    # Define metrics to compare
    comparison_metrics = [
        'avg_active_addresses',
        'total_tx_count',
        'avg_daily_tx',
        'current_merchant_count',
        'total_projects'
    ]
    
    # Filter metrics that are available
    available_metrics = [m for m in comparison_metrics if m in target_metrics]
    
    if not available_metrics:
        return {'error': 'No common metrics available for comparison'}
    
    # Calculate percentiles for each metric
    percentiles = {}
    
    for metric in available_metrics:
        # Skip if target doesn't have this metric
        if metric not in target_metrics or target_metrics[metric] is None:
            continue
            
        # Collect competitor values
        competitor_values = [
            comp.get(metric) for comp in competitor_metrics
            if metric in comp and comp[metric] is not None
        ]
        
        # Add target value
        all_values = competitor_values + [target_metrics[metric]]
        
        # Calculate percentile rank
        if len(all_values) > 1:
            percentile = stats.percentileofscore(all_values, target_metrics[metric])
            percentiles[metric] = percentile
    
    # Calculate overall adoption score
    target_score = calculate_adoption_score(target_metrics)
    competitor_scores = [calculate_adoption_score(comp) for comp in competitor_metrics]
    
    # Calculate average competitor score safely
    avg_competitor_score = sum(competitor_scores) / len(competitor_scores) if competitor_scores else 0
    
    # Determine relative adoption level
    if competitor_scores:
        if target_score > avg_competitor_score + 10:
            relative_adoption = 'High'
        elif target_score < avg_competitor_score - 10:
            relative_adoption = 'Low'
        else:
            relative_adoption = 'Medium'
    else:
        relative_adoption = 'Medium'  # Default when no competitors
    
    return {
        'target_score': target_score,
        'competitor_scores': competitor_scores,
        'avg_competitor_score': avg_competitor_score,
        'percentile_ranks': percentiles,
        'relative_adoption': relative_adoption
    }
