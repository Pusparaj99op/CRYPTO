import pandas as pd
import numpy as np
from datetime import datetime

def analyze_token_supply(max_supply, circulating_supply, total_supply):
    """
    Analyze token supply metrics to determine supply ratio and scarcity
    
    Args:
        max_supply (float): Maximum supply of tokens
        circulating_supply (float): Current circulating supply
        total_supply (float): Total supply created so far
        
    Returns:
        dict: Supply metrics and indicators
    """
    results = {
        'circulating_ratio': circulating_supply / max_supply if max_supply else None,
        'issuance_ratio': total_supply / max_supply if max_supply else None,
        'unreleased_supply': total_supply - circulating_supply,
        'is_deflationary': max_supply is not None,
        'supply_inflation_rate': (total_supply - circulating_supply) / circulating_supply if circulating_supply else 0
    }
    return results

def calculate_inflation_rate(supply_schedule, current_date=None):
    """
    Calculate inflation rate based on token release schedule
    
    Args:
        supply_schedule (dict): Dictionary with dates as keys and expected supply as values
        current_date (datetime, optional): Date to calculate from
        
    Returns:
        float: Annual inflation rate
    """
    if current_date is None:
        current_date = datetime.now()
    
    # Convert to DataFrame for analysis
    schedule_df = pd.DataFrame(list(supply_schedule.items()), columns=['date', 'supply'])
    schedule_df['date'] = pd.to_datetime(schedule_df['date'])
    schedule_df = schedule_df.sort_values('date')
    
    # Find closest dates before and after current date
    past_supply = schedule_df[schedule_df['date'] <= current_date]
    future_supply = schedule_df[schedule_df['date'] > current_date]
    
    if past_supply.empty or future_supply.empty:
        return None
    
    current_supply = past_supply.iloc[-1]['supply']
    next_year_date = current_date.replace(year=current_date.year + 1)
    
    # Find or interpolate future supply
    future_supply_year = None
    for i, row in future_supply.iterrows():
        if row['date'] >= next_year_date:
            future_supply_year = row['supply']
            break
    
    if future_supply_year is None:
        return None
    
    inflation_rate = (future_supply_year - current_supply) / current_supply
    return inflation_rate

def analyze_token_distribution(holder_data):
    """
    Analyze token distribution among holders
    
    Args:
        holder_data (list): List of dictionaries with address and balance
        
    Returns:
        dict: Distribution metrics
    """
    if not holder_data:
        return {}
    
    # Convert to DataFrame
    df = pd.DataFrame(holder_data)
    
    # Sort by balance
    df = df.sort_values('balance', ascending=False)
    
    # Calculate total supply in dataset
    total_balance = df['balance'].sum()
    
    # Calculate Gini coefficient
    df = df.sort_values('balance')
    cumulative_balance = df['balance'].cumsum()
    lorenz_curve = cumulative_balance / total_balance
    gini = 1 - 2 * np.trapz(lorenz_curve, dx=1/len(df))
    
    # Top holders analysis
    top10_share = df.iloc[:10]['balance'].sum() / total_balance
    top50_share = df.iloc[:50]['balance'].sum() / total_balance
    top100_share = df.iloc[:100]['balance'].sum() / total_balance
    
    return {
        'gini_coefficient': gini,
        'top10_concentration': top10_share,
        'top50_concentration': top50_share,
        'top100_concentration': top100_share,
        'unique_holders': len(df),
        'avg_balance': df['balance'].mean(),
        'median_balance': df['balance'].median()
    }

def calculate_velocity(transaction_volume, circulating_supply, period_days=30):
    """
    Calculate token velocity - how frequently tokens change hands
    
    Args:
        transaction_volume (float): Total transaction volume in period
        circulating_supply (float): Average circulating supply in period
        period_days (int): Number of days to calculate velocity for
        
    Returns:
        float: Token velocity (annualized)
    """
    if circulating_supply == 0:
        return 0
    
    # Daily velocity
    daily_velocity = transaction_volume / circulating_supply
    
    # Annualized velocity
    annual_velocity = daily_velocity * (365 / period_days)
    
    return annual_velocity
