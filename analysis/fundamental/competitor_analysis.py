import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def identify_competitors(token_data, token_universe, similarity_threshold=0.7):
    """
    Identify competitor tokens based on similarity of features
    
    Args:
        token_data (dict): Token features and metrics
        token_universe (list): List of dictionaries with token data for comparison
        similarity_threshold (float): Threshold to consider tokens as competitors
        
    Returns:
        list: List of competitors ranked by similarity
    """
    if not token_universe:
        return []
    
    # Convert token_universe to DataFrame
    universe_df = pd.DataFrame(token_universe)
    
    # Features to compare
    features = [
        'category', 'use_case', 'technology', 'target_market', 
        'consensus_mechanism', 'token_type'
    ]
    
    # Available features (intersection of token_data keys and defined features)
    available_features = [f for f in features if f in token_data and f in universe_df.columns]
    
    if not available_features:
        return []
    
    # Calculate similarity scores
    similarity_scores = []
    
    for idx, row in universe_df.iterrows():
        score = 0
        weights = {
            'category': 0.3,
            'use_case': 0.3,
            'technology': 0.2,
            'target_market': 0.1,
            'consensus_mechanism': 0.05,
            'token_type': 0.05
        }
        
        # Adjust weights if features are missing
        available_weights = {f: weights[f] for f in available_features}
        weight_sum = sum(available_weights.values())
        
        # Normalize weights
        if weight_sum > 0:
            available_weights = {f: w/weight_sum for f, w in available_weights.items()}
        
        # Calculate weighted similarity
        for feature in available_features:
            if isinstance(token_data[feature], (list, set)):
                # For list features, calculate Jaccard similarity
                set_a = set(token_data[feature])
                set_b = set(row[feature]) if isinstance(row[feature], (list, set)) else {row[feature]}
                if set_a or set_b:  # Avoid division by zero
                    feature_sim = len(set_a.intersection(set_b)) / len(set_a.union(set_b))
                else:
                    feature_sim = 0
            else:
                # For scalar features, exact match = 1, no match = 0
                feature_sim = 1 if token_data[feature] == row[feature] else 0
                
            score += feature_sim * available_weights[feature]
            
        similarity_scores.append({
            'token_id': row['token_id'],
            'name': row['name'],
            'similarity': score
        })
    
    # Sort by similarity and filter
    competitors = [
        comp for comp in sorted(similarity_scores, key=lambda x: x['similarity'], reverse=True)
        if comp['similarity'] >= similarity_threshold and comp['token_id'] != token_data.get('token_id')
    ]
    
    return competitors

def compare_market_metrics(token_data, competitors, metrics):
    """
    Compare token's market metrics against competitors
    
    Args:
        token_data (dict): Target token's data
        competitors (list): List of competitor tokens with their data
        metrics (list): List of metric names to compare
        
    Returns:
        dict: Comparison results
    """
    if not competitors or not metrics:
        return {}
    
    # Extract metrics from token and competitors
    all_tokens = [token_data] + competitors
    comparison_data = []
    
    for token in all_tokens:
        token_metrics = {
            'token_id': token['token_id'],
            'name': token['name']
        }
        
        for metric in metrics:
            if metric in token:
                token_metrics[metric] = token[metric]
        
        comparison_data.append(token_metrics)
    
    # Convert to DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Calculate percentiles for each metric
    percentiles = {}
    for metric in metrics:
        if metric in df.columns:
            # Skip if metric has missing values or all values are the same
            if df[metric].isna().any() or df[metric].nunique() <= 1:
                continue
                
            # Calculate percentile rank
            df[f'{metric}_percentile'] = df[metric].rank(pct=True) * 100
            
            # Get target token's percentile
            target_percentile = df[df['token_id'] == token_data['token_id']][f'{metric}_percentile'].iloc[0]
            percentiles[metric] = target_percentile
    
    # Calculate relative strength
    relative_strength = sum(percentiles.values()) / len(percentiles) if percentiles else 0
    
    return {
        'comparison_data': comparison_data,
        'percentiles': percentiles,
        'relative_strength': relative_strength
    }

def analyze_competitive_advantage(token_data, competitors, advantage_factors):
    """
    Analyze competitive advantages of a token compared to competitors
    
    Args:
        token_data (dict): Target token's data
        competitors (list): List of competitor tokens with their data
        advantage_factors (list): List of advantage factor definitions
        
    Returns:
        dict: Competitive advantage analysis
    """
    advantages = []
    disadvantages = []
    
    for factor in advantage_factors:
        factor_name = factor['name']
        metrics = factor['metrics']
        importance = factor['importance']
        
        # Calculate factor score for target token
        target_score = 0
        available_metrics = 0
        
        for metric, weight in metrics.items():
            if metric in token_data:
                target_score += token_data[metric] * weight
                available_metrics += 1
        
        # Skip if no metrics available
        if available_metrics == 0:
            continue
            
        # Calculate factor scores for competitors
        competitor_scores = []
        
        for comp in competitors:
            comp_score = 0
            comp_available_metrics = 0
            
            for metric, weight in metrics.items():
                if metric in comp:
                    comp_score += comp[metric] * weight
                    comp_available_metrics += 1
            
            if comp_available_metrics > 0:
                competitor_scores.append(comp_score)
        
        # Skip if no competitor data
        if not competitor_scores:
            continue
            
        # Compare scores
        avg_competitor_score = sum(competitor_scores) / len(competitor_scores)
        relative_advantage = target_score - avg_competitor_score
        
        result = {
            'factor': factor_name,
            'target_score': target_score,
            'avg_competitor_score': avg_competitor_score,
            'relative_advantage': relative_advantage,
            'importance': importance
        }
        
        if relative_advantage > 0:
            advantages.append(result)
        else:
            disadvantages.append(result)
    
    # Sort by importance and strength
    advantages.sort(key=lambda x: x['importance'] * x['relative_advantage'], reverse=True)
    disadvantages.sort(key=lambda x: x['importance'] * abs(x['relative_advantage']), reverse=True)
    
    return {
        'competitive_advantages': advantages,
        'competitive_disadvantages': disadvantages,
        'net_advantage': sum(a['relative_advantage'] * a['importance'] for a in advantages) -
                         sum(abs(d['relative_advantage']) * d['importance'] for d in disadvantages)
    }

def create_competitive_positioning_matrix(token_data, competitors, x_metric, y_metric):
    """
    Create competitive positioning matrix based on two metrics
    
    Args:
        token_data (dict): Target token's data
        competitors (list): List of competitor tokens with their data
        x_metric (str): Metric for x-axis
        y_metric (str): Metric for y-axis
        
    Returns:
        dict: Data for competitive positioning matrix
    """
    # Extract data
    all_tokens = [token_data] + competitors
    matrix_data = []
    
    for token in all_tokens:
        if x_metric in token and y_metric in token:
            matrix_data.append({
                'token_id': token['token_id'],
                'name': token['name'],
                'is_target': token['token_id'] == token_data['token_id'],
                x_metric: token[x_metric],
                y_metric: token[y_metric]
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(matrix_data)
    
    # Skip if not enough data
    if len(df) < 2:
        return {'error': 'Not enough data for positioning matrix'}
    
    # Normalize data for visualization
    scaler = MinMaxScaler()
    if len(df) >= 2:  # MinMaxScaler requires at least 2 samples
        df[[f'{x_metric}_scaled', f'{y_metric}_scaled']] = scaler.fit_transform(df[[x_metric, y_metric]])
    
    # Calculate quadrant
    x_median = df[x_metric].median()
    y_median = df[y_metric].median()
    
    # Assign quadrants
    df['quadrant'] = 0
    df.loc[(df[x_metric] >= x_median) & (df[y_metric] >= y_median), 'quadrant'] = 1  # Top right
    df.loc[(df[x_metric] < x_median) & (df[y_metric] >= y_median), 'quadrant'] = 2   # Top left
    df.loc[(df[x_metric] < x_median) & (df[y_metric] < y_median), 'quadrant'] = 3    # Bottom left
    df.loc[(df[x_metric] >= x_median) & (df[y_metric] < y_median), 'quadrant'] = 4   # Bottom right
    
    # Get target token's quadrant
    target_quadrant = df[df['is_target']]['quadrant'].iloc[0]
    
    # Calculate quadrant interpretations
    quadrant_names = {
        1: "Leader (High " + x_metric + ", High " + y_metric + ")",
        2: "Niche (Low " + x_metric + ", High " + y_metric + ")",
        3: "Laggard (Low " + x_metric + ", Low " + y_metric + ")",
        4: "Challenger (High " + x_metric + ", Low " + y_metric + ")"
    }
    
    return {
        'matrix_data': df.to_dict('records'),
        'x_metric': x_metric,
        'y_metric': y_metric,
        'target_quadrant': int(target_quadrant),
        'quadrant_name': quadrant_names[int(target_quadrant)],
        'x_median': x_median,
        'y_median': y_median
    }

def market_share_analysis(token_data, market_segment, universe_data):
    """
    Analyze market share within a specific market segment
    
    Args:
        token_data (dict): Target token's data
        market_segment (str): Market segment to analyze
        universe_data (list): List of tokens in the universe
        
    Returns:
        dict: Market share analysis
    """
    # Filter universe for the market segment
    segment_tokens = [
        t for t in universe_data 
        if t.get('category') == market_segment or 
        market_segment in t.get('categories', [])
    ]
    
    if not segment_tokens:
        return {'error': f'No tokens found in market segment: {market_segment}'}
    
    # Calculate total market cap of segment
    segment_market_cap = sum(t.get('market_cap', 0) for t in segment_tokens)
    
    if segment_market_cap == 0:
        return {'error': 'Market cap data not available'}
    
    # Calculate market shares
    market_shares = []
    for token in segment_tokens:
        if 'market_cap' in token and token['market_cap'] > 0:
            market_shares.append({
                'token_id': token['token_id'],
                'name': token['name'],
                'market_cap': token['market_cap'],
                'market_share': (token['market_cap'] / segment_market_cap) * 100,
                'is_target': token['token_id'] == token_data['token_id']
            })
    
    # Sort by market share
    market_shares.sort(key=lambda x: x['market_share'], reverse=True)
    
    # Calculate target token's metrics
    target_market_share = next(
        (item['market_share'] for item in market_shares if item['is_target']), 
        0
    )
    
    target_rank = next(
        (i+1 for i, item in enumerate(market_shares) if item['is_target']), 
        0
    )
    
    # Calculate concentration metrics
    top3_share = sum(item['market_share'] for item in market_shares[:3])
    top5_share = sum(item['market_share'] for item in market_shares[:5])
    top10_share = sum(item['market_share'] for item in market_shares[:10])
    
    # Calculate HHI (Herfindahl-Hirschman Index)
    hhi = sum((item['market_share'] / 100) ** 2 for item in market_shares) * 10000
    
    return {
        'market_segment': market_segment,
        'total_market_cap': segment_market_cap,
        'token_count': len(market_shares),
        'target_market_share': target_market_share,
        'target_rank': target_rank,
        'top3_concentration': top3_share,
        'top5_concentration': top5_share,
        'top10_concentration': top10_share,
        'hhi': hhi,
        'market_concentration': 'High' if hhi > 2500 else 'Moderate' if hhi > 1500 else 'Low',
        'market_shares': market_shares
    }
