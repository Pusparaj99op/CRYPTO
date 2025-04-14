import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats

def analyze_github_activity(repo_data, period_days=90):
    """
    Analyze GitHub activity for a repository
    
    Args:
        repo_data (dict): Repository data with commits, issues, pulls, etc.
        period_days (int): Period to analyze in days
        
    Returns:
        dict: GitHub activity metrics
    """
    # Set time window
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)
    
    # Filter data for the period
    commits = [c for c in repo_data.get('commits', []) if c['date'] >= start_date]
    issues = [i for i in repo_data.get('issues', []) if i['created_at'] >= start_date]
    pulls = [p for p in repo_data.get('pull_requests', []) if p['created_at'] >= start_date]
    contributors = set([c['author'] for c in commits if c['author']])
    
    # Calculate metrics
    metrics = {
        'commit_count': len(commits),
        'commits_per_day': len(commits) / period_days if period_days > 0 else 0,
        'active_contributors': len(contributors),
        'open_issues': len([i for i in issues if i['state'] == 'open']),
        'closed_issues': len([i for i in issues if i['state'] == 'closed']),
        'open_pulls': len([p for p in pulls if p['state'] == 'open']),
        'merged_pulls': len([p for p in pulls if p['state'] == 'merged']),
        'issue_close_rate': calculate_close_rate(issues),
        'pr_merge_rate': calculate_merge_rate(pulls),
        'avg_time_to_close_issue': calculate_avg_time_to_close(issues),
        'avg_time_to_merge_pr': calculate_avg_time_to_merge(pulls),
        'period_days': period_days
    }
    
    # Add commit frequency data if available
    if commits:
        metrics.update(analyze_commit_frequency(commits))
    
    return metrics

def calculate_close_rate(issues):
    """
    Calculate the rate at which issues are closed
    
    Args:
        issues (list): List of issue data
        
    Returns:
        float: Issue close rate as a percentage
    """
    if not issues:
        return 0
    
    closed_count = sum(1 for i in issues if i['state'] == 'closed')
    return (closed_count / len(issues)) * 100

def calculate_merge_rate(pulls):
    """
    Calculate the rate at which pull requests are merged
    
    Args:
        pulls (list): List of pull request data
        
    Returns:
        float: PR merge rate as a percentage
    """
    if not pulls:
        return 0
    
    merged_count = sum(1 for p in pulls if p['state'] == 'merged')
    return (merged_count / len(pulls)) * 100

def calculate_avg_time_to_close(issues):
    """
    Calculate average time to close issues
    
    Args:
        issues (list): List of issue data
        
    Returns:
        float: Average time to close in days
    """
    closed_issues = [i for i in issues if i['state'] == 'closed' and 'closed_at' in i]
    
    if not closed_issues:
        return None
    
    close_times = [(i['closed_at'] - i['created_at']).total_seconds() / 86400 for i in closed_issues]
    return sum(close_times) / len(close_times)

def calculate_avg_time_to_merge(pulls):
    """
    Calculate average time to merge pull requests
    
    Args:
        pulls (list): List of pull request data
        
    Returns:
        float: Average time to merge in days
    """
    merged_pulls = [p for p in pulls if p['state'] == 'merged' and 'merged_at' in p]
    
    if not merged_pulls:
        return None
    
    merge_times = [(p['merged_at'] - p['created_at']).total_seconds() / 86400 for p in merged_pulls]
    return sum(merge_times) / len(merge_times)

def analyze_commit_frequency(commits):
    """
    Analyze commit frequency
    
    Args:
        commits (list): List of commit data
        
    Returns:
        dict: Commit frequency metrics
    """
    if not commits:
        return {}
    
    # Extract dates
    dates = [c['date'] for c in commits]
    dates.sort()
    
    # Calculate days between first and last commit
    if len(dates) < 2:
        return {
            'commit_frequency': 0,
            'commit_consistency': 0
        }
    
    days_range = (dates[-1] - dates[0]).days
    if days_range == 0:
        return {
            'commit_frequency': len(commits),
            'commit_consistency': 1
        }
    
    # Calculate average commits per day
    commit_frequency = len(commits) / days_range
    
    # Calculate consistency (standard deviation of days between commits)
    days_between = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
    if not days_between:
        commit_consistency = 0
    else:
        # Normalize consistency score (0 to 1, higher is more consistent)
        consistency = 1 / (1 + np.std(days_between))
        commit_consistency = min(max(consistency, 0), 1)
    
    return {
        'commit_frequency': commit_frequency,
        'commit_consistency': commit_consistency
    }

def compare_developer_activity(target_repo, competitor_repos, period_days=90):
    """
    Compare developer activity across multiple repositories
    
    Args:
        target_repo (dict): Target repository data
        competitor_repos (list): List of competitor repository data
        period_days (int): Period to analyze in days
        
    Returns:
        dict: Comparative developer activity analysis
    """
    # Analyze each repo
    target_metrics = analyze_github_activity(target_repo, period_days)
    competitor_metrics = [
        {
            'name': repo.get('name', f'Competitor {i+1}'),
            'metrics': analyze_github_activity(repo, period_days)
        }
        for i, repo in enumerate(competitor_repos)
    ]
    
    # Calculate averages
    metrics_to_compare = [
        'commit_count', 'commits_per_day', 'active_contributors',
        'issue_close_rate', 'pr_merge_rate'
    ]
    
    competitor_averages = {}
    for metric in metrics_to_compare:
        values = [c['metrics'].get(metric, 0) for c in competitor_metrics]
        if values:
            competitor_averages[metric] = sum(values) / len(values)
        else:
            competitor_averages[metric] = 0
    
    # Calculate percentiles
    percentiles = {}
    for metric in metrics_to_compare:
        if metric in target_metrics:
            values = [c['metrics'].get(metric, 0) for c in competitor_metrics]
            values.append(target_metrics[metric])
            
            if len(values) > 1:
                percentiles[metric] = stats.percentileofscore(values, target_metrics[metric])
            else:
                percentiles[metric] = 50
    
    return {
        'target_metrics': target_metrics,
        'competitor_metrics': competitor_metrics,
        'competitor_averages': competitor_averages,
        'percentiles': percentiles
    }

def analyze_code_quality(repo_data):
    """
    Analyze code quality metrics
    
    Args:
        repo_data (dict): Repository data with code metrics
        
    Returns:
        dict: Code quality analysis
    """
    code_metrics = repo_data.get('code_metrics', {})
    
    if not code_metrics:
        return {'error': 'No code metrics available'}
    
    # Normalize metrics to 0-100 scale
    normalized_metrics = {}
    for metric, value in code_metrics.items():
        # Skip non-numeric values
        if not isinstance(value, (int, float)):
            continue
            
        # Apply appropriate transformations based on metric type
        if metric in ['test_coverage', 'documentation_coverage']:
            # Higher is better
            normalized_metrics[metric] = min(value, 100)
        elif metric in ['code_churn', 'complexity', 'technical_debt']:
            # Lower is better
            normalized_metrics[metric] = max(0, 100 - value)
        else:
            # Default case
            normalized_metrics[metric] = value
    
    # Calculate weighted quality score
    weights = {
        'test_coverage': 0.3,
        'code_churn': 0.1,
        'complexity': 0.2,
        'technical_debt': 0.25,
        'documentation_coverage': 0.15
    }
    
    weighted_score = 0
    total_weight = 0
    
    for metric, value in normalized_metrics.items():
        if metric in weights:
            weighted_score += value * weights[metric]
            total_weight += weights[metric]
    
    quality_score = weighted_score / total_weight if total_weight > 0 else 0
    
    # Interpret the score
    if quality_score >= 80:
        quality_level = 'Excellent'
    elif quality_score >= 60:
        quality_level = 'Good'
    elif quality_score >= 40:
        quality_level = 'Average'
    elif quality_score >= 20:
        quality_level = 'Poor'
    else:
        quality_level = 'Very Poor'
    
    return {
        'raw_metrics': code_metrics,
        'normalized_metrics': normalized_metrics,
        'quality_score': quality_score,
        'quality_level': quality_level
    }

def analyze_contributor_network(commits, min_contributions=5):
    """
    Analyze the contributor network and collaboration patterns
    
    Args:
        commits (list): List of commit data
        min_contributions (int): Minimum number of contributions to be included
        
    Returns:
        dict: Contributor network analysis
    """
    if not commits:
        return {'error': 'No commit data available'}
    
    # Count contributions by author
    author_contributions = {}
    for commit in commits:
        author = commit.get('author')
        if not author:
            continue
            
        if author not in author_contributions:
            author_contributions[author] = {
                'commits': 0,
                'files_changed': set(),
                'total_additions': 0,
                'total_deletions': 0
            }
            
        author_contributions[author]['commits'] += 1
        author_contributions[author]['files_changed'].update(commit.get('files_changed', []))
        author_contributions[author]['total_additions'] += commit.get('additions', 0)
        author_contributions[author]['total_deletions'] += commit.get('deletions', 0)
    
    # Filter for authors with minimum contributions
    active_contributors = {
        author: stats for author, stats in author_contributions.items()
        if stats['commits'] >= min_contributions
    }
    
    # Calculate centralization metrics
    total_commits = sum(stats['commits'] for stats in active_contributors.values())
    
    contributor_metrics = []
    for author, stats in active_contributors.items():
        contributor_metrics.append({
            'author': author,
            'commits': stats['commits'],
            'commit_share': (stats['commits'] / total_commits) if total_commits > 0 else 0,
            'files_changed': len(stats['files_changed']),
            'total_additions': stats['total_additions'],
            'total_deletions': stats['total_deletions'],
            'code_churn': stats['total_additions'] + stats['total_deletions']
        })
    
    # Sort by commit share
    contributor_metrics.sort(key=lambda x: x['commit_share'], reverse=True)
    
    # Calculate concentration metrics
    top_contributor_share = contributor_metrics[0]['commit_share'] if contributor_metrics else 0
    top3_share = sum(c['commit_share'] for c in contributor_metrics[:3]) if len(contributor_metrics) >= 3 else top_contributor_share
    
    # Calculate Gini coefficient for commit distribution
    if len(contributor_metrics) > 1:
        sorted_shares = sorted([c['commit_share'] for c in contributor_metrics])
        cumulative_shares = np.cumsum(sorted_shares)
        n = len(sorted_shares)
        gini = 1 - (2 * np.sum(cumulative_shares) / (n * np.sum(sorted_shares))) + (1 / n)
    else:
        gini = 0
    
    return {
        'active_contributors': len(active_contributors),
        'total_commits': total_commits,
        'top_contributor_share': top_contributor_share,
        'top3_contributors_share': top3_share,
        'gini_coefficient': gini,
        'centralization': 'High' if gini > 0.6 else 'Medium' if gini > 0.4 else 'Low',
        'contributor_metrics': contributor_metrics
    }
