import pandas as pd
import numpy as np
from datetime import datetime

def calculate_regulatory_risk_score(factors):
    """
    Calculate a regulatory risk score based on various factors
    
    Args:
        factors (dict): Dictionary with risk factors and their weights
        
    Returns:
        float: Regulatory risk score between 0-100
    """
    if not factors:
        return None
    
    # Extract values and weights
    values = np.array([v['value'] for v in factors.values()])
    weights = np.array([v['weight'] for v in factors.values()])
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Calculate weighted score
    risk_score = np.sum(values * weights) * 100
    
    return min(max(risk_score, 0), 100)  # Ensure score is between 0-100

def analyze_compliance_status(token_data, regulatory_frameworks):
    """
    Analyze compliance status across different regulatory frameworks
    
    Args:
        token_data (dict): Token metadata and characteristics
        regulatory_frameworks (dict): Dictionary of regulatory frameworks and requirements
        
    Returns:
        dict: Compliance analysis results
    """
    compliance_results = {}
    
    for framework, requirements in regulatory_frameworks.items():
        framework_compliance = {
            'compliant': True,
            'missing_requirements': [],
            'compliance_score': 0
        }
        
        met_requirements = 0
        total_requirements = len(requirements)
        
        for req_id, requirement in requirements.items():
            # Check if token meets this requirement
            if requirement['check_function'](token_data):
                met_requirements += 1
            else:
                framework_compliance['compliant'] = False
                framework_compliance['missing_requirements'].append({
                    'id': req_id,
                    'description': requirement['description'],
                    'importance': requirement['importance']
                })
        
        # Calculate compliance score as percentage
        if total_requirements > 0:
            framework_compliance['compliance_score'] = (met_requirements / total_requirements) * 100
        
        compliance_results[framework] = framework_compliance
    
    # Calculate overall compliance score
    all_frameworks_score = np.mean([
        res['compliance_score'] for res in compliance_results.values()
    ]) if compliance_results else 0
    
    return {
        'framework_compliance': compliance_results,
        'overall_compliance_score': all_frameworks_score
    }

def check_security_token_status(token_data):
    """
    Check if a token might be classified as a security using the Howey Test
    
    Args:
        token_data (dict): Token characteristics and properties
        
    Returns:
        dict: Security token analysis results
    """
    # Howey Test criteria
    howey_criteria = {
        'investment_of_money': {
            'result': None,
            'explanation': '',
            'risk_level': 0  # 0-10 scale
        },
        'common_enterprise': {
            'result': None,
            'explanation': '',
            'risk_level': 0
        },
        'expectation_of_profits': {
            'result': None,
            'explanation': '',
            'risk_level': 0
        },
        'efforts_of_others': {
            'result': None,
            'explanation': '',
            'risk_level': 0
        }
    }
    
    # Investment of money
    if token_data.get('ico_conducted', False):
        howey_criteria['investment_of_money']['result'] = True
        howey_criteria['investment_of_money']['explanation'] = 'Token was sold through an ICO/IEO/IDO'
        howey_criteria['investment_of_money']['risk_level'] = 8
    
    # Common enterprise
    if token_data.get('centralized_development', False):
        howey_criteria['common_enterprise']['result'] = True
        howey_criteria['common_enterprise']['explanation'] = 'Development is centralized'
        howey_criteria['common_enterprise']['risk_level'] = 7
    
    # Expectation of profits
    if token_data.get('marketed_as_investment', False):
        howey_criteria['expectation_of_profits']['result'] = True
        howey_criteria['expectation_of_profits']['explanation'] = 'Token was marketed with emphasis on price appreciation'
        howey_criteria['expectation_of_profits']['risk_level'] = 9
    
    # Efforts of others
    if token_data.get('value_dependent_on_team', False):
        howey_criteria['efforts_of_others']['result'] = True
        howey_criteria['efforts_of_others']['explanation'] = 'Token value primarily dependent on development team efforts'
        howey_criteria['efforts_of_others']['risk_level'] = 8
    
    # Calculate overall result
    criteria_met = sum(1 for c in howey_criteria.values() if c['result'] == True)
    avg_risk = np.mean([c['risk_level'] for c in howey_criteria.values() if c['risk_level'] > 0])
    
    security_risk_level = (criteria_met / len(howey_criteria)) * 10
    
    return {
        'howey_criteria': howey_criteria,
        'criteria_met': criteria_met,
        'security_risk_level': security_risk_level,
        'avg_risk_level': avg_risk,
        'is_likely_security': criteria_met >= 3,
        'recommendation': 'Seek legal counsel' if criteria_met >= 2 else 'Likely not a security, but monitor regulations'
    }

def analyze_jurisdictional_status(token_data, jurisdictions):
    """
    Analyze a token's regulatory status across different jurisdictions
    
    Args:
        token_data (dict): Token metadata and characteristics
        jurisdictions (list): List of jurisdictions to analyze
        
    Returns:
        dict: Jurisdictional analysis results
    """
    results = {}
    
    for jurisdiction in jurisdictions:
        # Skip if regulatory framework not defined
        if not jurisdiction.get('regulatory_framework'):
            continue
        
        status = {
            'allowed': True,
            'restrictions': [],
            'banned': False,
            'risk_level': 0,  # 0-10 scale
            'notes': ''
        }
        
        # Check if token type is banned
        if token_data.get('token_type') in jurisdiction.get('banned_token_types', []):
            status['allowed'] = False
            status['banned'] = True
            status['risk_level'] = 10
            status['notes'] = f"{token_data.get('token_type')} tokens are banned in this jurisdiction"
        
        # Check for restrictions
        for restriction in jurisdiction.get('restrictions', []):
            if restriction['check_function'](token_data):
                status['restrictions'].append(restriction['description'])
                status['risk_level'] = max(status['risk_level'], restriction['risk_level'])
        
        results[jurisdiction['name']] = status
    
    # Calculate overall risk level
    if results:
        avg_risk = np.mean([r['risk_level'] for r in results.values()])
        max_risk = max([r['risk_level'] for r in results.values()])
        banned_count = sum(1 for r in results.values() if r['banned'])
    else:
        avg_risk = 0
        max_risk = 0
        banned_count = 0
    
    return {
        'jurisdictional_status': results,
        'avg_risk_level': avg_risk,
        'max_risk_level': max_risk,
        'banned_count': banned_count,
        'high_risk_jurisdictions': [
            j for j, r in results.items() if r['risk_level'] >= 7
        ]
    }

def monitor_regulatory_changes(current_regulations, historical_regulations):
    """
    Monitor and analyze changes in regulations over time
    
    Args:
        current_regulations (dict): Current regulatory state
        historical_regulations (list): Historical regulatory states with timestamps
        
    Returns:
        dict: Analysis of regulatory changes
    """
    if not historical_regulations:
        return {'changes': [], 'trend': 'unknown', 'velocity': 0}
    
    # Sort historical regulations by timestamp
    sorted_regulations = sorted(
        historical_regulations,
        key=lambda x: x['timestamp']
    )
    
    changes = []
    for i in range(1, len(sorted_regulations)):
        prev_reg = sorted_regulations[i-1]['regulations']
        curr_reg = sorted_regulations[i]['regulations']
        
        for jurisdiction, curr_rules in curr_reg.items():
            if jurisdiction not in prev_reg:
                changes.append({
                    'timestamp': sorted_regulations[i]['timestamp'],
                    'jurisdiction': jurisdiction,
                    'type': 'new_jurisdiction',
                    'description': f"New regulations added for {jurisdiction}"
                })
                continue
                
            prev_rules = prev_reg[jurisdiction]
            
            # Detect changes in restrictions
            for restriction_id, restriction in curr_rules.get('restrictions', {}).items():
                if restriction_id not in prev_rules.get('restrictions', {}):
                    changes.append({
                        'timestamp': sorted_regulations[i]['timestamp'],
                        'jurisdiction': jurisdiction,
                        'type': 'new_restriction',
                        'description': f"New restriction: {restriction['description']}"
                    })
                elif restriction != prev_rules['restrictions'][restriction_id]:
                    changes.append({
                        'timestamp': sorted_regulations[i]['timestamp'],
                        'jurisdiction': jurisdiction,
                        'type': 'modified_restriction',
                        'description': f"Modified restriction: {restriction['description']}"
                    })
    
    # Analyze trend
    restrictive_changes = sum(1 for c in changes if c['type'] in ['new_restriction', 'modified_restriction'])
    permissive_changes = sum(1 for c in changes if c['type'] in ['removed_restriction'])
    
    if restrictive_changes > permissive_changes:
        trend = 'more_restrictive'
    elif restrictive_changes < permissive_changes:
        trend = 'more_permissive'
    else:
        trend = 'neutral'
    
    # Calculate change velocity (changes per month)
    if len(sorted_regulations) > 1:
        first_date = sorted_regulations[0]['timestamp']
        last_date = sorted_regulations[-1]['timestamp']
        months_diff = (last_date - first_date).days / 30
        velocity = len(changes) / months_diff if months_diff > 0 else 0
    else:
        velocity = 0
    
    return {
        'changes': changes,
        'trend': trend,
        'velocity': velocity
    }
