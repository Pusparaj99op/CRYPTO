import re
import pandas as pd
import numpy as np
from datetime import datetime
import spacy
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union

try:
    # Try to load spaCy model
    nlp = spacy.load("en_core_web_md")
except OSError:
    # If model not found, suggest installation
    print("Spacy model not found. Please install using: python -m spacy download en_core_web_md")
    # Create a simple pipeline as fallback
    nlp = spacy.blank("en")


def extract_ticker_symbols(text: str) -> List[str]:
    """
    Extract potential ticker symbols from text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        List[str]: List of extracted ticker symbols
    """
    # Pattern for ticker symbols: $XXX or $XXXX (1-5 capital letters)
    ticker_pattern = r'\$([A-Z]{1,5})\b'
    
    # Find all matches
    tickers = re.findall(ticker_pattern, text)
    
    return tickers


def extract_price_mentions(text: str) -> List[Dict[str, Union[str, float]]]:
    """
    Extract price mentions from text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        List[Dict]: List of dictionaries with price information
    """
    # Pattern for price mentions: $X, $X.XX, X dollars, etc.
    price_pattern = r'(?:(?:\$|USD|dollars)\s*)([\d,]+(?:\.\d+)?|(?:\.\d+))(?:\s*(?:USD|dollars))?|(?:([\d,]+(?:\.\d+)?|(?:\.\d+))\s*(?:USD|dollars|\$))'
    
    # Find all matches
    prices = []
    matches = re.finditer(price_pattern, text.lower())
    
    for match in matches:
        price_str = match.group(1) if match.group(1) else match.group(2)
        if price_str:
            # Remove commas and convert to float
            price_value = float(price_str.replace(',', ''))
            prices.append({
                "price": price_value,
                "original_text": match.group(0),
                "position": match.span()
            })
    
    return prices


def extract_percentage_changes(text: str) -> List[Dict[str, Union[str, float]]]:
    """
    Extract percentage changes from text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        List[Dict]: List of dictionaries with percentage information
    """
    # Pattern for percentage mentions: X%, up/down X%, increased/decreased by X%
    percentage_pattern = r'(?:up|down|increase[d]?|decrease[d]?|gain[ed]?|los[t]?|fell|drop[ped]?|jump[ed]?|surge[d]?|plummet[ed]?|rise|rose|growth)(?:\s+by)?\s+(\d+(?:\.\d+)?)%|(\d+(?:\.\d+)?)%\s+(?:up|down|increase|decrease|gain|loss|drop|jump|surge|plummet|rise|growth)'
    
    # Find all matches
    percentages = []
    matches = re.finditer(percentage_pattern, text.lower())
    
    for match in matches:
        pct_str = match.group(1) if match.group(1) else match.group(2)
        if pct_str:
            pct_value = float(pct_str)
            
            # Determine direction
            direction = None
            match_text = match.group(0).lower()
            if any(word in match_text for word in ['up', 'increase', 'gain', 'jump', 'surge', 'rise', 'rose', 'growth']):
                direction = 'positive'
            elif any(word in match_text for word in ['down', 'decrease', 'loss', 'lost', 'fell', 'drop', 'plummet']):
                direction = 'negative'
            
            percentages.append({
                "percentage": pct_value,
                "direction": direction,
                "original_text": match.group(0),
                "position": match.span()
            })
    
    return percentages


def extract_dates(text: str) -> List[Dict[str, Union[str, datetime]]]:
    """
    Extract date mentions from text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        List[Dict]: List of dictionaries with date information
    """
    # Use spaCy's entity recognition for dates
    doc = nlp(text)
    
    dates = []
    for ent in doc.ents:
        if ent.label_ == "DATE":
            dates.append({
                "date_text": ent.text,
                "position": (ent.start_char, ent.end_char)
            })
    
    return dates


def extract_named_entities(text: str, entity_types: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Union[str, Tuple[int, int]]]]]:
    """
    Extract named entities from text.
    
    Args:
        text (str): Input text to analyze
        entity_types (List[str], optional): Types of entities to extract
            (e.g., "PERSON", "ORG", "GPE", "MONEY", "PERCENT", "DATE")
            
    Returns:
        Dict[str, List[Dict]]: Dictionary with entity types as keys and lists of entities as values
    """
    doc = nlp(text)
    
    entities = defaultdict(list)
    
    for ent in doc.ents:
        if entity_types is None or ent.label_ in entity_types:
            entities[ent.label_].append({
                "text": ent.text,
                "position": (ent.start_char, ent.end_char)
            })
    
    return dict(entities)


def extract_financial_metrics(text: str) -> Dict[str, List[Dict]]:
    """
    Extract financial metrics from text (market cap, volume, supply, etc.).
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        Dict[str, List[Dict]]: Dictionary with extracted financial metrics
    """
    metrics = defaultdict(list)
    
    # Market cap pattern
    market_cap_pattern = r'(?:market\s+cap(?:italization)?|mcap)\s+(?:of\s+)?(?:\$|USD|dollars)?\s*([\d,]+(?:\.\d+)?)\s*(?:billion|million|trillion|B|M|T|USD|dollars|\$)?'
    market_cap_matches = re.finditer(market_cap_pattern, text.lower())
    
    for match in market_cap_matches:
        value_str = match.group(1).replace(',', '')
        value = float(value_str)
        
        # Determine multiplier
        multiplier = 1
        if 'billion' in match.group(0).lower() or 'B' in match.group(0):
            multiplier = 1_000_000_000
        elif 'million' in match.group(0).lower() or 'M' in match.group(0):
            multiplier = 1_000_000
        elif 'trillion' in match.group(0).lower() or 'T' in match.group(0):
            multiplier = 1_000_000_000_000
        
        metrics['market_cap'].append({
            "value": value * multiplier,
            "original_text": match.group(0),
            "position": match.span()
        })
    
    # Trading volume pattern
    volume_pattern = r'(?:(?:trading|24h|daily)\s+)?volume\s+(?:of\s+)?(?:\$|USD|dollars)?\s*([\d,]+(?:\.\d+)?)\s*(?:billion|million|trillion|B|M|T|USD|dollars|\$)?'
    volume_matches = re.finditer(volume_pattern, text.lower())
    
    for match in volume_matches:
        value_str = match.group(1).replace(',', '')
        value = float(value_str)
        
        # Determine multiplier
        multiplier = 1
        if 'billion' in match.group(0).lower() or 'B' in match.group(0):
            multiplier = 1_000_000_000
        elif 'million' in match.group(0).lower() or 'M' in match.group(0):
            multiplier = 1_000_000
        elif 'trillion' in match.group(0).lower() or 'T' in match.group(0):
            multiplier = 1_000_000_000_000
        
        metrics['volume'].append({
            "value": value * multiplier,
            "original_text": match.group(0),
            "position": match.span()
        })
    
    # Supply pattern
    supply_pattern = r'(?:circulating|total|max)\s+supply\s+(?:of\s+)?([\d,]+(?:\.\d+)?)\s*(?:billion|million|trillion|B|M|T)?'
    supply_matches = re.finditer(supply_pattern, text.lower())
    
    for match in supply_matches:
        value_str = match.group(1).replace(',', '')
        value = float(value_str)
        
        # Determine multiplier
        multiplier = 1
        if 'billion' in match.group(0).lower() or 'B' in match.group(0):
            multiplier = 1_000_000_000
        elif 'million' in match.group(0).lower() or 'M' in match.group(0):
            multiplier = 1_000_000
        elif 'trillion' in match.group(0).lower() or 'T' in match.group(0):
            multiplier = 1_000_000_000_000
        
        # Determine supply type
        supply_type = 'circulating_supply'
        if 'total' in match.group(0).lower():
            supply_type = 'total_supply'
        elif 'max' in match.group(0).lower():
            supply_type = 'max_supply'
        
        metrics[supply_type].append({
            "value": value * multiplier,
            "original_text": match.group(0),
            "position": match.span()
        })
    
    return dict(metrics)


def extract_token_metrics(text: str) -> Dict[str, List[Dict]]:
    """
    Extract specific token metrics from text (staking yield, APY, lockup periods).
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        Dict[str, List[Dict]]: Dictionary with extracted token metrics
    """
    metrics = defaultdict(list)
    
    # Staking yield/APY pattern
    apy_pattern = r'(?:staking\s+(?:yield|rewards|return)|apy|apr|annual\s+(?:percentage\s+(?:yield|rate)|returns?))(?:\s+of)?\s+(\d+(?:\.\d+)?)%'
    apy_matches = re.finditer(apy_pattern, text.lower())
    
    for match in apy_matches:
        value = float(match.group(1))
        
        # Determine metric type
        metric_type = 'staking_yield'
        if 'apr' in match.group(0).lower() or 'annual percentage rate' in match.group(0).lower():
            metric_type = 'apr'
        elif 'apy' in match.group(0).lower() or 'annual percentage yield' in match.group(0).lower():
            metric_type = 'apy'
        
        metrics[metric_type].append({
            "value": value,
            "original_text": match.group(0),
            "position": match.span()
        })
    
    # Lockup period pattern
    lockup_pattern = r'(?:lockup|lock-up|locking|vesting|staking)\s+period(?:\s+of)?\s+(\d+)\s+(days?|weeks?|months?|years?)'
    lockup_matches = re.finditer(lockup_pattern, text.lower())
    
    for match in lockup_matches:
        value = int(match.group(1))
        unit = match.group(2).rstrip('s')  # Remove plural 's'
        
        # Convert to days
        days = value
        if unit == 'week':
            days = value * 7
        elif unit == 'month':
            days = value * 30
        elif unit == 'year':
            days = value * 365
        
        metrics['lockup_period'].append({
            "value": value,
            "unit": unit,
            "days": days,
            "original_text": match.group(0),
            "position": match.span()
        })
    
    return dict(metrics)


def extract_financial_events(text: str) -> List[Dict[str, Union[str, List[str]]]]:
    """
    Extract financial events from text (listings, delistings, token burns, etc.).
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        List[Dict]: List of dictionaries with event information
    """
    events = []
    
    # Define event patterns
    event_patterns = {
        'listing': r'(?:listed|listing|launches|launches on|lists on|will list|will be listed)(?:\s+on)?\s+([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*)',
        'delisting': r'(?:delisted|delisting|removes|removes from|will delist|will be delisted)(?:\s+from)?\s+([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*)',
        'token_burn': r'(?:burns?|burning|burned|will burn)(?:\s+of)?\s+(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:million|billion|trillion|M|B|T)?\s+(?:tokens?|coins?|\$[A-Z]{2,5})',
        'airdrop': r'(?:airdrops?|distributes?|gives away)(?:\s+of)?\s+(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:million|billion|trillion|M|B|T)?\s+(?:tokens?|coins?|\$[A-Z]{2,5})',
        'fork': r'(?:fork(?:ing|ed)?|hard fork|soft fork)(?:\s+of)?\s+([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*)',
        'ico': r'(?:ICO|initial coin offering|token sale)(?:\s+of)?\s+([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*)',
        'partnership': r'(?:partners?|partnering|partnership|collaborates?|collaboration)(?:\s+with)?\s+([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*)'
    }
    
    # Extract events
    for event_type, pattern in event_patterns.items():
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            if match.group(1):
                entity = match.group(1).strip()
                
                # Special handling for token burn and airdrop amounts
                amount = None
                if event_type in ['token_burn', 'airdrop']:
                    amount_str = match.group(1).replace(',', '')
                    amount = float(amount_str)
                    
                    # Apply multiplier
                    multiplier = 1
                    if 'million' in match.group(0).lower() or 'M' in match.group(0):
                        multiplier = 1_000_000
                    elif 'billion' in match.group(0).lower() or 'B' in match.group(0):
                        multiplier = 1_000_000_000
                    elif 'trillion' in match.group(0).lower() or 'T' in match.group(0):
                        multiplier = 1_000_000_000_000
                    
                    amount *= multiplier
                
                event = {
                    "event_type": event_type,
                    "entity": entity,
                    "original_text": match.group(0),
                    "position": match.span()
                }
                
                if amount is not None:
                    event["amount"] = amount
                
                events.append(event)
    
    return events


def extract_sentiment_indicators(text: str) -> Dict[str, Union[float, str, List[str]]]:
    """
    Extract sentiment indicators from text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        Dict: Dictionary with sentiment indicators
    """
    # Define bullish and bearish terms
    bullish_terms = [
        'bullish', 'buy', 'long', 'support', 'upside', 'breakout', 'rally', 'surge',
        'positive', 'growth', 'gain', 'outperform', 'uptrend', 'recovery', 'momentum',
        'optimistic', 'confidence', 'strong', 'strength', 'boom', 'soar', 'climb'
    ]
    
    bearish_terms = [
        'bearish', 'sell', 'short', 'resistance', 'downside', 'breakdown', 'crash', 'dump',
        'negative', 'decline', 'loss', 'underperform', 'downtrend', 'weakness', 'pessimistic',
        'doubt', 'weak', 'drop', 'fall', 'plunge', 'collapse', 'tumble', 'slump'
    ]
    
    # Count occurrences of terms
    text_lower = text.lower()
    
    found_bullish_terms = []
    for term in bullish_terms:
        matches = re.finditer(r'\b' + re.escape(term) + r'\b', text_lower)
        for match in matches:
            found_bullish_terms.append({
                "term": term,
                "position": match.span()
            })
    
    found_bearish_terms = []
    for term in bearish_terms:
        matches = re.finditer(r'\b' + re.escape(term) + r'\b', text_lower)
        for match in matches:
            found_bearish_terms.append({
                "term": term,
                "position": match.span()
            })
    
    # Calculate sentiment score (-1 to 1)
    bullish_count = len(found_bullish_terms)
    bearish_count = len(found_bearish_terms)
    total_count = bullish_count + bearish_count
    
    sentiment_score = 0
    if total_count > 0:
        sentiment_score = (bullish_count - bearish_count) / total_count
    
    # Determine overall sentiment
    sentiment = 'neutral'
    if sentiment_score >= 0.2:
        sentiment = 'bullish'
    elif sentiment_score <= -0.2:
        sentiment = 'bearish'
    
    return {
        "sentiment_score": sentiment_score,
        "sentiment": sentiment,
        "bullish_terms": found_bullish_terms,
        "bearish_terms": found_bearish_terms,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count
    }


def analyze_text(text: str) -> Dict[str, any]:
    """
    Comprehensive analysis of financial text, extracting all relevant information.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        Dict: Dictionary with all extracted information
    """
    results = {
        "tickers": extract_ticker_symbols(text),
        "prices": extract_price_mentions(text),
        "percentages": extract_percentage_changes(text),
        "dates": extract_dates(text),
        "entities": extract_named_entities(text, ["PERSON", "ORG", "GPE", "MONEY"]),
        "financial_metrics": extract_financial_metrics(text),
        "token_metrics": extract_token_metrics(text),
        "events": extract_financial_events(text),
        "sentiment": extract_sentiment_indicators(text)
    }
    
    return results


def batch_analyze_texts(texts: List[str], include_original: bool = False) -> pd.DataFrame:
    """
    Analyze a batch of texts and return results as a DataFrame.
    
    Args:
        texts (List[str]): List of texts to analyze
        include_original (bool): Whether to include original texts in results
        
    Returns:
        pd.DataFrame: DataFrame with analysis results
    """
    results = []
    
    for i, text in enumerate(texts):
        analysis = analyze_text(text)
        
        # Flatten the analysis for DataFrame representation
        row = {
            'text_id': i,
            'tickers': ', '.join(analysis['tickers']) if analysis['tickers'] else None,
            'ticker_count': len(analysis['tickers']),
            'price_mentions': len(analysis['prices']),
            'percentage_mentions': len(analysis['percentages']),
            'date_mentions': len(analysis['dates']),
            'person_mentions': len(analysis['entities'].get('PERSON', [])),
            'org_mentions': len(analysis['entities'].get('ORG', [])),
            'sentiment_score': analysis['sentiment']['sentiment_score'],
            'sentiment': analysis['sentiment']['sentiment'],
            'bullish_term_count': analysis['sentiment']['bullish_count'],
            'bearish_term_count': analysis['sentiment']['bearish_count'],
            'event_count': len(analysis['events'])
        }
        
        # Add most common event types
        events_by_type = defaultdict(int)
        for event in analysis['events']:
            events_by_type[event['event_type']] += 1
        
        for event_type, count in events_by_type.items():
            row[f'event_{event_type}_count'] = count
        
        # Add original text if requested
        if include_original:
            row['original_text'] = text
        
        results.append(row)
    
    return pd.DataFrame(results)


def extract_information_from_file(file_path: str) -> Dict[str, any]:
    """
    Extract financial information from a text file.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        Dict: Dictionary with extracted information
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        return analyze_text(text)
    
    except Exception as e:
        print(f"Error extracting information from file: {e}")
        return {} 