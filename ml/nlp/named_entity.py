import spacy
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Set
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Try to load SpaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("SpaCy model not found. Please run: python -m spacy download en_core_web_md")
    # Create a blank model as fallback
    nlp = spacy.blank("en")


class CryptoEntityRecognizer:
    """
    Named Entity Recognition for cryptocurrency-related text
    """
    
    def __init__(self, custom_entities: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the entity recognizer
        
        Args:
            custom_entities (Dict[str, List[str]], optional): Custom entities to recognize
                Key: Entity type, Value: List of entity names
        """
        self.nlp = nlp
        
        # Default crypto-specific entities
        self.crypto_entities = {
            'BLOCKCHAIN': [
                'Bitcoin', 'Ethereum', 'Binance Smart Chain', 'Solana', 'Cardano', 'Polkadot',
                'Avalanche', 'Cosmos', 'Polygon', 'Tezos', 'Algorand', 'Fantom', 'Harmony',
                'BTC', 'ETH', 'BSC', 'SOL', 'ADA', 'DOT', 'AVAX', 'ATOM', 'MATIC', 'XTZ', 'ALGO',
                'FTM', 'ONE', 'Blockchain', 'Layer 1', 'Layer 2', 'L1', 'L2', 'Mainnet', 'Testnet'
            ],
            'TOKEN': [
                'USDT', 'USDC', 'DAI', 'BNB', 'XRP', 'DOGE', 'SHIB', 'LUNA', 'UNI', 'LINK',
                'Tether', 'USD Coin', 'Dai', 'Binance Coin', 'Ripple', 'Dogecoin', 'Shiba Inu',
                'Terra', 'Uniswap', 'Chainlink', 'Stablecoin', 'Memecoin', 'Governance token'
            ],
            'EXCHANGE': [
                'Binance', 'Coinbase', 'FTX', 'Kraken', 'Bitfinex', 'Huobi', 'KuCoin',
                'Gemini', 'Bitstamp', 'OKX', 'Bybit', 'Gate.io', 'Bittrex', 'Bithumb',
                'DEX', 'CEX', 'Decentralized Exchange', 'Centralized Exchange'
            ],
            'DEFI': [
                'Uniswap', 'SushiSwap', 'PancakeSwap', 'Curve', 'Aave', 'Compound', 'MakerDAO',
                'Yearn', 'Balancer', '1inch', 'Bancor', 'DeFi', 'Yield Farming', 'Liquidity Pool',
                'AMM', 'Automated Market Maker', 'Staking', 'Lending', 'Borrowing', 'LP Token',
                'Impermanent Loss', 'TVL', 'Total Value Locked', 'Yield', 'APY', 'APR'
            ],
            'NFT': [
                'NFT', 'Non-fungible Token', 'OpenSea', 'Rarible', 'Foundation', 'SuperRare',
                'Mintable', 'Nifty Gateway', 'NBA Top Shot', 'CryptoPunks', 'Bored Ape',
                'Azuki', 'Mintable', 'Minting', 'Floor Price', 'Royalty'
            ],
            'PERSON': [],  # Will use SpaCy's PERSON entities
            'ORG': []      # Will use SpaCy's ORG entities
        }
        
        # Add custom entities if provided
        if custom_entities:
            for entity_type, entity_list in custom_entities.items():
                if entity_type in self.crypto_entities:
                    self.crypto_entities[entity_type].extend(entity_list)
                else:
                    self.crypto_entities[entity_type] = entity_list
    
    def _create_entity_patterns(self):
        """
        Create entity matching patterns for use with spaCy
        
        Returns:
            List[Dict]: List of pattern dictionaries for spaCy's EntityRuler
        """
        patterns = []
        
        for entity_type, entities in self.crypto_entities.items():
            # Skip PERSON and ORG as they're handled by spaCy's model
            if entity_type in ['PERSON', 'ORG']:
                continue
                
            for entity in entities:
                # Create pattern for exact match
                pattern = {
                    "label": entity_type,
                    "pattern": entity
                }
                patterns.append(pattern)
                
                # Add lowercase version if it's different
                if entity.lower() != entity:
                    pattern = {
                        "label": entity_type,
                        "pattern": entity.lower()
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def add_custom_entities(self, entity_type: str, entities: List[str]):
        """
        Add custom entities to the recognizer
        
        Args:
            entity_type (str): Type of entity (e.g., 'BLOCKCHAIN', 'TOKEN')
            entities (List[str]): List of entity names
        """
        if entity_type in self.crypto_entities:
            self.crypto_entities[entity_type].extend(entities)
        else:
            self.crypto_entities[entity_type] = entities
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Union[str, Tuple[int, int]]]]]:
        """
        Extract entities from text
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, List[Dict]]: Dictionary with entity types as keys and lists of entities as values
        """
        if not isinstance(text, str) or not text.strip():
            return {}
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Extract entities recognized by spaCy
        entities = defaultdict(list)
        for ent in doc.ents:
            entities[ent.label_].append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        # Add custom entity recognition
        for entity_type, entity_list in self.crypto_entities.items():
            # Skip PERSON and ORG as they're handled by spaCy's model
            if entity_type in ['PERSON', 'ORG']:
                continue
                
            for entity in entity_list:
                # Look for exact matches (case insensitive)
                for match in re.finditer(r'\b' + re.escape(entity) + r'\b', text, re.IGNORECASE):
                    entities[entity_type].append({
                        "text": match.group(0),
                        "start": match.start(),
                        "end": match.end()
                    })
        
        # Convert defaultdict to regular dict
        return dict(entities)
    
    def extract_entity_relationships(self, text: str) -> List[Dict[str, Union[str, Dict]]]:
        """
        Extract relationships between entities
        
        Args:
            text (str): Input text
            
        Returns:
            List[Dict]: List of entity relationships
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # Process with spaCy for dependency parsing
        doc = self.nlp(text)
        
        # Find relationships between entities
        relationships = []
        
        # Extract all entity spans
        entity_spans = []
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                entity_spans.append({
                    "type": entity_type,
                    "text": entity["text"],
                    "start": entity["start"],
                    "end": entity["end"]
                })
        
        # Sort entity spans by start position
        entity_spans.sort(key=lambda x: x["start"])
        
        # Find consecutive entity pairs
        for i in range(len(entity_spans) - 1):
            entity1 = entity_spans[i]
            entity2 = entity_spans[i + 1]
            
            # Check if they're close enough (within 20 characters)
            if entity2["start"] - entity1["end"] <= 20:
                # Extract the text between entities
                relation_text = text[entity1["end"]:entity2["start"]]
                
                relationships.append({
                    "entity1": {
                        "type": entity1["type"],
                        "text": entity1["text"]
                    },
                    "entity2": {
                        "type": entity2["type"],
                        "text": entity2["text"]
                    },
                    "relation": relation_text.strip()
                })
        
        return relationships
    
    def extract_entities_from_dataframe(self, 
                                       df: pd.DataFrame, 
                                       text_column: str) -> pd.DataFrame:
        """
        Extract entities from texts in a DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame containing text data
            text_column (str): Name of the column containing text
            
        Returns:
            pd.DataFrame: DataFrame with entity counts added as columns
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        # Initialize entity count columns
        entity_types = list(self.crypto_entities.keys()) + ["GPE", "DATE", "MONEY", "PERCENT"]
        for entity_type in entity_types:
            df[f'{entity_type}_count'] = 0
            df[f'{entity_type}_entities'] = None
        
        # Extract entities for each row
        for i, row in df.iterrows():
            text = row[text_column]
            if not isinstance(text, str) or not text.strip():
                continue
                
            entities = self.extract_entities(text)
            
            # Update entity counts
            for entity_type, entity_list in entities.items():
                if f'{entity_type}_count' in df.columns:
                    df.at[i, f'{entity_type}_count'] = len(entity_list)
                    
                    # Store entity texts
                    entity_texts = [e["text"] for e in entity_list]
                    df.at[i, f'{entity_type}_entities'] = ", ".join(entity_texts) if entity_texts else None
        
        return df
    
    def extract_top_entities(self, 
                           texts: List[str], 
                           top_n: int = 10) -> Dict[str, List[Tuple[str, int]]]:
        """
        Extract top entities from a list of texts
        
        Args:
            texts (List[str]): List of texts to analyze
            top_n (int): Number of top entities to return for each type
            
        Returns:
            Dict[str, List[Tuple[str, int]]]: Dictionary with entity types as keys and lists of (entity, count) tuples
        """
        # Count entities
        entity_counts = defaultdict(lambda: defaultdict(int))
        
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
                
            entities = self.extract_entities(text)
            
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    entity_counts[entity_type][entity["text"].lower()] += 1
        
        # Get top entities for each type
        top_entities = {}
        for entity_type, counts in entity_counts.items():
            # Sort by count
            sorted_entities = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            top_entities[entity_type] = sorted_entities[:top_n]
        
        return top_entities
    
    def plot_entity_counts(self, 
                          entity_counts: Dict[str, List[Tuple[str, int]]], 
                          figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot entity counts
        
        Args:
            entity_counts (Dict[str, List[Tuple[str, int]]]): Dictionary with entity counts
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Determine number of entity types
        num_types = len(entity_counts)
        
        if num_types == 0:
            return plt.figure()
        
        # Calculate grid dimensions
        cols = min(3, num_types)
        rows = (num_types + cols - 1) // cols
        
        # Create figure and axes
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Flatten axes if there's more than one row
        if rows > 1:
            axes = axes.flatten()
        
        # Plot each entity type
        for i, (entity_type, entities) in enumerate(entity_counts.items()):
            # Skip if no entities
            if not entities:
                continue
                
            # Get axis
            if num_types == 1:
                ax = axes
            elif i < len(axes):
                ax = axes[i]
            else:
                break
            
            # Extract data for plotting
            labels = [e[0] for e in entities]
            values = [e[1] for e in entities]
            
            # Create bar chart
            ax.barh(labels, values)
            ax.set_title(f"{entity_type} Entities")
            ax.set_xlabel("Count")
            
            # Adjust tick labels if too long
            if any(len(label) > 15 for label in labels):
                ax.tick_params(axis='y', labelsize=8)
        
        # Hide unused subplots
        for i in range(num_types, len(axes) if isinstance(axes, np.ndarray) else 1):
            if isinstance(axes, np.ndarray):
                axes[i].set_visible(False)
        
        plt.tight_layout()
        
        return fig
    
    def analyze_entity_co_occurrence(self, 
                                   texts: List[str], 
                                   entity_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze entity co-occurrence in texts
        
        Args:
            texts (List[str]): List of texts to analyze
            entity_types (List[str], optional): Types of entities to include in co-occurrence analysis
            
        Returns:
            pd.DataFrame: Co-occurrence matrix
        """
        # Collect all unique entities
        all_entities = {}  # {entity_type: set(entity_text)}
        
        # Filter entity types if specified
        valid_entity_types = set(entity_types) if entity_types else None
        
        # Extract entities from all texts
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
                
            entities = self.extract_entities(text)
            
            for entity_type, entity_list in entities.items():
                # Skip if this entity type is not in valid_entity_types
                if valid_entity_types and entity_type not in valid_entity_types:
                    continue
                    
                # Initialize set for this entity type if not already
                if entity_type not in all_entities:
                    all_entities[entity_type] = set()
                
                # Add entity texts
                for entity in entity_list:
                    all_entities[entity_type].add(entity["text"].lower())
        
        # Create a dictionary to map entity texts to unique IDs
        entity_to_id = {}
        id_to_entity = {}
        id_to_type = {}
        
        next_id = 0
        for entity_type, entity_set in all_entities.items():
            for entity_text in entity_set:
                entity_to_id[(entity_type, entity_text)] = next_id
                id_to_entity[next_id] = entity_text
                id_to_type[next_id] = entity_type
                next_id += 1
        
        # Initialize co-occurrence matrix
        n_entities = len(entity_to_id)
        co_occurrence = np.zeros((n_entities, n_entities))
        
        # Fill co-occurrence matrix
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
                
            entities = self.extract_entities(text)
            
            # Collect entity IDs in this text
            entity_ids = []
            for entity_type, entity_list in entities.items():
                # Skip if this entity type is not in valid_entity_types
                if valid_entity_types and entity_type not in valid_entity_types:
                    continue
                    
                for entity in entity_list:
                    entity_text = entity["text"].lower()
                    if (entity_type, entity_text) in entity_to_id:
                        entity_ids.append(entity_to_id[(entity_type, entity_text)])
            
            # Update co-occurrence for all pairs of entities
            for i in range(len(entity_ids)):
                for j in range(i + 1, len(entity_ids)):
                    co_occurrence[entity_ids[i], entity_ids[j]] += 1
                    co_occurrence[entity_ids[j], entity_ids[i]] += 1
        
        # Create DataFrame
        co_occurrence_df = pd.DataFrame(co_occurrence)
        
        # Set row and column names
        co_occurrence_df.index = [f"{id_to_type[i]}:{id_to_entity[i]}" for i in range(n_entities)]
        co_occurrence_df.columns = [f"{id_to_type[i]}:{id_to_entity[i]}" for i in range(n_entities)]
        
        return co_occurrence_df
    
    def plot_entity_co_occurrence(self, 
                                co_occurrence_df: pd.DataFrame, 
                                threshold: int = 1,
                                figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Plot entity co-occurrence heatmap
        
        Args:
            co_occurrence_df (pd.DataFrame): Co-occurrence matrix from analyze_entity_co_occurrence
            threshold (int): Minimum co-occurrence count to include in plot
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Filter by threshold
        filtered_df = co_occurrence_df.copy()
        filtered_df[filtered_df < threshold] = 0
        
        # Remove rows and columns that are all zeros
        non_zero_rows = (filtered_df.sum(axis=1) > 0)
        non_zero_cols = (filtered_df.sum(axis=0) > 0)
        
        filtered_df = filtered_df.loc[non_zero_rows, non_zero_cols]
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Create heatmap
        ax = sns.heatmap(filtered_df, 
                         cmap='viridis', 
                         annot=True, 
                         fmt='.0f',
                         linewidths=0.5,
                         square=True)
        
        plt.title("Entity Co-occurrence Matrix")
        plt.tight_layout()
        
        return plt.gcf()
    
    def extract_entity_timeline(self, 
                              df: pd.DataFrame, 
                              text_column: str,
                              date_column: str,
                              entity_types: Optional[List[str]] = None,
                              freq: str = 'D') -> pd.DataFrame:
        """
        Extract entity mentions over time
        
        Args:
            df (pd.DataFrame): DataFrame containing text data
            text_column (str): Name of the column containing text
            date_column (str): Name of the column containing dates
            entity_types (List[str], optional): Types of entities to include
            freq (str): Frequency for grouping ('D' = daily, 'W' = weekly, etc.)
            
        Returns:
            pd.DataFrame: DataFrame with entity mentions over time
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        if date_column not in df.columns:
            raise ValueError(f"Column '{date_column}' not found in DataFrame")
        
        # Ensure date column is datetime
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Extract entities for each row
        entity_mentions = defaultdict(lambda: defaultdict(int))
        
        for i, row in df.iterrows():
            text = row[text_column]
            date = row[date_column]
            
            if not isinstance(text, str) or not text.strip() or pd.isna(date):
                continue
            
            entities = self.extract_entities(text)
            
            for entity_type, entity_list in entities.items():
                # Skip if entity_types is specified and this type is not in it
                if entity_types and entity_type not in entity_types:
                    continue
                
                for entity in entity_list:
                    entity_text = entity["text"].lower()
                    entity_key = f"{entity_type}:{entity_text}"
                    
                    entity_mentions[date][entity_key] += 1
        
        # Convert to DataFrame
        dates = sorted(entity_mentions.keys())
        all_entity_keys = set()
        
        for date_mentions in entity_mentions.values():
            all_entity_keys.update(date_mentions.keys())
        
        all_entity_keys = sorted(all_entity_keys)
        
        timeline_data = []
        for date in dates:
            row = {'date': date}
            for entity_key in all_entity_keys:
                row[entity_key] = entity_mentions[date].get(entity_key, 0)
            timeline_data.append(row)
        
        timeline_df = pd.DataFrame(timeline_data)
        
        # Group by frequency
        if len(timeline_df) > 0:
            timeline_df = timeline_df.set_index('date')
            timeline_df = timeline_df.groupby(pd.Grouper(freq=freq)).sum()
            timeline_df = timeline_df.reset_index()
        
        return timeline_df 