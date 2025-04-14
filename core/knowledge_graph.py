"""
Knowledge Graph - Market knowledge representation.

This module provides a graph-based representation of market knowledge,
relationships, and insights to support advanced market analysis and
decision making in the trading system.
"""

import logging
import networkx as nx
from typing import Dict, List, Set, Tuple, Any, Optional, Union
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """
    Knowledge graph for representing market knowledge, entities,
    relationships, and insights.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the knowledge graph.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.graph = nx.DiGraph()
        self.entity_types = set(['asset', 'exchange', 'indicator', 'news', 'event',
                              'person', 'company', 'concept', 'strategy', 'protocol'])
        self.relation_types = set(['correlates_with', 'influences', 'part_of',
                                'traded_on', 'developed_by', 'affects', 'owns',
                                'stronger_than', 'precedes', 'follows', 'signals'])
        
        # Configure persistence
        self.storage_directory = config.get('knowledge_graph_dir', 'knowledge')
        self.auto_save = config.get('auto_save_graph', False)
        self.auto_save_interval = config.get('auto_save_interval', 3600)  # seconds
        self.last_saved = datetime.now()
        
        os.makedirs(self.storage_directory, exist_ok=True)
        
        # Try to load existing graph
        self._load_graph()
        
        logger.info("Knowledge graph initialized")
    
    def add_entity(self, entity_id: str, entity_type: str, 
                  attributes: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an entity to the knowledge graph.
        
        Args:
            entity_id: Unique entity identifier
            entity_type: Type of entity
            attributes: Entity attributes
        """
        if entity_type not in self.entity_types:
            logger.warning(f"Unknown entity type: {entity_type}")
        
        if self.graph.has_node(entity_id):
            # Update existing entity
            current_attrs = self.graph.nodes[entity_id]
            if attributes:
                for k, v in attributes.items():
                    current_attrs[k] = v
        else:
            # Add new entity
            attrs = {'type': entity_type, 'created_at': datetime.now()}
            if attributes:
                attrs.update(attributes)
            self.graph.add_node(entity_id, **attrs)
            
        logger.debug(f"Added/updated entity: {entity_id} of type {entity_type}")
        
        if self.auto_save and (datetime.now() - self.last_saved).seconds >= self.auto_save_interval:
            self._save_graph()
    
    def add_relation(self, source_id: str, target_id: str, relation_type: str,
                    weight: float = 1.0, attributes: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a relationship between entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relation_type: Type of relationship
            weight: Relationship strength/weight
            attributes: Additional relationship attributes
        """
        if relation_type not in self.relation_types:
            logger.warning(f"Unknown relation type: {relation_type}")
            
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            logger.warning(f"Cannot create relation - missing entity: {source_id} or {target_id}")
            return
            
        attrs = {
            'type': relation_type,
            'weight': weight,
            'created_at': datetime.now()
        }
        
        if attributes:
            attrs.update(attributes)
            
        self.graph.add_edge(source_id, target_id, **attrs)
        logger.debug(f"Added relation: {source_id} --[{relation_type}]--> {target_id}")
        
        if self.auto_save and (datetime.now() - self.last_saved).seconds >= self.auto_save_interval:
            self._save_graph()
    
    def update_relation_weight(self, source_id: str, target_id: str, 
                              weight_delta: float) -> None:
        """
        Update the weight of a relationship based on new evidence.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            weight_delta: Change in weight (positive or negative)
        """
        if not self.graph.has_edge(source_id, target_id):
            logger.warning(f"Relation does not exist: {source_id} -> {target_id}")
            return
            
        current_weight = self.graph[source_id][target_id].get('weight', 1.0)
        new_weight = current_weight + weight_delta
        
        # Ensure weight stays in reasonable bounds
        new_weight = max(0.1, min(10.0, new_weight))
        
        self.graph[source_id][target_id]['weight'] = new_weight
        self.graph[source_id][target_id]['updated_at'] = datetime.now()
        
        logger.debug(f"Updated relation weight: {source_id} -> {target_id}, {current_weight} -> {new_weight}")
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get entity information.
        
        Args:
            entity_id: Entity ID to retrieve
            
        Returns:
            Entity attributes or None if not found
        """
        if self.graph.has_node(entity_id):
            return dict(self.graph.nodes[entity_id])
        return None
    
    def get_entities_by_type(self, entity_type: str) -> List[str]:
        """
        Get all entities of a specific type.
        
        Args:
            entity_type: Type of entities to retrieve
            
        Returns:
            List of entity IDs
        """
        return [n for n, attrs in self.graph.nodes(data=True)
                if attrs.get('type') == entity_type]
    
    def get_relations(self, entity_id: str, relation_type: Optional[str] = None,
                     direction: str = 'outgoing') -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Get relations for an entity.
        
        Args:
            entity_id: Entity ID
            relation_type: Optional filter by relation type
            direction: 'outgoing', 'incoming', or 'both'
            
        Returns:
            List of (source, target, attributes) tuples
        """
        relations = []
        
        if direction in ('outgoing', 'both'):
            for target in self.graph.successors(entity_id):
                edge_data = self.graph[entity_id][target]
                if relation_type is None or edge_data.get('type') == relation_type:
                    relations.append((entity_id, target, dict(edge_data)))
                    
        if direction in ('incoming', 'both'):
            for source in self.graph.predecessors(entity_id):
                edge_data = self.graph[source][entity_id]
                if relation_type is None or edge_data.get('type') == relation_type:
                    relations.append((source, entity_id, dict(edge_data)))
                    
        return relations
    
    def find_path(self, source_id: str, target_id: str, 
                 max_hops: int = 3) -> List[List[Tuple[str, str, str]]]:
        """
        Find connection paths between two entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            max_hops: Maximum path length
            
        Returns:
            List of paths, each path is a list of (source, relation, target) tuples
        """
        paths = []
        
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            return paths
            
        # Find simple paths with limited length
        for path in nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_hops):
            if len(path) > 1:
                path_details = []
                for i in range(len(path) - 1):
                    start, end = path[i], path[i + 1]
                    relation_type = self.graph[start][end].get('type', 'unknown')
                    path_details.append((start, relation_type, end))
                paths.append(path_details)
                
        return paths
    
    def query_related_entities(self, entity_id: str, max_hops: int = 2) -> Dict[str, List[str]]:
        """
        Get all entities related to the given entity within N hops.
        
        Args:
            entity_id: Center entity ID
            max_hops: Maximum path length
            
        Returns:
            Dictionary of related entities by type
        """
        if not self.graph.has_node(entity_id):
            return {}
            
        # Use BFS to find all nodes within max_hops
        related = set()
        visited = {entity_id}
        queue = [(entity_id, 0)]  # (node, distance)
        
        while queue:
            node, dist = queue.pop(0)
            
            if dist < max_hops:
                # Check outgoing links
                for successor in self.graph.successors(node):
                    if successor not in visited:
                        visited.add(successor)
                        related.add(successor)
                        queue.append((successor, dist + 1))
                        
                # Check incoming links
                for predecessor in self.graph.predecessors(node):
                    if predecessor not in visited:
                        visited.add(predecessor)
                        related.add(predecessor)
                        queue.append((predecessor, dist + 1))
        
        # Group by entity type
        result = {}
        for node_id in related:
            node_type = self.graph.nodes[node_id].get('type', 'unknown')
            if node_type not in result:
                result[node_type] = []
            result[node_type].append(node_id)
            
        return result
    
    def find_similar_entities(self, entity_id: str, 
                             min_similarity: float = 0.3) -> List[Tuple[str, float]]:
        """
        Find entities that are similar to the given entity based on shared connections.
        
        Args:
            entity_id: Target entity ID
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (entity_id, similarity_score) tuples
        """
        if not self.graph.has_node(entity_id):
            return []
            
        # Get neighbors of the target entity
        target_neighbors = set(self.graph.successors(entity_id)) | set(self.graph.predecessors(entity_id))
        if not target_neighbors:
            return []
            
        # Get all entities of the same type
        entity_type = self.graph.nodes[entity_id].get('type')
        similar_entities = []
        
        for candidate in self.get_entities_by_type(entity_type):
            if candidate == entity_id:
                continue
                
            # Get candidate neighbors
            candidate_neighbors = set(self.graph.successors(candidate)) | set(self.graph.predecessors(candidate))
            if not candidate_neighbors:
                continue
                
            # Calculate Jaccard similarity
            intersection = len(target_neighbors & candidate_neighbors)
            union = len(target_neighbors | candidate_neighbors)
            similarity = intersection / union if union > 0 else 0
            
            if similarity >= min_similarity:
                similar_entities.append((candidate, similarity))
                
        # Sort by similarity
        return sorted(similar_entities, key=lambda x: x[1], reverse=True)
    
    def get_influential_entities(self, entity_type: Optional[str] = None,
                                limit: int = 10) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Find the most influential entities (highest centrality).
        
        Args:
            entity_type: Optional entity type to filter
            limit: Maximum number of results
            
        Returns:
            List of (entity_id, metrics) tuples
        """
        # Calculate centrality measures
        degrees = dict(self.graph.degree())
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())
        
        try:
            # These can be computationally expensive for large graphs
            betweenness = nx.betweenness_centrality(self.graph)
            pagerank = nx.pagerank(self.graph)
        except Exception as e:
            logger.warning(f"Error calculating graph metrics: {str(e)}")
            betweenness = {node: 0.0 for node in self.graph.nodes()}
            pagerank = {node: 1.0 / len(self.graph) for node in self.graph.nodes()}
        
        # Combine metrics
        entities = []
        for node in self.graph.nodes():
            if entity_type and self.graph.nodes[node].get('type') != entity_type:
                continue
                
            metrics = {
                'degree': degrees.get(node, 0),
                'in_degree': in_degrees.get(node, 0),
                'out_degree': out_degrees.get(node, 0),
                'betweenness': betweenness.get(node, 0),
                'pagerank': pagerank.get(node, 0),
                'score': pagerank.get(node, 0) + 0.5 * betweenness.get(node, 0)
            }
            
            entities.append((node, metrics))
            
        # Sort by combined score
        sorted_entities = sorted(entities, key=lambda x: x[1]['score'], reverse=True)
        return sorted_entities[:limit]
    
    def find_communities(self) -> Dict[int, List[str]]:
        """
        Identify communities within the knowledge graph.
        
        Returns:
            Dictionary mapping community ID to list of entity IDs
        """
        try:
            communities = nx.community.greedy_modularity_communities(self.graph.to_undirected())
            result = {}
            for i, community in enumerate(communities):
                result[i] = list(community)
            return result
        except Exception as e:
            logger.error(f"Error finding communities: {str(e)}")
            return {}
    
    def export_subgraph(self, entity_ids: List[str], 
                       include_neighbors: bool = False) -> Dict[str, Any]:
        """
        Export a subgraph containing specified entities.
        
        Args:
            entity_ids: List of entity IDs to include
            include_neighbors: Whether to include direct neighbors
            
        Returns:
            Dictionary with nodes and edges for visualization
        """
        nodes_to_include = set(entity_ids)
        
        # Add neighbors if requested
        if include_neighbors:
            neighbors = set()
            for entity_id in entity_ids:
                if self.graph.has_node(entity_id):
                    neighbors.update(self.graph.successors(entity_id))
                    neighbors.update(self.graph.predecessors(entity_id))
            nodes_to_include.update(neighbors)
        
        # Create subgraph
        node_list = []
        for node in nodes_to_include:
            if self.graph.has_node(node):
                attrs = dict(self.graph.nodes[node])
                node_list.append({
                    'id': node,
                    'type': attrs.get('type', 'unknown'),
                    'attributes': attrs
                })
                
        edge_list = []
        for source in nodes_to_include:
            for target in self.graph.successors(source):
                if target in nodes_to_include:
                    attrs = dict(self.graph[source][target])
                    edge_list.append({
                        'source': source,
                        'target': target,
                        'type': attrs.get('type', 'unknown'),
                        'weight': attrs.get('weight', 1.0),
                        'attributes': attrs
                    })
                    
        return {
            'nodes': node_list,
            'edges': edge_list
        }
    
    def _save_graph(self) -> None:
        """
        Save the knowledge graph to disk.
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.storage_directory}/knowledge_graph_{timestamp}.json"
            
            # Prepare data for serialization
            data = {
                'nodes': [],
                'edges': []
            }
            
            for node, attrs in self.graph.nodes(data=True):
                node_data = {'id': node}
                
                # Handle non-serializable attributes
                serializable_attrs = {}
                for k, v in attrs.items():
                    if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                        serializable_attrs[k] = v
                    elif isinstance(v, datetime):
                        serializable_attrs[k] = v.isoformat()
                    else:
                        serializable_attrs[k] = str(v)
                        
                node_data['attributes'] = serializable_attrs
                data['nodes'].append(node_data)
                
            for source, target, attrs in self.graph.edges(data=True):
                edge_data = {
                    'source': source,
                    'target': target
                }
                
                # Handle non-serializable attributes
                serializable_attrs = {}
                for k, v in attrs.items():
                    if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                        serializable_attrs[k] = v
                    elif isinstance(v, datetime):
                        serializable_attrs[k] = v.isoformat()
                    else:
                        serializable_attrs[k] = str(v)
                        
                edge_data['attributes'] = serializable_attrs
                data['edges'].append(edge_data)
                
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.last_saved = datetime.now()
            logger.info(f"Saved knowledge graph to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {str(e)}")
    
    def _load_graph(self) -> None:
        """
        Load the knowledge graph from disk.
        """
        try:
            # Find the most recent graph file
            files = [f for f in os.listdir(self.storage_directory) 
                    if f.startswith('knowledge_graph_') and f.endswith('.json')]
                    
            if not files:
                logger.info("No existing knowledge graph found. Starting with empty graph.")
                return
                
            latest_file = sorted(files)[-1]
            filepath = os.path.join(self.storage_directory, latest_file)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Clear existing graph
            self.graph.clear()
            
            # Add nodes with attributes
            for node_data in data.get('nodes', []):
                node_id = node_data['id']
                attributes = node_data.get('attributes', {})
                
                # Convert back datetime strings
                for k, v in attributes.items():
                    if k in ('created_at', 'updated_at') and isinstance(v, str):
                        try:
                            attributes[k] = datetime.fromisoformat(v)
                        except ValueError:
                            pass
                            
                self.graph.add_node(node_id, **attributes)
                
            # Add edges with attributes
            for edge_data in data.get('edges', []):
                source = edge_data['source']
                target = edge_data['target']
                attributes = edge_data.get('attributes', {})
                
                # Convert back datetime strings
                for k, v in attributes.items():
                    if k in ('created_at', 'updated_at') and isinstance(v, str):
                        try:
                            attributes[k] = datetime.fromisoformat(v)
                        except ValueError:
                            pass
                            
                self.graph.add_edge(source, target, **attributes)
                
            logger.info(f"Loaded knowledge graph from {filepath} with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {str(e)}")
            # Initialize with empty graph on error
            self.graph.clear() 