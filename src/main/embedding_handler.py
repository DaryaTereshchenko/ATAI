"""
Embedding Handler for TransE-based Knowledge Graph Query Processing.
Loads and manages TransE embeddings for entities and relations.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle


class EmbeddingHandler:
    """Manages TransE embeddings for entities and relations from the knowledge graph."""
    
    def __init__(self, embeddings_dir: str):
        """
        Initialize the embedding handler.
        
        Args:
            embeddings_dir: Directory containing embedding files
                - entity_embeds.npy: Entity embeddings
                - entity_ids.del: Entity ID mappings
                - relation_embeds.npy: Relation embeddings
                - relation_ids.del: Relation ID mappings
        """
        self.embeddings_dir = embeddings_dir
        
        # Load embeddings and mappings
        self.entity_embeddings = None
        self.relation_embeddings = None
        self.entity_id_to_uri = {}
        self.entity_uri_to_id = {}
        self.relation_id_to_uri = {}
        self.relation_uri_to_id = {}
        
        self._load_embeddings()
        
        print(f"✅ Loaded {len(self.entity_id_to_uri)} entity embeddings")
        print(f"✅ Loaded {len(self.relation_id_to_uri)} relation embeddings")
        print(f"   Entity embedding dimension: {self.entity_embeddings.shape[1]}")
        print(f"   Relation embedding dimension: {self.relation_embeddings.shape[1]}")
    
    def _load_embeddings(self):
        """Load embeddings and ID mappings from files."""
        # Load entity embeddings
        entity_embeds_path = os.path.join(self.embeddings_dir, "entity_embeds.npy")
        entity_ids_path = os.path.join(self.embeddings_dir, "entity_ids.del")
        
        if not os.path.exists(entity_embeds_path):
            raise FileNotFoundError(f"Entity embeddings not found: {entity_embeds_path}")
        if not os.path.exists(entity_ids_path):
            raise FileNotFoundError(f"Entity IDs not found: {entity_ids_path}")
        
        self.entity_embeddings = np.load(entity_embeds_path)
        self.entity_id_to_uri, self.entity_uri_to_id = self._load_id_mappings(entity_ids_path)
        
        # Load relation embeddings
        relation_embeds_path = os.path.join(self.embeddings_dir, "relation_embeds.npy")
        relation_ids_path = os.path.join(self.embeddings_dir, "relation_ids.del")
        
        if not os.path.exists(relation_embeds_path):
            raise FileNotFoundError(f"Relation embeddings not found: {relation_embeds_path}")
        if not os.path.exists(relation_ids_path):
            raise FileNotFoundError(f"Relation IDs not found: {relation_ids_path}")
        
        self.relation_embeddings = np.load(relation_embeds_path)
        self.relation_id_to_uri, self.relation_uri_to_id = self._load_id_mappings(relation_ids_path)
    
    def _load_id_mappings(self, filepath: str) -> Tuple[Dict[int, str], Dict[str, int]]:
        """
        Load ID mappings from .del file.
        
        Args:
            filepath: Path to .del file
            
        Returns:
            Tuple of (id_to_uri, uri_to_id) dictionaries
        """
        id_to_uri = {}
        uri_to_id = {}
        
        try:
            # Try pickle format first
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    id_to_uri = data
                    uri_to_id = {v: k for k, v in data.items()}
                    return id_to_uri, uri_to_id
        except:
            pass
        
        # Try text format: "id\turi\n"
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        entity_id = int(parts[0])
                        uri = parts[1]
                        id_to_uri[entity_id] = uri
                        uri_to_id[uri] = entity_id
            return id_to_uri, uri_to_id
        except:
            pass
        
        # Try space-separated format
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        entity_id = int(parts[0])
                        uri = parts[1]
                        id_to_uri[entity_id] = uri
                        uri_to_id[uri] = entity_id
            return id_to_uri, uri_to_id
        except Exception as e:
            raise ValueError(f"Could not parse ID mapping file {filepath}: {e}")
    
    def get_entity_embedding(self, entity_uri: str) -> Optional[np.ndarray]:
        """
        Get embedding for an entity by URI.
        
        Args:
            entity_uri: Entity URI (e.g., "http://www.wikidata.org/entity/Q123")
            
        Returns:
            Entity embedding vector or None if not found
        """
        entity_id = self.entity_uri_to_id.get(entity_uri)
        if entity_id is None:
            return None
        return self.entity_embeddings[entity_id]
    
    def get_relation_embedding(self, relation_uri: str) -> Optional[np.ndarray]:
        """
        Get embedding for a relation by URI.
        
        Args:
            relation_uri: Relation URI (e.g., "http://www.wikidata.org/prop/direct/P57")
            
        Returns:
            Relation embedding vector or None if not found
        """
        relation_id = self.relation_uri_to_id.get(relation_uri)
        if relation_id is None:
            return None
        return self.relation_embeddings[relation_id]
    
    def find_nearest_entities(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 10,
        filter_uris: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Find nearest entities to a query embedding using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            filter_uris: Optional list of entity URIs to restrict search to
            
        Returns:
            List of (entity_uri, similarity_score) tuples, sorted by similarity (descending)
        """
        # Normalize query embedding
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        # If filter is provided, compute similarities only for those entities
        if filter_uris:
            entity_ids = [self.entity_uri_to_id[uri] for uri in filter_uris if uri in self.entity_uri_to_id]
            if not entity_ids:
                return []
            
            entity_embeds = self.entity_embeddings[entity_ids]
            entity_norms = entity_embeds / (np.linalg.norm(entity_embeds, axis=1, keepdims=True) + 1e-10)
            similarities = np.dot(entity_norms, query_norm)
            
            # Create results
            results = [(filter_uris[i], float(similarities[i])) for i in range(len(filter_uris)) if filter_uris[i] in self.entity_uri_to_id]
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        
        # Otherwise, compute similarities for all entities
        entity_norms = self.entity_embeddings / (np.linalg.norm(self.entity_embeddings, axis=1, keepdims=True) + 1e-10)
        similarities = np.dot(entity_norms, query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Map back to URIs
        results = [(self.entity_id_to_uri[idx], float(similarities[idx])) for idx in top_indices]
        return results
    
    def find_nearest_relations(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find nearest relations to a query embedding using cosine similarity.
        Useful for detecting which relation (director, actor, genre, etc.) the query asks about.
        
        Args:
            query_embedding: Query embedding vector (should be in TransE space)
            top_k: Number of top relations to return
            
        Returns:
            List of (relation_uri, similarity_score) tuples, sorted by similarity
        """
        # Normalize query embedding
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        # Normalize relation embeddings
        relation_norms = self.relation_embeddings / (np.linalg.norm(self.relation_embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Compute cosine similarities
        similarities = np.dot(relation_norms, query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Map back to URIs
        results = [(self.relation_id_to_uri[idx], float(similarities[idx])) for idx in top_indices]
        return results
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of entity embeddings."""
        return self.entity_embeddings.shape[1]
    
    def get_entity_label(self, entity_uri: str, graph=None) -> str:
        """
        Get human-readable label for an entity.
        
        Args:
            entity_uri: Entity URI
            graph: Optional RDFLib graph to query for labels
            
        Returns:
            Entity label or URI if label not found
        """
        if graph is None:
            return entity_uri
        
        # Try to get rdfs:label
        from rdflib import URIRef, RDFS
        entity_ref = URIRef(entity_uri)
        
        for label in graph.objects(entity_ref, RDFS.label):
            return str(label)
        
        # Fallback to URI
        return entity_uri
    
    def get_entities_by_type(self, entity_type_uri: str, graph) -> List[str]:
        """
        Get all entities of a specific type from the graph.
        
        Args:
            entity_type_uri: Type URI (e.g., "http://www.wikidata.org/entity/Q11424" for movies)
            graph: RDFLib graph
            
        Returns:
            List of entity URIs
        """
        from rdflib import URIRef
        from rdflib.namespace import RDF
        
        type_ref = URIRef(entity_type_uri)
        entities = []
        
        # Query: ?entity rdf:type <type_uri> or ?entity wdt:P31 <type_uri>
        P31 = URIRef("http://www.wikidata.org/prop/direct/P31")
        
        for entity in graph.subjects(P31, type_ref):
            entity_uri = str(entity)
            # Only include entities that have embeddings
            if entity_uri in self.entity_uri_to_id:
                entities.append(entity_uri)
        
        for entity in graph.subjects(RDF.type, type_ref):
            entity_uri = str(entity)
            if entity_uri in self.entity_uri_to_id:
                entities.append(entity_uri)
        
        return list(set(entities))  # Remove duplicates
    
    def get_entity_type_qcode(self, entity_uri: str, graph) -> Optional[str]:
        """
        Get the Wikidata Q-code for an entity's type.
        
        Args:
            entity_uri: Entity URI
            graph: RDFLib graph
            
        Returns:
            Q-code string (e.g., 'Q201658') or None
        """
        from rdflib import URIRef
        
        entity_ref = URIRef(entity_uri)
        P31 = URIRef("http://www.wikidata.org/prop/direct/P31")
        
        for type_uri in graph.objects(entity_ref, P31):
            type_str = str(type_uri)
            if '/Q' in type_str:
                # Extract Q-code
                return type_str.split('/Q')[-1].split('#')[0]
        
        return None
