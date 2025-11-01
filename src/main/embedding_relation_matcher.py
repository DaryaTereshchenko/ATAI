"""
Embedding-based relation matching using TransE relation embeddings.
Finds the closest relation to a query by computing cosine similarity.
"""

import numpy as np
from typing import List, Tuple, Optional
import re


class EmbeddingRelationMatcher:
    """Match natural language queries to relations using TransE embeddings."""
    
    def __init__(self, embedding_handler, query_embedder, aligner):
        """
        Initialize relation matcher.
        
        Args:
            embedding_handler: EmbeddingHandler with relation embeddings
            query_embedder: QueryEmbedder for NL queries
            aligner: EmbeddingAligner to align query space to TransE space
        """
        self.embedding_handler = embedding_handler
        self.query_embedder = query_embedder
        self.aligner = aligner
        
        # Build relation name cache
        self._build_relation_cache()
    
    def _build_relation_cache(self):
        """Build mapping from relation URIs to friendly names."""
        self.relation_uri_to_name = {}
        
        # Map Wikidata properties to names
        property_names = {
            'P57': 'director',
            'P161': 'cast_member',
            'P58': 'screenwriter',
            'P162': 'producer',
            'P136': 'genre',
            'P577': 'publication_date',
            'P495': 'country_of_origin',
            'P364': 'original_language',
            'P166': 'award_received',
        }
        
        # Populate cache
        for relation_uri in self.embedding_handler.relation_uri_to_id.keys():
            if '/P' in relation_uri:
                prop_id = relation_uri.split('/P')[-1].split('#')[0]
                name = property_names.get(prop_id, f"property_{prop_id}")
                self.relation_uri_to_name[relation_uri] = name
            elif 'ddis.ch/atai/' in relation_uri:
                name = relation_uri.split('/')[-1]
                self.relation_uri_to_name[relation_uri] = name
    
    def match_relation(
        self, 
        query: str, 
        extracted_entities: Optional[List[str]] = None,
        top_k: int = 5
    ) -> Tuple[str, str, float]:
        """
        Match query to most similar relation.
        """
        print(f"[EmbeddingRelationMatcher] 🔍 Matching relation for: '{query[:60]}...'")
        
        # Step 1: Clean query by removing entities
        clean_query = self._remove_entities(query, extracted_entities)
        print(f"[EmbeddingRelationMatcher] Cleaned query: '{clean_query}'")
        
        # Step 2: Embed cleaned query
        query_embedding = self.query_embedder.embed_query(clean_query)
        
        # Step 3: Align to TransE space
        aligned_embedding = self.aligner.align(query_embedding)
        
        # Step 4: Find nearest relations
        nearest_relations = self.embedding_handler.find_nearest_relations(
            aligned_embedding,
            top_k=top_k
        )
        
        if not nearest_relations:
            print(f"[EmbeddingRelationMatcher] ❌ No relations found")
            return None, None, 0.0
        
        # Step 5: Pick best match
        best_uri, best_similarity = nearest_relations[0]
        best_name = self.relation_uri_to_name.get(best_uri, best_uri)
        
        # ✅ Add logging for transparency
        print(f"[EmbeddingRelationMatcher] ✅ Matched Relation:")
        print(f"[EmbeddingRelationMatcher]    • Name: {best_name}")
        print(f"[EmbeddingRelationMatcher]    • URI: {best_uri}")
        print(f"[EmbeddingRelationMatcher]    • Confidence: {best_similarity:.3f}")
        print(f"[EmbeddingRelationMatcher]    • Method: TransE Embedding Similarity")
        
        print(f"[EmbeddingRelationMatcher]    Top Alternatives:")
        for i, (uri, sim) in enumerate(nearest_relations[1:3], 2):
            name = self.relation_uri_to_name.get(uri, uri)
            print(f"[EmbeddingRelationMatcher]      {i}. {name} ({sim:.3f})")
        
        return best_name, best_uri, best_similarity
    
    def _remove_entities(
        self, 
        query: str, 
        entities: Optional[List[str]]
    ) -> str:
        """
        Remove entity mentions from query to isolate relation.
        
        Args:
            query: Original query
            entities: List of entity strings to remove
            
        Returns:
            Query with entities removed
        """
        clean = query
        
        if entities:
            for entity in entities:
                # Remove quoted entity
                clean = re.sub(rf'["\']?{re.escape(entity)}["\']?', '', clean, flags=re.IGNORECASE)
        
        # Remove common question words and stopwords
        stopwords = ['the', 'a', 'an', 'is', 'was', 'are', 'were', 'of', 'in', 'for']
        for word in stopwords:
            clean = re.sub(rf'\b{word}\b', '', clean, flags=re.IGNORECASE)
        
        # Clean up whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        return clean
