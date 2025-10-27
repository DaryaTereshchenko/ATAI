"""
Embedding-based Answer Finder.

Uses TransE embeddings to find answers through vector arithmetic:
- Extract subject entity and relation from query
- Compute: subject_embedding + relation_embedding ≈ answer_embedding
- Find nearest entity to the computed embedding
"""

import sys
import os
from typing import Optional, Tuple, Dict, List
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)


class EmbeddingAnswerFinder:
    """Finds answers using TransE embedding arithmetic."""
    
    # Mapping from natural language relations to Wikidata properties
    RELATION_MAPPINGS = {
        "director": "http://www.wikidata.org/prop/direct/P57",
        "screenwriter": "http://www.wikidata.org/prop/direct/P58",
        "cast member": "http://www.wikidata.org/prop/direct/P161",
        "actor": "http://www.wikidata.org/prop/direct/P161",
        "producer": "http://www.wikidata.org/prop/direct/P162",
        "genre": "http://www.wikidata.org/prop/direct/P136",
        "country": "http://www.wikidata.org/prop/direct/P495",
        "publication date": "http://www.wikidata.org/prop/direct/P577",
        "release date": "http://www.wikidata.org/prop/direct/P577",
    }
    
    def __init__(self, embedding_handler, entity_extractor, sparql_handler):
        """
        Initialize the embedding answer finder.
        
        Args:
            embedding_handler: EmbeddingHandler with TransE embeddings
            entity_extractor: EntityExtractor for finding entities
            sparql_handler: SPARQLHandler for entity resolution
        """
        self.embedding_handler = embedding_handler
        self.entity_extractor = entity_extractor
        self.sparql_handler = sparql_handler
    
    def find_answer_by_embeddings(
        self, 
        question: str
    ) -> Tuple[Optional[str], Optional[str], float]:
        """
        Find answer using embedding-based similarity.
        
        Uses TransE formula: head + relation ≈ tail
        
        Args:
            question: Natural language question
            
        Returns:
            Tuple of (answer_label, entity_type, confidence)
        """
        print("\n🔢 Finding answer using TransE embeddings...")
        
        try:
            # Step 1: Extract subject entity from question
            subject_uri, subject_label = self._extract_subject_entity(question)
            if not subject_uri:
                print("❌ Could not extract subject entity")
                return None, None, 0.0
            
            print(f"   Subject: {subject_label} ({subject_uri})")
            
            # Step 2: Detect relation from question
            relation_uri = self._detect_relation(question)
            if not relation_uri:
                print("❌ Could not detect relation")
                return None, None, 0.0
            
            print(f"   Relation: {relation_uri}")
            
            # Step 3: Get embeddings
            subject_emb = self.embedding_handler.get_entity_embedding(subject_uri)
            relation_emb = self.embedding_handler.get_relation_embedding(relation_uri)
            
            if subject_emb is None:
                print(f"❌ No embedding found for subject: {subject_uri}")
                return None, None, 0.0
            
            if relation_emb is None:
                print(f"❌ No embedding found for relation: {relation_uri}")
                return None, None, 0.0
            
            # Step 4: Compute expected answer embedding using TransE
            # TransE formula: tail ≈ head + relation
            expected_answer_emb = subject_emb + relation_emb
            
            print(f"   Computing: answer ≈ subject + relation")
            
            # Step 5: Find nearest entity to expected answer
            nearest = self.embedding_handler.find_nearest_entities(
                expected_answer_emb,
                top_k=5
            )
            
            if not nearest:
                print("❌ No nearest entities found")
                return None, None, 0.0
            
            print(f"\n   Top 5 nearest entities:")
            for i, (uri, score) in enumerate(nearest, 1):
                label = self._get_entity_label(uri)
                print(f"   {i}. {label} ({uri[-10:]}) - similarity: {score:.4f}")
            
            # Return top answer
            top_uri, top_score = nearest[0]
            top_label = self._get_entity_label(top_uri)
            entity_type = self._get_entity_type(top_uri, question)
            
            print(f"\n✅ Selected answer: {top_label} (confidence: {top_score:.2%})")
            
            return top_label, entity_type, top_score
        
        except Exception as e:
            print(f"❌ Error in embedding-based answer finding: {e}")
            import traceback
            traceback.print_exc()
            return None, None, 0.0
    
    def _extract_subject_entity(self, question: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract the subject entity (e.g., movie title) from the question.
        
        Returns:
            Tuple of (entity_uri, entity_label)
        """
        # Use entity extractor to find entities
        entities = self.entity_extractor.extract_entities(question)
        
        if not entities:
            return None, None
        
        # Take the first entity (usually the movie or main subject)
        for entity in entities:
            if entity.get('uri'):
                return entity['uri'], entity.get('label', entity['uri'])
        
        return None, None
    
    def _detect_relation(self, question: str) -> Optional[str]:
        """
        Detect the relation being asked about from the question.
        
        Returns:
            Relation URI or None
        """
        question_lower = question.lower()
        
        # Check for relation keywords
        for relation_name, relation_uri in self.RELATION_MAPPINGS.items():
            if relation_name in question_lower:
                return relation_uri
        
        # Fallback patterns
        if "who directed" in question_lower or "director of" in question_lower:
            return self.RELATION_MAPPINGS["director"]
        elif "who wrote" in question_lower or "screenwriter" in question_lower:
            return self.RELATION_MAPPINGS["screenwriter"]
        elif "who produced" in question_lower or "producer" in question_lower:
            return self.RELATION_MAPPINGS["producer"]
        elif "what genre" in question_lower or "genre of" in question_lower:
            return self.RELATION_MAPPINGS["genre"]
        elif "what country" in question_lower or "from what country" in question_lower:
            return self.RELATION_MAPPINGS["country"]
        elif "when" in question_lower or "release" in question_lower or "publication" in question_lower:
            return self.RELATION_MAPPINGS["release date"]
        
        return None
    
    def _get_entity_label(self, entity_uri: str) -> str:
        """
        Get human-readable label for an entity URI.
        
        Args:
            entity_uri: Entity URI
            
        Returns:
            Entity label or shortened URI
        """
        try:
            # Query the graph for the label
            from rdflib import URIRef, RDFS
            
            uri_ref = URIRef(entity_uri)
            
            # Try to get English label
            for label in self.sparql_handler.graph.objects(uri_ref, RDFS.label):
                label_str = str(label)
                # Prefer English labels
                if hasattr(label, 'language') and label.language == 'en':
                    return label_str
                # Return first label if no language specified
                if not hasattr(label, 'language'):
                    return label_str
            
            # Try to get any label
            for label in self.sparql_handler.graph.objects(uri_ref, RDFS.label):
                return str(label)
            
            # Fallback: return last part of URI
            return entity_uri.split('/')[-1]
        
        except Exception as e:
            # Fallback: return last part of URI
            return entity_uri.split('/')[-1]
    
    def _get_entity_type(self, entity_uri: str, question: str) -> str:
        """
        Get the Wikidata entity type for an entity.
        
        Args:
            entity_uri: Entity URI
            question: Original question for context
            
        Returns:
            Wikidata entity type code
        """
        try:
            # Query the graph for instance-of (P31)
            from rdflib import URIRef
            
            uri_ref = URIRef(entity_uri)
            p31 = URIRef("http://www.wikidata.org/prop/direct/P31")
            
            for type_uri in self.sparql_handler.graph.objects(uri_ref, p31):
                # Extract QID from type URI
                type_id = str(type_uri).split('/')[-1]
                if type_id.startswith('Q'):
                    return type_id
            
            # Fallback: infer from question
            question_lower = question.lower()
            if any(word in question_lower for word in ["director", "screenwriter", "producer", "actor", "who"]):
                return "Q5"  # Person
            elif "genre" in question_lower:
                return "Q201658"  # Film genre
            elif "country" in question_lower:
                return "Q6256"  # Country
            else:
                return "Q35120"  # Entity (generic)
        
        except Exception as e:
            # Fallback
            return "Q35120"


# Example usage
if __name__ == "__main__":
    print("EmbeddingAnswerFinder module")
    print("This module provides TransE-based answer finding.")
