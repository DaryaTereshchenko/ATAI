"""
Query Analyzer - Understands query intent and structure.
NOW USES: Fine-tuned transformer model for pattern classification.
"""

import re
from typing import Optional
from dataclasses import dataclass

# Try to import transformer classifier
try:
    from src.main.sparql_pattern_classifier import (
        TransformerSPARQLClassifier,
        SPARQLPatternPrediction
    )
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False


@dataclass
class QueryPattern:
    """Represents a detected query pattern."""
    pattern_type: str  # 'forward', 'reverse', 'verification'
    relation: str      # 'director', 'cast_member', etc.
    subject_type: str  # 'movie', 'person', 'genre'
    object_type: str   # 'person', 'movie', 'date', 'string'
    confidence: float
    extracted_entities: list = None  # Entity hints


class QueryAnalyzer:
    """Analyzes natural language queries to extract patterns and entities."""
    
    def __init__(self, use_transformer_classifier: bool = True, transformer_model_path: str = None):
        """
        Initialize the query analyzer.
        
        Args:
            use_transformer_classifier: Whether to use transformer-based classification
            transformer_model_path: Path to the transformer model
        """
        self.use_transformer = use_transformer_classifier
        
        # Initialize transformer classifier if requested
        if self.use_transformer:
            if transformer_model_path is None:
                from src.config import SPARQL_CLASSIFIER_MODEL_PATH
                transformer_model_path = SPARQL_CLASSIFIER_MODEL_PATH
            
            self.transformer_classifier = TransformerSPARQLClassifier(
                model_path=transformer_model_path,
                # ✅ REMOVED: num_labels parameter - not accepted by TransformerSPARQLClassifier
                confidence_threshold=0.6
            )
        else:
            self.transformer_classifier = None
        
        # Define supported relations
        self.supported_relations = [
            'director',
            'cast_member',
            'genre',
            'publication_date',
            'screenwriter',
            'producer',
            'rating',
            'country',  # ✅ NEW: added country
        ]
        
        # Only keep minimal entity hint patterns and type mappings
        self._setup_entity_hints()
        self._setup_type_mappings()

    
    def _setup_type_mappings(self):
        """Map relations to subject/object types."""
        self.type_mappings = {
            'director': {'subject': 'movie', 'object': 'person'},
            'cast_member': {'subject': 'movie', 'object': 'person'},
            'screenwriter': {'subject': 'movie', 'object': 'person'},
            'producer': {'subject': 'movie', 'object': 'person'},
            'genre': {'subject': 'movie', 'object': 'string'},
            'publication_date': {'subject': 'movie', 'object': 'date'},
            'rating': {'subject': 'movie', 'object': 'string'},
            'country': {'subject': 'movie', 'object': 'string'},  # ✅ NEW
        }
    
    def _setup_entity_hints(self):
        """Define patterns for extracting entity hints from queries."""
        self.entity_hint_patterns = {
            'quoted_text': r'["\']([^"\']+)["\']',
            'title_case_span': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}\b',
            'after_directed': r'directed?\s+["\']?([^"\'?,\.]+)["\']?',
            'after_starred_in': r'starred?\s+in\s+["\']?([^"\'?,\.]+)["\']?',
            'person_context': r'(?:did|by|with)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:direct|star|write|produce)',
        }
    
    def analyze(self, query: str) -> Optional[QueryPattern]:
        """
        Analyze query to detect pattern and intent.
        Uses transformer model PRIMARILY, minimal rule-based fallback.
        """
        query_lower = query.lower()
        entity_hints = self._extract_entity_hints(query)
        
        # PRIMARY: Try transformer classification
        if self.use_transformer and self.transformer_classifier:
            pattern = self._transformer_classify(query, entity_hints)
            if pattern and pattern.confidence > 0.6:
                # ✅ Check if this is a superlative variant
                if pattern.pattern_type == 'forward' and self._is_superlative_query(query_lower):
                    print(f"[Analyzer] ✅ Detected superlative modifier on forward query")
                    pattern.extracted_entities = pattern.extracted_entities or {}
                    pattern.extracted_entities['superlative'] = self._extract_superlative_type(query_lower)
                return pattern
        
        # FALLBACK: Minimal rule-based classification for catastrophic failure
        print("[Analyzer] ⚠️  Transformer failed, using minimal fallback rules...")
        return self._minimal_fallback_classify(query, entity_hints)
    
    def _is_superlative_query(self, query: str) -> bool:
        """Check if query contains superlative modifiers (highest/lowest/best/worst)."""
        superlative_keywords = [
            'highest', 'lowest', 'best', 'worst', 'top', 'bottom',
            'maximum', 'minimum', 'greatest', 'least'
        ]
        return any(keyword in query for keyword in superlative_keywords)
    
    def _extract_superlative_type(self, query: str) -> str:
        """Extract superlative type (MAX or MIN) from query."""
        max_keywords = ['highest', 'best', 'top', 'maximum', 'greatest', 'most']
        min_keywords = ['lowest', 'worst', 'bottom', 'minimum', 'least']
        
        if any(keyword in query for keyword in max_keywords):
            return 'MAX'
        elif any(keyword in query for keyword in min_keywords):
            return 'MIN'
        return 'MAX'  # Default

    def _transformer_classify(
        self,
        query: str,
        entity_hints: dict
    ) -> Optional[QueryPattern]:
        """
        Classify using transformer model.
        
        Args:
            query: Natural language query
            entity_hints: Extracted entity hints
            
        Returns:
            QueryPattern or None
        """
        try:
            # Get prediction from transformer
            prediction = self.transformer_classifier.classify(query)
            
            print(f"[Transformer] Prediction: {prediction.pattern_type} + {prediction.relation}")
            print(f"[Transformer] Confidence: {prediction.confidence:.2%}")
            
            # Skip unknown predictions
            if prediction.pattern_type == 'unknown' or prediction.relation == 'unknown':
                print(f"[Transformer] Skipping unknown prediction")
                return None
            
            # ✅ CRITICAL: Map relation to type info correctly
            type_info = self.type_mappings.get(
                prediction.relation,
                {'subject': 'entity', 'object': 'entity'}
            )
            
            # ✅ CRITICAL: Adjust subject/object types based on pattern type
            if prediction.pattern_type == 'forward':
                # Forward: Movie → Property
                subject_type = type_info['subject']  # 'movie'
                object_type = type_info['object']    # 'person', 'date', 'string'
            elif prediction.pattern_type == 'reverse':
                # Reverse: Person → Movies
                subject_type = 'person'
                object_type = 'movie'
            elif prediction.pattern_type == 'verification':
                # Verification: Check relationship
                subject_type = 'mixed'
                object_type = 'mixed'
            else:
                print(f"[Transformer] Unknown pattern type: {prediction.pattern_type}")
                return None
            
            pattern = QueryPattern(
                pattern_type=prediction.pattern_type,
                relation=prediction.relation,
                subject_type=subject_type,
                object_type=object_type,
                confidence=prediction.confidence,
                extracted_entities=entity_hints
            )
            
            print(f"[Transformer] ✅ Created pattern: {pattern.pattern_type} + {pattern.relation}")
            print(f"[Transformer]    Subject: {pattern.subject_type} → Object: {pattern.object_type}")
            
            return pattern
            
        except Exception as e:
            print(f"[Transformer] Classification error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_entity_hints(self, query: str) -> dict:
        """
        Extract potential entity names from query to help pattern matching.
        
        Returns:
            Dict with 'quoted', 'capitalized', and 'contextual' entity lists
        """
        hints = {
            'quoted': [],
            'capitalized': [],
            'contextual': []
        }
        
        # Extract quoted text (highest priority)
        import re
        quoted = re.findall(self.entity_hint_patterns['quoted_text'], query)
        hints['quoted'] = [q.strip() for q in quoted if len(q.strip()) > 2]
        
        # Extract title case spans
        caps = re.findall(self.entity_hint_patterns['title_case_span'], query)
        # Filter out question words
        stop_words = {'Who', 'What', 'When', 'Where', 'Which', 'How', 'Did', 'Was', 'Is'}
        hints['capitalized'] = [c for c in caps if c not in stop_words]
        
        # Extract contextual entities (near keywords)
        for pattern_name, pattern in self.entity_hint_patterns.items():
            if pattern_name not in ['quoted_text', 'title_case_span']:
                matches = re.findall(pattern, query, re.IGNORECASE)
                hints['contextual'].extend([m.strip() for m in matches if len(m.strip()) > 2])
        
        return hints
    
    def _minimal_fallback_classify(self, query: str, entity_hints: dict) -> Optional[QueryPattern]:
        """
        Minimal rule-based fallback when transformer completely fails.
        Only handles most basic patterns with high confidence requirements.
        """
        q = query.lower()
        
        # ✅ NEW: Check for country queries (forward query about movie country)
        if any(word in q for word in ['country', 'from what country', 'which country']):
            print("[Analyzer] Detected country query pattern")
            return QueryPattern(
                pattern_type='forward',
                relation='country',
                subject_type='movie',
                object_type='string',
                confidence=0.75,
                extracted_entities=entity_hints
            )
        
        # Check for verification questions (yes/no)
        if any(word in q[:10] for word in ['did', 'is', 'was', 'does', 'has']):
            # Very basic verification detection
            if 'direct' in q:
                return QueryPattern(
                    pattern_type='verification',
                    relation='director',
                    subject_type='mixed',
                    object_type='mixed',
                    confidence=0.7,
                    extracted_entities=entity_hints
                )
        
        # Check for reverse queries (person → movies)
        if any(phrase in q for phrase in ['what movies', 'what films', 'which movies', 'which films']):
            if 'direct' in q:
                return QueryPattern(
                    pattern_type='reverse',
                    relation='director',
                    subject_type='person',
                    object_type='movie',
                    confidence=0.7,
                    extracted_entities=entity_hints
                )
            elif any(word in q for word in ['star', 'act']):
                return QueryPattern(
                    pattern_type='reverse',
                    relation='cast_member',
                    subject_type='person',
                    object_type='movie',
                    confidence=0.7,
                    extracted_entities=entity_hints
                )
        
        # Default: forward query about director
        if 'who' in q[:10] and 'direct' in q:
            return QueryPattern(
                pattern_type='forward',
                relation='director',
                subject_type='movie',
                object_type='person',
                confidence=0.6,
                extracted_entities=entity_hints
            )
        
        # If nothing matches, return None
        print("[Analyzer] ❌ No pattern detected even with fallback rules")
        return None

    def get_supported_relations(self) -> list:
        """
        Get list of all supported relations from transformer classifier.
        
        Returns:
            List of relation names
        """
        # Extract from type mappings (shared knowledge)
        return sorted(list(self.type_mappings.keys()))
    
    def get_pattern_info(self, pattern_type: str) -> dict:
        """
        Get information about patterns of a specific type.
        Now returns info based on transformer classifier capabilities.
        
        Args:
            pattern_type: 'forward', 'reverse', or 'verification'
        
        Returns:
            Dictionary with pattern statistics
        """
        if self.use_transformer and self.transformer_classifier:
            # Get info from transformer classifier
            relations = list(self.type_mappings.keys())
            
            return {
                'total_patterns': len(relations),
                'relations': {rel: 1 for rel in relations},
                'avg_confidence': 0.85,  # Typical transformer confidence
                'method': 'transformer'
            }
        else:
            return {
                'total_patterns': 0,
                'relations': {},
                'avg_confidence': 0.0,
                'method': 'none'
            }