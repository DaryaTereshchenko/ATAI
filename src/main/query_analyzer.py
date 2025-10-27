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
    """
    Analyzes queries to understand intent and structure.
    Uses fine-tuned transformer model OR falls back to rule-based patterns.
    """
    
    def __init__(
        self,
        use_transformer: bool = True,
        transformer_model_path: str = None
    ):
        """
        Initialize query analyzer.
        
        Args:
            use_transformer: Whether to use transformer classifier
            transformer_model_path: Path to fine-tuned SPARQL pattern classifier model (defaults to config value)
        """
        # Use config path if not specified
        if transformer_model_path is None:
            from src.config import SPARQL_CLASSIFIER_MODEL_PATH
            transformer_model_path = SPARQL_CLASSIFIER_MODEL_PATH
        
        self.use_transformer = use_transformer and TRANSFORMER_AVAILABLE
        self.transformer_classifier = None
        
        if self.use_transformer:
            try:
                import os
                if os.path.exists(transformer_model_path):
                    print(f"🤖 Loading transformer SPARQL pattern classifier...")
                    self.transformer_classifier = TransformerSPARQLClassifier(
                        model_path=transformer_model_path,
                        confidence_threshold=0.6
                    )
                    print(f"✅ SPARQL pattern classifier loaded from {transformer_model_path}\n")
                else:
                    print(f"⚠️  SPARQL classifier model not found: {transformer_model_path}")
                    print("   Falling back to rule-based pattern analysis\n")
                    self.use_transformer = False
            except Exception as e:
                print(f"⚠️  Failed to load SPARQL pattern classifier: {e}")
                print("   Falling back to rule-based pattern analysis\n")
                import traceback
                traceback.print_exc()
                self.use_transformer = False
        
        # Always initialize rule-based patterns as fallback
        self._setup_patterns()
        self._setup_entity_hints()
        
        # Mapping for subject/object types
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
            'country_of_origin': {'subject': 'movie', 'object': 'string'},  # ✅ NEW
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
    
    def _setup_patterns(self):
        """Define comprehensive query patterns for all supported query types."""
        
        # ==================== FORWARD PATTERNS ====================
        # Movie/Entity → Property (e.g., "Who directed The Matrix?")
        self.forward_patterns = [
            # Director queries
            {
                'regex': r'\b(?:who|what)\s+(?:is|was|are|were)?\s*(?:the)?\s*director[s]?\s+(?:of|for)\s+',
                'relation': 'director',
                'subject': 'movie',
                'object': 'person',
                'confidence': 0.95
            },
            {
                'regex': r'\b(?:who)\s+directed\s+',
                'relation': 'director',
                'subject': 'movie',
                'object': 'person',
                'confidence': 0.98
            },
            {
                'regex': r'\bdirector\s+of\s+',
                'relation': 'director',
                'subject': 'movie',
                'object': 'person',
                'confidence': 0.90
            },
            
            # Cast/Actor queries
            {
                'regex': r'\b(?:who|what)\s+(?:is|was|are|were)?\s*(?:the)?\s*(?:cast|actors?|stars?|actresses?)\s+(?:of|in|for)\s+',
                'relation': 'cast_member',
                'subject': 'movie',
                'object': 'person',
                'confidence': 0.95
            },
            {
                'regex': r'\b(?:who)\s+(?:acted|starred|plays?|appear(?:ed|s)?)\s+(?:in|on)\s+',
                'relation': 'cast_member',
                'subject': 'movie',
                'object': 'person',
                'confidence': 0.95
            },
            {
                'regex': r'\bcast\s+(?:of|in|for)\s+',
                'relation': 'cast_member',
                'subject': 'movie',
                'object': 'person',
                'confidence': 0.90
            },
            {
                'regex': r'\b(?:actors?|stars?)\s+(?:of|in)\s+',
                'relation': 'cast_member',
                'subject': 'movie',
                'object': 'person',
                'confidence': 0.90
            },
            
            # Genre queries
            {
                'regex': r'\b(?:what)\s+(?:is|was|are|were)?\s*(?:the)?\s*genre[s]?\s+(?:of|for|is)\s+',
                'relation': 'genre',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.95
            },
            {
                'regex': r'\bgenre[s]?\s+(?:of|for)\s+',
                'relation': 'genre',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.90
            },
            {
                'regex': r'\b(?:what)\s+(?:kind|type)\s+of\s+(?:movie|film)\s+is\s+',
                'relation': 'genre',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.85
            },
            
            # Release date queries
            {
                'regex': r'\b(?:when)\s+(?:was|is|did)\s+.*?\s+(?:released?|come\s+out|premiere[d]?)\b',
                'relation': 'publication_date',
                'subject': 'movie',
                'object': 'date',
                'confidence': 0.98
            },
            {
                'regex': r'\brelease\s+date\s+(?:of|for)\s+',
                'relation': 'publication_date',
                'subject': 'movie',
                'object': 'date',
                'confidence': 0.95
            },
            {
                'regex': r'\b(?:what)\s+year\s+(?:was|is|did)\s+.*?\s+(?:released?|come\s+out)\b',
                'relation': 'publication_date',
                'subject': 'movie',
                'object': 'date',
                'confidence': 0.95
            },
            
            # Screenwriter queries
            {
                'regex': r'\b(?:who)\s+(?:wrote|is\s+the\s+(?:screen)?writer)\s+',
                'relation': 'screenwriter',
                'subject': 'movie',
                'object': 'person',
                'confidence': 0.95
            },
            {
                'regex': r'\b(?:who)\s+(?:is|was|are|were)?\s*(?:the)?\s*(?:screen)?writer[s]?\s+(?:of|for)\s+',
                'relation': 'screenwriter',
                'subject': 'movie',
                'object': 'person',
                'confidence': 0.95
            },
            {
                'regex': r'\bscreenwriter[s]?\s+(?:of|for)\s+',
                'relation': 'screenwriter',
                'subject': 'movie',
                'object': 'person',
                'confidence': 0.90
            },
            
            # Producer queries
            {
                'regex': r'\b(?:who)\s+(?:produced|is\s+the\s+producer)\s+',
                'relation': 'producer',
                'subject': 'movie',
                'object': 'person',
                'confidence': 0.95
            },
            {
                'regex': r'\b(?:who)\s+(?:is|was|are|were)?\s*(?:the)?\s*producer[s]?\s+(?:of|for)\s+',
                'relation': 'producer',
                'subject': 'movie',
                'object': 'person',
                'confidence': 0.95
            },
            
            # Rating queries
            {
                'regex': r'\b(?:what)\s+(?:is|was)?\s*(?:the)?\s*rating\s+(?:of|for)\s+',
                'relation': 'rating',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.95
            },
            {
                'regex': r'\brating\s+(?:of|for)\s+',
                'relation': 'rating',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.90
            },
            
            # ✅ NEW: Country of origin queries
            {
                'regex': r'\b(?:from|of)\s+(?:what|which)\s+country\s+(?:is|was)\s+',
                'relation': 'country_of_origin',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.98
            },
            {
                'regex': r'\b(?:what|which)\s+country\s+(?:is|was|did)\s+.*?\s+from\b',
                'relation': 'country_of_origin',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.98
            },
            {
                'regex': r'\bcountry\s+of\s+origin\s+(?:of|for)\s+',
                'relation': 'country_of_origin',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.95
            },
            {
                'regex': r'\b(?:where|which\s+country)\s+(?:is|was|does)\s+.*?\s+(?:made|produced|filmed)\b',
                'relation': 'country_of_origin',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.90
            },
        ]
        
        # ==================== REVERSE PATTERNS ====================
        # Person → Movies (e.g., "What films did Christopher Nolan direct?")
        self.reverse_patterns = [
            # Director filmography
            {
                'regex': r'\b(?:what|which)\s+(?:films?|movies?)\s+(?:did|has|have)\s+.*?\s+direct(?:ed)?\b',
                'relation': 'director',
                'subject': 'person',
                'object': 'movie',
                'confidence': 0.98
            },
            {
                'regex': r'\b(?:films?|movies?)\s+directed\s+by\b',
                'relation': 'director',
                'subject': 'person',
                'object': 'movie',
                'confidence': 0.95
            },
            {
                'regex': r'\b(?:list|show|find|give\s+me)\s+.*?\s+(?:films?|movies?)\s+.*?\s+directed\b',
                'relation': 'director',
                'subject': 'person',
                'object': 'movie',
                'confidence': 0.90
            },
            {
                'regex': r'\b(?:films?|movies?)\s+by\s+(?:director)\b',
                'relation': 'director',
                'subject': 'person',
                'object': 'movie',
                'confidence': 0.85
            },
            {
                'regex': r'\b.*?\s+(?:directed|directs)\s+(?:which|what)\s+(?:films?|movies?)\b',
                'relation': 'director',
                'subject': 'person',
                'object': 'movie',
                'confidence': 0.90
            },
            
            # Actor filmography
            {
                'regex': r'\b(?:what|which)\s+(?:films?|movies?)\s+(?:did|has|have)\s+.*?\s+(?:star(?:red)?|act(?:ed)?)\s+(?:in|on)\b',
                'relation': 'cast_member',
                'subject': 'person',
                'object': 'movie',
                'confidence': 0.98
            },
            {
                'regex': r'\b(?:films?|movies?)\s+(?:starring|featuring|with)\b',
                'relation': 'cast_member',
                'subject': 'person',
                'object': 'movie',
                'confidence': 0.90
            },
            {
                'regex': r'\b(?:list|show|find)\s+.*?\s+(?:films?|movies?)\s+.*?\s+(?:starred?|acted?)\b',
                'relation': 'cast_member',
                'subject': 'person',
                'object': 'movie',
                'confidence': 0.90
            },
            {
                'regex': r'\b.*?\s+(?:starred?|acted?)\s+(?:in\s+)?(?:which|what)\s+(?:films?|movies?)\b',
                'relation': 'cast_member',
                'subject': 'person',
                'object': 'movie',
                'confidence': 0.90
            },
            
            # Screenwriter filmography
            {
                'regex': r'\b(?:what|which)\s+(?:films?|movies?)\s+(?:did|has|have)\s+.*?\s+(?:writ(?:e|ten)|screenplay)\b',
                'relation': 'screenwriter',
                'subject': 'person',
                'object': 'movie',
                'confidence': 0.95
            },
            {
                'regex': r'\b(?:films?|movies?)\s+written\s+by\b',
                'relation': 'screenwriter',
                'subject': 'person',
                'object': 'movie',
                'confidence': 0.95
            },
            
            # Producer filmography
            {
                'regex': r'\b(?:what|which)\s+(?:films?|movies?)\s+(?:did|has|have)\s+.*?\s+produce[d]?\b',
                'relation': 'producer',
                'subject': 'person',
                'object': 'movie',
                'confidence': 0.95
            },
            {
                'regex': r'\b(?:films?|movies?)\s+produced\s+by\b',
                'relation': 'producer',
                'subject': 'person',
                'object': 'movie',
                'confidence': 0.95
            }
        ]
        
        # ==================== VERIFICATION PATTERNS ====================
        # Does X have relation Y? (e.g., "Did Christopher Nolan direct Inception?")
        self.verification_patterns = [
            {
                'regex': r'\b(?:did|is|was)\s+(\w+(?:\s+\w+)*?)\s+(?:the\s+)?(direct(?:or)?|star|act(?:or)?|writ(?:e|er)|produc(?:e|er))\s+(?:of|in|for)\s+[\'""]?([^\'"",?]+)[\'""]?',
                'relation_map': {
                    'direct': 'director',
                    'director': 'director',
                    'star': 'cast_member',
                    'act': 'cast_member',
                    'actor': 'cast_member',
                    'write': 'screenwriter',
                    'writer': 'screenwriter',
                    'produce': 'producer',
                    'producer': 'producer'
                },
                'confidence': 0.95
            },
            {
                'regex': r'\b(?:is|was)\s+(\w+(?:\s+\w+)*?)\s+(?:in|a\s+cast\s+member\s+of)\s+[\'""]?([^\'"",?]+)[\'""]?',
                'relation': 'cast_member',
                'confidence': 0.95
            },
            {
                'regex': r'\b(?:did)\s+[\'""]?([^\'"",?]+)[\'""]?\s+(?:star|feature)\s+(\w+(?:\s+\w+)*?)',
                'relation': 'cast_member',
                'reverse': True,
                'confidence': 0.90
            }
        ]
        
        # ==================== COMPLEX/MULTI-CONSTRAINT PATTERNS ====================
        # Queries with multiple filters (country + award, genre + year, etc.)
        self.complex_patterns = [
            {
                'regex': r'(?:which|what)\s+movie.*?(?:from|of)\s+(?:the\s+)?country.*?(?:received?|won|got)\s+(?:the\s+)?award',
                'constraints': ['country', 'award'],
                'subject': 'movie',
                'confidence': 0.92
            },
            {
                'regex': r'(?:which|what)\s+(?:film|movie).*?award.*?country',
                'constraints': ['award', 'country'],
                'subject': 'movie',
                'confidence': 0.90
            }
        ]
    
    def analyze(self, query: str) -> Optional[QueryPattern]:
        """
        Analyze query to detect pattern and intent.
        Uses transformer model if available, otherwise rule-based.
        """
        query_lower = query.lower()
        entity_hints = self._extract_entity_hints(query)
        
        # Check complex patterns first (most specific)
        pattern = self._check_complex_patterns(query_lower, entity_hints)
        if pattern:
            print(f"[Analyzer] ✅ Detected complex pattern (pre-transformer check)")
            return pattern
        
        # PRIMARY: Try transformer classification for standard patterns
        if self.use_transformer and self.transformer_classifier:
            pattern = self._transformer_classify(query, entity_hints)
            if pattern and pattern.confidence > 0.6:
                # ✅ Check if this is a superlative variant of forward query
                if pattern.pattern_type == 'forward' and self._is_superlative_query(query_lower):
                    print(f"[Analyzer] ✅ Detected superlative modifier on forward query")
                    if not pattern.extracted_entities:
                        pattern.extracted_entities = {}
                    pattern.extracted_entities['superlative'] = self._extract_superlative_type(query_lower)
                return pattern
        
        # FALLBACK: Rule-based classification
        
        pattern = self._check_verification_patterns(query_lower, entity_hints)
        if pattern:
            return pattern
        
        pattern = self._check_reverse_patterns(query_lower, entity_hints)
        if pattern:
            return pattern
        
        pattern = self._check_forward_patterns(query_lower, entity_hints)
        if pattern:
            # ✅ Check if this is a superlative variant
            if self._is_superlative_query(query_lower):
                print(f"[Analyzer] ✅ Detected superlative modifier on forward query")
                if not pattern.extracted_entities:
                    pattern.extracted_entities = {}
                pattern.extracted_entities['superlative'] = self._extract_superlative_type(query_lower)
            return pattern
        
        return None
    
    def _is_superlative_query(self, query: str) -> bool:
        """Check if query contains superlative modifiers (highest/lowest/best/worst)."""
        superlative_keywords = [
            'highest', 'lowest', 'best', 'worst', 'top', 'bottom',
            'maximum', 'minimum', 'greatest', 'least', 'most', 'fewest'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in superlative_keywords)
    
    def _extract_superlative_type(self, query: str) -> str:
        """Extract superlative type (MAX or MIN) from query."""
        max_keywords = ['highest', 'best', 'top', 'maximum', 'greatest', 'most']
        min_keywords = ['lowest', 'worst', 'bottom', 'minimum', 'least', 'fewest']
        
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in max_keywords):
            return 'MAX'
        elif any(keyword in query_lower for keyword in min_keywords):
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
    
    def _check_forward_patterns(self, query: str, entity_hints: dict) -> Optional[QueryPattern]:
        """
        Check if query matches forward query patterns.
        Enhanced with entity hints for better confidence scoring.
        """
        for pattern in self.forward_patterns:
            if re.search(pattern['regex'], query, re.IGNORECASE):
                # Boost confidence if we have entity hints
                confidence = pattern['confidence']
                if entity_hints['quoted'] or entity_hints['capitalized']:
                    confidence = min(0.99, confidence + 0.05)
                
                return QueryPattern(
                    pattern_type='forward',
                    relation=pattern['relation'],
                    subject_type=pattern['subject'],
                    object_type=pattern['object'],
                    confidence=confidence,
                    extracted_entities=entity_hints
                )
        return None
    
    def _check_reverse_patterns(self, query: str, entity_hints: dict) -> Optional[QueryPattern]:
        """Check if query matches reverse query patterns with entity hints."""
        for pattern in self.reverse_patterns:
            if re.search(pattern['regex'], query, re.IGNORECASE):
                confidence = pattern['confidence']
                if entity_hints['quoted'] or entity_hints['capitalized']:
                    confidence = min(0.99, confidence + 0.05)
                
                return QueryPattern(
                    pattern_type='reverse',
                    relation=pattern['relation'],
                    subject_type=pattern['subject'],
                    object_type=pattern['object'],
                    confidence=confidence,
                    extracted_entities=entity_hints
                )
        return None
    
    def _check_verification_patterns(self, query: str, entity_hints: dict) -> Optional[QueryPattern]:
        """Check if query matches verification query patterns with entity hints."""
        for pattern in self.verification_patterns:
            match = re.search(pattern['regex'], query, re.IGNORECASE)
            if match:
                relation = pattern.get('relation')
                
                if not relation and 'relation_map' in pattern:
                    groups = match.groups()
                    if len(groups) >= 2:
                        verb = groups[1].lower()
                        verb_clean = re.sub(r'(or|er)$', '', verb)
                        relation = pattern['relation_map'].get(verb_clean) or pattern['relation_map'].get(verb)
                
                if relation:
                    confidence = pattern['confidence']
                    # Boost if we have two entities (movie + person)
                    if len(entity_hints['quoted']) >= 2 or len(entity_hints['capitalized']) >= 2:
                        confidence = min(0.99, confidence + 0.05)
                    
                    return QueryPattern(
                        pattern_type='verification',
                        relation=relation,
                        subject_type='mixed',
                        object_type='mixed',
                        confidence=confidence,
                        extracted_entities=entity_hints
                    )
        return None
    
    def _check_complex_patterns(self, query: str, entity_hints: dict) -> Optional[QueryPattern]:
        """
        Check if query matches complex multi-constraint patterns.
        These require special handling with multiple filters.
        """
        for pattern in self.complex_patterns:
            if re.search(pattern['regex'], query, re.IGNORECASE):
                return QueryPattern(
                    pattern_type='complex',  # Special type
                    relation='multi_constraint',
                    subject_type=pattern['subject'],
                    object_type='mixed',
                    confidence=pattern['confidence'],
                    extracted_entities={
                        'constraints': pattern['constraints'],
                        'quoted': entity_hints.get('quoted', [])
                    }
                )
        return None
    
    def get_supported_relations(self) -> list:
        """
        Get list of all supported relations.
        
        Returns:
            List of relation names
        """
        relations = set()
        for pattern in self.forward_patterns:
            relations.add(pattern['relation'])
        for pattern in self.reverse_patterns:
            relations.add(pattern['relation'])
        return sorted(list(relations))
    
    def get_pattern_info(self, pattern_type: str) -> dict:
        """
        Get information about patterns of a specific type.
        
        Args:
            pattern_type: 'forward', 'reverse', or 'verification'
            
        Returns:
            Dictionary with pattern statistics
        """
        if pattern_type == 'forward':
            patterns = self.forward_patterns
        elif pattern_type == 'reverse':
            patterns = self.reverse_patterns
        elif pattern_type == 'verification':
            patterns = self.verification_patterns
        else:
            return {}
        
        relations = {}
        for pattern in patterns:
            rel = pattern.get('relation', 'unknown')
            if rel not in relations:
                relations[rel] = 0
            relations[rel] += 1
        
        return {
            'total_patterns': len(patterns),
            'relations': relations,
            'avg_confidence': sum(p['confidence'] for p in patterns) / len(patterns) if patterns else 0.0
        }