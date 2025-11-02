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
        transformer_model_path: str = None,
        relation_manager = None  # ✅ NEW: Accept RelationManager
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
        
        # ✅ NEW: Store relation manager
        self.relation_manager = relation_manager
        
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
            'country_of_origin': {'subject': 'movie', 'object': 'string'},
            'country': {'subject': 'movie', 'object': 'string'},  # ✅ Already mapped
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
        """Set up rule-based patterns for NL to SPARQL conversion."""
        self.patterns = [
            # ==================== FORWARD PATTERNS ====================
            # Movie/Entity → Property (e.g., "Who directed The Matrix?")
            
            # ✅ NEW: Language queries - MUST come BEFORE genre queries for priority
            {
                'regex': r'\b(?:what|which)\s+language\s+(?:is|was|are)\s+',
                'relation': 'original_language_of_film_or_tv_show',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.98
            },
            {
                'regex': r'\b(?:in\s+)?(?:what|which)\s+language\b',
                'relation': 'original_language_of_film_or_tv_show',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.96
            },
            {
                'regex': r'\blanguage\s+(?:of|for|is)\s+',
                'relation': 'original_language_of_film_or_tv_show',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.95
            },
            {
                'regex': r'\b(?:spoken|original)\s+language\b',
                'relation': 'original_language_of_film_or_tv_show',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.95
            },
            
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
                'regex': r'\b(?:what)\s+(?:is|was)?\s*(?:the)?\s*(?:user\s+)?rating\s+(?:of|for)\s+',
                'relation': 'rating',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.95
            },
            {
                'regex': r'\brating\s+(?:of|for|does)\s+',
                'relation': 'rating',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.90
            },
            {
                'regex': r'\b(?:what|which)\s+rating\s+(?:does|did)\s+',
                'relation': 'rating',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.90
            },
            {
                'regex': r'\b(?:which|what)\s+(?:film|movie)\s+(?:has|have)\s+(?:the\s+)?.*?rating',
                'relation': 'rating',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.85
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
            
            # ✅ NEW: Award queries - ENHANCED patterns
            {
                'regex': r'\b(?:what|which)\s+award[s]?\s+(?:did|has|have)\s+',
                'relation': 'award_received',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.95
            },
            {
                'regex': r'\baward[s]?\s+(?:received|won|got)\s+(?:by|for)\s+',
                'relation': 'award_received',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.92
            },
            {
                'regex': r'\b(?:did|has)\s+.*?\s+(?:receive|win|get)\s+(?:any\s+)?award[s]?',
                'relation': 'award_received',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.90
            },
            {
                'regex': r'\b(?:what|which)\s+(?:award[s]?|prize[s]?)\s+(?:for|of)\s+',
                'relation': 'award_received',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.88
            },
            {
                'regex': r'\baward[s]?\s+(?:for|of|to)\s+',
                'relation': 'award_received',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.85
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
        
        # Convert patterns to dedicated lists for each type
        self.forward_patterns = [p for p in self.patterns]
        self.reverse_patterns = [p for p in self.reverse_patterns]
        self.verification_patterns = [p for p in self.verification_patterns]
        self.complex_patterns = [p for p in self.complex_patterns]
    
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
            'country_of_origin': {'subject': 'movie', 'object': 'string'},
            'country': {'subject': 'movie', 'object': 'string'},  # ✅ Already mapped
            'narrative_location': {'subject': 'movie', 'object': 'string'},
            'production_company': {'subject': 'movie', 'object': 'organization'},
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
            
            # ✅ LOG: What the transformer predicted
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
    
    def analyze(self, query: str) -> Optional[QueryPattern]:
        """
        Analyze query to detect pattern and intent.
        Uses transformer model if available, otherwise rule-based.
        
        ✅ CRITICAL: Works on ORIGINAL case-sensitive query.
        """
        # ✅ FIXED: Keep original query, only use lowercase for pattern matching
        query_original = query.strip()
        query_lower = query_original.lower()  # For pattern matching only
        
        entity_hints = self._extract_entity_hints(query_original)  # ✅ Use original
        
        # Check complex patterns first (most specific)
        pattern = self._check_complex_patterns(query_lower, entity_hints)
        if pattern:
            print(f"[Analyzer] ✅ Detected complex pattern (pre-transformer check)")
            # ✅ Store original query in pattern for later use
            pattern.extracted_entities['original_query'] = query_original
            return pattern
        
        # PRIMARY: Try transformer classification for standard patterns
        if self.use_transformer and self.transformer_classifier:
            pattern = self._transformer_classify(query_original, entity_hints)  # ✅ Use original
            if pattern and pattern.confidence > 0.6:
                # ✅ NEW: Validate and refine relation using RelationManager
                if self.relation_manager:
                    pattern = self._refine_relation_with_manager(pattern, query_original)  # ✅ Use original
                
                # ✅ NEW: GUARDRAIL - Override relation if "language" is explicitly mentioned
                if self._is_language_query(query_lower):
                    print(f"[Analyzer] 🔒 GUARDRAIL: Language query detected, overriding relation")
                    pattern.relation = 'original_language_of_film_or_tv_show'
                    pattern.object_type = 'string'
                    print(f"[Analyzer]    Relation set to: {pattern.relation}")
                
                # ✅ NEW: GUARDRAIL - Override relation if "rating" is explicitly mentioned (non-superlative)
                if self._is_rating_query(query_lower) and not self._is_superlative_query(query_lower):
                    print(f"[Analyzer] 🔒 GUARDRAIL: Rating query detected, overriding relation")
                    pattern.relation = 'rating'
                    pattern.object_type = 'string'
                    print(f"[Analyzer]    Relation set to: {pattern.relation}")
                
                # ✅ Check if this is a superlative variant of forward query
                if pattern.pattern_type == 'forward' and self._is_superlative_query(query_lower):
                    print(f"[Analyzer] ✅ Detected superlative modifier on forward query")
                    if not pattern.extracted_entities:
                        pattern.extracted_entities = {}
                    pattern.extracted_entities['superlative'] = self._extract_superlative_type(query_lower)
                
                # ✅ Store original query
                if not pattern.extracted_entities:
                    pattern.extracted_entities = {}
                pattern.extracted_entities['original_query'] = query_original
                return pattern
        
        # ✅ NEW: If transformer didn't work, try dynamic relation matching
        if self.relation_manager:
            pattern = self._dynamic_relation_matching(query_original, entity_hints)  # ✅ Use original
            if pattern:
                # ✅ NEW: GUARDRAIL - Also apply to dynamically matched patterns
                if self._is_language_query(query_lower):
                    print(f"[Analyzer] 🔒 GUARDRAIL: Language query detected, overriding relation")
                    pattern.relation = 'original_language_of_film_or_tv_show'
                    pattern.object_type = 'string'
                    print(f"[Analyzer]    Relation set to: {pattern.relation}")
                
                # ✅ NEW: GUARDRAIL - Apply rating override to dynamically matched patterns
                if self._is_rating_query(query_lower) and not self._is_superlative_query(query_lower):
                    print(f"[Analyzer] 🔒 GUARDRAIL: Rating query detected, overriding relation")
                    pattern.relation = 'rating'
                    pattern.object_type = 'string'
                    print(f"[Analyzer]    Relation set to: {pattern.relation}")
                
                pattern.extracted_entities['original_query'] = query_original
                return pattern
        
        # FALLBACK: Rule-based classification
        # ✅ Use lowercase ONLY for pattern matching, not for entity extraction
        pattern = self._check_verification_patterns(query_lower, entity_hints)
        if pattern:
            # ✅ NEW: GUARDRAIL - Apply to all pattern types
            if self._is_language_query(query_lower):
                print(f"[Analyzer] 🔒 GUARDRAIL: Language query detected, overriding relation")
                pattern.relation = 'original_language_of_film_or_tv_show'
                pattern.object_type = 'string'
            # ✅ NEW: GUARDRAIL - Apply rating override
            if self._is_rating_query(query_lower) and not self._is_superlative_query(query_lower):
                print(f"[Analyzer] 🔒 GUARDRAIL: Rating query detected, overriding relation")
                pattern.relation = 'rating'
                pattern.object_type = 'string'
            pattern.extracted_entities['original_query'] = query_original
            return pattern
        
        pattern = self._check_reverse_patterns(query_lower, entity_hints)
        if pattern:
            pattern.extracted_entities['original_query'] = query_original
            return pattern
        
        pattern = self._check_forward_patterns(query_lower, entity_hints)
        if pattern:
            # ✅ Check if this is a superlative variant
            if self._is_superlative_query(query_lower):
                print(f"[Analyzer] ✅ Detected superlative modifier on forward query")
                if not pattern.extracted_entities:
                    pattern.extracted_entities = {}
                pattern.extracted_entities['superlative'] = self._extract_superlative_type(query_lower)
            
            # ✅ NEW: GUARDRAIL - Apply to forward patterns
            if self._is_language_query(query_lower):
                print(f"[Analyzer] 🔒 GUARDRAIL: Language query detected, overriding relation")
                pattern.relation = 'original_language_of_film_or_tv_show'
                pattern.object_type = 'string'
            
            # ✅ NEW: GUARDRAIL - Apply rating override to forward patterns
            if self._is_rating_query(query_lower) and not self._is_superlative_query(query_lower):
                print(f"[Analyzer] 🔒 GUARDRAIL: Rating query detected, overriding relation")
                pattern.relation = 'rating'
                pattern.object_type = 'string'
            
            if not pattern.extracted_entities:
                pattern.extracted_entities = {}
            pattern.extracted_entities['original_query'] = query_original
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
            
            # ✅ LOG: What the transformer predicted
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
    
    def _refine_relation_with_manager(self, pattern: QueryPattern, query: str) -> QueryPattern:
        """
        Refine relation using RelationManager's fuzzy matching.
        
        ✅ ENHANCED: Better handling of generic relations and validation.
        """
        print(f"[Analyzer] 🔍 Refining relation '{pattern.relation}' using RelationManager")
        
        # ✅ NEW: Validate that relation manager can resolve this relation
        uri = self.relation_manager.get_relation_uri(pattern.relation)
        if uri:
            print(f"[Analyzer] ✅ Relation already resolvable: {pattern.relation} → {uri}")
            return pattern
        
        # ✅ NEW: List of generic relations that need refinement
        generic_relations = {
            'country': ['country_of_origin', 'filming_location', 'narrative_location'],
            'language': ['original_language_of_film_or_tv_show', 'language_of_work_or_name'],
            'rating': ['imda_rating', 'mpa_film_rating', 'fsk_film_rating'],
            'award': ['award_received', 'nominated_for'],
        }
        
        is_generic = pattern.relation in generic_relations
        
        if is_generic:
            print(f"[Analyzer] ⚠️  '{pattern.relation}' is generic, searching for specific property...")
            # Get candidates for this generic type
            candidates = generic_relations[pattern.relation]
            print(f"[Analyzer]    Candidates: {candidates}")
    
        # Get potential relations from query with higher top_k
        matches = self.relation_manager.find_relation(query, top_k=10)
    
        if not matches:
            print(f"[Analyzer] ⚠️  No relation matches found in query")
            return pattern
        
        print(f"[Analyzer] 📋 Top relation matches from query:")
        for i, (key, uri, confidence) in enumerate(matches[:5], 1):
            print(f"   {i}. {key} ({uri}) - confidence: {confidence:.2%}")
    
        # ✅ NEW: If generic, find first specific match
        if is_generic:
            for key, uri, conf in matches:
                if key in candidates or key.endswith('_of_origin') or '_language_' in key:
                    print(f"[Analyzer] 🔄 Refining generic '{pattern.relation}' → specific '{key}'")
                    print(f"[Analyzer]    Confidence: {conf:.2%}")
                    pattern.relation = key
                    return pattern
    
        # Check if transformer's relation matches any of the top matches
        for key, uri, confidence in matches:
            if key == pattern.relation or pattern.relation in self.relation_manager.relations.get(key, {}).get('aliases', []):
                print(f"[Analyzer] ✅ Confirmed relation: {pattern.relation} (matched {key})")
                return pattern
        
        # Use best match if confidence is high
        best_key, best_uri, best_confidence = matches[0]
        if best_confidence > 0.7:
            print(f"[Analyzer] 🔄 Refining relation: {pattern.relation} → {best_key}")
            print(f"[Analyzer]    Confidence: {best_confidence:.2%}")
            pattern.relation = best_key
        else:
            print(f"[Analyzer] ⚠️  Best match confidence too low ({best_confidence:.2%}), keeping original")
        
        return pattern
    
    def _dynamic_relation_matching(self, query: str, entity_hints: dict) -> Optional[QueryPattern]:
        """
        Fallback: Match relation dynamically using RelationManager.
        
        ✅ FIXED: Works on ORIGINAL case-sensitive query.
        
        Args:
            query: Query text (ORIGINAL CASE)
            entity_hints: Extracted entity hints
            
        Returns:
            QueryPattern or None
        """
        print(f"[Analyzer] 🎯 Attempting dynamic relation matching")
        
        # Extract potential relation keywords from query
        # ✅ Use case-insensitive stop words but preserve original text
        stop_words = ['who', 'what', 'when', 'where', 'which', 'how', 'is', 'was', 'are', 'were', 
                      'did', 'does', 'do', 'the', 'of', 'in', 'for', 'to', 'a', 'an']
        
        words = query.split()
        keywords = [w for w in words if w.lower() not in stop_words and len(w) > 2]
        print(f"[Analyzer]    Keywords: {keywords[:10]}")  # Show first 10
        
        # ✅ Try to find relation using ORIGINAL case-sensitive query
        matches = self.relation_manager.find_relation(query, top_k=1)
        
        if not matches or matches[0][2] < 0.5:
            print(f"[Analyzer] ❌ No suitable relation found (threshold: 0.5)")
            if matches:
                print(f"[Analyzer]    Best match: {matches[0][0]} (confidence: {matches[0][2]:.2%})")
            return None
        
        relation_key, relation_uri, confidence = matches[0]
        
        print(f"[Analyzer] ✅ Dynamic match found:")
        print(f"[Analyzer]    Relation: {relation_key}")
        print(f"[Analyzer]    URI: {relation_uri}")
        print(f"[Analyzer]    Confidence: {confidence:.2%}")
        
        # Determine pattern type from query structure (using lowercase for comparison)
        pattern_type = self._infer_pattern_type(query.lower(), entity_hints)
        
        # Get type info for relation
        relation_info = self.relation_manager.get_relation_info(relation_key)
        type_mapping = self.type_mappings.get(relation_key, {'subject': 'entity', 'object': 'entity'})
        
        # ✅ Store original query in entity hints
        entity_hints['original_query'] = query
        
        return QueryPattern(
            pattern_type=pattern_type,
            relation=relation_key,
            subject_type=type_mapping['subject'],
            object_type=type_mapping['object'],
            confidence=confidence,
            extracted_entities=entity_hints
        )
    
    def _get_expected_qcode(self, pattern: QueryPattern) -> Optional[str]:
        """
        Get expected Wikidata Q-code for the result based on relation.
        
        Args:
            pattern: Query pattern
            
        Returns:
            Expected Q-code string (e.g., 'Q201658' for genre)
        """
        # ✅ COMPREHENSIVE: Load from RelationManager if available
        if self.relation_manager:
            relation_info = self.relation_manager.get_relation_info(pattern.relation)
            if relation_info and 'expected_type' in relation_info:
                return relation_info['expected_type']
        
        # ✅ ENHANCED: Comprehensive mapping for common relations
        relation_to_qcode = {
            # People-related
            'director': 'Q5',             # Human
            'cast_member': 'Q5',          # Human
            'screenwriter': 'Q5',         # Human
            'producer': 'Q5',             # Human
            'voice_actor': 'Q5',          # Human
            'director_of_photography': 'Q5',  # Human
            'film_editor': 'Q5',          # Human
            'composer': 'Q5',             # Human
            'executive_producer': 'Q5',   # Human
            'costume_designer': 'Q5',     # Human
            'production_designer': 'Q5',  # Human
            'narrator': 'Q5',             # Human
            'animator': 'Q5',             # Human
            'sound_designer': 'Q5',       # Human
            'choreographer': 'Q5',        # Human
            'storyboard_artist': 'Q5',    # Human
            'art_director': 'Q5',         # Human
            'make_up_artist': 'Q5',       # Human
            'illustrator': 'Q5',          # Human
            
            # Media & content
            'genre': 'Q201658',           # Film genre
            'characters': 'Q15632617',    # Fictional character
            'based_on': 'Q7725634',       # Written work
            'derivative_work': 'Q11424',  # Film
            'part_of_the_series': 'Q24856',  # Series
            'follows': 'Q11424',          # Film
            'followed_by': 'Q11424',      # Film
            'present_in_work': 'Q11424',  # Film
            'media_franchise': 'Q130371093',  # Media franchise
            
            # Geographic
            'country_of_origin': 'Q6256', # Country
            'country': 'Q6256',           # Country
            'filming_location': 'Q208511', # Location
            'place_of_birth': 'Q1093829', # Place
            'place_of_death': 'Q745456',  # Place
            'narrative_location': 'Q6256', # Country/Location
            'headquarters_location': 'Q1093829',  # Place
            'location': 'Q1066984',       # Location
            
            # Organizations
            'production_company': 'Q1762059',  # Production company
            'distributed_by': 'Q59152282',     # Distributor
            'publisher': 'Q1762059',           # Publisher
            'original_broadcaster': 'Q1254874', # Broadcaster
            'record_label': 'Q18127',          # Record label
            
            # ✅ FIXED: Language & Culture - Use Q1097949 (natural language) as specified
            'original_language_of_film_or_tv_show': 'Q1097949',  # Natural language
            'language_of_work_or_name': 'Q1097949',  # Natural language
            'languages_spoken_written_or_signed': 'Q1097949',  # Natural language
            'native_language': 'Q1097949',     # Natural language
            'writing_language': 'Q1097949',    # Natural language
            'original_language': 'Q1097949',   # ✅ NEW: Alias for original_language_of_film_or_tv_show
            
            # Awards & Recognition
            'award_received': 'Q38033430',     # Award
            'nominated_for': 'Q38033430',      # Award
            
            # Ratings (string types - no Q-code validation)
            'rating': None,
            'fsk_film_rating': None,
            'medierådet_rating': None,
            'kijkwijzer_rating': None,
            'mpa_film_rating': None,
            'assessment': None,
            'classind_rating': None,
            'nmhh_film_rating': None,
            'cnc_film_rating_france': None,
            'australian_classification': None,
            'filmiroda_rating': None,
            'bbfc_rating': None,
            'eirin_film_rating': None,
            'jmk_film_rating': None,
            'icaa_rating': None,
            'mtrcb_rating': None,
            'bamid_film_rating': None,
            'rars_rating': None,
            'cnc_film_rating_romania': None,
            'igac_rating': None,
            'rcq_classification': None,
            'ifco_rating': None,
            'rtc_film_rating': None,
            'imda_rating': None,
            'kavi_rating': None,
            'fpb_rating': None,
            'incaa_film_rating': None,
            'kmrb_film_rating': None,
            'oflc_classification': None,
            
            # Dates (no specific Q-code validation)
            'publication_date': None,
            
            # Technical properties (mostly strings)
            'color': None,
            'aspect_ratio_wh': None,
            'distribution_format': None,
            'original_film_format': None,
            'platform': None,
            
            # Other entities
            'main_subject': 'Q813912',    # Academic discipline / Topic
            'form_of_creative_work': 'Q4263830',  # Form of creative work
            'time_period': 'Q578',        # Time period
            'described_by_source': 'Q186165',  # Source
            'from_narrative_universe': 'Q559618',  # Fictional universe
            'takes_place_in_fictional_universe': 'Q559618',  # Fictional universe
        }
        
        return relation_to_qcode.get(pattern.relation)
    
    def _is_language_query(self, query_lower: str) -> bool:
        """
        Check if query is explicitly asking about language.
        
        Args:
            query_lower: Lowercase query text
            
        Returns:
            True if this is a language query
        """
        # Strong language indicators
        language_patterns = [
            r'\b(?:what|which)\s+language\b',
            r'\bin\s+(?:what|which)\s+language\b',
            r'\blanguage\s+(?:is|was|of|for)\b',
            r'\b(?:spoken|original)\s+language\b',
            r'\blanguage\s+(?:does|did)\b',
        ]
        
        for pattern in language_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def _is_rating_query(self, query_lower: str) -> bool:
        """
        Check if query is explicitly asking about rating.
        
        Args:
            query_lower: Lowercase query text
            
        Returns:
            True if this is a rating query
        """
        # Strong rating indicators (excluding superlative patterns which are handled separately)
        rating_patterns = [
            r'\b(?:what|which)\s+(?:is|was)?\s*(?:the)?\s*(?:user\s+)?rating\b',
            r'\brating\s+(?:of|for|is)\b',
            r'\b(?:what|how)\s+(?:is|was)\s+(?:the\s+)?(?:movie|film)\s+rated\b',
            r'\brated\s+(?:at|as)\b',
            r'\buser\s+rating\b',
            r'\bmovie\s+rating\b',
            r'\bfilm\s+rating\b',
        ]
        
        for pattern in rating_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
