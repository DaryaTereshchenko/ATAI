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
            'genre': {'subject': 'movie', 'object': 'string'},  # ✅ Actually returns Q201658 (film genre)
            'publication_date': {'subject': 'movie', 'object': 'date'},
            'rating': {'subject': 'movie', 'object': 'string'},
            'country_of_origin': {'subject': 'movie', 'object': 'string'},  # ✅ Actually returns Q6256 (country)
            'country': {'subject': 'movie', 'object': 'string'},
            'narrative_location': {'subject': 'movie', 'object': 'string'},  # ✅ Can be Q515 (city) or other location types
            'production_company': {'subject': 'movie', 'object': 'organization'},
            'original_language_of_film_or_tv_show': {'subject': 'movie', 'object': 'language'},
            'language_of_work_or_name': {'subject': 'movie', 'object': 'language'},
            'official_language': {'subject': 'entity', 'object': 'language'},
        }
        
        # ✅ NEW: Expected Q-codes for result type validation in embeddings
        # Maps relations to expected Q-codes even if object_type is 'string'
        self.relation_to_qcode = {
            'genre': 'Q201658',  # Film genre
            'country_of_origin': 'Q6256',  # Country
            'country': 'Q6256',  # Country
            'award_received': None,  # Awards have various types
            'original_language_of_film_or_tv_show': 'Q34770',  # Language
            'language_of_work_or_name': 'Q34770',  # Language
            'official_language': 'Q34770',  # Language
            # Location properties can have multiple valid types
            'narrative_location': None,  # Can be city, country, region, etc.
            'filming_location': None,  # Can be city, country, region, etc.
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
            
            # ✅ CRITICAL: Language queries - MUST come FIRST with highest priority
            {
                'regex': r'\b(?:what|which)\s+language\s+(?:is|was|are|were|does)\s+',
                'relation': 'original_language_of_film_or_tv_show',
                'subject': 'movie',
                'object': 'language',
                'confidence': 0.99
            },
            {
                'regex': r'\b(?:in\s+)?(?:what|which)\s+language\s+(?:is|was|are|were)\s+',
                'relation': 'original_language_of_film_or_tv_show',
                'subject': 'movie',
                'object': 'language',
                'confidence': 0.99
            },
            {
                'regex': r'\blanguage\s+(?:of|for|is|was|in)\s+',
                'relation': 'original_language_of_film_or_tv_show',
                'subject': 'movie',
                'object': 'language',
                'confidence': 0.98
            },
            {
                'regex': r'\b(?:spoken|original|dialogue)\s+language\b',
                'relation': 'original_language_of_film_or_tv_show',
                'subject': 'movie',
                'object': 'language',
                'confidence': 0.98
            },
            {
                'regex': r'\b(?:what|which)\s+(?:is|was)\s+the\s+language\b',
                'relation': 'original_language_of_film_or_tv_show',
                'subject': 'movie',
                'object': 'language',
                'confidence': 0.98
            },
            {
                'regex': r'\bspoken\s+in\s+(?:what\s+)?language\b',
                'relation': 'original_language_of_film_or_tv_show',
                'subject': 'movie',
                'object': 'language',
                'confidence': 0.98
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
    
    def _is_superlative_query(self, query: str) -> bool:
        """
        Check if query contains superlative patterns (most, highest, best, etc.).
        
        ✅ FIXED: More precise matching to avoid false positives like "originally"
        
        Args:
            query: Query string (lowercase)
            
        Returns:
            True if superlative detected
        """
        # ✅ CRITICAL: Exclude words that contain superlatives but aren't superlatives
        # e.g., "originally" contains "most" but isn't a superlative
        
        # Words that contain superlative substrings but aren't superlatives
        false_positives = [
            'originally', 'almost', 'mostly', 'foremost', 'utmost', 'topmost',
            'innermost', 'outermost', 'uppermost', 'bottommost'
        ]
        
        # Check for false positives first
        for fp in false_positives:
            if fp in query:
                # If we find a false positive, be extra careful
                # Only return True if we also find a clear superlative context
                has_clear_superlative = False
                clear_patterns = [
                    r'\bhighest\s+\w+\b', r'\blowest\s+\w+\b',
                    r'\bbest\s+\w+\b', r'\bworst\s+\w+\b',
                    r'\bwhich\s+\w+\s+has\s+the\s+(highest|lowest|best|worst)\b'
                ]
                for pattern in clear_patterns:
                    if re.search(pattern, query, re.IGNORECASE):
                        has_clear_superlative = True
                        break
                
                if not has_clear_superlative:
                    return False
        
        # ✅ ENHANCED: More precise superlative patterns with context
        superlative_patterns = [
            # Standalone superlatives with proper word boundaries
            r'\b(highest|lowest)\s+\w+\b',  # "highest rating", "lowest score"
            r'\b(best|worst)\s+\w+\b',      # "best movie", "worst film"
            r'\b(top|bottom)\s+\d*\s*\w+\b',  # "top 10 movies", "bottom rated"
            r'\b(first|last)\s+\w+\b',      # "first movie", "last film"
            r'\b(earliest|latest)\s+\w+\b', # "earliest release", "latest movie"
            r'\b(longest|shortest)\s+\w+\b',  # "longest movie", "shortest film"
            r'\b(biggest|smallest|largest)\s+\w+\b',  # "biggest budget"
            r'\b(maximum|minimum)\s+\w+\b',  # "maximum rating"
            
            # Superlative with "the" (more confident)
            r'\bthe\s+(highest|lowest|best|worst|top|bottom|first|last)\b',
            
            # Question patterns with superlatives
            r'\bwhich\s+\w+\s+(has|have|is|are)\s+the\s+(highest|lowest|best|worst)\b',
            r'\bwhat\s+(is|are)\s+the\s+(highest|lowest|best|worst)\b',
            
            # Comparative constructions that imply superlatives
            r'\bmost\s+\w+\b(?!\s+originally)',  # "most recent" but not "most originally"
            r'\bleast\s+\w+\b',                  # "least expensive"
        ]
        
        for pattern in superlative_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        return False
    
    def _extract_superlative_type(self, query: str) -> str:
        """
        Extract the type of superlative (most, highest, etc.).
        
        Args:
            query: Query string (lowercase)
            
        Returns:
            Superlative type string ('MAX' or 'MIN' for simplicity)
        """
        # ✅ FIXED: Return normalized MAX/MIN instead of raw superlative
        superlative_map = {
            # MAX indicators (want highest/maximum value)
            r'\bhighest\b': 'MAX',
            r'\bbest\b': 'MAX',
            r'\bmost\b': 'MAX',
            r'\btop\b': 'MAX',
            r'\blatest\b': 'MAX',
            r'\blongest\b': 'MAX',
            r'\bbiggest\b': 'MAX',
            r'\blargest\b': 'MAX',
            r'\bmaximum\b': 'MAX',
            r'\bmax\b': 'MAX',
            # MIN indicators (want lowest/minimum value)
            r'\blowest\b': 'MIN',
            r'\bworst\b': 'MIN',
            r'\bleast\b': 'MIN',
            r'\bbottom\b': 'MIN',
            r'\bearliest\b': 'MIN',
            r'\bfirst\b': 'MIN',
            r'\bshortest\b': 'MIN',
            r'\bsmallest\b': 'MIN',
            r'\bminimum\b': 'MIN',
            r'\bmin\b': 'MIN',
        }
        
        for pattern, superlative_type in superlative_map.items():
            if re.search(pattern, query, re.IGNORECASE):
                return superlative_type
        
        return 'MAX'  # Default to maximum
    
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
            # Get prediction from transformer
            print(f"[Transformer] ✅ Created pattern: {pattern.pattern_type} + {pattern.relation}")
            print(f"[Transformer]    Subject: {pattern.subject_type} → Object: {pattern.object_type}")
            # ✅ LOG: What the transformer predicted
            return pattern
        except Exception as e:
            print(f"[Transformer] Classification error: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def _extract_entity_hints(self, query: str) -> dict:
        """
        Extract potential entity names from query to help pattern matching.
        
        Args:
            query: Query string (lowercase)
            
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
        """Check if query matches forward query patterns.
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
                # Boost confidence if we have entity hints
                confidence = min(0.99, pattern['confidence'] + 0.05)
                
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
                    object_type=pattern['object'],
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
        
        ✅ ENHANCED: Better detection of complex queries with quoted entities
        """
        # ✅ Check if we have multiple quoted entities (strong signal for complex query)
        quoted = entity_hints.get('quoted', [])
        if len(quoted) >= 2:
            print(f"[Analyzer] 🔍 Found {len(quoted)} quoted entities: {quoted}")
            
            # Analyze constraint types
            constraints = []
            
            # Check for country
            if any(kw in query for kw in ['country', 'from', 'originally from']):
                constraints.append('country')
            
            # Check for award
            if any(kw in query for kw in ['award', 'received', 'won', 'prize']):
                constraints.append('award')
            
            # Check for year
            if 'year' in query or re.search(r'\b\d{4}\b', query):
                constraints.append('year')
            
            # Check for genre
            if 'genre' in query:
                constraints.append('genre')
            
            # If we have multiple constraints, it's a complex query
            if len(constraints) >= 2:
                print(f"[Analyzer] ✅ Complex query with constraints: {constraints}")
                return QueryPattern(
                    pattern_type='complex',
                    relation='multi_constraint',
                    subject_type='movie',
                    object_type='mixed',
                    confidence=0.95,
                    extracted_entities={
                        'constraints': constraints,
                        'quoted': quoted
                    }
                )
        
        # Fallback to existing pattern matching
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
        
        ✅ ENHANCED: Better keyword detection for characters and production company.
        ✅ FIXED: Check complex patterns BEFORE superlative detection
        ✅ CRITICAL: Hard-coded language detection FIRST
        """
        # ✅ FIXED: Keep original query, only use lowercase for pattern matching
        query_original = query.strip()
        query_lower = query_original.lower()  # For pattern matching only
        
        entity_hints = self._extract_entity_hints(query_original)  # ✅ Use original
        
        # ✅ CRITICAL: Check complex patterns FIRST before superlative detection
        # Complex queries like "from country X received award Y" should not be treated as superlatives
        complex_pattern = self._check_complex_patterns(query_lower, entity_hints)
        if complex_pattern:
            print(f"[Analyzer] 🔀 COMPLEX PATTERN DETECTED")
            return complex_pattern
        
        # ✅ NEW: Detect superlative patterns (but after complex patterns)
        if self._is_superlative_query(query_lower):
            superlative_type = self._extract_superlative_type(query_lower)
            entity_hints['superlative'] = superlative_type
            print(f"[Analyzer] 🔝 SUPERLATIVE DETECTED: {superlative_type}")
        
        # ✅ CRITICAL: Detect "language" keyword and FORCE override - MUST come before other checks
        language_keywords = ['language', 'spoken', 'dialogue', 'in what language', 'which language', 'what language']
        if any(kw in query_lower for kw in language_keywords):
            print(f"[Analyzer] 🔒 GUARDRAIL: Language query detected")
            print(f"[Analyzer]    Forcing relation to: original_language_of_film_or_tv_show")
            # Force relation to original_language_of_film_or_tv_show (P364)
            pattern = QueryPattern(
                pattern_type='forward',
                relation='original_language_of_film_or_tv_show',
                subject_type='movie',
                object_type='language',
                confidence=0.99,
                extracted_entities=entity_hints
            )
            print(f"[Analyzer]    ✅ Language pattern enforced")
            return pattern
        
        # ✅ NEW: Detect "characters" keyword and override relation
        if 'character' in query_lower:
            print(f"[Analyzer] 🔒 GUARDRAIL: Characters query detected")
            # Force relation to characters (P674)
            pattern = QueryPattern(
                pattern_type='forward',
                relation='characters',
                subject_type='movie',
                object_type='person',  # Characters are fictional persons
                confidence=0.95,
                extracted_entities=entity_hints
            )
            print(f"[Analyzer]    Relation set to: characters")
            return pattern
        
        # ✅ NEW: Detect "production company" vs "producer" 
        if 'production company' in query_lower or 'production studio' in query_lower:
            print(f"[Analyzer] 🔒 GUARDRAIL: Production company query detected")
            pattern = QueryPattern(
                pattern_type='forward',
                relation='production_company',
                subject_type='movie',
                object_type='organization',
                confidence=0.95,
                extracted_entities=entity_hints
            )
            print(f"[Analyzer]    Relation set to: production_company")
            return pattern
        
        # Try transformer classifier first
        if self.use_transformer:
            pattern = self._transformer_classify(query_lower, entity_hints)
            if pattern:
                return pattern
        
        # Fallback to rule-based patterns
        # 1. Check forward patterns (Movie → Property)
        pattern = self._check_forward_patterns(query_lower, entity_hints)
        if pattern:
            return pattern
        
        # 2. Check reverse patterns (Person → Movies)
        pattern = self._check_reverse_patterns(query_lower, entity_hints)
        if pattern:
            return pattern
        
        # 3. Check verification patterns (Does X have relation Y?)
        pattern = self._check_verification_patterns(query_lower, entity_hints)
        if pattern:
            return pattern
        
        # If no patterns matched, return None
        return None