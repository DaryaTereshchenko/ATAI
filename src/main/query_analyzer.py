"""
Query Analyzer - Understands query intent and structure.
NOW USES: Relation-first approach.
"""

import re
from typing import Optional
from dataclasses import dataclass
import os

# No transformer SPARQL classifier is used anymore
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
    """Analyzes queries to understand intent and structure."""
    
    def __init__(
        self,
        use_transformer: bool = True,
        transformer_model_path: str = None,
        sparql_handler=None,
        embedding_matcher=None,
        relation_classifier_path: Optional[str] = None
    ):
        """
        Initialize query analyzer.
        
        Args:
            use_transformer: Deprecated (transformer classifier removed)
            transformer_model_path: Deprecated
            sparql_handler: SPARQLHandler for dynamic schema extraction
            embedding_matcher: EmbeddingRelationMatcher for fallback
            relation_classifier_path: Path to DistilBERT relation classifier model  # ✅ NEW
        """
        # Transformer-based SPARQL classifier has been removed — always use relation-first approach
        self.use_transformer = False
        self.transformer_classifier = None

        # Always initialize rule-based patterns as fallback
        self._setup_patterns()
        self._setup_entity_hints()
        
        # Mapping for subject/object types
        self._setup_type_mappings()
        
        # ✅ NEW: Extract supported relations from graph if handler provided
        self.sparql_handler = sparql_handler
        if sparql_handler:
            self._update_type_mappings_from_graph()
            self._build_property_synonyms()
        
        # ✅ NEW: Initialize hybrid relation analyzer with embedding matcher
        self.relation_analyzer = None
        if sparql_handler:
            try:
                from src.main.relation_classifier import HybridRelationAnalyzer
                
                # ✅ FIX: Ensure classifier path is passed and exists
                if relation_classifier_path and not os.path.exists(relation_classifier_path):
                    print(f"⚠️  Relation classifier path does not exist: {relation_classifier_path}")
                    relation_classifier_path = None
                
                # ✅ Try to use default path if not provided
                if not relation_classifier_path:
                    default_path = os.path.join(
                        os.path.dirname(__file__), 
                        '..', '..', 
                        'models', 
                        'relation_classifier'
                    )
                    default_path = os.path.abspath(default_path)
                    
                    if os.path.exists(default_path):
                        print(f"ℹ️  Using default classifier path: {default_path}")
                        relation_classifier_path = default_path
                
                self.relation_analyzer = HybridRelationAnalyzer(
                    classifier_path=relation_classifier_path,
                    sparql_handler=sparql_handler,
                    use_sbert=True,
                    embedding_matcher=embedding_matcher
                )
                
                # ✅ Check if BERT classifier was loaded
                if self.relation_analyzer.bert_classifier is not None:
                    print("✅ Hybrid relation analyzer initialized (with DistilBERT + SBERT + embedding fallback)\n")
                else:
                    print("✅ Hybrid relation analyzer initialized (SBERT + embedding fallback only)\n")
                
            except Exception as e:
                print(f"⚠️  Relation analyzer initialization failed: {e}")
                import traceback
                traceback.print_exc()
                
                # Fallback to basic keyword analyzer
                try:
                    from src.main.relation_analyzer import RelationAnalyzer
                    self.relation_analyzer = RelationAnalyzer(sparql_handler)
                    print("✅ Using basic keyword-based relation analyzer\n")
                except Exception as e2:
                    print(f"❌ Failed to load any relation analyzer: {e2}")
    
    def _update_type_mappings_from_graph(self):
        """Update type mappings dynamically from knowledge graph."""
        if not self.sparql_handler:
            return
        
        from rdflib import URIRef, RDFS
        P31 = URIRef("http://www.wikidata.org/prop/direct/P31")
        
        # Extract all Wikidata properties used in the graph
        all_properties = set()
        for s, p, o in self.sparql_handler.graph:
            pred_str = str(p)
            if 'wikidata.org/prop/direct/P' in pred_str:
                all_properties.add(pred_str)
        
        # Infer object types for each property
        property_to_object_types = {}
        
        for prop_uri in all_properties:
            prop_ref = URIRef(prop_uri)
            
            # Sample triples to infer object type
            object_types = {}
            sample_count = 0
            
            for s, p, o in self.sparql_handler.graph.triples((None, prop_ref, None)):
                if isinstance(o, URIRef):
                    for type_uri in self.sparql_handler.graph.objects(o, P31):
                        type_str = str(type_uri)
                        if 'Q5' in type_str:
                            object_types['person'] = object_types.get('person', 0) + 1
                        elif 'Q11424' in type_str:
                            object_types['movie'] = object_types.get('movie', 0) + 1
                        elif 'Q6256' in type_str:
                            object_types['country'] = object_types.get('country', 0) + 1
                        elif 'Q201658' in type_str:
                            object_types['genre'] = object_types.get('genre', 0) + 1
                
                sample_count += 1
                if sample_count >= 20:  # Sample 20 triples per property
                    break
            
            # Pick most common object type
            if object_types:
                most_common = max(object_types.items(), key=lambda x: x[1])[0]
                property_to_object_types[prop_uri] = most_common
        
        # Update type_mappings with inferred data
        for prop_uri, obj_type in property_to_object_types.items():
            if '/P' in prop_uri:
                prop_id = prop_uri.split('/P')[-1]
                
                # Map to friendly name if we have a pattern for it
                friendly_name_map = {
                    '57': 'director',
                    '161': 'cast_member',
                    '58': 'screenwriter',
                    '162': 'producer',
                    '136': 'genre',
                    '577': 'publication_date',
                    '495': 'country_of_origin',
                    '166': 'award_received'
                }
                
                if prop_id in friendly_name_map:
                    relation_name = friendly_name_map[prop_id]
                    
                    # Update type mapping with inferred object type
                    if relation_name in self.type_mappings:
                        # Update object type with inferred one
                        self.type_mappings[relation_name]['object'] = obj_type
    
    def _setup_type_mappings(self):
        """Map relations to subject/object types - ✅ NOW DYNAMIC."""
        
        # ✅ BASELINE: Start with common mappings
        self.type_mappings = {
            'director': {'subject': 'movie', 'object': 'person'},
            'cast_member': {'subject': 'movie', 'object': 'person'},
            'screenwriter': {'subject': 'movie', 'object': 'person'},
            'producer': {'subject': 'movie', 'object': 'person'},
            'genre': {'subject': 'movie', 'object': 'string'},
            'publication_date': {'subject': 'movie', 'object': 'date'},
            'rating': {'subject': 'movie', 'object': 'string'},
            'country_of_origin': {'subject': 'movie', 'object': 'string'},
            'original_language': {'subject': 'movie', 'object': 'string'},
            'language': {'subject': 'movie', 'object': 'string'},
        }
        
        # ✅ DYNAMIC: These will be updated/extended from graph in _update_type_mappings_from_graph()
        # This method is called later in __init__ if sparql_handler is available
    
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
        
        # ✅ CRITICAL CHANGE: Make patterns generic, not movie-specific
        # Use entity-agnostic language
        
        # ==================== FORWARD PATTERNS ====================
        self.forward_patterns = [
            # ✅ NEW: Specific "Who [verb]ed X?" patterns (highest priority)
            {
                'regex': r'\b(?:who)\s+(produced?|directed?|wrote|screenwrote|acted\s+in|starred\s+in)\s+',
                'relation_map': {
                    'produce': 'producer',
                    'produced': 'producer',
                    'direct': 'director',
                    'directed': 'director',
                    'wrote': 'screenwriter',
                    'screenwrote': 'screenwriter',
                    'acted in': 'cast_member',
                    'starred in': 'cast_member'
                },
                'subject': 'movie',
                'object': 'person',
                'confidence': 0.98
            },
            
            # Generic "what is X of Y" patterns
            {
                'regex': r'\b(?:what|which)\s+(?:is|was|are|were)\s+(?:the\s+)?(\w+)\s+(?:of|for)\s+',
                'relation_group': 1,  # Extract relation from regex group
                'subject': 'entity',
                'object': 'entity',
                'confidence': 0.85
            },
            {
                'regex': r'\b(?:who|what)\s+(\w+(?:\s+\w+)?)\s+',
                'relation_group': 1,
                'subject': 'entity',
                'object': 'entity',
                'confidence': 0.80
            },
            
            # Specific known patterns (higher confidence)
            {
                'regex': r'\b(?:who)\s+directed\s+',
                'relation': 'director',
                'subject': 'movie',
                'object': 'person',
                'confidence': 0.98
            },
            {
                'regex': r'\bfrom\s+(?:what|which)\s+country\s+',
                'relation': 'country_of_origin',
                'subject': 'entity',
                'object': 'string',
                'confidence': 0.98
            },
            
            # ✅ NEW: Language patterns (add BEFORE generic patterns for higher priority)
            {
                'regex': r'\b(?:what|which)\s+(?:is|was)\s+(?:the\s+)?(?:original\s+)?language\s+(?:of|in|for)\s+',
                'relation': 'original_language',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.98
            },
            {
                'regex': r'\b(?:in|from)\s+(?:what|which)\s+language\s+(?:is|was)\s+',
                'relation': 'original_language',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.98
            },
            {
                'regex': r'\blanguage\s+(?:is|of)\s+[\'""]?([^\'"",?]+)[\'""]?',
                'relation': 'original_language',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.95
            },
            {
                'regex': r'(?:spoken|filmed|made)\s+in\s+(?:what|which)\s+language',
                'relation': 'original_language',
                'subject': 'movie',
                'object': 'string',
                'confidence': 0.95
            },
            
            # ✅ Keep existing movie patterns but add generic fallbacks
            # ...existing code (keep all your current patterns)...
        ]
        
        # ==================== REVERSE PATTERNS ====================
        self.reverse_patterns = [
            # Generic reverse patterns
            {
                'regex': r'\b(?:what|which)\s+(\w+)\s+(?:did|has|have)\s+.*?\s+(\w+)\b',
                'relation_group': 1,
                'confidence': 0.75
            },
            
            # ✅ Keep existing specific patterns
            # ...existing code...
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
    
    def analyze(self, query: str) -> QueryPattern:
        """
        Analyze query to detect pattern, relation, and entities.
        """
        print(f"[Analyzer] 🔍 Starting relation-based analysis")
        
        # ✅ NEW: Check for superlative FIRST (before relation analysis)
        is_superlative = self._is_superlative_query(query)
        superlative_type = None
        
        if is_superlative:
            superlative_type = self._extract_superlative_type(query)
            print(f"[Analyzer] 🔝 Superlative query detected: {superlative_type}")
        
        # ✅ ADD: Keyword-based relation override BEFORE calling relation_analyzer
        query_lower = query.lower()
        keyword_overrides = {
            'director': ['who directed', 'director of', 'who was the director'],
            'cast_member': ['actors in', 'cast of', 'who starred', 'who acted'],
            'publication_date': ['when was', 'release date', 'when did', 'released'],
            'country_of_origin': ['what country', 'from what country', 'country of origin'],
            'composer': ['who composed', 'composer of', 'music by'],
            'award_received': ['what awards', 'awards did', 'won'],
            'filming_location': ['filming location', 'where was filmed', 'shot'],
            'director_of_photography': ['cinematographer', 'director of photography'],
            'production_company': ['production company', 'produced by'],
        }
        
        keyword_detected_relation = None
        for relation, keywords in keyword_overrides.items():
            if any(kw in query_lower for kw in keywords):
                keyword_detected_relation = relation
                break
        
        # Call relation analyzer
        relation_result = self.relation_analyzer.analyze(query)
        
        # ✅ FIX: Access RelationQuery attributes with dot notation, not subscript
        if keyword_detected_relation and relation_result.confidence < 85.0:
            print(f"[Analyzer] 🔧 Overriding low-confidence detection with keyword match: {keyword_detected_relation}")
            relation_result.relation = keyword_detected_relation
            relation_result.confidence = 95.0
            relation_result.method = 'KEYWORD OVERRIDE'
        
        # ✅ STEP 2: Infer query direction from entity mentions
        query_lower = query.lower()
        entity_hints = self._extract_entity_hints(query)
        
        # ✅ NEW: Log entity hints
        print(f"\n[Analyzer] 🔍 BREAKPOINT A3: Entity hints extracted")
        print(f"[Analyzer]    Quoted: {entity_hints.get('quoted', [])}")
        print(f"[Analyzer]    Capitalized: {entity_hints.get('capitalized', [])}")
        print(f"[Analyzer]    Contextual: {entity_hints.get('contextual', [])}")
        
        # Check for forward indicators (entity → property)
        has_entity = (entity_hints.get('quoted') or 
                      entity_hints.get('capitalized'))
        
        # Check for reverse indicators (property → entities)
        reverse_indicators = ['what movies', 'which films', 'list of', 
                             'all movies', 'films that']
        is_reverse = any(ind in query_lower for ind in reverse_indicators)
        
        # Check for verification (does X have Y?)
        verification_indicators = ['did', 'is', 'was', 'does', 'has']
        is_verification = (any(ind in query_lower for ind in verification_indicators) and
                           has_entity and len(entity_hints.get('quoted', [])) >= 2)
        
        # Determine pattern type
        if is_verification:
            pattern_type = 'verification'
        elif is_reverse:
            pattern_type = 'reverse'
        else:
            pattern_type = 'forward'
        
        # ✅ NEW: Normalize relation name with context
        print(f"\n[Analyzer] 🔍 BREAKPOINT A4: Normalizing relation name")
        print(f"[Analyzer]    Original relation: '{relation_result.relation}'")
        
        normalized_relation = self.normalize_relation_name(relation_result.relation, query)
        
        print(f"[Analyzer]    Normalized relation: '{normalized_relation}'")
        
        if normalized_relation != relation_result.relation:
            print(f"[Analyzer]    ⚠️ Relation name changed during normalization!")
            print(f"[Analyzer]       Before: {relation_result.relation}")
            print(f"[Analyzer]       After: {normalized_relation}")
            
            # Update the relation in the query
            relation_result.relation = normalized_relation
        
        # ✅ FIX: Access RelationQuery attributes with dot notation
        entity_hints['keywords'] = relation_result.keywords
        
        if is_superlative:
            entity_hints['superlative'] = superlative_type
            print(f"[Analyzer] 📊 Marked as superlative {pattern_type} query ({superlative_type})")
        
        print(f"[Analyzer] ✅ Detected pattern: {pattern_type}")
        print(f"[Analyzer]    Relation: {relation_result.relation}")
        print(f"[Analyzer]    Relation URI: {relation_result.relation_uri}\n")
        
        return QueryPattern(
            pattern_type=pattern_type,
            relation=relation_result.relation,
            subject_type=relation_result.subject_type,
            object_type=relation_result.object_type,
            confidence=relation_result.confidence,
            extracted_entities=entity_hints
        )
    
    def _pattern_based_analyze(self, query: str) -> Optional[QueryPattern]:
        """Fallback: Pattern-based analysis when relation detection fails."""
        print(f"[Analyzer] 🔍 Using pattern-based fallback")
        
        query_lower = query.lower()
        entity_hints = self._extract_entity_hints(query)
        
        # Try forward patterns first
        pattern = self._check_forward_patterns(query_lower, entity_hints)
        if pattern:
            return pattern
        
        # Try reverse patterns
        pattern = self._check_reverse_patterns(query_lower, entity_hints)
        if pattern:
            return pattern
        
        # Try verification patterns
        pattern = self._check_verification_patterns(query_lower, entity_hints)
        if pattern:
            return pattern
        
        print(f"[Analyzer] ❌ No patterns matched")
        return None
    
    def _check_forward_patterns(self, query: str, entity_hints: dict) -> Optional[QueryPattern]:
        """Check and apply forward patterns to the query."""
        for pattern in self.forward_patterns:
            regex = pattern['regex']
            relation = pattern.get('relation')
            relation_group = pattern.get('relation_group', 0)
            relation_map = pattern.get('relation_map', {})
            
            print(f"[Analyzer]   Testing forward pattern: {regex}")
            
            match = re.search(regex, query)
            if match:
                # ✅ NEW: Handle relation_map patterns
                if relation_map and match.lastindex and match.lastindex >= 1:
                    # Extract verb and map to relation
                    verb = match.group(1).lower()
                    for key, mapped_relation in relation_map.items():
                        if key in verb:
                            relation = mapped_relation
                            break
                
                # Extract subject/object if pattern expects them
                subject = None
                object = None
                
                try:
                    if 'subject' in pattern:
                        if 'entity' in str(pattern.get('subject', '')):
                            if match.lastindex and match.lastindex >= 1 and not relation_map:
                                subject = match.group(1)
                    
                    if 'object' in pattern:
                        if 'entity' in str(pattern.get('object', '')):
                            if match.lastindex and match.lastindex >= 2:
                                object = match.group(2)
                
                except IndexError:
                    print(f"[Analyzer]   ⚠️  Pattern has insufficient capture groups")
                    continue
                
                # Use entity hints if available
                if entity_hints.get('quoted') and not object:
                    object = entity_hints['quoted'][0]
                
                # Get relation from regex group if not set
                if not relation and relation_group > 0:
                    try:
                        if match.lastindex and match.lastindex >= relation_group:
                            relation = match.group(relation_group)
                    except IndexError:
                        pass
                
                # Ensure we have a valid relation
                if not relation:
                    print(f"[Analyzer]   ⚠️  No relation detected in pattern")
                    continue
                
                if relation not in self.type_mappings:
                    print(f"[Analyzer]   ⚠️  Relation '{relation}' not in type mappings")
                    continue
                
                print(f"[Analyzer]   ➡️  Match found: Subject='{subject}', Object='{object}', Relation='{relation}'")
                
                return QueryPattern(
                    pattern_type='forward',
                    relation=relation,
                    subject_type=self.type_mappings[relation]['subject'],
                    object_type=self.type_mappings[relation]['object'],
                    confidence=pattern['confidence'],
                    extracted_entities=entity_hints
                )
        
        return None
    
    def _check_reverse_patterns(self, query: str, entity_hints: dict) -> Optional[QueryPattern]:
        """Check and apply reverse patterns to the query."""
        for pattern in self.reverse_patterns:
            regex = pattern['regex']
            relation = pattern.get('relation')
            
            # Debug: Show the regex being applied
            print(f"[Analyzer]   Testing reverse pattern: {regex}")
            
            match = re.search(regex, query)
            if match:
                # ✅ FIX: Safely extract groups with bounds checking
                subject = None
                object = None
                
                try:
                    if match.lastindex and match.lastindex >= 1:
                        subject = match.group(1)
                    if match.lastindex and match.lastindex >= 2:
                        object = match.group(2)
                except IndexError:
                    print(f"[Analyzer]   ⚠️  Pattern has insufficient capture groups")
                    continue
                
                # ✅ Ensure we have a valid relation
                if not relation:
                    print(f"[Analyzer]   ⚠️  No relation detected in pattern")
                    continue
                
                if relation not in self.type_mappings:
                    print(f"[Analyzer]   ⚠️  Relation '{relation}' not in type mappings")
                    continue
                
                # Debug: Show the match results
                print(f"[Analyzer]   ➡️  Match found: Subject='{subject}', Object='{object}'")
                
                return QueryPattern(
                    pattern_type='reverse',
                    relation=relation,
                    subject_type=self.type_mappings[relation]['object'],
                    object_type=self.type_mappings[relation]['subject'],
                    confidence=pattern['confidence'],
                    extracted_entities=entity_hints
                )
        
        return None
    
    def _check_verification_patterns(self, query: str, entity_hints: dict) -> Optional[QueryPattern]:
        """Check and apply verification patterns to the query."""
        for pattern in self.verification_patterns:
            regex = pattern['regex']
            relation_map = pattern.get('relation_map', {})
            
            # Debug: Show the regex being applied
            print(f"[Analyzer]   Testing verification pattern: {regex}")
            
            match = re.search(regex, query)
            if match:
                # ✅ FIX: Safely extract groups with bounds checking
                subject = None
                object = None
                
                try:
                    if match.lastindex and match.lastindex >= 1:
                        subject = match.group(1)
                    if match.lastindex and match.lastindex >= 2:
                        object = match.group(2)
                except IndexError:
                    print(f"[Analyzer]   ⚠️  Pattern has insufficient capture groups")
                    continue
                
                # Determine relation from mapping
                relation = None
                for key, value in relation_map.items():
                    if key in query:
                        relation = value
                        break
                
                # If no relation found in map, try to get from pattern directly
                if not relation:
                    relation = pattern.get('relation')
                
                # ✅ Ensure we have a valid relation
                if not relation:
                    print(f"[Analyzer]   ⚠️  No relation detected")
                    continue
                
                if relation not in self.type_mappings:
                    print(f"[Analyzer]   ⚠️  Relation '{relation}' not in type mappings")
                    continue
                
                # Debug: Show the match results
                print(f"[Analyzer]   ➡️  Match found: Subject='{subject}', Object='{object}', Relation='{relation}'")
                
                return QueryPattern(
                    pattern_type='verification',
                    relation=relation,
                    subject_type=self.type_mappings[relation]['subject'],
                    object_type=self.type_mappings[relation]['object'],
                    confidence=pattern['confidence'],
                    extracted_entities=entity_hints
                )
        
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
    
    def get_supported_relations(self) -> list:
        """
        Get list of all supported relations.
        
        Returns:
            List of relation names
        """
        return list(self.type_mappings.keys())
    
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
    
    def normalize_relation_name(self, relation: str, query_context: str = "") -> str:
        """
        Normalize relation name using synonym mapping with priority AND context.
        ✅ NOW: Considers query context to determine subject type.
        """
        relation_lower = relation.lower()
        query_lower = query_context.lower()
        
        # Direct match
        if relation_lower in self.property_synonyms:
            canonical = self.property_synonyms[relation_lower]
            return canonical
        
        # ✅ ENHANCED: For 'country' queries, detect subject type from context
        if relation_lower == 'country' or 'country' in relation_lower:
            # Check for person-related keywords
            is_person_query = any(
                keyword in query_lower 
                for keyword in ['born', 'birth', 'citizenship', 'nationality', 'was born', 'is from']
            )
            
            # Check for movie-related keywords
            is_movie_query = any(
                keyword in query_lower
                for keyword in ['movie', 'film', 'produced', 'released', 'from what country']
            )
            
            # Detect subject type from context
            if is_person_query:
                print(f"[Analyzer] 🎯 Detected PERSON query for country → using P27 (citizenship)")
                return 'country_of_citizenship'
            elif is_movie_query:
                print(f"[Analyzer] 🎯 Detected MOVIE query for country → using P495 (country of origin)")
                return 'country_of_origin'
        
        # ✅ Check for partial matches with priority (existing logic)
        candidates = []
        for synonym, canonical in self.property_synonyms.items():
            if relation_lower in synonym or synonym in relation_lower:
                priority = self.property_priorities.get(canonical, 0)
                candidates.append((canonical, priority))
        
        if candidates:
            # Sort by priority (descending)
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_match = candidates[0][0]
            print(f"[Analyzer] 🎯 Disambiguated '{relation}' → '{best_match}' (priority-based)")
            return best_match
        
        # Fallback: return as-is
        return relation

    def _build_property_synonyms(self):
        """Build synonym mappings for properties to improve relation detection."""
        
        # ✅ DYNAMIC: Extract properties from SPARQLGenerator instead of hard-coding
        if not hasattr(self, 'sparql_handler') or not self.sparql_handler:
            print("⚠️  No SPARQL handler available for dynamic property extraction")
            return
        
        # Get relation URIs from SPARQLGenerator (already extracted dynamically)
        try:
            from src.main.sparql_generator import SPARQLGenerator
            generator = SPARQLGenerator(self.sparql_handler)
            available_relations = generator.relation_uris
            
            print(f"🔍 Building synonyms for {len(available_relations)} dynamically extracted properties")
        except Exception as e:
            print(f"⚠️  Could not get dynamic properties: {e}")
            available_relations = {}
        
        # Build comprehensive synonym map (only for properties that exist in graph)
        self.property_synonyms = {}
        self.property_priorities = {}
        
        # ✅ Define synonym mappings (independent of whether property exists)
        synonym_definitions = {
            'director': {
                'synonyms': ['director', 'directed', 'filmmaker', 'directed by', 'film director'],
                'priority': 10
            },
            'cast_member': {
                'synonyms': ['cast', 'cast member', 'actor', 'actress', 'starred', 'starring', 'acted', 'performer'],
                'priority': 9
            },
            'screenwriter': {
                'synonyms': ['screenwriter', 'writer', 'screenplay', 'wrote', 'written by'],
                'priority': 8
            },
            'producer': {
                'synonyms': ['producer', 'produced', 'produced by'],
                'priority': 8
            },
            'genre': {
                'synonyms': ['genre', 'type', 'category', 'kind'],
                'priority': 9
            },
            'country_of_origin': {
                'synonyms': ['country', 'country of origin', 'from what country', 'made in', 'produced in'],
                'priority': 10
            },
            'country_of_citizenship': {
                'synonyms': ['citizenship', 'nationality', 'born in'],
                'priority': 5
            },
            'original_language': {
                'synonyms': ['language', 'original language', 'spoken', 'dialogue', 'filmed in language'],
                'priority': 10
            },
            'publication_date': {
                'synonyms': ['release', 'release date', 'released', 'came out', 'published'],
                'priority': 9
            },
            'rating': {
                'synonyms': ['rating'],
                'priority': 8
            },
            'award_received': {
                'synonyms': ['award', 'won'],
                'priority': 8
            },
            'distributed_by': {
                'synonyms': ['distributed', 'distributor', 'distribution'],
                'priority': 8
            },
            'composer': {
                'synonyms': ['composed', 'composer', 'soundtrack', 'music', 'score'],
                'priority': 8
            },
            'director_of_photography': {
                'synonyms': ['cinematography', 'cinematographer', 'director of photography'],
                'priority': 8
            },
            'film_editor': {
                'synonyms': ['edited', 'editor', 'film editor'],
                'priority': 8
            },
            'production_company': {
                'synonyms': ['production company', 'studio', 'made by', 'produced by company'],
                'priority': 8
            },
            'filming_location': {
                'synonyms': ['location', 'filmed', 'shot'],
                'priority': 7
            },
        }
        
        # ✅ Only add synonyms for properties that actually exist in the graph
        for relation_name, relation_uri in available_relations.items():
            if relation_name in synonym_definitions:
                definition = synonym_definitions[relation_name]
                
                # Add all synonyms pointing to this canonical relation
                for synonym in definition['synonyms']:
                    self.property_synonyms[synonym] = relation_name
                
                # Set priority
                self.property_priorities[relation_name] = definition['priority']
                
                print(f"   ✅ Added {len(definition['synonyms'])} synonyms for '{relation_name}'")
        
        print(f"✅ Built {len(self.property_synonyms)} synonym mappings for {len(self.property_priorities)} properties")