"""
Dynamic Relation Manager - Discovers and manages relations from knowledge graph.
Provides intelligent relation matching and URI resolution.
"""

import re
from typing import Dict, List, Optional, Tuple
from rdflib import Graph, URIRef, RDFS
from collections import defaultdict
from difflib import SequenceMatcher


class RelationManager:
    """
    Manages relations dynamically from knowledge graph.
    Provides fuzzy matching and intelligent relation resolution.
    """
    
    def __init__(self, graph: Graph):
        """
        Initialize relation manager by discovering relations from graph.
        
        Args:
            graph: RDFLib graph containing knowledge base
        """
        self.graph = graph
        self.relations: Dict[str, Dict] = {}  # relation_key -> {uri, label, aliases, count}
        self.uri_to_key: Dict[str, str] = {}  # URI -> relation_key
        self._discover_relations()
    
    def _discover_relations(self):
        """Discover all relations from the knowledge graph."""
        print("🔍 Discovering relations from knowledge graph...")
        
        # Track property usage
        property_counts = defaultdict(int)
        property_labels = {}
        
        # Scan all triples to find used properties
        for s, p, o in self.graph:
            prop_uri = str(p)
            property_counts[prop_uri] += 1
            
            # Try to get label for property
            if prop_uri not in property_labels:
                prop_ref = URIRef(prop_uri)
                for label in self.graph.objects(prop_ref, RDFS.label):
                    property_labels[prop_uri] = str(label)
                    break
        
        # Build relations dictionary
        for prop_uri, count in property_counts.items():
            # Extract key from URI or label
            if prop_uri in property_labels:
                label = property_labels[prop_uri]
                key = self._normalize_relation_name(label)
            else:
                # Extract from URI
                key = self._extract_key_from_uri(prop_uri)
            
            # Store relation info
            self.relations[key] = {
                'uri': prop_uri,
                'label': property_labels.get(prop_uri, key),
                'aliases': self._generate_aliases(key, property_labels.get(prop_uri, '')),
                'count': count,
                'property_code': self._extract_property_code(prop_uri)
            }
            self.uri_to_key[prop_uri] = key
        
        print(f"✅ Discovered {len(self.relations)} relations")
        
        # Show top relations
        top_relations = sorted(self.relations.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
        print(f"📊 Top 10 relations by usage:")
        for key, info in top_relations:
            print(f"   • {key} ({info['property_code']}): {info['count']} triples")
    
    def _normalize_relation_name(self, name: str) -> str:
        """
        Normalize relation name to standard format.
        
        Examples:
            "cast member" -> "cast_member"
            "Director Of Photography" -> "director_of_photography"
        """
        # Convert to lowercase and replace spaces/hyphens with underscores
        normalized = name.lower()
        normalized = re.sub(r'[\s\-]+', '_', normalized)
        normalized = re.sub(r'[^\w_]', '', normalized)
        return normalized
    
    def _extract_key_from_uri(self, uri: str) -> str:
        """Extract relation key from URI."""
        # Extract last part of URI
        if '/' in uri:
            key = uri.split('/')[-1]
        elif '#' in uri:
            key = uri.split('#')[-1]
        else:
            key = uri
        
        # Remove property prefix (P57 -> director)
        if key.startswith('P') and key[1:].isdigit():
            # This is a Wikidata property code - keep it as is for now
            return f"property_{key}"
        
        return self._normalize_relation_name(key)
    
    def _extract_property_code(self, uri: str) -> Optional[str]:
        """Extract Wikidata property code (e.g., P57) from URI."""
        match = re.search(r'/P(\d+)$', uri)
        if match:
            return f"P{match.group(1)}"
        return None
    
    def _generate_aliases(self, key: str, label: str) -> List[str]:
        """Generate common aliases for a relation."""
        aliases = [key]
        
        # Add label variations
        if label:
            aliases.append(self._normalize_relation_name(label))
        
        # ✅ CRITICAL: Add common variations AND generic terms
        variations = {
            'cast_member': ['actor', 'actress', 'cast', 'stars', 'starring', 'acted', 'cast member'],
            'director': ['directed_by', 'filmmaker', 'directed', 'director'],
            'screenwriter': ['writer', 'screenplay_by', 'written_by', 'screenwriter', 'wrote'],
            'producer': ['produced_by', 'producer', 'produced'],
            'publication_date': ['release_date', 'released', 'premiere', 'date', 'release'],
            # ✅ CRITICAL: Map BOTH 'country' and 'country_of_origin' to P495
            'country_of_origin': ['country', 'origin_country', 'from_country', 'from', 'country_of', 'made_in', 'country of origin'],
            'genre': ['type', 'category', 'genre'],
            'award_received': ['award', 'won_award', 'awards', 'award received'],
            # ✅ ENHANCED: Language aliases
            'original_language_of_film_or_tv_show': [
                'language', 'original_language', 'spoken_language', 
                'language_of_work', 'dialogue_language', 'audio_language',
                'film_language', 'movie_language', 'original language'
            ],
            # ✅ Rating property aliases
            'imda_rating': ['rating', 'user_rating', 'movie_rating', 'film_rating'],
            'mpa_film_rating': ['rating', 'user_rating', 'movie_rating', 'mpaa_rating'],
            'fsk_film_rating': ['rating', 'user_rating', 'movie_rating'],
            'bbfc_rating': ['rating', 'user_rating', 'movie_rating'],
            'eirin_film_rating': ['rating', 'user_rating', 'movie_rating'],
            'australian_classification': ['rating', 'user_rating', 'movie_rating'],
            'kijkwijzer_rating': ['rating', 'user_rating', 'movie_rating'],
            'medierådet_rating': ['rating', 'user_rating', 'movie_rating'],
            'classind_rating': ['rating', 'user_rating', 'movie_rating'],
            'nmhh_film_rating': ['rating', 'user_rating', 'movie_rating'],
            'cnc_film_rating_france': ['rating', 'user_rating', 'movie_rating'],
            'filmiroda_rating': ['rating', 'user_rating', 'movie_rating'],
            'jmk_film_rating': ['rating', 'user_rating', 'movie_rating'],
            'icaa_rating': ['rating', 'user_rating', 'movie_rating'],
            'mtrcb_rating': ['rating', 'user_rating', 'movie_rating'],
            'bamid_film_rating': ['rating', 'user_rating', 'movie_rating'],
            'rars_rating': ['rating', 'user_rating', 'movie_rating'],
            'cnc_film_rating_romania': ['rating', 'user_rating', 'movie_rating'],
            'igac_rating': ['rating', 'user_rating', 'movie_rating'],
            'rcq_classification': ['rating', 'user_rating', 'movie_rating'],
            'ifco_rating': ['rating', 'user_rating', 'movie_rating'],
            'rtc_film_rating': ['rating', 'user_rating', 'movie_rating'],
            'kavi_rating': ['rating', 'user_rating', 'movie_rating'],
            'fpb_rating': ['rating', 'user_rating', 'movie_rating'],
            'incaa_film_rating': ['rating', 'user_rating', 'movie_rating'],
            'kmrb_film_rating': ['rating', 'user_rating', 'movie_rating'],
            'oflc_classification': ['rating', 'user_rating', 'movie_rating'],
            'narrative_location': ['setting', 'set_in', 'takes_place'],
        }
        
        if key in variations:
            aliases.extend(variations[key])
        
        # ✅ NEW: Also check if this key should be mapped TO a generic term
        # For example, if key is 'country_of_origin', also add 'country' as an alias
        reverse_generic_map = {
            'country_of_origin': ['country'],
            'original_language_of_film_or_tv_show': ['language'],
            'publication_date': ['date', 'release'],
        }
        
        if key in reverse_generic_map:
            aliases.extend(reverse_generic_map[key])
        
        return list(set(aliases))
    
    def find_relation(self, query_text: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """
        Find best matching relation(s) from query text using fuzzy matching.
        
        ✅ ENHANCED: Better generic keyword detection and context-aware matching.
        """
        matches = []
        
        query_original = query_text.strip()
        query_lower = query_original.lower()
        
        # ✅ ENHANCED: Detect context for generic keywords
        generic_keyword_contexts = {
            'country': {
                'keywords': ['country', 'nation'],
                'context_patterns': {
                    'origin': ['from', 'of', 'origin', 'made in', 'produced in'],
                    'filming': ['filmed', 'shot', 'location'],
                    'narrative': ['set in', 'takes place', 'story set'],
                }
            },
            'language': {
                'keywords': ['language', 'spoken', 'dialogue'],
                'specific_relations': ['original_language_of_film_or_tv_show', 'language_of_work_or_name']
            },
            'rating': {
                'keywords': ['rating', 'rated', 'classification'],
                'specific_relations': ['imda_rating', 'mpa_film_rating', 'fsk_film_rating']
            },
        }
        
        # ✅ Check for generic keywords with context
        for generic, config in generic_keyword_contexts.items():
            if any(kw in query_lower for kw in config['keywords']):
                print(f"[RelationManager] Detected generic keyword: '{generic}'")
                
                # Try to determine specific relation from context
                if 'context_patterns' in config:
                    for context_type, patterns in config['context_patterns'].items():
                        if any(pattern in query_lower for pattern in patterns):
                            print(f"[RelationManager]    Context: {context_type}")
                            # Map context to specific relation
                            context_to_relation = {
                                'origin': 'country_of_origin',
                                'filming': 'filming_location',
                                'narrative': 'narrative_location',
                            }
                            specific_rel = context_to_relation.get(context_type)
                            if specific_rel and specific_rel in self.relations:
                                print(f"[RelationManager]    Using context-specific: {specific_rel}")
                                matches.append((specific_rel, self.relations[specific_rel]['uri'], 0.95))
                                continue
                
                # Fallback: boost all specific relations for this generic type
                if 'specific_relations' in config:
                    for specific in config['specific_relations']:
                        if specific in self.relations:
                            matches.append((specific, self.relations[specific]['uri'], 0.90))
        
        # Continue with existing matching logic
        for key, info in self.relations.items():
            max_score = 0.0
            
            # Check all aliases
            for alias in info['aliases']:
                alias_lower = alias.lower()
                
                # Exact substring match
                if alias_lower in query_lower:
                    score = 1.0
                    max_score = max(max_score, score)
                    continue
                
                # ✅ ENHANCED: Word-level matching with position weighting
                alias_words = alias_lower.split()
                query_words = query_lower.split()
                
                if len(alias_words) > 1:
                    words_found = sum(1 for word in alias_words if word in query_words)
                    word_ratio = words_found / len(alias_words)
                    
                    # Boost score if words appear in order
                    if words_found == len(alias_words):
                        # Check if words appear in same order
                        positions = []
                        for word in alias_words:
                            try:
                                positions.append(query_words.index(word))
                            except ValueError:
                                break
                        
                        if len(positions) == len(alias_words) and positions == sorted(positions):
                            word_ratio = 0.95  # Almost exact match
                    
                    max_score = max(max_score, word_ratio)
                
                # Fuzzy similarity
                from difflib import SequenceMatcher
                ratio = SequenceMatcher(None, alias_lower, query_lower).ratio()
                max_score = max(max_score, ratio)
            
            # Boost for property code/label
            if info['property_code'] and info['property_code'].lower() in query_lower:
                max_score += 0.2
            
            if info['label'] and info['label'].lower() in query_lower:
                max_score += 0.1
            
            max_score = min(max_score, 1.0)
            
            if max_score > 0.3:
                matches.append((key, info['uri'], max_score))
        
        # Sort and deduplicate
        matches.sort(key=lambda x: x[2], reverse=True)
        
        seen_uris = {}
        for key, uri, score in matches:
            if uri not in seen_uris or score > seen_uris[uri][1]:
                seen_uris[uri] = (key, score)
        
        unique_matches = [(key, uri, score) for uri, (key, score) in seen_uris.items()]
        unique_matches.sort(key=lambda x: x[2], reverse=True)
        
        return unique_matches[:top_k]
    
    def get_relation_uri(self, relation_key: str) -> Optional[str]:
        """
        Get URI for a relation key.
        
        ✅ ENHANCED: Multiple fallback strategies for robust lookup.
        """
        if not relation_key:
            return None
        
        relation_key_lower = relation_key.lower().strip()
        
        # Strategy 1: Direct key lookup
        if relation_key in self.relations:
            return self.relations[relation_key]['uri']
        
        # Strategy 2: Lowercase key lookup
        for key, info in self.relations.items():
            if key.lower() == relation_key_lower:
                print(f"[RelationManager] Mapped '{relation_key}' → '{key}' via lowercase match")
                return info['uri']
        
        # Strategy 3: Check all relations' aliases for a match (case-insensitive)
        for key, info in self.relations.items():
            if relation_key_lower in [alias.lower() for alias in info['aliases']]:
                print(f"[RelationManager] Mapped '{relation_key}' → '{key}' via alias")
                return info['uri']
        
        # Strategy 4: Fuzzy match with higher threshold
        matches = self.find_relation(relation_key, top_k=1)
        if matches and matches[0][2] > 0.7:  # Confidence > 70%
            best_key, best_uri, best_conf = matches[0]
            print(f"[RelationManager] Mapped '{relation_key}' → '{best_key}' via fuzzy match ({best_conf:.2%})")
            return best_uri
        
        # Strategy 5: Try removing underscores and retrying
        if '_' in relation_key:
            relation_key_no_underscore = relation_key.replace('_', ' ')
            matches = self.find_relation(relation_key_no_underscore, top_k=1)
            if matches and matches[0][2] > 0.6:
                best_key, best_uri, best_conf = matches[0]
                print(f"[RelationManager] Mapped '{relation_key}' → '{best_key}' via space-normalized match ({best_conf:.2%})")
                return best_uri
        
        print(f"[RelationManager] ❌ No mapping found for '{relation_key}'")
        return None
    
    def get_relation_info(self, relation_key: str) -> Optional[Dict]:
        """
        Get full information about a relation.
        
        Args:
            relation_key: Relation key (e.g., 'director')
            
        Returns:
            Dictionary with uri, label, aliases, triple_count, expected_type (if available)
        """
        return self.relations.get(relation_key)
    
    def get_all_relations(self) -> List[str]:
        """Get list of all relation keys."""
        return list(self.relations.keys())
    
    def match_relation_from_keywords(self, keywords: List[str]) -> Optional[Tuple[str, str, float]]:
        """
        Match relation from a list of keywords.
        
        Args:
            keywords: List of potential relation keywords
            
        Returns:
            Best match as (relation_key, uri, confidence) or None
        """
        best_match = None
        best_score = 0.0
        
        for keyword in keywords:
            matches = self.find_relation(keyword, top_k=1)
            if matches and matches[0][2] > best_score:
                best_score = matches[0][2]
                best_match = matches[0]
        
        return best_match if best_score > 0.5 else None
