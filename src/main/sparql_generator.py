"""
Dynamic SPARQL Generator - Creates SPARQL queries based on query patterns.
Handles forward, reverse, and verification queries robustly.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from typing import Optional


class SPARQLGenerator:
    """Generates SPARQL queries dynamically based on query patterns."""
    
    def __init__(self, sparql_handler, relation_manager=None):
        """
        Initialize with SPARQLHandler for label normalization.
        
        Args:
            sparql_handler: SPARQLHandler instance for validation and label operations
            relation_manager: Optional RelationManager for dynamic relation resolution
        """
        self.sparql_handler = sparql_handler
        self.relation_manager = relation_manager
        
        # Legacy mappings (kept for backward compatibility)
        self.RELATION_URIS = {
            'director': 'http://www.wikidata.org/prop/direct/P57',
            'cast_member': 'http://www.wikidata.org/prop/direct/P161',
            'screenwriter': 'http://www.wikidata.org/prop/direct/P58',
            'producer': 'http://www.wikidata.org/prop/direct/P162',
            'genre': 'http://www.wikidata.org/prop/direct/P136',
            'publication_date': 'http://www.wikidata.org/prop/direct/P577',
            # ✅ RESTORED: Generic 'rating' uses custom ddis:rating property
            'rating': 'http://ddis.ch/atai/rating',
            'country_of_origin': 'http://www.wikidata.org/prop/direct/P495',
            'award_received': 'http://www.wikidata.org/prop/direct/P166',
            # ✅ Language properties
            'original_language_of_film_or_tv_show': 'http://www.wikidata.org/prop/direct/P364',
            'original_language': 'http://www.wikidata.org/prop/direct/P364',
            'language': 'http://www.wikidata.org/prop/direct/P364',
            # ✅ Specific rating properties (examples - there are many more)
            'imda_rating': 'http://www.wikidata.org/prop/direct/P5201',
            'mpa_film_rating': 'http://www.wikidata.org/prop/direct/P1657',
        }
    
    def generate(
        self,
        pattern,  # QueryPattern from query_analyzer
        subject_label: Optional[str] = None,
        object_label: Optional[str] = None
    ) -> str:
        """
        Generate SPARQL query based on pattern and entities.
        
        NOW HANDLES: Superlative forward queries (highest/lowest)
        """
        # ✅ Check if forward query has superlative modifier
        if (pattern.pattern_type == 'forward' and 
            pattern.extracted_entities and 
            'superlative' in pattern.extracted_entities):
            return self._generate_superlative_forward(
                pattern, 
                pattern.extracted_entities['superlative']
            )
        
        if pattern.pattern_type == 'forward':
            return self._generate_forward(pattern, subject_label)
        elif pattern.pattern_type == 'reverse':
            return self._generate_reverse(pattern, subject_label)
        elif pattern.pattern_type == 'verification':
            return self._generate_verification(pattern, subject_label, object_label)
        else:
            raise ValueError(f"Unknown pattern type: {pattern.pattern_type}")
    
    def _generate_superlative_forward(self, pattern, superlative: str) -> str:
        """
        Generate SPARQL for superlative forward queries (highest/lowest).
        This is a forward query WITHOUT entity extraction, using ORDER BY + LIMIT.
        
        Example: "Which movie has the highest rating?" 
                 → forward_rating + ORDER BY DESC(?rating) LIMIT 1
        """
        relation_uri = self.RELATION_URIS.get(pattern.relation)
        if not relation_uri:
            raise ValueError(f"Unknown relation: {pattern.relation}")
        
        # Determine ORDER direction
        order = "DESC" if superlative == "MAX" else "ASC"
        
        # Generate based on relation type
        if pattern.relation == 'rating':
            # Special case: ddis:rating is a literal value
            sparql = f"""PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX ddis: <http://ddis.ch/atai/>

SELECT ?movieLabel ?rating WHERE {{
    ?movieUri wdt:P31 wd:Q11424 .
    ?movieUri rdfs:label ?movieLabel .
    ?movieUri ddis:rating ?rating .
    FILTER(LANG(?movieLabel) = "en" || LANG(?movieLabel) = "")
}}
ORDER BY {order}(?rating)
LIMIT 1"""
        
        else:
            # Generic superlative for other properties
            sparql = f"""PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?movieLabel ?value WHERE {{
    ?movieUri wdt:P31 wd:Q11424 .
    ?movieUri rdfs:label ?movieLabel .
    ?movieUri <{relation_uri}> ?value .
    FILTER(LANG(?movieLabel) = "en" || LANG(?movieLabel) = "")
}}
ORDER BY {order}(?value)
LIMIT 1"""
        
        return sparql.strip()
    
    def _generate_forward(self, pattern, movie_label: str) -> str:
        """Generate forward query: Movie → Property"""
        
        print(f"[SPARQLGenerator] Generating FORWARD query")
        print(f"   Pattern relation: {pattern.relation}")
        print(f"   Subject (movie): {movie_label}")
        print(f"   Object type: {pattern.object_type}")
        
        # ✅ Use dynamic relation resolution
        relation_uri = self._get_relation_uri(pattern.relation)
        if not relation_uri:
            raise ValueError(f"Unknown relation: {pattern.relation}")
        
        print(f"[SPARQLGenerator] Using relation URI: {relation_uri}")
        
        # Normalize label using SPARQLHandler's snap_label
        normalized_label = self._escape_label(movie_label)
        
        # Build SPARQL based on object type
        if pattern.object_type == 'person':
            # Return person entities
            sparql = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT DISTINCT ?objectLabel ?objectUri WHERE {{
    ?movieUri wdt:P31 wd:Q11424 .
    ?movieUri rdfs:label ?movieLabel .
    FILTER(LCASE(STR(?movieLabel)) = LCASE("{normalized_label}"))
    
    ?movieUri <{relation_uri}> ?objectUri .
    ?objectUri rdfs:label ?objectLabel .
    FILTER(LANG(?objectLabel) = "en" || LANG(?objectLabel) = "")
}}
ORDER BY ?objectLabel
"""
        
        elif pattern.object_type == 'date':
            # Return date
            sparql = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT DISTINCT ?date WHERE {{
    ?movieUri wdt:P31 wd:Q11424 .
    ?movieUri rdfs:label ?movieLabel .
    FILTER(LCASE(STR(?movieLabel)) = LCASE("{normalized_label}"))
    
    ?movieUri <{relation_uri}> ?date .
}}
"""
        
        else:  # string (genre, rating, etc.)
            # ✅ Special handling for rating (ddis:rating is a literal, not an entity)
            if pattern.relation == 'rating':
                sparql = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX ddis: <http://ddis.ch/atai/>

SELECT DISTINCT ?value WHERE {{
    ?movieUri wdt:P31 wd:Q11424 .
    ?movieUri rdfs:label ?movieLabel .
    FILTER(LCASE(STR(?movieLabel)) = LCASE("{normalized_label}"))
    
    ?movieUri ddis:rating ?value .
}}
"""
            else:
                # For other string types (genre, country, etc.)
                sparql = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT DISTINCT ?value ?valueUri WHERE {{
    ?movieUri wdt:P31 wd:Q11424 .
    ?movieUri rdfs:label ?movieLabel .
    FILTER(LCASE(STR(?movieLabel)) = LCASE("{normalized_label}"))
    
    ?movieUri <{relation_uri}> ?valueUri .
    
    OPTIONAL {{ 
        ?valueUri rdfs:label ?value .
        FILTER(LANG(?value) = "en" || LANG(?value) = "")
    }}
}}
"""
        
        return sparql.strip()
    
    def _generate_reverse(self, pattern, person_label: str) -> str:
        """Generate reverse query: Person → Movies"""
        
        relation_uri = self.RELATION_URIS.get(pattern.relation)
        if not relation_uri:
            raise ValueError(f"Unknown relation: {pattern.relation}")
        
        normalized_label = self._escape_label(person_label)
        
        sparql = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT DISTINCT ?movieLabel ?movieUri WHERE {{
    ?personUri rdfs:label ?personLabel .
    FILTER(LCASE(STR(?personLabel)) = LCASE("{normalized_label}"))
    
    ?movieUri wdt:P31 wd:Q11424 .
    ?movieUri <{relation_uri}> ?personUri .
    ?movieUri rdfs:label ?movieLabel .
    FILTER(LANG(?movieLabel) = "en" || LANG(?movieLabel) = "")
}}
ORDER BY ?movieLabel
"""
        
        return sparql.strip()
    
    def _generate_verification(
        self,
        pattern,
        person_label: str,
        movie_label: str
    ) -> str:
        """Generate verification query: ASK if relationship exists"""
        
        relation_uri = self.RELATION_URIS.get(pattern.relation)
        if not relation_uri:
            raise ValueError(f"Unknown relation: {pattern.relation}")
        
        normalized_person = self._escape_label(person_label)
        normalized_movie = self._escape_label(movie_label)
        
        sparql = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

ASK {{
    ?movieUri wdt:P31 wd:Q11424 .
    ?movieUri rdfs:label ?movieLabel .
    FILTER(LCASE(STR(?movieLabel)) = LCASE("{normalized_movie}"))
    
    ?personUri rdfs:label ?personLabel .
    FILTER(LCASE(STR(?personLabel)) = LCASE("{normalized_person}"))
    
    ?movieUri <{relation_uri}> ?personUri .
}}
"""
        
        return sparql.strip()
    
    def _escape_label(self, label: str) -> str:
        """
        Escape label for SPARQL string literal.
        Uses SPARQLHandler's snap_label for case normalization.
        
        Args:
            label: Raw entity label
            
        Returns:
            Escaped and normalized label
        """
        # Use SPARQLHandler's snap_label for case normalization
        normalized = self.sparql_handler.snap_label(label)
        
        # Escape quotes and backslashes for SPARQL
        escaped = normalized.replace('\\', '\\\\').replace('"', '\\"')
        
        return escaped
    
    def _get_relation_uri(self, relation: str) -> str:
        """
        Get Wikidata property URI for a relation.
        
        Args:
            relation: Relation key (e.g., 'director', 'cast_member')
            
        Returns:
            Property URI
        """
        # ✅ CRITICAL FIX: Map 'country' to 'country_of_origin'
        if relation == 'country':
            relation = 'country_of_origin'
        
        # Try RelationManager first
        if self.relation_manager:
            uri = self.relation_manager.get_relation_uri(relation)
            if uri:
                return uri
        
        # Fallback to hardcoded mappings
        relation_to_property = {
            'director': 'http://www.wikidata.org/prop/direct/P57',
            'cast_member': 'http://www.wikidata.org/prop/direct/P161',
            'screenwriter': 'http://www.wikidata.org/prop/direct/P58',
            'producer': 'http://www.wikidata.org/prop/direct/P162',
            'genre': 'http://www.wikidata.org/prop/direct/P136',
            'publication_date': 'http://www.wikidata.org/prop/direct/P577',
            # ✅ RESTORED: Generic 'rating' uses custom ddis:rating property
            'rating': 'http://ddis.ch/atai/rating',
            'country_of_origin': 'http://www.wikidata.org/prop/direct/P495',
            'country': 'http://www.wikidata.org/prop/direct/P495',  # ✅ Alias
            'award_received': 'http://www.wikidata.org/prop/direct/P166',
            # ✅ Language properties
            'original_language_of_film_or_tv_show': 'http://www.wikidata.org/prop/direct/P364',
            'original_language': 'http://www.wikidata.org/prop/direct/P364',
            'language': 'http://www.wikidata.org/prop/direct/P364',
            # ✅ Specific rating properties (examples - there are many more)
            'imda_rating': 'http://www.wikidata.org/prop/direct/P5201',
            'mpa_film_rating': 'http://www.wikidata.org/prop/direct/P1657',
        }
        
        if relation in relation_to_property:
            return relation_to_property[relation]
        
        raise ValueError(f"Unknown relation: {relation}")