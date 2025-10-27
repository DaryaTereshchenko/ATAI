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
    
    # Relation URI mappings
    RELATION_URIS = {
        'director': 'http://www.wikidata.org/prop/direct/P57',
        'cast_member': 'http://www.wikidata.org/prop/direct/P161',
        'screenwriter': 'http://www.wikidata.org/prop/direct/P58',
        'producer': 'http://www.wikidata.org/prop/direct/P162',
        'genre': 'http://www.wikidata.org/prop/direct/P136',
        'publication_date': 'http://www.wikidata.org/prop/direct/P577',
        'rating': 'http://ddis.ch/atai/rating',
        'country': 'http://www.wikidata.org/prop/direct/P495',  # ✅ NEW: country of origin
    }
    
    # Type URIs
    TYPE_URIS = {
        'movie': 'http://www.wikidata.org/entity/Q11424',
        'person': 'http://www.wikidata.org/entity/Q5'
    }
    
    def __init__(self, sparql_handler):
        """
        Initialize with SPARQLHandler for label normalization.
        
        Args:
            sparql_handler: SPARQLHandler instance for validation and label operations
        """
        self.sparql_handler = sparql_handler
    
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
        
        relation_uri = self.RELATION_URIS.get(pattern.relation)
        if not relation_uri:
            raise ValueError(f"Unknown relation: {pattern.relation}")
        
        # ✅ VALIDATION: Ensure we're using the correct property
        print(f"[SPARQLGenerator] Forward query: {pattern.relation} → {relation_uri}")
        
        # Normalize label using SPARQLHandler's snap_label
        normalized_label = self._escape_label(movie_label)
        
        # ✅ OPTIMIZED: Special handling for country queries to avoid timeout
        if pattern.relation == 'country':
            sparql = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT DISTINCT ?countryLabel WHERE {{
    ?movieUri wdt:P31 wd:Q11424 .
    ?movieUri rdfs:label ?movieLabel .
    FILTER(LCASE(STR(?movieLabel)) = LCASE("{normalized_label}"))
    
    ?movieUri <{relation_uri}> ?countryUri .
    ?countryUri rdfs:label ?countryLabel .
    FILTER(LANG(?countryLabel) = "en" || LANG(?countryLabel) = "")
}}
LIMIT 10
"""
            return sparql.strip()
        
        # Build SPARQL based on object type
        if pattern.object_type == 'person':
            # ✅ VALIDATION: Log which property we're using
            print(f"[SPARQLGenerator]    Person query using: {relation_uri}")
            
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