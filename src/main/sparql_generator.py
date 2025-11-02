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
            # ✅ FIXED: Use imda_rating instead of generic rating
            'rating': 'http://www.wikidata.org/prop/direct/P5201',
            'country_of_origin': 'http://www.wikidata.org/prop/direct/P495',
            'award_received': 'http://www.wikidata.org/prop/direct/P166',
            # ✅ Language properties
            'original_language_of_film_or_tv_show': 'http://www.wikidata.org/prop/direct/P364',
            'original_language': 'http://www.wikidata.org/prop/direct/P364',
            'language': 'http://www.wikidata.org/prop/direct/P364',
            # ✅ Specific rating properties
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
        
        NOW HANDLES: Superlative forward queries (highest/lowest) AND complex multi-constraint queries
        """
        # ✅ NEW: Handle complex multi-constraint queries FIRST
        if pattern.pattern_type == 'complex':
            return self._generate_complex(pattern)
        
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
        relation_uri = self._get_relation_uri(pattern.relation)
        if not relation_uri:
            raise ValueError(f"Unknown relation: {pattern.relation}")
        
        # ✅ FIXED: Correct ORDER direction mapping
        # Superlatives that indicate we want MAXIMUM value → DESC (descending)
        max_superlatives = {'highest', 'best', 'most', 'top', 'latest', 'longest', 'biggest', 'largest', 'maximum', 'max'}
        # Superlatives that indicate we want MINIMUM value → ASC (ascending)
        min_superlatives = {'lowest', 'worst', 'least', 'bottom', 'earliest', 'shortest', 'smallest', 'minimum', 'min', 'first'}
        
        if superlative.lower() in max_superlatives:
            order = "DESC"
        elif superlative.lower() in min_superlatives:
            order = "ASC"
        else:
            # Default: assume they want the highest/maximum
            order = "DESC"
        
        print(f"[SPARQLGenerator] Superlative: '{superlative}' → ORDER BY {order}")
        
        # ✅ FIXED: Use imda_rating property directly (not ddis:rating)
        if pattern.relation in ['rating', 'imda_rating']:
            sparql = f"""PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?movieLabel ?rating WHERE {{
    ?movieUri wdt:P31 wd:Q11424 .
    ?movieUri rdfs:label ?movieLabel .
    ?movieUri <{relation_uri}> ?rating .
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
        
        # ✅ CRITICAL: Check for language relation and set correct object type
        is_language_query = 'language' in pattern.relation.lower()
        if is_language_query:
            print(f"[SPARQLGenerator] 🌐 Language query detected - ensuring correct property and type")
        
        # ✅ Use dynamic relation resolution
        relation_uri = self._get_relation_uri(pattern.relation)
        if not relation_uri:
            raise ValueError(f"Unknown relation: {pattern.relation}")
        
        print(f"[SPARQLGenerator] Using relation URI: {relation_uri}")
        
        # ✅ CRITICAL: For language queries, always use P364 (original language)
        if is_language_query:
            relation_uri = 'http://www.wikidata.org/prop/direct/P364'
            print(f"[SPARQLGenerator] Enforced language property: {relation_uri}")
        
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
        
        elif pattern.object_type == 'language' or is_language_query:
            # ✅ NEW: Special handling for language queries
            sparql = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT DISTINCT ?languageName ?languageUri WHERE {{
    ?movieUri wdt:P31 wd:Q11424 .
    ?movieUri rdfs:label ?movieLabel .
    FILTER(LCASE(STR(?movieLabel)) = LCASE("{normalized_label}"))
    
    ?movieUri wdt:P364 ?languageUri .
    ?languageUri rdfs:label ?languageName .
    FILTER(LANG(?languageName) = "en" || LANG(?languageName) = "")
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
    
    def _generate_complex(self, pattern) -> str:
        """
        Generate SPARQL for complex multi-constraint queries.
        
        Example: "Which movie from South Korea received Academy Award for Best Picture?"
                 → Filter by P495 (country) AND P166 (award)
        
        Args:
            pattern: QueryPattern with extracted_entities containing 'constraints' dict
            
        Returns:
            SPARQL query string
        """
        constraints = pattern.extracted_entities.get('constraints', {})
        
        print(f"[SPARQLGenerator] Generating COMPLEX query")
        print(f"   Constraints: {constraints}")
        
        # Build WHERE clause with all constraints
        where_clauses = []
        select_vars = ['?movieLabel', '?movieUri']
        
        # Base: Movie type
        where_clauses.append('?movieUri wdt:P31 wd:Q11424 .')
        where_clauses.append('?movieUri rdfs:label ?movieLabel .')
        where_clauses.append('FILTER(LANG(?movieLabel) = "en" || LANG(?movieLabel) = "")')
        
        # Constraint 1: Country
        if 'country' in constraints:
            country_name = self._escape_label(constraints['country'])
            where_clauses.append('?movieUri wdt:P495 ?countryUri .')
            where_clauses.append('?countryUri rdfs:label ?countryLabel .')
            where_clauses.append(f'FILTER(LCASE(STR(?countryLabel)) = LCASE("{country_name}"))')
            select_vars.append('?countryLabel')
        
        # Constraint 2: Award
        if 'award' in constraints:
            award_name = self._escape_label(constraints['award'])
            where_clauses.append('?movieUri wdt:P166 ?awardUri .')
            where_clauses.append('?awardUri rdfs:label ?awardLabel .')
            where_clauses.append(f'FILTER(LCASE(STR(?awardLabel)) = LCASE("{award_name}"))')
            select_vars.append('?awardLabel')
        
        # Constraint 3: Year
        if 'year' in constraints:
            year = constraints['year']
            where_clauses.append('?movieUri wdt:P577 ?date .')
            where_clauses.append(f'FILTER(YEAR(?date) = {year})')
            select_vars.append('?date')
        
        # Constraint 4: Genre
        if 'genre' in constraints:
            genre_name = self._escape_label(constraints['genre'])
            where_clauses.append('?movieUri wdt:P136 ?genreUri .')
            where_clauses.append('?genreUri rdfs:label ?genreLabel .')
            where_clauses.append(f'FILTER(LCASE(STR(?genreLabel)) = LCASE("{genre_name}"))')
            select_vars.append('?genreLabel')
        
        # Build complete query
        sparql = f"""PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT DISTINCT {' '.join(select_vars)} WHERE {{
    {chr(10).join('    ' + clause for clause in where_clauses)}
}}
ORDER BY ?movieLabel
"""
        
        return sparql.strip()
    
    def _escape_label(self, label: str) -> str:
        """
        Escape label for SPARQL string literal.
        
        Args:
            label: Raw entity label
            
        Returns:
            Escaped and normalized label
        """
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
        # Fallback to hardcoded mappings
        relation_to_property = {
            'director': 'http://www.wikidata.org/prop/direct/P57',
            'cast_member': 'http://www.wikidata.org/prop/direct/P161',
            'screenwriter': 'http://www.wikidata.org/prop/direct/P58',
            'producer': 'http://www.wikidata.org/prop/direct/P162',
            'genre': 'http://www.wikidata.org/prop/direct/P136',
            'publication_date': 'http://www.wikidata.org/prop/direct/P577',
            'rating': 'http://www.wikidata.org/prop/direct/P5201',
            'country': 'http://www.wikidata.org/prop/direct/P495',
            'award': 'http://www.wikidata.org/prop/direct/P166',
            # ✅ Language properties
            'original_language_of_film_or_tv_show': 'http://www.wikidata.org/prop/direct/P364',
            'original_language': 'http://www.wikidata.org/prop/direct/P364',
            'language': 'http://www.wikidata.org/prop/direct/P364',
            # ✅ Specific rating properties
            'imda_rating': 'http://www.wikidata.org/prop/direct/P5201',
            'mpa_film_rating': 'http://www.wikidata.org/prop/direct/P1657',
        }
        
        if relation in relation_to_property:
            return relation_to_property[relation]
        
        # Try RelationManager first
        if self.relation_manager:
            uri = self.relation_manager.get_relation_uri(relation)
            if uri:
                return uri
        
        raise ValueError(f"Unknown relation: {relation}")
    
    def _generate_complex(self, pattern) -> str:
        """
        Generate SPARQL for complex queries with multiple constraints.
        Example: "Find movies with a specific actor AND a specific genre"
        
        Complex queries are handled by creating OPTIONAL patterns for each
        extracted entity, allowing flexible matching.
        """
        # Base query: Find movies that match ALL extracted entities
        # Uses OPTIONAL {} to allow missing values for some entities
        sparql = f"""PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT DISTINCT ?movieLabel ?movieUri WHERE {{
    ?movieUri wdt:P31 wd:Q11424 .
    ?movieUri rdfs:label ?movieLabel .
    FILTER(LANG(?movieLabel) = "en" || LANG(?movieLabel) = "")
    
    """
        
        # Add OPTIONAL patterns for each extracted entity
        optional_patterns = []
        for entity, value in pattern.extracted_entities.items():
            if entity == 'superlative':
                continue  # Skip superlative, already handled
            
            # Determine relation URI
            relation_uri = self._get_relation_uri(entity)
            if not relation_uri:
                raise ValueError(f"Unknown relation: {entity}")
            
            # Handle specific cases (e.g., rating)
            if entity in ['rating', 'imda_rating']:
                optional_patterns.append(f"""
    OPTIONAL {{
        ?movieUri <{relation_uri}> ?rating .
        FILTER(?rating {">=" if value == "MAX" else "<="} 0)  # Ensure valid rating
    }}""")
            
            elif entity == 'publication_date':
                optional_patterns.append(f"""
    OPTIONAL {{
        ?movieUri <{relation_uri}> ?date .
        FILTER(?date >= "1900-01-01"^^xsd:date)  # Reasonable date range
    }}""")
            
            else:
                # Generic OPTIONAL pattern
                optional_patterns.append(f"""
    OPTIONAL {{
        ?movieUri <{relation_uri}> ?value .
    }}""")
        
        # Combine base query with OPTIONAL patterns
        sparql += " .\n".join(optional_patterns)
        
        # Close query
        sparql += """
}
ORDER BY ?movieLabel
LIMIT 100
"""
        
        return sparql.strip()