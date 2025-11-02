"""
Embedding-based Query Processor.
Orchestrates the complete embedding-based query answering pipeline:
1. Analyze query pattern (forward/reverse/verification)
2. Extract entities from query (case-insensitive)
3. Generate SPARQL dynamically based on pattern
4. Execute using cached graph
5. Format natural language response

✅ ENHANCED: Uses QueryAnalyzer + SPARQLGenerator for robust pattern handling
"""

import sys
import os
import traceback
from typing import List, Tuple, Optional, Dict
from rdflib import Graph, URIRef, RDFS
import numpy as np
import re

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.main.embedding_handler import EmbeddingHandler
from src.main.entity_extractor import EntityExtractor
from src.main.query_embedder import QueryEmbedder
from src.main.embedding_aligner import SimpleAligner
from src.main.sparql_handler import SPARQLHandler
from src.main.query_analyzer import QueryAnalyzer, QueryPattern
from src.main.sparql_generator import SPARQLGenerator


class EmbeddingQueryProcessor:
    """
    Main processor for embedding-based query answering.
    Uses hybrid approach: pattern analysis + entity extraction + dynamic SPARQL generation.
    """
    
    def __init__(
        self,
        embeddings_dir: str,
        graph_path: str,
        query_model: str = "all-MiniLM-L6-v2",
        alignment_matrix_path: Optional[str] = None,
        use_simple_aligner: bool = False,
        sparql_handler: Optional[SPARQLHandler] = None,
        relation_manager = None  # ✅ NEW
    ):
        """
        Initialize the embedding query processor.
        
        Args:
            embeddings_dir: Directory containing TransE embeddings
            graph_path: Path to knowledge graph file
            query_model: Sentence transformer model for query embedding
            alignment_matrix_path: Path to alignment matrix (optional)
            use_simple_aligner: Use simple normalization-based alignment
            sparql_handler: Optional SPARQLHandler instance (shared across components)
        """
        print("🔧 Initializing Embedding Query Processor (Hybrid Mode)...")
        
        # Initialize embedding handler for TransE embeddings
        print("   📊 Loading TransE embeddings...")
        self.embedding_handler = EmbeddingHandler(embeddings_dir)
        
        # Initialize query embedder for natural language
        print("   🔤 Loading query embedding model...")
        self.query_embedder = QueryEmbedder(model_name=query_model)
        
        # Initialize alignment between query and TransE spaces
        print("   🔗 Setting up embedding alignment...")
        query_dim = self.query_embedder.get_embedding_dimension()
        transe_dim = self.embedding_handler.get_embedding_dimension()
        
        if use_simple_aligner:
            self.aligner = SimpleAligner(query_dim=query_dim, transe_dim=transe_dim)
        else:
            from src.main.embedding_aligner import EmbeddingAligner
            self.aligner = EmbeddingAligner(
                query_dim=query_dim,
                transe_dim=transe_dim,
                projection_matrix_path=alignment_matrix_path
            )
        
        # Initialize SPARQL handler (shared or new)
        if sparql_handler is None:
            print("   🔍 Loading knowledge graph...")
            self.sparql_handler = SPARQLHandler(graph_file_path=graph_path)
        else:
            print("   🔍 Using shared SPARQL handler...")
            self.sparql_handler = sparql_handler
        
        # Initialize entity extractor
        print("   🏷️  Initializing entity extractor...")
        self.entity_extractor = EntityExtractor(self.sparql_handler.graph)
        
        # ✅ NEW: Store relation manager
        self.relation_manager = relation_manager
        
        # ✅ NEW: Initialize query analyzer and SPARQL generator with relation manager
        print("   🧠 Initializing query analysis components...")
        from src.config import SPARQL_CLASSIFIER_MODEL_PATH
        self.query_analyzer = QueryAnalyzer(
            use_transformer=True,
            transformer_model_path=SPARQL_CLASSIFIER_MODEL_PATH,
            relation_manager=relation_manager  # ✅ Pass relation manager
        )
        self.sparql_generator = SPARQLGenerator(
            self.sparql_handler,
            relation_manager=relation_manager  # ✅ Pass relation manager
        )
        
        # ✅ NEW: Initialize NL2SPARQL for LLM fallback
        print("   🤖 Initializing LLM fallback for SPARQL generation...")
        from src.main.nl_to_sparql import NLToSPARQL
        self.nl2sparql = NLToSPARQL(method="direct-llm", sparql_handler=self.sparql_handler)
        
        print("✅ Embedding Query Processor ready (hybrid mode)\n")
    
    def process_hybrid_factual_query(self, query: str) -> str:
        """
        Process factual query using ROBUST pattern analysis + dynamic SPARQL.
        
        Pipeline:
        1. Analyze query pattern (forward/reverse/verification/complex)
        2. Extract required entities based on pattern (or skip for superlative)
        3. Generate SPARQL query dynamically
        4. Execute and format response
        
        Args:
            query: Natural language query
            
        Returns:
            Natural language response
        """
        print(f"\n{'='*80}")
        print(f"🔍 PROCESSING HYBRID FACTUAL QUERY")
        print(f"{'='*80}\n")
        print(f"Query: {query}\n")
        
        try:
            # ==================== STEP 1: ANALYZE QUERY PATTERN ====================
            print("📝 Step 1: Analyzing query pattern...")
            pattern = self.query_analyzer.analyze(query)
            
            if not pattern:
                print("❌ No pattern detected - query structure not recognized\n")
                return self._handle_unrecognized_query(query)
            
            print(f"✅ Pattern detected:")
            print(f"   Type: {pattern.pattern_type}")
            print(f"   Relation: {pattern.relation}")
            print(f"   Subject: {pattern.subject_type} → Object: {pattern.object_type}")
            print(f"   Confidence: {pattern.confidence:.2%}\n")
            
            # ==================== STEP 2: EXTRACT ENTITIES ====================
            print("📝 Step 2: Extracting entities based on pattern...")
            
            if pattern.pattern_type == 'forward':
                # ✅ Check if this is a superlative variant
                if pattern.extracted_entities and 'superlative' in pattern.extracted_entities:
                    return self._process_superlative_forward_query(query, pattern)
                else:
                    return self._process_forward_query(query, pattern)
            
            elif pattern.pattern_type == 'reverse':
                return self._process_reverse_query(query, pattern)
            
            elif pattern.pattern_type == 'verification':
                return self._process_verification_query(query, pattern)
            
            elif pattern.pattern_type == 'complex':
                return self._process_complex_query(query, pattern)
            
            else:
                print(f"❌ Unknown pattern type: {pattern.pattern_type}\n")
                return "❌ I encountered an internal error processing your query pattern."
        
        except Exception as e:
            print(f"❌ Error in hybrid processing: {e}")
            traceback.print_exc()
            return f"❌ An error occurred while processing your query: {str(e)}"
    
    def _generate_sparql_with_fallback(
        self,
        pattern: QueryPattern,
        subject_label: str,
        object_label: Optional[str] = None
    ) -> Dict:
        """
        Generate SPARQL with LLM-first, template-fallback strategy.
        NOW ENHANCED: Passes pattern to LLM for better few-shot example selection.
        """
        print("📝 Generating SPARQL query...")
        
        # ✅ NEW: Log the relation being used
        print(f"   📌 Using relation: {pattern.relation}")
        
        # Try to get the URI for this relation
        relation_uri = None
        if self.relation_manager:
            relation_uri = self.relation_manager.get_relation_uri(pattern.relation)
            if relation_uri:
                print(f"   🔗 Relation URI: {relation_uri}")
            else:
                print(f"   ⚠️  No URI found for relation '{pattern.relation}' in RelationManager")
        
        # PRIMARY: Try LLM-based generation FIRST with pattern-specific examples
        try:
            print("   Attempting LLM-based generation...")
            
            # ✅ FIX: Map relation names correctly for LLM query construction
            relation_to_phrase = {
                'director': 'director',
                'cast_member': 'cast member',
                'screenwriter': 'screenwriter',
                'producer': 'producer',
                'genre': 'genre',
                'publication_date': 'release date',
                'country_of_origin': 'country of origin',  # ✅ FIXED
                'rating': 'rating'
            }
            
            # Construct a descriptive query based on pattern
            if pattern.pattern_type == 'forward':
                relation_phrase = relation_to_phrase.get(pattern.relation, pattern.relation.replace('_', ' '))
                llm_query = f"What is the {relation_phrase} of \"{subject_label}\"?"
            elif pattern.pattern_type == 'reverse':
                relation_verb = {
                    'director': 'direct',
                    'cast_member': 'act in',
                    'screenwriter': 'write',
                    'producer': 'produce'
                }.get(pattern.relation, pattern.relation)
                llm_query = f"What movies did \"{subject_label}\" {relation_verb}?"
            else:  # verification
                relation_verb = {
                    'director': 'direct',
                    'cast_member': 'act in',
                    'screenwriter': 'write',
                    'producer': 'produce'
                }.get(pattern.relation, pattern.relation)
                llm_query = f"Did \"{subject_label}\" {relation_verb} \"{object_label}\"?"
            
            print(f"   LLM query: {llm_query}")
            print(f"   Pattern: {pattern.pattern_type}_{pattern.relation}")
            
            # Generate with pattern context for better example selection AND validation
            result = self.nl2sparql.convert(llm_query, pattern=pattern)
            
            if result.confidence > 0.0:
                print(f"   ✅ LLM generation successful (confidence: {result.confidence:.2%})")
                return {
                    'query': result.query,
                    'method': 'llm',
                    'confidence': result.confidence
                }
            else:
                print(f"   ⚠️ LLM generated invalid query (confidence: {result.confidence:.2%})")
                raise ValueError("LLM generated invalid query")
                
        except Exception as e:
            print(f"   ⚠️ LLM generation failed: {e}")
        
        # FALLBACK: Use template-based generation
        try:
            print("   Attempting template-based generation (fallback)...")
            sparql = self.sparql_generator.generate(pattern, subject_label, object_label)
            print("   ✅ Template generation successful")
            return {
                'query': sparql,
                'method': 'template',
                'confidence': 0.95
            }
        except Exception as e:
            print(f"   ❌ Template fallback failed: {e}")
            raise ValueError(f"Both LLM and template generation failed: {str(e)}")
    
    def _process_forward_query(self, query: str, pattern: QueryPattern) -> str:
        """
        Process forward query: Movie → Property
        Example: "Who directed The Matrix?"
        
        Args:
            query: Natural language query
            pattern: Detected query pattern
            
        Returns:
            Natural language response
        """
        print(f"   Direction: Forward ({pattern.subject_type} → {pattern.object_type})")
        
        # ✅ CRITICAL: Check superlative FIRST before entity extraction
        if pattern.extracted_entities and 'superlative' in pattern.extracted_entities:
            print("   ℹ️  Superlative query detected - delegating to superlative handler")
            return self._process_superlative_forward_query(query, pattern)
        
        # Regular forward query - needs entity extraction
        # Use entity hints from pattern if available
        entity_hints = pattern.extracted_entities if pattern.extracted_entities else {}
        
        # Prioritize quoted entities for movie titles
        if entity_hints.get('quoted'):
            print(f"   Using quoted entity hint: {entity_hints['quoted'][0]}")
        
        # Extract movie entity
        movie_entities = self.entity_extractor.extract_entities(
            query,
            entity_type="http://www.wikidata.org/entity/Q11424",  # Q11424 = film
            threshold=75
        )
        
        if not movie_entities:
            print("❌ No movie entity found\n")
            return (
                "❌ I couldn't identify the movie in your question.\n\n"
                "**Tips:**\n"
                "- Use quotes around the movie title: \"The Matrix\"\n"
                "- Check the spelling of the movie name\n"
                "- Try a more complete title if it's ambiguous"
            )
        
        # Get best match
        movie_uri, movie_text, score = movie_entities[0]
        movie_label = self.entity_extractor.get_entity_label(movie_uri)
        print(f"✅ Movie identified: '{movie_label}' (confidence: {score}%)\n")
        
        # Generate SPARQL query with fallback
        print("📝 Step 3: Generating SPARQL query...")
        try:
            sparql_result = self._generate_sparql_with_fallback(
                pattern=pattern,
                subject_label=movie_label
            )
            sparql = sparql_result['query']
            method = sparql_result['method']
            confidence = sparql_result['confidence']
            
            print(f"✅ SPARQL generated using {method} (confidence: {confidence:.2%}):")
            print("-" * 80)
            print(sparql)
            print("-" * 80 + "\n")
        except Exception as e:
            print(f"❌ SPARQL generation failed: {e}\n")
            return f"❌ Failed to generate query for this request: {str(e)}"
        
        # Execute query
        print("📝 Step 4: Executing query against knowledge graph...")
        result = self._execute_sparql(sparql)
        
        if not result['success']:
            print(f"❌ Query execution failed: {result.get('error', 'Unknown error')}\n")
            return f"❌ Query execution failed: {result.get('error', 'Unknown error')}"
        
        # Format response
        print("📝 Step 5: Formatting response...")
        response = self._format_forward_response(
            pattern=pattern,
            movie_label=movie_label,
            data=result['data']
        )
        
        print(f"✅ Response generated\n")
        print("="*80)
        return response
    
    def _process_reverse_query(self, query: str, pattern: QueryPattern) -> str:
        """
        Process reverse query: Person → Movies
        Example: "What films did Christopher Nolan direct?"
        
        Args:
            query: Natural language query
            pattern: Detected query pattern
            
        Returns:
            Natural language response
        """
        print(f"   Direction: Reverse ({pattern.subject_type} → {pattern.object_type})")
        
        # Extract person entity
        person_entities = self.entity_extractor.extract_entities(
            query,
            entity_type="http://www.wikidata.org/entity/Q5",  # Q5 = human
            threshold=75
        )
        
        if not person_entities:
            print("❌ No person entity found\n")
            return (
                "❌ I couldn't identify the person in your question.\n\n"
                "**Tips:**\n"
                "- Check the spelling of the person's name\n"
                "- Use the full name if possible\n"
                "- Make sure the person is in the knowledge graph"
            )
        
        # Get best match
        person_uri, person_text, score = person_entities[0]
        person_label = self.entity_extractor.get_entity_label(person_uri)
        print(f"✅ Person identified: '{person_label}' (confidence: {score}%)\n")
        
        # Generate SPARQL query with fallback
        print("📝 Step 3: Generating SPARQL query...")
        try:
            sparql_result = self._generate_sparql_with_fallback(
                pattern=pattern,
                subject_label=person_label
            )
            sparql = sparql_result['query']
            method = sparql_result['method']
            confidence = sparql_result['confidence']
            
            print(f"✅ SPARQL generated using {method} (confidence: {confidence:.2%}):")
            print("-" * 80)
            print(sparql)
            print("-" * 80 + "\n")
        except Exception as e:
            print(f"❌ SPARQL generation failed: {e}\n")
            return f"❌ Failed to generate query for this request: {str(e)}"
        
        # Execute query
        print("📝 Step 4: Executing query against knowledge graph...")
        result = self._execute_sparql(sparql)
        
        if not result['success']:
            print(f"❌ Query execution failed: {result.get('error', 'Unknown error')}\n")
            return f"❌ Query execution failed: {result.get('error', 'Unknown error')}"
        
        # Format response
        print("📝 Step 5: Formatting response...")
        response = self._format_reverse_response(
            pattern=pattern,
            person_label=person_label,
            data=result['data']
        )
        
        print(f"✅ Response generated\n")
        print("="*80)
        return response
    
    def _process_verification_query(self, query: str, pattern: QueryPattern) -> str:
        """
        Process verification query: Does relationship exist?
        Example: "Did Christopher Nolan direct Inception?"
        
        Args:
            query: Natural language query
            pattern: Detected query pattern
            
        Returns:
            Natural language response (Yes/No)
        """
        print(f"   Direction: Verification (relationship check)")
        
        # Extract both movie and person entities
        movie_entities = self.entity_extractor.extract_entities(
            query,
            entity_type="http://www.wikidata.org/entity/Q11424",
            threshold=75
        )
        person_entities = self.entity_extractor.extract_entities(
            query,
            entity_type="http://www.wikidata.org/entity/Q5",
            threshold=75
        )
        
        if not movie_entities:
            print("❌ No movie entity found\n")
            return "❌ I couldn't identify the movie in your question."
        
        if not person_entities:
            print("❌ No person entity found\n")
            return "❌ I couldn't identify the person in your question."
        
        movie_uri, movie_text, movie_score = movie_entities[0]
        person_uri, person_text, person_score = person_entities[0]
        
        movie_label = self.entity_extractor.get_entity_label(movie_uri)
        person_label = self.entity_extractor.get_entity_label(person_uri)
        
        print(f"✅ Movie: '{movie_label}' (confidence: {movie_score}%)")
        print(f"✅ Person: '{person_label}' (confidence: {person_score}%)\n")
        
        # Generate SPARQL ASK query with fallback
        print("📝 Step 3: Generating SPARQL verification query...")
        try:
            sparql_result = self._generate_sparql_with_fallback(
                pattern=pattern,
                subject_label=person_label,
                object_label=movie_label
            )
            sparql = sparql_result['query']
            method = sparql_result['method']
            confidence = sparql_result['confidence']
            
            print(f"✅ SPARQL generated using {method} (confidence: {confidence:.2%}):")
            print("-" * 80)
            print(sparql)
            print("-" * 80 + "\n")
        except Exception as e:
            print(f"❌ SPARQL generation failed: {e}\n")
            return f"❌ Failed to generate verification query: {str(e)}"
        
        # Execute query
        print("📝 Step 4: Executing verification query...")
        result = self._execute_sparql(sparql)
        
        if not result['success']:
            print(f"❌ Query execution failed: {result.get('error', 'Unknown error')}\n")
            return f"❌ Query execution failed: {result.get('error', 'Unknown error')}"
        
        # Format response
        print("📝 Step 5: Formatting verification response...")
        response = self._format_verification_response(
            pattern=pattern,
            person_label=person_label,
            movie_label=movie_label,
            data=result['data']
        )
        
        print(f"✅ Response generated\n")
        print("="*80)
        return response
    
    def _process_superlative_forward_query(self, query: str, pattern: QueryPattern) -> str:
        """
        Process superlative forward query: Find movie with highest/lowest property value.
        Example: "Which movie has the highest user rating?" → forward_rating + MAX
        
        This is a **forward query variant** with ORDER BY + LIMIT.
        No entity extraction needed - we're querying all movies.
        
        Args:
            query: Natural language query
            pattern: Detected forward pattern with superlative modifier
            
        Returns:
            Natural language response
        """
        print(f"   Direction: Forward + Superlative ({pattern.relation})")
        
        # Get superlative type (MAX or MIN)
        superlative = pattern.extracted_entities.get('superlative', 'MAX')
        print(f"   Superlative: {superlative}")
        print(f"   (No entity extraction needed - querying all movies)\n")
        
        # Generate SPARQL with ORDER BY + LIMIT
        print("📝 Step 3: Generating superlative SPARQL query...")
        try:
            sparql = self._generate_superlative_sparql(pattern, superlative)
            
            print(f"✅ SPARQL generated:")
            print("-" * 80)
            print(sparql)
            print("-" * 80 + "\n")
        except Exception as e:
            print(f"❌ SPARQL generation failed: {e}\n")
            return f"❌ Failed to generate query for this request: {str(e)}"
        
        # Execute query
        print("📝 Step 4: Executing query against knowledge graph...")
        result = self._execute_sparql(sparql)
        
        if not result['success']:
            print(f"❌ Query execution failed: {result.get('error', 'Unknown error')}\n")
            return f"❌ Query execution failed: {result.get('error', 'Unknown error')}"
        
        # Format response
        print("📝 Step 5: Formatting response...")
        response = self._format_superlative_response(
            pattern=pattern,
            superlative=superlative,
            data=result['data']
        )
        
        print(f"✅ Response generated\n")
        print("="*80)
        return response

    def _generate_superlative_sparql(self, pattern: QueryPattern, superlative: str) -> str:
        """
        Generate SPARQL for superlative forward queries (highest/lowest).
        Example: "Which movie has the highest rating?"
        """
        relation_uri = self.sparql_generator._get_relation_uri(pattern.relation)
        if not relation_uri:
            raise ValueError(f"Unknown relation: {pattern.relation}")
        
        # Determine ORDER direction
        order = "DESC" if superlative == "MAX" else "ASC"
        
        # Generate based on relation type
        if pattern.relation == 'rating' or pattern.relation.endswith('_rating'):
            # Rating queries - use ddis:rating or specific rating property
            sparql = f"""PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX ddis: <http://ddis.ch/atai/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?movieLabel ?rating WHERE {{
    ?movieUri wdt:P31 wd:Q11424 .
    ?movieUri rdfs:label ?movieLabel .
    ?movieUri ddis:rating ?ratingRaw .
    FILTER(LANG(?movieLabel) = "en" || LANG(?movieLabel) = "")
    
    # Convert rating to decimal for proper sorting
    BIND(xsd:decimal(?ratingRaw) AS ?rating)
    
    # Filter out invalid ratings (legitimate movies have ratings between 1.0 and 9.5)
    FILTER(?rating >= 1.0 && ?rating <= 9.5)
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

    def _format_superlative_response(
        self,
        pattern: QueryPattern,
        superlative: str,
        data: str
    ) -> str:
        """Format response for superlative queries."""
        
        if not data or data == "No answer found in the database.":
            return f"❌ I couldn't find any movies with {pattern.relation} information in the knowledge graph."
        
        # Parse plain text results (format: "movie, rating")
        lines = [line.strip() for line in data.strip().split('\n') if line.strip()]
        
        if not lines:
            return f"❌ No results found for the query."
        
        # Extract movie and value from first line
        parts = lines[0].split(',')
        if len(parts) >= 2:
            movie = parts[0].strip()
            value_raw = parts[1].strip()
            
            # ✅ Format rating value nicely
            try:
                value_numeric = float(value_raw)
                value = f"{value_numeric:.1f}"
            except ValueError:
                value = value_raw
            
            descriptor = "highest" if superlative == "MAX" else "lowest"
            
            # ✅ Use proper article and rating description
            return f"✅ The movie with the **{descriptor} {pattern.relation}** is **'{movie}'** with a rating of **{value}**."
        
        return f"✅ Result: {lines[0]}"

    def _execute_sparql(self, sparql: str) -> Dict:
        """
        Execute SPARQL query with validation and error handling.
        
        Args:
            sparql: SPARQL query string
            
        Returns:
            Dictionary with success status and data/error
        """
        try:
            result = self.sparql_handler.execute_query(sparql, validate=True)
            return result
        except Exception as e:
            print(f"❌ SPARQL execution error: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def _format_forward_response(
        self,
        pattern: QueryPattern,
        movie_label: str,
        data: str
    ) -> str:
        """
        Format response for forward queries.
        
        Args:
            pattern: Query pattern
            movie_label: Movie name
            data: Query results (plain text from SPARQLHandler, NOT JSON)
            
        Returns:
            Formatted natural language response
        """
        try:
            # SPARQLHandler returns plain text, not JSON
            # Each line is: "label1, uri1" or just "label1"
            if not data or data == "No answer found in the database.":
                return f"❌ I couldn't find any {pattern.relation.replace('_', ' ')} information for '{movie_label}' in the knowledge graph."
            
            # Parse plain text results
            lines = [line.strip() for line in data.strip().split('\n') if line.strip()]
            
            if not lines:
                return f"❌ I couldn't find any {pattern.relation.replace('_', ' ')} information for '{movie_label}' in the knowledge graph."
            
            # Handle different object types
            if pattern.object_type == 'person':
                # Extract person names
                names = []
                for line in lines:
                    # Line format: "name, uri" or just "name"
                    parts = line.split(',')
                    name = parts[0].strip()
                    if name and name not in names:
                        names.append(name)
                
                if not names:
                    return f"❌ Found {pattern.relation} data but couldn't extract names."
                
                # Format based on relation
                relation_text = {
                    'director': 'directed by',
                    'cast_member': 'starring',
                    'screenwriter': 'written by',
                    'producer': 'produced by'
                }.get(pattern.relation, pattern.relation.replace('_', ' '))
                
                if len(names) == 1:
                    return f"✅ **'{movie_label}'** was {relation_text} **{names[0]}**."
                else:
                    names_str = ", ".join(names[:-1]) + f", and {names[-1]}"
                    return f"✅ **'{movie_label}'** was {relation_text}:\n\n{names_str}"
            
            elif pattern.object_type == 'date':
                # Extract date (first line)
                date_value = lines[0].split(',')[0].strip()
                # Extract year from date
                year = date_value.split('-')[0] if '-' in date_value else date_value
                return f"✅ **'{movie_label}'** was released in **{year}**."
            
            elif pattern.object_type == 'string':
                # Extract string values (genre, rating)
                values = []
                for line in lines:
                    # Line format: "value, uri" or just "value"
                    parts = line.split(',')
                    val = parts[0].strip()
                    if val and val not in values:
                        values.append(val)
                
                if not values:
                    return f"❌ Found {pattern.relation} data but couldn't extract values."
                
                if len(values) == 1:
                    return f"✅ **'{movie_label}'** {pattern.relation.replace('_', ' ')}: **{values[0]}**"
                else:
                    values_str = ", ".join(values)
                    return f"✅ **'{movie_label}'** {pattern.relation.replace('_', ' ')}:\n\n{values_str}"
            
            # Default fallback
            return f"✅ Found {len(lines)} result(s) for '{movie_label}':\n\n{data}"
            
        except Exception as e:
            print(f"❌ Error formatting forward response: {e}")
            traceback.print_exc()
            return f"✅ Query executed successfully. Results:\n\n{data}"
    
    def _format_reverse_response(
        self,
        pattern: QueryPattern,
        person_label: str,
        data: str
    ) -> str:
        """
        Format response for reverse queries.
        
        Args:
            pattern: Query pattern
            person_label: Person name
            data: Query results (plain text from SPARQLHandler, NOT JSON)
            
        Returns:
            Formatted natural language response
        """
        try:
            # SPARQLHandler returns plain text, not JSON
            if not data or data == "No answer found in the database.":
                return f"❌ I couldn't find any films where **{person_label}** was the {pattern.relation.replace('_', ' ')} in the knowledge graph."
            
            # Parse plain text results
            lines = [line.strip() for line in data.strip().split('\n') if line.strip()]
            
            if not lines:
                return f"❌ I couldn't find any films where **{person_label}** was the {pattern.relation.replace('_', ' ')} in the knowledge graph."
            
            # Extract movie names
            movies = []
            for line in lines:
                # Line format: "movie, uri" or just "movie"
                parts = line.split(',')
                movie = parts[0].strip()
                if movie and movie not in movies:
                    movies.append(movie)
            
            if not movies:
                return f"❌ Found data but couldn't extract movie names."
            
            # Format response based on relation
            relation_text = {
                'director': 'directed',
                'cast_member': 'starred in',
                'screenwriter': 'wrote',
                'producer': 'produced'
            }.get(pattern.relation, pattern.relation.replace('_', ' '))
            
            if len(movies) == 1:
                return f"✅ **{person_label}** {relation_text} **{movies[0]}**."
            else:
                movies_list = "\n".join([f"• {movie}" for movie in movies])
                return f"✅ **{person_label}** {relation_text} **{len(movies)} films**:\n\n{movies_list}"
            
        except Exception as e:
            print(f"❌ Error formatting reverse response: {e}")
            traceback.print_exc()
            return f"✅ Query executed successfully. Results:\n\n{data}"
    
    def _format_verification_response(
        self,
        pattern: QueryPattern,
        person_label: str,
        movie_label: str,
        data: str
    ) -> str:
        """
        Format response for verification queries.
        
        Args:
            pattern: Query pattern
            person_label: Person name
            movie_label: Movie name
            data: Query results (plain text from SPARQLHandler, NOT JSON)
            
        Returns:
            Yes/No natural language response
        """
        try:
            # For ASK queries, SPARQLHandler should return "true" or "false" or boolean result
            # Check if data contains affirmative indicators
            data_lower = data.lower().strip()
            is_true = data_lower in ['true', 'yes', '1'] or data_lower.startswith('true')
            
            # Format based on relation
            relation_text = {
                'director': 'directed',
                'cast_member': 'starred in',
                'screenwriter': 'wrote',
                'producer': 'produced'
            }.get(pattern.relation, pattern.relation.replace('_', ' '))
            
            if is_true:
                return f"✅ **Yes**, **{person_label}** {relation_text} **'{movie_label}'**."
            else:
                # Convert past tense to present for "did not" phrasing
                relation_negative = {
                    'directed': 'direct',
                    'starred in': 'star in',
                    'wrote': 'write',
                    'produced': 'produce'
                }.get(relation_text, relation_text)
                
                return f"❌ **No**, **{person_label}** did not {relation_negative} **'{movie_label}'**."
            
        except Exception as e:
            print(f"❌ Error formatting verification response: {e}")
            traceback.print_exc()
            return f"✅ Query executed. Result: {data}"

    def _handle_unrecognized_query(self, query: str) -> str:
        """
        Handle queries that don't match any known pattern.
        
        Args:
            query: Original query
            
        Returns:
            Helpful error message with supported query examples
        """
        supported_relations = self.query_analyzer.get_supported_relations()
        
        examples = {
            'director': 'Who directed "The Matrix"?',
            'cast_member': 'What films did Tom Hanks star in?',
            'genre': 'What genre is "Inception"?',
            'publication_date': 'When was "The Godfather" released?'
        }
        
        examples_text = "\n".join([
            f"• {examples.get(rel, f'Query about {rel}')}"
            for rel in supported_relations[:4]
        ])
        
        return (
            "❌ I couldn't understand the structure of your question.\n\n"
            "**Supported query types:**\n"
            f"{examples_text}\n\n"
            "**Tips:**\n"
            "• Use quotes around movie titles\n"
            "• Be specific about what you're asking\n"
            "• Check spelling of names and titles"
        )
    
    def _process_complex_query(self, query: str, pattern: QueryPattern) -> str:
        """
        Process complex multi-constraint query.
        Example: "Which movie from South Korea won Academy Award for Best Picture?"
        
        Args:
            query: Natural language query
            pattern: Detected query pattern with constraint info
            
        Returns:
            Natural language response
        """
        print(f"   Direction: Complex multi-constraint")
        
        # Extract quoted entities (country, award names)
        quoted_entities = pattern.extracted_entities.get('quoted', [])
        constraints = pattern.extracted_entities.get('constraints', [])
        
        print(f"   Constraints: {', '.join(constraints)}")
        print(f"   Quoted entities: {quoted_entities}\n")
        
        if len(quoted_entities) < 2:
            return (
                "❌ I couldn't identify all required entities in your complex query.\n\n"
                "**Tips:**\n"
                "- Use quotes around specific values: \"South Korea\", \"Academy Award for Best Picture\"\n"
                "- Make sure to specify both the country and the award"
            )
        
        # Generate SPARQL for complex query
        print("📝 Step 3: Generating complex SPARQL query...")
        try:
            sparql = self._generate_complex_sparql(constraints, quoted_entities)
            
            print(f"✅ SPARQL generated:")
            print("-" * 80)
            print(sparql)
            print("-" * 80 + "\n")
        except Exception as e:
            print(f"❌ SPARQL generation failed: {e}\n")
            return f"❌ Failed to generate query for this complex request: {str(e)}"
        
        # Execute query
        print("📝 Step 4: Executing query against knowledge graph...")
        result = self._execute_sparql(sparql)
        
        if not result['success']:
            print(f"❌ Query execution failed: {result.get('error', 'Unknown error')}\n")
            return f"❌ Query execution failed: {result.get('error', 'Unknown error')}"
        
        # Format response
        print("📝 Step 5: Formatting response...")
        response = self._format_complex_response(
            constraints=constraints,
            entities=quoted_entities,
            data=result['data']
        )
        
        print(f"✅ Response generated\n")
        print("="*80)
        return response

    def _generate_complex_sparql(self, constraints: List[str], entities: List[str]) -> str:
        """
        Generate SPARQL for complex multi-constraint queries.
        
        Args:
            constraints: List of constraint types (e.g., ['country', 'award'])
            entities: List of entity values from query (e.g., ['South Korea', 'Academy Award for Best Picture'])
            
        Returns:
            SPARQL query string
        """
        # Map constraints to properties
        property_map = {
            'country': 'http://www.wikidata.org/prop/direct/P495',  # country of origin
            'award': 'http://www.wikidata.org/prop/direct/P166',    # award received
            'genre': 'http://www.wikidata.org/prop/direct/P136',
            'year': 'http://www.wikidata.org/prop/direct/P577'
        }
        
        # Build filter clauses for each constraint
        filters = []
        for i, constraint in enumerate(constraints):
            if i < len(entities) and constraint in property_map:
                property_uri = property_map[constraint]
                entity_value = entities[i]
                
                # Escape the entity value for SPARQL
                escaped_value = entity_value.replace('\\', '\\\\').replace('"', '\\"')
                
                if constraint == 'country':
                    filters.append(f"""
    ?movieUri <{property_uri}> ?country .
    ?country rdfs:label ?countryLabel .
    FILTER(LCASE(STR(?countryLabel)) = LCASE("{escaped_value}"))""")
                
                elif constraint == 'award':
                    filters.append(f"""
    ?movieUri <{property_uri}> ?award .
    ?award rdfs:label ?awardLabel .
    FILTER(LCASE(STR(?awardLabel)) = LCASE("{escaped_value}"))""")
        
        if not filters:
            raise ValueError("No valid constraints found in complex query")
        
        # Combine filters into SPARQL query
        sparql = f"""PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT DISTINCT ?movieLabel WHERE {{
    ?movieUri wdt:P31 wd:Q11424 .
    ?movieUri rdfs:label ?movieLabel .
    FILTER(LANG(?movieLabel) = "en" || LANG(?movieLabel) = "")
{chr(10).join(filters)}
}}
LIMIT 10"""
        
        return sparql

    def _format_complex_response(
        self,
        constraints: List[str],
        entities: List[str],
        data: str
    ) -> str:
        """Format response for complex multi-constraint queries."""
        
        if not data or data == "No answer found in the database.":
            constraint_desc = " and ".join([f"{c} '{e}'" for c, e in zip(constraints, entities)])
            return f"❌ I couldn't find any movies matching {constraint_desc} in the knowledge graph."
        
        # Parse plain text results
        lines = [line.strip() for line in data.strip().split('\n') if line.strip()]
        
        if not lines:
            return f"❌ No results found for the complex query."
        
        # Extract movie names
        movies = []
        for line in lines:
            parts = line.split(',')
            movie = parts[0].strip()
            if movie and movie not in movies:
                movies.append(movie)
        
        if not movies:
            return f"❌ Found data but couldn't extract movie names."
        
        # Format description of constraints
        constraint_desc = " and ".join([f"{c} **'{e}'**" for c, e in zip(constraints, entities)])
        
        if len(movies) == 1:
            return f"✅ The movie matching {constraint_desc} is **'{movies[0]}'**."
        else:
            movies_list = "\n".join([f"• {movie}" for movie in movies])
            return f"✅ Found **{len(movies)} movies** matching {constraint_desc}:\n\n{movies_list}"
    
    def process_embedding_query(self, query: str) -> str:
        """
        Process query using pure embedding approach.
        
        Supports:
        - Forward queries: Movie → Property (director, genre, etc.)
        - Reverse queries: Person → Movies
        
        NOT supported:
        - Verification queries (yes/no) - use factual approach
        - Superlative queries (highest/lowest) - use factual approach
        
        Args:
            query: Natural language query
            
        Returns:
            Natural language response with entity type information
        """
        print(f"\n{'='*80}")
        print(f"🔍 PROCESSING EMBEDDING QUERY")
        print(f"{'='*80}\n")
        print(f"Query: {query}\n")
        
        try:
            # ✅ FIXED: Clean query by removing only the embedding instruction prefix
            clean_query = query
            
            # Remove "Please answer this question with an embedding approach:" (case-insensitive)
            clean_query = re.sub(
                r'^\s*please\s+answer\s+this\s+question\s+with\s+(?:a|an)\s+embedding\s+approach\s*:\s*',
                '',
                clean_query,
                flags=re.IGNORECASE
            ).strip()
            
            # If that didn't match, try simpler patterns
            if clean_query == query:
                # Remove "Please answer this question:"
                clean_query = re.sub(
                    r'^\s*please\s+answer\s+this\s+question\s*:\s*',
                    '',
                    clean_query,
                    flags=re.IGNORECASE
                ).strip()
            
            print(f"Cleaned query: {clean_query}\n")
            
            # Analyze query pattern
            print("📝 Step 1: Analyzing query pattern...")
            pattern = self.query_analyzer.analyze(clean_query)
            
            if not pattern:
                print("❌ No pattern detected\n")
                return self._format_embedding_error("Could not understand query structure")
            
            print(f"✅ Pattern: {pattern.pattern_type} + {pattern.relation}")
            print(f"   Confidence: {pattern.confidence:.2%}\n")
            
            # ✅ Check for unsupported query types
            if pattern.extracted_entities and 'superlative' in pattern.extracted_entities:
                print(f"⚠️  Superlative query detected in embedding pipeline")
                return self._format_embedding_error(
                    "Superlative queries (highest/lowest/best/worst) are not supported in embedding mode. "
                    "These queries require aggregation which embeddings cannot provide. "
                    "Please use the factual approach or rephrase your question to ask about a specific movie."
                )
            
            if pattern.pattern_type == 'verification':
                print(f"⚠️  Verification query detected in embedding pipeline")
                return self._format_embedding_error(
                    "Verification queries (yes/no questions) are not well-supported by embeddings. "
                    "Please use the factual approach for more accurate yes/no answers."
                )
            
            # Route to appropriate handler
            if pattern.pattern_type == 'forward':
                return self._embedding_forward_query(clean_query, pattern)
            elif pattern.pattern_type == 'reverse':
                return self._embedding_reverse_query(clean_query, pattern)
            else:
                # Unsupported pattern type
                return self._format_embedding_error(
                    f"Query pattern '{pattern.pattern_type}' is not supported in embedding mode. "
                    "Supported patterns: forward (movie → property), reverse (person → movies). "
                    "Please use the factual approach or rephrase your question."
                )
            
        except Exception as e:
            print(f"❌ Error in embedding processing: {e}")
            import traceback
            traceback.print_exc()
            return self._format_embedding_error(str(e))
    
    def _embedding_forward_query(self, query: str, pattern: QueryPattern) -> str:
        """
        Process forward query using embeddings: Movie → Property
        
        ✅ ENHANCED: Uses QueryAnalyzer's relation extraction + better k-NN with relation context
        """
        print(f"   Direction: Forward ({pattern.relation})")
        
        # ✅ CRITICAL: Check superlative FIRST before entity extraction
        if pattern.extracted_entities and 'superlative' in pattern.extracted_entities:
            print("   ℹ️  Superlative query detected - delegating to superlative handler")
            return self._process_superlative_forward_query(query, pattern)
        
        # Regular forward query - needs entity extraction
        # Extract movie entity
        movie_entities = self.entity_extractor.extract_entities(
            query,
            entity_type="http://www.wikidata.org/entity/Q11424",  # Q11424 = film
            threshold=75
        )
        
        if not movie_entities:
            print("❌ No movie entity found\n")
            return self._format_embedding_error("Could not identify movie in query")
        
        # Get best match
        movie_uri, movie_text, score = movie_entities[0]
        movie_label = self.entity_extractor.get_entity_label(movie_uri)
        print(f"✅ Movie identified: '{movie_label}' (confidence: {score}%)\n")
        
        # ✅ NEW: Get relation embedding for better k-NN calculation
        print("📝 Step 2: Resolving relation and computing target embedding...")
        
        # ✅ CRITICAL: For language queries, enforce P364 property
        is_language_query = 'language' in pattern.relation.lower()
        if is_language_query:
            print(f"   🌐 Language query detected - enforcing P364 (original language)")
            relation_uri = 'http://www.wikidata.org/prop/direct/P364'
        else:
            # Get relation URI using the same robust approach as SPARQL pipeline
            relation_uri = self._get_relation_uri(pattern.relation)
        
        if not relation_uri:
            print(f"⚠️  Could not resolve relation URI for '{pattern.relation}'")
            return self._format_embedding_error(
                f"Could not resolve the relation '{pattern.relation}'. "
                f"This relation might not be available in the knowledge graph."
            )
        
        print(f"   ✅ Relation URI: {relation_uri}")
        
        # Get relation embedding
        relation_embedding = self.embedding_handler.get_relation_embedding(relation_uri)
        if relation_embedding is None:
            print(f"⚠️  No embedding found for relation {relation_uri}")
            # Fallback to direct question embedding
            return self._embedding_forward_direct_similarity(query, pattern, movie_uri, movie_label)
        
        print(f"   ✅ Relation embedding dimension: {relation_embedding.shape[0]}")
        
        # Get movie embedding
        movie_embedding = self.embedding_handler.get_entity_embedding(movie_uri)
        if movie_embedding is None:
            print(f"⚠️  No embedding found for movie {movie_uri}")
            # Fallback to direct question embedding
            return self._embedding_forward_direct_similarity(query, pattern, movie_uri, movie_label)
        
        print(f"   ✅ Movie embedding dimension: {movie_embedding.shape[0]}")
        
        # ✅ ENHANCED: Use TransE formula (h + r ≈ t) for better k-NN
        print("   Computing target embedding using TransE formula: h + r ≈ t")
        target_embedding = movie_embedding + relation_embedding
        
        print(f"   Target embedding norm: {np.linalg.norm(target_embedding):.3f}")
        
        # ✅ ENHANCED: Determine expected entity type based on pattern
        if is_language_query:
            expected_entity_type = 'language'
            expected_qcode = 'Q34770'
            print(f"   🌐 Language query - enforcing entity type: {expected_entity_type} (Q-code: {expected_qcode})")
        else:
            expected_entity_type = self._get_expected_entity_type(pattern)
            expected_qcode = self._get_expected_qcode(pattern)
            print(f"   Expected result type: {expected_entity_type} (Q-code: {expected_qcode})")
        
        # ✅ NEW: Special handling for language queries
        if 'language' in pattern.relation.lower():
            print(f"   🌐 Language query detected - using specialized language entity type")
            expected_qcode = 'Q34770'  # Q34770 = language
            expected_entity_type = 'language'
        
        print(f"   Expected result type: {expected_entity_type} (Q-code: {expected_qcode})")
        
        # ✅ Search for nearest entities using TransE-guided k-NN
        print(f"📝 Step 3: Finding nearest entities using TransE-guided k-NN...")
        validated_result = self._search_with_retry(
            target_embedding,
            expected_qcode,
            expected_entity_type,
            max_attempts=3
        )
        
        if validated_result is None:
            print("⚠️  TransE-guided search failed, trying direct similarity fallback...")
            return self._embedding_forward_direct_similarity(query, pattern, movie_uri, movie_label)
        
        result_uri, similarity = validated_result
        
        # ✅ ENHANCED: Get proper label with specialized language handling
        result_label = self._get_entity_label_robust(result_uri, expected_entity_type)
        
        # Get entity type
        entity_type = self._get_entity_type_label(result_uri)
        
        print(f"✅ Result: '{result_label}' (type: {entity_type}, similarity: {similarity:.3f})\n")
        
        return f"The answer suggested by embeddings is: **{result_label}** (type: {entity_type})"
    
    def _embedding_forward_direct_similarity(
        self,
        query: str,
        pattern: QueryPattern,
        movie_uri: str,
        movie_label: str
    ) -> str:
        """
        Fallback: Use direct question embedding similarity when TransE fails.
        
        Args:
            query: Natural language query
            pattern: Query pattern
            movie_uri: Movie URI
            movie_label: Movie label
            
        Returns:
            Natural language response
        """
        print("   Using direct question embedding similarity (fallback)")
        
        # Embed the entire question
        question_embedding = self.query_embedder.embed_query(query)
        
        # Align to TransE space
        aligned_embedding = self.aligner.align(question_embedding)
        print(f"   Question embedding dimension: {aligned_embedding.shape[0]}")
        print(f"   Question embedding norm: {np.linalg.norm(aligned_embedding):.3f}")
        
        # Determine expected entity type
        expected_entity_type = self._get_expected_entity_type(pattern)
        expected_qcode = self._get_expected_qcode(pattern)
        
        # Search for nearest entities using DIRECT similarity
        validated_result = self._search_with_retry(
            aligned_embedding,
            expected_qcode,
            expected_entity_type,
            max_attempts=3
        )
        
        if validated_result is None:
            return self._format_embedding_error(
                f"Could not find valid {expected_entity_type} result. "
                f"The {pattern.relation} information might not be available in the embedding space."
            )
        
        result_uri, similarity = validated_result
        
        # Get proper label
        result_label = self.embedding_handler.get_entity_label(result_uri, self.sparql_handler.graph)
        
        if not result_label or result_label == result_uri or result_label.startswith('http'):
            if '/entity/' in result_uri:
                entity_id = result_uri.split('/entity/')[-1].split('#')[0].split('?')[0]
                result_label = entity_id
            elif '/' in result_uri:
                result_label = result_uri.split('/')[-1]
            else:
                result_label = result_uri
        
        # Get entity type
        entity_type = self._get_entity_type_label(result_uri)
        
        print(f"✅ Result: '{result_label}' (type: {entity_type}, similarity: {similarity:.3f})\n")
        
        return f"The answer suggested by embeddings is: **{result_label}** (type: {entity_type})"

    def _embedding_reverse_query(self, query: str, pattern: QueryPattern) -> str:
        """
        Process reverse query using embeddings: Person → Movies
        
        ✅ ENHANCED: Uses QueryAnalyzer's relation extraction + TransE formula
        """
        print(f"   Direction: Reverse ({pattern.relation})")
        
        # Extract person entity
        person_entities = self.entity_extractor.extract_entities(
            query,
            entity_type="http://www.wikidata.org/entity/Q5",
            threshold=75
        )
        
        if not person_entities:
            return self._format_embedding_error("Could not identify person in query")
        
        person_uri, person_text, score = person_entities[0]
        person_label = self.entity_extractor.get_entity_label(person_uri)
        person_qcode = self.embedding_handler.get_entity_type_qcode(person_uri, self.sparql_handler.graph) or "Q5"
        print(f"✅ Person: '{person_label}' ({person_qcode})\n")
        
        # ✅ NEW: Get relation embedding for better k-NN
        print("📝 Step 2: Resolving relation and computing target embedding...")
        
        # Get relation URI
        relation_uri = self._get_relation_uri(pattern.relation)
        if not relation_uri:
            print(f"⚠️  Could not resolve relation URI for '{pattern.relation}'")
            # Fallback to direct question embedding
            return self._embedding_reverse_direct_similarity(query, pattern, person_uri, person_label, person_qcode)
        
        print(f"   ✅ Relation URI: {relation_uri}")
        
        # Get relation embedding
        relation_embedding = self.embedding_handler.get_relation_embedding(relation_uri)
        if relation_embedding is None:
            print(f"⚠️  No embedding found for relation {relation_uri}")
            # Fallback to direct question embedding
            return self._embedding_reverse_direct_similarity(query, pattern, person_uri, person_label, person_qcode)
        
        # Get person embedding
        person_embedding = self.embedding_handler.get_entity_embedding(person_uri)
        if person_embedding is None:
            print(f"⚠️  No embedding found for person {person_uri}")
            # Fallback to direct question embedding
            return self._embedding_reverse_direct_similarity(query, pattern, person_uri, person_label, person_qcode)
        
        print(f"   ✅ Person embedding dimension: {person_embedding.shape[0]}")
        
        # ✅ ENHANCED: For reverse queries, we need to find movies where:
        # movie - relation ≈ person, so we search for: person - (-relation) = person + relation
        # OR: movie ≈ person + relation (depending on TransE training)
        # We'll try both approaches
        
        print("   Computing target embeddings using TransE formulas")
        
        # Approach 1: person + relation (forward direction)
        target_embedding_forward = person_embedding + relation_embedding
        print(f"   Target (forward): norm = {np.linalg.norm(target_embedding_forward):.3f}")
        
        # Approach 2: person - relation (reverse direction)
        target_embedding_reverse = person_embedding - relation_embedding
        print(f"   Target (reverse): norm = {np.linalg.norm(target_embedding_reverse):.3f}")
        
        # ✅ Try both approaches and merge results
        expected_qcode = 'Q11424'  # Movies
        movie_type_uri = "http://www.wikidata.org/entity/Q11424"
        
        print(f"📝 Step 3: Finding nearest movies using TransE-guided k-NN...")
        
        try:
            filter_uris = self.embedding_handler.get_entities_by_type(movie_type_uri, self.sparql_handler.graph)
            print(f"   Filtering to {len(filter_uris)} movies")
            
            # Get candidates from both approaches
            nearest_forward = self.embedding_handler.find_nearest_entities(
                target_embedding_forward,
                top_k=20,
                filter_uris=filter_uris
            )
            
            nearest_reverse = self.embedding_handler.find_nearest_entities(
                target_embedding_reverse,
                top_k=20,
                filter_uris=filter_uris
            )
            
            # Merge and re-rank by best similarity
            merged = {}
            for uri, sim in nearest_forward:
                merged[uri] = sim
            for uri, sim in nearest_reverse:
                if uri not in merged or sim > merged[uri]:
                    merged[uri] = sim
            
            nearest = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:10]
            
        except Exception as e:
            print(f"   ⚠️  TransE-guided search failed: {e}")
            # Fallback to direct similarity
            return self._embedding_reverse_direct_similarity(query, pattern, person_uri, person_label, person_qcode)
        
        if not nearest:
            return self._format_embedding_error("No results found in embedding space")

        # Format results
        results = []
        for movie_uri, similarity in nearest[:5]:
            movie_label = self.embedding_handler.get_entity_label(movie_uri, self.sparql_handler.graph)
            movie_qcode = self.embedding_handler.get_entity_type_qcode(movie_uri, self.sparql_handler.graph) or "Q11424"
            results.append(f"• **{movie_label}** (type: {movie_qcode}, similarity: {similarity:.3f})")

        results_text = "\n".join(results)
        
        return f"✅ Movies directed by **{person_label}** ({person_qcode}) according to embeddings (type: Q11424):\n\n{results_text}"
    
    def _embedding_reverse_direct_similarity(
        self,
        query: str,
        pattern: QueryPattern,
        person_uri: str,
        person_label: str,
        person_qcode: str
    ) -> str:
        """
        Fallback: Use direct question embedding similarity for reverse queries.
        """
        print("   Using direct question embedding similarity (fallback)")
        
        # Embed the entire question
        question_embedding = self.query_embedder.embed_query(query)
        
        # Align to TransE space
        aligned_embedding = self.aligner.align(question_embedding)
        
        # Expected result type: movies
        expected_qcode = 'Q11424'
        movie_type_uri = "http://www.wikidata.org/entity/Q11424"
        
        try:
            filter_uris = self.embedding_handler.get_entities_by_type(movie_type_uri, self.sparql_handler.graph)
            print(f"   Filtering to {len(filter_uris)} movies")
            
            nearest = self.embedding_handler.find_nearest_entities(
                aligned_embedding,
                top_k=10,
                filter_uris=filter_uris
            )
        except Exception as e:
            print(f"   ⚠️  Type filtering failed: {e}")
            nearest = self.embedding_handler.find_nearest_entities(aligned_embedding, top_k=10)
        
        if not nearest:
            return self._format_embedding_error("No results found in embedding space")

        # Format results
        results = []
        for movie_uri, similarity in nearest[:5]:
            movie_label = self.embedding_handler.get_entity_label(movie_uri, self.sparql_handler.graph)
            movie_qcode = self.embedding_handler.get_entity_type_qcode(movie_uri, self.sparql_handler.graph) or "Q11424"
            results.append(f"• **{movie_label}** (type: {movie_qcode}, similarity: {similarity:.3f})")

        results_text = "\n".join(results)
        
        return f"✅ Movies directed by **{person_label}** ({person_qcode}) according to embeddings (type: Q11424):\n\n{results_text}"
    
    def _get_relation_uri(self, relation_key: str) -> Optional[str]:
        """
        Get relation URI from relation key using RelationManager.
        
        Args:
            relation_key: Relation key (e.g., 'director', 'cast_member')
            
        Returns:
            Relation URI or None if not found
        """
        if self.relation_manager is None:
            print(f"[EmbeddingProcessor] ⚠️  RelationManager not available")
            return None
        
        uri = self.relation_manager.get_relation_uri(relation_key)
        if uri:
            return uri
        
        # Fallback: try to construct URI from common patterns
        property_map = {
            'director': 'http://www.wikidata.org/prop/direct/P57',
            'cast_member': 'http://www.wikidata.org/prop/direct/P161',
            'screenwriter': 'http://www.wikidata.org/prop/direct/P58',
            'producer': 'http://www.wikidata.org/prop/direct/P162',
            'genre': 'http://www.wikidata.org/prop/direct/P136',
            'publication_date': 'http://www.wikidata.org/prop/direct/P577',
            'country_of_origin': 'http://www.wikidata.org/prop/direct/P495',
            'rating': 'http://ddis.ch/atai/rating',
            'award_received': 'http://www.wikidata.org/prop/direct/P166',
            'characters': 'http://www.wikidata.org/prop/direct/P674',
            'production_company': 'http://www.wikidata.org/prop/direct/P272',
            'original_language_of_film_or_tv_show': 'http://www.wikidata.org/prop/direct/P364',
        }
        
        return property_map.get(relation_key)
    
    def _get_entity_label_robust(self, entity_uri: str, expected_type: str) -> str:
        """
        Get entity label with specialized handling for different entity types.
        
        Args:
            entity_uri: Entity URI
            expected_type: Expected entity type (e.g., 'language', 'person', 'movie')
            
        Returns:
            Human-readable label
        """
        # Try standard label lookup first
        result_label = self.embedding_handler.get_entity_label(entity_uri, self.sparql_handler.graph)
        
        # If label is valid and not just the URI, return it
        if result_label and result_label != entity_uri and not result_label.startswith('http'):
            return result_label
        
        # ✅ Special handling for language entities
        if expected_type == 'language' or 'Q34770' in entity_uri:
            print(f"   🌐 Language entity detected, trying comprehensive label lookup")
            
            # Try multiple label properties for languages
            from rdflib import URIRef, RDFS
            entity_ref = URIRef(entity_uri)
            
            # Try rdfs:label with language filter
            for lang in ['en', '', 'mul']:
                for label in self.sparql_handler.graph.objects(entity_ref, RDFS.label):
                    label_str = str(label)
                    # Filter by language if available
                    if hasattr(label, 'language'):
                        if lang and label.language == lang:
                            print(f"   ✅ Found language label: '{label_str}' (lang: {label.language})")
                            return label_str
                    elif not lang:  # No language tag
                        print(f"   ✅ Found language label: '{label_str}' (no lang tag)")
                        return label_str
            
            # Try wdt:P1705 (native label)
            P1705 = URIRef("http://www.wikidata.org/prop/direct/P1705")
            for label in self.sparql_handler.graph.objects(entity_ref, P1705):
                native_label = str(label)
                print(f"   ✅ Found native language label (P1705): '{native_label}'")
                return native_label
            
            # Try wdt:P1813 (short name)
            P1813 = URIRef("http://www.wikidata.org/prop/direct/P1813")
            for label in self.sparql_handler.graph.objects(entity_ref, P1813):
                short_name = str(label)
                print(f"   ✅ Found short language name (P1813): '{short_name}'")
                return short_name
        
        # Fallback: extract Q-code and make it more readable
        if '/Q' in entity_uri:
            qcode = entity_uri.split('/Q')[-1].split('#')[0].split('?')[0]
            qcode = 'Q' + qcode
            
            # Try to get label for this Q-code from graph
            from rdflib import URIRef, RDFS
            qcode_uri = f"http://www.wikidata.org/entity/{qcode}"
            qcode_ref = URIRef(qcode_uri)
            
            for label in self.sparql_handler.graph.objects(qcode_ref, RDFS.label):
                label_str = str(label)
                if label_str and not label_str.startswith('http'):
                    print(f"   ✅ Found label via Q-code lookup: '{label_str}'")
                    return label_str
            
            return qcode
        
        # Final fallback
        if '/entity/' in entity_uri:
            entity_id = entity_uri.split('/entity/')[-1].split('#')[0].split('?')[0]
            return entity_id
        elif '/' in entity_uri:
            return entity_uri.split('/')[-1]
        
        return entity_uri
    
    def _get_expected_entity_type(self, pattern: QueryPattern) -> str:
        """
        Get human-readable expected entity type from pattern.
        
        Args:
            pattern: Query pattern
            
        Returns:
            Entity type string (e.g., 'person', 'movie', 'date', 'language')
        """
        # ✅ Check for language-specific relations first
        if 'language' in pattern.relation.lower():
            return 'language'
        
        # Return the object type from pattern
        return pattern.object_type

    def _get_expected_qcode(self, pattern: QueryPattern) -> str:
        """
        Get expected Wikidata Q-code for result entity type.
        
        Args:
            pattern: Query pattern
            
        Returns:
            Q-code string (e.g., 'Q5' for person, 'Q11424' for movie)
        """
        qcode_map = {
            'person': 'Q5',
            'movie': 'Q11424',
            'string': None,  # No Q-code for strings
            'date': None,    # No Q-code for dates
            'organization': 'Q43229',
            'language': 'Q34770',  # ✅ NEW: Q34770 = language
            'entity': None   # Generic, no specific Q-code
        }
        
        # ✅ NEW: Check for language-specific relations
        if 'language' in pattern.relation.lower():
            return 'Q34770'
        
        return qcode_map.get(pattern.object_type)
    
    def _search_with_retry(
        self,
        target_embedding: np.ndarray,
        expected_qcode: Optional[str],
        expected_entity_type: str,
        max_attempts: int = 3
    ) -> Optional[Tuple[str, float]]:
        """
        Search for nearest entities with retry and validation.
        
        Args:
            target_embedding: Target embedding vector
            expected_qcode: Expected Q-code for result entities
            expected_entity_type: Expected entity type
            max_attempts: Maximum number of search attempts
            
        Returns:
            (entity_uri, similarity) tuple or None if no valid result found
        """
        # If no specific type expected, just return nearest
        if expected_qcode is None or expected_entity_type in ['string', 'date', 'entity']:
            nearest = self.embedding_handler.find_nearest_entities(target_embedding, top_k=1)
            if nearest:
                return nearest[0]
            return None
        
        # Otherwise, filter by entity type
        entity_type_uri = f"http://www.wikidata.org/entity/{expected_qcode}"
        
        try:
            filter_uris = self.embedding_handler.get_entities_by_type(
                entity_type_uri,
                self.sparql_handler.graph
            )
            
            if not filter_uris:
                print(f"   ⚠️  No entities found with type {expected_qcode}")
                # Fallback: try without filter
                nearest = self.embedding_handler.find_nearest_entities(target_embedding, top_k=1)
                if nearest:
                    return nearest[0]
                return None
            
            print(f"   Searching among {len(filter_uris)} entities of type {expected_qcode}")
            
            nearest = self.embedding_handler.find_nearest_entities(
                target_embedding,
                top_k=10,
                filter_uris=filter_uris
            )
            
            if nearest:
                return nearest[0]
            
            return None
            
        except Exception as e:
            print(f"   ⚠️  Type-filtered search failed: {e}")
            # Fallback: try without filter
            nearest = self.embedding_handler.find_nearest_entities(target_embedding, top_k=1)
            if nearest:
                return nearest[0]
            return None
    
    def _get_entity_type_label(self, entity_uri: str) -> str:
        """
        Get human-readable type label for an entity.
        
        Args:
            entity_uri: Entity URI
            
        Returns:
            Type label string (e.g., 'Q5', 'Q11424')
        """
        qcode = self.embedding_handler.get_entity_type_qcode(
            entity_uri,
            self.sparql_handler.graph
        )
        
        if qcode:
            return qcode
        
        # Fallback: try to infer from URI
        if 'Q' in entity_uri:
            parts = entity_uri.split('Q')
            if len(parts) > 1:
                qcode = 'Q' + parts[-1].split('/')[0].split('#')[0].split('?')[0]
                return qcode
        
        return 'unknown'
    
    def _format_embedding_error(self, message: str) -> str:
        """
        Format an error message for embedding processing failures.
        
        Args:
            message: Error message
            
        Returns:
            Formatted error string
        """
        return f"❌ **Embedding processing error**: {message}"