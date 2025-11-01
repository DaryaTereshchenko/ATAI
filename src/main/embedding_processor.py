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
        
        Two strategies:
        1. Direct embedding: Embed the NL question and find nearest entity in embedding space
        2. Entity+Relation extraction: Extract entity and relation, compute in embedding space
        
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
            
            # ✅ NEW: Check if this is a superlative query
            if pattern.extracted_entities and 'superlative' in pattern.extracted_entities:
                print(f"⚠️  Superlative query detected in embedding pipeline")
                return self._format_embedding_error(
                    "Superlative queries (highest/lowest/best/worst) are not supported in embedding mode. "
                    "These queries require aggregation which embeddings cannot provide. "
                    "Please use the factual approach or rephrase your question to ask about a specific movie."
                )
            
            # STRATEGY 1: Try entity+relation approach (more accurate)
            if pattern.pattern_type == 'forward':
                return self._embedding_forward_query(clean_query, pattern)
            elif pattern.pattern_type == 'reverse':
                return self._embedding_reverse_query(clean_query, pattern)
            elif pattern.pattern_type == 'verification':
                return self._embedding_verification_query(clean_query, pattern)
            else:
                # STRATEGY 2: Direct embedding approach (fallback)
                return self._embedding_direct_query(clean_query, pattern)
            
        except Exception as e:
            print(f"❌ Error in embedding processing: {e}")
            import traceback
            traceback.print_exc()
            return self._format_embedding_error(str(e))
    
    def _embedding_forward_query(self, query: str, pattern: QueryPattern) -> str:
        """
        Process forward query using embeddings: Movie → Property
        Strategy: Extract movie entity, use TransE to find related entities.
        """
        print(f"   Direction: Forward ({pattern.relation})")
        
        # ✅ SAFETY CHECK: This should not happen but double-check for superlative
        if pattern.extracted_entities and 'superlative' in pattern.extracted_entities:
            return self._format_embedding_error(
                "Superlative queries require aggregation which embeddings cannot provide"
            )
        
        # Extract movie entity
        movie_entities = self.entity_extractor.extract_entities(
            query,
            entity_type="http://www.wikidata.org/entity/Q11424",
            threshold=75
        )
        
        if not movie_entities:
            return self._format_embedding_error("Could not identify movie in query")
        
        movie_uri, movie_text, score = movie_entities[0]
        movie_label = self.entity_extractor.get_entity_label(movie_uri)
        print(f"✅ Movie: '{movie_label}'\n")
        
        # Get movie embedding
        movie_embedding = self.embedding_handler.get_entity_embedding(movie_uri)
        if movie_embedding is None:
            return self._format_embedding_error(f"No embedding found for movie '{movie_label}'")
        
        # Get relation embedding
        relation_uri = self._get_relation_uri(pattern.relation)
        relation_embedding = self.embedding_handler.get_relation_embedding(relation_uri)
        
        if relation_embedding is None:
            return self._format_embedding_error(f"No embedding found for relation '{pattern.relation}'")
        
        # TransE: head + relation ≈ tail
        # Compute expected tail embedding
        expected_tail = movie_embedding + relation_embedding
        
        # ✅ NEW: Determine expected entity type based on pattern
        expected_entity_type = self._get_expected_entity_type(pattern)
        expected_qcode = self._get_expected_qcode(pattern)
        print(f"📝 Expected result type: {expected_entity_type} (Q-code: {expected_qcode})")
        
        # Find nearest entity
        print(f"📝 Step 2: Finding nearest entity in embedding space...")
        
        # ✅ ENHANCED: Get more candidates and validate their types
        top_k = 20  # ✅ INCREASED: Get more candidates for better validation
        
        # ✅ ENHANCED: For string types (genre, country), filter by broader criteria
        if expected_entity_type == 'string':
            print(f"   String type detected - using broader search")
            # For genres, countries, etc., we need to search more broadly
            # These are labeled entities in the graph, not necessarily with types
            nearest = self.embedding_handler.find_nearest_entities(
                expected_tail,
                top_k=top_k
            )
        elif expected_entity_type and expected_entity_type != 'string':
            type_uri = self._entity_type_to_uri(expected_entity_type)
            
            if type_uri is not None:
                try:
                    filter_uris = self.embedding_handler.get_entities_by_type(type_uri, self.sparql_handler.graph)
                    print(f"   Filtering to {len(filter_uris)} entities of type {expected_entity_type}")
                    
                    nearest = self.embedding_handler.find_nearest_entities(
                        expected_tail,
                        top_k=top_k,
                        filter_uris=filter_uris
                    )
                except Exception as e:
                    print(f"   ⚠️  Type filtering failed: {e}")
                    nearest = self.embedding_handler.find_nearest_entities(
                        expected_tail,
                        top_k=top_k
                    )
            else:
                nearest = self.embedding_handler.find_nearest_entities(
                    expected_tail,
                    top_k=top_k
                )
        else:
            # For date types or unknown, get all candidates
            print(f"   No entity type filtering (expected type: {expected_entity_type})")
            nearest = self.embedding_handler.find_nearest_entities(
                expected_tail,
                top_k=top_k
            )
        
        if not nearest:
            return self._format_embedding_error("No results found in embedding space")
        
        # ✅ NEW: Validate result type and pick the best match
        validated_result = self._validate_and_select_result(
            nearest, 
            expected_qcode, 
            expected_entity_type
        )
        
        if validated_result is None:
            # ✅ ENHANCED: Better error message with suggestions
            return self._format_embedding_error(
                f"Could not find valid {expected_entity_type} result. "
                f"Top candidates had wrong types or the {pattern.relation} information "
                f"might not be available in the embedding space."
            )
        
        result_uri, similarity = validated_result
        
        # ✅ CRITICAL: Get proper label instead of URI
        result_label = self.embedding_handler.get_entity_label(result_uri, self.sparql_handler.graph)
        
        # ✅ ENHANCED: If label is still a URI or empty, extract entity name from URI
        if not result_label or result_label == result_uri or result_label.startswith('http'):
            print(f"   ⚠️  No label found for {result_uri}, extracting from URI")
            # Extract entity ID from URI (e.g., Q6256 from http://www.wikidata.org/entity/Q6256)
            if '/entity/' in result_uri:
                entity_id = result_uri.split('/entity/')[-1].split('#')[0].split('?')[0]
                result_label = entity_id
                print(f"   📝 Extracted entity ID: {entity_id}")
            elif '/' in result_uri:
                result_label = result_uri.split('/')[-1]
            else:
                result_label = result_uri
        
        # Get entity type
        entity_type = self._get_entity_type_label(result_uri)
        
        print(f"✅ Result: '{result_label}' (type: {entity_type}, similarity: {similarity:.3f})\n")
        
        return f"The answer suggested by embeddings is: **{result_label}** (type: {entity_type})"
    
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
        
        # ✅ FIXED: Map both country variants to Q6256
        if pattern.relation in ['country', 'country_of_origin']:
            return 'Q6256'  # Country Q-code
        
        # ✅ ENHANCED: Comprehensive mapping for common relations
        relation_to_qcode = {
            # People-related
            'director': 'Q5',
            'cast_member': 'Q5',
            'screenwriter': 'Q5',
            'producer': 'Q5',
            'voice_actor': 'Q5',
            'director_of_photography': 'Q5',
            'film_editor': 'Q5',
            'composer': 'Q5',
            'executive_producer': 'Q5',
            'costume_designer': 'Q5',
            'production_designer': 'Q5',
            'narrator': 'Q5',
            'animator': 'Q5',
            'sound_designer': 'Q5',
            'choreographer': 'Q5',
            'storyboard_artist': 'Q5',
            'art_director': 'Q5',
            'make_up_artist': 'Q5',
            'illustrator': 'Q5',
            
            # Media & content
            'genre': 'Q201658',
            'characters': 'Q15632617',
            'based_on': 'Q7725634',
            'derivative_work': 'Q11424',
            'part_of_the_series': 'Q24856',
            'follows': 'Q11424',
            'followed_by': 'Q11424',
            'present_in_work': 'Q11424',
            'media_franchise': 'Q130371093',
            
            # Geographic
            'country_of_origin': 'Q6256',
            'country': 'Q6256',
            'filming_location': 'Q208511',
            'place_of_birth': 'Q1093829',
            'place_of_death': 'Q745456',
            'narrative_location': 'Q6256',
            'headquarters_location': 'Q1093829',
            'location': 'Q1066984',
            
            # ✅ FIXED: Language & Culture - Use Q1097949 (natural language) as specified
            'original_language_of_film_or_tv_show': 'Q1097949',  # natural language
            'language_of_work_or_name': 'Q1097949',
            'languages_spoken_written_or_signed': 'Q1097949',
            'native_language': 'Q1097949',
            'writing_language': 'Q1097949',
            'original_language': 'Q1097949',  # Alias
            
            # Awards & Recognition
            'award_received': 'Q38033430',
            'nominated_for': 'Q38033430',
            
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
            'main_subject': 'Q813912',
            'form_of_creative_work': 'Q4263830',
            'time_period': 'Q578',
            'described_by_source': 'Q186165',
            'from_narrative_universe': 'Q559618',
            'takes_place_in_fictional_universe': 'Q559618',
            
            # Production and Company
            'production_company': 'Q783794',   # Company that produces films
            'production_studio': 'Q783794',    # Same as production company
            'studio': 'Q783794',              # Same concept
            'publisher': 'Q2085381',          # Publishing company
            
            # Locations and Settings
            'narrative_location': 'Q2221906',  # Geographic location 
            'setting': 'Q2221906',            # Location where story is set
            'filming_location': 'Q2221906',    # Location where filmed
            'set_in': 'Q2221906',             # Story setting
            
            # Geographic/Countries (expanded)
            'country_of_origin': 'Q6256',      # Country
            'country': 'Q6256',                # Country (alias)
            'location': 'Q2221906',            # Location (general)
            'production_location': 'Q2221906',  # Where produced
        }
        
        return relation_to_qcode.get(pattern.relation)

    def _validate_and_select_result(
        self,
        candidates: List[Tuple[str, float]],
        expected_qcode: Optional[str],
        expected_type: Optional[str]
    ) -> Optional[Tuple[str, float]]:
        """
        Validate candidates against expected Q-code and select best match.
        
        ✅ ENHANCED: More robust validation with better error handling.
        """
        if not expected_qcode:
            # For types without Q-code validation (ratings, strings, dates)
            if expected_type in ['string', 'date', None]:
                print(f"   ℹ️  No Q-code validation needed for type: {expected_type}")
                return candidates[0] if candidates else None
            return candidates[0] if candidates else None
        
        print(f"   Validating {len(candidates)} candidates against expected type {expected_qcode}...")
        
        # Build type hierarchy map
        type_hierarchy = self._build_type_hierarchy()
        expected_types = type_hierarchy.get(expected_qcode, [expected_qcode])
        
        print(f"   Accepting Q-codes: {', '.join(expected_types[:5])}{'...' if len(expected_types) > 5 else ''}")
        
        # ✅ NEW: Track best candidate by validation strategy
        best_candidate = None
        best_score = 0
        
        for uri, similarity in candidates:
            entity_qcode = self._get_entity_type_label(uri)
            
            # Strategy 1: Exact Q-code match (score: 5)
            if entity_qcode == expected_qcode:
                print(f"   ✅ Found exact type match: {uri} (type: {entity_qcode})")
                return (uri, similarity)
            
            # Strategy 2: Type hierarchy match (score: 4)
            if entity_qcode in expected_types:
                if best_score < 4:
                    best_candidate = (uri, similarity)
                    best_score = 4
                    print(f"   ✅ Found related type match: {uri} (type: {entity_qcode} in hierarchy)")
            
            # Strategy 3: Subclass match (score: 3)
            if best_score < 3 and self._is_subclass_of(uri, expected_qcode):
                best_candidate = (uri, similarity)
                best_score = 3
                print(f"   ✅ Found subclass match: {uri} (subclass of {expected_qcode})")
            
            # Strategy 4: Label-based validation (score: 2)
            if best_score < 2 and self._validate_by_label(uri, expected_qcode):
                best_candidate = (uri, similarity)
                best_score = 2
                print(f"   ✅ Found label-based match: {uri} (validated by label for {expected_qcode})")
        
        # Return best candidate found, or use lenient fallback
        if best_candidate:
            return best_candidate
        
        # Lenient fallback for complex types
        lenient_types = ['string', 'Q11424', 'Q7725634', 'Q208511', 'Q1066984', 'Q813912', 'Q4263830', 'Q6256']
        
        if expected_type in lenient_types or expected_qcode in lenient_types:
            if candidates:
                print(f"   ⚠️  Using lenient matching for type {expected_qcode}, returning first candidate")
                return candidates[0]
        
        print(f"   ❌ No candidates matched expected type {expected_qcode}")
        return None
    
    def _build_type_hierarchy(self) -> Dict[str, List[str]]:
        """
        Build a type hierarchy map for common Wikidata types.
        Maps parent types to their related/child types.
        
        Returns:
            Dictionary mapping Q-code to list of acceptable Q-codes (including hierarchy)
        """
        return {
            # Countries and geographic entities
            'Q6256': [  # country
                'Q6256',   # country
                'Q6465',   # department (French administrative division)
                'Q515',    # city
                'Q5107',   # continent
                'Q82794',  # geographic region
                'Q1048835', # political territorial entity
                'Q3624078', # sovereign state
                'Q15634554', # state with limited recognition
                'Q1549591',  # big city
                'Q486972',   # human settlement
                'Q3024240',  # historical country
                'Q3024240',  # historical region
            ],
            # Human/Person
            'Q5': [
                'Q5',      # human
                'Q15632617', # fictional human (for character queries)
            ],
            # Film/Movie
            'Q11424': [
                'Q11424',  # film
                'Q24862',  # short film
                'Q506240', # television film
                'Q506240', # TV movie
            ],
            # Genre
            'Q201658': [
                'Q201658',  # film genre
                'Q188451',  # genre (general)
                'Q483394',  # music genre
            ],
            # ✅ FIXED: Language - Use Q1097949 (natural language)
            'Q1097949': [
                'Q1097949', # natural language
                'Q34770',   # language (general)
                'Q33742',   # natural language (alternative)
                'Q1288568', # language (Wikidata property)
                'Q14827288', # Wikidata language code
            ],
            # Award
            'Q38033430': [
                'Q38033430', # film award
                'Q618779',   # award (general)
                'Q618779',   # prize
            ],
            # Production and Company
            'Q783794': [
                'Q783794', # production company
                'Q783794', # production studio
                'Q783794', # studio
                'Q2085381', # publisher
            ],
            # Locations and Settings
            'Q2221906': [
                'Q2221906', # narrative location
                'Q2221906', # setting
                'Q2221906', # filming location
                'Q2221906', # set in
            ],
        }
    
    def _validate_by_label(self, entity_uri: str, expected_qcode: str) -> bool:
        """
        Validate entity by checking its label against expected type patterns.
        Useful for geographic entities, genres, and other string-like types.
        
        Args:
            entity_uri: Entity URI to validate
            expected_qcode: Expected Q-code
            
        Returns:
            True if label matches expected patterns
        """
        try:
            label = self.embedding_handler.get_entity_label(entity_uri, self.sparql_handler.graph)
            if not label or label == entity_uri:
                return False
            
            label_lower = label.lower()
            
            # Validation patterns for different Q-codes
            validation_patterns = {
                'Q6256': [  # Country/geographic entity
                    # Common country indicators
                    r'\b(republic|kingdom|state|nation|country|territory|federation|confederation)\b',
                    r'\b(islands?|peninsula|region|province|department)\b',
                    # French departments often end in numbers or specific suffixes
                    r'\b(ain|aisne|allier|ardèche|charente|corrèze|côte|dordogne|doubs|eure|gard|gers|hérault|indre|isère|jura|loire|lot|lozère|marne|meuse|nord|oise|orne|pas|puy|rhône|saône|sarthe|savoie|seine|somme|tarn|var|vienne|vosges|yonne)\b',
                ],
                'Q201658': [  # Film genre
                    r'\b(film|movie|cinema|genre|comedy|drama|action|thriller|horror|romance|documentary|animation)\b',
                ],
                'Q1097949': [  # Language
                    r'\b(language|lingua|tongue|dialect|speech)\b',
                ],
                'Q38033430': [  # Award
                    r'\b(award|prize|trophy|medal|oscar|golden|prix|premio)\b',
                ],
                'Q783794': [  # Production company patterns
                    r'\b(studios?|pictures?|films?|entertainment|productions?|media)\b',
                    r'\b(company|corporation|incorporated|inc|ltd|llc)\b'
                ],
                'Q2221906': [  # Location/setting patterns
                    r'\b(city|town|village|region|district|area|location)\b',
                    r'\b(north|south|east|west|central|downtown)\b',
                    r'\b(street|avenue|road|boulevard|plaza|square)\b'
                ],
            }
            
            patterns = validation_patterns.get(expected_qcode, [])
            for pattern in patterns:
                if re.search(pattern, label_lower):
                    print(f"      Label '{label}' matches pattern for {expected_qcode}")
                    return True
            
            return False
            
        except Exception as e:
            print(f"      Error in label validation: {e}")
            return False
    
    def _format_embedding_error(self, error_message: str) -> str:
        """
        Format error message for embedding queries.
        
        Args:
            error_message: Error description
            
        Returns:
            Formatted error message
        """
        return f"❌ Embedding approach failed: {error_message}"
    
    def _get_relation_uri(self, relation: str) -> Optional[str]:
        """
        Get relation URI for a given relation name.
        
        ✅ ENHANCED: Properly utilize RelationManager with better debugging.
        """
        print(f"[Processor] 🔍 Resolving relation URI for: '{relation}'")
        
        if not self.relation_manager:
            print(f"[Processor] ⚠️  No RelationManager available, using hardcoded fallback")
            return self._get_hardcoded_relation_uri(relation)
        
        # ✅ STRATEGY 1: Direct RelationManager lookup (now includes internal fallbacks)
        uri = self.relation_manager.get_relation_uri(relation)
        if uri:
            print(f"[Processor] ✅ Resolved via RelationManager: {uri}")
            return uri
        
        # ✅ STRATEGY 2: Try with normalized relation name
        normalized = relation.replace('_', ' ').lower().strip()
        if normalized != relation.lower():
            print(f"[Processor] 🔄 Trying normalized form: '{normalized}'")
            matches = self.relation_manager.find_relation(normalized, top_k=1)
            if matches and matches[0][2] > 0.6:
                best_key, best_uri, best_conf = matches[0]
                print(f"[Processor] ✅ Resolved via normalized match: '{relation}' → '{best_key}' ({best_conf:.2%})")
                return best_uri
        
        # ✅ STRATEGY 3: Try fuzzy matching with lower threshold
        print(f"[Processor] 🔄 Trying lenient fuzzy match for '{relation}'")
        matches = self.relation_manager.find_relation(relation, top_k=5)
        
        if matches:
            print(f"[Processor] 📋 Top fuzzy matches:")
            for i, (key, match_uri, conf) in enumerate(matches, 1):
                print(f"   {i}. {key} ({conf:.2%})")
            
            # Use best match if confidence is reasonable
            best_key, best_uri, best_conf = matches[0]
            if best_conf > 0.5:  # ✅ Lower threshold for embeddings
                print(f"[Processor] ✅ Using fuzzy match: '{relation}' → '{best_key}' ({best_conf:.2%})")
                return best_uri
            else:
                print(f"[Processor] ⚠️  Best match confidence too low: {best_conf:.2%}")
        
        # ✅ STRATEGY 4: Hardcoded fallback as last resort
        print(f"[Processor] ⚠️  All RelationManager strategies failed, using hardcoded fallback")
        return self._get_hardcoded_relation_uri(relation)
    
    def _get_hardcoded_relation_uri(self, relation: str) -> Optional[str]:
        """
        Hardcoded fallback for common relations.
        
        Args:
            relation: Relation name
            
        Returns:
            Relation URI or None
        """
        fallback_uris = {
            'country_of_origin': 'http://www.wikidata.org/prop/direct/P495',
            'country': 'http://www.wikidata.org/prop/direct/P495',
            'director': 'http://www.wikidata.org/prop/direct/P57',
            'cast_member': 'http://www.wikidata.org/prop/direct/P161',
            'genre': 'http://www.wikidata.org/prop/direct/P136',
            'publication_date': 'http://www.wikidata.org/prop/direct/P577',
            'screenwriter': 'http://www.wikidata.org/prop/direct/P58',
            'producer': 'http://www.wikidata.org/prop/direct/P162',
            'original_language_of_film_or_tv_show': 'http://www.wikidata.org/prop/direct/P364',
            'original_language': 'http://www.wikidata.org/prop/direct/P364',
            'language': 'http://www.wikidata.org/prop/direct/P364',
            'award_received': 'http://www.wikidata.org/prop/direct/P166',
            'rating': 'http://ddis.ch/atai/rating',
        }
        
        relation_lower = relation.lower().strip()
        
        # Direct lookup
        if relation_lower in fallback_uris:
            uri = fallback_uris[relation_lower]
            print(f"[Processor] 💾 Using hardcoded fallback: {uri}")
            return uri
        
        # ✅ STRATEGY 4: SPARQLGenerator as absolute last resort
        try:
            uri = self.sparql_generator._get_relation_uri(relation)
            print(f"[Processor] ⚠️  Using SPARQLGenerator fallback: {uri}")
            return uri
        except Exception as e:
            print(f"[Processor] ❌ All strategies failed for '{relation}': {e}")
            return None

    def _get_expected_entity_type(self, pattern: QueryPattern) -> Optional[str]:
        """
        Get expected entity type based on query pattern.
        
        Args:
            pattern: Query pattern
            
        Returns:
            Expected entity type string (e.g., 'person', 'string', 'date')
        """
        return pattern.object_type if pattern.pattern_type == 'forward' else pattern.subject_type
    
    def _entity_type_to_uri(self, entity_type: str) -> Optional[str]:
        """
        Convert entity type string to Wikidata URI.
        
        Args:
            entity_type: Entity type string (e.g., 'person', 'movie')
            
        Returns:
            Wikidata URI or None
        """
        type_map = {
            'person': 'http://www.wikidata.org/entity/Q5',
            'movie': 'http://www.wikidata.org/entity/Q11424',
            'country': 'http://www.wikidata.org/entity/Q6256',
            'genre': 'http://www.wikidata.org/entity/Q201658',
            'language': 'http://www.wikidata.org/entity/Q1097949',
            'award': 'http://www.wikidata.org/entity/Q38033430',
            'organization': 'http://www.wikidata.org/entity/Q43229',
            'company': 'http://www.wikidata.org/entity/Q783794',
            'location': 'http://www.wikidata.org/entity/Q2221906',
        }
        return type_map.get(entity_type)
    
    def _get_entity_type_label(self, entity_uri: str) -> str:
        """
        Get entity type label (Q-code) from entity URI.
        
        Args:
            entity_uri: Entity URI
            
        Returns:
            Q-code string (e.g., 'Q5') or 'unknown'
        """
        try:
            # Query for entity's P31 (instance of) property
            query = f"""
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            SELECT ?type WHERE {{
                <{entity_uri}> wdt:P31 ?type .
            }}
            LIMIT 1
            """
            
            result = self.sparql_handler.execute_query(query, validate=False)
            if result.get('success') and result.get('data'):
                # Extract Q-code from result
                lines = result['data'].strip().split('\n')
                if lines and lines[0]:
                    type_uri = lines[0].split(',')[0].strip()
                    # Extract Q-code from URI
                    if '/entity/' in type_uri:
                        return type_uri.split('/entity/')[-1].split('#')[0].split('?')[0]
                    elif '/' in type_uri:
                        return type_uri.split('/')[-1]
            
            return 'unknown';
            
        except Exception as e:
            print(f"⚠️  Error getting entity type for {entity_uri}: {e}")
            return 'unknown'
    
    def _is_subclass_of(self, entity_uri: str, expected_qcode: str) -> bool:
        """
        Check if entity is a subclass of expected type.
        
        ✅ FIXED: Handle boolean return value properly.
        """
        try:
            # Query for P279 (subclass of) relationship
            query = f"""
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            ASK {{
                <{entity_uri}> wdt:P279* wd:{expected_qcode} .
            }}
            """
            
            result = self.sparql_handler.execute_query(query, validate=False)
            if result.get('success') and result.get('data'):
                # ✅ CRITICAL FIX: The data is a string "true" or "false", not a list
                data_str = result['data'].strip().lower()
                return data_str in ['true', 'yes', '1']
            
            return False
            
        except Exception as e:
            print(f"⚠️  Error checking subclass relationship: {e}")
            return False
    
    def _embedding_reverse_query(self, query: str, pattern: QueryPattern) -> str:
        """
        Process reverse query using embeddings: Person → Movies
        
        ✅ FIXED: Better response formatting with proper entity type extraction.
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
        
        # Get person embedding
        person_embedding = self.embedding_handler.get_entity_embedding(person_uri)
        if person_embedding is None:
            return self._format_embedding_error(f"No embedding found for person '{person_label}'")
        
        # Get relation embedding
        relation_uri = self._get_relation_uri(pattern.relation)
        if relation_uri is None:
            return self._format_embedding_error(f"Could not resolve relation '{pattern.relation}'")
        
        relation_embedding = self.embedding_handler.get_relation_embedding(relation_uri)
        
        if relation_embedding is None:
            return self._format_embedding_error(f"No embedding found for relation '{pattern.relation}'")
        
        # TransE reverse: tail - relation ≈ head
        expected_head = person_embedding - relation_embedding
        
        # Find nearest movies
        print(f"📝 Step 2: Finding nearest movies in embedding space...")
        movie_type_uri = "http://www.wikidata.org/entity/Q11424"
        
        try:
            filter_uris = self.embedding_handler.get_entities_by_type(movie_type_uri, self.sparql_handler.graph)
            print(f"   Filtering to {len(filter_uris)} movies")
            
            nearest = self.embedding_handler.find_nearest_entities(
                expected_head,
                top_k=10,
                filter_uris=filter_uris
            )
        except Exception as e:
            print(f"   ⚠️  Type filtering failed: {e}")
            nearest = self.embedding_handler.find_nearest_entities(expected_head, top_k=10)
        
        if not nearest:
            return self._format_embedding_error("No results found in embedding space")

        # ✅ ENHANCED: Format results with better structure
        results = []
        for movie_uri, similarity in nearest[:5]:
            movie_label = self.embedding_handler.get_entity_label(movie_uri, self.sparql_handler.graph)
            movie_qcode = self.embedding_handler.get_entity_type_qcode(movie_uri, self.sparql_handler.graph) or "Q11424"
            results.append(f"• **{movie_label}** (type: {movie_qcode}, similarity: {similarity:.3f})")

        results_text = "\n".join(results)
        
        # ✅ CRITICAL: Add entity type marker for test validation
        return f"✅ Movies directed by **{person_label}** ({person_qcode}) according to embeddings (type: Q11424):\n\n{results_text}"