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
        sparql_handler: Optional[SPARQLHandler] = None
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
        
        # ✅ NEW: Initialize query analyzer and SPARQL generator
        print("   🧠 Initializing query analysis components...")
        from src.config import SPARQL_CLASSIFIER_MODEL_PATH
        self.query_analyzer = QueryAnalyzer(
            use_transformer=True,
            transformer_model_path=SPARQL_CLASSIFIER_MODEL_PATH  # Use SPARQL classifier, not question classifier
        )
        self.sparql_generator = SPARQLGenerator(self.sparql_handler)
        
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
        
        # PRIMARY: Try LLM-based generation FIRST with pattern-specific examples
        try:
            print("   Attempting LLM-based generation...")
            
            # Construct a descriptive query based on pattern
            if pattern.pattern_type == 'forward':
                llm_query = f"What is the {pattern.relation.replace('_', ' ')} of \"{subject_label}\"?"
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
        """Generate SPARQL for superlative queries (highest/lowest)."""
        
        if pattern.relation == 'rating':
            # Query for highest/lowest rated movie
            order = "DESC" if superlative == "MAX" else "ASC"
            
            sparql = f"""PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX ddis: <http://ddis.ch/atai/>

SELECT ?movieLabel ?rating WHERE {{
    ?movieUri wdt:P31 wd:Q11424 .
    ?movieUri rdfs:label ?movieLabel .
    ?movieUri ddis:rating ?rating .
    FILTER(LANG(?movieLabel) = "en" || LANG(?movieLabel) = "")
    FILTER(DATATYPE(?rating) = xsd:decimal || DATATYPE(?rating) = xsd:double || DATATYPE(?rating) = xsd:float)
}}
ORDER BY {order}(?rating)
LIMIT 1"""
            
            return sparql
        
        else:
            raise ValueError(f"Superlative queries not supported for relation: {pattern.relation}")

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
            value = parts[1].strip()
            
            descriptor = "highest" if superlative == "MAX" else "lowest"
            return f"✅ The movie with the **{descriptor} {pattern.relation}** is **'{movie}'** with a {pattern.relation} of **{value}**."
        
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