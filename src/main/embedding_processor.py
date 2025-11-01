"""
Embedding-based Query Processor.
✅ NOW: Uses dynamically extracted relations and entities from knowledge graph.
"""

import sys
import os
import traceback
import numpy as np  # ✅ ADDED: Missing numpy import
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
    """
    
    def __init__(
        self,
        embeddings_dir: str,
        graph_path: str,
        query_model: str = "all-MiniLM-L6-v2",
        alignment_matrix_path: Optional[str] = None,
        use_simple_aligner: bool = False,
        sparql_handler: Optional[SPARQLHandler] = None,
        relation_classifier_path: Optional[str] = None  # ✅ NEW: Accept classifier path
    ):
        """
        Initialize the embedding query processor.
        """
        # Initialize embedding handler for TransE embeddings
        self.embedding_handler = EmbeddingHandler(embeddings_dir)
        
        # Initialize query embedder for natural language
        self.query_embedder = QueryEmbedder(model_name=query_model)
        
        # Initialize alignment between query and TransE spaces
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
            self.sparql_handler = SPARQLHandler(graph_file_path=graph_path)
        else:
            self.sparql_handler = sparql_handler
        
        # Initialize entity extractor
        self.entity_extractor = EntityExtractor(self.sparql_handler.graph)
        
        # ✅ REMOVED: No more relation analyzer for embeddings
        # ✅ REMOVED: No more query analyzer for embeddings
        
        # ✅ Keep SPARQL components for factual queries only
        from src.main.query_analyzer import QueryAnalyzer
        from src.main.embedding_relation_matcher import EmbeddingRelationMatcher
        
        self.embedding_relation_matcher = EmbeddingRelationMatcher(
            embedding_handler=self.embedding_handler,
            query_embedder=self.query_embedder,
            aligner=self.aligner
        )
        
        self.query_analyzer = QueryAnalyzer(
            use_transformer=False,
            transformer_model_path=None,
            sparql_handler=self.sparql_handler,
            embedding_matcher=self.embedding_relation_matcher,
            relation_classifier_path=relation_classifier_path  # ✅ NEW: Pass classifier path
        )
        self.sparql_generator = SPARQLGenerator(self.sparql_handler)
        
        # ✅ Extract dynamic schema from SPARQLGenerator
        self.relation_uris = self.sparql_generator.relation_uris
        self.type_uris = self.sparql_generator.type_uris
        
        # ✅ Build reverse mappings
        self._build_reverse_mappings()
        self._add_fallback_qcode_mappings()
        
        # Initialize NL2SPARQL for LLM fallback
        from src.main.nl_to_sparql import NLToSPARQL
        self.nl2sparql = NLToSPARQL(method="direct-llm", sparql_handler=self.sparql_handler)

    def _build_reverse_mappings(self):
        """Build reverse mappings from relation names to URIs and Q-codes."""
        # ✅ REMOVED: Initial print statement
        
        # Initialize empty mapping
        self.relation_to_qcode = {}
        
        # ✅ Only attempt dynamic extraction if graph has data
        if not self.sparql_handler.graph or len(self.sparql_handler.graph) == 0:
            print("   ⚠️  Graph is empty, skipping dynamic Q-code inference")
            return
        
        from rdflib import URIRef
        P31 = URIRef("http://www.wikidata.org/prop/direct/P31")
        
        for relation_name, relation_uri in self.relation_uris.items():
            relation_ref = URIRef(relation_uri)
            
            object_types = {}
            sample_count = 0
            max_samples = 50
            
            for s, p, o in self.sparql_handler.graph.triples((None, relation_ref, None)):
                if isinstance(o, URIRef):
                    for type_uri in self.sparql_handler.graph.objects(o, P31):
                        type_str = str(type_uri)
                        if '/Q' in type_str:
                            qcode = type_str.split('/Q')[-1].split('#')[0]
                            object_types[qcode] = object_types.get(qcode, 0) + 1
                
                sample_count += 1
                if sample_count >= max_samples:
                    break
            
            if object_types:
                most_common_qcode = max(object_types.items(), key=lambda x: x[1])[0]
                self.relation_to_qcode[relation_name] = most_common_qcode
                # ✅ REMOVED: Individual relation logging
        
        print(f"   ✅ Built mappings for {len(self.relation_to_qcode)} relations")

    def _add_fallback_qcode_mappings(self):
        """Add/merge fallback Q-code mappings (does not overwrite existing)."""
        fallback_mappings = {
            'director': '5',
            'cast_member': '5',
            'screenwriter': '5',
            'producer': '5',
            'genre': '201658',
            'country_of_origin': '6256',
        }
        
        # Merge without overwriting
        for relation, qcode in fallback_mappings.items():
            if relation not in self.relation_to_qcode:
                self.relation_to_qcode[relation] = qcode
        
        print(f"   ✅ Total Q-code mappings: {len(self.relation_to_qcode)}")
    
    def process_hybrid_factual_query(self, query: str) -> str:
        """
        Process factual query using ROBUST pattern analysis + dynamic SPARQL.
        """
        # ✅ FORCE PRINT - These should always be visible
        print(f"\n{'='*80}", flush=True)
        print(f"🔍 PROCESSING HYBRID FACTUAL QUERY", flush=True)
        print(f"{'='*80}\n", flush=True)
        print(f"Query: {query}\n", flush=True)
        
        try:
            # ==================== STEP 1: ANALYZE QUERY PATTERN ====================
            print("📝 Step 1: Analyzing query pattern...", flush=True)
            
            pattern = self.query_analyzer.analyze(query)
            
            if not pattern:
                return self._handle_unrecognized_query(query)
            
            # ✅ NEW: ENHANCED LOGGING - Show relation extraction method immediately
            print(f"\n{'='*80}", flush=True)
            print(f"📊 RELATION EXTRACTION RESULTS", flush=True)
            print(f"{'='*80}", flush=True)
            print(f"Query: '{query[:80]}...'", flush=True)
            print(f"\n✅ Pattern Detected:", flush=True)
            print(f"  • Type: {pattern.pattern_type}", flush=True)
            print(f"  • Relation: {pattern.relation}", flush=True)
            print(f"  • Subject Type: {pattern.subject_type}", flush=True)
            print(f"  • Object Type: {pattern.object_type}", flush=True)
            print(f"  • Confidence: {pattern.confidence:.2%}", flush=True)
            
            # ✅ NEW: Decode which method extracted the relation
            if hasattr(pattern, 'extracted_entities') and pattern.extracted_entities:
                if 'keywords' in pattern.extracted_entities:
                    keywords = pattern.extracted_entities['keywords']
                    print(f"\n📋 Extraction Method Details:", flush=True)
                    if keywords and len(keywords) > 0:
                        first_keyword = keywords[0]
                        
                        if first_keyword.startswith('bert:'):
                            print(f"  • Method: 🤖 DISTILBERT CLASSIFIER", flush=True)
                            print(f"  • Classification: {first_keyword.split(':')[1]}", flush=True)
                            print(f"  • Description: Fine-tuned transformer model", flush=True)
                        elif first_keyword.startswith('sbert:'):
                            print(f"  • Method: 🔢 SBERT ZERO-SHOT MATCHER", flush=True)
                            print(f"  • Matched Property: {first_keyword.split(':')[1]}", flush=True)
                            print(f"  • Description: Semantic similarity to property descriptions", flush=True)
                        elif first_keyword.startswith('embedding:'):
                            print(f"  • Method: 🎯 TRANSE EMBEDDING MATCHER", flush=True)
                            print(f"  • Matched Relation: {first_keyword.split(':')[1]}", flush=True)
                            print(f"  • Description: TransE embedding cosine similarity", flush=True)
                        else:
                            print(f"  • Method: 🔤 KEYWORD-BASED FALLBACK", flush=True)
                            print(f"  • Matched Keyword: '{first_keyword}'", flush=True)
                            print(f"  • Description: Rule-based pattern matching", flush=True)
                    else:
                        print(f"  • Method: ⚠️ UNKNOWN", flush=True)
                        print(f"  • Keywords list is empty", flush=True)
            
            # ✅ NEW: Show relation URI resolution
            relation_uri = self.relation_uris.get(pattern.relation)
            print(f"\n📋 Relation URI Mapping:", flush=True)
            if relation_uri:
                if '/P' in relation_uri:
                    prop_code = 'P' + relation_uri.split('/P')[-1].split('#')[0]
                    print(f"  • Relation Name: '{pattern.relation}'", flush=True)
                    print(f"  • Wikidata Property: {prop_code}", flush=True)
                    print(f"  • Full URI: {relation_uri}", flush=True)
                    print(f"  • Status: ✅ RESOLVED", flush=True)
                else:
                    prop_code = relation_uri.split('/')[-1]
                    print(f"  • Relation Name: '{pattern.relation}'", flush=True)
                    print(f"  • Custom Property: {prop_code}", flush=True)
                    print(f"  • Full URI: {relation_uri}", flush=True)
                    print(f"  • Status: ✅ RESOLVED", flush=True)
            else:
                print(f"  • Relation Name: '{pattern.relation}'", flush=True)
                print(f"  • Status: ❌ NOT FOUND IN MAPPING", flush=True)
                print(f"  • Available relations: {list(self.relation_uris.keys())[:10]}...", flush=True)
            
            print(f"{'='*80}\n", flush=True)
            
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

    def _process_forward_query(self, query: str, pattern: QueryPattern) -> str:
        """
        Process forward query: Entity → Property
        """
        # ✅ REMOVED: Duplicate relation extraction logging (now done earlier)
        # The detailed logging is now in process_hybrid_factual_query

        print(f"   Direction: Forward ({pattern.subject_type} → {pattern.object_type})")
        
        # ✅ Check superlative first
        if pattern.extracted_entities and 'superlative' in pattern.extracted_entities:
            print("   ℹ️  Superlative query detected - delegating to superlative handler")
            return self._process_superlative_forward_query(query, pattern)
        
        # ✅ CRITICAL: Always extract movie entities for movie queries
        # Determine entity type based on query keywords, not pattern.subject_type
        query_lower = query.lower()
        movie_keywords = ['movie', 'film', 'directed', 'genre', 'released', 'cast']
        
        if any(kw in query_lower for kw in movie_keywords) or pattern.subject_type == '11424':
            entity_type_uri = "http://www.wikidata.org/entity/Q11424"
            print(f"   🎬 Detected movie query, using Q11424 entity type")
        elif pattern.subject_type == 'person' or pattern.subject_type == '5':
            entity_type_uri = "http://www.wikidata.org/entity/Q5"
        elif pattern.subject_type in self.type_uris:
            entity_type_uri = self.type_uris[pattern.subject_type]
        else:
            entity_type_uri = None
        
        print(f"   🔍 Entity Extraction:")
        print(f"      Type filter: {entity_type_uri or 'None (any entity)'}")
        
        # Extract entity
        subject_entities = self.entity_extractor.extract_entities(
            query,
            entity_type=entity_type_uri,
            threshold=75
        )
        
        if not subject_entities:
            entity_type_name = pattern.subject_type if pattern.subject_type != 'entity' else 'entity'
            print(f"❌ No {entity_type_name} entity found\n")
            return (
                f"❌ I couldn't identify the {entity_type_name} in your question.\n\n"
                "**Tips:**\n"
                f"- Use quotes around the {entity_type_name} name\n"
                "- Check the spelling\n"
                "- Make sure it exists in the knowledge graph"
            )
        
        # Get best match
        subject_uri, subject_text, score = subject_entities[0]
        subject_label = self.entity_extractor.get_entity_label(subject_uri)
        print(f"✅ Entity identified: '{subject_label}' (confidence: {score}%)")
        print(f"   URI: {subject_uri}\n")
        
        # Generate SPARQL query with fallback
        print("📝 Step 3: Generating SPARQL query...")
        
        try:
            sparql_result = self._generate_sparql_with_fallback(
                pattern=pattern,
                subject_label=subject_label
            )
            
            # ✅ CRITICAL: Validate sparql_result structure
            if not isinstance(sparql_result, dict):
                raise ValueError(f"SPARQL generation returned invalid type: {type(sparql_result)}")
            
            if 'query' not in sparql_result:
                raise ValueError("SPARQL generation missing 'query' key")
            
            if 'method' not in sparql_result:
                raise ValueError("SPARQL generation missing 'method' key")
            
            sparql = sparql_result['query']
            method = sparql_result['method']
            confidence = sparql_result.get('confidence', 0.0)
            
            # ✅ Log the generated SPARQL
            print(f"\n{'='*80}")
            print(f"📄 GENERATED SPARQL QUERY")
            print(f"{'='*80}")
            print(f"Generation Method: {method}")
            print(f"Confidence: {confidence:.2%}")
            print(f"\nQuery:")
            print(f"{'─'*80}")
            print(sparql)
            print(f"{'─'*80}\n")
            
        except Exception as e:
            print(f"❌ SPARQL generation failed: {e}\n")
            traceback.print_exc()
            return f"❌ Failed to generate query for this request: {str(e)}"
        
        # Execute query
        print("📝 Step 4: Executing query against knowledge graph...")
        print(f"   Query being executed (full query):")
        print(f"{'─'*80}")
        print(sparql)
        print(f"{'─'*80}\n")
        
        result = self._execute_sparql(sparql)
        
        print(f"   🔍 BREAKPOINT 4.2: After SPARQL execution")
        print(f"      Success: {result['success']}")
        if result['success']:
            print(f"      Data type: {type(result.get('data'))}")
            print(f"      Data preview: {str(result.get('data', ''))[:200]}...")
            print(f"      Data is empty: {not result.get('data') or result.get('data') == 'No answer found in the database.'}")
        else:
            print(f"      Error: {result.get('error', 'Unknown error')}")
        
        if not result['success']:
            print(f"❌ Query execution failed: {result.get('error', 'Unknown error')}\n")
            return f"❌ Query execution failed: {result.get('error', 'Unknown error')}"
        
        # Format response
        print("📝 Step 5: Formatting response...")
        print(f"   🔍 BREAKPOINT 5.1: Before response formatting")
        print(f"      Raw data: {result['data'][:200] if result['data'] else 'None'}...")
        
        response = self._format_forward_response(
            pattern=pattern,
            movie_label=subject_label,
            data=result['data']
        )
        
        print(f"   🔍 BREAKPOINT 5.2: After response formatting")
        print(f"      Response length: {len(response)} chars")
        print(f"      Response preview: {response[:150]}...")
        
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
            
            # ✅ ENHANCED: Add more sophisticated filtering
            # Strategy: Filter out movies that are likely test data by requiring:
            # 1. Valid rating range (1.0 to 9.5)
            # 2. Movie has at least one other property (director, cast, or genre) - indicates it's a real entry
            # 3. English label
            
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
    
    # ✅ ENHANCED: Require at least one of: director, cast member, or genre
    # This filters out test/stub entries that only have title and rating
    {{
        ?movieUri wdt:P57 ?director .
    }} UNION {{
        ?movieUri wdt:P161 ?cast .
    }} UNION {{
        ?movieUri wdt:P136 ?genre .
    }}
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
                    
                    # ✅ FIX: Skip if name looks like a URI (starts with http://)
                    if name.startswith('http://'):
                        # This line only has URI, no label - skip it
                        print(f"[Formatter] ⚠️  Skipping result without label: {name}")
                        continue
                    
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
                    # ✅ FIXED: Use "and" for last item
                    names_str = ", ".join(names[:-1]) + f" and {names[-1]}"
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
                    
                    # ✅ FIX: Skip if value looks like a URI
                    if val.startswith('http://'):
                        print(f"[Formatter] ⚠️  Skipping result without label: {val}")
                        continue
                    
                    if val and val not in values:
                        values.append(val)
                
                if not values:
                    return f"❌ Found {pattern.relation} data but couldn't extract values."
                
                if len(values) == 1:
                    return f"✅ **'{movie_label}'** {pattern.relation.replace('_', ' ')}: **{values[0]}**"
                else:
                    # ✅ FIXED: Concatenate with "and" instead of newlines
                    values_str = " and ".join(values)
                    return f"✅ **'{movie_label}'** {pattern.relation.replace('_', ' ')}: **{values_str}**"
            
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
    FILTER(LANG(?countryLabel) = "en" || LANG(?countryLabel) = "")
    FILTER(LCASE(STR(?countryLabel)) = LCASE("{escaped_value}"))""")
                
                elif constraint == 'award':
                    filters.append(f"""
    ?movieUri <{property_uri}> ?award .
    ?award rdfs:label ?awardLabel .
    FILTER(LANG(?awardLabel) = "en" || LANG(?awardLabel) = "")
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
        ✅ SIMPLIFIED: Direct embedding matching without relation analysis.
        
        Strategy:
        1. Clean and embed the full question
        2. Find nearest entity in embedding space
        3. Detect expected type from query keywords
        4. Validate result type matches expected
        
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
            # ✅ Step 1: Clean and embed query
            print(f"📝 Step 1: Cleaning and embedding query...")
            clean_query = self._clean_query_for_embedding(query)
            print(f"   Cleaned query: '{clean_query}'")
            
            # ✅ Step 2: Detect expected type
            print(f"\n📝 Step 2: Detecting expected answer type...")
            expected_type, expected_qcode = self._detect_expected_type_from_query(query)
            print(f"   Expected type: {expected_type} (Q-code: {expected_qcode})")
            
            # ✅ Step 3: Embed query
            print(f"\n📝 Step 3: Embedding query...")
            query_embedding = self.query_embedder.embed(clean_query)
            
            # ✅ Step 4: Align to TransE space
            print(f"\n📝 Step 4: Aligning to TransE space...")
            aligned_embedding = self.aligner.align(query_embedding)
            
            # ✅ Step 5: Find nearest entities
            print(f"\n📝 Step 5: Finding nearest entities...")
            
            # Check if we should filter by type
            if expected_qcode and expected_qcode not in ['string', 'date', 'unknown']:
                entity_type_uri = self._entity_type_to_uri(expected_type)
                if entity_type_uri:
                    print(f"   Filtering by type: {expected_type}")
                    nearest = self.embedding_handler.find_nearest_entities(
                        aligned_embedding,
                        top_k=10,
                        entity_type=entity_type_uri
                    )
                else:
                    nearest = self.embedding_handler.find_nearest_entities(
                        aligned_embedding,
                        top_k=10
                    )
            else:
                nearest = self.embedding_handler.find_nearest_entities(
                    aligned_embedding,
                    top_k=10
                )
            
            if not nearest:
                return self._format_embedding_error("No results found in embedding space")
            
            # ✅ Step 6: Validate and select best match
            print(f"\n📝 Step 6: Validating results...")
            print(f"   Found {len(nearest)} candidates")
            
            # Show top 3 candidates with better formatting
            for i, (uri, sim) in enumerate(nearest[:3], 1):
                label = self.embedding_handler.get_entity_label(uri, self.sparql_handler.graph)
                qcode = self._get_entity_type_qcode(uri)
                print(f"   {i}. {label} (type: {qcode}, similarity: {sim:.3f})")
            
            # Select best match that matches expected type (if specified)
            best_match = None
            
            if expected_qcode and expected_qcode not in ['string', 'date', 'unknown']:
                # Try to find type match
                for uri, similarity in nearest:
                    actual_qcode = self._get_entity_type_qcode(uri)
                    if self._qcodes_match(actual_qcode, expected_qcode):
                        best_match = (uri, similarity, actual_qcode)
                        print(f"\n✅ Found type-matching result: {actual_qcode} = {expected_qcode}")
                        break
            
            # Fallback: use best similarity
            if not best_match:
                uri, similarity = nearest[0]
                actual_qcode = self._get_entity_type_qcode(uri)
                best_match = (uri, similarity, actual_qcode)
                
                if expected_qcode and expected_qcode not in ['string', 'date', 'unknown']:
                    print(f"\n⚠️  No exact type match found. Using best similarity.")
                    print(f"   Expected: {expected_qcode}, Got: {actual_qcode}")
            
            # ✅ Format response with better output
            result_uri, similarity, result_qcode = best_match
            result_label = self.embedding_handler.get_entity_label(
                result_uri, 
                self.sparql_handler.graph
            )
            
            print(f"\n✅ Final answer: {result_label} (type: {result_qcode}, similarity: {similarity:.3f})")
            
            # ✅ IMPROVED: Show label instead of URI
            if result_qcode == "unknown":
                # Try to infer type from expected type
                display_type = expected_qcode if expected_qcode != "unknown" else result_qcode
                return f"The answer suggested by embeddings is: **{result_label}** (type: {display_type})"
            else:
                return f"The answer suggested by embeddings is: **{result_label}** (type: {result_qcode})"
            
        except Exception as e:
            print(f"❌ Error in embedding processing: {e}")
            import traceback
            traceback.print_exc()
            return self._format_embedding_error(str(e))    
    
    def _clean_query_for_embedding(self, query: str) -> str:
        """Clean query for embedding (remove instruction prefixes)."""
        clean = query
        
        # Remove instruction prefixes
        clean = re.sub(
            r'^please\s+answer\s+this\s+question\s+with\s+(?:a|an)\s+embedding\s+approach:\s*',
            '',
            clean,
            flags=re.IGNORECASE
        )
        
        clean = re.sub(
            r'^please\s+answer\s+this\s+question:\s*',
            '',
            clean,
            flags=re.IGNORECASE
        )
        
        return clean.strip()
    
    def _detect_expected_type_from_query(self, query: str) -> Tuple[str, str]:
        """
        Detect expected answer type from query keywords.
        
        Returns:
            (type_name, q_code) tuple
        """
        query_lower = query.lower()
        
        # ✅ Person indicators (who, director, actor, etc.)
        person_keywords = ['who', 'director', 'actor', 'actress', 'screenwriter', 
                          'writer', 'producer', 'composer', 'cast']
        if any(kw in query_lower for kw in person_keywords):
            return ('person', 'Q5')
        
        # ✅ Country indicators
        country_keywords = ['country', 'nation', 'produced in', 'made in', 'from what country']
        if any(kw in query_lower for kw in country_keywords):
            return ('country', 'Q6256')
        
        # ✅ Language indicators
        language_keywords = ['language', 'spoken in', 'filmed in language', 'dialogue']
        if any(kw in query_lower for kw in language_keywords):
            return ('language', 'Q1288568')
        
        # ✅ Genre indicators
        genre_keywords = ['genre', 'type of movie', 'kind of film']
        if any(kw in query_lower for kw in genre_keywords):
            return ('genre', 'Q201658')
        
        # ✅ Movie indicators (reverse queries: "what films did X...")
        movie_keywords = ['what movies', 'what films', 'which films', 'filmography']
        if any(kw in query_lower for kw in movie_keywords):
            return ('movie', 'Q11424')
        
        # ✅ Date indicators
        date_keywords = ['when', 'release date', 'came out', 'published']
        if any(kw in query_lower for kw in date_keywords):
            return ('date', 'date')
        
        # Default: unknown type
        return ('unknown', 'unknown')
    
    def _qcodes_match(self, qcode1: str, qcode2: str) -> bool:
        """Check if two Q-codes match (handles formatting differences)."""
        if not qcode1 or not qcode2:
            return False
        
        # Normalize both
        q1 = qcode1.strip().upper()
        q2 = qcode2.strip().upper()
        
        if not q1.startswith('Q'):
            q1 = 'Q' + q1
        if not q2.startswith('Q'):
            q2 = 'Q' + q2
        
        return q1 == q2
    
    def _entity_type_to_uri(self, entity_type: str) -> Optional[str]:
        """
        Convert entity type name to URI.
        
        Args:
            entity_type: Entity type name (e.g., 'person', 'movie', 'country')
            
        Returns:
            Entity type URI or None
        """
        type_map = {
            'person': 'http://www.wikidata.org/entity/Q5',
            'movie': 'http://www.wikidata.org/entity/Q11424',
            'country': 'http://www.wikidata.org/entity/Q6256',
            'genre': 'http://www.wikidata.org/entity/Q201658',
            'language': 'http://www.wikidata.org/entity/Q1288568'
        }
        return type_map.get(entity_type)
    
    def _get_entity_type_qcode(self, entity_uri: str) -> str:
        """
        Get the Q-code for an entity's type.
        
        Args:
            entity_uri: Entity URI
            
        Returns:
            Q-code string (e.g., 'Q5') or 'unknown'
        """
        try:
            from rdflib import URIRef
            P31 = URIRef("http://www.wikidata.org/prop/direct/P31")
            entity_ref = URIRef(entity_uri)
            
            # Get the entity's type(s)
            for type_uri in self.sparql_handler.graph.objects(entity_ref, P31):
                type_str = str(type_uri)
                
                # ✅ FIX: Better Q-code extraction
                if '/entity/Q' in type_str:
                    # Standard Wikidata entity format
                    qcode = type_str.split('/entity/Q')[-1].split('#')[0].split('/')[0]
                    return f"Q{qcode}"
                elif '/Q' in type_str:
                    # Generic format
                    qcode = type_str.split('/Q')[-1].split('#')[0].split('/')[0]
                    return f"Q{qcode}"
            
            # ✅ FALLBACK: If no P31 found, check if the entity URI itself is a Q-code
            if '/entity/Q' in entity_uri:
                qcode = entity_uri.split('/entity/Q')[-1].split('#')[0].split('/')[0]
                return f"Q{qcode}"
            elif '/Q' in entity_uri:
                qcode = entity_uri.split('/Q')[-1].split('#')[0].split('/')[0]
                return f"Q{qcode}"
            
            return "unknown"
        
        except Exception as e:
            print(f"⚠️  Error getting entity type: {e}")
            import traceback
            traceback.print_exc()
            return "unknown"
    
    def _format_embedding_error(self, error_message: str) -> str:
        """
        Format error message for embedding queries.
        
        Args:
            error_message: Error message
            
        Returns:
            Formatted error response
        """
        return f"❌ **Embedding Query Error**\n\n{error_message}\n\nPlease try rephrasing your question."
    
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
            
            # ✅ FIX: Map relation names correctly for LLM query construction
            relation_to_phrase = {
                'director': 'director',
                'cast_member': 'cast member',
                'screenwriter': 'screenwriter',
                'producer': 'producer',
                'genre': 'genre',
                'publication_date': 'release date',
                'country_of_origin': 'country of origin',
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