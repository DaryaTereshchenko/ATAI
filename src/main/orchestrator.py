import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from src.main.sparql_handler import SPARQLHandler
from src.main.nl_to_sparql import NLToSPARQL
from src.main.workflow import QueryWorkflow

from src.config import (
    GRAPH_FILE_PATH, EMBEDDINGS_DIR, USE_EMBEDDINGS,
    EMBEDDING_QUERY_MODEL, EMBEDDING_ALIGNMENT_MATRIX_PATH,
    RELATION_CLASSIFIER_PATH,
    USE_LAZY_GRAPH_LOADING  # ✅ ADD
)

class QuestionType(str, Enum):
    FACTUAL = "factual"
    EMBEDDINGS = "embeddings"
    RECOMMENDATION = "recommendation"
    IMAGE = "image"
    HYBRID = "hybrid"

class QueryClassification(BaseModel):
    """Classification of a user query."""
    question_type: QuestionType = Field(
        description="The type of question: factual, embeddings, recommendation, image, or hybrid"
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence of the classification (0.0 to 1.0)"
    )

# Global instance for access by workflow
orchestrator_instance = None

class Orchestrator:
    """Routes user queries based on simple keyword matching."""

    def __init__(self, use_workflow: bool = True):
        """
        Initialize the orchestrator with rule-based classification.
        
        Args:
            use_workflow: Whether to use workflow processing
        """
        global orchestrator_instance

        # Initialize SPARQL handler with lazy loading if configured
        self.sparql_handler = SPARQLHandler(
            graph_file_path=GRAPH_FILE_PATH,
            use_lazy_loading=USE_LAZY_GRAPH_LOADING  # ✅ ADD
        )

        # Initialize NL-to-SPARQL
        self.nl_to_sparql = NLToSPARQL(
            method="direct-llm",
            sparql_handler=self.sparql_handler
        )

        # Initialize embedding processor
        self.embedding_processor = None
        if USE_EMBEDDINGS:
            try:
                print("\n🔢 Initializing embedding processor...")
                
                # ✅ Validate embeddings directory exists
                if not os.path.exists(EMBEDDINGS_DIR):
                    print(f"⚠️  Embeddings directory not found: {EMBEDDINGS_DIR}")
                    print(f"   Skipping embedding initialization")
                else:
                    # Check for required files
                    required_files = ["entity_embeds.npy", "entity_ids.del", "relation_embeds.npy", "relation_ids.del"]
                    missing = [f for f in required_files if not os.path.exists(os.path.join(EMBEDDINGS_DIR, f))]
                    
                    if missing:
                        print(f"⚠️  Missing embedding files: {', '.join(missing)}")
                        print(f"   Skipping embedding initialization")
                    else:
                        from src.main.embedding_processor import EmbeddingQueryProcessor
                        self.embedding_processor = EmbeddingQueryProcessor(
                            embeddings_dir=EMBEDDINGS_DIR,
                            graph_path=GRAPH_FILE_PATH,
                            query_model=EMBEDDING_QUERY_MODEL,
                            alignment_matrix_path=EMBEDDING_ALIGNMENT_MATRIX_PATH,
                            use_simple_aligner=True,
                            sparql_handler=self.sparql_handler,
                            relation_classifier_path=RELATION_CLASSIFIER_PATH
                        )
                        print("✅ Embedding processor initialized\n")
                
            except FileNotFoundError as e:
                print(f"⚠️  Embedding files not found: {e}")
                print(f"   Embeddings-based queries will not be available")
            except Exception as e:
                print(f"⚠️  Failed to initialize embedding processor: {e}")
                import traceback
                traceback.print_exc()
                print(f"   Embeddings-based queries will not be available")

        # Initialize workflow
        self.use_workflow = use_workflow
        if use_workflow:
            self.workflow = QueryWorkflow(self)

        orchestrator_instance = self
        # print("✅ Orchestrator initialized (rule-based mode)\n")  # REMOVED

    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify query using simple keyword matching.
        
        Priority:
        1. Check for explicit "factual approach" or "embedding approach" keywords
        2. Check for recommendation keywords
        3. Check for image-related keywords (with movie-context filtering)
        4. Default to hybrid (use both factual and embeddings)
        
        Args:
            query: User query string
            
        Returns:
            QueryClassification with question_type and confidence
        """
        query_lower = query.lower()
        
        print(f"\n{'='*80}")
        print(f"[CLASSIFICATION] Query: {query[:60]}...")
        
        # PRIORITY 1: Explicit approach keywords
        if 'factual approach' in query_lower or 'with a factual' in query_lower:
            print(f"[CLASSIFICATION] Type: factual (explicit)")
            print(f"[CLASSIFICATION] Confidence: 100%")
            print(f"[CLASSIFICATION] Method: Keyword matching")
            print(f"{'='*80}\n")
            return QueryClassification(
                question_type=QuestionType.FACTUAL,
                confidence=1.0
            )
        
        if 'embedding approach' in query_lower or 'with an embedding' in query_lower:
            print(f"[CLASSIFICATION] Type: embeddings (explicit)")
            print(f"[CLASSIFICATION] Confidence: 100%")
            print(f"[CLASSIFICATION] Method: Keyword matching")
            print(f"{'='*80}\n")
            return QueryClassification(
                question_type=QuestionType.EMBEDDINGS,
                confidence=1.0
            )
        
        # PRIORITY 2: Recommendation keywords
        recommendation_keywords = ['recommend', 'suggest', 'what should i watch', 'similar to', 'like']
        if any(keyword in query_lower for keyword in recommendation_keywords):
            print(f"[CLASSIFICATION] Type: recommendation")
            print(f"[CLASSIFICATION] Confidence: 95%")
            print(f"[CLASSIFICATION] Method: Keyword matching")
            print(f"{'='*80}\n")
            return QueryClassification(
                question_type=QuestionType.RECOMMENDATION,
                confidence=0.95
            )
        
        # PRIORITY 3: Image keywords (with movie-context filtering)
        # ✅ FIX: More specific image patterns and exclude movie queries
        image_keywords = ['image', 'picture', 'photo', 'poster', 'screenshot', 'visual']
        image_phrases = ['show me an image', 'show me a picture', 'display the poster']
        
        # Check for movie-related keywords that indicate factual query
        movie_factual_keywords = [
            'director', 'actor', 'actress', 'release', 'genre', 'screenwriter',
            'writer', 'star', 'cast', 'rating', 'country', 'award', 'won',
            'which movie', 'what movie', 'what film', 'which film'
        ]
        
        has_movie_context = any(keyword in query_lower for keyword in movie_factual_keywords)
        has_image_phrase = any(phrase in query_lower for phrase in image_phrases)
        has_image_keyword_only = any(keyword in query_lower for keyword in image_keywords)
        
        # Only classify as image if:
        # - Has image phrase (strong signal), OR
        # - Has image keyword AND no movie factual context
        if has_image_phrase or (has_image_keyword_only and not has_movie_context):
            print(f"[CLASSIFICATION] Type: image")
            print(f"[CLASSIFICATION] Confidence: 95%")
            print(f"[CLASSIFICATION] Method: Keyword matching")
            print(f"{'='*80}\n")
            return QueryClassification(
                question_type=QuestionType.IMAGE,
                confidence=0.95
            )
        
        # DEFAULT: Hybrid (use both factual and embeddings)
        print(f"[CLASSIFICATION] Type: hybrid (no explicit approach specified)")
        print(f"[CLASSIFICATION] Confidence: 80%")
        print(f"[CLASSIFICATION] Method: Keyword matching (default fallback)")
        print(f"{'='*80}\n")
        return QueryClassification(
            question_type=QuestionType.HYBRID,
            confidence=0.8
        )

    def process_query(self, query: str) -> str:
        """Process a query using the workflow."""
        if self.use_workflow:
            return self.workflow.run(query)
        else:
            # Direct processing without workflow
            classification = self.classify_query(query)
            
            if classification.question_type == QuestionType.FACTUAL:
                return self._process_factual(query)
            elif classification.question_type == QuestionType.EMBEDDINGS:
                return self._process_embeddings(query)
            elif classification.question_type == QuestionType.HYBRID:
                return self._process_hybrid(query)
            elif classification.question_type == QuestionType.IMAGE:
                return self._process_image(query)
            elif classification.question_type == QuestionType.RECOMMENDATION:
                return self._process_recommendation(query)
    
    def _clean_query_for_processing(self, query: str) -> str:
        """
        Remove classification prefixes from query before processing.
        
        ✅ FIXED: More precise patterns to preserve "From" at start of real questions.
        
        Args:
            query: Original query with possible prefixes
            
        Returns:
            Clean query without prefixes
        """
        import re
        
        clean = query
        
        # ✅ FIXED: Only remove when it's part of the full instruction phrase
        # Remove "Please answer this question with a factual approach:"
        clean = re.sub(
            r'^please\s+answer\s+this\s+question\s+with\s+(?:a|an)\s+factual\s+approach:\s*',
            '',
            clean,
            flags=re.IGNORECASE
        )
        
        # Remove "Please answer this question with an embedding approach:"
        clean = re.sub(
            r'^please\s+answer\s+this\s+question\s+with\s+(?:a|an)\s+embedding\s+approach:\s*',
            '',
            clean,
            flags=re.IGNORECASE
        )
        
        # Remove "Please answer this question:"
        clean = re.sub(
            r'^please\s+answer\s+this\s+question:\s*',
            '',
            clean,
            flags=re.IGNORECASE
        )
        
        return clean.strip()

    def _log_pipeline_step(self, step_name: str, details: dict) -> None:
        """
        Log detailed pipeline step information.
        
        Args:
            step_name: Name of the pipeline step
            details: Dictionary of details to log
        """
        print(f"\n{'─'*60}")
        print(f"📋 {step_name}")
        print(f"{'─'*60}")
        for key, value in details.items():
            # Format value for display
            if isinstance(value, str) and len(value) > 100:
                display_value = value[:97] + "..."
            elif isinstance(value, (list, tuple)) and len(value) > 3:
                display_value = f"[{len(value)} items] {value[:3]}..."
            else:
                display_value = value
            print(f"  {key}: {display_value}")

    def _process_factual(self, query: str) -> str:
        """Process factual query using SPARQL."""
        if self.embedding_processor is None:
            return "⚠️ **Processing not available**"
        
        try:
            # ✅ Clean query before processing
            clean_query = self._clean_query_for_processing(query)
            self._log_pipeline_step("Query Cleaning", {
                "Original Query": query,
                "Cleaned Query": clean_query
            })
            
            # Call the processor with detailed logging
            print("\n🔧 Calling embedding_processor.process_hybrid_factual_query()...")
            
            # Try to capture intermediate steps if possible
            try:
                # Extract entities first
                print("\n📍 Step: Entity Extraction")
                entities = self.embedding_processor._extract_entities_from_query(clean_query)
                self._log_pipeline_step("Entity Extraction Results", {
                    "Number of Entities": len(entities),
                    "Entities": entities if entities else "None found"
                })
                
                # Determine query type
                print("\n🔍 Step: Query Pattern Analysis")
                query_pattern = self.embedding_processor._determine_query_type(clean_query)
                self._log_pipeline_step("Query Pattern Detection", {
                    "Pattern": query_pattern,
                    "Query": clean_query
                })
                
                # Generate SPARQL
                print("\n⚡ Step: SPARQL Generation")
                sparql_query = self.embedding_processor._generate_sparql_for_pattern(
                    clean_query, query_pattern, entities
                )
                self._log_pipeline_step("SPARQL Query Generated", {
                    "Query Length": len(sparql_query) if sparql_query else 0,
                    "Query Preview": sparql_query if sparql_query else "Failed to generate"
                })
                
                if sparql_query:
                    print("\n📤 Step: SPARQL Execution")
                    print(f"Full SPARQL Query:")
                    print("─" * 60)
                    print(sparql_query)
                    print("─" * 60)
                    
            except AttributeError as ae:
                print(f"⚠️  Could not access internal methods: {ae}")
            except Exception as ie:
                print(f"⚠️  Error in intermediate logging: {ie}")
            
            # Execute the full pipeline
            response = self.embedding_processor.process_hybrid_factual_query(clean_query)
            
            self._log_pipeline_step("Final Response", {
                "Response Length": len(response),
                "Success": "❌" not in response and "⚠️" not in response
            })
            
            return response
            
        except Exception as e:
            error_msg = f"❌ **Error in factual processing**: {str(e)}"
            print(f"\n{error_msg}")
            import traceback
            print("\n📋 Full Stack Trace:")
            traceback.print_exc()
            return error_msg
    
    def _process_embeddings(self, query: str) -> str:
        """Process embeddings query using embedding space."""
        if self.embedding_processor is None:
            return "⚠️ **Embeddings processing not available**"
        
        try:
            # ✅ Clean query before processing
            clean_query = self._clean_query_for_processing(query)
            self._log_pipeline_step("Query Cleaning", {
                "Original Query": query,
                "Cleaned Query": clean_query
            })
            
            print("\n🔧 Calling embedding_processor.process_embedding_query()...")
            
            # Try to capture intermediate steps
            try:
                print("\n🔢 Step: Query Embedding")
                # Note: This assumes the processor has these methods
                query_embedding = self.embedding_processor.query_encoder.encode([clean_query])[0]
                self._log_pipeline_step("Query Embedding", {
                    "Embedding Dimension": len(query_embedding),
                    "Embedding Norm": float(sum(x**2 for x in query_embedding)**0.5)
                })
                
                print("\n🔍 Step: Embedding Space Search")
                # The processor should handle this internally
                
            except Exception as ie:
                print(f"⚠️  Error in intermediate logging: {ie}")
            
            response = self.embedding_processor.process_embedding_query(clean_query)
            
            self._log_pipeline_step("Final Response", {
                "Response Length": len(response),
                "Has Entity Type": "(type:" in response
            })
            
            return response
            
        except Exception as e:
            error_msg = f"❌ **Error in embeddings processing**: {str(e)}"
            print(f"\n{error_msg}")
            import traceback
            print("\n📋 Full Stack Trace:")
            traceback.print_exc()
            return error_msg
    
    def _process_hybrid(self, query: str) -> str:
        """Process hybrid query using both factual and embeddings."""
        if self.embedding_processor is None:
            return "⚠️ **Hybrid processing not available**"
        
        try:
            # ✅ Clean query before processing
            clean_query = self._clean_query_for_processing(query)
            self._log_pipeline_step("Query Cleaning", {
                "Original Query": query,
                "Cleaned Query": clean_query
            })
            
            # Run both approaches with detailed logging
            print("\n" + "="*60)
            print("🔵 FACTUAL PIPELINE (Hybrid Mode)")
            print("="*60)
            factual_result = self._process_factual_with_logging(clean_query)
            
            print("\n" + "="*60)
            print("🟢 EMBEDDING PIPELINE (Hybrid Mode)")
            print("="*60)
            embeddings_result = self._process_embedding_with_logging(clean_query)
            
            # Combine results
            print("\n🔗 Combining results...")
            response = f"**Factual Answer:**\n{factual_result}\n\n"
            response += f"**Embeddings Answer:**\n{embeddings_result}"
            
            return response
            
        except Exception as e:
            error_msg = f"❌ **Error in hybrid processing**: {str(e)}"
            print(f"\n{error_msg}")
            import traceback
            print("\n📋 Full Stack Trace:")
            traceback.print_exc()
            return error_msg
    
    def _process_factual_with_logging(self, clean_query: str) -> str:
        """Process factual query with detailed logging (for hybrid mode)."""
        try:
            return self.embedding_processor.process_hybrid_factual_query(clean_query)
        except Exception as e:
            return f"❌ Factual processing error: {str(e)}"
    
    def _process_embedding_with_logging(self, clean_query: str) -> str:
        """Process embedding query with detailed logging (for hybrid mode)."""
        try:
            return self.embedding_processor.process_embedding_query(clean_query)
        except Exception as e:
            return f"❌ Embedding processing error: {str(e)}"

    def _process_image(self, query: str) -> str:
        """Process image query."""
        return "🖼️ **Image queries are not yet supported**\n\nPlease ask a factual or embeddings question instead."
    
    def _process_recommendation(self, query: str) -> str:
        """Process recommendation query."""
        return "💡 **Recommendation queries are not yet supported**\n\nPlease ask a factual or embeddings question instead."