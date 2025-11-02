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
from src.main.answer_formatter import AnswerFormatter
from src.main.query_cache import QueryCache  # ✅ NEW

from src.config import (
    GRAPH_FILE_PATH, EMBEDDINGS_DIR, USE_EMBEDDINGS,
    EMBEDDING_QUERY_MODEL, EMBEDDING_ALIGNMENT_MATRIX_PATH
)

class QuestionType(str, Enum):
    FACTUAL = "factual"
    EMBEDDINGS = "embeddings"
    RECOMMENDATION = "recommendation"
    IMAGE = "image"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"  # ✅ NEW: For unclassified queries

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

        print("\n🔧 Initializing Orchestrator with rule-based classification...")

        # ✅ NEW: Initialize query cache
        print("\n📦 Initializing query cache...")
        self.query_cache = QueryCache()
        cache_stats = self.query_cache.get_stats()
        print(f"   Cache entries: {cache_stats['total_entries']}")
        print(f"   Cache file: {cache_stats['cache_file']}")

        # Initialize SPARQL handler
        self.sparql_handler = SPARQLHandler()
        
        # ✅ NEW: Initialize RelationManager
        print("\n🔗 Initializing Relation Manager...")
        from src.main.relation_manager import RelationManager
        self.relation_manager = RelationManager(self.sparql_handler.graph)
        
        # Initialize NL-to-SPARQL
        self.nl_to_sparql = NLToSPARQL(
            method="direct-llm",
            sparql_handler=self.sparql_handler
        )

        # Initialize embedding processor
        self.embedding_processor = None
        
        # ✅ FIXED: Always try to initialize if embeddings are enabled
        if USE_EMBEDDINGS:
            print("\n🔢 Initializing embedding processor...")
            try:
                from src.main.embedding_processor import EmbeddingQueryProcessor
                self.embedding_processor = EmbeddingQueryProcessor(
                    embeddings_dir=EMBEDDINGS_DIR,
                    graph_path=GRAPH_FILE_PATH,
                    query_model=EMBEDDING_QUERY_MODEL,
                    alignment_matrix_path=EMBEDDING_ALIGNMENT_MATRIX_PATH,
                    use_simple_aligner=True,
                    sparql_handler=self.sparql_handler,
                    relation_manager=self.relation_manager  # ✅ FIX: Pass relation manager here
                )
                print("✅ Embedding processor initialized successfully\n")
            except FileNotFoundError as e:
                print(f"⚠️  Embedding files not found: {e}")
                print(f"   Embeddings directory: {EMBEDDINGS_DIR}")
                print(f"   Graph path: {GRAPH_FILE_PATH}")
                print("   Embedding processor will not be available.\n")
                self.embedding_processor = None
            except Exception as e:
                print(f"⚠️  Failed to initialize embedding processor: {e}")
                print("   Embedding processor will not be available.\n")
                import traceback
                traceback.print_exc()
                self.embedding_processor = None
        else:
            print("ℹ️  Embeddings disabled in config (USE_EMBEDDINGS=False)\n")

        # Initialize workflow
        self.use_workflow = use_workflow
        if use_workflow:
            self.workflow = QueryWorkflow(self)

        orchestrator_instance = self
        print("✅ Orchestrator initialized (rule-based mode)\n")

    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify query using simple keyword matching.
        
        Priority:
        1. Check for explicit "factual approach" or "embedding approach" keywords
        2. Check for "Please answer this question:" (hybrid)
        3. Check for recommendation keywords
        4. Check for image-related keywords (with movie-context filtering)
        5. Check for movie factual keywords (director, actor, etc.)
        6. Default to unknown (polite rejection)
        
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
        
        # PRIORITY 2: Hybrid - must start with "Please answer this question:"
        if query_lower.startswith('please answer this question:'):
            print(f"[CLASSIFICATION] Type: hybrid (explicit)")
            print(f"[CLASSIFICATION] Confidence: 100%")
            print(f"[CLASSIFICATION] Method: Keyword matching")
            print(f"{'='*80}\n")
            return QueryClassification(
                question_type=QuestionType.HYBRID,
                confidence=1.0
            )
        
        # PRIORITY 3: Recommendation keywords
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
        
        # PRIORITY 4: Image keywords (with movie-context filtering)
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
        
        # PRIORITY 5: Check for movie factual keywords (queries we can handle)
        if has_movie_context:
            print(f"[CLASSIFICATION] Type: factual (movie context detected)")
            print(f"[CLASSIFICATION] Confidence: 85%")
            print(f"[CLASSIFICATION] Method: Keyword matching")
            print(f"{'='*80}\n")
            return QueryClassification(
                question_type=QuestionType.FACTUAL,
                confidence=0.85
            )
        
        # DEFAULT: Unknown (polite rejection)
        print(f"[CLASSIFICATION] Type: unknown (no recognizable query pattern)")
        print(f"[CLASSIFICATION] Confidence: 100%")
        print(f"[CLASSIFICATION] Method: Default fallback")
        print(f"{'='*80}\n")
        return QueryClassification(
            question_type=QuestionType.UNKNOWN,
            confidence=1.0
        )

    def process_query(self, query: str) -> str:
        """Process a query using the workflow."""
        # ✅ NEW: Check cache first
        cached_response = self.query_cache.get(query)
        if cached_response is not None:
            print("\n🎯 Returning cached response (no processing needed)\n")
            return cached_response
        
        # Process query normally
        if self.use_workflow:
            response = self.workflow.run(query)
        else:
            # Direct processing without workflow
            classification = self.classify_query(query)
            
            if classification.question_type == QuestionType.FACTUAL:
                response = self._process_factual(query)
            elif classification.question_type == QuestionType.EMBEDDINGS:
                response = self._process_embeddings(query)
            elif classification.question_type == QuestionType.HYBRID:
                response = self._process_hybrid(query)
            elif classification.question_type == QuestionType.IMAGE:
                response = self._process_image(query)
            elif classification.question_type == QuestionType.RECOMMENDATION:
                response = self._process_recommendation(query)
            elif classification.question_type == QuestionType.UNKNOWN:
                response = "🤖 **I'm sorry, I don't understand the question.**\n\n" \
                           "Please ask a factual question or request embeddings."
        
        # ✅ NEW: Cache successful responses (avoid caching errors)
        if response and not response.startswith("❌") and not response.startswith("⚠️"):
            self.query_cache.set(query, response)
        
        return response
    
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
            # Query is already cleaned by workflow
            self._log_pipeline_step("Processing Factual Query", {
                "Query": query,
                "Query Length": len(query)
            })
            
            # Call the processor with detailed logging
            print("\n🔧 Calling embedding_processor.process_hybrid_factual_query()...")
            
            # Execute the full pipeline
            response = self.embedding_processor.process_hybrid_factual_query(query)
            
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
            # Query is already cleaned by workflow
            self._log_pipeline_step("Processing Embeddings Query", {
                "Query": query,
                "Query Length": len(query)
            })
            
            print("\n🔧 Calling embedding_processor.process_embedding_query()...")
            
            # Try to capture intermediate steps
            try:
                print("\n🔢 Step: Query Embedding")
                # ✅ FIXED: Use correct attribute name
                if hasattr(self.embedding_processor, 'query_embedder'):
                    query_embedding = self.embedding_processor.query_embedder.embed_query(query)
                    self._log_pipeline_step("Query Embedding", {
                        "Embedding Dimension": len(query_embedding),
                        "Embedding Norm": float(sum(x**2 for x in query_embedding)**0.5)
                    })
                
                print("\n🔍 Step: Embedding Space Search")
                # The processor should handle this internally
                
            except Exception as ie:
                print(f"⚠️  Error in intermediate logging: {ie}")
            
            response = self.embedding_processor.process_embedding_query(query)
            
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
            # Query is already cleaned by workflow
            self._log_pipeline_step("Processing Hybrid Query", {
                "Query": query,
                "Query Length": len(query)
            })
            
            # Run both approaches
            print("\n" + "="*60)
            print("🔵 FACTUAL PIPELINE (Hybrid Mode)")
            print("="*60)
            factual_result = self._process_factual_with_logging(query)
            
            print("\n" + "="*60)
            print("🟢 EMBEDDING PIPELINE (Hybrid Mode)")
            print("="*60)
            embeddings_result = self._process_embedding_with_logging(query)
            
            # Use AnswerFormatter to combine results
            print("\n🔗 Combining results...")
            response = AnswerFormatter.format_hybrid_response(
                factual=factual_result,
                embedding=embeddings_result
            )
            
            return response
            
        except Exception as e:
            error_msg = f"❌ **Error in hybrid processing**: {str(e)}"
            print(f"\n{error_msg}")
            import traceback
            print("\n📋 Full Stack Trace:")
            traceback.print_exc()
            return error_msg
    
    def _process_factual_with_logging(self, query: str) -> str:
        """Process factual query with detailed logging (for hybrid mode)."""
        try:
            return self.embedding_processor.process_hybrid_factual_query(query)
        except Exception as e:
            return f"❌ Factual processing error: {str(e)}"
    
    def _process_embedding_with_logging(self, query: str) -> str:
        """Process embedding query with detailed logging (for hybrid mode)."""
        try:
            return self.embedding_processor.process_embedding_query(query)
        except Exception as e:
            return f"❌ Embedding processing error: {str(e)}"

    def _process_image(self, query: str) -> str:
        """Process image query."""
        return "🖼️ **Image queries are not yet supported**\n\nPlease ask a factual or embeddings question instead."
    
    def _process_recommendation(self, query: str) -> str:
        """Process recommendation query."""
        return "💡 **Recommendation queries are not yet supported**\n\nPlease ask a factual or embeddings question instead."