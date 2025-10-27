import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import re  # ✅ Add missing import

# Try to import LLM libraries
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  langchain_ollama not available")

try:
    from langchain_community.llms import LlamaCpp
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False

from src.main.sparql_handler import SPARQLHandler
from src.main.nl_to_sparql import NLToSPARQL
from src.main.workflow import QueryWorkflow

# ✅ NEW: Import transformer classifier
try:
    from src.main.transformer_classifier import TransformerQueryClassifier
    TRANSFORMER_CLASSIFIER_AVAILABLE = True
except ImportError:
    TRANSFORMER_CLASSIFIER_AVAILABLE = False
    print("⚠️  TransformerQueryClassifier not available")

from src.config import (
    LLM_TYPE, LLM_MODEL, LLM_MODEL_PATH, LLM_TEMPERATURE,
    LLM_MAX_TOKENS, LLM_CONTEXT_LENGTH, USE_LLM_CLASSIFICATION,
    GRAPH_FILE_PATH, EMBEDDINGS_DIR, USE_EMBEDDINGS,
    EMBEDDING_QUERY_MODEL, EMBEDDING_ALIGNMENT_MATRIX_PATH,
    TRANSFORMER_MODEL_PATH
)

class QuestionType(str, Enum):
    FACTUAL = "factual"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"  # ✅ NEW: both factual + embedding
    IMAGE = "image"     # ✅ NEW: renamed from multimedia
    RECOMMENDATION = "recommendation"

class QueryClassification(BaseModel):
    """Classification of a user query."""
    question_type: QuestionType = Field(
        description="The type of question: factual, multimedia, recommendation, or out_of_scope"
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence of the classification (0.0 to 1.0)"
    )

# Global instance for access by workflow
orchestrator_instance = None

class Orchestrator:
    """Routes user queries to appropriate processing nodes based on question type."""

    def __init__(
        self,
        llm=None,
        use_workflow: bool = True,
        use_transformer_classifier: bool = False,  # ✅ CHANGED: Default to False
        transformer_model_path: str = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            llm: Optional LLM instance
            use_workflow: Whether to use workflow processing
            use_transformer_classifier: Whether to use fine-tuned transformer classifier (deprecated)
            transformer_model_path: Path to fine-tuned model (deprecated)
        """
        global orchestrator_instance

        # ✅ SIMPLIFIED: No transformer classifier initialization
        print("\n🔧 Initializing orchestrator with keyword-based classification...")
        
        # Initialize SPARQL handler
        self.sparql_handler = SPARQLHandler()

        # Initialize NL-to-SPARQL
        self.nl_to_sparql = NLToSPARQL(
            method="direct-llm",
            sparql_handler=self.sparql_handler
        )

        # Initialize embedding processor
        self.embedding_processor = None
        if USE_EMBEDDINGS:
            try:
                print("\n🔢 Initializing embedding processor (hybrid approach)...")
                from src.main.embedding_processor import EmbeddingQueryProcessor
                self.embedding_processor = EmbeddingQueryProcessor(
                    embeddings_dir=EMBEDDINGS_DIR,
                    graph_path=GRAPH_FILE_PATH,
                    query_model=EMBEDDING_QUERY_MODEL,
                    alignment_matrix_path=EMBEDDING_ALIGNMENT_MATRIX_PATH,
                    use_simple_aligner=True,
                    sparql_handler=self.sparql_handler
                )
                print("✅ Embedding processor initialized (hybrid mode ready)\n")
            except Exception as e:
                print(f"⚠️  Failed to initialize embedding processor: {e}")
                import traceback
                traceback.print_exc()

        # Initialize workflow
        self.use_workflow = use_workflow
        if use_workflow:
            self.workflow = QueryWorkflow(self)

        orchestrator_instance = self

    def _initialize_llm(self):
        """Initialize LLM (fallback for classification)."""
        # ...existing LLM initialization code...
        try:
            if LLM_TYPE == "gguf" and LLAMACPP_AVAILABLE:
                print(f"📥 Loading GGUF model: {LLM_MODEL}")
                from llama_cpp import Llama
                
                class LlamaCppWrapper:
                    def __init__(self, model_path, **kwargs):
                        self.llm = Llama(
                            model_path=model_path,
                            n_ctx=kwargs.get('n_ctx', LLM_CONTEXT_LENGTH),
                            n_threads=kwargs.get('n_threads', 4),
                            verbose=False
                        )
                        self.temperature = kwargs.get('temperature', LLM_TEMPERATURE)
                        self.max_tokens = kwargs.get('max_tokens', LLM_MAX_TOKENS)

                    def __call__(self, prompt, **kwargs):
                        result = self.llm(
                            prompt,
                            max_tokens=kwargs.get('max_tokens', self.max_tokens),
                            temperature=kwargs.get('temperature', self.temperature),
                            stop=kwargs.get('stop', []),
                        )
                        return result['choices'][0]['text']

                    def invoke(self, inputs, **kwargs):
                        prompt = inputs.get('input', str(inputs)) if isinstance(inputs, dict) else str(inputs)
                        return self(prompt, **kwargs)

                return LlamaCppWrapper(
                    model_path=LLM_MODEL_PATH,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                    n_ctx=LLM_CONTEXT_LENGTH,
                    n_threads=4
                )
            elif LLM_TYPE == "ollama" and OLLAMA_AVAILABLE:
                return ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
            else:
                return None
        except Exception as e:
            print(f"❌ Error initializing LLM: {e}")
            return None

    def _setup_classifier(self):
        """Set up LLM classification (fallback)."""
        # ...existing LLM classifier setup...
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query classifier for a movie information system.
Classify into: factual, multimedia, recommendation, or out_of_scope."""),
            ("user", "{query}")
        ])
        self.classification_chain = classification_prompt | self.llm

    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify a user query using SIMPLE KEYWORD-BASED detection.
        
        5 classes:
        - factual: word "factual" present
        - embedding: word "embedding(s)" present
        - image: words related to images (show, picture, image, photo, display)
        - recommendation: words related to recommendations (recommend, suggest, "should i watch")
        - hybrid: none of the above (default) → use BOTH factual + embedding
        
        Priority (first match wins):
        1. Check for "factual" keyword
        2. Check for "embedding(s)" keyword
        3. Check for image keywords
        4. Check for recommendation keywords
        5. Default to hybrid (both approaches)
        """
        query_lower = query.lower()
        words = set(re.findall(r'\b\w+\b', query_lower))
        
        print(f"\n{'='*80}")
        print(f"[CLASSIFICATION] Query: {query[:60]}...")
        
        # 1. Check for "factual"
        if 'factual' in words:
            print(f"[CLASSIFICATION] Type: FACTUAL (keyword detected)")
            print(f"[CLASSIFICATION] Method: Keyword-based")
            print(f"{'='*80}\n")
            return QueryClassification(
                question_type=QuestionType.FACTUAL,
                confidence=1.0
            )
        
        # 2. Check for "embedding(s)"
        if {'embedding', 'embeddings'} & words:
            print(f"[CLASSIFICATION] Type: EMBEDDING (keyword detected)")
            print(f"[CLASSIFICATION] Method: Keyword-based")
            print(f"{'='*80}\n")
            return QueryClassification(
                question_type=QuestionType.EMBEDDING,
                confidence=1.0
            )
        
        # 3. Check for image keywords
        image_keywords = {'show', 'picture', 'image', 'photo', 'display', 'visualize'}
        if image_keywords & words:
            # Additional check: must have context suggesting images
            if any(word in query_lower for word in ['picture', 'image', 'photo']):
                print(f"[CLASSIFICATION] Type: IMAGE (keywords detected)")
                print(f"[CLASSIFICATION] Method: Keyword-based")
                print(f"{'='*80}\n")
                return QueryClassification(
                    question_type=QuestionType.IMAGE,
                    confidence=1.0
                )
        
        # 4. Check for recommendation keywords
        recommendation_keywords = {'recommend', 'suggestion', 'suggest', 'should i watch', 'what to watch'}
        if recommendation_keywords & words or any(phrase in query_lower for phrase in ['should i watch', 'what to watch']):
            print(f"[CLASSIFICATION] Type: RECOMMENDATION (keywords detected)")
            print(f"[CLASSIFICATION] Method: Keyword-based")
            print(f"{'='*80}\n")
            return QueryClassification(
                question_type=QuestionType.RECOMMENDATION,
                confidence=1.0
            )
        
        # 5. Default: HYBRID (both factual + embedding)
        print(f"[CLASSIFICATION] Type: HYBRID (default - will use both approaches)")
        print(f"[CLASSIFICATION] Method: Keyword-based (no specific keyword found)")
        print(f"{'='*80}\n")
        return QueryClassification(
            question_type=QuestionType.HYBRID,
            confidence=1.0
        )

    def process_query(self, query: str) -> str:
        """Process a query using the workflow."""
        if self.use_workflow:
            return self.workflow.run(query)
        else:
            return self._process_hybrid(query)
    
    def _process_hybrid(self, query: str) -> str:
        """Process query with hybrid approach."""
        if self.embedding_processor is None:
            return (
                "⚠️ **Hybrid processing not available**\n\n"
                "The embedding processor is not initialized."
            )
        
        try:
            return self.embedding_processor.process_hybrid_factual_query(query)
        except Exception as e:
            return f"⚠️ **Error in hybrid processing**: {str(e)}"