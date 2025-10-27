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
    MULTIMEDIA = "multimedia"
    RECOMMENDATION = "recommendation"
    OUT_OF_SCOPE = "out_of_scope"  # ✅ NEW: For rejection

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
        use_transformer_classifier: bool = True,
        transformer_model_path: str = None  # Changed to None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            llm: Optional LLM instance
            use_workflow: Whether to use workflow processing
            use_transformer_classifier: Whether to use fine-tuned transformer classifier
            transformer_model_path: Path to fine-tuned model (defaults to config value)
        """
        global orchestrator_instance

        # Use config path if not specified
        if transformer_model_path is None:
            transformer_model_path = TRANSFORMER_MODEL_PATH

        # ✅ NEW: Initialize classifier (transformer or LLM)
        self.use_transformer = use_transformer_classifier and TRANSFORMER_CLASSIFIER_AVAILABLE
        self.transformer_classifier = None
        
        if self.use_transformer:
            try:
                print("\n🤖 Initializing fine-tuned transformer classifier...")
                self.transformer_classifier = TransformerQueryClassifier(
                    model_path=transformer_model_path,
                    confidence_threshold=0.5
                )
                print("✅ Transformer classifier initialized\n")
            except Exception as e:
                print(f"⚠️  Failed to load transformer classifier: {e}")
                print("   Falling back to LLM/rule-based classification\n")
                self.use_transformer = False
        
        # Initialize LLM (fallback if transformer not available)
        if not self.use_transformer:
            self.llm = llm or self._initialize_llm()
            self.use_llm = self.llm is not None and USE_LLM_CLASSIFICATION

            if self.use_llm:
                self.parser = PydanticOutputParser(pydantic_object=QueryClassification)
                self._setup_classifier()
                print("ℹ️  Using LLM-based classification.")
            else:
                print("ℹ️  Using rule-based classification.")

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
        Classify a user query using the configured classifier.
        
        Priority:
        1. Fine-tuned transformer (most accurate)
        2. LLM-based classification (fallback)
        3. Rule-based classification (last resort)
        """
        # ✅ PRIMARY: Use fine-tuned transformer
        if self.use_transformer and self.transformer_classifier:
            try:
                result = self.transformer_classifier.classify(query)
                
                print(f"\n{'='*80}")
                print(f"[CLASSIFICATION] Query: {query[:60]}...")
                print(f"[CLASSIFICATION] Type: {result.question_type}")
                print(f"[CLASSIFICATION] Confidence: {result.confidence:.2%}")
                print(f"[CLASSIFICATION] Method: Fine-tuned Transformer")
                print(f"{'='*80}\n")
                
                return QueryClassification(
                    question_type=QuestionType(result.question_type),
                    confidence=result.confidence
                )
            except Exception as e:
                print(f"⚠️  Transformer classification failed: {e}")
                print("   Falling back to alternative method...")
        
        # FALLBACK 1: LLM-based classification
        if self.use_llm:
            try:
                raw_output = self.classification_chain.invoke({"query": query})
                output_text = raw_output if isinstance(raw_output, str) else str(raw_output)
                
                import re
                type_match = re.search(
                    r'\b(factual|multimedia|recommendation|out_of_scope)\b',
                    output_text.lower()
                )
                if type_match:
                    qt = type_match.group(1)
                    return QueryClassification(
                        question_type=QuestionType(qt),
                        confidence=0.8
                    )
            except Exception as e:
                print(f"⚠️  LLM classification failed: {e}")
        
        # FALLBACK 2: Rule-based classification
        print("ℹ️  Using rule-based classification")
        return self._rule_based_classify(query)

    def _rule_based_classify(self, query: str) -> QueryClassification:
        """Rule-based classification (last resort)."""
        q = query.lower()
        
        # Check for out-of-scope patterns (simple heuristics)
        out_of_scope_keywords = [
            'weather', 'temperature', 'forecast',
            'calculate', 'math', 'solve',
            'code', 'program', 'function',
            'capital', 'population', 'president',
            'recipe', 'cook', 'bake'
        ]
        if any(keyword in q for keyword in out_of_scope_keywords):
            return QueryClassification(
                question_type=QuestionType.OUT_OF_SCOPE,
                confidence=0.7
            )
        
        # Multimedia detection
        multimedia_keywords = ['show', 'picture', 'image', 'photo', 'display']
        if any(k in q for k in multimedia_keywords) and ('picture' in q or 'image' in q):
            return QueryClassification(
                question_type=QuestionType.MULTIMEDIA,
                confidence=0.8
            )
        
        # Recommendation detection
        recommendation_keywords = ['recommend', 'suggest', 'what should i watch']
        if any(k in q for k in recommendation_keywords):
            return QueryClassification(
                question_type=QuestionType.RECOMMENDATION,
                confidence=0.8
            )
        
        # Default: factual
        return QueryClassification(
            question_type=QuestionType.FACTUAL,
            confidence=0.6
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