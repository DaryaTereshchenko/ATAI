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
    # Try multiple import paths for LlamaCpp
    try:
        from langchain_community.llms import LlamaCpp
        LLAMACPP_AVAILABLE = True
        print("✅ Imported LlamaCpp from langchain_community.llms")
        LLAMACPP_DIRECT = False
    except ImportError:
        try:
            from langchain.llms import LlamaCpp
            LLAMACPP_AVAILABLE = True
            LLAMACPP_DIRECT = False
            print("✅ Imported LlamaCpp from langchain.llms")
        except ImportError:
            # Direct import as fallback
            from llama_cpp import Llama  # noqa: F401
            LLAMACPP_AVAILABLE = True
            LLAMACPP_DIRECT = True
            print("✅ Imported Llama directly from llama_cpp")
except ImportError as e:
    LLAMACPP_AVAILABLE = False
    LLAMACPP_DIRECT = False
    print(f"⚠️  llama-cpp-python not available: {e}")
    print("   Install langchain-community with: pip install langchain-community")

from src.main.sparql_handler import SPARQLHandler
from src.main.nl_to_sparql import NLToSPARQL
from src.main.workflow import QueryWorkflow
from src.config import (
    LLM_TYPE, LLM_MODEL, LLM_MODEL_PATH, LLM_TEMPERATURE,
    LLM_MAX_TOKENS, LLM_CONTEXT_LENGTH, USE_LLM_CLASSIFICATION,
    # Optional: flip this on if you want a language guard on labels by default
    # ADD_LABEL_LANG_FILTER, LABEL_LANG
)

class QuestionType(str, Enum):
    FACTUAL = "factual"
    EMBEDDING = "embedding"
    MULTIMEDIA = "multimedia"
    RECOMMENDATION = "recommendation"

class QueryClassification(BaseModel):
    """Classification of a user query into one of four types."""
    question_type: QuestionType = Field(
        description="The type of question: factual, embedding, multimedia, or recommendation"
    )

# Global instance for access by workflow
orchestrator_instance = None

class Orchestrator:
    """Routes user queries to appropriate processing nodes based on question type."""

    def __init__(self, llm=None, use_workflow: bool = True):
        """Initialize the orchestrator with a language model."""
        global orchestrator_instance

        # Initialize LLM based on configuration (used only for classification here)
        self.llm = llm or self._initialize_llm()
        self.use_llm = self.llm is not None and USE_LLM_CLASSIFICATION

        if self.use_llm:
            self.parser = PydanticOutputParser(pydantic_object=QueryClassification)
            self._setup_classifier()
        else:
            print("ℹ️  Using rule-based classification.")

        # Initialize SPARQL handler ONCE
        self.sparql_handler = SPARQLHandler()

        # ✅ Instantiate NL-to-SPARQL in MODEL-FIRST mode and reuse the same handler
        #    This ensures DeepSeek (llama-cpp) is tried first, with rules as fallback.
        self.nl_to_sparql = NLToSPARQL(
            method="direct-llm",
            sparql_handler=self.sparql_handler
        )

        # Initialize workflow system
        self.use_workflow = use_workflow
        if use_workflow:
            self.workflow = QueryWorkflow(self)

        orchestrator_instance = self

    def _initialize_llm(self):
        """Initialize LLM based on configuration (for classification only)."""
        try:
            if LLM_TYPE == "gguf" and LLAMACPP_AVAILABLE:
                print(f"📥 Loading GGUF model: {LLM_MODEL}")
                print(f"    Path: {LLM_MODEL_PATH}")

                if not os.path.exists(LLM_MODEL_PATH):
                    print(f"❌ Model file not found: {LLM_MODEL_PATH}")
                    print("   Please update LLM_MODEL_PATH in src/config.py")
                    return None

                if not LLAMACPP_DIRECT:
                    try:
                        llm = LlamaCpp(
                            model_path=LLM_MODEL_PATH,
                            temperature=LLM_TEMPERATURE,
                            max_tokens=LLM_MAX_TOKENS,
                            n_ctx=LLM_CONTEXT_LENGTH,
                            n_batch=512,
                            verbose=False,
                            n_threads=4,
                        )
                        print("✅ GGUF model loaded successfully via LangChain wrapper")
                        return llm
                    except Exception as e:
                        print(f"⚠️  LangChain wrapper failed: {e}")
                        print("   Trying direct llama-cpp-python...")

                # Fallback: thin wrapper to mimic LangChain interface
                try:
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

                    llm = LlamaCppWrapper(
                        model_path=LLM_MODEL_PATH,
                        temperature=LLM_TEMPERATURE,
                        max_tokens=LLM_MAX_TOKENS,
                        n_ctx=LLM_CONTEXT_LENGTH,
                        n_threads=4
                    )
                    print("✅ GGUF model loaded successfully via direct llama-cpp-python")
                    return llm

                except Exception as e:
                    print(f"❌ Failed to load GGUF model directly: {e}")
                    return None

            elif LLM_TYPE == "ollama" and OLLAMA_AVAILABLE:
                print(f"📥 Connecting to Ollama model: {LLM_MODEL}")
                llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
                print("✅ Ollama model connected")
                return llm

            elif LLM_TYPE == "none":
                print("ℹ️  LLM disabled in config (LLM_TYPE='none')")
                return None

            else:
                print(f"⚠️  LLM type '{LLM_TYPE}' not available or libraries not installed")
                if LLM_TYPE == "gguf":
                    print("   Make sure langchain-community is installed: pip install langchain-community")
                return None

        except Exception as e:
            print(f"❌ Error initializing LLM: {e}")
            print("   Falling back to rule-based classification")
            import traceback; traceback.print_exc()
            return None

    def _setup_classifier(self):
        """Set up the classification prompt and chain."""
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query classifier for a movie information system.

Classify the user's query into exactly ONE of these types:
- factual: Questions about specific facts (who, what, when, where)
- embedding: Semantic search or similarity questions  
- multimedia: Questions asking to see/show images
- recommendation: Asking for suggestions or similar items

Examples:
- "Who directed Star Wars?" → factual
- "Find movies similar to Inception" → embedding
- "Show me a picture of Tom Hanks" → multimedia
- "Recommend action movies" → recommendation

Respond with ONLY the classification type, nothing else."""),
            ("user", "{query}")
        ])
        self.classification_chain = classification_prompt | self.llm

    def classify_query(self, query: str) -> QueryClassification:
        """Classify a user query into one of the four types."""
        if self.use_llm:
            try:
                raw_output = self.classification_chain.invoke({"query": query})
                output_text = raw_output if isinstance(raw_output, str) else str(raw_output)
                print(f"[Classification] Raw LLM output: {output_text[:100]}...")
                
                # Clean up the output
                import re
                output_text = output_text.strip()
                output_text = re.sub(r'^(AI:|Assistant:)\s*', '', output_text, flags=re.IGNORECASE)
                output_text = output_text.strip('"\'')
                
                # Extract just the classification type
                type_match = re.search(r'\b(factual|embedding|multimedia|recommendation)\b', output_text.lower())
                if type_match:
                    qt = type_match.group(1)
                    print(f"[Classification] ✅ Parsed type: {qt}")
                    return QueryClassification(question_type=QuestionType(qt))
                
                # Fallback: check for keywords in output
                output_lower = output_text.lower()
                if 'factual' in output_lower:
                    return QueryClassification(question_type=QuestionType.FACTUAL)
                elif 'embedding' in output_lower:
                    return QueryClassification(question_type=QuestionType.EMBEDDING)
                elif 'multimedia' in output_lower:
                    return QueryClassification(question_type=QuestionType.MULTIMEDIA)
                elif 'recommendation' in output_lower:
                    return QueryClassification(question_type=QuestionType.RECOMMENDATION)
                
                raise ValueError(f"Could not extract classification from: {output_text}")
            except Exception as e:
                print(f"⚠️  LLM classification failed: {str(e)[:200]}")
                print("ℹ️  Falling back to rule-based classification.")
                return self._rule_based_classify(query)
        else:
            return self._rule_based_classify(query)

    def _rule_based_classify(self, query: str) -> QueryClassification:
        q = query.lower()
        multimedia_keywords = ['show', 'picture', 'image', 'photo', 'display', 'look like', 'see', 'view']
        if any(k in q for k in multimedia_keywords):
            return QueryClassification(question_type=QuestionType.MULTIMEDIA)
        recommendation_keywords = ['recommend', 'suggest', 'similar', 'like', 'what should i watch']
        if any(k in q for k in recommendation_keywords):
            return QueryClassification(question_type=QuestionType.RECOMMENDATION)
        factual_keywords = ['who', 'what', 'when', 'where', 'which', 'director', 'actor', 'release']
        if any(k in q for k in factual_keywords):
            return QueryClassification(question_type=QuestionType.FACTUAL)
        return QueryClassification(question_type=QuestionType.FACTUAL)

    def process_query(self, query: str) -> str:
        """Process a query by routing it through the workflow or directly to handlers."""
        if self.use_workflow:
            return self.workflow.run(query)
        else:
            return self._process_query_legacy(query)

    def _process_query_legacy(self, query: str) -> str:
        classification = self.classify_query(query)
        print(f"Query classified as: {classification.question_type.value}")
        if classification.question_type == QuestionType.FACTUAL:
            return self._handle_factual(query)
        elif classification.question_type == QuestionType.EMBEDDING:
            return self._handle_embedding(query)
        elif classification.question_type == QuestionType.MULTIMEDIA:
            return self._handle_multimedia(query)
        elif classification.question_type == QuestionType.RECOMMENDATION:
            return self._handle_recommendation(query)
        else:
            return "I'm not sure how to handle that question."

    def _handle_factual(self, query: str) -> str:
        """Handle factual questions using the knowledge graph."""
        try:
            if not self.nl_to_sparql.validate_question(query):
                return ("I'm sorry, I couldn't understand your question. "
                        "Please ask about movies, actors, directors, release dates, genres, or ratings.")

            print("Converting natural language to SPARQL (model-first)...")
            sparql_result = self.nl_to_sparql.convert(query)

            print(f"Generated SPARQL (confidence: {getattr(sparql_result, 'confidence', 0.0)}):")
            print(sparql_result.query)
            if getattr(sparql_result, 'explanation', None):
                print(f"Explanation: {sparql_result.explanation}")

            # Confidence hint (optional UX)
            if getattr(sparql_result, 'confidence', 1.0) < 0.5:
                return (
                    f"I'm not very confident about this query (confidence: {sparql_result.confidence:.2f}).\n\n"
                    f"Generated query:\n{sparql_result.query}\n\n"
                    "Would you like to rephrase your question?"
                )

            # Optionally add a language guard to labels (uncomment if needed)
            # if ADD_LABEL_LANG_FILTER:
            #     sparql_result.query = self.sparql_handler.add_lang_filter(
            #         sparql_result.query, ["?movieLabel", "?personLabel"], LABEL_LANG
            #     )

            print("Executing SPARQL query...")
            exec_result = self.sparql_handler.execute_query(sparql_result.query, validate=True)

            if not exec_result.get('success'):
                return (
                    f"I couldn't find an answer.\n\n"
                    f"Generated query:\n{sparql_result.query}\n\n"
                    f"Result: {exec_result.get('error', 'Unknown error')}\n\n"
                    "The information might not be available in the knowledge graph, or the query may need adjustment."
                )

            data = exec_result.get('data') or "No answer found in the database."
            return data

        except Exception as e:
            return (
                f"Error processing factual question: {e}\n\n"
                f"Original question: {query}\n\n"
                "Please try rephrasing your question."
            )

    def _handle_embedding(self, query: str) -> str:
        return f"[EMBEDDING NODE - Not yet implemented] Processing: {query}"

    def _handle_multimedia(self, query: str) -> str:
        return f"[MULTIMEDIA NODE - Not yet implemented] Processing: {query}"

    def _handle_recommendation(self, query: str) -> str:
        return f"[RECOMMENDATION NODE - Not yet implemented] Processing: {query}"
