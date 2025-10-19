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
    except ImportError:
        try:
            from langchain.llms import LlamaCpp
            LLAMACPP_AVAILABLE = True
            print("✅ Imported LlamaCpp from langchain.llms")
        except ImportError:
            # Direct import as fallback
            from llama_cpp import Llama
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
    LLM_MAX_TOKENS, LLM_CONTEXT_LENGTH, USE_LLM_CLASSIFICATION
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

class Orchestrator:
    """Routes user queries to appropriate processing nodes based on question type."""
    
    def __init__(self, llm=None, use_workflow: bool = True):
        """Initialize the orchestrator with a language model."""
        # Initialize LLM based on configuration
        self.llm = llm or self._initialize_llm()
        self.use_llm = self.llm is not None and USE_LLM_CLASSIFICATION
        
        if self.use_llm:
            self.parser = PydanticOutputParser(pydantic_object=QueryClassification)
            self._setup_classifier()
        else:
            print("⚠️  LLM not available. Using rule-based classification.")
            self.parser = None
        
        # Initialize SPARQL handler ONCE
        self.sparql_handler = SPARQLHandler()
        
        # Initialize NL-to-SPARQL converter (shares the SPARQLHandler to avoid loading graph twice)
        self.nl_to_sparql = NLToSPARQL(sparql_handler=self.sparql_handler)
        
        # Initialize workflow system
        self.use_workflow = use_workflow
        if use_workflow:
            self.workflow = QueryWorkflow(self)
    
    def _initialize_llm(self):
        """Initialize LLM based on configuration."""
        try:
            if LLM_TYPE == "gguf" and LLAMACPP_AVAILABLE:
                print(f"📥 Loading GGUF model: {LLM_MODEL}")
                print(f"    Path: {LLM_MODEL_PATH}")
                
                # Check if model file exists
                if not os.path.exists(LLM_MODEL_PATH):
                    print(f"❌ Model file not found: {LLM_MODEL_PATH}")
                    print("   Please update LLM_MODEL_PATH in src/config.py")
                    return None
                
                # Try using LangChain wrapper first
                if not hasattr(self, 'LLAMACPP_DIRECT') or not LLAMACPP_DIRECT:
                    try:
                        # Load GGUF model using LlamaCpp from langchain
                        llm = LlamaCpp(
                            model_path=LLM_MODEL_PATH,
                            temperature=LLM_TEMPERATURE,
                            max_tokens=LLM_MAX_TOKENS,
                            n_ctx=LLM_CONTEXT_LENGTH,
                            n_batch=512,
                            verbose=False,
                            n_threads=4,  # Adjust based on your CPU
                        )
                        print(f"✅ GGUF model loaded successfully via LangChain wrapper")
                        return llm
                    except Exception as e:
                        print(f"⚠️  LangChain wrapper failed: {e}")
                        print("   Trying direct llama-cpp-python...")
                
                # Fallback to direct llama-cpp-python
                try:
                    from llama_cpp import Llama
                    
                    # Create a simple wrapper to match LangChain interface
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
                            """Make the wrapper callable like LangChain models."""
                            result = self.llm(
                                prompt,
                                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                                temperature=kwargs.get('temperature', self.temperature),
                                stop=kwargs.get('stop', []),
                            )
                            return result['choices'][0]['text']
                        
                        def invoke(self, inputs, **kwargs):
                            """LangChain-style invoke method."""
                            if isinstance(inputs, dict):
                                prompt = inputs.get('input', str(inputs))
                            else:
                                prompt = str(inputs)
                            return self(prompt, **kwargs)
                    
                    llm = LlamaCppWrapper(
                        model_path=LLM_MODEL_PATH,
                        temperature=LLM_TEMPERATURE,
                        max_tokens=LLM_MAX_TOKENS,
                        n_ctx=LLM_CONTEXT_LENGTH,
                        n_threads=4
                    )
                    
                    print(f"✅ GGUF model loaded successfully via direct llama-cpp-python")
                    return llm
                    
                except Exception as e:
                    print(f"❌ Failed to load GGUF model directly: {e}")
                    return None
                
            elif LLM_TYPE == "ollama" and OLLAMA_AVAILABLE:
                print(f"📥 Connecting to Ollama model: {LLM_MODEL}")
                llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
                print(f"✅ Ollama model connected")
                return llm
            
            elif LLM_TYPE == "none":
                print("ℹ️  LLM disabled in config (LLM_TYPE='none')")
                return None
                
            else:
                print(f"⚠️  LLM type '{LLM_TYPE}' not available or libraries not installed")
                if LLM_TYPE == "gguf":
                    print("   Make sure langchain-community is installed:")
                    print("   pip install langchain-community")
                return None
                
        except Exception as e:
            print(f"❌ Error initializing LLM: {e}")
            print("   Falling back to rule-based classification")
            import traceback
            traceback.print_exc()
            return None
    
    def _setup_classifier(self):
        """Set up the classification prompt and chain."""
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query classifier for a movie information system.

Classify the user's query into exactly ONE of these types:
- factual
- embedding  
- multimedia
- recommendation

Respond with ONLY a JSON object containing the question_type. Nothing else.

Format: {{"question_type": "factual"}}

Rules:
1. FACTUAL: Questions about specific facts (who, what, when, where)
2. EMBEDDING: Semantic search or similarity questions
3. MULTIMEDIA: Questions asking to see/show images
4. RECOMMENDATION: Asking for suggestions

Examples:
- "Who directed Star Wars?" → {{"question_type": "factual"}}
- "Show me a picture of Tom Hanks" → {{"question_type": "multimedia"}}
- "Recommend movies like Inception" → {{"question_type": "recommendation"}}

Output ONLY the JSON, no other text."""),
            ("user", "{query}")
        ])
        
        # Don't use PydanticOutputParser, parse manually
        self.classification_chain = classification_prompt | self.llm
    
    def classify_query(self, query: str) -> QueryClassification:
        """Classify a user query into one of the four types."""
        if self.use_llm:
            try:
                # Invoke the LLM
                raw_output = self.classification_chain.invoke({"query": query})
                
                # Handle different output types
                if isinstance(raw_output, str):
                    output_text = raw_output
                else:
                    output_text = str(raw_output)
                
                print(f"[Classification] Raw LLM output: {output_text[:100]}...")
                
                # Parse JSON from output
                import json
                import re
                
                # Remove "AI:" prefix if present
                output_text = re.sub(r'^AI:\s*', '', output_text.strip())
                
                # Extract JSON
                json_match = re.search(r'\{[^}]*"question_type"\s*:\s*"(\w+)"[^}]*\}', output_text)
                if json_match:
                    full_json = json_match.group(0)
                    data = json.loads(full_json)
                    question_type = data.get('question_type', 'factual')
                    
                    print(f"[Classification] ✅ Parsed type: {question_type}")
                    
                    return QueryClassification(question_type=QuestionType(question_type))
                else:
                    raise ValueError(f"Could not find JSON in output: {output_text}")
                
            except Exception as e:
                print(f"⚠️  LLM classification failed: {str(e)[:200]}")
                print("⚠️  Using rule-based fallback")
                return self._rule_based_classify(query)
        else:
            return self._rule_based_classify(query)
    
    def _rule_based_classify(self, query: str) -> QueryClassification:
        """
        Fallback rule-based classification when LLM is not available.
        """
        query_lower = query.lower()
        
        # Multimedia keywords
        multimedia_keywords = ['show', 'picture', 'image', 'photo', 'display', 'look like', 'see', 'view']
        if any(keyword in query_lower for keyword in multimedia_keywords):
            return QueryClassification(
                question_type=QuestionType.MULTIMEDIA
            )
        
        # Recommendation keywords
        recommendation_keywords = ['recommend', 'suggest', 'similar', 'like', 'what should i watch']
        if any(keyword in query_lower for keyword in recommendation_keywords):
            return QueryClassification(
                question_type=QuestionType.RECOMMENDATION
            )
        
        # Factual keywords (who, what, when, where)
        factual_keywords = ['who', 'what', 'when', 'where', 'which', 'director', 'actor', 'release']
        if any(keyword in query_lower for keyword in factual_keywords):
            return QueryClassification(
                question_type=QuestionType.FACTUAL
            )
        
        # Default to factual for knowledge graph queries
        return QueryClassification(
            question_type=QuestionType.FACTUAL
        )

    def process_query(self, query: str) -> str:
        """
        Process a query by routing it through the workflow or directly to handlers.
        
        Args:
            query: User's natural language query
            
        Returns:
            Formatted response string
        """
        if self.use_workflow:
            # Use LangGraph-style workflow with validation and routing
            return self.workflow.run(query)
        else:
            # Legacy direct processing
            return self._process_query_legacy(query)
    
    def _process_query_legacy(self, query: str) -> str:
        """Legacy query processing without workflow (backward compatibility)."""
        classification = self.classify_query(query)
        
        print(f"Query classified as: {classification.question_type.value}")
        
        # Route to appropriate node
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
        """Handle factual questions using knowledge graph."""
        try:
            # Validate the question
            if not self.nl_to_sparql.validate_question(query):
                return "I'm sorry, but I couldn't understand your question. Please ask a question about movies, actors, directors, or related topics."
            
            # Convert natural language to SPARQL
            print("Converting natural language to SPARQL...")
            sparql_result = self.nl_to_sparql.convert(query)
            
            print(f"Generated SPARQL query (confidence: {sparql_result.confidence}):")
            print(sparql_result.query)
            print(f"Explanation: {sparql_result.explanation}")
            
            # Check confidence threshold
            if sparql_result.confidence < 0.5:
                return (
                    f"I'm not very confident about this query (confidence: {sparql_result.confidence:.2f}). "
                    f"The generated query might not be accurate.\n\n"
                    f"Generated query:\n{sparql_result.query}\n\n"
                    "Would you like to rephrase your question?"
                )
            
            # Execute the SPARQL query
            print("Executing SPARQL query...")
            result = self.sparql_handler.process_sparql_input(sparql_result.query)
            
            # Format the response
            if "No results found" in result or "Error" in result:
                return (
                    f"I couldn't find an answer to your question.\n\n"
                    f"Generated query:\n{sparql_result.query}\n\n"
                    f"Result: {result}\n\n"
                    "The information might not be available in the knowledge graph, or the query might need adjustment."
                )
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            return (
                f"Error processing factual question: {error_msg}\n\n"
                f"Original question: {query}\n\n"
                "Please try rephrasing your question or check if the information is available."
            )
    
    def _handle_embedding(self, query: str) -> str:
        """Handle embedding-based questions (placeholder)."""
        # TODO: Implement embedding search logic
        return f"[EMBEDDING NODE - Not yet implemented] Processing: {query}"
    
    def _handle_multimedia(self, query: str) -> str:
        """Handle multimedia questions (placeholder)."""
        # TODO: Implement multimedia retrieval logic
        return f"[MULTIMEDIA NODE - Not yet implemented] Processing: {query}"
    
    def _handle_recommendation(self, query: str) -> str:
        """Handle recommendation questions (placeholder)."""
        # TODO: Implement recommendation logic
        return f"[RECOMMENDATION NODE - Not yet implemented] Processing: {query}"
