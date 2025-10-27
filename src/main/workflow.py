"""
LangGraph-style workflow for query processing.
Implements a state machine with validation, routing, and formatting.

This version is streamlined to cooperate with a model-first NL→SPARQL pipeline:
- Light, lossless NL pre-processing only (no title-casing here).
- All casing/label matching is handled in NLToSPARQL (case-insensitive, anchored).
"""

import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from typing import TypedDict, Literal, Optional, List
from enum import Enum
import re
from dataclasses import dataclass
from pydantic import BaseModel, Field
from src.main.answer_formatter import AnswerFormatter


class ProcessingMethod(str, Enum):
    """Method used to process the query."""
    SPARQL = "sparql"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"
    FAILED = "failed"


class WorkflowState(TypedDict):
    """State object passed between workflow nodes."""
    # Input
    raw_query: str
    
    # ✅ NEW: Separate approach and clean question
    requested_approach: Optional[str]  # "factual", "embedding", or "both"
    clean_question: str  # Question WITHOUT approach prefix

    # Validation
    is_valid: bool
    validation_message: Optional[str]
    detected_threats: List[str]

    # Classification
    query_type: Optional[str]  # factual, embedding, multimedia, recommendation

    # Routing decision
    processing_method: Optional[ProcessingMethod]
    routing_reason: Optional[str]

    # SPARQL processing
    generated_sparql: Optional[str]
    sparql_confidence: float
    sparql_explanation: Optional[str]

    # Results
    raw_result: Optional[str]
    formatted_response: Optional[str]

    # Error handling
    error: Optional[str]
    current_node: str


class InputValidator:
    """
    Validates user input for security threats and processability,
    and applies *light* normalization only.
    """

    # Malicious patterns to detect (keep conservative)
    MALICIOUS_PATTERNS = [
        # SQL injection attempts
        r"(?i)(union\s+select|drop\s+table|insert\s+into|delete\s+from)",
        # Script injection
        r"(?i)(<script|javascript:|onerror=|onclick=)",
        # Command injection
        r"(?i)(;\s*rm\s+-rf|&&\s*cat\s+|`.*`|\$\(.*\))",
        # Path traversal
        r"(\.\./|\.\.\\|%2e%2e)",
        # SPARQL injection (modify operations)
        r"(?i)(;\s*drop|;\s*insert|;\s*delete|;\s*clear)",
        # Excessive special characters
        r"[<>{}|\[\]]{5,}",
    ]

    # Suspicious character sequences
    SUSPICIOUS_CHARS = [
        "\x00",         # Null byte
        "\r\n\r\n",     # HTTP header injection
        "<?php",        # PHP code
        "<%",           # ASP code
    ]

    MAX_QUERY_LENGTH = 1000
    MIN_QUERY_LENGTH = 2

    @classmethod
    def preprocess_query(cls, query: str) -> str:
        """
        Light, lossless normalization for user input:
        - Trim
        - Normalize smart quotes/dashes to ASCII
        - Collapse whitespace

        NOTE: Do NOT title-case or entity-normalize here.
        NLToSPARQL handles all casing and label matching.
        """
        if not query:
            return ""

        s = query.strip()
        # smart quotes/dashes → ASCII
        s = (s.replace("“", '"').replace("”", '"')
               .replace("‘", "'").replace("’", "'")
               .replace("—", "-").replace("–", "-"))
        # collapse spaces
        s = re.sub(r"\s+", " ", s)

        # Keep user punctuation; do NOT strip characters (movie titles need them)
        return s

    @classmethod
    def validate(cls, query: str) -> dict:
        """
        Validate user input for security and processability.

        Returns:
            dict with 'is_valid', 'message', 'threats', and 'cleaned_query' keys
        """
        threats: List[str] = []

        # Preprocess the query first (light, lossless)
        cleaned_query = cls.preprocess_query(query)

        # Empty after cleaning?
        if not cleaned_query:
            return {
                "is_valid": False,
                "message": "Query cannot be empty.",
                "threats": ["empty_input"],
                "cleaned_query": cleaned_query,
            }

        # Length (use cleaned query)
        if len(cleaned_query) > cls.MAX_QUERY_LENGTH:
            return {
                "is_valid": False,
                "message": f"Query is too long (max {cls.MAX_QUERY_LENGTH} characters).",
                "threats": ["excessive_length"],
                "cleaned_query": cleaned_query,
            }

        if len(cleaned_query) < cls.MIN_QUERY_LENGTH:
            return {
                "is_valid": False,
                "message": "Query is too short.",
                "threats": ["insufficient_length"],
                "cleaned_query": cleaned_query,
            }

        # Malicious patterns (on original input to avoid bypass)
        for pattern in cls.MALICIOUS_PATTERNS:
            if re.search(pattern, query):
                threats.append(f"malicious_pattern: {pattern}")

        # Suspicious sequences (on original)
        for char_seq in cls.SUSPICIOUS_CHARS:
            if char_seq in query:
                threats.append(f"suspicious_chars: {char_seq}")

        # Excessive repetition (on cleaned)
        if re.search(r"(.)\1{50,}", cleaned_query):
            threats.append("excessive_repetition")

        if threats:
            return {
                "is_valid": False,
                "message": "Query contains potentially malicious or invalid content.",
                "threats": threats,
                "cleaned_query": cleaned_query,
            }

        return {
            "is_valid": True,
            "message": "Query is valid.",
            "threats": [],
            "cleaned_query": cleaned_query,
        }
    
    @classmethod
    def extract_approach_and_question(cls, query: str) -> tuple[str, str]:
        """
        Extract approach indicator and clean question from user input.
        
        CRITICAL: This must happen BEFORE any other preprocessing to avoid losing information.
        
        Patterns:
        - "Please answer this question with a factual approach: <question>"
        - "Please answer this question with an embedding approach: <question>"
        - "<question>" (no approach indicator)
        
        Returns:
            Tuple of (approach, clean_question)
            - approach: "factual", "embedding", or "both" (if neither specified)
            - clean_question: The actual question without the approach prefix
        """
        # ✅ CRITICAL: Extract BEFORE any preprocessing
        query_stripped = query.strip()
        
        # Check for explicit approach indicator
        import re
        
        # Pattern: "Please answer this question with an? X approach: <question>"
        pattern = r'^[Pp]lease answer this question with an? (factual|embedding) approach:\s*(.+)$'
        match = re.match(pattern, query_stripped, re.IGNORECASE)
        
        if match:
            approach_word = match.group(1).lower()
            clean_question = match.group(2).strip()
            
            if approach_word == 'factual':
                return ('factual', clean_question)
            elif approach_word == 'embedding':
                return ('embedding', clean_question)
        
        # No explicit approach - check for keywords in the question itself
        query_lower = query_stripped.lower()
        words = set(re.findall(r'\b\w+\b', query_lower))
        
        has_factual = 'factual' in words
        has_embedding = {'embedding', 'embeddings'} & words
        
        if has_factual and not has_embedding:
            return ('factual', query_stripped)
        elif has_embedding and not has_factual:
            return ('embedding', query_stripped)
        
        # ✅ CHANGED: Default is now 'both' instead of 'factual'
        return ('both', query_stripped)


class QueryWorkflow:
    """LangGraph-style workflow for processing user queries."""

    def __init__(self, orchestrator):
        """Initialize workflow with orchestrator."""
        self.orchestrator = orchestrator
        self.validator = InputValidator()
        self.formatter = AnswerFormatter()

    # ==================== WORKFLOW NODES ====================

    def validate_input(self, state: WorkflowState) -> WorkflowState:
        """
        Node 1: Validate user input for security and processability.
        """
        print(f"\n[NODE: validate_input] Validating input...")
        
        validation = self.validator.validate(state['raw_query'])
        
        state['is_valid'] = validation['is_valid']
        state['validation_message'] = validation['message']
        state['detected_threats'] = validation['threats']
        state['current_node'] = 'validate_input'
        
        # Update cleaned query
        if validation['is_valid']:
            state['raw_query'] = validation['cleaned_query']
            print(f"[NODE: validate_input] ✅ Input is valid")
        else:
            print(f"[NODE: validate_input] ❌ Validation failed: {validation['message']}")
        
        return state
    
    def classify_query(self, state: WorkflowState) -> WorkflowState:
        """
        Node 2: Classify the query type.
        """
        print(f"\n[NODE: classify_query] Classifying query...")
        
        try:
            classification = self.orchestrator.classify_query(state['raw_query'])
            
            state['query_type'] = classification.question_type.value
            state['current_node'] = 'classify_query'
            
            print(f"[NODE: classify_query] ✅ Type: {state['query_type']}")
            
        except Exception as e:
            state['error'] = f"Classification error: {e}"
            state['current_node'] = 'classify_query'
            print(f"[NODE: classify_query] ❌ {state['error']}")
        
        return state

    def decide_processing_method(self, state: WorkflowState) -> WorkflowState:
        """
        Node 3: Decide processing method based on requested approach and classification.
        """
        print(f"\n[NODE: decide_processing_method] Routing query...")
        
        requested = state.get('requested_approach', 'both')  # Default to both
        query_type = state.get('query_type', 'hybrid')  # From classification
        
        # ✅ Priority: requested_approach overrides query_type
        if requested == 'embedding' or query_type == 'embedding':
            state['processing_method'] = 'embedding'
            state['routing_reason'] = "Embedding approach requested or detected"
            print(f"[NODE: decide_processing_method] ✅ Method: EMBEDDING")
        
        elif requested == 'factual' or query_type == 'factual':
            state['processing_method'] = ProcessingMethod.HYBRID
            state['routing_reason'] = "Factual approach: entity extraction → SPARQL → graph query"
            print(f"[NODE: decide_processing_method] ✅ Method: FACTUAL (hybrid SPARQL)")
        
        elif query_type == 'image':
            state['processing_method'] = 'image'
            state['routing_reason'] = "Image/multimedia query"
            print(f"[NODE: decide_processing_method] ✅ Method: IMAGE")
        
        elif query_type == 'recommendation':
            state['processing_method'] = 'recommendation'
            state['routing_reason'] = "Recommendation query"
            print(f"[NODE: decide_processing_method] ✅ Method: RECOMMENDATION")
        
        else:  # hybrid or both
            state['processing_method'] = 'both'
            state['routing_reason'] = "Hybrid: using both factual + embedding approaches"
            print(f"[NODE: decide_processing_method] ✅ Method: BOTH (factual + embedding)")
        
        print(f"[NODE: decide_processing_method] 📋 Reason: {state['routing_reason']}")
        state['current_node'] = 'decide_processing_method'
        return state

    def run(self, query: str) -> str:
        """
        Execute the complete workflow for a user query.
        
        Args:
            query: The user's natural language question
            
        Returns:
            Formatted response string
        """
        print(f"\n{'='*80}")
        print(f"🔄 WORKFLOW EXECUTION STARTED")
        print(f"{'='*80}\n")
        
        # ✅ STEP 0: Extract approach and clean question FIRST
        requested_approach, clean_question = InputValidator.extract_approach_and_question(query)
        
        print(f"📋 Approach detected: {requested_approach}")
        print(f"📋 Clean question: '{clean_question}'\n")
        
        # Initialize state with CLEAN question
        state: WorkflowState = {
            'raw_query': clean_question,
            'requested_approach': requested_approach,
            'clean_question': clean_question,
            'is_valid': False,
            'validation_message': None,
            'detected_threats': [],
            'query_type': None,
            'processing_method': None,
            'routing_reason': None,
            'generated_sparql': None,
            'sparql_confidence': 0.0,
            'sparql_explanation': None,
            'raw_result': None,
            'formatted_response': None,
            'error': None,
            'current_node': 'start'
        }
        
        # Execute workflow nodes in sequence
        try:
            # Node 1: Validate input
            state = self.validate_input(state)
            if not state['is_valid']:
                state = self.format_response(state)
                return state['formatted_response']
            
            # Node 2: Classify query (keyword-based now)
            state = self.classify_query(state)
            if state.get('error'):
                state = self.format_response(state)
                return state['formatted_response']
            
            # Node 3: Route based on requested approach and classification
            state = self.decide_processing_method(state)
            
            processing_method = state.get('processing_method')
            
            # Node 4: Process with selected approach
            if processing_method == ProcessingMethod.HYBRID or processing_method == 'both':
                state = self.process_with_hybrid(state)
            elif processing_method == 'embedding':
                state = self.process_with_embedding(state)
            elif processing_method == 'image':
                state['formatted_response'] = "ℹ️  Image queries are not yet implemented."
                state['current_node'] = 'process_image'
            elif processing_method == 'recommendation':
                state['formatted_response'] = "ℹ️  Recommendation queries are not yet implemented."
                state['current_node'] = 'process_recommendation'
            else:
                print(f"❌ Unexpected processing_method: {processing_method}")
                state['error'] = f"Internal error: unexpected processing method '{processing_method}'"
            
            # Node 5: Format response
            state = self.format_response(state)
            
            print(f"\n{'='*80}")
            print(f"✅ WORKFLOW EXECUTION COMPLETED")
            print(f"{'='*80}\n")
            
            return state['formatted_response']
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"❌ WORKFLOW EXECUTION FAILED")
            print(f"{'='*80}\n")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            
            state['error'] = f"Workflow error: {str(e)}"
            state = self.format_response(state)
            return state['formatted_response']
    
    def process_with_hybrid(self, state: WorkflowState) -> WorkflowState:
        """
        Node 4A: Hybrid processing - entity extraction + SPARQL.
        Handles both single factual and dual (factual + embedding) modes.
        """
        print(f"\n[NODE: process_with_hybrid] Processing with hybrid approach...")
        
        if not hasattr(self.orchestrator, 'embedding_processor') or self.orchestrator.embedding_processor is None:
            print(f"[NODE: process_with_hybrid] ❌ Embedding processor not available")
            state['error'] = "Embedding processor not initialized. Cannot process query."
            state["current_node"] = "process_with_hybrid"
            return state
        
        try:
            # ✅ Use clean question directly - no additional cleaning needed
            query = state['clean_question']
            
            print(f"[NODE: process_with_hybrid] Processing query: '{query}'")
            
            # Check if we need both approaches
            if state.get('processing_method') == 'both':
                print(f"[NODE: process_with_hybrid] 📊 Running BOTH approaches...")
                
                # Run factual first
                factual_result = self.orchestrator.embedding_processor.process_hybrid_factual_query(query)
                
                # Run embedding
                embedding_result = self.orchestrator.embedding_processor.process_embedding_factual_query(query)
                
                # Combine results with proper formatting
                combined = f"{factual_result}\n\n{embedding_result}"
                
                state['raw_result'] = combined
                state['formatted_response'] = combined
                print(f"[NODE: process_with_hybrid] ✅ Both approaches completed")
            
            else:
                # Run factual only
                result = self.orchestrator.embedding_processor.process_hybrid_factual_query(query)
                
                state['raw_result'] = result
                state['formatted_response'] = result
                print(f"[NODE: process_with_hybrid] ✅ Factual processing successful")
            
            state['processing_method'] = ProcessingMethod.HYBRID
            state['current_node'] = 'process_with_hybrid'
            
        except Exception as e:
            state['error'] = f"Hybrid processing error: {e}"
            state['current_node'] = 'process_with_hybrid'
            print(f"[NODE: process_with_hybrid] ❌ {state['error']}")
            import traceback
            traceback.print_exc()
        
        return state

    def process_with_embedding(self, state: WorkflowState) -> WorkflowState:
        """
        Node 4B: Process with pure embedding approach (TransE computations).
        """
        print(f"\n[NODE: process_with_embedding] Processing with TransE embeddings...")
        
        if not hasattr(self.orchestrator, 'embedding_processor') or self.orchestrator.embedding_processor is None:
            print(f"[NODE: process_with_embedding] ❌ Embedding processor not available")
            state['error'] = "Embedding processor not initialized. Cannot use embedding approach."
            state["current_node"] = "process_with_embedding"
            return state
        
        try:
            # ✅ Use clean question directly - no additional cleaning needed
            query = state['clean_question']
            
            print(f"[NODE: process_with_embedding] Processing query: '{query}'")
            
            # Use pure embedding approach
            result = self.orchestrator.embedding_processor.process_embedding_factual_query(query)
            
            state['raw_result'] = result
            state['formatted_response'] = result
            state['processing_method'] = 'embedding'
            state['current_node'] = 'process_with_embedding'
            
            print(f"[NODE: process_with_embedding] ✅ Embedding processing successful")
            
        except Exception as e:
            state['error'] = f"Embedding processing error: {e}"
            state['current_node'] = 'process_with_embedding'
            print(f"[NODE: process_with_embedding] ❌ {state['error']}")
            import traceback
            traceback.print_exc()
        
        return state

    def format_response(self, state: WorkflowState) -> WorkflowState:
        """
        Node 5: Format the response in a user-friendly way.
        """
        print(f"\n[NODE: format_response] Formatting response...")

        # Error?
        if state.get("error"):
            state["formatted_response"] = self._format_error_response(state)
            state["current_node"] = "format_response"
            return state

        # No result?
        if not state.get("raw_result"):
            state["formatted_response"] = (
                "I processed your query, but couldn't find a result. "
                "The information might not be available in the knowledge graph."
            )
            state["current_node"] = "format_response"
            return state

        # Check if response is already formatted (from hybrid mode)
        if state.get("formatted_response"):
            state["current_node"] = "format_response"
            print(f"[NODE: format_response] ✅ Response already formatted")
            return state

        # Hybrid responses are pre-formatted
        state["formatted_response"] = state["raw_result"]

        state["current_node"] = "format_response"
        print(f"[NODE: format_response] ✅ Response formatted")

        return state

    def _format_error_response(self, state: WorkflowState) -> str:
        """Format error response."""
        error = state.get("error", "Unknown error")
        if state.get("detected_threats"):
            return (
                "⚠️ **Security Warning**\n\n"
                "Your query contains potentially unsafe content and cannot be processed.\n\n"
                "Please rephrase your question using natural language only."
            )
        return self.formatter.format_error(error)
