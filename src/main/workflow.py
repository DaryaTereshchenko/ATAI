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
        - ✅ NEW: Remove explicit approach prefixes

        NOTE: Do NOT title-case or entity-normalize here.
        NLToSPARQL handles all casing and label matching.
        """
        if not query:
            return ""

        s = query.strip()
        
        # ✅ FIXED: More specific patterns to avoid removing "From" at the start of actual questions
        # These patterns now require the full phrase with "Please answer" or similar
        s = re.sub(
            r'^please\s+answer\s+this\s+question\s+with\s+(?:a|an)\s+factual\s+approach:\s*',
            '',
            s,
            flags=re.IGNORECASE
        )
        s = re.sub(
            r'^please\s+answer\s+this\s+question\s+with\s+(?:a|an)\s+embedding\s+approach:\s*',
            '',
            s,
            flags=re.IGNORECASE
        )
        s = re.sub(
            r'^please\s+answer\s+this\s+question:\s*',
            '',
            s,
            flags=re.IGNORECASE
        )
        
        # smart quotes/dashes → ASCII
        s = (s.replace(""", '"').replace(""", '"')
               .replace("'", "'").replace("'", "'")
               .replace("—", "-").replace("–", "-"))
        # collapse spaces
        s = re.sub(r"\s+", " ", s)

        # Keep user punctuation; do NOT strip characters (movie titles need them)
        return s.strip()

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
        print(f"\n[NODE: validate_input] Processing query: {state['raw_query'][:50]}...")

        validation_result = self.validator.validate(state["raw_query"])

        state["is_valid"] = validation_result["is_valid"]
        state["validation_message"] = validation_result["message"]
        state["detected_threats"] = validation_result["threats"]
        state["current_node"] = "validate_input"

        # Replace raw_query with cleaned version for downstream processing
        if validation_result.get("cleaned_query"):
            original_query = state["raw_query"]
            state["raw_query"] = validation_result["cleaned_query"]
            print(f"[NODE: validate_input] Cleaned query: {state['raw_query'][:80]}...")
            if original_query != state["raw_query"]:
                print(f"[NODE: validate_input] ℹ️  Query was normalized (quotes/dashes/spacing)")

        if not validation_result["is_valid"]:
            print(f"[NODE: validate_input] ❌ Validation failed: {validation_result['message']}")
            print(f"[NODE: validate_input] Threats: {validation_result['threats']}")
            state["error"] = validation_result["message"]
            state["processing_method"] = ProcessingMethod.FAILED
        else:
            print(f"[NODE: validate_input] ✅ Validation passed")

        return state

    def classify_query(self, state: WorkflowState) -> WorkflowState:
        """
        Node 2: Classify the query type using rule-based keywords.
        """
        print(f"\n[NODE: classify_query] Classifying query...")

        try:
            classification = self.orchestrator.classify_query(state["raw_query"])
            state["query_type"] = classification.question_type.value
            state["current_node"] = "classify_query"
            
            print(f"[NODE: classify_query] Type: {state['query_type']}")
            print(f"[NODE: classify_query] Confidence: {classification.confidence:.2%}")
            
        except Exception as e:
            print(f"[NODE: classify_query] ❌ Classification error: {e}")
            state["error"] = f"Classification failed: {str(e)}"
            state["query_type"] = "unknown"

        return state

    def decide_processing_method(self, state: WorkflowState) -> WorkflowState:
        """
        Node 3: Decide processing method based on query type.
        """
        print(f"\n[NODE: decide_processing_method] Routing based on query type...")

        query_type = state.get("query_type", "hybrid")
        
        # Map query type to processing method
        if query_type == "factual":
            state["processing_method"] = ProcessingMethod.SPARQL
            state["routing_reason"] = "Explicit factual approach requested - using SPARQL pipeline."
        elif query_type == "embeddings":
            state["processing_method"] = ProcessingMethod.EMBEDDING
            state["routing_reason"] = "Explicit embeddings approach requested - using embedding space."
        elif query_type == "hybrid":
            state["processing_method"] = ProcessingMethod.HYBRID
            state["routing_reason"] = "No explicit approach specified - using both factual and embeddings."
        elif query_type == "image":
            state["processing_method"] = ProcessingMethod.FAILED
            state["routing_reason"] = "Image queries not yet supported."
            state["error"] = "Image queries are not yet implemented."
        elif query_type == "recommendation":
            state["processing_method"] = ProcessingMethod.FAILED
            state["routing_reason"] = "Recommendation queries not yet supported."
            state["error"] = "Recommendation queries are not yet implemented."
        else:
            state["processing_method"] = ProcessingMethod.HYBRID
            state["routing_reason"] = "Unknown query type - defaulting to hybrid."

        state["current_node"] = "decide_processing_method"
        print(f"[NODE: decide_processing_method] ✅ Method: {state['processing_method'].value}")
        print(f"[NODE: decide_processing_method] 📋 Reason: {state['routing_reason']}")

        return state

    def process_with_factual(self, state: WorkflowState) -> WorkflowState:
        """
        Node 4a: Process with factual SPARQL approach.
        """
        print(f"\n[NODE: process_with_factual] Processing with factual approach...")
        
        if not hasattr(self.orchestrator, 'embedding_processor') or self.orchestrator.embedding_processor is None:
            print(f"[NODE: process_with_factual] ❌ Embedding processor not available")
            state['error'] = "Embedding processor not initialized. Cannot process query."
            state["current_node"] = "process_with_factual"
            return state
        
        try:
            query = state["raw_query"]
            
            # Use hybrid factual query processing (pattern-based SPARQL)
            result = self.orchestrator.embedding_processor.process_hybrid_factual_query(query)
            
            state["raw_result"] = result
            state["formatted_response"] = result
            state["processing_method"] = ProcessingMethod.SPARQL
            state["current_node"] = "process_with_factual"
            
            print(f"[NODE: process_with_factual] ✅ Factual processing successful")
            
        except Exception as e:
            state["error"] = f"Factual processing error: {e}"
            state["current_node"] = "process_with_factual"
            print(f"[NODE: process_with_factual] ❌ {state['error']}")
            import traceback
            traceback.print_exc()
        
        return state

    def process_with_embeddings(self, state: WorkflowState) -> WorkflowState:
        """
        Node 4b: Process with embeddings approach.
        """
        print(f"\n[NODE: process_with_embeddings] Processing with embeddings approach...")
        
        if not hasattr(self.orchestrator, 'embedding_processor') or self.orchestrator.embedding_processor is None:
            print(f"[NODE: process_with_embeddings] ❌ Embedding processor not available")
            state['error'] = "Embedding processor not initialized."
            state["current_node"] = "process_with_embeddings"
            return state
        
        try:
            query = state["raw_query"]
            
            result = self.orchestrator.embedding_processor.process_embedding_query(query)
            
            state["raw_result"] = result
            state["formatted_response"] = result
            state["processing_method"] = ProcessingMethod.EMBEDDING
            state["current_node"] = "process_with_embeddings"
            
            print(f"[NODE: process_with_embeddings] ✅ Embeddings processing successful")
            
        except Exception as e:
            state["error"] = f"Embeddings processing error: {e}"
            state["current_node"] = "process_with_embeddings"
            print(f"[NODE: process_with_embeddings] ❌ {state['error']}")
            import traceback
            traceback.print_exc()
        
        return state

    def process_with_both(self, state: WorkflowState) -> WorkflowState:
        """
        Node 4c: Process with both factual and embeddings approaches.
        """
        print(f"\n[NODE: process_with_both] Processing with hybrid approach (factual + embeddings)...")
        
        if not hasattr(self.orchestrator, 'embedding_processor') or self.orchestrator.embedding_processor is None:
            print(f"[NODE: process_with_both] ❌ Embedding processor not available")
            state['error'] = "Embedding processor not initialized."
            state["current_node"] = "process_with_both"
            return state
        
        try:
            query = state["raw_query"]
            
            # Get factual result
            print(f"[NODE: process_with_both] Running factual approach...")
            factual_result = self.orchestrator.embedding_processor.process_hybrid_factual_query(query)
            
            # Get embeddings result
            print(f"[NODE: process_with_both] Running embeddings approach...")
            embeddings_result = self.orchestrator.embedding_processor.process_embedding_query(query)
            
            # Combine results
            combined_result = f"**Factual Answer:**\n{factual_result}\n\n"
            combined_result += f"**Embeddings Answer:**\n{embeddings_result}"
            
            state["raw_result"] = combined_result
            state["formatted_response"] = combined_result
            state["processing_method"] = ProcessingMethod.HYBRID
            state["current_node"] = "process_with_both"
            
            print(f"[NODE: process_with_both] ✅ Hybrid processing successful")
            
        except Exception as e:
            state["error"] = f"Hybrid processing error: {e}"
            state["current_node"] = "process_with_both"
            print(f"[NODE: process_with_both] ❌ {state['error']}")
            import traceback
            traceback.print_exc()
        
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
        
        # Initialize state
        state: WorkflowState = {
            'raw_query': query,
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
            
            # Node 2: Classify query
            state = self.classify_query(state)
            if state.get('error'):
                state = self.format_response(state)
                return state['formatted_response']
            
            # Node 3: Decide processing method
            state = self.decide_processing_method(state)
            
            # Node 4: Process based on method
            if state['processing_method'] == ProcessingMethod.SPARQL:
                state = self.process_with_factual(state)  # ✅ FIXED: Use correct method name
            elif state['processing_method'] == ProcessingMethod.EMBEDDING:
                state = self.process_with_embeddings(state)
            elif state['processing_method'] == ProcessingMethod.HYBRID:
                state = self.process_with_both(state)
            elif state['processing_method'] == ProcessingMethod.FAILED:
                # Error already set
                pass
            
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

        # Check if response is already formatted
        if state.get("formatted_response"):
            state["current_node"] = "format_response"
            print(f"[NODE: format_response] ✅ Response already formatted")
            return state

        # Use raw result as formatted response
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
