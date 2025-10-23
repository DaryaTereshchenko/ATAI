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


class QueryWorkflow:
    """LangGraph-style workflow for processing user queries."""

    def __init__(self, orchestrator):
        """Initialize workflow with orchestrator."""
        self.orchestrator = orchestrator
        self.validator = InputValidator()
        self.formatter = AnswerFormatter()

    # ==================== WORKFLOW NODES ====================

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
                state = self.process_with_sparql(state)
            elif state['processing_method'] == ProcessingMethod.EMBEDDING:
                state = self.process_with_embeddings(state)
            else:
                state['error'] = "Unknown processing method"
            
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
        Node 2: Classify the query type using the orchestrator.
        """
        print(f"\n[NODE: classify_query] Classifying query...")

        try:
            classification = self.orchestrator.classify_query(state["raw_query"])
            state["query_type"] = classification.question_type.value
            state["current_node"] = "classify_query"
            print(f"[NODE: classify_query] Type: {state['query_type']}")
        except Exception as e:
            print(f"[NODE: classify_query] ❌ Classification error: {e}")
            state["error"] = f"Classification failed: {str(e)}"
            state["query_type"] = "unknown"

        return state

    def decide_processing_method(self, state: WorkflowState) -> WorkflowState:
        """
        Node 3: Decide whether to use SPARQL or embeddings based on query type.
        """
        print(f"\n[NODE: decide_processing_method] Deciding processing method...")

        query_type = state.get("query_type", "unknown")

        # Decision logic
        if query_type == "factual":
            state["processing_method"] = ProcessingMethod.SPARQL
            state["routing_reason"] = "Factual questions use SPARQL for precise data retrieval."
        elif query_type == "embedding":
            state["processing_method"] = ProcessingMethod.EMBEDDING
            state["routing_reason"] = "This question requires semantic search."
        elif query_type in ["multimedia", "recommendation"]:
            state["processing_method"] = ProcessingMethod.SPARQL
            state["routing_reason"] = f"{query_type.capitalize()} questions will use SPARQL for now."
        else:
            state["processing_method"] = ProcessingMethod.SPARQL
            state["routing_reason"] = "Defaulting to SPARQL-based processing."

        state["current_node"] = "decide_processing_method"
        print(f"[NODE: decide_processing_method] ✅ Method: {state['processing_method'].value}")

        return state

    def process_with_sparql(self, state: WorkflowState) -> WorkflowState:
        """
        Node 4a: Process query using SPARQL generation and execution.
        - Uses orchestrator.nl_to_sparql (model-first) to generate SPARQL.
        - Executes via SPARQLHandler with validation + timeout + caching.
        """
        try:
            print("[NODE: process_with_sparql] Generating SPARQL query...")
            
            # Generate SPARQL (NLToSPARQL is model-first per orchestrator)
            sparql_result = self.orchestrator.nl_to_sparql.convert(state["raw_query"])

            # Store generated query + metadata in state
            state["generated_sparql"] = sparql_result.query
            state["sparql_confidence"] = float(getattr(sparql_result, "confidence", 0.0))
            state["sparql_explanation"] = getattr(sparql_result, "explanation", None)

            # Log generation details
            print(f"[NODE: process_with_sparql] 📊 Confidence: {state['sparql_confidence']:.2f}")
            if state["sparql_explanation"]:
                print(f"[NODE: process_with_sparql] 💡 Method: {state['sparql_explanation']}")
            
            # Determine if it was rule-based or LLM
            explanation_lower = (state["sparql_explanation"] or "").lower()
            if "pattern-matched" in explanation_lower or "rule" in explanation_lower:
                generation_method = "🎯 Rule-based (Pattern Matching)"
            elif "llm" in explanation_lower or "model" in explanation_lower:
                generation_method = "🤖 LLM-generated (DeepSeek)"
            else:
                generation_method = "❓ Unknown method"
            
            print(f"[NODE: process_with_sparql] 🔧 Generation: {generation_method}")
            print(f"[NODE: process_with_sparql] 📝 Query preview: {state['generated_sparql'][:100]}...")

            # Optional: language guard for labels
            # state["generated_sparql"] = self.orchestrator.sparql_handler.add_lang_filter(
            #     state["generated_sparql"], ["?movieLabel", "?personLabel"], "en"
            # )

            print("[NODE: process_with_sparql] Executing SPARQL query...")
            exec_result = self.orchestrator.sparql_handler.execute_query(
                state["generated_sparql"], validate=True
            )

            if not exec_result.get("success"):
                state["error"] = exec_result.get("error", "Unknown error")
                state["current_node"] = "process_with_sparql"
                print(f"[NODE: process_with_sparql] ❌ Execution error: {state['error']}")
                return state

            state["raw_result"] = exec_result.get("data") or "No answer found in the database."
            state["current_node"] = "process_with_sparql"
            print("[NODE: process_with_sparql] ✅ Query executed successfully")

        except Exception as e:
            state["error"] = f"SPARQL processing error: {e}"
            state["current_node"] = "process_with_sparql"
            print(f"[NODE: process_with_sparql] ❌ {state['error']}")

        return state

    def process_with_embeddings(self, state: WorkflowState) -> WorkflowState:
        """
        Node 4b: Process query using embeddings (placeholder).
        """
        print(f"\n[NODE: process_with_embeddings] Processing with embeddings...")
        state["error"] = "Embedding-based processing not yet implemented."
        state["current_node"] = "process_with_embeddings"
        print(f"[NODE: process_with_embeddings] ⚠️ Not implemented yet")
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

        # SPARQL / Embedding responses
        if state["processing_method"] == ProcessingMethod.SPARQL:
            state["formatted_response"] = self._format_sparql_response(state)
        elif state["processing_method"] == ProcessingMethod.EMBEDDING:
            state["formatted_response"] = self._format_embedding_response(state)
        else:
            state["formatted_response"] = state["raw_result"]

        state["current_node"] = "format_response"
        print(f"[NODE: format_response] ✅ Response formatted")

        return state

    def _format_sparql_response(self, state: WorkflowState) -> str:
        """Format response for SPARQL processing using templates."""
        raw_result = state["raw_result"]
        explanation = state.get("sparql_explanation")
        # Template-based formatter for human-friendly output
        return self.formatter.format(raw_result, explanation)

    def _format_embedding_response(self, state: WorkflowState) -> str:
        """Format response for embedding processing (placeholder)."""
        response = "🔍 **Query processed using Semantic Search (Embeddings)**\n\n"
        response += "📊 **Answer:**\n" + (state["raw_result"] or "")
        return response

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
