"""
LangGraph-style workflow for query processing.
Implements a state machine with validation, routing, and formatting.
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
    """Validates user input for security threats and processability."""
    
    # Malicious patterns to detect
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
        '\x00',  # Null byte
        '\r\n\r\n',  # HTTP header injection
        '<?php',  # PHP code
        '<%',  # ASP code
    ]
    
    MAX_QUERY_LENGTH = 1000
    MIN_QUERY_LENGTH = 2
    
    @classmethod
    def validate(cls, query: str) -> dict:
        """
        Validate user input for security and processability.
        
        Returns:
            dict with 'is_valid', 'message', and 'threats' keys
        """
        threats = []
        
        # Check if query is empty or just whitespace
        if not query or not query.strip():
            return {
                'is_valid': False,
                'message': "Query cannot be empty.",
                'threats': ['empty_input']
            }
        
        # Check length
        if len(query) > cls.MAX_QUERY_LENGTH:
            return {
                'is_valid': False,
                'message': f"Query is too long (max {cls.MAX_QUERY_LENGTH} characters).",
                'threats': ['excessive_length']
            }
        
        if len(query.strip()) < cls.MIN_QUERY_LENGTH:
            return {
                'is_valid': False,
                'message': "Query is too short.",
                'threats': ['insufficient_length']
            }
        
        # Check for malicious patterns
        for pattern in cls.MALICIOUS_PATTERNS:
            if re.search(pattern, query):
                threats.append(f"malicious_pattern: {pattern}")
        
        # Check for suspicious characters
        for char_seq in cls.SUSPICIOUS_CHARS:
            if char_seq in query:
                threats.append(f"suspicious_chars: {char_seq}")
        
        # Check for excessive repetition (potential DoS)
        if re.search(r'(.)\1{50,}', query):
            threats.append('excessive_repetition')
        
        # Check for non-printable characters (except common whitespace)
        if re.search(r'[^\x20-\x7E\t\n\r]', query):
            threats.append('non_printable_chars')
        
        if threats:
            return {
                'is_valid': False,
                'message': "Query contains potentially malicious or invalid content.",
                'threats': threats
            }
        
        return {
            'is_valid': True,
            'message': "Query is valid.",
            'threats': []
        }


class QueryWorkflow:
    """LangGraph-style workflow for processing user queries."""
    
    def __init__(self, orchestrator):
        """Initialize workflow with orchestrator."""
        self.orchestrator = orchestrator
        self.validator = InputValidator()
        self.formatter = AnswerFormatter()  # Add answer formatter
    
    # ==================== WORKFLOW NODES ====================
    
    def validate_input(self, state: WorkflowState) -> WorkflowState:
        """
        Node 1: Validate user input for security and processability.
        """
        print(f"\n[NODE: validate_input] Processing query: {state['raw_query'][:50]}...")
        
        validation_result = self.validator.validate(state['raw_query'])
        
        state['is_valid'] = validation_result['is_valid']
        state['validation_message'] = validation_result['message']
        state['detected_threats'] = validation_result['threats']
        state['current_node'] = 'validate_input'
        
        if not validation_result['is_valid']:
            print(f"[NODE: validate_input] ❌ Validation failed: {validation_result['message']}")
            print(f"[NODE: validate_input] Threats: {validation_result['threats']}")
            state['error'] = validation_result['message']
            state['processing_method'] = ProcessingMethod.FAILED
        else:
            print(f"[NODE: validate_input] ✅ Validation passed")
        
        return state
    
    def classify_query(self, state: WorkflowState) -> WorkflowState:
        """
        Node 2: Classify the query type using the orchestrator.
        """
        print(f"\n[NODE: classify_query] Classifying query...")
        
        try:
            classification = self.orchestrator.classify_query(state['raw_query'])
            
            state['query_type'] = classification.question_type.value
            state['current_node'] = 'classify_query'
            
            print(f"[NODE: classify_query] Type: {state['query_type']}")
            
        except Exception as e:
            print(f"[NODE: classify_query] ❌ Classification error: {e}")
            state['error'] = f"Classification failed: {str(e)}"
            state['query_type'] = 'unknown'
        
        return state
    
    def decide_processing_method(self, state: WorkflowState) -> WorkflowState:
        """
        Node 3: Decide whether to use SPARQL or embeddings based on query type.
        """
        print(f"\n[NODE: decide_processing_method] Deciding processing method...")
        
        query_type = state.get('query_type', 'unknown')
        
        # Decision logic
        if query_type == 'factual':
            # Factual questions go through SPARQL
            state['processing_method'] = ProcessingMethod.SPARQL
            state['routing_reason'] = "Factual questions use SPARQL for precise data retrieval."
        
        elif query_type == 'embedding':
            # Embedding questions use semantic search (placeholder for now)
            state['processing_method'] = ProcessingMethod.EMBEDDING
            state['routing_reason'] = "This question requires semantic search."
        
        elif query_type in ['multimedia', 'recommendation']:
            # These might use hybrid approach in the future
            state['processing_method'] = ProcessingMethod.SPARQL
            state['routing_reason'] = f"{query_type.capitalize()} questions will use SPARQL for now."
        
        else:
            # Unknown - default to SPARQL
            state['processing_method'] = ProcessingMethod.SPARQL
            state['routing_reason'] = "Defaulting to SPARQL-based processing."
        
        state['current_node'] = 'decide_processing_method'
        
        print(f"[NODE: decide_processing_method] ✅ Method: {state['processing_method'].value}")
        
        return state
    
    def process_with_sparql(self, state: WorkflowState) -> WorkflowState:
        """
        Node 4a: Process query using SPARQL generation and execution.
        """
        try:
            # Convert natural language to SPARQL
            sparql_result = self.orchestrator.nl_to_sparql.convert(state['raw_query'])
            
            # ✅ Execute the SPARQL query against the database
            print("[NODE: process_with_sparql] Executing SPARQL query...")
            result = self.orchestrator.sparql_handler.execute_and_format(sparql_result.query)
            # ↑ This queries the RDF graph and returns the results
            
            state['raw_result'] = result
            state['current_node'] = 'process_with_sparql'
            
            print(f"[NODE: process_with_sparql] ✅ Query executed successfully")
            
        except Exception as e:
            error_msg = str(e)
            state['error'] = f"SPARQL processing error: {error_msg}"
            print(f"[NODE: process_with_sparql] ❌ {state['error']}")
        
        return state
    
    def process_with_embeddings(self, state: WorkflowState) -> WorkflowState:
        """
        Node 4b: Process query using embeddings (placeholder).
        """
        print(f"\n[NODE: process_with_embeddings] Processing with embeddings...")
        
        # Placeholder for embedding-based processing
        state['error'] = "Embedding-based processing not yet implemented."
        state['current_node'] = 'process_with_embeddings'
        
        print(f"[NODE: process_with_embeddings] ⚠️ Not implemented yet")
        
        return state
    
    def format_response(self, state: WorkflowState) -> WorkflowState:
        """
        Node 5: Format the response in a user-friendly way.
        """
        print(f"\n[NODE: format_response] Formatting response...")
        
        # Check if there was an error
        if state.get('error'):
            state['formatted_response'] = self._format_error_response(state)
            state['current_node'] = 'format_response'
            return state
        
        # Check if we have a result
        if not state.get('raw_result'):
            state['formatted_response'] = (
                "I processed your query, but couldn't find a result. "
                "The information might not be available in the knowledge graph."
            )
            state['current_node'] = 'format_response'
            return state
        
        # Format based on processing method
        if state['processing_method'] == ProcessingMethod.SPARQL:
            state['formatted_response'] = self._format_sparql_response(state)
        elif state['processing_method'] == ProcessingMethod.EMBEDDING:
            state['formatted_response'] = self._format_embedding_response(state)
        else:
            state['formatted_response'] = state['raw_result']
        
        state['current_node'] = 'format_response'
        
        print(f"[NODE: format_response] ✅ Response formatted")
        
        return state
    
    def _format_sparql_response(self, state: WorkflowState) -> str:
        """Format response for SPARQL processing using templates."""
        raw_result = state['raw_result']
        explanation = state.get('sparql_explanation')
        
        # Use the template-based formatter for human-friendly output
        return self.formatter.format(raw_result, explanation)
    
    def _format_embedding_response(self, state: WorkflowState) -> str:
        """Format response for embedding processing."""
        response = "🔍 **Query processed using Semantic Search (Embeddings)**\n\n"
        response += "📊 **Answer:**\n" + state['raw_result']
        return response
    
    def _format_error_response(self, state: WorkflowState) -> str:
        """Format error response."""
        error = state.get('error', 'Unknown error')
        
        # Check for validation errors
        if state.get('detected_threats'):
            return (
                "⚠️ **Security Warning**\n\n"
                "Your query contains potentially unsafe content and cannot be processed.\n\n"
                f"Details: {error}\n\n"
                "Please rephrase your question using natural language only."
            )
        
        # Use formatter for other errors
        return self.formatter.format_error(error)
    
    # ==================== WORKFLOW EXECUTION ====================
    
    def should_continue(self, state: WorkflowState) -> Literal["continue", "error", "end"]:
        """Determine if workflow should continue based on state."""
        if state.get('error'):
            return "error"
        if state.get('formatted_response'):
            return "end"
        return "continue"
    
    def route_processing(self, state: WorkflowState) -> Literal["sparql", "embeddings", "error"]:
        """Route to appropriate processing method."""
        if state.get('error'):
            return "error"
        
        method = state.get('processing_method')
        if method == ProcessingMethod.SPARQL:
            return "sparql"
        elif method == ProcessingMethod.EMBEDDING:
            return "embeddings"
        else:
            return "error"
    
    def run(self, query: str) -> str:
        """
        Execute the complete workflow.
        
        Args:
            query: User's natural language query
            
        Returns:
            Formatted response string
        """
        print("=" * 80)
        print("WORKFLOW STARTED")
        print("=" * 80)
        
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
        
        # Step 1: Validate input
        state = self.validate_input(state)
        if state.get('error'):
            state = self.format_response(state)
            return state['formatted_response']
        
        # Step 2: Classify query
        state = self.classify_query(state)
        if state.get('error'):
            state = self.format_response(state)
            return state['formatted_response']
        
        # Step 3: Decide processing method
        state = self.decide_processing_method(state)
        
        # Step 4: Process based on decision
        processing_method = state.get('processing_method')
        
        if processing_method == ProcessingMethod.SPARQL:
            state = self.process_with_sparql(state)
        elif processing_method == ProcessingMethod.EMBEDDING:
            state = self.process_with_embeddings(state)
        else:
            state['error'] = "Unable to determine processing method"
        
        # Step 5: Format response
        state = self.format_response(state)
        
        print("\n" + "=" * 80)
        print("WORKFLOW COMPLETED")
        print("=" * 80 + "\n")
        
        return state['formatted_response']
