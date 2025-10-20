"""
LangGraph-style workflow for query processing.
Implements a state machine with validation, routing, and formatting.

Requirements:
    pip install spacy
    python -m spacy download en_core_web_sm
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

# Add spaCy import
try:
    import spacy
    SPACY_AVAILABLE = True
    # Try to load the model
    try:
        # ✅ Enable GPU if available
        if spacy.prefer_gpu():
            print("✅ spaCy using GPU acceleration")
        else:
            print("ℹ️  spaCy using CPU (GPU not available)")
        
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("⚠️  spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        SPACY_AVAILABLE = False
        nlp = None
except ImportError:
    print("⚠️  spaCy not installed. Install with: pip install spacy")
    SPACY_AVAILABLE = False
    nlp = None


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
    def preprocess_query(cls, query: str) -> str:
        """
        Preprocess and normalize user query.
        
        - Remove leading/trailing whitespace
        - Normalize multiple spaces to single space
        - Remove unsupported special characters (keep only alphanumeric, basic punctuation)
        - Smart case normalization: use spaCy to detect proper nouns and normalize only if needed
        
        Returns:
            Cleaned query string
        """
        if not query:
            return ""
        
        # Remove leading/trailing whitespace
        query = query.strip()
        
        # Normalize multiple spaces to single space
        query = re.sub(r'\s+', ' ', query)
        
        # Remove unsupported characters - keep only:
        # - Letters (a-z, A-Z, including accented characters)
        # - Numbers (0-9)
        # - Basic punctuation (. , ! ? - ' " : ;)
        # - Whitespace
        query = re.sub(r'[^\w\s.,!?\-\'\":;àáâãäåèéêëìíîïòóôõöùúûüýÿñçÀÁÂÃÄÅÈÉÊËÌÍÎÏÒÓÔÕÖÙÚÛÜÝŸÑÇ]', '', query)
        
        # Remove trailing punctuation that might interfere with queries
        query = re.sub(r'[.!?]+$', '', query)
        
        # ✅ Use spaCy to detect and normalize proper nouns
        if SPACY_AVAILABLE and nlp is not None:
            query = cls._normalize_proper_nouns_with_spacy(query)
        else:
            # Fallback to quoted string normalization only
            query = cls._normalize_quoted_strings(query)
        
        return query.strip()
    
    @classmethod
    def _normalize_proper_nouns_with_spacy(cls, query: str) -> str:
        """
        Use spaCy NER to detect named entities (movies, people) and normalize them intelligently.
        Combines NER with pattern matching for better coverage.
        """
        try:
            # Process with spaCy
            doc = nlp(query)
            
            # Track replacements to apply (store as tuples of (start, end, replacement))
            replacements = []
            
            # Strategy 1: Use Named Entity Recognition
            for ent in doc.ents:
                # Look for entities that are movies, people, or works of art
                if ent.label_ in ['WORK_OF_ART', 'PERSON', 'ORG', 'GPE']:
                    original_text = ent.text
                    
                    # Only normalize if all lowercase or all uppercase
                    if not cls._is_mixed_case(original_text):
                        from src.main.nl_to_sparql import NLToSPARQL
                        normalized = NLToSPARQL._normalize_proper_name(original_text)
                        
                        if normalized != original_text:
                            replacements.append((ent.start_char, ent.end_char, normalized))
                            print(f"[Preprocessing] Normalizing entity ({ent.label_}): '{original_text}' → '{normalized}'")
            
            # Strategy 2: Pattern-based detection for missed entities
            # This catches "the movie X" or "director Y" patterns that NER might miss
            replacements.extend(cls._detect_contextual_entities(query, doc))
            
            # Strategy 3: Quoted strings (always considered proper nouns)
            replacements.extend(cls._detect_quoted_entities(query))
            
            # Remove overlapping replacements (keep longer spans)
            replacements = cls._remove_overlapping_replacements(replacements)
            
            # Apply replacements in reverse order to maintain correct indices
            result = query
            for start, end, replacement in reversed(sorted(replacements)):
                result = result[:start] + replacement + result[end:]
            
            return result
            
        except Exception as e:
            print(f"⚠️  spaCy processing failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to quoted string normalization
            return cls._normalize_quoted_strings(query)
    
    @classmethod
    def _detect_contextual_entities(cls, query: str, doc) -> list:
        """
        Detect entities based on context patterns like "the movie X" or "director Y".
        Returns list of (start, end, replacement) tuples.
        """
        replacements = []
        
        # Patterns: keywords followed by potential proper nouns
        # Match: (the movie|the film|director|actor|actress) ENTITY_TEXT
        pattern = r'\b((?:the\s+)?(?:movie|film)|director|actor|actress|person)\s+([a-zA-Z][a-zA-Z0-9\s\-:]+?)(?=\s*[?.!,]|$)'
        
        for match in re.finditer(pattern, query, re.IGNORECASE):
            keyword = match.group(1)
            entity_text = match.group(2).strip()
            
            # Only normalize if all lowercase or all uppercase
            if entity_text and not cls._is_mixed_case(entity_text):
                from src.main.nl_to_sparql import NLToSPARQL
                normalized = NLToSPARQL._normalize_proper_name(entity_text)
                
                if normalized != entity_text:
                    # Calculate character positions for the entity part only (not the keyword)
                    entity_start = match.start(2)
                    entity_end = match.end(2)
                    
                    replacements.append((entity_start, entity_end, normalized))
                    print(f"[Preprocessing] Normalizing contextual entity: '{entity_text}' → '{normalized}'")
        
        return replacements
    
    @classmethod
    def _detect_quoted_entities(cls, query: str) -> list:
        """
        Detect quoted strings and treat them as proper nouns.
        Returns list of (start, end, replacement) tuples.
        """
        replacements = []
        
        # Match single and double quoted strings
        for quote_char in ["'", '"']:
            pattern = f'{quote_char}([^{quote_char}]+){quote_char}'
            
            for match in re.finditer(pattern, query):
                quoted_text = match.group(1)
                
                # Only normalize if all lowercase or all uppercase
                if quoted_text and not cls._is_mixed_case(quoted_text):
                    from src.main.nl_to_sparql import NLToSPARQL
                    normalized = NLToSPARQL._normalize_proper_name(quoted_text)
                    
                    if normalized != quoted_text:
                        # Replace only the content inside quotes, keep the quotes
                        replacements.append((match.start(1), match.end(1), normalized))
                        print(f"[Preprocessing] Normalizing quoted entity: '{quoted_text}' → '{normalized}'")
        
        return replacements
    
    @staticmethod
    def _remove_overlapping_replacements(replacements: list) -> list:
        """
        Remove overlapping replacements, keeping the longer/more specific ones.
        Replacements are (start, end, text) tuples.
        """
        if not replacements:
            return []
        
        # Sort by start position, then by length (descending)
        sorted_replacements = sorted(replacements, key=lambda x: (x[0], -(x[1] - x[0])))
        
        result = []
        last_end = -1
        
        for start, end, text in sorted_replacements:
            # Skip if this replacement overlaps with the previous one
            if start < last_end:
                continue
            
            result.append((start, end, text))
            last_end = end
        
        return result
    
    @staticmethod
    def _is_mixed_case(text: str) -> bool:
        """
        Check if text has mixed case (some uppercase, some lowercase).
        This indicates the text is already properly capitalized.
        
        Returns:
            True if text has mixed case (already proper), False if all lower/upper
        """
        # Remove spaces and punctuation for checking
        letters_only = ''.join(c for c in text if c.isalpha())
        
        if not letters_only:
            return False
        
        has_upper = any(c.isupper() for c in letters_only)
        has_lower = any(c.islower() for c in letters_only)
        
        # Mixed case = already proper capitalization
        return has_upper and has_lower
    
    @classmethod
    def validate(cls, query: str) -> dict:
        """
        Validate user input for security and processability.
        
        Returns:
            dict with 'is_valid', 'message', 'threats', and 'cleaned_query' keys
        """
        threats = []
        
        # Preprocess the query first
        cleaned_query = cls.preprocess_query(query)
        
        # Check if query is empty or just whitespace after preprocessing
        if not cleaned_query:
            return {
                'is_valid': False,
                'message': "Query cannot be empty.",
                'threats': ['empty_input'],
                'cleaned_query': cleaned_query
            }
        
        # Check length (use cleaned query)
        if len(cleaned_query) > cls.MAX_QUERY_LENGTH:
            return {
                'is_valid': False,
                'message': f"Query is too long (max {cls.MAX_QUERY_LENGTH} characters).",
                'threats': ['excessive_length'],
                'cleaned_query': cleaned_query
            }
        
        if len(cleaned_query) < cls.MIN_QUERY_LENGTH:
            return {
                'is_valid': False,
                'message': "Query is too short.",
                'threats': ['insufficient_length'],
                'cleaned_query': cleaned_query
            }
        
        # Check for malicious patterns (use original for security check)
        for pattern in cls.MALICIOUS_PATTERNS:
            if re.search(pattern, query):
                threats.append(f"malicious_pattern: {pattern}")
        
        # Check for suspicious characters (use original)
        for char_seq in cls.SUSPICIOUS_CHARS:
            if char_seq in query:
                threats.append(f"suspicious_chars: {char_seq}")
        
        # Check for excessive repetition (use cleaned)
        if re.search(r'(.)\1{50,}', cleaned_query):
            threats.append('excessive_repetition')
        
        if threats:
            return {
                'is_valid': False,
                'message': "Query contains potentially malicious or invalid content.",
                'threats': threats,
                'cleaned_query': cleaned_query
            }
        
        return {
            'is_valid': True,
            'message': "Query is valid.",
            'threats': [],
            'cleaned_query': cleaned_query
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
        
        # Replace raw_query with cleaned version for downstream processing
        if validation_result.get('cleaned_query'):
            original_query = state['raw_query']
            state['raw_query'] = validation_result['cleaned_query']
            print(f"[NODE: validate_input] Cleaned query: {state['raw_query'][:50]}...")
            if original_query != state['raw_query']:
                print(f"[NODE: validate_input] ℹ️  Query was normalized (spaces, special chars)")
        
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
        
        # Check for validation errors (security-related)
        if state.get('detected_threats'):
            return (
                "⚠️ **Security Warning**\n\n"
                "Your query contains potentially unsafe content and cannot be processed.\n\n"
                "Please rephrase your question using natural language only."
            )
        # Use formatter for all other errors (now with generic messages)
        return self.formatter.format_error(error)
