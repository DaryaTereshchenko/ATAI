from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery
from typing import Dict, List, Any, Optional
from pyparsing import ParseException 
import re
import logging
import sys 
import os
import signal
from contextlib import contextmanager

# Add project root to path (go up two levels from this file)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from config import GRAPH_FILE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SPARQLHandler:
    """Handles SPARQL query execution against the local knowledge graph."""
    
    # Security constants
    MAX_QUERY_LENGTH = 10000
    MAX_TRIPLE_PATTERNS = 50
    MAX_QUERY_DEPTH = 10
    QUERY_TIMEOUT_SECONDS = 30
    
    # Dangerous SPARQL operations that modify data
    DANGEROUS_OPERATIONS = [
        'INSERT', 'DELETE', 'DROP', 'CLEAR', 'CREATE', 
        'LOAD', 'COPY', 'MOVE', 'ADD', 'UPDATE'
    ]
    
    # Patterns that may cause infinite loops or excessive computation
    RISKY_PATTERNS = [
        r'OPTIONAL\s*{\s*OPTIONAL',  # Nested OPTIONAL
        r'\*\s+\w+\s+\*',  # Multiple wildcards in property paths
        r'\+{2,}',  # Excessive path operators
        r'\*{2,}',  # Multiple consecutive wildcards
        r'UNION.*UNION.*UNION.*UNION',  # Excessive UNIONs
    ]
    
    def __init__(self, graph_file_path: str = None, return_format: str = "json"):
        """
        Initialize the SPARQL handler with a local graph file.
        
        Args:
            graph_file_path: Path to the N-Triples graph file.
            return_format: The desired return format for query results (e.g., "json", "xml").
        """
        self.graph_file_path = graph_file_path or GRAPH_FILE_PATH
        self.return_format = return_format
        self.graph = Graph()
        self._load_graph()
    
    def _load_graph(self):
        """Load the knowledge graph from file."""
        logger.info(f"Loading graph from {self.graph_file_path}...")
        try:
            self.graph.parse(self.graph_file_path, format="nt")
            logger.info(f"Graph loaded successfully. Contains {len(self.graph)} triples.")
        except Exception as e:
            logger.error(f"Error loading graph: {e}")
            raise

    def _check_security_violations(self, query: str) -> Dict[str, Any]:
        """
        Check for security violations in the query.
        
        Args:
            query: The SPARQL query string
            
        Returns:
            Dict with 'safe' (bool), 'violation_type' (str), and 'message' (str)
        """
        security_result = {
            'safe': True,
            'violation_type': None,
            'message': ''
        }
        
        # Check query length
        if len(query) > self.MAX_QUERY_LENGTH:
            security_result['safe'] = False
            security_result['violation_type'] = 'excessive_length'
            security_result['message'] = (
                f"Query exceeds maximum length of {self.MAX_QUERY_LENGTH} characters. "
                "Please provide a shorter query."
            )
            return security_result
        
        # Check for dangerous operations
        query_upper = query.upper()
        for operation in self.DANGEROUS_OPERATIONS:
            if re.search(rf'\b{operation}\b', query_upper):
                security_result['safe'] = False
                security_result['violation_type'] = 'dangerous_operation'
                security_result['message'] = (
                    f"Query contains forbidden operation '{operation}'. "
                    "Only SELECT, ASK, CONSTRUCT, and DESCRIBE queries are allowed. "
                    "Please provide a read-only query."
                )
                return security_result
        
        # Check for risky patterns that may cause infinite loops
        for pattern in self.RISKY_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                security_result['safe'] = False
                security_result['violation_type'] = 'infinite_loop_risk'
                security_result['message'] = (
                    "Query contains patterns that may cause infinite loops or excessive computation. "
                    "Please simplify your query and avoid nested optional clauses or excessive wildcards."
                )
                return security_result
        
        # Count triple patterns to prevent complexity attacks
        triple_pattern_count = len(re.findall(r'\{[^}]*\?', query))
        if triple_pattern_count > self.MAX_TRIPLE_PATTERNS:
            security_result['safe'] = False
            security_result['violation_type'] = 'excessive_complexity'
            security_result['message'] = (
                f"Query is too complex with {triple_pattern_count} triple patterns. "
                f"Maximum allowed is {self.MAX_TRIPLE_PATTERNS}. "
                "Please simplify your query."
            )
            return security_result
        
        # Check for excessive nesting depth
        max_depth = 0
        current_depth = 0
        for char in query:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth -= 1
        
        if max_depth > self.MAX_QUERY_DEPTH:
            security_result['safe'] = False
            security_result['violation_type'] = 'excessive_nesting'
            security_result['message'] = (
                f"Query has excessive nesting depth ({max_depth} levels). "
                f"Maximum allowed is {self.MAX_QUERY_DEPTH}. "
                "Please simplify your query structure."
            )
            return security_result
        
        return security_result

    @contextmanager
    def _timeout_handler(self, seconds: int):
        """
        Context manager to handle query timeout.
        
        Args:
            seconds: Timeout duration in seconds
        """
        def timeout_handler(signum, frame):
            raise TimeoutError("Query execution exceeded time limit")
        
        # Set the signal handler and alarm
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            # Restore the old handler and cancel the alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate a SPARQL query without executing it.
        
        Args:
            query: The SPARQL query string to validate
            
        Returns:
            Dict with 'valid' (bool) and 'message' (str) keys
        """
        validation_result = {
            'valid': False,
            'message': '',
            'query_type': None
        }
        
        # Basic string validation
        if not query or not isinstance(query, str):
            validation_result['message'] = "Query must be a non-empty string"
            return validation_result
        
        # Check if it's empty or just whitespace
        if not query.strip():
            validation_result['message'] = "Query cannot be empty"
            return validation_result
        
        # Security checks
        security_check = self._check_security_violations(query)
        if not security_check['safe']:
            validation_result['message'] = security_check['message']
            validation_result['violation_type'] = security_check['violation_type']
            return validation_result
        
        # Method 1: Use rdflib to parse the query
        try:
            parsed_query = prepareQuery(query)
            validation_result['valid'] = True
            validation_result['query_type'] = parsed_query.algebra.name
            validation_result['message'] = f"Valid {parsed_query.algebra.name} query"
            return validation_result
        except Exception as e:
            validation_result['message'] = f"Query parsing error: {str(e)}"
        
        # Method 2: Basic regex pattern matching (fallback)
        sparql_pattern = re.compile(
            r'(PREFIX\s+\w+:\s*<[^>]+>\s*)*(SELECT|ASK|CONSTRUCT|DESCRIBE)',
            re.IGNORECASE | re.MULTILINE
        )
        
        if not sparql_pattern.search(query):
            validation_result['message'] = "Query doesn't match basic SPARQL structure"
            return validation_result
        
        return validation_result

    def execute_query(self, query: str, validate: bool = True) -> Dict[str, Any]:
        """
        Execute a SPARQL query after optional validation.
        
        This actually queries the RDF graph loaded from graph.nt
        """
        # Validate first
        if validate:
            validation = self.validate_query(query)
            if not validation['valid']:
                return {'success': False, 'error': validation['message']}
    
        # ✅ Execute query against the loaded RDF graph
        query_result = self.graph.query(query)  # ← This runs the SPARQL query on the database
        results_list = list(query_result)
        
        # Extract and format results
        answers = []
        for row in results_list:
            values = [str(val) for val in row if val is not None]
            if values:
                answers.append(", ".join(values) if len(values) > 1 else values[0])
        
        return {
            'success': True,
            'data': "\n".join(answers) if answers else "No answer found in the database."
        }

    def execute_and_format(self, query: str) -> str:
        """Execute query and return formatted results."""
        execution_result = self.execute_query(query)
        
        if not execution_result['success']:
            raise Exception(f"Query execution failed: {execution_result['error']}")
        
        return execution_result['data']
