from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery
from typing import Dict, List, Any, Optional
from pyparsing import ParseException 
import re
import logging
import sys 
import os

# Add project root to path (go up two levels from this file)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from config import GRAPH_FILE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SPARQLHandler:
    """Handles SPARQL query execution against the local knowledge graph."""
    
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
            validation_result['message'] = "Please provide a valid query."
            return validation_result
        
        # Check if it's empty or just whitespace
        if not query.strip():
            validation_result['message'] = "Your query appears to be empty. Please try again."
            return validation_result
        
        # Method 1: Use rdflib to parse the query
        try:
            parsed_query = prepareQuery(query)
            validation_result['valid'] = True
            validation_result['query_type'] = parsed_query.algebra.name
            validation_result['message'] = f"Valid {parsed_query.algebra.name} query"
            return validation_result
        except Exception as e:
            validation_result['message'] = "I couldn't understand your query. Please check that it's properly formatted and try again."
        
        # Method 2: Basic regex pattern matching (fallback)
        sparql_pattern = re.compile(
            r'(PREFIX\s+\w+:\s*<[^>]+>\s*)*(SELECT|ASK|CONSTRUCT|DESCRIBE)',
            re.IGNORECASE | re.MULTILINE
        )
        
        if not sparql_pattern.search(query):
            validation_result['message'] = "Your query doesn't seem to be in the correct format. Please verify and try again."
            return validation_result
        
        return validation_result

    def execute_query(self, query: str, validate: bool = True) -> Dict[str, Any]:
        """
        Execute a SPARQL query after optional validation.
        
        Args:
            query: The SPARQL query string
            validate: Whether to validate before executing
            
        Returns:
            Dict with execution results or error information
        """
        result = {
            'success': False,
            'data': None,
            'error': None,
            'query_info': None
        }
        
        # Validate first if requested
        if validate:
            validation = self.validate_query(query)
            if not validation['valid']:
                result['error'] = validation['message']
                logger.warning(f"Validation failed: {validation['message']}")
                return result
            result['query_info'] = validation
        
        # Execute query using rdflib
        try:
            logger.info("Executing query against graph...")
            query_result = self.graph.query(query)
            
            # Check if there are any results
            results_list = list(query_result)
            logger.info(f"Query returned {len(results_list)} row(s)")
            
            if not results_list:
                result['data'] = "No answer found in the database."
                result['success'] = True
                logger.info("No results found")
                return result
            
            # Extract simple string values
            answers = []
            for i, row in enumerate(results_list):
                logger.debug(f"Processing row {i}: {row}")
                # Get all non-None values from the row
                values = [str(val) for val in row if val is not None]
                if values:
                    # If single value, add it directly; otherwise join with comma
                    if len(values) == 1:
                        answers.append(values[0])
                    else:
                        answers.append(", ".join(values))
            
            # Return concatenated answer or no answer message
            if answers:
                result['data'] = "\n".join(answers) if len(answers) > 1 else answers[0]
                logger.info(f"Formatted answer: {result['data']}")
            else:
                result['data'] = "No answer found in the database."
                logger.info("No values extracted from results")
            
            result['success'] = True
            
        except ParseException as e:
            result['error'] = f"Malformed query: {str(e)}"
            logger.error(f"ParseException: {e}", exc_info=True)
        except Exception as e:
            result['error'] = f"Unexpected error: {str(e)}"
            logger.error(f"Unexpected error: {e}", exc_info=True)
        
        return result
    
    def execute_and_format(self, query: str) -> str:
        """
        Execute query and return formatted results.
        
        Args:
            query: The SPARQL query string
            
        Returns:
            String with result or error message
        """
        execution_result = self.execute_query(query)
        
        if not execution_result['success']:
            raise Exception(f"Query execution failed: {execution_result['error']}")
        
        return execution_result['data']
