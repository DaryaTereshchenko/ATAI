from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery
from typing import Dict, List, Any, Optional
from pyparsing import ParseException 
import re
from config import GRAPH_FILE_PATH
import logging

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
            validation_result['message'] = "Query must be a non-empty string"
            return validation_result
        
        # Check if it's empty or just whitespace
        if not query.strip():
            validation_result['message'] = "Query cannot be empty"
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
                return result
            result['query_info'] = validation
        
        # Execute query using rdflib
        try:
            query_result = self.graph.query(query)
            
            # Convert to JSON-like format
            if self.return_format.lower() == "json":
                bindings = []
                for row in query_result:
                    binding = {}
                    for var in query_result.vars:
                        value = row[var]
                        if value is not None:
                            binding[str(var)] = {
                                'type': 'uri' if hasattr(value, 'n3') and value.n3().startswith('<') else 'literal',
                                'value': str(value)
                            }
                    bindings.append(binding)
                
                result['data'] = {
                    'results': {
                        'bindings': bindings
                    }
                }
            else:
                result['data'] = list(query_result)
            
            result['success'] = True
            
        except ParseException as e:
            result['error'] = f"Malformed query: {str(e)}"
        except Exception as e:
            result['error'] = f"Unexpected error: {str(e)}"
        
        return result
    
    def execute_and_format(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute query and return formatted results.
        
        Args:
            query: The SPARQL query string
            
        Returns:
            List of result dictionaries
        """
        execution_result = self.execute_query(query)
        
        if not execution_result['success']:
            raise Exception(f"Query execution failed: {execution_result['error']}")
        
        # Format JSON results
        if self.return_format.lower() == "json":
            data = execution_result['data']
            if 'results' in data and 'bindings' in data['results']:
                return data['results']['bindings']
        
        return execution_result['data']
    
    def get_query_results_as_list(self, query: str, variable: Optional[str] = None) -> List[str]:
        """
        Execute query and extract a specific variable as a list.
        
        Args:
            query: The SPARQL query string
            variable: The variable name to extract (without '?')
            
        Returns:
            List of values for the specified variable
        """
        results = self.execute_and_format(query)
        
        if not variable:
            # Auto-detect first variable
            if results and len(results) > 0:
                variable = list(results[0].keys())[0]
            else:
                return []
        
        return [
            result[variable]['value'] 
            for result in results 
            if variable in result
        ]


# Example usage
if __name__ == "__main__":
    # Your example query
    query = """
        PREFIX ddis: <http://ddis.ch/atai/>   

        PREFIX wd: <http://www.wikidata.org/entity/>   

        PREFIX wdt: <http://www.wikidata.org/prop/direct/>   

        PREFIX schema: <http://schema.org/>   

        

        SELECT ?director WHERE {  

            ?movie rdfs:label "Apocalypse Now"@en .  

                ?movie wdt:P57 ?directorItem . 

            ?directorItem rdfs:label ?director . 

        }  

        LIMIT 1  
    """
    
    # Initialize handler
    handler = SPARQLHandler(
        graph_file_path="data/graph.nt",
        return_format="json"
    )
    
    # Validate query
    validation = handler.validate_query(query)
    logging.info(f"Validation Result: {validation}")

    # Execute query
    result = handler.execute_query(query)
    if result['success']:
        logging.info("Query executed successfully!")
        logging.info(f"Results: {result['data']}")
    else:
        logging.info("Query execution failed:")
        logging.error(f"Error: {result['error']}")

    # Get specific results
    try:
        labels = handler.get_query_results_as_list(query, 'lbl')
        logging.info(f"Extracted labels: {labels}")
    except Exception as e:
        logging.error(f"Error getting results: {e}")