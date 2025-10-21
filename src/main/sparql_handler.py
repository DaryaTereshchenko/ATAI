from rdflib import Graph, Literal
from rdflib.namespace import RDFS
from rdflib.plugins.sparql import prepareQuery
from typing import Dict, List, Any, Optional
from pyparsing import ParseException
import re
import logging
import sys
import os
import signal
from contextlib import contextmanager
from functools import lru_cache
from collections import defaultdict

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
        r'OPTIONAL\s*{\s*OPTIONAL',       # Nested OPTIONAL
        r'\*\s+\w+\s+\*',                 # Multiple wildcards in property paths
        r'\+{2,}',                        # Excessive path operators
        r'\*{2,}',                        # Multiple consecutive wildcards
        r'UNION.*UNION.*UNION.*UNION',    # Excessive UNIONs
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

        # Lowercased label -> {canonical variants}
        # Enables fast snap-back to your graph's exact casing from NL layer.
        self.label_index: Dict[str, set] = defaultdict(set)

        self._load_graph()

    def _load_graph(self):
        """Load the knowledge graph from file and build label index."""
        logger.info(f"Loading graph from {self.graph_file_path}...")
        try:
            self.graph.parse(self.graph_file_path, format="nt")
            logger.info(f"Graph loaded successfully. Contains {len(self.graph)} triples.")

            # Build label index once (helps NL layer "snap" titles to canonical casing)
            cnt = 0
            for s, p, o in self.graph.triples((None, RDFS.label, None)):
                if isinstance(o, Literal):
                    lbl = str(o)
                    self.label_index[lbl.casefold()].add(lbl)
                    cnt += 1
            logger.info(f"Label index built with {cnt} labels.")
        except Exception as e:
            logger.error(f"Error loading graph: {e}")
            raise

    def _check_security_violations(self, query: str) -> Dict[str, Any]:
        """
        Check for security violations in the query.
        """
        security_result = {
            'safe': True,
            'violation_type': None,
            'message': ''
        }

        # Length
        if len(query) > self.MAX_QUERY_LENGTH:
            return {
                'safe': False,
                'violation_type': 'excessive_length',
                'message': (
                    f"Query exceeds maximum length of {self.MAX_QUERY_LENGTH} characters. "
                    "Please provide a shorter query."
                )
            }

        # Dangerous ops
        query_upper = query.upper()
        for operation in self.DANGEROUS_OPERATIONS:
            if re.search(rf'\b{operation}\b', query_upper):
                return {
                    'safe': False,
                    'violation_type': 'dangerous_operation',
                    'message': (
                        f"Query contains forbidden operation '{operation}'. "
                        "Only SELECT, ASK, CONSTRUCT, and DESCRIBE queries are allowed. "
                        "Please provide a read-only query."
                    )
                }

        # Risky patterns
        for pattern in self.RISKY_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return {
                    'safe': False,
                    'violation_type': 'infinite_loop_risk',
                    'message': (
                        "Query contains patterns that may cause infinite loops or excessive computation. "
                        "Please simplify your query and avoid nested optional clauses or excessive wildcards."
                    )
                }

        # Approx triple pattern count
        triple_pattern_count = len(re.findall(r'\{[^}]*\?', query))
        if triple_pattern_count > self.MAX_TRIPLE_PATTERNS:
            return {
                'safe': False,
                'violation_type': 'excessive_complexity',
                'message': (
                    f"Query is too complex with {triple_pattern_count} triple patterns. "
                    f"Maximum allowed is {self.MAX_TRIPLE_PATTERNS}. "
                    "Please simplify your query."
                )
            }

        # Nesting depth
        max_depth = 0
        current_depth = 0
        for char in query:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth -= 1

        if max_depth > self.MAX_QUERY_DEPTH:
            return {
                'safe': False,
                'violation_type': 'excessive_nesting',
                'message': (
                    f"Query has excessive nesting depth ({max_depth} levels). "
                    f"Maximum allowed is {self.MAX_QUERY_DEPTH}. "
                    "Please simplify your query structure."
                )
            }

        return security_result

    @contextmanager
    def _timeout_handler(self, seconds: int):
        """Context manager to enforce query timeout (POSIX)."""
        def timeout_handler(signum, frame):
            raise TimeoutError("Query execution exceeded time limit")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate a SPARQL query without executing it.
        """
        validation_result = {
            'valid': False,
            'message': '',
            'query_type': None
        }

        if not query or not isinstance(query, str):
            validation_result['message'] = "Query must be a non-empty string"
            return validation_result

        if not query.strip():
            validation_result['message'] = "Query cannot be empty"
            return validation_result

        # Security checks first
        security_check = self._check_security_violations(query)
        if not security_check['safe']:
            validation_result['message'] = security_check['message']
            validation_result['violation_type'] = security_check['violation_type']
            return validation_result

        # Parse with rdflib
        try:
            pq = prepareQuery(query)
            validation_result['valid'] = True
            validation_result['query_type'] = getattr(pq.algebra, "name", "SELECT")
            validation_result['message'] = f"Valid {validation_result['query_type']} query"
            return validation_result
        except Exception as e:
            validation_result['message'] = f"Query parsing error: {str(e)}"

        # Fallback: superficial structure check
        if not re.search(r'(PREFIX\s+\w+:\s*<[^>]+>\s*)*(SELECT|ASK|CONSTRUCT|DESCRIBE)', query, re.I):
            validation_result['message'] = "Query doesn't match basic SPARQL structure"
            return validation_result

        return validation_result

    # Small cache for repeat queries (speeds up iterative dev/testing)
    @lru_cache(maxsize=256)
    def _run_query_cached(self, query: str) -> List[Any]:
        return list(self.graph.query(query))

    def execute_query(self, query: str, validate: bool = True) -> Dict[str, Any]:
        """
        Execute a SPARQL query after optional validation.
        This actually queries the RDF graph loaded from graph.nt
        """
        if validate:
            validation = self.validate_query(query)
            if not validation['valid']:
                return {'success': False, 'error': validation.get('message', 'Invalid query')}

        try:
            with self._timeout_handler(self.QUERY_TIMEOUT_SECONDS):
                results_list = self._run_query_cached(query)
        except TimeoutError as te:
            return {'success': False, 'error': str(te)}
        except Exception as e:
            return {'success': False, 'error': f"Execution error: {e}"}

        # Extract and format results (simple text; keep as-is)
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

    # -------- Optional helpers you can use from NLToSPARQL --------

    def snap_label(self, text: str) -> str:
        """
        Given a user-provided label, return the graph's canonical-cased label if known.
        Otherwise return the input unchanged.
        """
        if not text:
            return text
        cand = self.label_index.get(text.casefold())
        if not cand:
            return text
        # choose the longest variant (often most complete)
        return sorted(cand, key=len, reverse=True)[0]

    def add_lang_filter(self, query: str, label_var_names: Optional[List[str]] = None, lang: str = "en") -> str:
        """
        (Optional) Add a LANG filter for given label variables (e.g., ?movieLabel, ?personLabel).
        Use this only if your data is multilingual and you want to prefer a language.
        """
        if not label_var_names:
            return query
        lines = query.splitlines()
        injected = []
        for i, line in enumerate(lines):
            injected.append(line)
            for v in label_var_names:
                # after a line that binds the label variable, add a language guard if not already present
                if re.search(rf'\b{re.escape(v)}\b', line) and 'rdfs:label' in line:
                    injected.append(f'FILTER(LANGMATCHES(LANG({v}), "{lang}") || LANG({v}) = "") .')
        return "\n".join(injected)
