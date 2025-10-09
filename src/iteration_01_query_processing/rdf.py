from __future__ import annotations
import os
import os
from pathlib import Path
import sys
from typing import List, Optional, Union

# Add project root to path (go up two levels from this file)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)


try:
    from rdflib import Graph, URIRef, Literal, BNode
except ImportError as exc:
    raise ImportError(
        "rdflib is required to use the RDFStore class.\n"
        "Install with: pip install rdflib"
    ) from exc


class RDFStore:
    _DEFAULT_FILENAMES: List[str] = ["graph.nt"]

    def __init__(self, graph_path: Optional[str] = None, graph_format: Optional[str] = None) -> None:
        candidate_path: Optional[str] = graph_path

        if candidate_path is None:
            env_path = os.getenv("ATAI_KG_PATH")
            if env_path and os.path.exists(env_path):
                candidate_path = env_path

        if candidate_path is None:
            # Recursively search for default filenames in the working directory
            cwd = Path.cwd()
            found: Optional[Path] = None
            for root, _, files in os.walk(cwd):
                for fname in self._DEFAULT_FILENAMES:
                    if fname in files:
                        found = Path(root) / fname
                        break
                if found:
                    break
            if found:
                candidate_path = str(found)

        if not candidate_path:
            raise FileNotFoundError(
                "No knowledge graph file was found. "
                "Set ATAI_KG_PATH, pass graph_path to RDFStore, or place 'graph.nt' somewhere under the working directory."
            )

        # Guess rdflib format when not provided
        chosen_format = graph_format
        if chosen_format is None:
            ext = os.path.splitext(candidate_path)[1].lower()
            if ext in {".ttl", ".turtle"}:
                chosen_format = "turtle"
            elif ext in {".nt", ".ntriples", ".nq", ".nt11"}:
                chosen_format = "nt"
            elif ext in {".rdf", ".xml"}:
                chosen_format = "xml"

  
        graph = Graph()
        path = Path(candidate_path).expanduser().resolve()
        graph.parse(path.as_uri(), format=chosen_format)
        self.graph = graph

    def run_query(self, query: str) -> List[Union[str, List[Union[str, int, float]]]]:
        
        result_rows = self.graph.query(query)
        processed: List[Union[str, List[Union[str, int, float]]]] = []
        for row in result_rows:
            if not isinstance(row, tuple):
                row = (row,)  # type: ignore
            converted: List[Union[str, int, float]] = []
            for term in row:
                if isinstance(term, Literal):
                    converted.append(term.toPython())
                elif isinstance(term, (URIRef, BNode)):
                    converted.append(str(term))
                else:
                    converted.append(str(term))
            processed.append(converted[0] if len(converted) == 1 else converted)
        return processed
