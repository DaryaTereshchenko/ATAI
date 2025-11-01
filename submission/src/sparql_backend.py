# src/sparql_backend.py
from __future__ import annotations
from typing import List, Optional
import re

try:
    from rdflib import Graph
except Exception:
    Graph = None  # graceful fallback if rdflib is not present


RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"

def _escape_quote(s: str) -> str:
    # Escape " characters inside literals
    return s.replace('"', '\\"')


class SPARQLBackend:
    """
    Minimal SPARQL frontend:
      • Builds VALID SPARQL strings for single-hop queries
      • Executes them via rdflib if available
    Works directly on your local graph.nt.
    """
    def __init__(self, nt_path: str, lang: str = "en"):
        self.nt_path = nt_path
        self.lang = lang
        self._g = None
        if Graph is not None:
            self._g = Graph()
            # rdflib autodetects N-Triples; this is streaming-friendly for large graphs.
            self._g.parse(self.nt_path, format="nt")

    # -----------------------------
    # Query construction utilities
    # -----------------------------

    def build_find_objects_query(
        self,
        subject_label: str,
        predicate_iri: str,
        return_labels: bool = True,
    ) -> str:
        """
        Single-hop: given S(label) and P, retrieve all O.
        If O is an IRI and return_labels=True, return English labels, else return the IRI/lexical value.
        """
        s_lit = _escape_quote(subject_label)
        # We support labels tagged with @en *or* untagged literals to be robust.
        if return_labels:
            return f"""
SELECT DISTINCT ?answer WHERE {{{{
  ?s <{RDFS_LABEL}> "{s_lit}"@{self.lang} .
  ?s <{predicate_iri}> ?o .
  OPTIONAL {{ ?o <{RDFS_LABEL}> ?oLabel .
            FILTER(LANG(?oLabel) = "{self.lang}")
          }}
  BIND(COALESCE(?oLabel, STR(?o)) AS ?answer)
}}}}
        """.strip()
    def build_find_subjects_query(
        self,
        object_label: str,
        predicate_iri: str,
        return_labels: bool = True,
    ) -> str:
        """
        Reverse single-hop: given O(label) and P, retrieve all S such that S P O.
        """
        o_lit = _escape_quote(object_label)
        if return_labels:
            return f"""
SELECT DISTINCT ?answer WHERE {{{{
  ?o <{RDFS_LABEL}> "{o_lit}"@{self.lang} .
  ?s <{predicate_iri}> ?o .
  OPTIONAL {{ ?s <{RDFS_LABEL}> ?sLabel .
            FILTER(LANG(?sLabel) = "{self.lang}")
          }}
  BIND(COALESCE(?sLabel, STR(?s)) AS ?answer)
}}}}
        """.strip()
    def build_find_literal_query(
        self,
        subject_label: str,
        predicate_iri: str,
    ) -> str:
        """
        For literal-valued properties (e.g., dates): given S(label) and P, get literal O.
        """
        s_lit = _escape_quote(subject_label)
        return f"""
SELECT DISTINCT ?o WHERE {{{{
  ?s <{RDFS_LABEL}> "{s_lit}"@{self.lang} .
  ?s <{predicate_iri}> ?o .
}}}}
        """.strip()

    def build_find_literal_query(
        self,
        subject_label: str,
        predicate_iri: str,
    ) -> str:
        """
        For literal-valued properties (e.g., dates): given S(label) and P, get literal O.
        """
        s_lit = _escape_quote(subject_label)
        return f"""
SELECT DISTINCT ?o WHERE {{
  ?s <{RDFS_LABEL}> "{s_lit}"@{self.lang} .
  ?s <{predicate_iri}> ?o .
}}
        """.strip()

    # -----------------------------
    # Execution
    # -----------------------------

    def execute(self, query: str) -> List[str]:
        """
        Execute the SPARQL query with rdflib if available, else raise a helpful error.
        """
        if self._g is None:
            raise RuntimeError(
                "rdflib is not installed in this environment; "
                "please `pip install rdflib` to execute SPARQL queries."
            )
        rows = self._g.query(query)
        out = []
        for row in rows:
            # We name the selected variable either ?answer, ?o, or ?s above.
            for var in ("answer", "o", "s"):
                if var in rows.vars:
                    val = row[rows.vars.index(var)]
                    out.append(str(val))
                    break
        return out
