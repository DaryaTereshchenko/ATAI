"""
kg_utils.py
-----------------

Utility functions and classes for working with the provided knowledge graph.

The knowledge graph is stored as an N‑Triples file containing subject–predicate–object
triples.  For the purposes of this project we do not depend on external RDF
libraries (such as rdflib) because they may not be available in the execution
environment.  Instead, we implement a lightweight loader that reads triples
line‑by‑line and stores them in a simple in‑memory structure.  Only a subset of
the graph is loaded – we index by subject and by object to support one‑hop
queries.

The loader also extracts English labels (rdfs:label) for entities to allow
looking up IRIs from natural language questions.  When multiple entities share
the same label the first encountered is used.

Example usage:

>>> kg = KnowledgeGraph("/path/to/graph.nt")
>>> kg.load()
>>> kg.get_objects_for("Fargo", "director")
['Joel Coen', 'Ethan Coen']

The graph is large; loading may take tens of seconds.  For production use you
may wish to serialize the index to disk for faster subsequent startup.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


class KnowledgeGraph:
    """Simple wrapper around a set of RDF triples stored in N‑Triples format.

    This class supports retrieving objects given a subject and predicate (for
    forward edges) and retrieving subjects given an object and predicate (for
    reverse edges).  It also maintains a mapping between entity IRIs and their
    English labels (rdfs:label) to enable name lookup from natural language.
    """

    # Regular expressions to parse N‑Triple lines.  This simplistic parser is
    # sufficient for our controlled dataset; it does not handle escaped
    # characters within literals or IRIs.  We intentionally avoid external
    # dependencies.
    _triple_re = re.compile(
        r"^\s*<([^>]*)>\s+<([^>]*)>\s+(?:<([^>]*)>|\"([^\"]*)\"(?:@([a-zA-Z\-]+))?)\s*\.\s*$"
    )

    def __init__(self, path: str) -> None:
        self.path = path
        # Mapping from (subject, predicate) to list of object IRIs or literal
        # strings.  We store all objects because many predicates are multi
        # valued (e.g., directors of a film).
        self._forward: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        # Reverse mapping from (predicate, object) to list of subject IRIs
        self._reverse: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        # Entity IRI → English label mapping.  Only one label per entity is
        # stored; additional labels are ignored.  This suffices for
        # disambiguation in most one‑hop queries.
        self._labels: Dict[str, str] = {}
        # Label (lowercased) → IRI mapping.  When multiple entities share the
        # same label we record only the first encountered.  Clients should
        # implement their own disambiguation if necessary.
        self._label_to_iri: Dict[str, str] = {}
        self._loaded = False

    def load(self) -> None:
        """Load triples from the N‑Triples file.

        Subsequent calls have no effect.
        """
        if self._loaded:
            return
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"Knowledge graph file not found: {self.path}")
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                match = self._triple_re.match(line)
                if not match:
                    continue
                subj = match.group(1)
                pred = match.group(2)
                obj_iri = match.group(3)
                obj_lit = match.group(4)
                obj_lang = match.group(5)
                # Store triples in forward and reverse indices.
                if obj_iri:
                    self._forward[(subj, pred)].append(obj_iri)
                    self._reverse[(pred, obj_iri)].append(subj)
                elif obj_lit:
                    literal = obj_lit
                    # Append language tag if available (e.g., "@en").
                    if obj_lang:
                        literal = f"{obj_lit}@{obj_lang}"
                    self._forward[(subj, pred)].append(literal)
                    # Literals do not populate the reverse mapping
                # Capture English rdfs:label
                # Detect rdfs:label predicates.  Labels may use either a fragment
                # ('#label') or path ('/label') suffix depending on the
                # namespace.  We normalise by checking both endings.
                if (pred.endswith("/label") or pred.endswith("#label")) and obj_lit and (obj_lang == "en" or obj_lang is None):
                    # Prefer first label encountered
                    if subj not in self._labels:
                        self._labels[subj] = obj_lit
                        key = obj_lit.strip().lower()
                        if key not in self._label_to_iri:
                            self._label_to_iri[key] = subj
        self._loaded = True

    def iri_for_label(self, label: str) -> Optional[str]:
        """Resolve a natural language label to a canonical IRI.

        The lookup is case insensitive.  Returns `None` if the label is not
        present in the graph.
        """
        if not self._loaded:
            self.load()
        return self._label_to_iri.get(label.strip().lower())

    def label_for_iri(self, iri: str) -> Optional[str]:
        """Return the English label for an IRI if available."""
        if not self._loaded:
            self.load()
        return self._labels.get(iri)

    def get_objects(self, subj_label: str, predicate_iri: str) -> List[str]:
        """Retrieve object labels/literals for a given subject and predicate.

        Parameters
        ----------
        subj_label: str
            The human readable label of the subject (e.g., a movie title).
        predicate_iri: str
            The full IRI of the predicate (e.g., http://www.wikidata.org/prop/direct/P57).

        Returns
        -------
        list of str
            A list of object labels or literal strings.  If an object is an
            entity IRI its English label is returned; otherwise the literal
            value is returned (without language tag).  If the subject or
            predicate is not found an empty list is returned.
        """
        if not self._loaded:
            self.load()
        subj_iri = self.iri_for_label(subj_label)
        if not subj_iri:
            return []
        objs = self._forward.get((subj_iri, predicate_iri), [])
        results: List[str] = []
        for o in objs:
            if o.startswith("http://") or o.startswith("https://"):
                # entity
                label = self._labels.get(o)
                if label:
                    results.append(label)
                else:
                    results.append(o)
            else:
                # literal: remove language tag if present
                if "@" in o:
                    lit_value, _lang = o.rsplit("@", 1)
                    results.append(lit_value)
                else:
                    results.append(o)
        return results

    def get_subjects(self, obj_label: str, predicate_iri: str) -> List[str]:
        """Retrieve subject labels for a given object and predicate.

        This is used when the question asks for the entity that has a relation
        pointing to a given object (e.g., "Which film did X direct?").
        """
        if not self._loaded:
            self.load()
        obj_iri = self.iri_for_label(obj_label)
        if not obj_iri:
            return []
        subj_iris = self._reverse.get((predicate_iri, obj_iri), [])
        results: List[str] = []
        for s in subj_iris:
            label = self._labels.get(s)
            if label:
                results.append(label)
            else:
                results.append(s)
        return results
