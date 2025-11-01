"""
relations_mapping.py
---------------------

Defines a mapping between high level relation names used by the question
parser and the full predicate IRIs found in the knowledge graph.  The
``RelationsMapping`` class reads from the ``relations_kg.txt`` file if
available to populate this mapping; otherwise it falls back to a small
hard‑coded dictionary for common film relations (director, screenwriter,
genre, etc.).  It also holds information about the expected answer type for
each relation (e.g., human, film, genre) derived from the same file when
available.

Clients can query this mapping to look up the predicate IRI for a given
relation keyword and to inspect the expected type of the corresponding
objects.  The latter is needed to annotate answers returned by embedding
queries.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional, Tuple


class RelationsMapping:
    """Mapping between short relation names and their predicate IRIs and types."""

    def __init__(self, relations_file: Optional[str] = None) -> None:
        """Initialize the mapping.  If a relations file is provided and
        readable, parse it to populate the mapping; otherwise use defaults."""
        self.relations: Dict[str, Tuple[str, str]] = {}
        if relations_file and os.path.isfile(relations_file):
            self._parse_relations_file(relations_file)
        else:
            self._populate_defaults()

    def _populate_defaults(self) -> None:
        """Populate mapping with a small set of defaults."""
        # The default mapping is based on widely used Wikidata properties.
        self.relations = {
            "director": ("http://www.wikidata.org/prop/direct/P57", "Q5"),  # human
            "screenwriter": ("http://www.wikidata.org/prop/direct/P58", "Q5"),  # human
            "producer": ("http://www.wikidata.org/prop/direct/P162", "Q5"),  # human
            "country_of_origin": ("http://www.wikidata.org/prop/direct/P495", "Q6256"),  # country
            "genre": ("http://www.wikidata.org/prop/direct/P136", "Q201658"),  # genre
            "publication_date": ("http://www.wikidata.org/prop/direct/P577", "xsd:dateTime"),
            "cast_member": ("http://www.wikidata.org/prop/direct/P161", "Q5"),  # human
        }

    def _parse_relations_file(self, path: str) -> None:
        """Parse the provided relations file (relations_kg.txt).

        The file contains lines of the form ``✓ label → iri (n triples)`` or
        ``label → Q5`` in the reverse mapping section.  We extract the
        relation label, predicate IRI and the expected object type (Q‑id) when
        possible.
        """
        pattern = re.compile(r"^\s*✓\s+([^\(]+?)\s+\(from graph label\)\s+→\s+(\S+)")
        type_pattern = re.compile(r"^\s*([a-zA-Z0-9_:]+)\s+→\s+(Q\d+)")
        rels: Dict[str, Tuple[str, str]] = {}
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        # First pass: gather predicate IRIs
        for line in lines:
            m = pattern.match(line)
            if m:
                label = m.group(1).strip()
                iri = m.group(2).strip()
                # normalise label
                rels[label] = (iri, "")
        # Second pass: gather expected types from reverse mapping
        for line in lines:
            m = type_pattern.match(line)
            if m:
                label = m.group(1).strip()
                qid = m.group(2).strip()
                if label in rels:
                    iri, _ = rels[label]
                    rels[label] = (iri, qid)
        # Map selected relations to our canonical names.  We only include
        # relations we care about; others can be added as needed.
        canonical = {
            "director": "director",
            "screenwriter": "screenwriter",
            "producer": "producer",
            "country_of_origin": "country_of_origin",
            "genre": "genre",
            "publication_date": "publication_date",
            "cast_member": "cast_member",
        }
        for label, canon in canonical.items():
            if label in rels:
                self.relations[canon] = rels[label]
        # If any of the above were missing set fallback values
        for canon, default in [
            ("director", ("http://www.wikidata.org/prop/direct/P57", "Q5")),
            ("screenwriter", ("http://www.wikidata.org/prop/direct/P58", "Q5")),
            ("producer", ("http://www.wikidata.org/prop/direct/P162", "Q5")),
            ("country_of_origin", ("http://www.wikidata.org/prop/direct/P495", "Q6256")),
            ("genre", ("http://www.wikidata.org/prop/direct/P136", "Q201658")),
            ("publication_date", ("http://www.wikidata.org/prop/direct/P577", "xsd:dateTime")),
            ("cast_member", ("http://www.wikidata.org/prop/direct/P161", "Q5")),
        ]:
            self.relations.setdefault(canon, default)

    def get_predicate(self, relation: str) -> Optional[str]:
        """Return the predicate IRI for a given canonical relation name."""
        return self.relations.get(relation, (None, None))[0]  # type: ignore[return-value]

    def get_object_type(self, relation: str) -> Optional[str]:
        """Return the expected object type Q‑id for a canonical relation."""
        return self.relations.get(relation, (None, None))[1]  # type: ignore[return-value]
