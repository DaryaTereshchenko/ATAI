"""
query_parser.py
----------------

Tools for classifying natural language questions and extracting salient
information such as entities and relations.  The heuristics defined here are
simple and deterministic, tailored for the specific types of questions
required by the assignment.  For more robust parsing one could integrate a
trained language model, but that is beyond the scope of this repository.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ParsedQuestion:
    """Structure representing the output of question parsing."""

    # The raw question string
    question: str
    # One of {"factual", "embedding", "recommendation", "multimedia"}
    qtype: str
    # List of extracted entity names (as they appear in the question)
    entities: List[str]
    # Relation keyword, e.g., "director" or "genre"
    relation: Optional[str]


class QuestionParser:
    """Heuristic parser for natural language questions."""

    # Simple keyword mapping from natural language relation terms to our
    # canonical relation names.  These names correspond to entries in
    # relations_mapping.RelationsMapping which further map to full predicate
    # IRIs and entity types.
    RELATION_KEYWORDS: Dict[str, List[str]] = {
        "director": ["director", "directed", "direct", "director of"],
        "screenwriter": ["screenwriter", "screenplay", "written by", "writer"],
        "producer": ["producer", "produced"],
        "country_of_origin": ["country", "country of origin"],
        "genre": ["genre", "type", "kind"],
        "publication_date": ["when", "release date", "come out", "published", "premiere"],
        "cast_member": ["actor", "star", "cast", "starring"],
    }

    def classify(self, question: str) -> str:
        """Classify the question type based on cue phrases."""
        q = question.lower()
        if "embedding approach" in q:
            return "embedding"
        if "factual approach" in q:
            return "factual"
        # Recommendation
        if q.startswith("given that i like") or q.startswith("recommend"):
            return "recommendation"
        # Multimedia (images)
        if any(kw in q for kw in ["show me a picture", "what does", "look like", "image of"]):
            return "multimedia"
        # Default to factual if relation keywords are present, else embedding
        return "factual"

    def extract_entities(self, question: str) -> List[str]:
        """Extract quoted substrings from the question as entity names.

        The parser recognises standard single/double quotes as well as
        left/right curly quotes (e.g., ‘ ’ and “ ”).  If no quoted substrings
        are found it falls back to detecting capitalised multi‑word
        sequences.
        """
        # Normalise curly quotes to straight quotes for easier matching
        normalized = (
            question.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
        )
        # Extract text inside single or double quotes
        pattern = r"['\"]([^'\"]+)['\"]"
        matches = re.findall(pattern, normalized)
        if matches:
            return [m.strip() for m in matches]
        # Fall back: detect capitalised sequences (two or more words)
        cap_pattern = r"\b([A-Z][^\s,]+(?:\s+[A-Z][^\s,]+)+)"
        cap_matches = re.findall(cap_pattern, normalized)
        return [m.strip() for m in cap_matches]

    def extract_relation(self, question: str) -> Optional[str]:
        """Identify the relation term within the question, if any."""
        q = question.lower()
        for rel, keywords in self.RELATION_KEYWORDS.items():
            for kw in keywords:
                if kw in q:
                    return rel
        return None

    def parse(self, question: str) -> ParsedQuestion:
        """Parse the question into its components."""
        qtype = self.classify(question)
        entities = self.extract_entities(question)
        relation = self.extract_relation(question) if qtype != "recommendation" and qtype != "multimedia" else None
        return ParsedQuestion(question=question, qtype=qtype, entities=entities, relation=relation)
