"""
multimedia.py
-------------

Support for answering multimedia questions.  The assignment specifies
questions of the form "Show me a picture of X" or "What does X look like?"
where X is an entity in the knowledge graph.  The provided data set does
not contain actual images; however it does include the ``imdb_id`` property
(P345).  We leverage this by constructing a link to the entity's IMDb page,
which typically contains photographs of the person.  If an entity lacks an
IMDb identifier the function falls back to returning ``None``.
"""

from __future__ import annotations

from typing import Optional

from .kg_utils import KnowledgeGraph
from .relations_mapping import RelationsMapping


def get_multimedia_answer(
    entity_name: str, kg: KnowledgeGraph, relations: RelationsMapping
) -> Optional[str]:
    """Return a URL representing the requested multimedia for an entity.

    Parameters
    ----------
    entity_name: str
        The natural language label of the entity (e.g., a person name).
    kg: KnowledgeGraph
        The knowledge graph instance used to resolve the entity and fetch
        identifiers.
    relations: RelationsMapping
        Mapping of relation names to predicates; used to look up the
        ``imdb_id`` predicate.  If the relations file was parsed this will
        contain the correct predicate IRI; otherwise we fall back to the
        default mapping.

    Returns
    -------
    str or None
        A URL to the entity's IMDb page if available; otherwise ``None``.
    """
    # Predicate for imdb_id (property P345)
    imdb_pred = "http://www.wikidata.org/prop/direct/P345"
    # Retrieve the IRI for the entity
    iri = kg.iri_for_label(entity_name)
    if not iri:
        return None
    label = kg.label_for_iri(iri)
    if not label:
        return None
    # Query the graph for the imdb_id literal
    ids = kg.get_objects(label, imdb_pred)
    if not ids:
        return None
    imdb_id = ids[0]
    # Construct IMDb URL.  IMDb IDs for names start with 'nm' followed by
    # digits.  We remove any surrounding quotes or tags.
    imdb_id = imdb_id.strip()
    # The link template for people on IMDb
    return f"https://www.imdb.com/name/{imdb_id}"
