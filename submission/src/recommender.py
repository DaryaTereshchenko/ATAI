"""
recommender.py
--------------

Simple movie recommender built on top of the knowledge graph and
embeddings.  Given a list of movies the user likes, it computes an
average embedding vector and returns the nearest neighbour movies in
embedding space, filtering out the input movies themselves.  Optionally
the recommender can use genre information from the graph to bias the
recommendations towards movies that share genres with the input set.

This recommender is simplistic but demonstrates how one might combine
structural information (genres) with embeddings.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .kg_utils import KnowledgeGraph
from .embeddings_utils import EmbeddingModel
from .relations_mapping import RelationsMapping


def recommend_movies(
    liked_titles: List[str],
    kg: KnowledgeGraph,
    embed_model: EmbeddingModel,
    relations: RelationsMapping,
    top_k: int = 5,
    genre_bias: bool = True,
) -> List[str]:
    """Return a list of recommended movie titles.

    Parameters
    ----------
    liked_titles: list of str
        Movie titles the user likes.
    kg: KnowledgeGraph
        The knowledge graph instance used to resolve movie IRIs and genres.
    embed_model: EmbeddingModel
        Pretrained embeddings for computing similarity between movies.
    relations: RelationsMapping
        Mapping of relation names to predicates; used to look up the 'genre'
        predicate IRI.
    top_k: int
        Number of recommendations to return.
    genre_bias: bool
        If True, candidate movies are filtered to those sharing at least one
        genre with the liked movies when possible.  If no genre information
        is available this filter is ignored.

    Returns
    -------
    list of str
        Recommended movie titles.  If insufficient candidates are found a
        shorter list may be returned.
    """
    # Resolve liked movie IRIs
    liked_iris = []
    for title in liked_titles:
        iri = kg.iri_for_label(title)
        if iri:
            liked_iris.append(iri)
    if not liked_iris:
        return []
    # Compute average embedding vector
    avg_vec = embed_model.average_vector(liked_iris)
    if avg_vec is None:
        return []
    # Collect genres for liked movies
    liked_genres: set[str] = set()
    if genre_bias:
        genre_pred = relations.get_predicate("genre")
        if genre_pred:
            for iri in liked_iris:
                label = kg.label_for_iri(iri)
                if not label:
                    continue
                # get genres for this movie
                genres = kg.get_objects(label, genre_pred)
                for g in genres:
                    liked_genres.add(g)
    # Determine candidate IRIs by nearest neighbours
    candidates = embed_model.nearest_entities(avg_vec, top_k=50)
    recommendations: List[str] = []
    for iri, _dist in candidates:
        label = kg.label_for_iri(iri)
        if not label:
            continue
        if label in liked_titles:
            continue
        # optionally filter by genre
        if genre_bias and liked_genres:
            if genre_pred:
                genres = kg.get_objects(label, genre_pred)
                if not any(g in liked_genres for g in genres):
                    continue
        if label not in recommendations:
            recommendations.append(label)
        if len(recommendations) >= top_k:
            break
    return recommendations
