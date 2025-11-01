"""
embeddings_utils.py
--------------------

Utilities for working with entity and relation embeddings.

The embeddings are provided as NumPy arrays stored in .npy files along with
mapping files that associate integer indices with the corresponding entity or
relation IRIs.  This module provides a simple API to load the embeddings,
resolve IRIs and labels to indices, and perform basic algebra for TransE style
models (h + r ≈ t).  It also includes helpers for computing nearest
neighbours and recommending entities similar to a given set of entities.
"""

from __future__ import annotations

import os
import numpy as np
from typing import Dict, List, Optional, Tuple


class EmbeddingModel:
    """Wrapper around TransE embeddings for entities and relations."""

    def __init__(
        self,
        entity_embeds_path: str,
        entity_ids_path: str,
        relation_embeds_path: str,
        relation_ids_path: str,
    ) -> None:
        """Initialize the model with paths to the embedding and mapping files."""
        self.entity_embeds_path = entity_embeds_path
        self.entity_ids_path = entity_ids_path
        self.relation_embeds_path = relation_embeds_path
        self.relation_ids_path = relation_ids_path
        self.entity_embeds: Optional[np.ndarray] = None
        self.relation_embeds: Optional[np.ndarray] = None
        self.iri_to_ent_idx: Dict[str, int] = {}
        self.ent_idx_to_iri: Dict[int, str] = {}
        self.iri_to_rel_idx: Dict[str, int] = {}
        self.rel_idx_to_iri: Dict[int, str] = {}
        self._loaded = False

    def _load_mapping(self, path: str) -> Dict[str, int]:
        """Load an ID mapping file.

        The mapping file contains one mapping per line.  The format is either
        ``entity<TAB>id`` or ``id<TAB>entity``.  We detect which column
        contains the IRI by checking if a field starts with ``http``.
        Returns a dict mapping IRI → index.
        """
        mapping: Dict[str, int] = {}
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ID mapping file not found: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                a, b = parts
                if a.startswith("http"):
                    iri, idx_str = a, b
                elif b.startswith("http"):
                    iri, idx_str = b, a
                else:
                    # fall back: assume first column is iri
                    iri, idx_str = a, b
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                mapping[iri] = idx
        return mapping

    def load(self) -> None:
        """Load embeddings and mappings into memory.  Subsequent calls are no‑ops."""
        if self._loaded:
            return
        # Load mappings
        self.iri_to_ent_idx = self._load_mapping(self.entity_ids_path)
        self.ent_idx_to_iri = {idx: iri for iri, idx in self.iri_to_ent_idx.items()}
        self.iri_to_rel_idx = self._load_mapping(self.relation_ids_path)
        self.rel_idx_to_iri = {idx: iri for iri, idx in self.iri_to_rel_idx.items()}
        # Load embeddings
        if not os.path.isfile(self.entity_embeds_path):
            raise FileNotFoundError(f"Entity embeddings file not found: {self.entity_embeds_path}")
        if not os.path.isfile(self.relation_embeds_path):
            raise FileNotFoundError(f"Relation embeddings file not found: {self.relation_embeds_path}")
        self.entity_embeds = np.load(self.entity_embeds_path)
        self.relation_embeds = np.load(self.relation_embeds_path)
        self._loaded = True

    def get_entity_vector(self, iri: str) -> Optional[np.ndarray]:
        """Return the embedding vector for the given entity IRI."""
        if not self._loaded:
            self.load()
        idx = self.iri_to_ent_idx.get(iri)
        if idx is None:
            return None
        return self.entity_embeds[idx]

    def get_relation_vector(self, iri: str) -> Optional[np.ndarray]:
        """Return the embedding vector for the given relation IRI."""
        if not self._loaded:
            self.load()
        idx = self.iri_to_rel_idx.get(iri)
        if idx is None:
            return None
        return self.relation_embeds[idx]

    def nearest_entities(self, vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return the top_k nearest entity IRIs to the provided vector.

        Distance is measured by Euclidean (L2) norm.  Returns a list of
        (entity_iri, distance) pairs sorted by ascending distance.
        """
        if not self._loaded:
            self.load()
        diffs = self.entity_embeds - vector  # (n_entities, dim)
        dists = np.linalg.norm(diffs, axis=1)
        nearest_indices = np.argsort(dists)[:top_k]
        return [(self.ent_idx_to_iri[idx], dists[idx]) for idx in nearest_indices]

    def predict_missing_entity(
        self,
        head_iri: Optional[str],
        relation_iri: str,
        tail_iri: Optional[str] = None,
        top_k: int = 5,
    ) -> List[str]:
        """Predict the missing entity in a triple (h, r, t) using TransE addition.

        At least one of ``head_iri`` or ``tail_iri`` must be provided.  If
        ``head_iri`` is provided, the method predicts possible tails; if
        ``tail_iri`` is provided, it predicts possible heads.  Returns the
        top_k predicted entity IRIs.
        """
        if not self._loaded:
            self.load()
        if head_iri is None and tail_iri is None:
            raise ValueError("Either head_iri or tail_iri must be provided")
        r_vec = self.get_relation_vector(relation_iri)
        if r_vec is None:
            return []
        if head_iri:
            h_vec = self.get_entity_vector(head_iri)
            if h_vec is None:
                return []
            target = h_vec + r_vec
        else:
            t_vec = self.get_entity_vector(tail_iri)  # type: ignore[arg-type]
            if t_vec is None:
                return []
            target = t_vec - r_vec
        nearest = self.nearest_entities(target, top_k=top_k)
        return [iri for iri, _ in nearest]

    def average_vector(self, iris: List[str]) -> Optional[np.ndarray]:
        """Compute the average embedding vector of a list of entity IRIs."""
        if not self._loaded:
            self.load()
        vectors = []
        for iri in iris:
            vec = self.get_entity_vector(iri)
            if vec is None:
                continue
            vectors.append(vec)
        if not vectors:
            return None
        return np.mean(np.stack(vectors, axis=0), axis=0)
