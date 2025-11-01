"""
orchestrator.py
---------------

This module ties together the knowledge graph, embeddings and various
subsystems (query parsing, recommendation, multimedia) to answer natural
language questions.  The primary entry point is the :func:`answer_question`
function which returns a human‑readable answer string.

The orchestrator maintains singleton instances of the heavy components
(`KnowledgeGraph`, `EmbeddingModel`, `RelationsMapping`) to avoid repeated
loading.  Paths to the underlying data files must be provided when
instantiating the orchestrator.
"""

from __future__ import annotations

from typing import List, Optional

from .kg_utils import KnowledgeGraph
from .embeddings_utils import EmbeddingModel
from .relations_mapping import RelationsMapping
from .query_parser import ParsedQuestion, QuestionParser
from .recommender import recommend_movies
from .multimedia import get_multimedia_answer


class QAOrchestrator:
    """High level orchestrator for question answering."""

    def __init__(
        self,
        graph_path: str,
        entity_embeds_path: str,
        entity_ids_path: str,
        relation_embeds_path: str,
        relation_ids_path: str,
        relations_file: Optional[str] = None,
    ) -> None:
        self.kg = KnowledgeGraph(graph_path)
        self.embed_model = EmbeddingModel(
            entity_embeds_path,
            entity_ids_path,
            relation_embeds_path,
            relation_ids_path,
        )
        self.relations = RelationsMapping(relations_file)
        self.parser = QuestionParser()

    def _answer_factual(self, pq: ParsedQuestion) -> str:
        """Answer factual questions via graph lookup."""
        if not pq.entities or not pq.relation:
            return "I'm sorry, I couldn't extract the necessary information from the question."
        predicate = self.relations.get_predicate(pq.relation)
        if not predicate:
            return f"I'm sorry, I don't know how to handle relation '{pq.relation}'."
        answers: List[str] = []
        # For questions with a single entity we retrieve objects; if multiple
        # entities are given we take the first as the subject (common for our
        # tasks).  Additional entities are ignored.
        subj_label = pq.entities[0]
        objs = self.kg.get_objects(subj_label, predicate)
        for o in objs:
            if o not in answers:
                answers.append(o)
        if not answers:
            return f"I couldn't find an answer for {subj_label}."
        # Format the answer: join multiple answers with ' and '
        answer_str = " and ".join(answers)
        # Return a simple sentence.
        # We attempt to incorporate the relation for readability.  Note: the
        # relation names use underscores; replace with spaces.
        readable_rel = pq.relation.replace("_", " ")
        return f"The {readable_rel} of {subj_label} is {answer_str}."

    def _answer_embedding(self, pq: ParsedQuestion) -> str:
        """Answer embedding questions via vector arithmetic."""
        if not pq.entities or not pq.relation:
            return "I'm sorry, I couldn't extract the necessary information from the question."
        predicate = self.relations.get_predicate(pq.relation)
        if not predicate:
            return f"I'm sorry, I don't know how to handle relation '{pq.relation}'."
        subj_label = pq.entities[0]
        subj_iri = self.kg.iri_for_label(subj_label)
        if not subj_iri:
            return f"I'm sorry, I couldn't find the entity '{subj_label}' in the knowledge graph."
        predicted = self.embed_model.predict_missing_entity(head_iri=subj_iri, relation_iri=predicate, top_k=5)
        if not predicted:
            return "I'm sorry, I couldn't compute an embedding prediction."
        # Choose the first prediction that has a label
        chosen_label = None
        for iri in predicted:
            label = self.kg.label_for_iri(iri)
            if label:
                chosen_label = label
                break
        if not chosen_label:
            chosen_label = predicted[0]
        obj_type = self.relations.get_object_type(pq.relation)
        if not obj_type:
            obj_type = "unknown"
        return f"The answer suggested by embeddings is: {chosen_label} (type: {obj_type})"

    def _answer_recommendation(self, pq: ParsedQuestion) -> str:
        """Answer recommendation questions."""
        if not pq.entities:
            # Entities will be extracted as liked titles when the question
            # starts with "Given that I like ...".  If none found the parser
            # uses capitalised phrases; else we cannot proceed.
            return "I'm sorry, I couldn't extract the movies you like."
        recommendations = recommend_movies(pq.entities, self.kg, self.embed_model, self.relations)
        if not recommendations:
            return "I couldn't find any suitable recommendations."
        return "You might also enjoy: " + ", ".join(recommendations)

    def _answer_multimedia(self, pq: ParsedQuestion) -> str:
        """Answer multimedia questions."""
        if not pq.entities:
            return "I'm sorry, I couldn't identify the entity you are asking about."
        # Multimedia questions typically involve a single entity
        entity_name = pq.entities[0]
        url = get_multimedia_answer(entity_name, self.kg, self.relations)
        if not url:
            return "I'm sorry, I couldn't find a picture for that entity."
        return url

    def answer_question(self, question: str) -> str:
        """Top level entry point: classify and answer a question."""
        pq = self.parser.parse(question)
        if pq.qtype == "factual":
            return self._answer_factual(pq)
        if pq.qtype == "embedding":
            return self._answer_embedding(pq)
        if pq.qtype == "recommendation":
            return self._answer_recommendation(pq)
        if pq.qtype == "multimedia":
            return self._answer_multimedia(pq)
        return "I'm sorry, I don't know how to answer that question."
