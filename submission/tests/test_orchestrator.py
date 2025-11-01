import os
import tempfile
import unittest

import numpy as np

from src.orchestrator import QAOrchestrator


def create_test_dataset(tmpdir: str):
    """Create a miniature knowledge graph, embeddings and mapping files for testing."""
    # Paths for files
    kg_path = os.path.join(tmpdir, "graph.nt")
    entity_embeds_path = os.path.join(tmpdir, "entity_embeds.npy")
    relation_embeds_path = os.path.join(tmpdir, "relation_embeds.npy")
    entity_ids_path = os.path.join(tmpdir, "entity_ids.del")
    relation_ids_path = os.path.join(tmpdir, "relation_ids.del")
    # Define entities and relations
    entities = [
        ("http://example.org/Good_Will_Hunting", "Good Will Hunting"),
        ("http://example.org/Gus_Van_Sant", "Gus Van Sant"),
        ("http://example.org/Harmony_Korine", "Harmony Korine"),
        ("http://example.org/Aro_Tolbukhin", "Aro Tolbukhin. En la mente del asesino"),
        ("http://example.org/Mexico", "Mexico"),
        ("http://example.org/Fargo", "Fargo"),
        ("http://example.org/Ethan_Coen", "Ethan Coen"),
        ("http://example.org/Joel_Coen", "Joel Coen"),
        ("http://example.org/Shortcut_to_Happiness", "Shortcut to Happiness"),
        ("http://example.org/Pete_Dexter", "Pete Dexter"),
        ("http://example.org/Bandit_Queen", "Bandit Queen"),
        ("http://example.org/Drama_Film", "drama film"),
        ("http://example.org/Biographical_Film", "biographical film"),
        ("http://example.org/Crime_Film", "crime film"),
        ("http://example.org/Miracles_Still_Happen", "Miracles Still Happen"),
        ("http://example.org/Apocalypse_Now", "Apocalypse Now"),
        ("http://example.org/John_Milius", "John Milius"),
        ("http://example.org/Monkeys_12", "12 Monkeys"),
        ("http://example.org/Carol_Florence", "Carol Florence"),
        ("http://example.org/Shoplifters", "Shoplifters"),
        ("http://example.org/Comedy_Film", "comedy film"),
    ]
    # Create entity index mapping
    ent_id_map = {iri: idx for idx, (iri, _) in enumerate(entities)}
    # Relations of interest
    relations = {
        "http://www.wikidata.org/prop/direct/P57": 0,  # director
        "http://www.wikidata.org/prop/direct/P58": 1,  # screenwriter
        "http://www.wikidata.org/prop/direct/P495": 2,  # country of origin
        "http://www.wikidata.org/prop/direct/P136": 3,  # genre
        "http://www.wikidata.org/prop/direct/P577": 4,  # publication date
        "http://www.wikidata.org/prop/direct/P345": 5,  # imdb id
    }
    # Define triples
    triples = []
    def add_label(iri, label):
        triples.append(f"<{iri}> <http://www.w3.org/2000/01/rdf-schema#label> \"{label}\"@en .\n")
    for iri, label in entities:
        add_label(iri, label)
    # Add factual relations
    triples.append(
        f"<http://example.org/Good_Will_Hunting> <http://www.wikidata.org/prop/direct/P57> <http://example.org/Gus_Van_Sant> .\n"
    )
    triples.append(
        f"<http://example.org/Aro_Tolbukhin> <http://www.wikidata.org/prop/direct/P495> <http://example.org/Mexico> .\n"
    )
    triples.append(
        f"<http://example.org/Fargo> <http://www.wikidata.org/prop/direct/P57> <http://example.org/Ethan_Coen> .\n"
    )
    triples.append(
        f"<http://example.org/Fargo> <http://www.wikidata.org/prop/direct/P57> <http://example.org/Joel_Coen> .\n"
    )
    triples.append(
        f"<http://example.org/Shortcut_to_Happiness> <http://www.wikidata.org/prop/direct/P58> <http://example.org/Pete_Dexter> .\n"
    )
    triples.append(
        f"<http://example.org/Bandit_Queen> <http://www.wikidata.org/prop/direct/P136> <http://example.org/Drama_Film> .\n"
    )
    triples.append(
        f"<http://example.org/Bandit_Queen> <http://www.wikidata.org/prop/direct/P136> <http://example.org/Biographical_Film> .\n"
    )
    triples.append(
        f"<http://example.org/Bandit_Queen> <http://www.wikidata.org/prop/direct/P136> <http://example.org/Crime_Film> .\n"
    )
    triples.append(
        f"<http://example.org/Miracles_Still_Happen> <http://www.wikidata.org/prop/direct/P577> \"1974-07-19\"@en .\n"
    )
    triples.append(
        f"<http://example.org/Gus_Van_Sant> <http://www.wikidata.org/prop/direct/P345> \"nm0000241\" .\n"
    )
    triples.append(
        f"<http://example.org/Ethan_Coen> <http://www.wikidata.org/prop/direct/P345> \"nm0001054\" .\n"
    )
    # Write graph file
    with open(kg_path, "w", encoding="utf-8") as f:
        f.writelines(triples)
    # Create embeddings (dimension 3) for entities
    np.random.seed(0)
    entity_embeds = np.random.randn(len(entities), 3).astype(np.float32)
    # Create mapping files
    with open(entity_ids_path, "w", encoding="utf-8") as f:
        for iri, idx in ent_id_map.items():
            f.write(f"{iri}\t{idx}\n")
    # Create relation embeddings (dimension 3)
    relation_embeds = np.random.randn(len(relations), 3).astype(np.float32)
    # Modify relation 'director' vector to ensure that Good Will Hunting + director ≈ Harmony Korine
    h_idx = ent_id_map["http://example.org/Good_Will_Hunting"]
    t_idx = ent_id_map["http://example.org/Harmony_Korine"]
    dir_idx = relations["http://www.wikidata.org/prop/direct/P57"]
    relation_embeds[dir_idx] = entity_embeds[t_idx] - entity_embeds[h_idx]
    np.save(entity_embeds_path, entity_embeds)
    np.save(relation_embeds_path, relation_embeds)
    # Write relation ids mapping
    with open(relation_ids_path, "w", encoding="utf-8") as f:
        for iri, idx in relations.items():
            f.write(f"{iri}\t{idx}\n")
    return (
        kg_path,
        entity_embeds_path,
        entity_ids_path,
        relation_embeds_path,
        relation_ids_path,
    )


class TestQAOrchestrator(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.tmpdir = tempfile.mkdtemp()
        (
            self.kg_path,
            self.ent_emb_path,
            self.ent_ids_path,
            self.rel_emb_path,
            self.rel_ids_path,
        ) = create_test_dataset(self.tmpdir)
        self.orch = QAOrchestrator(
            graph_path=self.kg_path,
            entity_embeds_path=self.ent_emb_path,
            entity_ids_path=self.ent_ids_path,
            relation_embeds_path=self.rel_emb_path,
            relation_ids_path=self.rel_ids_path,
            relations_file=None,
        )

    def test_factual_country(self):
        question = "Please answer this question with a factual approach: From what country is the movie 'Aro Tolbukhin. En la mente del asesino'?"
        answer = self.orch.answer_question(question)
        self.assertIn("Mexico", answer)

    def test_factual_director_multiple(self):
        q = "Please answer this question with a factual approach: Who directed ‘Fargo’?"
        answer = self.orch.answer_question(q)
        self.assertIn("Ethan Coen", answer)
        self.assertIn("Joel Coen", answer)

    def test_factual_genre(self):
        q = "Please answer this question with a factual approach: What genre is the movie 'Bandit Queen'?"
        answer = self.orch.answer_question(q)
        self.assertIn("drama film", answer)
        self.assertIn("biographical film", answer)
        self.assertIn("crime film", answer)

    def test_factual_publication_date(self):
        q = "Please answer this question with a factual approach: When did the movie 'Miracles Still Happen' come out?"
        answer = self.orch.answer_question(q)
        self.assertIn("1974-07-19", answer)

    def test_embedding_director_prediction(self):
        q = "Please answer this question with an embedding approach: Who is the director of ‘Good Will Hunting’?"
        answer = self.orch.answer_question(q)
        self.assertIn("Harmony Korine", answer)
        self.assertIn("type: Q5", answer)

    def test_multimedia_imdb_link(self):
        q = "Show me a picture of Gus Van Sant"
        answer = self.orch.answer_question(q)
        self.assertTrue(answer.startswith("https://www.imdb.com/name/nm0000241"))


if __name__ == "__main__":
    unittest.main()
