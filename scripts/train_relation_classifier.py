"""
Train DistilBERT Relation Classifier for Movie Knowledge Graph.
Extracts all 496+ properties from the database and generates synthetic training data.
"""

import sys
import os
import random
from typing import List, Dict, Tuple
from collections import defaultdict

# ✅ Add project root to path BEFORE importing from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
import pandas as pd
from rdflib import URIRef, RDFS

from src.config import GRAPH_FILE_PATH
from src.main.sparql_handler import SPARQLHandler


class RelationDatasetBuilder:
    """Builds training dataset by extracting relations from knowledge graph."""
    
    def __init__(self, sparql_handler: SPARQLHandler):
        self.sparql_handler = sparql_handler
        self.graph = sparql_handler.graph
        
        # Extract relations and their metadata
        self.relations = {}  # relation_uri -> metadata
        self.entity_labels = {}  # entity_uri -> label
        
        print("🔍 Extracting relations from knowledge graph...")
        self._extract_relations()
        self._extract_entity_labels()
    
    def _extract_relations(self):
        """Extract all relations (properties) used in the graph."""
        print("   Scanning graph for properties...")
        
        property_usage = defaultdict(int)  # Track usage frequency
        property_examples = defaultdict(list)  # Store example triples
        
        # Scan all triples
        for s, p, o in self.graph:
            pred_str = str(p)
            
            # Only Wikidata properties and custom properties
            if 'wikidata.org/prop/direct/P' not in pred_str and 'ddis.ch/atai/' not in pred_str:
                continue
            
            property_usage[pred_str] += 1
            
            # Store example triples (limit to 10 per property)
            if len(property_examples[pred_str]) < 10:
                property_examples[pred_str].append((str(s), pred_str, str(o)))
        
        print(f"   Found {len(property_usage)} unique properties")
        
        # Get labels for properties
        for prop_uri in property_usage.keys():
            prop_ref = URIRef(prop_uri)
            
            # Get label
            label = None
            for lbl in self.graph.objects(prop_ref, RDFS.label):
                label = str(lbl).lower()
                label = label.replace(' - wikidata', '').replace('wikidata property for ', '')
                break
            
            # Extract property ID
            if '/P' in prop_uri:
                prop_id = f"P{prop_uri.split('/P')[-1]}"
                friendly_name = label.replace(' ', '_').replace('-', '_') if label else prop_id
            else:
                prop_id = prop_uri.split('/')[-1]
                friendly_name = prop_id
            
            self.relations[prop_uri] = {
                'id': prop_id,
                'label': label or friendly_name,
                'friendly_name': friendly_name,
                'usage_count': property_usage[prop_uri],
                'examples': property_examples[prop_uri]
            }
        
        # Sort by usage frequency
        sorted_relations = sorted(
            self.relations.items(),
            key=lambda x: x[1]['usage_count'],
            reverse=True
        )
        
        print(f"   ✅ Extracted metadata for {len(self.relations)} relations")
        print(f"\n   📊 Top 10 most used relations:")
        for prop_uri, metadata in sorted_relations[:10]:
            print(f"      {metadata['id']:10s} - {metadata['label'][:50]:50s} ({metadata['usage_count']} uses)")
    
    def _extract_entity_labels(self, max_labels: int = 10000):
        """Extract entity labels for generating queries (sample for efficiency)."""
        print(f"\n   Extracting entity labels (sampling {max_labels})...")
        
        count = 0
        for s, p, o in self.graph.triples((None, RDFS.label, None)):
            if count >= max_labels:
                break
            
            entity_uri = str(s)
            label = str(o)
            
            self.entity_labels[entity_uri] = label
            count += 1
        
        print(f"   ✅ Cached {len(self.entity_labels)} entity labels")
    
    def generate_training_data(
        self,
        queries_per_relation: int = 50,
        min_usage: int = 10
    ) -> pd.DataFrame:
        """
        Generate synthetic training queries for each relation.
        
        Args:
            queries_per_relation: Number of training queries per relation
            min_usage: Minimum property usage to include in training
            
        Returns:
            DataFrame with columns: text, label, relation_id
        """
        print(f"\n🔧 Generating training data...")
        print(f"   Queries per relation: {queries_per_relation}")
        print(f"   Minimum usage threshold: {min_usage}")
        
        examples = []
        
        # Filter relations by usage
        active_relations = {
            uri: meta for uri, meta in self.relations.items()
            if meta['usage_count'] >= min_usage
        }
        
        print(f"   Using {len(active_relations)} relations (filtered by usage)")
        
        # Generate queries for each relation
        for prop_uri, metadata in active_relations.items():
            relation_id = metadata['id']
            relation_label = metadata['label']
            friendly_name = metadata['friendly_name']
            
            # Generate varied query templates
            queries = self._generate_queries_for_relation(
                relation_id=relation_id,
                relation_label=relation_label,
                friendly_name=friendly_name,
                examples_triples=metadata['examples'],
                num_queries=queries_per_relation
            )
            
            for query_text in queries:
                examples.append({
                    'text': query_text,
                    'label': friendly_name,
                    'relation_id': relation_id
                })
        
        df = pd.DataFrame(examples)
        
        print(f"   ✅ Generated {len(df)} training examples")
        print(f"   ✅ Covering {df['label'].nunique()} unique relations")
        
        return df
    
    def _generate_queries_for_relation(
        self,
        relation_id: str,
        relation_label: str,
        friendly_name: str,
        examples_triples: List[Tuple[str, str, str]],
        num_queries: int
    ) -> List[str]:
        """Generate diverse natural language queries for a specific relation."""
        
        queries = []
        
        # Get sample entities from example triples
        sample_entities = []
        for subj_uri, _, obj_uri in examples_triples[:5]:
            if subj_uri in self.entity_labels:
                sample_entities.append(self.entity_labels[subj_uri])
        
        # Use generic placeholders if no samples
        if not sample_entities:
            sample_entities = ["Inception", "The Matrix", "Titanic", "Pulp Fiction", "Forrest Gump"]
        
        # Template categories
        templates = self._get_query_templates(relation_label, friendly_name)
        
        # Generate queries by filling templates
        for _ in range(num_queries):
            template_category = random.choice(list(templates.keys()))
            template = random.choice(templates[template_category])
            entity = random.choice(sample_entities)
            
            query = template.format(entity=entity)
            queries.append(query)
        
        return queries
    
    def _get_query_templates(self, relation_label: str, friendly_name: str) -> Dict[str, List[str]]:
        """Get query templates based on relation type."""
        
        # Detect relation category
        relation_lower = relation_label.lower()
        
        templates = {
            'what': [],
            'who': [],
            'when': [],
            'which': [],
            'where': []
        }
        
        # Generic templates that work for most relations
        templates['what'].extend([
            f"What is the {relation_label} of {{entity}}?",
            f"What {relation_label} does {{entity}} have?",
            f"Tell me the {relation_label} for {{entity}}",
            f"{{entity}} - what is the {relation_label}?",
            f"What's the {relation_label} of {{entity}}?",
        ])
        
        # Relation-specific templates
        if 'director' in relation_lower:
            templates['who'].extend([
                "Who directed {entity}?",
                "Who is the director of {entity}?",
                "{entity} was directed by whom?",
                "Who made {entity}?",
                "Name the director of {entity}",
            ])
        
        elif 'cast' in relation_lower or 'actor' in relation_lower:
            templates['who'].extend([
                "Who acted in {entity}?",
                "Who stars in {entity}?",
                "Who are the cast members of {entity}?",
                "Which actors appear in {entity}?",
                "Who performed in {entity}?",
            ])
        
        elif 'screenwriter' in relation_lower or 'writer' in relation_lower:
            templates['who'].extend([
                "Who wrote {entity}?",
                "Who is the screenwriter of {entity}?",
                "Who wrote the screenplay for {entity}?",
                "Name the writer of {entity}",
                "{entity} was written by whom?",
            ])
        
        elif 'producer' in relation_lower:
            templates['who'].extend([
                "Who produced {entity}?",
                "Who is the producer of {entity}?",
                "Name the producers of {entity}",
                "{entity} was produced by whom?",
                "Who are the producers of {entity}?",
            ])
        
        elif 'genre' in relation_lower:
            templates['what'].extend([
                "What genre is {entity}?",
                "What type of movie is {entity}?",
                "What kind of film is {entity}?",
                "What category is {entity} in?",
                "What's the genre of {entity}?",
            ])
        
        elif 'country' in relation_lower and 'origin' in relation_lower:
            templates['what'].extend([
                "What country is {entity} from?",
                "From what country is {entity}?",
                "Which country produced {entity}?",
                "What is the country of origin of {entity}?",
                "Where was {entity} made?",
            ])
        
        elif 'language' in relation_lower:
            templates['what'].extend([
                "What language is {entity} in?",
                "In which language is {entity} filmed?",
                "What is the original language of {entity}?",
                "What language is spoken in {entity}?",
                "Which language is {entity} in?",
            ])
        
        elif 'date' in relation_lower or 'release' in relation_lower or 'publication' in relation_lower:
            templates['when'].extend([
                "When was {entity} released?",
                "What year was {entity} released?",
                "When did {entity} come out?",
                "What is the release date of {entity}?",
                "When was {entity} published?",
            ])
        
        elif 'award' in relation_lower:
            templates['what'].extend([
                "What awards did {entity} win?",
                "Which awards did {entity} receive?",
                "What prizes did {entity} get?",
                "Name the awards for {entity}",
                "What accolades did {entity} receive?",
            ])
        
        elif 'composer' in relation_lower or 'music' in relation_lower:
            templates['who'].extend([
                "Who composed the music for {entity}?",
                "Who wrote the soundtrack for {entity}?",
                "Who is the composer of {entity}?",
                "Who did the music for {entity}?",
                "Name the composer of {entity}",
            ])
        
        elif 'cinematography' in relation_lower or 'photography' in relation_lower:
            templates['who'].extend([
                "Who was the cinematographer for {entity}?",
                "Who did the cinematography for {entity}?",
                "Who shot {entity}?",
                "Name the director of photography for {entity}",
                "Who filmed {entity}?",
            ])
        
        elif 'location' in relation_lower or 'filmed' in relation_lower:
            templates['where'].extend([
                "Where was {entity} filmed?",
                "What is the filming location of {entity}?",
                "Where did they shoot {entity}?",
                "In which location was {entity} filmed?",
                "Where was {entity} shot?",
            ])
        
        # Remove empty categories
        templates = {k: v for k, v in templates.items() if v}
        
        # If no specific templates, use generic ones
        if not templates:
            templates = {
                'what': [
                    f"What is the {relation_label} of {{entity}}?",
                    f"What {relation_label} does {{entity}} have?",
                    f"Tell me about the {relation_label} of {{entity}}",
                ]
            }
        
        return templates


def train_distilbert_classifier(
    train_df: pd.DataFrame,
    output_dir: str = "./models/relation_classifier",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 5e-5
):
    """
    Train DistilBERT classifier on relation dataset.
    
    Args:
        train_df: Training data with columns: text, label
        output_dir: Output directory for model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
    """
    print(f"\n🎓 Training DistilBERT Relation Classifier")
    print(f"{'='*80}")
    
    # Create label mappings
    unique_labels = sorted(train_df['label'].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    print(f"   Number of classes: {len(unique_labels)}")
    print(f"   Training examples: {len(train_df)}")
    print(f"   Output directory: {output_dir}")
    
    # Convert labels to IDs
    train_df['label_id'] = train_df['label'].map(label2id)
    
    # Split train/validation
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(train_df) * 0.9)
    
    train_data = train_df[:split_idx]
    val_data = train_df[split_idx:]
    
    print(f"   Train set: {len(train_data)} examples")
    print(f"   Val set: {len(val_data)} examples")
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_data[['text', 'label_id']])
    val_dataset = Dataset.from_pandas(val_data[['text', 'label_id']])
    
    # Load tokenizer and model
    print(f"\n   Loading DistilBERT model...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id
    )
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=128
        )
    
    print(f"   Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Rename label column
    train_dataset = train_dataset.rename_column('label_id', 'labels')
    val_dataset = val_dataset.rename_column('label_id', 'labels')
    
    # Set format
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        warmup_steps=500,
        fp16=torch.cuda.is_available(),
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print(f"\n   🚀 Starting training...")
    trainer.train()
    
    # Save model
    print(f"\n   💾 Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label mappings
    import json
    label_mapping_path = os.path.join(output_dir, 'label_mapping.json')
    with open(label_mapping_path, 'w') as f:
        json.dump({
            'label2id': label2id,
            'id2label': id2label
        }, f, indent=2)
    
    print(f"   ✅ Model saved successfully")
    print(f"\n{'='*80}")
    print(f"✅ Training complete!")
    print(f"   Model location: {output_dir}")
    print(f"   To use: HybridRelationAnalyzer(classifier_path='{output_dir}')")


def main():
    """Main training pipeline."""
    print("="*80)
    print("RELATION CLASSIFIER TRAINING")
    print("="*80 + "\n")
    
    # Step 1: Load knowledge graph
    print("📥 Loading knowledge graph...")
    sparql_handler = SPARQLHandler(graph_file_path=GRAPH_FILE_PATH)
    print(f"   ✅ Loaded graph: {len(sparql_handler.graph)} triples\n")
    
    # Step 2: Build training dataset
    builder = RelationDatasetBuilder(sparql_handler)
    
    train_df = builder.generate_training_data(
        queries_per_relation=50,  # Generate 50 queries per relation
        min_usage=10  # Only include relations used at least 10 times
    )
    
    # Save dataset
    dataset_path = "data/relation_classifier_dataset.csv"
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    train_df.to_csv(dataset_path, index=False)
    print(f"\n   💾 Saved dataset to: {dataset_path}")
    
    # Step 3: Train DistilBERT classifier
    train_distilbert_classifier(
        train_df=train_df,
        output_dir="./models/relation_classifier",
        epochs=3,
        batch_size=16,
        learning_rate=5e-5
    )
    
    print(f"\n{'='*80}")
    print("✅ ALL DONE! Relation classifier trained successfully")
    print("="*80)
    print(f"\nTo use the trained model:")
    print(f"  from src.main.relation_classifier import HybridRelationAnalyzer")
    print(f"  analyzer = HybridRelationAnalyzer(")
    print(f"      classifier_path='./models/relation_classifier',")
    print(f"      sparql_handler=sparql_handler")
    print(f"  )")


if __name__ == "__main__":
    main()
