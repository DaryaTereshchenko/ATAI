"""
Train DistilBERT for SPARQL Pattern Classification.
Replaces rule-based pattern matching with learned classification.
"""

import json
import torch
from pathlib import Path
from typing import Dict

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


class SPARQLClassifierTrainer:
    """Trainer for SPARQL pattern classification."""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        batch_size: int = 16,
        num_epochs: int = 5,
        learning_rate: float = 5e-5,
        output_dir: str = "models/sparql_classifier",
        cache_dir: str = "./models/cache"
    ):
        """Initialize trainer."""
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
    
    def load_dataset(self, dataset_path: str) -> Dict:
        """Load dataset from JSON."""
        print(f"Loading dataset from {dataset_path}...")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract label mappings
        self.label2id = data['metadata']['label2id']
        self.id2label = {int(k): v for k, v in data['metadata']['id2label'].items()}
        self.num_labels = data['metadata']['num_classes']
        
        # Convert to HuggingFace Dataset
        train_data = {
            'text': [item['text'] for item in data['train']],
            'label': [self.label2id[item['label']] for item in data['train']]
        }
        
        val_data = {
            'text': [item['text'] for item in data['validation']],
            'label': [self.label2id[item['label']] for item in data['validation']]
        }
        
        datasets = {
            'train': Dataset.from_dict(train_data),
            'validation': Dataset.from_dict(val_data)
        }
        
        print(f"✅ Loaded {len(datasets['train'])} train / {len(datasets['validation'])} val samples")
        print(f"   Number of classes: {self.num_labels}")
        
        return datasets
    
    def preprocess_function(self, examples):
        """Tokenize examples."""
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
    
    def compute_metrics(self, pred: EvalPrediction) -> Dict[str, float]:
        """Compute evaluation metrics."""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )
        acc = accuracy_score(labels, preds)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
        }
    
    def train(self, dataset_path: str):
        """Train the model."""
        print(f"\n{'='*80}")
        print(f"TRAINING SPARQL PATTERN CLASSIFIER")
        print(f"{'='*80}\n")
        
        # Load dataset
        datasets = self.load_dataset(dataset_path)
        
        # Tokenize
        print("\nTokenizing dataset...")
        tokenized_datasets = {
            split: dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=['text']
            )
            for split, dataset in datasets.items()
        }
        
        # Initialize model
        print(f"\nInitializing model: {self.model_name}")
        model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            cache_dir=self.cache_dir
        )
        
        print(f"✅ Model initialized with {self.num_labels} classes")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            warmup_steps=100,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=50,
            eval_steps=100,
            save_steps=500,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            push_to_hub=False,
            report_to=["tensorboard"],
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        print("\n🚀 Starting training...")
        print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        trainer.train()
        
        # Evaluate
        print("\n📊 Final evaluation...")
        eval_results = trainer.evaluate()
        
        print("\n✅ Training complete!")
        print("\nFinal Metrics:")
        for key, value in eval_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        
        # Save
        print(f"\n💾 Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save label mappings
        label_mapping = {
            'label2id': self.label2id,
            'id2label': self.id2label
        }
        with open(f"{self.output_dir}/label_mapping.json", 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        print(f"\n✅ Model saved successfully!")


def main():
    """Main training pipeline."""
    # Check if dataset exists
    dataset_path = "sparql_pattern_dataset.json"
    if not Path(dataset_path).exists():
        print(f"⚠️  Dataset not found: {dataset_path}")
        print("\nGenerate it first:")
        print("  python -m src.main.classifier_fine_tuning.sparql_dataset_generator")
        return
    
    # Train
    trainer = SPARQLClassifierTrainer(
        model_name="distilbert-base-uncased",
        num_epochs=5,
        batch_size=16,
        output_dir="models/sparql_classifier"
    )
    
    trainer.train(dataset_path)
    
    print("\n✅ Training complete!")
    print("\nTo use the model:")
    print("  from src.main.sparql_pattern_classifier import TransformerSPARQLClassifier")
    print("  classifier = TransformerSPARQLClassifier('models/sparql_classifier')")
    print("  result = classifier.classify('Who directed The Matrix?')")


if __name__ == "__main__":
    main()
