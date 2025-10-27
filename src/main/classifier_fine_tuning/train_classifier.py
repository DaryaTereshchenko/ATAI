"""
Fine-tune DistilBERT for query classification.
Lightweight, fast, and accurate for text classification tasks.
"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


@dataclass
class ClassifierConfig:
    """Configuration for classifier training."""
    model_name: str = "distilbert-base-uncased"  # Lightweight (~66M params)
    # Alternative: "distilroberta-base" or "microsoft/deberta-v3-small"
    
    max_length: int = 128
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    output_dir: str = "models/query_classifier"
    cache_dir: str = "./models/cache"  # Local cache
    logging_steps: int = 50
    eval_steps: int = 100
    save_steps: int = 500


class QueryClassifierTrainer:
    """Trainer for fine-tuning transformer models on query classification."""
    
    def __init__(self, config: ClassifierConfig = None):
        """Initialize trainer with configuration."""
        self.config = config or ClassifierConfig()
        
        # Create cache directory if it doesn't exist
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # ✅ UPDATED: Label mapping with 4 classes (including out_of_scope)
        self.label2id = {
            'factual': 0,
            'multimedia': 1,
            'recommendation': 2,
            'out_of_scope': 3  # ✅ NEW: Negative class
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # Initialize tokenizer
        print(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir
        )
        
        # Model will be initialized during training
        self.model = None
    
    def load_dataset(self, dataset_path: str) -> Dict[str, Dataset]:
        """Load dataset from JSON file."""
        print(f"Loading dataset from {dataset_path}...")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to HuggingFace Dataset format
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
        
        print(f"✅ Loaded {len(datasets['train'])} training samples")
        print(f"✅ Loaded {len(datasets['validation'])} validation samples")
        
        return datasets
    
    def preprocess_function(self, examples):
        """Tokenize and prepare examples for training."""
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=self.config.max_length
        )
    
    def compute_metrics(self, pred: EvalPrediction) -> Dict[str, float]:
        """Compute evaluation metrics."""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted'
        )
        acc = accuracy_score(labels, preds)
        
        # Per-class metrics
        per_class_f1 = f1_score(labels, preds, average=None)
        
        metrics = {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
        }
        
        # Add per-class F1
        for i, label in self.id2label.items():
            if i < len(per_class_f1):
                metrics[f'f1_{label}'] = per_class_f1[i]
        
        return metrics
    
    def train(self, dataset_path: str):
        """
        Fine-tune the model on the dataset.
        
        Args:
            dataset_path: Path to JSON dataset file
        """
        print(f"\n{'='*80}")
        print(f"FINE-TUNING {self.config.model_name}")
        print(f"{'='*80}\n")
        
        # Load and preprocess dataset
        datasets = self.load_dataset(dataset_path)
        
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
        print(f"\nInitializing model: {self.config.model_name}")
        
        # Check if model is already cached
        cache_path = Path(self.config.cache_dir) / f"models--{self.config.model_name.replace('/', '--')}"
        if cache_path.exists():
            print(f"✅ Using cached model from: {cache_path}")
        else:
            print(f"⬇️  Downloading model from Hugging Face Hub...")
            print(f"   This may take a few minutes (~250MB download)")
            print(f"   Model will be cached in: {self.config.cache_dir}")
        
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
            cache_dir=self.config.cache_dir
        )
        
        print(f"✅ Model loaded successfully")
        print(f"ℹ️  Note: New classifier layers initialized (normal for fine-tuning)")
        
        # ✅ FIXED: Training arguments compatible with newer transformers versions
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            eval_strategy="steps",  # ✅ FIXED: Changed from 'evaluation_strategy'
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            push_to_hub=False,
            report_to=["tensorboard"],
            remove_unused_columns=True,
            disable_tqdm=False,  # Show progress bar
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        print("\n🚀 Starting training...")
        print(f"   Epochs: {self.config.num_epochs}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Learning rate: {self.config.learning_rate}")
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
            else:
                print(f"  {key}: {value}")
        
        # Save model
        print(f"\n💾 Saving model to {self.config.output_dir}")
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save label mappings
        import json
        label_mapping = {
            'label2id': self.label2id,
            'id2label': self.id2label
        }
        with open(f"{self.config.output_dir}/label_mapping.json", 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        print(f"\n✅ Model saved successfully!")
        print(f"   Model files: {self.config.output_dir}/")
        print(f"   - pytorch_model.bin")
        print(f"   - config.json")
        print(f"   - tokenizer files")
        print(f"   - label_mapping.json")
    
    def evaluate_on_rules(self, test_queries: List[str]):
        """
        Compare fine-tuned model with rule-based classifier.
        
        Args:
            test_queries: List of queries to test
        """
        if self.model is None:
            print("⚠️  Model not trained yet. Train the model first.")
            return
        
        try:
            from query_classifier import RobustQueryClassifier
        except ImportError:
            print("⚠️  RobustQueryClassifier not found. Skipping comparison.")
            return
        
        rule_classifier = RobustQueryClassifier()
        
        print(f"\n{'='*80}")
        print(f"COMPARING MODELS")
        print(f"{'='*80}\n")
        
        agreements = 0
        disagreements = []
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        
        for query in test_queries:
            # Rule-based prediction
            rule_result = rule_classifier.classify(query)
            rule_label = rule_result.question_type.value
            
            # Transformer prediction
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_id = outputs.logits.argmax(-1).item()
                confidence = probs[0, pred_id].item()
                transformer_label = self.id2label[pred_id]
            
            if rule_label == transformer_label:
                agreements += 1
                print(f"✅ AGREE: {query[:60]}...")
                print(f"   Both predicted: {rule_label} (confidence: {confidence:.2%})")
            else:
                disagreements.append({
                    'query': query,
                    'rule_based': rule_label,
                    'transformer': transformer_label,
                    'confidence': confidence
                })
                print(f"❌ DISAGREE: {query[:60]}...")
                print(f"   Rule-based: {rule_label}")
                print(f"   Transformer: {transformer_label} (confidence: {confidence:.2%})")
            print()
        
        print(f"\n{'='*80}")
        print(f"Agreement: {agreements}/{len(test_queries)} "
              f"({agreements/len(test_queries)*100:.1f}%)")
        print(f"{'='*80}")


def main():
    """Main training pipeline."""
    
    # Step 1: Generate synthetic dataset (if not exists)
    dataset_path = "synthetic_query_dataset.json"
    if not Path(dataset_path).exists():
        print("⚠️  Dataset not found. Please generate it first:")
        print("\n   Or place your dataset at: synthetic_query_dataset.json")
        return
    
    # Step 2: Fine-tune model
    config = ClassifierConfig(
        model_name="distilbert-base-uncased",  # Fast and lightweight
        num_epochs=3,
        batch_size=16,
        learning_rate=5e-5,
        output_dir="models/query_classifier",
        cache_dir="./models/cache",
    )
    
    trainer = QueryClassifierTrainer(config)
    trainer.train(dataset_path)
    
    # Step 3: Test on sample queries
    print("\n" + "="*80)
    print("TESTING ON SAMPLE QUERIES")
    print("="*80)
    
    test_queries = [
        "Who directed The Godfather?",
        "Show me a picture of Star Wars",
        "Recommend me a good action movie",
        "What genre is Inception?",
        "What should I watch tonight?",
        "When was Titanic released?",
        "Find movies directed by Christopher Nolan",
        "Display the poster for The Matrix",
    ]
    
    trainer.evaluate_on_rules(test_queries)
    
    print("\n✅ Training pipeline complete!")
    print(f"\nTo use the fine-tuned model:")

    print(f"  classifier = TransformerQueryClassifier('models/query_classifier')")
    print(f"  result = classifier.classify('your query')")


if __name__ == "__main__":
    main()