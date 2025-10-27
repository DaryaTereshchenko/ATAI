"""
Transformer-based SPARQL Pattern Classifier.
Uses fine-tuned DistilBERT to classify query patterns for SPARQL generation.
Replaces rule-based pattern matching with learned classification.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    from transformers import (
        DistilBertTokenizer,
        DistilBertForSequenceClassification,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers not available. Install with: pip install transformers torch")


@dataclass
class SPARQLPatternPrediction:
    """Result of SPARQL pattern classification."""
    pattern_type: str  # 'forward', 'reverse', 'verification', 'unknown'
    relation: str      # 'director', 'cast_member', 'genre', etc.
    confidence: float
    all_scores: Dict[str, float]  # All class probabilities


class TransformerSPARQLClassifier:
    """
    Fine-tuned transformer model for SPARQL pattern classification.
    Replaces regex-based pattern matching with learned classification.
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.6,
        device: Optional[str] = None
    ):
        """
        Initialize transformer SPARQL classifier.
        
        Args:
            model_path: Path to fine-tuned model directory
            confidence_threshold: Minimum confidence for prediction
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required. Install with: pip install transformers torch")
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"🤖 Loading SPARQL pattern classifier from: {model_path}")
        print(f"   Device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mappings
        self._load_label_mappings()
        
        print(f"✅ SPARQL classifier loaded successfully")
        print(f"   Classes: {list(self.id2label.values())}")
    
    def _load_label_mappings(self):
        """Load label mappings from model config."""
        import json
        
        # Try to load from label_mapping.json
        label_file = Path(self.model_path) / "label_mapping.json"
        if label_file.exists():
            with open(label_file, 'r') as f:
                mappings = json.load(f)
                self.label2id = mappings['label2id']
                self.id2label = {int(k): v for k, v in mappings['id2label'].items()}
        else:
            # Fallback to model config
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id
        
        # Parse composite labels (e.g., "forward_director" -> ("forward", "director"))
        self.label_components = {}
        for label_id, label in self.id2label.items():
            if '_' in label and label != 'unknown':
                parts = label.split('_', 1)
                if len(parts) == 2:
                    pattern_type, relation = parts
                    self.label_components[label] = (pattern_type, relation)
                else:
                    self.label_components[label] = (label, 'unknown')
            elif label == 'unknown':
                self.label_components[label] = ('unknown', 'unknown')
            else:
                self.label_components[label] = (label, 'general')
    
    def classify(self, query: str) -> SPARQLPatternPrediction:
        """
        Classify a natural language query into SPARQL pattern.
        
        Args:
            query: Natural language query
            
        Returns:
            SPARQLPatternPrediction with pattern type, relation, and confidence
        """
        # Tokenize
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Get prediction
        pred_id = logits.argmax(-1).item()
        confidence = probs[0, pred_id].item()
        predicted_label = self.id2label[pred_id]
        
        # Get all scores for debugging
        all_scores = {
            self.id2label[i]: probs[0, i].item()
            for i in range(len(self.id2label))
        }
        
        # Parse composite label
        if predicted_label in self.label_components:
            pattern_type, relation = self.label_components[predicted_label]
        else:
            pattern_type = predicted_label
            relation = 'unknown'
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            pattern_type = 'unknown'
            relation = 'unknown'
        
        return SPARQLPatternPrediction(
            pattern_type=pattern_type,
            relation=relation,
            confidence=confidence,
            all_scores=all_scores
        )
    
    def classify_batch(self, queries: List[str]) -> List[SPARQLPatternPrediction]:
        """
        Classify multiple queries in batch.
        
        Args:
            queries: List of natural language queries
            
        Returns:
            List of SPARQLPatternPrediction objects
        """
        # Tokenize batch
        inputs = self.tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Parse predictions
        predictions = []
        for i in range(len(queries)):
            pred_id = logits[i].argmax(-1).item()
            confidence = probs[i, pred_id].item()
            predicted_label = self.id2label[pred_id]
            
            all_scores = {
                self.id2label[j]: probs[i, j].item()
                for j in range(len(self.id2label))
            }
            
            if predicted_label in self.label_components:
                pattern_type, relation = self.label_components[predicted_label]
            else:
                pattern_type = predicted_label
                relation = 'unknown'
            
            if confidence < self.confidence_threshold:
                pattern_type = 'unknown'
                relation = 'unknown'
            
            predictions.append(SPARQLPatternPrediction(
                pattern_type=pattern_type,
                relation=relation,
                confidence=confidence,
                all_scores=all_scores
            ))
        
        return predictions
    
    def get_top_k_predictions(
        self,
        query: str,
        k: int = 3
    ) -> List[Tuple[str, str, float]]:
        """
        Get top-k predictions with pattern and relation.
        
        Args:
            query: Natural language query
            k: Number of top predictions to return
            
        Returns:
            List of (pattern_type, relation, confidence) tuples
        """
        prediction = self.classify(query)
        
        # Sort by score
        sorted_scores = sorted(
            prediction.all_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        results = []
        for label, score in sorted_scores:
            if label in self.label_components:
                pattern_type, relation = self.label_components[label]
            else:
                pattern_type = label
                relation = 'unknown'
            
            results.append((pattern_type, relation, score))
        
        return results
