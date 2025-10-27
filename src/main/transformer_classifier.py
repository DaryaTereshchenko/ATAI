import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
from typing import Dict

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


class ClassificationResult:
    """Result from transformer classification."""
    def __init__(self, question_type: str, confidence: float):
        self.question_type = question_type
        self.confidence = confidence

class TransformerQueryClassifier:
    """Fine-tuned transformer model for query classification."""
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.75
    ):
        """
        Initialize the transformer classifier.
        
        Args:
            model_path: Path to fine-tuned model directory
            confidence_threshold: Minimum confidence for classification
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Verify the path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        print(f"🤖 Loading fine-tuned transformer classifier...")
        print(f"   Model path: {model_path}")
        
        # Check for required files
        required_files = ['config.json', 'tokenizer_config.json', 'vocab.txt']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Device: {self.device}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                local_files_only=True
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Load label mapping
            label_mapping_path = os.path.join(model_path, 'label_mapping.json')
            if os.path.exists(label_mapping_path):
                with open(label_mapping_path, 'r') as f:
                    label_data = json.load(f)
                    # Extract id2label mapping (string IDs to label names)
                    if 'id2label' in label_data:
                        self.id2label = {int(k): v for k, v in label_data['id2label'].items()}
                    else:
                        # Fallback: try direct mapping
                        self.id2label = {int(k): v for k, v in label_data.items()}
            else:
                # Default mapping if file doesn't exist
                self.id2label = {
                    0: "factual",
                    1: "multimedia",
                    2: "recommendation",
                    3: "out_of_scope"
                }
            
            print(f"✅ Transformer classifier loaded successfully")
            print(f"   Labels: {list(self.id2label.values())}\n")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load transformer classifier: {e}")
    
    def classify(self, query: str) -> ClassificationResult:
        """
        Classify a query using the fine-tuned model.
        
        Args:
            query: User query to classify
            
        Returns:
            ClassificationResult with question_type and confidence
        """
        # Tokenize input
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
            # Get predicted class and confidence
            confidence, predicted_class = torch.max(probabilities, dim=-1)
            confidence = confidence.item()
            predicted_class = predicted_class.item()
        
        # Map to label
        question_type = self.id2label.get(predicted_class, "factual")
        
        return ClassificationResult(
            question_type=question_type,
            confidence=confidence
        )