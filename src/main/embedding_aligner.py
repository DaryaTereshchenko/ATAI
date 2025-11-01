"""
Embedding Aligner for mapping query embeddings to TransE embedding space.
Uses a learned linear projection to align different embedding spaces.
"""

import numpy as np
import os
from typing import Optional, Tuple
import pickle


class EmbeddingAligner:
    """
    Aligns query embeddings (e.g., from sentence transformers) to TransE embedding space.
    Uses a learned linear transformation matrix.
    """
    
    def __init__(
        self, 
        query_dim: int, 
        transe_dim: int,
        projection_matrix_path: Optional[str] = None
    ):
        """
        Initialize the embedding aligner.
        
        Args:
            query_dim: Dimension of query embeddings (e.g., 384 for MiniLM)
            transe_dim: Dimension of TransE embeddings
            projection_matrix_path: Path to saved projection matrix (optional)
        """
        self.query_dim = query_dim
        self.transe_dim = transe_dim
        self.projection_matrix = None
        self.bias = None
        
        if projection_matrix_path and os.path.exists(projection_matrix_path):
            self._load_projection_matrix(projection_matrix_path)
        else:
            # Initialize with identity-like projection
            self._initialize_projection()
    
    def _initialize_projection(self):
        """
        Initialize projection matrix with orthogonal initialization.
        For different dimensions, uses zero-padding or truncation.
        """
        print(f"🔧 Initializing projection matrix: {self.query_dim} → {self.transe_dim}")
        
        if self.query_dim == self.transe_dim:
            # Same dimension: start with identity
            self.projection_matrix = np.eye(self.query_dim, dtype=np.float32)
        elif self.query_dim < self.transe_dim:
            # Query embeddings are smaller: pad with small random values
            self.projection_matrix = np.random.randn(self.query_dim, self.transe_dim).astype(np.float32) * 0.01
            # Initialize main diagonal to 1
            for i in range(self.query_dim):
                self.projection_matrix[i, i] = 1.0
        else:
            # Query embeddings are larger: project down with random orthogonal matrix
            self.projection_matrix = np.random.randn(self.query_dim, self.transe_dim).astype(np.float32)
            # Orthogonalize using QR decomposition
            self.projection_matrix, _ = np.linalg.qr(self.projection_matrix)
        
        # Initialize bias to zero
        self.bias = np.zeros(self.transe_dim, dtype=np.float32)
        
    
    def align(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Align a query embedding to TransE space.
        
        Args:
            query_embedding: Query embedding vector (shape: query_dim)
            
        Returns:
            Aligned embedding in TransE space (shape: transe_dim)
        """
        if self.projection_matrix is None:
            raise RuntimeError("Projection matrix not initialized")
        
        # Linear projection: aligned = query @ projection_matrix + bias
        aligned = np.dot(query_embedding, self.projection_matrix) + self.bias
        
        return aligned
    
    def align_batch(self, query_embeddings: np.ndarray) -> np.ndarray:
        """
        Align multiple query embeddings to TransE space.
        
        Args:
            query_embeddings: Query embeddings (shape: n_queries, query_dim)
            
        Returns:
            Aligned embeddings (shape: n_queries, transe_dim)
        """
        if self.projection_matrix is None:
            raise RuntimeError("Projection matrix not initialized")
        
        # Batch projection
        aligned = np.dot(query_embeddings, self.projection_matrix) + self.bias
        
        return aligned
    
    def train(
        self, 
        query_embeddings: np.ndarray, 
        target_embeddings: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 100,
        verbose: bool = True
    ):
        """
        Train the alignment using paired query and target embeddings.
        Uses simple gradient descent to learn the projection matrix.
        
        Args:
            query_embeddings: Query embeddings (shape: n_samples, query_dim)
            target_embeddings: Target TransE embeddings (shape: n_samples, transe_dim)
            learning_rate: Learning rate for gradient descent
            epochs: Number of training epochs
            verbose: Print training progress
        """
        if query_embeddings.shape[0] != target_embeddings.shape[0]:
            raise ValueError("Number of query and target embeddings must match")
        
        if query_embeddings.shape[1] != self.query_dim:
            raise ValueError(f"Query embedding dimension mismatch: expected {self.query_dim}, got {query_embeddings.shape[1]}")
        
        if target_embeddings.shape[1] != self.transe_dim:
            raise ValueError(f"Target embedding dimension mismatch: expected {self.transe_dim}, got {target_embeddings.shape[1]}")

        
        # Solve using least squares: W = (X^T X)^-1 X^T Y
        # Where X = query_embeddings, Y = target_embeddings, W = projection_matrix
        try:
            # Add regularization for numerical stability
            lambda_reg = 0.01
            XTX = np.dot(query_embeddings.T, query_embeddings) + lambda_reg * np.eye(self.query_dim)
            XTY = np.dot(query_embeddings.T, target_embeddings)
            self.projection_matrix = np.linalg.solve(XTX, XTY).astype(np.float32)
            
            # Compute bias as mean residual
            predicted = np.dot(query_embeddings, self.projection_matrix)
            self.bias = np.mean(target_embeddings - predicted, axis=0).astype(np.float32)
            
            # Compute training loss (MSE)
            aligned = self.align_batch(query_embeddings)
            mse = np.mean((aligned - target_embeddings) ** 2)

            
        except np.linalg.LinAlgError as e:
            self._train_gradient_descent(query_embeddings, target_embeddings, learning_rate, epochs, verbose)
    
    def _train_gradient_descent(
        self,
        query_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        learning_rate: float,
        epochs: int,
        verbose: bool
    ):
        """Fallback training using gradient descent."""
        n_samples = query_embeddings.shape[0]
        
        for epoch in range(epochs):
            # Forward pass
            aligned = self.align_batch(query_embeddings)
            
            # Compute loss (MSE)
            loss = np.mean((aligned - target_embeddings) ** 2)
            
            # Backward pass
            residual = aligned - target_embeddings
            grad_W = (2.0 / n_samples) * np.dot(query_embeddings.T, residual)
            grad_b = (2.0 / n_samples) * np.sum(residual, axis=0)
            
            # Update parameters
            self.projection_matrix -= learning_rate * grad_W
            self.bias -= learning_rate * grad_b
            

        
    
    def save(self, filepath: str):
        """
        Save the projection matrix and bias to disk.
        
        Args:
            filepath: Path to save the model
        """
        data = {
            'projection_matrix': self.projection_matrix,
            'bias': self.bias,
            'query_dim': self.query_dim,
            'transe_dim': self.transe_dim
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        

    
    def _load_projection_matrix(self, filepath: str):
        """
        Load projection matrix and bias from disk.
        
        Args:
            filepath: Path to saved model
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.projection_matrix = data['projection_matrix']
            self.bias = data['bias']
            
            # Verify dimensions
            if data['query_dim'] != self.query_dim or data['transe_dim'] != self.transe_dim:
                raise ValueError(
                    f"Dimension mismatch: expected ({self.query_dim}, {self.transe_dim}), "
                    f"got ({data['query_dim']}, {data['transe_dim']})"
                )
            
            
        except Exception as e:
            self._initialize_projection()


class SimpleAligner:
    """
    Simple linear projection aligner.
    Uses a learned or identity transformation to map query embeddings to TransE space.
    """
    
    def __init__(self, query_dim: int, transe_dim: int):
        """
        Initialize simple aligner with identity or random projection.
        
        Args:
            query_dim: Dimension of query embeddings
            transe_dim: Dimension of TransE embeddings
        """
        self.query_dim = query_dim
        self.transe_dim = transe_dim
        
        # Use identity if dimensions match, otherwise random projection
        if query_dim == transe_dim:
            self.projection = np.eye(query_dim)
        else:
            # Random projection normalized by dimensions
            self.projection = np.random.randn(query_dim, transe_dim) / np.sqrt(query_dim)
        
    
    def align(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Align query embedding to TransE space.
        
        Args:
            query_embedding: Query embedding vector
            
        Returns:
            Aligned embedding in TransE space
        """
        aligned = np.dot(query_embedding, self.projection)
        
        # Normalize
        norm = np.linalg.norm(aligned)
        if norm > 0:
            aligned = aligned / norm
        
        return aligned
    
    def align_batch(self, query_embeddings: np.ndarray) -> np.ndarray:
        """Align batch of query embeddings."""
        return np.array([self.align(emb) for emb in query_embeddings])
