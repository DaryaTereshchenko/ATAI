"""
Approach Detector for Query Processing.

Detects whether a query requests factual or embedding approach,
and extracts the actual question from the query.
"""

import re
from typing import Tuple, Optional
from enum import Enum


class ApproachType(str, Enum):
    """Type of approach requested in the query."""
    FACTUAL = "factual"
    EMBEDDING = "embedding"
    BOTH = "both"  # When not specified, use both approaches


class ApproachDetector:
    """Detects the requested approach from a user query."""
    
    # Patterns to detect approach requests
    FACTUAL_PATTERNS = [
        r"(?:please\s+)?answer\s+(?:this\s+)?question\s+with\s+a\s+factual\s+approach",
        r"using\s+(?:a\s+)?factual\s+approach",
        r"with\s+(?:a\s+)?factual\s+approach",
        r"factual\s+approach",
    ]
    
    EMBEDDING_PATTERNS = [
        r"(?:please\s+)?answer\s+(?:this\s+)?question\s+with\s+an?\s+embeddings?\s+approach",
        r"using\s+(?:an?\s+)?embeddings?\s+approach",
        r"with\s+(?:an?\s+)?embeddings?\s+approach",
        r"embeddings?\s+approach",
    ]
    
    BOTH_PATTERNS = [
        r"(?:please\s+)?answer\s+(?:this\s+)?question(?:\s*:\s*)?",  # Just "Please answer this question:"
    ]
    
    @classmethod
    def detect(cls, query: str) -> Tuple[ApproachType, str]:
        """
        Detect the requested approach and extract the actual question.
        
        Args:
            query: Full user query including approach specification
            
        Returns:
            Tuple of (approach_type, cleaned_question)
        """
        query_lower = query.lower()
        
        # Check for factual approach
        for pattern in cls.FACTUAL_PATTERNS:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                # Extract the actual question (everything after the pattern)
                cleaned = cls._extract_question(query, match)
                return ApproachType.FACTUAL, cleaned
        
        # Check for embedding approach
        for pattern in cls.EMBEDDING_PATTERNS:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                # Extract the actual question (everything after the pattern)
                cleaned = cls._extract_question(query, match)
                return ApproachType.EMBEDDING, cleaned
        
        # Check for "both" pattern (not specified)
        for pattern in cls.BOTH_PATTERNS:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                # Extract the actual question (everything after the pattern)
                cleaned = cls._extract_question(query, match)
                return ApproachType.BOTH, cleaned
        
        # Default: treat as both approaches if no specific approach mentioned
        return ApproachType.BOTH, query
    
    @classmethod
    def _extract_question(cls, query: str, match: re.Match) -> str:
        """
        Extract the actual question from the query after removing approach specification.
        
        Args:
            query: Original query
            match: Regex match object for the approach pattern
            
        Returns:
            Cleaned question
        """
        # Get text after the match
        question = query[match.end():].strip()
        
        # Remove leading colon or question mark if present
        question = re.sub(r'^[:\s?]+', '', question).strip()
        
        # If question is empty, return the whole original query
        # (this handles edge cases)
        if not question:
            return query
        
        return question
    
    @classmethod
    def is_factual_approach(cls, query: str) -> bool:
        """Quick check if query requests factual approach."""
        approach, _ = cls.detect(query)
        return approach in [ApproachType.FACTUAL, ApproachType.BOTH]
    
    @classmethod
    def is_embedding_approach(cls, query: str) -> bool:
        """Quick check if query requests embedding approach."""
        approach, _ = cls.detect(query)
        return approach in [ApproachType.EMBEDDING, ApproachType.BOTH]


# Example usage
if __name__ == "__main__":
    test_queries = [
        "Please answer this question with a factual approach: Who is the director of 'Fargo'?",
        "Please answer this question with an embedding approach: Who is the director of 'Apocalypse Now'?",
        "Please answer this question: Who is the director of 'Good Will Hunting'?",
        "Who directed 'The Matrix'?",
    ]
    
    detector = ApproachDetector()
    
    for query in test_queries:
        approach, question = detector.detect(query)
        print(f"\nQuery: {query}")
        print(f"Approach: {approach}")
        print(f"Question: {question}")
