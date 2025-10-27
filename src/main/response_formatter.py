"""
Response Formatter for Factual and Embedding Approaches.

Formats answers according to the requirements:
- Factual: Multiple answers concatenated with "and"
- Embedding: Single answer with entity type
"""

from typing import List, Optional, Dict, Any
from src.main.approach_detector import ApproachType


class ResponseFormatter:
    """Formats query responses based on approach type."""
    
    @staticmethod
    def format_factual_response(
        question: str,
        answers: List[str],
        property_asked: Optional[str] = None
    ) -> str:
        """
        Format factual approach response.
        
        Multiple answers are concatenated with "and".
        
        Args:
            question: The original question
            answers: List of answer strings
            property_asked: Optional property name (e.g., "director", "genre")
            
        Returns:
            Formatted natural language response
        """
        if not answers:
            return "The factual answer is: No results found"
        
        # Sort answers for consistency (case-insensitive)
        sorted_answers = sorted(answers, key=lambda x: x.lower())
        
        # Join with " and "
        answer_text = " and ".join(sorted_answers)
        
        return f"The factual answer is: {answer_text}"
    
    @staticmethod
    def format_embedding_response(
        question: str,
        answer: str,
        entity_type: str,
        confidence: Optional[float] = None
    ) -> str:
        """
        Format embedding approach response.
        
        Single answer with entity type information.
        
        Args:
            question: The original question
            answer: Single answer string
            entity_type: Wikidata entity type (e.g., "Q5" for person, "Q201658" for genre)
            confidence: Optional confidence score
            
        Returns:
            Formatted natural language response
        """
        if not answer:
            return "The answer suggested by embeddings is: No results found"
        
        # Format entity type
        type_display = f"type: {entity_type}"
        
        # Add confidence if provided
        if confidence is not None:
            type_display += f", confidence: {confidence:.2%}"
        
        return f"The answer suggested by embeddings is: {answer} ({type_display})"
    
    @staticmethod
    def format_both_responses(
        question: str,
        factual_answers: List[str],
        embedding_answer: str,
        embedding_type: str,
        factual_property: Optional[str] = None,
        embedding_confidence: Optional[float] = None
    ) -> str:
        """
        Format response when both approaches are used.
        
        Args:
            question: The original question
            factual_answers: List of factual answers
            embedding_answer: Single embedding answer
            embedding_type: Entity type for embedding answer
            factual_property: Optional property name for factual
            embedding_confidence: Optional confidence for embedding
            
        Returns:
            Combined formatted response
        """
        factual_part = ResponseFormatter.format_factual_response(
            question, factual_answers, factual_property
        )
        
        embedding_part = ResponseFormatter.format_embedding_response(
            question, embedding_answer, embedding_type, embedding_confidence
        )
        
        # Keep both parts as is, add period separator
        return f"{factual_part}. {embedding_part}"
    
    @staticmethod
    def format_error(
        error_message: str,
        approach: Optional[ApproachType] = None
    ) -> str:
        """
        Format error response.
        
        Args:
            error_message: Error description
            approach: Optional approach type that failed
            
        Returns:
            Formatted error message
        """
        prefix = "Error"
        if approach == ApproachType.FACTUAL:
            prefix = "Factual approach error"
        elif approach == ApproachType.EMBEDDING:
            prefix = "Embedding approach error"
        
        return f"❌ {prefix}: {error_message}"


# Example usage
if __name__ == "__main__":
    formatter = ResponseFormatter()
    
    # Test factual format (single answer)
    print("Test 1: Single factual answer")
    result = formatter.format_factual_response(
        "Who directed Fargo?",
        ["Ethan Coen", "Joel Coen"]
    )
    print(result)
    print()
    
    # Test factual format (multiple answers)
    print("Test 2: Multiple factual answers")
    result = formatter.format_factual_response(
        "What genre is Bandit Queen?",
        ["drama film", "biographical film", "crime film"]
    )
    print(result)
    print()
    
    # Test embedding format
    print("Test 3: Embedding answer")
    result = formatter.format_embedding_response(
        "Who is the director of Apocalypse Now?",
        "John Milius",
        "Q5"
    )
    print(result)
    print()
    
    # Test both approaches
    print("Test 4: Both approaches")
    result = formatter.format_both_responses(
        "Who is the director of Good Will Hunting?",
        ["Gus Van Sant"],
        "Harmony Korine",
        "Q5"
    )
    print(result)
    print()
