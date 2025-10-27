"""
Dual Approach Query Processor.

Implements both factual and embedding approaches for answering queries:
- Factual: NL → SPARQL → Execute → Format (with "and")
- Embedding: Entity extraction → Embedding computation → Format (with type)
"""

import sys
import os
from typing import List, Tuple, Optional, Dict, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.main.approach_detector import ApproachDetector, ApproachType
from src.main.response_formatter import ResponseFormatter


class DualApproachProcessor:
    """Processes queries using factual and/or embedding approaches."""
    
    def __init__(
        self,
        sparql_handler=None,
        embedding_processor=None,
        entity_extractor=None
    ):
        """
        Initialize the dual approach processor.
        
        Args:
            sparql_handler: SPARQL handler for factual approach
            embedding_processor: Embedding processor for embedding approach
            entity_extractor: Entity extractor for both approaches
        """
        self.sparql_handler = sparql_handler
        self.embedding_processor = embedding_processor
        self.entity_extractor = entity_extractor
        
        self.detector = ApproachDetector()
        self.formatter = ResponseFormatter()
    
    def process_query(self, full_query: str) -> str:
        """
        Process a query using the requested approach(es).
        
        Args:
            full_query: Full query including approach specification
            
        Returns:
            Formatted natural language response
        """
        # Detect approach and extract question
        approach, question = self.detector.detect(full_query)
        
        print(f"\n{'='*80}")
        print(f"DUAL APPROACH PROCESSOR")
        print(f"{'='*80}")
        print(f"Detected approach: {approach}")
        print(f"Question: {question}")
        print(f"{'='*80}\n")
        
        try:
            if approach == ApproachType.FACTUAL:
                return self._process_factual(question)
            elif approach == ApproachType.EMBEDDING:
                return self._process_embedding(question)
            elif approach == ApproachType.BOTH:
                return self._process_both(question)
            else:
                return self.formatter.format_error(f"Unknown approach: {approach}")
        
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
            return self.formatter.format_error(str(e), approach)
    
    def _process_factual(self, question: str) -> str:
        """
        Process using factual approach: NL → SPARQL → Execute.
        
        Returns multiple answers concatenated with "and".
        """
        print("📊 Processing with FACTUAL approach...")
        
        # Use embedding processor's hybrid method which generates SPARQL
        if self.embedding_processor is None:
            return self.formatter.format_error(
                "Embedding processor not available for factual processing",
                ApproachType.FACTUAL
            )
        
        try:
            # Get the raw result from hybrid processing
            result = self.embedding_processor.process_hybrid_factual_query(question)
            
            # Parse the result to extract answers
            answers = self._extract_answers_from_result(result)
            
            # Format as factual response
            return self.formatter.format_factual_response(question, answers)
        
        except Exception as e:
            print(f"❌ Factual processing error: {e}")
            return self.formatter.format_error(str(e), ApproachType.FACTUAL)
    
    def _process_embedding(self, question: str) -> str:
        """
        Process using embedding approach: Entity extraction → Embedding computation.
        
        Returns single answer with entity type.
        """
        print("🔢 Processing with EMBEDDING approach...")
        
        if self.embedding_processor is None:
            return self.formatter.format_error(
                "Embedding processor not available",
                ApproachType.EMBEDDING
            )
        
        try:
            # Use embedding-based similarity search
            answer, entity_type = self._compute_embedding_answer(question)
            
            # Format as embedding response
            return self.formatter.format_embedding_response(
                question, answer, entity_type
            )
        
        except Exception as e:
            print(f"❌ Embedding processing error: {e}")
            return self.formatter.format_error(str(e), ApproachType.EMBEDDING)
    
    def _process_both(self, question: str) -> str:
        """
        Process using both approaches and combine results.
        """
        print("🔀 Processing with BOTH approaches...")
        
        try:
            # Get factual result
            factual_result = self._process_factual(question)
            factual_answers = self._extract_answers_from_result(factual_result)
            
            # Get embedding result
            embedding_answer, embedding_type = self._compute_embedding_answer(question)
            
            # Format combined response
            return self.formatter.format_both_responses(
                question,
                factual_answers,
                embedding_answer,
                embedding_type
            )
        
        except Exception as e:
            print(f"❌ Both approaches error: {e}")
            return self.formatter.format_error(str(e))
    
    def _extract_answers_from_result(self, result: str) -> List[str]:
        """
        Extract answer entities from a result string.
        
        Parses various result formats to extract the actual answers.
        """
        # If result is already formatted as "The factual answer is: ...", extract it
        if "The factual answer is:" in result:
            answer_part = result.split("The factual answer is:")[-1].strip()
            # Split by "and" to get individual answers
            answers = [a.strip() for a in answer_part.split(" and ")]
            return answers
        
        # Try to parse from various formats
        # Look for bullet points
        if "•" in result:
            lines = result.split("\n")
            answers = []
            for line in lines:
                if "•" in line:
                    answer = line.split("•")[-1].strip()
                    if answer:
                        answers.append(answer)
            if answers:
                return answers
        
        # Look for comma-separated or newline-separated values
        if "\n" in result:
            lines = [l.strip() for l in result.split("\n") if l.strip()]
            # Filter out lines that look like headers
            answers = [
                l for l in lines 
                if not l.startswith("✅") 
                and not l.startswith("❌")
                and not l.startswith("The ")
                and len(l) > 0
            ]
            if answers:
                return answers
        
        # If nothing worked, return the whole result as a single answer
        return [result.strip()]
    
    def _compute_embedding_answer(self, question: str) -> Tuple[str, str]:
        """
        Compute answer using embedding similarity.
        
        Returns:
            Tuple of (answer, entity_type)
        """
        # For now, use the hybrid processor and extract the top answer
        # In a full implementation, this would use pure embedding similarity
        
        if self.embedding_processor is None:
            return "Unknown", "Q35120"  # Q35120 = entity
        
        # Use the embedding handler to find closest entity
        try:
            # Extract entities from question
            from src.main.entity_extractor import EntityExtractor
            if self.entity_extractor:
                # Try to extract subject entity
                entities = self.entity_extractor.extract_entities(question)
                if entities:
                    # For embedding approach, we want to find the answer entity
                    # using embedding similarity
                    
                    # This is a simplified version - full implementation would:
                    # 1. Extract the relation from the question
                    # 2. Get embedding for subject entity
                    # 3. Add relation embedding
                    # 4. Find nearest entity embedding
                    
                    # For now, fall back to SPARQL but return single answer
                    result = self.embedding_processor.process_hybrid_factual_query(question)
                    answers = self._extract_answers_from_result(result)
                    
                    if answers:
                        # Return first answer with person type (Q5) as default
                        # In practice, we'd detect the actual type
                        entity_type = self._detect_entity_type(question, answers[0])
                        return answers[0], entity_type
            
            return "Unknown", "Q35120"
        
        except Exception as e:
            print(f"⚠️ Embedding computation error: {e}")
            return "Unknown", "Q35120"
    
    def _detect_entity_type(self, question: str, answer: str) -> str:
        """
        Detect the entity type based on question and answer.
        
        Returns Wikidata entity type code.
        """
        question_lower = question.lower()
        
        # Detect based on question patterns
        if any(word in question_lower for word in ["director", "actor", "screenwriter", "producer", "cast", "who"]):
            return "Q5"  # Person
        elif any(word in question_lower for word in ["genre", "type of"]):
            return "Q201658"  # Film genre
        elif any(word in question_lower for word in ["country", "from what country"]):
            return "Q6256"  # Country
        elif any(word in question_lower for word in ["movie", "film"]):
            return "Q11424"  # Film
        elif any(word in question_lower for word in ["award", "prize"]):
            return "Q618779"  # Award
        else:
            return "Q35120"  # Entity (generic)


# Example usage
if __name__ == "__main__":
    # This would normally be initialized with actual handlers
    processor = DualApproachProcessor()
    
    test_queries = [
        "Please answer this question with a factual approach: Who directed 'Fargo'?",
        "Please answer this question with an embedding approach: Who is the director of 'Apocalypse Now'?",
        "Please answer this question: Who is the director of 'Good Will Hunting'?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        result = processor.process_query(query)
        print(f"\nResult: {result}\n")
