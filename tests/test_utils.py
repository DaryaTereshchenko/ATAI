"""
Shared utility functions for test files.
Provides logging, answer extraction, and comparison utilities.
"""

import sys
import re


class TeeOutput:
    """Utility class to write output to both console and file."""
    
    def __init__(self, file_path: str):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()


def extract_answer_from_response(response: str) -> str:
    """
    Extract the actual answer from a formatted response.
    
    Args:
        response: Full formatted response string
        
    Returns:
        Extracted answer text
    """
    # Remove markdown formatting
    response = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)
    
    # For factual responses: extract text after "was" or "is"
    match = re.search(r'(?:was|is|are)\s+(?:released in\s+)?(.+?)(?:\.|$)', response, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        answer = re.sub(r'^(?:directed by|starring|written by|produced by)\s+', '', answer, flags=re.IGNORECASE)
        return answer
    
    # For embedding responses: extract "answer suggested by embeddings is: X (type: Y)"
    match = re.search(r'answer suggested by embeddings is:\s*([^(]+)\s*\(type:\s*([^)]+)\)', response, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        entity_type = match.group(2).strip()
        return f"{answer} (type: {entity_type})"
    
    # For list responses: extract items
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    for line in lines:
        if line.startswith('•') or line.startswith('-'):
            return line.lstrip('•-').strip()
    
    return response.strip()


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = ' '.join(answer.split())
    answer = answer.replace('"', '').replace("'", '')
    answer = answer.lower()
    return answer


def compare_answers(actual: str, expected: str) -> tuple:
    """
    Compare actual answer with expected answer.
    
    Returns:
        (is_match: bool, similarity_score: float)
    """
    actual_norm = normalize_answer(actual)
    expected_norm = normalize_answer(expected)
    
    # Exact match
    if actual_norm == expected_norm:
        return True, 1.0
    
    # Check if expected is contained in actual
    if expected_norm in actual_norm:
        return True, 0.9
    
    # Check word overlap for partial match
    actual_words = set(actual_norm.split())
    expected_words = set(expected_norm.split())
    
    if expected_words and actual_words:
        overlap = len(actual_words & expected_words)
        total = len(expected_words)
        similarity = overlap / total
        
        return similarity > 0.7, similarity
    
    return False, 0.0
