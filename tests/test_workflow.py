"""
Test the workflow system with various query types.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.main.orchestrator import Orchestrator
from src.main.workflow import InputValidator


def test_input_validation():
    """Test input validation."""
    validator = InputValidator()
    
    # Valid queries
    assert validator.validate("Who directed Star Wars?")['is_valid']
    assert validator.validate("What is the genre of Inception?")['is_valid']
    
    # Invalid queries
    assert not validator.validate("")['is_valid']
    assert not validator.validate("x")['is_valid']
    assert not validator.validate("DROP TABLE movies;")['is_valid']
    assert not validator.validate("<script>alert('xss')</script>")['is_valid']
    
    print("✅ Input validation tests passed")


def test_workflow_processing():
    """Test complete workflow processing."""
    orchestrator = Orchestrator(use_workflow=True)
    
    # Test queries
    test_queries = [
        "Who directed Star Wars?",
        "When was The Godfather released?",
        "What is the genre of Inception?",
    ]
    
    for query in test_queries:
        print(f"\nTesting: {query}")
        response = orchestrator.process_query(query)
        print(f"Response: {response[:200]}...")
        assert response
        assert "processed using" in response.lower()
    
    print("\n✅ Workflow processing tests passed")


def test_malicious_input():
    """Test handling of malicious input."""
    orchestrator = Orchestrator(use_workflow=True)
    
    malicious_queries = [
        "DROP TABLE movies;",
        "<script>alert('xss')</script>",
        "; DELETE FROM movies",
    ]
    
    for query in malicious_queries:
        print(f"\nTesting malicious: {query}")
        response = orchestrator.process_query(query)
        print(f"Response: {response[:200]}...")
        assert "Security Warning" in response or "unsafe" in response.lower()
    
    print("\n✅ Malicious input tests passed")


if __name__ == "__main__":
    print("Running workflow tests...\n")
    test_input_validation()
    test_workflow_processing()
    test_malicious_input()
    print("\n✅ All tests passed!")
