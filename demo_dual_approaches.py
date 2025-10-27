#!/usr/bin/env python3
"""
Demo script showing the dual approach system (without requiring actual data).

This demonstrates the approach detection and response formatting components.
"""

import sys
import os

# Add project root to path (this file is at project root)
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.main.approach_detector import ApproachDetector, ApproachType
from src.main.response_formatter import ResponseFormatter


def demo_approach_detection():
    """Demonstrate approach detection."""
    print("\n" + "="*80)
    print("DEMO 1: Approach Detection")
    print("="*80 + "\n")
    
    detector = ApproachDetector()
    
    queries = [
        "Please answer this question with a factual approach: Who directed 'Fargo'?",
        "Please answer this question with an embedding approach: Who is the director of 'Apocalypse Now'?",
        "Please answer this question: Who is the director of 'Good Will Hunting'?",
        "Who is the screenwriter of 'Shortcut to Happiness'?",
    ]
    
    for query in queries:
        approach, question = detector.detect(query)
        print(f"Original Query:")
        print(f"  {query}")
        print(f"\nDetected:")
        print(f"  Approach: {approach.value}")
        print(f"  Question: {question}")
        print()


def demo_response_formatting():
    """Demonstrate response formatting."""
    print("\n" + "="*80)
    print("DEMO 2: Response Formatting")
    print("="*80 + "\n")
    
    formatter = ResponseFormatter()
    
    # Factual - single answer
    print("Example 1: Factual approach with single answer")
    print("-" * 80)
    response = formatter.format_factual_response(
        "From what country is the movie 'Aro Tolbukhin'?",
        ["Mexico"]
    )
    print(response)
    print()
    
    # Factual - multiple answers
    print("Example 2: Factual approach with multiple answers")
    print("-" * 80)
    response = formatter.format_factual_response(
        "Who directed 'Fargo'?",
        ["Ethan Coen", "Joel Coen"]
    )
    print(response)
    print()
    
    # Factual - many answers
    print("Example 3: Factual approach with many answers (genres)")
    print("-" * 80)
    response = formatter.format_factual_response(
        "What genre is 'Bandit Queen'?",
        ["drama film", "biographical film", "crime film"]
    )
    print(response)
    print()
    
    # Embedding - person
    print("Example 4: Embedding approach (director - person)")
    print("-" * 80)
    response = formatter.format_embedding_response(
        "Who is the director of 'Apocalypse Now'?",
        "John Milius",
        "Q5"
    )
    print(response)
    print()
    
    # Embedding - genre
    print("Example 5: Embedding approach (genre)")
    print("-" * 80)
    response = formatter.format_embedding_response(
        "What is the genre of 'Shoplifters'?",
        "comedy film",
        "Q201658"
    )
    print(response)
    print()
    
    # Both approaches
    print("Example 6: Both approaches combined")
    print("-" * 80)
    response = formatter.format_both_responses(
        "Who is the director of 'Good Will Hunting'?",
        ["Gus Van Sant"],
        "Harmony Korine",
        "Q5"
    )
    print(response)
    print()


def demo_workflow():
    """Demonstrate the complete workflow logic."""
    print("\n" + "="*80)
    print("DEMO 3: Complete Workflow Logic")
    print("="*80 + "\n")
    
    detector = ApproachDetector()
    formatter = ResponseFormatter()
    
    # Simulate processing different query types
    test_cases = [
        {
            "query": "Please answer this question with a factual approach: Who directed 'Fargo'?",
            "mock_factual_results": ["Ethan Coen", "Joel Coen"],
            "mock_embedding_result": None,
            "mock_embedding_type": None
        },
        {
            "query": "Please answer this question with an embedding approach: Who is the director of 'Apocalypse Now'?",
            "mock_factual_results": None,
            "mock_embedding_result": "John Milius",
            "mock_embedding_type": "Q5"
        },
        {
            "query": "Please answer this question: Who is the director of 'Good Will Hunting'?",
            "mock_factual_results": ["Gus Van Sant"],
            "mock_embedding_result": "Harmony Korine",
            "mock_embedding_type": "Q5"
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print("-" * 80)
        print(f"Query: {test['query']}")
        print()
        
        # Detect approach
        approach, question = detector.detect(test['query'])
        print(f"Step 1: Detect approach")
        print(f"  → Approach: {approach.value}")
        print(f"  → Extracted question: {question}")
        print()
        
        # Route to appropriate handler
        print(f"Step 2: Route to handler")
        if approach == ApproachType.FACTUAL:
            print(f"  → Route to: Factual Processor (NL → SPARQL → Execute)")
        elif approach == ApproachType.EMBEDDING:
            print(f"  → Route to: Embedding Processor (TransE computation)")
        elif approach == ApproachType.BOTH:
            print(f"  → Route to: Both processors")
        print()
        
        # Format response
        print(f"Step 3: Format response")
        if approach == ApproachType.FACTUAL and test['mock_factual_results']:
            response = formatter.format_factual_response(question, test['mock_factual_results'])
        elif approach == ApproachType.EMBEDDING and test['mock_embedding_result']:
            response = formatter.format_embedding_response(
                question, 
                test['mock_embedding_result'],
                test['mock_embedding_type']
            )
        elif approach == ApproachType.BOTH:
            response = formatter.format_both_responses(
                question,
                test['mock_factual_results'],
                test['mock_embedding_result'],
                test['mock_embedding_type']
            )
        
        print(f"  → Response: {response}")
        print()


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("DUAL APPROACH SYSTEM DEMONSTRATION")
    print("="*80)
    print("\nThis demo shows how the system detects approaches and formats responses")
    print("without requiring actual knowledge graph data or embeddings.")
    
    demo_approach_detection()
    demo_response_formatting()
    demo_workflow()
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nFor actual query processing with data, use:")
    print("  python src/main/bot.py")
    print("\nOr run tests with:")
    print("  python tests/test_dual_approaches.py")
    print()


if __name__ == "__main__":
    main()
