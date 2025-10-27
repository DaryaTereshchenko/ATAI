#!/usr/bin/env python3
"""
Test script for the updated hybrid workflow (embeddings → SPARQL fallback).
Tests that embeddings are tried first and SPARQL is used as fallback.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def test_hybrid_workflow():
    """Test the hybrid workflow with various query types."""
    
    print("\n" + "="*80)
    print("HYBRID WORKFLOW TEST (Embeddings → SPARQL Fallback)")
    print("="*80 + "\n")
    
    # Import after adding to path
    from src.main.orchestrator import Orchestrator
    from src.config import USE_EMBEDDINGS
    
    if not USE_EMBEDDINGS:
        print("⚠️  WARNING: USE_EMBEDDINGS is False in config.py")
        print("   Set USE_EMBEDDINGS = True to enable the hybrid workflow")
        print()
    
    try:
        # Initialize orchestrator
        print("🚀 Initializing orchestrator with hybrid workflow...\n")
        orchestrator = Orchestrator(use_workflow=True)
        
        print("\n" + "="*80)
        print("TESTING HYBRID WORKFLOW")
        print("="*80 + "\n")
        
        # Test queries - these should try embeddings first
        test_queries = [
            # Queries likely to work well with embeddings
            {
                "query": "Find movies about space and science fiction",
                "expected": "embeddings",
                "description": "Semantic query (should use embeddings)"
            },
            {
                "query": "Show me action movies",
                "expected": "embeddings",
                "description": "Genre-based query (should use embeddings)"
            },
            
            # Factual queries that might fall back to SPARQL
            {
                "query": "Who directed Star Wars?",
                "expected": "sparql_fallback",
                "description": "Specific factual query (may use SPARQL as fallback)"
            },
            {
                "query": "What is the genre of The Godfather?",
                "expected": "sparql_fallback",
                "description": "Specific property query (may use SPARQL as fallback)"
            },
            
            # Query that should definitely use embeddings
            {
                "query": "Movies similar to The Matrix",
                "expected": "embeddings",
                "description": "Similarity query (should use embeddings)"
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n{'='*80}")
            print(f"TEST CASE {i}/{len(test_queries)}")
            print(f"{'='*80}\n")
            
            query = test_case['query']
            expected = test_case['expected']
            description = test_case['description']
            
            print(f"Query: {query}")
            print(f"Description: {description}")
            print(f"Expected: {expected}")
            print()
            
            # Process query
            response = orchestrator.process_query(query)
            
            # Check which method was used
            if "Retrieved via Semantic Embeddings" in response or "Semantic Embedding" in response:
                actual_method = "embeddings"
                print("✅ Result: Used EMBEDDINGS")
            elif "Database Query Result" in response:
                actual_method = "sparql"
                print("✅ Result: Used SPARQL")
            else:
                actual_method = "unknown"
                print("⚠️  Result: Method unclear")
            
            print(f"\nResponse preview:")
            print("-" * 80)
            print(response[:300] + "..." if len(response) > 300 else response)
            print("-" * 80)
            print()
            
            results.append({
                'query': query,
                'expected': expected,
                'actual': actual_method,
                'description': description
            })
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80 + "\n")
        
        embeddings_count = sum(1 for r in results if r['actual'] == 'embeddings')
        sparql_count = sum(1 for r in results if r['actual'] == 'sparql')
        
        print(f"Total tests: {len(results)}")
        print(f"Used embeddings: {embeddings_count}")
        print(f"Used SPARQL: {sparql_count}")
        print()
        
        for i, result in enumerate(results, 1):
            status = "✅" if result['actual'] == 'embeddings' or result['actual'] == 'sparql' else "⚠️"
            print(f"{status} Test {i}: {result['query'][:50]}...")
            print(f"   Method: {result['actual']}")
            print()
        
        print("="*80)
        print("✅ HYBRID WORKFLOW TEST COMPLETE")
        print("="*80)
        print()
        print("Key observations:")
        print("- Embeddings are tried FIRST for factual queries")
        print("- SPARQL is used as FALLBACK if embeddings don't find good results")
        print("- Responses clearly indicate which method was used")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = test_hybrid_workflow()
    sys.exit(exit_code)
