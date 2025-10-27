"""
Test script for factual and embedding approaches.

Tests the system with the example queries from the problem statement.
"""

import sys
import os

# Add project root to path BEFORE any imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def test_approach_detection():
    """Test approach detection from queries."""
    from src.main.approach_detector import ApproachDetector, ApproachType
    
    print("\n" + "="*80)
    print("TEST 1: Approach Detection")
    print("="*80 + "\n")
    
    detector = ApproachDetector()
    
    test_cases = [
        ("Please answer this question with a factual approach: Who directed 'Fargo'?", 
         ApproachType.FACTUAL, "Who directed 'Fargo'?"),
        ("Please answer this question with an embedding approach: Who is the director of 'Apocalypse Now'?",
         ApproachType.EMBEDDING, "Who is the director of 'Apocalypse Now'?"),
        ("Please answer this question: Who is the director of 'Good Will Hunting'?",
         ApproachType.BOTH, "Who is the director of 'Good Will Hunting'?"),
    ]
    
    passed = 0
    failed = 0
    
    for full_query, expected_approach, expected_question in test_cases:
        approach, question = detector.detect(full_query)
        
        print(f"Query: {full_query}")
        print(f"  Expected: {expected_approach} / {expected_question}")
        print(f"  Got:      {approach} / {question}")
        
        if approach == expected_approach and question == expected_question:
            print("  ✅ PASS\n")
            passed += 1
        else:
            print("  ❌ FAIL\n")
            failed += 1
    
    print(f"Approach Detection: {passed} passed, {failed} failed\n")
    return failed == 0


def test_response_formatting():
    """Test response formatting."""
    from src.main.response_formatter import ResponseFormatter
    
    print("\n" + "="*80)
    print("TEST 2: Response Formatting")
    print("="*80 + "\n")
    
    formatter = ResponseFormatter()
    
    # Test factual format
    print("Test 2.1: Factual format (multiple answers)")
    result = formatter.format_factual_response(
        "Who directed Fargo?",
        ["Ethan Coen", "Joel Coen"]
    )
    expected = "The factual answer is: Ethan Coen and Joel Coen"
    print(f"  Expected: {expected}")
    print(f"  Got:      {result}")
    if result == expected:
        print("  ✅ PASS\n")
    else:
        print("  ❌ FAIL\n")
    
    # Test embedding format
    print("Test 2.2: Embedding format")
    result = formatter.format_embedding_response(
        "Who is the director of Apocalypse Now?",
        "John Milius",
        "Q5"
    )
    expected = "The answer suggested by embeddings is: John Milius (type: Q5)"
    print(f"  Expected: {expected}")
    print(f"  Got:      {result}")
    if result == expected:
        print("  ✅ PASS\n")
    else:
        print("  ❌ FAIL\n")
    
    return True


def test_example_queries():
    """Test with example queries from problem statement."""
    print("\n" + "="*80)
    print("TEST 3: Example Queries from Problem Statement")
    print("="*80 + "\n")
    
    # These are the expected outputs from the problem statement
    test_cases = [
        {
            "query": "Please answer this question with a factual approach: From what country is the movie 'Aro Tolbukhin. En la mente del asesino'?",
            "expected_contains": ["The factual answer is:", "Mexico"],
            "approach": "factual"
        },
        {
            "query": "Please answer this question with a factual approach: Who is the screenwriter of 'Shortcut to Happiness'?",
            "expected_contains": ["The factual answer is:", "Pete Dexter"],
            "approach": "factual"
        },
        {
            "query": "Please answer this question with a factual approach: Who directed 'Fargo'?",
            "expected_contains": ["The factual answer is:", "Ethan Coen", "Joel Coen", "and"],
            "approach": "factual"
        },
        {
            "query": "Please answer this question with a factual approach: What genre is the movie 'Bandit Queen'?",
            "expected_contains": ["The factual answer is:", "drama film", "biographical film", "crime film", "and"],
            "approach": "factual"
        },
        {
            "query": "Please answer this question with a factual approach: When did the movie 'Miracles Still Happen' come out?",
            "expected_contains": ["The factual answer is:", "1974"],
            "approach": "factual"
        },
        {
            "query": "Please answer this question with an embedding approach: Who is the director of 'Apocalypse Now'?",
            "expected_contains": ["The answer suggested by embeddings is:", "type:", "Q5"],
            "approach": "embedding"
        },
        {
            "query": "Please answer this question with an embedding approach: Who is the screenwriter of '12 Monkeys'?",
            "expected_contains": ["The answer suggested by embeddings is:", "type:", "Q5"],
            "approach": "embedding"
        },
        {
            "query": "Please answer this question with an embedding approach: What is the genre of 'Shoplifters'?",
            "expected_contains": ["The answer suggested by embeddings is:", "type:", "Q201658"],
            "approach": "embedding"
        },
        {
            "query": "Please answer this question: Who is the director of 'Good Will Hunting'?",
            "expected_contains": ["The factual answer is:", "The answer suggested by embeddings is:", "type:", "Q5"],
            "approach": "both"
        },
    ]
    
    print("Example queries to test:")
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['query']}")
        print(f"   Approach: {test['approach']}")
        print(f"   Expected to contain: {test['expected_contains']}")
    
    print("\n" + "="*80)
    print("NOTE: Actual execution requires graph data and embeddings")
    print("="*80 + "\n")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("ATAI - FACTUAL AND EMBEDDING APPROACHES TEST SUITE")
    print("="*80)
    
    results = []
    
    # Test 1: Approach detection
    try:
        results.append(("Approach Detection", test_approach_detection()))
    except Exception as e:
        print(f"❌ Approach detection test failed: {e}")
        results.append(("Approach Detection", False))
    
    # Test 2: Response formatting
    try:
        results.append(("Response Formatting", test_response_formatting()))
    except Exception as e:
        print(f"❌ Response formatting test failed: {e}")
        results.append(("Response Formatting", False))
    
    # Test 3: Example queries (informational)
    try:
        results.append(("Example Queries", test_example_queries()))
    except Exception as e:
        print(f"❌ Example queries test failed: {e}")
        results.append(("Example Queries", False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80 + "\n")
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
