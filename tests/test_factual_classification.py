"""
Master test runner for all query processing approaches.
Runs factual, embedding, and hybrid test suites.
"""

import sys
import os
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def test_all_approaches():
    """Run all three test suites and report combined results."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE PIPELINE TEST - ALL APPROACHES")
    print("="*80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    from tests.test_factual_queries import test_factual_queries
    from tests.test_embedding_queries import test_embedding_queries
    from tests.test_hybrid_queries import test_hybrid_queries
    
    # Run each test suite
    print("\n" + "🔵" * 40)
    print("Running FACTUAL queries test suite...")
    print("🔵" * 40)
    factual_results = test_factual_queries()
    
    print("\n" + "🟢" * 40)
    print("Running EMBEDDING queries test suite...")
    print("🟢" * 40)
    embedding_results = test_embedding_queries()
    
    print("\n" + "🟡" * 40)
    print("Running HYBRID queries test suite...")
    print("🟡" * 40)
    hybrid_results = test_hybrid_queries()
    
    # Combined summary
    print("\n" + "="*80)
    print("COMBINED TEST SUMMARY - ALL APPROACHES")
    print("="*80)
    
    total_tests = factual_results['total'] + embedding_results['total'] + hybrid_results['total']
    total_correct = factual_results['answer_correct'] + embedding_results['answer_correct'] + hybrid_results['answer_correct']
    total_partial = factual_results['answer_partial'] + embedding_results['answer_partial'] + hybrid_results['answer_partial']
    
    print(f"\n📊 Overall Results:")
    print(f"  Total Tests:        {total_tests}")
    print(f"  ✅ Correct:         {total_correct}")
    print(f"  ⚠️  Partial:         {total_partial}")
    print(f"  ❌ Incorrect:       {total_tests - total_correct - total_partial}")
    
    print(f"\n  By Approach:")
    print(f"    🔵 Factual:       {factual_results['answer_correct']}/{factual_results['total']}")
    print(f"    🟢 Embedding:     {embedding_results['answer_correct']}/{embedding_results['total']}")
    print(f"    🟡 Hybrid:        {hybrid_results['answer_correct']}/{hybrid_results['total']}")
    
    overall_success = total_correct / total_tests if total_tests > 0 else 0
    print(f"\n  Overall Success:    {total_correct}/{total_tests} ({overall_success:.1%})")
    
    if total_correct == total_tests:
        print("\n🎉 ALL TESTS PASSED!\n")
    elif overall_success >= 0.7:
        print(f"\n✅ Most tests passed ({overall_success:.1%} success rate)\n")
    else:
        print(f"\n⚠️  {total_tests - total_correct} test(s) failed\n")
    
    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    test_all_approaches()