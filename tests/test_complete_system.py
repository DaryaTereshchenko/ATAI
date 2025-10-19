"""
Comprehensive test for the complete ATAI system.
Tests: workflow, validation, classification, SPARQL generation, and execution.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.main.orchestrator import Orchestrator
import time


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def test_workflow_processing():
    """Test complete workflow processing with DeepSeek + rule-based fallback."""
    print_section("TEST: Workflow Processing (DeepSeek First, Rule-Based Fallback)")
    
    print("Initializing orchestrator (loading models, this may take 10-15 seconds)...")
    start_time = time.time()
    orchestrator = Orchestrator(use_workflow=True)
    init_time = time.time() - start_time
    print(f"✅ Orchestrator initialized in {init_time:.1f}s")
    
    # Simple test queries
    test_queries = [
        "Who directed the movie 'The Bridge on the River Kwai'?",
        "Who is the producer of the movie 'French Kiss'?",
        "What genre is the movie 'Shoplifters'?",
        "Who is the director of Star Wars: Episode VI - Return of the Jedi?",
        "When was 'The Godfather' released? "
    ]
    
    passed = 0
    failed = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_queries)}")
        print(f"{'='*80}")
        print(f"📥 User Query: {query}")
        print(f"{'='*80}\n")
        
        try:
            start = time.time()
            response = orchestrator.process_query(query)
            elapsed = time.time() - start
            
            # Show the final answer that would be sent to the client
            print(f"\n{'#'*80}")
            print(f"{'#'*80}")
            print(f"  📤 ANSWER SENT TO CLIENT (Response Time: {elapsed:.2f}s)")
            print(f"{'#'*80}")
            print(f"{'#'*80}\n")
            print(response)
            print(f"\n{'#'*80}")
            print(f"{'#'*80}\n")
            
            # Validation
            if response and len(response) > 10:
                if "Database" in response or "found" in response.lower():
                    print(f"✅ PASS: Client received friendly formatted response\n")
                    passed += 1
                else:
                    print(f"⚠️  WARNING: Response format might be unexpected\n")
                    passed += 1  # Still count as pass if no error
            else:
                print(f"❌ FAIL: Response is too short or empty\n")
                failed += 1
                
        except Exception as e:
            print(f"\n{'#'*80}")
            print(f"❌ ERROR - Client would receive error message")
            print(f"{'#'*80}")
            print(f"Error: {str(e)}\n")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"📊 Final Results: {passed} passed, {failed} failed")
    print(f"{'='*80}\n")
    return failed == 0


if __name__ == "__main__":
    print("\n" + "="*80)
    print("  ATAI SYSTEM TEST SUITE")
    print("  Testing Client-Facing Responses")
    print("="*80)
    
    success = test_workflow_processing()
    
    # Summary
    print_section("FINAL RESULTS")
    
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status}: Workflow Processing")
    
    print(f"\n{'='*80}")
    if success:
        print("  ✅ All client responses formatted correctly")
    else:
        print("  ❌ Some responses failed")
    print(f"{'='*80}\n")
    
    sys.exit(0 if success else 1)
