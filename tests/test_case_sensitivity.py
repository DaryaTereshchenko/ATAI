"""
Test case-insensitive movie title matching.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.main.nl_to_sparql import NLToSPARQL
from src.main.sparql_handler import SPARQLHandler

def test_case_insensitive_queries():
    """Test that queries work regardless of case."""
    
    print("="*80)
    print("Testing Case-Insensitive Movie Title Matching")
    print("="*80)
    
    # Initialize components
    sparql_handler = SPARQLHandler()
    nl_to_sparql = NLToSPARQL(method="llm", sparql_handler=sparql_handler)
    
    # Test cases with different capitalizations
    test_cases = [
        ("Who directed 'The Bridge on the River Kwai'?", "Correct case"),
        ("Who directed 'the bridge on the river kwai'?", "All lowercase"),
        ("Who directed 'THE BRIDGE ON THE RIVER KWAI'?", "All uppercase"),
        ("Who directed 'ThE bRiDgE oN tHe RiVeR kWaI'?", "Mixed case"),
    ]
    
    for query, description in test_cases:
        print(f"\n{'='*80}")
        print(f"Test: {description}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        # Convert to SPARQL
        sparql_result = nl_to_sparql.convert(query)
        
        print(f"\nGenerated SPARQL:\n{sparql_result.query}\n")
        
        # Execute query
        try:
            result = sparql_handler.execute_and_format(sparql_result.query)
            print(f"Result: {result}")
            
            if "David Lean" in result or "Q55260" in result:
                print(f"✅ PASS: Found correct director")
            else:
                print(f"❌ FAIL: Expected 'David Lean' in result")
        except Exception as e:
            print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_case_insensitive_queries()
