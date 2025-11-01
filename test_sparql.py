"""
Test script to check if movie titles exist in the database.
Run this to diagnose entity lookup issues.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, project_root)

from src.main.sparql_handler import SPARQLHandler
from src.config import GRAPH_FILE_PATH


def test_movie_lookup():
    """Test various SPARQL queries to find the movie."""
    
    print("\n" + "="*80)
    print("TESTING MOVIE LOOKUP: 'Aro Tolbukhin. En la mente del asesino'")
    print("="*80 + "\n")
    
    # Initialize SPARQL handler
    print("Initializing SPARQL handler...")
    handler = SPARQLHandler(graph_file_path=GRAPH_FILE_PATH)
    print(f"Graph loaded: {len(handler.graph)} triples\n")
    
    # Test queries
    queries = [
        {
            "name": "Exact match (case-sensitive)",
            "query": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?movieUri ?movieLabel WHERE {
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(?movieLabel = "Aro Tolbukhin. En la mente del asesino")
}
LIMIT 10"""
        },
        {
            "name": "Partial match with 'Aro' or 'Tolbukhin' (case-insensitive)",
            "query": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?movieUri ?movieLabel WHERE {
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(?movieLabel, "Aro|Tolbukhin", "i"))
}
LIMIT 20"""
        },
        {
            "name": "Movies containing 'mente'",
            "query": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT DISTINCT ?movieLabel WHERE {
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(?movieLabel, "mente", "i"))
}
LIMIT 30"""
        },
        {
            "name": "Check label languages for 'Aro Tolbukhin'",
            "query": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?movieUri ?movieLabel (LANG(?movieLabel) AS ?lang) WHERE {
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(?movieLabel, "Aro.*Tolbukhin", "i"))
}
LIMIT 20"""
        },
        {
            "name": "Get country for movies matching 'Aro Tolbukhin'",
            "query": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?movieLabel ?countryLabel WHERE {
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(?movieLabel, "Aro.*Tolbukhin", "i"))
  ?movieUri wdt:P495 ?countryUri .
  ?countryUri rdfs:label ?countryLabel .
  FILTER(LANG(?countryLabel) = "en" || LANG(?countryLabel) = "")
}
LIMIT 10"""
        }
    ]
    
    # Execute each query
    for i, test in enumerate(queries, 1):
        print(f"\n{'─'*80}")
        print(f"TEST {i}: {test['name']}")
        print(f"{'─'*80}")
        
        try:
            result = handler.execute_query(test['query'], validate=True)
            
            if result['success']:
                data = result['data']
                
                if not data or data == "No answer found in the database.":
                    print("❌ No results found")
                else:
                    print(f"✅ Found {len(data.split(chr(10)))} results:")
                    print()
                    print(data)
            else:
                print(f"❌ Query failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error executing query: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_movie_lookup()
