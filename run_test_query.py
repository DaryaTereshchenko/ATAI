#!/usr/bin/env python3
"""
Script to run test SPARQL queries against the knowledge graph.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.main.sparql_handler import SPARQLHandler

def main():
    """Run test SPARQL query."""
    
    # Path to your graph file
    graph_path = "/home/dariast/WS2025/ATAI/2025/ATAI/dataset/graph.nt"
    
    # SPARQL query
    query = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?movieUri ?movieLabel WHERE {
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(?movieLabel = "Aro Tolbukhin. En la mente del asesino")
}
LIMIT 10"""
    
    print("="*80)
    print("LOADING KNOWLEDGE GRAPH")
    print("="*80)
    print(f"Graph file: {graph_path}\n")
    
    # Initialize SPARQL handler
    handler = SPARQLHandler(graph_file_path=graph_path)
    
    print("\n" + "="*80)
    print("EXECUTING QUERY")
    print("="*80)
    print(query)
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")
    
    # Execute query
    result = handler.execute_query(query, validate=True)
    
    if result['success']:
        print("✅ Query executed successfully\n")
        print(result['data'])
    else:
        print("❌ Query failed\n")
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
