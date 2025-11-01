#!/usr/bin/env python3
"""
Fuzzy search for movie titles in the knowledge graph.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.main.sparql_handler import SPARQLHandler

def search_movies_fuzzy(handler, search_term):
    """Search for movies with fuzzy matching."""
    
    query = f"""PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT DISTINCT ?movieUri ?movieLabel WHERE {{
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(?movieLabel, "{search_term}", "i"))
}}
LIMIT 20"""
    
    return handler.execute_query(query, validate=True)

def main():
    handler = SPARQLHandler(graph_file_path="data/graph.nt")
    
    # Try different search terms
    search_terms = [
        "Aro Tolbukhin",
        "Tolbukhin",
        "mente del asesino",
        "Aro.*asesino"
    ]
    
    for term in search_terms:
        print(f"\n{'='*80}")
        print(f"SEARCHING FOR: {term}")
        print('='*80)
        
        result = search_movies_fuzzy(handler, term)
        
        if result['success'] and result['data']:
            print(result['data'])
        else:
            print("No results found")

if __name__ == "__main__":
    main()
