#!/usr/bin/env python3
from src.main.sparql_handler import SPARQLHandler

handler = SPARQLHandler(graph_file_path="data/graph.nt")

# Count total movies
query = """PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd: <http://www.wikidata.org/entity/>

SELECT (COUNT(DISTINCT ?movie) AS ?count) WHERE {
  ?movie wdt:P31 wd:Q11424 .
}"""

result = handler.execute_query(query)
print("Total movies in graph:", result['data'])

# Sample movie labels
query2 = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd: <http://www.wikidata.org/entity/>

SELECT ?label WHERE {
  ?movie wdt:P31 wd:Q11424 .
  ?movie rdfs:label ?label .
  FILTER(regex(?label, "^A", "i"))
}
LIMIT 10"""

result2 = handler.execute_query(query2)
print("\nSample movies starting with 'A':")
print(result2['data'])
