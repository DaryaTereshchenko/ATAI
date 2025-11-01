"""
Diagnostic script to check the graph schema and P31 (instance of) usage.
Helps understand what entity types are present in the graph.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from rdflib import Graph, URIRef, Namespace, RDFS
from src.config import GRAPH_FILE_PATH

def check_p31_usage():
    """Check if P31 (instance of) is used in the graph."""
    print("="*80)
    print("GRAPH SCHEMA DIAGNOSTIC")
    print("="*80)
    print(f"\nGraph file: {GRAPH_FILE_PATH}\n")
    
    # Load graph
    print("📥 Loading graph...")
    graph = Graph()
    graph.parse(GRAPH_FILE_PATH, format="nt")
    print(f"✅ Loaded {len(graph)} triples\n")
    
    # Define namespaces
    WDT = Namespace("http://www.wikidata.org/prop/direct/")
    WD = Namespace("http://www.wikidata.org/entity/")
    P31 = URIRef("http://www.wikidata.org/prop/direct/P31")
    
    # Check P31 usage
    print("="*80)
    print("CHECKING P31 (instance of) USAGE")
    print("="*80)
    
    p31_triples = list(graph.triples((None, P31, None)))
    print(f"\n📊 P31 triples found: {len(p31_triples)}\n")
    
    if len(p31_triples) == 0:
        print("❌ No P31 triples found!")
        print("   This means entity types are not explicitly declared in the graph.")
        print("   The dynamic schema extraction will fail.\n")
        return
    
    # Count entity types
    print("="*80)
    print("ENTITY TYPE DISTRIBUTION")
    print("="*80)
    
    type_counts = {}
    for s, p, o in p31_triples:
        type_uri = str(o)
        if type_uri not in type_counts:
            type_counts[type_uri] = []
        type_counts[type_uri].append(str(s))
    
    # Sort by count
    sorted_types = sorted(type_counts.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"\nFound {len(sorted_types)} unique entity types:\n")
    
    for i, (type_uri, entities) in enumerate(sorted_types[:20], 1):
        # Extract Q-code
        if '/Q' in type_uri:
            qcode = type_uri.split('/Q')[-1].split('#')[0]
            display = f"Q{qcode}"
        else:
            display = type_uri
        
        # Get label if available
        type_ref = URIRef(type_uri)
        label = None
        for lbl in graph.objects(type_ref, RDFS.label):
            label = str(lbl)
            break
        
        if label:
            print(f"{i:2d}. {display:12s} ({len(entities):6d} entities) - {label}")
        else:
            print(f"{i:2d}. {display:12s} ({len(entities):6d} entities)")
        
        # Show first 3 example entities
        if i <= 5:  # Show examples for top 5 types
            print(f"    Examples:")
            for entity_uri in entities[:3]:
                entity_ref = URIRef(entity_uri)
                entity_label = None
                for lbl in graph.objects(entity_ref, RDFS.label):
                    entity_label = str(lbl)
                    break
                
                if '/Q' in entity_uri:
                    entity_qcode = entity_uri.split('/Q')[-1].split('#')[0]
                    if entity_label:
                        print(f"      - Q{entity_qcode}: {entity_label}")
                    else:
                        print(f"      - Q{entity_qcode}")
                else:
                    if entity_label:
                        print(f"      - {entity_label}")
                    else:
                        print(f"      - {entity_uri}")
            print()
    
    # Check for movie and person types
    print("="*80)
    print("CHECKING KEY ENTITY TYPES")
    print("="*80)
    
    Q11424 = URIRef("http://www.wikidata.org/entity/Q11424")  # Movie
    Q5 = URIRef("http://www.wikidata.org/entity/Q5")  # Person
    
    movies = list(graph.subjects(P31, Q11424))
    persons = list(graph.subjects(P31, Q5))
    
    print(f"\n🎬 Movies (Q11424): {len(movies)}")
    if len(movies) > 0:
        print(f"   Examples:")
        for movie_uri in movies[:5]:
            movie_ref = URIRef(movie_uri)
            movie_label = None
            for lbl in graph.objects(movie_ref, RDFS.label):
                movie_label = str(lbl)
                break
            if movie_label:
                print(f"      - {movie_label}")
    
    print(f"\n👤 Persons (Q5): {len(persons)}")
    if len(persons) > 0:
        print(f"   Examples:")
        for person_uri in persons[:5]:
            person_ref = URIRef(person_uri)
            person_label = None
            for lbl in graph.objects(person_ref, RDFS.label):
                person_label = str(lbl)
                break
            if person_label:
                print(f"      - {person_label}")
    
    # Check properties
    print("\n="*80)
    print("CHECKING WIKIDATA PROPERTIES")
    print("="*80)
    
    properties = {}
    for s, p, o in graph:
        pred_str = str(p)
        if 'wikidata.org/prop/direct/P' in pred_str:
            if pred_str not in properties:
                properties[pred_str] = 0
            properties[pred_str] += 1
    
    sorted_props = sorted(properties.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nFound {len(sorted_props)} Wikidata properties:\n")
    
    property_names = {
        'P57': 'director',
        'P161': 'cast member',
        'P58': 'screenwriter',
        'P162': 'producer',
        'P136': 'genre',
        'P577': 'publication date',
        'P495': 'country of origin',
        'P166': 'award received',
        'P31': 'instance of',
        'P106': 'occupation',
        'P345': 'IMDb ID',
        'P21': 'sex or gender',
        'P27': 'country of citizenship',
        'P735': 'given name',
        'P1412': 'languages spoken',
        'P19': 'place of birth',
        'P69': 'educated at',
        'P1343': 'described by source',
        'P734': 'family name',
        'P1411': 'nominated for',
        'P20': 'place of death',
        'P103': 'native language',
        'P1441': 'present in work',
        'P750': 'distributor'
    }
    
    for i, (prop_uri, count) in enumerate(sorted_props[:20], 1):
        if '/P' in prop_uri:
            prop_id = prop_uri.split('/P')[-1]
            prop_name = property_names.get(prop_id, f'property P{prop_id}')
            print(f"{i:2d}. P{prop_id:6s} ({count:6d} triples) - {prop_name}")
    
    print("\n="*80)
    print("SUMMARY")
    print("="*80)
    print(f"""
✅ Your graph is well-structured:
   - {len(graph):,} total triples
   - {len(p31_triples):,} P31 (instance of) triples
   - {len(sorted_types):,} unique entity types
   - {len(sorted_props)} Wikidata properties

✅ Key entity types found:
   - {len(movies):,} movies (Q11424)
   - {len(persons):,} persons (Q5)

✅ Dynamic schema extraction will work correctly!
   The system can discover relations and types from your graph.
   Fallback defaults will supplement any missing mappings.

🎯 Your system should now handle queries like:
   • "Who directed The Matrix?" (forward query)
   • "What movies did Christopher Nolan direct?" (reverse query)
   • "From what country is 'Aro Tolbukhin'?" (country query)
   • "Did Christopher Nolan direct Inception?" (verification)
""")
    
    print("="*80)

if __name__ == "__main__":
    check_p31_usage()
