"""
Test entity extraction for person names (directors, actors).
Diagnoses why reverse lookup queries fail.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from rdflib import Graph
from src.main.entity_extractor import EntityExtractor
from src.config import GRAPH_FILE_PATH


def test_person_extraction():
    """Test if entity extractor can find person entities."""
    
    print("\n" + "="*80)
    print("ENTITY EXTRACTION TEST - PERSON NAMES")
    print("="*80)
    print("Testing: Can entity extractor find directors and actors by name?")
    print("="*80 + "\n")
    
    # Load graph
    print("📂 Loading knowledge graph...")
    graph = Graph()
    try:
        graph.parse(GRAPH_FILE_PATH, format='nt')
        print(f"✅ Loaded {len(graph)} triples\n")
    except Exception as e:
        print(f"❌ Failed to load graph: {e}")
        return 1
    
    # Initialize entity extractor
    print("🔧 Initializing entity extractor...")
    extractor = EntityExtractor(graph)
    print()
    
    # Test cases for person extraction
    test_cases = [
        {
            "name": "Christopher Nolan",
            "query": "What films did Christopher Nolan direct?",
            "entity_type": "http://www.wikidata.org/entity/Q5",  # Person
            "expected_in_graph": True
        },
        {
            "name": "Tom Hanks", 
            "query": "What movies did Tom Hanks star in?",
            "entity_type": "http://www.wikidata.org/entity/Q5",  # Person
            "expected_in_graph": True
        },
        {
            "name": "Francis Ford Coppola",
            "query": "Who is Francis Ford Coppola?",
            "entity_type": "http://www.wikidata.org/entity/Q5",  # Person
            "expected_in_graph": True
        },
        {
            "name": "Steven Spielberg",
            "query": "What did Steven Spielberg direct?",
            "entity_type": "http://www.wikidata.org/entity/Q5",  # Person
            "expected_in_graph": True
        }
    ]
    
    print("="*80)
    print(f"RUNNING {len(test_cases)} PERSON EXTRACTION TESTS")
    print("="*80 + "\n")
    
    results = {'passed': 0, 'failed': 0}
    
    for i, test_case in enumerate(test_cases, 1):
        name = test_case['name']
        query = test_case['query']
        entity_type = test_case['entity_type']
        expected = test_case['expected_in_graph']
        
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_cases)}")
        print(f"{'='*80}\n")
        print(f"Person: {name}")
        print(f"Query: {query}")
        print(f"Expected in graph: {expected}")
        print()
        
        # Test 1: Check if person name is in cache (case-insensitive)
        print("📋 Test 1: Check entity cache")
        name_lower = name.lower()
        
        if name_lower in extractor.entity_cache:
            uris = extractor.entity_cache[name_lower]
            print(f"   ✅ Found '{name}' in cache (lowercase key)")
            print(f"   URIs: {len(uris)}")
            for uri in uris[:3]:
                label = extractor.get_entity_label(uri)
                print(f"      - {label} ({uri})")
        else:
            print(f"   ❌ '{name}' NOT in cache (key: '{name_lower}')")
            
            # Search for partial matches
            print(f"\n   🔍 Searching for partial matches...")
            partial_matches = [
                key for key in extractor.entity_cache.keys()
                if name_lower in key or any(word in key for word in name_lower.split())
            ]
            
            if partial_matches:
                print(f"   Found {len(partial_matches)} partial matches:")
                for match in partial_matches[:10]:
                    print(f"      - '{match}'")
            else:
                print(f"   No partial matches found")
        
        print()
        
        # Test 2: Try extracting from query (no type filter)
        print("📋 Test 2: Extract from query (no type filter)")
        all_entities = extractor.extract_entities(query, entity_type=None, threshold=75)
        
        if all_entities:
            print(f"   ✅ Found {len(all_entities)} entities:")
            for uri, text, score in all_entities[:5]:
                label = extractor.get_entity_label(uri)
                has_person_type = extractor._has_type(uri, entity_type)
                type_marker = "[Person]" if has_person_type else "[Other]"
                print(f"      {type_marker} '{label}' (match: {text}, score: {score})")
        else:
            print(f"   ❌ No entities extracted")
        
        print()
        
        # Test 3: Try extracting with person type filter
        print("📋 Test 3: Extract from query (Person type filter)")
        person_entities = extractor.extract_entities(query, entity_type=entity_type, threshold=75)
        
        if person_entities:
            print(f"   ✅ Found {len(person_entities)} person entities:")
            for uri, text, score in person_entities[:5]:
                label = extractor.get_entity_label(uri)
                print(f"      - '{label}' (match: {text}, score: {score})")
                
            # Test passed
            results['passed'] += 1
            print(f"\n✅ TEST PASSED - Person '{name}' can be extracted")
        else:
            print(f"   ❌ No person entities found")
            
            # Test failed
            results['failed'] += 1
            print(f"\n❌ TEST FAILED - Cannot extract person '{name}'")
        
        print()
        
        # Test 4: Check what pattern extraction finds
        print("📋 Test 4: Debug pattern extraction")
        query_lower = query.lower()
        
        # Simulate stop word removal
        stop_words = [
            'who', 'what', 'when', 'where', 'which', 'how', 'is', 'was', 'are', 'were',
            'show', 'find', 'list', 'get', 'tell', 'give', 'directed', 'director',
            'screenwriter', 'actor', 'released', 'the', 'of', 'in', 'for', 'about', 'did',
            'films', 'movies', 'star'
        ]
        
        import re
        query_cleaned = query_lower
        for word in stop_words:
            query_cleaned = re.sub(rf'\b{word}\b', ' ', query_cleaned, flags=re.IGNORECASE)
        query_cleaned = re.sub(r'\s+', ' ', query_cleaned).strip()
        
        print(f"   Original query: '{query}'")
        print(f"   After stop word removal: '{query_cleaned}'")
        
        # Check if cleaned query matches any cache keys
        matching_keys = [
            key for key in extractor.entity_cache.keys()
            if key in query_cleaned or query_cleaned in key
        ]
        
        if matching_keys:
            print(f"   ✅ Found {len(matching_keys)} potential matches in cache:")
            for key in matching_keys[:5]:
                print(f"      - '{key}'")
        else:
            print(f"   ❌ No matches found in cache for '{query_cleaned}'")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Passed: {results['passed']}/{len(test_cases)}")
    print(f"Failed: {results['failed']}/{len(test_cases)}")
    print("="*80 + "\n")
    
    if results['failed'] > 0:
        print("⚠️  ISSUES FOUND:")
        print("   1. Person names may not be in the knowledge graph")
        print("   2. Person names may use different spelling/formatting")
        print("   3. Entity cache may not include person entities")
        print("   4. Stop word removal may interfere with name extraction")
        print()
        print("💡 RECOMMENDATIONS:")
        print("   - Check if persons exist in graph with SPARQL query")
        print("   - Verify person names have rdfs:label properties")
        print("   - Consider adding person names without stop word removal")
        print("   - Add special handling for multi-word person names")
    else:
        print("🎉 All person extraction tests passed!")
    
    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    exit_code = test_person_extraction()
    sys.exit(exit_code)