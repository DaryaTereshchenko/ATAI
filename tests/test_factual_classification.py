"""
Comprehensive end-to-end pipeline test for transformer classifier.
Tests the complete flow: Classification → Pattern Analysis → Entity Extraction → 
                         LLM/Template SPARQL Generation → Execution → Response

Focus on:
1. FACTUAL queries: Full pipeline with LLM-first, template-fallback strategy
2. OUT-OF-SCOPE queries: Proper rejection at classification stage
3. Pattern-specific few-shot prompting for LLM

Uses real Wikidata entities that should exist in the knowledge graph.
"""

import sys
import os
import json

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.config import TRANSFORMER_MODEL_PATH

# Suppress verbose logging
import logging
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('src.main.sparql_handler').setLevel(logging.ERROR)


def diagnose_entity_extraction(query: str, orchestrator):
    """
    Diagnostic function to test entity extraction with pattern-aware architecture.
    
    Uses QueryAnalyzer to understand query pattern, then extracts appropriate entities.
    """
    embedding_proc = orchestrator.embedding_processor
    if not embedding_proc:
        return None
    
    # Pattern Analysis
    pattern = embedding_proc.query_analyzer.analyze(query)
    
    if not pattern:
        return None
    
    # Entity Extraction
    entities_found = {}
    
    if pattern.pattern_type == 'forward':
        entity_type = "http://www.wikidata.org/entity/Q11424"  # Q11424 = film
        entities = embedding_proc.entity_extractor.extract_entities(
            query,
            entity_type=entity_type,
            threshold=70
        )
        
        if entities:
            entities_found['movie'] = (entities[0], entity_type)
    
    elif pattern.pattern_type == 'reverse':
        entity_type = "http://www.wikidata.org/entity/Q5"  # Q5 = human
        entities = embedding_proc.entity_extractor.extract_entities(
            query,
            entity_type=entity_type,
            threshold=70
        )
        
        if entities:
            entities_found['person'] = (entities[0], entity_type)
    
    elif pattern.pattern_type == 'verification':
        movie_type = "http://www.wikidata.org/entity/Q11424"
        person_type = "http://www.wikidata.org/entity/Q5"
        
        movie_entities = embedding_proc.entity_extractor.extract_entities(
            query,
            entity_type=movie_type,
            threshold=70
        )
        
        person_entities = embedding_proc.entity_extractor.extract_entities(
            query,
            entity_type=person_type,
            threshold=70
        )
        
        if movie_entities:
            entities_found['movie'] = (movie_entities[0], movie_type)
        
        if person_entities:
            entities_found['person'] = (person_entities[0], person_type)  # Fixed: was personEntities
    
    return {
        'pattern': pattern,
        'entities': entities_found
    }


def test_sparql_generation(pattern, entities_found, embedding_proc):
    """
    Test SPARQL generation with LLM-first, template-fallback strategy.
    
    Args:
        pattern: QueryPattern from analyzer
        entities_found: Dict with extracted entities (now includes entity_type)
        embedding_proc: Embedding processor instance
    """
    if not entities_found:
        return None
    
    # Determine subject and object labels
    subject_label = None
    object_label = None
    
    if pattern.pattern_type == 'forward':
        if 'movie' in entities_found:
            (uri, text, score), entity_type = entities_found['movie']
            subject_label = embedding_proc.entity_extractor.get_entity_label(uri)
    
    elif pattern.pattern_type == 'reverse':
        if 'person' in entities_found:
            (uri, text, score), entity_type = entities_found['person']
            subject_label = embedding_proc.entity_extractor.get_entity_label(uri)
    
    elif pattern.pattern_type == 'verification':
        if 'person' in entities_found:
            (uri, text, score), entity_type = entities_found['person']
            subject_label = embedding_proc.entity_extractor.get_entity_label(uri)
        if 'movie' in entities_found:
            (uri, text, score), entity_type = entities_found['movie']
            object_label = embedding_proc.entity_extractor.get_entity_label(uri)
    
    if not subject_label:
        return None
    
    # Test SPARQL generation with fallback
    try:
        sparql_result = embedding_proc._generate_sparql_with_fallback(
            pattern=pattern,
            subject_label=subject_label,
            object_label=object_label
        )
        
        return sparql_result
    
    except Exception as e:
        return None


def test_transformer_pipeline():
    """Test complete pipeline with transformer classifier for factual and out-of-scope queries."""
    
    print("\n" + "="*80)
    print("TRANSFORMER CLASSIFIER - END-TO-END PIPELINE TEST")
    print("="*80 + "\n")
    
    from src.main.orchestrator import Orchestrator
    from src.config import USE_EMBEDDINGS
    
    # Initialize orchestrator (suppress initialization output)
    import io
    import contextlib
    
    f = io.StringIO()
    try:
        with contextlib.redirect_stdout(f):
            orchestrator = Orchestrator(
                use_workflow=True,
                use_transformer_classifier=True,
                transformer_model_path=TRANSFORMER_MODEL_PATH
            )
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # ==================== TEST CASES ====================
    test_cases = [
        {
            'query': 'What are the genres of the movie Even Cowgirls Get the Blues?',
            'expected_type': 'factual',
            'expected_pattern': 'forward_genre',
            'description': 'Multi-genre query',
            'should_process': True
        },
        {
            'query': 'Who produced the movie Tesis?',
            'expected_type': 'factual',
            'expected_pattern': 'forward_producer',
            'description': 'Producer query',
            'should_process': True
        },
        {
            'query': 'Which movie has the highest user rating?',
            'expected_type': 'factual',
            'expected_pattern': None,  # Complex query, pattern may vary
            'description': 'Superlative query (highest rating)',
            'should_process': True
        },
        {
            'query': "Who directed the movie 'The Bridge on the River Kwai'?",
            'expected_type': 'factual',
            'expected_pattern': 'forward_director',
            'description': 'Director query with title case',
            'should_process': True
        },
        {
            'query': "What genre is the movie 'Shoplifters'?",
            'expected_type': 'factual',
            'expected_pattern': 'forward_genre',
            'description': 'Genre query (single)',
            'should_process': True
        },
        {
            'query': "Who is the producer of the movie 'French Kiss'?",
            'expected_type': 'factual',
            'expected_pattern': 'forward_producer',
            'description': 'Producer query variant',
            'should_process': True
        },
        {
            'query': "Which movie, originally from the country 'South Korea', received the award 'Academy Award for Best Picture'?",
            'expected_type': 'factual',
            'expected_pattern': None,  # Complex multi-constraint query
            'description': 'Complex multi-constraint query (country + award)',
            'should_process': True
        },
        {
            'query': 'What is the weather today?',
            'expected_type': 'out_of_scope',
            'expected_pattern': None,
            'description': 'Out-of-scope query',
            'should_process': False
        }
    ]
    
    # Run tests
    results = {
        'total': len(test_cases),
        'classification_correct': 0,
        'pattern_correct': 0,
        'entity_extraction_success': 0,
        'sparql_generation_success': 0,
        'processing_correct': 0,
        'factual_success': 0,
        'oos_rejected': 0,
        'llm_used': 0,
        'template_used': 0
    }
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case['query']
        expected_type = test_case['expected_type']
        expected_pattern = test_case.get('expected_pattern')
        description = test_case['description']
        should_process = test_case['should_process']
        
        print(f"[{i}/{len(test_cases)}] {description}")
        print(f"Query: '{query}'")
        
        # Classification (suppress verbose output)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            classification = orchestrator.classify_query(query)
        
        # Check classification
        if classification.question_type.value == expected_type:
            results['classification_correct'] += 1
            print(f"✅ Classification: {classification.question_type.value}")
        else:
            print(f"❌ Classification: {classification.question_type.value} (expected: {expected_type})")
            print()
            continue
        
        # Processing (only for FACTUAL)
        if should_process and classification.question_type.value == 'factual':
            
            # Entity extraction diagnostics (suppress verbose output)
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                diagnostic_result = diagnose_entity_extraction(query, orchestrator)
            
            if diagnostic_result:
                pattern = diagnostic_result['pattern']
                entities_found = diagnostic_result['entities']
                
                # Check pattern (if expected pattern is specified)
                pattern_label = f"{pattern.pattern_type}_{pattern.relation}"
                if expected_pattern:
                    if pattern_label == expected_pattern:
                        results['pattern_correct'] += 1
                        print(f"✅ Pattern: {pattern_label}")
                    else:
                        print(f"⚠️  Pattern: {pattern_label} (expected: {expected_pattern})")
                else:
                    # No expected pattern, just show detected pattern
                    print(f"ℹ️  Pattern: {pattern_label}")
                    results['pattern_correct'] += 1  # Count as correct if no expectation
                
                # Check entity extraction - NOW WITH TYPE INFO
                if entities_found:
                    results['entity_extraction_success'] += 1
                    print(f"✅ Entities:")
                    for key, ((uri, text, score), entity_type) in entities_found.items():
                        label = orchestrator.embedding_processor.entity_extractor.get_entity_label(uri)
                        # Extract QID from entity type URI
                        type_qid = entity_type.split('/')[-1] if '/' in entity_type else entity_type
                        print(f"   • {key.capitalize()}: {label}")
                        print(f"     Type: {type_qid}, Score: {score}%")
                else:
                    print(f"❌ Entity extraction failed")
                
                # SPARQL generation (suppress verbose output)
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    sparql_result = test_sparql_generation(
                        pattern, 
                        entities_found, 
                        orchestrator.embedding_processor
                    )
                
                if sparql_result:
                    results['sparql_generation_success'] += 1
                    print(f"✅ SPARQL: {sparql_result['method'].upper()}")
                    
                    # SHOW FULL SPARQL QUERY
                    print(f"\n📝 Generated SPARQL Query:")
                    print("-" * 80)
                    print(sparql_result['query'])
                    print("-" * 80)
                    
                    if sparql_result['method'] == 'llm':
                        results['llm_used'] += 1
                    else:
                        results['template_used'] += 1
                else:
                    print(f"❌ SPARQL generation failed")
            
            # Full pipeline test - Execute and show results
            print(f"\n📊 Query Results:")
            print("-" * 80)
            try:
                # Suppress processing output, only show final results
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    response = orchestrator.process_query(query)
                
                # Extract and show only the results data
                if "✅" in response:
                    # Parse and show clean results
                    lines = response.split('\n')
                    result_lines = []
                    
                    for line in lines:
                        # Skip headers and formatting
                        if '📊' in line or '##' in line or '===' in line or '---' in line:
                            continue
                        # Skip status messages at start
                        if line.startswith('✅') or line.startswith('❌') or line.startswith('⚠️'):
                            continue
                        # Skip empty lines
                        if not line.strip():
                            continue
                        # Show actual data (lines with content)
                        stripped = line.strip()
                        if stripped and (stripped.startswith('•') or stripped.startswith('-') or ':' in stripped):
                            result_lines.append(stripped)
                    
                    if result_lines:
                        for line in result_lines[:10]:  # Show first 10 results
                            print(line)
                    else:
                        # Fallback: show the whole response
                        print(response.strip())
                    
                    results['factual_success'] += 1
                    results['processing_correct'] += 1
                else:
                    print("Query execution failed")
                
            except Exception as e:
                print(f"Error: {e}")
            
            print("-" * 80)
        
        elif not should_process and classification.question_type.value == 'out_of_scope':
            results['oos_rejected'] += 1
            results['processing_correct'] += 1
            print(f"✅ Correctly rejected")
        
        print()
    
    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    factual_tests = sum(1 for tc in test_cases if tc['should_process'])
    
    print(f"\n📊 Results:")
    print(f"  Classification:     {results['classification_correct']}/{results['total']}")
    
    if factual_tests > 0:
        print(f"  Pattern Analysis:   {results['pattern_correct']}/{factual_tests}")
        print(f"  Entity Extraction:  {results['entity_extraction_success']}/{factual_tests}")
        print(f"  SPARQL Generation:  {results['sparql_generation_success']}/{factual_tests} [LLM: {results['llm_used']}, Template: {results['template_used']}]")
        print(f"  Full Pipeline:      {results['factual_success']}/{factual_tests}")
    
    if results['oos_rejected'] > 0:
        print(f"  OOS Rejection:      {results['oos_rejected']}/{results['total'] - factual_tests}")
    
    print(f"\n  Overall Success:    {results['processing_correct']}/{results['total']}")
    
    if results['classification_correct'] == results['total'] and results['processing_correct'] == results['total']:
        print("\n🎉 ALL TESTS PASSED!\n")
    else:
        print(f"\n⚠️  {results['total'] - results['processing_correct']} test(s) failed\n")


if __name__ == "__main__":
    test_transformer_pipeline()