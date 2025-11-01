#!/usr/bin/env python3
"""
Test script for hybrid query processing (factual + embedding).
Tests queries using both approaches (no answer validation as correct answers are unknown).
"""

import sys
import os
import io
import contextlib
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Suppress verbose logging
import logging
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('src.main.sparql_handler').setLevel(logging.ERROR)


def test_hybrid_queries():
    """Test hybrid approach queries (output verification only, no answer validation)."""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(logs_dir, f'test_hybrid_{timestamp}.log')
    
    # Open log file
    with open(log_file, 'w', encoding='utf-8') as log:
        def log_print(msg):
            """Print to both console and log file."""
            print(msg)
            log.write(msg + '\n')
            log.flush()
        
        log_print("\n" + "="*80)
        log_print("TEST: Hybrid Approach Queries (Factual + Embedding)")
        log_print("="*80 + "\n")
        log_print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_print(f"Log file: {log_file}\n")
        
        try:
            from src.main.orchestrator import Orchestrator
            orchestrator = Orchestrator(use_workflow=True)
            
            if not hasattr(orchestrator, 'embedding_processor') or orchestrator.embedding_processor is None:
                log_print("⚠️  Embedding processor not available - skipping test")
                return
            
            # HYBRID TEST CASES
            test_cases = [
                {
                    'query': "Please answer this question: Who is the director of 'Good Will Hunting'?",
                    'description': 'Hybrid director query (P57)'
                },
                {
                    'query': "Which movie has the highest user rating?",
                    'description': 'Hybrid query - highest rated movie (rating property)'
                },
                {
                    'query': "Who directed the movie 'The Bridge on the River Kwai'?",
                    'description': 'Hybrid director query (P57)'
                },
                {
                    'query': "What genre is the movie 'Shoplifters'?",
                    'description': 'Hybrid genre query (P136)'
                },
                {
                    'query': "Who is the producer of the movie 'French Kiss'?",
                    'description': 'Hybrid producer query (P162)'
                },
                {
                    'query': "Which movie, originally from the country 'South Korea', received the award 'Academy Award for Best Picture'?",
                    'description': 'Hybrid complex query (P495 + P166)'
                },
                {
                    'query': "Who are the main cast members of 'Fight Club'?",
                    'description': 'Hybrid cast query (P161)'
                },
                {
                    'query': "What sequel followed 'The Terminator'?",
                    'description': 'Hybrid sequel query (P156)'
                },
                {
                    'query': "Which actor played the lead role in 'Forrest Gump'?",
                    'description': 'Hybrid cast member query (P161)'
                },
                {
                    'query': "Who composed the music for 'Star Wars'?",
                    'description': 'Hybrid composer query (P86)'
                },
                {
                    'query': "What language is 'Amélie' in?",
                    'description': 'Hybrid language query (P364)'
                },
                {
                    'query': "When was 'Pulp Fiction' released?",
                    'description': 'Hybrid publication date query (P577)'
                },
                {
                    'query': "Who wrote the screenplay for 'Eternal Sunshine of the Spotless Mind'?",
                    'description': 'Hybrid screenwriter query (P58)'
                },
                {
                    'query': "What country is 'Cinema Paradiso' from?",
                    'description': 'Hybrid country query (P495)'
                },
                {
                    'query': "Who did the cinematography for 'Birdman'?",
                    'description': 'Hybrid cinematographer query (P344)'
                },
                {
                    'query': "Who are the producers of 'Schindler's List'?",
                    'description': 'Hybrid multiple producers query (P162)'
                },
                {
                    'query': "Who is the production designer of 'The Grand Budapest Hotel'?",
                    'description': 'Hybrid production designer query (P2554)'
                },
                {
                    'query': "What franchise does 'Iron Man' belong to?",
                    'description': 'Hybrid media franchise query (P8345)'
                },
                {
                    'query': "Who narrated 'The Big Lebowski'?",
                    'description': 'Hybrid narrator query (P2438)'
                },
                {
                    'query': "Who did the costume design for 'The Great Gatsby'?",
                    'description': 'Hybrid costume designer query (P2515)'
                }
            ]
            
            results = {
                'total': len(test_cases),
                'classification_correct': 0,
                'factual_generated': 0,
                'embedding_generated': 0,
                'both_generated': 0
            }
            
            for i, test_case in enumerate(test_cases, 1):
                query = test_case['query']
                description = test_case['description']
                
                log_print(f"\n{'='*80}")
                log_print(f"TEST [{i}/{len(test_cases)}]: {description}")
                log_print(f"Query: '{query}'")
                
                # Classification
                classification = orchestrator.classify_query(query)
                classification_msg = f"  Classification: {classification.question_type.value} ({classification.confidence:.1%})"
                
                if classification.question_type.value == 'hybrid':
                    results['classification_correct'] += 1
                    log_print(classification_msg + " ✅")
                else:
                    log_print(classification_msg + f" ❌ Expected: hybrid")
                    continue
                
                # Query Processing
                try:
                    f = io.StringIO()
                    with contextlib.redirect_stdout(f):
                        clean_query = orchestrator._clean_query_for_processing(query)
                        
                        # Process both approaches
                        factual_result = orchestrator.embedding_processor.process_hybrid_factual_query(clean_query)
                        embedding_result = orchestrator.embedding_processor.process_embedding_query(clean_query)
                    
                    # Check factual response
                    factual_ok = factual_result and not factual_result.startswith('❌')
                    if factual_ok:
                        results['factual_generated'] += 1
                        log_print(f"  Factual:   ✅ Generated")
                        # Show factual answer
                        factual_preview = factual_result[:100] + "..." if len(factual_result) > 100 else factual_result
                        log_print(f"    Answer: {factual_preview}")
                    else:
                        log_print(f"  Factual:   ❌ Error")
                    
                    # Check embedding response
                    embedding_ok = embedding_result and not embedding_result.startswith('❌')
                    if embedding_ok:
                        results['embedding_generated'] += 1
                        log_print(f"  Embedding: ✅ Generated")
                        # Show embedding answer
                        embedding_preview = embedding_result[:100] + "..." if len(embedding_result) > 100 else embedding_result
                        log_print(f"    Answer: {embedding_preview}")
                    else:
                        log_print(f"  Embedding: ❌ Error")
                    
                    # Track if both succeeded
                    if factual_ok and embedding_ok:
                        results['both_generated'] += 1
                    
                except Exception as e:
                    log_print(f"  ❌ ERROR: {str(e)}")
                    import traceback
                    log_print(traceback.format_exc())
            
            # Summary
            log_print("\n" + "="*80)
            log_print("TEST SUMMARY - HYBRID QUERIES")
            log_print("="*80)
            
            log_print(f"\n📊 Results:")
            log_print(f"  Classification:        {results['classification_correct']}/{results['total']} ({results['classification_correct']/results['total']:.1%})")
            log_print(f"  Factual Generated:     {results['factual_generated']}/{results['total']} ({results['factual_generated']/results['total']:.1%})")
            log_print(f"  Embedding Generated:   {results['embedding_generated']}/{results['total']} ({results['embedding_generated']/results['total']:.1%})")
            log_print(f"  Both Generated:        {results['both_generated']}/{results['total']} ({results['both_generated']/results['total']:.1%})")
            
            success_rate = results['both_generated'] / results['total']
            
            if results['both_generated'] == results['total']:
                log_print("\n🎉 ALL TESTS PASSED!\n")
            elif success_rate >= 0.7:
                log_print(f"\n✅ Most tests passed ({success_rate:.1%} success rate)\n")
            else:
                log_print(f"\n⚠️  {results['total'] - results['both_generated']} test(s) failed\n")
            
            log_print("ℹ️  Note: Hybrid queries do not have ground truth answers.")
            log_print("   Manual comparison of factual vs embedding results is recommended.\n")
            
        finally:
            log_print(f"\n✅ Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            log_print(f"📄 Results saved to: {log_file}\n")


if __name__ == "__main__":
    test_hybrid_queries()
