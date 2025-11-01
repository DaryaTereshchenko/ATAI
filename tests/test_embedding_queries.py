#!/usr/bin/env python3
"""
Test script for embedding-based query processing.
Tests queries using the embedding approach (no answer validation as correct answers are unknown).
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


def test_embedding_queries():
    """Test embedding approach queries (output verification only, no answer validation)."""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(logs_dir, f'test_embedding_{timestamp}.log')
    
    # Open log file
    with open(log_file, 'w', encoding='utf-8') as log:
        def log_print(msg):
            """Print to both console and log file."""
            print(msg)
            log.write(msg + '\n')
            log.flush()
        
        log_print("\n" + "="*80)
        log_print("TEST: Embedding Approach Queries (Output Format Verification)")
        log_print("="*80 + "\n")
        log_print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_print(f"Log file: {log_file}\n")
        
        try:
            from src.main.orchestrator import Orchestrator
            orchestrator = Orchestrator(use_workflow=True)
            
            if not hasattr(orchestrator, 'embedding_processor') or orchestrator.embedding_processor is None:
                log_print("⚠️  Embedding processor not available - skipping test")
                return
            
            # EMBEDDING TEST CASES
            test_cases = [
                {
                    'query': "Please answer this question with an embedding approach: Who is the director of 'Apocalypse Now'?",
                    'description': 'Embedding director query (P57)'
                },
                {
                    'query': "Please answer this question with an embedding approach: Who is the screenwriter of '12 Monkeys'?",
                    'description': 'Embedding screenwriter query (P58)'
                },
                {
                    'query': "Please answer this question with an embedding approach: What is the genre of 'Shoplifters'?",
                    'description': 'Embedding genre query (P136)'
                },
                {
                    'query': "Please answer this question with an embedding approach: Who produced 'Inception'?",
                    'description': 'Embedding producer query (P162)'
                },
                {
                    'query': "Please answer this question with an embedding approach: What studio distributed 'Star Wars'?",
                    'description': 'Embedding distributor query (P750)'
                },
                {
                    'query': "Please answer this question with an embedding approach: Who composed the soundtrack for 'Interstellar'?",
                    'description': 'Embedding composer query (P86)'
                },
                {
                    'query': "Please answer this question with an embedding approach: What language is spoken in 'Life Is Beautiful'?",
                    'description': 'Embedding language query (P364)'
                },
                {
                    'query': "Please answer this question with an embedding approach: Who did the cinematography for 'The Revenant'?",
                    'description': 'Embedding cinematographer query (P344)'
                },
                {
                    'query': "Please answer this question with an embedding approach: What country produced 'Crouching Tiger, Hidden Dragon'?",
                    'description': 'Embedding country query (P495)'
                },
                {
                    'query': "Please answer this question with an embedding approach: What studio made 'Toy Story'?",
                    'description': 'Embedding production company query (P272)'
                },
                {
                    'query': "Please answer this question with an embedding approach: Who edited 'Mad Max: Fury Road'?",
                    'description': 'Embedding film editor query (P1040)'
                },
                {
                    'query': "Please answer this question with an embedding approach: What genre is '2001: A Space Odyssey'?",
                    'description': 'Embedding genre query (P136)'
                },
            ]
            
            results = {
                'total': len(test_cases),
                'classification_correct': 0,
                'response_generated': 0,
                'response_has_type': 0
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
                
                if classification.question_type.value == 'embeddings':
                    results['classification_correct'] += 1
                    log_print(classification_msg + " ✅")
                else:
                    log_print(classification_msg + f" ❌ Expected: embeddings")
                    continue
                
                # Query Processing
                try:
                    f = io.StringIO()
                    with contextlib.redirect_stdout(f):
                        clean_query = orchestrator._clean_query_for_processing(query)
                        response = orchestrator.embedding_processor.process_embedding_query(clean_query)
                    
                    # Check response format
                    if response and not response.startswith('❌'):
                        results['response_generated'] += 1
                        
                        # Check if response has entity type information
                        if '(type:' in response:
                            results['response_has_type'] += 1
                            log_print(f"  Response: ✅ Generated with type information")
                        else:
                            log_print(f"  Response: ⚠️  Generated but missing type information")
                        
                        # Show the answer (for manual verification)
                        answer_part = response.split('suggested by embeddings is:')[-1].strip()
                        log_print(f"  Suggested Answer: {answer_part}")
                    else:
                        log_print(f"  Response: ❌ Error in generation")
                        log_print(f"  Error: {response}")
                    
                except Exception as e:
                    log_print(f"  ❌ ERROR: {str(e)}")
                    import traceback
                    log_print(traceback.format_exc())
            
            # Summary
            log_print("\n" + "="*80)
            log_print("TEST SUMMARY - EMBEDDING QUERIES")
            log_print("="*80)
            
            log_print(f"\n📊 Results:")
            log_print(f"  Classification:        {results['classification_correct']}/{results['total']} ({results['classification_correct']/results['total']:.1%})")
            log_print(f"  Response Generation:   {results['response_generated']}/{results['total']} ({results['response_generated']/results['total']:.1%})")
            log_print(f"  Type Information:      {results['response_has_type']}/{results['total']} ({results['response_has_type']/results['total']:.1%})")
            
            success_rate = results['response_generated'] / results['total']
            
            if results['response_generated'] == results['total']:
                log_print("\n🎉 ALL TESTS PASSED!\n")
            elif success_rate >= 0.7:
                log_print(f"\n✅ Most tests passed ({success_rate:.1%} success rate)\n")
            else:
                log_print(f"\n⚠️  {results['total'] - results['response_generated']} test(s) failed\n")
            
            log_print("ℹ️  Note: Embedding queries do not have ground truth answers.")
            log_print("   Manual verification of suggested answers is recommended.\n")
            
        finally:
            log_print(f"\n✅ Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            log_print(f"📄 Results saved to: {log_file}\n")


if __name__ == "__main__":
    test_embedding_queries()
