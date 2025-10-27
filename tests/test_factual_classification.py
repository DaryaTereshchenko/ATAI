"""
Comprehensive end-to-end pipeline test for transformer classifier.
Tests the complete flow: Classification → Pattern Analysis → Entity Extraction → 
                         LLM/Template SPARQL Generation → Execution → Response

Focus on:
1. FACTUAL queries: Full pipeline with LLM-first, template-fallback strategy
2. EMBEDDING queries: Pure embedding-based approach with entity type reporting
3. HYBRID queries: Both factual and embedding answers
4. Answer validation against expected results

Uses real Wikidata entities from the provided test cases.
"""

import sys
import os
import json
import re
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.config import TRANSFORMER_MODEL_PATH

# Suppress verbose logging
import logging
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('src.main.sparql_handler').setLevel(logging.ERROR)


class TeeOutput:
    """Utility class to write output to both console and file."""
    
    def __init__(self, file_path: str):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()


def extract_answer_from_response(response: str) -> str:
    """
    Extract the actual answer from a formatted response.
    
    Args:
        response: Full formatted response string
        
    Returns:
        Extracted answer text
    """
    # Remove markdown formatting
    response = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)
    
    # For factual responses: extract text after "was" or "is"
    # Example: "✅ 'Movie' was released in **1974**."
    match = re.search(r'(?:was|is|are)\s+(?:released in\s+)?(.+?)(?:\.|$)', response, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        # Clean up common prefixes
        answer = re.sub(r'^(?:directed by|starring|written by|produced by)\s+', '', answer, flags=re.IGNORECASE)
        return answer
    
    # For embedding responses: extract "answer suggested by embeddings is: X (type: Y)"
    match = re.search(r'answer suggested by embeddings is:\s*([^(]+)\s*\(type:\s*([^)]+)\)', response, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        entity_type = match.group(2).strip()
        return f"{answer} (type: {entity_type})"
    
    # For list responses: extract items
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    for line in lines:
        # Look for bullet points or answer lines
        if line.startswith('•') or line.startswith('-'):
            return line.lstrip('•-').strip()
    
    # Fallback: return cleaned response
    return response.strip()


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    # Remove quotes
    answer = answer.replace('"', '').replace("'", '')
    # Lowercase
    answer = answer.lower()
    return answer


def compare_answers(actual: str, expected: str) -> tuple:
    """
    Compare actual answer with expected answer.
    
    Returns:
        (is_match: bool, similarity_score: float)
    """
    actual_norm = normalize_answer(actual)
    expected_norm = normalize_answer(expected)
    
    # Exact match
    if actual_norm == expected_norm:
        return True, 1.0
    
    # Check if expected is contained in actual (for multi-part answers)
    if expected_norm in actual_norm:
        return True, 0.9
    
    # Check word overlap for partial match
    actual_words = set(actual_norm.split())
    expected_words = set(expected_norm.split())
    
    if expected_words and actual_words:
        overlap = len(actual_words & expected_words)
        total = len(expected_words)
        similarity = overlap / total
        
        # Consider it a match if >70% word overlap
        return similarity > 0.7, similarity
    
    return False, 0.0


def test_transformer_pipeline():
    """Test complete pipeline with provided example queries."""
    
    # ✅ NEW: Setup file logging
    # Create logs directory if it doesn't exist
    logs_dir = Path(project_root) / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = logs_dir / f'test_pipeline_{timestamp}.log'
    
    # Setup tee output to write to both console and file
    tee = TeeOutput(str(log_file_path))
    original_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        print("\n" + "="*80)
        print("TRANSFORMER CLASSIFIER - FULL PIPELINE TEST")
        print("Testing Factual, Embedding, and Hybrid Approaches")
        print("="*80)
        print(f"\n📝 Log file: {log_file_path}")
        print(f"📅 Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        from src.main.orchestrator import Orchestrator
        
        # Initialize orchestrator (suppress initialization output)
        import io
        import contextlib
        
        print("🔧 Initializing Orchestrator...")
        f = io.StringIO()
        try:
            with contextlib.redirect_stdout(f):
                orchestrator = Orchestrator(use_workflow=True)
            print("✅ Orchestrator initialized\n")
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            return
        
        # ==================== TEST CASES ====================
        test_cases = [
            # FACTUAL APPROACH TESTS
            {
                'query': "Please answer this question with a factual approach: From what country is the movie 'Aro Tolbukhin. En la mente del asesino'?",
                'expected_type': 'factual',
                'expected_answer': 'Mexico',
                'description': 'Country of origin query'
            },
            {
                'query': "Please answer this question with a factual approach: Who is the screenwriter of 'Shortcut to Happiness'?",
                'expected_type': 'factual',
                'expected_answer': 'Pete Dexter',
                'description': 'Screenwriter query'
            },
            {
                'query': "Please answer this question with a factual approach: What country is 'The Bridge on the River Kwai' from?",
                'expected_type': 'factual',
                'expected_answer': 'United Kingdom',
                'description': 'Country of origin query (alternate phrasing)'
            },
            {
                'query': "Please answer this question with a factual approach: Who directed 'Fargo'?",
                'expected_type': 'factual',
                'expected_answer': 'Ethan Coen and Joel Coen',
                'description': 'Director query (multiple directors)'
            },
            {
                'query': "Please answer this question with a factual approach: What genre is the movie 'Bandit Queen'?",
                'expected_type': 'factual',
                'expected_answer': 'drama film and biographical film and crime film',
                'description': 'Genre query (multiple genres)'
            },
            {
                'query': "Please answer this question with a factual approach: When did the movie 'Miracles Still Happen' come out?",
                'expected_type': 'factual',
                'expected_answer': '1974-07-19',
                'description': 'Release date query'
            },
            
            # EMBEDDING APPROACH TESTS
            {
                'query': "Please answer this question with an embedding approach: Who is the director of 'Apocalypse Now'?",
                'expected_type': 'embeddings',
                'expected_answer': 'John Milius (type: Q5)',
                'description': 'Embedding director query'
            },
            {
                'query': "Please answer this question with an embedding approach: Who is the screenwriter of '12 Monkeys'?",
                'expected_type': 'embeddings',
                'expected_answer': 'Carol Florence (type: Q5)',
                'description': 'Embedding screenwriter query'
            },
            {
                'query': "Please answer this question with an embedding approach: What is the genre of 'Shoplifters'?",
                'expected_type': 'embeddings',
                'expected_answer': 'comedy film (type: Q201658)',
                'description': 'Embedding genre query'
            },
            
            # HYBRID APPROACH TEST (no explicit approach specified)
            {
                'query': "Please answer this question: Who is the director of 'Good Will Hunting'?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'Gus Van Sant',
                'expected_embedding_answer': 'Harmony Korine (type: Q5)',
                'description': 'Hybrid query (both approaches)'
            },
            {
                'query': "Which movie has the highest user rating?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'The Shawshank Redemption',
                'expected_embedding_answer': 'The Godfather (type: Q11424)',
                'description': 'Hybrid query - highest rated movie'
            },
            {
                'query': "Who directed the movie 'The Bridge on the River Kwai'?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'David Lean',
                'expected_embedding_answer': 'David Lean (type: Q5)',
                'description': 'Hybrid query - director of classic film'
            },
            {
                'query': "What genre is the movie 'Shoplifters'?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'drama film',
                'expected_embedding_answer': 'comedy film (type: Q201658)',
                'description': 'Hybrid query - genre identification'
            },
            {
                'query': "Who is the producer of the movie 'French Kiss'?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'Meg Ryan',
                'expected_embedding_answer': 'Tim Bevan (type: Q5)',
                'description': 'Hybrid query - producer query'
            },
            {
                'query': "Which movie, originally from the country 'South Korea', received the award 'Academy Award for Best Picture'?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'Parasite',
                'expected_embedding_answer': 'Parasite (type: Q11424)',
                'description': 'Hybrid query - complex multi-criteria query'
            }
        ]
        
        # Run tests
        results = {
            'total': len(test_cases),
            'classification_correct': 0,
            'answer_correct': 0,
            'answer_partial': 0,
            'factual_correct': 0,
            'embedding_correct': 0,
            'hybrid_correct': 0
        }
        
        for i, test_case in enumerate(test_cases, 1):
            query = test_case['query']
            expected_type = test_case['expected_type']
            description = test_case['description']
            
            print(f"\n{'='*80}")
            print(f"TEST CASE [{i}/{len(test_cases)}]: {description}")
            print(f"{'='*80}")
            print(f"📝 Query: '{query}'")
            print()
            
            # ==================== STAGE 1: CLASSIFICATION ====================
            print("🔍 STAGE 1: QUERY CLASSIFICATION")
            print("-" * 80)
            
            classification = orchestrator.classify_query(query)
            
            print(f"Classification Result:")
            print(f"  • Type:       {classification.question_type.value}")
            print(f"  • Confidence: {classification.confidence:.1%}")
            print(f"  • Expected:   {expected_type}")
            
            # Check classification
            if classification.question_type.value == expected_type:
                results['classification_correct'] += 1
                print(f"✅ Classification: CORRECT")
            else:
                print(f"❌ Classification: INCORRECT (got {classification.question_type.value}, expected {expected_type})")
                print()
                continue
            print()
            
            # ==================== STAGE 2: QUERY PROCESSING ====================
            print("⚙️  STAGE 2: QUERY PROCESSING")
            print("-" * 80)
            
            try:
                # Log which processor will be used
                if expected_type == 'factual':
                    print("📌 Pipeline: FACTUAL APPROACH")
                    print("  → Clean query → Extract entities → Generate SPARQL → Execute → Format response")
                    print()
                    
                    # Call process with detailed output
                    print("🔄 Step 2.1: Cleaning query...")
                    clean_query = orchestrator._clean_query_for_processing(query)
                    print(f"  Clean query: '{clean_query}'")
                    print()
                    
                    print("🔄 Step 2.2: Processing with embedding processor (factual mode)...")
                    response = orchestrator.embedding_processor.process_hybrid_factual_query(clean_query)
                    
                elif expected_type == 'embeddings':
                    print("📌 Pipeline: EMBEDDING APPROACH")
                    print("  → Clean query → Encode query → Search embedding space → Find nearest entities → Format response")
                    print()
                    
                    print("🔄 Step 2.1: Cleaning query...")
                    clean_query = orchestrator._clean_query_for_processing(query)
                    print(f"  Clean query: '{clean_query}'")
                    print()
                    
                    print("🔄 Step 2.2: Processing with embedding processor (embedding mode)...")
                    response = orchestrator.embedding_processor.process_embedding_query(clean_query)
                    
                elif expected_type == 'hybrid':
                    print("📌 Pipeline: HYBRID APPROACH")
                    print("  → Run FACTUAL pipeline")
                    print("  → Run EMBEDDING pipeline")
                    print("  → Combine results")
                    print()
                    
                    print("🔄 Step 2.1: Cleaning query...")
                    clean_query = orchestrator._clean_query_for_processing(query)
                    print(f"  Clean query: '{clean_query}'")
                    print()
                    
                    print("🔄 Step 2.2a: Running FACTUAL pipeline...")
                    factual_result = orchestrator.embedding_processor.process_hybrid_factual_query(clean_query)
                    print(f"  Factual result (preview): {factual_result[:100]}...")
                    print()
                    
                    print("🔄 Step 2.2b: Running EMBEDDING pipeline...")
                    embeddings_result = orchestrator.embedding_processor.process_embedding_query(clean_query)
                    print(f"  Embeddings result (preview): {embeddings_result[:100]}...")
                    print()
                    
                    print("🔄 Step 2.3: Combining results...")
                    response = f"**Factual Answer:**\n{factual_result}\n\n"
                    response += f"**Embeddings Answer:**\n{embeddings_result}"
                
                print("✅ Query processing completed")
                print()
                
                # ==================== STAGE 3: RESPONSE ANALYSIS ====================
                print("📊 STAGE 3: RESPONSE ANALYSIS")
                print("-" * 80)
                print(f"Full Response:")
                print("-" * 40)
                print(response)
                print("-" * 40)
                print()
                
                # ==================== STAGE 4: ANSWER EXTRACTION ====================
                print("🎯 STAGE 4: ANSWER EXTRACTION")
                print("-" * 80)
                
                if expected_type == 'factual':
                    expected_answer = test_case['expected_answer']
                    actual_answer = extract_answer_from_response(response)
                    
                    print(f"Extraction Method: Factual answer extraction")
                    print(f"Expected Answer:   '{expected_answer}'")
                    print(f"Extracted Answer:  '{actual_answer}'")
                    print()
                    
                    print("🔍 STAGE 5: ANSWER VALIDATION")
                    print("-" * 80)
                    is_match, similarity = compare_answers(actual_answer, expected_answer)
                    
                    print(f"Comparison Details:")
                    print(f"  • Normalized Expected: '{normalize_answer(expected_answer)}'")
                    print(f"  • Normalized Actual:   '{normalize_answer(actual_answer)}'")
                    print(f"  • Similarity Score:    {similarity:.1%}")
                    print(f"  • Match Threshold:     70%")
                    
                    if is_match:
                        results['answer_correct'] += 1
                        results['factual_correct'] += 1
                        print(f"✅ Answer VALIDATION: PASSED (similarity: {similarity:.1%})")
                    elif similarity > 0.5:
                        results['answer_partial'] += 1
                        print(f"⚠️  Answer VALIDATION: PARTIAL MATCH (similarity: {similarity:.1%})")
                    else:
                        print(f"❌ Answer VALIDATION: FAILED (similarity: {similarity:.1%})")
                
                elif expected_type == 'embeddings':
                    expected_answer = test_case['expected_answer']
                    actual_answer = extract_answer_from_response(response)
                    
                    print(f"Extraction Method: Embedding answer extraction (with entity type)")
                    print(f"Expected Answer:   '{expected_answer}'")
                    print(f"Extracted Answer:  '{actual_answer}'")
                    print()
                    
                    print("🔍 STAGE 5: ANSWER VALIDATION")
                    print("-" * 80)
                    
                    # For embedding answers, check if entity type is correct
                    if '(type:' in actual_answer:
                        type_match = re.search(r'\(type:\s*([^)]+)\)', actual_answer)
                        entity_match = re.search(r'^([^(]+)\s*\(type:', actual_answer)
                        
                        if type_match and entity_match:
                            actual_type = type_match.group(1).strip()
                            actual_entity = entity_match.group(1).strip()
                            expected_type_match = re.search(r'\(type:\s*([^)]+)\)', expected_answer)
                            expected_entity_match = re.search(r'^([^(]+)\s*\(type:', expected_answer)
                            
                            if expected_type_match and expected_entity_match:
                                expected_type_val = expected_type_match.group(1).strip()
                                expected_entity_val = expected_entity_match.group(1).strip()
                                
                                print(f"Entity Comparison:")
                                print(f"  • Expected Entity: '{expected_entity_val}'")
                                print(f"  • Actual Entity:   '{actual_entity}'")
                                print(f"Entity Type Comparison:")
                                print(f"  • Expected Type: '{expected_type_val}'")
                                print(f"  • Actual Type:   '{actual_type}'")
                                
                                if actual_type == expected_type_val:
                                    results['answer_correct'] += 1
                                    results['embedding_correct'] += 1
                                    print(f"✅ Answer VALIDATION: PASSED (entity type match: {actual_type})")
                                else:
                                    print(f"❌ Answer VALIDATION: FAILED (wrong entity type)")
                            else:
                                results['answer_partial'] += 1
                                print(f"⚠️  Answer VALIDATION: PARTIAL (has type: {actual_type})")
                        else:
                            print(f"❌ Answer VALIDATION: FAILED (malformed type information)")
                    else:
                        print(f"❌ Answer VALIDATION: FAILED (missing type information)")
                
                elif expected_type == 'hybrid':
                    expected_factual = test_case['expected_factual_answer']
                    expected_embedding = test_case['expected_embedding_answer']
                    
                    print(f"Extraction Method: Hybrid (both factual and embedding)")
                    print(f"Expected Factual Answer:   '{expected_factual}'")
                    print(f"Expected Embedding Answer: '{expected_embedding}'")
                    print()
                    
                    # Split response into factual and embedding parts
                    if 'Factual Answer:' in response and 'Embeddings Answer:' in response:
                        parts = response.split('Embeddings Answer:')
                        factual_part = parts[0]
                        embedding_part = parts[1] if len(parts) > 1 else ''
                        
                        factual_answer = extract_answer_from_response(factual_part)
                        embedding_answer = extract_answer_from_response(embedding_part)
                        
                        print(f"Extracted Factual Answer:   '{factual_answer}'")
                        print(f"Extracted Embedding Answer: '{embedding_answer}'")
                        print()
                        
                        print("🔍 STAGE 5: ANSWER VALIDATION")
                        print("-" * 80)
                        
                        # Check factual
                        print("Validating Factual Answer:")
                        factual_match, factual_sim = compare_answers(factual_answer, expected_factual)
                        print(f"  • Similarity: {factual_sim:.1%}")
                        print(f"  • Match: {factual_match}")
                        
                        # Check embedding type
                        print("Validating Embedding Answer:")
                        embedding_match = False
                        if '(type:' in embedding_answer:
                            type_match = re.search(r'\(type:\s*([^)]+)\)', embedding_answer)
                            if type_match:
                                actual_type = type_match.group(1).strip()
                                expected_type_match = re.search(r'\(type:\s*([^)]+)\)', expected_embedding)
                                if expected_type_match:
                                    expected_type_val = expected_type_match.group(1).strip()
                                    embedding_match = actual_type == expected_type_val
                                    print(f"  • Expected Type: {expected_type_val}")
                                    print(f"  • Actual Type: {actual_type}")
                                    print(f"  • Match: {embedding_match}")
                        
                        if factual_match and embedding_match:
                            results['answer_correct'] += 1
                            results['hybrid_correct'] += 1
                            print(f"✅ Answer VALIDATION: PASSED (both answers correct)")
                        elif factual_match or embedding_match:
                            results['answer_partial'] += 1
                            print(f"⚠️  Answer VALIDATION: PARTIAL (factual: {factual_match}, embedding: {embedding_match})")
                        else:
                            print(f"❌ Answer VALIDATION: FAILED (both answers incorrect)")
                    else:
                        print(f"❌ Response format incorrect (missing Factual/Embeddings sections)")
                
            except Exception as e:
                print(f"\n❌ ERROR during query processing:")
                print(f"   {str(e)}")
                print("\nStack trace:")
                import traceback
                traceback.print_exc()
        
        # Summary
        print("="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        print(f"\n📊 Results:")
        print(f"  Classification:     {results['classification_correct']}/{results['total']}")
        print(f"  Answer Validation:")
        print(f"    ✅ Correct:       {results['answer_correct']}/{results['total']}")
        print(f"    ⚠️  Partial:       {results['answer_partial']}/{results['total']}")
        print(f"    ❌ Incorrect:     {results['total'] - results['answer_correct'] - results['answer_partial']}/{results['total']}")
        
        print(f"\n  By Approach:")
        print(f"    Factual:          {results['factual_correct']}/{sum(1 for tc in test_cases if tc['expected_type'] == 'factual')}")
        print(f"    Embedding:        {results['embedding_correct']}/{sum(1 for tc in test_cases if tc['expected_type'] == 'embeddings')}")
        print(f"    Hybrid:           {results['hybrid_correct']}/{sum(1 for tc in test_cases if tc['expected_type'] == 'hybrid')}")
        
        success_rate = results['answer_correct'] / results['total']
        print(f"\n  Overall Success:    {results['answer_correct']}/{results['total']} ({success_rate:.1%})")
        
        if results['answer_correct'] == results['total']:
            print("\n🎉 ALL TESTS PASSED!\n")
        elif success_rate >= 0.7:
            print(f"\n✅ Most tests passed ({success_rate:.1%} success rate)\n")
        else:
            print(f"\n⚠️  {results['total'] - results['answer_correct']} test(s) failed or partially matched\n")
        
        print(f"📝 Full test log saved to: {log_file_path}")
        print(f"📅 Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    finally:
        # ✅ Restore original stdout and close log file
        sys.stdout = original_stdout
        tee.close()
        print(f"\n✅ Log saved to: {log_file_path}")


if __name__ == "__main__":
    test_transformer_pipeline()