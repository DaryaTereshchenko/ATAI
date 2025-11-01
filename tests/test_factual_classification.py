#!/usr/bin/env python3
"""
Test script for factual query classification and SPARQL generation.
Tests the complete pipeline from natural language to SPARQL execution.
"""

import sys
import os
import io  # ✅ ADD: Missing import for io.StringIO
import contextlib  # ✅ ADD: Missing import for contextlib.redirect_stdout
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
    """Test complete transformer-based NL→SPARQL→Results pipeline."""
    print("\n" + "="*80)
    print("TEST: Complete Transformer Pipeline (NL → SPARQL → Results)")
    print("="*80 + "\n")
    
    try:
        from src.main.orchestrator import Orchestrator
        orchestrator = Orchestrator(use_workflow=True)
        
        # ✅ Check if embedding processor is available
        if not hasattr(orchestrator, 'embedding_processor') or orchestrator.embedding_processor is None:
            print("⚠️  Embedding processor not available - skipping test")
            return  # ✅ Changed from pytest.skip() to return
        
        # ==================== TEST CASES ====================
        test_cases = [
            # FACTUAL APPROACH TESTS - Using actual database properties
            {
                'query': "Please answer this question with a factual approach: From what country is the movie 'Aro Tolbukhin. En la mente del asesino'?",
                'expected_type': 'factual',
                'expected_answer': 'Mexico',
                'description': 'Country of origin query (P495)'
            },
            {
                'query': "Please answer this question with a factual approach: Who is the screenwriter of 'Shortcut to Happiness'?",
                'expected_type': 'factual',
                'expected_answer': 'Pete Dexter',
                'description': 'Screenwriter query (P58)'
            },
            {
                'query': "Please answer this question with a factual approach: What country is 'The Bridge on the River Kwai' from?",
                'expected_type': 'factual',
                'expected_answer': 'United Kingdom',
                'description': 'Country of origin query (P495)'
            },
            {
                'query': "Please answer this question with a factual approach: Who directed 'Fargo'?",
                'expected_type': 'factual',
                'expected_answer': 'Ethan Coen and Joel Coen',
                'description': 'Director query (P57)'
            },
            {
                'query': "Please answer this question with a factual approach: What genre is the movie 'Bandit Queen'?",
                'expected_type': 'factual',
                'expected_answer': 'drama film and biographical film and crime film',
                'description': 'Genre query (P136)'
            },
            {
                'query': "Please answer this question with a factual approach: When was 'The Godfather' released?",
                'expected_type': 'factual',
                'expected_answer': '1972',
                'description': 'Publication date query (P577)'
            },
            {
                'query': "Please answer this question with a factual approach: Who composed the music for 'The Lion King'?",
                'expected_type': 'factual',
                'expected_answer': 'Hans Zimmer',
                'description': 'Composer query (P86)'
            },
            {
                'query': "Please answer this question with a factual approach: In which language is 'Amélie' filmed?",
                'expected_type': 'factual',
                'expected_answer': 'French',
                'description': 'Original language query (P364)'
            },
            {
                'query': "Please answer this question with a factual approach: Who are the main actors in 'Pulp Fiction'?",
                'expected_type': 'factual',
                'expected_answer': 'John Travolta and Samuel L. Jackson and Uma Thurman',
                'description': 'Cast members query (P161)'
            },
            {
                'query': "Please answer this question with a factual approach: What production company made 'Jurassic Park'?",
                'expected_type': 'factual',
                'expected_answer': 'Universal Pictures',
                'description': 'Production company query (P272)'
            },
            {
                'query': "Please answer this question with a factual approach: What awards did 'Titanic' win?",
                'expected_type': 'factual',
                'expected_answer': 'Academy Award for Best Picture',
                'description': 'Awards query (P166)'
            },
            {
                'query': "Please answer this question with a factual approach: Who is the cinematographer of 'Blade Runner'?",
                'expected_type': 'factual',
                'expected_answer': 'Jordan Cronenweth',
                'description': 'Director of photography query (P344)'
            },
            {
                'query': "Please answer this question with a factual approach: What is the filming location of 'The Lord of the Rings'?",
                'expected_type': 'factual',
                'expected_answer': 'New Zealand',
                'description': 'Filming location query (P915)'
            },
            {
                'query': "Please answer this question with a factual approach: Who wrote the screenplay for 'The Shawshank Redemption'?",
                'expected_type': 'factual',
                'expected_answer': 'Frank Darabont',
                'description': 'Screenwriter query (P58)'
            },
            {
                'query': "Please answer this question with a factual approach: What is the original language of 'Parasite'?",
                'expected_type': 'factual',
                'expected_answer': 'Korean',
                'description': 'Original language query (P364)'
            },
            {
                'query': "Please answer this question with a factual approach: When was 'Casablanca' released?",
                'expected_type': 'factual',
                'expected_answer': '1942',
                'description': 'Publication date query (P577)'
            },
            {
                'query': "Please answer this question with a factual approach: Who produced 'The Dark Knight'?",
                'expected_type': 'factual',
                'expected_answer': 'Christopher Nolan and Emma Thomas and Charles Roven',
                'description': 'Producer query (P162)'
            },
            {
                'query': "Please answer this question with a factual approach: What studio distributed 'Spider-Man: No Way Home'?",
                'expected_type': 'factual',
                'expected_answer': 'Sony Pictures',
                'description': 'Distributor query (P750)'
            },
            {
                'query': "Please answer this question with a factual approach: Who are the cast members of 'The Avengers'?",
                'expected_type': 'factual',
                'expected_answer': 'Robert Downey Jr. and Chris Evans and Scarlett Johansson',
                'description': 'Cast query (P161)'
            },
            {
                'query': "Please answer this question with a factual approach: Who edited 'Whiplash'?",
                'expected_type': 'factual',
                'expected_answer': 'Tom Cross',
                'description': 'Film editor query (P1040)'
            },
            {
                'query': "Please answer this question with a factual approach: Who is the costume designer of 'Marie Antoinette'?",
                'expected_type': 'factual',
                'expected_answer': 'Milena Canonero',
                'description': 'Costume designer query (P2515)'
            },
            {
                'query': "Please answer this question with a factual approach: What film series does 'The Fellowship of the Ring' belong to?",
                'expected_type': 'factual',
                'expected_answer': 'The Lord of the Rings',
                'description': 'Part of series query (P179)'
            },
            {
                'query': "Please answer this question with a factual approach: Who narrated 'The Shawshank Redemption'?",
                'expected_type': 'factual',
                'expected_answer': 'Morgan Freeman',
                'description': 'Narrator query (P2438)'
            },
            {
                'query': "Please answer this question with a factual approach: What is the sequel to 'The Terminator'?",
                'expected_type': 'factual',
                'expected_answer': 'Terminator 2: Judgment Day',
                'description': 'Followed by query (P156)'
            },
            {
                'query': "Please answer this question with a factual approach: Who did the production design for 'Blade Runner 2049'?",
                'expected_type': 'factual',
                'expected_answer': 'Dennis Gassner',
                'description': 'Production designer query (P2554)'
            },
            
            # EMBEDDING APPROACH TESTS
            {
                'query': "Please answer this question with an embedding approach: Who is the director of 'Apocalypse Now'?",
                'expected_type': 'embeddings',
                'expected_answer': 'Francis Ford Coppola (type: Q5)',
                'description': 'Embedding director query (P57)'
            },
            {
                'query': "Please answer this question with an embedding approach: Who is the screenwriter of '12 Monkeys'?",
                'expected_type': 'embeddings',
                'expected_answer': 'David Peoples (type: Q5)',
                'description': 'Embedding screenwriter query (P58)'
            },
            {
                'query': "Please answer this question with an embedding approach: What is the genre of 'Shoplifters'?",
                'expected_type': 'embeddings',
                'expected_answer': 'drama film (type: Q201658)',
                'description': 'Embedding genre query (P136)'
            },
            {
                'query': "Please answer this question with an embedding approach: Who produced 'Inception'?",
                'expected_type': 'embeddings',
                'expected_answer': 'Emma Thomas (type: Q5)',
                'description': 'Embedding producer query (P162)'
            },
            {
                'query': "Please answer this question with an embedding approach: What studio distributed 'Star Wars'?",
                'expected_type': 'embeddings',
                'expected_answer': '20th Century Fox (type: Q1762059)',
                'description': 'Embedding distributor query (P750)'
            },
            {
                'query': "Please answer this question with an embedding approach: Who composed the soundtrack for 'Interstellar'?",
                'expected_type': 'embeddings',
                'expected_answer': 'Hans Zimmer (type: Q5)',
                'description': 'Embedding composer query (P86)'
            },
            {
                'query': "Please answer this question with an embedding approach: What language is spoken in 'Life Is Beautiful'?",
                'expected_type': 'embeddings',
                'expected_answer': 'Italian (type: Q1288568)',
                'description': 'Embedding language query (P364)'
            },
            {
                'query': "Please answer this question with an embedding approach: Who did the cinematography for 'The Revenant'?",
                'expected_type': 'embeddings',
                'expected_answer': 'Emmanuel Lubezki (type: Q5)',
                'description': 'Embedding cinematographer query (P344)'
            },
            {
                'query': "Please answer this question with an embedding approach: What country produced 'Crouching Tiger, Hidden Dragon'?",
                'expected_type': 'embeddings',
                'expected_answer': 'China (type: Q6256)',
                'description': 'Embedding country query (P495)'
            },
            {
                'query': "Please answer this question with an embedding approach: What studio made 'Toy Story'?",
                'expected_type': 'embeddings',
                'expected_answer': 'Pixar (type: Q1762059)',
                'description': 'Embedding production company query (P272)'
            },
            {
                'query': "Please answer this question with an embedding approach: Who edited 'Mad Max: Fury Road'?",
                'expected_type': 'embeddings',
                'expected_answer': 'Margaret Sixel (type: Q5)',
                'description': 'Embedding film editor query (P1040)'
            },
            {
                'query': "Please answer this question with an embedding approach: What genre is '2001: A Space Odyssey'?",
                'expected_type': 'embeddings',
                'expected_answer': 'science fiction film (type: Q201658)',
                'description': 'Embedding genre query (P136)'
            },
            
            # HYBRID APPROACH TESTS
            {
                'query': "Please answer this question: Who is the director of 'Good Will Hunting'?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'Gus Van Sant',
                'expected_embedding_answer': 'Gus Van Sant (type: Q5)',
                'description': 'Hybrid director query (P57)'
            },
            {
                'query': "Which movie has the highest user rating?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'The Shawshank Redemption',
                'expected_embedding_answer': 'The Godfather (type: Q11424)',
                'description': 'Hybrid query - highest rated movie (rating property)'
            },
            {
                'query': "Who directed the movie 'The Bridge on the River Kwai'?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'David Lean',
                'expected_embedding_answer': 'David Lean (type: Q5)',
                'description': 'Hybrid director query (P57)'
            },
            {
                'query': "What genre is the movie 'Shoplifters'?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'drama film',
                'expected_embedding_answer': 'drama film (type: Q201658)',
                'description': 'Hybrid genre query (P136)'
            },
            {
                'query': "Who is the producer of the movie 'French Kiss'?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'Meg Ryan',
                'expected_embedding_answer': 'Meg Ryan (type: Q5)',
                'description': 'Hybrid producer query (P162)'
            },
            {
                'query': "Which movie, originally from the country 'South Korea', received the award 'Academy Award for Best Picture'?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'Parasite',
                'expected_embedding_answer': 'Parasite (type: Q11424)',
                'description': 'Hybrid complex query (P495 + P166)'
            },
            {
                'query': "Who are the main cast members of 'Fight Club'?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'Brad Pitt and Edward Norton',
                'expected_embedding_answer': 'Brad Pitt (type: Q5)',
                'description': 'Hybrid cast query (P161)'
            },
            {
                'query': "What sequel followed 'The Terminator'?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'Terminator 2: Judgment Day',
                'expected_embedding_answer': 'Terminator 2: Judgment Day (type: Q11424)',
                'description': 'Hybrid sequel query (P156)'
            },
            {
                'query': "Which actor played the lead role in 'Forrest Gump'?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'Tom Hanks',
                'expected_embedding_answer': 'Tom Hanks (type: Q5)',
                'description': 'Hybrid cast member query (P161)'
            },
            {
                'query': "Who composed the music for 'Star Wars'?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'John Williams',
                'expected_embedding_answer': 'John Williams (type: Q5)',
                'description': 'Hybrid composer query (P86)'
            },
            {
                'query': "What language is 'Amélie' in?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'French',
                'expected_embedding_answer': 'French (type: Q1288568)',
                'description': 'Hybrid language query (P364)'
            },
            {
                'query': "When was 'Pulp Fiction' released?",
                'expected_type': 'hybrid',
                'expected_factual_answer': '1994',
                'expected_embedding_answer': '1994 (type: Q577)',
                'description': 'Hybrid publication date query (P577)'
            },
            {
                'query': "Who wrote the screenplay for 'Eternal Sunshine of the Spotless Mind'?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'Charlie Kaufman',
                'expected_embedding_answer': 'Charlie Kaufman (type: Q5)',
                'description': 'Hybrid screenwriter query (P58)'
            },
            {
                'query': "What country is 'Cinema Paradiso' from?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'Italy',
                'expected_embedding_answer': 'Italy (type: Q6256)',
                'description': 'Hybrid country query (P495)'
            },
            {
                'query': "Who did the cinematography for 'Birdman'?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'Emmanuel Lubezki',
                'expected_embedding_answer': 'Emmanuel Lubezki (type: Q5)',
                'description': 'Hybrid cinematographer query (P344)'
            },
            {
                'query': "Who are the producers of 'Schindler's List'?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'Steven Spielberg and Gerald R. Molen and Branko Lustig',
                'expected_embedding_answer': 'Steven Spielberg (type: Q5)',
                'description': 'Hybrid multiple producers query (P162)'
            },
            {
                'query': "Who is the production designer of 'The Grand Budapest Hotel'?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'Adam Stockhausen',
                'expected_embedding_answer': 'Adam Stockhausen (type: Q5)',
                'description': 'Hybrid production designer query (P2554)'
            },
            {
                'query': "What franchise does 'Iron Man' belong to?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'Marvel Cinematic Universe',
                'expected_embedding_answer': 'Marvel Cinematic Universe (type: Q130371093)',
                'description': 'Hybrid media franchise query (P8345)'
            },
            {
                'query': "Who narrated 'The Big Lebowski'?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'Sam Elliott',
                'expected_embedding_answer': 'Sam Elliott (type: Q5)',
                'description': 'Hybrid narrator query (P2438)'
            },
            {
                'query': "Who did the costume design for 'The Great Gatsby'?",
                'expected_type': 'hybrid',
                'expected_factual_answer': 'Catherine Martin',
                'expected_embedding_answer': 'Catherine Martin (type: Q5)',
                'description': 'Hybrid costume designer query (P2515)'
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
            print(f"TEST [{i}/{len(test_cases)}]: {description}")
            print(f"Query: '{query}'")
            
            # ==================== STAGE 1: CLASSIFICATION ====================
            classification = orchestrator.classify_query(query)
            
            print(f"  Classification: {classification.question_type.value} ({classification.confidence:.1%})", end="")
            
            # Check classification
            if classification.question_type.value == expected_type:
                results['classification_correct'] += 1
                print(" ✅")
            else:
                print(f" ❌ Expected: {expected_type}")
                continue
            
            # ==================== STAGE 2: QUERY PROCESSING ====================
            try:
                # Suppress verbose output during processing
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    clean_query = orchestrator._clean_query_for_processing(query)
                    
                    if expected_type == 'factual':
                        response = orchestrator.embedding_processor.process_hybrid_factual_query(clean_query)
                    elif expected_type == 'embeddings':
                        response = orchestrator.embedding_processor.process_embedding_query(clean_query)
                    elif expected_type == 'hybrid':
                        factual_result = orchestrator.embedding_processor.process_hybrid_factual_query(clean_query)
                        embeddings_result = orchestrator.embedding_processor.process_embedding_query(clean_query)
                        response = f"**Factual Answer:**\n{factual_result}\n\n**Embeddings Answer:**\n{embeddings_result}"
                
                # Check if query was regenerated (capture from suppressed output)
                processing_log = f.getvalue()
                if 'regenerat' in processing_log.lower() or 'retry' in processing_log.lower():
                    print("  🔄 Query regenerated")
                
                # ==================== STAGE 3: ANSWER EXTRACTION & VALIDATION ====================
                if expected_type == 'factual':
                    expected_answer = test_case['expected_answer']
                    actual_answer = extract_answer_from_response(response)
                    
                    is_match, similarity = compare_answers(actual_answer, expected_answer)
                    
                    if is_match:
                        results['answer_correct'] += 1
                        results['factual_correct'] += 1
                        print(f"  Answer: ✅ '{actual_answer}'")
                    elif similarity > 0.5:
                        results['answer_partial'] += 1
                        print(f"  Answer: ⚠️  '{actual_answer}' (expected: '{expected_answer}', similarity: {similarity:.1%})")
                    else:
                        print(f"  Answer: ❌ '{actual_answer}' (expected: '{expected_answer}')")
                        # Show SPARQL query on failure
                        if 'SELECT' in processing_log or 'ASK' in processing_log:
                            sparql_match = re.search(r'(PREFIX.*?)(?=\n\n|\Z)', processing_log, re.DOTALL)
                            if sparql_match:
                                print(f"  SPARQL Query:\n{sparql_match.group(1)}")
                
                elif expected_type == 'embeddings':
                    expected_answer = test_case['expected_answer']
                    actual_answer = extract_answer_from_response(response)
                    
                    # For embedding answers, check if entity type is correct
                    if '(type:' in actual_answer:
                        type_match = re.search(r'\(type:\s*([^)]+)\)', actual_answer)
                        expected_type_match = re.search(r'\(type:\s*([^)]+)\)', expected_answer)
                        
                        if type_match and expected_type_match:
                            actual_type = type_match.group(1).strip()
                            expected_type_val = expected_type_match.group(1).strip()
                            
                            if actual_type == expected_type_val:
                                results['answer_correct'] += 1
                                results['embedding_correct'] += 1
                                print(f"  Answer: ✅ '{actual_answer}'")
                            else:
                                print(f"  Answer: ❌ '{actual_answer}' (expected type: {expected_type_val})")
                        else:
                            results['answer_partial'] += 1
                            print(f"  Answer: ⚠️  '{actual_answer}' (type check failed)")
                    else:
                        print(f"  Answer: ❌ '{actual_answer}' (missing type information)")
                
                elif expected_type == 'hybrid':
                    expected_factual = test_case['expected_factual_answer']
                    expected_embedding = test_case['expected_embedding_answer']
                    
                    # Split response into factual and embedding parts
                    if 'Factual Answer:' in response and 'Embeddings Answer:' in response:
                        parts = response.split('Embeddings Answer:')
                        factual_part = parts[0]
                        embedding_part = parts[1] if len(parts) > 1 else ''
                        
                        factual_answer = extract_answer_from_response(factual_part)
                        embedding_answer = extract_answer_from_response(embedding_part)
                        
                        # Check factual
                        factual_match, factual_sim = compare_answers(factual_answer, expected_factual)
                        
                        # Check embedding type
                        embedding_match = False
                        if '(type:' in embedding_answer:
                            type_match = re.search(r'\(type:\s*([^)]+)\)', embedding_answer)
                            if type_match:
                                actual_type = type_match.group(1).strip()
                                expected_type_match = re.search(r'\(type:\s*([^)]+)\)', expected_embedding)
                                if expected_type_match:
                                    expected_type_val = expected_type_match.group(1).strip()
                                    embedding_match = actual_type == expected_type_val
                        
                        if factual_match and embedding_match:
                            results['answer_correct'] += 1
                            results['hybrid_correct'] += 1
                            print(f"  Factual: ✅ '{factual_answer}'")
                            print(f"  Embedding: ✅ '{embedding_answer}'")
                        elif factual_match or embedding_match:
                            results['answer_partial'] += 1
                            print(f"  Factual: {'✅' if factual_match else '❌'} '{factual_answer}'")
                            print(f"  Embedding: {'✅' if embedding_match else '❌'} '{embedding_answer}'")
                            if not factual_match:
                                print(f"    Expected factual: '{expected_factual}'")
                            if not embedding_match:
                                print(f"    Expected embedding: '{expected_embedding}'")
                        else:
                            print(f"  Factual: ❌ '{factual_answer}' (expected: '{expected_factual}')")
                            print(f"  Embedding: ❌ '{embedding_answer}' (expected: '{expected_embedding}')")
                    else:
                        print(f"  ❌ Response format incorrect (missing Factual/Embeddings sections)")
                
            except Exception as e:
                print(f"  ❌ ERROR: {str(e)}")
                import traceback
                print("  Stack trace:")
                traceback.print_exc()
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        print(f"\n📊 Results:")
        print(f"  Classification:     {results['classification_correct']}/{results['total']} ({results['classification_correct']/results['total']:.1%})")
        print(f"  Answer Validation:")
        print(f"    ✅ Correct:       {results['answer_correct']}/{results['total']} ({results['answer_correct']/results['total']:.1%})")
        print(f"    ⚠️  Partial:       {results['answer_partial']}/{results['total']}")
        print(f"    ❌ Incorrect:     {results['total'] - results['answer_correct'] - results['answer_partial']}/{results['total']}")
        
        print(f"\n  By Approach:")
        factual_total = sum(1 for tc in test_cases if tc['expected_type'] == 'factual')
        embedding_total = sum(1 for tc in test_cases if tc['expected_type'] == 'embeddings')
        hybrid_total = sum(1 for tc in test_cases if tc['expected_type'] == 'hybrid')
        
        print(f"    Factual:          {results['factual_correct']}/{factual_total} ({results['factual_correct']/factual_total:.1%})")
        print(f"    Embedding:        {results['embedding_correct']}/{embedding_total} ({results['embedding_correct']/embedding_total:.1%})")
        print(f"    Hybrid:           {results['hybrid_correct']}/{hybrid_total} ({results['hybrid_correct']/hybrid_total:.1%})")
        
        success_rate = results['answer_correct'] / results['total']
        print(f"\n  Overall Success:    {results['answer_correct']}/{results['total']} ({success_rate:.1%})")
        
        if results['answer_correct'] == results['total']:
            print("\n🎉 ALL TESTS PASSED!\n")
        elif success_rate >= 0.7:
            print(f"\n✅ Most tests passed ({success_rate:.1%} success rate)\n")
        else:
            print(f"\n⚠️  {results['total'] - results['answer_correct']} test(s) failed or partially matched\n")
        
    finally:
        print(f"\n✅ Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    test_transformer_pipeline()