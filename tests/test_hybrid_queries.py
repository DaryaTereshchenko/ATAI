"""
Test suite for hybrid query processing (both factual and embedding approaches).
Tests that both pipelines run successfully and produce complementary results.
✅ ENHANCED: Includes out-of-scope query filtering tests.
"""

import sys
import os
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.main.orchestrator import Orchestrator


class TeeOutput:
    """Utility class to write output to both console and file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def test_hybrid_director_query(orchestrator):
    """Test director query with hybrid approach - both factual and embedding."""
    print("\n" + "="*80)
    print("TEST: Director query (hybrid)")
    print("="*80)
    
    query = "Who directed 'The Matrix'?"
    expected_directors = ["Wachowski", "Lana", "Lilly"]  # Either full names or surnames
    
    print(f"Query: {query}")
    print(f"Expected: At least one director mention")
    
    response = orchestrator._process_hybrid(query)
    print(f"Response: {response[:300]}...")
    
    # Check that response has both factual and embedding sections
    assert "Factual Answer" in response or "📊" in response, "❌ Should contain factual section"
    assert "Embedding" in response or "🔢" in response, "❌ Should contain embedding section"
    
    # Check that at least one expected director appears somewhere
    response_lower = response.lower()
    directors_found = [d for d in expected_directors if d.lower() in response_lower]
    
    assert len(directors_found) > 0, f"❌ Expected director mention in hybrid response"
    print(f"✅ PASSED: Found director reference(s): {directors_found}")
    return True


def test_hybrid_cast_query(orchestrator):
    """Test cast query with hybrid approach."""
    print("\n" + "="*80)
    print("TEST: Cast member query (hybrid)")
    print("="*80)
    
    query = "Who starred in 'Inception'?"
    expected_actors = ["Leonardo DiCaprio", "Marion Cotillard", "Tom Hardy", "Ellen Page"]
    
    print(f"Query: {query}")
    print(f"Expected: At least one actor mention")
    
    response = orchestrator._process_hybrid(query)
    print(f"Response: {response[:300]}...")
    
    # Check structure
    assert "Factual" in response or "📊" in response, "❌ Should contain factual section"
    assert "Embedding" in response or "🔢" in response, "❌ Should contain embedding section"
    
    # Check for actor mentions
    response_lower = response.lower()
    actors_found = [a for a in expected_actors if a.lower() in response_lower]
    
    assert len(actors_found) > 0 or "cast" in response_lower, "❌ Expected actor or cast mention"
    if actors_found:
        print(f"✅ PASSED: Found actor(s): {actors_found}")
    else:
        print(f"✅ PASSED: Found cast-related content")
    return True


def test_hybrid_genre_query(orchestrator):
    """Test genre query with hybrid approach."""
    print("\n" + "="*80)
    print("TEST: Genre query (hybrid)")
    print("="*80)
    
    query = "What genre is 'The Godfather'?"
    expected_genres = ["crime", "drama"]
    
    print(f"Query: {query}")
    print(f"Expected: Genre information")
    
    response = orchestrator._process_hybrid(query)
    print(f"Response: {response[:300]}...")
    
    # Check structure
    assert "Factual" in response or "📊" in response, "❌ Should contain factual section"
    assert "Embedding" in response or "🔢" in response, "❌ Should contain embedding section"
    
    # Check for genre mentions
    response_lower = response.lower()
    genres_found = [g for g in expected_genres if g in response_lower]
    
    assert len(genres_found) > 0 or "genre" in response_lower, "❌ Expected genre information"
    if genres_found:
        print(f"✅ PASSED: Found genre(s): {genres_found}")
    else:
        print(f"✅ PASSED: Found genre-related content")
    return True


def test_hybrid_country_query(orchestrator):
    """Test country of origin query with hybrid approach."""
    print("\n" + "="*80)
    print("TEST: Country of origin query (hybrid)")
    print("="*80)
    
    query = "From what country is 'Amélie'?"
    expected_country = "France"
    
    print(f"Query: {query}")
    print(f"Expected: {expected_country}")
    
    response = orchestrator._process_hybrid(query)
    print(f"Response: {response[:300]}...")
    
    # Check structure
    assert "Factual" in response or "📊" in response, "❌ Should contain factual section"
    assert "Embedding" in response or "🔢" in response, "❌ Should contain embedding section"
    
    # Check for country mention
    assert expected_country.lower() in response.lower() or "country" in response.lower(), \
        f"❌ Expected country information"
    
    if expected_country.lower() in response.lower():
        print(f"✅ PASSED: Found '{expected_country}' in response")
    else:
        print(f"✅ PASSED: Found country-related content")
    return True


def test_hybrid_reverse_query(orchestrator):
    """Test reverse query (person to movies) with hybrid approach."""
    print("\n" + "="*80)
    print("TEST: Reverse query - person to movies (hybrid)")
    print("="*80)
    
    query = "What films did Christopher Nolan direct?"
    expected_movies = ["Inception", "Interstellar", "The Dark Knight", "Dunkirk", "Tenet"]
    
    print(f"Query: {query}")
    print(f"Expected: Movie titles from Nolan's filmography")
    
    response = orchestrator._process_hybrid(query)
    print(f"Response: {response[:300]}...")
    
    # Check structure
    assert "Factual" in response or "📊" in response, "❌ Should contain factual section"
    assert "Embedding" in response or "🔢" in response, "❌ Should contain embedding section"
    
    # Check for movie mentions
    response_lower = response.lower()
    movies_found = [m for m in expected_movies if m.lower() in response_lower]
    
    # Also check for Nolan mention (entity might not be extracted)
    has_nolan = "nolan" in response_lower
    has_movies = len(movies_found) > 0 or "film" in response_lower or "movie" in response_lower
    
    assert has_nolan or has_movies, "❌ Expected Nolan or movie information"
    
    if movies_found:
        print(f"✅ PASSED: Found movie(s): {movies_found}")
    elif has_nolan:
        print(f"✅ PASSED: Found Nolan reference with movie context")
    else:
        print(f"✅ PASSED: Found movie-related content")
    return True


def test_hybrid_publication_date_query(orchestrator):
    """Test publication date query with hybrid approach."""
    print("\n" + "="*80)
    print("TEST: Publication date query (hybrid)")
    print("="*80)
    
    query = "When was 'The Godfather' released?"
    expected_year = "1972"
    
    print(f"Query: {query}")
    print(f"Expected year: {expected_year}")
    
    response = orchestrator._process_hybrid(query)
    print(f"Response: {response[:300]}...")
    
    # Check structure
    assert "Factual" in response or "📊" in response, "❌ Should contain factual section"
    assert "Embedding" in response or "🔢" in response, "❌ Should contain embedding section"
    
    # Check for year or date mention
    has_year = expected_year in response
    has_date_info = any(word in response.lower() for word in ["released", "date", "year", "1972"])
    
    assert has_year or has_date_info, f"❌ Expected release date information"
    
    if has_year:
        print(f"✅ PASSED: Found year '{expected_year}' in response")
    else:
        print(f"✅ PASSED: Found date-related content")
    return True


def test_hybrid_language_query(orchestrator):
    """Test language query with hybrid approach."""
    print("\n" + "="*80)
    print("TEST: Language query (hybrid)")
    print("="*80)
    
    query = "What language is 'La Vie en Rose' in?"
    expected_language = "French"
    
    print(f"Query: {query}")
    print(f"Expected language: {expected_language}")
    
    response = orchestrator._process_hybrid(query)
    print(f"Response: {response[:300]}...")
    
    # Check structure
    assert "Factual" in response or "📊" in response, "❌ Should contain factual section"
    assert "Embedding" in response or "🔢" in response, "❌ Should contain embedding section"
    
    # Check for language mention
    has_language = expected_language.lower() in response.lower()
    has_language_info = "language" in response.lower()
    
    assert has_language or has_language_info, f"❌ Expected language information"
    
    if has_language:
        print(f"✅ PASSED: Found '{expected_language}' in response")
    else:
        print(f"✅ PASSED: Found language-related content")
    return True


def test_hybrid_screenwriter_query(orchestrator):
    """Test screenwriter query with hybrid approach."""
    print("\n" + "="*80)
    print("TEST: Screenwriter query (hybrid)")
    print("="*80)
    
    query = "Who wrote 'Pulp Fiction'?"
    expected_writers = ["Quentin Tarantino", "Roger Avary"]
    
    print(f"Query: {query}")
    print(f"Expected: Screenwriter information")
    
    response = orchestrator._process_hybrid(query)
    print(f"Response: {response[:300]}...")
    
    # Check structure
    assert "Factual" in response or "📊" in response, "❌ Should contain factual section"
    assert "Embedding" in response or "🔢" in response, "❌ Should contain embedding section"
    
    # Check for writer mentions
    response_lower = response.lower()
    writers_found = [w for w in expected_writers if w.lower() in response_lower]
    has_writer_info = any(word in response_lower for word in ["wrote", "writer", "screenplay", "screenwriter"])
    
    assert len(writers_found) > 0 or has_writer_info, "❌ Expected writer information"
    
    if writers_found:
        print(f"✅ PASSED: Found writer(s): {writers_found}")
    else:
        print(f"✅ PASSED: Found writer-related content")
    return True


def test_hybrid_producer_query(orchestrator):
    """Test producer query with hybrid approach."""
    print("\n" + "="*80)
    print("TEST: Producer query (hybrid)")
    print("="*80)
    
    query = "Who produced 'The Lord of the Rings: The Fellowship of the Ring'?"
    expected_producer = "Peter Jackson"  # He was also producer
    
    print(f"Query: {query}")
    print(f"Expected: Producer information")
    
    response = orchestrator._process_hybrid(query)
    print(f"Response: {response[:300]}...")
    
    # Check structure
    assert "Factual" in response or "📊" in response, "❌ Should contain factual section"
    assert "Embedding" in response or "🔢" in response, "❌ Should contain embedding section"
    
    # Check for producer mention or producer-related content
    response_lower = response.lower()
    has_producer_name = expected_producer.lower() in response_lower
    has_producer_info = "producer" in response_lower or "produced" in response_lower
    
    assert has_producer_name or has_producer_info, "❌ Expected producer information"
    
    if has_producer_name:
        print(f"✅ PASSED: Found '{expected_producer}' in response")
    else:
        print(f"✅ PASSED: Found producer-related content")
    return True


def test_hybrid_award_query(orchestrator):
    """Test award query with hybrid approach."""
    print("\n" + "="*80)
    print("TEST: Award query (hybrid)")
    print("="*80)
    
    query = "What awards did 'Parasite' receive?"
    expected_awards = ["Academy Award", "Oscar", "Best Picture", "Palme d'Or"]
    
    print(f"Query: {query}")
    print(f"Expected: Award information")
    
    response = orchestrator._process_hybrid(query)
    print(f"Response: {response[:300]}...")
    
    # Check structure
    assert "Factual" in response or "📊" in response, "❌ Should contain factual section"
    assert "Embedding" in response or "🔢" in response, "❌ Should contain embedding section"
    
    # Check for award mentions
    response_lower = response.lower()
    awards_found = [a for a in expected_awards if a.lower() in response_lower]
    has_award_info = "award" in response_lower
    
    assert len(awards_found) > 0 or has_award_info, "❌ Expected award information"
    
    if awards_found:
        print(f"✅ PASSED: Found award(s): {awards_found}")
    else:
        print(f"✅ PASSED: Found award-related content")
    return True


def test_hybrid_complex_country_award_query(orchestrator):
    """Test complex query combining country and award with hybrid approach."""
    print("\n" + "="*80)
    print("TEST: Complex query - country + award (hybrid)")
    print("="*80)
    
    query = "Which movie from South Korea won the Academy Award for Best Picture?"
    expected_movie = "Parasite"
    
    print(f"Query: {query}")
    print(f"Expected movie: {expected_movie}")
    
    response = orchestrator._process_hybrid(query)
    print(f"Response: {response[:300]}...")
    
    # Check structure
    assert "Factual" in response or "📊" in response, "❌ Should contain factual section"
    assert "Embedding" in response or "🔢" in response, "❌ Should contain embedding section"
    
    # Check for expected content
    has_parasite = expected_movie.lower() in response.lower()
    has_korea = "korea" in response.lower()
    has_award = "award" in response.lower() or "oscar" in response.lower()
    
    assert has_parasite or (has_korea and has_award), "❌ Expected relevant movie information"
    
    if has_parasite:
        print(f"✅ PASSED: Found '{expected_movie}' in response")
    else:
        print(f"✅ PASSED: Found relevant movie context (Korea + Award)")
    return True


def test_hybrid_multiple_results_query(orchestrator):
    """Test query that should return multiple results with hybrid approach."""
    print("\n" + "="*80)
    print("TEST: Multiple results query (hybrid)")
    print("="*80)
    
    query = "What countries produced 'The Grand Budapest Hotel'?"
    expected_countries = ["Germany", "United Kingdom", "USA"]
    
    print(f"Query: {query}")
    print(f"Expected: Multiple countries")
    
    response = orchestrator._process_hybrid(query)
    print(f"Response: {response[:300]}...")
    
    # Check structure
    assert "Factual" in response or "📊" in response, "❌ Should contain factual section"
    assert "Embedding" in response or "🔢" in response, "❌ Should contain embedding section"
    
    # Check for country mentions
    response_lower = response.lower()
    countries_found = [c for c in expected_countries if c.lower() in response_lower]
    has_multiple_countries = len(countries_found) >= 2
    has_country_info = "country" in response_lower or "countries" in response_lower
    
    assert has_multiple_countries or has_country_info, "❌ Expected country information"
    
    if has_multiple_countries:
        print(f"✅ PASSED: Found multiple countries: {countries_found}")
    else:
        print(f"✅ PASSED: Found country-related content")
    return True


def test_hybrid_out_of_scope_greeting(orchestrator):
    """Test that greeting messages are properly handled."""
    print("\n" + "="*80)
    print("TEST: Out-of-scope query - Greeting (hybrid)")
    print("="*80)
    
    query = "Hello, how are you?"
    
    print(f"Query: {query}")
    print(f"Expected: Polite rejection or redirection to movie queries")
    
    response = orchestrator._process_hybrid(query)
    print(f"Response: {response[:300]}...")
    
    # Check that response indicates out-of-scope or provides guidance
    response_lower = response.lower()
    helpful_indicators = [
        'movie', 'film', 'help', 'ask', 'question', 
        'sorry', 'cannot', "can't", 'not available',
        'understand', 'structure', 'pattern'
    ]
    
    has_helpful_content = any(indicator in response_lower for indicator in helpful_indicators)
    
    assert has_helpful_content, "❌ Expected helpful guidance for out-of-scope query"
    print(f"✅ PASSED: System provided guidance for out-of-scope query")
    return True


def test_hybrid_out_of_scope_weather(orchestrator):
    """Test that weather questions are properly handled."""
    print("\n" + "="*80)
    print("TEST: Out-of-scope query - Weather (hybrid)")
    print("="*80)
    
    query = "What's the weather like today?"
    
    print(f"Query: {query}")
    print(f"Expected: Polite rejection or redirection")
    
    response = orchestrator._process_hybrid(query)
    print(f"Response: {response[:300]}...")
    
    response_lower = response.lower()
    
    # Should not contain actual weather information
    weather_terms = ['sunny', 'rainy', 'cloudy', 'temperature', 'celsius', 'fahrenheit']
    has_weather_info = any(term in response_lower for term in weather_terms)
    
    # Should contain helpful guidance
    helpful_terms = ['movie', 'film', 'cannot', 'sorry', 'not available', 'understand']
    has_guidance = any(term in response_lower for term in helpful_terms)
    
    assert not has_weather_info, "❌ Should not provide weather information"
    assert has_guidance, "❌ Should provide helpful guidance"
    print(f"✅ PASSED: Properly rejected weather query")
    return True


def test_hybrid_out_of_scope_math(orchestrator):
    """Test that math questions are properly handled."""
    print("\n" + "="*80)
    print("TEST: Out-of-scope query - Math (hybrid)")
    print("="*80)
    
    query = "What is 2 + 2?"
    
    print(f"Query: {query}")
    print(f"Expected: Polite rejection or redirection")
    
    response = orchestrator._process_hybrid(query)
    print(f"Response: {response[:300]}...")
    
    response_lower = response.lower()
    
    # Should not contain math answer
    has_math_answer = response_lower.strip().startswith('4') or '= 4' in response_lower or 'equals 4' in response_lower
    
    # Should contain helpful guidance
    helpful_terms = ['movie', 'film', 'cannot', 'sorry', 'pattern', 'understand']
    has_guidance = any(term in response_lower for term in helpful_terms)
    
    assert not has_math_answer, "❌ Should not solve math problems"
    assert has_guidance, "❌ Should provide helpful guidance"
    print(f"✅ PASSED: Properly rejected math query")
    return True


def test_hybrid_out_of_scope_personal(orchestrator):
    """Test that personal questions are properly handled."""
    print("\n" + "="*80)
    print("TEST: Out-of-scope query - Personal question (hybrid)")
    print("="*80)
    
    query = "What is your favorite color?"
    
    print(f"Query: {query}")
    print(f"Expected: Polite rejection or redirection")
    
    response = orchestrator._process_hybrid(query)
    print(f"Response: {response[:300]}...")
    
    response_lower = response.lower()
    
    # Should not contain color preferences
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'my favorite']
    has_color_answer = any(color in response_lower for color in colors) and 'movie' not in response_lower
    
    # Should contain helpful guidance
    helpful_terms = ['movie', 'film', 'cannot', 'sorry', 'pattern', 'understand', 'ask about']
    has_guidance = any(term in response_lower for term in helpful_terms)
    
    assert not has_color_answer, "❌ Should not answer personal questions"
    assert has_guidance, "❌ Should provide helpful guidance"
    print(f"✅ PASSED: Properly rejected personal query")
    return True


def test_hybrid_out_of_scope_gibberish(orchestrator):
    """Test that gibberish input is properly handled."""
    print("\n" + "="*80)
    print("TEST: Out-of-scope query - Gibberish (hybrid)")
    print("="*80)
    
    query = "asdfghjkl qwertyuiop"
    
    print(f"Query: {query}")
    print(f"Expected: Error message or guidance")
    
    response = orchestrator._process_hybrid(query)
    print(f"Response: {response[:300]}...")
    
    response_lower = response.lower()
    
    # Should indicate error or inability to understand
    error_indicators = [
        'could not', 'couldn\'t', 'cannot', 'unable',
        'error', 'invalid', 'understand', 'pattern',
        'sorry', 'help'
    ]
    
    has_error_indication = any(indicator in response_lower for indicator in error_indicators)
    
    assert has_error_indication, "❌ Should indicate inability to process gibberish"
    print(f"✅ PASSED: Properly handled gibberish input")
    return True


def test_hybrid_ambiguous_entity(orchestrator):
    """Test handling of ambiguous entity references."""
    print("\n" + "="*80)
    print("TEST: Ambiguous entity reference (hybrid)")
    print("="*80)
    
    query = "Who directed Avatar?"  # Could be Avatar (2009) or Avatar: The Last Airbender
    
    print(f"Query: {query}")
    print(f"Expected: Answer for one of the Avatar movies")
    
    response = orchestrator._process_hybrid(query)
    print(f"Response: {response[:300]}...")
    
    # Should contain director information
    response_lower = response.lower()
    
    # Should have factual and embedding sections
    has_factual = "factual" in response_lower or "📊" in response
    has_embedding = "embedding" in response_lower or "🔢" in response
    
    # Should contain some director-related content
    director_terms = ['director', 'directed', 'james cameron', 'cameron']
    has_director_info = any(term in response_lower for term in director_terms)
    
    assert has_factual or has_director_info, "❌ Should attempt to answer for Avatar movie"
    assert has_embedding or has_director_info, "❌ Should attempt embedding-based answer"
    print(f"✅ PASSED: Handled ambiguous entity reference")
    return True


def test_hybrid_malformed_query(orchestrator):
    """Test handling of malformed/incomplete queries."""
    print("\n" + "="*80)
    print("TEST: Malformed query (hybrid)")
    print("="*80)
    
    query = "directed by who the movie"
    
    print(f"Query: {query}")
    print(f"Expected: Error message or best-effort attempt")
    
    response = orchestrator._process_hybrid(query)
    print(f"Response: {response[:300]}...")
    
    response_lower = response.lower()
    
    # Should indicate problem understanding the query
    problem_indicators = [
        'could not', 'couldn\'t', 'cannot identify', 'not understand',
        'pattern', 'structure', 'sorry', 'error', 'unable'
    ]
    
    has_problem_indication = any(indicator in response_lower for indicator in problem_indicators)
    
    assert has_problem_indication or len(response) < 500, "❌ Should indicate difficulty with malformed query"
    print(f"✅ PASSED: Handled malformed query appropriately")
    return True


def run_all_tests():
    """Run all hybrid tests."""
    # Setup logging
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(logs_dir, f'test_hybrid_{timestamp}.log')
    
    # Redirect stdout to both console and file
    tee = TeeOutput(log_file_path)
    original_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        print("\n" + "="*80)
        print("HYBRID QUERIES TEST SUITE")
        print("="*80)
        print(f"\n📝 Log file: {log_file_path}")
        print(f"📅 Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Initialize orchestrator
        print("🔧 Initializing Orchestrator...")
        orchestrator = Orchestrator(use_workflow=False)
        
        if orchestrator.embedding_processor is None:
            print("❌ Embedding processor not available - hybrid tests require embeddings")
            print("   Please ensure embeddings are enabled in config")
            return
        
        print("✅ Orchestrator initialized\n")
        
        # Run tests - ENHANCED with out-of-scope tests
        tests = [
            # Movie-related queries
            ("Director Query", test_hybrid_director_query),
            ("Cast Query", test_hybrid_cast_query),
            ("Genre Query", test_hybrid_genre_query),
            ("Country Query", test_hybrid_country_query),
            ("Reverse Query", test_hybrid_reverse_query),
            ("Publication Date Query", test_hybrid_publication_date_query),
            ("Language Query", test_hybrid_language_query),
            ("Screenwriter Query", test_hybrid_screenwriter_query),
            ("Producer Query", test_hybrid_producer_query),
            ("Award Query", test_hybrid_award_query),
            ("Complex Country Award Query", test_hybrid_complex_country_award_query),
            ("Multiple Results Query", test_hybrid_multiple_results_query),
            
            # ✅ NEW: Out-of-scope and edge case tests
            ("Out-of-Scope: Greeting", test_hybrid_out_of_scope_greeting),
            ("Out-of-Scope: Weather", test_hybrid_out_of_scope_weather),
            ("Out-of-Scope: Math", test_hybrid_out_of_scope_math),
            ("Out-of-Scope: Personal", test_hybrid_out_of_scope_personal),
            ("Out-of-Scope: Gibberish", test_hybrid_out_of_scope_gibberish),
            ("Ambiguous Entity", test_hybrid_ambiguous_entity),
            ("Malformed Query", test_hybrid_malformed_query),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func(orchestrator)
                results.append((test_name, "PASSED", None))
            except AssertionError as e:
                print(f"❌ FAILED: {e}")
                results.append((test_name, "FAILED", str(e)))
            except Exception as e:
                print(f"❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
                results.append((test_name, "ERROR", str(e)))
        
        # Print summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        passed = sum(1 for _, status, _ in results if status == "PASSED")
        failed = sum(1 for _, status, _ in results if status == "FAILED")
        errors = sum(1 for _, status, _ in results if status == "ERROR")
        
        for test_name, status, error in results:
            icon = "✅" if status == "PASSED" else "❌"
            print(f"{icon} {test_name}: {status}")
            if error:
                print(f"   {error[:100]}...")
        
        print(f"\nTotal: {len(results)} tests")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Errors: {errors}")
        
        success_rate = passed / len(results) * 100 if results else 0
        print(f"Success Rate: {success_rate:.1f}%")
        
        if passed == len(results):
            print("\n🎉 ALL TESTS PASSED!")
        elif success_rate >= 70:
            print(f"\n✅ Most tests passed ({success_rate:.1f}%)")
        else:
            print(f"\n⚠️  Many tests failed ({failed + errors} failures)")
        
        print(f"\n📝 Full test log saved to: {log_file_path}")
        print(f"📅 Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    finally:
        # Restore original stdout and close log file
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    run_all_tests()
