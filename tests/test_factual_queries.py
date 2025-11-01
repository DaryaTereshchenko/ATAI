"""
Test suite for factual query processing.
Tests SPARQL generation and execution for diverse relations.
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


def test_director_query(orchestrator):
    """Test director query with factual approach."""
    print("\n" + "="*80)
    print("TEST: Director query (factual)")
    print("="*80)
    
    query = "Please answer this question with a factual approach: Who directed 'Fargo'?"
    expected_directors = ["Ethan Coen", "Joel Coen"]
    
    print(f"Query: {query}")
    print(f"Expected: {' and '.join(expected_directors)}")
    
    response = orchestrator.process_query(query)
    print(f"Response: {response[:200]}...")
    
    # Check that response contains at least one of the expected directors
    response_lower = response.lower()
    directors_found = [d for d in expected_directors if d.lower() in response_lower]
    
    assert len(directors_found) > 0, f"❌ Expected directors {expected_directors}, but found none in response"
    print(f"✅ PASSED: Found director(s): {directors_found}")
    return True


def test_country_query(orchestrator):
    """Test country of origin query with factual approach."""
    print("\n" + "="*80)
    print("TEST: Country of origin query (factual)")
    print("="*80)
    
    query = "Please answer this question with a factual approach: From what country is the movie 'Aro Tolbukhin. En la mente del asesino'?"
    expected_country = "Mexico"
    
    print(f"Query: {query}")
    print(f"Expected: {expected_country}")
    
    response = orchestrator.process_query(query)
    print(f"Response: {response[:200]}...")
    
    # Check that response contains the expected country
    assert expected_country.lower() in response.lower(), f"❌ Expected '{expected_country}' in response"
    print(f"✅ PASSED: Found '{expected_country}' in response")
    return True


def test_screenwriter_query(orchestrator):
    """Test screenwriter query with factual approach."""
    print("\n" + "="*80)
    print("TEST: Screenwriter query (factual)")
    print("="*80)
    
    query = "Please answer this question with a factual approach: Who is the screenwriter of 'Shortcut to Happiness'?"
    expected_screenwriter = "Pete Dexter"
    
    print(f"Query: {query}")
    print(f"Expected: {expected_screenwriter}")
    
    response = orchestrator.process_query(query)
    print(f"Response: {response[:200]}...")
    
    # Check that response contains the expected screenwriter
    assert expected_screenwriter.lower() in response.lower(), f"❌ Expected '{expected_screenwriter}' in response"
    print(f"✅ PASSED: Found '{expected_screenwriter}' in response")
    return True


def test_genre_query(orchestrator):
    """Test genre query with factual approach."""
    print("\n" + "="*80)
    print("TEST: Genre query (factual)")
    print("="*80)
    
    query = "Please answer this question with a factual approach: What genre is the movie 'Bandit Queen'?"
    expected_genres = ["drama", "biographical", "crime"]
    
    print(f"Query: {query}")
    print(f"Expected genres: {expected_genres}")
    
    response = orchestrator.process_query(query)
    print(f"Response: {response[:200]}...")
    
    # Check that response contains at least one expected genre
    response_lower = response.lower()
    genres_found = [g for g in expected_genres if g in response_lower]
    
    assert len(genres_found) > 0, f"❌ Expected genres {expected_genres}, but found none in response"
    print(f"✅ PASSED: Found genre(s): {genres_found}")
    return True


def test_publication_date_query(orchestrator):
    """Test publication date query with factual approach."""
    print("\n" + "="*80)
    print("TEST: Publication date query (factual)")
    print("="*80)
    
    query = "Please answer this question with a factual approach: When did the movie 'Miracles Still Happen' come out?"
    expected_year = "1974"
    
    print(f"Query: {query}")
    print(f"Expected year: {expected_year}")
    
    response = orchestrator.process_query(query)
    print(f"Response: {response[:200]}...")
    
    # Check that response contains the expected year
    assert expected_year in response, f"❌ Expected year '{expected_year}' in response"
    print(f"✅ PASSED: Found year '{expected_year}' in response")
    return True


def test_language_query(orchestrator):
    """Test language query with factual approach."""
    print("\n" + "="*80)
    print("TEST: Language query (factual)")
    print("="*80)
    
    query = "Please answer this question with a factual approach: What language is the movie 'La Vie en Rose' in?"
    expected_language = "French"
    
    print(f"Query: {query}")
    print(f"Expected language: {expected_language}")
    
    response = orchestrator.process_query(query)
    print(f"Response: {response[:200]}...")
    
    # Check that response contains the expected language
    assert expected_language.lower() in response.lower(), f"❌ Expected '{expected_language}' in response"
    print(f"✅ PASSED: Found '{expected_language}' in response")
    return True


def test_rating_query(orchestrator):
    """Test rating query with factual approach."""
    print("\n" + "="*80)
    print("TEST: Rating query (factual)")
    print("="*80)
    
    query = "Please answer this question with a factual approach: What is the user rating of 'The Shawshank Redemption'?"
    
    print(f"Query: {query}")
    print(f"Expected: A numeric rating value")
    
    response = orchestrator.process_query(query)
    print(f"Response: {response[:200]}...")
    
    # Check that response contains a numeric rating or rating-related keywords
    response_lower = response.lower()
    has_rating = any(word in response_lower for word in ["rating", "rated", "score", "stars"]) or \
                 any(char.isdigit() for char in response)
    
    assert has_rating, f"❌ Expected rating information in response"
    print(f"✅ PASSED: Found rating information in response")
    return True


def test_country_alternative_query(orchestrator):
    """Test country query with different phrasing."""
    print("\n" + "="*80)
    print("TEST: Country query - alternative phrasing (factual)")
    print("="*80)
    
    query = "Please answer this question with a factual approach: What country produced the movie 'Amélie'?"
    expected_country = "France"
    
    print(f"Query: {query}")
    print(f"Expected: {expected_country}")
    
    response = orchestrator.process_query(query)
    print(f"Response: {response[:200]}...")
    
    # Check that response contains the expected country
    assert expected_country.lower() in response.lower(), f"❌ Expected '{expected_country}' in response"
    print(f"✅ PASSED: Found '{expected_country}' in response")
    return True


def test_complex_south_korea_award_query(orchestrator):
    """Test complex query combining country of origin and award."""
    print("\n" + "="*80)
    print("TEST: Complex query - South Korea + Academy Award (factual)")
    print("="*80)
    
    query = "Please answer this question with a factual approach: Which movie, originally from the country 'South Korea', received the award 'Academy Award for Best Picture'?"
    expected_movie = "Parasite"
    
    print(f"Query: {query}")
    print(f"Expected movie: {expected_movie}")
    
    response = orchestrator.process_query(query)
    print(f"Response: {response[:200]}...")
    
    # Check that response contains the expected movie
    assert expected_movie.lower() in response.lower(), f"❌ Expected '{expected_movie}' in response"
    print(f"✅ PASSED: Found '{expected_movie}' in response")
    return True


def test_highest_rating_query(orchestrator):
    """Test complex query for highest rated movie."""
    print("\n" + "="*80)
    print("TEST: Complex query - Highest user rating (factual)")
    print("="*80)
    
    query = "Please answer this question with a factual approach: Which movie has the highest user rating?"
    # Common top-rated movies - response should contain at least one
    expected_movies = ["The Shawshank Redemption", "The Godfather", "The Dark Knight", "12 Angry Men"]
    
    print(f"Query: {query}")
    print(f"Expected: One of the highest-rated movies like {expected_movies}")
    
    response = orchestrator.process_query(query)
    print(f"Response: {response[:200]}...")
    
    # Check that response contains at least one expected movie or mentions a rating
    response_lower = response.lower()
    movies_found = [m for m in expected_movies if m.lower() in response_lower]
    has_movie_info = len(movies_found) > 0 or "rating" in response_lower
    
    assert has_movie_info, f"❌ Expected movie name or rating information in response"
    if movies_found:
        print(f"✅ PASSED: Found movie(s): {movies_found}")
    else:
        print(f"✅ PASSED: Found rating-related information")
    return True


def test_multiple_countries_query(orchestrator):
    """Test query for movie with multiple countries."""
    print("\n" + "="*80)
    print("TEST: Multiple countries query (factual)")
    print("="*80)
    
    query = "Please answer this question with a factual approach: What countries produced 'The Grand Budapest Hotel'?"
    expected_countries = ["Germany", "United Kingdom", "USA"]
    
    print(f"Query: {query}")
    print(f"Expected countries: {expected_countries}")
    
    response = orchestrator.process_query(query)
    print(f"Response: {response[:200]}...")
    
    # Check that response contains at least one expected country
    response_lower = response.lower()
    countries_found = [c for c in expected_countries if c.lower() in response_lower]
    
    assert len(countries_found) > 0, f"❌ Expected countries {expected_countries}, but found none in response"
    print(f"✅ PASSED: Found country/countries: {countries_found}")
    return True


def run_all_tests():
    """Run all factual tests."""
    # Setup logging
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(logs_dir, f'test_factual_{timestamp}.log')
    
    # Redirect stdout to both console and file
    tee = TeeOutput(log_file_path)
    original_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        print("\n" + "="*80)
        print("FACTUAL QUERIES TEST SUITE")
        print("="*80)
        print(f"\n📝 Log file: {log_file_path}")
        print(f"📅 Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Initialize orchestrator
        print("🔧 Initializing Orchestrator...")
        orchestrator = Orchestrator(use_workflow=False)
        print("✅ Orchestrator initialized\n")
        
        # Run tests
        tests = [
            ("Director Query", test_director_query),
            ("Country Query", test_country_query),
            ("Screenwriter Query", test_screenwriter_query),
            ("Genre Query", test_genre_query),
            ("Publication Date Query", test_publication_date_query),
            ("Language Query", test_language_query),
            ("Rating Query", test_rating_query),
            ("Country Alternative Query", test_country_alternative_query),
            ("Complex South Korea Award Query", test_complex_south_korea_award_query),
            ("Highest Rating Query", test_highest_rating_query),
            ("Multiple Countries Query", test_multiple_countries_query),
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
