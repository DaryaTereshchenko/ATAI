"""
Test to verify the fix for the AttributeError: 'EmbeddingQueryProcessor' object has no attribute '_get_expected_entity_type'
"""

import sys
import os
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import Optional

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


@dataclass
class MockQueryPattern:
    """Mock QueryPattern for testing."""
    pattern_type: str
    relation: str
    subject_type: Optional[str] = None
    object_type: Optional[str] = None
    confidence: float = 0.9
    extracted_entities: Optional[dict] = None


def test_embedding_forward_query_calls_get_expected_entity_type():
    """
    Test that _embedding_forward_query successfully calls _get_expected_entity_type
    without AttributeError.
    """
    print("\n" + "="*80)
    print("TEST: Verify _get_expected_entity_type is accessible")
    print("="*80 + "\n")
    
    # Import after mocking to avoid import-time dependencies
    from src.main.embedding_processor import EmbeddingQueryProcessor
    
    # Create mock pattern
    pattern = MockQueryPattern(
        pattern_type='forward',
        relation='director',
        subject_type='movie',
        object_type='person',
        confidence=0.95
    )
    
    # Create a minimally mocked processor instance
    with patch.object(EmbeddingQueryProcessor, '__init__', lambda x, *args, **kwargs: None):
        processor = EmbeddingQueryProcessor()
        
        # Mock the required attributes and methods
        processor.entity_extractor = Mock()
        processor.embedding_handler = Mock()
        processor.query_analyzer = Mock()
        processor.aligner = Mock()
        processor.query_embedder = Mock()
        processor.sparql_handler = Mock()
        processor.sparql_generator = Mock()
        processor.relation_manager = None
        
        # Mock entity extraction to return a movie
        processor.entity_extractor.extract_entities.return_value = [
            ('http://www.wikidata.org/entity/Q83422', 'The Matrix', 95)
        ]
        processor.entity_extractor.get_entity_label.return_value = 'The Matrix'
        
        # Mock embedding handler
        import numpy as np
        processor.embedding_handler.get_entity_embedding.return_value = np.array([0.1] * 100)
        processor.embedding_handler.get_relation_embedding.return_value = np.array([0.2] * 100)
        processor.embedding_handler.find_nearest_entities.return_value = [
            ('http://www.wikidata.org/entity/Q51564', 0.95)
        ]
        processor.embedding_handler.get_entity_label.return_value = 'Wachowski Brothers'
        processor.embedding_handler.get_entity_type_qcode.return_value = 'Q5'
        processor.embedding_handler.get_entities_by_type.return_value = []
        
        processor.sparql_handler.graph = Mock()
        processor.sparql_handler.execute_query.return_value = {
            'success': True,
            'data': 'Q5'
        }
        
        # Create a mock _get_relation_uri to avoid RelationManager dependency
        processor._get_relation_uri = Mock(return_value='http://www.wikidata.org/prop/direct/P57')
        
        # Create mock for validate and select
        processor._validate_and_select_result = Mock(return_value=(
            'http://www.wikidata.org/entity/Q51564', 0.95
        ))
        processor._get_entity_type_label = Mock(return_value='Q5')
        
        # This is the critical test - calling _get_expected_entity_type should not raise AttributeError
        try:
            result_type = processor._get_expected_entity_type(pattern)
            print(f"✅ _get_expected_entity_type returned: {result_type}")
            
            result_qcode = processor._get_expected_qcode(pattern)
            print(f"✅ _get_expected_qcode returned: {result_qcode}")
            
            # Now test the full forward query flow
            print("\n📝 Testing full _embedding_forward_query flow...")
            result = processor._embedding_forward_query("Who directed The Matrix?", pattern)
            
            print(f"✅ Forward query completed without AttributeError")
            print(f"   Result preview: {result[:100]}...")
            
            return True
            
        except AttributeError as e:
            print(f"❌ AttributeError occurred: {e}")
            import traceback
            traceback.print_exc()
            raise


def test_embedding_verification_query_exists():
    """Test that _embedding_verification_query method exists and works."""
    print("\n" + "="*80)
    print("TEST: Verify _embedding_verification_query exists")
    print("="*80 + "\n")
    
    from src.main.embedding_processor import EmbeddingQueryProcessor
    
    # Create mock pattern
    pattern = MockQueryPattern(
        pattern_type='verification',
        relation='director',
        confidence=0.85
    )
    
    with patch.object(EmbeddingQueryProcessor, '__init__', lambda x, *args, **kwargs: None):
        processor = EmbeddingQueryProcessor()
        
        try:
            result = processor._embedding_verification_query("Did X direct Y?", pattern)
            print(f"✅ _embedding_verification_query executed successfully")
            print(f"   Result: {result[:150]}...")
            assert "not supported" in result.lower(), "Should indicate verification is not supported"
            return True
        except AttributeError as e:
            print(f"❌ AttributeError occurred: {e}")
            raise


def test_embedding_direct_query_exists():
    """Test that _embedding_direct_query method exists and works."""
    print("\n" + "="*80)
    print("TEST: Verify _embedding_direct_query exists")
    print("="*80 + "\n")
    
    from src.main.embedding_processor import EmbeddingQueryProcessor
    
    pattern = MockQueryPattern(
        pattern_type='unknown',
        relation='',
        confidence=0.2
    )
    
    with patch.object(EmbeddingQueryProcessor, '__init__', lambda x, *args, **kwargs: None):
        processor = EmbeddingQueryProcessor()
        
        # Mock required dependencies
        processor.query_embedder = Mock()
        processor.aligner = Mock()
        processor.embedding_handler = Mock()
        processor.sparql_handler = Mock()
        
        import numpy as np
        processor.query_embedder.embed_query.return_value = np.array([0.1] * 384)
        processor.aligner.align.return_value = np.array([0.1] * 100)
        processor.embedding_handler.find_nearest_entities.return_value = [
            ('http://www.wikidata.org/entity/Q83422', 0.85)
        ]
        processor.embedding_handler.get_entity_label.return_value = 'The Matrix'
        processor.embedding_handler.get_entity_type_qcode.return_value = 'Q11424'
        processor.sparql_handler.graph = Mock()
        
        try:
            result = processor._embedding_direct_query("Some query", pattern)
            print(f"✅ _embedding_direct_query executed successfully")
            print(f"   Result preview: {result[:150]}...")
            return True
        except AttributeError as e:
            print(f"❌ AttributeError occurred: {e}")
            raise


if __name__ == "__main__":
    print("="*80)
    print("EMBEDDING PROCESSOR ATTRIBUTION ERROR FIX TESTS")
    print("="*80)
    
    all_passed = True
    
    try:
        test_embedding_forward_query_calls_get_expected_entity_type()
        print("\n✅ Test 1 passed\n")
    except Exception as e:
        print(f"\n❌ Test 1 failed: {e}\n")
        all_passed = False
    
    try:
        test_embedding_verification_query_exists()
        print("\n✅ Test 2 passed\n")
    except Exception as e:
        print(f"\n❌ Test 2 failed: {e}\n")
        all_passed = False
    
    try:
        test_embedding_direct_query_exists()
        print("\n✅ Test 3 passed\n")
    except Exception as e:
        print(f"\n❌ Test 3 failed: {e}\n")
        all_passed = False
    
    print("="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED - AttributeError fix verified")
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
    print("="*80)
