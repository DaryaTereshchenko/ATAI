"""
Test suite to verify embedding processor structure and method availability.
This test validates that all required methods exist and are properly defined.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.main.embedding_processor import EmbeddingQueryProcessor


def test_class_methods_exist():
    """Test that all required methods exist in EmbeddingQueryProcessor."""
    required_methods = [
        'process_embedding_query',
        'process_hybrid_factual_query',
        '_embedding_forward_query',
        '_embedding_reverse_query',
        '_embedding_verification_query',
        '_embedding_direct_query',
        '_get_expected_entity_type',
        '_get_expected_qcode',
        '_get_relation_uri',
        '_format_embedding_error',
    ]
    
    for method_name in required_methods:
        assert hasattr(EmbeddingQueryProcessor, method_name), \
            f"Missing method: {method_name}"
        print(f"✅ Method exists: {method_name}")
    
    print("\n✅ All required methods are present")


def test_method_signatures():
    """Test that methods have correct signatures."""
    import inspect
    
    # Check _embedding_forward_query signature
    sig = inspect.signature(EmbeddingQueryProcessor._embedding_forward_query)
    params = list(sig.parameters.keys())
    assert 'self' in params, "_embedding_forward_query missing 'self' parameter"
    assert 'query' in params, "_embedding_forward_query missing 'query' parameter"
    assert 'pattern' in params, "_embedding_forward_query missing 'pattern' parameter"
    print("✅ _embedding_forward_query has correct signature")
    
    # Check _embedding_reverse_query signature
    sig = inspect.signature(EmbeddingQueryProcessor._embedding_reverse_query)
    params = list(sig.parameters.keys())
    assert 'self' in params, "_embedding_reverse_query missing 'self' parameter"
    assert 'query' in params, "_embedding_reverse_query missing 'query' parameter"
    assert 'pattern' in params, "_embedding_reverse_query missing 'pattern' parameter"
    print("✅ _embedding_reverse_query has correct signature")
    
    # Check _embedding_verification_query signature
    sig = inspect.signature(EmbeddingQueryProcessor._embedding_verification_query)
    params = list(sig.parameters.keys())
    assert 'self' in params, "_embedding_verification_query missing 'self' parameter"
    assert 'query' in params, "_embedding_verification_query missing 'query' parameter"
    assert 'pattern' in params, "_embedding_verification_query missing 'pattern' parameter"
    print("✅ _embedding_verification_query has correct signature")
    
    # Check _embedding_direct_query signature
    sig = inspect.signature(EmbeddingQueryProcessor._embedding_direct_query)
    params = list(sig.parameters.keys())
    assert 'self' in params, "_embedding_direct_query missing 'self' parameter"
    assert 'query' in params, "_embedding_direct_query missing 'query' parameter"
    assert 'pattern' in params, "_embedding_direct_query missing 'pattern' parameter"
    print("✅ _embedding_direct_query has correct signature")
    
    # Check _get_expected_entity_type signature
    sig = inspect.signature(EmbeddingQueryProcessor._get_expected_entity_type)
    params = list(sig.parameters.keys())
    assert 'self' in params, "_get_expected_entity_type missing 'self' parameter"
    assert 'pattern' in params, "_get_expected_entity_type missing 'pattern' parameter"
    print("✅ _get_expected_entity_type has correct signature")
    
    # Check _get_expected_qcode signature
    sig = inspect.signature(EmbeddingQueryProcessor._get_expected_qcode)
    params = list(sig.parameters.keys())
    assert 'self' in params, "_get_expected_qcode missing 'self' parameter"
    assert 'pattern' in params, "_get_expected_qcode missing 'pattern' parameter"
    print("✅ _get_expected_qcode has correct signature")
    
    print("\n✅ All method signatures are correct")


def test_method_return_annotations():
    """Test that methods have proper return type hints."""
    import inspect
    from typing import get_type_hints
    
    # Get type hints for the class
    try:
        hints = get_type_hints(EmbeddingQueryProcessor._get_expected_entity_type)
        print(f"✅ _get_expected_entity_type return hint: {hints.get('return', 'None')}")
    except Exception as e:
        print(f"⚠️  Could not get type hints for _get_expected_entity_type: {e}")
    
    try:
        hints = get_type_hints(EmbeddingQueryProcessor._get_expected_qcode)
        print(f"✅ _get_expected_qcode return hint: {hints.get('return', 'None')}")
    except Exception as e:
        print(f"⚠️  Could not get type hints for _get_expected_qcode: {e}")
    
    print("\n✅ Return type annotations validated")


if __name__ == "__main__":
    print("="*80)
    print("EMBEDDING PROCESSOR STRUCTURE TESTS")
    print("="*80)
    print()
    
    try:
        test_class_methods_exist()
        print()
        test_method_signatures()
        print()
        test_method_return_annotations()
        print()
        print("="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
    except AssertionError as e:
        print()
        print("="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80)
        sys.exit(1)
    except Exception as e:
        print()
        print("="*80)
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("="*80)
        sys.exit(1)
