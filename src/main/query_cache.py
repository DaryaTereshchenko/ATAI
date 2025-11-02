"""
Local file-based query cache for storing question-answer pairs.
Uses JSON file storage and SHA-256 hashing for cache keys.
"""

import json
import hashlib
import os
from datetime import datetime
from typing import Optional, Dict
from pathlib import Path


class QueryCache:
    """Local file-based cache for storing query results."""
    
    def __init__(self, cache_file: str = None):
        """
        Initialize the query cache.
        
        Args:
            cache_file: Path to cache file (default: .cache/query_cache.json in project root)
        """
        if cache_file is None:
            project_root = Path(__file__).parent.parent.parent
            cache_dir = project_root / ".cache"
            cache_dir.mkdir(exist_ok=True)
            cache_file = str(cache_dir / "query_cache.json")
        
        self.cache_file = cache_file
        self._cache: Dict[str, dict] = {}
        self._load_cache()
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for consistent cache lookup.
        
        Args:
            query: Raw query string
            
        Returns:
            Normalized query (lowercase, trimmed, collapsed whitespace)
        """
        # Convert to lowercase, strip, collapse whitespace
        normalized = ' '.join(query.lower().strip().split())
        return normalized
    
    def _get_cache_key(self, query: str) -> str:
        """
        Generate cache key from query using SHA-256 hash.
        
        Args:
            query: Query string
            
        Returns:
            SHA-256 hash of normalized query
        """
        normalized = self._normalize_query(query)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def _load_cache(self) -> None:
        """Load cache from file if it exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                print(f"📦 Loaded {len(self._cache)} cached queries from {self.cache_file}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  Could not load cache file: {e}")
                self._cache = {}
        else:
            print(f"📦 Cache file not found, starting with empty cache")
            self._cache = {}
    
    def _save_cache(self) -> None:
        """Save cache to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"⚠️  Could not save cache file: {e}")
    
    def get(self, query: str) -> Optional[str]:
        """
        Get cached response for a query.
        
        Args:
            query: User query
            
        Returns:
            Cached response if found, None otherwise
        """
        cache_key = self._get_cache_key(query)
        
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            print(f"✅ Cache HIT for query: {query[:60]}...")
            print(f"   Cached on: {entry.get('timestamp', 'unknown')}")
            print(f"   Cache key: {cache_key[:16]}...")
            return entry['response']
        
        print(f"❌ Cache MISS for query: {query[:60]}...")
        return None
    
    def set(self, query: str, response: str) -> None:
        """
        Store query-response pair in cache.
        
        Args:
            query: User query
            response: System response
        """
        cache_key = self._get_cache_key(query)
        normalized_query = self._normalize_query(query)
        
        self._cache[cache_key] = {
            'query': query,  # Original query for reference
            'normalized_query': normalized_query,
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
        
        self._save_cache()
        print(f"💾 Cached response for query: {query[:60]}...")
        print(f"   Cache key: {cache_key[:16]}...")
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache = {}
        self._save_cache()
        print(f"🗑️  Cache cleared")
    
    def size(self) -> int:
        """Get number of cached entries."""
        return len(self._cache)
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            'total_entries': len(self._cache),
            'cache_file': self.cache_file,
            'file_exists': os.path.exists(self.cache_file)
        }
