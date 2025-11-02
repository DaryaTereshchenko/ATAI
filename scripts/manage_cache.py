"""
Utility script for managing the query cache.
Usage: python scripts/manage_cache.py [stats|clear|list]
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main.query_cache import QueryCache


def main():
    cache = QueryCache()
    
    if len(sys.argv) < 2:
        command = 'stats'
    else:
        command = sys.argv[1]
    
    if command == 'stats':
        stats = cache.get_stats()
        print("\n📊 Cache Statistics:")
        print(f"   Total entries: {stats['total_entries']}")
        print(f"   Cache file: {stats['cache_file']}")
        print(f"   File exists: {stats['file_exists']}\n")
    
    elif command == 'clear':
        print("\n🗑️  Clearing cache...")
        cache.clear()
        print("✅ Cache cleared\n")
    
    elif command == 'list':
        print(f"\n📋 Cached Queries ({cache.size()} total):\n")
        for key, entry in cache._cache.items():
            print(f"Key: {key[:16]}...")
            print(f"  Query: {entry['query'][:80]}...")
            print(f"  Timestamp: {entry['timestamp']}")
            print(f"  Response length: {len(entry['response'])} chars\n")
    
    else:
        print(f"Unknown command: {command}")
        print("Usage: python scripts/manage_cache.py [stats|clear|list]")


if __name__ == '__main__':
    main()
