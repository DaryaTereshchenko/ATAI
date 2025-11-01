#!/usr/bin/env python3
"""
✅ OPTIMIZED: Test factual query processing with memory management.
"""

import sys
import os
import gc
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# ✅ Configure logging BEFORE imports
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"test_factual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
print(f"📄 Logging to: {log_file}")

# Redirect stdout/stderr to file
sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

try:
    from src.main.orchestrator import Orchestrator
    
    print("="*80)
    print("FACTUAL QUERY TEST")
    print("="*80)
    
    # ✅ Initialize orchestrator once
    print("\n🔧 Initializing orchestrator...")
    orchestrator = Orchestrator(use_workflow=False)
    
    # ✅ Test cases (reduced set for faster execution)
    test_cases = [
        "Who directed The Matrix?",
        "From what country is 'Aro Tolbukhin. En la mente del asesino'?",
        "What language is spoken in 'Life Is Beautiful'?",
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_cases)}: {query}")
        print(f"{'='*80}")
        
        try:
            response = orchestrator._process_factual(query)
            print(f"\n✅ RESPONSE:\n{response}")
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        # ✅ Force garbage collection between tests
        gc.collect()
        print(f"\n🧹 Memory cleaned (test {i} complete)")
    
    print(f"\n{'='*80}")
    print("ALL TESTS COMPLETED")
    print(f"{'='*80}")

except Exception as e:
    print(f"\n❌ FATAL ERROR: {e}")
    import traceback
    traceback.print_exc()

finally:
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f"✅ Test complete. Check log: {log_file}")
