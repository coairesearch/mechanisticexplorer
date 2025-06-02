#!/usr/bin/env python
"""
Test runner for the mechanistic explorer backend.
Run all tests to verify the implementation is working correctly.
"""

import sys
import os
import importlib.util

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_test_file(test_file_path):
    """Run a single test file and return success status"""
    print(f"\n{'='*50}")
    print(f"Running {os.path.basename(test_file_path)}")
    print(f"{'='*50}")
    
    try:
        # Load and run the test module
        spec = importlib.util.spec_from_file_location("test_module", test_file_path)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        return True
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define test files in order of importance
    test_files = [
        "test_nnsight_basic.py",           # Basic nnsight functionality
        "test_logit_lens.py",              # LogitLensExtractor functionality
        "test_api_functions.py",           # API function testing
        "test_api_integration.py",         # Full integration test
        "test_caching_system.py",          # Caching system tests
        "test_cached_api_integration.py",  # Cached API integration tests
    ]
    
    print("🔬 Running Mechanistic Explorer Backend Tests")
    print("=" * 60)
    
    passed = 0
    total = len(test_files)
    
    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)
        if os.path.exists(test_path):
            if run_test_file(test_path):
                passed += 1
        else:
            print(f"❌ Test file not found: {test_file}")
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The implementation is working correctly.")
        print("✅ Ready for production use.")
    else:
        print("❌ Some tests failed. Check the implementation.")
        sys.exit(1)

if __name__ == "__main__":
    main()