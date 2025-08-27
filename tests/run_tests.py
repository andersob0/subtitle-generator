#!/usr/bin/env python3
"""
Test Suite Runner for Subtitle Generator
Runs all essential tests in the correct order.
"""

import sys
import os
import unittest
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def run_all_tests():
    """Run all tests in the test suite"""
    print("🧪 SUBTITLE GENERATOR TEST SUITE")
    print("=" * 50)
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = str(Path(__file__).parent)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print(f"\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n❌ Some tests failed")
    
    return success

def run_specific_test(test_name):
    """Run a specific test file"""
    print(f"🧪 Running {test_name}")
    print("=" * 50)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_name)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run subtitle generator tests")
    parser.add_argument("--test", help="Run specific test (e.g., test_integration)")
    parser.add_argument("--list", action="store_true", help="List available tests")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available tests:")
        test_files = [f.stem for f in Path(__file__).parent.glob("test_*.py")]
        for test in test_files:
            print(f"  - {test}")
    elif args.test:
        success = run_specific_test(args.test)
        sys.exit(0 if success else 1)
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)
