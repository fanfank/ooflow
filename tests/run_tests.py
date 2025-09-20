#!/usr/bin/env python3
"""
One-click run OoFlow test suite
Author: fanfank@github
Date: 2025-09-19

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py --verbose          # Verbose output
    python tests/run_tests.py --quick            # Run quick tests only
    python tests/run_tests.py --integration      # Run integration tests only
"""

import asyncio
import sys
import os
import argparse
import time

# Add project root directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import test modules
from test_ooflow import run_all_tests


def print_banner():
    """Print test banner"""
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 16 + "OoFlow Test Suite" + " " * 25 + "║")
    print("║" + " " * 10 + "Complete Unit and Integration Tests" + " " * 13 + "║")
    print("╚" + "═" * 58 + "╝")
    print()


def print_help():
    """Print help information"""
    print("OoFlow test suite contains the following test modules:")
    print()
    print("📋 Test Modules:")
    print("  • TestLogger          - Logger setup and configuration")
    print("  • TestContext         - Context class methods")
    print("  • TestNode            - Node decorator functionality")
    print("  • TestEdge            - Edge class and queue mechanism")
    print("  • TestOoFlow          - OoFlow core functionality")
    print("  • TestOoFlowIntegration - Complete workflow scenarios")
    print("  • TestCreateFunction  - create function")
    print()
    print("🧪 Test Coverage:")
    print("  • Parameter validation and error handling")
    print("  • Asynchronous message passing")
    print("  • Graph construction and topology validation")
    print("  • Concurrent execution and synchronization")
    print("  • Boundary conditions and exception cases")
    print()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="OoFlow test suite runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                # Run all tests
  python run_tests.py -v             # Verbose output
  python run_tests.py --help-tests   # Show test module information
        """
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose test output"
    )
    
    parser.add_argument(
        "--help-tests",
        action="store_true",
        help="Show detailed test module information"
    )
    
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Don't show banner"
    )
    
    parser.add_argument(
        "--time",
        action="store_true",
        help="Show test execution time"
    )
    
    return parser.parse_args()


async def main():
    """Main function"""
    args = parse_args()
    
    if args.help_tests:
        print_help()
        return
    
    if not args.no_banner:
        print_banner()
    
    if args.verbose:
        print("🚀 Starting verbose mode tests...")
        print()
    
    # Record start time
    start_time = time.time() if args.time else None
    
    try:
        # Run tests
        success = await run_all_tests()
        
        # Show execution time
        if args.time:
            elapsed = time.time() - start_time
            print(f"\n⏱️  Total execution time: {elapsed:.2f} seconds")
        
        # Show final result
        if success:
            print("\n🎉 All tests passed! OoFlow is working properly.")
            return 0
        else:
            print("\n❌ Some tests failed, please check the error messages above.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Test runner error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTests interrupted")
        sys.exit(130)
