#!/usr/bin/env python3
"""
Comprehensive test runner for TranscriMatic
Provides convenient commands for running different test suites with enhanced features
"""

import sys
import subprocess
import argparse
import time
import os
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], capture_output: bool = False) -> int:
    """Run a command and return the exit code"""
    print(f"🔧 Running: {' '.join(cmd)}")
    
    if capture_output:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode
    else:
        return subprocess.call(cmd)


def cleanup_test_outputs():
    """Clean up test output directories"""
    from test_utils import cleanup_test_outputs
    print("🧹 Cleaning test outputs...")
    cleanup_test_outputs()
    print("✅ Test outputs cleaned")


def check_dependencies():
    """Check if required test dependencies are available"""
    required_packages = ['pytest', 'pytest-cov', 'pytest-xdist']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"⚠️  Missing test dependencies: {', '.join(missing)}")
        print("Install with: pip install " + ' '.join(missing))
        return False
    
    return True


def show_test_summary(suite: str, exit_code: int, duration: float):
    """Show test execution summary"""
    status = "✅ PASSED" if exit_code == 0 else "❌ FAILED"
    print(f"\n{'='*60}")
    print(f"Test Suite: {suite.upper()}")
    print(f"Status: {status}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Exit Code: {exit_code}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="TranscriMatic Comprehensive Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py unit --verbose     # Run unit tests with verbose output
  python run_tests.py integration -x     # Run integration tests, stop on first failure
  python run_tests.py performance        # Run performance benchmarks
  python run_tests.py coverage --html    # Generate HTML coverage report
  python run_tests.py --clean            # Clean test outputs only
  python run_tests.py --check-deps       # Check test dependencies
        """
    )
    
    # Test suite selection
    parser.add_argument(
        "suite",
        nargs="?",
        default="all",
        choices=["all", "unit", "integration", "performance", "coverage", "main", "output"],
        help="Test suite to run (default: all)"
    )
    
    # Test execution options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--failfast", "-x",
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "--parallel", "-n",
        type=int,
        metavar="WORKERS",
        help="Run tests in parallel with N workers"
    )
    parser.add_argument(
        "--pattern", "-k",
        type=str,
        help="Run tests matching pattern"
    )
    
    # Coverage options
    parser.add_argument(
        "--nocov",
        action="store_true",
        help="Disable coverage reporting"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report"
    )
    parser.add_argument(
        "--xml",
        action="store_true",
        help="Generate XML coverage report"
    )
    
    # Utility options
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean test outputs and exit"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check test dependencies and exit"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tests without running"
    )
    
    args = parser.parse_args()
    
    # Handle utility commands
    if args.clean:
        cleanup_test_outputs()
        return 0
    
    if args.check_deps:
        if check_dependencies():
            print("✅ All test dependencies are available")
            return 0
        else:
            return 1
    
    # Check dependencies before running tests
    if not check_dependencies():
        return 1
    
    # Clean test outputs before running
    cleanup_test_outputs()
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test discovery and output options
    cmd.extend([
        "--tb=short",  # Shorter traceback format
        "--strict-markers",  # Require all markers to be defined
        "--strict-config",  # Require valid configuration
    ])
    
    # Add verbosity
    if args.verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")
    
    # Add failfast
    if args.failfast:
        cmd.append("-x")
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    # Add pattern matching
    if args.pattern:
        cmd.extend(["-k", args.pattern])
    
    # Add listing option
    if args.list:
        cmd.append("--collect-only")
    
    # Coverage configuration
    if not args.nocov and args.suite != "performance":
        cmd.extend([
            "--cov=src",
            "--cov-branch",
            "--cov-report=term-missing"
        ])
        
        if args.html or args.suite == "coverage":
            cmd.append("--cov-report=html")
        
        if args.xml:
            cmd.append("--cov-report=xml")
    
    # Select test suite
    if args.suite == "unit":
        cmd.extend(["-m", "unit or not integration and not performance"])
    elif args.suite == "integration":
        cmd.extend(["-m", "integration"])
    elif args.suite == "performance":
        cmd.extend(["-m", "performance", "--benchmark-only"])
    elif args.suite == "main":
        cmd.append("tests/test_main_controller.py")
    elif args.suite == "output":
        cmd.append("tests/test_output_formatter.py")
    elif args.suite == "coverage":
        # Run all tests with detailed coverage
        cmd.extend([
            "--cov-report=html",
            "--cov-report=xml",
            "--cov-report=term"
        ])
    # For "all", run all tests (no additional filters)
    
    # Record start time
    start_time = time.time()
    
    # Run the tests
    exit_code = run_command(cmd)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Show summary
    show_test_summary(args.suite, exit_code, duration)
    
    # Show coverage report locations if generated
    if not args.nocov and not args.list:
        coverage_html = Path("htmlcov/index.html")
        coverage_xml = Path("coverage.xml")
        
        if coverage_html.exists():
            print(f"📊 HTML Coverage Report: {coverage_html.absolute()}")
        
        if coverage_xml.exists():
            print(f"📊 XML Coverage Report: {coverage_xml.absolute()}")
    
    # Show test outputs location
    test_outputs = Path("tests/test_outputs")
    if test_outputs.exists():
        print(f"📁 Test Outputs: {test_outputs.absolute()}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())