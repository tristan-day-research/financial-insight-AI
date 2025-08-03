#!/usr/bin/env python3
"""
Test runner script for Financial Insight AI project.
Provides easy commands to run different types of tests.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Command not found. Make sure pytest is installed.")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run tests for Financial Insight AI")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "all", "coverage", "performance"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--parallel", "-n",
        type=int,
        help="Number of parallel processes (requires pytest-xdist)"
    )
    parser.add_argument(
        "--file",
        help="Run tests from specific file"
    )
    parser.add_argument(
        "--class",
        help="Run tests from specific class"
    )
    parser.add_argument(
        "--method",
        help="Run specific test method"
    )
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Add parallel processing
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    # Add specific test selection
    if args.file:
        cmd.append(args.file)
    elif args.class:
        cmd.append(f"knowledge_base/tests/{args.class}")
    elif args.method:
        cmd.append(f"knowledge_base/tests/{args.method}")
    else:
        # Add test type markers
        if args.type == "unit":
            cmd.extend(["-m", "unit"])
        elif args.type == "integration":
            cmd.extend(["-m", "integration"])
        elif args.type == "performance":
            cmd.extend(["-m", "performance"])
        elif args.type == "coverage":
            cmd.extend([
                "--cov=knowledge_base",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-fail-under=70"
            ])
    
    # Run the command
    success = run_command(cmd, f"{args.type.title()} tests")
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 