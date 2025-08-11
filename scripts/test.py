#!/usr/bin/env python3
"""
Development script for running tests
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status"""
    print(f"Running {description}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {description} passed")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"✗ {description} failed")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ {description} failed with error: {e}")
        return False


def main():
    """Main test script"""
    print("🧪 Running tests...")
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    
    commands = [
        (["uv", "run", "pytest", "backend/tests/", "-v"], "Unit tests (pytest)"),
    ]
    
    success_count = 0
    total_count = len(commands)
    
    for cmd, description in commands:
        if run_command(cmd, description):
            success_count += 1
    
    print(f"\n📊 Testing complete: {success_count}/{total_count} test suites passed")
    
    if success_count == total_count:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())