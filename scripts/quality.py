#!/usr/bin/env python3
"""
Development script for running all code quality checks
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_name: str, description: str) -> bool:
    """Run a quality script and return success status"""
    print(f"\n{'='*50}")
    print(f"Running {description}")
    print('='*50)
    
    try:
        script_path = Path(__file__).parent / f"{script_name}.py"
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"✗ Failed to run {description}: {e}")
        return False


def main():
    """Main quality check script"""
    print("🚀 Running comprehensive code quality checks...")
    
    # List of quality scripts to run in order
    quality_scripts = [
        ("format", "Code Formatting"),
        ("lint", "Code Linting"),
    ]
    
    success_count = 0
    total_count = len(quality_scripts)
    
    for script_name, description in quality_scripts:
        if run_script(script_name, description):
            success_count += 1
        else:
            print(f"❌ {description} failed")
    
    print(f"\n{'='*50}")
    print(f"📊 Quality Check Summary: {success_count}/{total_count} checks passed")
    print('='*50)
    
    if success_count == total_count:
        print("🎉 All quality checks passed! Code is ready for commit.")
        return 0
    else:
        print("❌ Some quality checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())