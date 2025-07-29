#!/usr/bin/env python3
"""
Quick Demo and Testing Script for GitHub
=======================================

This script demonstrates all the key features of the backtesting engine
and creates visual assets for GitHub.
"""

import sys
import subprocess
from pathlib import Path
import time

def run_command(cmd, description):
    """Run a command and show results."""
    print(f"\n🔄 {description}")
    print("-" * 50)
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ SUCCESS")
            # Show last few lines of output
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    print(f"   {line}")
        else:
            print("❌ FAILED")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}...")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT (>60s)")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Run comprehensive demo."""
    print("🚀 BACKTESTING ENGINE - QUICK DEMO")
    print("=" * 60)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test commands
    tests = [
        ("python validate_framework.py", "Framework Validation"),
        ("python -m pytest test_suite.py -x", "Core Test Suite"),
        ("python examples/simple_strategy.py", "Example Strategy"),
    ]
    
    results = []
    for cmd, desc in tests:
        success = run_command(cmd, desc)
        results.append((desc, success))
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DEMO SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for desc, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {desc}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - FRAMEWORK IS DEMO READY!")
        print("\n📋 Next Steps for GitHub:")
        print("   1. Record this demo script running")
        print("   2. Screenshot the generated HTML reports")
        print("   3. Capture notebook execution")
        print("   4. Create 30-60 second highlight video")
        
        # Check for generated reports
        results_dir = Path('results')
        if results_dir.exists():
            html_files = list(results_dir.glob('*.html'))
            if html_files:
                latest_report = max(html_files, key=lambda p: p.stat().st_mtime)
                print(f"\n📄 Latest report: {latest_report}")
                print(f"🌐 Open in browser: file://{latest_report.absolute()}")
        
        # Check for web demo
        web_demo = Path('web_demo/index.html')
        if web_demo.exists():
            print(f"🌐 Web demo available: {web_demo.absolute()}")
            print("   Run: cd web_demo && python run_demo.py")
        
    else:
        print(f"\n⚠️  {total-passed} tests failed - check output above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
