#!/usr/bin/env python3
"""
APERA Backend Diagnostic Tool
Tests all aspects of the backend and identifies specific failure points
"""

import requests
import json
import sys
import time
from typing import Dict, Any, Tuple

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.END}")

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 30  # seconds

def test_connection() -> bool:
    """Test if backend is reachable"""
    print_header("TEST 1: Backend Connectivity")
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        
        if response.status_code == 200:
            print_success(f"Backend is reachable at {BASE_URL}")
            data = response.json()
            print_info(f"Service: {data.get('service', 'Unknown')}")
            print_info(f"Version: {data.get('version', 'Unknown')}")
            print_info(f"Status: {data.get('status', 'Unknown')}")
            return True
        else:
            print_error(f"Backend returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to backend")
        print_info("Make sure the backend is running:")
        print_info("  python apera_backend_production.py")
        return False
    except requests.exceptions.Timeout:
        print_error("Connection timed out")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {type(e).__name__}: {str(e)}")
        return False

def test_health_check() -> bool:
    """Test health check endpoint"""
    print_header("TEST 2: Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success("Health check passed")
            
            # Check status
            status = data.get('status', 'unknown')
            if status == 'healthy':
                print_success(f"Status: {status}")
            else:
                print_warning(f"Status: {status}")
            
            # Check individual components
            checks = data.get('checks', {})
            print_info("\nComponent Status:")
            for component, status in checks.items():
                if status:
                    print_success(f"  {component}: OK")
                else:
                    print_error(f"  {component}: FAILED")
            
            return data.get('status') == 'healthy'
        else:
            print_error(f"Health check failed with status: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Health check error: {type(e).__name__}: {str(e)}")
        return False

def test_chat_endpoint() -> Tuple[bool, Dict[str, Any]]:
    """Test chat endpoint with a simple query"""
    print_header("TEST 3: Chat Endpoint (Live Research Mode)")
    
    payload = {
        "query": "machine learning",
        "session_id": "diagnostic_test",
        "mode": "Live Research (ArXiv)"
    }
    
    print_info(f"Sending request with query: '{payload['query']}'")
    print_info("This may take 10-30 seconds as it searches ArXiv...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        
        print_info(f"Response Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print_success("Chat endpoint returned 200 OK")
            
            try:
                data = response.json()
                
                # Validate response structure
                print_info("\nValidating response structure...")
                
                if "response" in data:
                    print_success("✓ 'response' field present")
                    print_info(f"  Length: {len(data['response'])} characters")
                else:
                    print_error("✗ 'response' field missing")
                    return False, {}
                
                if "citations" in data:
                    print_success(f"✓ 'citations' field present ({len(data['citations'])} items)")
                    
                    # Check citation structure
                    if len(data['citations']) > 0:
                        print_info("\nValidating first citation...")
                        first_citation = data['citations'][0]
                        
                        required_fields = ['file', 'text', 'url', 'type']
                        for field in required_fields:
                            if field in first_citation:
                                print_success(f"  ✓ '{field}' field present")
                                if field == 'file':
                                    print_info(f"    Value: {first_citation[field][:50]}...")
                            else:
                                print_error(f"  ✗ '{field}' field MISSING")
                                print_error("  THIS IS THE PROBLEM!")
                                return False, data
                else:
                    print_warning("✗ 'citations' field missing (may be empty)")
                
                if "meta" in data:
                    print_success("✓ 'meta' field present")
                    meta = data['meta']
                    print_info(f"  Intent: {meta.get('intent', 'N/A')}")
                    print_info(f"  Confidence: {meta.get('confidence', 'N/A')}")
                else:
                    print_warning("✗ 'meta' field missing")
                
                print_success("\n✅ All validation checks passed!")
                return True, data
                
            except json.JSONDecodeError as e:
                print_error(f"Failed to parse JSON response: {e}")
                print_info(f"Raw response: {response.text[:200]}...")
                return False, {}
                
        elif response.status_code == 500:
            print_error("Server returned 500 Internal Server Error")
            print_info("Response content:")
            try:
                error_data = response.json()
                print_error(json.dumps(error_data, indent=2))
            except:
                print_error(response.text)
            return False, {}
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            print_info(f"Response: {response.text[:200]}...")
            return False, {}
            
    except requests.exceptions.Timeout:
        print_error("Request timed out after 30 seconds")
        print_info("ArXiv might be slow or unreachable")
        return False, {}
    except requests.exceptions.ConnectionError:
        print_error("Connection error - backend might have crashed")
        print_info("Check the backend console for error messages")
        return False, {}
    except Exception as e:
        print_error(f"Unexpected error: {type(e).__name__}: {str(e)}")
        import traceback
        print_info(traceback.format_exc())
        return False, {}

def test_local_mode() -> bool:
    """Test chat endpoint in local mode"""
    print_header("TEST 4: Chat Endpoint (Local Mode)")
    
    payload = {
        "query": "test query",
        "session_id": "diagnostic_test",
        "mode": "local"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            print_success("Local mode endpoint works")
            data = response.json()
            print_info(f"Response: {data['response'][:100]}...")
            return True
        else:
            print_error(f"Failed with status: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error: {type(e).__name__}: {str(e)}")
        return False

def test_feedback_endpoint() -> bool:
    """Test feedback endpoint"""
    print_header("TEST 5: Feedback Endpoint")
    
    payload = {
        "query": "test query",
        "rating": "positive"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/feedback",
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            print_success("Feedback endpoint works")
            return True
        else:
            print_error(f"Failed with status: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error: {type(e).__name__}: {str(e)}")
        return False

def check_backend_logs():
    """Check if backend logs exist and show last few lines"""
    print_header("TEST 6: Backend Logs")
    
    import os
    log_file = "logs/apera_backend.log"
    
    if os.path.exists(log_file):
        print_success(f"Log file found: {log_file}")
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            print_info(f"Total log lines: {len(lines)}")
            print_info("\nLast 10 log entries:")
            print("-" * 80)
            for line in lines[-10:]:
                print(line.rstrip())
            print("-" * 80)
            
        except Exception as e:
            print_error(f"Could not read log file: {e}")
    else:
        print_warning(f"Log file not found: {log_file}")
        print_info("Backend may not have been started yet")

def main():
    """Run all diagnostic tests"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                  APERA BACKEND DIAGNOSTIC TOOL                             ║")
    print("║                  Comprehensive System Testing                              ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}\n")
    
    print_info(f"Testing backend at: {BASE_URL}")
    print_info(f"Timeout: {TIMEOUT} seconds")
    print_info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Run tests
    results['connection'] = test_connection()
    
    if not results['connection']:
        print_error("\n❌ Backend is not accessible. Cannot continue testing.")
        print_info("\nTroubleshooting steps:")
        print_info("1. Make sure you started the backend:")
        print_info("   python apera_backend_production.py")
        print_info("2. Check if port 8000 is available:")
        print_info("   lsof -i :8000")
        print_info("3. Check firewall settings")
        return 1
    
    results['health'] = test_health_check()
    results['chat'], chat_data = test_chat_endpoint()
    results['local'] = test_local_mode()
    results['feedback'] = test_feedback_endpoint()
    
    check_backend_logs()
    
    # Summary
    print_header("TEST SUMMARY")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test_name, passed_test in results.items():
        status = f"{Colors.GREEN}PASS{Colors.END}" if passed_test else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  {test_name.upper():.<40} {status}")
    
    print(f"\n{Colors.BOLD}Overall: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{'='*80}")
        print("✅ ALL TESTS PASSED - Backend is working correctly!")
        print(f"{'='*80}{Colors.END}\n")
        return 0
    else:
        print(f"\n{Colors.RED}{'='*80}")
        print("❌ SOME TESTS FAILED - Check the output above for details")
        print(f"{'='*80}{Colors.END}\n")
        
        if not results.get('chat'):
            print_error("\n🔍 DIAGNOSIS:")
            print_error("The chat endpoint is failing. This is likely causing your 500 errors.")
            print_info("\nNext steps:")
            print_info("1. Check backend console for detailed error messages")
            print_info("2. Look at logs/apera_backend.log for stack traces")
            print_info("3. Verify ArXiv is accessible:")
            print_info("   python -c 'import arxiv; print(\"ArXiv OK\")'")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
