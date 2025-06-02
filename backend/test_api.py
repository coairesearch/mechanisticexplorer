#!/usr/bin/env python
"""Test script for the backend API"""

import requests
import json
import time

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8000/api/health")
        print(f"Health check status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Connection error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    try:
        response = requests.get("http://localhost:8000/api/model_info")
        print(f"\nModel info status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Connection error: {e}")
        return False

def test_chat():
    """Test the chat endpoint"""
    try:
        payload = {
            "text": "Hello, how are you?",
            "messages": []
        }
        response = requests.post(
            "http://localhost:8000/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"\nChat test status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response text: {data.get('text', 'No text')}")
            print(f"Number of tokens: {len(data.get('tokens', []))}")
            print(f"Number of user tokens: {len(data.get('userTokens', []))}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Connection error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Backend API...")
    print("=" * 50)
    
    # Wait a moment for server to be ready
    time.sleep(2)
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_health():
        tests_passed += 1
    
    if test_model_info():
        tests_passed += 1
    
    if test_chat():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")