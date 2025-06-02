#!/usr/bin/env python
"""Test the backend API directly to debug the issue"""

import requests
import json

def test_simple_chat():
    """Test a simple chat request"""
    url = "http://localhost:8000/api/chat"
    
    payload = {
        "text": "Hello",
        "messages": []
    }
    
    print("Sending request to:", url)
    print("Payload:", json.dumps(payload, indent=2))
    
    try:
        response = requests.post(url, json=payload)
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print("\nSuccess! Response:")
            print(f"Text: {data.get('text', 'No text')}")
            print(f"Tokens: {len(data.get('tokens', []))} tokens")
            print(f"User Tokens: {len(data.get('userTokens', []))} user tokens")
        else:
            print(f"\nError Response: {response.text}")
            
    except Exception as e:
        print(f"\nException: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    test_simple_chat()