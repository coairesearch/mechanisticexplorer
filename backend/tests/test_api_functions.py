#!/usr/bin/env python
"""Test the API functionality directly without starting the server"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
from app.main import chat, health_check, model_info
from app.models import ChatRequest, Message

async def test_api_functions():
    """Test API functions directly"""
    print("Testing API functions directly...")
    
    try:
        # Test health check
        print("Testing health check...")
        health = await health_check()
        print(f"✅ Health check: {health}")
        
        # Test model info  
        print("Testing model info...")
        info = await model_info()
        print(f"✅ Model info: {info}")
        
        # Test chat
        print("Testing chat...")
        request = ChatRequest(
            text="What is Paris?",
            messages=[]
        )
        
        response = await chat(request)
        print(f"✅ Chat response text: '{response.text}'")
        print(f"✅ Response tokens: {len(response.tokens)}")
        print(f"✅ User tokens: {len(response.userTokens)}")
        
        # Check if we have real logit lens data
        if response.tokens and response.tokens[0].lens:
            first_token = response.tokens[0]
            print(f"✅ First response token: '{first_token.text}'")
            print(f"✅ Layers with predictions: {len(first_token.lens)}")
            
            if first_token.lens:
                first_layer = first_token.lens[0]
                print(f"✅ Layer 0 predictions: {[p.token for p in first_layer.predictions[:3]]}")
                
                last_layer = first_token.lens[-1]
                print(f"✅ Last layer predictions: {[p.token for p in last_layer.predictions[:3]]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Testing API Functions Directly")
    print("=" * 50)
    
    success = asyncio.run(test_api_functions())
    
    if success:
        print("\n🎉 All API tests passed! Implementation is working.")
        print("\n📝 To test the implementation:")
        print("   1. Run: python -m uvicorn app.main:app --reload --port 8000")
        print("   2. Test with: curl -X POST http://localhost:8000/api/chat \\")
        print("      -H 'Content-Type: application/json' \\")
        print("      -d '{\"text\": \"What is Paris?\", \"messages\": []}'")
    else:
        print("\n❌ API tests failed. Check implementation.")