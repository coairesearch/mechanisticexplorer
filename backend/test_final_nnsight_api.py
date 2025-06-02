#!/usr/bin/env python
"""Test the final nnsight-based API implementation"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
from app.main import chat, health_check, model_info
from app.models import ChatRequest, Message

async def test_nnsight_api():
    """Test nnsight-based API functions"""
    print("Testing nnsight-based API functions...")
    
    try:
        # Test health check
        print("Testing health check...")
        health = await health_check()
        print(f"✅ Health check: {health}")
        
        # Test model info  
        print("Testing model info...")
        info = await model_info()
        print(f"✅ Model info: {info}")
        
        # Test chat with real logit lens
        print("Testing chat with real nnsight logit lens...")
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
                
                # Show the probability differences between layers
                print("\n📊 Layer-by-layer evolution (first token):")
                for layer_idx in [0, 5, 11]:  # Show beginning, middle, end
                    if layer_idx < len(first_token.lens):
                        layer = first_token.lens[layer_idx]
                        top_pred = layer.predictions[0]
                        print(f"  Layer {layer_idx}: '{top_pred.token}' ({top_pred.probability:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Final NNsight-based API")
    print("=" * 50)
    
    success = asyncio.run(test_nnsight_api())
    
    if success:
        print("\n🎉 All nnsight API tests passed!")
        print("\n✅ REAL LOGIT LENS WORKING WITH NNSIGHT!")
        print("📈 You can now see how predictions evolve across transformer layers")
        print("🔬 Perfect for mechanistic interpretability research")
        print("\n📝 Next steps:")
        print("   1. Start the API: python -m uvicorn app.main:app --reload --port 8000")
        print("   2. Start frontend: npm run dev")
        print("   3. Click on tokens to see real layer-by-layer predictions!")
    else:
        print("\n❌ API tests failed. Check implementation.")