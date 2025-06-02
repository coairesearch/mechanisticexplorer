#!/usr/bin/env python
"""Test the real logit lens implementation"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from nnsight import LanguageModel
from app.logit_lens import LogitLensExtractor, ConversationTokenizer

def test_basic_extraction():
    """Test basic logit lens extraction"""
    print("Testing basic logit lens extraction...")
    
    # Initialize model
    torch.set_grad_enabled(False)
    model = LanguageModel("openai-community/gpt2", device_map="auto")
    
    # Initialize extractor
    extractor = LogitLensExtractor(model)
    
    # Test text
    text = "The capital of France is"
    
    print(f"Extracting activations for: '{text}'")
    
    try:
        # Extract activations
        activations = extractor.extract_activations(text, top_k=3)
        
        print(f"✅ Successfully extracted activations for {len(activations)} tokens")
        
        # Test formatting for each token
        for token_pos in activations.keys():
            layer_data = extractor.format_layer_predictions(activations, token_pos)
            
            token_text = model.tokenizer.decode([model.tokenizer.encode(text)[token_pos]])
            print(f"\nToken {token_pos} ('{token_text}'):")
            print(f"  Layers: {len(layer_data)}")
            
            if layer_data:
                # Show predictions for first and last layer
                first_layer = layer_data[0]
                last_layer = layer_data[-1]
                
                print(f"  Layer 0 predictions: {[p.token for p in first_layer.predictions[:3]]}")
                print(f"  Layer {len(layer_data)-1} predictions: {[p.token for p in last_layer.predictions[:3]]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_conversation_tracking():
    """Test conversation tokenizer"""
    print("\nTesting conversation tracking...")
    
    # Initialize model and tokenizer
    torch.set_grad_enabled(False)
    model = LanguageModel("openai-community/gpt2", device_map="auto")
    
    tracker = ConversationTokenizer(model)
    
    try:
        # Add messages
        user_msg = "What is the capital of France?"
        assistant_msg = "The capital of France is Paris."
        
        user_tokens = tracker.add_message(user_msg, is_user=True)
        assistant_tokens = tracker.add_message(assistant_msg, is_user=False)
        
        print(f"✅ User tokens: {len(user_tokens)}")
        print(f"✅ Assistant tokens: {len(assistant_tokens)}")
        
        # Test context window
        total_tokens = len(user_tokens) + len(assistant_tokens)
        last_token_pos = total_tokens - 1
        
        context = tracker.get_context_window(last_token_pos, size=5)
        print(f"✅ Context window: {len(context)} tokens")
        
        # Show context
        print("Context tokens:")
        for token in context:
            marker = "→" if token['is_target'] else " "
            print(f"  {marker} {token['global_position']}: '{token['text']}' ({'user' if token['is_user'] else 'assistant'})")
        
        # Test full text
        full_text = tracker.get_full_text()
        print(f"✅ Full text: '{full_text}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test full integration"""
    print("\nTesting full integration...")
    
    try:
        # Initialize
        torch.set_grad_enabled(False)
        model = LanguageModel("openai-community/gpt2", device_map="auto")
        
        extractor = LogitLensExtractor(model)
        tracker = ConversationTokenizer(model)
        
        # Simulate conversation
        user_msg = "Hello"
        assistant_msg = "Hi there!"
        
        tracker.add_message(user_msg, is_user=True)
        tracker.add_message(assistant_msg, is_user=False)
        
        # Extract activations for full conversation
        full_text = tracker.get_full_text()
        activations = extractor.extract_activations(full_text, top_k=3)
        
        print(f"✅ Extracted activations for conversation: {len(activations)} tokens")
        
        # Test getting predictions for a specific token
        target_pos = len(tracker.global_tokens) - 1  # Last token
        predictions = extractor.format_layer_predictions(activations, target_pos)
        
        if predictions:
            print(f"✅ Last token predictions across {len(predictions)} layers")
            last_layer = predictions[-1]
            print(f"  Final layer top prediction: '{last_layer.predictions[0].token}' ({last_layer.predictions[0].probability:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Real Logit Lens Implementation")
    print("=" * 50)
    
    tests = [
        test_basic_extraction,
        test_conversation_tracking,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 30)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for frontend testing.")
    else:
        print("❌ Some tests failed. Check implementation.")