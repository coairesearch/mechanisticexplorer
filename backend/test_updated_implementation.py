#!/usr/bin/env python
"""Test the updated implementation with HuggingFace transformers"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from app.logit_lens import LogitLensExtractor, ConversationTokenizer

def test_updated_implementation():
    """Test the updated implementation"""
    print("Testing updated implementation...")
    
    try:
        # Initialize
        print("Initializing extractor...")
        extractor = LogitLensExtractor("openai-community/gpt2")
        tracker = ConversationTokenizer(extractor.tokenizer)
        
        print("✅ Initialization successful")
        
        # Test conversation tracking
        user_msg = "What is the capital of France?"
        assistant_msg = "The capital of France is Paris."
        
        print("Adding messages to tracker...")
        user_tokens = tracker.add_message(user_msg, is_user=True)
        assistant_tokens = tracker.add_message(assistant_msg, is_user=False)
        
        print(f"✅ User tokens: {len(user_tokens)}")
        print(f"✅ Assistant tokens: {len(assistant_tokens)}")
        
        # Test logit lens extraction
        print("Extracting logit lens for full conversation...")
        full_text = tracker.get_full_text()
        print(f"Full text: '{full_text}'")
        
        activations = extractor.extract_activations(full_text, top_k=3)
        print(f"✅ Extracted activations for {len(activations)} tokens")
        
        # Test formatting for last token
        last_token_pos = len(tracker.global_tokens) - 1
        predictions = extractor.format_layer_predictions(activations, last_token_pos)
        
        if predictions:
            print(f"✅ Formatted predictions for last token across {len(predictions)} layers")
            
            # Show predictions for first and last layer
            first_layer = predictions[0]
            last_layer = predictions[-1]
            
            print(f"First layer (0) predictions: {[p.token for p in first_layer.predictions]}")
            print(f"Last layer ({len(predictions)-1}) predictions: {[p.token for p in last_layer.predictions]}")
            
            # Show context window
            context = tracker.get_context_window(last_token_pos, size=5)
            print(f"✅ Context window: {len(context)} tokens")
            for token in context:
                marker = "→" if token['is_target'] else " "
                print(f"  {marker} {token['global_position']}: '{token['text']}' ({'user' if token['is_user'] else 'assistant'})")
            
            return True
        else:
            print("❌ No predictions formatted")
            return False
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_text_generation():
    """Test text generation"""
    print("\nTesting text generation...")
    
    try:
        extractor = LogitLensExtractor("openai-community/gpt2")
        
        prompt = "The capital of France is"
        
        # Tokenize input
        inputs = extractor.tokenizer(prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = extractor.model.generate(
                inputs["input_ids"],
                max_new_tokens=10,
                do_sample=True,
                temperature=0.7,
                pad_token_id=extractor.tokenizer.eos_token_id
            )
        
        # Decode the output
        full_text = extractor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text[len(prompt):].strip()
        
        print(f"✅ Generated response: '{response}'")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Updated Implementation")
    print("=" * 50)
    
    tests = [
        test_updated_implementation,
        test_text_generation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 30)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Implementation is working.")
    else:
        print("❌ Some tests failed. Check implementation.")