#!/usr/bin/env python
"""Test the corrected nnsight-based LogitLensExtractor"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from app.logit_lens import LogitLensExtractor, ConversationTokenizer

def test_nnsight_implementation():
    """Test the nnsight-based implementation"""
    print("Testing nnsight-based LogitLensExtractor...")
    
    try:
        # Initialize
        print("Initializing extractor...")
        extractor = LogitLensExtractor("openai-community/gpt2")
        tracker = ConversationTokenizer(extractor.model)
        
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
            
            # Show probability changes across layers for the last token
            last_token_text = tracker.global_tokens[last_token_pos]['text']
            print(f"\nLayer-by-layer predictions for last token '{last_token_text}':")
            
            for layer_idx in [0, 5, 11]:  # Show beginning, middle, end
                if layer_idx < len(predictions):
                    layer = predictions[layer_idx]
                    top_pred = layer.predictions[0]
                    print(f"  Layer {layer_idx}: '{top_pred.token}' ({top_pred.probability:.1f}%)")
            
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

def test_nnsight_text_generation():
    """Test text generation with nnsight"""
    print("\nTesting text generation with nnsight...")
    
    try:
        extractor = LogitLensExtractor("openai-community/gpt2")
        
        prompt = "The capital of France is"
        
        # Use nnsight's generation capabilities
        with extractor.model.generate(prompt, max_new_tokens=10) as generator:
            output = extractor.model.generator.output.save()
        
        # Decode the output
        full_text = extractor.model.tokenizer.decode(output[0])
        response = full_text[len(prompt):].strip()
        
        print(f"✅ Generated response: '{response}'")
        
        # Test logit lens on the generated text
        print("Testing logit lens on generated text...")
        activations = extractor.extract_activations(full_text, top_k=3)
        print(f"✅ Extracted activations for {len(activations)} tokens in generated text")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Testing NNsight-based Implementation")
    print("=" * 50)
    
    tests = [
        test_nnsight_implementation,
        test_nnsight_text_generation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 30)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! NNsight implementation is working.")
        print("✅ Ready to replace the HuggingFace version with nnsight.")
    else:
        print("❌ Some tests failed. Check implementation.")