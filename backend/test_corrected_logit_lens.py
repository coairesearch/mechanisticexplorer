#!/usr/bin/env python
"""Test corrected logit lens implementation for local GPT-2"""

import torch
from nnsight import LanguageModel

def test_corrected_approach():
    """Test the corrected approach for local GPT-2"""
    print("Testing corrected approach for local GPT-2...")
    
    # Initialize model
    torch.set_grad_enabled(False)
    model = LanguageModel("openai-community/gpt2", device_map="auto")
    
    prompt = "The capital of France is"
    layers = model.transformer.h  # Use transformer.h for GPT-2
    probs_layers = []
    
    try:
        # Use the tracer.invoke pattern like in playground but for local model
        with model.trace(prompt) as tracer:
            with tracer.invoke(prompt) as invoker:
                for layer_idx, layer in enumerate(layers):
                    # Process layer output through the model's head and layer normalization
                    layer_output = model.lm_head(model.transformer.ln_f(layer.output[0]))
                    
                    # Apply softmax to obtain probabilities and save the result
                    probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
                    probs_layers.append(probs)
        
        # Extract the actual values
        probs_tensor = torch.cat([probs.value for probs in probs_layers])
        print(f"Probabilities shape: {probs_tensor.shape}")
        
        # Find the maximum probability and corresponding tokens for each position
        max_probs, tokens = probs_tensor.max(dim=-1)
        
        print(f"Max probs shape: {max_probs.shape}")
        print(f"Tokens shape: {tokens.shape}")
        
        # Get input tokens from invoker
        input_ids = invoker.inputs[0][0]["input_ids"][0]
        input_words = [model.tokenizer.decode([t]) for t in input_ids]
        
        print(f"Input words: {input_words}")
        print(f"Number of layers: {len(probs_layers)}")
        print(f"Number of tokens: {len(input_words)}")
        
        # Show predictions for each token at the last layer
        last_layer_probs = probs_layers[-1].value[0]  # [seq_len, vocab_size]
        
        for token_idx, word in enumerate(input_words):
            token_probs = last_layer_probs[token_idx]
            top_probs, top_indices = torch.topk(token_probs, 3)
            
            top_words = [model.tokenizer.decode([idx.item()]) for idx in top_indices]
            top_probs_list = [prob.item() * 100 for prob in top_probs]
            
            print(f"Token {token_idx} ('{word}'): {list(zip(top_words, top_probs_list))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_approach():
    """Test the simplest possible approach"""
    print("\nTesting simplest approach...")
    
    try:
        # Initialize model
        torch.set_grad_enabled(False)
        model = LanguageModel("openai-community/gpt2", device_map="auto")
        
        prompt = "The capital of France is"
        
        # Try the simplest possible extraction
        with model.trace(prompt) as tracer:
            # Just get the last layer output
            last_layer = model.transformer.h[-1]
            layer_output = model.lm_head(model.transformer.ln_f(last_layer.output[0]))
            probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
        
        # Check if we get real data
        print(f"Probs shape: {probs.value.shape}")
        print(f"Probs type: {type(probs.value)}")
        print(f"Is meta tensor: {probs.value.is_meta}")
        
        if not probs.value.is_meta:
            # Get top predictions for first token
            token_probs = probs.value[0, 0, :]  # First token
            top_probs, top_indices = torch.topk(token_probs, 3)
            
            top_words = [model.tokenizer.decode([idx.item()]) for idx in top_indices]
            top_probs_list = [prob.item() * 100 for prob in top_probs]
            
            print(f"✅ Top predictions for first token: {list(zip(top_words, top_probs_list))}")
            return True
        else:
            print("❌ Still getting meta tensors")
            return False
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Corrected Logit Lens Implementation")
    print("=" * 50)
    
    tests = [
        test_simple_approach,
        test_corrected_approach
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 30)
    
    print(f"\nResults: {passed}/{total} tests passed")