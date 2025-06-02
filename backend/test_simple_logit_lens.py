#!/usr/bin/env python
"""Test simple logit lens implementation directly from playground"""

import torch
from nnsight import LanguageModel

def test_playground_approach():
    """Test the exact approach from playground"""
    print("Testing playground approach...")
    
    # Initialize model
    torch.set_grad_enabled(False)
    model = LanguageModel("openai-community/gpt2", device_map="auto")
    
    prompt = "The capital of France is"
    layers = model.transformer.h  # Use transformer.h for GPT-2
    probs_layers = []
    
    try:
        with model.trace(prompt) as tracer:
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
        
        # Get input tokens
        input_ids = model.tokenizer.encode(prompt)
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

if __name__ == "__main__":
    test_playground_approach()