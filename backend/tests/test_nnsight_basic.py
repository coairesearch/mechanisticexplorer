#!/usr/bin/env python
"""Test nnsight with corrected access to input tokens"""

import torch
from nnsight import LanguageModel

def test_nnsight_fixed():
    """Test nnsight with corrected token access"""
    print("Testing nnsight with fixed token access...")
    
    try:
        model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
        
        prompt = "The Eiffel Tower is in the city of"
        layers = model.transformer.h
        probs_layers = []

        with model.trace() as tracer:
            with tracer.invoke(prompt) as invoker:
                for layer_idx, layer in enumerate(layers):
                    # Process layer output through the model's head and layer normalization
                    layer_output = model.lm_head(model.transformer.ln_f(layer.output[0]))

                    # Apply softmax to obtain probabilities and save the result
                    probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
                    probs_layers.append(probs)

        probs = torch.cat([probs.value for probs in probs_layers])

        # Find the maximum probability and corresponding tokens for each position
        max_probs, tokens = probs.max(dim=-1)

        # Decode token IDs to words for each layer
        words = [[model.tokenizer.decode(t.cpu()).encode("unicode_escape").decode() for t in layer_tokens]
            for layer_tokens in tokens]

        # CORRECTED: Access the input token IDs properly
        input_token_ids = invoker.inputs[0][0].ids  # Access .ids attribute
        input_words = [model.tokenizer.decode([t]) for t in input_token_ids]
        
        print(f"✅ Success! Got results for {len(input_words)} input tokens")
        print(f"✅ Processed {len(probs_layers)} layers")
        print(f"✅ Probs shape: {probs.shape}")
        print(f"✅ Input token IDs: {input_token_ids}")
        print(f"✅ Input words: {input_words}")
        
        # Show predictions for each input token at the last layer
        print("\nPredictions at last layer:")
        last_layer_probs = probs_layers[-1].value[0]  # [seq_len, vocab_size]
        
        for token_idx, word in enumerate(input_words):
            token_probs = last_layer_probs[token_idx]
            top_probs, top_indices = torch.topk(token_probs, 3)
            
            top_words = [model.tokenizer.decode([idx.item()]) for idx in top_indices]
            top_probs_list = [prob.item() * 100 for prob in top_probs]
            
            print(f"  Token {token_idx} ('{word}'): {list(zip(top_words, top_probs_list))}")
        
        # Test API-compatible extraction
        print("\nTesting API-compatible extraction...")
        token_activations = {}
        num_tokens = len(input_words)
        
        for token_pos in range(num_tokens):
            token_data = {}
            
            for layer_idx, probs in enumerate(probs_layers):
                # Get probabilities for this token at this layer
                token_probs = probs.value[0, token_pos, :]
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(token_probs, 5)
                
                # Store layer data
                token_data[layer_idx] = {
                    'top_k_probs': top_probs.detach().cpu(),
                    'top_k_indices': top_indices.detach().cpu()
                }
            
            token_activations[token_pos] = token_data
        
        print(f"✅ Extracted activations for {len(token_activations)} tokens")
        
        # Test first token formatting
        token_pos = 0
        print(f"\nToken {token_pos} ('{input_words[token_pos]}') layer predictions:")
        token_data = token_activations[token_pos]
        
        # Show first and last layer
        for layer_idx in [0, len(probs_layers) - 1]:
            layer_data = token_data[layer_idx]
            probs = layer_data['top_k_probs']
            indices = layer_data['top_k_indices']
            words = [model.tokenizer.decode([idx.item()]) for idx in indices]
            percentages = [prob.item() * 100 for prob in probs]
            
            print(f"  Layer {layer_idx}: {list(zip(words, percentages))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Testing NNsight with Fixed Token Access")
    print("=" * 50)
    
    success = test_nnsight_fixed()
    
    if success:
        print("\n🎉 Test passed! NNsight is working correctly.")
        print("✅ Ready to update the LogitLensExtractor to use nnsight properly.")
    else:
        print("\n❌ Test failed.")