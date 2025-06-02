#!/usr/bin/env python
"""Test nnsight with the correct pattern from the official tutorial"""

import torch
from nnsight import LanguageModel

def test_nnsight_official_pattern():
    """Test using the exact pattern from the official nnsight logit lens tutorial"""
    print("Testing nnsight with official pattern...")
    
    try:
        # Load model exactly like in the tutorial
        model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
        
        prompt = "The Eiffel Tower is in the city of"
        layers = model.transformer.h
        probs_layers = []

        # Use the exact pattern from the tutorial
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

        # Access the 'input_ids' attribute of the invoker object to get the input words
        input_words = [model.tokenizer.decode(t) for t in invoker.inputs[0][0]["input_ids"][0]]
        
        print(f"✅ Success! Got results for {len(input_words)} input tokens")
        print(f"✅ Processed {len(probs_layers)} layers")
        print(f"✅ Probs shape: {probs.shape}")
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
        
        return True, model, probs_layers, input_words, invoker
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, None, None, None

def test_nnsight_for_api_usage(model, probs_layers, input_words, invoker):
    """Test extracting data in the format needed for our API"""
    print("\nTesting API-compatible data extraction...")
    
    try:
        # Extract activations in the format our API needs
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
        
        # Test formatting for API response
        for token_pos in range(min(2, num_tokens)):  # Test first 2 tokens
            print(f"\nToken {token_pos} ('{input_words[token_pos]}'):")
            token_data = token_activations[token_pos]
            
            # Show first and last layer predictions
            first_layer = token_data[0]
            last_layer = token_data[len(probs_layers) - 1]
            
            # Format first layer
            first_probs = first_layer['top_k_probs']
            first_indices = first_layer['top_k_indices']
            first_words = [model.tokenizer.decode([idx.item()]) for idx in first_indices]
            first_percentages = [prob.item() * 100 for prob in first_probs]
            
            print(f"  Layer 0: {list(zip(first_words, first_percentages))}")
            
            # Format last layer
            last_probs = last_layer['top_k_probs']
            last_indices = last_layer['top_k_indices']
            last_words = [model.tokenizer.decode([idx.item()]) for idx in last_indices]
            last_percentages = [prob.item() * 100 for prob in last_probs]
            
            print(f"  Layer 11: {list(zip(last_words, last_percentages))}")
        
        return True, token_activations
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    print("=" * 50)
    print("Testing NNsight with Official Pattern")
    print("=" * 50)
    
    # Test the official pattern
    success, model, probs_layers, input_words, invoker = test_nnsight_official_pattern()
    
    if success:
        # Test API-compatible extraction
        api_success, token_activations = test_nnsight_for_api_usage(model, probs_layers, input_words, invoker)
        
        if api_success:
            print("\n🎉 Both tests passed! NNsight is working correctly.")
            print("✅ Ready to update the LogitLensExtractor to use this pattern.")
        else:
            print("\n❌ API extraction failed.")
    else:
        print("\n❌ Official pattern test failed.")