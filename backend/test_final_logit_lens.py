#!/usr/bin/env python
"""Test final working logit lens implementation for local GPT-2"""

import torch
from nnsight import LanguageModel

def test_final_approach():
    """Test the final working approach - process everything within trace context"""
    print("Testing final approach for local GPT-2...")
    
    # Initialize model
    torch.set_grad_enabled(False)
    model = LanguageModel("openai-community/gpt2", device_map="auto")
    
    prompt = "The capital of France is"
    layers = model.transformer.h
    
    try:
        # Process everything within the trace context
        probs_layers = []
        
        with model.trace(prompt) as tracer:
            for layer_idx, layer in enumerate(layers):
                # Process layer output through the model's head and layer normalization
                layer_output = model.lm_head(model.transformer.ln_f(layer.output[0]))
                
                # Apply softmax to obtain probabilities and save the result
                probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
                probs_layers.append(probs)
        
        print(f"✅ Traced and saved {len(probs_layers)} layers")
        
        # Now access the values after trace context is complete
        actual_probs = []
        for layer_idx, probs in enumerate(probs_layers):
            print(f"Layer {layer_idx}: shape={probs.value.shape}, is_meta={probs.value.is_meta}")
            if not probs.value.is_meta:
                actual_probs.append(probs.value)
            else:
                print(f"  ❌ Layer {layer_idx} is still meta")
        
        if actual_probs:
            print(f"✅ Successfully extracted {len(actual_probs)} layers")
            
            # Show predictions for last layer
            last_layer_probs = actual_probs[-1][0]  # [seq_len, vocab_size]
            input_ids = model.tokenizer.encode(prompt)
            input_words = [model.tokenizer.decode([t]) for t in input_ids]
            
            print(f"Input words: {input_words}")
            print(f"Last layer probs shape: {last_layer_probs.shape}")
            
            for token_idx, word in enumerate(input_words):
                if token_idx < last_layer_probs.shape[0]:
                    token_probs = last_layer_probs[token_idx]
                    top_probs, top_indices = torch.topk(token_probs, 3)
                    
                    top_words = [model.tokenizer.decode([idx.item()]) for idx in top_indices]
                    top_probs_list = [prob.item() * 100 for prob in top_probs]
                    
                    print(f"Token {token_idx} ('{word}'): {list(zip(top_words, top_probs_list))}")
            
            return True
        else:
            print("❌ No layers extracted successfully")
            return False
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_huggingface():
    """Test direct HuggingFace approach without nnsight"""
    print("\nTesting direct HuggingFace approach...")
    
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        # Load model and tokenizer directly
        model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        
        prompt = "The capital of France is"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        print(f"Input shape: {input_ids.shape}")
        
        # Forward pass to get hidden states
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of (num_layers+1) tensors
        
        print(f"✅ Got {len(hidden_states)} hidden state tensors")
        
        # Process last few layers
        for layer_idx in [-3, -2, -1]:  # Last 3 layers
            layer_output = hidden_states[layer_idx]
            
            # Apply layer norm and head
            normed = model.transformer.ln_f(layer_output)
            logits = model.lm_head(normed)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            print(f"Layer {layer_idx} probs shape: {probs.shape}")
            
            if layer_idx == -1:  # Last layer
                input_words = [tokenizer.decode([t]) for t in input_ids[0]]
                
                print(f"Input words: {input_words}")
                
                for token_idx, word in enumerate(input_words):
                    token_probs = probs[0, token_idx]
                    top_probs, top_indices = torch.topk(token_probs, 3)
                    
                    top_words = [tokenizer.decode([idx.item()]) for idx in top_indices]
                    top_probs_list = [prob.item() * 100 for prob in top_probs]
                    
                    print(f"Token {token_idx} ('{word}'): {list(zip(top_words, top_probs_list))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Final Logit Lens Implementation")
    print("=" * 50)
    
    tests = [
        test_final_approach,
        test_direct_huggingface
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 30)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed > 0:
        print("🎉 At least one approach works! Can proceed with implementation.")