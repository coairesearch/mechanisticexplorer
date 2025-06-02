#!/usr/bin/env python
"""Test working logit lens implementation for local GPT-2"""

import torch
from nnsight import LanguageModel

def test_working_approach():
    """Test a working approach for local GPT-2 based on nnsight docs"""
    print("Testing working approach for local GPT-2...")
    
    # Initialize model
    torch.set_grad_enabled(False)
    model = LanguageModel("openai-community/gpt2", device_map="auto")
    
    prompt = "The capital of France is"
    layers = model.transformer.h  # Use transformer.h for GPT-2
    
    try:
        # For local models, don't use nested invoke
        with model.trace(prompt) as tracer:
            # Save intermediate outputs at each layer
            saved_outputs = []
            for layer_idx, layer in enumerate(layers):
                # Get the output of this layer
                layer_out = layer.output[0].save()
                saved_outputs.append(layer_out)
        
        print(f"✅ Traced {len(saved_outputs)} layers")
        
        # Now process the saved outputs
        probs_layers = []
        for layer_idx, layer_out in enumerate(saved_outputs):
            print(f"Processing layer {layer_idx}")
            print(f"  Layer output shape: {layer_out.value.shape}")
            print(f"  Is meta: {layer_out.value.is_meta}")
            
            if not layer_out.value.is_meta:
                # Apply layer norm and model head
                normed = model.transformer.ln_f(layer_out.value)
                logits = model.lm_head(normed)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                probs_layers.append(probs)
            else:
                print(f"  ❌ Layer {layer_idx} output is meta tensor")
        
        if probs_layers:
            print(f"✅ Successfully processed {len(probs_layers)} layers")
            
            # Show predictions for last layer
            last_layer_probs = probs_layers[-1][0]  # [seq_len, vocab_size]
            input_ids = model.tokenizer.encode(prompt)
            input_words = [model.tokenizer.decode([t]) for t in input_ids]
            
            print(f"Input words: {input_words}")
            
            for token_idx, word in enumerate(input_words):
                if token_idx < last_layer_probs.shape[0]:
                    token_probs = last_layer_probs[token_idx]
                    top_probs, top_indices = torch.topk(token_probs, 3)
                    
                    top_words = [model.tokenizer.decode([idx.item()]) for idx in top_indices]
                    top_probs_list = [prob.item() * 100 for prob in top_probs]
                    
                    print(f"Token {token_idx} ('{word}'): {list(zip(top_words, top_probs_list))}")
            
            return True
        else:
            print("❌ No layers processed successfully")
            return False
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_forward():
    """Test direct forward pass without nnsight tracing"""
    print("\nTesting direct forward pass...")
    
    try:
        # Initialize model
        torch.set_grad_enabled(False)
        model = LanguageModel("openai-community/gpt2", device_map="auto")
        
        prompt = "The capital of France is"
        
        # Tokenize input
        inputs = model.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        print(f"Input shape: {input_ids.shape}")
        
        # Get the actual model (not nnsight wrapper)
        hf_model = model.model
        
        # Forward pass to get hidden states
        with torch.no_grad():
            outputs = hf_model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of (num_layers+1) tensors
        
        print(f"✅ Got {len(hidden_states)} hidden state tensors")
        
        # Process each layer
        for layer_idx in range(len(hidden_states) - 1):  # Skip input embeddings
            layer_output = hidden_states[layer_idx + 1]  # +1 to skip embeddings
            
            # Apply layer norm and head
            normed = hf_model.transformer.ln_f(layer_output)
            logits = hf_model.lm_head(normed)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            if layer_idx == len(hidden_states) - 2:  # Last layer
                print(f"Last layer probs shape: {probs.shape}")
                
                input_words = [model.tokenizer.decode([t]) for t in input_ids[0]]
                
                for token_idx, word in enumerate(input_words):
                    token_probs = probs[0, token_idx]
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
    print("=" * 50)
    print("Testing Working Logit Lens Implementation")
    print("=" * 50)
    
    tests = [
        test_working_approach,
        test_direct_forward
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 30)
    
    print(f"\nResults: {passed}/{total} tests passed")