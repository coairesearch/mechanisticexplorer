#!/usr/bin/env python
"""Test nnsight trace functionality"""

import torch
from nnsight import LanguageModel

torch.set_grad_enabled(False)
model = LanguageModel("openai-community/gpt2", device_map="auto")

prompt = "Hello world"

# Test 1: Check tracer.input structure
print("Test 1: Checking tracer.input structure")
with model.trace(prompt) as tracer:
    print(f"Tracer type: {type(tracer)}")
    print(f"Tracer input: {tracer.input}")
    print(f"Input keys: {tracer.input.keys() if hasattr(tracer.input, 'keys') else 'No keys method'}")
    
    # Try different ways to access input_ids
    try:
        print(f"tracer.input.input_ids: {tracer.input.input_ids}")
    except Exception as e:
        print(f"Error accessing tracer.input.input_ids: {e}")
    
    try:
        print(f"tracer.inputs: {tracer.inputs}")
    except Exception as e:
        print(f"Error accessing tracer.inputs: {e}")

# Test 2: Correct way to do logit lens
print("\nTest 2: Correct logit lens approach")
with model.trace(prompt) as tracer:
    # Access hidden states through the model's forward pass
    hidden_states = []
    
    # Get hidden state after each layer
    for i, layer in enumerate(model.transformer.h):
        hidden = layer.output[0].save()
        hidden_states.append(hidden)
    
    # Get final hidden state
    final_hidden = model.transformer.ln_f.output.save()

print("Collected hidden states from layers")

# Test 3: Alternative approach
print("\nTest 3: Using invoker")
with model.trace() as tracer:
    with tracer.invoke(prompt) as invoker:
        # This gives us access to the actual inputs
        print(f"Invoker input_ids shape: {invoker.input['input_ids'].shape}")