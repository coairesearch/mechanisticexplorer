#!/usr/bin/env python
"""Debug the model generation issue"""

import torch
from nnsight import LanguageModel

# Test basic model functionality
print("Testing GPT-2 model...")
torch.set_grad_enabled(False)
model = LanguageModel("openai-community/gpt2", device_map="auto")

# Test 1: Basic generation
print("\nTest 1: Basic generation")
try:
    prompt = "Hello"
    with model.generate(prompt, max_new_tokens=5) as generator:
        output = model.generator.output.save()
    
    result = model.tokenizer.decode(output[0])
    print(f"Prompt: {prompt}")
    print(f"Generated: {result}")
    print("✅ Basic generation works!")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {str(e)}")

# Test 2: Trace functionality
print("\nTest 2: Trace functionality")
try:
    with model.trace("Hello") as tracer:
        hidden = model.transformer.h[0].output[0].save()
    print(f"Hidden shape: {hidden.value.shape}")
    print("✅ Trace works!")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {str(e)}")

# Test 3: Token extraction
print("\nTest 3: Tokenization")
try:
    text = "Hello world"
    encoding = model.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    print(f"Tokens: {encoding.input_ids}")
    print(f"Decoded: {[model.tokenizer.decode([t]) for t in encoding.input_ids]}")
    print("✅ Tokenization works!")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {str(e)}")