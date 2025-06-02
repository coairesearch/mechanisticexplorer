#!/usr/bin/env python
"""Debug nnsight invoker inputs structure"""

import torch
from nnsight import LanguageModel

def debug_invoker_structure():
    """Debug the structure of invoker.inputs"""
    print("Debugging invoker inputs structure...")
    
    try:
        model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
        
        prompt = "The Eiffel Tower is in the city of"
        
        with model.trace() as tracer:
            with tracer.invoke(prompt) as invoker:
                # Let's inspect the invoker structure
                print(f"invoker type: {type(invoker)}")
                print(f"invoker.inputs type: {type(invoker.inputs)}")
                print(f"invoker.inputs length: {len(invoker.inputs) if hasattr(invoker.inputs, '__len__') else 'No length'}")
                
                if hasattr(invoker.inputs, '__len__') and len(invoker.inputs) > 0:
                    print(f"invoker.inputs[0] type: {type(invoker.inputs[0])}")
                    
                    if hasattr(invoker.inputs[0], '__len__') and len(invoker.inputs[0]) > 0:
                        print(f"invoker.inputs[0][0] type: {type(invoker.inputs[0][0])}")
                        print(f"invoker.inputs[0][0]: {invoker.inputs[0][0]}")
                        
                        # If it's an Encoding object, let's see what's available
                        encoding = invoker.inputs[0][0]
                        print(f"Encoding attributes: {dir(encoding)}")
                        
                        if hasattr(encoding, 'input_ids'):
                            print(f"encoding.input_ids: {encoding.input_ids}")
                            print(f"encoding.input_ids type: {type(encoding.input_ids)}")
                
                # Let's also check if we can get tokens directly from the model
                print(f"\nDirect tokenization:")
                tokens = model.tokenizer.encode(prompt)
                print(f"Direct tokens: {tokens}")
                words = [model.tokenizer.decode([t]) for t in tokens]
                print(f"Direct words: {words}")
                
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_invoker_structure()