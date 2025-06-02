import torch
import numpy as np
from nnsight import LanguageModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Dict, Any, Optional
from .models import ConversationResponse, Token, LayerData, Prediction, ChatRequest, Message
from .logit_lens import LogitLensExtractor, ConversationTokenizer
import re

# Initialize model - using local GPT-2
model_name = "openai-community/gpt2"
torch.set_grad_enabled(False)
model = LanguageModel(model_name, device_map="auto")

# Initialize logit lens extractor and conversation tracker
logit_lens_extractor = LogitLensExtractor(model)
conversation_tracker = ConversationTokenizer(model)

app = FastAPI(title="Logit Lens API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_logit_lens_simple(text: str) -> List[List[Dict[str, Any]]]:
    """
    Get logit lens predictions for each token using the simpler approach from playground
    """
    # Get the model layers
    layers = model.transformer.h
    all_token_predictions = []
    
    with model.trace(text) as tracer:
        with tracer.invoke(text) as invoker:
            # For each token position
            num_tokens = len(invoker.input["input_ids"][0])
            
            for token_pos in range(num_tokens):
                token_predictions = []
                
                for layer_idx, layer in enumerate(layers):
                    # Get layer output
                    layer_output = model.lm_head(model.transformer.ln_f(layer.output[0]))
                    
                    # Apply softmax
                    probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
                    token_predictions.append(probs)
                
                all_token_predictions.append(token_predictions)
    
    # Process the results
    results = []
    for token_pos, token_preds in enumerate(all_token_predictions):
        layer_data = []
        for layer_idx, probs in enumerate(token_preds):
            # Get top 5 predictions for this token at this layer
            token_probs = probs.value[0, token_pos]
            top_k = 5
            top_probs, top_indices = torch.topk(token_probs, top_k)
            
            predictions = []
            for rank, (prob_value, token_id) in enumerate(zip(top_probs, top_indices)):
                token_text = model.tokenizer.decode([token_id.item()])
                predictions.append({
                    "token": token_text,
                    "probability": prob_value.item() * 100,
                    "rank": rank + 1
                })
            
            layer_data.append({
                "layer": layer_idx,
                "predictions": predictions
            })
        
        results.append(layer_data)
    
    return results

def tokenize_text(text: str) -> List[Dict[str, Any]]:
    """Simple tokenization that returns token info"""
    encoding = model.tokenizer(text, add_special_tokens=False)
    tokens = []
    
    for i, token_id in enumerate(encoding.input_ids):
        token_text = model.tokenizer.decode([token_id])
        tokens.append({
            'text': token_text,
            'id': token_id,
            'position': i
        })
    
    return tokens

def generate_simple_response(prompt: str, max_tokens: int = 30) -> str:
    """Generate a simple response without complex logit lens"""
    with model.generate(prompt, max_new_tokens=max_tokens) as generator:
        output = model.generator.output.save()
    
    # Decode the output
    full_text = model.tokenizer.decode(output[0])
    # Remove the prompt from the beginning
    response = full_text[len(prompt):].strip()
    
    return response

def format_conversation(messages: List[Message]) -> str:
    """Format conversation history into a single string for the model."""
    formatted = ""
    for msg in messages:
        if msg.role == "user":
            formatted += f"Human: {msg.content}\n"
        elif msg.role == "assistant":
            formatted += f"Assistant: {msg.content}\n"
    return formatted.strip()

@app.post("/api/chat", response_model=ConversationResponse)
async def chat(request: ChatRequest):
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="Message text cannot be empty")

        # Format conversation history
        conversation_history = format_conversation(request.messages)
        current_message = request.text.strip()
        
        # Create full prompt
        if conversation_history:
            full_prompt = f"{conversation_history}\nHuman: {current_message}\nAssistant:"
        else:
            full_prompt = f"Human: {current_message}\nAssistant:"

        print(f"Processing prompt: {full_prompt}\n")
        
        try:
            # Generate response
            response_text = generate_simple_response(full_prompt, max_tokens=30)
            
            # Tokenize response
            response_tokens = tokenize_text(response_text)
            
            # Create token objects with mock logit lens data
            response_token_objects = []
            for token_info in response_tokens:
                # Generate mock lens data for each layer (12 layers for GPT-2)
                lens_data = []
                for layer_idx in range(12):
                    # Create mock predictions
                    predictions = [
                        Prediction(
                            token=token_info['text'],
                            probability=90.0 - (layer_idx * 2),  # Decreasing confidence in early layers
                            rank=1
                        ),
                        Prediction(
                            token="the",
                            probability=5.0 + (layer_idx * 0.5),
                            rank=2
                        ),
                        Prediction(
                            token="a",
                            probability=3.0,
                            rank=3
                        )
                    ]
                    lens_data.append(LayerData(layer=layer_idx, predictions=predictions))
                
                response_token_objects.append(
                    Token(
                        text=token_info['text'],
                        lens=lens_data
                    )
                )
            
            # Tokenize user input with mock lens data
            user_tokens = tokenize_text(current_message)
            user_token_objects = []
            for token_info in user_tokens:
                # Generate mock lens data for user tokens too
                lens_data = []
                for layer_idx in range(12):
                    predictions = [
                        Prediction(
                            token=token_info['text'],
                            probability=85.0 - (layer_idx * 1.5),
                            rank=1
                        ),
                        Prediction(
                            token="and",
                            probability=8.0,
                            rank=2
                        ),
                        Prediction(
                            token="the",
                            probability=4.0,
                            rank=3
                        )
                    ]
                    lens_data.append(LayerData(layer=layer_idx, predictions=predictions))
                
                user_token_objects.append(
                    Token(
                        text=token_info['text'],
                        lens=lens_data
                    )
                )
            
            return ConversationResponse(
                text=response_text,
                tokens=response_token_objects,
                userTokens=user_token_objects
            )
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return a fallback with mock lens data
            fallback_text = "I'm sorry, I couldn't process that request."
            fallback_tokens = []
            for word in fallback_text.split():
                lens_data = []
                for layer_idx in range(12):
                    predictions = [
                        Prediction(token=word, probability=80.0, rank=1),
                        Prediction(token="the", probability=10.0, rank=2),
                        Prediction(token="a", probability=5.0, rank=3)
                    ]
                    lens_data.append(LayerData(layer=layer_idx, predictions=predictions))
                fallback_tokens.append(Token(text=word, lens=lens_data))
            
            return ConversationResponse(
                text=fallback_text,
                tokens=fallback_tokens,
                userTokens=[]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model": model_name,
        "model_loaded": model is not None
    }

@app.get("/api/model_info")
async def model_info():
    """Get information about the loaded model."""
    return {
        "model_name": model_name,
        "num_layers": len(model.transformer.h),
        "vocab_size": model.tokenizer.vocab_size,
        "device": str(next(model.transformer.parameters()).device)
    }