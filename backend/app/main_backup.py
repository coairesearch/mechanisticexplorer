import torch
import numpy as np
from nnsight import LanguageModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Dict, Any, Optional
from .models import ConversationResponse, Token, LayerData, Prediction, ChatRequest, Message
import re

# Initialize model - using local GPT-2
model_name = "openai-community/gpt2"
torch.set_grad_enabled(False)
model = LanguageModel(model_name, device_map="auto")

app = FastAPI(title="Logit Lens API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_logit_lens_predictions(prompt: str, token_position: int = -1) -> List[LayerData]:
    """
    Extract real logit lens predictions from GPT-2 model for a specific token position.
    
    Args:
        prompt: Input text to analyze
        token_position: Which token position to analyze (-1 for last token)
    
    Returns:
        List of LayerData with predictions from each layer
    """
    layers = model.transformer.h  # GPT-2 uses 'h' for transformer blocks
    probs_layers = []
    
    # Tokenize outside of trace context
    input_ids = model.tokenizer.encode(prompt, return_tensors="pt")
    
    with model.trace(prompt) as tracer:
        # Get embeddings from the traced input
        embeddings = model.transformer.wte(tracer.input.input_ids)
        position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long).unsqueeze(0)
        position_embeds = model.transformer.wpe(position_ids)
        hidden_state = embeddings + position_embeds
        
        for layer_idx, layer in enumerate(layers):
            # Pass through layer
            layer_output = layer(hidden_state)
            hidden_state = layer_output[0]
            
            # Apply layer norm and language model head
            normalized = model.transformer.ln_f(hidden_state)
            logits = model.lm_head(normalized)
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1).save()
            probs_layers.append(probs)
    
    # Extract predictions for the specified token position
    layer_predictions = []
    
    for layer_idx, probs in enumerate(probs_layers):
        # Get probabilities for the specific token position
        token_probs = probs.value[0, token_position]
        
        # Get top 5 predictions
        top_k = 5
        top_probs, top_indices = torch.topk(token_probs, top_k)
        
        predictions = []
        for rank, (prob_value, token_id) in enumerate(zip(top_probs, top_indices)):
            token_text = model.tokenizer.decode([token_id.item()])
            predictions.append(
                Prediction(
                    token=token_text,
                    probability=prob_value.item() * 100,
                    rank=rank + 1
                )
            )
        
        layer_predictions.append(
            LayerData(
                layer=layer_idx,
                predictions=predictions
            )
        )
    
    return layer_predictions

def tokenize_with_positions(text: str) -> List[Dict[str, Any]]:
    """
    Tokenize text and return tokens with their positions.
    """
    # Use GPT-2 tokenizer
    encoding = model.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = []
    
    for i, (token_id, (start, end)) in enumerate(zip(encoding.input_ids, encoding.offset_mapping)):
        token_text = text[start:end]
        tokens.append({
            'text': token_text,
            'id': token_id,
            'position': i,
            'start': start,
            'end': end
        })
    
    return tokens

def generate_response_with_logits(prompt: str, max_tokens: int = 50) -> Dict[str, Any]:
    """
    Generate response and extract logit lens data for each generated token.
    """
    # Generate response
    with model.generate(prompt, max_new_tokens=max_tokens) as generator:
        output = model.generator.output.save()
    
    # Decode full output
    full_output = model.tokenizer.decode(output[0])
    
    # Separate prompt from generated text
    generated_text = full_output[len(prompt):].strip()
    
    # Tokenize the generated text
    generated_tokens = tokenize_with_positions(generated_text)
    
    # For each generated token, get logit lens predictions
    tokens_with_lens = []
    current_context = prompt
    
    for token_info in generated_tokens:
        # Get logit lens for this token given the context
        lens_data = extract_logit_lens_predictions(current_context, token_position=-1)
        
        tokens_with_lens.append(
            Token(
                text=token_info['text'],
                lens=lens_data
            )
        )
        
        # Update context for next token
        current_context += token_info['text']
    
    return {
        'text': generated_text,
        'tokens': tokens_with_lens
    }

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
            # Generate response with logit lens data
            response_data = generate_response_with_logits(full_prompt, max_tokens=30)
            
            # Process user tokens
            user_tokens = []
            user_token_data = tokenize_with_positions(current_message)
            
            for token_info in user_token_data:
                # For user tokens, we analyze them in the context of the conversation
                context_for_token = conversation_history + "\nHuman: " + current_message[:token_info['end']]
                lens_data = extract_logit_lens_predictions(context_for_token, token_position=-1)
                
                user_tokens.append(
                    Token(
                        text=token_info['text'],
                        lens=lens_data
                    )
                )
            
            return ConversationResponse(
                text=response_data['text'],
                tokens=response_data['tokens'],
                userTokens=user_tokens
            )
            
        except Exception as model_error:
            print(f"Model error: {str(model_error)}")
            import traceback
            traceback.print_exc()
            # Fallback to simple response without logit lens
            simple_response = "I apologize, but I'm having trouble processing your request. Please try again."
            
            # Create simple tokens without lens data
            simple_tokens = [
                Token(text=word, lens=[])
                for word in simple_response.split()
            ]
            
            return ConversationResponse(
                text=simple_response,
                tokens=simple_tokens,
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