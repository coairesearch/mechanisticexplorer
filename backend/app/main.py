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

# Initialize model - using local GPT-2 with nnsight
model_name = "openai-community/gpt2"
torch.set_grad_enabled(False)

# Initialize logit lens extractor and conversation tracker
logit_lens_extractor = LogitLensExtractor(model_name)
conversation_tracker = ConversationTokenizer(logit_lens_extractor.model)

app = FastAPI(title="Logit Lens API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_simple_response(prompt: str, max_tokens: int = 30) -> str:
    """Generate a simple response using nnsight"""
    # Use nnsight's generation capabilities
    with logit_lens_extractor.model.generate(prompt, max_new_tokens=max_tokens) as generator:
        output = logit_lens_extractor.model.generator.output.save()
    
    # Decode the output
    full_text = logit_lens_extractor.model.tokenizer.decode(output[0])
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

        # Validate message roles
        valid_roles = {"system", "user", "assistant"}
        for message in request.messages:
            if message.role not in valid_roles:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid role: {message.role}. Must be one of: {', '.join(valid_roles)}"
                )

        current_message = request.text.strip()
        
        # Reset conversation tracker for new conversation
        # (In production, you'd want to maintain state per conversation ID)
        conversation_tracker.reset()
        
        # Add previous messages to conversation tracker
        for message in request.messages:
            is_user = message.role == "user"
            conversation_tracker.add_message(message.content, is_user)
        
        # Add current user message
        user_tokens = conversation_tracker.add_message(current_message, is_user=True)
        
        # Format conversation for model
        conversation_history = format_conversation(request.messages)
        if conversation_history:
            full_prompt = f"{conversation_history}\nHuman: {current_message}\nAssistant:"
        else:
            full_prompt = f"Human: {current_message}\nAssistant:"

        print(f"Processing prompt: {full_prompt}\n")
        
        try:
            # Generate response
            response_text = generate_simple_response(full_prompt, max_tokens=30)
            
            # Add response tokens to conversation tracker
            response_tokens = conversation_tracker.add_message(response_text, is_user=False)
            
            # Get the full conversation text for logit lens extraction
            full_conversation = conversation_tracker.get_full_text()
            
            # Extract real logit lens data for all tokens
            print("Extracting logit lens activations...")
            all_activations = logit_lens_extractor.extract_activations(full_conversation, top_k=5)
            
            # Create response token objects with real lens data
            response_token_objects = []
            for token_info in response_tokens:
                global_pos = token_info['global_position']
                
                # Get real logit lens predictions
                lens_data = logit_lens_extractor.format_layer_predictions(all_activations, global_pos)
                
                response_token_objects.append(
                    Token(
                        text=token_info['text'],
                        lens=lens_data
                    )
                )
            
            # Create user token objects with real lens data
            user_token_objects = []
            for token_info in user_tokens:
                global_pos = token_info['global_position']
                
                # Get real logit lens predictions
                lens_data = logit_lens_extractor.format_layer_predictions(all_activations, global_pos)
                
                user_token_objects.append(
                    Token(
                        text=token_info['text'],
                        lens=lens_data
                    )
                )
            
            print(f"Generated {len(response_token_objects)} response tokens with real logit lens data")
            
            return ConversationResponse(
                text=response_text,
                tokens=response_token_objects,
                userTokens=user_token_objects
            )
            
        except Exception as e:
            print(f"Error during generation or logit lens extraction: {str(e)}")
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
        "model_loaded": logit_lens_extractor.model is not None
    }

@app.get("/api/model_info")
async def model_info():
    """Get information about the loaded model."""
    return {
        "model_name": model_name,
        "num_layers": logit_lens_extractor.num_layers,
        "vocab_size": logit_lens_extractor.model.tokenizer.vocab_size,
        "device": str(next(logit_lens_extractor.model.transformer.parameters()).device)
    }

# New endpoint for getting token context
@app.get("/api/token/{global_position}/context")
async def get_token_context(global_position: int, context_size: int = 15):
    """Get context window for a specific token position."""
    try:
        context_tokens = conversation_tracker.get_context_window(global_position, context_size)
        
        if not context_tokens:
            raise HTTPException(status_code=404, detail="Token not found")
        
        # Get conversation text and extract logit lens for context
        full_text = conversation_tracker.get_full_text()
        
        # Extract activations for the target token
        target_lens = logit_lens_extractor.extract_for_single_token(
            full_text, global_position, top_k=5
        )
        
        return {
            "token_position": global_position,
            "context_tokens": context_tokens,
            "logit_lens": target_lens,
            "conversation_stats": conversation_tracker.get_conversation_stats()
        }
        
    except Exception as e:
        print(f"Error getting token context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))