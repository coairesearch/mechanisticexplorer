import torch
import numpy as np
from nnsight import LanguageModel
import transformers
from nnsight import CONFIG
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Dict, Any
from .models import ConversationResponse, Token, LayerData, Prediction, ChatRequest, Message
import requests

# Configure nnsight
CONFIG.API.HOST = "localhost:5001"
CONFIG.API.SSL = False
CONFIG.API.APIKEY = "0Bb6oUQxj2TuPtlrTkny"

# Initialize model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
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

def load_model(model_name: str):
    torch.set_grad_enabled(False)
    model = LanguageModel(model_name, device_map="auto")
    return model

def get_alternative_tokens(token: str) -> List[str]:
    """Get alternative tokens for a given token."""
    alternatives: Dict[str, List[str]] = {
        "Paris": ["London", "Rome", "Berlin"],
        "France": ["Europe", "country", "nation"],
        "is": ["was", "remains", "becomes"],
        "the": ["a", "this", "that"],
        "capital": ["city", "metropolis", "center"],
        "weather": ["climate", "forecast", "conditions"],
        "cloudy": ["sunny", "rainy", "clear"],
        "high": ["temperature", "maximum", "peak"],
        "recommend": ["suggest", "advise", "propose"],
        "machine": ["deep", "artificial", "computer"],
        "learning": ["intelligence", "education", "training"]
    }
    return alternatives.get(token, [f"word_{token}", "the", "and"])

def generate_logit_lens_data(token: str) -> List[LayerData]:
    """Generate mock logit lens data for a token."""
    layers = []
    for i in range(24):
        predictions = []
        actual_token_prob = 0.1 + (i / 24) * 0.89

        # Add alternatives in early layers with varying probabilities
        if i < 20:
            alternatives = get_alternative_tokens(token)
            # In early layers, alternatives might be more likely
            alt_prob_1 = max((20 - i) / 20 * 100, actual_token_prob * 100 * 0.8)
            alt_prob_2 = max((20 - i) / 20 * 50, actual_token_prob * 100 * 0.4)
            
            predictions.extend([
                Prediction(
                    token=alternatives[0],
                    probability=alt_prob_1,
                    rank=1
                ),
                Prediction(
                    token=alternatives[1],
                    probability=alt_prob_2,
                    rank=2
                )
            ])

        # Add actual token with increasing probability in later layers
        predictions.append(
            Prediction(
                token=token,
                probability=actual_token_prob * 100,
                rank=len(predictions) + 1
            )
        )

        # Sort by probability
        predictions.sort(key=lambda x: x.probability, reverse=True)
        for idx, pred in enumerate(predictions):
            pred.rank = idx + 1

        layers.append(
            LayerData(
                layer=i,
                predictions=predictions
            )
        )

    return layers

def tokenize_text(text: str) -> List[str]:
    """Simple tokenization of text."""
    import re
    return [t for t in re.split(r'([.,!?;:]|\s+)', text) if t and t.strip()]

def generate_response(message: str) -> str:
    """Generate a response based on the input message."""
    message_lower = message.lower()
    if "capital of france" in message_lower:
        return "The capital of France is Paris."
    if "hello" in message_lower or "hi" in message_lower:
        return "Hello! How can I help you today?"
    if "weather" in message_lower:
        return "The weather forecast shows partly cloudy skies with a high of 72°F."
    if "recommend" in message_lower or "suggestion" in message_lower:
        return "I would recommend trying the new machine learning course that was just released last month."
    return f"I understand your question about {' '.join(message.split()[-3:])}. Let me think about that."

def format_conversation(messages: List[Message]) -> str:
    """Format conversation history into a single string for the model."""
    formatted = ""
    for msg in messages:
        if msg.role == "system":
            formatted += f"System: {msg.content}\n"
        elif msg.role == "user":
            formatted += f"User: {msg.content}\n"
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

        # Format the entire conversation history including the new message
        conversation_history = format_conversation(request.messages)
        current_message = request.text.strip()
        
        # Combine history with current message
        full_prompt = f"{conversation_history}\nUser: {current_message}\nAssistant:"

        print(f"Full conversation context being sent to model:\n{full_prompt}\n")
        
        try:
            # Generate response using the model
            n_new_tokens = 50  # Adjust this value based on desired response length
            with model.generate(full_prompt, max_new_tokens=n_new_tokens, remote=True) as tracer:
                out = model.generator.output.save()

            # Decode the generated response
            decoded_prompt = model.tokenizer.decode(out[0][0:-n_new_tokens].cpu())
            response_text = model.tokenizer.decode(out[0][-n_new_tokens:].cpu()).strip()
            
            if not response_text:
                raise ValueError("Model generated empty response")

            print(f"Model response: {response_text}\n")

        except Exception as model_error:
            print(f"Model error: {str(model_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Model generation error: {str(model_error)}"
            )
        
        # Generate token data for both user message and response
        try:
            user_tokens = [
                Token(text=token, lens=generate_logit_lens_data(token))
                for token in tokenize_text(current_message)
            ]
            
            response_tokens = [
                Token(text=token, lens=generate_logit_lens_data(token))
                for token in tokenize_text(response_text)
            ]
        except Exception as token_error:
            print(f"Token processing error: {str(token_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Token processing error: {str(token_error)}"
            )
        
        return ConversationResponse(
            text=response_text,
            tokens=response_tokens,
            userTokens=user_tokens
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/available_models")
def get_ndif_models(host="localhost", port=5001):
    """Get list of models running on NDIF cluster"""
    url = f"http://{host}:{port}/stats"
    response = requests.get(url)
    
    if response.status_code == 200:
        raw_models = response.json()
        # Restructure the response to use repo_id as the main key
        models = {}
        for _, model_data in raw_models.items():
            repo_id = model_data.get('repo_id')
            if repo_id:
                models[repo_id] = {
                    'num_running_replicas': model_data['num_running_replicas'],
                    'config_json_string': model_data['config_json_string']
                }
        return models
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None