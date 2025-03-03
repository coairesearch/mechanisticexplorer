import torch
import numpy as np
from nnsight import LanguageModel
import transformers
from nnsight import CONFIG
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Dict, Any
from .models import ConversationResponse, Token, LayerData, Prediction

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

@app.post("/api/chat", response_model=ConversationResponse)
async def chat(message: Dict[str, Any]):
    try:
        user_message = message.get("text", "")
        
        # Generate response using the model
        n_new_tokens = 50  # Adjust this value based on desired response length
        with model.generate(user_message, max_new_tokens=n_new_tokens, remote=True) as tracer:
            out = model.generator.output.save()

        # Decode the generated response
        decoded_prompt = model.tokenizer.decode(out[0][0:-n_new_tokens].cpu())
        response_text = model.tokenizer.decode(out[0][-n_new_tokens:].cpu())
        
        # Generate token data for both user message and response
        user_tokens = [
            Token(text=token, lens=generate_logit_lens_data(token))
            for token in tokenize_text(user_message)
        ]
        
        response_tokens = [
            Token(text=token, lens=generate_logit_lens_data(token))
            for token in tokenize_text(response_text)
        ]
        
        return ConversationResponse(
            text=response_text,
            tokens=response_tokens,
            userTokens=user_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}
