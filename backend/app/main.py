from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Dict, Any
from .models import ConversationResponse, Token, LayerData, Prediction

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

        # Add alternatives in early layers
        if i < 20:
            alternatives = get_alternative_tokens(token)
            predictions.extend([
                Prediction(
                    token=alternatives[0],
                    probability=(1 - actual_token_prob) * 70,
                    rank=1
                ),
                Prediction(
                    token=alternatives[1],
                    probability=(1 - actual_token_prob) * 30,
                    rank=2
                )
            ])

        # Add actual token
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
        response_text = generate_response(user_message)
        
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
