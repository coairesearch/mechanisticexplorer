from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Prediction(BaseModel):
    token: str
    probability: float
    rank: int

class LayerData(BaseModel):
    layer: int
    predictions: List[Prediction]

class Token(BaseModel):
    text: str
    lens: List[LayerData]

class Message(BaseModel):
    role: str = Field(..., description="The role of the message sender (system, user, or assistant)")
    content: str = Field(..., description="The content of the message")

class ChatRequest(BaseModel):
    messages: List[Message] = Field(default_factory=list, description="List of previous messages in the conversation")
    text: str = Field(..., description="The current message text")

class ConversationResponse(BaseModel):
    text: str
    tokens: List[Token]
    userTokens: List[Token]

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello! How can I help you today?",
                "tokens": [],
                "userTokens": []
            }
        }
