from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
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

class CachedChatRequest(BaseModel):
    messages: List[Message] = Field(default_factory=list, description="List of previous messages in the conversation")
    text: str = Field(..., description="The current message text")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID to resume existing conversation")
    enable_caching: bool = Field(True, description="Whether to cache activations for this conversation")

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

class CachedConversationResponse(BaseModel):
    conversation_id: str
    text: str
    tokens: List[Token]
    userTokens: List[Token]
    cached: bool = Field(True, description="Whether activations were cached")
    cache_stats: Optional[Dict[str, Any]] = Field(None, description="Cache statistics")

class TokenContextResponse(BaseModel):
    token_position: int
    activations: List[LayerData]
    context_tokens: List[Dict[str, Any]]
    conversation_stats: Dict[str, Any]

class CacheStatusResponse(BaseModel):
    memory_used_mb: float
    memory_limit_mb: float
    memory_utilization_percent: float
    disk_used_mb: float
    disk_conversations: int
    cache_type: str

class ConversationListResponse(BaseModel):
    conversations: List[Dict[str, Any]]
    total_count: int
