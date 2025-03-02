from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Prediction(BaseModel):
    token: str
    probability: float
    rank: Optional[int] = None

class LayerData(BaseModel):
    layer: int
    predictions: List[Prediction]

class Token(BaseModel):
    text: str
    lens: List[LayerData]

class Message(BaseModel):
    text: str
    isUser: bool
    timestamp: datetime
    tokens: Optional[List[Token]] = None
    isError: Optional[bool] = None

class ConversationResponse(BaseModel):
    text: str
    tokens: List[Token]
    userTokens: List[Token]
