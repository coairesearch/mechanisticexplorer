"""
Enhanced FastAPI server with activation caching for multi-turn conversations.
Supports persistent conversations and fast token analysis through caching.
"""

import torch
import numpy as np
import logging
from nnsight import LanguageModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import models
from .models import (
    ConversationResponse, Token, LayerData, Prediction, ChatRequest, Message,
    CachedChatRequest, CachedConversationResponse, TokenContextResponse,
    CacheStatusResponse, ConversationListResponse
)

# Import caching and conversation management
from .cache import create_default_cache
from .conversation import PersistentConversationTokenizer, CachedLogitLensExtractor
from .logit_lens import LogitLensExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model - using local GPT-2 with nnsight
model_name = "openai-community/gpt2"
torch.set_grad_enabled(False)

# Initialize caching system
cache = create_default_cache(
    memory_mb=512,  # 512MB memory cache
    disk_gb=2.0,    # 2GB disk cache
    cache_dir="./cache"
)

# Initialize enhanced components
model = LanguageModel(model_name, device_map="auto", dispatch=True)
conversation_manager = PersistentConversationTokenizer(
    model=model, 
    cache=cache,
    persistence_dir="./conversations"
)
cached_extractor = CachedLogitLensExtractor(model=model, cache=cache)

# Fallback extractor for non-cached operations
fallback_extractor = LogitLensExtractor(model_name)

app = FastAPI(title="Logit Lens API with Caching")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_with_caching(prompt: str, conversation_id: str, max_tokens: int = 30) -> str:
    """Generate response using nnsight with conversation context."""
    try:
        # Use nnsight's generation capabilities
        with model.generate(prompt, max_new_tokens=max_tokens) as generator:
            output = model.generator.output.save()
        
        # Decode the output
        full_text = model.tokenizer.decode(output[0])
        # Remove the prompt from the beginning
        response = full_text[len(prompt):].strip()
        
        return response
        
    except Exception as e:
        logger.error(f"Error in generation: {str(e)}")
        # Fallback response
        return "I apologize, but I encountered an error while generating a response."

def format_conversation(messages: List[Message]) -> str:
    """Format conversation history into a single string for the model."""
    formatted = ""
    for msg in messages:
        if msg.role == "user":
            formatted += f"Human: {msg.content}\n"
        elif msg.role == "assistant":
            formatted += f"Assistant: {msg.content}\n"
    return formatted.strip()

@app.post("/api/chat/cached", response_model=CachedConversationResponse)
async def chat_with_cache(request: CachedChatRequest):
    """Enhanced chat endpoint with activation caching."""
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
        
        # Start or resume conversation
        conversation_id = conversation_manager.start_conversation(request.conversation_id)
        
        # Add previous messages if this is a new conversation
        if not conversation_manager.global_tokens:
            for message in request.messages:
                is_user = message.role == "user"
                conversation_manager.add_message(message.content, is_user)
        
        # Add current user message
        user_tokens = conversation_manager.add_message(current_message, is_user=True)
        
        # Format conversation for model
        conversation_history = format_conversation(request.messages)
        if conversation_history:
            full_prompt = f"{conversation_history}\nHuman: {current_message}\nAssistant:"
        else:
            full_prompt = f"Human: {current_message}\nAssistant:"

        logger.info(f"Processing conversation {conversation_id} with {len(conversation_manager.global_tokens)} total tokens")
        
        # Generate response
        response_text = generate_with_caching(full_prompt, conversation_id, max_tokens=30)
        
        # Add response tokens to conversation
        response_tokens = conversation_manager.add_message(response_text, is_user=False)
        
        # Get the full conversation text for logit lens extraction
        full_conversation = conversation_manager.get_full_text()
        
        # Extract activations with caching for the new tokens only
        if request.enable_caching:
            logger.info("Extracting and caching activations...")
            
            # Find the starting position for new tokens (user + assistant)
            start_position = len(conversation_manager.global_tokens) - len(user_tokens) - len(response_tokens)
            
            # Extract activations for the new segment
            new_activations = cached_extractor.extract_activations_with_caching(
                text=current_message + response_text,
                conversation_id=conversation_id,
                start_token_idx=start_position,
                top_k=5
            )
            
            # Mark tokens as cached
            conversation_manager.mark_tokens_cached(
                start_position, 
                len(conversation_manager.global_tokens)
            )
        
        # Create response token objects with lens data
        response_token_objects = []
        user_token_objects = []
        
        for token_info in response_tokens:
            global_pos = token_info['global_position']
            
            # Try to get cached predictions first
            lens_data = cached_extractor.get_cached_predictions(conversation_id, global_pos)
            
            if lens_data is None and request.enable_caching:
                # Fallback: extract on demand (shouldn't happen with caching)
                logger.warning(f"Cache miss for response token {global_pos}, extracting on demand")
                lens_data = []
            
            response_token_objects.append(
                Token(
                    text=token_info['text'],
                    lens=lens_data or []
                )
            )
        
        for token_info in user_tokens:
            global_pos = token_info['global_position']
            
            # Try to get cached predictions
            lens_data = cached_extractor.get_cached_predictions(conversation_id, global_pos)
            
            user_token_objects.append(
                Token(
                    text=token_info['text'],
                    lens=lens_data or []
                )
            )
        
        # Get conversation and cache statistics
        conv_stats = conversation_manager.get_conversation_stats()
        
        logger.info(f"Generated {len(response_token_objects)} response tokens for conversation {conversation_id}")
        
        return CachedConversationResponse(
            conversation_id=conversation_id,
            text=response_text,
            tokens=response_token_objects,
            userTokens=user_token_objects,
            cached=request.enable_caching,
            cache_stats=conv_stats.get('cache')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in cached chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ConversationResponse)
async def chat(request: ChatRequest):
    """Original chat endpoint without caching (for compatibility)."""
    # Convert to cached request with caching disabled
    cached_request = CachedChatRequest(
        messages=request.messages,
        text=request.text,
        conversation_id=None,
        enable_caching=False
    )
    
    # Use cached endpoint but return non-cached response
    cached_response = await chat_with_cache(cached_request)
    
    return ConversationResponse(
        text=cached_response.text,
        tokens=cached_response.tokens,
        userTokens=cached_response.userTokens
    )

@app.get("/api/cache/token/{conversation_id}/{token_idx}", response_model=TokenContextResponse)
async def get_cached_token(conversation_id: str, token_idx: int, context_size: int = 15):
    """Get cached activation for a specific token with context."""
    try:
        # Check if conversation exists
        if not conversation_manager.conversation_id == conversation_id:
            # Try to load the conversation
            if not conversation_manager.start_conversation(conversation_id):
                raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get cached activations
        lens_data = cached_extractor.get_cached_predictions(conversation_id, token_idx)
        if lens_data is None:
            raise HTTPException(status_code=404, detail="Token activations not cached")
        
        # Get context tokens
        context_tokens = conversation_manager.get_context_window(token_idx, context_size)
        
        # Get conversation stats
        conv_stats = conversation_manager.get_conversation_stats()
        
        return TokenContextResponse(
            token_position=token_idx,
            activations=lens_data,
            context_tokens=context_tokens,
            conversation_stats=conv_stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cached token {conversation_id}:{token_idx}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cache/status", response_model=CacheStatusResponse)
async def cache_status():
    """Get cache system status and statistics."""
    try:
        stats = cache.get_stats()
        
        if stats['type'] == 'hybrid':
            memory_stats = stats['memory']
            disk_stats = stats['disk']
            
            return CacheStatusResponse(
                memory_used_mb=memory_stats['size_mb'],
                memory_limit_mb=memory_stats['max_size_mb'],
                memory_utilization_percent=memory_stats['utilization_percent'],
                disk_used_mb=disk_stats['size_mb'],
                disk_conversations=disk_stats['conversations'],
                cache_type="hybrid"
            )
        else:
            # Fallback for other cache types
            return CacheStatusResponse(
                memory_used_mb=stats.get('size_mb', 0),
                memory_limit_mb=stats.get('max_size_mb', 0),
                memory_utilization_percent=stats.get('utilization_percent', 0),
                disk_used_mb=0,
                disk_conversations=0,
                cache_type=stats['type']
            )
            
    except Exception as e:
        logger.error(f"Error getting cache status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations", response_model=ConversationListResponse)
async def list_conversations():
    """List all saved conversations."""
    try:
        conversations = conversation_manager.list_conversations()
        
        return ConversationListResponse(
            conversations=conversations,
            total_count=len(conversations)
        )
        
    except Exception as e:
        logger.error(f"Error listing conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and its cached data."""
    try:
        # Load and clear the conversation
        original_id = conversation_manager.conversation_id
        conversation_manager.start_conversation(conversation_id)
        success = conversation_manager.clear_conversation()
        
        # Restore original conversation if different
        if original_id and original_id != conversation_id:
            conversation_manager.start_conversation(original_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {"message": f"Conversation {conversation_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": model_name,
        "model_loaded": model is not None,
        "cache_enabled": cache is not None,
        "features": ["activation_caching", "persistent_conversations", "fast_token_analysis"]
    }

@app.get("/api/model_info")
async def model_info():
    """Get information about the loaded model and cache."""
    cache_stats = cache.get_stats() if cache else None
    
    return {
        "model_name": model_name,
        "num_layers": len(model.transformer.h),
        "vocab_size": model.tokenizer.vocab_size,
        "device": str(next(model.transformer.parameters()).device),
        "cache_stats": cache_stats,
        "cache_enabled": cache is not None
    }

# Legacy endpoint for backward compatibility
@app.get("/api/token/{global_position}/context")
async def get_token_context_legacy(global_position: int, context_size: int = 15):
    """Legacy endpoint for token context (redirects to cached version)."""
    if conversation_manager.conversation_id is None:
        raise HTTPException(status_code=400, detail="No active conversation")
    
    return await get_cached_token(
        conversation_manager.conversation_id, 
        global_position, 
        context_size
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)