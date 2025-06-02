"""
Enhanced conversation management with caching support.
Handles persistent conversations across sessions with activation caching.
"""

import uuid
import time
import json
import torch
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from nnsight import LanguageModel
from .cache import ActivationCache


class PersistentConversationTokenizer:
    """
    Enhanced conversation tokenizer with persistence and caching support.
    Manages conversations across sessions with unique IDs and global token tracking.
    """
    
    def __init__(self, model: LanguageModel, cache: Optional[ActivationCache] = None, 
                 persistence_dir: str = "./conversations"):
        """
        Initialize with model and optional caching.
        
        Args:
            model: nnsight LanguageModel instance
            cache: Optional activation cache for storing computed activations
            persistence_dir: Directory for saving conversation metadata
        """
        self.model = model
        self.tokenizer = model.tokenizer
        self.cache = cache
        self.persistence_dir = Path(persistence_dir)
        self.persistence_dir.mkdir(exist_ok=True)
        
        # Current conversation state
        self.conversation_id: Optional[str] = None
        self.global_tokens: List[Dict[str, Any]] = []
        self.message_boundaries: List[int] = []
        self.conversation_metadata: Dict[str, Any] = {}
        
    def start_conversation(self, conversation_id: Optional[str] = None) -> str:
        """
        Start a new conversation or resume an existing one.
        
        Args:
            conversation_id: Optional existing conversation ID to resume
            
        Returns:
            Conversation ID
        """
        if conversation_id is None:
            # Create new conversation
            conversation_id = str(uuid.uuid4())
            self._initialize_new_conversation(conversation_id)
        else:
            # Resume existing conversation
            if not self._load_conversation(conversation_id):
                # If loading fails, create new conversation
                self._initialize_new_conversation(conversation_id)
        
        self.conversation_id = conversation_id
        return conversation_id
    
    def add_message(self, text: str, is_user: bool, message_id: Optional[str] = None, 
                   cache_activations: bool = True) -> List[Dict[str, Any]]:
        """
        Add a message to the conversation with optional activation caching.
        
        Args:
            text: Message text
            is_user: Whether this is a user message
            message_id: Optional message identifier
            cache_activations: Whether to cache activations for this message
            
        Returns:
            List of token dictionaries with global positions
        """
        if self.conversation_id is None:
            raise ValueError("No active conversation. Call start_conversation() first.")
        
        # Mark message boundary
        self.message_boundaries.append(len(self.global_tokens))
        
        # Tokenize the message
        encoding = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=False)
        
        message_tokens = []
        for local_pos, token_id in enumerate(encoding.input_ids):
            token_text = self.tokenizer.decode([token_id])
            global_pos = len(self.global_tokens)
            
            token_info = {
                'text': token_text,
                'id': token_id,
                'global_position': global_pos,
                'local_position': local_pos,
                'is_user': is_user,
                'message_id': message_id,
                'message_start': local_pos == 0,
                'timestamp': time.time(),
                'cached': False  # Will be set to True if activations are cached
            }
            
            self.global_tokens.append(token_info)
            message_tokens.append(token_info)
        
        # Update conversation metadata
        self.conversation_metadata.update({
            'last_updated': time.time(),
            'total_tokens': len(self.global_tokens),
            'total_messages': len(self.message_boundaries),
            'user_tokens': sum(1 for t in self.global_tokens if t['is_user']),
            'assistant_tokens': sum(1 for t in self.global_tokens if not t['is_user'])
        })
        
        # Save conversation state
        self._save_conversation()
        
        return message_tokens
    
    def get_context_window(self, position: int, size: int = 15) -> List[Dict[str, Any]]:
        """
        Get context window of tokens around a position.
        
        Args:
            position: Global token position
            size: Number of tokens to include before the target
            
        Returns:
            List of token info dictionaries including context
        """
        if position >= len(self.global_tokens):
            return []
        
        # Get tokens from start of window to target position (inclusive)
        start_pos = max(0, position - size)
        end_pos = position + 1
        
        context_tokens = []
        for i in range(start_pos, end_pos):
            if i < len(self.global_tokens):
                token_info = self.global_tokens[i].copy()
                token_info['is_target'] = (i == position)
                token_info['relative_position'] = i - position  # Negative for context, 0 for target
                
                # Check if activations are cached
                if self.cache:
                    cached_activations = self.cache.retrieve(self.conversation_id, i)
                    token_info['has_cached_activations'] = cached_activations is not None
                else:
                    token_info['has_cached_activations'] = False
                
                context_tokens.append(token_info)
        
        return context_tokens
    
    def get_full_text(self) -> str:
        """
        Get the full conversation text for model processing.
        
        Returns:
            Complete conversation text
        """
        return ''.join(token['text'] for token in self.global_tokens)
    
    def get_tokens_since_position(self, start_position: int) -> List[Dict[str, Any]]:
        """
        Get all tokens from a starting position to the end.
        Useful for incremental processing.
        
        Args:
            start_position: Starting global token position
            
        Returns:
            List of tokens from start position onwards
        """
        if start_position >= len(self.global_tokens):
            return []
        
        return self.global_tokens[start_position:]
    
    def mark_tokens_cached(self, start_position: int, end_position: int):
        """
        Mark a range of tokens as having cached activations.
        
        Args:
            start_position: Starting token position (inclusive)
            end_position: Ending token position (exclusive)
        """
        for i in range(start_position, min(end_position, len(self.global_tokens))):
            if i < len(self.global_tokens):
                self.global_tokens[i]['cached'] = True
        
        # Save updated state
        self._save_conversation()
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the current conversation.
        
        Returns:
            Dictionary with conversation statistics including cache info
        """
        stats = self.conversation_metadata.copy()
        
        # Add cache statistics if available
        if self.cache:
            cache_stats = self.cache.get_stats()
            stats['cache'] = cache_stats
            
            # Count cached tokens
            cached_count = 0
            if self.conversation_id:
                for i in range(len(self.global_tokens)):
                    if self.cache.retrieve(self.conversation_id, i) is not None:
                        cached_count += 1
            
            stats['cached_tokens'] = cached_count
            stats['cache_coverage_percent'] = (
                round(cached_count / len(self.global_tokens) * 100, 1) 
                if self.global_tokens else 0
            )
        
        return stats
    
    def clear_conversation(self) -> bool:
        """
        Clear the current conversation from memory and cache.
        
        Returns:
            True if successful
        """
        if self.conversation_id is None:
            return True
        
        # Clear from cache
        if self.cache:
            self.cache.clear_conversation(self.conversation_id)
        
        # Clear from disk
        conv_file = self.persistence_dir / f"{self.conversation_id}.json"
        if conv_file.exists():
            conv_file.unlink()
        
        # Reset state
        self.conversation_id = None
        self.global_tokens.clear()
        self.message_boundaries.clear()
        self.conversation_metadata.clear()
        
        return True
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        List all saved conversations.
        
        Returns:
            List of conversation metadata
        """
        conversations = []
        
        for conv_file in self.persistence_dir.glob("*.json"):
            try:
                with open(conv_file, 'r') as f:
                    metadata = json.load(f)
                    conversations.append({
                        'conversation_id': conv_file.stem,
                        'created': metadata.get('created', 0),
                        'last_updated': metadata.get('last_updated', 0),
                        'total_tokens': metadata.get('total_tokens', 0),
                        'total_messages': metadata.get('total_messages', 0)
                    })
            except Exception:
                continue  # Skip corrupted files
        
        # Sort by last updated (newest first)
        conversations.sort(key=lambda x: x['last_updated'], reverse=True)
        return conversations
    
    def _initialize_new_conversation(self, conversation_id: str):
        """Initialize a new conversation with metadata."""
        self.global_tokens = []
        self.message_boundaries = []
        self.conversation_metadata = {
            'conversation_id': conversation_id,
            'created': time.time(),
            'last_updated': time.time(),
            'total_tokens': 0,
            'total_messages': 0,
            'user_tokens': 0,
            'assistant_tokens': 0
        }
    
    def _load_conversation(self, conversation_id: str) -> bool:
        """
        Load conversation from persistence.
        
        Args:
            conversation_id: Conversation ID to load
            
        Returns:
            True if loaded successfully
        """
        conv_file = self.persistence_dir / f"{conversation_id}.json"
        
        if not conv_file.exists():
            return False
        
        try:
            with open(conv_file, 'r') as f:
                data = json.load(f)
            
            self.global_tokens = data.get('global_tokens', [])
            self.message_boundaries = data.get('message_boundaries', [])
            self.conversation_metadata = data.get('metadata', {})
            
            return True
            
        except Exception as e:
            print(f"Error loading conversation {conversation_id}: {str(e)}")
            return False
    
    def _save_conversation(self):
        """Save current conversation to persistence."""
        if self.conversation_id is None:
            return
        
        conv_file = self.persistence_dir / f"{self.conversation_id}.json"
        
        try:
            data = {
                'global_tokens': self.global_tokens,
                'message_boundaries': self.message_boundaries,
                'metadata': self.conversation_metadata
            }
            
            with open(conv_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving conversation {self.conversation_id}: {str(e)}")


class CachedLogitLensExtractor:
    """
    LogitLensExtractor enhanced with activation caching for fast repeated access.
    """
    
    def __init__(self, model: LanguageModel, cache: Optional[ActivationCache] = None):
        """
        Initialize with model and cache.
        
        Args:
            model: nnsight LanguageModel instance
            cache: Optional activation cache
        """
        self.model = model
        self.cache = cache
        self.num_layers = len(self.model.transformer.h)
    
    def extract_activations_with_caching(self, text: str, conversation_id: str, 
                                       start_token_idx: int = 0, top_k: int = 5) -> Dict[int, Dict[str, Any]]:
        """
        Extract activations with caching support for incremental processing.
        
        Args:
            text: Input text to analyze
            conversation_id: Conversation ID for cache storage
            start_token_idx: Starting token index for this text segment
            top_k: Number of top predictions to return per layer
            
        Returns:
            Dict mapping token positions to their layer activations
        """
        layers = self.model.transformer.h
        probs_layers = []

        # Use the correct nnsight pattern with tracer.invoke()
        with self.model.trace() as tracer:
            with tracer.invoke(text) as invoker:
                for layer_idx, layer in enumerate(layers):
                    # Process layer output through the model's head and layer normalization
                    layer_output = self.model.lm_head(self.model.transformer.ln_f(layer.output[0]))

                    # Apply softmax to obtain probabilities and save the result
                    probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
                    probs_layers.append(probs)

        # Extract token activations in API-compatible format
        token_activations = {}
        
        # Get input token information
        input_token_ids = invoker.inputs[0][0].ids
        num_tokens = len(input_token_ids)
        
        for token_pos in range(num_tokens):
            global_token_pos = start_token_idx + token_pos
            layer_activations = {}
            
            for layer_idx, probs in enumerate(probs_layers):
                # Get probabilities for this token at this layer
                token_probs = probs.value[0, token_pos, :]
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(token_probs, top_k)
                
                # Store layer data
                layer_activations[layer_idx] = {
                    'top_k_probs': top_probs.detach().cpu(),
                    'top_k_indices': top_indices.detach().cpu()
                }
            
            token_activations[global_token_pos] = layer_activations
            
            # Cache the activations if cache is available
            if self.cache:
                self.cache.store(conversation_id, global_token_pos, layer_activations)
        
        return token_activations
    
    def get_cached_predictions(self, conversation_id: str, token_position: int) -> Optional[List]:
        """
        Get cached predictions for a token position.
        
        Args:
            conversation_id: Conversation ID
            token_position: Global token position
            
        Returns:
            Formatted predictions or None if not cached
        """
        if not self.cache:
            return None
        
        activations = self.cache.retrieve(conversation_id, token_position)
        if activations is None:
            return None
        
        # Format into LayerData objects manually
        from .models import LayerData, Prediction
        
        layer_predictions = []
        
        for layer_idx in range(self.num_layers):
            if layer_idx not in activations:
                continue
                
            layer_data = activations[layer_idx]
            top_k_probs = layer_data['top_k_probs']
            top_k_indices = layer_data['top_k_indices']
            
            # Create predictions list
            predictions = []
            for rank, (prob_value, token_id) in enumerate(zip(top_k_probs, top_k_indices)):
                # Ensure token_id is an integer
                token_id_int = int(token_id.item())
                token_text = self.model.tokenizer.decode([token_id_int])
                predictions.append(
                    Prediction(
                        token=token_text,
                        probability=prob_value.item() * 100,  # Convert to percentage
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