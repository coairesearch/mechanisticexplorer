"""
Real Logit Lens implementation using nnsight for activation extraction.
CORRECTED VERSION - Uses proper nnsight patterns.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from nnsight import LanguageModel
from .models import LayerData, Prediction


class LogitLensExtractor:
    """
    Extracts real logit lens predictions from language model layers using nnsight.
    """
    
    def __init__(self, model_name: str = "openai-community/gpt2"):
        """
        Initialize the extractor with an nnsight LanguageModel.
        
        Args:
            model_name: HuggingFace model identifier
        """
        torch.set_grad_enabled(False)
        self.model = LanguageModel(model_name, device_map="auto", dispatch=True)
        self.num_layers = len(self.model.transformer.h)
        
    def extract_activations(self, text: str, top_k: int = 5) -> Dict[int, Dict[str, Any]]:
        """
        Extract activations for all tokens in text using nnsight.
        Uses the correct pattern from nnsight's official logit lens tutorial.
        
        Args:
            text: Input text to analyze
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
            token_data = {}
            
            for layer_idx, probs in enumerate(probs_layers):
                # Get probabilities for this token at this layer
                token_probs = probs.value[0, token_pos, :]
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(token_probs, top_k)
                
                # Store layer data
                token_data[layer_idx] = {
                    'top_k_probs': top_probs.detach().cpu(),
                    'top_k_indices': top_indices.detach().cpu()
                }
            
            token_activations[token_pos] = token_data
        
        return token_activations
    
    def format_layer_predictions(self, token_activations: Dict[int, Dict[str, Any]], 
                                token_position: int) -> List[LayerData]:
        """
        Format token activations into LayerData objects for API response.
        
        Args:
            token_activations: Raw activations from extract_activations
            token_position: Which token to format predictions for
            
        Returns:
            List of LayerData objects, one per layer
        """
        if token_position not in token_activations:
            return []
        
        token_data = token_activations[token_position]
        layer_predictions = []
        
        for layer_idx in range(self.num_layers):
            if layer_idx not in token_data:
                continue
                
            layer_data = token_data[layer_idx]
            top_k_probs = layer_data['top_k_probs']
            top_k_indices = layer_data['top_k_indices']
            
            # Create predictions list
            predictions = []
            for rank, (prob_value, token_id) in enumerate(zip(top_k_probs, top_k_indices)):
                token_text = self.model.tokenizer.decode([token_id.item()])
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
    
    def extract_for_single_token(self, full_text: str, target_position: int, 
                                top_k: int = 5) -> List[LayerData]:
        """
        Extract logit lens predictions for a single token position.
        More efficient than extracting for all tokens.
        
        Args:
            full_text: Complete text context
            target_position: Token position to analyze
            top_k: Number of top predictions per layer
            
        Returns:
            List of LayerData for the target token
        """
        try:
            # Extract activations for all tokens (we need full context)
            all_activations = self.extract_activations(full_text, top_k)
            
            # Format predictions for the target token
            return self.format_layer_predictions(all_activations, target_position)
            
        except Exception as e:
            print(f"Error extracting logit lens for token {target_position}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []


class ConversationTokenizer:
    """
    Manages token tracking across multi-turn conversations with global positioning.
    """
    
    def __init__(self, model: LanguageModel):
        """
        Initialize with an nnsight LanguageModel.
        
        Args:
            model: nnsight LanguageModel instance
        """
        self.model = model
        self.tokenizer = model.tokenizer
        self.global_tokens = []  # All tokens in conversation order
        self.token_metadata = []  # Metadata for each token
        self.message_boundaries = []  # Indices where messages start
        
    def add_message(self, text: str, is_user: bool, message_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Add a message to the conversation and return token information.
        
        Args:
            text: Message text
            is_user: Whether this is a user message
            message_id: Optional message identifier
            
        Returns:
            List of token dictionaries with global positions
        """
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
                'message_start': local_pos == 0
            }
            
            self.global_tokens.append(token_info)
            self.token_metadata.append(token_info)
            message_tokens.append(token_info)
        
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
                context_tokens.append(token_info)
        
        return context_tokens
    
    def get_full_text(self) -> str:
        """
        Get the full conversation text for model processing.
        
        Returns:
            Complete conversation text
        """
        return ''.join(token['text'] for token in self.global_tokens)
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current conversation.
        
        Returns:
            Dictionary with conversation statistics
        """
        return {
            'total_tokens': len(self.global_tokens),
            'total_messages': len(self.message_boundaries),
            'user_tokens': sum(1 for t in self.global_tokens if t['is_user']),
            'assistant_tokens': sum(1 for t in self.global_tokens if not t['is_user']),
            'memory_usage_mb': len(self.global_tokens) * 0.001  # Rough estimate
        }
    
    def reset(self):
        """Reset the conversation state."""
        self.global_tokens.clear()
        self.token_metadata.clear()
        self.message_boundaries.clear()