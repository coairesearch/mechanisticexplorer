"""
Activation caching system for multi-turn conversations.
Implements memory and disk caching with compression for fast token analysis.
"""

import time
import uuid
import pickle
import lz4.frame
import torch
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class ActivationCache(ABC):
    """Abstract base class for activation caching."""
    
    @abstractmethod
    def store(self, conv_id: str, token_idx: int, activations: Dict[int, Dict[str, Any]]) -> bool:
        """Store activations for a token at all layers."""
        pass
    
    @abstractmethod
    def retrieve(self, conv_id: str, token_idx: int) -> Optional[Dict[int, Dict[str, Any]]]:
        """Retrieve activations for a token across all layers."""
        pass
    
    @abstractmethod
    def clear_conversation(self, conv_id: str) -> bool:
        """Clear all activations for a conversation."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCache(ActivationCache):
    """In-memory LRU cache with compression."""
    
    def __init__(self, max_size_mb: int = 512):
        """
        Initialize memory cache with size limit.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
        """
        self.cache = OrderedDict()  # For LRU behavior
        self.max_size = max_size_mb * 1024 * 1024  # Convert to bytes
        self.current_size = 0
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        
    def store(self, conv_id: str, token_idx: int, activations: Dict[int, Dict[str, Any]]) -> bool:
        """Store compressed activations in memory."""
        key = f"{conv_id}:{token_idx}"
        
        try:
            # Compress activations
            compressed = self._compress_activations(activations)
            size = len(compressed)
            
            # Remove old entry if exists
            if key in self.cache:
                old_size = len(self.cache[key])
                self.current_size -= old_size
                del self.cache[key]
            
            # Evict LRU items if needed
            while self.current_size + size > self.max_size and self.cache:
                self._evict_lru()
            
            # Check if we have space
            if self.current_size + size > self.max_size:
                logger.warning(f"Cannot store activation {key}: would exceed cache limit")
                return False
            
            # Store compressed data
            self.cache[key] = compressed
            self.current_size += size
            self.access_times[key] = time.time()
            
            logger.debug(f"Stored activation {key} ({size} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Error storing activation {key}: {str(e)}")
            return False
    
    def retrieve(self, conv_id: str, token_idx: int) -> Optional[Dict[int, Dict[str, Any]]]:
        """Retrieve and decompress activations from memory."""
        key = f"{conv_id}:{token_idx}"
        
        if key not in self.cache:
            self.miss_count += 1
            return None
        
        try:
            # Move to end for LRU
            self.cache.move_to_end(key)
            self.access_times[key] = time.time()
            self.hit_count += 1
            
            # Decompress and return
            compressed_data = self.cache[key]
            activations = self._decompress_activations(compressed_data)
            
            logger.debug(f"Retrieved activation {key}")
            return activations
            
        except Exception as e:
            logger.error(f"Error retrieving activation {key}: {str(e)}")
            # Remove corrupted entry
            if key in self.cache:
                self.current_size -= len(self.cache[key])
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
            return None
    
    def clear_conversation(self, conv_id: str) -> bool:
        """Clear all tokens for a conversation."""
        keys_to_remove = [key for key in self.cache.keys() if key.startswith(f"{conv_id}:")]
        
        for key in keys_to_remove:
            self.current_size -= len(self.cache[key])
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
        
        logger.info(f"Cleared {len(keys_to_remove)} tokens for conversation {conv_id}")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "type": "memory",
            "size_mb": round(self.current_size / 1024 / 1024, 2),
            "max_size_mb": round(self.max_size / 1024 / 1024, 2),
            "utilization_percent": round(self.current_size / self.max_size * 100, 1),
            "entries": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate_percent": round(hit_rate, 1)
        }
    
    def _evict_lru(self):
        """Remove least recently used item."""
        if not self.cache:
            return
        
        # Remove oldest item
        oldest_key, oldest_data = self.cache.popitem(last=False)
        self.current_size -= len(oldest_data)
        
        if oldest_key in self.access_times:
            del self.access_times[oldest_key]
        
        logger.debug(f"Evicted LRU item {oldest_key}")
    
    def _compress_activations(self, activations: Dict[int, Dict[str, Any]]) -> bytes:
        """Compress activations for storage."""
        # Convert tensors to half precision and move to CPU for space efficiency
        compressed_acts = {}
        
        for layer_idx, layer_data in activations.items():
            compressed_layer = {}
            
            for key, value in layer_data.items():
                if isinstance(value, torch.Tensor):
                    # Convert to half precision and CPU
                    compressed_layer[key] = value.half().cpu().numpy()
                else:
                    # Keep as-is for non-tensors
                    compressed_layer[key] = value
            
            compressed_acts[layer_idx] = compressed_layer
        
        # Serialize and compress with LZ4
        serialized = pickle.dumps(compressed_acts, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = lz4.frame.compress(serialized)
        
        return compressed
    
    def _decompress_activations(self, compressed_data: bytes) -> Dict[int, Dict[str, Any]]:
        """Decompress activations from storage."""
        # Decompress and deserialize
        decompressed = lz4.frame.decompress(compressed_data)
        activations = pickle.loads(decompressed)
        
        # Convert numpy arrays back to tensors
        result = {}
        for layer_idx, layer_data in activations.items():
            result_layer = {}
            
            for key, value in layer_data.items():
                if isinstance(value, np.ndarray):
                    # Convert back to tensor
                    result_layer[key] = torch.from_numpy(value).float()
                else:
                    result_layer[key] = value
            
            result[layer_idx] = result_layer
        
        return result


class DiskCache(ActivationCache):
    """Persistent disk cache with per-conversation directories."""
    
    def __init__(self, cache_dir: str = "./cache", max_size_gb: float = 5.0):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Directory for cache storage
            max_size_gb: Maximum cache size in gigabytes
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size_gb * 1024 * 1024 * 1024  # Convert to bytes
        
    def store(self, conv_id: str, token_idx: int, activations: Dict[int, Dict[str, Any]]) -> bool:
        """Store activations to disk."""
        try:
            conv_dir = self.cache_dir / conv_id
            conv_dir.mkdir(exist_ok=True)
            
            # Store each layer in separate file for efficient access
            for layer_idx, layer_data in activations.items():
                layer_file = conv_dir / f"layer_{layer_idx:03d}_token_{token_idx:06d}.pt"
                
                # Convert tensors to half precision for space savings
                compressed_data = {}
                for key, value in layer_data.items():
                    if isinstance(value, torch.Tensor):
                        compressed_data[key] = value.half().cpu()
                    else:
                        compressed_data[key] = value
                
                torch.save(compressed_data, layer_file, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            
            # Check and enforce size limits
            self._enforce_size_limit()
            
            logger.debug(f"Stored activation to disk: {conv_id}:{token_idx}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing to disk {conv_id}:{token_idx}: {str(e)}")
            return False
    
    def retrieve(self, conv_id: str, token_idx: int) -> Optional[Dict[int, Dict[str, Any]]]:
        """Retrieve activations from disk."""
        try:
            conv_dir = self.cache_dir / conv_id
            if not conv_dir.exists():
                return None
            
            activations = {}
            
            # Load all layer files for this token
            pattern = f"layer_*_token_{token_idx:06d}.pt"
            layer_files = list(conv_dir.glob(pattern))
            
            if not layer_files:
                return None
            
            for layer_file in layer_files:
                # Extract layer index from filename
                layer_idx = int(layer_file.stem.split('_')[1])
                
                # Load and convert back to float
                layer_data = torch.load(layer_file, map_location='cpu', weights_only=False)
                
                # Convert back to full precision
                result_data = {}
                for key, value in layer_data.items():
                    if isinstance(value, torch.Tensor):
                        result_data[key] = value.float()
                    else:
                        result_data[key] = value
                
                activations[layer_idx] = result_data
            
            logger.debug(f"Retrieved activation from disk: {conv_id}:{token_idx}")
            return activations
            
        except Exception as e:
            logger.error(f"Error retrieving from disk {conv_id}:{token_idx}: {str(e)}")
            return None
    
    def clear_conversation(self, conv_id: str) -> bool:
        """Clear all files for a conversation."""
        try:
            conv_dir = self.cache_dir / conv_id
            if conv_dir.exists():
                # Remove all files in conversation directory
                for file in conv_dir.iterdir():
                    file.unlink()
                conv_dir.rmdir()
                
                logger.info(f"Cleared conversation directory: {conv_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing conversation {conv_id}: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get disk cache statistics."""
        try:
            total_size = 0
            conversation_count = 0
            file_count = 0
            
            for conv_dir in self.cache_dir.iterdir():
                if conv_dir.is_dir():
                    conversation_count += 1
                    for file in conv_dir.rglob('*.pt'):
                        total_size += file.stat().st_size
                        file_count += 1
            
            return {
                "type": "disk",
                "size_mb": round(total_size / 1024 / 1024, 2),
                "max_size_gb": round(self.max_size / 1024 / 1024 / 1024, 2),
                "utilization_percent": round(total_size / self.max_size * 100, 1),
                "conversations": conversation_count,
                "files": file_count,
                "directory": str(self.cache_dir)
            }
            
        except Exception as e:
            logger.error(f"Error getting disk stats: {str(e)}")
            return {"type": "disk", "error": str(e)}
    
    def _enforce_size_limit(self):
        """Remove oldest conversations if size limit exceeded."""
        try:
            current_size = sum(
                file.stat().st_size 
                for file in self.cache_dir.rglob('*.pt')
            )
            
            if current_size <= self.max_size:
                return
            
            # Get conversations sorted by modification time (oldest first)
            conversations = []
            for conv_dir in self.cache_dir.iterdir():
                if conv_dir.is_dir():
                    # Use the oldest file in the conversation as the conversation age
                    oldest_time = min(
                        file.stat().st_mtime 
                        for file in conv_dir.iterdir() 
                        if file.is_file()
                    )
                    conversations.append((oldest_time, conv_dir))
            
            conversations.sort()  # Sort by time (oldest first)
            
            # Remove oldest conversations until under limit
            for _, conv_dir in conversations:
                if current_size <= self.max_size:
                    break
                
                # Calculate conversation size
                conv_size = sum(
                    file.stat().st_size 
                    for file in conv_dir.iterdir()
                    if file.is_file()
                )
                
                # Remove conversation
                self.clear_conversation(conv_dir.name)
                current_size -= conv_size
                
                logger.info(f"Evicted conversation {conv_dir.name} ({conv_size} bytes) due to size limit")
                
        except Exception as e:
            logger.error(f"Error enforcing size limit: {str(e)}")


class HybridCache(ActivationCache):
    """Hybrid cache using memory for speed and disk for persistence."""
    
    def __init__(self, memory_cache: MemoryCache, disk_cache: DiskCache):
        """
        Initialize hybrid cache.
        
        Args:
            memory_cache: Fast memory cache
            disk_cache: Persistent disk cache
        """
        self.memory = memory_cache
        self.disk = disk_cache
        
    def store(self, conv_id: str, token_idx: int, activations: Dict[int, Dict[str, Any]]) -> bool:
        """Store in both memory and disk caches."""
        memory_success = self.memory.store(conv_id, token_idx, activations)
        disk_success = self.disk.store(conv_id, token_idx, activations)
        
        # Success if at least one cache worked
        return memory_success or disk_success
    
    def retrieve(self, conv_id: str, token_idx: int) -> Optional[Dict[int, Dict[str, Any]]]:
        """Retrieve from memory first, fall back to disk."""
        # Try memory first (fastest)
        result = self.memory.retrieve(conv_id, token_idx)
        if result is not None:
            return result
        
        # Fall back to disk
        result = self.disk.retrieve(conv_id, token_idx)
        if result is not None:
            # Promote to memory cache for future access
            self.memory.store(conv_id, token_idx, result)
        
        return result
    
    def clear_conversation(self, conv_id: str) -> bool:
        """Clear conversation from both caches."""
        memory_success = self.memory.clear_conversation(conv_id)
        disk_success = self.disk.clear_conversation(conv_id)
        
        return memory_success and disk_success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        memory_stats = self.memory.get_stats()
        disk_stats = self.disk.get_stats()
        
        return {
            "type": "hybrid",
            "memory": memory_stats,
            "disk": disk_stats
        }


def create_default_cache(memory_mb: int = 512, disk_gb: float = 5.0, cache_dir: str = "./cache") -> HybridCache:
    """
    Create a default hybrid cache configuration.
    
    Args:
        memory_mb: Memory cache size in MB
        disk_gb: Disk cache size in GB
        cache_dir: Disk cache directory
        
    Returns:
        Configured HybridCache instance
    """
    memory_cache = MemoryCache(max_size_mb=memory_mb)
    disk_cache = DiskCache(cache_dir=cache_dir, max_size_gb=disk_gb)
    
    return HybridCache(memory_cache, disk_cache)