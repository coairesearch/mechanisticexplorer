# Issue 2: Implement Activation Caching for Multi-Turn Conversations

## Summary
Design and implement a caching system to store model activations for instant token analysis in multi-turn conversations, especially important for larger models with slow inference.

## Problem
For larger models (7B+ parameters), generating activations on-demand is too slow for interactive exploration. Users should be able to click any token in the conversation history and instantly see its logit lens visualization. This requires:
1. Caching all activations during generation
2. Efficient storage for potentially gigabytes of data
3. Fast retrieval for any token in the conversation

## Requirements

### Functional Requirements
1. **Activation Storage**
   - Cache all layer activations during text generation
   - Store both raw activations and top-k predictions
   - Support conversations with 1000+ tokens

2. **Instant Retrieval**
   - Click any historical token → immediate visualization
   - No re-computation needed
   - Work across page refreshes

3. **Memory Management**
   - Configurable cache size limits
   - Automatic eviction of old conversations
   - Option to persist important conversations

### Technical Requirements
- Two-tier cache: memory (fast) + disk (persistent)
- Support for models up to 70B parameters
- Handle concurrent access
- Graceful degradation when cache full

## Implementation Details

### 1. Cache Architecture

```python
from abc import ABC, abstractmethod
import pickle
import lz4.frame
from pathlib import Path

class ActivationCache(ABC):
    @abstractmethod
    def store(self, conv_id: str, token_idx: int, activations: Dict):
        pass
    
    @abstractmethod
    def retrieve(self, conv_id: str, token_idx: int) -> Optional[Dict]:
        pass

class MemoryCache(ActivationCache):
    def __init__(self, max_size_mb: int = 1024):
        self.cache = {}
        self.max_size = max_size_mb * 1024 * 1024
        self.current_size = 0
        self.access_times = {}  # For LRU
        
    def store(self, conv_id: str, token_idx: int, activations: Dict):
        key = f"{conv_id}:{token_idx}"
        
        # Compress activations
        compressed = self._compress(activations)
        size = len(compressed)
        
        # Evict if needed
        while self.current_size + size > self.max_size:
            self._evict_lru()
        
        self.cache[key] = compressed
        self.current_size += size
        self.access_times[key] = time.time()
    
    def _compress(self, activations: Dict) -> bytes:
        # Convert tensors to float16 for space saving
        compressed_acts = {}
        for layer, data in activations.items():
            compressed_acts[layer] = {
                'hidden': data['hidden'].half().cpu().numpy(),
                'top_k_tokens': data['top_k_tokens'],  # Already small
                'top_k_probs': data['top_k_probs'].half().cpu().numpy()
            }
        
        # Serialize and compress
        serialized = pickle.dumps(compressed_acts)
        return lz4.frame.compress(serialized)

class DiskCache(ActivationCache):
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def store(self, conv_id: str, token_idx: int, activations: Dict):
        conv_dir = self.cache_dir / conv_id
        conv_dir.mkdir(exist_ok=True)
        
        # Store each layer separately for efficient access
        for layer_idx, layer_data in activations.items():
            layer_file = conv_dir / f"layer_{layer_idx}_token_{token_idx}.pt"
            torch.save(layer_data, layer_file, pickle_protocol=4)
    
    def retrieve(self, conv_id: str, token_idx: int) -> Optional[Dict]:
        conv_dir = self.cache_dir / conv_id
        if not conv_dir.exists():
            return None
        
        activations = {}
        for layer_file in conv_dir.glob(f"layer_*_token_{token_idx}.pt"):
            layer_idx = int(layer_file.stem.split('_')[1])
            activations[layer_idx] = torch.load(layer_file)
        
        return activations if activations else None

class HybridCache(ActivationCache):
    def __init__(self, memory_cache: MemoryCache, disk_cache: DiskCache):
        self.memory = memory_cache
        self.disk = disk_cache
        
    def store(self, conv_id: str, token_idx: int, activations: Dict):
        # Store in both caches
        self.memory.store(conv_id, token_idx, activations)
        self.disk.store(conv_id, token_idx, activations)
    
    def retrieve(self, conv_id: str, token_idx: int) -> Optional[Dict]:
        # Try memory first
        result = self.memory.retrieve(conv_id, token_idx)
        if result:
            return result
        
        # Fall back to disk
        result = self.disk.retrieve(conv_id, token_idx)
        if result:
            # Promote to memory cache
            self.memory.store(conv_id, token_idx, result)
        
        return result
```

### 2. Integration with Generation

```python
class CachedLogitLensModel:
    def __init__(self, model: LanguageModel, cache: ActivationCache):
        self.model = model
        self.cache = cache
        self.conv_id = None
        
    def generate_with_caching(self, prompt: str, conv_id: str, 
                            existing_tokens: int = 0) -> str:
        self.conv_id = conv_id
        
        with self.model.generate(prompt, max_new_tokens=50) as generator:
            # Hook to save activations
            for layer_idx, layer in enumerate(self.model.transformer.h):
                layer.register_forward_hook(
                    lambda m, i, o: self._cache_activation(layer_idx, o)
                )
            
            output = generator.output.save()
        
        return self.model.tokenizer.decode(output[0])
    
    def _cache_activation(self, layer_idx: int, output):
        # Extract and cache for each token
        hidden_states = output[0]
        
        for token_idx in range(hidden_states.size(1)):
            token_hidden = hidden_states[:, token_idx, :]
            
            # Get top-k predictions
            logits = self.model.lm_head(self.model.transformer.ln_f(token_hidden))
            probs = torch.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = probs.topk(10)
            
            activation_data = {
                'hidden': token_hidden,
                'top_k_tokens': top_k_indices,
                'top_k_probs': top_k_probs
            }
            
            global_token_idx = self.existing_tokens + token_idx
            self.cache.store(self.conv_id, global_token_idx, 
                           {layer_idx: activation_data})
```

### 3. API Endpoints

```python
@app.post("/api/chat/cached")
async def chat_with_cache(request: CachedChatRequest):
    conv_id = request.conversation_id or str(uuid.uuid4())
    
    # Count existing tokens
    existing_tokens = sum(len(tokenize(msg.content)) 
                         for msg in request.messages)
    
    # Generate with caching
    cached_model = CachedLogitLensModel(model, hybrid_cache)
    response = cached_model.generate_with_caching(
        prompt, conv_id, existing_tokens
    )
    
    return {
        "conversation_id": conv_id,
        "response": response,
        "cached": True
    }

@app.get("/api/cache/token/{conv_id}/{token_idx}")
async def get_cached_activation(conv_id: str, token_idx: int, 
                               context_size: int = 15):
    # Retrieve from cache
    activations = hybrid_cache.retrieve(conv_id, token_idx)
    if not activations:
        raise HTTPException(404, "Activation not found")
    
    # Get context tokens
    context_tokens = []
    for ctx_idx in range(max(0, token_idx - context_size), token_idx):
        ctx_act = hybrid_cache.retrieve(conv_id, ctx_idx)
        if ctx_act:
            context_tokens.append({
                'position': ctx_idx,
                'predictions': extract_top_predictions(ctx_act)
            })
    
    return {
        "token_position": token_idx,
        "activations": format_activations(activations),
        "context": context_tokens
    }

@app.get("/api/cache/status")
async def cache_status():
    return {
        "memory_used_mb": memory_cache.current_size / 1024 / 1024,
        "memory_limit_mb": memory_cache.max_size / 1024 / 1024,
        "disk_conversations": len(list(disk_cache.cache_dir.iterdir())),
        "disk_used_mb": sum(f.stat().st_size for f in 
                           disk_cache.cache_dir.rglob('*')) / 1024 / 1024
    }
```

### 4. Frontend Integration

```typescript
// Frontend caching service
class ActivationCacheService {
  private conversationId: string;
  private tokenCache: Map<number, LayerData[]>;
  
  async onTokenClick(tokenPosition: number) {
    // Check local cache first
    if (this.tokenCache.has(tokenPosition)) {
      return this.tokenCache.get(tokenPosition);
    }
    
    // Fetch from backend
    const response = await fetch(
      `/api/cache/token/${this.conversationId}/${tokenPosition}?context_size=15`
    );
    
    const data = await response.json();
    
    // Cache locally
    this.tokenCache.set(tokenPosition, data.activations);
    
    // Prefetch nearby tokens
    this.prefetchNearby(tokenPosition);
    
    return data;
  }
  
  private async prefetchNearby(position: number) {
    const range = 5;
    for (let i = position - range; i <= position + range; i++) {
      if (i !== position && !this.tokenCache.has(i)) {
        // Prefetch in background
        fetch(`/api/cache/token/${this.conversationId}/${i}`)
          .then(r => r.json())
          .then(data => this.tokenCache.set(i, data.activations));
      }
    }
  }
}
```

## Storage Estimates

### GPT-2 (768 hidden, 12 layers)
- Per token: ~37KB raw → ~9KB compressed
- 1000 tokens: ~9MB compressed
- Full conversation: ~3-5MB on disk

### 7B Model (4096 hidden, 32 layers)
- Per token: ~524KB raw → ~131KB compressed  
- 1000 tokens: ~131MB compressed
- Full conversation: ~40-50MB on disk

### 70B Model (8192 hidden, 80 layers)
- Per token: ~2.6MB raw → ~650KB compressed
- 1000 tokens: ~650MB compressed
- Full conversation: ~200-250MB on disk

## Testing

1. **Performance Tests**
   - Cache write speed during generation
   - Cache read speed for random access
   - Memory usage under load

2. **Stress Tests**
   - 10 concurrent conversations
   - 10,000 token conversation
   - Cache eviction behavior

3. **Integration Tests**
   - Multi-turn conversation flow
   - Page refresh persistence
   - Cache miss handling

## Acceptance Criteria

- [ ] Activations cached during generation with <10% overhead
- [ ] Token click → visualization in <100ms
- [ ] Support conversations up to 10,000 tokens
- [ ] Automatic cache eviction when full
- [ ] Persist across page refreshes
- [ ] Work with models up to 70B parameters
- [ ] Graceful degradation when cache unavailable

## Configuration Options

```yaml
cache:
  memory:
    max_size_mb: 1024
    eviction_policy: lru
  disk:
    max_size_gb: 10
    directory: ./cache
    compression: lz4
  prefetch:
    enabled: true
    range: 5
  ttl:
    memory_minutes: 60
    disk_days: 7
```

## Future Enhancements

1. **Distributed Caching**: Redis/Memcached for multi-server
2. **Selective Caching**: Only cache "interesting" tokens
3. **Progressive Loading**: Stream activations as available
4. **Export/Import**: Save conversations for sharing