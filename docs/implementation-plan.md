# Implementation Plan: Real Logit Lens with Caching

## Overview

This document outlines the plan to implement real logit lens functionality with efficient caching for multi-turn conversations in the Mechanistic Explorer.

## Current Issues

1. **Mock Data**: Currently using fake logit lens predictions
2. **Context Problem**: When clicking a token, we need the previous 15 tokens' activations (including from the question)
3. **Performance**: Larger models are slow; need caching for instant token clicks
4. **Scalability**: Multi-turn conversations can accumulate large amounts of activation data

## Proposed Solutions

### 1. Real Logit Lens Implementation

#### Architecture Changes

```python
# New data structure for activation storage
class TokenActivation:
    token_id: int
    token_text: str
    position: int  # Global position in conversation
    layer_activations: List[torch.Tensor]  # Raw activations per layer
    layer_predictions: List[LayerData]  # Top-k predictions per layer
    context_window: List[int]  # IDs of previous 15 tokens

class ConversationCache:
    tokens: List[TokenActivation]
    model_name: str
    created_at: datetime
    total_tokens: int
```

#### Implementation Steps

1. **Activation Extraction During Generation**
   - Hook into each layer during forward pass
   - Store raw activations for each token
   - Calculate top-k predictions immediately

2. **Context Window Management**
   - Track global token positions across entire conversation
   - When token clicked, retrieve previous 15 tokens from cache
   - Include tokens from both questions and answers

### 2. Caching Strategy

#### Two-Level Cache System

**Level 1: In-Memory Cache (Fast)**
- Store current conversation's activations
- Keep last N conversations in memory
- Immediate access for token clicks

**Level 2: Disk Cache (Scalable)**
- Serialize activations to disk for older conversations
- Use memory-mapped files for efficient access
- Compress using torch.save with compression

#### Cache Structure

```
cache/
├── conversations/
│   ├── conv_123456/
│   │   ├── metadata.json
│   │   ├── tokens.pkl
│   │   └── activations/
│   │       ├── layer_0.pt
│   │       ├── layer_1.pt
│   │       └── ...
```

### 3. Efficient Storage

#### Optimization Techniques

1. **Quantization**: Store activations as float16 instead of float32
2. **Selective Storage**: Only store top-k predictions, not full vocabulary
3. **Compression**: Use zlib compression for disk storage
4. **Lazy Loading**: Load only requested layers when needed

#### Memory Estimates

For GPT-2 (768 hidden dim, 12 layers):
- Per token: ~37KB (raw) → ~9KB (optimized)
- 1000 tokens: ~37MB → ~9MB
- With compression: ~3-4MB

For larger models (e.g., 7B params):
- Per token: ~200KB → ~50KB (optimized)
- 1000 tokens: ~200MB → ~50MB
- With compression: ~15-20MB

### 4. API Changes

#### New Endpoints

```python
# Generate with activation caching
POST /api/chat/with-activations
{
    "text": "...",
    "messages": [...],
    "cache_activations": true,
    "conversation_id": "optional-id"
}

# Retrieve activations for token
GET /api/activations/{conversation_id}/{token_position}
{
    "include_context": true,
    "context_size": 15,
    "layers": [0, 1, 2, ...]  # Optional: specific layers
}

# Manage cache
DELETE /api/cache/{conversation_id}
GET /api/cache/status
```

### 5. Frontend Updates

#### Progressive Loading
1. Show placeholder while loading activations
2. Stream layer predictions as they arrive
3. Cache visualizations client-side

#### Smart Prefetching
- Prefetch activations for nearby tokens
- Predict likely clicks based on hover

## Implementation Phases

### Phase 1: Basic Real Logit Lens (Week 1)
- Implement activation extraction
- Store in-memory for current conversation
- Update API to return real predictions

### Phase 2: Caching System (Week 2)
- Implement two-level cache
- Add conversation persistence
- Create cache management endpoints

### Phase 3: Optimization (Week 3)
- Add compression and quantization
- Implement lazy loading
- Optimize for larger models

### Phase 4: Advanced Features (Week 4)
- Add activation search
- Implement attention visualization
- Export functionality

## Technical Considerations

### Memory Management
- Set max cache size (e.g., 1GB)
- LRU eviction policy
- Monitor memory usage

### Concurrency
- Thread-safe cache access
- Async activation extraction
- Parallel layer processing

### Error Handling
- Graceful degradation if cache fails
- Fallback to on-demand computation
- Clear error messages

## Performance Targets

- Token click → visualization: <100ms
- Full conversation caching: <2s per 100 tokens
- Memory usage: <1GB for 10 conversations
- Disk usage: <100MB per conversation

## Testing Strategy

1. **Unit Tests**
   - Cache operations
   - Activation extraction
   - Compression/decompression

2. **Integration Tests**
   - Multi-turn conversations
   - Large model handling
   - Cache eviction

3. **Performance Tests**
   - Response time benchmarks
   - Memory usage monitoring
   - Concurrent access