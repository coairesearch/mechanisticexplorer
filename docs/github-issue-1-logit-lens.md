# Issue 1: Implement Real Logit Lens Extraction with Context Window

## Summary
Replace mock logit lens data with real activation extraction from model layers, including proper context window handling for clicked tokens.

## Problem
Currently, the application uses mock data for logit lens visualization. When a user clicks on a token (e.g., "capital" in the response "The capital of France is Paris"), the system should:
1. Extract real layer activations for that token
2. Show the previous 15 tokens as context (including tokens from the user's question)
3. Display actual predictions from each layer

## Requirements

### Functional Requirements
1. **Real Activation Extraction**
   - Hook into GPT-2 (and other models) during forward pass
   - Extract hidden states after each transformer layer
   - Apply LM head to get vocabulary predictions

2. **Context Window**
   - Track token positions globally across conversation
   - When token clicked, retrieve 15 previous tokens
   - Context should span across user/assistant messages

3. **API Updates**
   - Modify `/api/chat` to extract and store activations
   - Return real layer predictions instead of mock data
   - Include context token information

### Technical Requirements
- Use nnsight's tracing capabilities efficiently
- Minimize memory usage during extraction

## Implementation Details

### 1. Activation Extraction Module

```python
class LogitLensExtractor:
    def __init__(self, model: LanguageModel):
        self.model = model
        self.layers = model.transformer.h
        
    def extract_activations(self, text: str) -> Dict[int, LayerActivations]:
        """Extract activations for all tokens in text"""
        activations = {}
        
        with self.model.trace(text) as tracer:
            with tracer.invoke(text) as invoker:
                # Hook into each layer
                for layer_idx, layer in enumerate(self.layers):
                    # Get hidden states after layer
                    hidden = layer.output[0].save()
                    
                    # Apply LN + LM head for predictions
                    normalized = self.model.transformer.ln_f(hidden)
                    logits = self.model.lm_head(normalized)
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Store for each token position
                    activations[layer_idx] = {
                        'hidden': hidden,
                        'probs': probs
                    }
        
        return activations
```

### 2. Context Management

```python
class ConversationTokenizer:
    def __init__(self):
        self.global_tokens = []  # All tokens in conversation
        self.token_metadata = []  # Position, speaker, etc.
        
    def add_message(self, text: str, is_user: bool):
        """Add message tokens with global positioning"""
        tokens = tokenize(text)
        for token in tokens:
            self.global_tokens.append({
                'text': token.text,
                'position': len(self.global_tokens),
                'is_user': is_user,
                'message_boundary': token.position == 0
            })
    
    def get_context_window(self, position: int, size: int = 15):
        """Get previous N tokens from position"""
        start = max(0, position - size)
        return self.global_tokens[start:position + 1]
```

### 3. Updated API Response

```python
@app.post("/api/chat")
async def chat(request: ChatRequest):
    # ... existing code ...
    
    # Extract activations during generation
    extractor = LogitLensExtractor(model)
    
    # Generate response
    response_text = generate_response(prompt)
    
    # Extract activations for full context
    full_context = conversation_history + response_text
    activations = extractor.extract_activations(full_context)
    
    # Build tokens with real lens data
    for token_idx, token in enumerate(tokens):
        token.lens = build_lens_data(activations, token_idx)
        token.context_window = get_context_tokens(token_idx)
```

## Testing

1. **Unit Tests**
   - Verify activation shapes match model architecture
   - Test context window extraction across message boundaries
   - Validate probability distributions sum to 1

2. **Integration Tests**
   - Multi-turn conversation with 100+ tokens
   - Click tokens at different positions
   - Verify context includes user tokens

3. **Performance Tests**
   - Measure extraction time for different text lengths
   - Memory usage during activation storage
   - API response time with real data

## Acceptance Criteria

- [ ] Clicking any token shows real layer predictions
- [ ] Context window shows previous 15 tokens
- [ ] Context spans across user/assistant messages
- [ ] Works with different model architectures

## References
- [nnsight documentation](https://nnsight.net/)
- [Logit Lens paper](https://arxiv.org/abs/2104.03073)
- Current mock implementation in `backend/app/main.py`