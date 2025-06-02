# Mechanistic Explorer - Usage Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Understanding the Interface](#understanding-the-interface)
3. [Using the Logit Lens Feature](#using-the-logit-lens-feature)
4. [Visualization Modes](#visualization-modes)
5. [Advanced Features](#advanced-features)
6. [API Reference](#api-reference)
7. [Troubleshooting](#troubleshooting)

## Getting Started

### First Launch

1. Ensure both backend and frontend servers are running:
   ```bash
   # Terminal 1 - Backend
   cd backend && python run.py
   
   # Terminal 2 - Frontend
   cd frontend && npm run dev
   ```

2. Open your browser to `http://localhost:5173`

3. You'll see the chat interface with "Logit Lens Explorer" header

### Basic Interaction

1. **Send a Message**: Type in the chat input and press Enter
2. **View Response**: The model will generate a response token by token
3. **Explore Tokens**: Click on any token (highlighted on hover) to see its logit lens analysis

## Understanding the Interface

### Main Components

#### Chat Window
- **Message History**: Shows all messages in the conversation
- **Token Highlighting**: Hover over tokens to see they're clickable
- **Context Preservation**: The model remembers previous messages

#### Logit Lens Panel
- **Opens on Token Click**: Appears on the right side when you click a token
- **Layer Predictions**: Shows what the model predicted at each layer
- **Probability Scores**: Displays confidence percentages

### Color Coding

- **High Probability (>70%)**: Green
- **Medium Probability (30-70%)**: Yellow
- **Low Probability (<30%)**: Red

## Using the Logit Lens Feature

### What is Logit Lens?

The logit lens technique allows you to see what the model would predict at each intermediate layer, not just the final output. This helps understand how the model builds up its understanding.

### How to Use It

1. **Click a Token**: Any token in either user or assistant messages
2. **View Predictions**: See top 5 predictions at each layer
3. **Track Evolution**: Notice how predictions change from layer 0 to 11

### Understanding the Display

Each layer shows:
- **Layer Number**: From 0 (earliest) to 11 (final)
- **Top Predictions**: Up to 5 most likely tokens
- **Probabilities**: Percentage chance for each prediction

Example:
```
Layer 11 (Final):
1. "Paris" - 89.2%
2. "London" - 4.3%
3. "France" - 2.1%
```

## Visualization Modes

### Standard View
- **List Format**: Shows predictions in a vertical list
- **Detailed Information**: Full token names and exact probabilities
- **Easy Comparison**: Compare predictions across layers

### Heatmap View
- **Visual Overview**: Color-coded grid showing probability distributions
- **Pattern Recognition**: Quickly spot where predictions stabilize
- **Context Tokens**: Shows surrounding tokens for context

To switch modes, use the toggle buttons in the panel header.

## Advanced Features

### Multi-turn Conversations

The system maintains conversation context:

```
User: What is the capital of France?
Assistant: The capital of France is Paris.
User: What about Germany?
Assistant: The capital of Germany is Berlin.
```

The model remembers previous exchanges and uses them for context.

### Token Analysis Patterns

Look for these patterns in logit lens:

1. **Early Uncertainty**: Lower layers often show diverse predictions
2. **Convergence**: Higher layers typically converge on final answer
3. **Sudden Shifts**: Sometimes predictions change dramatically at specific layers

### Comparing Tokens

To compare how different tokens are processed:
1. Click on a token to open its analysis
2. Note the patterns
3. Close the panel (X button)
4. Click another token to compare

## API Reference

### Chat Endpoint

**POST** `/api/chat`

Request body:
```json
{
  "text": "Your message here",
  "messages": [
    {"role": "user", "content": "Previous message"},
    {"role": "assistant", "content": "Previous response"}
  ]
}
```

Response includes:
- `text`: Generated response
- `tokens`: Response tokens with logit lens data
- `userTokens`: Input tokens with logit lens data

### Model Information

**GET** `/api/model_info`

Returns:
```json
{
  "model_name": "openai-community/gpt2",
  "num_layers": 12,
  "vocab_size": 50257,
  "device": "cuda:0"
}
```

## Troubleshooting

### Performance Issues

**Slow Response Times**
- First request loads the model (can take 30-60 seconds)
- Subsequent requests should be faster
- Consider using GPU if available

**Memory Issues**
- GPT-2 requires ~2GB RAM
- Close other applications if needed
- Monitor system resources

### Display Issues

**Tokens Not Clickable**
- Ensure JavaScript is enabled
- Try refreshing the page
- Check browser console for errors

**Panel Not Opening**
- Click directly on the token text
- Ensure backend is running
- Check network tab for API errors

### API Errors

**500 Internal Server Error**
- Check backend logs
- Ensure model loaded correctly
- Verify Python dependencies

**CORS Errors**
- Ensure frontend runs on `localhost:5173`
- Backend should be on `localhost:8000`
- Check CORS configuration

## Tips for Research

1. **Focus on Transitions**: Pay attention to layers where predictions change significantly
2. **Context Matters**: The same token can have different predictions based on context
3. **Early vs Late Layers**: Early layers often capture syntax, later layers capture semantics
4. **Probability Patterns**: High confidence isn't always in final layers

## Keyboard Shortcuts

- **Enter**: Send message
- **Escape**: Close logit lens panel (when focused)
- **Click Outside**: Also closes the panel

## Best Practices

1. **Start Simple**: Begin with short, clear prompts
2. **Build Context**: Use multi-turn conversations for complex topics
3. **Compare Similar Tokens**: Click on related words to see processing differences
4. **Document Findings**: Screenshot interesting patterns for later analysis

## Further Resources

- [nnsight Documentation](https://nnsight.net/)
- [Logit Lens Paper](https://arxiv.org/abs/2104.03073)
- [GPT-2 Model Card](https://huggingface.co/openai-community/gpt2)