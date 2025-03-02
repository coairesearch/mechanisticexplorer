// Simulates message submission and returns model response with token lens data
export const sendMessage = async (message: string, conversationHistory: any[]) => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  // For each response, generate dummy token data with lens information
  const response = generateDummyResponse(message, conversationHistory);
  const tokens = tokenizeResponse(response);
  const userTokens = tokenizeResponse(message);
  
  return {
    text: response,
    tokens: tokens.map(token => ({
      text: token,
      lens: generateLogitLensData(token)
    })),
    userTokens: userTokens.map(token => ({
      text: token,
      lens: generateLogitLensData(token)
    }))
  };
};

// Generate dummy logit lens data for a token
const generateLogitLensData = (token: string) => {
  // Create simulated layers (e.g., 24 layers)
  const layers = [];
  for (let i = 0; i < 24; i++) {
    // Generate top 3 predictions for each layer
    // Earlier layers have more variety, later layers converge to the actual token
    const predictions = [];
    
    // Add the actual token with increasing probability as layers progress
    const actualTokenProb = 0.1 + (i / 24) * 0.89; // Increases from 10% to 99%
    
    // Add some alternatives with decreasing probability
    if (i < 20) { // In early layers, show alternatives
      const alternatives = getAlternativeTokens(token);
      predictions.push(
        { token: alternatives[0], probability: (1 - actualTokenProb) * 0.7 },
        { token: alternatives[1], probability: (1 - actualTokenProb) * 0.3 }
      );
    }
    
    // Add the actual token (with higher rank in later layers)
    predictions.push({ token, probability: actualTokenProb });
    
    // Sort by probability
    predictions.sort((a, b) => b.probability - a.probability);
    
    layers.push({
      layer: i,
      predictions: predictions.map((p, idx) => ({
        ...p,
        rank: idx + 1,
        probability: parseFloat((p.probability * 100).toFixed(1))
      }))
    });
  }
  
  return layers;
};

// Helper functions for generating responses and alternative tokens
function generateDummyResponse(message: string, conversationHistory?: any[]) {
  // Simple rule-based responses for demo purposes
  if (message.toLowerCase().includes("capital of france")) {
    return "The capital of France is Paris.";
  }
  if (message.toLowerCase().includes("hello") || message.toLowerCase().includes("hi")) {
    return "Hello! How can I help you today?";
  }
  if (message.toLowerCase().includes("weather")) {
    return "The weather forecast shows partly cloudy skies with a high of 72°F.";
  }
  if (message.toLowerCase().includes("recommend") || message.toLowerCase().includes("suggestion")) {
    return "I would recommend trying the new machine learning course that was just released last month.";
  }
  // Add more canned responses as needed
  return "I understand your question about " + message.split(" ").slice(-3).join(" ") + ". Let me think about that.";
}

function tokenizeResponse(text: string) {
  // Simple word-based tokenization for the prototype
  // Improved to properly handle spaces and punctuation
  return text.split(/([.,!?;:]|\s+)/).filter(t => t !== "");
}

function getAlternativeTokens(token: string) {
  // Map of common tokens to plausible alternatives
  const alternatives: Record<string, string[]> = {
    "Paris": ["London", "Rome", "Berlin"],
    "France": ["Europe", "country", "nation"],
    "is": ["was", "remains", "becomes"],
    "the": ["a", "this", "that"],
    "capital": ["city", "metropolis", "center"],
    "weather": ["climate", "forecast", "conditions"],
    "cloudy": ["sunny", "rainy", "clear"],
    "high": ["temperature", "maximum", "peak"],
    "recommend": ["suggest", "advise", "propose"],
    "machine": ["deep", "artificial", "computer"],
    "learning": ["intelligence", "education", "training"]
    // Add more mappings as needed
  };
  
  // Return alternatives or generic alternatives if not found
  return alternatives[token] || ["word_" + token, "the", "and"];
}