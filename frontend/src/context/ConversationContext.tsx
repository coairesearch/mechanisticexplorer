import React, { createContext, useContext, useState } from 'react';
import { sendMessage } from '../api/api';
import { Message, SelectedToken, ConversationContextType, LayerData, ViewMode } from '../types';

const ConversationContext = createContext<ConversationContextType | undefined>(undefined);

export const ConversationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedToken, setSelectedToken] = useState<SelectedToken | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('standard');
  
  const addMessage = async (text: string, isUser = true) => {
    // Add user message (will be updated with tokens after API response)
    const userMessage: Message = { text, isUser, timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    
    if (isUser) {
      // Get response from API
      setIsLoading(true);
      try {
        const conversationHistory = messages.map(m => ({ 
          role: m.isUser ? 'user' : 'assistant', 
          content: m.text 
        }));
        
        const { data, error } = await sendMessage(text, conversationHistory);
        
        if (error || !data) {
          throw new Error(error || 'Failed to get response');
        }
        
        // Update the user message with tokens
        setMessages(prev => prev.map((msg, idx) => 
          idx === prev.length - 1 && msg.isUser
            ? { ...msg, tokens: data.userTokens }
            : msg
        ));
        
        // Add the model's response
        const modelMessage: Message = {
          text: data.text,
          isUser: false,
          timestamp: new Date(),
          tokens: data.tokens
        };
        
        setMessages(prev => [...prev, modelMessage]);
      } catch (error) {
        console.error("Error getting response:", error);
        // Add error message
        setMessages(prev => [...prev, {
          text: "Sorry, I encountered an error. Please try again.",
          isUser: false,
          timestamp: new Date(),
          isError: true
        }]);
      } finally {
        setIsLoading(false);
      }
    }
  };
  
  const selectToken = (messageIndex: number, tokenText: string, lensData: LayerData[]) => {
    const contextTokens = [];
    let tokensNeeded = 15;
    let currentMessageIndex = messageIndex;
    
    // Start with the current message
    while (tokensNeeded > 0 && currentMessageIndex >= 0) {
      const message = messages[currentMessageIndex];
      if (message && message.tokens) {
        let tokenIndex = currentMessageIndex === messageIndex ? 
          message.tokens.findIndex(t => t.text === tokenText) :
          message.tokens.length;
        
        // Get tokens from this message
        while (tokenIndex > 0 && tokensNeeded > 0) {
          tokenIndex--;
          const token = message.tokens[tokenIndex];
          if (token.text.trim() !== '' && token.text !== ' ') {
            contextTokens.unshift({
              text: token.text,
              probability: token.lens[token.lens.length - 1].predictions[0].probability,
              lensData: token.lens
            });
            tokensNeeded--;
          }
        }
      }
      currentMessageIndex--;
    }
    
    setSelectedToken({
      messageIndex,
      tokenText,
      lensData,
      contextTokens
    });
  };
  
  const clearSelectedToken = () => {
    setSelectedToken(null);
  };
  
  return (
    <ConversationContext.Provider value={{
      messages,
      isLoading,
      selectedToken,
      viewMode,
      addMessage,
      selectToken,
      clearSelectedToken,
      setViewMode
    }}>
      {children}
    </ConversationContext.Provider>
  );
};

export const useConversation = (): ConversationContextType => {
  const context = useContext(ConversationContext);
  if (context === undefined) {
    throw new Error('useConversation must be used within a ConversationProvider');
  }
  return context;
};