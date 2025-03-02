import React, { useRef, useEffect } from 'react';
import { useConversation } from '../../context/ConversationContext';
import MessageBubble from './MessageBubble';
import ChatInput from './ChatInput';

const ChatWindow: React.FC = () => {
  const { messages, isLoading } = useConversation();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-8">
            <p className="text-lg">Welcome to Logit Lens</p>
            <p className="text-sm mt-2">Start a conversation and click on any word in the response to see what the model was predicting at each layer.</p>
          </div>
        )}
        {messages.map((message, index) => (
          <MessageBubble 
            key={index} 
            message={message} 
            messageIndex={index} 
          />
        ))}
        {isLoading && (
          <div className="text-gray-500 italic">The model is thinking...</div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <ChatInput />
    </div>
  );
};

export default ChatWindow;