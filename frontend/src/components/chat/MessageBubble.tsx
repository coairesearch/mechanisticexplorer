import React from 'react';
import { Message } from '../../types';
import TokenizedText from './TokenizedText';

interface MessageBubbleProps {
  message: Message;
  messageIndex: number;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message, messageIndex }) => {
  const { isUser, text, tokens, isError } = message;
  
  // Different styling based on the message source
  const bubbleClass = isUser 
    ? "bg-blue-100 ml-auto rounded-lg p-3 max-w-[75%]" 
    : "bg-gray-100 rounded-lg p-3 max-w-[75%]";
  
  return (
    <div className={`${bubbleClass} ${isError ? 'bg-red-100' : ''}`}>
      {!tokens ? (
        <p>{text}</p>
      ) : (
        <TokenizedText 
          tokens={tokens} 
          messageIndex={messageIndex} 
        />
      )}
    </div>
  );
};

export default MessageBubble;