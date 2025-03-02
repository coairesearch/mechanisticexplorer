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
    ? "bg-blue-100 ml-auto rounded-lg p-3 max-w-[75%] inline-block" 
    : "bg-gray-100 rounded-lg p-3 max-w-[75%] inline-block";
  
  const containerClass = isUser
    ? "flex flex-col items-end"
    : "flex flex-col items-start";

  return (
    <div className={containerClass}>
      <div className="text-[10px] text-gray-500 mb-1 px-1">
        {isUser ? 'User' : 'Assistant'}
      </div>
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
    </div>
  );
};

export default MessageBubble;