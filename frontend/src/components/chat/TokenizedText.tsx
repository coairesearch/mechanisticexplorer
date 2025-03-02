import React from 'react';
import { useConversation } from '../../context/ConversationContext';
import { Token } from '../../types';

interface TokenizedTextProps {
  tokens: Token[];
  messageIndex: number;
}

const TokenizedText: React.FC<TokenizedTextProps> = ({ tokens, messageIndex }) => {
  const { selectToken } = useConversation();
  
  // Handle clicking on a token to show lens data
  const handleTokenClick = (token: Token) => {
    selectToken(messageIndex, token.text, token.lens);
  };
  
  return (
    <div>
      {tokens.map((token, index) => {
        // For whitespace tokens, preserve them
        if (token.text === ' ') {
          return <span key={index}>{token.text}</span>;
        }
        
        // For punctuation, don't make clickable
        if (/^[.,!?;:]$/.test(token.text)) {
          return <span key={index}>{token.text}</span>;
        }
        
        // For regular tokens, make them interactive
        return (
          <React.Fragment key={index}>
            <span 
              className="cursor-pointer hover:bg-blue-200 underline decoration-dotted underline-offset-2"
              onClick={() => handleTokenClick(token)}
              title="Click to see layer predictions"
            >
              {token.text}
            </span>
            {/* Add a space after each token unless it's followed by punctuation */}
            {index < tokens.length - 1 && !/^[.,!?;:]$/.test(tokens[index + 1].text) && (
              <span> </span>
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
};

export default TokenizedText;