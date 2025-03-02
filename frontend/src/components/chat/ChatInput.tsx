import React, { useState } from 'react';
import { useConversation } from '../../context/ConversationContext';
import { Send } from 'lucide-react';

const ChatInput: React.FC = () => {
  const [inputText, setInputText] = useState('');
  const { addMessage, isLoading } = useConversation();
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputText.trim() && !isLoading) {
      addMessage(inputText);
      setInputText('');
    }
  };
  
  return (
    <form onSubmit={handleSubmit} className="p-4 border-t">
      <div className="flex">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Ask something..."
          disabled={isLoading}
          className="flex-1 p-2 border rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button
          type="submit"
          disabled={isLoading || !inputText.trim()}
          className="bg-blue-500 text-white px-4 py-2 rounded-r-lg hover:bg-blue-600 disabled:bg-gray-300 flex items-center justify-center"
        >
          <Send size={18} />
        </button>
      </div>
    </form>
  );
};

export default ChatInput;