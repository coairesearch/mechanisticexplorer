import React from 'react';
import { ConversationProvider } from './context/ConversationContext';
import MainLayout from './components/MainLayout';

function App() {
  return (
    <div className="min-h-screen bg-gray-100">
      <ConversationProvider>
        <MainLayout />
      </ConversationProvider>
    </div>
  );
}

export default App;