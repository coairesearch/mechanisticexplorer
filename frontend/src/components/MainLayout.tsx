import React, { useState } from 'react';
import ChatWindow from './chat/ChatWindow';
import LogitLensPanel from './logitLens/LogitLensPanel';
import { useConversation } from '../context/ConversationContext';
import { Brain, Columns as LayoutColumns, LayoutGrid } from 'lucide-react';

const MainLayout: React.FC = () => {
  const { selectedToken, viewMode, setViewMode } = useConversation();
  const [panelWidth, setPanelWidth] = useState(33); // Default width 33%
  const hasSidePanel = !!selectedToken;
  
  const handleResize = (e: React.MouseEvent<HTMLDivElement>) => {
    const startX = e.clientX;
    const startWidth = panelWidth;
    
    const handleMouseMove = (moveEvent: MouseEvent) => {
      const containerWidth = window.innerWidth;
      const newWidth = Math.min(Math.max(20, startWidth + ((startX - moveEvent.clientX) / containerWidth * 100)), 60);
      setPanelWidth(newWidth);
    };
    
    const handleMouseUp = () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
    
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };
  
  return (
    <div className="flex h-screen bg-gray-50">
      <div 
        className={`transition-all duration-300 ${hasSidePanel ? '' : 'w-full'}`} 
        style={{ width: hasSidePanel ? `${100 - panelWidth}%` : '100%' }}
      >
        <div className="bg-white h-full flex flex-col shadow">
          <header className="p-4 border-b flex items-center justify-between">
            <div className="flex items-center">
              <Brain className="text-blue-600 mr-2" size={24} />
              <h1 className="text-xl font-bold">Logit Lens Explorer</h1>
            </div>
            <div className="text-sm text-gray-500">
              Visualize model layer predictions
            </div>
          </header>
          <ChatWindow />
        </div>
      </div>
      
      {hasSidePanel && (
        <>
          <div 
            className="w-1 bg-gray-300 hover:bg-blue-400 cursor-col-resize relative z-10"
            onMouseDown={handleResize}
          >
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="h-8 w-1 bg-gray-400 rounded"></div>
            </div>
          </div>
          <div style={{ width: `${panelWidth}%` }}>
            <LogitLensPanel />
          </div>
        </>
      )}
    </div>
  );
};

export default MainLayout;