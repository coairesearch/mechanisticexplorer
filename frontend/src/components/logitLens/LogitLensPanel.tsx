import React from 'react';
import { useConversation } from '../../context/ConversationContext';
import LayerPrediction from './LayerPrediction';
import TokenHeatmap from './TokenHeatmap';
import { X, Columns2, LayoutGrid } from 'lucide-react';

const LogitLensPanel: React.FC = () => {
  const { selectedToken, clearSelectedToken, viewMode, setViewMode } = useConversation();
  
  if (!selectedToken) return null;
  
  // Sort layers in descending order (highest layer first)
  const sortedLensData = [...selectedToken.lensData].sort((a, b) => b.layer - a.layer);
  
  return (
    <div className="border-l bg-white h-full p-4 overflow-y-auto">
      <div className="flex flex-col space-y-4 mb-4">
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-semibold">
            Logit Lens: "{selectedToken.tokenText}"
          </h2>
          <button 
            onClick={clearSelectedToken}
            className="text-gray-500 hover:text-gray-700 p-1 rounded-full hover:bg-gray-100"
            aria-label="Close panel"
          >
            <X size={20} />
          </button>
        </div>
        <div className="flex items-center justify-end space-x-2">
          <button 
            onClick={() => setViewMode('standard')}
            className={`p-2 rounded ${viewMode === 'standard' ? 'bg-blue-100 text-blue-600' : 'text-gray-500 hover:bg-gray-100'}`}
            title="Standard View"
          >
            <Columns2 size={18} />
          </button>
          <button 
            onClick={() => setViewMode('heatmap')}
            className={`p-2 rounded ${viewMode === 'heatmap' ? 'bg-blue-100 text-blue-600' : 'text-gray-500 hover:bg-gray-100'}`}
            title="Heatmap View"
          >
            <LayoutGrid size={18} />
          </button>
        </div>
      </div>
      
      {viewMode === 'standard' ? (
        <>
          <p className="text-sm text-gray-600 mb-4">
            This visualization shows what the model predicted at each layer before producing the final token.
          </p>
          
          <div className="space-y-2">
            {sortedLensData.map((layerData) => (
              <LayerPrediction 
                key={layerData.layer} 
                layerData={layerData} 
                actualToken={selectedToken.tokenText} 
              />
            ))}
          </div>
        </>
      ) : (
        <TokenHeatmap 
          lensData={sortedLensData} 
          actualToken={selectedToken.tokenText}
          contextTokens={selectedToken.contextTokens || []}
        />
      )}
    </div>
  );
};

export default LogitLensPanel;