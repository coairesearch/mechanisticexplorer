import React from 'react';
import { LayerData, TokenWithProbability } from '../../types';

interface TokenHeatmapProps {
  lensData: LayerData[];
  actualToken: string;
  contextTokens: TokenWithProbability[];
}

const TokenHeatmap: React.FC<TokenHeatmapProps> = ({ lensData, actualToken, contextTokens }) => {
  // Get color for probability
  const getProbabilityColor = (probability: number) => {
    if (probability >= 90) return 'bg-blue-900 text-white';
    if (probability >= 70) return 'bg-blue-700 text-white';
    if (probability >= 50) return 'bg-blue-500 text-white';
    if (probability >= 30) return 'bg-blue-300';
    if (probability >= 10) return 'bg-blue-200';
    return 'bg-blue-100';
  };

  // Sort layers in descending order (highest layer first)
  const sortedLayers = [...lensData].sort((a, b) => b.layer - a.layer);
  
  // Get all context tokens plus the current token
  const allTokens = [...contextTokens, { text: actualToken, lensData, probability: 0 }];
  
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium mb-2">Layer Predictions Heatmap</h3>
        <p className="text-sm text-gray-600 mb-4">
          This heatmap shows how the model's predictions evolve through the layers for each token position.
          Color intensity indicates prediction probability.
        </p>
        
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse">
            <thead>
              <tr>
                <th className="p-2 border bg-gray-100 sticky left-0 z-10 text-xs">Layer</th>
                {allTokens.map((token, idx) => (
                  <th 
                    key={idx} 
                    className={`p-2 border bg-gray-50 text-xs ${token.text === actualToken ? 'font-bold bg-blue-50' : ''}`}
                  >
                    {token.text}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedLayers.map(layer => (
                <tr key={layer.layer}>
                  <td className="p-2 border bg-gray-100 font-medium sticky left-0 z-10 text-xs">
                    {layer.layer}
                  </td>
                  {allTokens.map((token, idx) => {
                    const tokenLensData = token.lensData || lensData;
                    const layerData = tokenLensData.find(l => l.layer === layer.layer);
                    const topPrediction = layerData?.predictions[0];
                    
                    return (
                      <td 
                        key={idx}
                        className={`p-2 border text-center text-xs ${getProbabilityColor(topPrediction?.probability || 0)}`}
                        title={`${topPrediction?.token}: ${topPrediction?.probability.toFixed(1)}%`}
                      >
                        <div className="font-medium whitespace-nowrap overflow-hidden text-ellipsis" style={{ maxWidth: '100px' }}>
                          {topPrediction?.token || '-'}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      {contextTokens.length > 0 && (
        <div>
          <h3 className="text-lg font-medium mb-2">Context Tokens</h3>
          <p className="text-sm text-gray-600 mb-4">
            The last 15 tokens before the selected token and their probabilities.
          </p>
          
          <div className="flex flex-wrap gap-2">
            {contextTokens.map((token, index) => (
              <div 
                key={index}
                className={`p-2 rounded text-xs ${getProbabilityColor(token.probability)}`}
                title={`Probability: ${token.probability}%`}
              >
                {token.text}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default TokenHeatmap;