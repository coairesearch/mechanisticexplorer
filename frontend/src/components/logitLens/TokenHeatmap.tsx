import React from 'react';
import { LayerData, TokenWithProbability } from '../../types';

interface TokenHeatmapProps {
  lensData: LayerData[];
  actualToken: string;
  contextTokens: TokenWithProbability[];
}

const TokenHeatmap: React.FC<TokenHeatmapProps> = ({ lensData, actualToken, contextTokens }) => {
  // Include all context tokens plus the actual token
  const displayTokens = [...contextTokens.map(t => t.text), actualToken];
  
  // Create a map of token to its lens data
  const tokenLensMap = new Map<string, LayerData[]>();
  contextTokens.forEach(token => {
    if (token.lensData) {
      tokenLensMap.set(token.text, token.lensData);
    }
  });
  tokenLensMap.set(actualToken, lensData);
  
  // Get color for probability
  const getProbabilityColor = (probability: number) => {
    if (probability >= 90) return 'bg-blue-900 text-white';
    if (probability >= 70) return 'bg-blue-700 text-white';
    if (probability >= 50) return 'bg-blue-500 text-white';
    if (probability >= 30) return 'bg-blue-300';
    if (probability >= 10) return 'bg-blue-200';
    return 'bg-blue-100';
  };
  
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium mb-2">Layer Predictions Heatmap</h3>
        <p className="text-sm text-gray-600 mb-4">
          This heatmap shows how the probability for different tokens changes across layers.
        </p>
        
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse">
            <thead>
              <tr>
                <th className="p-2 border bg-gray-100 sticky left-0 z-10">Layer</th>
                {displayTokens.map(token => (
                  <th 
                    key={token} 
                    className={`p-2 border ${token === actualToken ? 'bg-blue-50 font-bold' : 'bg-gray-50'}`}
                  >
                    {token}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: 24 }, (_, i) => i).map(layerIndex => {
                return (
                  <tr key={layerIndex}>
                    <td className="p-2 border bg-gray-100 font-medium sticky left-0 z-10">
                      {layerIndex}
                    </td>
                    {displayTokens.map(token => {
                      const tokenLens = tokenLensMap.get(token);
                      const layerData = tokenLens?.[layerIndex];
                      const probability = layerData?.predictions.find(p => p.token === token)?.probability || 0;
                      
                      return (
                        <td 
                          key={token} 
                          className={`p-2 border text-center ${getProbabilityColor(probability)}`}
                          title={`${token}: ${probability}%`}
                        >
                          {probability > 0 ? `${probability}%` : '-'}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
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
                className={`p-2 rounded ${getProbabilityColor(token.probability)}`}
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