import React from 'react';
import { LayerData } from '../../types';

interface LayerPredictionProps {
  layerData: LayerData;
  actualToken: string;
}

const LayerPrediction: React.FC<LayerPredictionProps> = ({ layerData, actualToken }) => {
  const { layer, predictions } = layerData;
  
  // Format layer number with padding for alignment
  const layerNum = `Layer ${layer.toString().padStart(2, '0')}`;
  
  return (
    <div className="border rounded p-2 hover:bg-gray-50 transition-colors">
      <div className="font-medium text-gray-700">{layerNum}</div>
      <div className="mt-1">
        {predictions.map((pred, index) => {
          // Determine if this prediction matches the final token
          const isActual = pred.token === actualToken;
          
          // Style based on whether it's the actual token and its rank
          const tokenStyle = isActual 
            ? "font-bold text-blue-700" 
            : "text-gray-800";
          
          // Calculate width for probability bar
          const barWidth = `${pred.probability}%`;
          
          return (
            <div key={index} className="flex items-center mb-1">
              <div className="w-8 text-xs text-gray-500">
                {pred.rank}.
              </div>
              <div className={`flex-1 ${tokenStyle}`}>
                {pred.token}
              </div>
              <div className="w-12 text-right text-xs">
                {pred.probability}%
              </div>
              <div className="w-20 ml-2 bg-gray-200 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full ${isActual ? 'bg-blue-500' : 'bg-gray-400'}`}
                  style={{ width: barWidth }}
                ></div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default LayerPrediction;