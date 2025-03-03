import { LogitLensRequest, LogitLensResponse, Message } from '../types/logitLens';

const API_BASE_URL = 'http://localhost:8000';

export const fetchLogitLens = async (
  messages: Message[],
  position: number
): Promise<LogitLensResponse> => {
  const request: LogitLensRequest = {
    messages,
    target_position: position
  };

  console.log('Sending logit lens request:', request);

  const response = await fetch(`${API_BASE_URL}/api/logit-lens`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    console.error('Error response:', error);
    throw new Error(error.detail || 'Failed to fetch logit lens data');
  }

  return response.json();
}; 