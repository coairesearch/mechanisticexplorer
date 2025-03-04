const API_BASE_URL = 'http://localhost:8000/api';

export interface ApiResponse<T> {
  data: T;
  error?: string;
}

export const sendMessage = async (message: string, conversationHistory: any[]): Promise<ApiResponse<any>> => {
  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: message,
        messages: conversationHistory,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return { data };
  } catch (error) {
    console.error('API Error:', error);
    return {
      data: null,
      error: error instanceof Error ? error.message : 'An unknown error occurred',
    };
  }
}; 