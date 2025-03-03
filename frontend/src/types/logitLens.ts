export interface Message {
    role: 'user' | 'assistant';
    content: string;
}

export interface LogitLensRequest {
    messages: Message[];
    target_position: number;
}

export interface LogitLensResponse {
    logit_probabilities: number[][];
    tokens: string[][];
    layers: number[];
    selected_token: string;
}

export interface TokenPosition {
    text: string;
    position: number;
} 