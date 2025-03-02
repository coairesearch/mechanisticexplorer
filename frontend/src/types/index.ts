export interface Message {
  text: string;
  isUser: boolean;
  timestamp: Date;
  tokens?: Token[];
  isError?: boolean;
}

export interface Token {
  text: string;
  lens: LayerData[];
}

export interface LayerData {
  layer: number;
  predictions: Prediction[];
}

export interface Prediction {
  token: string;
  probability: number;
  rank: number;
}

export interface TokenWithProbability {
  text: string;
  probability: number;
  lensData?: LayerData[];
}

export interface SelectedToken {
  messageIndex: number;
  tokenText: string;
  lensData: LayerData[];
  contextTokens?: TokenWithProbability[];
}

export type ViewMode = 'standard' | 'heatmap';

export interface ConversationContextType {
  messages: Message[];
  isLoading: boolean;
  selectedToken: SelectedToken | null;
  viewMode: ViewMode;
  addMessage: (text: string, isUser?: boolean) => Promise<void>;
  selectToken: (messageIndex: number, tokenText: string, lensData: LayerData[]) => void;
  clearSelectedToken: () => void;
  setViewMode: (mode: ViewMode) => void;
}