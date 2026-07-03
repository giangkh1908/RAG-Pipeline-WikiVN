export interface Citation {
  doc_id: string;
  title: string;
  url: string;
  score: number;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  confidence: number;
  passages_used: number;
  latency_ms: number;
}

export interface StreamToken {
  type: 'token';
  content: string;
}

export interface StreamDone {
  type: 'done';
  citations: Citation[];
  confidence: number;
}

export interface StreamError {
  type: 'error';
  message: string;
}

export type StreamEvent = StreamToken | StreamDone | StreamError;

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  confidence?: number;
  latency_ms?: number;
  isStreaming?: boolean;
}
