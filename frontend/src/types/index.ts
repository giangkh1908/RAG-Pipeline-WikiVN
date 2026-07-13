export interface Source {
  citation: string;
  title: string;
  content: string;
  chunk_id: string;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
  intent: string;
  latency_ms: number;
}

export interface StreamProgress {
  type: 'progress';
  step: string;
  message: string;
}

export interface StreamToken {
  type: 'token';
  content: string;
}

export interface StreamDone {
  type: 'done';
  answer: string;
  sources: Source[];
  intent: string;
}

export interface StreamError {
  type: 'error';
  message: string;
}

export type StreamEvent = StreamProgress | StreamToken | StreamDone | StreamError;

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  intent?: string;
  latency_ms?: number;
  isStreaming?: boolean;
  progress?: string;
}
