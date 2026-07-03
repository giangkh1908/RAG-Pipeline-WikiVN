import type { ChatResponse, StreamEvent } from '../types';

const API_BASE = '';

export async function chat(question: string): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function chatStream(
  question: string,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  const params = new URLSearchParams({ question });
  const res = await fetch(`${API_BASE}/api/chat/stream?${params}`);

  if (!res.ok) throw new Error(`API error: ${res.status}`);

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      try {
        const event: StreamEvent = JSON.parse(line.slice(6));
        onEvent(event);
      } catch {
        // skip malformed JSON
      }
    }
  }
}

export async function healthCheck(): Promise<{ status: string; qdrant_connected: boolean }> {
  const res = await fetch(`${API_BASE}/api/health`);
  return res.json();
}
