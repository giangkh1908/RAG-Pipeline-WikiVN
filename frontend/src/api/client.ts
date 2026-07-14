import type { ChatResponse, StreamEvent } from '../types';

const API_BASE = '';
const SESSION_STORAGE_KEY = 'rag.session.id';

function generateSessionId(): string {
  const c: Crypto | undefined =
    typeof crypto !== 'undefined' ? (crypto as Crypto) : undefined;
  if (c && typeof c.randomUUID === 'function') {
    return c.randomUUID().replace(/-/g, '');
  }
  const bytes = new Uint8Array(16);
  if (c && typeof c.getRandomValues === 'function') {
    c.getRandomValues(bytes);
  } else {
    for (let i = 0; i < bytes.length; i += 1) {
      bytes[i] = Math.floor(Math.random() * 256);
    }
  }
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

export function getOrCreateSessionId(): string {
  try {
    const existing = sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (existing) return existing;
    const fresh = generateSessionId();
    sessionStorage.setItem(SESSION_STORAGE_KEY, fresh);
    return fresh;
  } catch {
    return generateSessionId();
  }
}

export function setSessionId(sessionId: string): void {
  try {
    sessionStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  } catch {
    /* sessionStorage unavailable */
  }
}

export function clearSessionId(): string {
  const fresh = generateSessionId();
  try {
    sessionStorage.setItem(SESSION_STORAGE_KEY, fresh);
  } catch {
    /* sessionStorage unavailable */
  }
  return fresh;
}

export async function chat(
  question: string,
  sessionId?: string | null,
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId ?? undefined }),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function chatStream(
  question: string,
  sessionId: string | null | undefined,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId ?? undefined }),
  });

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

export async function healthCheck(): Promise<{ status: string; qdrant: string }> {
  const res = await fetch(`${API_BASE}/api/health`);
  return res.json();
}

export async function fetchSuggestions(
  sessionId: string | null | undefined,
  lastQuestion: string,
  lastAnswer: string,
): Promise<{ suggestions: string[]; fallback: boolean }> {
  const res = await fetch(`${API_BASE}/api/suggestions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId ?? undefined,
      last_question: lastQuestion,
      last_answer: lastAnswer,
    }),
  });
  if (!res.ok) return { suggestions: [], fallback: true };
  return res.json();
}
