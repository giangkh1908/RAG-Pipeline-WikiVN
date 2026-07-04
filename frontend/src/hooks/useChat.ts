import { useState, useCallback } from 'react';
import type { ChatHistoryEntry, Message, StreamEvent } from '../types';
import { chatStream } from '../api/client';

let msgId = 0;
const nextId = () => `msg-${++msgId}`;

const MAX_HISTORY_TURNS = 5;

function extractHistory(messages: Message[]): ChatHistoryEntry[] {
  // Take last N completed turns (user + assistant pairs)
  const completed = messages.filter(m => !m.isStreaming);
  const history: ChatHistoryEntry[] = [];
  for (const m of completed) {
    if (m.role === 'user' || (m.role === 'assistant' && m.content)) {
      history.push({ role: m.role, content: m.content });
    }
  }
  // Limit to last MAX_HISTORY_TURNS * 2 entries (user + assistant per turn)
  return history.slice(-(MAX_HISTORY_TURNS * 2));
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const sendMessage = useCallback(async (question: string) => {
    if (isStreaming) return;

    const userMsg: Message = { id: nextId(), role: 'user', content: question };
    const assistantMsg: Message = {
      id: nextId(),
      role: 'assistant',
      content: '',
      isStreaming: true,
    };

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setIsStreaming(true);

    // Extract history from previous messages (before adding new ones)
    const history = extractHistory(messages);

    try {
      await chatStream(question, (event: StreamEvent) => {
        if (event.type === 'token') {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMsg.id
                ? { ...m, content: m.content + event.content }
                : m,
            ),
          );
        } else if (event.type === 'done') {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMsg.id
                ? {
                    ...m,
                    citations: event.citations,
                    confidence: event.confidence,
                    isStreaming: false,
                  }
                : m,
            ),
          );
        }
      }, history);
    } catch {
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantMsg.id
            ? { ...m, content: 'Lỗi kết nối server.', isStreaming: false }
            : m,
        ),
      );
    } finally {
      setIsStreaming(false);
    }
  }, [isStreaming, messages]);

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, isStreaming, sendMessage, clearMessages };
}
