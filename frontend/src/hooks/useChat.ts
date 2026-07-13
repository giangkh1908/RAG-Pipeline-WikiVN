import { useState, useCallback } from 'react';
import type { Message, StreamEvent } from '../types';
import { chatStream } from '../api/client';

let msgId = 0;
const nextId = () => `msg-${++msgId}`;

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

    try {
      await chatStream(question, (event: StreamEvent) => {
        if (event.type === 'progress') {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMsg.id
                ? { ...m, progress: event.message }
                : m,
            ),
          );
        } else if (event.type === 'token') {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMsg.id
                ? { ...m, content: m.content + event.content, progress: undefined }
                : m,
            ),
          );
        } else if (event.type === 'done') {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMsg.id
                ? {
                    ...m,
                    content: event.answer,
                    sources: event.sources,
                    intent: event.intent,
                    isStreaming: false,
                    progress: undefined,
                  }
                : m,
            ),
          );
        } else if (event.type === 'error') {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMsg.id
                ? { ...m, content: event.message, isStreaming: false, progress: undefined }
                : m,
            ),
          );
        }
      });
    } catch {
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantMsg.id
            ? { ...m, content: 'Lỗi kết nối server.', isStreaming: false, progress: undefined }
            : m,
        ),
      );
    } finally {
      setIsStreaming(false);
    }
  }, [isStreaming]);

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, isStreaming, sendMessage, clearMessages };
}
