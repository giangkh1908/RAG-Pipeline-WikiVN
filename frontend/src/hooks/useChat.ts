import { useCallback, useRef, useState } from 'react';
import type { Message, StreamEvent } from '../types';
import {
  chatStream,
  clearSessionId,
  fetchSuggestions,
  getOrCreateSessionId,
  setSessionId,
} from '../api/client';

let msgId = 0;
const nextId = () => `msg-${++msgId}`;

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const sessionIdRef = useRef<string>(getOrCreateSessionId());
  const lastDoneRef = useRef<{ question: string; answer: string } | null>(null);

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
    setSuggestions([]);

    let doneAnswer = '';

    try {
      await chatStream(question, sessionIdRef.current, (event: StreamEvent) => {
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
                ? {
                    ...m,
                    content: m.content + event.content,
                    progress: undefined,
                  }
                : m,
            ),
          );
        } else if (event.type === 'done') {
          if (event.session_id && event.session_id !== sessionIdRef.current) {
            sessionIdRef.current = event.session_id;
            setSessionId(event.session_id);
          }
          doneAnswer = event.answer.replace(/\s+$/, '');
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMsg.id
                ? {
                    ...m,
                    content: doneAnswer,
                    sources: event.sources,
                    intent: event.intent,
                    turn_no: event.turn_no ?? undefined,
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

      // Fetch contextual suggestions after the answer is done.
      if (doneAnswer) {
        lastDoneRef.current = { question, answer: doneAnswer };
        try {
          const result = await fetchSuggestions(
            sessionIdRef.current,
            question,
            doneAnswer,
          );
          if (result.suggestions.length > 0) {
            setSuggestions(result.suggestions);
          }
        } catch {
          // Silently fall back — ChatInput has its own defaults.
        }
      }
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

  const resetSession = useCallback(() => {
    sessionIdRef.current = clearSessionId();
    setMessages([]);
    setSuggestions([]);
    lastDoneRef.current = null;
  }, []);

  return { messages, isStreaming, suggestions, sendMessage, resetSession };
}
