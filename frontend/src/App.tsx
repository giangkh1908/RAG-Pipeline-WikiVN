import { useEffect, useRef } from 'react';
import { useChat } from './hooks/useChat';
import { ChatInput } from './components/ChatInput';
import { MessageBubble } from './components/MessageBubble';

export default function App() {
  const { messages, isStreaming, sendMessage, clearMessages } = useChat();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex h-[100dvh] flex-col bg-white">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-gray-100 px-4 py-2.5
                         sm:px-6">
        <h1 className="text-sm font-semibold text-gray-800 sm:text-base">
          Vietnam Tourism RAG
        </h1>
        {messages.length > 0 && (
          <button
            onClick={clearMessages}
            className="rounded-lg px-2.5 py-1 text-xs text-gray-400 hover:text-gray-600
                       hover:bg-gray-50 transition-colors sm:px-3"
          >
            Mới
          </button>
        )}
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto overscroll-contain">
        <div className="mx-auto max-w-3xl px-3 py-4 space-y-5 sm:px-4 sm:py-6 sm:space-y-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center py-16 sm:py-20">
              <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-full
                              bg-gray-100 sm:h-12 sm:w-12">
                <svg className="h-5 w-5 text-gray-400 sm:h-6 sm:w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                        d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
              </div>
              <p className="text-base font-medium text-gray-600 sm:text-lg">
                Hỏi bất cứ điều gì
              </p>
              <p className="mt-1 text-xs text-gray-400 sm:text-sm">
                Hỏi đáp về du lịch Việt Nam
              </p>
            </div>
          )}

          {messages.map(msg => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input */}
      <ChatInput onSend={sendMessage} disabled={isStreaming} />
    </div>
  );
}
