import type { Message } from '../types';
import { CitationCard } from './CitationCard';

interface Props {
  message: Message;
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user';

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-3xl bg-gray-100 px-3.5 py-2 sm:max-w-[70%] sm:px-4 sm:py-2.5">
          <p className="text-sm leading-relaxed text-gray-900 whitespace-pre-wrap">
            {message.content}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[90%] space-y-2.5 sm:max-w-[85%] sm:space-y-3">
        {/* Answer text */}
        <div className="text-sm leading-relaxed text-gray-800 whitespace-pre-wrap">
          {message.content}
          {message.isStreaming && message.content.length === 0 && (
            <span className="inline-flex gap-1 py-1">
              <span className="h-2 w-2 rounded-full bg-gray-400 animate-bounce [animation-delay:0ms]"></span>
              <span className="h-2 w-2 rounded-full bg-gray-400 animate-bounce [animation-delay:150ms]"></span>
              <span className="h-2 w-2 rounded-full bg-gray-400 animate-bounce [animation-delay:300ms]"></span>
            </span>
          )}
          {message.isStreaming && message.content.length > 0 && (
            <span className="ml-0.5 inline-block h-4 w-0.5 bg-gray-500 animate-pulse"></span>
          )}
        </div>

        {/* Citations */}
        {!message.isStreaming && message.citations && message.citations.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {message.citations.map((c, i) => (
              <CitationCard key={i} citation={c} index={i + 1} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
