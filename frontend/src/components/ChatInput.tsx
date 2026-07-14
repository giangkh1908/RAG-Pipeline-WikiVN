import { useState, useRef, useEffect } from 'react';

interface Props {
  onSend: (message: string) => void;
  disabled: boolean;
  suggestions?: string[];
}

const MAX_CHARS = 500;
const WARN_THRESHOLD = MAX_CHARS - 50;

const DEFAULT_SUGGESTIONS = [
  'Vịnh Hạ Long nằm ở đâu?',
  'Du lịch Hội An nên đi mùa nào?',
  'Có món ăn đặc sản nào ở Đà Nẵng?',
  'Nha Trang có bãi biển nổi tiếng nào?',
];

export function ChatInput({ onSend, disabled, suggestions = [] }: Props) {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  }, [value]);

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const remaining = MAX_CHARS - value.length;
  const showCounter = value.length > WARN_THRESHOLD;

  return (
    <div className="border-t border-gray-100 bg-white pb-3 pt-2 sm:pb-4 sm:pt-2">
      {/* Suggestions — horizontal scroll on mobile */}
      {value === '' && !disabled && (
        <div className="mx-auto max-w-3xl px-3 pb-2.5 sm:px-4 sm:pb-3">
          <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-hide sm:flex-wrap sm:justify-center">
            {(suggestions.length > 0 ? suggestions : DEFAULT_SUGGESTIONS).map((s, i) => (
              <button
                key={i}
                onClick={() => onSend(s)}
                className="flex-shrink-0 rounded-full border border-gray-200 px-3 py-1.5 text-xs text-gray-500
                           hover:bg-gray-50 hover:border-gray-300 active:bg-gray-100 transition-colors"
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input area */}
      <div className="mx-auto max-w-3xl px-3 sm:px-4">
        <div className="flex items-end gap-1.5 rounded-2xl border border-gray-200 bg-white
                        px-3 py-2 sm:px-4 sm:py-2.5 focus-within:border-gray-400 transition-colors">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={e => setValue(e.target.value.slice(0, MAX_CHARS))}
            onKeyDown={handleKeyDown}
            placeholder="Hỏi bất cứ điều gì..."
            disabled={disabled}
            rows={1}
            maxLength={MAX_CHARS}
            className="flex-1 resize-none border-0 bg-transparent p-0 text-sm leading-relaxed
                       placeholder:text-gray-400 focus:outline-none focus:ring-0
                       disabled:text-gray-400 min-h-[24px] max-h-[160px]"
          />
          <button
            onClick={handleSubmit}
            disabled={disabled || !value.trim()}
            className="flex-shrink-0 rounded-xl p-1.5 transition-colors
                       disabled:text-gray-300 disabled:cursor-not-allowed
                       text-gray-400 hover:text-gray-600 hover:bg-gray-100
                       active:bg-gray-200"
          >
            {disabled ? (
              <svg className="h-5 w-5 animate-spin" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
              </svg>
            ) : (
              <svg className="h-5 w-5" viewBox="0 0 24 24" fill="currentColor">
                <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
              </svg>
            )}
          </button>
        </div>

        <div className="mt-1.5 flex items-center justify-between text-[11px] text-gray-400 sm:mt-2">
          <span className="flex-1 text-center">Hỏi đáp về du lịch Việt Nam</span>
          {showCounter && (
            <span className={remaining < 20 ? 'text-red-500' : ''}>
              {remaining}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
