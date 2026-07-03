import type { Citation } from '../types';

interface Props {
  citation: Citation;
  index: number;
}

export function CitationCard({ citation, index }: Props) {
  return (
    <a
      href={citation.url}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 rounded-md bg-gray-50 px-2 py-1 text-xs
                 border border-gray-200 hover:bg-gray-100 transition-colors group"
    >
      <span className="flex h-4 w-4 items-center justify-center rounded bg-gray-200 text-[10px]
                       font-semibold text-gray-500 group-hover:bg-gray-300">
        {index}
      </span>
      <span className="text-gray-600 group-hover:text-gray-800 truncate max-w-[150px]">
        {citation.title}
      </span>
    </a>
  );
}
