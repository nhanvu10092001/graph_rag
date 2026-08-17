import React, { useState } from 'react';
import { Brain, ChevronDown, ChevronRight } from 'lucide-react';
import MarkdownRenderer from './MarkdownRenderer';

interface ThinkingPanelProps {
  thinking: string;
}

export default function ThinkingPanel({ thinking }: ThinkingPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!thinking) return null;

  return (
    <div className="mb-3 border border-slate-200 rounded-xl overflow-hidden bg-slate-50/50">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-3 text-xs font-semibold text-slate-600 hover:bg-slate-100 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-slate-500" />
          <span>Quá trình suy nghĩ</span>
        </div>
        {isExpanded ? (
          <ChevronDown className="w-4 h-4 text-slate-400" />
        ) : (
          <ChevronRight className="w-4 h-4 text-slate-400" />
        )}
      </button>

      {isExpanded && (
        <div className="p-4 border-t border-slate-200 bg-white text-sm text-slate-600">
          <MarkdownRenderer content={thinking} />
        </div>
      )}
    </div>
  );
}