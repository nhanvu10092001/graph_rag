import React, { useState } from 'react';
import { Database, ChevronDown, ChevronRight, Loader2, CheckCircle2, AlertCircle } from 'lucide-react';
import { ToolCallState } from '../types';

interface ToolCallBadgeProps {
  toolCall: ToolCallState;
}

export const ToolCallBadge: React.FC<ToolCallBadgeProps> = ({ toolCall }) => {
  const [isOpen, setIsOpen] = useState(false);

  // Friendly human-readable tool name mapping
  const formatToolName = (name?: string) => {
    if (!name) return 'Đang khởi chạy công cụ...';
    if (name === 'query_knowledge_graph') return 'Truy vấn Neo4j Knowledge Graph';
    return name;
  };

  // Helper to extract query text or formatted inputs
  const getQueryPreview = (): string => {
    if (toolCall.input) {
      if (typeof toolCall.input === 'string') return toolCall.input;
      if (typeof toolCall.input === 'object' && toolCall.input.query) {
        return toolCall.input.query;
      }
      return JSON.stringify(toolCall.input);
    }
    if (toolCall.args) {
      try {
        const parsed = JSON.parse(toolCall.args);
        if (parsed.query) return parsed.query;
      } catch {
        // Fallback to raw streamed args string if incomplete JSON
      }
      return toolCall.args;
    }
    return '';
  };

  const queryPreview = getQueryPreview();

  return (
    <div className="my-2 border border-slate-200/80 rounded-xl bg-slate-50/70 overflow-hidden text-xs transition-all duration-200 shadow-sm max-w-full">
      {/* Header Bar */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3 py-2 text-left text-slate-700 hover:bg-slate-100/70 transition-colors focus:outline-none"
      >
        <div className="flex items-center gap-2 min-w-0 pr-2">
          {/* Status Icon */}
          {toolCall.status === 'completed' ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
          ) : toolCall.status === 'error' ? (
            <AlertCircle className="w-4 h-4 text-rose-500 shrink-0" />
          ) : (
            <Loader2 className="w-4 h-4 text-indigo-600 animate-spin shrink-0" />
          )}

          {/* Tool Icon & Title */}
          <div className="flex items-center gap-1.5 min-w-0">
            <Database className="w-3.5 h-3.5 text-indigo-500 shrink-0" />
            <span className="font-medium text-slate-800 truncate">
              {formatToolName(toolCall.name)}
            </span>
          </div>

          {/* Inline query preview string */}
          {queryPreview && (
            <span className="text-slate-400 font-mono truncate text-[11px] max-w-[200px] sm:max-w-[320px]">
              &quot;{queryPreview}&quot;
            </span>
          )}
        </div>

        <div className="flex items-center gap-1 text-slate-400 shrink-0">
          <span className="text-[10px] font-medium uppercase tracking-wider">
            {toolCall.status === 'completed'
              ? 'Hoàn tất'
              : toolCall.status === 'error'
              ? 'Thất bại'
              : 'Đang chạy'}
          </span>
          {isOpen ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
        </div>
      </button>

      {/* Collapsible Details Body */}
      {isOpen && (
        <div className="px-3 py-2 border-t border-slate-200/60 bg-white/80 font-mono text-[11px] text-slate-600 space-y-1.5">
          <div className="flex justify-between items-center text-slate-400 text-[10px] uppercase font-sans">
            <span>Tham số đầu vào (Input):</span>
            <span>Tool ID: {toolCall.id || toolCall.name || 'N/A'}</span>
          </div>
          <pre className="p-2 bg-slate-900 text-slate-200 rounded-lg overflow-x-auto whitespace-pre-wrap break-all leading-relaxed">
            {toolCall.input
              ? JSON.stringify(toolCall.input, null, 2)
              : toolCall.args || '(Đang nhận tham số stream...)'}
          </pre>
        </div>
      )}
    </div>
  );
};
