import React, { useState } from 'react';
import { Database, ChevronDown, ChevronRight, Loader2, CheckCircle2, AlertCircle } from 'lucide-react';
import { ToolCallState } from '../types';

interface ToolCallBadgeProps {
  toolCall: ToolCallState;
}

export const ToolCallBadge: React.FC<ToolCallBadgeProps> = ({ toolCall }) => {
  const [isOpen, setIsOpen] = useState(false);

  const formatToolName = (name?: string) => {
    if (!name) return 'Đang khởi chạy công cụ...';
    if (name === 'query_knowledge_graph') return 'Truy vấn Neo4j Knowledge Graph';
    return name;
  };

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

  const statusConfig = {
    completed: { label: 'Hoàn tất', color: 'text-emerald-600 bg-emerald-50 border-emerald-100' },
    error: { label: 'Thất bại', color: 'text-rose-600 bg-rose-50 border-rose-100' },
    calling: { label: 'Đang gọi', color: 'text-indigo-600 bg-indigo-50 border-indigo-100' },
    executing: { label: 'Đang chạy', color: 'text-amber-600 bg-amber-50 border-amber-100' },
  };

  const status = statusConfig[toolCall.status] || statusConfig.calling;

  return (
    <div className="my-2 border border-slate-200 rounded-xl bg-white overflow-hidden text-xs transition-all duration-200 shadow-sm max-w-full">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3 py-2.5 text-left text-slate-700 hover:bg-slate-50 transition-colors focus:outline-none cursor-pointer"
      >
        <div className="flex items-center gap-2 min-w-0 pr-2">
          {toolCall.status === 'completed' ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
          ) : toolCall.status === 'error' ? (
            <AlertCircle className="w-4 h-4 text-rose-500 shrink-0" />
          ) : (
            <Loader2 className="w-4 h-4 text-indigo-600 animate-spin shrink-0" />
          )}

          <div className="flex items-center gap-1.5 min-w-0">
            <Database className="w-3.5 h-3.5 text-indigo-500 shrink-0" />
            <span className="font-semibold text-slate-800 truncate">
              {formatToolName(toolCall.name)}
            </span>
          </div>

          {queryPreview && (
            <span className="text-slate-400 font-mono truncate text-[10px] max-w-[180px] sm:max-w-[300px]">
              &quot;{queryPreview}&quot;
            </span>
          )}
        </div>

        <div className="flex items-center gap-1.5 shrink-0">
          <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold tracking-wider font-mono border ${status.color}`}>
            {status.label.toUpperCase()}
          </span>
          {isOpen ? <ChevronDown className="w-3.5 h-3.5 text-slate-400" /> : <ChevronRight className="w-3.5 h-3.5 text-slate-400" />}
        </div>
      </button>

      {isOpen && (
        <div className="px-3 py-2.5 border-t border-slate-100 bg-slate-50/50 font-mono text-[11px] text-slate-600 space-y-1.5">
          <div className="flex justify-between items-center text-slate-400 text-[10px] uppercase font-sans font-bold tracking-wider">
            <span>Tham số đầu vào</span>
            <span className="font-mono normal-case">{toolCall.id || toolCall.name || 'N/A'}</span>
          </div>
          <pre className="p-2.5 bg-slate-900 text-slate-200 rounded-lg overflow-x-auto whitespace-pre-wrap break-all leading-relaxed text-[10px]">
            {toolCall.input
              ? JSON.stringify(toolCall.input, null, 2)
              : toolCall.args || '(Đang nhận tham số stream...)'}
          </pre>
        </div>
      )}
    </div>
  );
};
