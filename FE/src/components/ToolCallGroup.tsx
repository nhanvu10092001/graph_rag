import React, { useState, useEffect, useRef } from 'react';
import { ChevronDown, ChevronRight, Loader2, CheckCircle2 } from 'lucide-react';
import { ToolCallBadge } from './ToolCallBadge';
import { ToolCallState } from '../types';

interface ToolCallGroupProps {
  toolCalls: Record<number, ToolCallState>;
  isStreaming: boolean;
}

export const ToolCallGroup: React.FC<ToolCallGroupProps> = ({ toolCalls, isStreaming }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const prevStreamingRef = useRef(isStreaming);
  const collapseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const entries = Object.values(toolCalls).filter(Boolean);
  const total = entries.length;
  const completedCount = entries.filter((tc) => tc.status === 'completed').length;
  const hasActive = entries.some((tc) => tc.status === 'executing' || tc.status === 'calling');
  const allCompleted = total > 0 && completedCount === total;

  useEffect(() => {
    if (isStreaming && hasActive) {
      setIsExpanded(true);
      if (collapseTimerRef.current) {
        clearTimeout(collapseTimerRef.current);
        collapseTimerRef.current = null;
      }
    }
  }, [isStreaming, hasActive]);

  useEffect(() => {
    if (allCompleted && prevStreamingRef.current && !hasActive) {
      collapseTimerRef.current = setTimeout(() => {
        setIsExpanded(false);
        collapseTimerRef.current = null;
      }, 1000);
    }
    prevStreamingRef.current = isStreaming;

    return () => {
      if (collapseTimerRef.current) {
        clearTimeout(collapseTimerRef.current);
      }
    };
  }, [allCompleted, isStreaming, hasActive]);

  if (total === 0) return null;

  const getSummary = () => {
    if (total === 1) {
      const tc = entries[0];
      const name = tc.name || 'tool';
      const status = tc.status === 'completed' ? 'completed' : tc.status === 'executing' ? 'executing...' : 'calling...';
      return `${name} — ${status}`;
    }
    if (allCompleted) {
      return `${total} tool calls completed`;
    }
    const parts: string[] = [];
    if (completedCount > 0) parts.push(`${completedCount} completed`);
    const executingCount = entries.filter((tc) => tc.status === 'executing').length;
    if (executingCount > 0) parts.push(`${executingCount} executing`);
    const callingCount = entries.filter((tc) => tc.status === 'calling').length;
    if (callingCount > 0) parts.push(`${callingCount} calling`);
    return `${total} tool calls (${parts.join(', ')})`;
  };

  return (
    <div className="border border-slate-200 rounded-xl bg-white overflow-hidden shadow-sm">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between px-3 py-2 text-left text-xs text-slate-700 hover:bg-slate-50 transition-colors focus:outline-none cursor-pointer"
      >
        <div className="flex items-center gap-2">
          {hasActive ? (
            <Loader2 className="w-3.5 h-3.5 text-indigo-600 animate-spin shrink-0" />
          ) : (
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
          )}
          <span className="font-medium text-slate-600">{getSummary()}</span>
        </div>
        <div className="shrink-0">
          {isExpanded ? (
            <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
          ) : (
            <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
          )}
        </div>
      </button>

      {isExpanded && (
        <div className="px-2 pb-2 space-y-1">
          {entries.map((tc, idx) => (
            <ToolCallBadge key={tc.id || idx} toolCall={tc} />
          ))}
        </div>
      )}
    </div>
  );
};
