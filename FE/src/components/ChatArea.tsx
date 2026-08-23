/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  Send, Bot, User, Loader2, Sparkles,
  Menu, Terminal, HelpCircle, FileText, ArrowDown,
  Check, CheckCheck, Clock, Square
} from 'lucide-react';
import { ChatSession, Message } from '../types';
import MarkdownRenderer from './MarkdownRenderer';
import { ToolCallGroup } from './ToolCallGroup';
import ThinkingPanel from './ThinkingPanel';

interface ChatAreaProps {
  session: ChatSession | null;
  onSendMessage: (text: string) => void;
  isStreaming: boolean;
  onToggleMobile: () => void;
  onStopGeneration?: () => void;
}

const STARTER_PROMPTS = [
  {
    icon: <Terminal className="w-4 h-4 text-indigo-500" />,
    label: "Write Code",
    detail: "Create a JavaScript string reversal function with detailed error handling and optimization.",
    prompt: "Write a JavaScript function to reverse a string with handling for invalid input, along with detailed explanation."
  },
  {
    icon: <HelpCircle className="w-4 h-4 text-sky-500" />,
    label: "Explain Concept",
    detail: "Explain how WebSockets work compared to REST APIs.",
    prompt: "Please explain how WebSockets work compared to traditional REST APIs, and when a business should prioritize each."
  },
  {
    icon: <FileText className="w-4 h-4 text-amber-500" />,
    label: "Draft Document",
    detail: "Write an email to a business partner explaining a server outage and compensation policy.",
    prompt: "Write a professional email to a business partner explaining this morning's cloud server outage and proposing a service fee refund policy."
  },
  {
    icon: <Sparkles className="w-4 h-4 text-violet-500" />,
    label: "System Design",
    detail: "Propose a database schema for an e-commerce system.",
    prompt: "Outline a standard SQL database schema for a basic e-commerce system (including Users, Products, Orders, OrderItems)."
  }
];

export default function ChatArea({
  session,
  onSendMessage,
  isStreaming,
  onToggleMobile,
  onStopGeneration,
}: ChatAreaProps) {
  const [input, setInput] = useState('');
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = (behavior: ScrollBehavior = 'smooth') => {
    messagesEndRef.current?.scrollIntoView({ behavior });
  };

  useEffect(() => {
    scrollToBottom('smooth');
  }, [session?.messages, isStreaming]);

  const handleScroll = () => {
    const container = scrollContainerRef.current;
    if (!container) return;
    const isNearBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight < 250;
    setShowScrollBtn(!isNearBottom);
  };

  const handleSend = () => {
    if (!input.trim() || isStreaming) return;
    onSendMessage(input.trim());
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <section className="flex-1 flex flex-col bg-white text-slate-900 h-full relative overflow-hidden" id="chat-area">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3.5 bg-white border-b border-slate-200 z-10">
        <div className="flex items-center gap-3 min-w-0">
          <button
            onClick={onToggleMobile}
            className="p-2 -ml-2 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-lg lg:hidden transition cursor-pointer"
            id="mobile-sidebar-toggle"
          >
            <Menu className="w-5 h-5" />
          </button>

          <div className="min-w-0">
            {session ? (
              <>
                <div className="flex items-center gap-2">
                  <h2 className="text-sm font-semibold text-slate-800 truncate max-w-[200px] sm:max-w-xs md:max-w-md lg:max-w-lg">{session.title}</h2>
                </div>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                  <span className="text-[10px] text-slate-400 font-medium">Ready</span>
                </div>
              </>
            ) : (
              <>
                <div className="flex items-center gap-2">
                  <h2 className="text-sm font-semibold text-slate-800">Graph RAG Assistant</h2>
                </div>
                <p className="text-[10px] text-slate-400 font-medium">Select or create a conversation to start</p>
              </>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {isStreaming && (
            <div className="flex items-center gap-1.5 px-3 py-1 bg-indigo-50 border border-indigo-100 text-indigo-600 rounded-full text-[10px] font-semibold">
              <Loader2 className="w-3 h-3 animate-spin" />
              Processing
            </div>
          )}
        </div>
      </header>

      {/* Message List */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-4 py-6 md:p-8 space-y-6 scrollbar-thin scrollbar-thumb-slate-200 scrollbar-track-transparent bg-slate-50/30"
        id="messages-container"
      >
        {!session || session.messages.length === 0 ? (
          <div className="max-w-2xl mx-auto flex flex-col items-center justify-center py-10 space-y-8">
            <div className="flex flex-col items-center text-center space-y-3">
              <div className="p-4 bg-gradient-to-br from-indigo-50 to-indigo-100/60 border border-indigo-100 rounded-2xl shadow-sm">
                <Bot className="w-10 h-10 text-indigo-600" />
              </div>
              <h1 className="text-2xl font-bold tracking-tight text-slate-800 mt-2">Hello! How can I help you today?</h1>
              <p className="text-slate-500 text-sm max-w-md leading-relaxed">
                Graph RAG Knowledge Retrieval System powered by Neo4j and LangGraph is ready to assist you.
              </p>
            </div>

            <div className="w-full space-y-3">
              <h3 className="text-[10px] font-bold text-slate-400 tracking-wider font-mono text-center mb-1">EXPERT SUGGESTIONS</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {STARTER_PROMPTS.map((starter, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      if (!session) {
                        onSendMessage(starter.prompt);
                      } else {
                        setInput(starter.prompt);
                      }
                    }}
                    className="flex flex-col text-left p-4 bg-white hover:bg-indigo-50/30 border border-slate-200 hover:border-indigo-200 rounded-xl transition-all duration-200 group cursor-pointer hover:shadow-sm"
                  >
                    <div className="flex items-center gap-2 mb-1.5 font-semibold text-slate-700 text-xs">
                      {starter.icon}
                      <span>{starter.label}</span>
                    </div>
                    <p className="text-xs text-slate-500 group-hover:text-slate-600 line-clamp-2 leading-relaxed">{starter.detail}</p>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto space-y-6">
            {session.messages.map((message) => {
              const isUser = message.role === 'user';
              return (
                <div
                  key={message.id}
                  className={`flex gap-3.5 ${isUser ? 'justify-end' : 'justify-start'}`}
                  id={`msg-block-${message.id}`}
                >
                  {!isUser && (
                    <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-indigo-50 to-indigo-100/60 border border-indigo-100 flex items-center justify-center shrink-0 shadow-sm">
                      <Bot className="w-4 h-4 text-indigo-600" />
                    </div>
                  )}

                  <div className={`flex flex-col max-w-[85%] min-w-0 space-y-1.5 ${isUser ? 'items-end' : 'items-start'}`}>
                    <div className="flex items-center gap-2 text-[10px] text-slate-400 font-mono flex-wrap overflow-hidden">
                      <span>{isUser ? 'You' : (message.modelUsed || 'Graph RAG Assistant')}</span>
                      <span>•</span>
                      <span>{new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                      {isUser && (
                        <>
                          <span>•</span>
                          <span className="flex items-center gap-1">
                            {(!message.status || message.status === 'sending') && (
                              <>
                                <Clock className="w-3 h-3 text-slate-400 animate-pulse" />
                                <span>Sending</span>
                              </>
                            )}
                            {message.status === 'sent' && (
                              <>
                                <Check className="w-3 h-3 text-slate-400" />
                                <span>Sent</span>
                              </>
                            )}
                            {message.status === 'delivered' && (
                              <>
                                <CheckCheck className="w-3 h-3 text-slate-400" />
                                <span>Delivered</span>
                              </>
                            )}
                            {message.status === 'read' && (
                              <>
                                <CheckCheck className="w-3 h-3 text-indigo-600" />
                                <span className="text-indigo-600 font-semibold">Read</span>
                              </>
                            )}
                          </span>
                        </>
                      )}
                    </div>

                    <div className={`p-4 rounded-2xl shadow-sm border ${
                      isUser
                        ? 'bg-slate-100 text-slate-800 rounded-tr-none border-slate-200/60'
                        : message.error
                          ? 'bg-rose-50 border-rose-200 text-rose-700 rounded-tl-none'
                          : 'bg-white border-slate-200 rounded-tl-none text-slate-800'
                    }`}>
                      {isUser ? (
                        <p className="whitespace-pre-wrap text-sm break-words leading-relaxed">{message.content}</p>
                      ) : (
                        <>
                          {message.toolCalls && typeof message.toolCalls === 'object' && Object.keys(message.toolCalls).length > 0 && (
                            <div className="mb-3">
                              <ToolCallGroup
                                toolCalls={message.toolCalls as Record<number, import('../types').ToolCallState>}
                                isStreaming={isStreaming && message.id === session?.messages[session.messages.length - 1]?.id}
                              />
                            </div>
                          )}
                          {message.thinking && <ThinkingPanel thinking={message.thinking} />}
                          {message.content && <MarkdownRenderer content={message.content} />}
                        </>
                      )}
                    </div>
                  </div>

                  {isUser && (
                    <div className="w-8 h-8 rounded-xl bg-slate-200 flex items-center justify-center shrink-0 shadow-sm">
                      <User className="w-4 h-4 text-slate-600" />
                    </div>
                  )}
                </div>
              );
            })}

            {isStreaming && (
              <div className="flex gap-3.5 justify-start">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-indigo-50 to-indigo-100/60 border border-indigo-100 flex items-center justify-center shrink-0 shadow-sm animate-pulse">
                  <Bot className="w-4 h-4 text-indigo-600" />
                </div>
                <div className="flex flex-col items-start space-y-1">
                  <div className="text-[10px] text-slate-400 font-mono">
                    <span>Thinking...</span>
                  </div>
                  <div className="flex items-center justify-center px-4 py-3 bg-white border border-slate-200 rounded-2xl rounded-tl-none shadow-sm">
                    <Loader2 className="w-4 h-4 text-indigo-600 animate-spin" />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Floating Scroll To Bottom Button */}
      {showScrollBtn && (
        <button
          onClick={() => scrollToBottom('smooth')}
          className="absolute bottom-24 right-6 md:right-10 p-2.5 bg-white hover:bg-slate-50 border border-slate-200 text-slate-500 hover:text-slate-800 rounded-xl shadow-md transition hover:scale-105 cursor-pointer z-10"
          title="Scroll to bottom"
          id="scroll-to-bottom-btn"
        >
          <ArrowDown className="w-4 h-4" />
        </button>
      )}

      {/* Input Tray */}
      <footer className="p-4 md:p-6 bg-gradient-to-t from-white via-white/95 to-transparent border-t border-slate-200/60">
        <div className="max-w-3xl mx-auto relative">
          <div className="bg-slate-50 border border-slate-200 focus-within:border-indigo-500/80 focus-within:bg-white rounded-2xl shadow-sm transition-all p-2 flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                !session
                  ? "Select a suggestion above or enter a question to start..."
                  : "Type a message... (Shift+Enter for new line)"
              }
              rows={1}
              className="flex-1 max-h-36 min-h-[36px] bg-transparent resize-none border-0 text-slate-800 placeholder:text-slate-400 focus:outline-none focus-visible:outline-none px-3 py-2 text-sm leading-relaxed"
              id="chat-input-textarea"
            />

            <div className="flex items-center gap-2 pr-1 pb-1">
              {input.length > 0 && (
                <span className="text-[10px] text-slate-400 font-mono font-medium hidden sm:inline-block">
                  {input.length} characters
                </span>
              )}

              <button
                type="button"
                onClick={isStreaming ? onStopGeneration : handleSend}
                disabled={!isStreaming && !input.trim()}
                className={`p-2 font-medium rounded-xl transition cursor-pointer active:scale-[0.97] shadow-sm ${
                  isStreaming
                    ? 'bg-red-500 hover:bg-red-600 text-white'
                    : 'bg-indigo-600 hover:bg-indigo-700 disabled:opacity-30 text-white'
                }`}
                id={isStreaming ? "stop-msg-btn" : "send-msg-btn"}
                title={isStreaming ? "Stop generation" : "Send message"}
              >
                {isStreaming ? (
                  <Square className="w-4 h-4 fill-current" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </button>
            </div>
          </div>

          <div className="flex justify-between items-center mt-2 px-1 text-[10px] text-slate-400 leading-relaxed font-mono">
            <span>Press Enter to send, Shift + Enter for a new line.</span>
            <span>Graph RAG Assistant</span>
          </div>
        </div>
      </footer>
    </section>
  );
}
