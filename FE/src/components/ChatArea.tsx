/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, Bot, User, Loader2, Sparkles, AlertTriangle, 
  Menu, Terminal, HelpCircle, FileText, ArrowDown, Settings, AlertCircle,
  Check, CheckCheck, Clock
} from 'lucide-react';
import { ChatSession, Message, AVAILABLE_MODELS } from '../types';
import MarkdownRenderer from './MarkdownRenderer';

interface ChatAreaProps {
  session: ChatSession | null;
  onSendMessage: (text: string) => void;
  isStreaming: boolean;
  onToggleMobile: () => void;
  hasSystemKey: boolean;
  customApiKey: string;
  onOpenSettings: () => void;
}

const STARTER_PROMPTS = [
  {
    icon: <Terminal className="w-4 h-4 text-indigo-500" />,
    label: "Viết mã nguồn",
    detail: "Tạo hàm JavaScript đảo ngược chuỗi có xử lý lỗi chi tiết và tối ưu.",
    prompt: "Viết cho tôi một hàm JavaScript đảo ngược chuỗi (reverse string) có xử lý trường hợp đầu vào không hợp lệ, kèm giải thích chi tiết."
  },
  {
    icon: <HelpCircle className="w-4 h-4 text-sky-500" />,
    label: "Giải thích khái niệm",
    detail: "Giải thích cơ chế hoạt động của WebSockets so với REST API.",
    prompt: "Hãy giải thích cơ chế hoạt động của WebSockets so với REST API truyền thống, khi nào doanh nghiệp nên ưu tiên sử dụng mỗi loại?"
  },
  {
    icon: <FileText className="w-4 h-4 text-amber-500" />,
    label: "Soạn thảo văn bản",
    detail: "Viết email gửi đối tác giải thích sự cố máy chủ và chính sách đền bù.",
    prompt: "Hãy viết một email chuyên nghiệp gửi đối tác kinh doanh để giải thích về sự cố sập máy chủ đám mây sáng nay và đề xuất chính sách đền bù hoàn phí dịch vụ."
  },
  {
    icon: <Sparkles className="w-4 h-4 text-violet-500" />,
    label: "Tư vấn thiết kế",
    detail: "Đề xuất cấu trúc cơ sở dữ liệu cho hệ thống thương mại điện tử.",
    prompt: "Hãy phác thảo một cấu trúc bảng cơ sở dữ liệu SQL chuẩn cho hệ thống thương mại điện tử cơ bản (gồm Users, Products, Orders, OrderItems)."
  }
];

export default function ChatArea({
  session,
  onSendMessage,
  isStreaming,
  onToggleMobile,
  hasSystemKey,
  customApiKey,
  onOpenSettings,
}: ChatAreaProps) {
  const [input, setInput] = useState('');
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Auto scroll to bottom
  const scrollToBottom = (behavior: ScrollBehavior = 'smooth') => {
    messagesEndRef.current?.scrollIntoView({ behavior });
  };

  useEffect(() => {
    scrollToBottom('smooth');
  }, [session?.messages, isStreaming]);

  // Handle scroll position to show/hide "scroll to bottom" button
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

  const activeModelDetails = session 
    ? AVAILABLE_MODELS.find(m => m.id === session.model) 
    : null;

  return (
    <section className="flex-1 flex flex-col bg-white text-slate-900 h-full relative overflow-hidden" id="chat-area">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 bg-white border-b border-slate-200 backdrop-blur-sm z-10">
        <div className="flex items-center gap-3 min-w-0">
          <button 
            onClick={onToggleMobile}
            className="p-2 -ml-2 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-lg lg:hidden"
            id="mobile-sidebar-toggle"
          >
            <Menu className="w-5 h-5" />
          </button>
          
          <div className="min-w-0">
            {session ? (
              <>
                <h2 className="text-sm font-semibold text-slate-800 truncate max-w-[200px] sm:max-w-xs md:max-w-md lg:max-w-lg">{session.title}</h2>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-indigo-600"></span>
                  <span className="text-[10px] text-slate-400 font-medium">
                    Đang sử dụng: <span className="text-indigo-600 font-mono font-semibold">{activeModelDetails?.name || session.model}</span>
                  </span>
                </div>
              </>
            ) : (
              <>
                <h2 className="text-sm font-semibold text-slate-800">Graph RAG Hub</h2>
                <p className="text-[10px] text-slate-400">Chưa tải cuộc trò chuyện</p>
              </>
            )}
          </div>
        </div>

        {/* Top Header Actions */}
        <div className="flex items-center gap-2">
          {!customApiKey && !hasSystemKey && (
            <button
              onClick={onOpenSettings}
              className="flex items-center gap-1.5 px-3 py-1 bg-amber-500/10 hover:bg-amber-500/20 text-amber-600 border border-amber-500/30 rounded-full text-[11px] font-semibold transition cursor-pointer animate-pulse"
              id="header-missing-key-btn"
            >
              <AlertCircle className="w-3.5 h-3.5" />
              <span>Chưa cấu hình API Key</span>
            </button>
          )}
        </div>
      </header>

      {/* Message List */}
      <div 
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-4 py-6 md:p-8 space-y-6 scrollbar-thin scrollbar-thumb-slate-200 scrollbar-track-transparent bg-white"
        id="messages-container"
      >
        {!session || session.messages.length === 0 ? (
          // Welcome / Landing screen with suggestions
          <div className="max-w-2xl mx-auto flex flex-col items-center justify-center py-10 space-y-8">
            <div className="flex flex-col items-center text-center space-y-3">
              <div className="p-3.5 bg-indigo-50 border border-indigo-100/80 rounded-2xl">
                <Bot className="w-10 h-10 text-indigo-600" />
              </div>
              <h1 className="text-2xl font-bold tracking-tight text-slate-800 mt-2">Xin chào! Tôi có thể giúp gì cho bạn?</h1>
              <p className="text-slate-500 text-sm max-w-md">
                Hệ thống tích hợp mô hình ngôn ngữ lớn thế hệ mới nhất của OpenAI kết hợp với cơ sở dữ liệu đồ thị Neo4j và LangGraph.
              </p>
            </div>

            <div className="w-full space-y-3">
              <h3 className="text-xs font-bold text-slate-400 tracking-wider font-mono text-center mb-1">GỢI Ý CHUYÊN MÔN</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {STARTER_PROMPTS.map((starter, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      if (!session) {
                        // Creates session automatically, handled by parent
                        onSendMessage(starter.prompt);
                      } else {
                        setInput(starter.prompt);
                      }
                    }}
                    className="flex flex-col text-left p-4 bg-slate-50 hover:bg-slate-100/80 border border-slate-200 hover:border-indigo-500/30 rounded-xl transition-all duration-300 group cursor-pointer hover:shadow-sm"
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

            {/* Empty Key alert inside the empty chat */}
            {!customApiKey && !hasSystemKey && (
              <div className="w-full p-4 bg-amber-50 border border-amber-200 rounded-xl text-xs text-slate-700 flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5 animate-bounce" />
                <div className="space-y-1">
                  <h4 className="font-semibold text-amber-600">Không tìm thấy API Key máy chủ hoặc khóa cá nhân!</h4>
                  <p className="leading-relaxed text-slate-500">Vui lòng nhấp vào nút dưới đây để nhập khóa API OpenAI của bạn trước khi tiến hành gửi yêu cầu trò chuyện.</p>
                  <button 
                    onClick={onOpenSettings} 
                    className="mt-2 flex items-center gap-1 px-3 py-1.5 bg-amber-600 hover:bg-amber-700 text-white font-semibold rounded-lg transition text-[11px]"
                  >
                    <Settings className="w-3.5 h-3.5" />
                    Thiết lập API Key ngay
                  </button>
                </div>
              </div>
            )}
          </div>
        ) : (
          // Message Bubbles
          <div className="max-w-3xl mx-auto space-y-6">
            {session.messages.map((message) => {
              const isUser = message.role === 'user';
              return (
                <div 
                  key={message.id}
                  className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start'}`}
                  id={`msg-block-${message.id}`}
                >
                  {/* Left Avatar for Model */}
                  {!isUser && (
                    <div className="w-8 h-8 rounded-xl bg-indigo-50 border border-indigo-100 flex items-center justify-center shrink-0 shadow-sm">
                      <Bot className="w-4 h-4 text-indigo-600" />
                    </div>
                  )}

                  {/* Bubble Container */}
                  <div className={`flex flex-col max-w-[85%] space-y-1 ${isUser ? 'items-end' : 'items-start'}`}>
                    <div className="flex items-center gap-2 text-[10px] text-slate-400 font-mono">
                      <span>{isUser ? 'Bạn' : (message.modelUsed || 'RAG Assistant')}</span>
                      <span>•</span>
                      <span>{new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                      {isUser && (
                        <>
                          <span>•</span>
                          <span className="flex items-center gap-1">
                            {(!message.status || message.status === 'sending') && (
                              <>
                                <Clock className="w-3 h-3 text-slate-400 animate-pulse" />
                                <span>Đang gửi</span>
                              </>
                            )}
                            {message.status === 'sent' && (
                              <>
                                <Check className="w-3 h-3 text-slate-400" />
                                <span>Đã gửi</span>
                              </>
                            )}
                            {message.status === 'delivered' && (
                              <>
                                <CheckCheck className="w-3 h-3 text-slate-400" />
                                <span>Đã nhận</span>
                              </>
                            )}
                            {message.status === 'read' && (
                              <>
                                <CheckCheck className="w-3 h-3 text-indigo-600" />
                                <span className="text-indigo-600 font-semibold">Đã xem</span>
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
                          : 'bg-white border border-slate-200 rounded-tl-none text-slate-800 shadow-sm'
                    }`}>
                      {isUser ? (
                        <p className="whitespace-pre-wrap text-sm break-words leading-relaxed">{message.content}</p>
                      ) : (
                        <MarkdownRenderer content={message.content} />
                      )}
                    </div>
                  </div>

                  {/* Right Avatar for User */}
                  {isUser && (
                    <div className="w-8 h-8 rounded-xl bg-slate-200 flex items-center justify-center shrink-0 shadow-sm">
                      <User className="w-4 h-4 text-slate-600" />
                    </div>
                  )}
                </div>
              );
            })}

            {/* Streaming Placeholder indicator */}
            {isStreaming && (
              <div className="flex gap-4 justify-start">
                <div className="w-8 h-8 rounded-xl bg-indigo-50 border border-indigo-100 flex items-center justify-center shrink-0 shadow-sm animate-pulse">
                  <Bot className="w-4 h-4 text-indigo-600" />
                </div>
                <div className="flex flex-col items-start space-y-1">
                  <div className="text-[10px] text-slate-400 font-mono">
                    <span>Đang suy nghĩ...</span>
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
          className="absolute bottom-24 right-6 md:right-10 p-2.5 bg-white hover:bg-slate-50 border border-slate-200 text-slate-500 hover:text-slate-850 rounded-xl shadow-md transition hover:scale-105 cursor-pointer z-10"
          title="Cuộn xuống dưới"
          id="scroll-to-bottom-btn"
        >
          <ArrowDown className="w-4 h-4" />
        </button>
      )}

      {/* Input Tray */}
      <footer className="p-4 md:p-6 bg-gradient-to-t from-white via-white/95 to-transparent border-t border-slate-200/60">
        <div className="max-w-3xl mx-auto relative">
          <div className="bg-slate-50 border border-slate-200 focus-within:border-indigo-500/85 focus-within:bg-white rounded-2xl shadow-sm transition-all p-2 flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                !session 
                  ? "Chọn hoặc gửi gợi ý ở trên để bắt đầu hội thoại..." 
                  : `Gửi tin nhắn đến ${activeModelDetails?.name || 'mô hình'}... (Shift+Enter để xuống dòng)`
              }
              rows={1}
              className="flex-1 max-h-36 min-h-[36px] bg-transparent resize-none border-0 text-slate-800 placeholder:text-slate-400 focus:outline-none px-3 py-2 text-sm leading-relaxed"
              id="chat-input-textarea"
            />

            <div className="flex items-center gap-2 pr-1 pb-1">
              {/* Token Counter placeholder or similar detail */}
              {input.length > 0 && (
                <span className="text-[10px] text-slate-400 font-mono font-medium hidden sm:inline-block">
                  {input.length} ký tự
                </span>
              )}

              <button
                type="button"
                onClick={handleSend}
                disabled={!input.trim() || isStreaming}
                className="p-2 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-30 text-white font-medium rounded-xl transition cursor-pointer active:scale-[0.97] shadow-sm"
                id="send-msg-btn"
              >
                {isStreaming ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </button>
            </div>
          </div>

          <div className="flex justify-between items-center mt-2 px-1 text-[10px] text-slate-400 leading-relaxed font-mono">
            <span>Nhấn Enter để gửi, Shift + Enter để thêm dòng mới.</span>
            <span>OpenAI & LangGraph Agent</span>
          </div>
        </div>
      </footer>
    </section>
  );
}
