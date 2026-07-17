/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { 
  Plus, MessageSquare, Trash2, Edit3, Settings, Check, X, 
  Sparkles, ShieldCheck, Key, Bot, ChevronLeft, ChevronRight, Menu, Upload
} from 'lucide-react';
import { ChatSession, AVAILABLE_MODELS } from '../types';

interface SidebarProps {
  sessions: ChatSession[];
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewSession: (modelId: string) => void;
  onDeleteSession: (id: string) => void;
  onRenameSession: (id: string, newTitle: string) => void;
  onOpenSettings: () => void;
  hasSystemKey: boolean;
  customApiKey: string;
  selectedModelId: string;
  onSelectModel: (id: string) => void;
  isMobileOpen: boolean;
  onToggleMobile: () => void;
  documents: any[];
  isUploading: boolean;
  onUploadFile: (file: File) => void;
  onDeleteDocument: (id: number) => void;
  onRefreshDocuments: () => void;
}

export default function Sidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onRenameSession,
  onOpenSettings,
  hasSystemKey,
  customApiKey,
  selectedModelId,
  onSelectModel,
  isMobileOpen,
  onToggleMobile,
  documents,
  isUploading,
  onUploadFile,
  onDeleteDocument,
  onRefreshDocuments,
}: SidebarProps) {
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [deletingDocId, setDeletingDocId] = useState<number | null>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onUploadFile(e.target.files[0]);
    }
  };

  const startRename = (session: ChatSession, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingSessionId(session.id);
    setEditTitle(session.title);
  };

  const cancelRename = (e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingSessionId(null);
  };

  const submitRename = (id: string, e: React.FormEvent | React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    if (editTitle.trim()) {
      onRenameSession(id, editTitle.trim());
    }
    setEditingSessionId(null);
  };

  return (
    <>
      {/* Mobile Backdrop Overlay */}
      {isMobileOpen && (
        <div 
          onClick={onToggleMobile}
          className="fixed inset-0 z-40 bg-black/25 backdrop-blur-sm lg:hidden"
        />
      )}

      {/* Sidebar Container */}
      <aside 
        className={`fixed inset-y-0 left-0 z-40 flex flex-col w-72 bg-slate-50 border-r border-slate-200 text-slate-700 transition-transform duration-300 transform 
          lg:translate-x-0 lg:static lg:h-full lg:flex-shrink-0
          ${isMobileOpen ? 'translate-x-0' : '-translate-x-full'}`}
        id="app-sidebar"
      >
        {/* Brand Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-white">
          <div className="flex items-center gap-2">
            <div className="p-1.5 bg-indigo-50 border border-indigo-100/80 rounded-lg">
              <Bot className="w-5 h-5 text-indigo-600" />
            </div>
            <div>
              <h1 className="text-sm font-semibold text-slate-800 tracking-tight">Graph RAG Hub</h1>
              <p className="text-[10px] text-slate-400 font-mono tracking-wider">ENTERPRISE CHAT</p>
            </div>
          </div>
          <button 
            onClick={onToggleMobile}
            className="p-1 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded lg:hidden"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Action: Create New Conversation */}
        <div className="p-4 space-y-3 bg-white border-b border-slate-100">
          <button
            onClick={() => {
              onNewSession(selectedModelId);
              if (isMobileOpen) onToggleMobile();
            }}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-xl text-xs shadow-sm hover:shadow-md transition active:scale-[0.98] cursor-pointer"
            id="new-chat-btn"
          >
            <Plus className="w-4 h-4" />
            Tạo hội thoại mới
          </button>

          {/* Model Selector in Sidebar */}
          <div className="space-y-1.5">
            <label className="text-[10px] font-bold text-slate-400 tracking-wider block font-mono">DÒNG MÔ HÌNH CHUẨN</label>
            <div className="grid grid-cols-2 gap-1 bg-slate-100 p-1 border border-slate-200 rounded-xl">
              {AVAILABLE_MODELS.map((model) => (
                <button
                  key={model.id}
                  onClick={() => onSelectModel(model.id)}
                  className={`px-2 py-1.5 text-[11px] font-medium rounded-lg transition-all cursor-pointer ${
                    selectedModelId === model.id
                      ? 'bg-white text-indigo-600 font-semibold border border-slate-200/80 shadow-sm'
                      : 'text-slate-500 hover:text-slate-800'
                  }`}
                  title={model.description}
                >
                  {model.name}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Navigation Section: List of Sessions */}
        <div className="flex-1 overflow-y-auto px-3 py-4 space-y-1">
          <div className="flex items-center justify-between px-3 pb-2 text-[10px] font-bold text-slate-400 tracking-wider font-mono">
            <span>LỊCH SỬ HỘI THOẠI</span>
            <span>{sessions.length}</span>
          </div>

          {sessions.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-center text-slate-400 space-y-2">
              <MessageSquare className="w-6 h-6 stroke-[1.5]" />
              <p className="text-xs">Chưa có cuộc trò chuyện nào</p>
            </div>
          ) : (
            <div className="space-y-1" id="chat-sessions-list">
              {sessions.map((session) => {
                const isActive = session.id === activeSessionId;
                const isEditing = session.id === editingSessionId;

                return (
                  <div
                    key={session.id}
                    onClick={() => {
                      onSelectSession(session.id);
                      if (isMobileOpen) onToggleMobile();
                    }}
                    className={`group relative flex items-center justify-between px-3 py-2 rounded-xl text-xs transition cursor-pointer ${
                      isActive 
                        ? 'bg-slate-200/60 border border-slate-200/80 text-slate-900 font-semibold' 
                        : 'text-slate-600 hover:bg-slate-200/30 hover:text-slate-900'
                    }`}
                  >
                    <div className="flex items-center gap-2.5 min-w-0 flex-1">
                      <MessageSquare className={`w-4 h-4 shrink-0 ${isActive ? 'text-indigo-600' : 'text-slate-400'}`} />
                      
                      {isEditing ? (
                        <form 
                          onSubmit={(e) => submitRename(session.id, e)} 
                          className="flex items-center gap-1 flex-1 min-w-0"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <input
                            type="text"
                            value={editTitle}
                            onChange={(e) => setEditTitle(e.target.value)}
                            className="bg-white border border-indigo-500 rounded px-1.5 py-0.5 text-xs text-slate-900 focus:outline-none w-full"
                            autoFocus
                            id={`rename-input-${session.id}`}
                          />
                          <button 
                            type="button" 
                            onClick={(e) => submitRename(session.id, e)}
                            className="text-indigo-600 hover:text-indigo-500 p-0.5"
                          >
                            <Check className="w-3.5 h-3.5" />
                          </button>
                          <button 
                            type="button" 
                            onClick={cancelRename}
                            className="text-slate-400 hover:text-slate-700 p-0.5"
                          >
                            <X className="w-3.5 h-3.5" />
                          </button>
                        </form>
                      ) : (
                        <span className="truncate pr-4">{session.title}</span>
                      )}
                    </div>

                    {/* Quick Session Actions */}
                    {!isEditing && (
                      <div className="absolute right-2 hidden group-hover:flex items-center gap-1 bg-slate-100 pl-2 py-0.5 rounded-l border-l border-slate-100">
                        <button
                          onClick={(e) => startRename(session, e)}
                          className="text-slate-500 hover:text-indigo-600 p-1 rounded hover:bg-slate-200 transition"
                          title="Đổi tên"
                          id={`rename-btn-${session.id}`}
                        >
                          <Edit3 className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteSession(session.id);
                          }}
                          className="text-slate-500 hover:text-rose-600 p-1 rounded hover:bg-slate-200 transition"
                          title="Xóa"
                          id={`delete-btn-${session.id}`}
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* RAG Documents Indexing Manager */}
        <div className="border-t border-slate-200 bg-white p-4 space-y-3 shrink-0">
          <div className="flex items-center justify-between text-[10px] font-bold text-slate-400 tracking-wider font-mono">
            <span>TÀI LIỆU GRAPH RAG</span>
            <button 
              onClick={onRefreshDocuments}
              className="text-indigo-600 hover:text-indigo-700 font-semibold cursor-pointer hover:underline text-[9px] bg-transparent border-0 p-0"
            >
              Làm mới
            </button>
          </div>

          <div className="relative">
            <input
              type="file"
              accept=".txt,.md,.pdf"
              onChange={handleFileUpload}
              className="hidden"
              id="rag-file-upload"
              disabled={isUploading}
            />
            <label
              htmlFor="rag-file-upload"
              className={`w-full flex items-center justify-center gap-2 px-3 py-2 border border-dashed rounded-xl text-xs font-medium cursor-pointer transition-all ${
                isUploading 
                  ? 'bg-slate-50 border-slate-200 text-slate-400 cursor-not-allowed' 
                  : 'border-indigo-200 hover:border-indigo-400 bg-indigo-50/20 hover:bg-indigo-50/40 text-indigo-600'
              }`}
            >
              <Upload className="w-3.5 h-3.5" />
              {isUploading ? 'Đang tải lên...' : 'Tải tài liệu (.pdf, .txt, .md)'}
            </label>
          </div>

          <div className="max-h-[140px] overflow-y-auto space-y-1.5 pr-1">
            {documents.length === 0 ? (
              <p className="text-[10px] text-slate-400 text-center py-2 font-medium">Chưa có tài liệu nào</p>
            ) : (
              documents.map((doc) => (
                <div key={doc.id} className="flex flex-col p-2 bg-slate-50 border border-slate-100 rounded-lg text-[10px] space-y-0.5 shadow-sm">
                  <div className="flex items-center justify-between font-medium">
                    <span className="truncate max-w-[110px] text-slate-700 font-semibold" title={doc.filename}>{doc.filename}</span>
                    <div className="flex items-center gap-1">
                      {deletingDocId === doc.id ? (
                        <div className="flex items-center gap-1">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              onDeleteDocument(doc.id);
                              setDeletingDocId(null);
                            }}
                            className="bg-rose-500 hover:bg-rose-600 text-white px-1.5 py-0.5 rounded text-[8px] font-bold transition cursor-pointer"
                            title="Xác nhận xóa"
                          >
                            Xóa
                          </button>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setDeletingDocId(null);
                            }}
                            className="bg-slate-200 hover:bg-slate-300 text-slate-700 px-1.5 py-0.5 rounded text-[8px] font-bold transition cursor-pointer"
                            title="Hủy"
                          >
                            Hủy
                          </button>
                        </div>
                      ) : (
                        <>
                          <span className={`px-1.5 py-0.5 rounded text-[8px] font-semibold tracking-wider font-mono ${
                            doc.status === 'indexed' ? 'bg-emerald-50 text-emerald-600 border border-emerald-100' :
                            doc.status === 'processing' ? 'bg-indigo-50 text-indigo-600 border border-indigo-100 animate-pulse' :
                            doc.status === 'failed' ? 'bg-rose-50 text-rose-600 border border-rose-100' :
                            'bg-slate-100 text-slate-600 border border-slate-200'
                          }`}>
                            {doc.status.toUpperCase()}
                          </span>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setDeletingDocId(doc.id);
                            }}
                            className="text-slate-400 hover:text-rose-600 p-0.5 rounded hover:bg-slate-200/50 transition cursor-pointer"
                            title="Xoá tài liệu"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                  {(doc.entity_count > 0 || doc.relationship_count > 0) && (
                    <div className="text-[9px] text-slate-400 font-mono flex gap-2">
                      <span>Nút: {doc.entity_count}</span>
                      <span>Cạnh: {doc.relationship_count}</span>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Sidebar Footer Info & Trigger Settings */}
        <div className="p-4 bg-slate-50 border-t border-slate-200 space-y-3">
          {/* Status Indicator */}
          <div className="p-3 bg-slate-100 border border-slate-200 rounded-xl space-y-2">
            <div className="flex items-center justify-between text-[10px]">
              <span className="text-slate-400 font-mono">BẢO MẬT API</span>
              <span className="flex items-center gap-1 text-indigo-600">
                <ShieldCheck className="w-3.5 h-3.5" /> Bảo mật
              </span>
            </div>
            
            <div className="flex items-center gap-2 text-xs">
              {customApiKey ? (
                <div className="flex items-center gap-1.5 text-indigo-600 font-medium">
                  <Key className="w-3.5 h-3.5" />
                  <span className="truncate max-w-[120px]">Key Cá Nhân</span>
                </div>
              ) : hasSystemKey ? (
                <div className="flex items-center gap-1.5 text-indigo-600 font-medium">
                  <Bot className="w-3.5 h-3.5" />
                  <span>Key Máy Chủ</span>
                </div>
              ) : (
                <div className="flex items-center gap-1.5 text-amber-600 font-medium">
                  <X className="w-3.5 h-3.5" />
                  <span>Thiếu API Key</span>
                </div>
              )}
            </div>
          </div>

          {/* Trigger Settings Button */}
          <button
            onClick={onOpenSettings}
            className="w-full flex items-center justify-between px-4 py-2.5 bg-white hover:bg-slate-100 text-slate-600 hover:text-slate-800 border border-slate-200 rounded-xl text-xs font-medium transition cursor-pointer shadow-sm"
            id="open-settings-btn"
          >
            <span className="flex items-center gap-2">
              <Settings className="w-4 h-4 text-indigo-600" />
              Cấu hình hệ thống
            </span>
            <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
          </button>
        </div>
      </aside>
    </>
  );
}
