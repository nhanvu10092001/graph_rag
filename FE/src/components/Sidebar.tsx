/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import {
  Plus, MessageSquare, Trash2, Edit3, Settings, Check, X,
  Bot, ChevronRight, Upload, Globe2
} from 'lucide-react';
import { ChatSession } from '../types';

interface SidebarProps {
  sessions: ChatSession[];
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
  onRenameSession: (id: string, newTitle: string) => void;
  onOpenSettings: () => void;
  isMobileOpen: boolean;
  onToggleMobile: () => void;
  documents: any[];
  isUploading: boolean;
  onUploadFile: (file: File) => void;
  onDeleteDocument: (id: number) => void;
  onRefreshDocuments: () => void;
  onOpenCommunity: () => void;
}

export default function Sidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onRenameSession,
  onOpenSettings,
  isMobileOpen,
  onToggleMobile,
  documents,
  isUploading,
  onUploadFile,
  onDeleteDocument,
  onRefreshDocuments,
  onOpenCommunity,
}: SidebarProps) {
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [confirmDeletingDoc, setConfirmDeletingDoc] = useState<any | null>(null);

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
      {/* Mobile Backdrop */}
      {isMobileOpen && (
        <div
          onClick={onToggleMobile}
          className="fixed inset-0 z-40 bg-black/25 backdrop-blur-sm lg:hidden"
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex flex-col w-72 bg-slate-50 border-r border-slate-200 text-slate-700 transition-transform duration-300 transform
          lg:translate-x-0 lg:static lg:h-full lg:flex-shrink-0
          ${isMobileOpen ? 'translate-x-0' : '-translate-x-full'}`}
        id="app-sidebar"
      >
        {/* Brand Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-200 bg-white">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-indigo-500 to-indigo-600 rounded-xl shadow-sm">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-bold text-slate-800 tracking-tight">Graph RAG</h1>
              <p className="text-[10px] text-slate-400 font-medium">Knowledge Assistant</p>
            </div>
          </div>
          <button
            onClick={onToggleMobile}
            className="p-1.5 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg lg:hidden transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* New Chat Button */}
        <div className="p-4 bg-white border-b border-slate-100">
          <button
            onClick={() => {
              onNewSession();
              if (isMobileOpen) onToggleMobile();
            }}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-700 hover:to-indigo-800 text-white font-semibold rounded-xl text-xs shadow-sm hover:shadow-md transition-all active:scale-[0.98] cursor-pointer"
            id="new-chat-btn"
          >
            <Plus className="w-4 h-4" />
            New Chat
          </button>
        </div>

        {/* Sessions List */}
        <div className="flex-1 overflow-y-auto px-3 py-4 space-y-1">
          <div className="flex items-center justify-between px-3 pb-2 text-[10px] font-bold text-slate-400 tracking-wider font-mono">
            <span>CHAT HISTORY</span>
            <span>{sessions.length}</span>
          </div>

          {sessions.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-center text-slate-400 space-y-2">
              <MessageSquare className="w-6 h-6 stroke-[1.5]" />
              <p className="text-xs">No conversations yet</p>
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
                    className={`group relative flex items-center justify-between px-3 py-2.5 rounded-xl text-xs transition cursor-pointer ${
                      isActive
                        ? 'bg-white border border-indigo-100 text-slate-900 font-semibold shadow-sm'
                        : 'text-slate-600 hover:bg-white/60 hover:text-slate-900'
                    }`}
                  >
                    <div className="flex items-center gap-2.5 min-w-0 flex-1">
                      {isActive && (
                        <div className="w-0.5 h-5 bg-indigo-500 rounded-full shrink-0" />
                      )}
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

                    {!isEditing && (
                      <div className="absolute right-2 hidden group-hover:flex items-center gap-1 bg-slate-50 pl-2 py-0.5 rounded-l border-l border-slate-100">
                        <button
                          onClick={(e) => startRename(session, e)}
                          className="text-slate-500 hover:text-indigo-600 p-1 rounded hover:bg-slate-200 transition"
                          title="Rename"
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
                          title="Delete"
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

        {/* Document Manager */}
        <div className="border-t border-slate-200 bg-white p-4 space-y-3 shrink-0">
          <div className="flex items-center justify-between text-[10px] font-bold text-slate-400 tracking-wider font-mono">
            <span>GRAPH RAG DOCUMENTS</span>
            <button
              onClick={onRefreshDocuments}
              className="text-indigo-600 hover:text-indigo-700 font-semibold cursor-pointer hover:underline text-[9px] bg-transparent border-0 p-0"
            >
              Refresh
            </button>
          </div>

          {/* File Upload */}
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
              {isUploading ? 'Uploading...' : 'Upload document (.pdf, .txt, .md)'}
            </label>
          </div>

          {/* Document List */}
          <div className="max-h-[140px] overflow-y-auto space-y-1.5 pr-1">
            {documents.length === 0 ? (
              <p className="text-[10px] text-slate-400 text-center py-2 font-medium">No documents uploaded</p>
            ) : (
              documents.map((doc) => (
                <div key={doc.id} className="flex flex-col p-2 bg-slate-50 border border-slate-100 rounded-lg text-[10px] space-y-0.5 shadow-sm">
                  <div className="flex items-center justify-between font-medium">
                    <span className="truncate max-w-[110px] text-slate-700 font-semibold" title={doc.filename}>{doc.filename}</span>
                    <div className="flex items-center gap-1">
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
                          setConfirmDeletingDoc(doc);
                        }}
                        className="text-slate-400 hover:text-rose-600 p-0.5 rounded hover:bg-slate-200/50 transition cursor-pointer flex items-center justify-center border-0 bg-transparent"
                        title="Delete document"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                  {(doc.entity_count > 0 || doc.relationship_count > 0) && (
                    <div className="flex items-center justify-end text-[9px] text-slate-400 font-mono gap-1 pt-0.5">
                      <span>N: {doc.entity_count}</span>
                      <span>C: {doc.relationship_count}</span>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Footer Actions */}
        <div className="p-4 bg-slate-50 border-t border-slate-200 space-y-2">
          <button
            onClick={onOpenCommunity}
            className="w-full flex items-center justify-between px-4 py-2.5 bg-white hover:bg-indigo-50 text-slate-600 hover:text-indigo-700 border border-slate-200 rounded-xl text-xs font-medium transition cursor-pointer shadow-sm"
            id="open-community-btn"
          >
            <span className="flex items-center gap-2">
              <Globe2 className="w-4 h-4 text-indigo-600" />
              Community Detection
            </span>
            <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
          </button>

          <button
            onClick={onOpenSettings}
            className="w-full flex items-center justify-between px-4 py-2.5 bg-white hover:bg-slate-100 text-slate-600 hover:text-slate-800 border border-slate-200 rounded-xl text-xs font-medium transition cursor-pointer shadow-sm"
            id="open-settings-btn"
          >
            <span className="flex items-center gap-2">
              <Settings className="w-4 h-4 text-indigo-600" />
              System Settings
            </span>
            <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
          </button>
        </div>
      </aside>

      {/* Modal: Delete Document */}
      {confirmDeletingDoc && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/25 backdrop-blur-sm transition-opacity duration-200">
          <div className="relative w-full max-w-sm bg-white border border-slate-200 rounded-2xl shadow-xl overflow-hidden flex flex-col p-6 animate-in fade-in zoom-in-95 duration-200 text-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-rose-600 tracking-wider uppercase font-mono">Delete Document</h3>
              <button
                type="button"
                onClick={() => setConfirmDeletingDoc(null)}
                className="text-slate-400 hover:text-slate-700 hover:bg-slate-100 p-1.5 rounded-lg transition border-0 cursor-pointer bg-transparent"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed font-medium">
              Are you sure you want to delete document <strong>"{confirmDeletingDoc.filename}"</strong>? This action will remove the document from the database and all related entities/relationships in Neo4j.
            </p>
            <div className="flex justify-end gap-2 text-xs pt-1">
              <button
                type="button"
                onClick={() => setConfirmDeletingDoc(null)}
                className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold rounded-xl border-0 cursor-pointer transition"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => {
                  onDeleteDocument(confirmDeletingDoc.id);
                  setConfirmDeletingDoc(null);
                }}
                className="px-4 py-2 bg-rose-600 hover:bg-rose-700 text-white font-semibold rounded-xl border-0 cursor-pointer transition shadow-sm"
              >
                Confirm Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
