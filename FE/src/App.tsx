/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import CommunityPanel from './components/CommunityPanel';
import SettingsModal from './components/SettingsModal';
import { ChatSession, Message, ModelConfig, ToolCallState } from './types';

const DEFAULT_SYSTEM_INSTRUCTION =
  "Bạn là một Trợ lý AI cao cấp chuyên nghiệp, lịch sự và đáng tin cậy. " +
  "Hãy cung cấp câu trả lời rõ ràng, chính xác, cấu trúc mạch lạc sử dụng định dạng Markdown " +
  "và viết bằng tiếng Việt tự nhiên trừ khi người dùng yêu cầu ngôn ngữ khác.";

const DEFAULT_CONFIG: ModelConfig = {
  systemInstruction: DEFAULT_SYSTEM_INSTRUCTION,
  temperature: 0.7,
  topP: 0.95,
  topK: 40,
};

const API_BASE = window.location.hostname === 'localhost' ? 'http://localhost:8000' : '';

export default function App() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [config, setConfig] = useState<ModelConfig>(DEFAULT_CONFIG);

  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeView, setActiveView] = useState<'chat' | 'community'>('chat');

  const [documents, setDocuments] = useState<any[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [groups, setGroups] = useState<any[]>([]);
  const [selectedGroupId, setSelectedGroupId] = useState<number | null>(null);

  const fetchDocuments = async () => {
    try {
      const url = selectedGroupId ? `/api/documents?group_id=${selectedGroupId}` : '/api/documents';
      const res = await fetch(`${API_BASE}${url}`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data);
      }
    } catch (e) {
      console.error('Error fetching documents', e);
    }
  };

  const fetchGroups = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/groups`);
      if (res.ok) {
        const data = await res.json();
        setGroups(data);
      }
    } catch (e) {
      console.error('Error fetching groups:', e);
    }
  };

  const handleCreateGroup = async (name: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/groups`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      if (res.ok) {
        const newGroup = await res.json();
        await fetchGroups();
        setSelectedGroupId(newGroup.id);
      } else {
        const err = await res.json();
        alert(`Tạo nhóm thất bại: ${err.detail || 'Lỗi không xác định'}`);
      }
    } catch (e) {
      console.error('Error creating group:', e);
      alert('Gặp lỗi khi kết nối tạo nhóm!');
    }
  };

  const handleDeleteGroup = async (id: number) => {
    try {
      const res = await fetch(`${API_BASE}/api/groups/${id}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        await fetchGroups();
        setSelectedGroupId(null);
      } else {
        const err = await res.json();
        alert(`Xóa nhóm thất bại: ${err.detail || 'Lỗi không xác định'}`);
      }
    } catch (e) {
      console.error('Error deleting group:', e);
      alert('Gặp lỗi khi kết nối xóa nhóm!');
    }
  };

  useEffect(() => {
    fetchDocuments();
    const interval = setInterval(fetchDocuments, 5000);
    return () => clearInterval(interval);
  }, [selectedGroupId]);

  useEffect(() => {
    fetchGroups();
  }, []);

  const handleUploadFile = async (file: File) => {
    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    if (selectedGroupId !== null) {
      formData.append('group_id', selectedGroupId.toString());
    }
    try {
      const res = await fetch(`${API_BASE}/api/documents/upload`, {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        await fetchDocuments();
      } else {
        alert('Tải tài liệu lên thất bại!');
      }
    } catch (e) {
      console.error('Error uploading document:', e);
      alert('Lỗi kết nối khi tải tài liệu lên!');
    } finally {
      setIsUploading(false);
    }
  };

  useEffect(() => {
    const savedConfig = localStorage.getItem('openai_chat_config');
    if (savedConfig) {
      try {
        setConfig(JSON.parse(savedConfig));
      } catch (e) {
        console.error('Failed to parse saved config, using default');
      }
    }

    const savedSessions = localStorage.getItem('openai_chat_sessions');
    if (savedSessions) {
      try {
        const parsed = JSON.parse(savedSessions);
        setSessions(parsed);
        if (parsed.length > 0) {
          setActiveSessionId(parsed[0].id);
        }
      } catch (e) {
        console.error('Failed to parse saved sessions');
      }
    }
  }, []);

  const saveSessionsToLocal = (newSessions: ChatSession[]) => {
    setSessions(newSessions);
    localStorage.setItem('openai_chat_sessions', JSON.stringify(newSessions));
  };

  const handleSaveConfig = (newConfig: ModelConfig) => {
    setConfig(newConfig);
    localStorage.setItem('openai_chat_config', JSON.stringify(newConfig));
  };

  const handleSelectSession = (id: string) => {
    setActiveSessionId(id);
  };

  const handleNewSession = (): ChatSession => {
    const newSession: ChatSession = {
      id: crypto.randomUUID(),
      title: 'Hội thoại mới',
      messages: [],
      systemInstruction: config.systemInstruction,
      temperature: config.temperature,
      createdAt: Date.now(),
    };

    const updatedSessions = [newSession, ...sessions];
    saveSessionsToLocal(updatedSessions);
    setActiveSessionId(newSession.id);
    return newSession;
  };

  const handleDeleteSession = (id: string) => {
    const updatedSessions = sessions.filter((s) => s.id !== id);
    saveSessionsToLocal(updatedSessions);

    if (activeSessionId === id) {
      if (updatedSessions.length > 0) {
        setActiveSessionId(updatedSessions[0].id);
      } else {
        setActiveSessionId(null);
      }
    }
  };

  const handleRenameSession = (id: string, newTitle: string) => {
    const updatedSessions = sessions.map((s) =>
      s.id === id ? { ...s, title: newTitle } : s
    );
    saveSessionsToLocal(updatedSessions);
  };

  const handleSendMessage = async (text: string) => {
    if (isStreaming) return;

    let currentSession = sessions.find((s) => s.id === activeSessionId);
    let sessionToUse: ChatSession;
    let sessionsListToMap = sessions;

    if (!currentSession) {
      sessionToUse = {
        id: crypto.randomUUID(),
        title: 'Hội thoại mới',
        messages: [],
        systemInstruction: config.systemInstruction,
        temperature: config.temperature,
        createdAt: Date.now(),
      };
      sessionsListToMap = [sessionToUse, ...sessions];
      setActiveSessionId(sessionToUse.id);
    } else {
      sessionToUse = currentSession;
    }

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: Date.now(),
      status: 'sending',
    };

    const isFirstMessage = sessionToUse.messages.length === 0;
    const originalTitle = sessionToUse.title;
    const newTitle = isFirstMessage
      ? text.length > 30
        ? text.substring(0, 30) + '...'
        : text
      : originalTitle;

    const updatedMessages = [...sessionToUse.messages, userMessage];

    const assistantMessageId = crypto.randomUUID();
    const assistantPlaceholderMessage: Message = {
      id: assistantMessageId,
      role: 'model',
      content: '',
      timestamp: Date.now(),
      modelUsed: 'Graph RAG Assistant',
    };

    const initialSessionsState = sessionsListToMap.map((s) => {
      if (s.id === sessionToUse.id) {
        return {
          ...s,
          title: newTitle,
          messages: [...updatedMessages, assistantPlaceholderMessage],
        };
      }
      return s;
    });

    saveSessionsToLocal(initialSessionsState);
    setIsStreaming(true);

    try {
      const response = await fetch(`${API_BASE}/api/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: updatedMessages,
          config: {
            systemInstruction: config.systemInstruction,
            temperature: config.temperature,
            topP: config.topP,
            topK: config.topK,
          },
          groupId: selectedGroupId,
        }),
      });

      if (!response.ok) {
        throw new Error('Không thể kết nối đến máy chủ API.');
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder('utf-8');

      if (!reader) {
        throw new Error('Dữ liệu luồng không phản hồi.');
      }

      setSessions((prevSessions) => {
        const next = prevSessions.map((s) => {
          if (s.id === sessionToUse.id) {
            return {
              ...s,
              messages: s.messages.map((m) =>
                m.id === userMessage.id
                  ? { ...m, status: 'sent' as const }
                  : m
              ),
            };
          }
          return s;
        });
        localStorage.setItem('openai_chat_sessions', JSON.stringify(next));
        return next;
      });

      let accumulatedText = '';
      const currentToolCalls: Record<number, ToolCallState> = {};
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        let boundary = buffer.indexOf('\n\n');

        while (boundary !== -1) {
          const rawMessage = buffer.slice(0, boundary).trim();
          buffer = buffer.slice(boundary + 2);

          const lines = rawMessage.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.slice(6).trim();
              if (dataStr === '[DONE]') {
                break;
              }
              let parsed: any = null;
              try {
                parsed = JSON.parse(dataStr);
              } catch (e: any) {
                continue;
              }

              if (parsed.error) {
                throw new Error(parsed.error);
              }

              const msgType = parsed.type || (parsed.text !== undefined ? 'text_delta' : 'unknown');

              if (msgType === 'text_delta') {
                const chunkText = parsed.content || parsed.text || '';
                accumulatedText += chunkText;
              } else if (msgType === 'tool_call_delta') {
                const index = parsed.index ?? 0;
                const existing = currentToolCalls[index] || { args: '', status: 'calling' };
                currentToolCalls[index] = {
                  ...existing,
                  id: parsed.id || existing.id,
                  name: parsed.name || existing.name,
                  args: existing.args + (parsed.args || ''),
                  status: existing.status === 'completed' ? 'completed' : 'calling',
                };
              } else if (msgType === 'tool_call_executing') {
                let targetIndex: number | null = null;
                for (const [k, v] of Object.entries(currentToolCalls)) {
                  const idx = Number(k);
                  if ((v as any).runId === parsed.id || v.id === parsed.id) {
                    targetIndex = idx;
                    break;
                  }
                }
                if (targetIndex === null) {
                  for (const [k, v] of Object.entries(currentToolCalls)) {
                    const idx = Number(k);
                    if (v.name === parsed.name && v.status !== 'completed') {
                      targetIndex = idx;
                      break;
                    }
                  }
                }
                if (targetIndex === null) {
                  const keys = Object.keys(currentToolCalls).map(Number);
                  targetIndex = keys.length > 0 ? Math.max(...keys) + 1 : 0;
                }

                const existing = currentToolCalls[targetIndex] || { args: '', status: 'executing' };
                currentToolCalls[targetIndex] = {
                  ...existing,
                  id: existing.id || parsed.id,
                  ...(parsed.id ? { runId: parsed.id } : {}),
                  name: parsed.name || existing.name,
                  input: parsed.input || existing.input,
                  status: 'executing',
                };
              } else if (msgType === 'tool_call_end') {
                let targetIndex: number | null = null;
                for (const [k, v] of Object.entries(currentToolCalls)) {
                  const idx = Number(k);
                  if ((v as any).runId === parsed.id || v.id === parsed.id) {
                    targetIndex = idx;
                    break;
                  }
                }
                if (targetIndex === null) {
                  for (const [k, v] of Object.entries(currentToolCalls)) {
                    const idx = Number(k);
                    if (v.name === parsed.name && v.status === 'executing') {
                      targetIndex = idx;
                      break;
                    }
                  }
                }
                if (targetIndex === null) {
                  for (const [k, v] of Object.entries(currentToolCalls)) {
                    const idx = Number(k);
                    if (v.name === parsed.name && v.status !== 'completed') {
                      targetIndex = idx;
                      break;
                    }
                  }
                }
                if (targetIndex === null) {
                  targetIndex = 0;
                }
                const existing = currentToolCalls[targetIndex] || { args: '', status: 'completed' };
                currentToolCalls[targetIndex] = {
                  ...existing,
                  status: 'completed',
                };
              }

              const toolCallsSnapshot = Object.keys(currentToolCalls).length > 0 ? { ...currentToolCalls } : undefined;

              setSessions((prevSessions) => {
                const next = prevSessions.map((s) => {
                  if (s.id === sessionToUse.id) {
                    return {
                      ...s,
                      messages: s.messages.map((m) => {
                        if (m.id === assistantMessageId) {
                          return {
                            ...m,
                            content: accumulatedText,
                            ...(toolCallsSnapshot ? { toolCalls: toolCallsSnapshot } : {})
                          };
                        }
                        if (m.id === userMessage.id && m.status !== 'delivered' && m.status !== 'read') {
                          return { ...m, status: 'delivered' as const };
                        }
                        return m;
                      }),
                    };
                  }
                  return s;
                });
                localStorage.setItem('openai_chat_sessions', JSON.stringify(next));
                return next;
              });
            }
          }
          boundary = buffer.indexOf('\n\n');
        }
      }

      // Mark any uncompleted tool calls as completed when stream ends
      if (Object.keys(currentToolCalls).length > 0) {
        for (const k of Object.keys(currentToolCalls)) {
          const idx = Number(k);
          if (currentToolCalls[idx] && currentToolCalls[idx].status !== 'completed') {
            currentToolCalls[idx].status = 'completed';
          }
        }
      }

      const finalToolCallsSnapshot = Object.keys(currentToolCalls).length > 0 ? { ...currentToolCalls } : undefined;

      setSessions((prevSessions) => {
        const next = prevSessions.map((s) => {
          if (s.id === sessionToUse.id) {
            return {
              ...s,
              messages: s.messages.map((m) => {
                if (m.id === userMessage.id) {
                  return { ...m, status: 'read' as const };
                }
                if (m.id === assistantMessageId && finalToolCallsSnapshot) {
                  return { ...m, toolCalls: finalToolCallsSnapshot };
                }
                return m;
              }),
            };
          }
          return s;
        });
        localStorage.setItem('openai_chat_sessions', JSON.stringify(next));
        return next;
      });
    } catch (error: any) {
      console.error('Streaming error:', error);

      setSessions((prevSessions) => {
        const next = prevSessions.map((s) => {
          if (s.id === sessionToUse.id) {
            return {
              ...s,
              messages: s.messages.map((m) =>
                m.id === assistantMessageId
                  ? {
                      ...m,
                      content: error.message || 'Đã có lỗi xảy ra trong quá trình xử lý. Vui lòng thử lại.',
                      error: true
                    }
                  : m
              ),
            };
          }
          return s;
        });
        localStorage.setItem('openai_chat_sessions', JSON.stringify(next));
        return next;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  const handleDeleteDocument = async (id: number) => {
    try {
      const res = await fetch(`${API_BASE}/api/documents/${id}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        await fetchDocuments();
      } else {
        alert('Xoá tài liệu thất bại!');
      }
    } catch (e) {
      console.error('Error deleting document:', e);
      alert('Lỗi kết nối khi xoá tài liệu!');
    }
  };

  const activeSession = sessions.find((s) => s.id === activeSessionId) || null;

  return (
    <main className="flex w-screen h-screen overflow-hidden bg-white font-sans antialiased text-slate-900" id="main-layout">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={handleSelectSession}
        onNewSession={handleNewSession}
        onDeleteSession={handleDeleteSession}
        onRenameSession={handleRenameSession}
        onOpenSettings={() => setIsSettingsOpen(true)}
        isMobileOpen={isMobileSidebarOpen}
        onToggleMobile={() => setIsMobileSidebarOpen(!isMobileSidebarOpen)}
        documents={documents}
        isUploading={isUploading}
        onUploadFile={handleUploadFile}
        onDeleteDocument={handleDeleteDocument}
        onRefreshDocuments={fetchDocuments}
        groups={groups}
        selectedGroupId={selectedGroupId}
        onSelectGroup={setSelectedGroupId}
        onCreateGroup={handleCreateGroup}
        onDeleteGroup={handleDeleteGroup}
        onOpenCommunity={() => setActiveView('community')}
      />

      {activeView === 'community' ? (
        <CommunityPanel onBack={() => setActiveView('chat')} />
      ) : (
        <ChatArea
          session={activeSession}
          onSendMessage={handleSendMessage}
          isStreaming={isStreaming}
          onToggleMobile={() => setIsMobileSidebarOpen(!isMobileSidebarOpen)}
        />
      )}

      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        config={config}
        onSaveConfig={handleSaveConfig}
      />
    </main>
  );
}
