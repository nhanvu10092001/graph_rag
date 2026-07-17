/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import SettingsModal from './components/SettingsModal';
import { ChatSession, Message, ModelConfig, AVAILABLE_MODELS } from './types';

// Default corporate AI Assistant instruction
const DEFAULT_SYSTEM_INSTRUCTION = 
  "Bạn là một Trợ lý AI cao cấp chuyên nghiệp, lịch sự và đáng tin cậy. " +
  "Hãy cung cấp câu trả lời rõ ràng, chính xác, cấu trúc mạch lạc sử dụng định dạng Markdown " +
  "và viết bằng tiếng Việt tự nhiên trừ khi người dùng yêu cầu ngôn ngữ khác.";

const DEFAULT_CONFIG: ModelConfig = {
  model: 'gpt-4o-mini',
  systemInstruction: DEFAULT_SYSTEM_INSTRUCTION,
  temperature: 0.7,
  topP: 0.95,
  topK: 40,
};

const API_BASE = window.location.hostname === 'localhost' ? 'http://localhost:8000' : '';

export default function App() {
  // --- States ---
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<string>('gpt-4o-mini');
  const [config, setConfig] = useState<ModelConfig>(DEFAULT_CONFIG);
  const [customApiKey, setCustomApiKey] = useState<string>('');
  const [hasSystemKey, setHasSystemKey] = useState<boolean>(true);
  
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);

  // --- Document Indexing States & Actions ---
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
    console.log('App: handleCreateGroup called with:', name);
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
    // Poll documents status every 5 seconds
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

  // --- Initial Mount & State Loading ---
  useEffect(() => {
    // 1. Load config & API Key from localStorage
    const savedApiKey = localStorage.getItem('openai_chat_api_key') || '';
    setCustomApiKey(savedApiKey);

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

    // 2. Query System API Key availability on server
    const checkSystemConfig = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/config`);
        if (res.ok) {
          const data = await res.json();
          setHasSystemKey(!!data.hasSystemKey);
        }
      } catch (e) {
        console.error('Error contacting config API', e);
      }
    };
    checkSystemConfig();
  }, []);

  // --- State Persistence ---
  const saveSessionsToLocal = (newSessions: ChatSession[]) => {
    setSessions(newSessions);
    localStorage.setItem('openai_chat_sessions', JSON.stringify(newSessions));
  };

  const handleSaveConfig = (newConfig: ModelConfig) => {
    setConfig(newConfig);
    localStorage.setItem('openai_chat_config', JSON.stringify(newConfig));
  };

  const handleSaveApiKey = (newKey: string) => {
    setCustomApiKey(newKey);
    localStorage.setItem('openai_chat_api_key', newKey);
  };

  // --- Chat Handlers ---
  const handleSelectSession = (id: string) => {
    setActiveSessionId(id);
    const session = sessions.find((s) => s.id === id);
    if (session) {
      setSelectedModelId(session.model);
    }
  };

  const handleNewSession = (modelId: string): ChatSession => {
    const newSession: ChatSession = {
      id: crypto.randomUUID(),
      title: 'Hội thoại mới',
      messages: [],
      model: modelId,
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

  const handleSelectModel = (id: string) => {
    setSelectedModelId(id);
    // If we have an active session, update its model
    if (activeSessionId) {
      const updatedSessions = sessions.map((s) => 
        s.id === activeSessionId ? { ...s, model: id } : s
      );
      saveSessionsToLocal(updatedSessions);
    }
  };

  // --- Message Sending and SSE Streaming Logic ---
  const handleSendMessage = async (text: string) => {
    if (isStreaming) return;

    let currentSession = sessions.find((s) => s.id === activeSessionId);
    let sessionToUse: ChatSession;
    let sessionsListToMap = sessions;

    // 1. Create a session on the fly if none is active or exists
    if (!currentSession) {
      sessionToUse = {
        id: crypto.randomUUID(),
        title: 'Hội thoại mới',
        messages: [],
        model: selectedModelId,
        systemInstruction: config.systemInstruction,
        temperature: config.temperature,
        createdAt: Date.now(),
      };
      sessionsListToMap = [sessionToUse, ...sessions];
      setActiveSessionId(sessionToUse.id);
    } else {
      sessionToUse = currentSession;
    }

    // 2. Construct user message
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: Date.now(),
      status: 'sending',
    };

    // Auto rename conversation based on first prompt
    const isFirstMessage = sessionToUse.messages.length === 0;
    const originalTitle = sessionToUse.title;
    const newTitle = isFirstMessage 
      ? text.length > 30 
        ? text.substring(0, 30) + '...' 
        : text 
      : originalTitle;

    const updatedMessages = [...sessionToUse.messages, userMessage];
    
    // Create assistant placeholder message
    const assistantMessageId = crypto.randomUUID();
    const assistantPlaceholderMessage: Message = {
      id: assistantMessageId,
      role: 'model',
      content: '',
      timestamp: Date.now(),
      modelUsed: AVAILABLE_MODELS.find(m => m.id === sessionToUse.model)?.name || sessionToUse.model,
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
      // 3. Request SSE stream from the server
      const response = await fetch(`${API_BASE}/api/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: updatedMessages,
          model: sessionToUse.model,
          config: {
            systemInstruction: config.systemInstruction,
            temperature: config.temperature,
            topP: config.topP,
            topK: config.topK,
          },
          apiKey: customApiKey,
          groupId: selectedGroupId,
        }),
      });

      if (!response.ok) {
        throw new Error('Không thể kết nối đến máy chủ API hoặc khóa của bạn bị lỗi.');
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder('utf-8');

      if (!reader) {
        throw new Error('Dữ liệu luồng không phản hồi.');
      }

      // Transition user message status to 'sent'
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
                // Ignore parsing incomplete JSON lines
                continue;
              }

              if (parsed.error) {
                throw new Error(parsed.error);
              }

              if (parsed.text) {
                accumulatedText += parsed.text;

                // Update session's assistant message and user message status (to 'delivered')
                setSessions((prevSessions) => {
                  const next = prevSessions.map((s) => {
                    if (s.id === sessionToUse.id) {
                      return {
                        ...s,
                        messages: s.messages.map((m) => {
                          if (m.id === assistantMessageId) {
                            return { ...m, content: accumulatedText };
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
          }
          boundary = buffer.indexOf('\n\n');
        }
      }

      // Stream completed successfully, mark user message as 'read'
      setSessions((prevSessions) => {
        const next = prevSessions.map((s) => {
          if (s.id === sessionToUse.id) {
            return {
              ...s,
              messages: s.messages.map((m) => 
                m.id === userMessage.id 
                  ? { ...m, status: 'read' as const } 
                  : m
              ),
            };
          }
          return s;
        });
        localStorage.setItem('openai_chat_sessions', JSON.stringify(next));
        return next;
      });
    } catch (error: any) {
      console.error('Streaming error:', error);
      
      // Update assistant message with error state
      setSessions((prevSessions) => {
        const next = prevSessions.map((s) => {
          if (s.id === sessionToUse.id) {
            return {
              ...s,
              messages: s.messages.map((m) => 
                m.id === assistantMessageId 
                  ? { 
                      ...m, 
                      content: error.message || 'Đã có lỗi xảy ra trong quá trình truyền dữ liệu từ OpenAI API. Vui lòng kiểm tra lại cấu hình API Key.', 
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
      {/* Sidebar Navigation */}
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={handleSelectSession}
        onNewSession={handleNewSession}
        onDeleteSession={handleDeleteSession}
        onRenameSession={handleRenameSession}
        onOpenSettings={() => setIsSettingsOpen(true)}
        hasSystemKey={hasSystemKey}
        customApiKey={customApiKey}
        selectedModelId={selectedModelId}
        onSelectModel={handleSelectModel}
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
      />

      {/* Main Conversation Center Canvas */}
      <ChatArea
        session={activeSession}
        onSendMessage={handleSendMessage}
        isStreaming={isStreaming}
        onToggleMobile={() => setIsMobileSidebarOpen(!isMobileSidebarOpen)}
        hasSystemKey={hasSystemKey}
        customApiKey={customApiKey}
        onOpenSettings={() => setIsSettingsOpen(true)}
      />

      {/* Corporate Advanced Settings Panel Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        config={config}
        onSaveConfig={handleSaveConfig}
        customApiKey={customApiKey}
        onSaveApiKey={handleSaveApiKey}
        hasSystemKey={hasSystemKey}
      />
    </main>
  );
}
