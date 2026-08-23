/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import CommunityPanel from './components/CommunityPanel';
import SettingsModal from './components/SettingsModal';
import { ChatSession, Message, ModelConfig, ToolCallState } from './types';

const DEFAULT_SYSTEM_INSTRUCTION =
  "You are a professional, polite, and reliable AI assistant. " +
  "Provide clear, accurate, well-structured answers using Markdown format " +
  "and write in natural English unless the user requests a different language.";

const DEFAULT_CONFIG: ModelConfig = {
  systemInstruction: DEFAULT_SYSTEM_INSTRUCTION,
  temperature: 0.7,
  topP: 0.95,
  topK: 40,
};

const API_BASE = window.location.hostname === 'localhost' ? 'http://localhost:8000' : '';

function getWsUrl(): string {
  const loc = window.location;
  const protocol = loc.protocol === 'https:' ? 'wss:' : 'ws:';
  if (loc.hostname === 'localhost' && loc.port !== '8000') {
    return 'ws://localhost:8000/ws/chat';
  }
  return `${protocol}//${loc.host}/ws/chat`;
}

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

  const wsRef = useRef<WebSocket | null>(null);
  const messageHandlerRef = useRef<((data: any) => void) | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    const ws = new WebSocket(getWsUrl());
    wsRef.current = ws;

    ws.onopen = () => {
      reconnectAttemptRef.current = 0;
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        messageHandlerRef.current?.(data);
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };

    ws.onclose = (event) => {
      wsRef.current = null;
      if (!event.wasClean) {
        const delay = Math.min(1000 * Math.pow(2, reconnectAttemptRef.current), 30000);
        reconnectAttemptRef.current++;
        console.log(`WebSocket closed unexpectedly. Reconnecting in ${delay}ms...`);
        reconnectTimerRef.current = setTimeout(connectWebSocket, delay);
      }
    };

    ws.onerror = () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
    };
  }, [connectWebSocket]);

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data);
      }
    } catch (e) {
      console.error('Error fetching documents', e);
    }
  };

  useEffect(() => {
    fetchDocuments();
    const interval = setInterval(fetchDocuments, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleUploadFile = async (file: File) => {
    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch(`${API_BASE}/api/documents/upload`, {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        await fetchDocuments();
      } else {
        const err = await res.json().catch(() => ({}));
        alert(`Document upload failed: ${err.detail || 'Unknown error'}`);
      }
    } catch (e) {
      console.error('Error uploading document:', e);
      alert('Connection error when uploading document!');
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
      title: 'New Conversation',
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
        title: 'New Conversation',
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

    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      connectWebSocket();
      await new Promise<void>((resolve, reject) => {
        const checkInterval = setInterval(() => {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            clearInterval(checkInterval);
            resolve();
          }
        }, 100);
        setTimeout(() => {
          clearInterval(checkInterval);
          reject(new Error('WebSocket connection timeout'));
        }, 5000);
      }).catch((err) => {
        setIsStreaming(false);
        setSessions((prev) => {
          const next = prev.map((s) => {
            if (s.id === sessionToUse.id) {
              return {
                ...s,
                messages: s.messages.map((m) =>
                  m.id === assistantMessageId
                    ? { ...m, content: err.message || 'Connection failed.', error: true }
                    : m
                ),
              };
            }
            return s;
          });
          localStorage.setItem('openai_chat_sessions', JSON.stringify(next));
          return next;
        });
        return;
      });
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
    let accumulatedThinking = '';
    const currentToolCalls: Record<number, ToolCallState> = {};

    const updateSessionMessages = () => {
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
                    ...(accumulatedThinking ? { thinking: accumulatedThinking } : {}),
                    ...(toolCallsSnapshot ? { toolCalls: toolCallsSnapshot } : {}),
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
    };

    const finalizeStream = () => {
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
                if (m.id === assistantMessageId) {
                  return {
                    ...m,
                    content: accumulatedText,
                    ...(accumulatedThinking ? { thinking: accumulatedThinking } : {}),
                    ...(finalToolCallsSnapshot ? { toolCalls: finalToolCallsSnapshot } : {}),
                  };
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

      setIsStreaming(false);
      messageHandlerRef.current = null;
    };

    messageHandlerRef.current = (parsed: any) => {
      if (parsed.error && parsed.type !== 'error') {
        setSessions((prev) => {
          const next = prev.map((s) => {
            if (s.id === sessionToUse.id) {
              return {
                ...s,
                messages: s.messages.map((m) =>
                  m.id === assistantMessageId
                    ? { ...m, content: parsed.error, error: true }
                    : m
                ),
              };
            }
            return s;
          });
          localStorage.setItem('openai_chat_sessions', JSON.stringify(next));
          return next;
        });
        setIsStreaming(false);
        messageHandlerRef.current = null;
        return;
      }

      const msgType = parsed.type;

      if (msgType === 'text_delta') {
        const chunkText = parsed.content || parsed.text || '';
        accumulatedText += chunkText;
        updateSessionMessages();
      } else if (msgType === 'thinking_delta') {
        const chunkThinking = parsed.content || '';
        accumulatedThinking += chunkThinking;
        updateSessionMessages();
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
        updateSessionMessages();
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
        } as ToolCallState;
        updateSessionMessages();
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
          output: parsed.output || undefined,
        };
        updateSessionMessages();
      } else if (msgType === 'done') {
        finalizeStream();
      } else if (msgType === 'cancelled') {
        finalizeStream();
      } else if (msgType === 'error') {
        setSessions((prev) => {
          const next = prev.map((s) => {
            if (s.id === sessionToUse.id) {
              return {
                ...s,
                messages: s.messages.map((m) =>
                  m.id === assistantMessageId
                    ? {
                        ...m,
                        content: parsed.error || 'An error occurred during processing. Please try again.',
                        error: true,
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
        setIsStreaming(false);
        messageHandlerRef.current = null;
      }
    };

    wsRef.current!.send(JSON.stringify({
      type: 'chat',
      messages: updatedMessages.map((m) => ({ role: m.role, content: m.content })),
      config: {
        systemInstruction: config.systemInstruction,
        temperature: config.temperature,
        topP: config.topP,
        topK: config.topK,
      },
    }));
  };

  const handleStopGeneration = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'cancel' }));
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
        const err = await res.json().catch(() => ({}));
        alert(`Document deletion failed: ${err.detail || 'Unknown error'}`);
      }
    } catch (e) {
      console.error('Error deleting document:', e);
      alert('Connection error when deleting document!');
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
          onStopGeneration={handleStopGeneration}
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
