/**
 * CommunityPanel — Trang quản lý Community Detection & Global Search
 * Cho phép người dùng tương tác với các API community detection từ UI.
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Network, Play, RefreshCw, Search, ChevronDown, ChevronUp,
  Loader2, CheckCircle2, AlertCircle, Sparkles, Layers, Globe2,
  BarChart3, Hash, ArrowLeft
} from 'lucide-react';

const API_BASE = window.location.hostname === 'localhost' ? 'http://localhost:8000' : '';

interface Community {
  id: string;
  title: string;
  summary: string;
  findings: string;
  importance_score: number;
  entity_count: number;
  level: number;
  score?: number;
}

interface CommunityPanelProps {
  onBack: () => void;
}

type ActionStatus = 'idle' | 'loading' | 'success' | 'error';

export default function CommunityPanel({ onBack }: CommunityPanelProps) {
  // --- State ---
  const [communities, setCommunities] = useState<Community[]>([]);
  const [selectedLevel, setSelectedLevel] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<Community[]>([]);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // Action statuses
  const [detectStatus, setDetectStatus] = useState<ActionStatus>('idle');
  const [summarizeStatus, setSummarizeStatus] = useState<ActionStatus>('idle');
  const [rebuildStatus, setRebuildStatus] = useState<ActionStatus>('idle');
  const [listStatus, setListStatus] = useState<ActionStatus>('idle');
  const [searchStatus, setSearchStatus] = useState<ActionStatus>('idle');

  // Detection stats
  const [detectionStats, setDetectionStats] = useState<any>(null);
  const [lastAction, setLastAction] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string>('');

  // --- API Calls ---

  const fetchCommunities = useCallback(async (level: number) => {
    setListStatus('loading');
    setErrorMessage('');
    try {
      const res = await fetch(`${API_BASE}/api/community/list?level=${level}`);
      if (res.ok) {
        const data = await res.json();
        setCommunities(data.communities || []);
        setListStatus('success');
      } else {
        const err = await res.json();
        setErrorMessage(err.detail || 'Lỗi khi tải danh sách communities');
        setListStatus('error');
      }
    } catch (e) {
      console.error('Error fetching communities:', e);
      setErrorMessage('Không thể kết nối đến server');
      setListStatus('error');
    }
  }, []);

  useEffect(() => {
    fetchCommunities(selectedLevel);
  }, [selectedLevel, fetchCommunities]);

  const handleDetect = async () => {
    setDetectStatus('loading');
    setLastAction('detect');
    setErrorMessage('');
    try {
      const res = await fetch(`${API_BASE}/api/community/detect`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setDetectionStats(data.data);
        setDetectStatus('success');
        // Refresh community list
        await fetchCommunities(selectedLevel);
      } else {
        const err = await res.json();
        setErrorMessage(err.detail || 'Community detection thất bại');
        setDetectStatus('error');
      }
    } catch (e) {
      console.error('Error detecting communities:', e);
      setErrorMessage('Không thể kết nối đến server');
      setDetectStatus('error');
    }
  };

  const handleSummarize = async () => {
    setSummarizeStatus('loading');
    setLastAction('summarize');
    setErrorMessage('');
    try {
      const res = await fetch(`${API_BASE}/api/community/summarize`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setDetectionStats(data.data);
        setSummarizeStatus('success');
        await fetchCommunities(selectedLevel);
      } else {
        const err = await res.json();
        setErrorMessage(err.detail || 'Tạo tóm tắt thất bại');
        setSummarizeStatus('error');
      }
    } catch (e) {
      console.error('Error summarizing communities:', e);
      setErrorMessage('Không thể kết nối đến server');
      setSummarizeStatus('error');
    }
  };

  const handleRebuild = async () => {
    if (!confirm('Thao tác này sẽ xóa toàn bộ communities cũ và xây dựng lại. Tiếp tục?')) return;
    setRebuildStatus('loading');
    setLastAction('rebuild');
    setErrorMessage('');
    try {
      const res = await fetch(`${API_BASE}/api/community/rebuild`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setDetectionStats(data.data);
        setRebuildStatus('success');
        await fetchCommunities(selectedLevel);
      } else {
        const err = await res.json();
        setErrorMessage(err.detail || 'Rebuild thất bại');
        setRebuildStatus('error');
      }
    } catch (e) {
      console.error('Error rebuilding communities:', e);
      setErrorMessage('Không thể kết nối đến server');
      setRebuildStatus('error');
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setSearchStatus('loading');
    setErrorMessage('');
    try {
      const res = await fetch(
        `${API_BASE}/api/community/search?query=${encodeURIComponent(searchQuery)}&top_k=5&level=${selectedLevel}`
      );
      if (res.ok) {
        const data = await res.json();
        setSearchResults(data.results || []);
        setSearchStatus('success');
      } else {
        const err = await res.json();
        setErrorMessage(err.detail || 'Tìm kiếm thất bại');
        setSearchStatus('error');
      }
    } catch (e) {
      console.error('Error searching communities:', e);
      setErrorMessage('Không thể kết nối đến server');
      setSearchStatus('error');
    }
  };

  const parseFindingsJson = (findings: string): string[] => {
    try {
      return JSON.parse(findings);
    } catch {
      return findings ? [findings] : [];
    }
  };

  const getImportanceColor = (score: number) => {
    if (score >= 0.8) return 'text-emerald-600 bg-emerald-50 border-emerald-100';
    if (score >= 0.6) return 'text-indigo-600 bg-indigo-50 border-indigo-100';
    if (score >= 0.4) return 'text-amber-600 bg-amber-50 border-amber-100';
    return 'text-slate-500 bg-slate-50 border-slate-200';
  };

  const renderStatusIcon = (status: ActionStatus) => {
    switch (status) {
      case 'loading': return <Loader2 className="w-4 h-4 animate-spin" />;
      case 'success': return <CheckCircle2 className="w-4 h-4 text-emerald-500" />;
      case 'error': return <AlertCircle className="w-4 h-4 text-rose-500" />;
      default: return null;
    }
  };

  const renderCommunityCard = (community: Community, showScore = false) => {
    const isExpanded = expandedId === community.id;
    const findings = parseFindingsJson(community.findings);

    return (
      <div
        key={community.id}
        className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm hover:shadow-md transition-shadow"
      >
        {/* Header */}
        <button
          type="button"
          onClick={() => setExpandedId(isExpanded ? null : community.id)}
          className="w-full flex items-center justify-between px-4 py-3 text-left cursor-pointer hover:bg-slate-50 transition bg-transparent border-0"
        >
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <div className="p-1.5 bg-indigo-50 border border-indigo-100 rounded-lg shrink-0">
              <Network className="w-4 h-4 text-indigo-600" />
            </div>
            <div className="min-w-0 flex-1">
              <h4 className="text-sm font-semibold text-slate-800 truncate">{community.title}</h4>
              <div className="flex items-center gap-2 mt-0.5">
                <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-bold tracking-wider font-mono border ${getImportanceColor(community.importance_score)}`}>
                  <BarChart3 className="w-2.5 h-2.5 mr-0.5" />
                  {(community.importance_score * 100).toFixed(0)}%
                </span>
                <span className="text-[10px] text-slate-400 font-mono flex items-center gap-0.5">
                  <Hash className="w-2.5 h-2.5" />{community.entity_count} entities
                </span>
                {showScore && community.score !== undefined && (
                  <span className="text-[10px] text-indigo-500 font-mono">
                    sim: {community.score.toFixed(3)}
                  </span>
                )}
              </div>
            </div>
          </div>
          {isExpanded ? <ChevronUp className="w-4 h-4 text-slate-400 shrink-0" /> : <ChevronDown className="w-4 h-4 text-slate-400 shrink-0" />}
        </button>

        {/* Expanded Content */}
        {isExpanded && (
          <div className="px-4 pb-4 space-y-3 border-t border-slate-100 pt-3">
            <div>
              <h5 className="text-[10px] font-bold text-slate-400 tracking-wider font-mono mb-1">TÓM TẮT</h5>
              <p className="text-xs text-slate-600 leading-relaxed">{community.summary}</p>
            </div>

            {findings.length > 0 && (
              <div>
                <h5 className="text-[10px] font-bold text-slate-400 tracking-wider font-mono mb-1">PHÁT HIỆN CHÍNH</h5>
                <ul className="space-y-1">
                  {findings.map((f, i) => (
                    <li key={i} className="flex items-start gap-2 text-xs text-slate-600">
                      <Sparkles className="w-3 h-3 text-amber-500 mt-0.5 shrink-0" />
                      <span>{f}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="flex items-center gap-3 text-[10px] text-slate-400 font-mono pt-1 border-t border-slate-100">
              <span>ID: {community.id}</span>
              <span>Level: {community.level ?? selectedLevel}</span>
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-50 overflow-hidden">
      {/* Top Bar */}
      <div className="flex items-center justify-between px-6 py-4 bg-white border-b border-slate-200 shrink-0">
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="p-1.5 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition cursor-pointer border-0 bg-transparent"
            title="Quay lại Chat"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div className="p-2 bg-indigo-50 border border-indigo-100 rounded-xl">
            <Globe2 className="w-5 h-5 text-indigo-600" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-slate-800">Community Detection</h2>
            <p className="text-[10px] text-slate-400 font-mono tracking-wider">LEIDEN ALGORITHM · GRAPH RAG</p>
          </div>
        </div>
      </div>

      {/* Main Content — Scrollable */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-6 py-6 space-y-6">

          {/* Action Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {/* Detect */}
            <button
              onClick={handleDetect}
              disabled={detectStatus === 'loading'}
              className={`group relative flex flex-col items-center gap-2 p-5 border rounded-xl text-center transition-all cursor-pointer ${
                detectStatus === 'loading'
                  ? 'bg-slate-50 border-slate-200 text-slate-400 cursor-wait'
                  : 'bg-white border-slate-200 hover:border-indigo-300 hover:shadow-md hover:bg-indigo-50/20 text-slate-700'
              }`}
            >
              <div className="p-2.5 bg-indigo-50 border border-indigo-100 rounded-xl group-hover:bg-indigo-100 transition">
                {detectStatus === 'loading' ? (
                  <Loader2 className="w-5 h-5 text-indigo-600 animate-spin" />
                ) : (
                  <Play className="w-5 h-5 text-indigo-600" />
                )}
              </div>
              <div>
                <span className="text-xs font-semibold">Phát hiện Communities</span>
                <p className="text-[10px] text-slate-400 mt-0.5">Chạy thuật toán Leiden</p>
              </div>
              {detectStatus !== 'idle' && (
                <div className="absolute top-2 right-2">{renderStatusIcon(detectStatus)}</div>
              )}
            </button>

            {/* Summarize */}
            <button
              onClick={handleSummarize}
              disabled={summarizeStatus === 'loading'}
              className={`group relative flex flex-col items-center gap-2 p-5 border rounded-xl text-center transition-all cursor-pointer ${
                summarizeStatus === 'loading'
                  ? 'bg-slate-50 border-slate-200 text-slate-400 cursor-wait'
                  : 'bg-white border-slate-200 hover:border-amber-300 hover:shadow-md hover:bg-amber-50/20 text-slate-700'
              }`}
            >
              <div className="p-2.5 bg-amber-50 border border-amber-100 rounded-xl group-hover:bg-amber-100 transition">
                {summarizeStatus === 'loading' ? (
                  <Loader2 className="w-5 h-5 text-amber-600 animate-spin" />
                ) : (
                  <Sparkles className="w-5 h-5 text-amber-600" />
                )}
              </div>
              <div>
                <span className="text-xs font-semibold">Tạo Tóm tắt</span>
                <p className="text-[10px] text-slate-400 mt-0.5">LLM sinh summary cho mỗi nhóm</p>
              </div>
              {summarizeStatus !== 'idle' && (
                <div className="absolute top-2 right-2">{renderStatusIcon(summarizeStatus)}</div>
              )}
            </button>

            {/* Rebuild */}
            <button
              onClick={handleRebuild}
              disabled={rebuildStatus === 'loading'}
              className={`group relative flex flex-col items-center gap-2 p-5 border rounded-xl text-center transition-all cursor-pointer ${
                rebuildStatus === 'loading'
                  ? 'bg-slate-50 border-slate-200 text-slate-400 cursor-wait'
                  : 'bg-white border-slate-200 hover:border-rose-300 hover:shadow-md hover:bg-rose-50/20 text-slate-700'
              }`}
            >
              <div className="p-2.5 bg-rose-50 border border-rose-100 rounded-xl group-hover:bg-rose-100 transition">
                {rebuildStatus === 'loading' ? (
                  <Loader2 className="w-5 h-5 text-rose-600 animate-spin" />
                ) : (
                  <RefreshCw className="w-5 h-5 text-rose-600" />
                )}
              </div>
              <div>
                <span className="text-xs font-semibold">Xây dựng lại</span>
                <p className="text-[10px] text-slate-400 mt-0.5">Xóa cũ → Detect → Summarize</p>
              </div>
              {rebuildStatus !== 'idle' && (
                <div className="absolute top-2 right-2">{renderStatusIcon(rebuildStatus)}</div>
              )}
            </button>
          </div>

          {/* Error / Stats Banner */}
          {errorMessage && (
            <div className="flex items-center gap-2 p-3 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-700">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{errorMessage}</span>
            </div>
          )}

          {detectionStats && (
            <div className="p-4 bg-indigo-50/50 border border-indigo-100 rounded-xl space-y-2">
              <h4 className="text-[10px] font-bold text-indigo-400 tracking-wider font-mono">KẾT QUẢ THỰC THI</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {Object.entries(detectionStats).map(([key, value]) => {
                  if (typeof value === 'object') return null;
                  return (
                    <div key={key} className="p-2 bg-white rounded-lg border border-indigo-100 text-center">
                      <div className="text-[10px] text-slate-400 font-mono truncate">{key}</div>
                      <div className="text-sm font-bold text-indigo-700">{String(value)}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Level Selector + Search */}
          <div className="flex flex-col md:flex-row gap-3">
            {/* Level Selector */}
            <div className="flex items-center gap-2 bg-white border border-slate-200 rounded-xl px-3 py-2 shrink-0">
              <Layers className="w-4 h-4 text-slate-400" />
              <span className="text-[10px] font-bold text-slate-400 tracking-wider font-mono">LEVEL</span>
              <div className="flex gap-1 ml-1">
                {[0, 1, 2].map((lvl) => (
                  <button
                    key={lvl}
                    onClick={() => setSelectedLevel(lvl)}
                    className={`px-2.5 py-1 text-xs font-semibold rounded-lg transition cursor-pointer border-0 ${
                      selectedLevel === lvl
                        ? 'bg-indigo-600 text-white shadow-sm'
                        : 'bg-slate-100 text-slate-500 hover:bg-slate-200'
                    }`}
                  >
                    {lvl}
                  </button>
                ))}
              </div>
            </div>

            {/* Search Bar */}
            <div className="flex-1 flex items-center gap-2 bg-white border border-slate-200 rounded-xl px-3 py-2">
              <Search className="w-4 h-4 text-slate-400 shrink-0" />
              <input
                type="text"
                placeholder="Tìm kiếm community theo nội dung..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                className="flex-1 text-xs text-slate-700 placeholder:text-slate-300 bg-transparent border-0 outline-none"
              />
              <button
                onClick={handleSearch}
                disabled={searchStatus === 'loading' || !searchQuery.trim()}
                className="px-3 py-1 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-medium rounded-lg transition cursor-pointer border-0 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1"
              >
                {searchStatus === 'loading' ? <Loader2 className="w-3 h-3 animate-spin" /> : <Search className="w-3 h-3" />}
                Tìm
              </button>
            </div>
          </div>

          {/* Search Results */}
          {searchResults.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-[10px] font-bold text-indigo-400 tracking-wider font-mono flex items-center gap-1.5">
                <Search className="w-3.5 h-3.5" />
                KẾT QUẢ TÌM KIẾM ({searchResults.length})
              </h3>
              <div className="space-y-2">
                {searchResults.map((c) => renderCommunityCard(c, true))}
              </div>
            </div>
          )}

          {/* Community List */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="text-[10px] font-bold text-slate-400 tracking-wider font-mono flex items-center gap-1.5">
                <Network className="w-3.5 h-3.5" />
                COMMUNITIES TẠI LEVEL {selectedLevel} ({communities.length})
              </h3>
              {listStatus === 'loading' && <Loader2 className="w-4 h-4 text-indigo-500 animate-spin" />}
            </div>

            {communities.length === 0 && listStatus !== 'loading' ? (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <div className="p-4 bg-slate-100 rounded-2xl mb-3">
                  <Network className="w-8 h-8 text-slate-300" />
                </div>
                <p className="text-sm text-slate-400 font-medium">Chưa có community nào</p>
                <p className="text-xs text-slate-300 mt-1">Nhấn "Phát hiện Communities" để bắt đầu</p>
              </div>
            ) : (
              <div className="space-y-2">
                {communities.map((c) => renderCommunityCard(c))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
