/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { X, Key, Sliders, Info, ShieldAlert, CheckCircle, AlertCircle, HelpCircle } from 'lucide-react';
import { ModelConfig } from '../types';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: ModelConfig;
  onSaveConfig: (newConfig: ModelConfig) => void;
  customApiKey: string;
  onSaveApiKey: (key: string) => void;
  hasSystemKey: boolean;
}

export default function SettingsModal({
  isOpen,
  onClose,
  config,
  onSaveConfig,
  customApiKey,
  onSaveApiKey,
  hasSystemKey,
}: SettingsModalProps) {
  const [localApiKey, setLocalApiKey] = useState(customApiKey);
  const [systemInstruction, setSystemInstruction] = useState(config.systemInstruction);
  const [temperature, setTemperature] = useState(config.temperature);
  const [topP, setTopP] = useState(config.topP || 0.95);
  const [topK, setTopK] = useState(config.topK || 40);

  const [testingKey, setTestingKey] = useState(false);
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null);

  // Sync state with props
  useEffect(() => {
    if (isOpen) {
      setLocalApiKey(customApiKey);
      setSystemInstruction(config.systemInstruction);
      setTemperature(config.temperature);
      setTopP(config.topP || 0.95);
      setTopK(config.topK || 40);
      setTestResult(null);
    }
  }, [isOpen, customApiKey, config]);

  if (!isOpen) return null;

  const handleTestKey = async () => {
    setTestingKey(true);
    setTestResult(null);
    try {
      const response = await fetch('/api/verify-key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ apiKey: localApiKey }),
      });
      const data = await response.json();
      if (response.ok && data.valid) {
        setTestResult({ success: true, message: 'Khóa API hợp lệ! Bạn có thể sử dụng.' });
      } else {
        setTestResult({ success: false, message: data.message || 'Khóa API không hợp lệ.' });
      }
    } catch (err: any) {
      setTestResult({ success: false, message: 'Không thể kết nối đến máy chủ xác thực.' });
    } finally {
      setTestingKey(false);
    }
  };

  const handleSave = () => {
    onSaveApiKey(localApiKey.trim());
    onSaveConfig({
      model: config.model,
      systemInstruction: systemInstruction.trim(),
      temperature,
      topP,
      topK,
    });
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/25 backdrop-blur-sm transition-opacity duration-300">
      <div 
        className="relative w-full max-w-2xl bg-white border border-slate-200 rounded-2xl shadow-xl overflow-hidden flex flex-col max-h-[90vh] animate-in fade-in zoom-in-95 duration-200 text-slate-800"
        id="settings-modal"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 bg-slate-50 border-b border-slate-200">
          <div className="flex items-center gap-2">
            <Sliders className="w-5 h-5 text-indigo-600" />
            <h2 className="text-lg font-semibold text-slate-900">Cấu Hình Doanh Nghiệp</h2>
          </div>
          <button 
            onClick={onClose}
            className="text-slate-400 hover:text-slate-700 hover:bg-slate-100 p-1.5 rounded-lg transition"
            id="close-settings-btn"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Scrollable Content */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1 bg-white">
          {/* Section 1: API Key */}
          <div className="space-y-3 pb-6 border-b border-slate-200">
            <div className="flex items-center gap-2 text-slate-800 font-semibold">
              <Key className="w-4 h-4 text-indigo-600" />
              <h3>Quản Lý OpenAI API Key</h3>
            </div>
            
            <p className="text-xs text-slate-500 leading-relaxed">
              Mặc định ứng dụng sử dụng <strong>API Key Hệ Thống</strong> được cấu hình sẵn bởi quản trị viên. Nếu muốn dùng hạn ngạch (quota) riêng của bạn, vui lòng cấu hình khóa cá nhân bên dưới. Khóa của bạn chỉ lưu tạm trên trình duyệt (LocalStorage) và không lưu trên máy chủ của chúng tôi.
            </p>

            <div className="space-y-2 mt-2">
              <label className="block text-xs font-medium text-slate-600">API Key Cá Nhân của bạn (Tùy chọn)</label>
              <div className="flex gap-2">
                <input
                  type="password"
                  placeholder={hasSystemKey ? "Đã bật API Key Hệ Thống (Bỏ trống để sử dụng)" : "Nhập API Key OpenAI của bạn..."}
                  value={localApiKey}
                  onChange={(e) => setLocalApiKey(e.target.value)}
                  className="flex-1 bg-slate-50 border border-slate-200 focus:border-indigo-500 rounded-xl px-4 py-2 text-sm text-slate-800 placeholder:text-slate-300 focus:outline-none transition font-mono"
                  id="api-key-input"
                />
                <button
                  type="button"
                  onClick={handleTestKey}
                  disabled={testingKey}
                  className="px-4 py-2 bg-slate-100 hover:bg-slate-200 disabled:opacity-50 text-slate-700 text-xs font-medium rounded-xl border border-slate-200 transition flex items-center justify-center gap-1 min-w-[100px] cursor-pointer shadow-sm"
                  id="test-key-btn"
                >
                  {testingKey ? "Đang thử..." : "Kiểm tra"}
                </button>
              </div>

              {testResult && (
                <div className={`p-3 rounded-xl border text-xs flex items-start gap-2 ${
                  testResult.success 
                    ? "bg-emerald-50 border-emerald-200 text-emerald-700" 
                    : "bg-rose-50 border-rose-200 text-rose-700"
                }`}>
                  {testResult.success ? (
                    <CheckCircle className="w-4 h-4 mt-0.5 shrink-0" />
                  ) : (
                    <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
                  )}
                  <span>{testResult.message}</span>
                </div>
              )}

              {hasSystemKey && !localApiKey && (
                <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 text-xs text-indigo-600/85 flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-indigo-600 shrink-0" />
                  <span>Hệ thống đã sẵn sàng: Đang sử dụng API Key máy chủ.</span>
                </div>
              )}
            </div>
          </div>

          {/* Section 2: System Instruction */}
          <div className="space-y-3 pb-6 border-b border-slate-200">
            <div className="flex items-center justify-between text-slate-800 font-semibold">
              <div className="flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-indigo-600" />
                <h3>Chỉ Thị Hệ Thống (System Instruction)</h3>
              </div>
              <span title="Quy định vai trò, phong cách trả lời và định dạng phản hồi của mô hình AI.">
                <HelpCircle className="w-4 h-4 text-slate-400 hover:text-slate-600 transition cursor-help" />
              </span>
            </div>
            
            <p className="text-xs text-slate-500">
              Thiết lập chỉ thị để định hình tính cách và hành vi phản hồi của trợ lý AI trước khi bắt đầu cuộc hội thoại.
            </p>

            <textarea
              placeholder="Ví dụ: Bạn là một trợ lý lập trình viên chuyên nghiệp, luôn viết code sạch, giải thích súc tích và sử dụng ngôn ngữ tiếng Việt lịch sự..."
              value={systemInstruction}
              onChange={(e) => setSystemInstruction(e.target.value)}
              className="w-full h-24 bg-slate-50 border border-slate-200 focus:border-indigo-500 rounded-xl px-4 py-2 text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none transition resize-none leading-relaxed"
              id="system-instruction-textarea"
            />
          </div>

          {/* Section 3: Advanced Hyperparameters */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-slate-800 font-semibold">
              <Sliders className="w-4 h-4 text-indigo-600" />
              <h3>Tham Số Mô Hình (Hyperparameters)</h3>
            </div>

            {/* Temperature Slider */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-750 font-medium">Độ Sáng Tạo (Temperature): {temperature.toFixed(1)}</span>
                <span className="text-slate-400">Thấp = Chính xác | Cao = Sáng tạo</span>
              </div>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-indigo-600 border border-slate-200/50"
                id="temperature-slider"
              />
            </div>

            {/* Top P and Top K */}
            <div className="grid grid-cols-2 gap-4 pt-2">
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-750 font-medium">Top P: {topP.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={topP}
                  onChange={(e) => setTopP(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-indigo-600 border border-slate-200/50"
                  id="topp-slider"
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-750 font-medium">Top K: {topK}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="100"
                  step="1"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-indigo-600 border border-slate-200/50"
                  id="topk-slider"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 px-6 py-4 bg-slate-50 border-t border-slate-200">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-white hover:bg-slate-100 border border-slate-200 text-slate-600 hover:text-slate-800 text-sm font-medium rounded-xl transition cursor-pointer shadow-sm"
            id="cancel-settings-btn"
          >
            Hủy bỏ
          </button>
          <button
            onClick={handleSave}
            className="px-5 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-semibold rounded-xl shadow-sm hover:shadow-md transition cursor-pointer"
            id="save-settings-btn"
          >
            Lưu cấu hình
          </button>
        </div>
      </div>
    </div>
  );
}
