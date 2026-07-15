/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export type Role = 'user' | 'model';

export interface Message {
  id: string;
  role: Role;
  content: string;
  timestamp: number;
  modelUsed?: string;
  error?: boolean;
  status?: 'sending' | 'sent' | 'delivered' | 'read';
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  model: string;
  systemInstruction?: string;
  temperature?: number;
  createdAt: number;
}

export interface ModelOption {
  id: string;
  name: string;
  description: string;
  category: 'flash' | 'pro' | 'experimental';
  isPaid: boolean;
  maxTokens?: number;
}

export interface ModelConfig {
  model: string;
  systemInstruction: string;
  temperature: number;
  topP?: number;
  topK?: number;
}

export const AVAILABLE_MODELS: ModelOption[] = [
  {
    id: 'gemini-3.5-flash',
    name: 'Gemini 3.5 Flash',
    description: 'Bản phát hành mới nhất, phản hồi cực nhanh, tối ưu hóa cho hầu hết các tác vụ hội thoại và đa phương tiện.',
    category: 'flash',
    isPaid: false,
  },
  {
    id: 'gemini-3.1-pro-preview',
    name: 'Gemini 3.1 Pro (Preview)',
    description: 'Mô hình tiên tiến nhất cho các tác vụ phức tạp, viết code, tư duy thuật toán và phân tích chuyên sâu.',
    category: 'pro',
    isPaid: true,
  }
];
