/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export type Role = 'user' | 'model';

export interface ToolCallState {
  id?: string;
  name?: string;
  args: string;
  input?: any;
  status: 'calling' | 'executing' | 'completed' | 'error';
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  toolCalls?: Record<number, ToolCallState>;
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
    id: 'gpt-4o-mini',
    name: 'GPT-4o Mini',
    description: 'Mô hình nhanh, nhẹ, chi phí thấp, tối ưu hóa cho hầu hết các tác vụ hội thoại và phân tích văn bản.',
    category: 'flash',
    isPaid: false,
  },
  {
    id: 'gpt-4o',
    name: 'GPT-4o',
    description: 'Mô hình tiên tiến nhất của OpenAI cho các tác vụ phức tạp, viết code, tư duy thuật toán và phân tích chuyên sâu.',
    category: 'pro',
    isPaid: true,
  }
];
