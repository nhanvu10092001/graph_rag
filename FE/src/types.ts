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
  thinking?: string;
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
  systemInstruction?: string;
  temperature?: number;
  createdAt: number;
}

export interface ModelConfig {
  systemInstruction: string;
  temperature: number;
  topP?: number;
  topK?: number;
}
