export interface Document {
  doc_id: string;
  status?: string;
}

export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

export interface ChatRequest {
  user_input: string;
  doc_ids?: string[];
  n_results?: number;
  model?: string;
  system_instructions?: string;
}

export interface ContextRetrieveResponse {
  found: boolean;
  context: string;
}

export interface ChatResponse {
  answer: string;
  context: string;
  doc_ids_used: string[];
  model: string;
}