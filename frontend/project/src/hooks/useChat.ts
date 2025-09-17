import { useState } from 'react';
import { Message } from '../types';

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);

  const clearChat = () => {
    setMessages([]);
  };

  return {
    messages,
    setMessages,
    clearChat,
  };
}