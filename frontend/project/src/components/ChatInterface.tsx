import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2 } from 'lucide-react';
import { Message } from '../types';
import { ApiService } from '../services/api';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';

interface ChatInterfaceProps {
  selectedDocs: string[];
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
}

export function ChatInterface({ selectedDocs, messages, setMessages }: ChatInterfaceProps) {
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState('');
  const [tabStates, setTabStates] = useState<Record<string, number>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingMessage]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      role: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    setStreamingMessage('');

    try {
      const assistantMessageId = (Date.now() + 1).toString();

      const response = await ApiService.sendChatMessage({
        user_input: inputMessage,
        doc_ids: selectedDocs.length > 0 ? selectedDocs : undefined,
        n_results: 5,
        model: 'deepseek-r1:latest',
        system_instructions: 'You are "CITER", an assistant for a college student, answering quiz questions or essays. You must answer based on the provided context. Try as hard as you can to fit the answer and question by the provided context. If there is none you may think your own answer',
      });

      // Faster streaming: batch updates per chunk instead of per character
      const fullResponse = response.answer || '';
      const CHUNK_SIZE = 12;        // adjust larger -> fewer updates, smaller -> more real-time
      const YIELD_DELAY_MS = 0;     // 0 yields to event loop, set to 1 or higher to slow down

      let buffer = '';
      let displayed = '';

      for (let i = 0; i < fullResponse.length; i++) {
        buffer += fullResponse[i];

        // When buffer reaches CHUNK_SIZE or it's the last char, flush to UI
        if (buffer.length >= CHUNK_SIZE || i === fullResponse.length - 1) {
          displayed += buffer;
          setStreamingMessage(displayed);
          buffer = '';

          // yield to UI so browser can repaint (non-blocking)
          if (YIELD_DELAY_MS > 0) {
            await new Promise(resolve => setTimeout(resolve, YIELD_DELAY_MS));
          } else {
            await new Promise(resolve => setTimeout(resolve, 0));
          }
        }
      }

      // Add final message (full)
      const assistantMessage: Message = {
        id: assistantMessageId,
        content: fullResponse,
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
      setStreamingMessage('');
    } catch (error) {
      console.error('Chat failed:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: 'Sorry, I encountered an error. Please make sure your backend is running and try again.',
        role: 'assistant',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Extract think & main (already in your original)
  const extractThinkAndMain = (content: string) => {
    const thinkMatch = content.match(/<think>([\s\S]*?)<\/think>/i);
    const thinkContent = thinkMatch ? thinkMatch[1].trim() : null;
    const mainContent = content.replace(/<think>[\s\S]*?<\/think>/i, '').trim();
    return { thinkContent, mainContent };
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const adjustTextareaHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [inputMessage]);

  // Tab control per message
  const handleTabChange = (messageId: string, tabIndex: number) => {
    setTabStates(prev => ({ ...prev, [messageId]: tabIndex }));
  };

  return (
    <div style={{height: '93%'}} className="flex flex-col w-4/6 justify-center mx-auto">
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 && !streamingMessage && (
          <div className="text-center text-gray-500 mt-8">
            <Bot size={48} className="mx-auto mb-4 text-gray-400" />
            <p className="text-lg">Yok gas gada limit ni</p>
            <p className="text-sm mt-2">
              {selectedDocs.length > 0
                ? `Using ${selectedDocs.length} document(s) as context`
                : 'No documents selected - general chat mode'
              }
            </p>
          </div>
        )}

        {messages.map((message) => {
          const isUser = message.role === 'user';
          const { thinkContent, mainContent } = extractThinkAndMain(message.content);
          const currentTab = tabStates[message.id] ?? 0;

          return (
            <div
              key={message.id}
              className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start'}`}
            >
              {!isUser && (
                <div className="flex-shrink-0 w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                  <Bot size={16} className="text-white" />
                </div>
              )}

              <div
                className={`  ${isUser ? 'bg-blue-600 text-white px-4 py-3 rounded-2xl max-w-3xl' : ' text-gray-800 col-span-full text-lg'}`}
              >
                <div className="whitespace-pre-wrap">
                  {/* If assistant and has think -> render tabs */}
                  {!isUser && thinkContent ? (
                    <div>
                      <div className="flex items-center gap-2 mb-3">
                        <button
                          onClick={() => handleTabChange(message.id, 0)}
                          className={`text-xs px-2 py-1 rounded ${currentTab === 0 ? 'bg-blue-600 text-white' : 'bg-white text-gray-800 border'}`}
                          aria-pressed={currentTab === 0}
                        >
                          📝 Final Answer
                        </button>
                        <button
                          onClick={() => handleTabChange(message.id, 1)}
                          className={`text-xs px-2 py-1 rounded ${currentTab === 1 ? 'bg-blue-600 text-white' : 'bg-white text-gray-800 border'}`}
                          aria-pressed={currentTab === 1}
                        >
                          🧠 Process
                        </button>
                      </div>

                      <div>
                        {currentTab === 0 ? (
                          <Markdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
                            {mainContent}
                          </Markdown>
                        ) : (
                          <Markdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
                            {thinkContent}
                          </Markdown>
                        )}
                      </div>
                    </div>
                  ) : (
                    // normal rendering (user or assistant without think)
                    <Markdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
                      {message.content}
                    </Markdown>
                  )}
                </div>

                <div className="text-xs opacity-70 mt-2">
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>

              {isUser && (
                <div className="flex-shrink-0 w-8 h-8 bg-gray-600 rounded-full flex items-center justify-center">
                  <User size={16} className="text-white" />
                </div>
              )}
            </div>
          );
        })}

        {streamingMessage && (
          <div className="flex gap-4 justify-start">
            <div className="flex-shrink-0 w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
              <Bot size={16} className="text-white" />
            </div>
            <div className="text-gray-800 col-span-full text-lg">
              <div className="whitespace-pre-wrap">
                <Markdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
                  {streamingMessage}
                </Markdown>
              </div>
              <div className="inline-block w-1 h-5 bg-black animate-pulse ml-1"></div>
            </div>
          </div>
        )}

        {isLoading && !streamingMessage && (
          <div className="flex gap-4 justify-start">
            <div className="flex-shrink-0 w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
              <Bot size={16} className="text-white" />
            </div>
            <div className="max-w-3xl px-4 py-3 rounded-2xl bg-gray-100 text-gray-800 flex items-center gap-2">
              <Loader2 size={16} className="animate-spin" />
              <span>Sabar jons sebats dlu..</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="border-t bg-slate-50 border-gray-200">
        {selectedDocs.length > 0 && (
          <div className="mb-3 p-2 bg-blue-50 rounded-lg">
            <p className="text-xs text-blue-700 font-medium mb-1">Using contexts:</p>
            <div className="flex flex-wrap gap-1">
              {selectedDocs.map(docId => (
                <span key={docId} className="text-xs bg-blue-200 text-blue-800 px-2 py-1 rounded">
                  {docId}
                </span>
              ))}
            </div>
          </div>
        )}

        <form onSubmit={handleSubmit} className="flex gap-3">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message..."
              className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none min-h-[48px] max-h-[120px]"
              rows={1}
              disabled={isLoading}
            />
          </div>
          <button
            type="submit"
            disabled={!inputMessage.trim() || isLoading}
            className="flex-shrink-0 w-12 h-12 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center transition-colors"
          >
            <Send size={18} />
          </button>
        </form>
      </div>
    </div>
  );
}
