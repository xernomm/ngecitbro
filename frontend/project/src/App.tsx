import React, { useState } from 'react';
import { MessageSquare, FileText, Trash2 } from 'lucide-react';
import { DocumentManager } from './components/DocumentManager';
import { ChatInterface } from './components/ChatInterface';
import { BackendStatus } from './components/BackendStatus';
import { useDocuments } from './hooks/useDocuments';
import { useChat } from './hooks/useChat';

function App() {
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const { documents, refetch: refetchDocuments } = useDocuments();
  const { messages, setMessages, clearChat } = useChat();

  return (
    <div className="h-screen bg-gray-50 flex">
      {/* Sidebar */}
      <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center gap-2 mb-4">
            <MessageSquare className="text-blue-600" size={24} />
            <h1 className="text-xl font-bold text-gray-800">NGECIT BRO?</h1>
          </div>
          <BackendStatus />
        </div>

        <DocumentManager
          documents={documents}
          selectedDocs={selectedDocs}
          onDocumentsChange={refetchDocuments}
          onSelectedDocsChange={setSelectedDocs}
        />

        <div className="p-4 border-t border-gray-200">
          <button
            onClick={clearChat}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
          >
            <Trash2 size={16} />
            Clear Chat
          </button>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col justify-center">
        <div className="bg-white border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-800">Chat</h2>
            <div className="text-sm text-gray-600">
              {messages.length} message{messages.length !== 1 ? 's' : ''}
            </div>
          </div>
        </div>

        <ChatInterface
          selectedDocs={selectedDocs}
          messages={messages}
          setMessages={setMessages}
        />
      </div>
    </div>
  );
}

export default App;