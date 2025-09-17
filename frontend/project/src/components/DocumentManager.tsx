import React, { useState, useRef } from 'react';
import { Upload, FileText, Trash2, Plus, X, AlertTriangle } from 'lucide-react';
import { ApiService } from '../services/api';

// --- Definisi tipe untuk respons API yang lebih baik ---
interface UploadResponse {
  status: 'ok' | 'exists' | 'error';
  doc_id: string;
  message?: string;
  saved_to: string;
  index_result?: any;
}

// documents sekarang adalah array objek { doc_id, filename }
interface DocEntry {
  doc_id: string;
  filename?: string | null;
}

interface DocumentManagerProps {
  documents: DocEntry[];                 // updated shape
  selectedDocs: string[];                // still list of doc_id
  onDocumentsChange: () => void;
  onSelectedDocsChange: (docs: string[]) => void;
}

export function DocumentManager({
  documents,
  selectedDocs,
  onDocumentsChange,
  onSelectedDocsChange,
}: DocumentManagerProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [showManualInput, setShowManualInput] = useState(false);
  const [manualText, setManualText] = useState('');
  const [docToDelete, setDocToDelete] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  console.log(documents)
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    try {
      const response: UploadResponse = await ApiService.uploadDocument(file);
      
      if (response.status === 'exists') {
        alert(`File uploaded. Note: This document ID ('${response.doc_id}') was already indexed. No new indexing was performed.`);
      } else {
        // optional success notification
      }
      onDocumentsChange();
    } catch (error: any) {
      console.error('Upload failed:', error);
      alert(`Upload failed: ${error?.message ?? String(error)}`);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleManualSubmit = async () => {
    if (!manualText.trim()) return;
    try {
      await ApiService.addManualContext(manualText);
      setManualText('');
      setShowManualInput(false);
      onDocumentsChange();
    } catch (error: any) {
      console.error('Manual context failed:', error);
      alert(`Failed to add context: ${error?.message ?? String(error)}`);
    }
  };

  const confirmDeleteDocument = async () => {
    if (!docToDelete) return;

    try {
      await ApiService.deleteDocument(docToDelete);
      onDocumentsChange();
      onSelectedDocsChange(selectedDocs.filter(id => id !== docToDelete));
    } catch (error: any) {
      console.error('Delete failed:', error);
      alert(`Delete failed: ${error?.message ?? String(error)}`);
    } finally {
      setDocToDelete(null);
    }
  };

  const handleDocSelection = (docId: string) => {
    if (selectedDocs.includes(docId)) {
      onSelectedDocsChange(selectedDocs.filter(id => id !== docId));
    } else {
      onSelectedDocsChange([...selectedDocs, docId]);
    }
  };
  
  const supportedFileTypes = ".pdf,.docx,.txt,.md,.py,.json,.csv,.html";

  return (
    <div className="h-full flex flex-col bg-gray-50">
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-800 mb-4">Males baca? upload sini!</h2>
        
        <div className="space-y-3">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors shadow-sm"
          >
            <Upload size={16} />
            {isUploading ? 'Uploading...' : 'Upload File'}
          </button>

          <button
            onClick={() => setShowManualInput(true)}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors shadow-sm"
          >
            <Plus size={16} />
            Add Text Context
          </button>

          <input
            ref={fileInputRef}
            type="file"
            accept={supportedFileTypes}
            onChange={handleFileUpload}
            className="hidden"
          />
        </div>
      </div>

      {showManualInput && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-5xl h-[80vh] mx-4 p-6 flex flex-col relative">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-gray-800">Gas copas bro, gw tau lu males baca awokwokwok.</h3>
              <button
                onClick={() => setShowManualInput(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                <X size={24} />
              </button>
            </div>
            <div className="flex-1">
              <textarea
                placeholder="Enter context text..."
                value={manualText}
                onChange={(e) => setManualText(e.target.value)}
                className="w-full h-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
              />
            </div>
            <div className="mt-4 flex justify-end gap-3">
              <button
                onClick={() => setShowManualInput(false)}
                className="px-4 py-2 rounded-lg bg-gray-200 text-gray-700 hover:bg-gray-300 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleManualSubmit}
                disabled={!manualText.trim()}
                className="px-4 py-2 rounded-lg bg-green-600 text-white hover:bg-green-700 disabled:opacity-50 transition-colors"
              >
                Add Context
              </button>
            </div>
          </div>
        </div>
      )}
      
      {docToDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4 p-6 flex flex-col">
            <div className="flex items-center mb-4">
              <div className="mr-4 flex-shrink-0 flex items-center justify-center h-12 w-12 rounded-full bg-red-100">
                  <AlertTriangle className="h-6 w-6 text-red-600" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-gray-900">Hapus jejak bro</h3>
                <p className="text-sm text-gray-600 mt-1">
                  Kalo lu males biar gw aja yang tau, wkwkwk.
                </p>
              </div>
            </div>
            <div className="bg-gray-100 p-2 my-2 rounded-md">
                {/* tampilkan filename jika ada + doc_id */}
                <p className="text-sm text-center font-mono break-all">
                  {(() => {
                    const entry = documents.find(d => d.doc_id === docToDelete);
                    if (!entry) return docToDelete;
                    return entry.filename ? `${entry.filename} — (${entry.doc_id})` : entry.doc_id;
                  })()}
                </p>
            </div>
            <div className="mt-4 flex justify-end gap-3">
              <button
                onClick={() => setDocToDelete(null)}
                className="px-4 py-2 rounded-lg bg-gray-200 text-gray-800 hover:bg-gray-300 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={confirmDeleteDocument}
                className="px-4 py-2 rounded-lg bg-red-600 text-white hover:bg-red-700 transition-colors"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-4">
        <h3 className="font-medium text-gray-800 mb-3">List yang lu males baca:</h3>
        {documents.length === 0 ? (
          <p className="text-gray-500 text-sm">Anjay, gausah sok rajin lo literasi minim awokwokwok</p>
        ) : (
          <div className="space-y-2">
            {documents.map((doc) => (
              <div
                key={doc.doc_id}
                className={`p-3 rounded-lg border cursor-pointer transition-all ${
                  selectedDocs.includes(doc.doc_id)
                    ? 'bg-blue-50 border-blue-300'
                    : 'bg-white border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => handleDocSelection(doc.doc_id)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 min-w-0">
                    <FileText size={16} className="text-gray-600 flex-shrink-0" />
                    <span className="text-sm font-medium text-gray-800 truncate">
                      {doc.filename && doc.filename.trim() ? doc.filename : doc.doc_id}
                    </span>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setDocToDelete(doc.doc_id);
                    }}
                    className="text-gray-400 hover:text-red-600 transition-colors ml-2"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
                {selectedDocs.includes(doc.doc_id) && (
                  <div className="mt-1 text-xs text-blue-600">Mantap :v</div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
