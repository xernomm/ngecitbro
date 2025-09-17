import { useState, useEffect, useCallback } from 'react';
import { ApiService } from '../services/api';

export type DocEntry = { doc_id: string; filename?: string | null };

export function useDocuments() {
  const [documents, setDocuments] = useState<DocEntry[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);

  const fetchDocuments = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      // ApiService.listDocuments() should return DocEntry[] (not a Response)
      const docs: DocEntry[] = await ApiService.listDocuments();
      if (!Array.isArray(docs)) {
        throw new Error('Unexpected response shape from listDocuments');
      }
      setDocuments(docs);
    } catch (err: any) {
      console.error('Failed to fetch documents:', err);
      setDocuments([]);
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  return {
    documents,
    isLoading,
    error,
    refetch: fetchDocuments,
  };
}
