import React, { useState, useEffect } from 'react';
import { AlertCircle, CheckCircle } from 'lucide-react';
import { ApiService } from '../services/api';

export function BackendStatus() {
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const healthy = await ApiService.healthCheck();
        setIsHealthy(healthy);
      } catch {
        setIsHealthy(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, []);

  if (isHealthy === null) return null;

  return (
    <div className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm ${
      isHealthy 
        ? 'bg-green-50 text-green-800 border border-green-200' 
        : 'bg-red-50 text-red-800 border border-red-200'
    }`}>
      {isHealthy ? (
        <CheckCircle size={14} />
      ) : (
        <AlertCircle size={14} />
      )}
      <span>
        Backend: {isHealthy ? 'Connected' : 'Disconnected'}
      </span>
    </div>
  );
}