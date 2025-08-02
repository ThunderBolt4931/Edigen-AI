
import { useState, useCallback } from 'react';

interface ErrorItem {
  id: string;
  message: string;
  timestamp: Date;
  type: 'error' | 'warning';
}

export const useErrorManager = () => {
  const [errors, setErrors] = useState<ErrorItem[]>([]);

  const addError = useCallback((message: string, type: 'error' | 'warning' = 'error') => {
    const newError: ErrorItem = {
      id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
      message,
      timestamp: new Date(),
      type
    };
    setErrors(prev => [...prev, newError]);
  }, []);

  const clearError = useCallback((id: string) => {
    setErrors(prev => prev.filter(error => error.id !== id));
  }, []);

  const clearAllErrors = useCallback(() => {
    setErrors([]);
  }, []);

  return {
    errors,
    addError,
    clearError,
    clearAllErrors
  };
};
