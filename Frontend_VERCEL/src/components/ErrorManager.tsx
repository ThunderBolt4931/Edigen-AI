
import { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { AlertCircle, X, Clock } from "lucide-react";

interface ErrorItem {
  id: string;
  message: string;
  timestamp: Date;
  type: 'error' | 'warning';
}

interface ErrorManagerProps {
  errors: ErrorItem[];
  onClearError: (id: string) => void;
  onClearAll: () => void;
}

const ErrorManager = ({ errors, onClearError, onClearAll }: ErrorManagerProps) => {
  const [isOpen, setIsOpen] = useState(false);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour12: false, 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit' 
    });
  };

  if (errors.length === 0) return null;

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className="fixed bottom-4 right-4 z-50 bg-red-900/20 border-red-500/30 text-red-300 hover:bg-red-900/40 hover:border-red-500/50 transition-all duration-200 shadow-lg"
        >
          <AlertCircle className="w-4 h-4 mr-2" />




        </Button>
      </DialogTrigger>
      <DialogContent className="z-[1000] bg-slate-800 border-slate-700 text-white max-w-2xl max-h-[80vh]">
        <DialogHeader>
          <DialogTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-red-400" />
              Error Log ({errors.length})
            </div>
            {errors.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onClearAll}
                className="text-slate-400 hover:text-white hover:bg-slate-700"
              >
                Clear All
              </Button>
            )}
          </DialogTitle>
        </DialogHeader>
        <ScrollArea className="max-h-96">
          <div className="space-y-3">
            {errors.length === 0 ? (
              <div className="text-center py-8 text-slate-400">
                <AlertCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>🚫 No current errors</p>
              </div>
            ) : (
              errors.map((error) => (
                <div
                  key={error.id}
                  className="p-3 bg-red-900/20 border border-red-500/30 rounded-lg hover:bg-red-900/30 transition-colors duration-200"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
                        <span className="text-sm text-red-300 font-medium">
                          {error.type === 'error' ? 'Error' : 'Warning'}
                        </span>
                        <div className="flex items-center gap-1 text-xs text-slate-400">
                          <Clock className="w-3 h-3" />
                          {formatTime(error.timestamp)}
                        </div>
                      </div>
                      <p className="text-sm text-slate-300 leading-relaxed">
                        {error.message}
                      </p>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => onClearError(error.id)}
                      className="text-slate-400 hover:text-white hover:bg-slate-700 h-6 w-6 p-0 flex-shrink-0"
                    >
                      <X className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
              ))
            )}
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
};

export default ErrorManager;
