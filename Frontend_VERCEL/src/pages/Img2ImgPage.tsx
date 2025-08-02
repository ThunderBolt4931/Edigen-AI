import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import AppHeader from "@/components/AppHeader";
import ErrorManager from "@/components/ErrorManager";
import { useErrorManager } from "@/hooks/useErrorManager";
import Img2ImgTab from "@/components/Img2ImgTab";
import { Button } from "@/components/ui/button";
import { Monitor } from "lucide-react";

const Img2ImgPage = () => {
  const navigate = useNavigate();
  const { errors, addError, clearError, clearAllErrors } = useErrorManager();

  const handleSystemDashboard = () => {
    navigate('/systemdashboard');
  };

  return (
    <>
      {/* Background Spline 3D iframe */}
      <div className="fixed inset-0 z-0">
        <iframe 
          src='https://my.spline.design/retrofuturisticcircuitloop-mMGk86WKmMW9qCpLyPQkAZR0/' 
          frameBorder='0' 
          width='100%' 
          height='100%'
          className="pointer-events-none"
        />
      </div>
      <AppHeader />
      <div className="min-h-screen bg-transparent relative z-10">
        <div className="container mx-auto p-4 pt-20">
          

          <div className="w-full">
            <div className="mb-6 flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold text-white mb-2">Image to Image</h1>
                <p className="text-gray-300">Transform existing images with AI-powered modifications and style transfers.</p>
              </div>
              <Button
                onClick={handleSystemDashboard}
                className="inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium transition-all bg-green-600/20 text-green-300 hover:bg-green-600/30"
              >
                <Monitor className="w-4 h-4 mr-2" />
                System Dashboard
              </Button>
            </div>

            <Img2ImgTab onError={addError} />
          </div>
        </div>

        <ErrorManager
          errors={errors}
          onClearError={clearError}
          onClearAll={clearAllErrors}
        />
      </div>
    </>
  );
};

export default Img2ImgPage;
