import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import ScrollToTop from "./components/ScrollToTop";
import LandingPage from "./pages/LandingPage";
import AppPage from "./pages/AppPage";
import Txt2ImgPage from "./pages/Txt2ImgPage";
import Img2ImgPage from "./pages/Img2ImgPage";
import InpaintingPage from "./pages/InpaintingPage";
import TryOnPage from "./pages/TryOnPage";
import SystemDashboardPage from "./pages/SystemDashboardPage";

import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <ScrollToTop />
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/app" element={<AppPage />} />
          <Route path="/txt2img" element={<Txt2ImgPage />} />
          <Route path="/img2img" element={<Img2ImgPage />} />
          <Route path="/inpainting" element={<InpaintingPage />} />
          <Route path="/tryon" element={<TryOnPage />} />
          <Route path="/systemdashboard" element={<SystemDashboardPage />} />
          
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
