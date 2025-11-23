import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { MessageCircle, LayoutDashboard, Upload } from "lucide-react";
import MedicalLogo from "@/components/ui/medical-logo";

interface LayoutProps {
  children: React.ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
  const location = useLocation();
  const navigate = useNavigate();
  const [hasInsurance, setHasInsurance] = useState(false);
  
  // Different header styles based on current page
  const isLandingPage = location.pathname === "/";
  const isDashboard = location.pathname === "/dashboard";
  const isChat = location.pathname === "/chat";
  const isLogin = location.pathname === "/login";

  // Check for insurance upload status
  React.useEffect(() => {
    const userData = localStorage.getItem("userProfile");
    if (userData) {
      const data = JSON.parse(userData);
      if (data.insuranceUploaded) {
        setHasInsurance(true);
      }
    }
  }, []);

  // Don't show header on login page
  if (isLogin) {
    return <>{children}</>;
  }

  const handleToggle = () => {
    if (isChat) {
      navigate("/dashboard");
    } else if (isDashboard) {
      navigate("/chat");
    } else if (isLandingPage) {
      // Default to chat from landing page
      navigate("/chat");
    }
  };

  const handleInsuranceUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setHasInsurance(true);
      // Store in localStorage
      const userData = {
        insuranceUploaded: true,
        insuranceFile: file.name
      };
      localStorage.setItem("userProfile", JSON.stringify(userData));
      
      // Redirect to chat after successful upload
      navigate("/chat");
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Persistent Header */}
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto flex h-16 items-center justify-between px-4">
          <Link to="/" className="flex items-center gap-2">
            <MedicalLogo size={32} />
            <span className="text-xl font-bold">HEAL.AI</span>
          </Link>
          
          {/* Center Toggle for Chat/Dashboard */}
          <div className="absolute left-1/2 transform -translate-x-1/2">
            <div className="relative bg-white/10 backdrop-blur-md border border-white/20 rounded-full p-1 shadow-lg">
              <div className="relative flex items-center">
                {/* Sliding Background Indicator */}
                <div
                  className={`absolute top-0 bottom-0 bg-gradient-to-r from-blue-500/30 to-blue-600/30 backdrop-blur-sm rounded-full transition-all duration-300 ease-in-out shadow-inner border border-blue-400/30 ${
                    isChat
                      ? "left-0 w-[calc(50%-2px)]" 
                      : isDashboard
                      ? "left-[calc(50%+2px)] w-[calc(50%-2px)]"
                      : "opacity-0 left-0 w-[calc(50%-2px)]"
                  }`}
                />
                
                {/* Chat Option */}
                <button
                  onClick={handleToggle}
                  className={`relative z-10 flex items-center gap-2 px-4 py-2.5 rounded-full transition-all duration-300 min-w-[120px] justify-center ${
                    isChat
                      ? "text-white font-semibold shadow-sm"
                      : "text-muted-foreground hover:text-foreground cursor-pointer"
                  }`}
                >
                  <MessageCircle className={`h-4 w-4 ${isChat ? 'text-white' : ''}`} />
                  <span className="text-sm font-medium">Chat</span>
                </button>
                
                {/* Dashboard Option */}
                <button
                  onClick={handleToggle}
                  className={`relative z-10 flex items-center gap-2 px-4 py-2.5 rounded-full transition-all duration-300 min-w-[120px] justify-center ${
                    isDashboard
                      ? "text-white font-semibold shadow-sm"
                      : "text-muted-foreground hover:text-foreground cursor-pointer"
                  }`}
                >
                  <LayoutDashboard className={`h-4 w-4 ${isDashboard ? 'text-white' : ''}`} />
                  <span className="text-sm font-medium">Dashboard</span>
                </button>
              </div>
              
              {/* Visual Enhancement - Active State Glow */}
              {isChat && (
                <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/20 to-blue-600/20 rounded-full blur opacity-30"></div>
              )}
              {isDashboard && (
                <div className="absolute -inset-0.5 bg-gradient-to-r from-green-500/20 to-emerald-600/20 rounded-full blur opacity-30"></div>
              )}
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Show upload button */}
            <div className="flex items-center gap-2">
              <label htmlFor="header-insurance-upload" className="cursor-pointer">
                <Button 
                  variant="outline" 
                  size="sm" 
                  className={`flex items-center gap-2 ${hasInsurance ? 'text-green-600 border-green-600 hover:bg-green-50' : 'text-blue-600 border-blue-600 hover:bg-blue-50'}`}
                  asChild
                >
                  <div>
                    <Upload className="h-4 w-4" />
                    {hasInsurance ? 'Insurance Uploaded ✓' : 'Upload Insurance'}
                  </div>
                </Button>
                <input
                  id="header-insurance-upload"
                  type="file"
                  accept=".pdf,.png,.jpg,.jpeg"
                  onChange={handleInsuranceUpload}
                  className="hidden"
                />
              </label>
            </div>
          </div>
        </div>
      </header>

      {/* Page Content */}
      {children}
    </div>
  );
};

export default Layout;
