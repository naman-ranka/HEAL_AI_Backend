import { Button } from "@/components/ui/button";
import { Menu, User, Shield } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import MedicalLogo from "@/components/ui/medical-logo";

const Header = () => {
  const location = useLocation();
  const isDevelopment = process.env.NODE_ENV === 'development';

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/80 backdrop-blur-sm">
      <div className="container mx-auto flex h-16 items-center justify-between px-4">
        <div className="flex items-center gap-2">
          <Link to="/" className="flex items-center gap-2">
            <MedicalLogo size={32} />
            <span className="text-xl font-bold text-foreground">HEAL.AI</span>
          </Link>
        </div>
        
        <nav className="hidden md:flex items-center gap-6">
          <Link 
            to="/" 
            className={`text-muted-foreground hover:text-foreground transition-colors ${
              location.pathname === '/' ? 'text-foreground font-medium' : ''
            }`}
          >
            Home
          </Link>
          <Link 
            to="/dashboard" 
            className={`text-muted-foreground hover:text-foreground transition-colors ${
              location.pathname === '/dashboard' ? 'text-foreground font-medium' : ''
            }`}
          >
            Dashboard
          </Link>
          <Link 
            to="/chat" 
            className={`text-muted-foreground hover:text-foreground transition-colors ${
              location.pathname === '/chat' ? 'text-foreground font-medium' : ''
            }`}
          >
            Chat
          </Link>
          {isDevelopment && (
            <Link 
              to="/admin" 
              className={`text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1 ${
                location.pathname === '/admin' ? 'text-foreground font-medium' : ''
              }`}
            >
              <Shield className="h-4 w-4" />
              Admin
            </Link>
          )}
        </nav>

        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" className="hidden md:flex">
            <User className="h-4 w-4" />
            Sign In
          </Button>
          <Button variant="hero" size="sm">
            Get Started
          </Button>
          <Button variant="ghost" size="icon" className="md:hidden">
            <Menu className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </header>
  );
};

export default Header;