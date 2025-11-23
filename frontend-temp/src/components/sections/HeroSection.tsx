import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Upload, Shield, AlertTriangle, QrCode } from "lucide-react";
import heroImage from "@/assets/hero-medical-dashboard.jpg";

const HeroSection = () => {
  return (
    <section className="min-h-screen bg-gradient-to-br from-background via-muted/30 to-accent/20 flex items-center">
      <div className="container mx-auto px-4 py-16">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Content */}
          <div className="space-y-8 animate-fade-up">
            <div className="space-y-4">
              <h1 className="text-4xl md:text-6xl font-bold text-foreground leading-tight">
                Simplify Your
                <span className="bg-gradient-primary bg-clip-text text-transparent"> Medical Bills</span>
              </h1>
              <p className="text-xl text-muted-foreground leading-relaxed">
                Transform confusing medical bills into clear, actionable insights. 
                Know exactly what you owe and why with AI-powered analysis.
              </p>
            </div>

            <div className="flex flex-col sm:flex-row gap-4">
              <Button variant="hero" size="lg" className="text-lg px-8">
                <Upload className="h-5 w-5" />
                Upload Your Bill
              </Button>
              <Button variant="outline" size="lg" className="text-lg">
                Watch Demo
              </Button>
            </div>

            {/* Trust Indicators */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-8">
              <Card className="p-4 text-center bg-gradient-card shadow-card">
                <Shield className="h-8 w-8 text-primary mx-auto mb-2" />
                <p className="text-sm font-medium">HIPAA Secure</p>
              </Card>
              <Card className="p-4 text-center bg-gradient-card shadow-card">
                <AlertTriangle className="h-8 w-8 text-warning mx-auto mb-2" />
                <p className="text-sm font-medium">Error Detection</p>
              </Card>
              <Card className="p-4 text-center bg-gradient-card shadow-card">
                <QrCode className="h-8 w-8 text-accent-foreground mx-auto mb-2" />
                <p className="text-sm font-medium">Emergency QR</p>
              </Card>
              <Card className="p-4 text-center bg-gradient-card shadow-card">
                <div className="h-8 w-8 bg-gradient-primary rounded-full mx-auto mb-2 flex items-center justify-center">
                  <span className="text-white text-xs font-bold">AI</span>
                </div>
                <p className="text-sm font-medium">AI Powered</p>
              </Card>
            </div>
          </div>

          {/* Right Content */}
          <div className="relative">
            <div className="relative rounded-2xl overflow-hidden shadow-elevated">
              <img 
                src={heroImage} 
                alt="HEAL.AI Dashboard showing simplified medical bill breakdown"
                className="w-full h-auto"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-primary/20 to-transparent"></div>
            </div>
            
            {/* Floating Cards */}
            <Card className="absolute -top-4 -left-4 p-4 bg-success text-success-foreground shadow-elevated">
              <p className="text-sm font-semibold">✓ $250 Saved</p>
              <p className="text-xs opacity-90">Error Found</p>
            </Card>
            
            <Card className="absolute -bottom-4 -right-4 p-4 bg-primary text-primary-foreground shadow-elevated">
              <p className="text-sm font-semibold">Insurance: 80%</p>
              <p className="text-xs opacity-90">Coverage Clear</p>
            </Card>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;