import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { MessageSquare, BarChart3, ArrowRight, Search, Upload, Shield, AlertCircle, CheckCircle } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { useState, useEffect } from "react";
import { useApp } from "@/contexts/AppContext";
import { useToast } from "@/hooks/use-toast";
import MedicalLogo from "@/components/ui/medical-logo";

const Index = () => {
  const { userProfile, uploadInsurance, isLoading, error, backendHealthy, healthCheckLoading } = useApp();
  const [uploading, setUploading] = useState(false);
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleInsuranceUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    
    try {
      // Upload and analyze with real backend
      await uploadInsurance(file);
      
      toast({
        title: "Insurance Uploaded Successfully! 🎉",
        description: `Your insurance document "${file.name}" has been analyzed and processed.`,
      });
      
      // Redirect to dashboard after successful upload to show parsed info
      setTimeout(() => navigate("/dashboard"), 1500);
      
    } catch (error) {
      console.error('Upload failed:', error);
      toast({
        title: "Upload Failed",
        description: error instanceof Error ? error.message : "Failed to upload insurance document. Please try again.",
        variant: "destructive",
      });
    } finally {
      setUploading(false);
    }
  };

  // Show backend status warning if not healthy (only after initial health check)
  useEffect(() => {
    if (!healthCheckLoading && !backendHealthy) {
      toast({
        title: "Backend Connection Issue",
        description: "Unable to connect to HEAL backend. Some features may not work properly.",
        variant: "destructive",
      });
    }
  }, [backendHealthy, healthCheckLoading, toast]);
  return (
    <>
      {/* Hero Section */}
      <section className="py-20 px-4">
        <div className="container mx-auto text-center max-w-5xl">
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold mb-6 leading-tight">
            <span className="bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
              <span className="text-pink-500">HEAL</span>thcare Expenses Analyzer & Logger
            </span>
          </h1>
          <p className="text-lg sm:text-xl text-muted-foreground mb-8 max-w-3xl mx-auto leading-relaxed px-4">
            Simplify your medical bills, understand insurance coverage, and detect billing errors with AI-powered analysis
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            {healthCheckLoading && (
              <div className="flex items-center gap-2 text-blue-600 text-sm mb-4">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                <span>Checking backend connection...</span>
              </div>
            )}
            {!healthCheckLoading && !backendHealthy && (
              <div className="flex items-center gap-2 text-red-600 text-sm mb-4">
                <AlertCircle className="h-4 w-4" />
                <span>Backend connection issue - Please check if the server is running</span>
              </div>
            )}
            <label htmlFor="hero-insurance-upload" className="cursor-pointer">
              <Button 
                size="lg" 
                className="text-lg px-8" 
                disabled={uploading || isLoading}
                asChild
              >
                <div>
                  <Upload className="mr-2 h-5 w-5" />
                  {uploading ? 'Uploading...' : userProfile?.insuranceUploaded ? 'Insurance Uploaded ✓' : 'Upload Insurance'}
                </div>
              </Button>
              <input
                id="hero-insurance-upload"
                type="file"
                accept=".pdf,.png,.jpg,.jpeg"
                onChange={handleInsuranceUpload}
                className="hidden"
                disabled={uploading || isLoading}
              />
            </label>
            {userProfile?.insuranceUploaded && (
              <Link to="/dashboard">
                <Button size="lg" variant="outline" className="text-lg px-8">
                  <BarChart3 className="mr-2 h-5 w-5" />
                  View Dashboard
                </Button>
              </Link>
            )}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 px-4 bg-muted/50">
        <div className="container mx-auto max-w-6xl">
          <h2 className="text-3xl font-bold text-center mb-12">How <span className="text-pink-500">HEAL</span>.AI Helps You</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <Card>
              <CardHeader>
                <MessageSquare className="h-10 w-10 text-primary mb-4" />
                <CardTitle>AI-Powered Chat</CardTitle>
                <CardDescription>
                  Upload bills, ask questions, and get instant analysis through our conversational AI interface
                </CardDescription>
              </CardHeader>
            </Card>
            <Card>
              <CardHeader>
                <BarChart3 className="h-10 w-10 text-primary mb-4" />
                <CardTitle>Comprehensive Dashboard</CardTitle>
                <CardDescription>
                  Track insurance, medical bills, health history, and billing disputes all in one place
                </CardDescription>
              </CardHeader>
            </Card>
            <Card>
              <CardHeader>
                <Search className="h-10 w-10 text-primary mb-4" />
                <CardTitle>Error Detection</CardTitle>
                <CardDescription>
                  Automatically flag billing errors, duplicate charges, and insurance coverage issues
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4">
        <div className="container mx-auto text-center max-w-3xl">
          <h2 className="text-3xl font-bold mb-6">Ready to Take Control of Your Healthcare Costs?</h2>
          <p className="text-muted-foreground mb-8 text-lg">
            Start by uploading your insurance documents to unlock personalized medical bill analysis
          </p>
          <div className="flex justify-center">
            <label htmlFor="cta-insurance-upload" className="cursor-pointer">
              <Button 
                size="lg" 
                className="text-lg px-8" 
                disabled={uploading || isLoading}
                asChild
              >
                <div>
                  <Upload className="mr-2 h-5 w-5" />
                  {uploading ? 'Processing...' : userProfile?.insuranceUploaded ? 'Insurance Uploaded - Get Started!' : 'Upload Insurance to Begin'}
                  <ArrowRight className="ml-2 h-5 w-5" />
                </div>
              </Button>
              <input
                id="cta-insurance-upload"
                type="file"
                accept=".pdf,.png,.jpg,.jpeg"
                onChange={handleInsuranceUpload}
                className="hidden"
                disabled={uploading || isLoading}
              />
            </label>
          </div>
          {userProfile?.insuranceUploaded && (
            <div className="mt-6 flex items-center justify-center gap-2 text-green-600">
              <Shield className="h-5 w-5" />
              <span>Your insurance is uploaded and secure. Ready to analyze your bills!</span>
            </div>
          )}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t py-8 px-4">
        <div className="container mx-auto text-center">
          <div className="flex items-center justify-center gap-2 mb-4">
            <MedicalLogo size={24} />
            <span className="font-semibold"><span className="text-pink-500">HEAL</span>.AI</span>
          </div>
          <p className="text-muted-foreground">
            Healthcare Expenses Analyzer and Logger - Simplifying medical bills with AI
          </p>
        </div>
      </footer>
    </>
  );
};

export default Index;