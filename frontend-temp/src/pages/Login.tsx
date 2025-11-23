import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Link, useNavigate } from "react-router-dom";
import { Upload, FileCheck, AlertCircle } from "lucide-react";
import MedicalLogo from "@/components/ui/medical-logo";

const Login = () => {
  const [formData, setFormData] = useState({
    name: "",
    medicalHistory: "",
    emergencyContact: "",
    insuranceUploaded: false
  });
  
  const [insuranceFile, setInsuranceFile] = useState<File | null>(null);
  const navigate = useNavigate();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Store user data (in a real app, this would be sent to a server)
    const userData = {
      ...formData,
      insuranceFile: insuranceFile?.name || null
    };
    localStorage.setItem("userProfile", JSON.stringify(userData));
    navigate("/dashboard");
  };

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setInsuranceFile(file);
      setFormData(prev => ({
        ...prev,
        insuranceUploaded: true
      }));
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center px-4">
      <div className="w-full max-w-2xl">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex justify-center items-center gap-3 mb-4">
            <MedicalLogo size={40} />
            <h1 className="text-3xl font-bold">HEAL.AI</h1>
          </div>
          <h2 className="text-2xl font-semibold mb-2">Get Started</h2>
          <p className="text-muted-foreground">
            Please provide your information to personalize your healthcare experience
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Welcome to HEAL.AI</CardTitle>
            <CardDescription>
              Your AI-powered healthcare expenses analyzer and logger
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Name */}
              <div className="space-y-2">
                <Label htmlFor="name">Full Name *</Label>
                <Input
                  id="name"
                  type="text"
                  placeholder="Enter your full name"
                  value={formData.name}
                  onChange={(e) => handleInputChange("name", e.target.value)}
                  required
                />
              </div>

              {/* Medical History */}
              <div className="space-y-2">
                <Label htmlFor="medicalHistory">Medical History *</Label>
                <Textarea
                  id="medicalHistory"
                  placeholder="Please describe your medical history, current conditions, medications, allergies, etc."
                  value={formData.medicalHistory}
                  onChange={(e) => handleInputChange("medicalHistory", e.target.value)}
                  rows={4}
                  required
                />
              </div>

              {/* Insurance Upload */}
              <div className="space-y-2">
                <Label htmlFor="insuranceUpload">Insurance Card/Policy Document *</Label>
                <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6">
                  <div className="text-center space-y-4">
                    <div className="mx-auto w-12 h-12 flex items-center justify-center rounded-full bg-muted">
                      {insuranceFile ? (
                        <FileCheck className="h-6 w-6 text-green-600" />
                      ) : (
                        <Upload className="h-6 w-6 text-muted-foreground" />
                      )}
                    </div>
                    <div>
                      <p className="text-sm font-medium">
                        {insuranceFile ? insuranceFile.name : "Upload your insurance card or policy"}
                      </p>
                      <p className="text-xs text-muted-foreground mt-1">
                        {insuranceFile 
                          ? "Insurance details will be automatically parsed" 
                          : "PDF, PNG, JPG up to 10MB - We'll extract your policy details automatically"
                        }
                      </p>
                    </div>
                    <div>
                      <Input
                        id="insuranceUpload"
                        type="file"
                        accept=".pdf,.png,.jpg,.jpeg"
                        onChange={handleFileUpload}
                        className="hidden"
                      />
                      <Button 
                        type="button" 
                        variant="outline" 
                        onClick={() => document.getElementById('insuranceUpload')?.click()}
                      >
                        {insuranceFile ? "Change File" : "Choose File"}
                      </Button>
                    </div>
                    {!insuranceFile && (
                      <div className="flex items-center justify-center gap-1 text-xs text-amber-600">
                        <AlertCircle className="h-3 w-3" />
                        <span>Required for personalized insurance analysis</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Emergency Contact */}
              <div className="space-y-2">
                <Label htmlFor="emergencyContact">Emergency Contact</Label>
                <Input
                  id="emergencyContact"
                  type="text"
                  placeholder="Emergency contact name and phone number"
                  value={formData.emergencyContact}
                  onChange={(e) => handleInputChange("emergencyContact", e.target.value)}
                />
              </div>

              {/* Submit Button */}
              <Button 
                type="submit" 
                className="w-full"
                disabled={!formData.name || !formData.medicalHistory || !formData.insuranceUploaded}
              >
                Complete Setup & Continue to Dashboard
              </Button>
            </form>

            {/* Footer */}
            <div className="mt-6 text-center">
              <p className="text-sm text-muted-foreground">
                Already have an account?{" "}
                <Link to="/dashboard" className="text-primary hover:underline">
                  Go to Dashboard
                </Link>
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Privacy Note */}
        <div className="mt-6 text-center">
          <p className="text-xs text-muted-foreground">
            Your information is encrypted and secure. We comply with HIPAA regulations.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;
