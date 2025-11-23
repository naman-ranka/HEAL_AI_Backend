import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Upload, FileText, ChevronRight } from "lucide-react";
import { useState } from "react";

const UploadSection = () => {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    // Handle file drop logic here
    console.log("Files dropped:", e.dataTransfer.files);
  };

  return (
    <section className="py-16 bg-background">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              Get Started in Seconds
            </h2>
            <p className="text-xl text-muted-foreground">
              Upload your medical bill and get instant clarity on your expenses
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-8 items-center">
            {/* Upload Area */}
            <Card
              className={`p-8 border-2 border-dashed transition-all duration-300 cursor-pointer ${
                isDragOver 
                  ? "border-primary bg-primary/5 shadow-glow" 
                  : "border-border hover:border-primary/50"
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <div className="text-center space-y-4">
                <div className={`mx-auto w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center transition-colors ${
                  isDragOver ? "bg-primary/20" : ""
                }`}>
                  <Upload className="h-8 w-8 text-primary" />
                </div>
                
                <div>
                  <h3 className="text-xl font-semibold text-foreground mb-2">
                    Upload Your Medical Bill
                  </h3>
                  <p className="text-muted-foreground mb-4">
                    Drag and drop your PDF, image, or scan here
                  </p>
                  <Button variant="outline" className="w-full">
                    <FileText className="h-4 w-4" />
                    Choose File
                  </Button>
                </div>

                <div className="text-xs text-muted-foreground">
                  Supports PDF, JPG, PNG • Max 10MB • HIPAA Secure
                </div>
              </div>
            </Card>

            {/* Process Steps */}
            <div className="space-y-6">
              <div className="flex items-start gap-4">
                <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-semibold">
                  1
                </div>
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Upload Your Bill</h4>
                  <p className="text-muted-foreground text-sm">
                    Securely upload your medical bill in any format
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-4">
                <div className="w-8 h-8 rounded-full bg-success text-success-foreground flex items-center justify-center text-sm font-semibold">
                  2
                </div>
                <div>
                  <h4 className="font-semibold text-foreground mb-1">AI Analysis</h4>
                  <p className="text-muted-foreground text-sm">
                    Our AI breaks down charges and detects potential errors
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-4">
                <div className="w-8 h-8 rounded-full bg-accent-foreground text-white flex items-center justify-center text-sm font-semibold">
                  3
                </div>
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Get Clear Results</h4>
                  <p className="text-muted-foreground text-sm">
                    Receive a simplified breakdown with actionable insights
                  </p>
                </div>
              </div>

              <Button variant="hero" className="w-full mt-6">
                Start Analysis
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default UploadSection;