import { useState, useEffect } from "react";
import * as React from "react";
import { useNavigate } from "react-router-dom";
import { useApp } from "@/contexts/AppContext";
import { useToast } from "@/hooks/use-toast";
import { apiService, PolicyAnalysisResponse } from "@/services/api";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Shield, FileText, Upload, Download, Edit, Mail, DollarSign, Calendar, CheckCircle, XCircle, Clock, Search, ArrowRight, ChevronDown, ChevronRight, QrCode, Share, Copy, Smartphone, Shield as ShieldIcon, User, MapPin, Phone, AlertCircle, MessageSquare } from "lucide-react";
import PolicySummary from "@/components/PolicySummary";
import BillAnalysisLoader from "@/components/BillAnalysisLoader";

const Dashboard = () => {
  const { userProfile, insuranceData, uploadInsurance } = useApp();
  const { toast } = useToast();
  const navigate = useNavigate();
  const [expandedBills, setExpandedBills] = useState<Set<number>>(new Set());
  const [bills, setBills] = useState<any[]>([]);
  const [loadingBills, setLoadingBills] = useState(false);
  const [uploadingBill, setUploadingBill] = useState(false);
  const [billAnalysisStep, setBillAnalysisStep] = useState<'uploading' | 'extracting' | 'analyzing' | 'generating' | 'complete'>('uploading');
  const [currentBillFileName, setCurrentBillFileName] = useState<string>('');
  const [activeTab, setActiveTab] = useState("insurance");
  const [insuranceFile, setInsuranceFile] = useState<File | null>(null);

  // Professional Policy Display Flag - now always true since backend returns structured data
  const USE_PROFESSIONAL_POLICY_DISPLAY = true;

  // Type guard to check if insurance data is in new format
  const isNewFormatData = (data: any): data is PolicyAnalysisResponse => {
    return data && 
           typeof data === 'object' && 
           data.policyDetails && 
           data.coverageCosts && 
           data.commonServices && 
           data.prescriptions && 
           data.importantNotes;
  };

  // Check for existing insurance data on component mount
  React.useEffect(() => {
    // Allow access regardless of insurance upload status
  }, [userProfile, navigate]);

  const toggleBill = (index: number) => {
    const newExpanded = new Set(expandedBills);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedBills(newExpanded);
  };

  // Load bills when tab changes to bills
  useEffect(() => {
    if (activeTab === "bills" && userProfile?.insuranceUploaded) {
      loadBills();
    }
  }, [activeTab, userProfile?.insuranceUploaded]);

  const loadBills = async () => {
    setLoadingBills(true);
    try {
      const response = await apiService.getBillHistory(20);
      setBills(response.analyses || []);
    } catch (error) {
      console.error("Failed to load bills:", error);
      toast({
        title: "Error",
        description: "Failed to load bill history",
        variant: "destructive",
      });
    } finally {
      setLoadingBills(false);
    }
  };

  const handleBillUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    const allowedTypes = ['application/pdf', 'image/png', 'image/jpeg', 'image/jpg'];
    if (!allowedTypes.includes(file.type)) {
      toast({
        title: "Invalid file type",
        description: "Please upload a PDF or image file (PNG, JPG, JPEG)",
        variant: "destructive",
      });
      return;
    }

    setUploadingBill(true);
    setCurrentBillFileName(file.name);
    setBillAnalysisStep('uploading');

    try {
      // Upload the bill
      setBillAnalysisStep('extracting');
      const uploadResponse = await apiService.uploadBill(file);
      
      // Start analysis
      setBillAnalysisStep('analyzing');
      await new Promise(resolve => setTimeout(resolve, 800)); // Brief pause for better UX
      
      setBillAnalysisStep('generating');
      const analysisResponse = await apiService.analyzeBill(uploadResponse.bill_id);
      
      setBillAnalysisStep('complete');
      
      toast({
        title: "Bill analysis complete! ✅",
        description: `Your bill has been analyzed. Total due: $${analysisResponse.financial_breakdown?.patient_responsibility || 'N/A'}`,
      });

      // Reload bills list
      await loadBills();
      
      // Reset the file input
      event.target.value = '';
      
    } catch (error) {
      console.error("Bill upload/analysis failed:", error);
      toast({
        title: "Upload failed",
        description: error instanceof Error ? error.message : "Failed to upload bill",
        variant: "destructive",
      });
    } finally {
      setUploadingBill(false);
      setCurrentBillFileName('');
      setBillAnalysisStep('uploading');
    }
  };

  const handleInsuranceUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      await uploadInsurance(file);
      toast({
        title: "Insurance uploaded successfully!",
        description: "Your insurance document has been processed.",
      });
    } catch (error) {
      toast({
        title: "Upload failed",
        description: error instanceof Error ? error.message : "Failed to upload insurance",
        variant: "destructive",
      });
    }
  };

  const handleTabChange = (value: string) => {
    setActiveTab(value);
  };

  // Bills List Component
  const BillsList = () => {
    if (loadingBills) {
      return (
        <div className="flex items-center justify-center py-12">
          <div className="flex items-center gap-3 text-slate-500">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-slate-400"></div>
            <span className="text-sm font-medium">Loading bills...</span>
          </div>
        </div>
      );
    }

    if (bills.length === 0) {
      return (
        <div className="flex flex-col items-center justify-center py-16 space-y-6">
          <div className="text-center space-y-4">
            <div className="w-20 h-20 mx-auto rounded-full bg-blue-100 flex items-center justify-center">
              <FileText className="h-10 w-10 text-blue-600" />
            </div>
            <h3 className="text-xl font-semibold">No Bills Yet</h3>
            <p className="text-muted-foreground max-w-md">
              Upload your first medical bill to get started with AI-powered analysis.
            </p>
          </div>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        {bills.map((bill, index) => (
          <Card key={bill.analysis_id} className="hover:shadow-md transition-shadow cursor-pointer">
            <CardHeader className="pb-3">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <CardTitle className="text-lg">{bill.bill_filename}</CardTitle>
                    <Badge variant="outline" className="text-emerald-700 border-emerald-300 bg-emerald-50">
                      {(bill.confidence_score * 100).toFixed(0)}% confidence
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-4">
                    Analyzed on {new Date(bill.analysis_date).toLocaleDateString()}
                  </p>
                  
                  {/* Financial Summary */}
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <p className="text-muted-foreground">Total Billed</p>
                      <p className="font-semibold text-slate-900">
                        ${bill.total_charges?.toFixed(2) || '0.00'}
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Insurance Paid</p>
                      <p className="font-semibold text-blue-600">
                        ${bill.insurance_payment?.toFixed(2) || '0.00'}
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">You Owe</p>
                      <p className={`font-semibold ${bill.patient_responsibility === 0 ? "text-green-600" : "text-red-600"}`}>
                        ${bill.patient_responsibility?.toFixed(2) || '0.00'}
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center gap-2 ml-4">
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => navigate(`/bill-summary/${bill.analysis_id}`)}
                  >
                    <ArrowRight className="h-4 w-4 mr-1" />
                    View Details
                  </Button>
                  <Button variant="ghost" size="sm">
                    <Download className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardHeader>
          </Card>
        ))}
      </div>
    );
  };

  return (
    <>
      <div className="max-w-7xl mx-auto p-6">
        <div className="mb-8">
          <div className="flex justify-between items-start">
            <div>
              <h1 className="text-3xl font-bold mb-2">Dashboard</h1>
              <p className="text-muted-foreground">Manage your insurance and medical bills</p>
            </div>
            <div className="flex items-center gap-4">
              <Badge variant="outline" className="text-green-600 border-green-300 bg-green-50">
                ✅ Insurance Uploaded
              </Badge>
              <Button 
                className="bg-blue-600 hover:bg-blue-700 text-white"
                onClick={() => navigate("/chat")}
              >
                <MessageSquare className="h-4 w-4 mr-2" />
                Start Chat with AI
              </Button>
            </div>
          </div>
        </div>

        <Tabs value={activeTab} onValueChange={handleTabChange} className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="insurance">Insurance</TabsTrigger>
            <TabsTrigger value="bills">Medical Bills</TabsTrigger>
            <TabsTrigger value="qr">Emergency QR</TabsTrigger>
          </TabsList>

          {/* Insurance Tab */}
          <TabsContent value="insurance" className="space-y-6">
            {!userProfile?.insuranceUploaded ? (
              // Insurance Upload Prompt
              <div className="flex flex-col items-center justify-center py-16 space-y-6">
                <div className="text-center space-y-4">
                  <div className="w-20 h-20 mx-auto rounded-full bg-blue-100 flex items-center justify-center">
                    <Shield className="h-10 w-10 text-blue-600" />
                  </div>
                  <h2 className="text-2xl font-bold">Upload Your Insurance</h2>
                  <p className="text-muted-foreground max-w-md">
                    To access your dashboard and all features, please upload your insurance documents first. 
                    This will enable AI-powered analysis of your medical bills.
                  </p>
                </div>
                <div className="flex flex-col gap-4">
                  <input
                    type="file"
                    id="insurance-upload"
                    accept=".pdf,.png,.jpg,.jpeg"
                    className="hidden"
                    onChange={handleInsuranceUpload}
                  />
                  <Button 
                    size="lg" 
                    className="bg-blue-600 hover:bg-blue-700"
                    onClick={() => document.getElementById('insurance-upload')?.click()}
                  >
                    <Upload className="h-5 w-5 mr-2" />
                    Upload Insurance Document
                  </Button>
                  <p className="text-xs text-muted-foreground text-center">
                    Supports PDF, PNG, JPG, JPEG files
                  </p>
                </div>
              </div>
            ) : (
              <>
                {/* Professional Policy Display */}
                {USE_PROFESSIONAL_POLICY_DISPLAY && insuranceData && isNewFormatData(insuranceData) ? (
                  <PolicySummary data={insuranceData} />
                ) : insuranceData ? (
                  // This handles old format data - show message to re-upload
                  <div className="text-center py-12">
                    <div className="p-3 bg-amber-100 rounded-full w-fit mx-auto mb-4">
                      <AlertCircle className="h-6 w-6 text-amber-600" />
                    </div>
                    <h3 className="text-lg font-semibold text-slate-900 mb-2">Insurance Data Format Updated</h3>
                    <p className="text-slate-600 mb-4">
                      Please re-upload your insurance document to see the enhanced policy summary.
                    </p>
                    <Button 
                      onClick={() => {
                        // Clear old data
                        localStorage.removeItem('insuranceData');
                        localStorage.removeItem('userProfile');
                        window.location.reload();
                      }}
                      className="bg-amber-600 hover:bg-amber-700"
                    >
                      Clear Data & Re-upload
                    </Button>
                  </div>
                ) : (
                  <>
                    <div className="mb-6">
                      <h2 className="text-2xl font-bold text-foreground mb-2">Insurance Analysis</h2>
                      <p className="text-muted-foreground">AI-powered analysis of your insurance coverage from uploaded document.</p>
                    </div>

                    {/* Real Insurance Data */}
                <Card className="mb-8 border-0 shadow-sm bg-gradient-to-br from-slate-50 to-blue-50/30">
                  <CardHeader className="pb-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-blue-600 rounded-lg">
                          <Shield className="h-5 w-5 text-white" />
                        </div>
                        <div>
                          <CardTitle className="text-xl font-semibold text-slate-800">Policy Analysis</CardTitle>
                          <CardDescription className="text-sm text-slate-600">
                            From "{userProfile?.insuranceFile || 'your document'}"
                          </CardDescription>
                        </div>
                      </div>
                      {insuranceData?.confidence_score && (
                        <Badge variant="outline" className="text-emerald-700 border-emerald-300 bg-emerald-50 font-medium">
                          {(insuranceData.confidence_score * 100).toFixed(0)}% confidence
                        </Badge>
                      )}
                    </div>
                  </CardHeader>
                  <CardContent className="pt-0">
                    {insuranceData ? (
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {/* Deductible */}
                        <div className="group bg-white rounded-xl p-5 border border-slate-200/60 hover:border-blue-300 hover:shadow-md transition-all duration-200">
                          <div className="flex items-center gap-3 mb-2">
                            <div className="p-1.5 bg-blue-100 rounded-lg group-hover:bg-blue-200 transition-colors">
                              <DollarSign className="h-4 w-4 text-blue-600" />
                            </div>
                            <h3 className="text-sm font-medium text-slate-700">Annual Deductible</h3>
                          </div>
                          <p className="text-2xl font-bold text-slate-900 tracking-tight">
                            ${insuranceData.coverageCosts.inNetwork.deductible.individual}
                          </p>
                          <p className="text-xs text-slate-500 mt-1">Amount paid before coverage</p>
                        </div>

                        {/* Out-of-Pocket Max */}
                        <div className="group bg-white rounded-xl p-5 border border-slate-200/60 hover:border-emerald-300 hover:shadow-md transition-all duration-200">
                          <div className="flex items-center gap-3 mb-2">
                            <div className="p-1.5 bg-emerald-100 rounded-lg group-hover:bg-emerald-200 transition-colors">
                              <Shield className="h-4 w-4 text-emerald-600" />
                            </div>
                            <h3 className="text-sm font-medium text-slate-700">Out-of-Pocket Max</h3>
                          </div>
                          <p className="text-2xl font-bold text-slate-900 tracking-tight">
                            ${insuranceData.coverageCosts.inNetwork.outOfPocketMax.individual}
                          </p>
                          <p className="text-xs text-slate-500 mt-1">Maximum annual cost</p>
                        </div>

                        {/* Copay */}
                        <div className="group bg-white rounded-xl p-5 border border-slate-200/60 hover:border-purple-300 hover:shadow-md transition-all duration-200">
                          <div className="flex items-center gap-3 mb-2">
                            <div className="p-1.5 bg-purple-100 rounded-lg group-hover:bg-purple-200 transition-colors">
                              <Calendar className="h-4 w-4 text-purple-600" />
                            </div>
                            <h3 className="text-sm font-medium text-slate-700">Copay</h3>
                          </div>
                          <p className="text-2xl font-bold text-slate-900 tracking-tight">
                            {insuranceData.commonServices.length > 0 ? insuranceData.commonServices[0].cost : 'N/A'}
                          </p>
                          <p className="text-xs text-slate-500 mt-1">Fixed fee per visit</p>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-12">
                        <div className="inline-flex items-center gap-3 text-slate-500">
                          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-slate-400"></div>
                          <span className="text-sm font-medium">Analyzing insurance document...</span>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
                  </>
                )}

                {/* Call to Action for Chat */}
                <Card className="mb-8 bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
                  <CardContent className="p-6 text-center">
                    <h3 className="text-xl font-bold mb-2">Ready to Ask Questions?</h3>
                    <p className="text-muted-foreground mb-4">
                      Chat with our AI to understand your coverage, analyze bills, and get personalized insurance advice.
                    </p>
                    <Button 
                      size="lg" 
                      className="bg-blue-600 hover:bg-blue-700"
                      onClick={() => navigate("/chat")}
                    >
                      <MessageSquare className="h-5 w-5 mr-2" />
                      Start Chat Session
                    </Button>
                  </CardContent>
                </Card>
              </>
            )}
          </TabsContent>

          {/* Medical Bills Tab */}
          <TabsContent value="bills" className="space-y-6">
            {!userProfile?.insuranceUploaded ? (
              // Show disabled state if no insurance uploaded
              <div className="flex flex-col items-center justify-center py-16 space-y-6">
                <div className="text-center space-y-4">
                  <div className="w-20 h-20 mx-auto rounded-full bg-gray-100 flex items-center justify-center">
                    <FileText className="h-10 w-10 text-gray-400" />
                  </div>
                  <h2 className="text-2xl font-bold text-gray-400">Medical Bills</h2>
                  <p className="text-muted-foreground max-w-md">
                    Upload your insurance first to start analyzing medical bills.
                  </p>
                  <Button disabled variant="outline">
                    <Upload className="h-4 w-4 mr-2" />
                    Upload Insurance First
                  </Button>
                </div>
              </div>
            ) : (
              <>
                {/* Header with Upload Button */}
                <div className="flex justify-between items-center">
                  <div>
                    <h2 className="text-2xl font-bold">Medical Bills</h2>
                    <p className="text-muted-foreground">Upload and analyze your medical bills against your insurance policy</p>
                  </div>
                  <div className="flex gap-2">
                    <input
                      type="file"
                      id="bill-upload"
                      accept=".pdf,.png,.jpg,.jpeg"
                      className="hidden"
                      onChange={handleBillUpload}
                    />
                    <Button onClick={() => document.getElementById('bill-upload')?.click()}>
                      <Upload className="h-4 w-4 mr-2" />
                      Upload New Bill
                    </Button>
                  </div>
                </div>

                {/* Loading Overlay during Bill Analysis */}
                {uploadingBill && (
                  <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <BillAnalysisLoader 
                      currentStep={billAnalysisStep} 
                      fileName={currentBillFileName}
                    />
                  </div>
                )}

                {/* Bills List */}
                <BillsList />
              </>
            )}
          </TabsContent>

          {/* QR Code Tab */}
          <TabsContent value="qr" className="space-y-6">
            <div className="text-center">
              <div className="mb-6">
                <h2 className="text-2xl font-bold mb-2">Emergency Medical QR Code</h2>
                <p className="text-muted-foreground">Share your essential medical information instantly in emergencies</p>
              </div>

              {/* QR Code Section */}
              <div className="flex flex-col items-center space-y-6">
                <Card className="p-8 max-w-md mx-auto">
                  <div className="space-y-4 text-center">
                    <div className="w-48 h-48 mx-auto bg-white border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center">
                      <div className="text-center space-y-2">
                        <QrCode className="h-12 w-12 text-gray-400 mx-auto" />
                        <p className="text-sm text-gray-500">Your Emergency QR Code</p>
                        <p className="text-xs text-gray-400">Scan to access medical info</p>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <p className="text-sm font-medium">John Doe</p>
                      <p className="text-xs text-muted-foreground">ID: XXX-XX-1234</p>
                    </div>

                    <div className="flex gap-2 text-xs">
                      <Button variant="outline" size="sm" className="flex-1">
                        <Download className="h-3 w-3 mr-1" />
                        Download
                      </Button>
                      <Button variant="outline" size="sm" className="flex-1">
                        <Share className="h-3 w-3 mr-1" />
                        Share
                      </Button>
                    </div>
                  </div>
                </Card>

                {/* Information Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-4xl mt-8">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2 text-lg">
                        <ShieldIcon className="h-5 w-5 text-blue-600" />
                        What's Included
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="flex items-center gap-2 text-sm">
                        <div className="w-2 h-2 rounded-full bg-green-500"></div>
                        <span>Insurance policy details</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <div className="w-2 h-2 rounded-full bg-green-500"></div>
                        <span>Emergency contact information</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <div className="w-2 h-2 rounded-full bg-green-500"></div>
                        <span>Medical alert conditions</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <div className="w-2 h-2 rounded-full bg-green-500"></div>
                        <span>Current medications</span>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2 text-lg">
                        <Smartphone className="h-5 w-5 text-green-600" />
                        How to Use
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="flex items-start gap-3 text-sm">
                        <div className="w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                          <span className="text-xs font-semibold text-blue-600">1</span>
                        </div>
                        <span>Save QR code to your phone's photo gallery</span>
                      </div>
                      <div className="flex items-start gap-3 text-sm">
                        <div className="w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                          <span className="text-xs font-semibold text-blue-600">2</span>
                        </div>
                        <span>Print a copy for your wallet or car</span>
                      </div>
                      <div className="flex items-start gap-3 text-sm">
                        <div className="w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                          <span className="text-xs font-semibold text-blue-600">3</span>
                        </div>
                        <span>Updates automatically when you change info</span>
                      </div>
                      <div className="flex items-start gap-3 text-sm">
                        <div className="w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                          <span className="text-xs font-semibold text-blue-600">4</span>
                        </div>
                        <span>Works offline - no internet required</span>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Privacy Notice */}
                <Card className="w-full max-w-4xl bg-yellow-50 border-yellow-200">
                  <CardContent className="p-4">
                    <div className="flex items-start gap-3">
                      <AlertCircle className="h-5 w-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                      <div className="text-sm">
                        <p className="font-medium text-yellow-800 mb-1">Privacy & Security</p>
                        <p className="text-yellow-700">
                          Your QR code contains only essential emergency information. Sensitive data like 
                          full medical records and billing information are not included. The QR code 
                          expires after 6 months and can be regenerated at any time.
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </>
  );
};

export default Dashboard;
