import React, { useState } from "react";
import { useToast } from "@/hooks/use-toast";
import { apiService } from "@/services/api";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Separator } from "@/components/ui/separator";
import { Copy, Mail, FileText, AlertTriangle, CheckCircle } from "lucide-react";

interface DisputeEmailModalProps {
  isOpen: boolean;
  onClose: () => void;
  analysisId: string;
  billData: {
    bill_filename: string;
    bill_summary?: {
      provider_name?: string;
      patient_name?: string;
      date_of_service?: string;
    };
    financial_breakdown?: {
      patient_responsibility?: number;
    };
  };
}

const DisputeEmailModal: React.FC<DisputeEmailModalProps> = ({
  isOpen,
  onClose,
  analysisId,
  billData
}) => {
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [emailGenerated, setEmailGenerated] = useState(false);
  const [emailContent, setEmailContent] = useState("");
  const [formData, setFormData] = useState({
    patient_name: billData.bill_summary?.patient_name || "",
    provider_name: billData.bill_summary?.provider_name || "",
    service_date: billData.bill_summary?.date_of_service || "",
    dispute_reason: "I have identified discrepancies in the billing calculation that require immediate review and correction.",
    disputed_amount: billData.financial_breakdown?.patient_responsibility || 0
  });

  const handleInputChange = (field: string, value: string | number) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const generateEmail = async () => {
    if (!formData.patient_name || !formData.provider_name || !formData.service_date) {
      toast({
        title: "Missing Information",
        description: "Please fill in all required fields",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    try {
      const response = await apiService.generateDisputeEmail(analysisId, formData);
      setEmailContent(response.email_content);
      setEmailGenerated(true);
      
      toast({
        title: "Dispute Email Generated! ✉️",
        description: "Your professional dispute email is ready to copy.",
      });
    } catch (error) {
      console.error("Failed to generate dispute email:", error);
      toast({
        title: "Generation Failed",
        description: error instanceof Error ? error.message : "Failed to generate dispute email",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(emailContent);
      toast({
        title: "Email Copied! 📋",
        description: "The dispute email has been copied to your clipboard.",
      });
    } catch (error) {
      toast({
        title: "Copy Failed",
        description: "Unable to copy to clipboard. Please select and copy manually.",
        variant: "destructive",
      });
    }
  };

  const handleClose = () => {
    setEmailGenerated(false);
    setEmailContent("");
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-orange-600" />
            File Medical Bill Dispute
          </DialogTitle>
          <DialogDescription>
            Generate a professional dispute email with medical billing terminology and legal references.
          </DialogDescription>
        </DialogHeader>

        {!emailGenerated ? (
          // Form Step
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="patient_name">Patient Name *</Label>
                <Input
                  id="patient_name"
                  value={formData.patient_name}
                  onChange={(e) => handleInputChange("patient_name", e.target.value)}
                  placeholder="Enter patient full name"
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="provider_name">Provider/Facility Name *</Label>
                <Input
                  id="provider_name"
                  value={formData.provider_name}
                  onChange={(e) => handleInputChange("provider_name", e.target.value)}
                  placeholder="Enter healthcare provider name"
                  required
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="service_date">Service Date *</Label>
                <Input
                  id="service_date"
                  type="date"
                  value={formData.service_date}
                  onChange={(e) => handleInputChange("service_date", e.target.value)}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="disputed_amount">Disputed Amount ($)</Label>
                <Input
                  id="disputed_amount"
                  type="number"
                  step="0.01"
                  value={formData.disputed_amount}
                  onChange={(e) => handleInputChange("disputed_amount", parseFloat(e.target.value) || 0)}
                  placeholder="0.00"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="dispute_reason">Dispute Reason</Label>
              <Textarea
                id="dispute_reason"
                value={formData.dispute_reason}
                onChange={(e) => handleInputChange("dispute_reason", e.target.value)}
                rows={3}
                placeholder="Describe the specific issues with the billing..."
                className="resize-none"
              />
            </div>

            <div className="bg-blue-50 p-4 rounded-lg">
              <h4 className="font-semibold text-blue-900 mb-2 flex items-center gap-2">
                <FileText className="h-4 w-4" />
                What will be included:
              </h4>
              <ul className="text-sm text-blue-800 space-y-1">
                <li>• Professional medical billing terminology and legal references</li>
                <li>• Citation of FDCPA, No Surprises Act, and patient rights</li>
                <li>• Detailed service-by-service breakdown from analysis</li>
                <li>• Specific requests for documentation and corrections</li>
                <li>• Reference to analysis ID for tracking</li>
              </ul>
            </div>

            <DialogFooter>
              <Button variant="outline" onClick={handleClose}>
                Cancel
              </Button>
              <Button onClick={generateEmail} disabled={loading}>
                {loading ? "Generating..." : "Generate Dispute Email"}
                <Mail className="h-4 w-4 ml-2" />
              </Button>
            </DialogFooter>
          </div>
        ) : (
          // Email Display Step
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-green-50 rounded-lg">
              <div className="flex items-center gap-2 text-green-800">
                <CheckCircle className="h-5 w-5" />
                <span className="font-medium">Professional Dispute Email Generated</span>
              </div>
              <Button onClick={copyToClipboard} size="sm">
                <Copy className="h-4 w-4 mr-2" />
                Copy Email
              </Button>
            </div>

            <Separator />

            <div className="space-y-2">
              <Label>Generated Email Content:</Label>
              <Textarea
                value={emailContent}
                readOnly
                rows={20}
                className="font-mono text-sm resize-none"
                style={{ whiteSpace: 'pre-wrap' }}
              />
            </div>

            <div className="bg-amber-50 p-4 rounded-lg">
              <h4 className="font-semibold text-amber-900 mb-2">📧 Next Steps:</h4>
              <ol className="text-sm text-amber-800 space-y-1 list-decimal list-inside">
                <li>Copy the email content using the button above</li>
                <li>Paste into your preferred email client</li>
                <li>Send to the billing department of your healthcare provider</li>
                <li>Keep a copy for your records</li>
                <li>Await response within 30 days as per FDCPA requirements</li>
              </ol>
            </div>

            <DialogFooter>
              <Button variant="outline" onClick={handleClose}>
                Close
              </Button>
              <Button onClick={copyToClipboard}>
                <Copy className="h-4 w-4 mr-2" />
                Copy & Close
              </Button>
            </DialogFooter>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default DisputeEmailModal;


