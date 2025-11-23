import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { apiService, BillAnalysisResponse } from "@/services/api";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import DisputeEmailModal from "@/components/DisputeEmailModal";
import BillSummaryCard from "@/components/BillSummaryCard";

const BillSummary = () => {
  const { analysisId } = useParams<{ analysisId: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [analysis, setAnalysis] = useState<BillAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [disputeModalOpen, setDisputeModalOpen] = useState(false);

  useEffect(() => {
    if (analysisId) {
      loadBillAnalysis(analysisId);
    }
  }, [analysisId]);

  const loadBillAnalysis = async (id: string) => {
    try {
      setLoading(true);
      const response = await apiService.getBillAnalysis(id);
      setAnalysis(response);
    } catch (error) {
      console.error("Failed to load bill analysis:", error);
      toast({
        title: "Error",
        description: "Failed to load bill analysis",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto p-6">
        <div className="flex items-center justify-center py-16">
          <div className="flex items-center gap-3 text-slate-500">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-slate-400"></div>
            <span className="text-lg font-medium">Loading bill analysis...</span>
          </div>
        </div>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="max-w-6xl mx-auto p-6">
        <div className="text-center py-16">
          <h2 className="text-2xl font-bold mb-4">Analysis Not Found</h2>
          <Button onClick={() => navigate("/dashboard")}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Dashboard
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      {/* Header */}
      <div className="mb-6">
        <Button variant="ghost" onClick={() => navigate("/dashboard")} className="mb-4">
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Dashboard
        </Button>
      </div>

      {/* Bill Analysis Content */}
      <BillSummaryCard 
        data={analysis} 
        onFileDispute={() => setDisputeModalOpen(true)}
      />

      {/* Dispute Email Modal */}
      {analysis && (
        <DisputeEmailModal
          isOpen={disputeModalOpen}
          onClose={() => setDisputeModalOpen(false)}
          analysisId={analysisId || ""}
          billData={{
            bill_filename: analysis.bill_filename || "Medical Bill",
            bill_summary: analysis.bill_summary,
            financial_breakdown: analysis.financial_breakdown
          }}
        />
      )}
    </div>
  );
};

export default BillSummary;