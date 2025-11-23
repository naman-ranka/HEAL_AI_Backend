import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Send, Upload, Shield, Loader2 } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useApp } from "@/contexts/AppContext";
import { useToast } from "@/hooks/use-toast";
import { apiService } from "@/services/api";
import MarkdownMessage from "@/components/MarkdownMessage";

interface Message {
  id: string;
  content: string;
  sender: "user" | "ai";
  timestamp: Date;
  sources?: Array<{
    document: string;
    chunk_id: number;
    preview: string;
    similarity: number;
  }>;
}

const suggestedPrompts = [
  "What's my deductible?",
  "What does my policy cover?",
  "How much is my copay?",
  "What's my out-of-pocket maximum?",
  "Do I need referrals?",
  "What's covered for prescriptions?",
  "Explain my in-network benefits",
  "What's not covered by my plan?"
];

const Chat = () => {
  const { userProfile, uploadInsurance, uploadBill, createChatSession, currentSessionId, setCurrentSessionId } = useApp();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      content: "Hello! I'm HEAL.AI, your healthcare expenses analyzer. I can help you understand your medical bills, check insurance coverage, and identify potential errors. How can I assist you today?",
      sender: "ai",
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [sessionLoading, setSessionLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const { toast } = useToast();

  // Allow access regardless of insurance upload status
  useEffect(() => {
    // Chat is now accessible without insurance requirement
  }, [userProfile, navigate]);

  // Initialize chat session when user has insurance
  useEffect(() => {
    if (userProfile?.insuranceUploaded && !currentSessionId) {
      const initSession = async () => {
        setSessionLoading(true);
        try {
          await createChatSession();
        } catch (error) {
          console.error('Failed to create chat session:', error);
          toast({
            title: "Session Error",
            description: "Failed to initialize chat session. Some features may not work properly.",
            variant: "destructive",
          });
        } finally {
          setSessionLoading(false);
        }
      };
      
      initSession();
    }
  }, [userProfile, currentSessionId, createChatSession, toast]);

  const handleInsuranceUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      await uploadInsurance(file);
      
      // Add AI message about successful upload
      const successMessage: Message = {
        id: Date.now().toString(),
        content: `Great! I've received your insurance document "${file.name}". Now I can provide personalized assistance with your medical bills and coverage questions. You can also access all dashboard features. What would you like me to help you with?`,
        sender: "ai",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, successMessage]);
      
      toast({
        title: "Insurance Uploaded",
        description: "Your insurance document has been analyzed successfully!",
      });
    } catch (error) {
      toast({
        title: "Upload Failed",
        description: error instanceof Error ? error.message : "Failed to upload insurance document.",
        variant: "destructive",
      });
    }
  };

  const handleBillUpload = async (file: File) => {
    try {
      const result = await uploadBill(file);
      
      // Add user message about upload
      const uploadMessage: Message = {
        id: Date.now().toString(),
        content: `I've uploaded my medical bill: ${file.name}`,
        sender: "user",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, uploadMessage]);
      
      // Add AI processing message
      const processingMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: `I've received your medical bill "${file.name}". Let me analyze it against your insurance policy...`,
        sender: "ai",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, processingMessage]);
      
      // Analyze the bill
      setIsTyping(true);
      try {
        const analysis = await apiService.analyzeBill(result.bill_id);
        
        // Create detailed analysis response
        const analysisContent = `
**Bill Analysis Complete! 📊**

**Provider:** ${analysis.bill_summary.provider_name}
**Date of Service:** ${analysis.bill_summary.date_of_service}
**Network Status:** ${analysis.coverage_analysis.network_status}

**Financial Breakdown:**
• Total Charges: $${analysis.financial_breakdown.total_charges.toFixed(2)}
• Insurance Payment: $${analysis.financial_breakdown.insurance_payment.toFixed(2)}
• Your Responsibility: $${analysis.financial_breakdown.patient_responsibility.toFixed(2)}
• Amount Saved: $${analysis.financial_breakdown.amount_saved.toFixed(2)}

**Coverage Summary:** ${analysis.coverage_analysis.summary}

${analysis.dispute_recommendations.length > 0 ? '**⚠️ Potential Issues Found:**\n' + analysis.dispute_recommendations.map(rec => `• ${rec.description} (${rec.priority} priority)`).join('\n') : '✅ No billing discrepancies detected.'}

Would you like me to explain any part of this analysis in more detail?
        `.trim();
        
        const analysisMessage: Message = {
          id: (Date.now() + 2).toString(),
          content: analysisContent,
          sender: "ai",
          timestamp: new Date()
        };
        setMessages(prev => [...prev, analysisMessage]);
        
        toast({
          title: "Bill Analysis Complete",
          description: `Found ${analysis.dispute_recommendations.length} potential issues to review.`,
        });
        
      } catch (analysisError) {
        const errorMessage: Message = {
          id: (Date.now() + 2).toString(),
          content: "I encountered an issue analyzing your bill. Please try again or contact support if the problem persists.",
          sender: "ai",
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
        
        toast({
          title: "Analysis Failed",
          description: "Failed to analyze the medical bill. Please try again.",
          variant: "destructive",
        });
      } finally {
        setIsTyping(false);
      }
      
    } catch (error) {
      toast({
        title: "Upload Failed",
        description: error instanceof Error ? error.message : "Failed to upload medical bill.",
        variant: "destructive",
      });
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || !currentSessionId) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      sender: "user",
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const messageContent = inputValue;
    setInputValue("");
    setIsTyping(true);

    try {
      // Send message to backend chat system
      const response = await apiService.sendChatMessage(currentSessionId, messageContent);
      
      // Format AI response - sources are optional and can be toggled
      let aiContent = response.message;
      
      // Only show sources if they have meaningful similarity scores (> 0.5)
      const meaningfulSources = response.sources?.filter(source => source.similarity > 0.5) || [];
      if (meaningfulSources.length > 0) {
        aiContent += "\n\n---\n*Sources: " + meaningfulSources.map(source => 
          `${source.document || 'Policy Document'}`
        ).join(', ') + "*";
      }
      
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: aiContent,
        sender: "ai",
        timestamp: new Date(),
        sources: response.sources
      };
      
      setMessages(prev => [...prev, aiResponse]);
      
    } catch (error) {
      console.error('Failed to send message:', error);
      const errorResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: "I'm sorry, I'm having trouble connecting to my knowledge base right now. Please try again in a moment or upload a medical bill for analysis.",
        sender: "ai",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorResponse]);
      
      toast({
        title: "Message Failed",
        description: "Failed to send message. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsTyping(false);
    }
  };

  const handlePromptClick = (prompt: string) => {
    setInputValue(prompt);
  };

  const handleFileUpload = (type: "file" | "camera") => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.pdf,.png,.jpg,.jpeg';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        handleBillUpload(file);
      }
    };
    input.click();
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-50 to-blue-50/30">
      {/* Chat Container */}
      <div className="flex-1 flex flex-col max-w-5xl mx-auto w-full px-4 py-4">
        {/* Messages Container with Fixed Height and Scroll */}
        <div className="flex-1 overflow-y-auto space-y-4 pr-2 min-h-0">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[80%] rounded-lg px-4 py-2 ${
                  message.sender === "user"
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted"
                }`}
              >
                {message.sender === "ai" ? (
                  <MarkdownMessage content={message.content} />
                ) : (
                  <p className="text-sm">{message.content}</p>
                )}
                <span className="text-xs opacity-70">
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Suggested Prompts and Session Status */}
        {messages.length === 1 && (
          <div className="py-4 flex-shrink-0">
            {sessionLoading ? (
              <div className="flex items-center justify-center gap-2 text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Initializing chat session...</span>
              </div>
            ) : !userProfile?.insuranceUploaded ? (
              <div className="mb-4">
                <p className="text-sm text-muted-foreground mb-3">First, let's get your insurance information:</p>
                <label htmlFor="chat-insurance-upload" className="cursor-pointer">
                  <Card className="p-4 border-2 border-dashed border-blue-300 hover:border-blue-400 transition-colors bg-blue-50/50">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center">
                        <Upload className="h-5 w-5 text-blue-600" />
                      </div>
                      <div className="flex-1">
                        <p className="font-medium text-blue-600">Upload Insurance Documents</p>
                        <p className="text-xs text-muted-foreground">PDF, PNG, JPG up to 10MB</p>
                      </div>
                      <Shield className="h-5 w-5 text-blue-600" />
                    </div>
                  </Card>
                  <input
                    id="chat-insurance-upload"
                    type="file"
                    accept=".pdf,.png,.jpg,.jpeg"
                    onChange={handleInsuranceUpload}
                    className="hidden"
                  />
                </label>
              </div>
            ) : (
              <>
                <p className="text-sm text-muted-foreground mb-3">Popular insurance questions:</p>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                {suggestedPrompts.map((prompt, index) => (
                  <Card
                    key={index}
                    className="p-3 cursor-pointer hover:bg-blue-50 hover:border-blue-200 transition-all duration-200 border-slate-200"
                    onClick={() => handlePromptClick(prompt)}
                  >
                    <p className="text-sm font-medium text-slate-700">{prompt}</p>
                  </Card>
                ))}
                </div>
              </>
            )}
          </div>
        )}

        {/* Typing indicator */}
        {isTyping && (
          <div className="flex justify-start mb-4">
            <div className="bg-muted rounded-lg px-4 py-2 flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="text-sm text-muted-foreground">HEAL.AI is typing...</span>
            </div>
          </div>
        )}

        {/* Input Area - Fixed at Bottom like ChatGPT */}
        <div className="py-4 border-t flex-shrink-0 bg-background">
          <div className="relative max-w-3xl mx-auto">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={currentSessionId ? "Ask about your insurance policy..." : "Upload insurance first to start chatting"}
              onKeyPress={(e) => e.key === "Enter" && !e.shiftKey && handleSendMessage()}
              disabled={isTyping || !currentSessionId}
              className="w-full pr-12 py-3 rounded-xl border-2 focus:border-primary/50 shadow-sm"
            />
            <Button 
              onClick={handleSendMessage} 
              disabled={!inputValue.trim() || isTyping || !currentSessionId}
              className="absolute right-2 top-1/2 transform -translate-y-1/2 p-2 rounded-lg"
              size="sm"
            >
              {isTyping ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
          
          {/* Help text */}
          <div className="text-center mt-2">
            <p className="text-xs text-muted-foreground">
              Ask questions about your insurance policy • Press Enter to send
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chat;