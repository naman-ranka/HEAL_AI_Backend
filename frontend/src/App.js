import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [results, setResults] = useState(null);
  const [currentState, setCurrentState] = useState('initial'); // 'initial', 'loading', 'results', 'chat', 'bill-upload', 'bill-loading', 'bill-results'
  
  // Chat state
  const [chatSession, setChatSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [documentId, setDocumentId] = useState(null);
  const messagesEndRef = useRef(null);

  // Bill checker state
  const [billId, setBillId] = useState(null);
  const [billAnalysis, setBillAnalysis] = useState(null);
  const [billHistory, setBillHistory] = useState([]);
  const [showBillHistory, setShowBillHistory] = useState(false);

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load bill history on component mount
  useEffect(() => {
    loadBillHistory();
  }, []);

  // Create chat session
  const createChatSession = async (docId = null) => {
    try {
      const response = await fetch('http://localhost:8000/chat/sessions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          document_ids: docId ? [docId] : null
        })
      });

      if (!response.ok) {
        throw new Error('Failed to create chat session');
      }

      const data = await response.json();
      setChatSession(data.session_id);
      return data.session_id;
    } catch (error) {
      console.error('Error creating chat session:', error);
      return null;
    }
  };

  // Send chat message
  const sendMessage = async () => {
    if (!chatInput.trim()) return;
    
    const userMessage = chatInput.trim();
    setChatInput('');
    setChatLoading(true);

    // Add user message to chat
    setMessages(prev => [...prev, {
      type: 'user',
      content: userMessage,
      timestamp: new Date()
    }]);

    try {
      // Create session if not exists
      let sessionId = chatSession;
      if (!sessionId) {
        sessionId = await createChatSession(documentId);
        if (!sessionId) {
          throw new Error('Failed to create chat session');
        }
      }

      const response = await fetch(`http://localhost:8000/chat/sessions/${sessionId}/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage
        })
      });

      if (!response.ok) {
        throw new Error('Failed to send message');
      }

      const data = await response.json();
      
      // Add assistant response to chat
      setMessages(prev => [...prev, {
        type: 'assistant',
        content: data.message,
        sources: data.sources,
        confidence: data.confidence,
        timestamp: new Date()
      }]);
      
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, {
        type: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date()
      }]);
    } finally {
      setChatLoading(false);
    }
  };

  // Handle key press in chat input
  const handleChatKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Start chat from results
  const startChat = () => {
    setCurrentState('chat');
    // Add welcome message
    setMessages([{
      type: 'assistant',
      content: "Hi! I'm HEAL, your insurance policy assistant. I've analyzed your document and I'm ready to answer questions about your policy. What would you like to know?",
      timestamp: new Date()
    }]);
  };

  const handleUpload = async () => {
    // Create file input element
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*,application/pdf';
    
    input.onchange = async (event) => {
      const file = event.target.files[0];
      if (!file) return;

      setCurrentState('loading');
      setResults(null);

      try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('http://localhost:8000/upload', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('Upload failed');
        }

        const data = await response.json();
        setResults(data);
        // Store document ID for RAG and bill analysis
        if (data.document_id) {
          setDocumentId(data.document_id);
        } else if (data.additional_info && data.additional_info.rag_document_id) {
          setDocumentId(data.additional_info.rag_document_id);
        }
        setCurrentState('results');
      } catch (error) {
        console.error('Error uploading file:', error);
        setResults({
          error: 'Failed to process document. Please try again.'
        });
        setCurrentState('results');
      } finally {
        // Loading state handled by state transitions
      }
    };

    // Trigger file dialog
    input.click();
  };

  // Bill checker functions
  const handleBillUpload = async () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*,application/pdf';
    
    input.onchange = async (event) => {
      const file = event.target.files[0];
      if (!file) return;

      setCurrentState('bill-loading');
      setBillAnalysis(null);

      try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('http://localhost:8000/bill-checker/upload', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('Bill upload failed');
        }

        const data = await response.json();
        setBillId(data.bill_id);
        
        // Automatically analyze the bill
        await analyzeBill(data.bill_id);
        
      } catch (error) {
        console.error('Error uploading bill:', error);
        setBillAnalysis({
          error: 'Failed to process medical bill. Please try again.'
        });
        setCurrentState('bill-results');
      }
    };

    input.click();
  };

  const analyzeBill = async (billIdToAnalyze) => {
    try {
      const response = await fetch('http://localhost:8000/bill-checker/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          bill_id: billIdToAnalyze,
          policy_id: documentId, // Use the uploaded policy document ID
          include_dispute_recommendations: true
        }),
      });

      if (!response.ok) {
        throw new Error('Bill analysis failed');
      }

      const analysisData = await response.json();
      setBillAnalysis(analysisData);
      setCurrentState('bill-results');
      
      // Refresh bill history
      loadBillHistory();
      
    } catch (error) {
      console.error('Error analyzing bill:', error);
      setBillAnalysis({
        error: 'Failed to analyze medical bill. Please try again.'
      });
      setCurrentState('bill-results');
    }
  };

  const loadBillHistory = async () => {
    try {
      const response = await fetch('http://localhost:8000/bill-checker/history?limit=10');
      if (response.ok) {
        const data = await response.json();
        setBillHistory(data.analyses || []);
      }
    } catch (error) {
      console.error('Error loading bill history:', error);
    }
  };

  const resetToInitial = () => {
    setCurrentState('initial');
    setResults(null);
    // Reset chat state
    setChatSession(null);
    setMessages([]);
    setChatInput('');
    setChatLoading(false);
    setDocumentId(null);
    // Reset bill state
    setBillId(null);
    setBillAnalysis(null);
    setShowBillHistory(false);
  };

  return (
    <div className="app">
      <div className="container">
        {/* App Title */}
        <h1 className="app-title">HEAL</h1>
        
        {/* Initial State */}
        {currentState === 'initial' && (
          <div className="initial-state">
            <button 
              className="upload-button"
              onClick={handleUpload}
            >
              📄 Upload Insurance Policy
            </button>
            
            {billHistory.length > 0 && (
              <div className="bill-history-toggle">
                <button 
                  className="history-button" 
                  onClick={() => setShowBillHistory(!showBillHistory)}
                >
                  {showBillHistory ? 'Hide' : 'Show'} Previous Bill Analyses ({billHistory.length})
                </button>
                
                {showBillHistory && (
                  <div className="bill-history">
                    <h3>Recent Bill Analyses</h3>
                    {billHistory.slice(0, 5).map((bill, index) => (
                      <div key={bill.analysis_id} className="history-item">
                        <span className="bill-name">{bill.bill_filename}</span>
                        <span className="bill-date">{new Date(bill.analysis_date).toLocaleDateString()}</span>
                        {bill.patient_responsibility && (
                          <span className="patient-cost">${bill.patient_responsibility.toFixed(2)}</span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
        
        {/* Loading State */}
        {currentState === 'loading' && (
          <div className="loading-state">
            <div className="spinner"></div>
            <p className="loading-text">Analyzing your policy...</p>
          </div>
        )}

        {/* Bill Loading State */}
        {currentState === 'bill-loading' && (
          <div className="loading-state">
            <div className="spinner"></div>
            <p className="loading-text">Analyzing your medical bill...</p>
          </div>
        )}
        
        {/* Results State */}
        {currentState === 'results' && (
          <div className="results-state">
            {results && results.error ? (
              <div className="error-card">
                <p>{results.error}</p>
                <button className="retry-button" onClick={resetToInitial}>
                  Try Again
                </button>
              </div>
            ) : (
              <div className="summary-card">
                <h2 className="card-title">Policy Summary</h2>
                <div className="summary-grid">
                  <div className="summary-item">
                    <span className="label">Deductible</span>
                    <span className="value">{results?.deductible || 'Not found'}</span>
                  </div>
                  <div className="summary-item">
                    <span className="label">Out-of-Pocket Max</span>
                    <span className="value">{results?.out_of_pocket_max || 'Not found'}</span>
                  </div>
                  <div className="summary-item">
                    <span className="label">Co-pay</span>
                    <span className="value">{results?.copay || 'Not found'}</span>
                  </div>
                </div>
                <div className="action-buttons">
                  <button className="chat-button" onClick={startChat}>
                    💬 Ask Questions About Policy
                  </button>
                  <button className="bill-checker-button" onClick={handleBillUpload}>
                    🏥 Check Medical Bill Against This Policy
                  </button>
                  <button className="new-upload-button" onClick={resetToInitial}>
                    Upload Another Document
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Bill Results State */}
        {currentState === 'bill-results' && (
          <div className="results-state">
            {billAnalysis && billAnalysis.error ? (
              <div className="error-card">
                <p>{billAnalysis.error}</p>
                <button className="retry-button" onClick={resetToInitial}>
                  Try Again
                </button>
              </div>
            ) : (
              <div className="summary-card bill-analysis-card">
                <h2 className="card-title">🏥 Medical Bill Analysis</h2>
                {documentId && (
                  <div className="analysis-context">
                    <p>✅ Analyzed against your uploaded insurance policy</p>
                  </div>
                )}
                
                {billAnalysis && (
                  <div className="bill-analysis">
                    <div className="analysis-section">
                      <h3>👤 Patient & Provider</h3>
                      <div className="summary-grid">
                        <div className="summary-item">
                          <span className="label">Provider</span>
                          <span className="value">{billAnalysis.bill_summary?.provider_name || 'Provider not specified'}</span>
                        </div>
                        {billAnalysis.bill_summary?.patient_name && (
                          <div className="summary-item">
                            <span className="label">Patient</span>
                            <span className="value">{billAnalysis.bill_summary.patient_name}</span>
                          </div>
                        )}
                        {billAnalysis.bill_summary?.date_of_service && (
                          <div className="summary-item">
                            <span className="label">Date of Service</span>
                            <span className="value">{billAnalysis.bill_summary.date_of_service}</span>
                          </div>
                        )}
                        {billAnalysis.coverage_analysis?.network_status && (
                          <div className="summary-item">
                            <span className="label">Network Status</span>
                            <span className="value">{billAnalysis.coverage_analysis.network_status}</span>
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="analysis-section">
                      <h3>💰 Financial Summary</h3>
                      <div className="financial-summary">
                        <div className="financial-item total">
                          <span className="label">Total amount billed by the provider:</span>
                          <span className="value">${billAnalysis.financial_breakdown?.total_charges?.toFixed(2) || '0.00'}</span>
                        </div>
                        <div className="financial-item patient">
                          <span className="label">What you are paying:</span>
                          <span className="value highlight">${billAnalysis.financial_breakdown?.patient_responsibility?.toFixed(2) || '0.00'}</span>
                        </div>
                        <div className="financial-item insurance">
                          <span className="label">What the insurance is paying:</span>
                          <span className="value">${billAnalysis.financial_breakdown?.insurance_payment?.toFixed(2) || '0.00'}</span>
                        </div>
                        {billAnalysis.financial_breakdown?.amount_saved > 0 && (
                          <div className="financial-item savings">
                            <span className="label">Amount saved:</span>
                            <span className="value">${billAnalysis.financial_breakdown.amount_saved.toFixed(2)}</span>
                          </div>
                        )}
                      </div>
                    </div>

                    {billAnalysis.service_details?.length > 0 && (
                      <div className="analysis-section">
                        <h3>🔍 Service Details</h3>
                        <div className="service-details">
                          {billAnalysis.service_details.map((service, idx) => (
                            <div key={idx} className="service-item">
                              <div className="service-header">
                                <strong>{service.serviceDescription}</strong>
                                {service.serviceCode && <span className="service-code">({service.serviceCode})</span>}
                              </div>
                              <div className="service-amounts">
                                <span>Billed: ${service.providerBilled?.toFixed(2) || '0.00'}</span>
                                <span>Plan Paid: ${service.planPaid?.toFixed(2) || '0.00'}</span>
                                <span>Your Cost: ${service.patientOwed?.toFixed(2) || '0.00'}</span>
                              </div>
                              {service.notes && (
                                <div className="service-notes">{service.notes}</div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="analysis-section">
                      <h3>🔍 Coverage Analysis</h3>
                      <div className="coverage-details">
                        {billAnalysis.coverage_analysis?.summary && (
                          <p><strong>Summary:</strong> {billAnalysis.coverage_analysis.summary}</p>
                        )}
                        {billAnalysis.coverage_analysis?.benefits_applied && (
                          <p><strong>Benefits Applied:</strong> {billAnalysis.coverage_analysis.benefits_applied}</p>
                        )}
                        {billAnalysis.coverage_analysis?.deductible_status && (
                          <p><strong>Deductible Status:</strong> {billAnalysis.coverage_analysis.deductible_status}</p>
                        )}
                      </div>
                    </div>

                    {billAnalysis.discrepancy_check && billAnalysis.discrepancy_check !== "No discrepancies found." && (
                      <div className="analysis-section discrepancy">
                        <h3>⚠️ Discrepancy Check</h3>
                        <div className="discrepancy-content">
                          <p>{billAnalysis.discrepancy_check}</p>
                        </div>
                      </div>
                    )}

                    {billAnalysis.dispute_recommendations?.length > 0 && (
                      <div className="analysis-section disputes">
                        <h3>📋 Recommendations</h3>
                        {billAnalysis.dispute_recommendations.map((dispute, idx) => (
                          <div key={idx} className={`dispute-item ${dispute.priority}`}>
                            <strong>{dispute.issue_type}</strong>
                            <p>{dispute.description}</p>
                            {dispute.recommended_action && (
                              <p><em>Recommended: {dispute.recommended_action}</em></p>
                            )}
                          </div>
                        ))}
                      </div>
                    )}

                    {billAnalysis.full_analysis && (
                      <div className="analysis-section">
                        <h3>📋 Full Analysis</h3>
                        <div className="raw-analysis">
                          <pre>{JSON.stringify(billAnalysis.full_analysis, null, 2)}</pre>
                        </div>
                      </div>
                    )}

                    <div className="confidence-info">
                      <span>Analysis Confidence: {(billAnalysis.confidence_score * 100).toFixed(0)}%</span>
                      {billAnalysis.processing_time_ms && (
                        <span>Processing Time: {billAnalysis.processing_time_ms}ms</span>
                      )}
                    </div>
                  </div>
                )}
                
                <div className="action-buttons">
                  <button className="new-upload-button" onClick={handleBillUpload}>
                    Check Another Bill
                  </button>
                  <button className="new-upload-button" onClick={resetToInitial}>
                    Back to Home
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
        
        {/* Chat State */}
        {currentState === 'chat' && (
          <div className="chat-state">
            <div className="chat-header">
              <h2>Chat with HEAL</h2>
              <button className="back-button" onClick={() => setCurrentState('results')}>
                ← Back to Summary
              </button>
            </div>
            
            <div className="chat-container">
              <div className="messages-container">
                {messages.map((message, index) => (
                  <div key={index} className={`message ${message.type}`}>
                    <div className="message-content">
                      <div className="message-text">{message.content}</div>
                      {message.sources && message.sources.length > 0 && (
                        <div className="message-sources">
                          <span className="sources-label">Sources:</span>
                          {message.sources.map((source, idx) => (
                            <span key={idx} className="source-tag">
                              {source.document} (similarity: {source.similarity})
                            </span>
                          ))}
                        </div>
                      )}
                      {message.confidence && (
                        <div className="confidence-score">
                          Confidence: {Math.round(message.confidence * 100)}%
                        </div>
                      )}
                    </div>
                    <div className="message-time">
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                ))}
                
                {chatLoading && (
                  <div className="message assistant">
                    <div className="message-content">
                      <div className="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>
              
              <div className="chat-input-container">
                <div className="chat-input-wrapper">
                  <textarea
                    className="chat-input"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyPress={handleChatKeyPress}
                    placeholder="Ask me about your insurance policy..."
                    rows="2"
                    disabled={chatLoading}
                  />
                  <button 
                    className="send-button" 
                    onClick={sendMessage}
                    disabled={chatLoading || !chatInput.trim()}
                  >
                    Send
                  </button>
                </div>
                <div className="chat-suggestions">
                  <button 
                    className="suggestion" 
                    onClick={() => setChatInput("What is my deductible?")}
                    disabled={chatLoading}
                  >
                    What is my deductible?
                  </button>
                  <button 
                    className="suggestion" 
                    onClick={() => setChatInput("What does my policy cover?")}
                    disabled={chatLoading}
                  >
                    What does my policy cover?
                  </button>
                  <button 
                    className="suggestion" 
                    onClick={() => setChatInput("How do I file a claim?")}
                    disabled={chatLoading}
                  >
                    How do I file a claim?
                  </button>
                </div>
              </div>
            </div>
            
            <div className="chat-footer">
              <button className="new-upload-button" onClick={resetToInitial}>
                Upload New Document
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
