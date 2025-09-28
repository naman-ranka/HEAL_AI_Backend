import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [results, setResults] = useState(null);
  const [currentState, setCurrentState] = useState('initial'); // 'initial', 'loading', 'results', 'chat'
  
  // Chat state
  const [chatSession, setChatSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [documentId, setDocumentId] = useState(null);
  const messagesEndRef = useRef(null);

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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
        // Store document ID for RAG
        if (data.additional_info && data.additional_info.rag_document_id) {
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

  const resetToInitial = () => {
    setCurrentState('initial');
    setResults(null);
    // Reset chat state
    setChatSession(null);
    setMessages([]);
    setChatInput('');
    setChatLoading(false);
    setDocumentId(null);
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
              Upload Insurance Document
            </button>
          </div>
        )}
        
        {/* Loading State */}
        {currentState === 'loading' && (
          <div className="loading-state">
            <div className="spinner"></div>
            <p className="loading-text">Analyzing your policy...</p>
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
                    Ask Questions About Policy
                  </button>
                  <button className="new-upload-button" onClick={resetToInitial}>
                    Upload Another Document
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
