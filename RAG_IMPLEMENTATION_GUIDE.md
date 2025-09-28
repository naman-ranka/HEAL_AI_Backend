# HEAL RAG Implementation Guide

## 🎯 **Overview**

This document outlines the complete implementation of a Retrieval-Augmented Generation (RAG) system for HEAL using Genkit for Python. The system enables intelligent chatbot functionality by creating a knowledge base from uploaded insurance documents.

## 🏗️ **Architecture**

### **System Flow**
```
Document Upload → Text Extraction → Chunking → Embedding → Vector Storage → RAG Retrieval → Chatbot Response
```

### **Components**
1. **Document Ingestion Pipeline**
2. **Genkit Embedder Integration** 
3. **Local Vector Store**
4. **RAG Retrieval System**
5. **Chatbot with Context**
6. **Debug Dashboard**

## 📊 **Database Schema Extensions**

### **New Tables**
```sql
-- Store uploaded documents
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    original_name TEXT NOT NULL,
    file_size INTEGER,
    mime_type TEXT,
    upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    document_type TEXT, -- 'image' or 'pdf'
    extracted_text TEXT,
    processing_status TEXT DEFAULT 'pending', -- 'pending', 'processed', 'failed'
    chunk_count INTEGER DEFAULT 0
);

-- Store document chunks with embeddings
CREATE TABLE document_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_type TEXT DEFAULT 'paragraph', -- 'paragraph', 'section', 'table'
    embedding BLOB, -- Serialized embedding vector
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Store chat sessions
CREATE TABLE chat_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    user_id TEXT, -- Future user management
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
    document_context TEXT -- JSON array of document IDs in context
);

-- Store chat messages with RAG context
CREATE TABLE chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_type TEXT NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    relevant_chunks TEXT, -- JSON array of chunk IDs used for context
    confidence_score REAL,
    model_used TEXT,
    tokens_used INTEGER,
    processing_time_ms INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
);

-- Store RAG debug information
CREATE TABLE rag_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT NOT NULL,
    embedding BLOB,
    top_chunks TEXT, -- JSON array of retrieved chunks with scores
    execution_time_ms INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## 🔧 **Implementation Phases**

### **Phase 1: Document Ingestion & Storage**

#### **Features:**
- File upload with storage to disk
- Text extraction from images (OCR) and PDFs
- Document chunking strategies
- Metadata extraction and storage

#### **Endpoints:**
```python
POST /upload                    # Enhanced with RAG ingestion
GET /documents                  # List all documents
GET /documents/{id}             # Get document details
DELETE /documents/{id}          # Delete document and chunks
POST /documents/{id}/reprocess  # Reprocess document chunks
```

### **Phase 2: Genkit Embeddings & Vector Storage**

#### **Features:**
- Genkit embedder integration with Gemini
- Local vector store setup
- Chunk embedding generation
- Similarity search implementation

#### **Endpoints:**
```python
POST /embeddings/generate       # Generate embeddings for text
POST /embeddings/search         # Search similar chunks
GET /embeddings/stats          # Embedding statistics
```

### **Phase 3: RAG Retrieval System**

#### **Features:**
- Query embedding and similarity search
- Context ranking and filtering
- Multi-document context aggregation
- Source attribution

#### **Endpoints:**
```python
POST /rag/query                # Test RAG retrieval
GET /rag/context/{query}       # Get context for query
POST /rag/evaluate             # Evaluate retrieval quality
```

### **Phase 4: Chatbot Integration**

#### **Features:**
- Session management
- Conversation history
- Context-aware responses
- Multi-turn conversations

#### **Endpoints:**
```python
POST /chat/sessions            # Create new chat session
GET /chat/sessions/{id}        # Get session details
POST /chat/sessions/{id}/messages  # Send message
GET /chat/sessions/{id}/history    # Get conversation history
DELETE /chat/sessions/{id}     # Delete session
```

### **Phase 5: Debug Dashboard**

#### **Features:**
- Document inspection
- Chunk visualization
- Embedding similarity testing
- Chat conversation debugging
- Performance metrics

#### **Endpoints:**
```python
GET /debug/documents           # List documents with stats
GET /debug/chunks/{doc_id}     # View document chunks
POST /debug/similarity         # Test similarity search
GET /debug/chat/{session_id}   # Debug chat session
GET /debug/embeddings/{chunk_id} # Inspect embeddings
GET /debug/performance         # System performance metrics
```

## 🛠️ **Technical Implementation Details**

### **Genkit Configuration**
```python
from genkit import genkit
from genkit.plugins.google_genai import GoogleAI

# Initialize Genkit with Google AI
ai = genkit(
    plugins=[GoogleAI()],
    model='gemini-2.5-pro'
)

# Configure embedder
embedder = ai.embedder('text-embedding-004')  # Latest Gemini embedding model
```

### **Document Processing Pipeline**
```python
@ai.flow()
async def process_document(file_path: str, document_id: int) -> ProcessingResult:
    """Complete document processing pipeline"""
    
    # 1. Extract text
    text = await extract_text_from_file(file_path)
    
    # 2. Chunk text
    chunks = chunk_text_intelligently(text)
    
    # 3. Generate embeddings
    embeddings = []
    for chunk in chunks:
        embedding = await embedder.embed(chunk.text)
        embeddings.append(embedding)
    
    # 4. Store in database
    store_chunks_with_embeddings(document_id, chunks, embeddings)
    
    return ProcessingResult(
        chunks_created=len(chunks),
        embeddings_generated=len(embeddings),
        processing_time=time.time() - start_time
    )
```

### **RAG Retrieval Flow**
```python
@ai.flow()
async def rag_retrieve(query: str, top_k: int = 5) -> RetrievalResult:
    """Retrieve relevant chunks for query"""
    
    # 1. Generate query embedding
    query_embedding = await embedder.embed(query)
    
    # 2. Search similar chunks
    similar_chunks = search_similar_chunks(query_embedding, top_k)
    
    # 3. Rank and filter results
    ranked_chunks = rank_chunks_by_relevance(similar_chunks, query)
    
    return RetrievalResult(
        query=query,
        chunks=ranked_chunks,
        total_found=len(similar_chunks)
    )
```

### **Chat Flow with RAG**
```python
@ai.flow()
async def chat_with_rag(
    message: str, 
    session_id: str,
    context_limit: int = 3
) -> ChatResponse:
    """Chat with RAG context"""
    
    # 1. Retrieve relevant context
    context = await rag_retrieve(message, top_k=context_limit)
    
    # 2. Build contextual prompt
    prompt = build_rag_prompt(message, context.chunks)
    
    # 3. Generate response
    response = await ai.generate(
        prompt=prompt,
        model='gemini-2.5-pro'
    )
    
    # 4. Save conversation
    save_chat_message(session_id, message, response.text, context.chunks)
    
    return ChatResponse(
        message=response.text,
        sources=[chunk.source for chunk in context.chunks],
        confidence=calculate_confidence(context.chunks),
        session_id=session_id
    )
```

## 🔍 **Debugging Features**

### **1. Genkit Developer UI Integration**
- Visual flow debugging
- Embedding inspection
- Performance profiling
- Error tracing

### **2. Custom Debug Endpoints**
```python
# Document debugging
GET /debug/documents/{id}/chunks    # View all chunks
GET /debug/documents/{id}/embeddings # View embeddings

# Search debugging  
POST /debug/search/test             # Test similarity search
GET /debug/search/history           # Search history

# Chat debugging
GET /debug/chat/{session}/trace     # Full conversation trace
GET /debug/chat/{session}/context   # Context used in responses
```

### **3. Performance Monitoring**
```python
# Metrics tracked:
- Document processing time
- Embedding generation speed
- Search query latency
- Chat response time
- Token usage and costs
- Memory usage
- Database query performance
```

## 📱 **Frontend Integration**

### **New UI Components**

#### **1. Document Manager**
```jsx
// Document upload and management
<DocumentManager>
  <UploadZone />
  <DocumentList />
  <ProcessingStatus />
</DocumentManager>
```

#### **2. Chat Interface**
```jsx
// RAG-powered chatbot
<ChatInterface>
  <MessageHistory />
  <ContextSources />
  <InputArea />
  <TypingIndicator />
</ChatInterface>
```

#### **3. Debug Dashboard**
```jsx
// Development debugging tools
<DebugDashboard>
  <DocumentInspector />
  <EmbeddingVisualizer />
  <ChatTracer />
  <PerformanceMetrics />
</DebugDashboard>
```

### **Frontend Features**
- Real-time chat with typing indicators
- Source attribution for responses
- Document context highlighting
- Conversation history
- Debug mode toggle
- Performance metrics display

## 🚀 **Development Timeline**

### **Week 1: Foundation**
- [ ] Database schema updates
- [ ] Document storage system
- [ ] Basic text extraction
- [ ] Genkit embedder setup

### **Week 2: RAG Core**
- [ ] Chunking algorithms
- [ ] Embedding generation
- [ ] Vector similarity search
- [ ] Basic retrieval system

### **Week 3: Chatbot**
- [ ] Chat session management
- [ ] RAG-powered responses
- [ ] Conversation history
- [ ] Context management

### **Week 4: Frontend & Debug**
- [ ] Chat UI components
- [ ] Document management UI
- [ ] Debug dashboard
- [ ] Performance optimization

## 🎯 **Success Metrics**

### **Technical Metrics**
- Document processing speed: < 30 seconds per document
- Chat response time: < 5 seconds
- Retrieval accuracy: > 80% relevant results
- System uptime: > 99%

### **User Experience Metrics**
- Chat response relevance
- Source attribution accuracy
- Conversation flow quality
- Debug tool usability

## 🔒 **Security Considerations**

### **Data Protection**
- Secure file storage
- Embedding data encryption
- Session management security
- API rate limiting

### **Privacy**
- Document access controls
- Chat history privacy
- Embedding anonymization
- Audit logging

## 📚 **Resources & References**

- [Genkit Documentation](https://genkit.dev/)
- [Genkit Python API](https://python.api.genkit.dev/)
- [Gemini Embedding Models](https://ai.google.dev/gemini-api/docs/embeddings)
- [RAG Best Practices](https://genkit.dev/docs/rag/)

---

**This implementation provides a complete, debuggable RAG system that enhances HEAL's capabilities while maintaining the existing document analysis functionality.**
