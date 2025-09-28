#!/usr/bin/env python3
"""
FastAPI Integration for Genkit-style RAG Chat System
Provides REST API endpoints following Genkit patterns
"""

import asyncio
import logging
import os
import tempfile
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from genkit_rag_chat import GenkitRAGChat, ChatResponse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HEAL Genkit RAG Chat API",
    description="Insurance policy assistant powered by Genkit-style RAG and conversational AI",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG chat instance
rag_chat: Optional[GenkitRAGChat] = None

# Pydantic models for API
class ChatMessage(BaseModel):
    message: str
    system_prompt: Optional[str] = None

class SessionCreate(BaseModel):
    document_context: Optional[List[str]] = None

class DocumentUploadResponse(BaseModel):
    success: bool
    document_id: str
    chunks_created: int
    processing_time_ms: int
    message: str

class ChatSessionResponse(BaseModel):
    session_id: str
    message: str

class GenkitChatResponse(BaseModel):
    message: str
    sources: List[Dict[str, Any]]
    confidence: float
    session_id: str
    processing_time_ms: int
    metadata: Optional[Dict[str, Any]] = None

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize Genkit RAG Chat system"""
    global rag_chat
    try:
        rag_chat = GenkitRAGChat(db_path="genkit_heal.db")
        logger.info("🚀 Genkit RAG Chat system initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Genkit system: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Genkit RAG Chat system")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    api_key = os.getenv("GEMINI_API_KEY")
    
    return {
        "status": "healthy",
        "system": "genkit-rag-chat",
        "version": "2.0.0",
        "genkit_style": True,
        "api_key_configured": bool(api_key),
        "rag_chat_ready": rag_chat is not None
    }

# Document upload endpoint
@app.post("/genkit/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document using Genkit-style RAG"""
    if not rag_chat:
        raise HTTPException(status_code=500, detail="RAG Chat system not initialized")
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Validate file type
        allowed_types = ['application/pdf', 'image/jpeg', 'image/png', 'text/plain']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Process document using Genkit-style processing
            success = await rag_chat.process_document(
                file_path=tmp_file_path,
                chunk_size=750,
                chunk_overlap=100
            )
            
            if not success:
                raise HTTPException(status_code=500, detail="Document processing failed")
            
            # Calculate chunks created (estimate based on file size)
            chunks_estimate = max(1, len(content) // 3000)  # Rough estimate
            
            processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            
            logger.info(f"✅ Successfully processed {file.filename} using Genkit RAG")
            
            return DocumentUploadResponse(
                success=True,
                document_id=tmp_file_path,  # In production, use proper ID
                chunks_created=chunks_estimate,
                processing_time_ms=processing_time,
                message=f"Successfully processed {file.filename} using Genkit-style RAG"
            )
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Chat session endpoints
@app.post("/genkit/chat/sessions", response_model=ChatSessionResponse)
async def create_chat_session(session_data: SessionCreate):
    """Create new Genkit-style chat session"""
    if not rag_chat:
        raise HTTPException(status_code=500, detail="RAG Chat system not initialized")
    
    try:
        session_id = await rag_chat.create_chat_session(
            document_context=session_data.document_context
        )
        
        return ChatSessionResponse(
            session_id=session_id,
            message="Genkit chat session created successfully"
        )
        
    except Exception as e:
        logger.error(f"❌ Session creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")

@app.post("/genkit/chat/sessions/{session_id}/messages", response_model=GenkitChatResponse)
async def send_chat_message(session_id: str, message_data: ChatMessage):
    """Send message to Genkit-style chat session"""
    if not rag_chat:
        raise HTTPException(status_code=500, detail="RAG Chat system not initialized")
    
    try:
        # Use Genkit-style chat processing
        response = await rag_chat.chat(
            session_id=session_id,
            message=message_data.message,
            top_k=5,
            system_prompt=message_data.system_prompt
        )
        
        # Convert to API response format
        return GenkitChatResponse(
            message=response.message,
            sources=response.sources,
            confidence=response.confidence,
            session_id=response.session_id,
            processing_time_ms=response.processing_time_ms,
            metadata=response.metadata
        )
        
    except Exception as e:
        logger.error(f"❌ Chat message failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

# Genkit-specific debug endpoints
@app.get("/genkit/debug/system")
async def debug_genkit_system():
    """Debug information about Genkit system"""
    if not rag_chat:
        return {"error": "RAG Chat system not initialized"}
    
    return {
        "system": "genkit-rag-chat",
        "embedder_model": rag_chat.embedder.model_name,
        "generator_model": rag_chat.generator.model_name,
        "database_path": rag_chat.db_path,
        "genkit_style": True,
        "components": {
            "embedder": "GenkitStyleEmbedder",
            "retriever": "GenkitStyleRetriever", 
            "generator": "GenkitStyleGenerator",
            "chat": "GenkitRAGChat"
        }
    }

@app.post("/genkit/debug/test-embedding")
async def debug_test_embedding(data: Dict[str, str]):
    """Test Genkit-style embedding generation"""
    if not rag_chat:
        raise HTTPException(status_code=500, detail="RAG Chat system not initialized")
    
    try:
        text = data.get("text", "Test embedding text")
        
        # Test embedding
        embedding = await rag_chat.embedder.embed_query(text)
        
        return {
            "success": True,
            "text": text,
            "embedding_dimension": len(embedding),
            "embedding_preview": embedding[:5],  # First 5 values
            "model": rag_chat.embedder.model_name,
            "genkit_style": True
        }
        
    except Exception as e:
        logger.error(f"❌ Embedding test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/genkit/debug/test-retrieval")
async def debug_test_retrieval(data: Dict[str, Any]):
    """Test Genkit-style document retrieval"""
    if not rag_chat:
        raise HTTPException(status_code=500, detail="RAG Chat system not initialized")
    
    try:
        query = data.get("query", "What is my deductible?")
        top_k = data.get("top_k", 3)
        
        # Test retrieval
        result = await rag_chat.retriever.retrieve(
            query=query,
            top_k=top_k,
            similarity_threshold=0.1
        )
        
        return {
            "success": True,
            "query": result.query,
            "documents_found": result.total_found,
            "documents_returned": len(result.documents),
            "execution_time_ms": result.execution_time_ms,
            "documents": [
                {
                    "content_preview": doc.content[:100] + "..." if len(doc.content) > 100 else doc.content,
                    "similarity": doc.metadata.get('similarity_score', 0),
                    "source": doc.metadata.get('source_document', 'Unknown')
                }
                for doc in result.documents
            ],
            "genkit_style": True
        }
        
    except Exception as e:
        logger.error(f"❌ Retrieval test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/genkit/debug/test-generation")
async def debug_test_generation(data: Dict[str, Any]):
    """Test Genkit-style response generation"""
    if not rag_chat:
        raise HTTPException(status_code=500, detail="RAG Chat system not initialized")
    
    try:
        prompt = data.get("prompt", "Hello, this is a test prompt.")
        system_prompt = data.get("system_prompt", "You are a helpful assistant.")
        
        # Test generation
        response = await rag_chat.generator.generate(
            prompt=prompt,
            system_prompt=system_prompt
        )
        
        return {
            "success": True,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "response": response,
            "model": rag_chat.generator.model_name,
            "genkit_style": True
        }
        
    except Exception as e:
        logger.error(f"❌ Generation test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Comparison endpoint (old vs new)
@app.get("/genkit/compare")
async def compare_implementations():
    """Compare old implementation vs Genkit-style implementation"""
    return {
        "comparison": {
            "old_implementation": {
                "architecture": "Custom RAG with manual components",
                "embedder": "Custom GeminiEmbedder with hash fallback",
                "retriever": "Manual SQLite cosine similarity search",
                "generator": "Direct Gemini API calls",
                "conversation": "Manual history management",
                "patterns": "Custom patterns"
            },
            "genkit_implementation": {
                "architecture": "Genkit-style patterns and flows",
                "embedder": "GenkitStyleEmbedder with proper task types",
                "retriever": "GenkitStyleRetriever with structured documents",
                "generator": "GenkitStyleGenerator with flow patterns",
                "conversation": "Genkit-style conversation management",
                "patterns": "Following Genkit best practices"
            },
            "benefits_of_genkit": [
                "Standardized patterns and interfaces",
                "Better structured document handling",
                "Proper task-specific embeddings",
                "Flow-based architecture",
                "Easier testing and debugging",
                "More maintainable code structure",
                "Better separation of concerns",
                "Future compatibility with full Genkit"
            ]
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting HEAL Genkit-style RAG Chat API")
    logger.info("📚 Genkit patterns implemented for production-ready RAG")
    logger.info("🔗 Visit http://localhost:8001/docs for API documentation")
    
    uvicorn.run(
        "genkit_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
