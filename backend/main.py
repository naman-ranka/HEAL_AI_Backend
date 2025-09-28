from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sqlite3
import json
import os
import logging
import base64
import time
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv

# Import AI flows and schemas
from ai.flows.policy_analysis import analyze_insurance_policy, summarize_policy_document
from ai.genkit_config import ai_config
from ai.schemas import (
    PolicyAnalysisInput, 
    PolicyAnalysisOutput,
    DocumentType,
    HealthCheckOutput
)

# Import RAG system
from rag import DocumentProcessor, RAGRetriever, InsuranceChatbot
from database import create_rag_tables
from ai.embedder import get_embedder

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HEAL API - AI Powered",
    description="Insurance Policy Analyzer with structured document analysis and chatbot using Genkit-inspired patterns",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check AI availability
ai_available = ai_config.is_available()
if ai_available:
    logger.info("AI services initialized successfully")
else:
    logger.warning("AI services not available - using mock responses")

# Initialize RAG components
document_processor = DocumentProcessor()
rag_retriever = RAGRetriever()
chatbot = InsuranceChatbot()

# Initialize database
def init_db():
    conn = sqlite3.connect("heal.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS policies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            summary_json TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    
    # Initialize RAG tables
    create_rag_tables()

# Initialize database on startup
init_db()
logger.info("Database and RAG system initialized")

@app.post("/upload", response_model=PolicyAnalysisOutput)
async def upload_document(file: UploadFile = File(...)) -> PolicyAnalysisOutput:
    """
    Upload and analyze insurance document using Genkit AI flows
    
    Args:
        file: Uploaded file (image or PDF)
        
    Returns:
        PolicyAnalysisOutput with structured policy information
    """
    try:
        # Check if AI is available
        if not ai_available:
            logger.warning("AI not available, returning mock data")
            mock_response = PolicyAnalysisOutput(
                deductible="$1,000 (mock)",
                out_of_pocket_max="$5,000 (mock)", 
                copay="$25 (mock)",
                confidence_score=0.5,
                additional_info={"note": "Mock data - configure GEMINI_API_KEY for real analysis"}
            )
            save_to_database(mock_response.model_dump())
            return mock_response
        
        # Validate file type
        allowed_types = ["image/", "application/pdf"]
        if not any(file.content_type.startswith(file_type) for file_type in allowed_types):
            raise HTTPException(status_code=400, detail="File must be an image or PDF")
        
        # Read and encode file data
        file_data = await file.read()
        encoded_data = base64.b64encode(file_data).decode('utf-8')
        
        # Determine document type
        doc_type = DocumentType.IMAGE if file.content_type.startswith("image/") else DocumentType.PDF
        
        # Create Genkit input
        analysis_input = PolicyAnalysisInput(
            document_data=encoded_data,
            document_type=doc_type,
            filename=file.filename or "unknown"
        )
        
        logger.info(f"Analyzing {doc_type.value} file: {file.filename}")
        
        # Use Genkit flow for analysis - guaranteed structured output
        analysis_result = await analyze_insurance_policy(analysis_input)
        
        # Save to database (original functionality)
        save_to_database(analysis_result.model_dump())
        
        # Process document for RAG system
        logger.info(f"🚀 Starting RAG processing for {file.filename}")
        try:
            rag_result = await document_processor.process_uploaded_file(
                file_data=file_data,
                filename=file.filename or "unknown",
                mime_type=file.content_type
            )
            logger.info(f"✅ RAG processing completed: {rag_result['chunks_created']} chunks created")
            
            # Add RAG info to response
            if hasattr(analysis_result, 'additional_info') and analysis_result.additional_info:
                analysis_result.additional_info.update({
                    "rag_document_id": rag_result["document_id"],
                    "rag_chunks_created": rag_result["chunks_created"]
                })
            else:
                analysis_result.additional_info = {
                    "rag_document_id": rag_result["document_id"],
                    "rag_chunks_created": rag_result["chunks_created"]
                }
                
        except Exception as rag_error:
            logger.error(f"❌ RAG processing failed: {rag_error}")
            logger.exception("Full RAG error traceback:")
            # Don't fail the main upload, just log the error
            if hasattr(analysis_result, 'additional_info') and analysis_result.additional_info:
                analysis_result.additional_info["rag_error"] = str(rag_error)
            else:
                analysis_result.additional_info = {"rag_error": str(rag_error)}
        
        logger.info(f"Successfully analyzed document: {file.filename}")
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document {getattr(file, 'filename', 'unknown')}: {e}")
        # Return structured error response
        error_response = PolicyAnalysisOutput(
            deductible="Analysis failed",
            out_of_pocket_max="Analysis failed",
            copay="Analysis failed", 
            confidence_score=0.0,
            additional_info={"error": str(e)}
        )
        return error_response


def validate_analysis_result(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean analysis result from Gemini
    
    Args:
        analysis_result: Raw analysis result from Gemini
        
    Returns:
        Cleaned and validated result
    """
    required_keys = ["deductible", "out_of_pocket_max", "copay"]
    
    # If we got a valid response with the required keys
    if all(key in analysis_result for key in required_keys):
        return {
            "deductible": analysis_result["deductible"],
            "out_of_pocket_max": analysis_result["out_of_pocket_max"],
            "copay": analysis_result["copay"]
        }
    
    # If we got a raw response, try to extract what we can
    if "raw_response" in analysis_result:
        logger.warning("Received raw response, using fallback extraction")
        return {
            "deductible": "Could not extract",
            "out_of_pocket_max": "Could not extract",
            "copay": "Could not extract",
            "note": "Analysis completed but data extraction was incomplete"
        }
    
    # Default fallback
    return {
        "deductible": analysis_result.get("deductible", "Not found"),
        "out_of_pocket_max": analysis_result.get("out_of_pocket_max", "Not found"),
        "copay": analysis_result.get("copay", "Not found")
    }


def save_to_database(data: Dict[str, Any]) -> None:
    """
    Save analysis result to database
    
    Args:
        data: Analysis result to save
    """
    try:
        conn = sqlite3.connect("heal.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO policies (summary_json) VALUES (?)",
            (json.dumps(data),)
        )
        conn.commit()
        conn.close()
        logger.info("Data saved to database successfully")
    except Exception as e:
        logger.error(f"Error saving to database: {e}")
        # Don't raise exception here, as the analysis was successful

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "HEAL - Insurance Policy Analyzer API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/upload": "POST - Upload and analyze insurance documents",
            "/health": "GET - Health check",
            "/chat": "POST - Chat with AI assistant (future feature)",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health", response_model=HealthCheckOutput)
async def health_check() -> HealthCheckOutput:
    """
    Health check endpoint with structured response
    
    Returns:
        HealthCheckOutput with system status
    """
    try:
        # Check database
        conn = sqlite3.connect("heal.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM policies")
        policy_count = cursor.fetchone()[0]
        conn.close()
        database_status = "connected"
        
        # Check AI status
        ai_status = "available" if ai_available else "unavailable"
        
        # Check model availability
        model_status = "available" if ai_available else "unavailable"
        
        # Overall status
        overall_status = "healthy" if ai_available and database_status == "connected" else "degraded"
        
        return HealthCheckOutput(
            status=overall_status,
            genkit_status=ai_status,
            model_status=model_status,
            database_status=database_status,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckOutput(
            status="unhealthy",
            genkit_status="error",
            model_status="error", 
            database_status="error",
            timestamp=datetime.now().isoformat()
        )


@app.post("/chat/sessions")
async def create_chat_session(request: Dict[str, Any]) -> Dict[str, str]:
    """
    Create a new chat session
    
    Args:
        request: May contain document_ids to limit context
        
    Returns:
        Session information
    """
    try:
        document_ids = request.get("document_ids")
        session_id = await chatbot.create_session(document_ids)
        
        return {
            "session_id": session_id,
            "status": "created"
        }
        
    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/sessions/{session_id}/messages")
async def send_chat_message(session_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a message to the chatbot
    
    Args:
        session_id: Chat session ID
        request: Contains the user message
        
    Returns:
        Chatbot response with sources
    """
    try:
        message = request.get("message", "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Get chatbot response
        response = await chatbot.chat(message, session_id)
        
        return {
            "message": response.message,
            "sources": response.sources,
            "confidence": response.confidence,
            "processing_time_ms": response.processing_time_ms,
            "session_id": response.session_id
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in chat for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/sessions/{session_id}/history")
async def get_chat_history(session_id: str, limit: int = 50) -> Dict[str, Any]:
    """
    Get chat history for a session
    
    Args:
        session_id: Chat session ID
        limit: Maximum number of messages to return
        
    Returns:
        Chat history
    """
    try:
        history = chatbot.get_chat_history(session_id, limit)
        return {
            "session_id": session_id,
            "messages": history,
            "total_messages": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error getting chat history for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str) -> Dict[str, str]:
    """
    Delete a chat session
    
    Args:
        session_id: Chat session ID
        
    Returns:
        Deletion status
    """
    try:
        deleted = chatbot.delete_session(session_id)
        if deleted:
            return {"status": "deleted", "session_id": session_id}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/sessions")
async def list_chat_sessions(limit: int = 20) -> Dict[str, Any]:
    """
    List recent chat sessions
    
    Args:
        limit: Maximum number of sessions to return
        
    Returns:
        List of chat sessions
    """
    try:
        sessions = chatbot.list_sessions(limit)
        return {
            "sessions": sessions,
            "total": len(sessions)
        }
        
    except Exception as e:
        logger.error(f"Error listing chat sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize")
async def summarize_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of an insurance document
    
    Args:
        file: Uploaded insurance document
        
    Returns:
        Document summary and analysis
    """
    try:
        if not genkit_available:
            return {
                "summary": "Document summarization is currently unavailable. Please configure Genkit.",
                "document_type": "unknown",
                "filename": file.filename
            }
        
        # Validate file type
        allowed_types = ["image/", "application/pdf"]
        if not any(file.content_type.startswith(file_type) for file_type in allowed_types):
            raise HTTPException(status_code=400, detail="File must be an image or PDF")
        
        # Read and encode file data
        file_data = await file.read()
        encoded_data = base64.b64encode(file_data).decode('utf-8')
        
        # Determine document type
        doc_type = DocumentType.IMAGE if file.content_type.startswith("image/") else DocumentType.PDF
        
        # Create input for summarization
        summary_input = PolicyAnalysisInput(
            document_data=encoded_data,
            document_type=doc_type,
            filename=file.filename or "unknown"
        )
        
        logger.info(f"Summarizing document: {file.filename}")
        
        # Use Genkit flow for document summarization
        summary_result = await summarize_policy_document(summary_input)
        
        return summary_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing document: {e}")
        return {
            "summary": f"Failed to generate summary: {str(e)}",
            "document_type": "unknown",
            "filename": file.filename or "unknown"
        }


@app.post("/generate-questions")
async def generate_questions(request: Dict[str, str]) -> List[str]:
    """
    Generate relevant questions about a policy
    
    Args:
        request: Dictionary containing policy_text
        
    Returns:
        List of suggested questions
    """
    try:
        if not ai_available:
            return [
                "What is my deductible?",
                "What does my policy cover?", 
                "How do I file a claim?",
                "What are my copay amounts?",
                "What is not covered by my policy?"
            ]
        
        policy_text = request.get("policy_text", "")
        if not policy_text:
            raise HTTPException(status_code=400, detail="Policy text is required")
        
        # Use Genkit flow to generate questions
        questions = await generate_policy_questions(policy_text)
        
        return questions
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        return ["Sorry, I couldn't generate questions at this time."]


@app.get("/models")
async def get_available_models() -> Dict[str, Any]:
    """
    Get available Genkit models and configuration
    
    Returns:
        Dictionary with model information
    """
    try:
        if not ai_available:
            return {
                "models": [],
                "status": "AI not available",
                "configured_models": {
                    "flash": "gemini-1.5-flash",
                    "pro": "gemini-1.5-pro"
                }
            }
        
        return {
            "models": [
                "googleai/gemini-1.5-flash",
                "googleai/gemini-1.5-pro"
            ],
            "status": "available",
            "configured_models": {
                "vision": "googleai/gemini-1.5-flash", 
                "pro": "googleai/gemini-1.5-pro"
            },
            "framework": "Genkit for Python"
        }
        
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return {
            "models": [],
            "status": "error",
            "error": str(e)
        }


def save_to_database(data: Dict[str, Any]) -> None:
    """
    Save analysis result to database
    
    Args:
        data: Analysis result to save
    """
    try:
        conn = sqlite3.connect("heal.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO policies (summary_json) VALUES (?)",
            (json.dumps(data),)
        )
        conn.commit()
        conn.close()
        logger.info("Data saved to database successfully")
    except Exception as e:
        logger.error(f"Error saving to database: {e}")


# Document Management Endpoints
@app.get("/documents")
async def list_documents() -> Dict[str, Any]:
    """List all uploaded documents"""
    try:
        documents = document_processor.list_documents()
        return {"documents": documents, "total": len(documents)}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{document_id}")
async def get_document_info(document_id: int) -> Dict[str, Any]:
    """Get detailed information about a document"""
    try:
        doc_info = document_processor.get_document_info(document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document info for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# RAG System Endpoints
@app.post("/rag/search")
async def search_rag(request: Dict[str, Any]) -> Dict[str, Any]:
    """Search for relevant chunks using RAG"""
    try:
        query = request.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        result = await rag_retriever.retrieve(
            query=query,
            top_k=request.get("top_k", 5),
            similarity_threshold=request.get("similarity_threshold", 0.1),
            document_ids=request.get("document_ids")
        )
        
        return {
            "query": result.query,
            "chunks": [{
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "similarity_score": chunk.similarity_score,
                "source_document": chunk.source_document
            } for chunk in result.chunks],
            "total_found": result.total_found,
            "execution_time_ms": result.execution_time_ms
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in RAG search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/stats")
async def get_rag_stats() -> Dict[str, Any]:
    """Get RAG system statistics"""
    try:
        return rag_retriever.get_retrieval_stats()
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Comprehensive Debug Endpoints
@app.get("/debug/upload-process/{document_id}")
async def debug_upload_process(document_id: int) -> Dict[str, Any]:
    """
    Debug: Get complete upload process information for a document
    """
    try:
        # Get document info
        doc_info = document_processor.get_document_info(document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get chunks info
        from database import get_db_connection
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    id, chunk_text, chunk_index, chunk_type,
                    CASE WHEN embedding IS NOT NULL THEN 'Yes' ELSE 'No' END as has_embedding,
                    length(chunk_text) as text_length
                FROM document_chunks 
                WHERE document_id = ?
                ORDER BY chunk_index
            """, (document_id,))
            chunks = [dict(row) for row in cursor.fetchall()]
            
            # Get policy analysis result
            cursor.execute("""
                SELECT summary_json FROM policies 
                ORDER BY id DESC LIMIT 1
            """)
            policy_result = cursor.fetchone()
            
        finally:
            conn.close()
        
        return {
            "document_info": doc_info,
            "chunks": chunks,
            "chunk_statistics": {
                "total_chunks": len(chunks),
                "chunks_with_embeddings": len([c for c in chunks if c['has_embedding'] == 'Yes']),
                "average_chunk_length": sum(c['text_length'] for c in chunks) / len(chunks) if chunks else 0,
                "total_text_length": sum(c['text_length'] for c in chunks)
            },
            "policy_analysis": json.loads(policy_result[0]) if policy_result else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in debug upload process for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/latest-upload")
async def debug_latest_upload() -> Dict[str, Any]:
    """
    Debug: Get information about the most recent upload
    """
    try:
        # Get latest document
        from database import get_db_connection
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM documents 
                ORDER BY upload_timestamp DESC 
                LIMIT 1
            """)
            
            latest_doc = cursor.fetchone()
            if not latest_doc:
                return {"message": "No documents found"}
            
            doc_dict = dict(latest_doc)
            
        finally:
            conn.close()
        
        # Get full debug info for this document
        return await debug_upload_process(doc_dict['id'])
        
    except Exception as e:
        logger.error(f"Error getting latest upload debug info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/gemini-calls")
async def debug_gemini_calls() -> Dict[str, Any]:
    """
    Debug: Get information about recent Gemini API calls
    """
    try:
        # This would require adding logging to the AI flows
        # For now, return basic AI configuration info
        return {
            "ai_available": ai_available,
            "models_configured": ai_config.models if ai_available else {},
            "api_key_configured": bool(ai_config.api_key) if ai_available else False,
            "recent_calls": "Logging not yet implemented - check server logs"
        }
        
    except Exception as e:
        logger.error(f"Error getting Gemini debug info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/test-gemini")
async def debug_test_gemini(request: Dict[str, str]) -> Dict[str, Any]:
    """
    Debug: Test Gemini API with a simple prompt
    """
    try:
        if not ai_available:
            return {
                "success": False,
                "error": "AI services not available",
                "ai_config": {
                    "api_key_configured": False,
                    "models": {}
                }
            }
        
        test_prompt = request.get("prompt", "Hello, this is a test. Please respond with 'Test successful'.")
        
        # Test with flash model
        model = ai_config.get_model('flash')
        start_time = time.time()
        
        response = model.generate_content(test_prompt)
        end_time = time.time()
        
        return {
            "success": True,
            "prompt": test_prompt,
            "response": response.text,
            "model_used": "gemini-2.5-flash",
            "response_time_ms": int((end_time - start_time) * 1000),
            "ai_config": {
                "api_key_configured": True,
                "models": ai_config.models
            }
        }
        
    except Exception as e:
        logger.error(f"Error testing Gemini: {e}")
        return {
            "success": False,
            "error": str(e),
            "prompt": request.get("prompt", ""),
            "ai_config": {
                "api_key_configured": bool(ai_config.api_key) if ai_available else False,
                "models": ai_config.models if ai_available else {}
            }
        }


@app.get("/debug/embeddings/stats")
async def debug_embedding_stats() -> Dict[str, Any]:
    """
    Debug: Get embedding system statistics
    """
    try:
        embedder = get_embedder()
        stats = embedder.get_stats()
        
        return {
            "embedding_stats": stats,
            "api_available": ai_available,
            "system_status": "operational" if stats["success_rate_percent"] > 50 else "degraded"
        }
        
    except Exception as e:
        logger.error(f"Error getting embedding stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/embeddings/test")
async def debug_test_embedding(request: Dict[str, str]) -> Dict[str, Any]:
    """
    Debug: Test embedding generation for a text
    """
    try:
        text = request.get("text", "Test embedding text")
        task_type = request.get("task_type", "retrieval_document")
        
        embedder = get_embedder()
        
        # Import task type enum
        from ai.embedder import EmbeddingTaskType
        task_enum = EmbeddingTaskType(task_type)
        
        result = await embedder.embed_text(text, task_enum)
        
        return {
            "text": text,
            "task_type": task_type,
            "success": result.success,
            "model_used": result.model_used,
            "execution_time_ms": result.execution_time_ms,
            "embedding_dimension": len(result.embedding) if result.embedding is not None else 0,
            "embedding_preview": result.embedding[:10].tolist() if result.embedding is not None else None,
            "error_message": result.error_message
        }
        
    except Exception as e:
        logger.error(f"Error testing embedding: {e}")
        return {
            "success": False,
            "error": str(e),
            "text": request.get("text", ""),
            "task_type": request.get("task_type", "")
        }


@app.post("/debug/embeddings/compare")
async def debug_compare_embeddings(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Debug: Compare embeddings between two texts
    """
    try:
        text1 = request.get("text1", "")
        text2 = request.get("text2", "")
        
        if not text1 or not text2:
            raise HTTPException(status_code=400, detail="Both text1 and text2 are required")
        
        embedder = get_embedder()
        from ai.embedder import EmbeddingTaskType
        
        # Generate embeddings
        result1 = await embedder.embed_text(text1, EmbeddingTaskType.SEMANTIC_SIMILARITY)
        result2 = await embedder.embed_text(text2, EmbeddingTaskType.SEMANTIC_SIMILARITY)
        
        if not (result1.success and result2.success):
            return {
                "success": False,
                "error": "Failed to generate embeddings",
                "result1": {"success": result1.success, "error": result1.error_message},
                "result2": {"success": result2.success, "error": result2.error_message}
            }
        
        # Calculate similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(
            result1.embedding.reshape(1, -1),
            result2.embedding.reshape(1, -1)
        )[0][0]
        
        return {
            "text1": text1,
            "text2": text2,
            "similarity_score": float(similarity),
            "model_used": result1.model_used,
            "execution_time_ms": result1.execution_time_ms + result2.execution_time_ms,
            "interpretation": "Very similar" if similarity > 0.8 else "Similar" if similarity > 0.6 else "Somewhat similar" if similarity > 0.4 else "Different"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/chat/context")
async def debug_chat_context(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Debug: Test the enhanced context building for chat
    """
    try:
        query = request.get("query", "What is my deductible?")
        context_limit = request.get("context_limit", 5)
        
        # Get retrieval results
        result = await rag_retriever.retrieve(
            query=query,
            top_k=context_limit,
            similarity_threshold=0.3
        )
        
        # Get policy summary
        chatbot_instance = chatbot
        policy_summary = await chatbot_instance._get_policy_summary([])
        
        # Build both types of context
        basic_context = chatbot_instance._build_context_from_chunks(result.chunks)
        enhanced_context = chatbot_instance._build_enhanced_context(result.chunks, policy_summary)
        
        return {
            "query": query,
            "chunks_found": len(result.chunks),
            "execution_time_ms": result.execution_time_ms,
            "policy_summary": policy_summary,
            "basic_context": basic_context,
            "enhanced_context": enhanced_context,
            "chunks_details": [
                {
                    "similarity": chunk.similarity_score,
                    "source": chunk.source_document,
                    "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                }
                for chunk in result.chunks
            ]
        }
        
    except Exception as e:
        logger.error(f"Error testing chat context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Start the Genkit development server for enhanced debugging
    logger.info("Starting HEAL API with Genkit integration")
    logger.info("Visit http://localhost:8000/docs for API documentation")
    logger.info("Use 'genkit start -- python main.py' for Genkit Developer UI")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
