#!/usr/bin/env python3
"""
HEAL - Single LangChain Implementation
Complete RAG + Conversational Chat system using LangChain
Replaces all previous implementations with production-ready LangChain solution
"""

import os
import logging
import json
import uuid
import time
import tempfile
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

# FastAPI
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain Core
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# LangChain Chat Models
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# LangChain Document Processing
from langchain_community.document_loaders import PyPDFLoader, UnstructuredImageLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain Vector Store
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Database for session management
import sqlite3

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="HEAL - LangChain RAG Chat",
    description="Insurance policy assistant powered by LangChain RAG and conversational AI",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str

class SessionCreate(BaseModel):
    document_context: Optional[List[str]] = None

class ChatResponse(BaseModel):
    message: str
    sources: List[Dict[str, Any]]
    confidence: float
    session_id: str
    processing_time_ms: int

class DocumentUploadResponse(BaseModel):
    success: bool
    document_id: str
    chunks_created: int
    processing_time_ms: int
    message: str

# Global components
chat_model: Optional[ChatGoogleGenerativeAI] = None
embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
vector_store: Optional[FAISS] = None
retriever = None
conversational_rag_chain = None

# Database setup
DB_PATH = "langchain_heal.db"

def init_database():
    """Initialize SQLite database for session management"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Chat sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            document_context TEXT
        )
    """)
    
    # Chat messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
        )
    """)
    
    # Document metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            chunks_created INTEGER,
            upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info("📊 Database initialized")

def get_chat_history(session_id: str, limit: int = 10) -> List:
    """Get chat history for session as LangChain messages"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT role, content FROM chat_messages 
        WHERE session_id = ? 
        ORDER BY timestamp ASC 
        LIMIT ?
    """, (session_id, limit))
    
    messages = []
    for role, content in cursor.fetchall():
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
    
    conn.close()
    return messages

def store_message(session_id: str, role: str, content: str):
    """Store message in database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO chat_messages (session_id, role, content)
        VALUES (?, ?, ?)
    """, (session_id, role, content))
    
    # Update session activity
    cursor.execute("""
        UPDATE chat_sessions 
        SET last_activity = CURRENT_TIMESTAMP 
        WHERE session_id = ?
    """, (session_id,))
    
    conn.commit()
    conn.close()

def get_policy_summary() -> str:
    """Get policy summary from uploaded documents or return default"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get the most recent document
        cursor.execute("""
            SELECT filename FROM documents 
            ORDER BY upload_timestamp DESC 
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            filename = row[0]
            
            # Basic policy summary based on typical insurance structure
            return f"""
Policy Document: {filename}

Key Coverage Areas:
• Deductible: Amount you pay before insurance coverage begins
• Out-of-pocket Maximum: Maximum amount you'll pay in a policy year  
• Copays: Fixed amounts for specific services (primary care, specialist visits)
• Preventive Care: Often covered at 100% (annual exams, screenings)
• Prescription Drugs: Coverage with different tiers and copays
• Dental Services: May include preventive, basic, and major procedures
• Vision Services: Eye exams, frames, and lens coverage
• Emergency Services: Coverage for urgent medical care

Important: Specific amounts, limits, and exclusions are detailed in your policy documents.
For exact coverage details, refer to your Schedule of Benefits or contact your insurance provider.
"""
        else:
            return """
No policy document currently uploaded.

General Insurance Information:
• Insurance policies typically include deductibles, copays, and coverage limits
• Preventive care is often covered at 100%
• Different services may have different coverage levels
• Always verify specific coverage with your insurance provider

To get personalized information, please upload your insurance policy document.
"""
            
    except Exception as e:
        logger.error(f"Error getting policy summary: {e}")
        return "Policy information temporarily unavailable. Please try again or contact support."

@app.on_event("startup")
async def startup_event():
    """Initialize LangChain components"""
    global chat_model, embeddings, vector_store, retriever, conversational_rag_chain
    
    try:
        # Initialize database
        init_database()
        
        # Check API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("❌ GEMINI_API_KEY not found")
            return
        
        # Initialize LangChain components
        chat_model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.1,
            convert_system_message_to_human=True
        )
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )
        
        # Initialize empty vector store
        vector_store = FAISS.from_texts(
            ["Placeholder text for initialization"],
            embeddings,
            metadatas=[{"source": "init"}]
        )
        retriever = vector_store.as_retriever(search_k=5)
        
        # Create conversational RAG chain using LCEL
        system_prompt = """You are HEAL, an expert insurance policy assistant. 
        
Use BOTH the policy overview and specific document context to provide comprehensive answers.
The input may contain a Policy Overview section followed by the current question.
Also use the retrieved context from insurance documents to provide specific details.

Guidelines:
1. Start with information from the policy overview when available
2. Use specific document context to provide detailed answers
3. Maintain conversational flow by referencing previous discussions
4. If information is not available, clearly state what you don't have
5. Be helpful, clear, and use simple language
6. Always be accurate and conservative in your responses
7. Reference both overview and specific sections when relevant

Context from insurance documents:
{context}

Previous conversation:
{chat_history}

Input (may include policy overview + question): {input}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Create the chain using LCEL
        document_chain = create_stuff_documents_chain(chat_model, prompt)
        conversational_rag_chain = create_retrieval_chain(retriever, document_chain)
        
        logger.info("🚀 LangChain HEAL system initialized successfully")
        logger.info(f"🤖 Model: {chat_model.model}")
        logger.info(f"🧠 Embeddings: {embeddings.model}")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    api_key = os.getenv("GEMINI_API_KEY")
    
    return {
        "status": "healthy",
        "system": "langchain-heal",
        "version": "3.0.0",
        "framework": "LangChain",
        "api_key_configured": bool(api_key),
        "components_ready": all([chat_model, embeddings, vector_store, conversational_rag_chain]),
        "model": chat_model.model if chat_model else None
    }

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document using LangChain"""
    if not all([embeddings, vector_store]):
        raise HTTPException(status_code=500, detail="System not initialized")
    
    start_time = time.time()
    
    try:
        # Validate file type
        allowed_types = ['application/pdf', 'image/jpeg', 'image/png']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}"
            )
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        documents = []
        
        try:
            # Load document using LangChain loaders
            if file.content_type == 'application/pdf':
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
            else:  # Image files
                loader = UnstructuredImageLoader(tmp_file_path)
                docs = loader.load()
            
            # Split documents using LangChain text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=750,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            split_docs = text_splitter.split_documents(docs)
            
            # Add metadata
            for i, doc in enumerate(split_docs):
                doc.metadata.update({
                    'source_document': file.filename,
                    'chunk_index': i,
                    'upload_time': datetime.now().isoformat()
                })
            
            # Add to vector store using LangChain
            if len(split_docs) > 0:
                vector_store.add_documents(split_docs)
                logger.info(f"✅ Added {len(split_docs)} chunks to vector store")
            
            # Store document metadata
            doc_id = str(uuid.uuid4())
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO documents (id, filename, file_path, chunks_created)
                VALUES (?, ?, ?, ?)
            """, (doc_id, file.filename, tmp_file_path, len(split_docs)))
            conn.commit()
            conn.close()
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return DocumentUploadResponse(
                success=True,
                document_id=doc_id,
                chunks_created=len(split_docs),
                processing_time_ms=processing_time,
                message=f"Successfully processed {file.filename} using LangChain RAG"
            )
            
        finally:
            # Clean up temp file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"❌ Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/chat/sessions")
async def create_chat_session(session_data: SessionCreate):
    """Create new chat session"""
    session_id = str(uuid.uuid4())
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO chat_sessions (session_id, document_context)
        VALUES (?, ?)
    """, (session_id, json.dumps(session_data.document_context) if session_data.document_context else None))
    
    conn.commit()
    conn.close()
    
    logger.info(f"💬 Created LangChain chat session: {session_id}")
    
    return {
        "session_id": session_id,
        "message": "Chat session created successfully"
    }

@app.post("/chat/sessions/{session_id}/messages", response_model=ChatResponse)
async def send_chat_message(session_id: str, message_data: ChatMessage):
    """Send message using LangChain conversational RAG with policy summary"""
    if not conversational_rag_chain:
        raise HTTPException(status_code=500, detail="RAG chain not initialized")
    
    start_time = time.time()
    
    try:
        # Get conversation history
        chat_history = get_chat_history(session_id, limit=10)
        
        # Get policy summary to enhance context
        policy_summary = get_policy_summary()
        
        # Enhanced input with policy summary
        enhanced_input = f"""Policy Overview: {policy_summary}

Current Question: {message_data.message}"""
        
        # Use LangChain conversational RAG chain with enhanced input
        response = await conversational_rag_chain.ainvoke({
            "input": enhanced_input,
            "chat_history": chat_history
        })
        
        # Extract answer and source documents
        answer = response["answer"]
        source_documents = response.get("context", [])
        
        # Store messages
        store_message(session_id, "human", message_data.message)
        store_message(session_id, "ai", answer)
        
        # Calculate confidence based on retrieval score
        confidence = 0.8  # Default confidence
        if source_documents:
            # Simple confidence based on number of relevant documents
            confidence = min(0.95, 0.5 + (len(source_documents) * 0.1))
        
        # Prepare sources
        sources = []
        for i, doc in enumerate(source_documents[:5]):  # Top 5 sources
            sources.append({
                "document": doc.metadata.get("source_document", f"Document {i+1}"),
                "content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                "metadata": doc.metadata
            })
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"💬 LangChain response generated in {processing_time}ms")
        
        return ChatResponse(
            message=answer,
            sources=sources,
            confidence=confidence,
            session_id=session_id,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Chat message failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/chat/sessions/{session_id}/history")
async def get_chat_history_endpoint(session_id: str):
    """Get chat history for session"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT role, content, timestamp FROM chat_messages 
        WHERE session_id = ? 
        ORDER BY timestamp ASC
    """, (session_id,))
    
    messages = []
    for role, content, timestamp in cursor.fetchall():
        messages.append({
            "role": role,
            "content": content,
            "timestamp": timestamp
        })
    
    conn.close()
    
    return {
        "session_id": session_id,
        "messages": messages,
        "total_messages": len(messages)
    }

@app.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete chat session"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Delete messages first
    cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
    
    # Delete session
    cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
    
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    if deleted:
        logger.info(f"🗑️ Deleted session: {session_id}")
    
    return {"success": deleted}

# Debug endpoints
@app.get("/debug/langchain")
async def debug_langchain():
    """Debug LangChain components"""
    return {
        "framework": "LangChain",
        "components": {
            "chat_model": chat_model.__class__.__name__ if chat_model else None,
            "embeddings": embeddings.__class__.__name__ if embeddings else None,
            "vector_store": vector_store.__class__.__name__ if vector_store else None,
            "chain": conversational_rag_chain.__class__.__name__ if conversational_rag_chain else None
        },
        "models": {
            "chat": chat_model.model if chat_model else None,
            "embedding": embeddings.model if embeddings else None
        },
        "vector_store_info": {
            "total_docs": len(vector_store.docstore._dict) if vector_store else 0
        },
        "database": {
            "path": DB_PATH,
            "exists": os.path.exists(DB_PATH)
        }
    }

@app.post("/debug/test-chat")
async def debug_test_chat(data: Dict[str, str]):
    """Test LangChain chat model directly"""
    if not chat_model:
        raise HTTPException(status_code=500, detail="Chat model not initialized")
    
    try:
        prompt = data.get("prompt", "Hello, this is a test.")
        
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=prompt)
        ]
        
        response = await chat_model.ainvoke(messages)
        
        return {
            "success": True,
            "prompt": prompt,
            "response": response.content,
            "model": chat_model.model
        }
        
    except Exception as e:
        logger.error(f"❌ Chat test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug/test-retrieval")
async def debug_test_retrieval(data: Dict[str, Any]):
    """Test LangChain retrieval"""
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    
    try:
        query = data.get("query", "What is my deductible?")
        k = data.get("k", 3)
        
        # Update retriever k value
        retriever.search_kwargs = {"k": k}
        
        docs = await retriever.ainvoke(query)
        
        return {
            "success": True,
            "query": query,
            "documents_found": len(docs),
            "documents": [
                {
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Retrieval test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug/test-enhanced-context")
async def debug_test_enhanced_context(data: Dict[str, str]):
    """Test enhanced context with policy summary + RAG chunks"""
    try:
        query = data.get("query", "What is my deductible?")
        
        # Get policy summary
        policy_summary = get_policy_summary()
        
        # Test retrieval
        if retriever:
            retriever.search_kwargs = {"k": 3}
            docs = await retriever.ainvoke(query)
            
            # Build enhanced input
            enhanced_input = f"""Policy Overview: {policy_summary}

Current Question: {query}"""
            
            return {
                "success": True,
                "query": query,
                "policy_summary": policy_summary,
                "retrieved_docs": len(docs),
                "enhanced_input": enhanced_input,
                "doc_previews": [
                    {
                        "content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in docs[:3]
                ]
            }
        else:
            return {
                "success": False,
                "error": "Retriever not initialized",
                "policy_summary": policy_summary,
                "enhanced_input": f"Policy Overview: {policy_summary}\n\nCurrent Question: {query}"
            }
            
    except Exception as e:
        logger.error(f"❌ Enhanced context test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting HEAL - LangChain RAG Chat System")
    logger.info("📚 Single implementation using LangChain framework")
    logger.info("🔗 Visit http://localhost:8000/docs for API documentation")
    
    uvicorn.run(
        "langchain_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
