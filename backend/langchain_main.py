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
import base64
import io
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

# For policy analysis
import google.generativeai as genai
from PIL import Image
import fitz  # PyMuPDF

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

# Policy Analysis Models (need to be defined early)
class PolicyAnalysisResponse(BaseModel):
    """Response model for policy analysis"""
    deductible: str
    out_of_pocket_max: str
    copay: str
    confidence_score: float
    additional_info: Optional[Dict[str, Any]] = None

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
    policy_analysis: Optional[PolicyAnalysisResponse] = None

# Bill Checker Pydantic Models
class BillUploadResponse(BaseModel):
    """Response model for bill upload"""
    success: bool
    bill_id: str
    filename: str
    file_size: int
    upload_timestamp: str
    message: str

class BillAnalysisRequest(BaseModel):
    """Request model for bill analysis"""
    bill_id: str
    policy_id: Optional[str] = None
    include_dispute_recommendations: bool = True

class BillSummary(BaseModel):
    """Structured bill summary"""
    provider_name: str
    patient_name: Optional[str] = None
    service_date: Optional[str] = None
    total_charges: Optional[float] = None
    services_provided: List[str] = []
    billing_codes: List[str] = []

class CoverageAnalysis(BaseModel):
    """Insurance coverage analysis"""
    covered_services: List[str] = []
    non_covered_services: List[str] = []
    deductible_applied: Optional[float] = None
    copay_amount: Optional[float] = None
    coinsurance_percentage: Optional[float] = None

class FinancialBreakdown(BaseModel):
    """Financial responsibility breakdown"""
    total_charges: float
    insurance_payment: float
    patient_responsibility: float
    deductible_amount: float = 0.0
    copay_amount: float = 0.0
    coinsurance_amount: float = 0.0

class DisputeRecommendation(BaseModel):
    """Dispute recommendation"""
    issue_type: str
    description: str
    recommended_action: str
    priority: str  # "high", "medium", "low"

class BillAnalysisResponse(BaseModel):
    """Complete bill analysis response"""
    analysis_id: str
    bill_summary: BillSummary
    coverage_analysis: CoverageAnalysis
    financial_breakdown: FinancialBreakdown
    dispute_recommendations: List[DisputeRecommendation] = []
    confidence_score: float
    analysis_timestamp: str
    processing_time_ms: int

class BillAnalysisHistoryItem(BaseModel):
    """Bill analysis history item"""
    analysis_id: str
    bill_filename: str
    analysis_date: str
    total_charges: Optional[float] = None
    patient_responsibility: Optional[float] = None
    confidence_score: float

class BillAnalysisHistory(BaseModel):
    """Bill analysis history response"""
    analyses: List[BillAnalysisHistoryItem]
    total_count: int

class DocumentSummaryResponse(BaseModel):
    """Response model for document summary"""
    summary: str
    document_type: str
    filename: str
    processing_time_ms: Optional[int] = None

# Global components
chat_model: Optional[ChatGoogleGenerativeAI] = None
embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
vector_store: Optional[FAISS] = None
retriever = None
conversational_rag_chain = None
bill_analysis_service = None

# Database setup
DB_PATH = "langchain_heal.db"

def init_database():
    """Initialize SQLite database for session management and bill checker"""
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
    
    # Enhanced documents table with bill checker support
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            raw_file_path TEXT,
            document_type TEXT DEFAULT 'policy',
            chunks_created INTEGER,
            upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_size INTEGER,
            mime_type TEXT
        )
    """)
    
    # Bill analyses table for tracking bill checker results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bill_analyses (
            id TEXT PRIMARY KEY,
            bill_document_id TEXT NOT NULL,
            policy_document_id TEXT,
            analysis_result TEXT NOT NULL,
            analysis_summary TEXT,
            patient_responsibility REAL,
            insurance_payment REAL,
            potential_disputes TEXT,
            confidence_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (bill_document_id) REFERENCES documents(id),
            FOREIGN KEY (policy_document_id) REFERENCES documents(id)
        )
    """)
    
    # Policy analysis results table (matching main.py)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS policies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            summary_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Add indexes for performance
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_type 
        ON documents(document_type)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_bill_analyses_created 
        ON bill_analyses(created_at DESC)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_policies_created 
        ON policies(created_at DESC)
    """)
    
    conn.commit()
    conn.close()
    logger.info("📊 Database initialized with bill checker support and policy analysis")

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

# Policy Analysis Functions (matching main.py functionality)
async def analyze_insurance_policy_langchain(
    file_data: bytes, 
    filename: str, 
    content_type: str
) -> PolicyAnalysisResponse:
    """
    Analyze insurance policy document and extract key information using Gemini
    Similar to main.py's analyze_insurance_policy but integrated with LangChain
    """
    try:
        logger.info(f"🔍 Starting policy analysis for {filename}")
        
        # Check if AI is available
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("⚠️ AI not available, returning mock data")
            return PolicyAnalysisResponse(
                deductible="$1,000 (mock)",
                out_of_pocket_max="$5,000 (mock)",
                copay="$25 (mock)",
                confidence_score=0.5,
                additional_info={"note": "Mock data - configure GEMINI_API_KEY for real analysis"}
            )
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Prepare the analysis prompt
        analysis_prompt = """
        You are an expert insurance policy analyst. Analyze the provided insurance document and extract the following information with high accuracy:

        1. **Deductible**: The amount the policyholder must pay before insurance coverage begins
        2. **Out-of-Pocket Maximum**: The maximum amount the policyholder will pay in a year
        3. **Copay**: The fixed amount paid for covered services

        **Instructions:**
        - Extract exact amounts with currency symbols when available
        - If information is not clearly stated, use "Not found"
        - Be precise and conservative in your extraction
        - Look for terms like "deductible", "out-of-pocket max", "copay", "copayment"
        - Consider both individual and family amounts if present

        **Return your response as a JSON object with these exact keys:**
        {
            "deductible": "extracted deductible amount or 'Not found'",
            "out_of_pocket_max": "extracted out-of-pocket maximum or 'Not found'",
            "copay": "extracted copay amount or 'Not found'"
        }

        **Document to analyze:**
        """
        
        # Handle different document types (using same models as main.py)
        if content_type.startswith("image/"):
            logger.info("🖼️ Processing image document with OCR")
            image = Image.open(io.BytesIO(file_data))
            logger.info(f"📸 Image size: {image.size}, Mode: {image.mode}")
            
            model = genai.GenerativeModel('gemini-2.5-flash')  # Match main.py
            logger.info("🤖 Using Gemini 2.5 Flash model for image analysis")
            response = model.generate_content([analysis_prompt, image])
            
        elif content_type == "application/pdf":
            logger.info("📑 Processing PDF document")
            extracted_text = _extract_pdf_text(file_data)
            logger.info(f"📖 Extracted text length: {len(extracted_text)} chars")
            
            full_prompt = f"{analysis_prompt}\n\nExtracted Text:\n{extracted_text}"
            model = genai.GenerativeModel('gemini-2.5-pro')  # Match main.py
            logger.info("🤖 Using Gemini 2.5 Pro model for PDF text analysis")
            response = model.generate_content(full_prompt)
            
        else:
            raise ValueError(f"Unsupported document type: {content_type}")
        
        # Parse the response with detailed logging (matching main.py)
        logger.info(f"⚡ Gemini response received")
        logger.info(f"📄 Response length: {len(response.text)} chars")
        logger.debug(f"📄 Raw response preview: {response.text[:200]}...")
        
        logger.info("🔧 Parsing Gemini response for structured data")
        policy_analysis = _parse_analysis_response(response.text)
        
        logger.info(f"✅ Successfully analyzed policy: {filename}")
        logger.info(f"📋 Extracted - Deductible: {policy_analysis.deductible}, Out-of-pocket: {policy_analysis.out_of_pocket_max}, Copay: {policy_analysis.copay}")
        logger.info(f"🎯 Confidence score: {policy_analysis.confidence_score}")
        
        return policy_analysis
        
    except Exception as e:
        logger.error(f"❌ Error analyzing policy {filename}: {e}")
        logger.exception("Full error traceback:")
        return PolicyAnalysisResponse(
            deductible="Analysis failed",
            out_of_pocket_max="Analysis failed",
            copay="Analysis failed",
            confidence_score=0.0,
            additional_info={"error": str(e), "error_type": type(e).__name__}
        )

async def summarize_policy_document_langchain(
    file_data: bytes,
    filename: str,
    content_type: str
) -> DocumentSummaryResponse:
    """
    Generate a comprehensive summary of the insurance policy document
    Similar to main.py's summarize_policy_document
    """
    start_time = time.time()
    
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return DocumentSummaryResponse(
                summary="AI summarization not available. Please configure GEMINI_API_KEY.",
                document_type=content_type,
                filename=filename,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
        
        genai.configure(api_key=api_key)
        
        summary_prompt = """
        Provide a comprehensive summary of this insurance policy document. Include:
        
        1. **Policy Type**: What type of insurance (health, auto, home, etc.)
        2. **Coverage Overview**: Main benefits and coverage areas
        3. **Key Terms**: Important policy terms and conditions
        4. **Limitations**: Notable exclusions or limitations
        5. **Important Dates**: Policy periods, renewal dates
        6. **Contact Information**: Insurance company details
        
        Format your response as a clear, organized summary that a policyholder can easily understand.
        """
        
        if content_type.startswith("image/"):
            image = Image.open(io.BytesIO(file_data))
            model = genai.GenerativeModel('gemini-2.5-flash')  # Match main.py
            response = model.generate_content([summary_prompt, image])
            
        else:  # PDF
            extracted_text = _extract_pdf_text(file_data)
            full_prompt = f"{summary_prompt}\n\nDocument Content:\n{extracted_text}"
            model = genai.GenerativeModel('gemini-2.5-pro')  # Match main.py
            response = model.generate_content(full_prompt)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return DocumentSummaryResponse(
            summary=response.text,
            document_type=content_type,
            filename=filename,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error summarizing document {filename}: {e}")
        processing_time = int((time.time() - start_time) * 1000)
        return DocumentSummaryResponse(
            summary=f"Failed to generate summary: {str(e)}",
            document_type=content_type,
            filename=filename,
            processing_time_ms=processing_time
        )

def _parse_analysis_response(response_text: str) -> PolicyAnalysisResponse:
    """Parse AI response and create structured PolicyAnalysisResponse"""
    try:
        # Try to extract JSON from response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            json_text = response_text[json_start:json_end].strip()
        else:
            json_text = response_text.strip()
        
        # Parse JSON
        parsed_data = json.loads(json_text)
        
        # Create structured output
        return PolicyAnalysisResponse(
            deductible=parsed_data.get("deductible", "Not found"),
            out_of_pocket_max=parsed_data.get("out_of_pocket_max", "Not found"),
            copay=parsed_data.get("copay", "Not found"),
            confidence_score=0.9,
            additional_info={"parsing_method": "json_extraction"}
        )
        
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON response, using fallback extraction")
        return _extract_values_from_text(response_text)
    except Exception as e:
        logger.error(f"Error parsing response: {e}")
        return PolicyAnalysisResponse(
            deductible="Parsing failed",
            out_of_pocket_max="Parsing failed",
            copay="Parsing failed",
            confidence_score=0.0,
            additional_info={"error": str(e)}
        )

def _extract_values_from_text(text: str) -> PolicyAnalysisResponse:
    """Fallback method to extract values from unstructured text"""
    import re
    
    # Simple patterns to extract common values
    deductible_pattern = r'deductible["\s:]*([^,\n}]+)'
    oop_pattern = r'out[_-]?of[_-]?pocket[_-]?max["\s:]*([^,\n}]+)'
    copay_pattern = r'copay["\s:]*([^,\n}]+)'
    
    deductible = "Not found"
    out_of_pocket_max = "Not found"
    copay = "Not found"
    
    # Try to find values using regex
    deductible_match = re.search(deductible_pattern, text, re.IGNORECASE)
    if deductible_match:
        deductible = deductible_match.group(1).strip().strip('"').strip("'")
    
    oop_match = re.search(oop_pattern, text, re.IGNORECASE)
    if oop_match:
        out_of_pocket_max = oop_match.group(1).strip().strip('"').strip("'")
    
    copay_match = re.search(copay_pattern, text, re.IGNORECASE)
    if copay_match:
        copay = copay_match.group(1).strip().strip('"').strip("'")
    
    return PolicyAnalysisResponse(
        deductible=deductible,
        out_of_pocket_max=out_of_pocket_max,
        copay=copay,
        confidence_score=0.6,
        additional_info={"parsing_method": "text_extraction"}
    )

def _extract_pdf_text(pdf_data: bytes) -> str:
    """Extract text content from PDF bytes"""
    try:
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        text_content = ""
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text_content += f"\n--- Page {page_num + 1} ---\n"
            text_content += page.get_text()
        
        pdf_document.close()
        return text_content
        
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        return f"Failed to extract text from PDF: {str(e)}"

def save_policy_analysis_to_database(policy_analysis: PolicyAnalysisResponse) -> None:
    """Save policy analysis result to database (matching main.py functionality)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Convert to dict format matching main.py
        analysis_data = {
            "deductible": policy_analysis.deductible,
            "out_of_pocket_max": policy_analysis.out_of_pocket_max,
            "copay": policy_analysis.copay,
            "confidence_score": policy_analysis.confidence_score,
            "additional_info": policy_analysis.additional_info or {}
        }
        
        cursor.execute(
            "INSERT INTO policies (summary_json) VALUES (?)",
            (json.dumps(analysis_data),)
        )
        conn.commit()
        conn.close()
        logger.info("💾 Policy analysis saved to database successfully")
    except Exception as e:
        logger.error(f"❌ Error saving policy analysis to database: {e}")
        # Don't raise exception here, as the analysis was successful

@app.on_event("startup")
async def startup_event():
    """Initialize LangChain components and bill analysis service"""
    global chat_model, embeddings, vector_store, retriever, conversational_rag_chain, bill_analysis_service
    
    try:
        # Initialize database
        init_database()
        
        # Check API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("❌ GEMINI_API_KEY not found")
            logger.warning("⚠️ Continuing with limited functionality")
            # Don't return, continue with limited functionality
        
        if api_key:
            # Initialize LangChain components (using same models as main.py)
            chat_model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",  # Match main.py
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
            
            # Initialize bill analysis service
            try:
                from services.bill_analysis_service import BillAnalysisService
                bill_analysis_service = BillAnalysisService(DB_PATH)
                logger.info("🏥 Bill analysis service initialized")
            except Exception as bill_error:
                logger.error(f"⚠️ Bill analysis service failed to initialize: {bill_error}")
                bill_analysis_service = None
            
            logger.info("🚀 LangChain HEAL system initialized successfully")
            logger.info(f"🤖 Model: {chat_model.model}")
            logger.info(f"🧠 Embeddings: {embeddings.model}")
        else:
            logger.warning("⚠️ AI components not initialized - limited functionality")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.warning("⚠️ Continuing with limited functionality")
        # Don't raise, continue with limited functionality

@app.get("/health")
async def health_check():
    """Health check endpoint (matching main.py format)"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        
        # Check database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM policies")
        policy_count = cursor.fetchone()[0]
        conn.close()
        database_status = "connected"
        
        # Check AI status
        ai_status = "available" if api_key and all([chat_model, embeddings]) else "unavailable"
        
        # Check model availability
        model_status = "available" if chat_model else "unavailable"
        
        # Overall status
        overall_status = "healthy" if ai_status == "available" and database_status == "connected" else "degraded"
        
        return {
            "status": overall_status,
            "system": "langchain-heal",
            "version": "3.0.0", 
            "framework": "LangChain + Direct Gemini",
            "ai_status": ai_status,
            "model_status": model_status,
            "database_status": database_status,
            "api_key_configured": bool(api_key),
            "components_ready": all([chat_model, embeddings, vector_store, conversational_rag_chain]),
            "models": {
                "chat": chat_model.model if chat_model else None,
                "embeddings": embeddings.model if embeddings else None
            },
            "policy_count": policy_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "system": "langchain-heal",
            "version": "3.0.0",
            "framework": "LangChain + Direct Gemini", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def store_file_permanently(content: bytes, filename: str, document_type: str = "policy") -> str:
    """Store file permanently for multimodal analysis"""
    
    # Create uploads directory if it doesn't exist
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_extension = Path(filename).suffix
    stored_filename = f"{file_id}_{document_type}{file_extension}"
    file_path = uploads_dir / stored_filename
    
    # Write file
    with open(file_path, 'wb') as f:
        f.write(content)
    
    logger.info(f"💾 Stored {document_type} file: {file_path}")
    return str(file_path)

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process insurance policy document using LangChain"""
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
        
        # Read file content
        content = await file.read()
        
        # Store file permanently for multimodal analysis
        raw_file_path = store_file_permanently(content, file.filename, "policy")
        
        # Also create temporary file for LangChain processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
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
            
            # Perform policy analysis in parallel with RAG processing
            policy_analysis = None
            try:
                logger.info("🔍 Starting policy analysis...")
                policy_analysis = await analyze_insurance_policy_langchain(
                    content, file.filename, file.content_type
                )
                logger.info(f"✅ Policy analysis completed: {policy_analysis.deductible}, {policy_analysis.out_of_pocket_max}, {policy_analysis.copay}")
                
                # Save policy analysis to database (matching main.py)
                save_policy_analysis_to_database(policy_analysis)
                
            except Exception as analysis_error:
                logger.error(f"❌ Policy analysis failed: {analysis_error}")
                # Continue without policy analysis
            
            # Store document metadata with enhanced schema
            doc_id = str(uuid.uuid4())
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO documents (
                    id, filename, file_path, raw_file_path, document_type,
                    chunks_created, file_size, mime_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id, file.filename, tmp_file_path, raw_file_path, "policy",
                len(split_docs), len(content), file.content_type
            ))
            conn.commit()
            conn.close()
            
            processing_time = int((time.time() - start_time) * 1000)
            
            logger.info(f"✅ Successfully processed policy document: {file.filename}")
            
            return DocumentUploadResponse(
                success=True,
                document_id=doc_id,
                chunks_created=len(split_docs),
                processing_time_ms=processing_time,
                message=f"Successfully processed {file.filename} using LangChain RAG + Policy Analysis",
                policy_analysis=policy_analysis
            )
            
        finally:
            # Clean up temp file (but keep the permanent raw file)
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"❌ Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/summarize", response_model=DocumentSummaryResponse)
async def summarize_document(file: UploadFile = File(...)):
    """
    Generate a comprehensive summary of an insurance document
    Similar to main.py's summarize endpoint
    """
    try:
        # Validate file type
        allowed_types = ['application/pdf', 'image/jpeg', 'image/png']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}"
            )
        
        # Read file content
        content = await file.read()
        
        logger.info(f"📄 Summarizing document: {file.filename}")
        
        # Use LangChain-integrated summarization
        summary_result = await summarize_policy_document_langchain(
            content, file.filename, file.content_type
        )
        
        logger.info(f"✅ Successfully summarized document: {file.filename}")
        return summary_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Document summarization failed: {e}")
        return DocumentSummaryResponse(
            summary=f"Failed to generate summary: {str(e)}",
            document_type=file.content_type or "unknown",
            filename=file.filename or "unknown",
            processing_time_ms=0
        )

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

# Bill Checker Endpoints
@app.post("/bill-checker/upload", response_model=BillUploadResponse)
async def upload_medical_bill(file: UploadFile = File(...)):
    """Upload medical bill for analysis"""
    
    start_time = time.time()
    
    try:
        # Validate file type
        allowed_types = ['application/pdf', 'image/jpeg', 'image/png']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}"
            )
        
        # Read file content
        content = await file.read()
        
        # Store file permanently for multimodal analysis
        raw_file_path = store_file_permanently(content, file.filename, "bill")
        
        # Store bill metadata in database
        bill_id = str(uuid.uuid4())
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO documents (
                id, filename, file_path, raw_file_path, document_type,
                file_size, mime_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            bill_id, file.filename, "", raw_file_path, "bill",
            len(content), file.content_type
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Medical bill uploaded: {file.filename}")
        
        return BillUploadResponse(
            success=True,
            bill_id=bill_id,
            filename=file.filename,
            file_size=len(content),
            upload_timestamp=datetime.now().isoformat(),
            message=f"Successfully uploaded medical bill: {file.filename}"
        )
        
    except Exception as e:
        logger.error(f"❌ Bill upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/bill-checker/analyze", response_model=BillAnalysisResponse)
async def analyze_medical_bill(request: BillAnalysisRequest):
    """Analyze medical bill against insurance policy"""
    
    if not bill_analysis_service:
        raise HTTPException(status_code=500, detail="Bill analysis service not initialized")
    
    try:
        logger.info(f"🔍 Starting bill analysis: {request.bill_id}")
        
        # Perform analysis using the service
        result = await bill_analysis_service.analyze_bill_vs_policy(
            bill_id=request.bill_id,
            policy_id=request.policy_id,
            include_dispute_recommendations=request.include_dispute_recommendations
        )
        
        # For now, return structured response with raw analysis
        # In production, you'd parse the analysis into structured fields
        structured_analysis = result["structured_analysis"]
        
        return BillAnalysisResponse(
            analysis_id=result["analysis_id"],
            bill_summary=BillSummary(
                provider_name="Analysis available in raw format",
                services_provided=["See detailed analysis below"]
            ),
            coverage_analysis=CoverageAnalysis(
                covered_services=["Detailed analysis available"]
            ),
            financial_breakdown=FinancialBreakdown(
                total_charges=0.0,
                insurance_payment=0.0,
                patient_responsibility=0.0
            ),
            dispute_recommendations=[],
            confidence_score=0.85,
            analysis_timestamp=result["timestamp"],
            processing_time_ms=result["processing_time_ms"]
        )
        
    except Exception as e:
        logger.error(f"❌ Bill analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/bill-checker/analysis/{analysis_id}")
async def get_bill_analysis(analysis_id: str):
    """Get specific bill analysis by ID"""
    
    if not bill_analysis_service:
        raise HTTPException(status_code=500, detail="Bill analysis service not initialized")
    
    try:
        analysis = await bill_analysis_service.get_analysis_by_id(analysis_id)
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return {
            "analysis_id": analysis_id,
            "raw_analysis": analysis["raw_analysis"],
            "structured_analysis": analysis["structured_analysis"],
            "confidence_score": analysis["confidence_score"],
            "created_at": analysis["created_at"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/bill-checker/history", response_model=BillAnalysisHistory)
async def get_bill_analysis_history(limit: int = 20):
    """Get history of bill analyses"""
    
    if not bill_analysis_service:
        raise HTTPException(status_code=500, detail="Bill analysis service not initialized")
    
    try:
        history = await bill_analysis_service.get_analysis_history(limit)
        
        return BillAnalysisHistory(
            analyses=[
                BillAnalysisHistoryItem(
                    analysis_id=item["analysis_id"],
                    bill_filename=item["bill_filename"],
                    analysis_date=item["analysis_date"],
                    total_charges=item.get("total_charges"),
                    patient_responsibility=item.get("patient_responsibility"),
                    confidence_score=item["confidence_score"]
                )
                for item in history["analyses"]
            ],
            total_count=history["total_count"]
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

# Additional Debug Endpoints (matching main.py)
@app.post("/debug/test-gemini")
async def debug_test_gemini(request: Dict[str, str]) -> Dict[str, Any]:
    """Debug: Test Gemini API with a simple prompt (matching main.py)"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {
                "success": False,
                "error": "AI services not available",
                "api_key_configured": False
            }
        
        test_prompt = request.get("prompt", "Hello, this is a test. Please respond with 'Test successful'.")
        
        # Configure Gemini and test with Flash model
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')  # Use same model as main.py
        start_time = time.time()
        
        response = model.generate_content(test_prompt)
        end_time = time.time()
        
        return {
            "success": True,
            "prompt": test_prompt,
            "response": response.text,
            "model_used": "gemini-2.5-flash",
            "response_time_ms": int((end_time - start_time) * 1000),
            "api_key_configured": True
        }
        
    except Exception as e:
        logger.error(f"❌ Error testing Gemini: {e}")
        return {
            "success": False,
            "error": str(e),
            "prompt": request.get("prompt", ""),
            "api_key_configured": bool(os.getenv("GEMINI_API_KEY"))
        }

@app.get("/debug/gemini-calls")
async def debug_gemini_calls() -> Dict[str, Any]:
    """Debug: Get information about Gemini configuration (matching main.py)"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        return {
            "api_key_configured": bool(api_key),
            "models_configured": {
                "flash": "gemini-2.5-flash",
                "pro": "gemini-2.5-pro"
            },
            "framework": "LangChain + Direct Gemini",
            "langchain_models": {
                "chat": chat_model.model if chat_model else None,
                "embeddings": embeddings.model if embeddings else None
            },
            "components_ready": all([chat_model, embeddings, vector_store, conversational_rag_chain]),
            "recent_calls": "Check server logs for detailed call information"
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting Gemini debug info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/latest-upload")
async def debug_latest_upload() -> Dict[str, Any]:
    """Debug: Get information about the most recent upload (matching main.py)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get latest document
        cursor.execute("""
            SELECT * FROM documents 
            ORDER BY upload_timestamp DESC 
            LIMIT 1
        """)
        
        latest_doc = cursor.fetchone()
        if not latest_doc:
            return {"message": "No documents found"}
        
        # Convert to dict
        columns = [description[0] for description in cursor.description]
        doc_dict = dict(zip(columns, latest_doc))
        
        # Get latest policy analysis
        cursor.execute("""
            SELECT summary_json FROM policies 
            ORDER BY id DESC 
            LIMIT 1
        """)
        
        policy_result = cursor.fetchone()
        policy_analysis = json.loads(policy_result[0]) if policy_result else None
        
        conn.close()
        
        return {
            "latest_document": doc_dict,
            "policy_analysis": policy_analysis,
            "message": f"Latest upload: {doc_dict.get('filename', 'Unknown')}"
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting latest upload debug info: {e}")
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
