#!/usr/bin/env python3
"""
Full Genkit Implementation for HEAL RAG + Conversational Chat
Implements proper Genkit patterns for RAG and conversation management
"""

import os
import logging
import json
import uuid
import time
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# Genkit imports (when available)
try:
    from genkit.ai import Genkit, Document
    from genkit.plugins.google_genai import (
        GoogleGenAI, 
        gemini_15_flash,
        gemini_15_pro,
        text_embedding_004
    )
    GENKIT_AVAILABLE = True
except ImportError:
    print("⚠️ Genkit not available, using Genkit-style patterns with direct API")
    GENKIT_AVAILABLE = False

# Fallback imports for Genkit-style implementation
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import numpy as np
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key
from dotenv import load_dotenv
load_dotenv()

@dataclass
class ConversationMessage:
    """Genkit-style conversation message"""
    role: str  # 'user' or 'model'
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class RAGDocument:
    """Genkit-style document representation"""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class RetrievalResult:
    """Result from RAG retrieval"""
    documents: List[RAGDocument]
    query: str
    total_found: int
    execution_time_ms: int

@dataclass
class ChatResponse:
    """Genkit-style chat response"""
    message: str
    sources: List[Dict[str, Any]]
    confidence: float
    session_id: str
    processing_time_ms: int
    metadata: Optional[Dict[str, Any]] = None

class GenkitStyleEmbedder:
    """Genkit-style embedder implementation"""
    
    def __init__(self, model_name: str = "text-embedding-004"):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable required")
        
        genai.configure(api_key=self.api_key)
        logger.info(f"🧠 Initialized Genkit-style embedder with {model_name}")
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents following Genkit patterns"""
        embeddings = []
        
        for text in texts:
            try:
                result = genai.embed_content(
                    model=f"models/{self.model_name}",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                logger.error(f"❌ Embedding failed for text: {e}")
                # Fallback to zero vector
                embeddings.append([0.0] * 768)
        
        return embeddings
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed query text following Genkit patterns"""
        try:
            result = genai.embed_content(
                model=f"models/{self.model_name}",
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"❌ Query embedding failed: {e}")
            return [0.0] * 768

class GenkitStyleRetriever:
    """Genkit-style document retriever"""
    
    def __init__(self, embedder: GenkitStyleEmbedder, db_path: str = "genkit_rag.db"):
        self.embedder = embedder
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize Genkit-style RAG database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables following Genkit patterns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS genkit_documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS genkit_chat_sessions (
                session_id TEXT PRIMARY KEY,
                document_context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS genkit_chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES genkit_chat_sessions(session_id)
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("🗄️ Genkit-style database initialized")
    
    async def index_documents(self, documents: List[RAGDocument]) -> bool:
        """Index documents following Genkit patterns"""
        try:
            # Extract content for embedding
            contents = [doc.content for doc in documents]
            
            # Generate embeddings
            embeddings = await self.embedder.embed_documents(contents)
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for doc, embedding in zip(documents, embeddings):
                doc_id = str(uuid.uuid4())
                embedding_blob = json.dumps(embedding).encode()
                
                cursor.execute("""
                    INSERT INTO genkit_documents (id, content, metadata, embedding)
                    VALUES (?, ?, ?, ?)
                """, (
                    doc_id,
                    doc.content,
                    json.dumps(doc.metadata),
                    embedding_blob
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Indexed {len(documents)} documents in Genkit-style store")
            return True
            
        except Exception as e:
            logger.error(f"❌ Indexing failed: {e}")
            return False
    
    async def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> RetrievalResult:
        """Retrieve documents following Genkit patterns"""
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = await self.embedder.embed_query(query)
            
            # Get all documents with embeddings
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, content, metadata, embedding 
                FROM genkit_documents 
                WHERE embedding IS NOT NULL
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return RetrievalResult(
                    documents=[],
                    query=query,
                    total_found=0,
                    execution_time_ms=int((time.time() - start_time) * 1000)
                )
            
            # Calculate similarities
            retrieved_docs = []
            query_vec = np.array(query_embedding).reshape(1, -1)
            
            for row in rows:
                doc_id, content, metadata_str, embedding_blob = row
                
                # Decode embedding
                embedding = json.loads(embedding_blob.decode())
                doc_vec = np.array(embedding).reshape(1, -1)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(query_vec, doc_vec)[0][0]
                
                if similarity >= similarity_threshold:
                    metadata = json.loads(metadata_str)
                    metadata['similarity_score'] = float(similarity)
                    metadata['document_id'] = doc_id
                    
                    retrieved_docs.append(RAGDocument(
                        content=content,
                        metadata=metadata,
                        embedding=embedding
                    ))
            
            # Sort by similarity and take top k
            retrieved_docs.sort(key=lambda x: x.metadata['similarity_score'], reverse=True)
            top_docs = retrieved_docs[:top_k]
            
            execution_time = int((time.time() - start_time) * 1000)
            
            logger.info(f"🔍 Retrieved {len(top_docs)} documents in {execution_time}ms")
            
            return RetrievalResult(
                documents=top_docs,
                query=query,
                total_found=len(retrieved_docs),
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            return RetrievalResult(
                documents=[],
                query=query,
                total_found=0,
                execution_time_ms=int((time.time() - start_time) * 1000)
            )

class GenkitStyleGenerator:
    """Genkit-style response generator"""
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable required")
        
        genai.configure(api_key=self.api_key)
        
        # Configure safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            safety_settings=self.safety_settings
        )
        
        logger.info(f"🤖 Initialized Genkit-style generator with {model_name}")
    
    async def generate(
        self,
        prompt: str,
        documents: List[RAGDocument] = None,
        conversation_history: List[ConversationMessage] = None,
        system_prompt: str = None
    ) -> str:
        """Generate response following Genkit patterns"""
        try:
            # Build full prompt following Genkit patterns
            full_prompt = self._build_genkit_prompt(
                prompt, documents, conversation_history, system_prompt
            )
            
            # Generate response
            response = self.model.generate_content(full_prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."
    
    def _build_genkit_prompt(
        self,
        user_prompt: str,
        documents: List[RAGDocument] = None,
        conversation_history: List[ConversationMessage] = None,
        system_prompt: str = None
    ) -> str:
        """Build prompt following Genkit patterns"""
        
        parts = []
        
        # System prompt
        if system_prompt:
            parts.append(f"**System Instructions:**\n{system_prompt}\n")
        
        # Conversation history
        if conversation_history:
            parts.append("**Conversation History:**")
            for msg in conversation_history[-6:]:  # Last 6 messages
                role_name = "User" if msg.role == "user" else "Assistant"
                parts.append(f"{role_name}: {msg.content}")
            parts.append("")
        
        # Retrieved documents
        if documents:
            parts.append("**Retrieved Context:**")
            for i, doc in enumerate(documents, 1):
                similarity = doc.metadata.get('similarity_score', 0)
                source = doc.metadata.get('source_document', 'Unknown')
                parts.append(f"[Document {i} - {source} (similarity: {similarity:.3f})]")
                parts.append(doc.content)
                parts.append("")
        
        # Current user prompt
        parts.append(f"**Current Question:**\n{user_prompt}")
        parts.append("\n**Your Response:**")
        
        return "\n".join(parts)

class GenkitRAGChat:
    """Main Genkit-style RAG + Chat implementation"""
    
    def __init__(self, db_path: str = "genkit_rag.db"):
        self.embedder = GenkitStyleEmbedder()
        self.retriever = GenkitStyleRetriever(self.embedder, db_path)
        self.generator = GenkitStyleGenerator()
        self.db_path = db_path
        
        logger.info("🚀 Genkit-style RAG Chat initialized")
    
    async def process_document(
        self,
        file_path: str,
        chunk_size: int = 750,
        chunk_overlap: int = 100
    ) -> bool:
        """Process document following Genkit patterns"""
        try:
            logger.info(f"📄 Processing document: {file_path}")
            
            # Extract text based on file type
            if file_path.lower().endswith('.pdf'):
                text = self._extract_pdf_text(file_path)
            elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                text = self._extract_image_text(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            # Chunk text following Genkit patterns
            chunks = self._chunk_text(text, chunk_size, chunk_overlap)
            
            # Create RAG documents
            documents = []
            for i, chunk in enumerate(chunks):
                doc = RAGDocument(
                    content=chunk,
                    metadata={
                        'source_file': file_path,
                        'chunk_index': i,
                        'source_document': Path(file_path).name,
                        'chunk_type': 'text'
                    }
                )
                documents.append(doc)
            
            # Index documents
            success = await self.retriever.index_documents(documents)
            
            if success:
                logger.info(f"✅ Successfully processed {len(documents)} chunks from {file_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            return False
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF"""
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    
    def _extract_image_text(self, file_path: str) -> str:
        """Extract text from image using OCR"""
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)
    
    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Chunk text following Genkit patterns"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_words = words[i:i + chunk_size]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    async def create_chat_session(self, document_context: List[str] = None) -> str:
        """Create chat session following Genkit patterns"""
        session_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO genkit_chat_sessions (session_id, document_context)
            VALUES (?, ?)
        """, (
            session_id,
            json.dumps(document_context) if document_context else None
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"💬 Created Genkit chat session: {session_id}")
        return session_id
    
    async def chat(
        self,
        session_id: str,
        message: str,
        top_k: int = 5,
        system_prompt: str = None
    ) -> ChatResponse:
        """Chat following Genkit patterns"""
        start_time = time.time()
        
        try:
            # Get conversation history
            history = self._get_conversation_history(session_id)
            
            # Retrieve relevant documents
            retrieval_result = await self.retriever.retrieve(
                query=message,
                top_k=top_k,
                similarity_threshold=0.3
            )
            
            # Default system prompt for insurance
            if not system_prompt:
                system_prompt = """You are HEAL, an expert insurance policy assistant. 
Use the provided context to answer questions about insurance policies.
Maintain conversational flow by referencing previous discussions when relevant.
If information is not in the context, clearly state that you don't have that information."""
            
            # Generate response
            response = await self.generator.generate(
                prompt=message,
                documents=retrieval_result.documents,
                conversation_history=history,
                system_prompt=system_prompt
            )
            
            # Store messages
            self._store_conversation_message(session_id, "user", message)
            self._store_conversation_message(session_id, "model", response)
            
            # Calculate confidence
            confidence = self._calculate_confidence(retrieval_result.documents)
            
            # Prepare sources
            sources = [
                {
                    'document': doc.metadata.get('source_document', 'Unknown'),
                    'similarity': doc.metadata.get('similarity_score', 0),
                    'content_preview': doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
                }
                for doc in retrieval_result.documents
            ]
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return ChatResponse(
                message=response,
                sources=sources,
                confidence=confidence,
                session_id=session_id,
                processing_time_ms=processing_time,
                metadata={
                    'retrieval_time_ms': retrieval_result.execution_time_ms,
                    'documents_found': retrieval_result.total_found
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Chat failed: {e}")
            return ChatResponse(
                message="I apologize, but I encountered an error processing your message.",
                sources=[],
                confidence=0.0,
                session_id=session_id,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
    
    def _get_conversation_history(self, session_id: str) -> List[ConversationMessage]:
        """Get conversation history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT role, content, timestamp, metadata
            FROM genkit_chat_messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
            LIMIT 20
        """, (session_id,))
        
        messages = []
        for row in cursor.fetchall():
            role, content, timestamp_str, metadata_str = row
            
            messages.append(ConversationMessage(
                role=role,
                content=content,
                timestamp=datetime.fromisoformat(timestamp_str),
                metadata=json.loads(metadata_str) if metadata_str else None
            ))
        
        conn.close()
        return messages
    
    def _store_conversation_message(self, session_id: str, role: str, content: str):
        """Store conversation message"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO genkit_chat_messages (session_id, role, content)
            VALUES (?, ?, ?)
        """, (session_id, role, content))
        
        conn.commit()
        conn.close()
    
    def _calculate_confidence(self, documents: List[RAGDocument]) -> float:
        """Calculate response confidence"""
        if not documents:
            return 0.0
        
        # Average similarity score
        similarities = [doc.metadata.get('similarity_score', 0) for doc in documents]
        return sum(similarities) / len(similarities)

# Demo/Test functionality
async def demo_genkit_rag_chat():
    """Demo the Genkit-style RAG Chat system"""
    print("🚀 GENKIT-STYLE RAG CHAT DEMO")
    print("=" * 50)
    
    # Initialize system
    rag_chat = GenkitRAGChat()
    
    # Demo document processing (you can replace with actual file)
    demo_doc = RAGDocument(
        content="""INSURANCE POLICY SUMMARY
        
        Deductible: $500 individual, $1000 family
        Out-of-pocket maximum: $2000 individual, $4000 family
        Copay: $25 primary care, $50 specialist
        
        Dental Coverage:
        - Preventive care: 100% covered
        - Basic procedures: 80% covered after deductible
        - Major procedures: 50% covered after deductible
        
        Vision Coverage:
        - Annual eye exam: $25 copay
        - Frames: $150 allowance every 24 months
        """,
        metadata={
            'source_document': 'demo_policy.pdf',
            'chunk_index': 0,
            'source_file': 'demo_policy.pdf'
        }
    )
    
    # Index demo document
    print("📄 Indexing demo document...")
    await rag_chat.retriever.index_documents([demo_doc])
    
    # Create chat session
    session_id = await rag_chat.create_chat_session()
    print(f"💬 Created session: {session_id}")
    
    # Demo conversation
    questions = [
        "What is my deductible?",
        "What about dental coverage?",
        "I thought dental was only covered for preventive care?"
    ]
    
    for question in questions:
        print(f"\n👤 User: {question}")
        
        response = await rag_chat.chat(session_id, question)
        
        print(f"🤖 HEAL: {response.message}")
        print(f"📊 Confidence: {response.confidence:.2f}")
        print(f"📚 Sources: {len(response.sources)}")
        print(f"⏱️  Time: {response.processing_time_ms}ms")

if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_genkit_rag_chat())
