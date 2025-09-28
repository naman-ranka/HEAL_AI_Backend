"""
RAG-powered chatbot for insurance policy questions
"""

import logging
import json
import uuid
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .retriever import RAGRetriever, RetrievedChunk
from database import get_db_connection
from ai.genkit_config import ai_config

logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Represents a chat message"""
    content: str
    message_type: str  # 'user' or 'assistant'
    timestamp: datetime
    relevant_chunks: Optional[List[int]] = None
    confidence_score: Optional[float] = None

@dataclass
class ChatResponse:
    """Response from the chatbot"""
    message: str
    sources: List[Dict[str, Any]]
    confidence: float
    session_id: str
    processing_time_ms: int
    model_used: str
    tokens_used: Optional[int] = None

class InsuranceChatbot:
    """RAG-powered chatbot for insurance policy questions"""
    
    def __init__(self):
        self.retriever = RAGRetriever()
    
    async def create_session(self, document_ids: Optional[List[int]] = None) -> str:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_sessions (
                    session_id, document_context
                ) VALUES (?, ?)
            """, (
                session_id,
                json.dumps(document_ids) if document_ids else None
            ))
            conn.commit()
            
            logger.info(f"Created chat session {session_id}")
            return session_id
            
        finally:
            conn.close()
    
    async def chat(
        self, 
        message: str, 
        session_id: str,
        context_limit: int = 3
    ) -> ChatResponse:
        """
        Process a chat message and return AI response with RAG context
        
        Args:
            message: User's message
            session_id: Chat session ID
            context_limit: Number of relevant chunks to retrieve
            
        Returns:
            ChatResponse with AI-generated answer
        """
        start_time = time.time()
        
        try:
            # Get session context (document IDs)
            session_info = self._get_session_info(session_id)
            if not session_info:
                raise ValueError(f"Session {session_id} not found")
            
            document_ids = json.loads(session_info['document_context']) if session_info['document_context'] else None
            
            # Retrieve relevant context using RAG
            retrieval_result = await self.retriever.retrieve(
                query=message,
                top_k=context_limit,
                document_ids=document_ids
            )
            
            # Build context from retrieved chunks
            context = self._build_context_from_chunks(retrieval_result.chunks)
            
            # Generate AI response
            ai_response = await self._generate_ai_response(message, context, retrieval_result.chunks)
            
            # Calculate confidence based on retrieval quality
            confidence = self._calculate_response_confidence(retrieval_result.chunks)
            
            # Prepare sources information
            sources = self._prepare_sources(retrieval_result.chunks)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Save conversation to database
            chunk_ids = [chunk.chunk_id for chunk in retrieval_result.chunks]
            self._save_chat_message(session_id, message, 'user', None, None)
            self._save_chat_message(
                session_id, 
                ai_response, 
                'assistant', 
                chunk_ids, 
                confidence,
                processing_time
            )
            
            # Update session activity
            self._update_session_activity(session_id)
            
            response = ChatResponse(
                message=ai_response,
                sources=sources,
                confidence=confidence,
                session_id=session_id,
                processing_time_ms=processing_time,
                model_used='gemini-2.5-pro',
                tokens_used=None  # Would be available in real Genkit implementation
            )
            
            logger.info(f"Generated chat response for session {session_id} in {processing_time}ms")
            return response
            
        except Exception as e:
            logger.error(f"Error in chat for session {session_id}: {e}")
            processing_time = int((time.time() - start_time) * 1000)
            
            # Return error response
            return ChatResponse(
                message="I apologize, but I'm having trouble processing your question right now. Please try again.",
                sources=[],
                confidence=0.0,
                session_id=session_id,
                processing_time_ms=processing_time,
                model_used='error'
            )
    
    def _build_context_from_chunks(self, chunks: List[RetrievedChunk]) -> str:
        """Build context string from retrieved chunks"""
        if not chunks:
            return "No relevant policy information found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Source {i} - {chunk.source_document}]:\n{chunk.text}")
        
        return "\n\n".join(context_parts)
    
    async def _generate_ai_response(
        self, 
        user_message: str, 
        context: str,
        chunks: List[RetrievedChunk]
    ) -> str:
        """Generate AI response using context"""
        
        if not ai_config.is_available():
            return self._generate_fallback_response(user_message, chunks)
        
        try:
            # Build the prompt with context
            prompt = self._build_rag_prompt(user_message, context)
            
            # Generate response using Gemini
            model = ai_config.get_model('pro')
            response = model.generate_content(prompt)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return self._generate_fallback_response(user_message, chunks)
    
    def _build_rag_prompt(self, user_message: str, context: str) -> str:
        """Build the RAG prompt with context"""
        return f"""You are HEAL, an expert insurance policy assistant. Your role is to help users understand their insurance policies based on the provided policy information.

**Guidelines:**
1. Answer questions based ONLY on the provided policy context
2. If information is not in the policy context, clearly state "I don't have that information in your policy documents"
3. Be helpful, clear, and concise
4. Use simple language that policyholders can understand
5. Always be accurate and conservative in your responses
6. If you're uncertain, say so and suggest contacting the insurance company

**Policy Information:**
{context}

**User Question:**
{user_message}

**Your Response:**
Based on the policy information provided, """
    
    def _generate_fallback_response(self, user_message: str, chunks: List[RetrievedChunk]) -> str:
        """Generate a fallback response when AI is not available"""
        if not chunks:
            return "I don't have enough information in your policy documents to answer that question. Please contact your insurance company for specific details."
        
        # Simple keyword-based response
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ['deductible', 'deduct']):
            return "Based on your policy documents, I found information about deductibles. Please review the policy details or contact your insurance company for specific amounts."
        elif any(word in message_lower for word in ['copay', 'co-pay', 'copayment']):
            return "I found information about copayments in your policy. Please check the specific policy sections or contact your insurance company for exact amounts."
        elif any(word in message_lower for word in ['out of pocket', 'maximum', 'limit']):
            return "Your policy contains information about out-of-pocket maximums. Please review the policy details or contact your insurance company for specific limits."
        else:
            return f"I found {len(chunks)} relevant sections in your policy documents. Please review the policy details or contact your insurance company for specific information about your question."
    
    def _calculate_response_confidence(self, chunks: List[RetrievedChunk]) -> float:
        """Calculate confidence score based on retrieval quality"""
        if not chunks:
            return 0.0
        
        # Average similarity score of top chunks
        avg_similarity = sum(chunk.similarity_score for chunk in chunks) / len(chunks)
        
        # Boost confidence if we have multiple relevant chunks
        chunk_bonus = min(len(chunks) * 0.1, 0.3)
        
        confidence = min(avg_similarity + chunk_bonus, 1.0)
        return round(confidence, 2)
    
    def _prepare_sources(self, chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        """Prepare source information for response"""
        sources = []
        for chunk in chunks:
            sources.append({
                "document": chunk.source_document,
                "similarity": round(chunk.similarity_score, 3),
                "preview": chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text,
                "chunk_id": chunk.chunk_id
            })
        return sources
    
    def _get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information from database"""
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM chat_sessions WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
            
        finally:
            conn.close()
    
    def _save_chat_message(
        self, 
        session_id: str, 
        content: str, 
        message_type: str,
        relevant_chunks: Optional[List[int]] = None,
        confidence_score: Optional[float] = None,
        processing_time_ms: Optional[int] = None
    ):
        """Save chat message to database"""
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_messages (
                    session_id, message_type, content, relevant_chunks,
                    confidence_score, model_used, processing_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                message_type,
                content,
                json.dumps(relevant_chunks) if relevant_chunks else None,
                confidence_score,
                'gemini-2.5-pro' if message_type == 'assistant' else None,
                processing_time_ms
            ))
            conn.commit()
            
        finally:
            conn.close()
    
    def _update_session_activity(self, session_id: str):
        """Update session last activity timestamp"""
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE chat_sessions 
                SET last_activity = CURRENT_TIMESTAMP 
                WHERE session_id = ?
            """, (session_id,))
            conn.commit()
            
        finally:
            conn.close()
    
    def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM chat_messages 
                WHERE session_id = ? 
                ORDER BY created_at ASC 
                LIMIT ?
            """, (session_id, limit))
            
            messages = []
            for row in cursor.fetchall():
                message = dict(row)
                # Parse relevant_chunks JSON
                if message['relevant_chunks']:
                    message['relevant_chunks'] = json.loads(message['relevant_chunks'])
                messages.append(message)
            
            return messages
            
        finally:
            conn.close()
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and its messages"""
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            
            # Delete messages first (foreign key constraint)
            cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            
            # Delete session
            cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
            
            deleted = cursor.rowcount > 0
            conn.commit()
            
            if deleted:
                logger.info(f"Deleted chat session {session_id}")
            
            return deleted
            
        finally:
            conn.close()
    
    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent chat sessions"""
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    cs.*,
                    COUNT(cm.id) as message_count
                FROM chat_sessions cs
                LEFT JOIN chat_messages cm ON cs.session_id = cm.session_id
                GROUP BY cs.session_id
                ORDER BY cs.last_activity DESC
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
            
        finally:
            conn.close()
