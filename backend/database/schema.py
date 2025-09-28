"""
Database schema for HEAL RAG system
"""

import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def create_rag_tables(db_path: str = "heal.db"):
    """Create all tables for RAG system"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Original policies table (keep existing)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS policies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_json TEXT NOT NULL
            )
        """)
        
        # Documents table - store uploaded files
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
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
            )
        """)
        
        # Document chunks table - store text chunks with embeddings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_type TEXT DEFAULT 'paragraph', -- 'paragraph', 'section', 'table'
                embedding BLOB, -- Serialized embedding vector
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
        
        # Bill analyses table for tracking bill checker results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bill_analyses (
                id TEXT PRIMARY KEY,
                bill_document_id INTEGER NOT NULL,
                policy_document_id INTEGER,
                analysis_result TEXT NOT NULL,
                analysis_summary TEXT,
                patient_responsibility REAL,
                insurance_payment REAL,
                total_charges REAL,
                potential_disputes TEXT,
                confidence_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (bill_document_id) REFERENCES documents(id),
                FOREIGN KEY (policy_document_id) REFERENCES documents(id)
            )
        """)

        # Chat sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                user_id TEXT, -- Future user management
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                document_context TEXT -- JSON array of document IDs in context
            )
        """)
        
        # Chat messages table with RAG context
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
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
            )
        """)
        
        # RAG queries table for debugging
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                embedding BLOB,
                top_chunks TEXT, -- JSON array of retrieved chunks with scores
                execution_time_ms INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_timestamp ON chat_messages(created_at)")
        
        conn.commit()
        logger.info("RAG database tables created successfully")
        
    except Exception as e:
        logger.error(f"Error creating RAG tables: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def get_db_connection(db_path: str = "heal.db") -> sqlite3.Connection:
    """Get database connection with proper configuration"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn

if __name__ == "__main__":
    create_rag_tables()
