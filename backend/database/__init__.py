"""
Database package for HEAL RAG system
"""

from .schema import create_rag_tables, get_db_connection

__all__ = ['create_rag_tables', 'get_db_connection']
