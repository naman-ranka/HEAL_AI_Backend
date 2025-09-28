"""
Pydantic Schemas for HEAL AI Flows
Defines structured input and output schemas for insurance document analysis
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types"""
    IMAGE = "image"
    PDF = "pdf"


class PolicyAnalysisInput(BaseModel):
    """Input schema for insurance policy analysis"""
    document_data: str = Field(description="Base64 encoded document data or file path")
    document_type: DocumentType = Field(description="Type of document (image or PDF)")
    filename: str = Field(description="Original filename of the document")


class PolicyAnalysisOutput(BaseModel):
    """Structured output for insurance policy analysis"""
    deductible: str = Field(
        description="The deductible amount from the insurance policy. Use 'Not found' if not available."
    )
    out_of_pocket_max: str = Field(
        description="The out-of-pocket maximum amount. Use 'Not found' if not available."
    )
    copay: str = Field(
        description="The copay amount for services. Use 'Not found' if not available."
    )
    confidence_score: float = Field(
        description="Confidence score from 0.0 to 1.0 indicating extraction accuracy",
        ge=0.0,
        le=1.0,
        default=0.8
    )
    additional_info: Optional[Dict[str, str]] = Field(
        description="Any additional relevant policy information found",
        default=None
    )


class ChatMessage(BaseModel):
    """Individual chat message"""
    role: str = Field(description="Role of the message sender (user or assistant)")
    content: str = Field(description="Content of the message")
    timestamp: Optional[str] = Field(description="Timestamp of the message", default=None)


class ChatInput(BaseModel):
    """Input schema for chat functionality"""
    message: str = Field(description="User's message to the AI assistant")
    policy_context: Optional[str] = Field(
        description="Insurance policy context for grounded responses",
        default=None
    )
    conversation_history: List[ChatMessage] = Field(
        description="Previous conversation messages for context",
        default=[]
    )


class ChatOutput(BaseModel):
    """Structured output for chat responses"""
    response: str = Field(description="AI assistant's response to the user")
    confidence: float = Field(
        description="Confidence in the response accuracy",
        ge=0.0,
        le=1.0,
        default=0.9
    )
    sources_used: List[str] = Field(
        description="List of policy sections or information used in the response",
        default=[]
    )
    follow_up_questions: List[str] = Field(
        description="Suggested follow-up questions for the user",
        default=[]
    )


class DocumentSummaryInput(BaseModel):
    """Input for document summarization"""
    document_text: str = Field(description="Extracted text from the document")
    summary_type: str = Field(
        description="Type of summary requested (brief, detailed, key_points)",
        default="key_points"
    )


class DocumentSummaryOutput(BaseModel):
    """Output for document summarization"""
    summary: str = Field(description="Generated summary of the document")
    key_points: List[str] = Field(description="List of key points from the document")
    document_type_detected: str = Field(description="Detected type of insurance document")
    completeness_score: float = Field(
        description="Score indicating how complete the document appears",
        ge=0.0,
        le=1.0,
        default=0.8
    )


class HealthCheckOutput(BaseModel):
    """Health check response schema"""
    model_config = {"protected_namespaces": ()}
    
    status: str = Field(description="Overall system status")
    genkit_status: str = Field(description="Genkit framework status")
    model_status: str = Field(description="AI model availability status")
    database_status: str = Field(description="Database connection status")
    timestamp: str = Field(description="Health check timestamp")


class ErrorResponse(BaseModel):
    """Standardized error response"""
    error: str = Field(description="Error message")
    error_code: str = Field(description="Specific error code for debugging")
    details: Optional[Dict[str, Any]] = Field(description="Additional error details", default=None)
    timestamp: str = Field(description="Error timestamp")


# Export all schemas
__all__ = [
    'DocumentType',
    'PolicyAnalysisInput',
    'PolicyAnalysisOutput', 
    'ChatMessage',
    'ChatInput',
    'ChatOutput',
    'DocumentSummaryInput',
    'DocumentSummaryOutput',
    'HealthCheckOutput',
    'ErrorResponse'
]
