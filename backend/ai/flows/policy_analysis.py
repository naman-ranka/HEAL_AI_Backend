"""
Insurance Policy Analysis Flow using Genkit-Inspired Patterns
Provides structured analysis of insurance documents with guaranteed output format
"""

import base64
import logging
import json
import time
from typing import Dict, Any
from PIL import Image
import io
import fitz  # PyMuPDF

from ..genkit_config import ai_config
from ..schemas import PolicyAnalysisInput, PolicyAnalysisOutput

logger = logging.getLogger(__name__)


async def analyze_insurance_policy(input_data: PolicyAnalysisInput) -> PolicyAnalysisOutput:
    """
    Analyze insurance policy document and extract key information
    
    Args:
        input_data: PolicyAnalysisInput containing document data and metadata
        
    Returns:
        PolicyAnalysisOutput with structured policy information
    """
    try:
        logger.info(f"🔍 Starting policy analysis for {input_data.filename}")
        logger.info(f"📊 Document type: {input_data.document_type}, Data size: {len(input_data.document_data)} bytes")
        
        # Check if AI is available
        if not ai_config.is_available():
            logger.warning("⚠️ AI not available, returning mock data")
            return PolicyAnalysisOutput(
                deductible="$1,000 (mock)",
                out_of_pocket_max="$5,000 (mock)",
                copay="$25 (mock)",
                confidence_score=0.5,
                additional_info={"note": "Mock data - configure GEMINI_API_KEY for real analysis"}
            )
        
        # Prepare the analysis prompt with JSON schema instructions
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
        
        # Handle different document types
        if input_data.document_type == "image":
            logger.info("🖼️ Processing image document with OCR")
            # For images, decode and analyze
            image_data = base64.b64decode(input_data.document_data)
            image = Image.open(io.BytesIO(image_data))
            logger.info(f"📸 Image size: {image.size}, Mode: {image.mode}")
            
            # Get model and generate response
            model = ai_config.get_model('flash')
            logger.info("🤖 Using Gemini Flash model for image analysis")
            logger.info(f"📝 Sending prompt to Gemini (length: {len(analysis_prompt)} chars)")
            
            start_time = time.time()
            response = model.generate_content([analysis_prompt, image])
            end_time = time.time()
            
            logger.info(f"⚡ Gemini response received in {(end_time - start_time):.2f}s")
            logger.info(f"📄 Response length: {len(response.text)} chars")
            logger.debug(f"📄 Raw response preview: {response.text[:200]}...")
            
        elif input_data.document_type == "pdf":
            logger.info("📑 Processing PDF document")
            # For PDFs, extract text first then analyze
            pdf_data = base64.b64decode(input_data.document_data)
            extracted_text = _extract_pdf_text(pdf_data)
            logger.info(f"📖 Extracted text length: {len(extracted_text)} chars")
            logger.debug(f"📖 First 200 chars: {extracted_text[:200]}...")
            
            # Combine prompt with extracted text
            full_prompt = f"{analysis_prompt}\n\nExtracted Text:\n{extracted_text}"
            
            # Get model and generate response
            model = ai_config.get_model('pro')  # Use Pro for complex text analysis
            logger.info("🤖 Using Gemini Pro model for PDF text analysis")
            logger.info(f"📝 Sending combined prompt to Gemini (length: {len(full_prompt)} chars)")
            
            start_time = time.time()
            response = model.generate_content(full_prompt)
            end_time = time.time()
            
            logger.info(f"⚡ Gemini response received in {(end_time - start_time):.2f}s")
            logger.info(f"📄 Response length: {len(response.text)} chars")
            logger.debug(f"📄 Raw response preview: {response.text[:200]}...")
            
        else:
            logger.error(f"❌ Unsupported document type: {input_data.document_type}")
            raise ValueError(f"Unsupported document type: {input_data.document_type}")
        
        # Parse the response
        logger.info("🔧 Parsing Gemini response for structured data")
        policy_analysis = _parse_analysis_response(response.text)
        
        logger.info(f"✅ Successfully analyzed policy: {input_data.filename}")
        logger.info(f"📋 Extracted - Deductible: {policy_analysis.deductible}, Out-of-pocket: {policy_analysis.out_of_pocket_max}, Copay: {policy_analysis.copay}")
        logger.info(f"🎯 Confidence score: {policy_analysis.confidence_score}")
        
        return policy_analysis
        
    except Exception as e:
        logger.error(f"Error analyzing policy {input_data.filename}: {e}")
        # Return a structured error response that still matches our schema
        return PolicyAnalysisOutput(
            deductible="Analysis failed",
            out_of_pocket_max="Analysis failed", 
            copay="Analysis failed",
            confidence_score=0.0,
            additional_info={"error": str(e)}
        )


async def summarize_policy_document(input_data: PolicyAnalysisInput) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of the insurance policy document
    
    Args:
        input_data: PolicyAnalysisInput containing document data
        
    Returns:
        Dictionary containing detailed policy summary
    """
    try:
        if not ai_config.is_available():
            return {
                "summary": "AI summarization not available. Please configure GEMINI_API_KEY.",
                "document_type": input_data.document_type,
                "filename": input_data.filename
            }
        
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
        
        if input_data.document_type == "image":
            image_data = base64.b64decode(input_data.document_data)
            image = Image.open(io.BytesIO(image_data))
            
            model = ai_config.get_model('flash')
            response = model.generate_content([summary_prompt, image])
            
        else:  # PDF
            pdf_data = base64.b64decode(input_data.document_data)
            extracted_text = _extract_pdf_text(pdf_data)
            full_prompt = f"{summary_prompt}\n\nDocument Content:\n{extracted_text}"
            
            model = ai_config.get_model('pro')
            response = model.generate_content(full_prompt)
        
        return {
            "summary": response.text,
            "document_type": input_data.document_type,
            "filename": input_data.filename
        }
        
    except Exception as e:
        logger.error(f"Error summarizing document {input_data.filename}: {e}")
        return {
            "summary": f"Failed to generate summary: {str(e)}",
            "document_type": input_data.document_type,
            "filename": input_data.filename
        }


def _parse_analysis_response(response_text: str) -> PolicyAnalysisOutput:
    """
    Parse AI response and create structured PolicyAnalysisOutput
    
    Args:
        response_text: Raw response from AI model
        
    Returns:
        PolicyAnalysisOutput with extracted data
    """
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
        return PolicyAnalysisOutput(
            deductible=parsed_data.get("deductible", "Not found"),
            out_of_pocket_max=parsed_data.get("out_of_pocket_max", "Not found"),
            copay=parsed_data.get("copay", "Not found"),
            confidence_score=0.9,  # High confidence for successful parsing
            additional_info={"parsing_method": "json_extraction"}
        )
        
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON response, using fallback extraction")
        # Fallback: try to extract values using text patterns
        return _extract_values_from_text(response_text)
    except Exception as e:
        logger.error(f"Error parsing response: {e}")
        return PolicyAnalysisOutput(
            deductible="Parsing failed",
            out_of_pocket_max="Parsing failed",
            copay="Parsing failed",
            confidence_score=0.0,
            additional_info={"error": str(e)}
        )


def _extract_values_from_text(text: str) -> PolicyAnalysisOutput:
    """
    Fallback method to extract values from unstructured text
    
    Args:
        text: Raw text response
        
    Returns:
        PolicyAnalysisOutput with best-effort extraction
    """
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
    
    return PolicyAnalysisOutput(
        deductible=deductible,
        out_of_pocket_max=out_of_pocket_max,
        copay=copay,
        confidence_score=0.6,  # Lower confidence for text extraction
        additional_info={"parsing_method": "text_extraction"}
    )


def _extract_pdf_text(pdf_data: bytes) -> str:
    """
    Extract text content from PDF bytes
    
    Args:
        pdf_data: Raw PDF bytes
        
    Returns:
        Extracted text content
    """
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


# Export the flows
__all__ = ['analyze_insurance_policy', 'summarize_policy_document']
