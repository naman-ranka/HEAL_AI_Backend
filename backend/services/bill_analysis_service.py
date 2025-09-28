#!/usr/bin/env python3
"""
Bill Analysis Service
Handles multimodal analysis of medical bills against insurance policies
Following standard service patterns and best practices
"""

import os
import json
import logging
import time
import sqlite3
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

class BillAnalysisService:
    """
    Service for analyzing medical bills against insurance policies
    Using direct multimodal Gemini integration for maximum accuracy
    """
    
    def __init__(self, db_path: str = "langchain_heal.db"):
        """
        Initialize the bill analysis service
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable required")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize chat model for analysis
        self.chat_model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",  # Use Pro for complex analysis
            google_api_key=self.api_key,
            temperature=0.1
        )
        
        logger.info("🏥 BillAnalysisService initialized")
    
    async def analyze_bill_vs_policy(
        self, 
        bill_id: str, 
        policy_id: Optional[str] = None,
        include_dispute_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze medical bill against insurance policy using multimodal Gemini
        
        Args:
            bill_id: ID of uploaded bill document
            policy_id: ID of policy document (if None, uses most recent)
            include_dispute_recommendations: Whether to include dispute analysis
            
        Returns:
            Complete analysis results as structured data
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Starting bill analysis: bill_id={bill_id}, policy_id={policy_id}")
            
            # Get file paths
            bill_path, policy_path = await self._get_document_paths(bill_id, policy_id)
            
            if not bill_path or not policy_path:
                raise ValueError("Could not find required documents for analysis")
            
            # Perform multimodal analysis
            analysis_result = await self._perform_multimodal_analysis(
                bill_path, 
                policy_path, 
                include_dispute_recommendations
            )
            
            # Parse and structure the results
            structured_analysis = await self._parse_analysis_result(analysis_result)
            
            # Store analysis in database
            analysis_id = await self._store_analysis_result(
                bill_id, 
                policy_id, 
                analysis_result, 
                structured_analysis
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            logger.info(f"✅ Bill analysis completed in {processing_time}ms: {analysis_id}")
            
            return {
                "analysis_id": analysis_id,
                "structured_analysis": structured_analysis,
                "processing_time_ms": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Bill analysis failed: {e}")
            raise
    
    async def _get_document_paths(
        self, 
        bill_id: str, 
        policy_id: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """Get file paths for bill and policy documents"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get bill document
            cursor.execute("""
                SELECT raw_file_path, filename FROM documents 
                WHERE id = ? AND document_type = 'bill'
            """, (bill_id,))
            
            bill_row = cursor.fetchone()
            if not bill_row:
                logger.error(f"Bill document not found: {bill_id}")
                return None, None
            
            bill_path = bill_row[0]
            
            # Get policy document - prefer specific policy_id if provided
            if policy_id:
                cursor.execute("""
                    SELECT raw_file_path, filename FROM documents 
                    WHERE id = ? AND document_type = 'policy'
                """, (policy_id,))
                policy_row = cursor.fetchone()
                
                if policy_row:
                    policy_path = policy_row[0]
                    logger.info(f"📄 Using specific policy: {policy_row[1]}")
                else:
                    logger.warning(f"Specific policy {policy_id} not found, falling back to most recent")
                    # Fall back to most recent if specific policy not found
                    cursor.execute("""
                        SELECT raw_file_path, filename FROM documents 
                        WHERE document_type = 'policy' 
                        ORDER BY upload_timestamp DESC 
                        LIMIT 1
                    """)
                    policy_row = cursor.fetchone()
            else:
                # Get most recent policy
                cursor.execute("""
                    SELECT raw_file_path, filename FROM documents 
                    WHERE document_type = 'policy' 
                    ORDER BY upload_timestamp DESC 
                    LIMIT 1
                """)
                policy_row = cursor.fetchone()
            
            if not policy_row:
                logger.error("No policy document found")
                return None, None
            
            policy_path = policy_row[0]
            
            logger.info(f"📄 Found documents - Bill: {bill_path}, Policy: {policy_path}")
            
            return bill_path, policy_path
            
        finally:
            conn.close()
    
    async def _perform_multimodal_analysis(
        self, 
        bill_path: str, 
        policy_path: str,
        include_dispute_recommendations: bool
    ) -> str:
        """Perform the actual multimodal analysis using Gemini"""
        
        try:
            # Create comprehensive analysis prompt
            analysis_prompt = self._build_analysis_prompt(include_dispute_recommendations)
            
            # Upload files to Gemini
            logger.info("📤 Uploading files to Gemini for analysis...")
            
            bill_file = genai.upload_file(bill_path)
            policy_file = genai.upload_file(policy_path)
            
            # Generate analysis using multimodal Gemini
            logger.info("🤖 Generating multimodal analysis...")
            
            model = genai.GenerativeModel('gemini-1.5-pro')
            
            response = model.generate_content([
                analysis_prompt,
                bill_file,
                policy_file
            ])
            
            # Clean up uploaded files
            bill_file.delete()
            policy_file.delete()
            
            logger.info("✅ Multimodal analysis completed")
            
            return response.text
            
        except Exception as e:
            logger.error(f"❌ Multimodal analysis failed: {e}")
            raise
    
    def _build_analysis_prompt(self, include_dispute_recommendations: bool) -> str:
        """Build comprehensive analysis prompt for Gemini"""
        
        base_prompt = """
You are an expert medical billing and insurance analyst. You have been provided with:
1. A medical bill (first document)
2. The patient's specific insurance policy document (second document)

IMPORTANT: Use the EXACT policy document provided to analyze this specific bill. This policy contains the patient's actual coverage details, deductibles, copays, and benefits that apply to this bill.

Please analyze the medical bill against this specific insurance policy and provide a comprehensive analysis.

**ANALYSIS REQUIREMENTS:**

1. **BILL SUMMARY:**
   - Provider name and contact information
   - Patient name (if visible)
   - Date(s) of service
   - Total charges amount
   - List of services/procedures provided
   - Any procedure codes (CPT, ICD-10, etc.)

2. **COVERAGE ANALYSIS:**
   - Which services are covered by the policy
   - Which services are not covered (and why)
   - Deductible requirements and current status
   - Copay amounts for different services
   - Coinsurance percentages that apply
   - Network status (in-network vs out-of-network)

3. **FINANCIAL BREAKDOWN:**
   - Total charges from the bill
   - Expected insurance payment amount
   - Patient responsibility amount
   - Breakdown of: deductible + copay + coinsurance
   - Any remaining deductible amount

4. **POLICY COMPLIANCE:**
   - Whether the bill follows proper billing practices
   - If prior authorization was required and obtained
   - Appropriate use of procedure codes
"""

        if include_dispute_recommendations:
            base_prompt += """
5. **DISPUTE RECOMMENDATIONS:**
   - Any billing errors or questionable charges
   - Services that should be covered but appear denied
   - Incorrect procedure coding
   - Overcharges compared to standard rates
   - Recommended actions for the patient
   - Priority level for each issue (high/medium/low)
"""

        base_prompt += """
**OUTPUT FORMAT:**
Please provide your analysis in a clear, structured format with specific sections.
Use bullet points and clear headings.
Be specific with dollar amounts and percentages where possible.
If information is not available in the documents, clearly state "Information not available in provided documents."

**IMPORTANT:**
- Base your analysis ONLY on the information visible in the provided documents
- Do not make assumptions about coverage not explicitly stated in the policy
- Be conservative in your recommendations
- Focus on factual analysis rather than legal advice
"""

        return base_prompt
    
    async def _parse_analysis_result(self, analysis_text: str) -> Dict[str, Any]:
        """Parse the Gemini analysis into structured data"""
        
        try:
            # For now, return the raw analysis
            # In a production system, you might use another LLM call to structure this
            # or implement proper text parsing logic
            
            return {
                "raw_analysis": analysis_text,
                "analysis_length": len(analysis_text),
                "sections_detected": self._detect_sections(analysis_text)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to parse analysis result: {e}")
            return {
                "raw_analysis": analysis_text,
                "parsing_error": str(e)
            }
    
    def _detect_sections(self, text: str) -> List[str]:
        """Detect which analysis sections are present in the response"""
        
        sections = []
        section_keywords = {
            "bill_summary": ["bill summary", "provider name", "total charges"],
            "coverage_analysis": ["coverage", "covered", "not covered", "deductible"],
            "financial_breakdown": ["financial", "patient responsibility", "insurance payment"],
            "dispute_recommendations": ["dispute", "error", "questionable", "recommend"]
        }
        
        text_lower = text.lower()
        
        for section, keywords in section_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                sections.append(section)
        
        return sections
    
    async def _store_analysis_result(
        self, 
        bill_id: str, 
        policy_id: Optional[str], 
        raw_analysis: str,
        structured_analysis: Dict[str, Any]
    ) -> str:
        """Store analysis result in database"""
        
        analysis_id = f"analysis_{int(time.time())}_{bill_id[:8]}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO bill_analyses (
                    id, bill_document_id, policy_document_id, 
                    analysis_result, analysis_summary, confidence_score
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                analysis_id,
                bill_id,
                policy_id,
                raw_analysis,
                json.dumps(structured_analysis),
                0.85  # Default confidence score
            ))
            
            conn.commit()
            logger.info(f"💾 Stored analysis result: {analysis_id}")
            
            return analysis_id
            
        finally:
            conn.close()
    
    async def get_analysis_history(self, limit: int = 20) -> Dict[str, Any]:
        """Get history of bill analyses"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    ba.id,
                    d.filename,
                    ba.created_at,
                    ba.patient_responsibility,
                    ba.confidence_score
                FROM bill_analyses ba
                JOIN documents d ON ba.bill_document_id = d.id
                ORDER BY ba.created_at DESC
                LIMIT ?
            """, (limit,))
            
            analyses = []
            for row in cursor.fetchall():
                analyses.append({
                    "analysis_id": row[0],
                    "bill_filename": row[1],
                    "analysis_date": row[2],
                    "patient_responsibility": row[3],
                    "confidence_score": row[4]
                })
            
            return {
                "analyses": analyses,
                "total_count": len(analyses)
            }
            
        finally:
            conn.close()
    
    async def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get specific analysis by ID"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    analysis_result,
                    analysis_summary,
                    confidence_score,
                    created_at
                FROM bill_analyses
                WHERE id = ?
            """, (analysis_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                "analysis_id": analysis_id,
                "raw_analysis": row[0],
                "structured_analysis": json.loads(row[1]) if row[1] else {},
                "confidence_score": row[2],
                "created_at": row[3]
            }
            
        finally:
            conn.close()
