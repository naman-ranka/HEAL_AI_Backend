"""
Insurance Policy Chatbot Flow using Genkit
Provides conversational AI for insurance policy questions with RAG approach
"""

import logging
from typing import List
from datetime import datetime

from ..genkit_config import ai
from ..schemas import ChatInput, ChatOutput, ChatMessage

logger = logging.getLogger(__name__)


@ai.flow()
async def insurance_policy_chat(input_data: ChatInput) -> ChatOutput:
    """
    Chat with AI assistant about insurance policies using RAG approach
    
    Args:
        input_data: ChatInput containing user message and context
        
    Returns:
        ChatOutput with structured AI response
    """
    try:
        logger.info(f"Processing chat message: {input_data.message[:50]}...")
        
        # Build the system prompt with policy context
        system_prompt = """
        You are HEAL, an expert insurance policy assistant. Your role is to help users understand their insurance policies and answer questions based ONLY on the provided policy information.

        **Guidelines:**
        1. Answer questions based solely on the provided policy context
        2. If information is not in the policy context, clearly state "I don't have that information in your policy"
        3. Be helpful, clear, and concise
        4. Suggest relevant follow-up questions when appropriate
        5. Use simple language that policyholders can understand
        6. Always be accurate and conservative in your responses

        **Policy Context:**
        {policy_context}

        **Conversation History:**
        {conversation_history}

        **Current User Question:**
        {user_message}

        Provide a helpful response based on the policy information available.
        """
        
        # Format conversation history
        history_text = ""
        for msg in input_data.conversation_history:
            history_text += f"{msg.role}: {msg.content}\n"
        
        # Format the complete prompt
        formatted_prompt = system_prompt.format(
            policy_context=input_data.policy_context or "No policy information provided",
            conversation_history=history_text or "No previous conversation",
            user_message=input_data.message
        )
        
        # Generate response using Genkit with structured output
        result = await ai.generate(
            prompt=formatted_prompt,
            output_schema=ChatOutput,
            model='googleai/gemini-1.5-pro'  # Use Pro for better reasoning
        )
        
        # Genkit ensures structured output
        chat_response = result.output
        
        logger.info("Successfully generated chat response")
        return chat_response
        
    except Exception as e:
        logger.error(f"Error in chat flow: {e}")
        # Return structured error response
        return ChatOutput(
            response="I apologize, but I'm having trouble processing your question right now. Please try again.",
            confidence=0.0,
            sources_used=[],
            follow_up_questions=["Could you rephrase your question?", "Is there something specific about your policy you'd like to know?"]
        )


@ai.flow()
async def generate_policy_questions(policy_text: str) -> List[str]:
    """
    Generate relevant questions a user might ask about their policy
    
    Args:
        policy_text: The insurance policy text content
        
    Returns:
        List of suggested questions
    """
    try:
        questions_prompt = f"""
        Based on the following insurance policy information, generate 5-7 relevant questions that a policyholder might want to ask about their coverage.

        Focus on:
        - Coverage details and limits
        - Costs (deductibles, copays, premiums)
        - Claims process
        - Exclusions or limitations
        - Important policy terms

        Policy Information:
        {policy_text}

        Generate practical, specific questions that would help the policyholder understand their coverage better.
        Return the questions as a simple list, one per line.
        """
        
        result = await ai.generate(
            prompt=questions_prompt,
            model='googleai/gemini-1.5-flash'
        )
        
        # Parse the response into a list of questions
        questions_text = result.text()
        questions = [q.strip() for q in questions_text.split('\n') if q.strip() and not q.strip().startswith('#')]
        
        return questions[:7]  # Limit to 7 questions
        
    except Exception as e:
        logger.error(f"Error generating policy questions: {e}")
        return [
            "What is my deductible?",
            "What does my policy cover?",
            "How do I file a claim?",
            "What are my copay amounts?",
            "What is not covered by my policy?"
        ]


@ai.flow()
async def explain_policy_term(term: str, policy_context: str) -> str:
    """
    Explain a specific insurance term in the context of the user's policy
    
    Args:
        term: The insurance term to explain
        policy_context: The user's policy information
        
    Returns:
        Clear explanation of the term
    """
    try:
        explanation_prompt = f"""
        Explain the insurance term "{term}" in simple, easy-to-understand language. 
        
        If this term appears in the user's policy context below, explain how it specifically applies to their coverage.
        If the term doesn't appear in their policy, provide a general explanation but note that they should check their specific policy documents.

        Policy Context:
        {policy_context}

        Provide a clear, concise explanation that a typical policyholder would understand.
        """
        
        result = await ai.generate(
            prompt=explanation_prompt,
            model='googleai/gemini-1.5-flash'
        )
        
        return result.text()
        
    except Exception as e:
        logger.error(f"Error explaining term '{term}': {e}")
        return f"I'm sorry, I couldn't provide an explanation for '{term}' at this time. Please try again or consult your policy documents."


# Export the flows
__all__ = ['insurance_policy_chat', 'generate_policy_questions', 'explain_policy_term']
