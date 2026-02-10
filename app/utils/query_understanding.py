"""
Query understanding and reformulation utilities
"""
from typing import List, Dict


def analyze_question_type(question: str) -> str:
    """
    Analyze the type of question being asked.
    
    Args:
        question: The user's question
    
    Returns:
        Question type: 'definition', 'how_to', 'comparison', 'example', 
                      'explanation', 'list', 'other'
    """
    question_lower = question.lower().strip()
    
    # Definition questions
    if any(q in question_lower for q in ['what is', 'what are', 'define', 'meaning of', 'definition of']):
        return 'definition'
    
    # How-to/Process questions
    if any(q in question_lower for q in ['how to', 'how do', 'how does', 'how can', 'steps to', 'process of', 'way to']):
        return 'how_to'
    
    # Comparison questions
    if any(q in question_lower for q in ['difference between', 'compare', 'vs', 'versus', 'differ from', 'contrast']):
        return 'comparison'
    
    # Example questions
    if any(q in question_lower for q in ['example', 'give me', 'show me', 'demonstrate', 'instance of']):
        return 'example'
    
    # List questions
    if any(q in question_lower for q in ['list', 'types of', 'kinds of', 'categories', 'what are the']):
        return 'list'
    
    # Explanation questions (why/describe)
    if any(q in question_lower for q in ['explain', 'why', 'describe', 'tell me about', 'elaborate']):
        return 'explanation'
    
    return 'other'


def get_question_context(question: str) -> Dict[str, str]:
    """
    Get contextual information about the question to guide answer generation.
    
    Args:
        question: The user's question
    
    Returns:
        Dictionary with question type and guidance for the LLM
    """
    q_type = analyze_question_type(question)
    
    # Guidance for each question type
    guidance_map = {
        'definition': 'Provide a clear, concise definition followed by an explanation. Use simple language and examples from the context if available.',
        
        'how_to': 'Provide step-by-step instructions or explain the process clearly. Break down complex procedures into numbered steps. Include any prerequisites or important notes.',
        
        'comparison': 'Highlight key similarities and differences in a structured way. Use a comparison format (e.g., "X does... while Y does..."). Be specific and reference the context.',
        
        'example': 'Provide concrete, specific examples from the context. If multiple examples exist, choose the most relevant or illustrative one. Explain why the example demonstrates the concept.',
        
        'list': 'Provide a clear, organized list. Use bullet points or numbering. Include brief explanations for each item if helpful.',
        
        'explanation': 'Explain the concept thoroughly with reasoning. Break down complex ideas into digestible parts. Use analogies or examples from the context when helpful.',
        
        'other': 'Answer the question comprehensively based on the context. Be clear, helpful, and educational in your response.'
    }
    
    return {
        'type': q_type,
        'guidance': guidance_map.get(q_type, guidance_map['other'])
    }


def expand_query(question: str) -> List[str]:
    """
    Generate alternative phrasings of the question for better retrieval.
    
    Args:
        question: The original question
    
    Returns:
        List of query variations including the original
    """
    variations = [question]
    question_type = analyze_question_type(question)
    question_lower = question.lower()
    
    # Add variations based on question type
    if question_type == 'definition':
        # "What is X?" → also search for "X definition", "X meaning"
        for phrase in ['what is', 'what are', 'define']:
            if phrase in question_lower:
                topic = question_lower.replace(phrase, '').strip('?').strip()
                if topic:
                    variations.append(f"{topic} definition")
                    variations.append(f"{topic} explanation")
                break
    
    elif question_type == 'how_to':
        # "How to X?" → also search for "X process", "X steps"
        for phrase in ['how to', 'how do', 'how does']:
            if phrase in question_lower:
                topic = question_lower.replace(phrase, '').strip('?').strip()
                if topic:
                    variations.append(f"{topic} process")
                    variations.append(f"{topic} steps")
                    variations.append(f"{topic} tutorial")
                break
    
    elif question_type == 'comparison':
        # "Difference between X and Y" → also search for "X vs Y"
        if 'difference between' in question_lower:
            topic = question_lower.replace('difference between', '').strip('?').strip()
            variations.append(f"{topic} comparison")
    
    return variations
