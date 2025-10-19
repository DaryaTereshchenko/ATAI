"""
Template-based answer formatter for SPARQL results.
Provides human-friendly responses without requiring an LLM.
"""

import random
from typing import Optional


class AnswerFormatter:
    """Formats SPARQL query results into human-friendly responses using templates."""
    
    # Templates for successful queries (single result)
    SINGLE_RESULT_TEMPLATES = [
        "According to our database, the answer is: {answer}",
        "Based on the data, I found: {answer}",
        "The database shows: {answer}",
        "Here's what I found: {answer}",
        "From our movie database: {answer}",
    ]
    
    # Templates for successful queries (multiple results)
    MULTIPLE_RESULTS_TEMPLATES = [
        "I found {count} results in our database:\n{answers}",
        "According to the database, here are {count} results:\n{answers}",
        "The data shows {count} matches:\n{answers}",
        "Here's what I found ({count} results):\n{answers}",
        "From our movie database ({count} entries):\n{answers}",
    ]
    
    # Templates for no results
    NO_RESULTS_TEMPLATES = [
        "I couldn't find any results in our database for that query.",
        "The database doesn't contain information matching your question.",
        "No matching data was found in our movie database.",
        "Sorry, I don't have that information in the database.",
        "The database search didn't return any results.",
    ]
    
    # Prefix for all responses to indicate data source
    DB_PREFIX = "🔍 **Database Query Result**\n\n"
    
    @classmethod
    def format(cls, raw_result: str, query_explanation: Optional[str] = None) -> str:
        """
        Format a SPARQL query result into a human-friendly response.
        
        Args:
            raw_result: The raw result string from SPARQL execution
            query_explanation: Optional explanation (ignored to hide technical details)
            
        Returns:
            Formatted, human-friendly response string
        """
        # Check if no results
        if not raw_result or "No answer found" in raw_result or "No results" in raw_result:
            template = random.choice(cls.NO_RESULTS_TEMPLATES)
            response = cls.DB_PREFIX + template
            
            # Add helpful hint
            response += "\n\n💡 *Tip: Try rephrasing your question or check the movie title spelling.*"
            return response
        
        # Parse the results
        lines = [line.strip() for line in raw_result.strip().split('\n') if line.strip()]
        
        if len(lines) == 0:
            # Empty result
            template = random.choice(cls.NO_RESULTS_TEMPLATES)
            return cls.DB_PREFIX + template
        
        elif len(lines) == 1:
            # Single result
            template = random.choice(cls.SINGLE_RESULT_TEMPLATES)
            answer = lines[0]
            response = cls.DB_PREFIX + template.format(answer=answer)
            
        else:
            # Multiple results
            template = random.choice(cls.MULTIPLE_RESULTS_TEMPLATES)
            
            # Format multiple results as a bulleted list
            formatted_answers = "\n".join(f"• {line}" for line in lines[:10])  # Limit to 10 results
            
            if len(lines) > 10:
                formatted_answers += f"\n• ... and {len(lines) - 10} more"
            
            response = cls.DB_PREFIX + template.format(
                count=min(len(lines), 10),
                answers=formatted_answers
            )
        
        # Do not add any technical explanation
        
        return response
    
    @classmethod
    def format_error(cls, error_message: str) -> str:
        """
        Format an error message in a friendly way.
        
        Args:
            error_message: The error message
            
        Returns:
            Formatted error response
        """
        return (
            "⚠️ **Processing Error**\n\n"
            f"I encountered an issue: {error_message}\n\n"
            "Please try rephrasing your question or contact support if the problem persists."
        )
