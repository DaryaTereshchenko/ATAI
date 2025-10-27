"""
Template-based answer formatter for SPARQL results.
Provides human-friendly responses without requiring an LLM.
"""

import random
import re
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
    
    @classmethod
    def _clean_result_line(cls, line: str) -> str:
        """
        Clean a single result line by extracting entity IDs from Wikidata URLs.
        Also removes extra whitespace and normalizes the output.
        """
        # Quick check: if line doesn't contain 'http', no cleaning needed
        if 'http' not in line:
            return line.strip()
        
        # Pattern to match Wikidata entity URLs
        wikidata_pattern = r'http://www\.wikidata\.org/entity/(Q\d+)'
        
        # Find Wikidata URL in the line
        match = re.search(wikidata_pattern, line)
        
        if match:
            entity_id = match.group(1)  # Extract Q12345
            # Remove the URL from the line
            cleaned_line = re.sub(r',?\s*http://[^\s,]+', '', line).strip()
            # Remove extra spaces
            cleaned_line = re.sub(r'\s+', ' ', cleaned_line)
            # Add entity ID in parentheses
            return f"{cleaned_line} ({entity_id})"
        
        # No Wikidata URL found, return as-is (but still strip and normalize whitespace)
        return re.sub(r'\s+', ' ', line.strip())
    
    @classmethod
    def format(cls, raw_result: str, query_explanation: Optional[str] = None, approach: str = "factual") -> str:
        """
        Format a SPARQL query result into a human-friendly response.
        
        Args:
            raw_result: The raw result string from SPARQL execution
            query_explanation: Optional explanation (ignored to hide technical details)
            approach: The approach used - "factual" (SPARQL) or "embedding" (TransE)
            
        Returns:
            Formatted, human-friendly response string
        """
        # Determine prefix based on approach
        if approach == "embedding":
            prefix = "🔢 **Embedding-Based Answer**\n\n"
        elif approach == "both":
            prefix = "📊 **Hybrid Answer (Factual + Embeddings)**\n\n"
        else:  # factual
            prefix = "✅ **Factual Answer (SPARQL Query)**\n\n"
        
        # Check if no results
        if not raw_result or "No answer found" in raw_result or "No results" in raw_result:
            template = random.choice(cls.NO_RESULTS_TEMPLATES)
            response = prefix + template
            
            # Add helpful hint
            response += "\n\n💡 *Tip: Try rephrasing your question or check the movie title spelling.*"
            return response
        
        # Parse the results and clean each line
        lines = [cls._clean_result_line(line) for line in raw_result.strip().split('\n') if line.strip()]
        
        if len(lines) == 0:
            # Empty result
            template = random.choice(cls.NO_RESULTS_TEMPLATES)
            return prefix + template
        
        elif len(lines) == 1:
            # Single result
            template = random.choice(cls.SINGLE_RESULT_TEMPLATES)
            answer = lines[0]
            response = prefix + template.format(answer=answer)
            
        else:
            # Multiple results
            template = random.choice(cls.MULTIPLE_RESULTS_TEMPLATES)
            
            # Format multiple results as a bulleted list
            formatted_answers = "\n".join(f"• {line}" for line in lines[:10])  # Limit to 10 results
            
            if len(lines) > 10:
                formatted_answers += f"\n• ... and {len(lines) - 10} more"
            
            response = prefix + template.format(
                count=min(len(lines), 10),
                answers=formatted_answers
            )
        
        return response
    
    @classmethod
    def format_error(cls, error_message: str) -> str:
        """
        Format an error message in a friendly way without exposing technical details.
        
        Args:
            error_message: The error message
            
        Returns:
            Formatted error response
        """
        # Generic user-friendly error message
        return (
            "⚠️ **Something went wrong**\n\n"
            "I'm having trouble processing your request right now.\n\n"
            "Please try:\n"
            "• Rephrasing your question\n"
            "• Being more specific about the movie or person\n"
            "• Asking a different question\n\n"
            "If the problem persists, please contact support."
        )
