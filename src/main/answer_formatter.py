"""
Answer Formatter - Formats query results into polite, human-friendly responses.
Handles both factual SPARQL results and embedding-based results.
"""

from typing import List, Optional


class AnswerFormatter:
    """Formats query answers into natural, polite responses."""
    
    @staticmethod
    def format_list(items: List[str], conjunction: str = "and") -> str:
        """
        Format a list of items with proper grammar.
        
        Examples:
            ["Alice"] -> "Alice"
            ["Alice", "Bob"] -> "Alice and Bob"
            ["Alice", "Bob", "Carol"] -> "Alice, Bob, and Carol"
        
        Args:
            items: List of items to format
            conjunction: Conjunction to use ("and" or "or")
            
        Returns:
            Formatted string
        """
        if not items:
            return ""
        
        items = [str(item).strip() for item in items if item]
        
        if len(items) == 1:
            return items[0]
        elif len(items) == 2:
            return f"{items[0]} {conjunction} {items[1]}"
        else:
            # Oxford comma style: "A, B, and C"
            return ", ".join(items[:-1]) + f", {conjunction} {items[-1]}"
    
    @staticmethod
    def format_factual_response(
        movie_title: Optional[str] = None,
        person_name: Optional[str] = None,
        relation: Optional[str] = None,
        results: List[str] = None,
        is_verification: bool = False,
        verification_result: bool = False
    ) -> str:
        """
        Format a factual SPARQL response.
        
        Args:
            movie_title: Movie title (if applicable)
            person_name: Person name (if applicable)
            relation: Relation type (director, cast_member, etc.)
            results: List of results
            is_verification: Whether this is a yes/no question
            verification_result: Result of verification (True/False)
            
        Returns:
            Formatted natural language response
        """
        # Verification queries
        if is_verification:
            relation_text = {
                'director': 'directed',
                'cast_member': 'starred in',
                'screenwriter': 'wrote',
                'producer': 'produced'
            }.get(relation, relation.replace('_', ' '))
            
            if verification_result:
                return f"✅ Yes, **{person_name}** {relation_text} **'{movie_title}'**."
            else:
                relation_negative = {
                    'directed': 'direct',
                    'starred in': 'star in',
                    'wrote': 'write',
                    'produced': 'produce'
                }.get(relation_text, relation_text)
                return f"❌ No, **{person_name}** did not {relation_negative} **'{movie_title}'**."
        
        # Empty results
        if not results:
            if movie_title:
                return f"❌ I couldn't find any {relation.replace('_', ' ')} information for **'{movie_title}'** in the knowledge graph."
            elif person_name:
                return f"❌ I couldn't find any films where **{person_name}** was the {relation.replace('_', ' ')} in the knowledge graph."
            else:
                return "❌ I couldn't find any matching results in the knowledge graph."
        
        # Format based on relation type
        relation_map = {
            'director': ('directed by', 'directed'),
            'cast_member': ('starring', 'starred in'),
            'screenwriter': ('written by', 'wrote'),
            'producer': ('produced by', 'produced'),
            'genre': ('genre', 'genres'),
            'publication_date': ('released in', 'released'),
            'country_of_origin': ('from', 'from'),
            'original_language_of_film_or_tv_show': ('in', 'in'),
            'language': ('in', 'in'),
            'award_received': ('received', 'received'),
            'rating': ('rating', 'rating')
        }
        
        # Forward query (Movie → Property)
        if movie_title:
            forward_text, _ = relation_map.get(relation, (relation.replace('_', ' '), relation.replace('_', ' ')))
            formatted_results = AnswerFormatter.format_list(results)
            
            if relation in ['director', 'cast_member', 'screenwriter', 'producer']:
                return f"✅ **'{movie_title}'** was {forward_text} **{formatted_results}**."
            elif relation == 'genre':
                genre_word = 'genre is' if len(results) == 1 else 'genres are'
                return f"✅ The {genre_word} of **'{movie_title}'**: **{formatted_results}**."
            elif relation == 'publication_date':
                # Extract year from date
                year = results[0].split('-')[0] if '-' in results[0] else results[0]
                return f"✅ **'{movie_title}'** was released in **{year}**."
            elif relation in ['country_of_origin', 'country']:
                return f"✅ **'{movie_title}'** is from **{formatted_results}**."
            elif relation in ['original_language_of_film_or_tv_show', 'language', 'original_language']:
                return f"✅ **'{movie_title}'** is in **{formatted_results}**."
            elif relation == 'award_received':
                award_word = 'award' if len(results) == 1 else 'awards'
                return f"✅ **'{movie_title}'** received the following {award_word}: **{formatted_results}**."
            elif relation == 'rating':
                return f"✅ The rating of **'{movie_title}'** is **{formatted_results}**."
            else:
                return f"✅ The {relation.replace('_', ' ')} of **'{movie_title}'** is **{formatted_results}**."
        
        # Reverse query (Person → Movies)
        elif person_name:
            _, reverse_text = relation_map.get(relation, (relation.replace('_', ' '), relation.replace('_', ' ')))
            
            if len(results) == 1:
                return f"✅ **{person_name}** {reverse_text} **{results[0]}**."
            else:
                formatted_results = AnswerFormatter.format_list(results)
                film_word = 'film' if len(results) == 1 else 'films'
                return f"✅ **{person_name}** {reverse_text} **{len(results)} {film_word}**: {formatted_results}."
        
        # Default format
        formatted_results = AnswerFormatter.format_list(results)
        return f"✅ **Results:** {formatted_results}"
    
    @staticmethod
    def format_embedding_response(result: str, entity_type: Optional[str] = None) -> str:
        """
        Format an embedding-based response.
        
        Args:
            result: Result from embedding search
            entity_type: Type of entity (optional)
            
        Returns:
            Formatted response indicating embedding source
        """
        type_suffix = f" (type: {entity_type})" if entity_type else ""
        return f"🔢 **Embedding-based answer:** {result}{type_suffix}\n\n_Note: This answer was found using semantic similarity in the embedding space._"
    
    @staticmethod
    def format_hybrid_response(factual: str, embedding: str) -> str:
        """
        Format a hybrid response showing both factual and embedding results.
        
        Args:
            factual: Factual SPARQL result
            embedding: Embedding-based result
            
        Returns:
            Combined formatted response
        """
        return f"""**📊 Factual Answer (from knowledge graph):**
{factual}

**🔢 Embedding-based Answer (from semantic similarity):**
{embedding}"""
    
    @staticmethod
    def format_error(message: str, suggestions: Optional[List[str]] = None) -> str:
        """
        Format an error message with helpful suggestions.
        
        Args:
            message: Error message
            suggestions: Optional list of suggestions
            
        Returns:
            Formatted error message
        """
        response = f"❌ {message}"
        
        if suggestions:
            response += "\n\n**💡 Suggestions:**\n"
            response += "\n".join([f"• {suggestion}" for suggestion in suggestions])
        
        return response
