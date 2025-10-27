"""
Query Validator - Guardrails for detecting and rejecting unrelated questions.
Ensures queries are about movies and within the system's capabilities.
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of query validation."""
    is_valid: bool
    reason: Optional[str] = None
    category: Optional[str] = None  # 'movie_related', 'off_topic', 'unsafe', etc.


class QueryValidator:
    """
    Validates queries to ensure they are movie-related and appropriate.
    Provides guardrails against off-topic, unsafe, or unanswerable questions.
    """
    
    def __init__(self):
        """Initialize the query validator with patterns and keywords."""
        self._setup_movie_indicators()
        self._setup_off_topic_patterns()
        self._setup_unsafe_patterns()
    
    def _setup_movie_indicators(self):
        """Set up indicators that suggest movie-related queries."""
        
        # Strong movie indicators (high confidence)
        self.strong_movie_keywords = {
            'movie', 'movies', 'film', 'films', 'cinema',
            'director', 'directed', 'actor', 'actress', 'cast',
            'screenwriter', 'screenplay', 'producer', 'produced',
            'genre', 'genres', 'release', 'released', 'premiere',
            'starring', 'starred', 'acted', 'performance',
            'blockbuster', 'hollywood', 'sequel', 'prequel',
            'box office', 'oscar', 'award', 'nominated',
            'rating', 'mpaa', 'pg-13', 'r-rated',
            'trailer', 'plot', 'storyline', 'character'
        }
        
        # Weak movie indicators (context-dependent)
        self.weak_movie_keywords = {
            'watch', 'watched', 'watching', 'show', 'entertainment',
            'recommend', 'suggestion', 'good', 'best', 'favorite',
            'year', 'date', 'when', 'who', 'what', 'where'
        }
        
        # Famous movie titles (partial list for detection)
        self.known_movie_patterns = [
            r'\b(star wars|the godfather|pulp fiction|inception|matrix|avatar)\b',
            r'\b(lord of the rings|harry potter|marvel|batman|superman)\b',
            r'\b(titanic|forrest gump|shawshank|dark knight|fight club)\b',
        ]
        
        # Movie-related question patterns
        self.movie_question_patterns = [
            r'who (directed|wrote|produced|starred in|acted in)',
            r'what (genre|rating|year|date) (is|was)',
            r'when (was|did) .+ (released|made|come out)',
            r'(movies?|films?) (about|like|similar to|with|starring)',
            r'(recommend|suggest).+(movie|film)',
            r'(show|find|list).+(movie|film)',
        ]
    
    def _setup_off_topic_patterns(self):
        """Set up patterns for clearly off-topic queries."""
        
        # Categories of off-topic queries
        self.off_topic_patterns = {
            'mathematics': [
                r'\d+\s*[\+\-\*\/\^]\s*\d+',  # Math expressions
                r'\bcalculate\b|\bsolve\b|\bequation\b',
                r'\bderivative\b|\bintegral\b|\balgebra\b',
            ],
            'programming': [
                r'\bcode\b|\bprogram\b|\bfunction\b|\balgorithm\b',
                r'\bpython\b|\bjava\b|\bc\+\+\b|\bjavascript\b',
                r'\bdebug\b|\bcompile\b|\bsyntax error\b',
                r'(write|create|implement).+(function|class|script)',
            ],
            'general_knowledge': [
                r'\bcapital of\b|\bpopulation of\b',
                r'\bpresident of\b|\bprime minister\b',
                r'\b(what|who) is.+(definition|meaning)\b',
                r'\bhow (tall|old|many) is\b',
            ],
            'weather': [
                r'\bweather\b|\btemperature\b|\bforecast\b',
                r'\brain\b|\bsnow\b|\bsunny\b|\bcloudy\b',
            ],
            'sports': [
                r'\bfootball\b|\bsoccer\b|\bbasketball\b|\btennis\b',
                r'\b(nfl|nba|fifa|olympics)\b',
                r'\bscore\b|\bmatch\b|\bgame\b|\btournament\b',
            ],
            'health': [
                r'\bsymptoms?\b|\btreatment\b|\bmedic(ine|al)\b',
                r'\bdoctor\b|\bhospital\b|\bdisease\b',
            ],
            'cooking': [
                r'\brecipe\b|\bcook\b|\bbake\b|\bingredients?\b',
                r'\bhow to (make|prepare|cook)',
            ],
            'personal': [
                r'\bhow are you\b|\bhow do you feel\b',
                r"\bwhat('?s| is) your (name|age|favorite)\b",
                r'\bdo you (like|love|hate|prefer)\b',
            ],
        }
    
    def _setup_unsafe_patterns(self):
        """Set up patterns for unsafe or inappropriate content."""
        
        self.unsafe_patterns = [
            # Harmful instructions
            r'\b(how to|ways to) (harm|hurt|kill|attack|bomb)\b',
            r'\b(make|create|build) (weapon|explosive|bomb)\b',
            
            # Illegal activities
            r'\b(hack|crack|steal|pirate|torrent)\b',
            r'\b(drugs?|cocaine|heroin|meth)\b',
            
            # Hate speech indicators
            r'\b(hate|racist|sexist|homophobic)\b',
            
            # Sexual content
            r'\b(porn|sex|nude|xxx)\b',
        ]
    
    def validate(self, query: str) -> ValidationResult:
        """
        Validate a query against all guardrails.
        
        Args:
            query: User's natural language query
            
        Returns:
            ValidationResult with is_valid flag and reason
        """
        if not query or not query.strip():
            return ValidationResult(
                is_valid=False,
                reason="Query is empty",
                category='empty'
            )
        
        query_lower = query.lower().strip()
        
        # Check 1: Unsafe content (highest priority)
        unsafe_check = self._check_unsafe_content(query_lower)
        if not unsafe_check.is_valid:
            return unsafe_check
        
        # Check 2: Off-topic detection
        off_topic_check = self._check_off_topic(query_lower)
        if not off_topic_check.is_valid:
            return off_topic_check
        
        # Check 3: Movie-related validation
        movie_check = self._check_movie_related(query_lower)
        if not movie_check.is_valid:
            return movie_check
        
        # All checks passed
        return ValidationResult(
            is_valid=True,
            category='movie_related'
        )
    
    def _check_unsafe_content(self, query: str) -> ValidationResult:
        """Check for unsafe or inappropriate content."""
        for pattern in self.unsafe_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return ValidationResult(
                    is_valid=False,
                    reason="Your query contains inappropriate content. Please ask movie-related questions only.",
                    category='unsafe'
                )
        
        return ValidationResult(is_valid=True)
    
    def _check_off_topic(self, query: str) -> ValidationResult:
        """Check if query is clearly off-topic."""
        for category, patterns in self.off_topic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    # Exception: if query also contains strong movie keywords, allow it
                    if self._has_strong_movie_indicators(query):
                        continue
                    
                    return ValidationResult(
                        is_valid=False,
                        reason=self._get_off_topic_message(category),
                        category=f'off_topic_{category}'
                    )
        
        return ValidationResult(is_valid=True)
    
    def _check_movie_related(self, query: str) -> ValidationResult:
        """Check if query is sufficiently movie-related."""
        
        # Strong indicators (any one is enough)
        if self._has_strong_movie_indicators(query):
            return ValidationResult(is_valid=True, category='movie_related')
        
        # Movie title patterns
        for pattern in self.known_movie_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return ValidationResult(is_valid=True, category='movie_related')
        
        # Movie question patterns
        for pattern in self.movie_question_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return ValidationResult(is_valid=True, category='movie_related')
        
        # Weak indicators need multiple matches
        weak_matches = sum(1 for kw in self.weak_movie_keywords if kw in query)
        if weak_matches >= 2:
            return ValidationResult(is_valid=True, category='movie_related')
        
        # Query doesn't seem movie-related
        return ValidationResult(
            is_valid=False,
            reason="I'm a movie information assistant. I can only answer questions about movies, actors, directors, and related topics.",
            category='not_movie_related'
        )
    
    def _has_strong_movie_indicators(self, query: str) -> bool:
        """Check if query has strong movie indicators."""
        return any(keyword in query for keyword in self.strong_movie_keywords)
    
    def _get_off_topic_message(self, category: str) -> str:
        """Get appropriate message for off-topic category."""
        messages = {
            'mathematics': "I'm a movie information assistant and can't help with math problems. Please ask about movies instead!",
            'programming': "I'm specialized in movies, not programming. Try asking about films, actors, or directors!",
            'general_knowledge': "I focus on movie-related information. Please ask about films, actors, directors, or movie details!",
            'weather': "I can't help with weather information. I'm here to answer questions about movies!",
            'sports': "I'm a movie assistant, not a sports expert. Ask me about films instead!",
            'health': "I can't provide medical advice. I'm here to help with movie-related questions!",
            'cooking': "I'm not a recipe assistant. Try asking about movies, actors, or directors!",
            'personal': "I'm an AI focused on movie information. Let's talk about films instead!",
        }
        
        return messages.get(
            category,
            "I'm a movie information assistant. Please ask questions about movies, actors, directors, or related topics."
        )
    
    def get_standard_rejection_message(self) -> str:
        """Get the standard message for rejected queries."""
        return (
            "🎬 **Movie Information Assistant**\n\n"
            "I'm specialized in answering questions about movies! I can help you with:\n\n"
            "• **Movie details**: directors, actors, release dates, genres\n"
            "• **People in film**: actors, directors, screenwriters, producers\n"
            "• **Search queries**: find movies by genre, year, or topic\n"
            "• **Recommendations**: suggest movies based on your preferences\n\n"
            "Please ask a movie-related question and I'll be happy to help! 🍿"
        )