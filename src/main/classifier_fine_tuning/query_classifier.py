"""
Robust rule-based query classifier for movie knowledge graph queries.
Provides deterministic classification without relying on LLMs.
"""

import re
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass


class QuestionType(str, Enum):
    """Types of questions the system can handle."""
    FACTUAL = "factual"
    MULTIMEDIA = "multimedia"
    RECOMMENDATION = "recommendation"


@dataclass
class ClassificationResult:
    """Result of query classification."""
    question_type: QuestionType
    confidence: float  # 0.0 to 1.0
    matched_patterns: List[str]
    extracted_info: Dict[str, any]


class RobustQueryClassifier:
    """
    Rule-based classifier using pattern matching, keyword analysis, and linguistic features.
    Much more reliable than LLM classification for structured domains.
    """
    
    def __init__(self):
        """Initialize classifier with comprehensive pattern sets."""
        self._setup_patterns()
        self._setup_keywords()
        self._setup_linguistic_features()
    
    def _setup_patterns(self):
        """Define regex patterns for each question type."""
        
        # FACTUAL question patterns (most common)
        self.factual_patterns = [
            # Who questions
            (r'\bwho\s+(directed|made|wrote|produced|acted\s+in|starred\s+in|was\s+in)', 0.95),
            (r'\bwho\s+is\s+the\s+(director|actor|writer|producer|screenwriter)', 0.95),
            (r'\bwho\s+plays?\s+', 0.90),
            
            # What questions
            (r'\bwhat\s+is\s+the\s+(genre|rating|release\s+date|country)', 0.95),
            (r'\bwhat\s+(genre|type|kind|category)', 0.90),
            (r'\bwhat\s+year\s+was', 0.95),
            (r'\bwhat\s+awards?\s+did', 0.90),
            
            # When questions
            (r'\bwhen\s+(was|did|were)\s+.+\s+(released|made|produced|come\s+out)', 0.95),
            (r'\brelease\s+date', 0.90),
            
            # Where questions
            (r'\bwhere\s+was\s+.+\s+(filmed|made|produced)', 0.90),
            (r'\bcountry\s+of\s+origin', 0.90),
            
            # Which questions (usually factual)
            (r'\bwhich\s+(movies?|films?)\s+', 0.85),
            (r'\bwhich\s+.+\s+(directed|acted|starred)', 0.90),
            
            # Information requests
            (r'\b(tell|give)\s+me\s+(about|info|information)', 0.85),
            (r'\bfind\s+(movies?|films?)\s+(about|with|by|from)', 0.90),
            (r'\blist\s+(all\s+)?(movies?|films?)', 0.90),
            (r'\bsearch\s+for\s+', 0.85),
            
            # Similarity/comparison (still factual - using knowledge graph)
            (r'\b(similar|like|comparable)\s+to\s+', 0.85),
            (r'\bmovies?\s+like\s+', 0.85),
            (r'\b(same|similar)\s+(genre|style|type)', 0.85),
        ]
        
        # MULTIMEDIA patterns (explicit visual requests)
        self.multimedia_patterns = [
            (r'\b(show|display|view|see)\s+(me\s+)?(a\s+)?(picture|image|photo|poster)', 0.95),
            (r'\bwhat\s+does\s+.+\s+look\s+like', 0.90),
            (r'\b(picture|image|photo|poster)\s+of\s+', 0.90),
            (r'\bvisual\s+(representation|display)', 0.85),
            (r'\bcan\s+i\s+see\s+(a\s+)?(picture|image|photo)', 0.90),
        ]
        
        # RECOMMENDATION patterns (asking for suggestions)
        self.recommendation_patterns = [
            (r'\b(recommend|suggest|propose)\s+(me\s+)?(a\s+)?(movie|film|something)', 0.95),
            (r'\bwhat\s+should\s+i\s+watch', 0.95),
            (r'\bany\s+(good\s+)?(movie|film)\s+(suggestions?|recommendations?)', 0.95),
            (r'\bgive\s+me\s+.+\s+(suggestion|recommendation)', 0.90),
            (r'\bhelp\s+me\s+(find|choose|pick|select)\s+.+\s+to\s+watch', 0.90),
            (r'\b(looking\s+for|want)\s+.+\s+(recommendation|suggestion)', 0.85),
            (r'\bwhat\s+(are\s+)?(some|any)\s+good\s+(movies?|films?)', 0.85),
        ]
    
    def _setup_keywords(self):
        """Define keyword sets for each question type."""
        
        # Factual keywords
        self.factual_keywords = {
            'high_confidence': {
                'director', 'directed', 'screenwriter', 'writer', 'wrote',
                'actor', 'actress', 'cast', 'starred', 'acted',
                'producer', 'produced', 'genre', 'rating', 'release',
                'released', 'year', 'date', 'country', 'award', 'awards'
            },
            'medium_confidence': {
                'who', 'what', 'when', 'where', 'which',
                'movie', 'film', 'about', 'info', 'information',
                'find', 'search', 'list', 'show', 'tell'
            }
        }
        
        # Multimedia keywords
        self.multimedia_keywords = {
            'high_confidence': {
                'picture', 'image', 'photo', 'poster', 'visual',
                'display', 'show', 'see', 'view', 'look'
            }
        }
        
        # Recommendation keywords
        self.recommendation_keywords = {
            'high_confidence': {
                'recommend', 'suggest', 'recommendation', 'suggestion',
                'should', 'watch', 'good', 'best', 'help', 'choose', 'pick'
            }
        }
    
    def _setup_linguistic_features(self):
        """Define linguistic features for classification."""
        
        # Modal verbs indicating recommendations
        self.modal_verbs = {'should', 'could', 'would', 'might'}
        
        # Imperative verbs (often factual requests)
        self.imperative_verbs = {
            'tell', 'show', 'give', 'find', 'list', 'get', 'display'
        }
        
        # Question words
        self.question_words = {
            'who', 'what', 'when', 'where', 'which', 'why', 'how'
        }
    
    def classify(self, query: str) -> ClassificationResult:
        """
        Classify query using multi-stage analysis.
        
        Args:
            query: Natural language query
            
        Returns:
            ClassificationResult with type, confidence, and metadata
        """
        query_lower = query.lower().strip()
        
        # Stage 1: Pattern matching (highest priority)
        pattern_result = self._classify_by_patterns(query_lower)
        if pattern_result and pattern_result.confidence > 0.85:
            return pattern_result
        
        # Stage 2: Keyword analysis
        keyword_result = self._classify_by_keywords(query_lower)
        if keyword_result and keyword_result.confidence > 0.75:
            return keyword_result
        
        # Stage 3: Linguistic features
        linguistic_result = self._classify_by_linguistics(query_lower)
        if linguistic_result:
            return linguistic_result
        
        # Default: factual (most queries are factual)
        return ClassificationResult(
            question_type=QuestionType.FACTUAL,
            confidence=0.6,
            matched_patterns=['default_factual'],
            extracted_info={'reason': 'Default classification for movie queries'}
        )
    
    def _classify_by_patterns(self, query: str) -> Optional[ClassificationResult]:
        """Classify using regex patterns."""
        
        # Check multimedia patterns first (most specific)
        for pattern, confidence in self.multimedia_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return ClassificationResult(
                    question_type=QuestionType.MULTIMEDIA,
                    confidence=confidence,
                    matched_patterns=[pattern],
                    extracted_info={'pattern_type': 'multimedia'}
                )
        
        # Check recommendation patterns
        for pattern, confidence in self.recommendation_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return ClassificationResult(
                    question_type=QuestionType.RECOMMENDATION,
                    confidence=confidence,
                    matched_patterns=[pattern],
                    extracted_info={'pattern_type': 'recommendation'}
                )
        
        # Check factual patterns
        matched_factual = []
        max_confidence = 0.0
        for pattern, confidence in self.factual_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                matched_factual.append(pattern)
                max_confidence = max(max_confidence, confidence)
        
        if matched_factual:
            return ClassificationResult(
                question_type=QuestionType.FACTUAL,
                confidence=max_confidence,
                matched_patterns=matched_factual,
                extracted_info={'pattern_type': 'factual'}
            )
        
        return None
    
    def _classify_by_keywords(self, query: str) -> Optional[ClassificationResult]:
        """Classify using keyword analysis."""
        
        words = set(query.split())
        
        # Count keyword matches for each type
        scores = {
            QuestionType.FACTUAL: 0.0,
            QuestionType.MULTIMEDIA: 0.0,
            QuestionType.RECOMMENDATION: 0.0
        }
        
        # Factual keywords
        for word in words:
            if word in self.factual_keywords['high_confidence']:
                scores[QuestionType.FACTUAL] += 0.15
            elif word in self.factual_keywords['medium_confidence']:
                scores[QuestionType.FACTUAL] += 0.05
        
        # Multimedia keywords
        for word in words:
            if word in self.multimedia_keywords['high_confidence']:
                scores[QuestionType.MULTIMEDIA] += 0.2
        
        # Recommendation keywords
        for word in words:
            if word in self.recommendation_keywords['high_confidence']:
                scores[QuestionType.RECOMMENDATION] += 0.2
        
        # Check for multimedia combination (e.g., "show picture")
        if any(w in words for w in ['show', 'display', 'see']) and \
           any(w in words for w in ['picture', 'image', 'photo', 'poster']):
            scores[QuestionType.MULTIMEDIA] += 0.3
        
        # Find highest score
        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]
        
        if max_score > 0.3:  # Threshold for keyword-based classification
            return ClassificationResult(
                question_type=max_type,
                confidence=min(max_score, 0.9),
                matched_patterns=['keyword_analysis'],
                extracted_info={'keyword_scores': scores}
            )
        
        return None
    
    def _classify_by_linguistics(self, query: str) -> Optional[ClassificationResult]:
        """Classify using linguistic features."""
        
        words = query.split()
        
        # Check for modal verbs (recommendations)
        if any(word in self.modal_verbs for word in words):
            if 'watch' in words or 'see' in words:
                return ClassificationResult(
                    question_type=QuestionType.RECOMMENDATION,
                    confidence=0.75,
                    matched_patterns=['modal_verb_analysis'],
                    extracted_info={'feature': 'modal_verb'}
                )
        
        # Check question word at start (usually factual)
        if words and words[0] in self.question_words:
            return ClassificationResult(
                question_type=QuestionType.FACTUAL,
                confidence=0.70,
                matched_patterns=['question_word_analysis'],
                extracted_info={'feature': 'question_word', 'word': words[0]}
            )
        
        # Check imperative verbs (factual requests)
        if words and words[0] in self.imperative_verbs:
            # Exception: "show me a picture" is multimedia
            if 'picture' in words or 'image' in words or 'photo' in words:
                return ClassificationResult(
                    question_type=QuestionType.MULTIMEDIA,
                    confidence=0.85,
                    matched_patterns=['imperative_multimedia'],
                    extracted_info={'feature': 'imperative_visual'}
                )
            
            return ClassificationResult(
                question_type=QuestionType.FACTUAL,
                confidence=0.70,
                matched_patterns=['imperative_analysis'],
                extracted_info={'feature': 'imperative_verb', 'verb': words[0]}
            )
        
        return None
    
    def get_classification_explanation(self, result: ClassificationResult) -> str:
        """Generate human-readable explanation of classification."""
        
        explanation = f"Classified as {result.question_type.value} "
        explanation += f"(confidence: {result.confidence:.0%})\n"
        
        if result.matched_patterns:
            explanation += f"Matched patterns: {', '.join(result.matched_patterns[:3])}\n"
        
        if result.extracted_info:
            explanation += f"Additional info: {result.extracted_info}"
        
        return explanation