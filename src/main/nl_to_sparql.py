import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from typing import Optional, Dict, List, Any
import re
from pydantic import BaseModel, Field

# Note: sparql-llm is a SPARQL examples loader from SHACL ontology,
# not an LLM-based SPARQL generator. We use direct llama-cpp-python instead.

# Try to import llama-cpp-python for direct LLM integration
try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False
    print("⚠️  llama-cpp-python not available. Install with: pip install llama-cpp-python")

from src.config import (
    NL2SPARQL_METHOD, 
    NL2SPARQL_LLM_MODEL_PATH,
    NL2SPARQL_LLM_TEMPERATURE,
    NL2SPARQL_LLM_MAX_TOKENS,
    NL2SPARQL_LLM_CONTEXT_LENGTH
)

# Import SPARQLHandler for validation
from src.main.sparql_handler import SPARQLHandler


class SPARQLQuery(BaseModel):
    """Generated SPARQL query from natural language."""
    query: str = Field(
        description="The generated SPARQL query"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    explanation: str = Field(
        description="Brief explanation of the query logic"
    )


class NLToSPARQL:
    """Converts natural language questions to SPARQL queries using LLM or rule-based patterns."""
    
    def __init__(
        self, 
        use_transformer: bool = None,
        model_name: str = None,
        use_spbert: bool = False,
        method: str = None,
        sparql_handler: Optional[SPARQLHandler] = None
    ):
        """
        Initialize the NL to SPARQL converter.
        
        Args:
            use_transformer: Deprecated
            model_name: Deprecated
            use_spbert: Deprecated
            method: Conversion method - "direct-llm" or "rule-based"
            sparql_handler: Optional SPARQLHandler instance for validation (to avoid loading graph twice)
        """
        # Determine method
        if method is None:
            method = NL2SPARQL_METHOD
        self.method = method
        
        # Initialize based on method
        self.llm = None
        
        # Use provided SPARQL handler or create a new one
        # If provided, we avoid loading the graph twice
        self.sparql_validator = sparql_handler if sparql_handler is not None else SPARQLHandler()
        
        # Rule-based patterns (always available as fallback)
        self._setup_patterns()
        self._setup_schema()
        
        # Initialize chosen method
        if self.method == "direct-llm":
            self._initialize_direct_llm()
        else:
            # Rule-based only
            print("ℹ️  Using rule-based SPARQL generation only (no model)")

    def _initialize_direct_llm(self):
        """Initialize direct llama-cpp-python integration."""
        if not LLAMACPP_AVAILABLE:
            print("⚠️  llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            print("   Falling back to rule-based approach")
            self.method = "rule-based"
            return
        
        try:
            print(f"📥 Loading model for SPARQL generation...")
            print(f"    Model: {NL2SPARQL_LLM_MODEL_PATH}")
            
            # Check if model exists
            if not os.path.exists(NL2SPARQL_LLM_MODEL_PATH):
                print(f"❌ Model file not found: {NL2SPARQL_LLM_MODEL_PATH}")
                print("   Falling back to rule-based approach")
                self.method = "rule-based"
                return
            
            # Load model directly with llama-cpp-python
            self.llm = Llama(
                model_path=NL2SPARQL_LLM_MODEL_PATH,
                n_ctx=NL2SPARQL_LLM_CONTEXT_LENGTH,
                n_threads=4,
                verbose=False
            )
            
            print(f"✅ Model loaded successfully (Deepseek-Coder-1.3B)")
            print(f"    Using direct few-shot prompting for SPARQL generation")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("   Falling back to rule-based approach")
            import traceback
            traceback.print_exc()
            self.method = "rule-based"
            self.llm = None

    def _setup_patterns(self):
        """Set up rule-based patterns for NL to SPARQL conversion."""
        self.patterns = [
            {
                'pattern': r'who (?:directed|was the director of) (.+)',
                'type': 'director',
                'sparql_template': '''SELECT ?directorName ?directorUri WHERE {{
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^{movie}$", "i")) .
  ?movieUri wdt:P57 ?directorUri .
  ?directorUri rdfs:label ?directorName .
  FILTER(LANG(?directorName) = "en" || LANG(?directorName) = "")
}}'''
            },
            {
                'pattern': r'who (?:acted in|starred in|was in) (.+)',
                'type': 'cast',
                'sparql_template': '''SELECT ?actorName ?actorUri WHERE {{
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^{movie}$", "i")) .
  ?movieUri wdt:P161 ?actorUri .
  ?actorUri rdfs:label ?actorName .
  FILTER(LANG(?actorName) = "en" || LANG(?actorName) = "")
}}'''
            },
            {
                'pattern': r'what (?:is|was) the genre of (.+)',
                'type': 'genre',
                'sparql_template': '''SELECT ?genreName ?genreUri WHERE {{
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^{movie}$", "i")) .
  ?movieUri wdt:P136 ?genreUri .
  ?genreUri rdfs:label ?genreName .
  FILTER(LANG(?genreName) = "en" || LANG(?genreName) = "")
}}'''
            },
            {
                'pattern': r'when (?:was|did) (.+?) (?:released|come out)',
                'type': 'release_date',
                'sparql_template': '''SELECT ?date WHERE {{
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^{movie}$", "i")) .
  ?movieUri wdt:P577 ?date .
}}'''
            },
            {
                'pattern': r'what (?:movies|films) did (.+?) direct',
                'type': 'director_filmography',
                'sparql_template': '''SELECT ?movieLabel ?movieUri WHERE {{
  ?personUri rdfs:label ?personLabel .
  FILTER(regex(str(?personLabel), "^{person}$", "i")) .
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri wdt:P57 ?personUri .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(LANG(?movieLabel) = "en" || LANG(?movieLabel) = "")
}}
ORDER BY ?movieLabel'''
            },
            {
                'pattern': r'did (.+?) direct (.+)',
                'type': 'director_verification',
                'sparql_template': '''ASK WHERE {{
  ?personUri rdfs:label ?personLabel .
  FILTER(regex(str(?personLabel), "^{person}$", "i")) .
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^{movie}$", "i")) .
  ?movieUri wdt:P57 ?personUri .
}}'''
            }
        ]

    def _setup_schema(self):
        """Set up the movie ontology schema for NL2SPARQL prompting."""
        self.schema_info = {
            'prefixes': {
                'ddis': 'http://ddis.ch/atai/',
                'wd': 'http://www.wikidata.org/entity/',
                'wdt': 'http://www.wikidata.org/prop/direct/',
                'schema': 'http://schema.org/',
                'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
                'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
            },
            'classes': [
                'wd:Q11424',  # Movie
                'wd:Q5',      # Human/Person
            ],
            'properties': {
                'wdt:P31': {'description': 'instance of (type)', 'example': 'wd:Q11424 for movie'},
                'wdt:P57': {'description': 'director', 'property_id': 'P57'},
                'wdt:P161': {'description': 'cast member (actor)', 'property_id': 'P161'},
                'wdt:P58': {'description': 'screenwriter', 'property_id': 'P58'},
                'wdt:P162': {'description': 'producer', 'property_id': 'P162'},
                'wdt:P577': {'description': 'publication date (release date)', 'property_id': 'P577'},
                'wdt:P136': {'description': 'genre', 'property_id': 'P136'},
                'wdt:P495': {'description': 'country of origin', 'property_id': 'P495'},
                'wdt:P166': {'description': 'award received', 'property_id': 'P166'},
                'ddis:rating': {'description': 'MPAA rating or movie rating'},
                'rdfs:label': {'description': 'label/name of entity'},
            }
        }
        
        # Schema description for NL2SPARQL prompting
        self.schema_description = self._format_schema_for_nl2sparql()
    
    def _format_schema_for_nl2sparql(self) -> str:
        """Format schema for NL2SPARQL prompting."""
        schema_lines = []
        
        # Add prefixes
        for prefix, uri in self.schema_info['prefixes'].items():
            schema_lines.append(f"PREFIX {prefix}: <{uri}>")
        
        # Add schema description
        schema_lines.append("")
        schema_lines.append("# Movie Knowledge Graph Schema (Wikidata-based):")
        schema_lines.append("# Key entity types:")
        schema_lines.append("#   - wd:Q11424 = Movie (use: ?movie wdt:P31 wd:Q11424)")
        schema_lines.append("#   - Items have labels via rdfs:label")
        schema_lines.append("")
        schema_lines.append("# Key properties:")
        schema_lines.append("#   - wdt:P31 = instance of (type)")
        schema_lines.append("#   - wdt:P57 = director")
        schema_lines.append("#   - wdt:P161 = cast member (actor)")
        schema_lines.append("#   - wdt:P58 = screenwriter")
        schema_lines.append("#   - wdt:P162 = producer")
        schema_lines.append("#   - wdt:P577 = publication date")
        schema_lines.append("#   - wdt:P136 = genre")
        schema_lines.append("#   - wdt:P495 = country of origin")
        schema_lines.append("#   - wdt:P166 = award received")
        schema_lines.append("#   - ddis:rating = movie rating")
        schema_lines.append("#   - rdfs:label = name/label")
        schema_lines.append("")
        schema_lines.append("# Important patterns:")
        schema_lines.append("#   - Use rdfs:label for matching: ?item rdfs:label \"Title\"")
        schema_lines.append("#   - Return both label and item: SELECT ?label ?item")
        schema_lines.append("#   - Use FILTER(regex(?label, \"Title\", \"i\")) for partial matches")
        
        return "\n".join(schema_lines)
    
    def _get_ontology_description(self) -> str:
        """Get MINIMAL ontology description for small model."""
        return """Movie Knowledge Graph - Key Facts:
- Movies: wdt:P31 wd:Q11424
- Labels: rdfs:label
- Properties: P57=director, P161=actor, P136=genre, P162=producer, P577=date
"""
    
    def _get_few_shot_examples(self) -> List[Dict[str, str]]:
        """Get few-shot examples - IMPROVED with diverse patterns."""
        return [
            {
                "question": "Who directed The Matrix?",
                "reasoning": "Forward query: movie → director. Need to find director of 'The Matrix'.",
                "sparql": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?directorName ?directorUri WHERE {
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^The Matrix$", "i")) .
  ?movieUri wdt:P57 ?directorUri .
  ?directorUri rdfs:label ?directorName .
  FILTER(LANG(?directorName) = "en" || LANG(?directorName) = "")
}"""
            },
            {
                "question": "What movies did Christopher Nolan direct?",
                "reasoning": "Reverse query: person → movies. Need to find movies directed by 'Christopher Nolan'.",
                "sparql": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?movieLabel ?movieUri WHERE {
  ?personUri rdfs:label ?personLabel .
  FILTER(regex(str(?personLabel), "^Christopher Nolan$", "i")) .
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri wdt:P57 ?personUri .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(LANG(?movieLabel) = "en" || LANG(?movieLabel) = "")
}
ORDER BY ?movieLabel"""
            },
            {
                "question": "What is the genre of Inception?",
                "reasoning": "Forward query: movie → genre. Need to find genre of 'Inception'.",
                "sparql": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?genreName ?genreUri WHERE {
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^Inception$", "i")) .
  ?movieUri wdt:P136 ?genreUri .
  ?genreUri rdfs:label ?genreName .
  FILTER(LANG(?genreName) = "en" || LANG(?genreName) = "")
}"""
            },
            {
                "question": "Did Christopher Nolan direct Inception?",
                "reasoning": "Verification: check if relationship exists.",
                "sparql": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

ASK WHERE {
  ?personUri rdfs:label ?personLabel .
  FILTER(regex(str(?personLabel), "^Christopher Nolan$", "i")) .
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^Inception$", "i")) .
  ?movieUri wdt:P57 ?personUri .
}"""
            }
        ]

    def _get_few_shot_examples_by_pattern(self, pattern_type: str) -> List[Dict[str, str]]:
        """Get pattern-specific few-shot examples based on classification."""
        
        # Map from QueryPattern types to our classification labels
        pattern_map = {
            'forward_director': [
                {
                    "question": "Who directed The Matrix?",
                    "sparql": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?directorName ?directorUri WHERE {
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^The Matrix$", "i")) .
  ?movieUri wdt:P57 ?directorUri .
  ?directorUri rdfs:label ?directorName .
  FILTER(LANG(?directorName) = "en" || LANG(?directorName) = "")
}"""
                }
            ],
            'forward_cast_member': [
                {
                    "question": "Who acted in Inception?",
                    "sparql": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?actorName ?actorUri WHERE {
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^Inception$", "i")) .
  ?movieUri wdt:P161 ?actorUri .
  ?actorUri rdfs:label ?actorName .
  FILTER(LANG(?actorName) = "en" || LANG(?actorName) = "")
}"""
                }
            ],
            'forward_genre': [
                {
                    "question": "What is the genre of Inception?",
                    "sparql": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?genreName ?genreUri WHERE {
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^Inception$", "i")) .
  ?movieUri wdt:P136 ?genreUri .
  ?genreUri rdfs:label ?genreName .
  FILTER(LANG(?genreName) = "en" || LANG(?genreName) = "")
}"""
                }
            ],
            'reverse_director': [
                {
                    "question": "What movies did Christopher Nolan direct?",
                    "sparql": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?movieLabel ?movieUri WHERE {
  ?personUri rdfs:label ?personLabel .
  FILTER(regex(str(?personLabel), "^Christopher Nolan$", "i")) .
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri wdt:P57 ?personUri .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(LANG(?movieLabel) = "en" || LANG(?movieLabel) = "")
}
ORDER BY ?movieLabel"""
                }
            ],
            'verification_director': [
                {
                    "question": "Did Christopher Nolan direct Inception?",
                    "sparql": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

ASK WHERE {
  ?personUri rdfs:label ?personLabel .
  FILTER(regex(str(?personLabel), "^Christopher Nolan$", "i")) .
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^Inception$", "i")) .
  ?movieUri wdt:P57 ?personUri .
}"""
                }
            ]
        }
        
        # Return examples for the specific pattern, or default to director examples
        return pattern_map.get(pattern_type, pattern_map['forward_director'])

    def _create_few_shot_prompt(self, question: str, pattern=None) -> str:
        """Create pattern-specific few-shot prompt for 1.3B model."""
        
        # Determine pattern type from QueryPattern if provided
        pattern_label = None
        if pattern:
            # Map QueryPattern to classification label
            if pattern.pattern_type == 'forward':
                pattern_label = f"forward_{pattern.relation}"
            elif pattern.pattern_type == 'reverse':
                pattern_label = f"reverse_{pattern.relation}"
            elif pattern.pattern_type == 'verification':
                pattern_label = f"verification_{pattern.relation}"
        
        # Get pattern-specific examples
        if pattern_label:
            examples = self._get_few_shot_examples_by_pattern(pattern_label)
        else:
            # Fallback to original heuristic selection
            examples = self._get_few_shot_examples()
            selected_examples = []
            
            question_lower = question.lower()
            if any(word in question_lower for word in ['did', 'was', 'is', 'does', 'has']):
                selected_examples.append(examples[3])  # Verification
                selected_examples.append(examples[0])  # Forward
            elif any(word in question_lower for word in ['what movies', 'which films', 'films did']):
                selected_examples.append(examples[1])  # Reverse
                selected_examples.append(examples[0])  # Forward
            else:
                selected_examples.append(examples[0])  # Forward director
                selected_examples.append(examples[2])  # Forward genre
            
            examples = selected_examples
        
        examples_text = "\n\n".join([
            f"Question: {ex['question']}\nSPARQL:\n{ex['sparql']}"
            for ex in examples
        ])
        
        # FIX: Double the curly braces in the rules to escape them in f-string
        prompt = f"""Generate SPARQL for movie questions.

{self._get_ontology_description()}

RULES:
1. End triple patterns with period (.)
2. ALWAYS use FILTER for text matching: ?var rdfs:label ?varLabel . FILTER(regex(str(?varLabel), "^MovieTitle$", "i"))
3. Use proper English title case: "The Bridge on the River Kwai"
4. NEVER use exact match like: ?var rdfs:label "Text" (database is case-sensitive)
5. For YES/NO questions, use ASK queries
6. For reverse queries (person→movies), put person FILTER first

EXAMPLES:

{examples_text}

Question: {question}
SPARQL:
"""
        
        return prompt

    def _direct_llm_convert(self, question: str, pattern=None) -> Optional[SPARQLQuery]:
        """
        Convert question to SPARQL using direct LLM generation.
        
        Args:
            question: Natural language question
            pattern: Optional QueryPattern from QueryAnalyzer
            
        Returns:
            SPARQLQuery or None if generation fails
        """
        try:
            # Create few-shot prompt
            prompt = self._create_few_shot_prompt(question, pattern)
            
            print(f"[LLM] Generating SPARQL with DeepSeek...")
            
            # Generate with llama-cpp-python
            response = self.llm(
                prompt,
                max_tokens=NL2SPARQL_LLM_MAX_TOKENS,
                temperature=NL2SPARQL_LLM_TEMPERATURE,
                stop=["Question:", "\n\n\n"],  # Stop at next question or triple newline
            )
            
            # Extract text from response
            if isinstance(response, dict):
                output = response.get('choices', [{}])[0].get('text', '')
            else:
                output = str(response)
            
            print(f"[LLM] Generated {len(output)} characters")
            
            # Extract SPARQL from output
            sparql_query = self._extract_sparql_from_output(output)
            
            # Post-process the query
            sparql_query = self._postprocess_sparql(sparql_query)
            
            # Validate structure
            if not self._is_valid_sparql_structure(sparql_query):
                print(f"[LLM] ❌ Invalid SPARQL structure")
                return None
            
            # Validate and secure
            validation = self._validate_and_secure_sparql(sparql_query)
            
            if not validation['valid']:
                print(f"[LLM] ❌ Validation failed: {validation['message']}")
                return None
            
            print(f"[LLM] ✅ Generated valid SPARQL")
            
            return SPARQLQuery(
                query=validation['cleaned_query'],
                confidence=0.85,
                explanation=f"Generated by DeepSeek-Coder-1.3B using few-shot prompting"
            )
            
        except Exception as e:
            print(f"[LLM] ❌ Generation error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _rule_based_convert(self, question: str) -> Optional[SPARQLQuery]:
        """
        Convert question using rule-based pattern matching.
        
        Args:
            question: Natural language question
            
        Returns:
            SPARQLQuery or None if no pattern matches
        """
        question_lower = question.lower().strip()
        
        for pattern_def in self.patterns:
            match = re.search(pattern_def['pattern'], question_lower, re.IGNORECASE)
            if match:
                # Extract entities from match groups
                if pattern_def['type'] == 'director_verification':
                    person = self._normalize_proper_name(match.group(1))
                    movie = self._normalize_proper_name(match.group(2))
                    sparql = pattern_def['sparql_template'].format(person=person, movie=movie)
                elif pattern_def['type'] == 'director_filmography':
                    person = self._normalize_proper_name(match.group(1))
                    sparql = pattern_def['sparql_template'].format(person=person)
                else:
                    movie = self._normalize_proper_name(match.group(1))
                    sparql = pattern_def['sparql_template'].format(movie=movie)
                
                # Add prefixes
                prefixes = """PREFIX ddis: <http://ddis.ch/atai/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

"""
                sparql = prefixes + sparql
                
                return SPARQLQuery(
                    query=sparql,
                    confidence=0.8,
                    explanation=f"Rule-based conversion using {pattern_def['type']} pattern"
                )
        
        return None
    
    def _normalize_proper_name(self, text: str) -> str:
        """
        Normalize text to proper English title case.
        Handles articles, prepositions, and special cases.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Properly capitalized text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Articles and prepositions that should be lowercase (unless first/last word)
        lowercase_words = {
            'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor',
            'on', 'at', 'to', 'from', 'by', 'in', 'of', 'with'
        }
        
        words = text.split()
        result = []
        
        for i, word in enumerate(words):
            # Always capitalize first and last word
            if i == 0 or i == len(words) - 1:
                result.append(word.capitalize())
            # Lowercase articles/prepositions in middle
            elif word.lower() in lowercase_words:
                result.append(word.lower())
            # Capitalize other words
            else:
                result.append(word.capitalize())
        
        return ' '.join(result)
    
    def _escape_regex_special_chars(self, text: str) -> str:
        """Escape special regex characters in text."""
        special_chars = r'\.[]{}()*+?|^$'
        for char in special_chars:
            text = text.replace(char, '\\' + char)
        return text

    def convert(self, question: str, pattern=None) -> SPARQLQuery:
        """
        Convert a natural language question to a SPARQL query.
        NOW ACCEPTS: Optional pattern from QueryAnalyzer for better validation.
        
        Args:
            question: Natural language question
            pattern: Optional QueryPattern from QueryAnalyzer
        """
        
        # Step 1: Try DeepSeek LLM FIRST if enabled
        if self.method == "direct-llm" and self.llm is not None:
            print(f"[NL2SPARQL] Trying DeepSeek LLM first...")
            llm_result = self._direct_llm_convert(question)
            
            # Validate against pattern if provided
            if llm_result and pattern:
                if self._validate_sparql_for_pattern(llm_result.query, pattern):
                    print(f"[NL2SPARQL] ✅ DeepSeek result validated against pattern")
                    return llm_result
                else:
                    print(f"[NL2SPARQL] ⚠️  DeepSeek result doesn't match expected pattern, trying fallback...")
                    llm_result = None
            elif llm_result:
                # No pattern to validate against, use result as-is
                print(f"[NL2SPARQL] ✅ Using DeepSeek result")
                return llm_result
            else:
                print(f"[NL2SPARQL] ⚠️  DeepSeek failed, trying rule-based fallback...")
        
        # Step 2: Try rule-based approach as fallback
        print(f"[NL2SPARQL] Trying rule-based approach...")
        rule_result = self._rule_based_convert(question)
        
        if rule_result:
            print(f"[NL2SPARQL] Validating rule-based query...")
            validation = self._validate_and_secure_sparql(rule_result.query)
            
            if validation['valid']:
                print(f"[NL2SPARQL] ✅ Using rule-based result")
                rule_result.query = validation['cleaned_query']
                rule_result.explanation += f" Validated: {validation['message']}"
                return rule_result
            else:
                print(f"[NL2SPARQL] ❌ Rule-based validation failed: {validation['message']}")
        
        # Fallback - no valid query generated
        print("[NL2SPARQL] ❌ Both DeepSeek and rule-based failed")
        return SPARQLQuery(
            query="# Could not generate valid query",
            confidence=0.0,
            explanation=f"Could not generate a valid SPARQL query for: {question}"
        )

    def validate_question(self, question: str) -> bool:
        """
        Validate if the question is appropriate for the movie knowledge graph.
        Returns True if valid, False otherwise.
        """
        # Basic validation - check if question is non-empty and reasonable length
        if not question or len(question.strip()) < 3:
            return False
        
        # Check if it contains movie-related keywords (basic heuristic)
        movie_keywords = [
            'movie', 'film', 'director', 'actor', 'actress', 'release',
            'genre', 'screenwriter', 'writer', 'star', 'cast', 'rating'
        ]
        
        question_lower = question.lower()
        
        # Check for question words
        question_words = ['who', 'what', 'when', 'where', 'which', 'how']
        has_question_word = any(word in question_lower for word in question_words)
        
        # Check for movie-related content or patterns
        has_movie_context = any(keyword in question_lower for keyword in movie_keywords)
        
        # Check if matches any pattern
        matches_pattern = any(
            re.search(pattern['pattern'], question_lower, re.IGNORECASE)
            for pattern in self.patterns
        )
        
        return has_question_word or has_movie_context or matches_pattern

    def _extract_sparql_from_output(self, output: str) -> str:
        """Extract SPARQL query from model output with multiple strategies."""
        print(f"[Extract] Full output length: {len(output)} chars")
        print(f"[Extract] First 500 chars: {output[:500]}")
        
        # Strategy 1: Try to find complete SELECT...WHERE{...} block with proper nesting
        # Look for balanced braces (.)
        select_match = re.search(
            r'(PREFIX[^\n]*\n)*\s*SELECT\s+[^{]+WHERE\s*\{',
            output,
            re.IGNORECASE | re.DOTALL
        )
        
        if select_match:
            # Find the matching closing brace
            start_idx = select_match.start()
            brace_count = 0
            in_where = False
            end_idx = len(output)
            
            for i, char in enumerate(output[select_match.start():], start=select_match.start()):
                if char == '{':
                    brace_count += 1
                    in_where = True
                elif char == '}':
                    brace_count -= 1
                    if in_where and brace_count == 0:
                        end_idx = i + 1
                        break
            
            complete_query = output[start_idx:end_idx]
            
            # Validate that it's complete
            if complete_query.count('{') == complete_query.count('}'):
                print(f"[Extract] ✅ Found complete query with balanced braces")
                return complete_query
            else:
                print(f"[Extract] ⚠️  Query has unbalanced braces")
        
        # Strategy 2: Look for query between code markers
        code_block = re.search(r'```(?:sparql)?\s*([^`]+)```', output, re.IGNORECASE | re.DOTALL)
        if code_block:
            print(f"[Extract] Found code block")
            return code_block.group(1).strip()
        
        # Strategy 3: Try to extract line by line with proper brace counting
        sparql_lines = []
        in_query = False
        brace_count = 0
        
        for line in output.split('\n'):
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                continue
            
            # Check if we're entering WHERE clause
            if re.search(r'\bWHERE\s*\{', line_stripped, re.IGNORECASE):
                in_query = True
            
            # Check if we're exiting WHERE clause
            if in_query and line_stripped.strip() == '}':
                in_query = False
            
            # If we're in WHERE clause and line contains a triple pattern
            if in_query:
                # Check if line looks like a triple pattern (has predicate)
                if re.search(r'\s+(wdt:|rdfs:|wd:|ddis:|rdf:|schema:|a\s)', line_stripped):
                    sparql_lines.append(line_stripped)
        
        if sparql_lines:
            extracted = '\n'.join(sparql_lines)
            print(f"[Extract] Line-by-line extraction: {len(extracted)} chars")
            return extracted
        
        # Fallback: return the whole output
        print(f"[Extract] ⚠️  Using fallback (whole output)")
        return output.strip()
    
    def _validate_and_secure_sparql(self, query: str) -> Dict[str, Any]:
        """
        Validate SPARQL query using SPARQLHandler validation.
        
        Args:
            query: The SPARQL query string
            
        Returns:
            Dict with 'valid' (bool), 'message' (str), and 'cleaned_query' (str)
        """
        try:
            # Use SPARQLHandler's validation
            validation_result = self.sparql_validator.validate_query(query)
            
            if validation_result['valid']:
                return {
                    'valid': True,
                    'message': validation_result['message'],
                    'cleaned_query': query.strip(),
                    'query_type': validation_result.get('query_type', 'SELECT')
                }
            else:
                return {
                    'valid': False,
                    'message': validation_result['message'],
                    'cleaned_query': None,
                    'violation_type': validation_result.get('violation_type')
                }
        except Exception as e:
            return {
                'valid': False,
                'message': f"Validation error: {str(e)}",
                'cleaned_query': None
            }
    
    def _is_valid_sparql_structure(self, query: str) -> bool:
        """Check if generated text has valid SPARQL structure."""
        query_upper = query.upper()
        
        # Must have a query type
        has_query_type = any(qt in query_upper for qt in ['SELECT', 'ASK', 'CONSTRUCT', 'DESCRIBE'])
        
        # Must have WHERE clause or graph pattern
        has_where = 'WHERE' in query_upper or '{' in query
        
        # Should not be empty or just comments
        has_content = len(query.strip()) > 10
        
        return has_query_type and has_where and has_content
    
    def _postprocess_sparql(self, query: str) -> str:
        """
        Post-process generated SPARQL query to ensure correct prefixes and proper name capitalization.
        Handles all SPARQL query types correctly.
        """
        # Remove any markdown code blocks
        query = re.sub(r'```sparql\s*', '', query)
        query = re.sub(r'```\s*', '', query)
        
        # Remove any leading/trailing quotes
        query = query.strip('"\'')
        
        # Replace old movie ontology prefixes
        replacements = {
            r'mo:hasDirector': 'wdt:P57',
            r'mo:hasActor': 'wdt:P161',
            r'mo:hasScreenwriter': 'wdt:P58',
            r'mo:hasTitle': 'rdfs:label',
            r'mo:releaseDate': 'wdt:P577',
            r'mo:hasGenre': 'wdt:P136',
            r'mo:mpaaRating': 'ddis:rating',
            r'mo:personName': 'rdfs:label',
            r'mo:Movie': 'wd:Q11424',
            r'mo:Person': 'wd:Q5',
        }
        
        for old, new in replacements.items():
            query = re.sub(old, new, query, flags=re.IGNORECASE)
        
        # ✅ CRITICAL: Validate FILTER logic FIRST
        if 'wdt:P136' in query:  # Genre query
            if re.search(r'FILTER\(regex\(str\(\?genre[LI]', query, re.IGNORECASE):
                print("[Postprocess] ⚠️  Detected wrong FILTER - fixing genre query")
                query = re.sub(
                    r'FILTER\(regex\(str\(\?(genre[LI]\w*)\)',
                    r'FILTER(regex(str(?movieLabel)',
                    query,
                    flags=re.IGNORECASE
                )
        
        if 'wdt:P57' in query:  # Director query
            if re.search(r'FILTER\(regex\(str\(\?director[LI]', query, re.IGNORECASE):
                print("[Postprocess] ⚠️  Detected wrong FILTER - fixing director query")
                query = re.sub(
                    r'FILTER\(regex\(str\(\?(director[LI]\w*)\)',
                    r'FILTER(regex(str(?movieLabel)',
                    query,
                    flags=re.IGNORECASE
                )
        
        # ✅ Remove "Movie " prefix if accidentally added
        query = re.sub(
            r'FILTER\(regex\(str\(\?movieLabel\),\s*"[\^]?Movie\s+([^"]+)"',
            r'FILTER(regex(str(?movieLabel), "^\1"',
            query,
            flags=re.IGNORECASE
        )
        
        # ✅ NEW: Convert ALL exact rdfs:label matches to case-insensitive FILTER
        # This handles both user input variations AND database capitalization differences
        def replace_exact_label_match(match):
            """Replace exact label match with case-insensitive FILTER for ANY entity."""
            var_name = match.group(1)
            label_value = match.group(2)
            
            # ✅ CRITICAL: Use _normalize_proper_name for correct English title case
            label_title = self._normalize_proper_name(label_value)
            
            print(f"[Postprocess] Normalizing '{label_value}' → '{label_title}'")
            
            # Escape regex special characters
            label_escaped = self._escape_regex_special_chars(label_title)
            
            # Create a label variable name
            var_base = var_name.strip('?')
            label_var = f"?{var_base}Label"
            
            # ✅ CRITICAL: Ensure regex has ^ and $ anchors for exact match
            if not label_escaped.startswith('^'):
                label_escaped = '^' + label_escaped
            if not label_escaped.endswith('$'):
                label_escaped = label_escaped + '$'
            
            print(f"[Postprocess] FILTER pattern: {label_escaped}")
            
            # Generate replacement with case-insensitive FILTER
            return f'{var_name} rdfs:label {label_var} .\n  FILTER(regex(str({label_var}), "{label_escaped}", "i")) .'
        
        # Find and replace ALL exact rdfs:label matches (movies, actors, countries, awards, etc.)
        query = re.sub(
            r'(\?\w+)\s+rdfs:label\s+"([^"]+)"\s*\.',
            replace_exact_label_match,
            query,
            flags=re.IGNORECASE
        )
        
        # ✅ ALSO convert simple equality FILTERs to regex FILTERs
        # Pattern: ?var rdfs:label ?varLabel . FILTER(?varLabel = "Text")
        # Should be: ?var rdfs:label ?varLabel . FILTER(regex(str(?varLabel), "^Text$", "i"))
        def replace_equality_filter(match):
            """Replace FILTER equality with case-insensitive regex."""
            var_name = match.group(1)
            label_value = match.group(2)
            
            # ✅ CRITICAL: Use _normalize_proper_name for correct English title case
            label_title = self._normalize_proper_name(label_value)
            
            # Escape regex special characters
            label_escaped = self._escape_regex_special_chars(label_title)
            
            # ✅ CRITICAL: Ensure regex has ^ and $ anchors
            if not label_escaped.startswith('^'):
                label_escaped = '^' + label_escaped
            if not label_escaped.endswith('$'):
                label_escaped = label_escaped + '$'
            
            # Return case-insensitive regex FILTER
            return f'FILTER(regex(str({var_name}), "{label_escaped}", "i"))'
        
        # Replace FILTER(?var = "Text") with FILTER(regex(str(?var), "^Text$", "i"))
        query = re.sub(
            r'FILTER\s*\(\s*(\?\w+)\s*=\s*"([^"]+)"\s*\)',
            replace_equality_filter,
            query,
            flags=re.IGNORECASE
        )
        
        # ✅ IMPORTANT: Normalize line breaks FIRST before processing
        query = re.sub(r'\r\n', '\n', query)
        
        # ✅ Smart period addition - process line by line
        SPARQL_KEYWORDS = [
            'PREFIX', 'SELECT', 'ASK', 'CONSTRUCT', 'DESCRIBE',
            'WHERE', 'FROM', 'OPTIONAL', 'UNION', 'GRAPH',
            'ORDER', 'BY', 'GROUP', 'HAVING', 'LIMIT', 'OFFSET',
            'DISTINCT', 'REDUCED', 'BASE'
        ]
        
        lines = []
        in_where_clause = False
        
        for line in query.split('\n'):
            stripped = line.rstrip()
            
            # Skip empty lines
            if not stripped:
                lines.append(stripped)
                continue
            
            # Check if we're entering WHERE clause
            if re.search(r'\bWHERE\s*\{', stripped, re.IGNORECASE):
                in_where_clause = True
                lines.append(stripped)
                continue
            
            # Check if we're exiting WHERE clause
            if in_where_clause and stripped.strip() == '}':
                in_where_clause = False
                lines.append(stripped)
                continue
            
            # Skip if already has period, comma, brace
            if stripped.endswith(('.', ',', '{', '}')):
                lines.append(stripped)
                continue
            
            # Check if line is a SPARQL keyword (don't add period)
            first_word = stripped.strip().split()[0] if stripped.strip() else ""
            if any(first_word.upper().startswith(kw) for kw in SPARQL_KEYWORDS):
                lines.append(stripped)
                continue
            
            # ✅ CRITICAL: Handle FILTER statements specially
            if 'FILTER' in stripped.upper():
                # FILTER should end with period
                if not stripped.endswith('.'):
                    stripped += ' .'
                lines.append(stripped)
                continue
            
            # If we're in WHERE clause and line contains a triple pattern
            if in_where_clause:
                # Check if line looks like a triple pattern (has predicate)
                if re.search(r'\s+(wdt:|rdfs:|wd:|ddis:|rdf:|schema:|a\s)', stripped):
                    # Add period if it doesn't have special ending
                    if not re.search(r'[{},.;]$', stripped):
                        stripped += ' .'
            
            lines.append(stripped)
        
        query = '\n'.join(lines)
        
        # ✅ CRITICAL FIX: Ensure ALL FILTER(regex(...)) patterns use _normalize_proper_name
        # AND ensure they have proper anchors
        def fix_filter_capitalization(match):
            """Fix capitalization in FILTER regex patterns using proper English title case."""
            var_name = match.group(1)
            label_text = match.group(2).strip('^$')  # Remove existing anchors
            
            # ✅ Use _normalize_proper_name for correct capitalization
            normalized_text = self._normalize_proper_name(label_text)
            
            print(f"[Postprocess] Final FILTER: '{label_text}' → '{normalized_text}'")
            
            # ✅ Always add anchors for exact match
            return f'FILTER(regex(str(?{var_name}), "^{normalized_text}$", "i")) .'
        
        query = re.sub(
            r'FILTER\(regex\(str\(\?(\w+)\),\s*"([^"]+)",\s*"i"\)\)(\s*\.)?',
            fix_filter_capitalization,
            query,
            flags=re.IGNORECASE
        )
        
        # ✅ Clean up any double periods
        query = re.sub(r'\.\s*\.', '.', query)
        
        # ✅ Remove incorrectly placed periods after SELECT/WHERE/ORDER BY/etc
        query = re.sub(r'^(\s*(?:SELECT|ASK|CONSTRUCT|DESCRIBE)\s+[^\.{]+?)\s*\.$', r'\1', query, flags=re.MULTILINE | re.IGNORECASE)
        query = re.sub(r'^(\s*(?:WHERE|FROM|OPTIONAL)\s*)\s*\.$', r'\1', query, flags=re.MULTILINE | re.IGNORECASE)
        query = re.sub(r'^(\s*(?:ORDER\s+BY|GROUP\s+BY|LIMIT|OFFSET)[^\.]*?)\s*\.$', r'\1', query, flags=re.MULTILINE | re.IGNORECASE)
        
        # Ensure proper prefixes
        if 'PREFIX' not in query.upper():
            prefixes = """PREFIX ddis: <http://ddis.ch/atai/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

"""
            query = prefixes + query
        
        # Clean up excessive whitespace
        query = re.sub(r'\n{3,}', '\n\n', query)
        query = query.strip()
        
        print(f"[Postprocess] Final query preview:")
        print(query[:500])
        
        return query
    
    def _validate_sparql_for_pattern(self, sparql: str, pattern) -> bool:
        """
        Validate that generated SPARQL matches the expected pattern structure.
        
        Args:
            sparql: Generated SPARQL query
            pattern: QueryPattern from QueryAnalyzer
            
        Returns:
            True if structure matches expected pattern
        """
        sparql_upper = sparql.upper()
        
        # Forward queries should have movie type constraint and specific property
        if pattern.pattern_type == 'forward':
            has_movie_type = 'WDT:P31' in sparql_upper and 'WD:Q11424' in sparql_upper
            has_movie_label = 'RDFS:LABEL' in sparql_upper and '?MOVIELABEL' in sparql_upper.replace(' ', '')
            
            # Check for expected property based on relation
            property_map = {
                'director': 'P57',
                'cast_member': 'P161',
                'genre': 'P136',
                'publication_date': 'P577',
                'screenwriter': 'P58',
                'producer': 'P162',
                'rating': 'RATING'
            }
            expected_prop = property_map.get(pattern.relation, '')
            has_property = expected_prop in sparql_upper if expected_prop else True
            
            if not (has_movie_type and has_movie_label and has_property):
                print(f"[Validation] Forward query missing expected structure:")
                print(f"  Movie type: {has_movie_type}, Label: {has_movie_label}, Property {expected_prop}: {has_property}")
                return False
        
        # Reverse queries should have person label and movie type
        elif pattern.pattern_type == 'reverse':
            has_person_label = '?PERSONLABEL' in sparql_upper.replace(' ', '')
            has_movie_type = 'WDT:P31' in sparql_upper and 'WD:Q11424' in sparql_upper
            
            if not (has_person_label and has_movie_type):
                print(f"[Validation] Reverse query missing expected structure:")
                print(f"  Person label: {has_person_label}, Movie type: {has_movie_type}")
                return False
        
        # Verification queries should be ASK queries
        elif pattern.pattern_type == 'verification':
            if not sparql_upper.strip().startswith('PREFIX') or 'ASK' not in sparql_upper:
                print(f"[Validation] Verification query should be ASK query")
                return False
        
        return True
