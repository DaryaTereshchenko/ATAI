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

        # ✅ NEW: Retry configuration
        self.max_retries = 2  # Try up to 3 times total (initial + 2 retries)
        self.temperature_increment = 0.15  # Increase temperature by this amount per retry

    def _initialize_direct_llm(self):
        """Initialize direct llama-cpp-python integration."""
        if not LLAMACPP_AVAILABLE:
            print("⚠️  llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            print("   Falling back to rule-based approach")
            self.method = "rule-based"
            return
        
        try:
            # print(f"📥 Loading model for SPARQL generation...")  # REMOVED
            # print(f"    Model: {NL2SPARQL_LLM_MODEL_PATH}")  # REMOVED
            
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
            
            # print(f"✅ Model loaded successfully (Deepseek-Coder-1.3B)")  # REMOVED
            # print(f"    Using direct few-shot prompting for SPARQL generation")  # REMOVED
            
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
                # ✅ NEW
                'wdt:P364': {'description': 'original language', 'property_id': 'P364'},
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
        schema_lines.append("#   - wdt:P495 = country of origin")  # ✅ NEW
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
        """Get MINIMAL ontology description for small model - ✅ NOW with person query guidance."""
        return """Knowledge Graph - Key Facts:
- Entities have types: wdt:P31
- Entities have labels: rdfs:label
- Properties connect entities: wdt:P57, wdt:P161, wdt:P136, etc.

CRITICAL PROPERTY DISAMBIGUATION:
- For MOVIE COUNTRY queries → use wdt:P495 (country of origin)
- For PERSON NATIONALITY/BIRTHPLACE queries → use wdt:P27 (country of citizenship)
- For MOVIE LANGUAGE queries → use wdt:P364 (original language)
  Example: "What language is 'Parasite' in?"
  
- For FILMING LOCATION → use wdt:P915
- For DIRECTOR of movie → use wdt:P57
- For ACTORS/CAST → use wdt:P161

Common patterns: ?subject wdt:Pxxx ?object
"""
    
    def _get_few_shot_examples(self) -> List[Dict[str, str]]:
        """Get few-shot examples - ✅ NOW: Mix of movie and generic examples."""
        return [
            {
                "question": "Who directed The Matrix?",
                "reasoning": "Forward query: entity → property value.",
                "sparql": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?directorName WHERE {
  ?entity rdfs:label ?label .
  FILTER(regex(str(?label), "^The Matrix$", "i")) .
  ?entity wdt:P57 ?director .
  ?director rdfs:label ?directorName .
  FILTER(LANG(?directorName) = "en" || LANG(?directorName) = "")
}"""
            },
            {
                "question": "From what country is 'Aro Tolbukhin'?",
                "reasoning": "Forward query: entity → country property.",
                "sparql": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?countryName WHERE {
  ?entity rdfs:label ?label .
  FILTER(regex(str(?label), "^Aro Tolbukhin", "i")) .
  ?entity wdt:P495 ?country .
  ?country rdfs:label ?countryName .
  FILTER(LANG(?countryName) = "en" || LANG(?countryName) = "")
}"""
            },
            {
                "question": "What did Christopher Nolan direct?",
                "reasoning": "Reverse query: find entities related to a person.",
                "sparql": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?entityLabel WHERE {
  ?person rdfs:label ?personLabel .
  FILTER(regex(str(?personLabel), "^Christopher Nolan$", "i")) .
  ?entity wdt:P57 ?person .
  ?entity rdfs:label ?entityLabel .
  FILTER(LANG(?entityLabel) = "en" || LANG(?entityLabel) = "")
}"""
            },
        ]

    def _get_few_shot_examples_by_pattern(self, pattern_type: str) -> List[Dict[str, str]]:
        """Get pattern-specific few-shot examples based on classification."""
        
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
            # ✅ NEW: Country of origin examples
            'forward_country_of_origin': [
                {
                    "question": "From what country is the movie 'Aro Tolbukhin. En la mente del asesino'?",
                    "reasoning": "Country of origin query - use P495 (NOT P27 which is citizenship)",
                    "sparql": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?countryName ?countryUri WHERE {
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^Aro Tolbukhin. En la mente del asesino$", "i")) .
  ?movieUri wdt:P495 ?countryUri .
  ?countryUri rdfs:label ?countryName .
  FILTER(LANG(?countryName) = "en" || LANG(?countryName) = "")
}"""
                },
                {
                    "question": "What country is 'The Bridge on the River Kwai' from?",
                    "reasoning": "Movie country query - MUST use P495 for country of origin",
                    "sparql": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?countryName ?countryUri WHERE {
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^The Bridge on the River Kwai$", "i")) .
  ?movieUri wdt:P495 ?countryUri .
  ?countryUri rdfs:label ?countryName .
  FILTER(LANG(?countryName) = "en" || LANG(?countryName) = "")
}"""
                },
                {
                    "question": "Which country produced 'Parasite'?",
                    "reasoning": "Production country = P495 (country of origin), NOT other country properties",
                    "sparql": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?countryName ?countryUri WHERE {
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label "Parasite" .
  ?movieUri wdt:P495 ?countryUri .
  ?countryUri rdfs:label ?countryName .
  FILTER(LANG(?countryName) = "en" || LANG(?countryName) = "")
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
  FILTER(regex(str(?personLabel), "^{person}$", "i")) .
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^{movie}$", "i")) .
  ?movieUri wdt:P57 ?personUri .
}"""
                }
            ],
            # ✅ NEW: Language patterns
            'forward_original_language': [
                {
                    "question": "What is the original language of 'Parasite'?",
                    "reasoning": "Language query - use P364 (original language)",
                    "sparql": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?languageName ?languageUri WHERE {
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^Parasite$", "i")) .
  ?movieUri wdt:P364 ?languageUri .
  ?languageUri rdfs:label ?languageName .
  FILTER(LANG(?languageName) = "en" || LANG(?languageName) = "")
}"""
                },
                {
                    "question": "In which language is 'Amélie' filmed?",
                    "reasoning": "Language query - use P364",
                    "sparql": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?languageName ?languageUri WHERE {
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^Amélie$", "i")) .
  ?movieUri wdt:P364 ?languageUri .
  ?languageUri rdfs:label ?languageName .
  FILTER(LANG(?languageName) = "en" || LANG(?languageName) = "")
}"""
                }
            ],
            
            'forward_language': [  # Alias
                {
                    "question": "What language is spoken in 'Life Is Beautiful'?",
                    "reasoning": "Language query - use P364",
                    "sparql": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?languageName ?languageUri WHERE {
  ?movieUri wdt:P31 wd:Q11424 .
  ?movieUri rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^Life Is Beautiful$", "i")) .
  ?movieUri wdt:P364 ?languageUri .
  ?languageUri rdfs:label ?languageName .
  FILTER(LANG(?languageName) = "en" || LANG(?languageName) = "")
}"""
                }
            ],
            
            # ...rest of existing patterns...
        }
        
        return pattern_map.get(pattern_type, pattern_map.get('forward_director', []))  # Fixed: Added fallback

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
            # Fallback to original heuristic selection with country detection
            all_examples = self._get_few_shot_examples()
            selected_examples = []
            
            question_lower = question.lower()
            
            # ✅ FIX: Safer example selection with bounds checking
            if 'country' in question_lower:
                selected_examples.append(all_examples[1] if len(all_examples) > 1 else all_examples[0])  # Country example
                if len(all_examples) > 0:
                    selected_examples.append(all_examples[0])  # Forward director
            elif any(word in question_lower for word in ['did', 'was', 'is', 'does', 'has']):
                # Verification pattern - use examples 0 and 1
                if len(all_examples) > 0:
                    selected_examples.append(all_examples[0])  # Forward
                if len(all_examples) > 1:
                    selected_examples.append(all_examples[1])  # Another forward
            elif any(word in question_lower for word in ['what movies', 'which films', 'films did']):
                selected_examples.append(all_examples[2] if len(all_examples) > 2 else all_examples[0])  # Reverse
                if len(all_examples) > 0:
                    selected_examples.append(all_examples[0])  # Forward
            else:
                # Default: use first two examples
                if len(all_examples) > 0:
                    selected_examples.append(all_examples[0])  # Forward director
                if len(all_examples) > 1:
                    selected_examples.append(all_examples[1])  # Forward country
            
            examples = selected_examples
        
        examples_text = "\n\n".join([
            f"Question: {ex['question']}\nSPARQL:\n{ex['sparql']}"
            for ex in examples
        ])
        
        # ✅ ENHANCED: More explicit rules about property selection
        prompt = f"""Generate SPARQL for movie questions.

{self._get_ontology_description()}

CRITICAL RULES:
1. End triple patterns with period (.)
2. ALWAYS use FILTER for text matching: ?var rdfs:label ?varLabel . FILTER(regex(str(?varLabel), "^Title$", "i"))
3. Use proper English title case: "The Bridge on the River Kwai"
4. NEVER use exact match like: ?var rdfs:label "Text" (database is case-sensitive)
5. For YES/NO questions, use ASK queries
6. For reverse queries (person→movies), put person FILTER first

PROPERTY DISAMBIGUATION (CRITICAL):
7. For MOVIE COUNTRY queries → ALWAYS use wdt:P495 (country of origin)
   Keywords: "movie", "film", "produced", "released", "from what country"
   
8. For PERSON NATIONALITY/BIRTHPLACE queries → ALWAYS use wdt:P27 (country of citizenship)
   Keywords: "born", "birthplace", "citizenship", "nationality", "is from"
   Example: "Where was Bruce Willis born?" → use P27
   
9. For FILMING LOCATION → use wdt:P915
10. When query says "from what country" about a MOVIE → P495 ONLY
11. When query says "where was [person] born" → P27 ONLY

EXAMPLES:

{examples_text}

Question: {question}
SPARQL:
"""
        
        return prompt

    def _direct_llm_convert(self, question: str, pattern=None) -> Optional[SPARQLQuery]:
        """
        Convert question to SPARQL using direct LLM generation.
        ✅ NOW: Includes retry mechanism with temperature escalation.
        
        Args:
            question: Natural language question
            pattern: Optional QueryPattern from QueryAnalyzer
            
        Returns:
            SPARQLQuery or None if generation fails
        """
        base_temperature = NL2SPARQL_LLM_TEMPERATURE
        
        for attempt in range(self.max_retries + 1):
            # Calculate temperature for this attempt
            current_temperature = base_temperature + (attempt * self.temperature_increment)
            current_temperature = min(current_temperature, 1.0)  # Cap at 1.0
            
            attempt_label = f"attempt {attempt + 1}/{self.max_retries + 1}"
            print(f"[LLM] Generating SPARQL ({attempt_label}, temp={current_temperature:.2f})...")
            
            try:
                # Create few-shot prompt
                prompt = self._create_few_shot_prompt(question, pattern)
                
                # ✅ NEW: Log the prompt for country queries
                if pattern and pattern.relation == 'country_of_origin':
                    print(f"\n[LLM] 🌍 COUNTRY QUERY PROMPT:")
                    print(f"─" * 80)
                    print(prompt)
                    print(f"─" * 80)
                
                # Generate with llama-cpp-python
                response = self.llm(
                    prompt,
                    max_tokens=NL2SPARQL_LLM_MAX_TOKENS,
                    temperature=current_temperature,
                    stop=["Question:", "\n\n\n"],
                )
                
                # Extract text from response
                if isinstance(response, dict):
                    output = response.get('choices', [{}])[0].get('text', '')
                else:
                    output = str(response)
                
                print(f"[LLM] Generated {len(output)} characters")
                
                # ✅ NEW: Log raw LLM output for country queries
                if pattern and pattern.relation == 'country_of_origin':
                    print(f"\n[LLM] 🌍 COUNTRY QUERY RAW OUTPUT:")
                    print(f"─" * 80)
                    print(output[:500] if len(output) > 500 else output)
                    print(f"─" * 80)
                
                # Extract SPARQL from output
                sparql_query = self._extract_sparql_from_output(output)
                
                # ✅ NEW: Log extracted SPARQL
                if pattern and pattern.relation == 'country_of_origin':
                    print(f"\n[LLM] 🌍 EXTRACTED SPARQL:")
                    print(f"─" * 80)
                    print(sparql_query[:500] if len(sparql_query) > 500 else sparql_query)
                    print(f"─" * 80)
                
                # Post-process the query
                sparql_query = self._postprocess_sparql(sparql_query)
                
                # ✅ NEW: Log post-processed SPARQL
                if pattern and pattern.relation == 'country_of_origin':
                    print(f"\n[LLM] 🌍 POST-PROCESSED SPARQL:")
                    print(f"─" * 80)
                    print(sparql_query)
                    print(f"─" * 80)
                
                # Validate structure
                if not self._is_valid_sparql_structure(sparql_query):
                    print(f"[LLM] ❌ Invalid SPARQL structure on {attempt_label}")
                    if attempt < self.max_retries:
                        print(f"[LLM] 🔄 Retrying with higher temperature...")
                        continue
                    else:
                        print(f"[LLM] ❌ Max retries reached")
                        return None
                
                # Validate and secure
                validation = self._validate_and_secure_sparql(sparql_query)
                
                if not validation['valid']:
                    print(f"[LLM] ❌ Validation failed on {attempt_label}: {validation['message']}")
                    if attempt < self.max_retries:
                        print(f"[LLM] 🔄 Retrying with higher temperature...")
                        continue
                    else:
                        print(f"[LLM] ❌ Max retries reached")
                        return None
                
                print(f"[LLM] ✅ Generated valid SPARQL on {attempt_label}")
                
                return SPARQLQuery(
                    query=validation['cleaned_query'],
                    confidence=0.85 - (attempt * 0.05),  # Decrease confidence for retries
                    explanation=f"Generated by DeepSeek-Coder-1.3B using few-shot prompting ({attempt_label}, temp={current_temperature:.2f})"
                )
                
            except Exception as e:
                print(f"[LLM] ❌ Generation error on {attempt_label}: {e}")
                if attempt < self.max_retries:
                    print(f"[LLM] 🔄 Retrying with higher temperature...")
                    continue
                else:
                    print(f"[LLM] ❌ Max retries reached")
                    import traceback
                    traceback.print_exc()
                    return None
    
    def _rule_based_convert(self, question: str) -> Optional[SPARQLQuery]:
        """
        Convert question using rule-based pattern matching.
        """
        question_lower = question.lower().strip()
        
        for pattern_def in self.patterns:
            match = re.search(pattern_def['pattern'], question_lower, re.IGNORECASE)
            if match:
                # Extract entities from match groups - ✅ NO NORMALIZATION
                if pattern_def['type'] == 'director_verification':
                    person = match.group(1).strip()  # ✅ Keep as-is
                    movie = match.group(2).strip()   # ✅ Keep as-is
                    sparql = pattern_def['sparql_template'].format(person=person, movie=movie)
                elif pattern_def['type'] == 'director_filmography':
                    person = match.group(1).strip()  # ✅ Keep as-is
                    sparql = pattern_def['sparql_template'].format(person=person)
                else:
                    movie = match.group(1).strip()  # ✅ Keep as-is
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
        Post-process generated SPARQL query.
        ✅ NOW: Uses EXACT case-sensitive matching (no LCASE).
        """
        # ✅ CRITICAL: Replace smart quotes with regular quotes FIRST
        query = query.replace('"', '"').replace('"', '"')
        query = query.replace(''', "'").replace(''', "'")
        
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
        
        # ✅ CRITICAL FIX: Convert case-insensitive FILTER to exact match
        # Pattern: FILTER(regex(str(?var), "text", "i")) or FILTER(LCASE(STR(?var)) = LCASE("text"))
        # Replace with: ?var rdfs:label "text"
        
        def convert_filter_to_exact_match(match):
            """Convert case-insensitive FILTER to exact label match."""
            var_name = match.group(1)
            label_text = match.group(2).strip('^$')
            
            print(f"[Postprocess] Converting to exact match: ?{var_name} rdfs:label \"{label_text}\"")
            
            # Return exact label match (case-sensitive)
            return f'{var_name} rdfs:label "{label_text}" .'
        
        # Replace regex FILTER patterns
        query = re.sub(
            r'FILTER\(regex\(str\(\?(\w+)\),\s*"[\^]?([^"]+)[\$]?",\s*"i"\)\)\s*\.',
            convert_filter_to_exact_match,
            query,
            flags=re.IGNORECASE
        )
        
        # Replace LCASE FILTER patterns
        query = re.sub(
            r'FILTER\(LCASE\(STR\(\?(\w+)\)\)\s*=\s*LCASE\("([^"]+)"\)\)\s*\.',
            convert_filter_to_exact_match,
            query,
            flags=re.IGNORECASE
        )
        
        # ✅ Also handle patterns where label variable is already bound
        # Pattern: ?var rdfs:label ?varLabel . FILTER(regex(?varLabel, "text", "i"))
        # Replace with: ?var rdfs:label "text" .
        def simplify_label_filter(match):
            """Simplify pattern with separate label variable to direct match."""
            var_name = match.group(1)
            label_var = match.group(2)
            label_text = match.group(3).strip('^$')
            
            print(f"[Postprocess] Simplifying to: ?{var_name} rdfs:label \"{label_text}\"")
            
            return f'{var_name} rdfs:label "{label_text}" .'
        
        # Pattern: ?movie rdfs:label ?movieLabel . FILTER(regex(?movieLabel, "text", "i"))
        query = re.sub(
            r'(\?\w+)\s+rdfs:label\s+\?(\w+Label)\s*\.\s*FILTER\(regex\(str\(\?\2\),\s*"[\^]?([^"]+)[\$]?",\s*"i"\)\)\s*\.',
            simplify_label_filter,
            query,
            flags=re.IGNORECASE
        )
        
        # ✅ REMOVE any remaining FILTER(LANG(...)) on variables that now have direct label match
        # Pattern: ?var rdfs:label "text" . FILTER(LANG(?varLabel) = "en" || ...)
        # Just remove the FILTER line
        query = re.sub(
            r'\?\w+\s+rdfs:label\s+"[^"]+"\s*\.\s*FILTER\(LANG\(\?\w+\)[^\n]*\)\s*\.',
            lambda m: m.group(0).split('FILTER')[0].strip() + ' .',
            query,
            flags=re.IGNORECASE
        )
        
        # ✅ Ensure proper periods
        lines = []
        for line in query.split('\n'):
            stripped = line.rstrip()
            if not stripped:
                lines.append(stripped)
                continue
            
            if stripped.endswith(('.', ',', '{', '}')):
                lines.append(stripped)
            elif re.match(r'^\s*(?:PREFIX|SELECT|ASK|WHERE|OPTIONAL|UNION|ORDER|LIMIT)', stripped, re.IGNORECASE):
                lines.append(stripped)
            elif re.search(r'\s+(wdt:|rdfs:|wd:|ddis:)', stripped):
                if not stripped.endswith('.'):
                    stripped += ' .'
                lines.append(stripped)
            else:
                lines.append(stripped)
        
        query = '\n'.join(lines)
        
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
                'rating': 'RATING',
                'country_of_origin': 'P495'  # ✅ NEW
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
