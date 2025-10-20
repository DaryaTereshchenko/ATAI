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
        """Get few-shot examples - REDUCED to 2 for small model."""
        return [
            {
                "question": "Who directed Star Wars?",
                "sparql": """SELECT ?directorName ?directorItem WHERE {
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^Star Wars$", "i")) .
  ?movieItem wdt:P57 ?directorItem .
  ?directorItem rdfs:label ?directorName .
}"""
            },
            {
                "question": "What is the genre of The Godfather?",
                "sparql": """SELECT ?genreName ?genreItem WHERE {
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^The Godfather$", "i")) .
  ?movieItem wdt:P136 ?genreItem .
  ?genreItem rdfs:label ?genreName .
}"""
            }
        ]

    def _setup_patterns(self):
        """Set up rule-based patterns for common question types."""
        self.patterns = [
            # Director questions - improved to capture movie title
            {
                'pattern': r'(?:who (?:directed|is the director of)|director of)\s+(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$',
                'type': 'director',
                'confidence': 0.9
            },
            # Producer questions
            {
                'pattern': r'(?:who (?:is|was) the producer of|producer of)\s+(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$',
                'type': 'producer',
                'confidence': 0.9
            },
            # Actor questions
            {
                'pattern': r'(?:who (?:acted|starred|plays?) in|actors? (?:in|of)|cast of)\s+(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$',
                'type': 'actor',
                'confidence': 0.9
            },
            # Screenwriter questions
            {
                'pattern': r'(?:who (?:wrote|is the writer|screenwriter)|screenwriter of|written by)\s+(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$',
                'type': 'screenwriter',
                'confidence': 0.9
            },
            # Release date questions
            {
                'pattern': r'(?:when was|release date of|released)\s+(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$',
                'type': 'release_date',
                'confidence': 0.95
            },
            # Genre questions - improved pattern
            {
                'pattern': r'(?:what (?:is the )?genre|genre (?:of|is))\s+(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$',
                'type': 'genre',
                'confidence': 0.9
            },
            # Rating questions
            {
                'pattern': r'(?:what (?:is the )?rating|rating of|mpaa rating)\s+(?:of\s+)?(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$',
                'type': 'rating',
                'confidence': 0.9
            },
        ]
    
    @staticmethod
    def _is_mixed_case(text: str) -> bool:
        """
        Check if text has mixed case (some uppercase, some lowercase).
        This indicates the text is already properly capitalized.
        
        Returns:
            True if text has mixed case (already proper), False if all lower/upper
        """
        # Remove spaces and punctuation for checking
        letters_only = ''.join(c for c in text if c.isalpha())
        
        if not letters_only:
            return False
        
        has_upper = any(c.isupper() for c in letters_only)
        has_lower = any(c.islower() for c in letters_only)
        
        # Mixed case = already proper capitalization
        return has_upper and has_lower
    
    @staticmethod
    def _normalize_proper_name(name: str) -> str:
        """
        Normalize any proper name using PROPER English title case rules.
        - Remove extra quotes
        - Strip whitespace
        - Apply smart title casing (lowercase articles, conjunctions, prepositions)
        
        Examples:
            "the bridge on the river kwai" → "The Bridge on the River Kwai"
            "star wars: episode vi - return of the jedi" → "Star Wars: Episode VI - Return of the Jedi"
            "lord of the rings" → "Lord of the Rings"
        """
        name = name.strip()
        name = name.strip('"\'')  # Remove surrounding quotes
        name = re.sub(r'\s+', ' ', name)  # Normalize spaces
        
        # Words that should be lowercase in title case (unless first/last word)
        lowercase_words = {
            'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'from', 'in', 
            'into', 'of', 'on', 'or', 'over', 'the', 'to', 'up', 'with', 'via'
        }
        
        # Split into words
        words = name.split()
        
        # Apply title case rules
        result_words = []
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # Always capitalize first and last word
            if i == 0 or i == len(words) - 1:
                # Capitalize first letter, keep rest lowercase
                result_words.append(word_lower[0].upper() + word_lower[1:] if len(word_lower) > 1 else word_lower.upper())
            # Check if word should be lowercase
            elif word_lower in lowercase_words:
                result_words.append(word_lower)
            # Capitalize other words
            else:
                # Capitalize first letter, keep rest lowercase
                result_words.append(word_lower[0].upper() + word_lower[1:] if len(word_lower) > 1 else word_lower.upper())
        
        return ' '.join(result_words)
    
    def _normalize_movie_title(self, title: str) -> str:
        """
        Normalize movie title - delegates to _normalize_proper_name.
        Kept for backward compatibility.
        """
        return self._normalize_proper_name(title)
    
    def _escape_regex_special_chars(self, text: str) -> str:
        """
        Escape special regex characters for use in SPARQL FILTER regex.
        
        Characters that need escaping in regex: . ^ $ * + ? { } [ ] \ | ( )
        """
        # Escape backslashes first
        text = text.replace('\\', '\\\\')
        # Escape other special regex characters
        special_chars = ['.', '^', '$', '*', '+', '?', '{', '}', '[', ']', '|', '(', ')']
        for char in special_chars:
            text = text.replace(char, '\\' + char)
        return text
    
    def _generate_sparql_from_pattern(self, question_type: str, movie_title: str) -> str:
        """Generate SPARQL query with case-insensitive FILTER for movie matching."""
        # Normalize the title AND convert to Title Case for database matching
        movie_title = self._normalize_proper_name(movie_title)
        
        print(f"[SPARQL Generation] Normalized title (Title Case): '{movie_title}'")
        
        # Escape regex special characters BEFORE escaping quotes for SPARQL
        movie_title_regex = self._escape_regex_special_chars(movie_title)
        
        print(f"[SPARQL Generation] After regex escaping: '{movie_title_regex}'")
        
        # Then escape quotes for SPARQL string literal
        movie_title_escaped = movie_title_regex.replace('"', '\\"')
        
        print(f"[SPARQL Generation] Final escaped for SPARQL: '{movie_title_escaped}'")
        
        # Always include prefixes with correct URIs
        prefix_block = """PREFIX ddis: <http://ddis.ch/atai/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

"""
        
        # ✅ Now we use properly capitalized title with case-insensitive fallback
        # This ensures proper names match database format
        
        templates = {
            'director': f'''{prefix_block}SELECT ?directorName ?directorItem WHERE {{
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^{movie_title_escaped}$", "i")) .
  ?movieItem wdt:P57 ?directorItem .
  ?directorItem rdfs:label ?directorName .
}}''',
            'actor': f'''{prefix_block}SELECT ?actorName ?actorItem WHERE {{
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^{movie_title_escaped}$", "i")) .
  ?movieItem wdt:P161 ?actorItem .
  ?actorItem rdfs:label ?actorName .
}}''',
            'screenwriter': f'''{prefix_block}SELECT ?writerName ?writerItem WHERE {{
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^{movie_title_escaped}$", "i")) .
  ?movieItem wdt:P58 ?writerItem .
  ?writerItem rdfs:label ?writerName .
}}''',
            'producer': f'''{prefix_block}SELECT ?producerName ?producerItem WHERE {{
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^{movie_title_escaped}$", "i")) .
  ?movieItem wdt:P162 ?producerItem .
  ?producerItem rdfs:label ?producerName .
}}''',
            'release_date': f'''{prefix_block}SELECT ?releaseDate WHERE {{
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^{movie_title_escaped}$", "i")) .
  ?movieItem wdt:P577 ?releaseDate .
}}''',
            'genre': f'''{prefix_block}SELECT ?genreName ?genreItem WHERE {{
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^{movie_title_escaped}$", "i")) .
  ?movieItem wdt:P136 ?genreItem .
  ?genreItem rdfs:label ?genreName .
}}''',
            'rating': f'''{prefix_block}SELECT ?rating WHERE {{
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^{movie_title_escaped}$", "i")) .
  ?movieItem ddis:rating ?rating .
}}''',
        }
        
        query = templates.get(question_type, '')
        
        if query:
            print(f"[SPARQL Generation] Generated query for '{movie_title}':")
            print(query[:300] + "...")  # Print first 300 chars
        
        return query
    
    def _rule_based_convert(self, question: str) -> Optional[SPARQLQuery]:
        """Convert question using rule-based pattern matching."""
        question_lower = question.lower().strip()
        
        for pattern_info in self.patterns:
            match = re.search(pattern_info['pattern'], question_lower, re.IGNORECASE)
            if match:
                movie_title = match.group(1).strip()
                query = self._generate_sparql_from_pattern(
                    pattern_info['type'], 
                    movie_title
                )
                
                if query:
                    return SPARQLQuery(
                        query=query,
                        confidence=pattern_info['confidence'],
                        explanation=f"Pattern-matched as {pattern_info['type']} question for '{movie_title}' (case-insensitive)"
                    )
        
        return None
    
    def _nl2sparql_convert(self, question: str) -> Optional[SPARQLQuery]:
        """
        Convert question using T5 model with SPARQL-aware prompting.
        Uses schema context to guide generation for movie ontology.
        """
        if not self.use_transformer or self.model is None:
            return None
        
        try:
            print(f"[T5-SPARQL] Generating SPARQL for: {question}")
            
            # Create prompt with schema context
            prompt = self._create_nl2sparql_prompt(question)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate SPARQL query
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=5,  
                    num_return_sequences=1,
                    early_stopping=True,
                    temperature=0.7,
                    do_sample=False, 
                    no_repeat_ngram_size=3 
                )
            
            # Decode the output
            generated_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"[T5-SPARQL] Raw output: {generated_query[:200]}...")
            
            # Post-process the query
            query = self._postprocess_sparql(generated_query)
            
            print(f"[T5-SPARQL] Post-processed query: {query[:200]}...")
            
            # Validate SPARQL structure
            if self._is_valid_sparql_structure(query):
                return SPARQLQuery(
                    query=query,
                    confidence=0.75,  # Moderate confidence for general T5 model
                    explanation="Generated using T5 with SPARQL-aware prompting"
                )
            else:
                print(f"[T5-SPARQL] ⚠️  Generated invalid SPARQL structure")
                return None
                
        except Exception as e:
            print(f"[T5-SPARQL] ❌ Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _sparql_llm_convert(self, question: str) -> Optional[SPARQLQuery]:
        """
        Convert question using sparql-llm library.
        Uses few-shot learning with local model for SPARQL generation.
        """
        if self.sparql_llm is None:
            return None
        
        try:
            print(f"[sparql-llm] Generating SPARQL for: {question}")
            
            # Generate SPARQL using sparql-llm
            result = self.sparql_llm.generate_sparql(
                question=question,
                prefixes=self.schema_info['prefixes']
            )
            
            if result and result.get('sparql'):
                query = result['sparql']
                
                print(f"[sparql-llm] Generated query: {query[:200]}...")
                
                # Validate structure
                if self._is_valid_sparql_structure(query):
                    return SPARQLQuery(
                        query=query,
                        confidence=0.90,  # High confidence for specialized library
                        explanation="Generated using sparql-llm with few-shot learning"
                    )
                else:
                    print(f"[sparql-llm] ⚠️  Invalid SPARQL structure")
                    return None
            else:
                print(f"[sparql-llm] ⚠️  No query generated")
                return None
                
        except Exception as e:
            print(f"[sparql-llm] ❌ Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _direct_llm_convert(self, question: str) -> Optional[SPARQLQuery]:
        """Generate SPARQL using DeepSeek LLM."""
        if self.llm is None:
            return None
        
        try:
            print(f"[Direct-LLM] Generating SPARQL for: {question}")
            
            # Create few-shot prompt
            prompt = self._create_few_shot_prompt(question)
            
            print(f"[Direct-LLM] Prompt length: {len(prompt)} chars")
            
            # Generate with llama-cpp-python with increased tokens
            result = self.llm(
                prompt,
                max_tokens=NL2SPARQL_LLM_MAX_TOKENS,
                temperature=NL2SPARQL_LLM_TEMPERATURE,
                stop=["</s>", "Question:", "\n\n\nQuestion", "Example"],  # Better stop sequences
                echo=False
            )
            
            generated_text = result['choices'][0]['text'].strip()
            
            print(f"[Direct-LLM] Generated {len(generated_text)} chars")
            print(f"[Direct-LLM] Full output:\n{generated_text}")  # Print full output for debugging
            
            # Extract SPARQL query from output
            query = self._extract_sparql_from_output(generated_text)
            
            print(f"[Direct-LLM] Extracted query length: {len(query)} chars")
            print(f"[Direct-LLM] Extracted query:\n{query}")
            
            # Post-process the query
            query = self._postprocess_sparql(query)
            
            print(f"[Direct-LLM] Post-processed query:\n{query}")
            
            # Validate structure first (basic check)
            if not self._is_valid_sparql_structure(query):
                print(f"[Direct-LLM] ⚠️  Invalid SPARQL structure (basic check)")
                print(f"[Direct-LLM] Query: {query[:200]}...")
                return None
            
            # Check for complete SELECT clause
            if query.upper().startswith('SELECT'):
                # Verify SELECT has variables
                select_match = re.search(r'SELECT\s+(\?[\w\s]+)', query, re.IGNORECASE)
                if not select_match:
                    print(f"[Direct-LLM] ⚠️  SELECT clause has no variables")
                    return None
                
                # Verify WHERE clause exists and is complete
                if 'WHERE' not in query.upper():
                    print(f"[Direct-LLM] ⚠️  Missing WHERE clause")
                    return None
                
                # Check balanced braces
                if query.count('{') != query.count('}'):
                    print(f"[Direct-LLM] ⚠️  Unbalanced braces: {query.count('{')} open, {query.count('}')} close")
                    return None
            
            # ✅ Validate with SPARQLHandler (syntax check only, no execution)
            validation_result = self._validate_and_secure_sparql(query)
            # ↑ This only validates syntax, does NOT execute the query
            
            if validation_result['valid']:
                print(f"[Direct-LLM] ✅ Query validated successfully")
                return SPARQLQuery(
                    query=validation_result['cleaned_query'],
                    confidence=0.85,
                    explanation=f"Generated using Deepseek-Coder"
                )
            else:
                print(f"[Direct-LLM] ❌ Validation failed: {validation_result['message']}")
                
                # If validation failed due to security issues, return with low confidence
                if validation_result.get('violation_type'):
                    print(f"[Direct-LLM] Security violation: {validation_result['violation_type']}")
                
                return None
                
        except Exception as e:
            print(f"[Direct-LLM] ❌ Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_few_shot_prompt(self, question: str) -> str:
        """Create MINIMAL few-shot prompt for 1.3B model."""
        examples = self._get_few_shot_examples()
        
        prompt = f"""Generate SPARQL for movie questions.

{self._get_ontology_description()}

RULES:
1. End triple patterns with period (.)
2. ALWAYS use FILTER for text matching: ?var rdfs:label ?varLabel . FILTER(regex(str(?varLabel), "^Text$", "i"))
3. Use proper English title case: "The Bridge on the River Kwai"
4. NEVER use exact match like: ?var rdfs:label "Text" (database is case-sensitive)

EXAMPLES:

Question: {examples[0]['question']}
SPARQL:
{examples[0]['sparql']}

Question: {examples[1]['question']}
SPARQL:
{examples[1]['sparql']}

Question: {question}
SPARQL:
"""
        
        return prompt
    
    def _extract_sparql_from_output(self, output: str) -> str:
        """Extract SPARQL query from model output with multiple strategies."""
        print(f"[Extract] Full output length: {len(output)} chars")
        print(f"[Extract] First 500 chars: {output[:500]}")
        
        # Strategy 1: Try to find complete SELECT...WHERE{...} block with proper nesting
        # Look for balanced braces
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
            
            # Start of query
            if re.match(r'^(PREFIX|SELECT|ASK|CONSTRUCT|DESCRIBE)', line_stripped, re.IGNORECASE):
                in_query = True
            
            if in_query:
                sparql_lines.append(line)
                
                # Count braces
                brace_count += line.count('{') - line.count('}')
                
                # Check if we've closed all braces and have a WHERE clause
                if brace_count == 0 and any(re.search(r'\bWHERE\b', l, re.IGNORECASE) for l in sparql_lines):
                    if '}' in line:
                        break
        
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

    def convert(self, question: str) -> SPARQLQuery:
        """
        Convert a natural language question to a SPARQL query.
        
        SIMPLIFIED Strategy:
        1. Try DeepSeek LLM FIRST if enabled
        2. If DeepSeek fails validation, try rule-based as fallback
        3. Return first valid result
        """
        
        # Step 1: Try DeepSeek LLM FIRST if enabled
        if self.method == "direct-llm" and self.llm is not None:
            print(f"[NL2SPARQL] Trying DeepSeek LLM first...")
            llm_result = self._direct_llm_convert(question)
            
            # If LLM succeeded and validated, use it immediately
            if llm_result:
                print(f"[NL2SPARQL] ✅ Using DeepSeek result")
                return llm_result
            else:
                print(f"[NL2SPARQL] ❌ DeepSeek failed, trying rule-based fallback...")
        
        # Step 2: Try rule-based approach as fallback
        print(f"[NL2SPARQL] Trying rule-based approach...")
        rule_result = self._rule_based_convert(question)
        
        # Validate rule-based result if available
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
        
        # Fallback: Could not generate valid query
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
