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
        """Get ontology description for few-shot prompting."""
        return f"""{self.schema_description}

Movie Ontology Rules:
- All movies are of type wd:Q11424 (use: ?movieItem wdt:P31 wd:Q11424)
- Match movie titles using: ?movieItem rdfs:label "Title"
- Always return both labels and items: SELECT ?label ?item
- Common patterns:
  * Director: ?movie wdt:P57 ?director
  * Producer: ?movie wdt:P162 ?producer
  * Genre: ?movie wdt:P136 ?genre
  * Country: ?movie wdt:P495 ?country
  * Award: ?movie wdt:P166 ?award
- Use ORDER BY DESC(?rating) LIMIT 1 for "highest/best"
"""
    
    def _get_few_shot_examples(self) -> List[Dict[str, str]]:
        """Get few-shot examples with correct prefixes and real database patterns."""
        return [
            {
                "question": "Which movie, originally from the country 'South Korea', received the award 'Academy Award for Best Picture'?",
                "sparql": """SELECT ?movieLabel ?movieItem WHERE {
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem wdt:P495 ?countryItem .
  ?countryItem rdfs:label "South Korea" .
  ?movieItem wdt:P166 ?awardItem .
  ?awardItem rdfs:label "Academy Award for Best Picture" .
  ?movieItem rdfs:label ?movieLabel .
}"""
            },
            {
                "question": "What is the highest rated movie?",
                "sparql": """SELECT ?movieLabel ?movieItem WHERE {
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem ddis:rating ?rating .
  ?movieItem rdfs:label ?movieLabel .
}
ORDER BY DESC(?rating)
LIMIT 1"""
            },
            {
                "question": "Who directed Star Wars?",
                "sparql": """SELECT ?directorName ?directorItem WHERE {
  ?movieItem rdfs:label "Star Wars" .
  ?movieItem wdt:P57 ?directorItem .
  ?directorItem rdfs:label ?directorName .
}"""
            }
        ]

    def _setup_patterns(self):
        """Set up rule-based patterns for common question types."""
        self.patterns = [
            # Director questions - improved to capture movie title after "director of" or before "directed"
            {
                'pattern': r'(?:who (?:directed|is the director of)|director of)\s+(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$',
                'type': 'director',
                'confidence': 0.9
            },
            # Producer questions - NEW pattern for producers
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
    
    def _generate_sparql_from_pattern(self, question_type: str, movie_title: str) -> str:
        """Generate SPARQL query based on question type and movie title with correct prefixes."""
        movie_title_escaped = movie_title.strip().replace('"', '\\"')
        
        # Always include prefixes with correct URIs
        prefix_block = """PREFIX ddis: <http://ddis.ch/atai/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

"""
        
        templates = {
            'director': f'''{prefix_block}SELECT ?directorName ?directorItem WHERE {{
  ?movieItem rdfs:label "{movie_title_escaped}" .
  ?movieItem wdt:P57 ?directorItem .
  ?directorItem rdfs:label ?directorName .
}}''',
            'actor': f'''{prefix_block}SELECT ?actorName ?actorItem WHERE {{
  ?movieItem rdfs:label "{movie_title_escaped}" .
  ?movieItem wdt:P161 ?actorItem .
  ?actorItem rdfs:label ?actorName .
}}''',
            'screenwriter': f'''{prefix_block}SELECT ?writerName ?writerItem WHERE {{
  ?movieItem rdfs:label "{movie_title_escaped}" .
  ?movieItem wdt:P58 ?writerItem .
  ?writerItem rdfs:label ?writerName .
}}''',
            'producer': f'''{prefix_block}SELECT ?producerName ?producerItem WHERE {{
  ?movieItem rdfs:label "{movie_title_escaped}" .
  ?movieItem wdt:P162 ?producerItem .
  ?producerItem rdfs:label ?producerName .
}}''',
            'release_date': f'''{prefix_block}SELECT ?releaseDate WHERE {{
  ?movieItem rdfs:label "{movie_title_escaped}" .
  ?movieItem wdt:P577 ?releaseDate .
}}''',
            'genre': f'''{prefix_block}SELECT ?genreName ?genreItem WHERE {{
  ?movieItem rdfs:label "{movie_title_escaped}" .
  ?movieItem wdt:P136 ?genreItem .
  ?genreItem rdfs:label ?genreName .
}}''',
            'rating': f'''{prefix_block}SELECT ?rating WHERE {{
  ?movieItem rdfs:label "{movie_title_escaped}" .
  ?movieItem ddis:rating ?rating .
}}''',
        }
        
        return templates.get(question_type, '')
    
    def _create_nl2sparql_prompt(self, question: str) -> str:
        """
        Create a prompt for T5 model with SPARQL task specification.
        The prompt explicitly tells T5 to generate SPARQL for our movie ontology.
        """
        # T5 works well with task prefixes
        # We provide schema context and clear instructions
        prompt = f"""translate natural language to SPARQL query using this schema:

{self.schema_description}

Natural language question: {question}

SPARQL query:"""
        
        return prompt
    
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
                        explanation=f"Pattern-matched as {pattern_info['type']} question for '{movie_title}'"
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
                    num_beams=5,  # Use beam search for better quality
                    num_return_sequences=1,
                    early_stopping=True,
                    temperature=0.7,
                    do_sample=False,  # Deterministic for consistency
                    no_repeat_ngram_size=3  # Avoid repetitions
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
            
            # Generate with llama-cpp-python
            result = self.llm(
                prompt,
                max_tokens=NL2SPARQL_LLM_MAX_TOKENS,
                temperature=NL2SPARQL_LLM_TEMPERATURE,
                stop=["</s>", "Question:", "\n\n\n", "Example"],
                echo=False
            )
            
            generated_text = result['choices'][0]['text'].strip()
            
            print(f"[Direct-LLM] Raw output: {generated_text[:200]}...")
            
            # Extract SPARQL query from output
            query = self._extract_sparql_from_output(generated_text)
            
            print(f"[Direct-LLM] Extracted query: {query[:200]}...")
            
            # Post-process the query
            query = self._postprocess_sparql(query)
            
            print(f"[Direct-LLM] Post-processed query: {query[:200]}...")
            
            # Validate structure first (basic check)
            if not self._is_valid_sparql_structure(query):
                print(f"[Direct-LLM] ⚠️  Invalid SPARQL structure (basic check)")
                return None
            
            # ✅ Validate with SPARQLHandler (syntax check only, no execution)
            validation_result = self._validate_and_secure_sparql(query)
            # ↑ This only validates syntax, does NOT execute the query
            
            if validation_result['valid']:
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
        """Create a few-shot prompt for SPARQL generation with clear instructions."""
        examples = self._get_few_shot_examples()
        
        prompt = f"""You are a SPARQL query generator. Generate valid SPARQL queries for questions about movies.

{self._get_ontology_description()}

Here are example questions and their correct SPARQL queries:

"""
        
        # Add examples with clear separation
        for i, example in enumerate(examples[:3], 1):  # Use first 3 examples
            prompt += f"Question: {example['question']}\n"
            prompt += f"SPARQL:\n{example['sparql']}\n\n"
        
        # Add the actual question with clear instruction
        prompt += f"Now generate ONLY the SPARQL query for this question (no explanations):\n"
        prompt += f"Question: {question}\n"
        prompt += f"SPARQL:\n"
        
        return prompt
    
    def _extract_sparql_from_output(self, output: str) -> str:
        """Extract SPARQL query from model output with multiple strategies."""
        # Strategy 1: Try to find complete SELECT...WHERE{...} block
        select_match = re.search(
            r'(PREFIX[^\n]*\n)*\s*(SELECT\s+[^{]+WHERE\s*\{[^}]+\})',
            output,
            re.IGNORECASE | re.DOTALL
        )
        if select_match:
            return select_match.group(0)
        
        # Strategy 2: Find SELECT statement with any content up to closing brace
        select_simple = re.search(
            r'(SELECT\s+.+?WHERE\s*\{.+?\})',
            output,
            re.IGNORECASE | re.DOTALL
        )
        if select_simple:
            return select_simple.group(1)
        
        # Strategy 3: Look for query between code markers
        code_block = re.search(r'```(?:sparql)?\s*([^`]+)```', output, re.IGNORECASE | re.DOTALL)
        if code_block:
            return code_block.group(1).strip()
        
        # Strategy 4: Take everything that looks like SPARQL
        # Look for lines that start with SELECT, PREFIX, or contain WHERE
        sparql_lines = []
        in_query = False
        
        for line in output.split('\n'):
            line = line.strip()
            if re.match(r'^(PREFIX|SELECT|WHERE|FILTER)', line, re.IGNORECASE):
                in_query = True
            if in_query:
                sparql_lines.append(line)
                if '}' in line and not line.endswith(','):
                    break
        
        if sparql_lines:
            return '\n'.join(sparql_lines)
        
        # Fallback: return the whole output (will likely fail validation)
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
        Post-process generated SPARQL query to ensure correct prefixes.
        """
        # Remove any markdown code blocks
        query = re.sub(r'```sparql\s*', '', query)
        query = re.sub(r'```\s*', '', query)
        
        # Remove any leading/trailing quotes
        query = query.strip('"\'')
        
        # Replace old movie ontology prefixes with correct Wikidata prefixes
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
        
        # Ensure proper prefixes are included at the beginning
        if 'PREFIX' not in query.upper():
            prefixes = """PREFIX ddis: <http://ddis.ch/atai/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

"""
            query = prefixes + query
        
        # Clean up whitespace
        query = query.strip()
        
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
