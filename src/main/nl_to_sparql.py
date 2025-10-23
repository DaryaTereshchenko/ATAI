import sys
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional, Dict, List, Any, Tuple
from pydantic import BaseModel, Field

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Try to import llama-cpp-python for direct LLM integration
LlamaCache = None
try:
    from llama_cpp import Llama  # type: ignore
    try:
        from llama_cpp import LlamaCache  # type: ignore
    except ImportError:
        LlamaCache = None
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False
    Llama = None  # type: ignore
    LlamaCache = None
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
    query: str = Field(description="The generated SPARQL query")
    confidence: float = Field(description="Confidence score between 0 and 1", ge=0.0, le=1.0)
    explanation: str = Field(description="Brief explanation of the query logic")


class NLToSPARQL:
    """
    Converts natural language to SPARQL.
    - Fast rule-based generator with strict case-insensitive matching (no regex by default).
    - Optional local LLM (llama-cpp) with caching + timeouts.
    """

    # ------- precompiled helpers (module-level reuse across instances) -------
    _LOWERCASE_WORDS = {
        'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'from', 'in',
        'into', 'of', 'on', 'or', 'over', 'the', 'to', 'up', 'with', 'via'
    }

    def __init__(
        self,
        use_transformer: bool = None,   # deprecated, kept to avoid breaking signatures
        model_name: str = None,         # deprecated
        use_spbert: bool = False,       # deprecated
        method: str = None,
        sparql_handler: Optional[SPARQLHandler] = None
    ):
        # Determine method
        self.method = method or NL2SPARQL_METHOD or "rule-based"

        # Public knobs
        ctx_length = int(NL2SPARQL_LLM_CONTEXT_LENGTH or 2048)
        max_tokens_cfg = NL2SPARQL_LLM_MAX_TOKENS or 256
        temp_cfg = NL2SPARQL_LLM_TEMPERATURE if NL2SPARQL_LLM_TEMPERATURE is not None else 0.1

        # Runtime config
        self._llm_max_tokens = int(os.environ.get("NL2SPARQL_LLM_MAX_OUTPUT_TOKENS", max_tokens_cfg))
        self._llm_max_tokens = max(64, min(self._llm_max_tokens, ctx_length // 2))
        self._llm_temperature = float(os.environ.get("NL2SPARQL_LLM_TEMPERATURE", temp_cfg))
        # Set to None to disable timeout, or a very large number
        self._llm_timeout_seconds = None  # No timeout - wait for LLM to complete
        self._llm_stop_sequences = ["</s>", "\n\nQuestion:", "\nQuestion:", "Example", "EXAMPLES"]

        # Internals
        self.llm = None
        self._llama_prompt_cache = None
        self._llm_cache: Dict[str, str] = {}          # prompt -> raw llm text
        self._convert_cache: Dict[str, SPARQLQuery] = {}  # nl -> SPARQLQuery
        self._llm_executor: Optional[ThreadPoolExecutor] = None
        self._llm_lock = threading.Lock()

        # Back-compat placeholders to avoid AttributeErrors
        self.use_transformer = False
        self.model = None
        self.tokenizer = None
        self.device = None
        self.sparql_llm = None

        # Validation handler
        self.sparql_validator = sparql_handler if sparql_handler is not None else SPARQLHandler()

        # Setup
        self._setup_patterns()
        self._setup_schema()
        self._prompt_header = self._build_prompt_header()

        if self.method == "direct-llm":
            self._initialize_direct_llm()
        else:
            print("ℹ️  Using rule-based SPARQL generation only (no model)")

    # ---------------------------------------------------------------------
    # Initialization & configuration
    # ---------------------------------------------------------------------
    def _initialize_direct_llm(self):
        """Initialize llama-cpp with conservative defaults, caching, and mmap."""
        if not LLAMACPP_AVAILABLE:
            print("⚠️  llama-cpp-python not installed. Falling back to rule-based.")
            self.method = "rule-based"
            return

        model_path = NL2SPARQL_LLM_MODEL_PATH
        if not model_path or not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}. Falling back to rule-based.")
            self.method = "rule-based"
            return

        try:
            print("📥 Loading local model for SPARQL generation…")
            n_threads = int(os.environ.get("NL2SPARQL_LLM_THREADS", max(1, (os.cpu_count() or 4) // 2)))
            n_batch = int(os.environ.get("NL2SPARQL_LLM_BATCH", 256))
            n_gpu_layers = int(os.environ.get("NL2SPARQL_LLM_N_GPU_LAYERS", 0))
            use_mmap = bool(int(os.environ.get("NL2SPARQL_LLM_USE_MMAP", 1)))
            use_mlock = bool(int(os.environ.get("NL2SPARQL_LLM_USE_MLOCK", 0)))

            llama_kwargs: Dict[str, Any] = {
                "model_path": model_path,
                "n_ctx": int(NL2SPARQL_LLM_CONTEXT_LENGTH or 2048),
                "n_threads": max(1, n_threads),
                "n_batch": max(32, n_batch),
                "verbose": False,
                "n_gpu_layers": n_gpu_layers,
                "use_mmap": use_mmap,
                "use_mlock": use_mlock,
            }

            if LlamaCache is not None:
                try:
                    # Try to initialize cache without capacity parameter
                    self._llama_prompt_cache = LlamaCache()
                    llama_kwargs["cache"] = self._llama_prompt_cache
                    print(f"   Prompt cache enabled")
                except TypeError:
                    # If that fails, try without any parameters or skip cache
                    try:
                        # Some versions might use different initialization
                        self._llama_prompt_cache = LlamaCache(capacity_bytes=2048*1024*1024)  # 2GB
                        llama_kwargs["cache"] = self._llama_prompt_cache
                        print(f"   Prompt cache enabled (2GB)")
                    except Exception:
                        self._llama_prompt_cache = None
                        print(f"⚠️  Prompt cache unavailable in this llama-cpp version")
                except Exception as cache_err:
                    self._llama_prompt_cache = None
                    print(f"⚠️  Prompt cache unavailable: {cache_err}")

            self.llm = Llama(**llama_kwargs)
            self._llm_executor = ThreadPoolExecutor(max_workers=1)
            timeout_msg = "no timeout" if self._llm_timeout_seconds is None else f"{self._llm_timeout_seconds:.1f}s timeout"
            print(f"✅ Model ready | threads={llama_kwargs['n_threads']} batch={llama_kwargs['n_batch']} "
                  f"ctx={llama_kwargs['n_ctx']} gpu_layers={n_gpu_layers} "
                  f"{timeout_msg} max_new={self._llm_max_tokens}")
        except Exception as e:
            print(f"❌ Error loading model: {e}. Falling back to rule-based.")
            try:
                if self._llm_executor:
                    self._llm_executor.shutdown(wait=False)
            finally:
                self.llm = None
                self._llm_executor = None
                self.method = "rule-based"

    def _build_prompt_header(self) -> str:
        """A tiny header used in the LLM prompt."""
        return (
            "You generate SPARQL for a Wikidata-based movie graph.\n"
            "Key: P31=type(Movie=wd:Q11424), P57=director, P161=actor, P58=writer, "
            "P162=producer, P577=release, P136=genre, rdfs:label=name.\n"
            "Always match movie titles case-insensitively and exactly.\n"
        )

    def _setup_schema(self):
        self.schema_info = {
            'prefixes': {
                'ddis': 'http://ddis.ch/atai/',
                'wd': 'http://www.wikidata.org/entity/',
                'wdt': 'http://www.wikidata.org/prop/direct/',
                'schema': 'http://schema.org/',
                'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
                'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
            }
        }

    def _setup_patterns(self):
        """Precompile rule-based patterns."""
        pats: List[Tuple[re.Pattern, str, float]] = []
        def add(pat: str, t: str, conf: float):
            pats.append((re.compile(pat, re.IGNORECASE), t, conf))

        add(r'(?:who (?:directed|is the director of)|director of)\s+(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$', 'director', 0.95)
        add(r'(?:who (?:is|was) the producer of|producer of)\s+(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$', 'producer', 0.9)
        add(r'(?:who (?:acted|starred|plays?) in|actors? (?:in|of)|cast of)\s+(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$', 'actor', 0.9)
        add(r'(?:who (?:wrote|is the writer|screenwriter)|screenwriter of|written by)\s+(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$', 'screenwriter', 0.9)
        add(r'(?:when was|release date of|released)\s+(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$', 'release_date', 0.96)
        add(r'(?:what (?:is the )?genre|genre (?:of|is))\s+(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$', 'genre', 0.9)
        add(r'(?:what (?:is the )?rating|rating of|mpaa rating)\s+(?:of\s+)?(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$', 'rating', 0.9)

        self.patterns = pats

    # ---------------------------------------------------------------------
    # String normalization
    # ---------------------------------------------------------------------
    @staticmethod
    def _normalize_proper_name(name: str) -> str:
        if not name:
            return name
        name = name.strip().strip('"\'')
        name = re.sub(r'\s+', ' ', name)
        words = name.split()
        out = []
        for i, w in enumerate(words):
            lw = w.lower()
            if i == 0 or i == len(words) - 1 or lw not in NLToSPARQL._LOWERCASE_WORDS:
                out.append(lw.capitalize())
            else:
                out.append(lw)
        return ' '.join(out)

    @staticmethod
    def _escape_regex_special_chars(text: str) -> str:
        text = text.replace('\\', '\\\\')
        special = ['.', '^', '$', '*', '+', '?', '{', '}', '[', ']', '|', '(', ')']
        for ch in special:
            text = text.replace(ch, '\\' + ch)
        return text

    # ---------------------------------------------------------------------
    # SPARQL generation (fast, case-insensitive, non-regex by default)
    # ---------------------------------------------------------------------
    def _prefix_block(self) -> str:
        return """PREFIX ddis: <http://ddis.ch/atai/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

"""

    def _movie_title_filters(self, var_label: str, title: str, prefer_lang: str = None, use_regex: bool = False) -> str:
        """
        Build a case-insensitive EXACT match filter for labels:
        Default (fast):  LCASE(STR(?label)) = LCASE("Title")
        Optional:        LANGMATCHES(LANG(?label), "en") if prefer_lang given
        If use_regex=True: keep your previous regex("^Title$", "i") path.
        """
        title_norm = self._normalize_proper_name(title)
        if use_regex:
            esc = self._escape_regex_special_chars(title_norm)
            if not esc.startswith('^'):
                esc = '^' + esc
            if not esc.endswith('$'):
                esc = esc + '$'
            lang = f'FILTER(LANG({var_label}) = "{prefer_lang}") .' if prefer_lang else ''
            return f'{lang}\n  FILTER(regex(str({var_label}), "{esc}", "i")) .'
        # non-regex (faster)
        lang = f'FILTER(LANG({var_label}) = "{prefer_lang}") .' if prefer_lang else ''
        # ensure EXACT match ignoring case
        return f'{lang}\n  FILTER(LCASE(STR({var_label})) = LCASE("{title_norm}")) .'

    def _generate_sparql_from_pattern(self, qtype: str, movie_title: str, prefer_lang: str = None) -> str:
        movie_title = self._normalize_proper_name(movie_title)
        p = self._prefix_block()
        # label var for movie
        filters = self._movie_title_filters("?movieLabel", movie_title, prefer_lang=prefer_lang, use_regex=False)

        templates = {
            'director': f"""{p}SELECT ?directorName ?directorItem WHERE {{
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  {filters}
  ?movieItem wdt:P57 ?directorItem .
  ?directorItem rdfs:label ?directorName .
}}""",
            'actor': f"""{p}SELECT ?actorName ?actorItem WHERE {{
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  {filters}
  ?movieItem wdt:P161 ?actorItem .
  ?actorItem rdfs:label ?actorName .
}}""",
            'screenwriter': f"""{p}SELECT ?writerName ?writerItem WHERE {{
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  {filters}
  ?movieItem wdt:P58 ?writerItem .
  ?writerItem rdfs:label ?writerName .
}}""",
            'producer': f"""{p}SELECT ?producerName ?producerItem WHERE {{
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  {filters}
  ?movieItem wdt:P162 ?producerItem .
  ?producerItem rdfs:label ?producerName .
}}""",
            'release_date': f"""{p}SELECT ?releaseDate WHERE {{
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  {filters}
  ?movieItem wdt:P577 ?releaseDate .
}}""",
            'genre': f"""{p}SELECT ?genreName ?genreItem WHERE {{
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  {filters}
  ?movieItem wdt:P136 ?genreItem .
  ?genreItem rdfs:label ?genreName .
}}""",
            'rating': f"""{p}SELECT ?rating WHERE {{
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  {filters}
  ?movieItem ddis:rating ?rating .
}}""",
        }
        return templates.get(qtype, "")

    def _rule_based_convert(self, question: str) -> Optional[SPARQLQuery]:
        q = question.strip()
        q_lower = q.lower()

        for pat, qtype, conf in self.patterns:
            m = pat.search(q_lower)
            if m:
                movie_title = m.group(1).strip()
                query = self._generate_sparql_from_pattern(qtype, movie_title, prefer_lang=os.environ.get("NL2SPARQL_PREFER_LANG"))
                if query:
                    return SPARQLQuery(
                        query=query,
                        confidence=conf,
                        explanation=f"Rule-based: Pattern-matched as {qtype} for '{movie_title}' with case-insensitive exact title match"
                    )
        return None

    # ---------------------------------------------------------------------
    # Direct LLM path (kept lean, cached, with timeout)
    # ---------------------------------------------------------------------
    def _create_few_shot_prompt(self, question: str, intent: str, title: str) -> str:
        title_norm = self._normalize_proper_name(title) if title else "MOVIE_TITLE"
        # tiny, deterministic prompt — the model mostly fills variables
        return (
            f"{self._prompt_header}\n"
            f"Question: {question}\n"
            "Return a single SPARQL SELECT query. Use variables named *Name and *Item when appropriate.\n"
            "Match the movie by exact title, case-insensitive, NOT partial.\n"
            f"HINT intent={intent} title={title_norm}\n"
            "SPARQL:\n"
        )

    def _intent_hint(self, question: str) -> Tuple[str, Optional[str]]:
        q = question.lower().strip()
        for pat, qtype, _ in self.patterns:
            m = pat.search(q)
            if m:
                return (qtype, m.group(1).strip())
        return ("unknown", None)

    def _direct_llm_convert(self, question: str) -> Optional[SPARQLQuery]:
        if self.llm is None or self._llm_executor is None:
            return None

        intent, title = self._intent_hint(question)
        if not title:
            return None

        prompt = self._create_few_shot_prompt(question, intent, title)

        # cache
        with self._llm_lock:
            if prompt in self._llm_cache:
                generated_text = self._llm_cache[prompt]
            else:
                def _call():
                    return self.llm(
                        prompt,
                        max_tokens=self._llm_max_tokens,
                        temperature=self._llm_temperature,
                        stop=self._llm_stop_sequences,
                        echo=False,
                    )
                try:
                    fut = self._llm_executor.submit(_call)
                    # Wait indefinitely if timeout is None
                    if self._llm_timeout_seconds is None:
                        print("⏳ Waiting for LLM to generate SPARQL (no timeout)...")
                        result = fut.result()  # No timeout
                    else:
                        result = fut.result(timeout=self._llm_timeout_seconds)
                    generated_text = (result.get('choices', [{}])[0].get('text') or "").strip()
                    self._llm_cache[prompt] = generated_text
                    print(f"✅ LLM generation completed")
                except FuturesTimeoutError:
                    print(f"⏱️  LLM timed out after {self._llm_timeout_seconds}s")
                    return None
                except Exception as e:
                    print(f"❌ LLM generation error: {e}")
                    return None

        if not generated_text:
            return None

        # Extract minimal SELECT … WHERE { … }
        query = self._extract_sparql_from_output(generated_text)
        if not query:
            return None

        query = self._postprocess_sparql(query)

        if not self._is_valid_sparql_structure(query):
            return None

        validation_result = self._validate_and_secure_sparql(query)
        if validation_result.get('valid') and validation_result.get('cleaned_query'):
            return SPARQLQuery(
                query=validation['cleaned_query'],
                confidence=0.85,
                explanation="LLM-generated: Local DeepSeek model with timeout and caching"
            )
        return None

    # ---------------------------------------------------------------------
    # Extraction, validation, post-process
    # ---------------------------------------------------------------------
    @staticmethod
    def _extract_sparql_from_output(output: str) -> str:
        # prefer SELECT...WHERE{ } with balanced braces
        m = re.search(r'(?:PREFIX[^\n]*\n)*\s*SELECT\s+[^{]+WHERE\s*\{', output, re.IGNORECASE | re.DOTALL)
        if m:
            start = m.start()
            brace = 0
            end = len(output)
            for i, ch in enumerate(output[start:], start=start):
                if ch == '{':
                    brace += 1
                elif ch == '}':
                    brace -= 1
                    if brace == 0:
                        end = i + 1
                        break
            candidate = output[start:end]
            if candidate.count('{') == candidate.count('}'):
                return candidate.strip()

        block = re.search(r'```(?:sparql)?\s*([^`]+)```', output, re.IGNORECASE | re.DOTALL)
        if block:
            return block.group(1).strip()

        # fallback: try lines until braces close
        lines = []
        found_select = False
        brace = 0
        for line in output.splitlines():
            if not found_select and re.search(r'^\s*(PREFIX|SELECT)\b', line, re.IGNORECASE):
                found_select = True
            if found_select:
                lines.append(line)
                brace += line.count('{') - line.count('}')
                if brace == 0 and 'WHERE' in '\n'.join(lines).upper():
                    break
        return '\n'.join(lines).strip()

    def _validate_and_secure_sparql(self, query: str) -> Dict[str, Any]:
        try:
            return self.sparql_validator.validate_query(query)
        except Exception as e:
            return {'valid': False, 'message': f'Validation error: {e}', 'cleaned_query': None}

    @staticmethod
    def _is_valid_sparql_structure(query: str) -> bool:
        qU = query.upper()
        has_type = any(t in qU for t in ['SELECT', 'ASK', 'CONSTRUCT', 'DESCRIBE'])
        has_where = 'WHERE' in qU or '{' in query
        return has_type and has_where and len(query.strip()) > 10

    def _postprocess_sparql(self, query: str) -> str:
        # strip code fences/quotes
        query = re.sub(r'```sparql\s*|```\s*', '', query).strip('"\' \n\r\t')

        # ensure prefixes
        if 'PREFIX' not in query.upper():
            query = self._prefix_block() + query

        # normalize newlines + remove duplicate periods
        query = re.sub(r'\r\n?', '\n', query)
        query = re.sub(r'\.\s*\.', '.', query)
        query = re.sub(r'\n{3,}', '\n\n', query).strip()

        # ensure triple lines end with '.' inside WHERE (light pass)
        out = []
        in_where = False
        for ln in query.split('\n'):
            s = ln.rstrip()
            if re.search(r'\bWHERE\s*\{', s, re.IGNORECASE):
                in_where = True
                out.append(s)
                continue
            if in_where and s.strip() == '}':
                in_where = False
                out.append(s)
                continue
            if in_where:
                if ('FILTER' in s.upper()) and not s.endswith('.'):
                    s += ' .'
                elif re.search(r'\s+(wdt:|rdfs:|wd:|ddis:|rdf:|schema:| a )', s) and not re.search(r'[{};,\.]$', s):
                    s += ' .'
            out.append(s)
        query = '\n'.join(out)
        return query

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def convert(self, question: str) -> SPARQLQuery:
        """Fast path: LLM (if enabled) with timeout+cache -> rule-based fallback, both validated."""
        qkey = question.strip()
        if qkey in self._convert_cache:
            return self._convert_cache[qkey]

        # try LLM first only if configured and loaded
        if self.method == "direct-llm" and self.llm is not None:
            llm_result = self._direct_llm_convert(question)
            if llm_result:
                print(f"✅ Using LLM-generated SPARQL (confidence: {llm_result.confidence:.2f})")
                self._convert_cache[qkey] = llm_result
                return llm_result
            else:
                print("⚠️  LLM generation failed or timed out, falling back to rules...")

        # rule-based
        rule = self._rule_based_convert(question)
        if rule:
            print(f"✅ Using rule-based SPARQL (confidence: {rule.confidence:.2f})")
            validation = self._validate_and_secure_sparql(rule.query)
            if validation.get('valid'):
                rule.query = validation['cleaned_query']
                rule.explanation += f" | Validated: {validation.get('message')}"
                self._convert_cache[qkey] = rule
                return rule

        # final fallback
        res = SPARQLQuery(
            query="# Could not generate valid query",
            confidence=0.0,
            explanation=f"Could not generate a valid SPARQL query for: {question}"
        )
        self._convert_cache[qkey] = res
        return res

    def validate_question(self, question: str) -> bool:
        if not question or len(question.strip()) < 3:
            return False
        movie_keywords = ['movie', 'film', 'director', 'actor', 'actress', 'release',
                          'genre', 'screenwriter', 'writer', 'star', 'cast', 'rating']
        ql = question.lower()
        if any(w in ql for w in ['who', 'what', 'when', 'where', 'which', 'how']):
            return True
        if any(k in ql for k in movie_keywords):
            return True
        for pat, _, __ in self.patterns:
            if pat.search(ql):
                return True
        return False