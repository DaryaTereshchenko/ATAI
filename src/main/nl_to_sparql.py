import sys 
import os
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import signal
from contextlib import contextmanager

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.main.sparql_handler import SPARQLHandler
from src.config import (
    NL2SPARQL_METHOD,
    NL2SPARQL_LLM_MODEL_PATH,
    NL2SPARQL_LLM_CONTEXT_LENGTH,
    NL2SPARQL_LLM_MAX_TOKENS,
    NL2SPARQL_LLM_TEMPERATURE
)

try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False
    print("⚠️ llama-cpp-python not available. Install with: pip install llama-cpp-python")

@dataclass
class SPARQLQuery:
    query: str
    confidence: float
    explanation: str

class NLToSPARQL:
    """
    Converts natural language questions to SPARQL queries.
    - Model-first (DeepSeek via llama-cpp) with rule-based fallback.
    - Robust user-input normalization (quotes/dashes/spacing).
    - Case-insensitive, anchored label matching in SPARQL.
    """

    # -------- precompiled regexes / helpers --------
    _RE_SPACES = re.compile(r"\s+")
    _RE_CURLY_QUOTES = [
        (re.compile("[“”]"), '"'),
        (re.compile("[‘’]"), "'"),
        (re.compile("[—–]"), "-"),
    ]
    _RE_ROMAN = re.compile(r"(?:[ivxlcdm]+|[IVXLCDM]+)\.?$")
    _RE_MCPREFIX = re.compile(r"^mc([a-z])", flags=re.I)
    _RE_CODEBLOCK = re.compile(r"```(?:sparql|ttl|turtle|sql|text)?\s*(.*?)```", re.S | re.I)

    def __init__(
        self,
        use_transformer: bool = None,   # deprecated
        model_name: str = None,         # deprecated
        use_spbert: bool = False,       # deprecated
        method: str = None,             # "direct-llm" or "rule-based"
        sparql_handler: Optional[SPARQLHandler] = None
    ):
        # Prefer model first by default
        self.method = method or NL2SPARQL_METHOD or "direct-llm"

        # SPARQL validator/runner
        self.sparql_validator = sparql_handler if sparql_handler is not None else SPARQLHandler()

        # caches
        self._llm = None
        self._llm_cache: Dict[str, str] = {}
        self._convert_cache: Dict[str, SPARQLQuery] = {}

        # rules always available as fallback
        self._setup_patterns()
        self._setup_schema()

        # init model if requested/available
        if self.method == "direct-llm":
            self._initialize_direct_llm()
        else:
            print("ℹ️  Using rule-based SPARQL generation only (no model)")

    # -------------------- init LLM --------------------

    def _initialize_direct_llm(self):
        if not LLAMACPP_AVAILABLE:
            print("⚠️  llama-cpp-python not installed. Falling back to rule-based.")
            self.method = "rule-based"
            return

        try:
            model_path = NL2SPARQL_LLM_MODEL_PATH
            if not model_path or not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path!r}. Falling back to rule-based.")
                self.method = "rule-based"
                return

            print("📥 Loading model for SPARQL generation...")
            print(f"    Model: {model_path}")

            # Prefer all CPU cores; keep context from config
            self._llm = Llama(
                model_path=model_path,
                n_ctx=int(NL2SPARQL_LLM_CONTEXT_LENGTH or 4096),
                n_threads=os.cpu_count() or 4,
                verbose=False
            )
            print("✅ Model loaded (llama-cpp). Using model-first strategy.")
        except Exception as e:
            print(f"❌ Error loading model: {e}\nFalling back to rule-based.")
            self._llm = None
            self.method = "rule-based"

    # -------------------- schema / few-shot --------------------

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
        self.schema_description = self._format_schema_for_nl2sparql()

    def _format_schema_for_nl2sparql(self) -> str:
        lines = []
        for p, uri in self.schema_info['prefixes'].items():
            lines.append(f"PREFIX {p}: <{uri}>")
        lines += [
            "",
            "# Movie Knowledge Graph (Wikidata-style):",
            "#  wd:Q11424 Movie; labels via rdfs:label",
            "#  wdt:P57 director, wdt:P161 actor, wdt:P58 screenwriter,",
            "#  wdt:P162 producer, wdt:P577 release date, wdt:P136 genre",
            "# Matching pattern:",
            '#  ?x rdfs:label ?xLabel . FILTER(regex(str(?xLabel), "^Text$", "i"))',
        ]
        return "\n".join(lines)

    def _get_ontology_description(self) -> str:
        return (
            "Movie KG facts:\n"
            "- Movies: ?movieItem wdt:P31 wd:Q11424\n"
            "- Labels: rdfs:label\n"
            "- Properties: P57=director, P161=actor, P58=writer, P162=producer, P577=date, P136=genre\n"
        )

    def _get_few_shot_examples(self) -> List[Dict[str, str]]:
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

    # -------------------- input cleaning / normalization --------------------

    def _preclean(self, q: str) -> str:
        if not q:
            return ""
        s = q.strip()
        for rx, repl in self._RE_CURLY_QUOTES:
            s = rx.sub(repl, s)
        s = self._RE_SPACES.sub(" ", s)
        return s

    @staticmethod
    def _is_mixed_case(text: str) -> bool:
        letters_only = ''.join(c for c in text if c.isalpha())
        if not letters_only:
            return False
        return any(c.isupper() for c in letters_only) and any(c.islower() for c in letters_only)

    def _normalize_proper_name(self, name: str) -> str:
        """
        Fast, deterministic English title-case with common exceptions.
        Respects already-mixed case and roman numerals; supports Mc- prefix and hyphenations.
        """
        if not name:
            return name
        name = name.strip().strip('"\'')
        name = re.sub(r'\s+', ' ', name)

        if self._is_mixed_case(name):
            return self._snap_to_graph_casing(name)

        LOWER = {
            'a','an','and','as','at','but','by','for','from','in','into',
            'of','on','or','over','the','to','up','with','via','nor','so','yet'
        }

        def cap_word(w: str, first: bool, last: bool) -> str:
            if w.isupper() and len(w) > 1:
                return w
            if self._RE_ROMAN.fullmatch(w):
                return w.upper().rstrip(".")
            w = self._RE_MCPREFIX.sub(lambda m: "Mc" + m.group(1).upper(), w)
            parts = re.split(r"(-)", w)
            for i in range(0, len(parts), 2):
                p = parts[i]
                if not p:
                    continue
                if not first and not last and p.lower() in LOWER:
                    parts[i] = p.lower()
                else:
                    parts[i] = p[:1].upper() + p[1:].lower()
            return "".join(parts)

        tokens = re.split(r"(\s+)", name)
        words = [t for i, t in enumerate(tokens) if i % 2 == 0]
        seps  = [t for i, t in enumerate(tokens) if i % 2 == 1] + [""]

        out = []
        for i, w in enumerate(words):
            out.append(cap_word(w, i == 0, i == len(words)-1))
        rebuilt = "".join(a + b for a, b in zip(out, seps)).strip()
        return self._snap_to_graph_casing(rebuilt)

    def _snap_to_graph_casing(self, text: str) -> str:
        """
        If your SPARQLHandler exposes a .label_index {casefold(label)->{variants}},
        snap to the canonical label. Otherwise return text unchanged.
        """
        idx = getattr(self.sparql_validator, "label_index", None)
        if not idx or not text:
            return text
        cand = idx.get(text.casefold())
        if not cand:
            return text
        return sorted(cand, key=len, reverse=True)[0]

    # -------------------- rule patterns (fallback) --------------------

    def _setup_patterns(self):
        self.patterns = [
            {'pattern': r'(?:who (?:directed|is the director of)|director of)\s+(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$', 'type': 'director', 'confidence': 0.9},
            {'pattern': r'(?:who (?:is|was) the producer of|producer of)\s+(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$', 'type': 'producer', 'confidence': 0.9},
            {'pattern': r'(?:who (?:acted|starred|plays?) in|actors? (?:in|of)|cast of)\s+(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$', 'type': 'actor', 'confidence': 0.9},
            {'pattern': r'(?:who (?:wrote|is the writer|screenwriter)|screenwriter of|written by)\s+(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$', 'type': 'screenwriter', 'confidence': 0.9},
            {'pattern': r'(?:when was|release date of|released)\s+(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$', 'type': 'release_date', 'confidence': 0.95},
            {'pattern': r'(?:what (?:is the )?genre|genre (?:of|is))\s+(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$', 'type': 'genre', 'confidence': 0.9},
            {'pattern': r'(?:what (?:is the )?rating|rating of|mpaa rating)\s+(?:of\s+)?(?:the\s+)?(?:movie\s+)?["\']?([^"\'?\.]+?)["\']?\s*[\?\.]*$', 'type': 'rating', 'confidence': 0.9},
        ]

    # -------------------- rule-based query builder (fallback) --------------------

    def _escape_regex_special_chars(self, text: str) -> str:
        text = text.replace('\\', '\\\\')
        for ch in ['.', '^', '$', '*', '+', '?', '{', '}', '[', ']', '|', '(', ')']:
            text = text.replace(ch, '\\' + ch)
        return text

    def _generate_sparql_from_pattern(self, question_type: str, movie_title: str) -> str:
        title = self._normalize_proper_name(movie_title)
        lit = self._escape_regex_special_chars(title).replace('"', '\\"')
        prefix_block = """PREFIX ddis: <http://ddis.ch/atai/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

"""
        templates = {
            'director': f'''{prefix_block}SELECT ?directorName ?directorItem WHERE {{
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^{lit}$", "i")) .
  ?movieItem wdt:P57 ?directorItem .
  ?directorItem rdfs:label ?directorName .
}}''',
            'actor': f'''{prefix_block}SELECT ?actorName ?actorItem WHERE {{
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^{lit}$", "i")) .
  ?movieItem wdt:P161 ?actorItem .
  ?actorItem rdfs:label ?actorName .
}}''',
            'screenwriter': f'''{prefix_block}SELECT ?writerName ?writerItem WHERE {{
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^{lit}$", "i")) .
  ?movieItem wdt:P58 ?writerItem .
  ?writerItem rdfs:label ?writerName .
}}''',
            'producer': f'''{prefix_block}SELECT ?producerName ?producerItem WHERE {{
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^{lit}$", "i")) .
  ?movieItem wdt:P162 ?producerItem .
  ?producerItem rdfs:label ?producerName .
}}''',
            'release_date': f'''{prefix_block}SELECT ?releaseDate WHERE {{
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^{lit}$", "i")) .
  ?movieItem wdt:P577 ?releaseDate .
}}''',
            'genre': f'''{prefix_block}SELECT ?genreName ?genreItem WHERE {{
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^{lit}$", "i")) .
  ?movieItem wdt:P136 ?genreItem .
  ?genreItem rdfs:label ?genreName .
}}''',
            'rating': f'''{prefix_block}SELECT ?rating WHERE {{
  ?movieItem wdt:P31 wd:Q11424 .
  ?movieItem rdfs:label ?movieLabel .
  FILTER(regex(str(?movieLabel), "^{lit}$", "i")) .
  ?movieItem ddis:rating ?rating .
}}''',
        }
        return templates.get(question_type, "")

    def _rule_based_convert(self, question: str) -> Optional[SPARQLQuery]:
        q = self._preclean(question).lower()
        for pat in self.patterns:
            m = re.search(pat['pattern'], q, re.IGNORECASE)
            if not m:
                continue
            movie_title = m.group(1).strip()
            query = self._generate_sparql_from_pattern(pat['type'], movie_title)
            if query:
                return SPARQLQuery(
                    query=query,
                    confidence=pat['confidence'],
                    explanation=f"Rule-based as {pat['type']} for '{movie_title}'"
                )
        return None

    # -------------------- LLM path (DeepSeek via llama-cpp) --------------------

    def _create_few_shot_prompt(self, question: str) -> str:
        ex = self._get_few_shot_examples()
        return f"""Generate SPARQL for movie questions.

{self._get_ontology_description()}

RULES:
1) End triple patterns with period (.).
2) ALWAYS match labels case-insensitively with anchors:
   ?x rdfs:label ?xLabel . FILTER(regex(str(?xLabel), "^Text$", "i"))
3) NEVER use ?x rdfs:label "Text".
4) Return readable labels where relevant.

EXAMPLES

Question: {ex[0]['question']}
SPARQL:
{ex[0]['sparql']}

Question: {ex[1]['question']}
SPARQL:
{ex[1]['sparql']}

Question: {question}
SPARQL:
"""

    def _extract_sparql_from_output(self, output: str) -> str:
        m = self._RE_CODEBLOCK.search(output)
        if m:
            return m.group(1).strip()
        # Try to cut from first SELECT to last closing brace
        sel = re.search(r'(PREFIX[^\n]*\n)*\s*SELECT\s+[^{]+WHERE\s*\{', output, re.I | re.S)
        if sel:
            start = sel.start()
            # balance braces
            depth = 0; end = len(output)
            for i, ch in enumerate(output[start:], start=start):
                if ch == '{': depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            candidate = output[start:end]
            if candidate.count('{') == candidate.count('}'):
                return candidate.strip()
        # Fallback: best effort
        return output.strip()

    def _force_case_insensitive(self, query: str) -> str:
        if not query:
            return query

        # rdfs:label "Literal"  ->  rdfs:label ?vLabel . FILTER(regex(str(?vLabel), "^Literal$", "i"))
        def repl_label(m):
            var, lit = m.group("var"), m.group("lit")
            lit_norm = self._normalize_proper_name(lit)
            pat = f'^{re.escape(lit_norm)}$'
            vlabel = f'{var}Label'
            return f'{var} rdfs:label {vlabel} .\n  FILTER(regex(str({vlabel}), "{pat}", "i")) .'

        query = re.sub(
            r'(?P<var>\?[A-Za-z_]\w*)\s+rdfs:label\s+"(?P<lit>[^"]+)"\s*\.',
            repl_label,
            query
        )

        # FILTER(?x = "Literal") -> FILTER(regex(str(?x), "^Literal$", "i"))
        def repl_eq(m):
            var, lit = m.group("var"), m.group("lit")
            lit_norm = self._normalize_proper_name(lit)
            pat = f'^{re.escape(lit_norm)}$'
            return f'FILTER(regex(str({var}), "{pat}", "i"))'

        query = re.sub(
            r'FILTER\(\s*(?P<var>\?[A-Za-z_]\w*)\s*=\s*"(?P<lit>[^"]+)"\s*\)',
            repl_eq,
            query
        )
        return query

    def _postprocess_sparql(self, query: str) -> str:
        # strip code fences / quotes
        q = re.sub(r'```sparql\s*', '', query)
        q = re.sub(r'```\s*', '', q).strip('"\'')

        # enforce CI label matching everywhere
        q = self._force_case_insensitive(q)

        # normalize EOLs and ensure FILTER lines end with '.'
        q = re.sub(r'\r\n', '\n', q)
        lines = []
        in_where = False
        for line in q.split('\n'):
            s = line.rstrip()
            if re.search(r'\bWHERE\s*\{', s, re.I):
                in_where = True; lines.append(s); continue
            if in_where and s.strip() == '}':
                in_where = False; lines.append(s); continue
            if 'FILTER' in s.upper() and not s.endswith('.'):
                s += ' .'
            elif in_where and (not re.search(r'[{},.;]$', s)) and re.search(r'\s+(wdt:|rdfs:|wd:|ddis:|rdf:|schema:| a )', s):
                s += ' .'
            lines.append(s)
        q = '\n'.join(lines)

        # Prefix block if missing
        if 'PREFIX' not in q.upper():
            prefixes = """PREFIX ddis: <http://ddis.ch/atai/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

"""
            q = prefixes + q

        # cleanup
        q = re.sub(r'\.\s*\.', '.', q)
        q = re.sub(r'\n{3,}', '\n\n', q).strip()
        return q

    def _direct_llm_convert(self, question: str) -> Optional[SPARQLQuery]:
        if not self._llm:
            return None

        cleaned = self._preclean(question)
        if cleaned in self._llm_cache:
            raw = self._llm_cache[cleaned]
        else:
            prompt = self._create_few_shot_prompt(cleaned)
            out = self._llm(
                prompt,
                max_tokens=int(NL2SPARQL_LLM_MAX_TOKENS or 512),
                temperature=float(NL2SPARQL_LLM_TEMPERATURE or 0.1),
                stop=["</s>", "Question:", "\n\nQuestion", "EXAMPLES", "Examples:"],
                echo=False,
            )
            raw = out["choices"][0]["text"].strip()
            self._llm_cache[cleaned] = raw

        extracted = self._extract_sparql_from_output(raw)
        post = self._postprocess_sparql(extracted)

        if not self._is_valid_sparql_structure(post):
            return None

        validation = self._validate_and_secure_sparql(post)
        if not validation["valid"]:
            return None

        return SPARQLQuery(
            query=validation["cleaned_query"],
            confidence=0.85,
            explanation="Generated using DeepSeek (model-first)"
        )

    # -------------------- validation helpers --------------------

    def _validate_and_secure_sparql(self, query: str) -> Dict[str, Any]:
        try:
            return {
                'valid': self.sparql_validator.validate_query(query)['valid'],
                'message': self.sparql_validator.validate_query(query)['message'],
                'cleaned_query': query,
                'query_type': self.sparql_validator.validate_query(query).get('query_type', 'SELECT')
            }
        except Exception as e:
            return {'valid': False, 'message': f"Validation error: {e}", 'cleaned_query': None}

    def _is_valid_sparql_structure(self, query: str) -> bool:
        qu = query.upper()
        has_type = any(t in qu for t in ['SELECT','ASK','CONSTRUCT','DESCRIBE'])
        has_where = 'WHERE' in qu or '{' in query
        return has_type and has_where and len(query.strip()) > 10

    # -------------------- public entrypoint --------------------

    def convert(self, question: str) -> SPARQLQuery:
        """
        MODEL-FIRST STRATEGY:
          1) Pre-clean input
          2) Try DeepSeek (llama-cpp) first, if available
          3) Fall back to rule-based templates
        """
        cleaned = self._preclean(question)
        if cleaned in self._convert_cache:
            return self._convert_cache[cleaned]

        # 1) Model first
        if self._llm is not None:
            llm_result = self._direct_llm_convert(cleaned)
            if llm_result:
                self._convert_cache[cleaned] = llm_result
                return llm_result

        # 2) Rule-based fallback
        rule_result = self._rule_based_convert(cleaned)
        if rule_result:
            validation = self._validate_and_secure_sparql(rule_result.query)
            if validation['valid']:
                rule_result.query = validation['cleaned_query']
                rule_result.explanation += f" (validated)"
                self._convert_cache[cleaned] = rule_result
                return rule_result

        # 3) Hard failure
        fail = SPARQLQuery(
            query="# Could not generate valid query",
            confidence=0.0,
            explanation=f"Could not generate a valid SPARQL query for: {question}"
        )
        self._convert_cache[cleaned] = fail
        return fail

    # -------------------- misc --------------------

    def validate_question(self, question: str) -> bool:
        if not question or len(question.strip()) < 3:
            return False
        movie_keywords = ['movie','film','director','actor','actress','release','genre','screenwriter','writer','star','cast','rating']
        ql = question.lower()
        question_words = ['who','what','when','where','which','how']
        has_qw = any(w in ql for w in question_words)
        has_ctx = any(k in ql for k in movie_keywords)
        matches = any(re.search(p['pattern'], ql, re.I) for p in self.patterns)
        return has_qw or has_ctx or matches
