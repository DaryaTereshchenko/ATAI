"""
Entity Extractor for extracting entities from natural language queries.
Optimized for quoted movie title extraction with proper English title case normalization.
Enhanced with spaCy NER for better person/entity name extraction.
"""

import re
from typing import List, Dict, Tuple, Optional
from rdflib import Graph, URIRef, RDFS, RDF, Literal


class EntityExtractor:
    """Extract entities from natural language queries with focus on quoted text and NER."""
    
    def __init__(self, graph: Graph):
        """
        Initialize entity extractor with knowledge graph.
        
        Args:
            graph: RDFLib graph containing entities
        """
        self.graph = graph
        self.entity_cache = self._build_entity_cache()
        self.spacy_nlp = None  # Will be set by orchestrator if available
    
    def _build_entity_cache(self) -> Dict[str, List[str]]:
        """Build cache of entity labels to URIs for fast lookup."""
        print("🔨 Building entity cache from knowledge graph...")
        cache = {}
        
        try:
            # Get all entities with labels
            for s, p, o in self.graph.triples((None, RDFS.label, None)):
                if isinstance(o, Literal):
                    label = str(o).strip()
                    uri = str(s)
                    
                    # ✅ CRITICAL: Store ONLY lowercase keys for case-insensitive lookup
                    label_lower = label.lower()
                    
                    if label_lower not in cache:
                        cache[label_lower] = []
                    cache[label_lower].append(uri)
            
            print(f"✅ Cached {len(cache)} entity labels (case-insensitive)")
            
            # Debug: Show some sample entries
            sample_labels = list(cache.keys())[:5]
            print(f"📋 Sample labels: {sample_labels}")
            
        except Exception as e:
            print(f"❌ Error building entity cache: {e}")
        
        return cache
    
    def _normalize_title(self, title: str) -> str:
        """
        Normalize movie title to proper English title case.
        
        Rules:
        - First and last words always capitalized
        - Articles, conjunctions, short prepositions lowercase (unless first/last)
        - Roman numerals preserved as uppercase
        - Proper nouns capitalized
        
        Examples:
            "the bridge on the river kwai" → "The Bridge on the River Kwai"
            "star wars: episode vi - return of the jedi" → "Star Wars: Episode VI - Return of the Jedi"
            "lord of the rings" → "Lord of the Rings"
        """
        if not title:
            return title
        
        # Words that should be lowercase in title case (unless first/last)
        lowercase_words = {
            'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'from', 'in',
            'into', 'of', 'on', 'or', 'over', 'the', 'to', 'up', 'with', 'via',
            'vs', 'vs.', 'v.', 'v'
        }
        
        # Roman numerals should stay uppercase
        roman_numerals = {'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x'}
        
        words = title.split()
        result_words = []
        
        for i, word in enumerate(words):
            # Preserve existing punctuation
            # Split word and punctuation
            match = re.match(r'^([^\w]*)(\w+)([^\w]*)$', word)
            if not match:
                # Word is all punctuation or empty
                result_words.append(word)
                continue
            
            prefix, core, suffix = match.groups()
            core_lower = core.lower()
            
            # Check if it's a Roman numeral
            if core_lower in roman_numerals:
                result_words.append(prefix + core.upper() + suffix)
                continue
            
            # Always capitalize first and last word
            if i == 0 or i == len(words) - 1:
                result_words.append(prefix + core_lower.capitalize() + suffix)
            # Check if word should be lowercase
            elif core_lower in lowercase_words:
                result_words.append(prefix + core_lower + suffix)
            # Capitalize other words
            else:
                result_words.append(prefix + core_lower.capitalize() + suffix)
        
        return ' '.join(result_words)
    
    def _extract_quoted_text(self, query: str) -> List[str]:
        """
        Extract text within quotes (single or double, including smart quotes).
        
        Args:
            query: User query
            
        Returns:
            List of quoted strings (normalized to title case)
        """
        quoted_texts = []
        
        # ✅ FIXED: Use character class for smart quotes instead of alternation
        patterns = [
            r"'([^']+)'",                           # Single quotes
            r'"([^"]+)"',                           # Double quotes
            r"'([^']+)'",                           # Smart single quotes (U+2018, U+2019)
            r"[\u201c\u201d]([^\u201c\u201d]+)[\u201c\u201d]"  # Smart double quotes (U+201C, U+201D)
        ]
        
        for pattern in patterns:
            try:
                matches = re.findall(pattern, query)
                for match in matches:
                    normalized = self._normalize_title(match)
                    if normalized:
                        quoted_texts.append(normalized)
            except re.error as e:
                # Skip patterns that cause regex errors
                print(f"[EntityExtractor] Warning: Regex pattern error for pattern '{pattern}': {e}")
                continue
        
        # Remove duplicates while preserving order
        unique_texts = []
        seen = set()
        
        for text in quoted_texts:
            text_lower = text.lower()
            if text_lower not in seen:
                seen.add(text_lower)
                unique_texts.append(text)
        
        if unique_texts:
            print(f"[EntityExtractor] Extracted {len(unique_texts)} quoted text(s): {unique_texts}")
        
        return unique_texts
    
    def _extract_entities_with_spacy(self, query: str) -> List[str]:
        """
        Extract named entities using spaCy NER.
        
        Args:
            query: User query
            
        Returns:
            List of entity text spans detected by NER
        """
        if self.spacy_nlp is None:
            return []
        
        try:
            doc = self.spacy_nlp(query)
            entities = []
            
            for ent in doc.ents:
                # Focus on entity types relevant to movies
                if ent.label_ in ["PERSON", "WORK_OF_ART", "ORG", "GPE"]:
                    entities.append(ent.text.strip())
            
            if entities:
                print(f"[EntityExtractor] 🤖 spaCy NER found: {entities}")
            
            return entities
            
        except Exception as e:
            print(f"[EntityExtractor] ⚠️ spaCy NER error: {e}")
            return []
    
    def _extract_capitalized_spans(self, query: str) -> List[str]:
        """
        Extract consecutive capitalized word spans as potential entity names.
        
        Args:
            query: User query
            
        Returns:
            List of capitalized spans (e.g., "Christopher Nolan", "Martin Scorsese")
        """
        # Match sequences of capitalized words (including possessives like "O'Connor")
        # Pattern: Capitalized word followed by 0-5 more capitalized words
        pattern = r"\b[A-Z][\w']+(?:\s+[A-Z][\w']+){0,5}\b"
        
        matches = re.findall(pattern, query)
        
        # Filter out common false positives
        stop_words = {'What', 'Who', 'When', 'Where', 'Which', 'How', 'Why', 'Show', 'Tell', 'Find', 'List', 'Get'}
        
        filtered = [m for m in matches if m not in stop_words and len(m.split()) <= 6]
        
        if filtered:
            print(f"[EntityExtractor] 📝 Capitalized spans: {filtered}")
        
        return filtered

    def extract_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        threshold: int = 85
    ) -> List[Tuple[str, str, int]]:
        """
        Extract entities from query with PRIORITY on quoted text, then NER, then patterns.
        
        Strategy:
        1. Try quoted text (highest priority, exact match required)
        2. Try spaCy NER entities (high confidence)
        3. Try capitalized spans (person names) ✅ ENHANCED
        4. Try whole-word pattern matching (longest first)
        
        Args:
            query: Natural language query
            entity_type: Optional entity type URI to filter by
            threshold: (unused) kept for API compatibility
            
        Returns:
            List of (entity_uri, matched_text, confidence_score) tuples
        """
        print(f"\n[EntityExtractor] {'='*60}")
        print(f"[EntityExtractor] Extracting entities from: '{query}'")
        print(f"[EntityExtractor] Entity type filter: {entity_type or 'None'}")
        print(f"[EntityExtractor] {'='*60}")
        
        matches = []
        
        # PRIORITY 1: Extract quoted text first
        quoted_titles = self._extract_quoted_text(query)
        
        if quoted_titles:
            print(f"[EntityExtractor] 📝 Found quoted text: {quoted_titles}")
            
            for quoted_title in quoted_titles:
                quoted_lower = quoted_title.lower()
                
                print(f"[EntityExtractor] 🔍 Looking up quoted: '{quoted_title}'")
                
                # Try exact match (case-insensitive via lowercase key)
                if quoted_lower in self.entity_cache:
                    for uri in self.entity_cache[quoted_lower]:
                        if entity_type is None or self._has_type(uri, entity_type):
                            original_label = self.get_entity_label(uri)
                            print(f"[EntityExtractor] ✅ Exact match: '{quoted_title}' → '{original_label}'")
                            matches.append((uri, original_label, 100))
                else:
                    print(f"[EntityExtractor] ❌ No exact match for quoted '{quoted_title}'")
        
        # If we found matches from quoted text, return them immediately
        if matches:
            print(f"[EntityExtractor] ✅ Returning {len(matches)} quoted text matches")
            return self._deduplicate_matches(matches)
        
        # PRIORITY 2: Try spaCy NER
        ner_entities = self._extract_entities_with_spacy(query)
        
        for ner_entity in ner_entities:
            ner_lower = ner_entity.lower()
            
            # Try exact match
            if ner_lower in self.entity_cache:
                for uri in self.entity_cache[ner_lower]:
                    if entity_type is None or self._has_type(uri, entity_type):
                        original_label = self.get_entity_label(uri)
                        print(f"[EntityExtractor] ✅ NER match: '{ner_entity}' → '{original_label}'")
                        matches.append((uri, original_label, 98))
        
        if matches:
            print(f"[EntityExtractor] ✅ Returning {len(matches)} NER matches")
            return self._deduplicate_matches(matches)
        
        # PRIORITY 3: Try capitalized spans (person names) - ENHANCED
        cap_spans = self._extract_capitalized_spans(query)
        
        for cap_span in cap_spans:
            cap_lower = cap_span.lower()
            
            # ✅ ENHANCED: Try exact match first
            if cap_lower in self.entity_cache:
                for uri in self.entity_cache[cap_lower]:
                    if entity_type is None or self._has_type(uri, entity_type):
                        original_label = self.get_entity_label(uri)
                        print(f"[EntityExtractor] ✅ Capitalized exact match: '{cap_span}' → '{original_label}'")
                        matches.append((uri, original_label, 96))
            else:
                # ✅ ENHANCED: Try comprehensive fuzzy matching
                print(f"[EntityExtractor] 🔍 Trying comprehensive fuzzy match for: '{cap_span}'")
                
                fuzzy_matches = self._fuzzy_match_person_name(cap_span, entity_type)
                if fuzzy_matches:
                    matches.extend(fuzzy_matches)
                    print(f"[EntityExtractor] ✅ Found {len(fuzzy_matches)} fuzzy matches")
        
        if matches:
            print(f"[EntityExtractor] ✅ Returning {len(matches)} capitalized span matches")
            return self._deduplicate_matches(matches)
        
        # FALLBACK: Pattern-based extraction with whole-word matching
        print(f"[EntityExtractor] ⚠️  No high-confidence matches, trying pattern extraction...")
        return self._pattern_based_extraction(query, entity_type, threshold)
    
    def _fuzzy_match_person_name(
        self,
        name: str,
        entity_type: Optional[str]
    ) -> List[Tuple[str, str, int]]:
        """
        Comprehensive fuzzy matching for person names.
        Handles various name formats and variations.
        
        Args:
            name: Person name to match (e.g., "Quentin Tarantino")
            entity_type: Optional entity type filter
            
        Returns:
            List of (uri, label, confidence) tuples
        """
        matches = []
        name_lower = name.lower()
        parts = name_lower.split()
        
        if len(parts) < 2:
            return matches
        
        # Generate variations
        variations = []
        
        # 1. Original: "quentin tarantino"
        variations.append(name_lower)
        
        # 2. Reversed: "tarantino quentin"
        variations.append(f"{parts[-1]} {parts[0]}")
        
        # 3. Comma-separated: "tarantino, quentin"
        variations.append(f"{parts[-1]}, {parts[0]}")
        
        # 4. Last name only: "tarantino"
        variations.append(parts[-1])
        
        # 5. First name only: "quentin"
        variations.append(parts[0])
        
        # 6. All parts reversed with commas: "tarantino, quentin jerome"
        if len(parts) > 2:
            variations.append(f"{parts[-1]}, {' '.join(parts[:-1])}")
        
        # 7. Middle name variations
        if len(parts) == 3:
            # "quentin jerome tarantino" → "tarantino, quentin j."
            variations.append(f"{parts[-1]}, {parts[0]} {parts[1][0]}.")
            # "quentin j tarantino"
            variations.append(f"{parts[0]} {parts[1][0]} {parts[-1]}")
        
        print(f"[EntityExtractor]    Generated {len(variations)} name variations")
        print(f"[EntityExtractor]    Variations: {variations[:5]}...")  # Show first 5
        
        # Try each variation
        for variation in variations:
            if variation in self.entity_cache:
                for uri in self.entity_cache[variation]:
                    if entity_type is None or self._has_type(uri, entity_type):
                        original_label = self.get_entity_label(uri)
                        print(f"[EntityExtractor]    ✅ Variation '{variation}' → '{original_label}'")
                        # Confidence decreases with more transformations
                        confidence = 94 if variation == name_lower else 92
                        matches.append((uri, original_label, confidence))
                        break  # Take first valid match for this variation
        
        # If still no matches, try partial substring matching on cache keys
        if not matches:
            print(f"[EntityExtractor]    Trying substring matching...")
            last_name = parts[-1]
            first_name = parts[0]
            
            # Search cache keys that contain both first and last name
            for cache_key in self.entity_cache.keys():
                if last_name in cache_key and first_name in cache_key:
                    for uri in self.entity_cache[cache_key]:
                        if entity_type is None or self._has_type(uri, entity_type):
                            original_label = self.get_entity_label(uri)
                            print(f"[EntityExtractor]    ✅ Substring match '{cache_key}' → '{original_label}'")
                            matches.append((uri, original_label, 90))
                            break
                    if matches:
                        break  # Stop after finding first match
        
        return matches

    def _deduplicate_matches(
        self,
        matches: List[Tuple[str, str, int]]
    ) -> List[Tuple[str, str, int]]:
        """
        Remove duplicate matches, keeping highest score for each URI.
        
        Args:
            matches: List of (uri, text, score) tuples
            
        Returns:
            Deduplicated list
        """
        seen_uris = {}
        for uri, text, score in matches:
            if uri not in seen_uris or score > seen_uris[uri][1]:
                seen_uris[uri] = (text, score)
        
        unique_matches = [(uri, text, score) for uri, (text, score) in seen_uris.items()]
        unique_matches.sort(key=lambda x: x[2], reverse=True)
        
        print(f"[EntityExtractor] 🎯 Returning {len(unique_matches)} unique matches")
        return unique_matches
    
    def _pattern_based_extraction(
        self,
        query: str,
        entity_type: Optional[str],
        threshold: int
    ) -> List[Tuple[str, str, int]]:
        """
        Fallback: Extract entities using whole-word matching.
        
        ✅ FIXED: Works on ORIGINAL case-sensitive query, not lowercase.
        
        Strategy:
        - Sort labels by length (longest first) to prefer complete multi-word names
        - Use word boundary matching to avoid partial matches
        - Skip very short labels to reduce false positives
        """
        matches = []
        query_original = query.strip()  # ✅ Keep original case
        
        # Remove common question words and verbs for better matching
        # ✅ Use case-insensitive regex but preserve original query
        stop_words = [
            'who', 'what', 'when', 'where', 'which', 'how', 'is', 'was', 'are', 'were',
            'show', 'find', 'list', 'get', 'tell', 'give', 'directed', 'director',
            'screenwriter', 'actor', 'released', 'the', 'of', 'in', 'for', 'about', 'did'
        ]
        
        query_cleaned = query_original
        for word in stop_words:
            # ✅ Remove stop words but preserve case of remaining text
            query_cleaned = re.sub(rf'\b{word}\b', ' ', query_cleaned, flags=re.IGNORECASE)
        query_cleaned = re.sub(r'\s+', ' ', query_cleaned).strip()
        
        print(f"[EntityExtractor] 🔍 Pattern matching on: '{query_cleaned}'")
        
        try:
            # Sort labels by length (longest first) to prefer multi-word matches
            labels_sorted = sorted(
                (lbl for lbl in self.entity_cache.keys() if len(lbl) >= 3),
                key=lambda x: len(x),
                reverse=True
            )
            
            found_labels = []
            
            for label_lower in labels_sorted:
                # ✅ CRITICAL: Use case-insensitive matching
                # Match in both original and cleaned query
                pattern_original = re.compile(rf'\b{re.escape(label_lower)}\b', re.IGNORECASE)
                pattern_cleaned = re.compile(rf'\b{re.escape(label_lower)}\b', re.IGNORECASE)
                
                if pattern_original.search(query_original) or pattern_cleaned.search(query_cleaned):
                    for uri in self.entity_cache[label_lower]:
                        if entity_type is None or self._has_type(uri, entity_type):
                            original_label = self.get_entity_label(uri)
                            matches.append((uri, original_label, 90))
                            found_labels.append(label_lower)
                            # Once we find a match for this label, stop checking URIs
                            break
            
            if found_labels:
                print(f"[EntityExtractor] ✅ Pattern found: {found_labels[:5]}")  # Show first 5
            else:
                print(f"[EntityExtractor] ❌ No pattern matches found")
                
        except Exception as e:
            print(f"[EntityExtractor] ⚠️ Error in pattern extraction: {e}")
            import traceback
            traceback.print_exc()
        
        return self._deduplicate_matches(matches) if matches else []
    
    def _has_type(self, entity_uri: str, type_uri: str) -> bool:
        """Check if entity has a specific type."""
        try:
            entity_ref = URIRef(entity_uri)
            type_ref = URIRef(type_uri)
            
            # Check wdt:P31 (instance of)
            P31 = URIRef("http://www.wikidata.org/prop/direct/P31")
            
            for obj in self.graph.objects(entity_ref, P31):
                if obj == type_ref:
                    return True
            
            # Check rdf:type
            for obj in self.graph.objects(entity_ref, RDF.type):
                if obj == type_ref:
                    return True
        except Exception as e:
            print(f"[EntityExtractor] ⚠️ Error checking type for {entity_uri}: {e}")
        
        return False
    
    def get_entity_label(self, entity_uri: str) -> str:
        """Get label for an entity."""
        try:
            entity_ref = URIRef(entity_uri)
            
            for label in self.graph.objects(entity_ref, RDFS.label):
                return str(label)
        except Exception as e:
            print(f"[EntityExtractor] ⚠️ Error getting label for {entity_uri}: {e}")
        
        # Fallback: extract from URI
        if '#' in entity_uri:
            return entity_uri.split('#')[-1]
        elif '/' in entity_uri:
            return entity_uri.split('/')[-1]
        
        return entity_uri