"""
Entity Extractor for extracting entities from natural language queries.
"""

from typing import List, Dict, Tuple, Optional
from rdflib import Graph
from src.main.entity_utils import EntityUtils
from src.main.schema_config import SchemaConfig


class EntityExtractor:
    """Extract entities from natural language queries."""
    
    def __init__(self, graph: Graph):
        """Initialize entity extractor with knowledge graph."""
        self.graph = graph
        self.entity_cache = self._build_entity_cache()
        self.spacy_nlp = None
        self.utils = EntityUtils()
    
    def _build_entity_cache(self) -> Dict[str, List[str]]:
        """Build cache of entity labels to URIs for fast lookup."""
        cache = {}
        
        try:
            count = 0
            for s, p, o in self.graph.triples((None, RDFS.label, None)):
                if isinstance(o, Literal):
                    # ✅ CRITICAL: Store ORIGINAL label from database (no normalization)
                    original_label = str(o)
                    uri = str(s)
                    
                    # ✅ Use lowercase ONLY for lookup key
                    lookup_key = original_label.lower().strip()
                    
                    if lookup_key not in cache:
                        cache[lookup_key] = []
                    
                    # ✅ Store tuple of (uri, original_label) - preserve exact database label
                    entry = (uri, original_label)
                    if entry not in cache[lookup_key]:
                        cache[lookup_key].append(entry)
                    
                    count += 1
            
        except Exception as e:
            print(f"❌ Error building entity cache: {e}")
            import traceback
            traceback.print_exc()
        
        return cache
    
    def _extract_quoted_text(self, query: str) -> List[str]:
        """
        Extract text within quotes (single or double, including smart quotes).
        ✅ NOW: Returns raw text WITHOUT normalization.
        """
        quoted_texts = []
        
        patterns = [
            r'"([^"]+)"',
            r"'([^']+)'",
            r"'([^']+)'",
            r"[\u201c\u201d]([^\u201c\u201d]+)[\u201c\u201d]"
        ]
        
        for pattern in patterns:
            try:
                matches = re.findall(pattern, query)
                for match in matches:
                    cleaned = match.strip()
                    if cleaned:
                        quoted_texts.append(cleaned)
                        print(f"[EntityExtractor] 📝 Extracted quoted: '{cleaned}'")
            except re.error as e:
                print(f"[EntityExtractor] Warning: Regex pattern error: {e}")
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
            print(f"[EntityExtractor] Extracted {len(unique_texts)} unique quoted text(s)")
        
        return unique_texts
    
    def _extract_entities_with_spacy(self, query: str) -> List[str]:
        """Extract named entities using spaCy NER."""
        if self.spacy_nlp is None:
            return []
        
        try:
            doc = self.spacy_nlp(query)
            entities = []
            
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "WORK_OF_ART", "ORG", "GPE"]:
                    entities.append(ent.text.strip())
            
            if entities:
                print(f"[EntityExtractor] 🤖 spaCy NER found: {entities}")
            
            return entities
            
        except Exception as e:
            print(f"[EntityExtractor] ⚠️ spaCy NER error: {e}")
            return []
    
    def _extract_capitalized_spans(self, query: str) -> List[str]:
        """Extract consecutive capitalized word spans as potential entity names."""
        pattern = r"\b[A-Z][\w']+(?:\s+[A-Z][\w']+){0,5}\b"
        matches = re.findall(pattern, query)
        
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
        ✅ NOW: Returns ONLY the most relevant entity when multiple matches exist
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
                # ✅ Normalize for lookup ONLY
                quoted_normalized = ' '.join(quoted_title.lower().split())
                
                print(f"[EntityExtractor] 🔍 Looking up quoted: '{quoted_title}'")
                print(f"[EntityExtractor] 🔍 Normalized for cache: '{quoted_normalized}'")
                
                # Try exact match with normalized key
                if quoted_normalized in self.entity_cache:
                    # ✅ FIX: Prioritize entries by type if filter is provided
                    entries = self.entity_cache[quoted_normalized]
                    
                    # If entity_type filter is provided, prefer matching types
                    if entity_type:
                        matching_entries = [(uri, label) for uri, label in entries 
                                           if self._has_type(uri, entity_type)]
                        if matching_entries:
                            entries = matching_entries
                    
                    # ✅ CRITICAL: Return ONLY the first match to avoid ambiguity
                    for uri, original_label in entries[:1]:  # Take only first
                        print(f"[EntityExtractor] ✅ Exact match: '{quoted_title}' → '{original_label}'")
                        matches.append((uri, original_label, 100))
                        break  # Stop after first match
                else:
                    print(f"[EntityExtractor] ❌ No exact match for '{quoted_normalized}'")
                    
                    # ✅ ENHANCED: Try fuzzy matching
                    print(f"[EntityExtractor] 🔍 Trying fuzzy match...")
                    quoted_words = set(quoted_normalized.split())
                    best_match = None
                    best_similarity = 0.0
                    
                    for cached_label_lower, entries in self.entity_cache.items():
                        cached_words = set(cached_label_lower.split())
                        
                        # Calculate word overlap
                        common = quoted_words & cached_words
                        total = quoted_words | cached_words
                        
                        if len(total) > 0:
                            similarity = len(common) / len(total)
                            
                            if similarity > best_similarity and similarity > 0.7:
                                best_similarity = similarity
                                best_match = (cached_label_lower, entries)
                    
                    if best_match:
                        cached_label, entries = best_match
                        print(f"[EntityExtractor] ✅ Fuzzy match: '{quoted_title}' → '{cached_label}' (similarity: {best_similarity:.2f})")
                        # ✅ FIX: Use original_label from entries
                        for uri, original_label in entries:
                            if entity_type is None or self._has_type(uri, entity_type):
                                matches.append((uri, original_label, int(best_similarity * 95)))
        
        # If we found matches from quoted text, return them immediately
        if matches:
            print(f"[EntityExtractor] ✅ Returning {len(matches)} quoted text matches")
            return self._deduplicate_matches(matches)
        
        # PRIORITY 2: Try spaCy NER
        ner_entities = self._extract_entities_with_spacy(query)
        
        for ner_entity in ner_entities:
            ner_lower = ner_entity.lower()
            
            if ner_lower in self.entity_cache:
                # ✅ FIX: Use (uri, original_label) tuples
                for uri, original_label in self.entity_cache[ner_lower]:
                    if entity_type is None or self._has_type(uri, entity_type):
                        print(f"[EntityExtractor] ✅ NER match: '{ner_entity}' → '{original_label}'")
                        matches.append((uri, original_label, 98))
        
        if matches:        
            print(f"[EntityExtractor] ✅ Returning {len(matches)} NER matches")        
            return self._deduplicate_matches(matches)
        
        # PRIORITY 3: Try capitalized spans
        cap_spans = self._extract_capitalized_spans(query)
        
        for cap_span in cap_spans:
            cap_lower = cap_span.lower()
            
            if cap_lower in self.entity_cache:
                # ✅ FIX: Use (uri, original_label) tuples
                for uri, original_label in self.entity_cache[cap_lower]:
                    if entity_type is None or self._has_type(uri, entity_type):
                        print(f"[EntityExtractor] ✅ Capitalized match: '{cap_span}' → '{original_label}'")
                        matches.append((uri, original_label, 96))
        
        # If we found matches from capitalized spans, return them immediately
        if matches:
            print(f"[EntityExtractor] ✅ Returning {len(matches)} capitalized span matches")
            return self._deduplicate_matches(matches)
        
        # FALLBACK: Pattern-based extraction
        print(f"[EntityExtractor] ⚠️ No high-confidence matches, trying pattern extraction...")
        return self._pattern_based_extraction(query, entity_type, threshold)

    def _deduplicate_matches(
        self,
        matches: List[Tuple[str, str, int]]
    ) -> List[Tuple[str, str, int]]:
        """Remove duplicate matches, keeping highest score for each URI."""
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
        """Fallback: Extract entities using whole-word matching."""
        matches = []
        query_original = query.strip()
        query_lower = query_original.lower()
        
        # Remove common question words
        stop_words = [
            'who', 'what', 'when', 'where', 'which', 'how', 'is', 'was', 'are', 'were',
            'show', 'find', 'list', 'get', 'tell', 'give', 'directed', 'director',
            'screenwriter', 'actor', 'released', 'the', 'of', 'in', 'for', 'about', 'did'
        ]
        
        query_cleaned = query_lower
        for word in stop_words:
            query_cleaned = re.sub(rf'\b{word}\b', ' ', query_cleaned, flags=re.IGNORECASE)
        query_cleaned = re.sub(r'\s+', ' ', query_cleaned).strip()
        
        print(f"[EntityExtractor] 🔍 Pattern matching on: '{query_cleaned}'")
        
        try:
            # Sort labels by length (longest first)
            labels_sorted = sorted(
                (lbl for lbl in self.entity_cache.keys() if len(lbl) >= 3),
                key=lambda x: len(x),
                reverse=True
            )
            
            found_labels = []
            
            for label_lower in labels_sorted:
                if (re.search(rf'\b{re.escape(label_lower)}\b', query_lower) or
                    re.search(rf'\b{re.escape(label_lower)}\b', query_cleaned)):
                    
                    # ✅ FIX: Use (uri, original_label) tuples
                    for uri, original_label in self.entity_cache[label_lower]:
                        if entity_type is None or self._has_type(uri, entity_type):
                            matches.append((uri, original_label, 90))
                            found_labels.append(label_lower)
                            break
            
            if found_labels:
                print(f"[EntityExtractor] ✅ Pattern found: {found_labels[:5]}")
            else:
                print(f"[EntityExtractor] ❌ No pattern matches found")
                
        except Exception as e:
            print(f"[EntityExtractor] ⚠️ Error in pattern extraction: {e}")
            import traceback
            traceback.print_exc()
        
        return self._deduplicate_matches(matches) if matches else []
    
    def _has_type(self, entity_uri: str, type_uri: str) -> bool:
        """Check if entity has a specific type."""
        # Use SchemaConfig for type URIs
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
        return self.utils.get_entity_label(entity_uri, self.graph)