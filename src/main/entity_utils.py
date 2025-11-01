"""
Utility functions for entity extraction and normalization.
Consolidates common entity handling logic.
"""

import re
from typing import List, Tuple, Optional
from rdflib import Graph, URIRef, RDFS


class EntityUtils:
    """Common entity extraction and normalization utilities."""
    
    @staticmethod
    def extract_quoted_text(query: str) -> List[str]:
        """Extract text within quotes (single, double, or smart quotes)."""
        quoted_texts = []
        
        patterns = [
            r'"([^"]+)"',
            r"'([^']+)'",
            r"'([^']+)'",
            r"[\u201c\u201d]([^\u201c\u201d]+)[\u201c\u201d]"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query)
            quoted_texts.extend(match.strip() for match in matches if match.strip())
        
        return list(dict.fromkeys(quoted_texts))  # Remove duplicates, preserve order
    
    @staticmethod
    def extract_capitalized_spans(query: str, max_words: int = 5) -> List[str]:
        """Extract consecutive capitalized word spans."""
        pattern = rf"\b[A-Z][\w']+(?:\s+[A-Z][\w']+){{0,{max_words-1}}}\b"
        matches = re.findall(pattern, query)
        
        stop_words = {'What', 'Who', 'When', 'Where', 'Which', 'How', 'Why', 
                     'Show', 'Tell', 'Find', 'List', 'Get', 'Did', 'Is', 'Was'}
        
        return [m for m in matches if m not in stop_words]
    
    @staticmethod
    def get_entity_label(entity_uri: str, graph: Graph) -> str:
        """Get human-readable label for an entity."""
        entity_ref = URIRef(entity_uri)
        
        for label in graph.objects(entity_ref, RDFS.label):
            return str(label)
        
        # Fallback to URI fragments
        if '#' in entity_uri:
            return entity_uri.split('#')[-1]
        elif '/' in entity_uri:
            return entity_uri.split('/')[-1]
        
        return entity_uri
    
    @staticmethod
    def get_entity_qcode(entity_uri: str, graph: Graph) -> str:
        """Get Wikidata Q-code for an entity's type."""
        from rdflib import URIRef
        
        entity_ref = URIRef(entity_uri)
        P31 = URIRef("http://www.wikidata.org/prop/direct/P31")
        
        for type_uri in graph.objects(entity_ref, P31):
            type_str = str(type_uri)
            if '/entity/Q' in type_str:
                return 'Q' + type_str.split('/entity/Q')[-1].split('#')[0].split('/')[0]
            elif '/Q' in type_str:
                return 'Q' + type_str.split('/Q')[-1].split('#')[0].split('/')[0]
        
        # Fallback: check entity URI itself
        if '/entity/Q' in entity_uri:
            return 'Q' + entity_uri.split('/entity/Q')[-1].split('#')[0].split('/')[0]
        elif '/Q' in entity_uri:
            return 'Q' + entity_uri.split('/Q')[-1].split('#')[0].split('/')[0]
        
        return "unknown"
    
    @staticmethod
    def escape_sparql_string(text: str) -> str:
        """Escape string for SPARQL query."""
        return text.replace('\\', '\\\\').replace('"', '\\"')
    
    @staticmethod
    def normalize_for_lookup(text: str) -> str:
        """Normalize text for cache lookup (lowercase, clean whitespace)."""
        return ' '.join(text.lower().split())
