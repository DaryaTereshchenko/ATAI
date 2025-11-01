"""
Dynamic SPARQL Generator - Creates SPARQL queries based on query patterns.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from typing import Optional
from rdflib import URIRef
from src.main.schema_config import SchemaConfig


class SPARQLGenerator:
    """Generates SPARQL queries dynamically based on query patterns."""
    
    def __init__(self, sparql_handler):
        """Initialize with SPARQLHandler for label normalization."""
        self.sparql_handler = sparql_handler
        
        # Use centralized schema
        self.relation_uris = SchemaConfig.get_relation_uris()
        self.type_uris = SchemaConfig.ENTITY_TYPES.copy()
        
        # Extract dynamic relations from graph and merge
        dynamic_relations = self._extract_relation_uris()
        if dynamic_relations:
            self.relation_uris.update(dynamic_relations)
    
    def _extract_relation_uris(self):
        """Extract additional relations from graph."""
        # ...existing code...
        pass
    
    def generate(self, pattern, subject_label: Optional[str] = None, object_label: Optional[str] = None) -> str:
        """Generate SPARQL query based on pattern."""
        # ...existing code...
        pass
    
    def _generate_forward(self, pattern, subject_label: str) -> str:
        """Generate forward query: Entity → Property"""
        relation_uri = SchemaConfig.get_property_uri(pattern.relation)
        if not relation_uri:
            relation_uri = self.relation_uris.get(pattern.relation)
        
        if not relation_uri:
            raise ValueError(f"Unknown relation: {pattern.relation}")
        
        # ...existing code...