"""
Relation-Based Query Analyzer.
Identifies which relation the query asks about, then infers query structure.
"""

import re
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class RelationQuery:
    """Represents a query about a specific relation."""
    relation: str  # The property name (e.g., 'director', 'genre')
    relation_uri: str  # Full URI
    subject_type: str  # Expected subject type (Q-code or 'unknown')
    object_type: str  # Expected object type (Q-code or 'unknown')
    confidence: float
    keywords: List[str]  # Keywords that triggered this relation


class RelationAnalyzer:
    """Analyzes queries to detect which relation they ask about."""
    
    def __init__(self, sparql_handler):
        """
        Initialize with knowledge graph schema.
        
        Args:
            sparql_handler: SPARQLHandler with loaded graph
        """
        self.sparql_handler = sparql_handler
        
        # ✅ CRITICAL: Use SchemaConfig as single source of truth
        from src.main.schema_config import SchemaConfig
        self.schema = SchemaConfig
        
        # Build relation mappings from graph
        self._build_relation_mappings()
        self._build_keyword_index()
    
    def _build_relation_mappings(self):
        """Extract all relations and their type constraints from graph."""
        print("🔍 Building relation mappings from graph...")
        
        from rdflib import URIRef, RDFS
        
        self.relation_to_types = {}
        self.relation_names = {}
        
        # ✅ Start with SchemaConfig relations as baseline
        for prop_name, prop_meta in self.schema.PROPERTIES.items():
            self.relation_names[prop_name] = prop_meta.uri
            
            # Store type info
            subject_type = prop_meta.subject_types[0] if prop_meta.subject_types else 'unknown'
            object_type = prop_meta.object_types[0] if prop_meta.object_types else 'unknown'
            self.relation_to_types[prop_meta.uri] = (subject_type, object_type)
        
        print(f"✅ Loaded {len(self.relation_names)} relations from SchemaConfig")
        
        # Get all properties from graph (for dynamic extension)
        P31 = URIRef("http://www.wikidata.org/prop/direct/P31")
        
        properties = set()
        for s, p, o in self.sparql_handler.graph:
            pred_str = str(p)
            if 'wikidata.org/prop/direct/P' in pred_str or 'ddis.ch/atai/' in pred_str:
                properties.add(pred_str)
        
        print(f"   Found {len(properties)} total properties in graph")
        
        # Sample each property to infer types
        for prop_uri in properties:
            prop_ref = URIRef(prop_uri)
            
            # Sample subjects and objects
            subject_types = {}
            object_types = {}
            
            for s, p, o in list(self.sparql_handler.graph.triples((None, prop_ref, None)))[:20]:
                # Get subject type
                for subj_type in self.sparql_handler.graph.objects(s, P31):
                    type_qcode = self._extract_qcode(str(subj_type))
                    if type_qcode:
                        subject_types[type_qcode] = subject_types.get(type_qcode, 0) + 1
                
                # Get object type
                if isinstance(o, URIRef):
                    for obj_type in self.sparql_handler.graph.objects(o, P31):
                        type_qcode = self._extract_qcode(str(obj_type))
                        if type_qcode:
                            object_types[type_qcode] = object_types.get(type_qcode, 0) + 1
            
            # Store most common types
            if subject_types:
                most_common_subj = max(subject_types.items(), key=lambda x: x[1])[0]
            else:
                most_common_subj = 'unknown'
            
            if object_types:
                most_common_obj = max(object_types.items(), key=lambda x: x[1])[0]
            else:
                most_common_obj = 'unknown'
            
            self.relation_to_types[prop_uri] = (most_common_subj, most_common_obj)
            
            # ✅ CRITICAL: Get property label from graph (dynamic)
            property_label = self._get_property_label(prop_ref)
            
            if property_label:
                # ✅ Store BOTH the full label AND simplified versions
                self.relation_names[property_label] = prop_uri
                
                # Also store simplified version for better matching
                simplified = property_label.replace('_of_film_or_tv_show', '')
                if simplified != property_label:
                    self.relation_names[simplified] = prop_uri
                    print(f"   ✅ Mapped: '{property_label}' → {prop_uri}")
                    print(f"      Also: '{simplified}' → {prop_uri}")
            else:
                # Fallback: extract from URI
                if '/P' in prop_uri:
                    prop_id = prop_uri.split('/P')[-1]
                    self.relation_names[f"P{prop_id}"] = prop_uri
                else:
                    prop_name = prop_uri.split('/')[-1]
                    self.relation_names[prop_name] = prop_uri
        
        print(f"✅ Total relation mappings: {len(self.relation_names)}")
    
    def _get_property_label(self, prop_ref) -> Optional[str]:
        """Get rdfs:label for a property."""
        from rdflib import RDFS
        
        for label in self.sparql_handler.graph.objects(prop_ref, RDFS.label):
            label_str = str(label).lower()
            # Clean up
            label_str = label_str.replace(' - wikidata', '').replace('wikidata property for ', '')
            return label_str.strip().replace(' ', '_').replace('-', '_')
        return None
    
    def _extract_qcode(self, uri: str) -> Optional[str]:
        """Extract Q-code from Wikidata URI."""
        if '/Q' in uri:
            return uri.split('/Q')[-1].split('#')[0]
        return None
    
    def _build_keyword_index(self):
        """Build keyword → relation mapping for fast lookup."""
        self.keyword_to_relations = {}
        
        # ✅ STEP 1: Build from schema synonyms (PRIMARY SOURCE)
        print(f"🔍 Building keyword index from SchemaConfig...")
        for prop_name, prop_meta in self.schema.PROPERTIES.items():
            # Add all keywords with high priority
            for keyword in prop_meta.keywords:
                if keyword not in self.keyword_to_relations:
                    self.keyword_to_relations[keyword] = []
                self.keyword_to_relations[keyword].append((
                    prop_name,
                    prop_meta.uri,
                    prop_meta.priority
                ))
            
            # Add all synonyms with slightly lower priority
            for synonym in prop_meta.synonyms:
                if synonym not in self.keyword_to_relations:
                    self.keyword_to_relations[synonym] = []
                # Check if not already added (avoid duplicates)
                if not any(r[0] == prop_name for r in self.keyword_to_relations[synonym]):
                    self.keyword_to_relations[synonym].append((
                        prop_name,
                        prop_meta.uri,
                        prop_meta.priority - 1  # Slightly lower priority than keywords
                    ))
            
            print(f"   ✅ {prop_name}: {len(prop_meta.keywords)} keywords + {len(prop_meta.synonyms)} synonyms")
        
        # ✅ STEP 2: Add dynamic keywords from relation names (SECONDARY)
        for relation_name, relation_uri in self.relation_names.items():
            # Generate keywords from relation name
            keywords = self._generate_keywords_from_relation(relation_name)
            
            for keyword, priority in keywords:
                if keyword not in self.keyword_to_relations:
                    self.keyword_to_relations[keyword] = []
                
                # Only add if not already present with higher priority
                existing_priorities = [r[2] for r in self.keyword_to_relations[keyword] if r[0] == relation_name]
                if not existing_priorities or priority > max(existing_priorities):
                    self.keyword_to_relations[keyword].append((
                        relation_name,
                        relation_uri,
                        priority
                    ))
        
        # ✅ STEP 3: Add explicit high-priority phrase patterns (OVERRIDES)
        explicit_phrases = {
            # Release date variations
            'release date': [('publication_date', None, 20)],
            'came out': [('publication_date', None, 20)],
            'come out': [('publication_date', None, 20)],
            'when was released': [('publication_date', None, 20)],
            'when did release': [('publication_date', None, 20)],
            'when was it released': [('publication_date', None, 20)],
            
            # Cinematographer (NEVER director)
            'cinematographer': [('director_of_photography', None, 20)],
            'director of photography': [('director_of_photography', None, 20)],
            
            # Screenplay writer
            'wrote the screenplay': [('screenwriter', None, 20)],
            'who wrote the screenplay': [('screenwriter', None, 20)],
            'screenplay by': [('screenwriter', None, 20)],
            
            # Producer
            'produced by': [('producer', None, 18)],
            'producer': [('producer', None, 18)],
            
            # Production company
            'which studio': [('production_company', None, 18)],
            'what studio': [('production_company', None, 18)],
            'studio produced': [('production_company', None, 18)],
            'production company': [('production_company', None, 18)],
            'film studio': [('production_company', None, 18)],
            
            # Distributor
            'distributed by': [('distributed_by', None, 18)],
            'distributor': [('distributed_by', None, 18)],
            'which distributor': [('distributed_by', None, 18)],
            
            # Composer
            'music by': [('composer', None, 18)],
            'composed': [('composer', None, 18)],
            'composer': [('composer', None, 18)],
            'who composed': [('composer', None, 18)],
            'who scored': [('composer', None, 18)],
        }
        
        phrase_count = 0
        for phrase, relations in explicit_phrases.items():
            if phrase not in self.keyword_to_relations:
                self.keyword_to_relations[phrase] = []
            # Add with very high priority
            for relation_name, relation_uri, priority in relations:
                # Only add if not already present
                if not any(r[0] == relation_name for r in self.keyword_to_relations[phrase]):
                    self.keyword_to_relations[phrase].append((relation_name, relation_uri, priority))
                    phrase_count += 1
        
        print(f"✅ Built keyword index: {len(self.keyword_to_relations)} total keywords")
        print(f"   • {phrase_count} explicit phrase patterns added")
        print(f"   • Synonyms from schema: active")
        print(f"   • Dynamic relation keywords: active")
    
    def _generate_keywords_from_relation(self, relation_name: str) -> List[Tuple[str, int]]:
        """
        Generate search keywords from a relation name.
        
        Args:
            relation_name: Relation name (e.g., 'original_language_of_film_or_tv_show')
            
        Returns:
            List of (keyword, priority) tuples
        """
        keywords = []
        
        # Split by underscores
        parts = relation_name.split('_')
        
        # Add full name (highest priority)
        keywords.append((relation_name, 10))
        
        # Add combinations of parts
        if len(parts) > 1:
            # First N words (e.g., 'original_language')
            for i in range(1, len(parts)):
                partial = '_'.join(parts[:i+1])
                keywords.append((partial, 9 - i))
            
            # Key words to
