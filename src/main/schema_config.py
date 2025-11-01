"""
Centralized schema configuration for the knowledge graph.
Contains all property mappings, type definitions, and relation metadata.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PropertyMetadata:
    """Metadata for a Wikidata property."""
    property_id: str
    uri: str
    label: str
    synonyms: List[str]
    subject_types: List[str]
    object_types: List[str]
    keywords: List[str]
    priority: int


class SchemaConfig:
    """Centralized schema configuration."""
    
    # Wikidata property definitions
    PROPERTIES: Dict[str, PropertyMetadata] = {
        'director': PropertyMetadata(
            property_id='P57',
            uri='http://www.wikidata.org/prop/direct/P57',
            label='director',
            synonyms=['directed', 'filmmaker', 'film director'],
            subject_types=['movie'],
            object_types=['person'],
            keywords=['directed', 'director of', 'who directed'],
            priority=10
        ),
        'cast_member': PropertyMetadata(
            property_id='P161',
            uri='http://www.wikidata.org/prop/direct/P161',
            label='cast member',
            synonyms=['actor', 'actress', 'starred', 'starring', 'acted', 'performer'],
            subject_types=['movie'],
            object_types=['person'],
            keywords=['actors', 'acted', 'starred', 'main actors', 'cast'],
            priority=9
        ),
        'screenwriter': PropertyMetadata(
            property_id='P58',
            uri='http://www.wikidata.org/prop/direct/P58',
            label='screenwriter',
            synonyms=['writer', 'screenplay', 'wrote', 'written by'],
            subject_types=['movie'],
            object_types=['person'],
            keywords=['wrote', 'written', 'screenplay', 'script'],
            priority=8
        ),
        'producer': PropertyMetadata(
            property_id='P162',
            uri='http://www.wikidata.org/prop/direct/P162',
            label='producer',
            synonyms=['produced', 'produced by'],
            subject_types=['movie'],
            object_types=['person'],
            keywords=['produced', 'producer'],
            priority=8
        ),
        'genre': PropertyMetadata(
            property_id='P136',
            uri='http://www.wikidata.org/prop/direct/P136',
            label='genre',
            synonyms=['type', 'category', 'kind'],
            subject_types=['movie'],
            object_types=['genre'],
            keywords=['genre', 'type of movie'],
            priority=9
        ),
        'publication_date': PropertyMetadata(
            property_id='P577',
            uri='http://www.wikidata.org/prop/direct/P577',
            label='publication date',
            synonyms=['release', 'release date', 'released', 'came out'],
            subject_types=['movie'],
            object_types=['date'],
            keywords=['when', 'released', 'release date', 'came out'],
            priority=9
        ),
        'country_of_origin': PropertyMetadata(
            property_id='P495',
            uri='http://www.wikidata.org/prop/direct/P495',
            label='country of origin',
            synonyms=['country', 'from what country', 'made in', 'produced in'],
            subject_types=['movie'],
            object_types=['country'],
            keywords=['from what country', 'country is', 'country of origin', 'produced in'],
            priority=10
        ),
        'original_language': PropertyMetadata(
            property_id='P364',
            uri='http://www.wikidata.org/prop/direct/P364',
            label='original language',
            synonyms=['language', 'spoken', 'dialogue', 'filmed in language'],
            subject_types=['movie'],
            object_types=['language'],
            keywords=['language', 'spoken in', 'filmed in language'],
            priority=10
        ),
        'composer': PropertyMetadata(
            property_id='P86',
            uri='http://www.wikidata.org/prop/direct/P86',
            label='composer',
            synonyms=['music by', 'composed', 'soundtrack', 'score'],
            subject_types=['movie'],
            object_types=['person'],
            keywords=['composed', 'music by', 'composer'],
            priority=8
        ),
        'country_of_citizenship': PropertyMetadata(
            property_id='P27',
            uri='http://www.wikidata.org/prop/direct/P27',
            label='country of citizenship',
            synonyms=['citizenship', 'nationality', 'born in'],
            subject_types=['person'],
            object_types=['country'],
            keywords=['citizen', 'nationality', 'born in'],
            priority=5
        ),
        'rating': PropertyMetadata(
            property_id='rating',
            uri='http://ddis.ch/atai/rating',
            label='rating',
            synonyms=['rating', 'rated'],
            subject_types=['movie'],
            object_types=['string'],
            keywords=['rating', 'rated'],
            priority=8
        ),
    }
    
    # Entity type definitions
    ENTITY_TYPES = {
        'movie': 'http://www.wikidata.org/entity/Q11424',
        'person': 'http://www.wikidata.org/entity/Q5',
        'country': 'http://www.wikidata.org/entity/Q6256',
        'genre': 'http://www.wikidata.org/entity/Q201658',
        'language': 'http://www.wikidata.org/entity/Q1288568'
    }
    
    # Q-code mappings
    QCODES = {
        'Q11424': 'movie',
        'Q5': 'person',
        'Q6256': 'country',
        'Q201658': 'genre',
        'Q1288568': 'language'
    }
    
    @classmethod
    def get_property_uri(cls, property_name: str) -> str:
        """Get URI for a property name."""
        prop = cls.PROPERTIES.get(property_name)
        return prop.uri if prop else None
    
    @classmethod
    def get_property_by_uri(cls, uri: str) -> str:
        """Get property name from URI."""
        for name, prop in cls.PROPERTIES.items():
            if prop.uri == uri:
                return name
        return None
    
    @classmethod
    def get_relation_uris(cls) -> Dict[str, str]:
        """Get mapping of property names to URIs."""
        return {name: prop.uri for name, prop in cls.PROPERTIES.items()}
    
    @classmethod
    def get_type_uri(cls, type_name: str) -> str:
        """Get URI for entity type."""
        return cls.ENTITY_TYPES.get(type_name)
    
    @classmethod
    def build_keyword_index(cls) -> Dict[str, List[Tuple[str, int]]]:
        """Build keyword to property mapping with priorities."""
        index = {}
        for prop_name, prop in cls.PROPERTIES.items():
            # Add all keywords
            for keyword in prop.keywords:
                if keyword not in index:
                    index[keyword] = []
                index[keyword].append((prop_name, prop.priority))
            
            # Add synonyms
            for synonym in prop.synonyms:
                if synonym not in index:
                    index[synonym] = []
                index[synonym].append((prop_name, prop.priority - 1))
        
        return index
