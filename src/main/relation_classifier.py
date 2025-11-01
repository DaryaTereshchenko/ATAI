"""
Lightweight Relation Classifier using DistilBERT or zero-shot SBERT.
Handles all 496+ properties dynamically extracted from the knowledge graph.
"""

import torch
from typing import List, Tuple, Optional, Dict
import numpy as np
import os


class HybridRelationAnalyzer:
    """
    Hybrid relation analysis: BERT classifier (if available) + SBERT zero-shot fallback + Embedding fallback.
    Handles all properties dynamically extracted from graph.
    """
    
    def __init__(
        self,
        classifier_path: Optional[str] = None,
        sparql_handler = None,
        use_sbert: bool = True,
        embedding_matcher = None
    ):
        """
        Initialize hybrid analyzer.
        
        Args:
            classifier_path: Path to fine-tuned DistilBERT (optional)
            sparql_handler: SPARQLHandler for property schema
            use_sbert: Whether to use SBERT zero-shot as fallback
        """
        self.sparql_handler = sparql_handler
        self.use_sbert = use_sbert
        
        # Extract property descriptions from graph
        self.property_descriptions = {}
        if sparql_handler:
            self._build_property_descriptions()
        
        # Try to load fine-tuned BERT classifier
        self.bert_classifier = None
        self.tokenizer = None
        self.device = None
        
        # ✅ FIX: Load DistilBERT model if path provided
        if classifier_path and os.path.exists(classifier_path):
            print(f"🤖 Loading fine-tuned relation classifier from {classifier_path}...")
            
            # ✅ Check for required files first
            config_path = os.path.join(classifier_path, "config.json")
            if not os.path.exists(config_path):
                print(f"⚠️  config.json not found in {classifier_path}")
                print(f"   Skipping DistilBERT classifier")
            else:
                try:
                    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
                    
                    # Load tokenizer
                    self.tokenizer = DistilBertTokenizer.from_pretrained(classifier_path)
                    
                    # Load model
                    self.bert_classifier = DistilBertForSequenceClassification.from_pretrained(classifier_path)
                    self.bert_classifier.eval()
                    
                    # Move to GPU if available
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.bert_classifier.to(self.device)
                    
                    # Get label mappings
                    self.id2label = self.bert_classifier.config.id2label
                    self.label2id = self.bert_classifier.config.label2id
                    
                    print(f"✅ BERT relation classifier loaded successfully")
                    print(f"   Model classes: {len(self.id2label)}")
                    print(f"   Device: {self.device}")
                    
                    # ✅ REMOVED: Don't disable SBERT - keep it as fallback
                    # use_sbert = False  # DELETED THIS LINE
                    
                except Exception as e:
                    print(f"⚠️  Failed to load DistilBERT: {e}")
                    import traceback
                    traceback.print_exc()
                    self.bert_classifier = None
        elif classifier_path:
            print(f"⚠️  Classifier path does not exist: {classifier_path}")
        
        # ✅ Initialize SBERT zero-shot matcher (as fallback for low-confidence BERT results)
        self.sbert_matcher = None
        if use_sbert:  # ✅ CHANGED: Always try to init if use_sbert=True, regardless of BERT status
            try:
                from sentence_transformers import SentenceTransformer, util
                print(f"🔢 Loading SBERT for zero-shot relation matching...")
                self.sbert_matcher = SentenceTransformer('all-MiniLM-L6-v2')
                
                # ✅ CRITICAL: Import util module for cosine similarity
                self.sbert_util = util
                
                # Pre-encode property descriptions
                self._encode_property_descriptions()
                print(f"✅ SBERT matcher ready for {len(self.property_descriptions)} properties")
            except ImportError as e:
                print(f"⚠️  SBERT not available: {e}")
                print(f"   Install with: pip install sentence-transformers")
                self.sbert_matcher = None
            except Exception as e:
                print(f"⚠️  SBERT matcher initialization failed: {e}")
                import traceback
                traceback.print_exc()
                self.sbert_matcher = None
        
        # ✅ NEW: Add embedding matcher as final fallback
        self.embedding_matcher = embedding_matcher
        
        # Status summary
        if self.bert_classifier:
            print(f"✅ Using trained DistilBERT relation classifier ({len(self.id2label)} classes)")
        elif self.sbert_matcher:
            print(f"✅ Using zero-shot SBERT relation matcher ({len(self.property_descriptions)} properties)")
        elif self.embedding_matcher:
            print(f"✅ Using TransE embedding relation matcher (all relations)")
        else:
            print(f"⚠️  Using keyword-only fallback (limited coverage)")
    
    def _build_property_descriptions(self):
        """Build human-readable descriptions for all properties in graph."""
        from rdflib import URIRef, RDFS
        
        # print(f"🔍 Building property descriptions from graph...")  # REMOVED
        
        # Extract all properties and their labels
        for s, p, o in self.sparql_handler.graph:
            pred_str = str(p)
            
            # Only process Wikidata/custom properties
            if 'wikidata.org/prop/direct/P' not in pred_str and 'ddis.ch/atai/' not in pred_str:
                continue
            
            if pred_str not in self.property_descriptions:
                # Get property label
                prop_ref = URIRef(pred_str)
                label = None
                
                for lbl in self.sparql_handler.graph.objects(prop_ref, RDFS.label):
                    label = str(lbl).lower()
                    # Clean up
                    label = label.replace(' - wikidata', '').replace('wikidata property for ', '')
                    break
                
                # Create searchable description
                if label:
                    # Extract property ID for reference
                    if '/P' in pred_str:
                        prop_id = pred_str.split('/P')[-1]
                        description = f"{label} (P{prop_id})"
                    else:
                        prop_name = pred_str.split('/')[-1]
                        description = f"{label} ({prop_name})"
                    
                    self.property_descriptions[pred_str] = {
                        'label': label,
                        'description': description,
                        'uri': pred_str
                    }
        
        # print(f"✅ Built descriptions for {len(self.property_descriptions)} properties")  # REMOVED
    
    def _encode_property_descriptions(self):
        """Pre-encode all property descriptions with SBERT."""
        if not self.sbert_matcher:
            return
        
        self.property_uris = list(self.property_descriptions.keys())
        self.property_texts = [
            info['description'] 
            for info in self.property_descriptions.values()
        ]
        
        # print(f"   Encoding {len(self.property_texts)} property descriptions...")  # REMOVED
        self.property_embeddings = self.sbert_matcher.encode(
            self.property_texts,
            convert_to_tensor=True,
            show_progress_bar=False  # Changed to False
        )
    
    def analyze(self, query: str):
        """
        Analyze query to detect which relation it asks about.
        
        Strategy:
        1. Try SBERT zero-shot matching (PRIMARY - boosted confidence)
        2. Fall back to fine-tuned BERT classifier
        3. Fall back to TransE embedding matching
        4. Final fallback to keyword matching
        """
        from src.main.relation_analyzer import RelationQuery
        
        print(f"\n{'─'*80}")
        print(f"🔍 HYBRID RELATION ANALYSIS PIPELINE")
        print(f"{'─'*80}")
        print(f"Query: '{query[:60]}...'")
        print(f"\nAvailable methods:")
        print(f"  • SBERT zero-shot: {'✅ LOADED (PRIMARY)' if self.sbert_matcher else '❌ NOT AVAILABLE'}")
        print(f"  • DistilBERT classifier: {'✅ LOADED (FALLBACK)' if self.bert_classifier else '❌ NOT AVAILABLE'}")
        print(f"  • Keyword fallback: ✅ ALWAYS AVAILABLE")
        print(f"{'─'*80}\n")
        
        # Strategy 1: SBERT zero-shot (PRIMARY with boosted confidence)
        if self.sbert_matcher:
            print(f"[HybridAnalyzer] 🔢 STEP 1: Trying SBERT zero-shot matching (PRIMARY)...")
            result = self._match_with_sbert(query)
            if result and result.confidence > 0.35:  # ✅ LOWERED threshold from 0.5 to 0.35
                # ✅ BOOST: Increase confidence by 15%
                boosted_confidence = min(result.confidence * 1.15, 0.99)
                result.confidence = boosted_confidence
                
                print(f"[HybridAnalyzer] ✅ SUCCESS: SBERT zero-shot matcher")
                print(f"[HybridAnalyzer]    Relation: {result.relation}")
                print(f"[HybridAnalyzer]    Original Confidence: {result.confidence / 1.15:.2%}")
                print(f"[HybridAnalyzer]    Boosted Confidence: {result.confidence:.2%}")
                print(f"[HybridAnalyzer]    Method: SBERT SEMANTIC SIMILARITY (BOOSTED)")
                print(f"{'─'*80}\n")
                return result
            else:
                if result:
                    print(f"[HybridAnalyzer] ⚠️  SBERT confidence too low: {result.confidence:.2%} < 35%")
                else:
                    print(f"[HybridAnalyzer] ⚠️  SBERT matching failed")
                print(f"[HybridAnalyzer]    Falling back to next method...")
        else:
            print(f"[HybridAnalyzer] ⏭️  STEP 1: SBERT not available, skipping")
        
        # Strategy 2: BERT classifier (FALLBACK)
        if self.bert_classifier:
            print(f"\n[HybridAnalyzer] 🤖 STEP 2: Trying DistilBERT classifier (FALLBACK)...")
            result = self._classify_with_bert(query)
            if result and result.confidence > 0.7:
                print(f"[HybridAnalyzer] ✅ SUCCESS: DistilBERT classifier")
                print(f"[HybridAnalyzer]    Relation: {result.relation}")
                print(f"[HybridAnalyzer]    Confidence: {result.confidence:.2%}")
                print(f"[HybridAnalyzer]    Method: TRAINED DISTILBERT CLASSIFIER")
                print(f"{'─'*80}\n")
                return result
            else:
                if result:
                    print(f"[HybridAnalyzer] ⚠️  DistilBERT confidence too low: {result.confidence:.2%} < 70%")
                else:
                    print(f"[HybridAnalyzer] ⚠️  DistilBERT classification failed")
                print(f"[HybridAnalyzer]    Falling back to next method...")
        else:
            print(f"\n[HybridAnalyzer] ⏭️  STEP 2: DistilBERT not available, skipping")
        
        # Strategy 3: TransE embedding matching
        if self.embedding_matcher:
            print(f"\n[HybridAnalyzer] 🎯 STEP 3: Trying TransE embedding matching...")
            result = self._match_with_embeddings(query)
            if result and result.confidence > 0.4:
                print(f"[HybridAnalyzer] ✅ SUCCESS: TransE embedding matcher")
                print(f"[HybridAnalyzer]    Relation: {result.relation}")
                print(f"[HybridAnalyzer]    Confidence: {result.confidence:.2%}")
                print(f"[HybridAnalyzer]    Method: TRANSE EMBEDDING SIMILARITY")
                print(f"{'─'*80}\n")
                return result
            else:
                if result:
                    print(f"[HybridAnalyzer] ⚠️  Embedding confidence too low: {result.confidence:.2%} < 40%")
                else:
                    print(f"[HybridAnalyzer] ⚠️  Embedding matching failed")
                print(f"[HybridAnalyzer]    Falling back to final method...")
        else:
            print(f"\n[HybridAnalyzer] ⏭️  STEP 3: Embedding matcher not available, skipping")
        
        # Strategy 4: Keyword fallback
        print(f"\n[HybridAnalyzer] 🔤 STEP 4: Using keyword-based fallback...")
        result = self._keyword_fallback(query)
        if result:
            print(f"[HybridAnalyzer] ✅ SUCCESS: Keyword matcher")
            print(f"[HybridAnalyzer]    Relation: {result.relation}")
            print(f"[HybridAnalyzer]    Confidence: {result.confidence:.2%}")
            print(f"[HybridAnalyzer]    Method: KEYWORD MATCHING")
            print(f"{'─'*80}\n")
        else:
            print(f"[HybridAnalyzer] ❌ FAILED: No method could detect relation")
            print(f"{'─'*80}\n")
        
        return result
    
    def _load_bert_classifier(self, model_path: str):
        """
        Load fine-tuned DistilBERT classifier for relation classification.
        """
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        
        try:
            # ✅ FIX: Check if directory exists
            if not os.path.exists(model_path):
                print(f"⚠️  Model directory not found: {model_path}")
                return None
            
            # ✅ FIX: Check if required files exist
            config_path = os.path.join(model_path, "config.json")
            model_weights = os.path.join(model_path, "pytorch_model.bin")
            
            if not os.path.exists(config_path):
                print(f"⚠️  Model config not found: {config_path}")
                return None
            
            if not os.path.exists(model_weights):
                print(f"⚠️  Model weights not found: {model_weights}")
                return None
            
            print(f"🤖 Loading DistilBERT relation classifier from: {model_path}")
            
            # Load tokenizer
            tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            
            # Load model
            model = DistilBertForSequenceClassification.from_pretrained(model_path)
            model.eval()
            
            # Move to GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # Store for later use
            self.tokenizer = tokenizer
            self.device = device
            
            # Get label mappings (property ID → class)
            self.id2label = model.config.id2label
            self.label2id = model.config.label2id
            
            print(f"✅ Loaded DistilBERT classifier: {len(self.id2label)} classes")
            
            return model
            
        except Exception as e:
            print(f"❌ Failed to load DistilBERT classifier: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _classify_with_bert(self, query: str) -> Optional['RelationQuery']:
        """
        Classify query using fine-tuned DistilBERT.
        """
        if not self.bert_classifier or not self.tokenizer:
            return None
        
        from src.main.relation_analyzer import RelationQuery
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                query,
                return_tensors='pt',
                truncation=True,
                max_length=128,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.bert_classifier(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[0]
            
            # Get top prediction
            top_prob, top_idx = torch.max(probs, dim=0)
            property_id = self.id2label[top_idx.item()]
            confidence = top_prob.item()
            
            print(f"[BERT] Classified as: {property_id} (confidence: {confidence:.3f})")
            
            # Get property URI and metadata
            if property_id.startswith('P') and property_id[1:].isdigit():
                property_uri = f"http://www.wikidata.org/prop/direct/{property_id}"
                relation_name = property_id.lower()
            else:
                relation_name = property_id
                property_uri = None
                for uri, info in self.property_descriptions.items():
                    if info['label'] == property_id or uri.endswith(property_id):
                        property_uri = uri
                        break
                
                if property_uri is None:
                    name_to_pid = {
                        'director': 'P57',
                        'cast_member': 'P161',
                        'screenwriter': 'P58',
                        'producer': 'P162',
                        'genre': 'P136',
                        'country_of_origin': 'P495',
                        'original_language': 'P364',
                    }
                    pid = name_to_pid.get(relation_name)
                    if pid:
                        property_uri = f"http://www.wikidata.org/prop/direct/{pid}"
                    else:
                        print(f"[BERT] Warning: Could not map '{property_id}' to URI")
                        return None
            
            # Infer types
            if property_uri:
                subject_type, object_type = self._infer_types(property_uri)
            else:
                subject_type, object_type = 'unknown', 'unknown'
            
            return RelationQuery(
                relation=relation_name,
                relation_uri=property_uri,
                subject_type=subject_type,
                object_type=object_type,
                confidence=float(confidence),
                keywords=[f"bert:{property_id}"]  # ✅ Method indicator
            )
            
        except Exception as e:
            print(f"[BERT] Classification error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _match_with_sbert(self, query: str) -> Optional['RelationQuery']:
        """
        Match query to properties using SBERT zero-shot semantic similarity.
        ✅ ENHANCED: Better context-aware matching with query preprocessing
        """
        if not self.sbert_matcher:
            print(f"[SBERT] ⚠️  SBERT matcher not initialized")
            return None
        
        from src.main.relation_analyzer import RelationQuery
        
        try:
            # ✅ NEW: Preprocess query to emphasize key semantic features
            processed_query = self._preprocess_query_for_matching(query)
            print(f"[SBERT] Preprocessed query: '{processed_query}'")
            
            # Encode query
            query_embedding = self.sbert_matcher.encode(processed_query, convert_to_tensor=True)
            
            # Compute cosine similarity with all properties
            similarities = self.sbert_util.cos_sim(query_embedding, self.property_embeddings)[0]
            
            # Get top-10 matches for analysis
            top_results = torch.topk(similarities, k=min(10, len(similarities)))
            
            # ✅ NEW: Show top 5 candidates for debugging
            print(f"[SBERT] Top 5 candidates:")
            for i in range(min(5, len(top_results.indices))):
                idx = top_results.indices[i].item()
                score = top_results.values[i].item()
                prop_uri = self.property_uris[idx]
                prop_info = self.property_descriptions[prop_uri]
                print(f"[SBERT]   {i+1}. {prop_info['label']} (score: {score:.3f})")
            
            # ✅ NEW: Apply context-based filtering
            best_idx, best_score = self._select_best_match_with_context(
                query, 
                top_results.indices.tolist(), 
                top_results.values.tolist()
            )
            
            if best_score < 0.25:
                print(f"[SBERT] Best score {best_score:.3f} below threshold 0.25")
                return None
            
            # Get property info
            property_uri = self.property_uris[best_idx]
            property_info = self.property_descriptions[property_uri]
            
            print(f"[SBERT] Best match after context filtering: {property_info['label']} (score: {best_score:.3f})")
            
            # Infer subject/object types
            subject_type, object_type = self._infer_types(property_uri)
            
            # Extract friendly name
            relation_name = property_info['label'].replace(' ', '_')
            
            # ✅ BOOST: Increase raw score by 20% (but cap at 0.95)
            boosted_score = min(best_score * 1.20, 0.95)
            
            print(f"[SBERT] Confidence boost: {best_score:.3f} → {boosted_score:.3f}")
            
            return RelationQuery(
                relation=relation_name,
                relation_uri=property_uri,
                subject_type=subject_type,
                object_type=object_type,
                confidence=float(boosted_score),
                keywords=[f"sbert:{property_info['label']}"]
            )
            
        except Exception as e:
            print(f"[SBERT] Matching error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _preprocess_query_for_matching(self, query: str) -> str:
        """
        Preprocess query to emphasize key semantic features for SBERT matching.
        ✅ NEW: Extract and emphasize the core relation being asked about.
        """
        import re
        
        query_lower = query.lower()
        
        # ✅ Strategy 1: Detect "what/which X" patterns and emphasize X
        # Example: "From what country" → "country origin"
        what_which_patterns = [
            (r'from\s+what\s+(country|language|genre|date)', r'\1 origin'),
            (r'in\s+what\s+(country|language|genre)', r'\1'),
            (r'what\s+(?:is|was)\s+the\s+(\w+)', r'\1'),
            (r'which\s+(\w+)', r'\1'),
        ]
        
        processed = query_lower
        for pattern, replacement in what_which_patterns:
            processed = re.sub(pattern, replacement, processed)
        
        # ✅ Strategy 2: Remove movie titles (in quotes) to focus on relation
        processed = re.sub(r'["\']([^"\']+)["\']', '', processed)
        
        # ✅ Strategy 3: Remove question words but keep semantic keywords
        words_to_remove = ['from', 'what', 'is', 'the', 'of', 'movie', 'film']
        for word in words_to_remove:
            processed = re.sub(rf'\b{word}\b', '', processed)
        
        # Clean up whitespace
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed if processed else query
    
    def _select_best_match_with_context(
        self, 
        query: str, 
        candidate_indices: list, 
        candidate_scores: list
    ) -> tuple:
        """
        Select best property match using query context.
        ✅ NEW: Apply semantic rules to disambiguate similar properties.
        """
        query_lower = query.lower()
        
        # ✅ RULE 1: If query asks "from what country" → prefer country_of_origin (P495) over language (P364)
        if 'from what country' in query_lower or 'from which country' in query_lower:
            print(f"[SBERT] Context rule: 'from what country' → prefer P495 (country_of_origin)")
            for idx, score in zip(candidate_indices, candidate_scores):
                prop_uri = self.property_uris[idx]
                if 'P495' in prop_uri:  # country_of_origin
                    print(f"[SBERT] ✅ Found P495 in candidates, boosting")
                    return idx, score * 1.3  # Boost by 30%
        
        # ✅ RULE 2: If query asks about "language" explicitly → prefer P364 (original_language)
        if 'language' in query_lower and 'country' not in query_lower:
            print(f"[SBERT] Context rule: 'language' query → prefer P364 (original_language)")
            for idx, score in zip(candidate_indices, candidate_scores):
                prop_uri = self.property_uris[idx]
                if 'P364' in prop_uri:
                    print(f"[SBERT] ✅ Found P364 in candidates, boosting")
                    return idx, score * 1.3
        
        # ✅ RULE 3: If query mentions "born" or "citizenship" → prefer P27 (country_of_citizenship)
        if any(word in query_lower for word in ['born', 'citizenship', 'nationality']):
            print(f"[SBERT] Context rule: person nationality → prefer P27 (citizenship)")
            for idx, score in zip(candidate_indices, candidate_scores):
                prop_uri = self.property_uris[idx]
                if 'P27' in prop_uri:
                    print(f"[SBERT] ✅ Found P27 in candidates, boosting")
                    return idx, score * 1.3
        
        # Default: return best match
        return candidate_indices[0], candidate_scores[0]
    
    def _match_with_embeddings(self, query: str) -> Optional['RelationQuery']:
        """
        Match query to relations using TransE embedding similarity.
        """
        if not self.embedding_matcher:
            return None
        
        from src.main.relation_analyzer import RelationQuery
        
        try:
            # Extract entities first (to remove them from query)
            extracted_entities = self._extract_entities_for_removal(query)
            
            # Match relation
            relation_name, relation_uri, confidence = self.embedding_matcher.match_relation(
                query,
                extracted_entities=extracted_entities,
                top_k=5
            )
            
            if relation_name is None:
                return None
            
            # Infer types
            subject_type, object_type = self._infer_types(relation_uri)
            
            return RelationQuery(
                relation=relation_name,
                relation_uri=relation_uri,
                subject_type=subject_type,
                object_type=object_type,
                confidence=float(confidence),
                keywords=[f"embedding:{relation_name}"]  # ✅ Method indicator
            )
            
        except Exception as e:
            print(f"[Embedding] Matching error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_entities_for_removal(self, query: str) -> List[str]:
        """Extract entity strings from query for removal."""
        entities = []
        
        # Extract quoted text
        import re
        quoted = re.findall(r'["\']([^"\']+)["\']', query)
        entities.extend(quoted)
        
        # Extract capitalized spans
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b', query)
        stop_words = {'What', 'Who', 'When', 'Where', 'Which', 'How', 'Why'}
        entities.extend([c for c in caps if c not in stop_words])
        
        return entities
    
    def _infer_types(self, property_uri: str) -> Tuple[str, str]:
        """Infer subject and object types by sampling triples."""
        from rdflib import URIRef
        
        P31 = URIRef("http://www.wikidata.org/prop/direct/P31")
        prop_ref = URIRef(property_uri)
        
        subject_types = {}
        object_types = {}
        
        # Sample 20 triples
        for i, (s, p, o) in enumerate(self.sparql_handler.graph.triples((None, prop_ref, None))):
            if i >= 20:
                break
            
            # Get subject type
            for subj_type in self.sparql_handler.graph.objects(s, P31):
                type_str = str(subj_type)
                if '/Q' in type_str:
                    qcode = type_str.split('/Q')[-1].split('#')[0]
                    subject_types[qcode] = subject_types.get(qcode, 0) + 1
            
            # Get object type
            if isinstance(o, URIRef):
                for obj_type in self.sparql_handler.graph.objects(o, P31):
                    type_str = str(obj_type)
                    if '/Q' in type_str:
                        qcode = type_str.split('/Q')[-1].split('#')[0]
                        object_types[qcode] = object_types.get(qcode, 0) + 1
        
        # Pick most common
        subject_qcode = max(subject_types.items(), key=lambda x: x[1])[0] if subject_types else 'unknown'
        object_qcode = max(object_types.items(), key=lambda x: x[1])[0] if object_types else 'unknown'
        
        return subject_qcode, object_qcode
    
    def _keyword_fallback(self, query: str):
        """Final fallback: keyword matching on common properties."""
        from src.main.relation_analyzer import RelationAnalyzer
        
        # Use existing RelationAnalyzer as fallback
        fallback = RelationAnalyzer(self.sparql_handler)
        return fallback.analyze(query)
