"""
SPARQL Pattern Dataset Generator - ENHANCED VERSION
Creates synthetic training data for classifying query patterns into SPARQL types.
Now includes comprehensive coverage of ALL movie-related relations.
"""

import json
import random
from typing import List, Tuple, Dict
from pathlib import Path


class SPARQLPatternDatasetGenerator:
    """Generate synthetic dataset for SPARQL pattern classification."""
    
    def __init__(self):
        """Initialize with templates for each pattern-relation combination."""
        self._setup_templates()
        self._setup_entities()
    
    def _setup_templates(self):
        """Define comprehensive templates for ALL pattern-relation combinations."""
        
        # Format: (template, pattern_type, relation)
        self.templates = [
            # ==================== FORWARD - DIRECTOR ====================
            ("Who directed {movie}?", "forward", "director"),
            ("Who is the director of {movie}?", "forward", "director"),
            ("What director made {movie}?", "forward", "director"),
            ("Tell me the director of {movie}", "forward", "director"),
            ("Director of {movie}?", "forward", "director"),
            ("Who made {movie}?", "forward", "director"),
            ("Who helmed {movie}?", "forward", "director"),
            ("Who was behind {movie}?", "forward", "director"),
            ("{movie} was directed by whom?", "forward", "director"),
            ("Who is {movie}'s director?", "forward", "director"),
            
            # ==================== FORWARD - CAST ====================
            ("Who starred in {movie}?", "forward", "cast_member"),
            ("Who acted in {movie}?", "forward", "cast_member"),
            ("What actors are in {movie}?", "forward", "cast_member"),
            ("List the cast of {movie}", "forward", "cast_member"),
            ("Who plays in {movie}?", "forward", "cast_member"),
            ("Who appeared in {movie}?", "forward", "cast_member"),
            ("What is the cast of {movie}?", "forward", "cast_member"),
            ("Actors in {movie}?", "forward", "cast_member"),
            ("{movie} cast members?", "forward", "cast_member"),
            ("Who was in {movie}?", "forward", "cast_member"),
            
            # ==================== FORWARD - GENRE ====================
            ("What genre is {movie}?", "forward", "genre"),
            ("What type of movie is {movie}?", "forward", "genre"),
            ("What kind of film is {movie}?", "forward", "genre"),
            ("Genre of {movie}?", "forward", "genre"),
            ("What category is {movie}?", "forward", "genre"),
            ("{movie} genre?", "forward", "genre"),
            ("What style is {movie}?", "forward", "genre"),
            ("Is {movie} a drama or comedy?", "forward", "genre"),
            
            # ==================== FORWARD - RELEASE DATE ====================
            ("When was {movie} released?", "forward", "publication_date"),
            ("What year did {movie} come out?", "forward", "publication_date"),
            ("Release date of {movie}?", "forward", "publication_date"),
            ("When did {movie} premiere?", "forward", "publication_date"),
            ("{movie} release year?", "forward", "publication_date"),
            ("When did {movie} debut?", "forward", "publication_date"),
            ("What year was {movie} made?", "forward", "publication_date"),
            ("Publication date of {movie}?", "forward", "publication_date"),
            
            # ==================== FORWARD - SCREENWRITER ====================
            ("Who wrote {movie}?", "forward", "screenwriter"),
            ("Who is the screenwriter of {movie}?", "forward", "screenwriter"),
            ("Screenwriter of {movie}?", "forward", "screenwriter"),
            ("Who wrote the script for {movie}?", "forward", "screenwriter"),
            ("{movie} was written by whom?", "forward", "screenwriter"),
            ("Who penned {movie}?", "forward", "screenwriter"),
            ("Script writer of {movie}?", "forward", "screenwriter"),
            ("Who authored {movie}?", "forward", "screenwriter"),
            
            # ==================== FORWARD - PRODUCER ====================
            ("Who produced {movie}?", "forward", "producer"),
            ("Who is the producer of {movie}?", "forward", "producer"),
            ("Producer of {movie}?", "forward", "producer"),
            ("{movie} producers?", "forward", "producer"),
            ("Who financed {movie}?", "forward", "producer"),
            ("Who backed {movie}?", "forward", "producer"),
            
            # ==================== FORWARD - COUNTRY ====================
            ("What country is {movie} from?", "forward", "country"),
            ("Where is {movie} from?", "forward", "country"),
            ("Country of origin for {movie}?", "forward", "country"),
            ("{movie} country?", "forward", "country"),
            ("From what country is {movie}?", "forward", "country"),
            ("What is the country of origin of {movie}?", "forward", "country"),
            ("Which country produced {movie}?", "forward", "country"),
            # ✅ NEW: "From" starting questions
            ("From what country is the movie {movie}?", "forward", "country"),
            ("From which country does {movie} come?", "forward", "country"),
            ("From where is {movie}?", "forward", "country"),
            ("From what nation is {movie}?", "forward", "country"),
            
            # ==================== FORWARD - RATING ====================
            ("What is the rating of {movie}?", "forward", "rating"),
            ("Rating of {movie}?", "forward", "rating"),
            ("MPAA rating for {movie}?", "forward", "rating"),
            ("{movie} rating?", "forward", "rating"),
            ("What rating does {movie} have?", "forward", "rating"),
            ("Film rating of {movie}?", "forward", "rating"),
            
            # ==================== REVERSE - DIRECTOR ====================
            ("What films did {person} direct?", "reverse", "director"),
            ("What movies did {person} direct?", "reverse", "director"),
            ("List films directed by {person}", "reverse", "director"),
            ("Show me {person}'s directorial work", "reverse", "director"),
            ("{person} directed which movies?", "reverse", "director"),
            ("Movies by {person}?", "reverse", "director"),
            ("{person} filmography as director?", "reverse", "director"),
            ("What did {person} direct?", "reverse", "director"),
            ("Films helmed by {person}?", "reverse", "director"),
            ("{person}'s directed films?", "reverse", "director"),
            
            # ==================== REVERSE - CAST ====================
            ("What films did {person} star in?", "reverse", "cast_member"),
            ("What movies did {person} act in?", "reverse", "cast_member"),
            ("List films starring {person}", "reverse", "cast_member"),
            ("{person} starred in which movies?", "reverse", "cast_member"),
            ("Movies with {person}?", "reverse", "cast_member"),
            ("{person} filmography as actor?", "reverse", "cast_member"),
            ("What did {person} act in?", "reverse", "cast_member"),
            ("Films featuring {person}?", "reverse", "cast_member"),
            ("{person}'s movies?", "reverse", "cast_member"),
            ("Show me {person}'s acting work", "reverse", "cast_member"),
            
            # ==================== REVERSE - SCREENWRITER ====================
            ("What films did {person} write?", "reverse", "screenwriter"),
            ("What movies did {person} write?", "reverse", "screenwriter"),
            ("List films written by {person}", "reverse", "screenwriter"),
            ("{person} wrote which movies?", "reverse", "screenwriter"),
            ("Scripts by {person}?", "reverse", "screenwriter"),
            ("{person} filmography as writer?", "reverse", "screenwriter"),
            ("What did {person} pen?", "reverse", "screenwriter"),
            
            # ==================== REVERSE - PRODUCER ====================
            ("What films did {person} produce?", "reverse", "producer"),
            ("What movies did {person} produce?", "reverse", "producer"),
            ("List films produced by {person}", "reverse", "producer"),
            ("{person} produced which movies?", "reverse", "producer"),
            ("{person} filmography as producer?", "reverse", "producer"),
            
            # ==================== VERIFICATION - DIRECTOR ====================
            ("Did {person} direct {movie}?", "verification", "director"),
            ("Was {movie} directed by {person}?", "verification", "director"),
            ("Is {person} the director of {movie}?", "verification", "director"),
            ("Did {person} helm {movie}?", "verification", "director"),
            ("Was {person} behind {movie}?", "verification", "director"),
            ("{person} directed {movie}, right?", "verification", "director"),
            ("Is it true that {person} directed {movie}?", "verification", "director"),
            
            # ==================== VERIFICATION - CAST ====================
            ("Did {person} star in {movie}?", "verification", "cast_member"),
            ("Was {person} in {movie}?", "verification", "cast_member"),
            ("Is {person} in the cast of {movie}?", "verification", "cast_member"),
            ("Did {person} act in {movie}?", "verification", "cast_member"),
            ("Was {person} part of {movie}?", "verification", "cast_member"),
            ("{person} starred in {movie}, correct?", "verification", "cast_member"),
            ("Did {person} appear in {movie}?", "verification", "cast_member"),
            
            # ==================== VERIFICATION - SCREENWRITER ====================
            ("Did {person} write {movie}?", "verification", "screenwriter"),
            ("Was {movie} written by {person}?", "verification", "screenwriter"),
            ("Is {person} the writer of {movie}?", "verification", "screenwriter"),
            ("Did {person} pen {movie}?", "verification", "screenwriter"),
            ("{person} wrote {movie}, right?", "verification", "screenwriter"),
            
            # ==================== VERIFICATION - PRODUCER ====================
            ("Did {person} produce {movie}?", "verification", "producer"),
            ("Was {movie} produced by {person}?", "verification", "producer"),
            ("Is {person} the producer of {movie}?", "verification", "producer"),
            ("{person} produced {movie}, correct?", "verification", "producer"),
        ]
    
    def _setup_entities(self):
        """Define diverse entities for template filling."""
        # ✅ EXPANDED: More diverse movie titles (including non-English)
        self.movies = [
            # Classic English
            "The Godfather", "Star Wars", "Inception", "The Matrix",
            "Pulp Fiction", "The Dark Knight", "Forrest Gump",
            "The Shawshank Redemption", "Fight Club", "Interstellar",
            "The Lord of the Rings", "Titanic", "Avatar", "Gladiator",
            "The Silence of the Lambs", "Goodfellas", "Casablanca",
            "Schindler's List", "The Departed", "The Bridge on the River Kwai",
            "Psycho", "Citizen Kane", "Apocalypse Now", "Fargo",
            
            # ✅ International films
            "Parasite", "Amélie", "Crouching Tiger Hidden Dragon",
            "Cinema Paradiso", "Life Is Beautiful", "Pan's Labyrinth",
            "Spirited Away", "Oldboy", "The Raid", "City of God",
            "Shoplifters", "Roma", "Amores Perros",
            
            # ✅ Films with special characters/punctuation
            "12 Monkeys", "Se7en", "21 Grams", "127 Hours",
            "The King's Speech", "There Will Be Blood",
            "No Country for Old Men", "O Brother, Where Art Thou?",
            
            # ✅ Less common titles for diversity
            "Shortcut to Happiness", "French Kiss", "Bandit Queen",
            "Miracles Still Happen", "Good Will Hunting",
            "Aro Tolbukhin. En la mente del asesino"
        ]
        
        # ✅ EXPANDED: More diverse person names
        self.people = [
            # Famous directors
            "Steven Spielberg", "Martin Scorsese", "Christopher Nolan",
            "Quentin Tarantino", "Francis Ford Coppola", "James Cameron",
            "Alfred Hitchcock", "Stanley Kubrick", "Ridley Scott",
            "David Fincher", "Wes Anderson", "Paul Thomas Anderson",
            "Kathryn Bigelow", "Sofia Coppola", "Greta Gerwig",
            "Denis Villeneuve", "Guillermo del Toro", "Park Chan-wook",
            "Akira Kurosawa", "Wong Kar-wai", "Bong Joon-ho",
            
            # Famous actors
            "Tom Hanks", "Leonardo DiCaprio", "Robert De Niro",
            "Meryl Streep", "Al Pacino", "Brad Pitt",
            "Cate Blanchett", "Daniel Day-Lewis", "Joaquin Phoenix",
            "Frances McDormand", "Denzel Washington", "Tilda Swinton",
            
            # Writers/Producers
            "Aaron Sorkin", "Charlie Kaufman", "Nora Ephron",
            "Pete Dexter", "William Goldman", "Paddy Chayefsky",
            
            # ✅ International names
            "Pedro Almodóvar", "Alejandro González Iñárritu",
            "Hayao Miyazaki", "Jean-Pierre Jeunet"
        ]
        
        # ✅ NEW: Add countries for country queries
        self.countries = [
            "United States", "United Kingdom", "France", "Germany",
            "Italy", "Spain", "Japan", "South Korea", "China",
            "India", "Mexico", "Brazil", "Canada", "Australia",
            "Indonesia", "Thailand", "Argentina"
        ]
        
        # ✅ NEW: Add genres for genre queries
        self.genres = [
            "drama", "comedy", "action", "thriller", "horror",
            "science fiction", "romance", "documentary", "animation",
            "crime", "adventure", "mystery", "fantasy", "war",
            "biographical", "historical", "musical", "western"
        ]
    
    def generate_dataset(
        self,
        samples_per_template: int = 15,
        include_variations: bool = True,
        unknown_ratio: float = 0.12,
        add_misspellings: bool = True
    ) -> List[Tuple[str, str]]:
        """
        Generate comprehensive synthetic dataset with composite labels.
        
        Args:
            samples_per_template: Number of samples per template
            include_variations: Add case/punctuation variations
            unknown_ratio: Proportion of unknown/ambiguous queries
            add_misspellings: Add common misspelling patterns
            
        Returns:
            List of (query, label) tuples where label is "pattern_relation"
        """
        dataset = []
        
        print(f"Generating COMPREHENSIVE SPARQL pattern dataset...")
        print(f"  Templates: {len(self.templates)}")
        print(f"  Samples per template: {samples_per_template}")
        print(f"  Movies: {len(self.movies)}")
        print(f"  People: {len(self.people)}")
        
        # Generate from templates
        for template, pattern_type, relation in self.templates:
            for _ in range(samples_per_template):
                query = self._fill_template(template)
                label = f"{pattern_type}_{relation}"
                
                dataset.append((query, label))
                
                if include_variations:
                    # ✅ Case variations
                    dataset.append((query.lower(), label))
                    
                    # ✅ Punctuation variations (30% chance)
                    if random.random() > 0.7:
                        query_no_punct = query.rstrip('?!.')
                        dataset.append((query_no_punct, label))
                    
                    # ✅ Extra spaces (20% chance)
                    if random.random() > 0.8:
                        query_spaces = '  '.join(query.split())
                        dataset.append((query_spaces, label))
        
        # ✅ Add misspelling variations
        if add_misspellings:
            num_misspellings = len(dataset) // 20  # 5% of dataset
            print(f"  Adding {num_misspellings} misspelling variations...")
            
            for _ in range(num_misspellings):
                original_query, label = random.choice(dataset)
                misspelled = self._add_misspelling(original_query)
                if misspelled != original_query:
                    dataset.append((misspelled, label))
        
        # ✅ Add unknown/ambiguous queries
        num_unknown = int(len(dataset) * unknown_ratio)
        print(f"  Adding {num_unknown} unknown/ambiguous queries...")
        
        unknown_templates = [
            "Tell me about {movie}",
            "What about {movie}?",
            "Info on {movie}",
            "{movie} details",
            "Is {movie} good?",
            "Should I watch {movie}?",
            "Anything about {person}?",
            "{person} filmography",
            "I want to know about {movie}",
            "Facts about {movie}",
            "{movie} information",
            "Review of {movie}",
            "{movie} synopsis",
            "Plot of {movie}",
            "Summary of {movie}",
            "Trailer for {movie}",
            "Watch {movie}",
            "Stream {movie}",
            "{person} biography",
            "{person} age",
            "{person} net worth",
            "Best {genre} movies",
            "Top films",
            "Recommend a movie",
        ]
        
        for _ in range(num_unknown):
            template = random.choice(unknown_templates)
            query = self._fill_template(template)
            dataset.append((query, "unknown"))
        
        # ✅ Shuffle
        random.shuffle(dataset)
        
        # ✅ Print comprehensive statistics
        label_counts = {}
        for _, label in dataset:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\n✅ Generated {len(dataset)} total samples")
        print(f"   Unique labels: {len(label_counts)}")
        
        # Group by pattern type
        pattern_stats = {}
        for label, count in label_counts.items():
            if label == "unknown":
                pattern_type = "unknown"
            else:
                pattern_type = label.split('_')[0]
            
            if pattern_type not in pattern_stats:
                pattern_stats[pattern_type] = 0
            pattern_stats[pattern_type] += count
        
        print(f"\n   Distribution by pattern type:")
        for pattern, count in sorted(pattern_stats.items()):
            pct = count / len(dataset) * 100
            print(f"     - {pattern:15s}: {count:5d} ({pct:5.1f}%)")
        
        print(f"\n   Top 15 relation labels:")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            pct = count / len(dataset) * 100
            print(f"     - {label:30s}: {count:5d} ({pct:4.1f}%)")
        
        return dataset
    
    def _fill_template(self, template: str) -> str:
        """Fill template with random entities."""
        query = template
        
        if "{movie}" in query:
            query = query.replace("{movie}", random.choice(self.movies))
        
        if "{person}" in query:
            query = query.replace("{person}", random.choice(self.people))
        
        if "{country}" in query:
            query = query.replace("{country}", random.choice(self.countries))
        
        if "{genre}" in query:
            query = query.replace("{genre}", random.choice(self.genres))
        
        return query
    
    def _add_misspelling(self, query: str) -> str:
        """Add common misspelling patterns to query."""
        # Common misspelling transformations
        transformations = [
            # Double letters
            ('directed', 'directted'),
            ('starring', 'staring'),
            ('written', 'writen'),
            ('released', 'releaced'),
            
            # Missing letters
            ('director', 'directer'),
            ('actor', 'acter'),
            ('movie', 'moive'),
            ('film', 'flim'),
            
            # Transpositions
            ('the', 'teh'),
            ('from', 'form'),
            ('who', 'hwo'),
        ]
        
        # Apply random transformation (30% chance)
        if random.random() > 0.7:
            old, new = random.choice(transformations)
            query = query.replace(old, new)
        
        return query
    
    def save_dataset(
        self,
        dataset: List[Tuple[str, str]],
        output_path: str,
        train_split: float = 0.85,
        val_split: float = 0.10
    ):
        """
        Save dataset in HuggingFace format with train/val/test splits.
        
        Args:
            dataset: List of (query, label) tuples
            output_path: Output JSON file path
            train_split: Proportion for training (default 85%)
            val_split: Proportion for validation (default 10%, test gets remainder)
        """
        # Shuffle for good split distribution
        random.shuffle(dataset)
        
        # Calculate split indices
        train_idx = int(len(dataset) * train_split)
        val_idx = train_idx + int(len(dataset) * val_split)
        
        train_data = dataset[:train_idx]
        val_data = dataset[train_idx:val_idx]
        test_data = dataset[val_idx:]
        
        # Count labels
        all_labels = list(set(label for _, label in dataset))
        label2id = {label: i for i, label in enumerate(sorted(all_labels))}
        id2label = {i: label for label, i in label2id.items()}
        
        label_counts = {}
        for _, label in dataset:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Format with metadata
        formatted_dataset = {
            'train': [
                {'text': query, 'label': label}
                for query, label in train_data
            ],
            'validation': [
                {'text': query, 'label': label}
                for query, label in val_data
            ],
            'test': [
                {'text': query, 'label': label}
                for query, label in test_data
            ],
            'metadata': {
                'num_classes': len(all_labels),
                'classes': sorted(all_labels),
                'label2id': label2id,
                'id2label': id2label,
                'train_size': len(train_data),
                'val_size': len(val_data),
                'test_size': len(test_data),
                'total_size': len(dataset),
                'label_distribution': label_counts,
                'description': 'SPARQL pattern classification dataset for movie queries',
                'version': '2.0',
                'relations': [
                    'director', 'cast_member', 'screenwriter', 'producer',
                    'genre', 'publication_date', 'rating', 'country'
                ],
                'pattern_types': ['forward', 'reverse', 'verification', 'unknown']
            }
        }
        
        # Save
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Dataset saved to {output_path}")
        print(f"   Train: {len(train_data)} ({train_split*100:.0f}%)")
        print(f"   Val:   {len(val_data)} ({val_split*100:.0f}%)")
        print(f"   Test:  {len(test_data)} ({(1-train_split-val_split)*100:.0f}%)")
        print(f"   Total: {len(dataset)}")
        print(f"   Classes: {len(all_labels)}")


def main():
    """Generate comprehensive SPARQL pattern dataset."""
    generator = SPARQLPatternDatasetGenerator()
    
    # ✅ Generate larger, more diverse dataset
    dataset = generator.generate_dataset(
        samples_per_template=20,  # Increased from 15
        include_variations=True,
        unknown_ratio=0.12,
        add_misspellings=True
    )
    
    generator.save_dataset(
        dataset,
        "sparql_pattern_dataset_v2.json",
        train_split=0.85,
        val_split=0.10
    )
    
    print("\n" + "="*80)
    print("✅ Comprehensive dataset generation complete!")
    print("="*80)
    print("\n📋 Dataset includes:")
    print("   • 8 relations: director, cast_member, screenwriter, producer,")
    print("                  genre, publication_date, rating, country")
    print("   • 3 pattern types: forward, reverse, verification")
    print("   • 1 unknown class for out-of-scope queries")
    print("   • Diverse movie titles (60+ including international films)")
    print("   • Diverse person names (40+ including international names)")
    print("   • Case variations, punctuation variations, misspellings")
    print("\n📝 Next steps:")
    print("   1. Review the dataset: cat sparql_pattern_dataset_v2.json | jq '.metadata'")
    print("   2. Train the model:")
    print("      python -m src.main.classifier_fine_tuning.train_sparql_classifier")
    print("   3. Evaluate on test set")
    print("="*80)


if __name__ == "__main__":
    main()
