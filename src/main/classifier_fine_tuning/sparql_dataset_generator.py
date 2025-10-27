"""
SPARQL Pattern Dataset Generator.
Creates synthetic training data for classifying query patterns into SPARQL types.
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
        """Define templates for each pattern-relation combination."""
        
        # Format: (template, pattern_type, relation)
        self.templates = [
            # FORWARD - DIRECTOR
            ("Who directed {movie}?", "forward", "director"),
            ("Who is the director of {movie}?", "forward", "director"),
            ("What director made {movie}?", "forward", "director"),
            ("Tell me the director of {movie}", "forward", "director"),
            ("Director of {movie}?", "forward", "director"),
            
            # FORWARD - CAST
            ("Who starred in {movie}?", "forward", "cast_member"),
            ("Who acted in {movie}?", "forward", "cast_member"),
            ("What actors are in {movie}?", "forward", "cast_member"),
            ("List the cast of {movie}", "forward", "cast_member"),
            ("Who plays in {movie}?", "forward", "cast_member"),
            
            # FORWARD - GENRE
            ("What genre is {movie}?", "forward", "genre"),
            ("What type of movie is {movie}?", "forward", "genre"),
            ("What kind of film is {movie}?", "forward", "genre"),
            ("Genre of {movie}?", "forward", "genre"),
            
            # FORWARD - RELEASE DATE
            ("When was {movie} released?", "forward", "publication_date"),
            ("What year did {movie} come out?", "forward", "publication_date"),
            ("Release date of {movie}?", "forward", "publication_date"),
            ("When did {movie} premiere?", "forward", "publication_date"),
            
            # FORWARD - SCREENWRITER
            ("Who wrote {movie}?", "forward", "screenwriter"),
            ("Who is the screenwriter of {movie}?", "forward", "screenwriter"),
            ("Screenwriter of {movie}?", "forward", "screenwriter"),
            
            # FORWARD - PRODUCER
            ("Who produced {movie}?", "forward", "producer"),
            ("Who is the producer of {movie}?", "forward", "producer"),
            ("Producer of {movie}?", "forward", "producer"),
            
            # FORWARD - RATING
            ("What is the rating of {movie}?", "forward", "rating"),
            ("Rating of {movie}?", "forward", "rating"),
            ("MPAA rating for {movie}?", "forward", "rating"),
            
            # REVERSE - DIRECTOR
            ("What films did {person} direct?", "reverse", "director"),
            ("What movies did {person} direct?", "reverse", "director"),
            ("List films directed by {person}", "reverse", "director"),
            ("Show me {person}'s directorial work", "reverse", "director"),
            ("{person} directed which movies?", "reverse", "director"),
            
            # REVERSE - CAST
            ("What films did {person} star in?", "reverse", "cast_member"),
            ("What movies did {person} act in?", "reverse", "cast_member"),
            ("List films starring {person}", "reverse", "cast_member"),
            ("{person} starred in which movies?", "reverse", "cast_member"),
            
            # REVERSE - SCREENWRITER
            ("What films did {person} write?", "reverse", "screenwriter"),
            ("What movies did {person} write?", "reverse", "screenwriter"),
            ("List films written by {person}", "reverse", "screenwriter"),
            
            # REVERSE - PRODUCER
            ("What films did {person} produce?", "reverse", "producer"),
            ("What movies did {person} produce?", "reverse", "producer"),
            ("List films produced by {person}", "reverse", "producer"),
            
            # VERIFICATION - DIRECTOR
            ("Did {person} direct {movie}?", "verification", "director"),
            ("Was {movie} directed by {person}?", "verification", "director"),
            ("Is {person} the director of {movie}?", "verification", "director"),
            
            # VERIFICATION - CAST
            ("Did {person} star in {movie}?", "verification", "cast_member"),
            ("Was {person} in {movie}?", "verification", "cast_member"),
            ("Is {person} in the cast of {movie}?", "verification", "cast_member"),
            
            # VERIFICATION - SCREENWRITER
            ("Did {person} write {movie}?", "verification", "screenwriter"),
            ("Was {movie} written by {person}?", "verification", "screenwriter"),
            
            # VERIFICATION - PRODUCER
            ("Did {person} produce {movie}?", "verification", "producer"),
            ("Was {movie} produced by {person}?", "verification", "producer"),
        ]
    
    def _setup_entities(self):
        """Define entities for template filling."""
        self.movies = [
            "The Godfather", "Star Wars", "Inception", "The Matrix",
            "Pulp Fiction", "The Dark Knight", "Forrest Gump",
            "The Shawshank Redemption", "Fight Club", "Interstellar",
            "The Lord of the Rings", "Titanic", "Avatar", "Gladiator",
            "The Silence of the Lambs", "Goodfellas", "Casablanca",
            "Schindler's List", "The Departed", "Parasite",
            "The Bridge on the River Kwai", "Psycho", "Citizen Kane"
        ]
        
        self.people = [
            "Steven Spielberg", "Martin Scorsese", "Christopher Nolan",
            "Quentin Tarantino", "Francis Ford Coppola", "James Cameron",
            "Tom Hanks", "Leonardo DiCaprio", "Robert De Niro",
            "Meryl Streep", "Al Pacino", "Brad Pitt",
            "Alfred Hitchcock", "Stanley Kubrick", "Ridley Scott"
        ]
    
    def generate_dataset(
        self,
        samples_per_template: int = 10,
        include_variations: bool = True,
        unknown_ratio: float = 0.15
    ) -> List[Tuple[str, str]]:
        """
        Generate synthetic dataset with composite labels.
        
        Args:
            samples_per_template: Number of samples per template
            include_variations: Add case variations
            unknown_ratio: Proportion of unknown/ambiguous queries
            
        Returns:
            List of (query, label) tuples where label is "pattern_relation"
        """
        dataset = []
        
        print(f"Generating SPARQL pattern dataset...")
        print(f"  Templates: {len(self.templates)}")
        print(f"  Samples per template: {samples_per_template}")
        
        # Generate from templates
        for template, pattern_type, relation in self.templates:
            for _ in range(samples_per_template):
                query = self._fill_template(template)
                label = f"{pattern_type}_{relation}"
                
                dataset.append((query, label))
                
                if include_variations:
                    dataset.append((query.lower(), label))
                    if random.random() > 0.7:
                        dataset.append((query.upper(), label))
        
        # Add unknown/ambiguous queries
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
        ]
        
        for _ in range(num_unknown):
            template = random.choice(unknown_templates)
            query = self._fill_template(template)
            dataset.append((query, "unknown"))
        
        # Shuffle
        random.shuffle(dataset)
        
        # Print statistics
        label_counts = {}
        for _, label in dataset:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\n✅ Generated {len(dataset)} samples")
        print(f"   Unique labels: {len(label_counts)}")
        print(f"\n   Top 10 labels by count:")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"     - {label}: {count}")
        
        return dataset
    
    def _fill_template(self, template: str) -> str:
        """Fill template with random entities."""
        query = template
        
        if "{movie}" in query:
            query = query.replace("{movie}", random.choice(self.movies))
        
        if "{person}" in query:
            query = query.replace("{person}", random.choice(self.people))
        
        return query
    
    def save_dataset(
        self,
        dataset: List[Tuple[str, str]],
        output_path: str,
        train_split: float = 0.8
    ):
        """Save dataset in HuggingFace format."""
        # Split
        split_idx = int(len(dataset) * train_split)
        train_data = dataset[:split_idx]
        val_data = dataset[split_idx:]
        
        # Count labels
        all_labels = list(set(label for _, label in dataset))
        label2id = {label: i for i, label in enumerate(sorted(all_labels))}
        id2label = {i: label for label, i in label2id.items()}
        
        label_counts = {}
        for _, label in dataset:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Format
        formatted_dataset = {
            'train': [
                {'text': query, 'label': label}
                for query, label in train_data
            ],
            'validation': [
                {'text': query, 'label': label}
                for query, label in val_data
            ],
            'metadata': {
                'num_classes': len(all_labels),
                'classes': sorted(all_labels),
                'label2id': label2id,
                'id2label': id2label,
                'train_size': len(train_data),
                'val_size': len(val_data),
                'label_distribution': label_counts
            }
        }
        
        # Save
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Dataset saved to {output_path}")
        print(f"   Train: {len(train_data)} | Val: {len(val_data)}")
        print(f"   Classes: {len(all_labels)}")


def main():
    """Generate SPARQL pattern dataset."""
    generator = SPARQLPatternDatasetGenerator()
    
    dataset = generator.generate_dataset(
        samples_per_template=15,
        include_variations=True,
        unknown_ratio=0.10
    )
    
    generator.save_dataset(
        dataset,
        "sparql_pattern_dataset.json",
        train_split=0.8
    )
    
    print("\n✅ Dataset generation complete!")
    print("\nNext: Train the model with:")
    print("  python -m src.main.classifier_fine_tuning.train_sparql_classifier")


if __name__ == "__main__":
    main()
