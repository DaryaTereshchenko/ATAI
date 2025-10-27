"""
Synthetic Dataset Generator for Query Classification.
Generates training data using rule-based classifier and templates.
NOW INCLUDES: Negative examples (out-of-scope queries) for rejection training.
"""

import json
import random
from typing import List, Tuple, Dict
from pathlib import Path

import sys
import os
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from query_classifier import RobustQueryClassifier, QuestionType


class SyntheticQueryGenerator:
    """Generates synthetic movie queries for training a classifier."""
    
    def __init__(self):
        """Initialize with query templates."""
        self.rule_based_classifier = RobustQueryClassifier()
        self._setup_templates()
        self._setup_entities()
        self._setup_negative_templates()  # ✅ NEW: Negative examples
    
    def _setup_templates(self):
        """Define query templates for each question type."""
        
        # FACTUAL query templates
        self.factual_templates = [
            # Director questions
            "Who directed {movie}?",
            "Who is the director of {movie}?",
            "What director made {movie}?",
            "Tell me who directed {movie}",
            "Find the director of {movie}",
            
            # Actor questions
            "Who starred in {movie}?",
            "Who acted in {movie}?",
            "Who plays in {movie}?",
            "What actors are in {movie}?",
            "List the cast of {movie}",
            
            # Genre questions
            "What genre is {movie}?",
            "What type of movie is {movie}?",
            "What kind of film is {movie}?",
            "Tell me the genre of {movie}",
            
            # Release date questions
            "When was {movie} released?",
            "What year did {movie} come out?",
            "What is the release date of {movie}?",
            "When did {movie} premiere?",
            
            # Search/find questions
            "Find movies directed by {person}",
            "List films starring {person}",
            "What movies did {person} direct?",
            "Show me films with {person}",
            "Search for movies about {topic}",
            
            # Comparison questions
            "What movies are similar to {movie}?",
            "Find films like {movie}",
            "What are movies comparable to {movie}?",
        ]
        
        # MULTIMEDIA query templates
        self.multimedia_templates = [
            "Show me a picture of {movie}",
            "Display the poster for {movie}",
            "Can I see an image of {movie}?",
            "What does {movie} look like?",
            "Show me the poster of {movie}",
            "Display a photo of {movie}",
            "I want to see the image of {movie}",
        ]
        
        # RECOMMENDATION query templates
        self.recommendation_templates = [
            "Recommend me a {genre} movie",
            "Suggest a good film to watch",
            "What should I watch tonight?",
            "Give me a movie recommendation",
            "What are some good {genre} movies?",
            "Help me find a movie to watch",
            "What's a good movie like {movie}?",
            "Suggest movies similar to {movie}",
        ]
    
    def _setup_negative_templates(self):
        """
        ✅ NEW: Define templates for OUT-OF-SCOPE queries.
        These should be rejected by the system.
        """
        
        # Category: MATHEMATICS
        self.negative_math_templates = [
            "What is 2 + 2?",
            "Calculate 15 * 23",
            "Solve for x: 2x + 5 = 15",
            "What's the derivative of x squared?",
            "How do you calculate the area of a circle?",
            "What is the square root of 144?",
        ]
        
        # Category: PROGRAMMING
        self.negative_programming_templates = [
            "How do I write a Python function?",
            "What is a for loop in Java?",
            "Debug my code please",
            "How to implement binary search?",
            "What does this error mean: NullPointerException?",
            "Write a function to sort an array",
        ]
        
        # Category: GENERAL KNOWLEDGE
        self.negative_general_templates = [
            "What is the capital of France?",
            "Who is the president of the United States?",
            "How tall is Mount Everest?",
            "What is the speed of light?",
            "When did World War II end?",
            "What is the population of China?",
        ]
        
        # Category: WEATHER
        self.negative_weather_templates = [
            "What's the weather today?",
            "Is it going to rain tomorrow?",
            "What's the temperature outside?",
            "Give me the weather forecast",
            "How hot will it be this weekend?",
        ]
        
        # Category: SPORTS
        self.negative_sports_templates = [
            "Who won the Super Bowl?",
            "What's the NBA score?",
            "When is the next Olympics?",
            "Who is the best soccer player?",
            "What time is the football match?",
        ]
        
        # Category: HEALTH
        self.negative_health_templates = [
            "What are the symptoms of flu?",
            "How do I treat a headache?",
            "Should I see a doctor?",
            "What medication should I take?",
            "How to lose weight fast?",
        ]
        
        # Category: COOKING
        self.negative_cooking_templates = [
            "How do I make pasta?",
            "Give me a recipe for chocolate cake",
            "How long to bake chicken?",
            "What ingredients do I need for pizza?",
            "How to cook rice?",
        ]
        
        # Category: PERSONAL/CONVERSATIONAL
        self.negative_personal_templates = [
            "How are you?",
            "What's your name?",
            "Do you like movies?",
            "How old are you?",
            "What's your favorite color?",
            "Tell me about yourself",
        ]
        
        # Category: NEWS/CURRENT EVENTS
        self.negative_news_templates = [
            "What happened today?",
            "Give me the latest news",
            "What's happening in the world?",
            "Tell me about recent events",
            "What's trending on social media?",
        ]
        
        # Category: TRAVEL
        self.negative_travel_templates = [
            "Where should I go on vacation?",
            "How much does a flight to Paris cost?",
            "What are the best hotels in London?",
            "How do I get to the airport?",
            "What documents do I need to travel?",
        ]
        
        # Combine all negative templates
        self.all_negative_templates = (
            self.negative_math_templates +
            self.negative_programming_templates +
            self.negative_general_templates +
            self.negative_weather_templates +
            self.negative_sports_templates +
            self.negative_health_templates +
            self.negative_cooking_templates +
            self.negative_personal_templates +
            self.negative_news_templates +
            self.negative_travel_templates
        )
    
    def _setup_entities(self):
        """Define entity lists for template filling."""
        
        # Sample movies (expand this list)
        self.movies = [
            "The Godfather", "Star Wars", "Inception", "The Matrix",
            "Pulp Fiction", "The Dark Knight", "Forrest Gump",
            "The Shawshank Redemption", "Fight Club", "Interstellar",
            "The Lord of the Rings", "Titanic", "Avatar", "Gladiator",
            "The Silence of the Lambs", "Goodfellas", "Casablanca",
            "Schindler's List", "The Departed", "Parasite"
        ]
        
        # Sample people (expand this list)
        self.people = [
            "Steven Spielberg", "Martin Scorsese", "Christopher Nolan",
            "Quentin Tarantino", "Francis Ford Coppola", "James Cameron",
            "Tom Hanks", "Leonardo DiCaprio", "Robert De Niro",
            "Meryl Streep", "Al Pacino", "Brad Pitt"
        ]
        
        # Genres
        self.genres = [
            "action", "drama", "comedy", "thriller", "sci-fi",
            "horror", "romance", "documentary", "adventure",
            "mystery", "fantasy", "crime"
        ]
        
        # Topics
        self.topics = [
            "war", "crime", "love", "space", "time travel",
            "artificial intelligence", "superheroes", "history",
            "music", "sports", "family", "revenge"
        ]
    
    def generate_queries(
        self, 
        num_per_class: int = 100,
        include_variations: bool = True,
        negative_ratio: float = 0.25  # ✅ NEW: 25% negative examples
    ) -> List[Tuple[str, str]]:
        """
        Generate synthetic queries with labels INCLUDING negative examples.
        
        Args:
            num_per_class: Number of queries to generate per positive class
            include_variations: Add lowercase/uppercase variations
            negative_ratio: Proportion of negative examples (0.25 = 25%)
            
        Returns:
            List of (query, label) tuples
        """
        queries = []
        
        # Generate FACTUAL queries
        print(f"Generating {num_per_class} factual queries...")
        for _ in range(num_per_class):
            template = random.choice(self.factual_templates)
            query = self._fill_template(template)
            queries.append((query, "factual"))
            
            if include_variations:
                queries.append((query.lower(), "factual"))
                if random.random() > 0.7:
                    queries.append((query.upper(), "factual"))
        
        # Generate MULTIMEDIA queries
        print(f"Generating {num_per_class} multimedia queries...")
        for _ in range(num_per_class):
            template = random.choice(self.multimedia_templates)
            query = self._fill_template(template)
            queries.append((query, "multimedia"))
            
            if include_variations:
                queries.append((query.lower(), "multimedia"))
        
        # Generate RECOMMENDATION queries
        print(f"Generating {num_per_class} recommendation queries...")
        for _ in range(num_per_class):
            template = random.choice(self.recommendation_templates)
            query = self._fill_template(template)
            queries.append((query, "recommendation"))
            
            if include_variations:
                queries.append((query.lower(), "recommendation"))
        
        # ✅ NEW: Generate NEGATIVE (out-of-scope) queries
        num_positive = len(queries)
        num_negative = int(num_positive * negative_ratio)
        print(f"Generating {num_negative} NEGATIVE (out-of-scope) queries...")
        
        for _ in range(num_negative):
            query = random.choice(self.all_negative_templates)
            queries.append((query, "out_of_scope"))  # ✅ NEW label
            
            if include_variations:
                queries.append((query.lower(), "out_of_scope"))
        
        # Shuffle
        random.shuffle(queries)
        
        print(f"\n✅ Generated {len(queries)} total queries:")
        print(f"   - Positive (movie-related): {num_positive}")
        print(f"   - Negative (out-of-scope): {len(queries) - num_positive}")
        
        return queries
    
    def _fill_template(self, template: str) -> str:
        """Fill template placeholders with entities."""
        query = template
        
        if "{movie}" in query:
            query = query.replace("{movie}", random.choice(self.movies))
        
        if "{person}" in query:
            query = query.replace("{person}", random.choice(self.people))
        
        if "{genre}" in query:
            query = query.replace("{genre}", random.choice(self.genres))
        
        if "{topic}" in query:
            query = query.replace("{topic}", random.choice(self.topics))
        
        return query
    
    def validate_with_rules(
        self, 
        queries: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
        """
        Validate synthetic queries using rule-based classifier.
        For negative examples, check if they're correctly rejected.
        
        Args:
            queries: List of (query, label) tuples
            
        Returns:
            Filtered queries and statistics
        """
        from query_validator import QueryValidator
        
        validated = []
        stats = {
            'total': len(queries),
            'agreed': 0,
            'disagreed': 0,
            'disagreements_by_type': {}
        }
        
        validator = QueryValidator()
        
        for query, expected_label in queries:
            if expected_label == "out_of_scope":
                # ✅ For negative examples, check if validator rejects them
                validation_result = validator.validate(query)
                
                if not validation_result.is_valid:
                    # Correctly rejected!
                    validated.append((query, expected_label))
                    stats['agreed'] += 1
                else:
                    # Incorrectly accepted (disagreement)
                    stats['disagreed'] += 1
                    key = f"negative->accepted"
                    stats['disagreements_by_type'][key] = \
                        stats['disagreements_by_type'].get(key, 0) + 1
            else:
                # For positive examples, use rule-based classifier
                result = self.rule_based_classifier.classify(query)
                predicted_label = result.question_type.value
                
                if predicted_label == expected_label:
                    validated.append((query, expected_label))
                    stats['agreed'] += 1
                else:
                    stats['disagreed'] += 1
                    key = f"{expected_label}->{predicted_label}"
                    stats['disagreements_by_type'][key] = \
                        stats['disagreements_by_type'].get(key, 0) + 1
        
        return validated, stats
    
    def save_dataset(
        self, 
        queries: List[Tuple[str, str]], 
        output_path: str,
        train_split: float = 0.8
    ):
        """
        Save dataset in JSON format for training.
        
        Args:
            queries: List of (query, label) tuples
            output_path: Path to save dataset
            train_split: Proportion for training set
        """
        # Split into train/val
        split_idx = int(len(queries) * train_split)
        train_data = queries[:split_idx]
        val_data = queries[split_idx:]
        
        # Count labels
        label_counts = {}
        for _, label in queries:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Format for training
        dataset = {
            'train': [
                {'text': query, 'label': label} 
                for query, label in train_data
            ],
            'validation': [
                {'text': query, 'label': label} 
                for query, label in val_data
            ],
            'metadata': {
                'num_classes': 4,  # ✅ NOW 4: factual, multimedia, recommendation, out_of_scope
                'classes': ['factual', 'multimedia', 'recommendation', 'out_of_scope'],
                'train_size': len(train_data),
                'val_size': len(val_data),
                'label_distribution': label_counts
            }
        }
        
        # Save
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Dataset saved to {output_path}")
        print(f"   Train samples: {len(train_data)}")
        print(f"   Validation samples: {len(val_data)}")
        print(f"\n   Label distribution:")
        for label, count in label_counts.items():
            print(f"     - {label}: {count} ({count/len(queries)*100:.1f}%)")


def main():
    """Generate synthetic dataset with negative examples."""
    generator = SyntheticQueryGenerator()
    
    print("="*80)
    print("GENERATING SYNTHETIC DATASET WITH NEGATIVE EXAMPLES")
    print("="*80 + "\n")
    
    # Generate queries (including negative examples)
    queries = generator.generate_queries(
        num_per_class=150,  # 150 per positive class
        include_variations=True,
        negative_ratio=0.30  # 30% negative examples
    )
    
    print("\n" + "="*80)
    print("VALIDATING WITH RULE-BASED CLASSIFIER")
    print("="*80 + "\n")
    
    validated_queries, stats = generator.validate_with_rules(queries)
    
    print(f"\nValidation Statistics:")
    print(f"  Total: {stats['total']}")
    print(f"  Agreed: {stats['agreed']} ({stats['agreed']/stats['total']*100:.1f}%)")
    print(f"  Disagreed: {stats['disagreed']} ({stats['disagreed']/stats['total']*100:.1f}%)")
    
    if stats['disagreements_by_type']:
        print(f"\n  Disagreement breakdown:")
        for key, count in stats['disagreements_by_type'].items():
            print(f"    {key}: {count}")
    
    # Save dataset
    output_path = "synthetic_query_dataset.json"
    generator.save_dataset(validated_queries, output_path, train_split=0.8)
    
    print(f"\n{'='*80}")
    print(f"✅ DATASET GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nNext step: Train the model with:")
    print(f"  python -m src.main.classifier_fine_tuning.train_classifier")


if __name__ == "__main__":
    main()