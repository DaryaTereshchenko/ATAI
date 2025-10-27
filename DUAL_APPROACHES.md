# Factual and Embedding Approaches Implementation

This document describes the implementation of factual and embedding approaches for answering natural language questions about movies.

## Overview

The system implements two distinct approaches for answering questions:

1. **Factual Approach**: Converts natural language to SPARQL queries and retrieves answers from the knowledge graph
2. **Embedding Approach**: Uses TransE embeddings to compute answers through vector arithmetic

## Architecture

### Components

#### 1. Approach Detector (`approach_detector.py`)

Detects which approach is requested from the query text:

```python
# Example queries
"Please answer this question with a factual approach: Who directed 'Fargo'?"
→ ApproachType.FACTUAL

"Please answer this question with an embedding approach: Who is the director of 'Apocalypse Now'?"
→ ApproachType.EMBEDDING

"Please answer this question: Who is the director of 'Good Will Hunting'?"
→ ApproachType.BOTH (use both approaches)
```

#### 2. Response Formatter (`response_formatter.py`)

Formats responses according to the requirements:

**Factual Approach:**
- Multiple answers concatenated with "and"
- Format: `The factual answer is: <answer1> and <answer2> and ...`
- Example: `The factual answer is: Ethan Coen and Joel Coen`

**Embedding Approach:**
- Single answer with entity type
- Format: `The answer suggested by embeddings is: <answer> (type: <QID>)`
- Example: `The answer suggested by embeddings is: John Milius (type: Q5)`

**Both Approaches:**
- Combines both formats
- Example: `The factual answer is: Gus Van Sant. The answer suggested by embeddings is: Harmony Korine (type: Q5)`

#### 3. Embedding Answer Finder (`embedding_answer_finder.py`)

Implements pure embedding-based answering using TransE:

```python
# TransE formula: tail ≈ head + relation
answer_embedding = subject_embedding + relation_embedding

# Find nearest entity to computed embedding
nearest_entities = find_nearest(answer_embedding)
```

Features:
- Entity extraction from questions
- Relation detection (director, screenwriter, genre, etc.)
- TransE computation: `subject + relation ≈ answer`
- Entity type detection (Q5 for person, Q201658 for genre, etc.)

#### 4. Dual Approach Processor (`dual_approach_processor.py`)

Routes queries to the appropriate approach:

```python
1. Detect approach from query
2. Extract actual question
3. Route to:
   - Factual: NL → SPARQL → Execute → Format
   - Embedding: Entity extraction → TransE computation → Format
   - Both: Execute both and combine results
```

### Integration

The dual approach processor is integrated into the orchestrator and workflow:

```
User Query
    ↓
[Approach Detector] → Detect factual/embedding/both
    ↓
[Dual Processor] → Route to appropriate handler
    ↓
[Response Formatter] → Format according to approach
    ↓
Natural Language Response
```

## Example Queries and Expected Outputs

### Factual Approach Examples

1. **Single answer:**
   - Query: `Please answer this question with a factual approach: From what country is the movie 'Aro Tolbukhin. En la mente del asesino'?`
   - Expected: `The factual answer is: Mexico`

2. **Multiple answers:**
   - Query: `Please answer this question with a factual approach: Who directed 'Fargo'?`
   - Expected: `The factual answer is: Ethan Coen and Joel Coen`

3. **Multiple genres:**
   - Query: `Please answer this question with a factual approach: What genre is the movie 'Bandit Queen'?`
   - Expected: `The factual answer is: drama film and biographical film and crime film`

4. **Date:**
   - Query: `Please answer this question with a factual approach: When did the movie 'Miracles Still Happen' come out?`
   - Expected: `The factual answer is: 1974-07-19`

### Embedding Approach Examples

1. **Director (person):**
   - Query: `Please answer this question with an embedding approach: Who is the director of 'Apocalypse Now'?`
   - Expected: `The answer suggested by embeddings is: John Milius (type: Q5)`

2. **Screenwriter (person):**
   - Query: `Please answer this question with an embedding approach: Who is the screenwriter of '12 Monkeys'?`
   - Expected: `The answer suggested by embeddings is: Carol Florence (type: Q5)`

3. **Genre:**
   - Query: `Please answer this question with an embedding approach: What is the genre of 'Shoplifters'?`
   - Expected: `The answer suggested by embeddings is: comedy film (type: Q201658)`

### Both Approaches Example

- Query: `Please answer this question: Who is the director of 'Good Will Hunting'?`
- Expected: `The factual answer is: Gus Van Sant. The answer suggested by embeddings is: Harmony Korine (type: Q5)`

## Entity Type Codes

Common Wikidata entity types:
- `Q5`: Human/Person (directors, actors, screenwriters)
- `Q11424`: Film/Movie
- `Q201658`: Film genre
- `Q6256`: Country
- `Q618779`: Award
- `Q35120`: Entity (generic fallback)

## Implementation Details

### Factual Approach

1. **Query Analysis**: Detect pattern (forward/reverse/verification)
2. **Entity Extraction**: Find movies, people, properties from text
3. **SPARQL Generation**: Template-based or LLM-based generation
4. **Execution**: Query the knowledge graph
5. **Formatting**: Join multiple results with "and"

### Embedding Approach

1. **Entity Extraction**: Find subject entity (e.g., movie title)
2. **Relation Detection**: Identify property (director, genre, etc.)
3. **TransE Computation**:
   ```python
   subject_emb = get_embedding(subject_uri)
   relation_emb = get_embedding(relation_uri)
   answer_emb = subject_emb + relation_emb
   ```
4. **Nearest Neighbor Search**: Find closest entity to `answer_emb`
5. **Type Detection**: Determine entity type from context
6. **Formatting**: Single answer with type

## Testing

Run the test suite:

```bash
python tests/test_dual_approaches.py
```

This tests:
1. Approach detection from query text
2. Response formatting for both approaches
3. Example queries from the problem statement

## Configuration

Key configuration in `src/config.py`:

```python
# Enable embeddings
USE_EMBEDDINGS = True

# Embeddings directory (contains TransE embeddings)
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "data", "embeddings")

# Knowledge graph path
GRAPH_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "14_graph.nt")
```

## Dependencies

- **rdflib**: Knowledge graph processing
- **numpy**: Vector operations for embeddings
- **sentence-transformers**: Query embeddings (optional)
- **transformers**: Query classification (optional)

## Usage

```python
from src.main.orchestrator import Orchestrator

# Initialize orchestrator
orchestrator = Orchestrator(use_workflow=True)

# Process factual query
response = orchestrator.process_query(
    "Please answer this question with a factual approach: Who directed 'Fargo'?"
)
# Output: "The factual answer is: Ethan Coen and Joel Coen"

# Process embedding query
response = orchestrator.process_query(
    "Please answer this question with an embedding approach: Who is the director of 'Apocalypse Now'?"
)
# Output: "The answer suggested by embeddings is: John Milius (type: Q5)"

# Process with both approaches
response = orchestrator.process_query(
    "Please answer this question: Who is the director of 'Good Will Hunting'?"
)
# Output: "The factual answer is: Gus Van Sant. The answer suggested by embeddings is: Harmony Korine (type: Q5)"
```

## Notes

- **Factual approach** may return multiple answers (order doesn't matter)
- **Embedding approach** returns a single answer with entity type
- **Entity types** are crucial for embedding responses
- **TransE embeddings** must be pre-computed and stored in the embeddings directory
- The system gracefully falls back to SPARQL if embeddings are unavailable
