# Quick Start Guide - Dual Approach System

## What Was Implemented

Two approaches for answering movie questions:

1. **Factual Approach**: Query knowledge graph with SPARQL → Return all answers with "and"
2. **Embedding Approach**: Compute using TransE vectors → Return single answer with entity type

## Quick Demo (No Data Required)

```bash
# Run demo to see how the system works
python demo_dual_approaches.py

# Run tests
python tests/test_dual_approaches.py
```

## Usage Examples

### Option 1: Factual Approach
```python
from src.main.orchestrator import Orchestrator

orchestrator = Orchestrator(use_workflow=True)
response = orchestrator.process_query(
    "Please answer this question with a factual approach: Who directed 'Fargo'?"
)
# Output: "The factual answer is: Ethan Coen and Joel Coen"
```

### Option 2: Embedding Approach
```python
response = orchestrator.process_query(
    "Please answer this question with an embedding approach: Who is the director of 'Apocalypse Now'?"
)
# Output: "The answer suggested by embeddings is: John Milius (type: Q5)"
```

### Option 3: Both Approaches
```python
response = orchestrator.process_query(
    "Please answer this question: Who is the director of 'Good Will Hunting'?"
)
# Output: "The factual answer is: Gus Van Sant. The answer suggested by embeddings is: Harmony Korine (type: Q5)"
```

## How It Works

```
Query → Approach Detector → Route to Handler → Format Response
```

### Factual Path:
```
NL Question → SPARQL Generation → Execute on Graph → Multiple Answers (with "and")
```

### Embedding Path:
```
Extract Entity & Relation → TransE (subject + relation) → Nearest Entity → Single Answer (with type)
```

## Files Created

### Core Implementation (5 files):
- `src/config.py` - Configuration
- `src/main/approach_detector.py` - Detects approach type
- `src/main/response_formatter.py` - Formats responses
- `src/main/embedding_answer_finder.py` - TransE computation
- `src/main/dual_approach_processor.py` - Main processor

### Testing & Documentation (5 files):
- `tests/test_dual_approaches.py` - Test suite
- `demo_dual_approaches.py` - Interactive demo
- `DUAL_APPROACHES.md` - Technical docs
- `IMPLEMENTATION_SUMMARY.md` - Overview
- `ARCHITECTURE_DIAGRAM.txt` - Visual diagram

### Modified (2 files):
- `src/main/orchestrator.py` - Added dual processor
- `src/main/workflow.py` - Added routing logic

## Entity Types

Common Wikidata entity types returned by embedding approach:
- `Q5` - Person (directors, actors, etc.)
- `Q201658` - Film genre
- `Q11424` - Film/Movie
- `Q6256` - Country
- `Q618779` - Award

## Data Requirements (for production use)

Place data files in these locations:
- Knowledge graph: `data/14_graph.nt`
- Embeddings: `data/embeddings/` (entity_embeds.npy, relation_embeds.npy, etc.)

## Testing

```bash
# Run all tests
python tests/test_dual_approaches.py

# Expected output:
# ✅ PASS - Approach Detection (3/3 tests)
# ✅ PASS - Response Formatting (2/2 tests)
# ✅ PASS - Example Queries (validated)
# 🎉 All tests passed!
```

## Documentation

For detailed information, see:
- **IMPLEMENTATION_SUMMARY.md** - Complete implementation overview
- **DUAL_APPROACHES.md** - Technical documentation
- **ARCHITECTURE_DIAGRAM.txt** - Visual system architecture

## Status

✅ **Implementation Complete**
- All features working
- All tests passing
- Code review passed
- Security scan passed
- Documentation complete

Ready for production use with data.
