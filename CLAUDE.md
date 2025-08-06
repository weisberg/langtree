# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LangTree is a Tree-of-Thought (ToT) orchestration framework built on LangChain that enables multi-branch reasoning for complex tasks like coding, math, and planning.

## Development Commands

### Running the Application
```bash
# Basic usage
python main.py --task "<task description>" --print-tree

# With custom parameters
python main.py --task "<task>" --constraints "<constraints>" \
  --model-expand gpt-4o-mini --model-score gpt-4o-mini \
  --k-children 3 --max-select 3 --max-depth 4 --max-nodes 50 \
  --target-score 0.9 --beam 3 --min-score 0.0
```

### Dependencies
Install required packages:
```bash
pip install langchain langchain-core langchain-openai numpy rich
```

Set environment variable:
```bash
export OPENAI_API_KEY=sk-...
```

## Architecture

### Core Components

1. **ThoughtNode**: Represents a node in the reasoning tree with fields for state, score, confidence, and relationships.

2. **TreeManager**: Manages the tree structure, frontier nodes, and provides tree operations like adding nodes, pruning, and snapshots.

3. **Policies**: Modular components that control the search behavior:
   - **ExpansionPolicy**: Generates diverse next-step candidates using LLMs
   - **ScoringPolicy**: Evaluates nodes with combined LLM self-grading and deterministic checks
   - **SelectionPolicy**: Prioritizes nodes using score + uncertainty + novelty
   - **PruningPolicy**: Maintains beam width and minimum score thresholds
   - **TerminationPolicy**: Determines when to stop the search

4. **ToTOrchestrator**: Main controller that coordinates the tree search loop (selection → expansion → scoring → pruning).

### Key Design Patterns

- **JSON-structured outputs**: All LLM interactions use strict JSON schemas for reliability
- **Conservative fallbacks**: Graceful degradation when LLM calls or parsing fails
- **Pluggable policies**: Each policy is independently configurable and replaceable
- **Provider-agnostic**: Uses LangChain abstractions, easily swap LLM providers

### Future Structure (from roadmap)
```
langtree/
  __init__.py
  orchestrator.py
  policies/
    expansion.py
    scoring.py
    selection.py
    pruning.py
    termination.py
  tools/
  storage/
  cli.py
```

Currently all code is in `main.py` as an MVP implementation.