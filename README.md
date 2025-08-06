# LangTree

LangTree is a Tree-of-Thought (ToT) orchestration framework built on LangChain. It enables multi-branch reasoning, controlled exploration, and systematic selection of the best path using pluggable policies for expansion, scoring, selection, pruning, and termination.

LangTree is designed for developers who want higher accuracy on complex tasks (coding, math, planning, RAG) with predictable cost and latency.

## Features

- Modular policies
  - `ExpansionPolicy`: Generate diverse next-step candidates with LLMs or agents/tools.
  - `ScoringPolicy`: Combine deterministic checks with LLM self-grading to produce a score and confidence.
  - `SelectionPolicy`: Best-first or exploration-aware (uncertainty/novelty bonuses).
  - `PruningPolicy`: Beam width per depth, score thresholds, diversity pruning hooks.
  - `TerminationPolicy`: Depth/node/time budgets and target-confidence exits.
- Orchestration
  - `ToTOrchestrator.run()` drives selection → expansion → scoring → pruning loops.
  - Maintains a tree of `ThoughtNode`s with snapshots for audit/replay.
- Observability and reproducibility
  - JSON-structured outputs; optional pretty tree rendering.
  - Deterministic parsing with conservative fallbacks on errors.
- Provider-agnostic
  - Uses LangChain models; swap `ChatOpenAI` with your provider.

## Quickstart

Prerequisites
- Python 3.9+
- API key for your chosen LLM provider (OpenAI-compatible by default)

Install
```
pip install langchain langchain-core langchain-openai numpy rich
```

Environment
```
export OPENAI_API_KEY=sk-...
```

Run an example
```
python main.py --task "Solve: A boat goes 30km upstream in 5h and downstream in 3h. Find speed of boat in still water." --print-tree
```

Minimal usage (programmatic)
```python
from main import ToTOrchestrator, build_llm

llm_expand = build_llm("gpt-4o-mini", temperature=0.7)
llm_score = build_llm("gpt-4o-mini", temperature=0.0)

orchestrator = ToTOrchestrator(
    llm_expand=llm_expand,
    llm_score=llm_score,
    k_children=3,
    max_select=3,
    max_depth=4,
    max_nodes=50,
    target_score=0.9,
    beam_per_depth=3,
    min_score=0.0,
)

result = orchestrator.run(task="Prove: sum of first n odd numbers is n^2", constraints="")
print(result["best_state"])
```

## Repository Structure

Current MVP is in a single `main.py`. Planned package structure:
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
    code_exec.py
    calculator.py
    search.py
  storage/
    file_store.py
  cli.py
main.py   # MVP entrypoint (current)
README.md
```

## Concepts

- `ThoughtNode`
  - Represents a node in the reasoning tree.
  - Fields: `id`, `parent_id`, `depth`, `state`, `action`, `score`, `confidence`, `components`, `metadata`, `children`, `status`.
- `TreeManager`
  - Registry of nodes and indices; manages frontier and snapshots.
- Policies
  - `ExpansionPolicy.expand(node, task, constraints) -> List[ThoughtNode]`
  - `ScoringPolicy.score(node, tree, task, constraints) -> (score, confidence, components, rationale)`
  - `SelectionPolicy.select(tree, k) -> List[ThoughtNode]`
  - `PruningPolicy.prune(tree, depth) -> None`
  - `TerminationPolicy.should_stop(tree) -> bool`
- `ToTOrchestrator`
  - High-level controller with `run()` API that returns the best node and tree snapshot.

## CLI

Basic usage
```
python main.py --task "<task text>" [--constraints "<constraints>"] \
  --model-expand gpt-4o-mini --model-score gpt-4o-mini \
  --k-children 3 --max-select 3 --max-depth 4 --max-nodes 50 \
  --target-score 0.9 --beam 3 --min-score 0.0 --print-tree
```

Flags
- `--task`: problem statement (required)
- `--constraints`: optional constraints or rubric
- `--model-expand`: model used for candidate generation
- `--model-score`: model used for grading
- `--k-children`: candidates per expansion
- `--max-select`: nodes to expand per loop
- `--max-depth`: maximum tree depth
- `--max-nodes`: hard limit on total nodes
- `--target-score`: early exit when best score ≥ target
- `--beam`: beam width per depth for pruning
- `--min-score`: minimum score to keep nodes at a depth
- `--print-tree`: pretty tree output via Rich

Outputs
- If `--print-tree` is set, prints a Rich tree and path summaries.
- Otherwise prints compact JSON with `best_score`, `best_confidence`, `best_state`, `best_actions`, `best_path`.

## How It Works

1) Initialize root node from the task.
2) Loop:
   - Selection: pick top nodes by priority = score + β*(1 - confidence) + γ*novelty.
   - Expansion: `ExpansionPolicy` creates `k` candidate children using an LLM prompt that enforces JSON schema.
   - Scoring: `ScoringPolicy` calls an LLM grader to produce a normalized score, confidence, and component breakdown (accuracy, consistency, progress, novelty, cost).
   - Insert children and mark parent as expanded.
   - Pruning: beam and threshold per depth to keep the search tractable.
   - Check termination: max depth/nodes or sufficiently high score.
3) Return the best node and reconstruct the path.

## Heuristics

Default aggregate score
- Score = 0.4*accuracy + 0.2*consistency + 0.25*progress + 0.1*novelty - 0.05*cost
- Confidence is tracked separately to support exploration bonuses.

Selection priority
- priority = score + 0.2*(1 - confidence) + 0.1*novelty

You can adjust weights in `ScoringPolicy` or implement custom policies.

## Extending LangTree

- Swap model provider
  - Replace `build_llm()` with your LangChain-compatible model, or pass any `Runnable` compatible with `ChatPromptTemplate`.
- Tool-enabled expansion
  - Replace `ExpansionPolicy.chain` with an `AgentExecutor` and `Tool`s (e.g., calculator, web search, code exec).
- Deterministic scoring
  - Extend `ScoringPolicy.score` to run unit tests (code), numeric checks (math), or constraint validators (planning) before LLM grading.
- Diversity pruning
  - Add embeddings to compute novelty against siblings/ancestors; dedup similar branches.
- Persistence and observability
  - Persist `tree.snapshot()` to disk/DB; integrate tracing via LangSmith; log token cost and latencies per node.

## Examples

Math reasoning
```
python main.py --task "Find integer solutions to x^2 - y^2 = 45" --print-tree
```

Coding (pseudo)
- Add a code-exec `Tool` and unit tests within `ScoringPolicy`.
- Score combines pass rate with self-grading to prioritize compilable, test-passing branches.

Planning
```
python main.py --task "Plan a 3-day ML sprint with tasks, dependencies, and time estimates." --constraints "Budget 20 hours total; 2 engineers"
```

## Configuration (Roadmap)

Planned YAML config (Phase 5):
```
model_expand: gpt-4o-mini
model_score: gpt-4o-mini
k_children: 3
selection:
  type: ucb
  beta: 0.2
  gamma: 0.1
pruning:
  beam_per_depth: 3
  min_score: 0.0
termination:
  max_depth: 4
  max_nodes: 50
  target_score: 0.9
profiles:
  coding:
    scoring_weights: {accuracy: 0.5, consistency: 0.1, progress: 0.2, novelty: 0.1, cost: -0.1}
```

## Development Roadmap

See ROADMAP.md (or the roadmap in the PRD). Highlights:
- MVP complete in `main.py` (core policies, orchestrator, CLI).
- Next: packaging, tests, retries/backoff, tool-enabled expansion, deterministic scoring, advanced selection (UCB/PUCT), embeddings-based diversity, caching and persistence, YAML configs, CLI `run/replay/eval`, docs and examples.

## What’s Implemented Now (from main.py)

- `ThoughtNode`, `TreeManager` with frontier, depth index, and snapshots.
- `ExpansionPolicy`: LLM-based, structured JSON outputs with fallback on errors.
- `ScoringPolicy`: LLM self-grader with component breakdown, score aggregation, and fallbacks.
- `SelectionPolicy`: score + uncertainty + novelty prioritization.
- `PruningPolicy`: beam-per-depth and min-score thresholding.
- `TerminationPolicy`: depth/node/score/no-open-nodes stop logic.
- `ToTOrchestrator`: end-to-end loop and result packaging (best path, actions, state).
- CLI flags, model builders, pretty tree output via Rich, compact JSON output mode.

## Design Choices

- Structured prompting and parsing
  - Strict JSON outputs via `JsonOutputParser` to make nodes machine-auditable.
- Separation of score and uncertainty
  - Allows exploration vs exploitation tradeoffs and more stable pruning.
- Pluggable policies
  - Keeps the system extensible and testable across domains and models.

## Contributing

We welcome issues and PRs:
- Open an issue describing the use case or bug.
- For PRs, include tests and a short design note.
- Follow typing and linting rules; keep prompts inlined and documented.

Planned dev setup
```
pip install -e .[dev]
pytest -q
ruff check .
mypy .
```

## License

MIT (proposed). See `LICENSE` when available.

## Acknowledgments

- Built on LangChain abstractions and inspired by Tree-of-Thought research and Self-Consistency strategies.
- Thanks to contributors and the broader LLM ops community for patterns on grading, pruning, and structured prompting.

## FAQ

Q: Can I use local models?
- Yes. Replace `ChatOpenAI` in `build_llm()` with your local LangChain chat model or any `Runnable`.

Q: How do I add tools?
- Implement LangChain `Tool`s and wrap them in an `AgentExecutor` used by `ExpansionPolicy` or in `ScoringPolicy` verifiers.

Q: How do I reduce cost?
- Lower `k_children`, `max_select`, and `max_depth`; raise pruning thresholds; turn on caching; use cheaper models for scoring; early-exit on target score.

Q: Is this safe for code execution?
- Only with sandboxing. Use a restricted environment with time/memory limits, and never run untrusted code in production hosts.