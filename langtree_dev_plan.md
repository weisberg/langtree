# LangTree Development Roadmap

A staged, actionable roadmap with milestones, tasks, and checkmarks. Includes completed work from the provided `main.py`.

## Phase 0: Foundations and MVP (Weeks 1–2)

- Core abstractions and orchestrator
  - [x] Define `ThoughtNode` dataclass with fields: `id`, `parent_id`, `depth`, `state`, `action`, `score`, `confidence`, `components`, `metadata`, `children`, `status`
  - [x] Implement `TreeManager` with node registry, frontier management, depth index, best-nodes, and `snapshot()`
  - [x] Implement `ExpansionPolicy` using LangChain `ChatPromptTemplate` + `JsonOutputParser`
  - [x] Implement `ScoringPolicy` with LLM self-grader returning `score`, `confidence`, `components`, `rationale`
  - [x] Implement `SelectionPolicy` (priority = score + beta*(1-confidence) + gamma*novelty)
  - [x] Implement `PruningPolicy` (beam width per depth + min score threshold)
  - [x] Implement `TerminationPolicy` (max depth, max nodes, target score, no-open-nodes)
  - [x] Implement `ToTOrchestrator` with `run()`, `init_root()`, `step_expand_and_score()`
  - [x] Seed root with trivial score/novelty for initial selection
- CLI and bootstrap
  - [x] CLI arguments for task, constraints, models, k-children, depth, nodes, beam, min-score, print-tree
  - [x] `build_llm()` helper using `ChatOpenAI` (provider-agnostic via LangChain)
  - [x] Pretty-print results with Rich tree (optional) and compact JSON to stdout
- Prompts and parsing
  - [x] JSON-only outputs for expansion and scoring
  - [x] Parse-guard with LLM exceptions handled via conservative fallbacks
- Documentation and examples
  - [x] Inline usage notes and run command examples in `main.py`
  - [ ] Add README quickstart with sample tasks and screenshots of the tree

## Phase 1: Developer Experience and Reliability (Weeks 2–3)

- Packaging and structure
  - [ ] Convert to package layout: `langtree/` with modules: `orchestrator.py`, `policies/`, `models/`, `cli.py`
  - [ ] Add `pyproject.toml` / `setup.cfg` with dependencies and extras (e.g., `[rich]`)
  - [ ] Type hints across all public APIs; `mypy` config and clean pass
- Testing
  - [ ] Unit tests for `TreeManager` (add/remove nodes, frontier consistency, snapshot)
  - [ ] Unit tests for `SelectionPolicy`, `PruningPolicy`, and `TerminationPolicy`
  - [ ] Golden tests for prompts + parsers using a mock LLM
  - [ ] Integration test that runs end-to-end with small mock tasks
- Reliability and error handling
  - [ ] Retry and backoff on LLM errors (parse-repair prompt, structured re-ask)
  - [ ] Deterministic seeds for sampling where supported; document reproducibility
  - [ ] Guard against runaway frontier growth with hard caps per iteration

## Phase 2: Tooling and Deterministic Scoring (Weeks 3–4)

- Tool-enabled expansion
  - [ ] Optional `AgentExecutor`-based `ExpansionPolicy` variant that can call tools
  - [ ] Example tools: calculator, web search stub, code execution sandbox
- Deterministic scoring hooks
  - [ ] `ScoringPolicy` plugins for: unit tests pass rate (code), numeric checks (math), constraint validator (planning)
  - [ ] Composite scoring: combine deterministic metrics with LLM self-grade
  - [ ] Configurable weights per task domain; YAML-based profiles
- Safety and sandboxing
  - [ ] Sandboxed code exec tool with time/memory limits and safe imports
  - [ ] Policy filters for tool outputs and prompt content (basic allowlist/denylist)

## Phase 3: Advanced Search and Efficiency (Weeks 5–6)

- Search strategies
  - [ ] UCB/PUCT-style `SelectionPolicy` balancing exploration-exploitation
  - [ ] Dynamic branching: adapt `k_children` and temperature based on node score/confidence
  - [ ] Depth-aware normalization to avoid length bias
- Diversity and deduplication
  - [ ] Embedding-based novelty: compute distance vs siblings/ancestors
  - [ ] Structural dedup for code (AST hashing) and math (normalized expression forms)
  - [ ] Diversity pruning thresholds and configs
- Performance and budgets
  - [ ] Parallelize expansions per depth with configurable concurrency
  - [ ] Batch LLM calls where provider supports it
  - [ ] Token and cost budgeting: track per node/run; enforce stop conditions

## Phase 4: Observability, Persistence, and Caching (Weeks 6–7)

- Observability
  - [ ] LangSmith tracing integration with spans for prompts, generations, scores, tool calls
  - [ ] Structured logging: costs, token usage, latencies; log levels and sampling
  - [ ] Node-level audit trail: store prompts, outputs, and rationales in `metadata`
- Persistence
  - [ ] `StorageProvider` interface: file-based JSON and pluggable DB backends
  - [ ] `replay` utility to reconstruct a tree and best path from snapshot
- Caching
  - [ ] LLM call cache keyed by prompt hash and model params
  - [ ] Tool result cache keyed by input and tool version
  - [ ] Dedup across nodes to reduce redundant calls

## Phase 5: Configuration, CLI Suite, and Docs (Weeks 7–8)

- Configuration
  - [ ] YAML config loader with env overrides; validation via `pydantic`
  - [ ] Policy presets per domain (coding, math, planning, RAG)
  - [ ] Budget profiles (low-cost, balanced, high-accuracy)
- CLI
  - [ ] `langtree run` with config file support
  - [ ] `langtree replay --from snapshot.json`
  - [ ] `langtree eval --suite <path> --config <yaml> --out <csv>`
- Documentation
  - [ ] Comprehensive README and docs site (Quickstart, Policy Cookbook, API reference)
  - [ ] Example notebooks for coding, math, planning tasks
  - [ ] Troubleshooting guide and performance tips

## Phase 6: Evaluation and Tuning (Weeks 8–9)

- Offline evaluation
  - [ ] Benchmark suite (GSM8K-like math, small code katas, planning tasks)
  - [ ] Baselines: single CoT, self-consistency, ToT variants
  - [ ] Metrics: accuracy, cost, latency, expansions per success, best-depth
- Weight tuning
  - [ ] Fit logistic regression mapping component metrics to success; derive weights
  - [ ] Bandit search over weight vectors and branching parameters
  - [ ] Save tuned profiles and rollout playbooks

## Phase 7: Production Hardening (Weeks 9–10)

- Reliability
  - [ ] Circuit breakers on repeated tool/LLM failures
  - [ ] Rate limiting and backpressure for concurrency spikes
  - [ ] Graceful degradation to simpler strategies under heavy load
- Security and compliance
  - [ ] Secrets management integration (e.g., AWS/GCP secrets)
  - [ ] PII scrubbing and safe-logging defaults
  - [ ] SBOM and dependency pinning; supply chain checks

## Stretch Goals

- Learned ranker
  - [ ] Optional ML ranker for scoring nodes using collected features
- Interactive human-in-the-loop
  - [ ] Pause/resume with manual branch curation
  - [ ] Lightweight web UI for inspecting and steering trees
- Multi-modal support
  - [ ] Image/table-aware expansion/scoring prompts where models support it

## Completed (from main.py)

- [x] Core data model: `ThoughtNode`
- [x] Tree management: `TreeManager` with frontier, depth index, and snapshots
- [x] Policies
  - [x] `ExpansionPolicy` (LLM-based, JSON-structured outputs)
  - [x] `ScoringPolicy` (LLM self-grading with components and confidence)
  - [x] `SelectionPolicy` (score + uncertainty + novelty)
  - [x] `PruningPolicy` (beam + min score)
  - [x] `TerminationPolicy` (depth, node count, target score, no open nodes)
- [x] Orchestrator: `ToTOrchestrator` with `run`, `init_root`, `step_expand_and_score`
- [x] CLI entrypoint, argument parsing, model builders, and result rendering
- [x] Error fallback for parsing/LLM exceptions in expansion and scoring
- [x] Pretty tree output via Rich (optional) and compact JSON output

## Dependencies and Environment

- Core
  - [x] LangChain, langchain-core
  - [x] OpenAI-compatible chat model via `langchain-openai` (swappable)
  - [x] numpy (for potential scoring/normalization)
  - [x] rich (optional for tree UI)
- Next steps
  - [ ] pydantic for config validation
  - [ ] pytest + coverage for tests
  - [ ] typing/mypy for type checks

## Release Plan

- v0.1.0 (MVP): Phases 0–1
- v0.2.0: Phase 2 (tools, deterministic scoring)
- v0.3.0: Phase 3 (advanced search, diversity)
- v0.4.0: Phase 4–5 (observability, persistence, CLI suite, docs)
- v0.5.0: Phase 6–7 (evaluation, production hardening)