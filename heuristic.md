# Tree-of-Thought Branch-Value Heuristic Playbook (LangChain-Oriented)

---

## 1. **Core Principles**

-  **Value = Expected Utility**:  
  Score each node by its expected payoff:  
  \[
  \text{Value} = (\text{Likelihood of Success}) \times (\text{Payoff}) - (\text{Cost/Risk})
  \]

-  **Task Awareness**:  
  Use domain-specific checks/tests when possible; only use LLM self-grading when ground truth is unavailable.

-  **Separate Score & Uncertainty**:  
  Track both a mean score and a confidence/variance measure to balance exploration and exploitation.

---

## 2. **Signals to Compute Per Node**

| Signal Type         | How to Compute                                                                                         |
|---------------------|-------------------------------------------------------------------------------------------------------|
| **Intrinsic Quality** | - LLM self-grade (task rubric) <br> - Rule/constraint checks <br> - Consistency with facts <br> - Logical soundness |
| **External Verifications** | - Code: unit tests, static analysis, sandbox exec <br> - Math: numeric/algebraic checks <br> - Planning: feasibility, dependencies |
| **Information Gain / Novelty** | - Embedding distance from siblings/ancestors <br> - Uncertainty reduction <br> - Subgoal coverage |
| **Progress & Completeness** | - Subgoal completion % <br> - Milestone attainment <br> - Depth-normalized progress <br> - Heuristic distance to goal |
| **Cost & Risk** | - Token/tool cost so far <br> - Projected continuation cost <br> - Latency impact <br> - Safety/risk flags |

---

## 3. **Scoring Function**

\[
\text{Score(node)} = w_\text{acc} \times \text{Acc} + w_\text{cons} \times \text{Consistency} + w_\text{prog} \times \text{Progress} + w_\text{nov} \times \text{Novelty} - w_\text{cost} \times \text{NormCost} - w_\text{risk} \times \text{Risk}
\]

-  **Confidence**: Value in \([0,1]\), either from LLM or derived from components.
-  **Normalization**: Normalize each component per-depth or use z-scores over siblings for comparability.

---

## 4. **Choosing Weights**

| Task Type  | Example Weights (Acc, Cons, Prog, Nov, Cost, Risk) |
|------------|----------------------------------------------------|
| Coding     | 0.5, 0.1, 0.2, 0.1, 0.05, 0.05                     |
| Math       | 0.4, 0.3, 0.2, 0.05, 0.05                          |
| Planning   | 0.4 (feasibility), 0.3, 0.2, 0.1                   |

-  **Calibration**:  
  - Start with defaults.  
  - Offline: regress final success on metrics to refine weights.  
  - Online: use bandit optimization for weights.

---

## 5. **Selection Policy Interface**

| Policy      | Priority Formula                                                                           |
|-------------|--------------------------------------------------------------------------------------------|
| Best-first  | Top-K by Score                                                                             |
| UCB/PUCT    | \(\text{Priority} = \text{Score} + \beta \times \text{UncertaintyBonus} + \gamma \times \text{NoveltyBonus}\) <br> - UncertaintyBonus: \(\sqrt{\frac{\log(N_\text{parent}+1)}{n_\text{node}+1}}\) or LLM confidence <br> - NoveltyBonus: Embedding distance from top sibling(s) |
| Dynamic K   | If top score & confidence are high, reduce K; if scores are clustered/uncertain, increase K |

---

## 6. **Heuristic Details by Task**

### **Coding**
-  **Acc**: Test pass rate, runtime checks.
-  **Consistency**: Type/static analysis.
-  **Progress**: Failing tests decreased.
-  **Novelty**: AST similarity.
-  **Risk/Cost**: Sandbox errors, timeouts.

### **Math/Reasoning**
-  **Acc**: Numeric/unit checks.
-  **Consistency**: No contradictions.
-  **Progress**: Sub-lemmas solved.
-  **Novelty**: Different solution strategies.
-  **Risk**: Speculative leaps.

### **Planning/Agents**
-  **Feasibility**: Constraints, prerequisites.
-  **Progress**: Completed tasks, dependencies reduced.
-  **Novelty**: Distinct plan structures.
-  **Risk**: Unsafe actions, low-ROI steps.

---

## 7. **LLM Self-Grading Prompt Tips**

-  Ask for JSON:  
  ```json
  {
    "score": 0.83,
    "confidence": 0.7,
    "fail_reasons": [],
    "constraint_violations": [],
    "novelty_reasoning": "Explores a new approach by ..."
  }
  ```
-  Provide strict rubrics and examples.
-  Penalize unverifiable assertions, reward explicit, checkable claims.

---

## 8. **Combating Failure Modes**

| Failure Mode    | Mitigation                                                                   |
|-----------------|------------------------------------------------------------------------------|
| Length bias     | Normalize by depth; use progress, not token count                            |
| Redundancy      | Diversity pruning (embeddings, AST/structure hashes)                         |
| Hallucination   | Require citations/evidence; cross-check with retrieval; fact-consistency subscore |
| Early convergence | Keep small exploration bonus for low-confidence, diverse nodes                |

---

## 9. **Implementation Sketch (LangChain)**

```python
class ScoringPolicy:
    def score(self, node) -> dict:
        # 1. Run verifiers/tools for deterministic metrics
        # 2. If needed, call LLM grader via ChatPromptTemplate + StructuredOutputParser
        # 3. Aggregate: {'score': float, 'confidence': float, 'components': dict, 'rationale': str}
        return result

class SelectionPolicy:
    def select(self, frontier, k):
        # Maintain min-heap: priority = score + β*uncertainty + γ*novelty
        # Update per expansion; store components in node.metadata
        return top_k_nodes
```
-  Normalize components per-depth using rolling stats in `TreeManager`.

---

## 10. **Adaptive Heuristic Tuning**

-  **Online**: Multi-armed bandit over weights, optimizing success@k under cost.
-  **Offline**: Logistic regression of final success on metrics; use coefficients as weights.
-  **Per-domain**: Load weight sets/templates based on task tags.

---

## 11. **Least Valuable Branch Definition**

Flag a branch as low-value if:
-  Any hard constraint is violated.
-  Strictly dominated by a sibling (worse on all major metrics, not more novel).
-  Low score with high confidence of being wrong.
-  High projected cost with little progress potential.

---

## 12. **Summary Table: Signals and Weighting**

| Metric        | Coding         | Math/Reasoning | Planning/Agents |
|:--------------|:--------------|:--------------|:---------------|
| Accuracy      | 0.5           | 0.4           | 0.4 (feasibility) |
| Consistency   | 0.1           | 0.3           | 0.0            |
| Progress      | 0.2           | 0.2           | 0.3            |
| Novelty       | 0.1           | 0.05          | 0.2            |
| Cost          | 0.05          | 0.05          | 0.05           |
| Risk          | 0.05          | 0.0           | 0.05           |

---

## 13. **How to Start**

-  Implement a simple weighted sum with a few signals (accuracy, cost, novelty).
-  Use LLM self-grading only when deterministic checks aren’t available.
-  Add confidence and rationale fields for transparency.
-  Tune weights over time via offline/online experiments.

---

This playbook gives you a modular, explainable, and tunable heuristic for branch scoring and selection in Tree-of-Thought pipelines with LangChain. Start simple; evolve it as your domains and cost constraints demand.

