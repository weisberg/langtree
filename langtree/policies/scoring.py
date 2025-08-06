"""Scoring policy for evaluating nodes."""

import json
from typing import Any, Dict, Optional, Tuple

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..models import ThoughtNode, TreeManager


class ScoringPolicy:
    """
    Combines simple deterministic checks with LLM self-grading.
    Produces score in [0,1] and a confidence in [0,1].
    """
    
    def __init__(self, llm: Runnable, weights: Optional[Dict[str, float]] = None):
        self.llm = llm
        self.weights = weights or {
            "accuracy": 0.4,
            "consistency": 0.2,
            "progress": 0.25,
            "novelty": 0.1,
            "cost": -0.05,
        }
        self.parser = JsonOutputParser()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "You are a strict evaluator. Given a task and a candidate partial solution, "
                 "return JSON with fields: score (0-1), confidence (0-1), components "
                 "(accuracy, consistency, progress, novelty, cost), rationale (short). Be conservative."),
                ("user",
                 "Task:\n{task}\n\nCandidate state (JSON):\n{state_json}\n\n"
                 "Parent state (JSON):\n{parent_state_json}\n\n"
                 "Constraints:\n{constraints}\n\n"
                 "Return JSON only.")
            ]
        )
        self.chain: Runnable = self.prompt | self.llm | self.parser

    def score(self, node: ThoughtNode, tree: TreeManager, task: str, constraints: str = "") -> Tuple[float, float, Dict[str, float], str]:
        """Score a node and return (score, confidence, components, rationale)."""
        parent_state = tree.nodes[node.parent_id].state if node.parent_id else {}
        state_json = json.dumps(node.state or {}, ensure_ascii=False)
        parent_state_json = json.dumps(parent_state, ensure_ascii=False)

        try:
            res: Dict[str, Any] = self.chain.invoke(
                {"task": task, "state_json": state_json, "parent_state_json": parent_state_json, "constraints": constraints}
            )
        except Exception as e:
            # Fallback conservative scoring
            res = {
                "score": 0.3,
                "confidence": 0.3,
                "components": {"accuracy": 0.3, "consistency": 0.3, "progress": 0.2, "novelty": 0.1, "cost": 0.1},
                "rationale": f"LLM grading failed: {e}",
            }

        comp = res.get("components") or {}
        # Weighted aggregate (normalize any missing fields and handle invalid values)
        def safe_float(value, default=0.0):
            """Safely convert value to float, handling invalid cases."""
            try:
                result = float(value)
                if not (result == result):  # Check for NaN
                    return default
                if result == float('inf') or result == float('-inf'):
                    return default
                return result
            except (TypeError, ValueError):
                return default
        
        acc = safe_float(comp.get("accuracy", 0.0))
        cons = safe_float(comp.get("consistency", 0.0))
        prog = safe_float(comp.get("progress", 0.0))
        nov = safe_float(comp.get("novelty", 0.0))
        cost = safe_float(comp.get("cost", 0.0))  # higher cost => lower score via negative weight

        score = (
            self.weights["accuracy"] * acc
            + self.weights["consistency"] * cons
            + self.weights["progress"] * prog
            + self.weights["novelty"] * nov
            + self.weights["cost"] * cost
        )
        score = max(0.0, min(1.0, score))
        conf = max(0.0, min(1.0, safe_float(res.get("confidence", 0.0))))
        rationale = res.get("rationale", "")
        return score, conf, {"accuracy": acc, "consistency": cons, "progress": prog, "novelty": nov, "cost": cost}, rationale