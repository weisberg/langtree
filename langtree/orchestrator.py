"""Tree-of-Thought orchestrator."""

import time
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.runnables import Runnable

from .models import ThoughtNode, TreeManager
from .policies import (
    ExpansionPolicy,
    ScoringPolicy,
    SelectionPolicy,
    PruningPolicy,
    TerminationPolicy,
)


class ToTOrchestrator:
    """Main orchestrator for Tree-of-Thought reasoning."""
    
    def __init__(
        self,
        llm_expand: Runnable,
        llm_score: Runnable,
        k_children: int = 3,
        max_select: int = 3,
        max_depth: int = 4,
        max_nodes: int = 50,
        target_score: float = 0.9,
        beam_per_depth: int = 3,
        min_score: float = 0.0,
    ):
        self.tree = TreeManager()
        self.expansion = ExpansionPolicy(llm_expand, k=k_children)
        self.scoring = ScoringPolicy(llm_score)
        self.selection = SelectionPolicy()
        self.pruning = PruningPolicy(beam_per_depth=beam_per_depth, min_score=min_score)
        self.termination = TerminationPolicy(max_depth=max_depth, max_nodes=max_nodes, target_score=target_score)
        self.max_select = max_select

    def init_root(self, task: str, initial_state: Optional[Dict[str, Any]] = None) -> ThoughtNode:
        """Initialize the root node."""
        root = ThoughtNode(
            id=str(uuid.uuid4()),
            parent_id=None,
            depth=0,
            state=initial_state or {"reasoning_steps": [], "scratchpad": "", "partial_solution": ""},
            status="open",
            metadata={"created_at": time.time()},
        )
        self.tree.add_node(root)
        return root

    def step_expand_and_score(self, node: ThoughtNode, task: str, constraints: str = "") -> None:
        """Expand a node and score its children."""
        children = self.expansion.expand(node, task, constraints)
        
        # Score each child
        for c in children:
            score, conf, comps, rationale = self.scoring.score(c, self.tree, task, constraints)
            c.score = score
            c.confidence = conf
            c.components = comps
            c.metadata["score_rationale"] = rationale
            
        self.tree.add_children(node.id, children)
        self.tree.set_expanded(node.id)

    def run(self, task: str, constraints: str = "", initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the Tree-of-Thought search."""
        root = self.init_root(task, initial_state)

        # Score root trivially to seed selection
        root.score = 0.0
        root.confidence = 0.0
        root.components = {"novelty": 1.0}

        while not self.termination.should_stop(self.tree):
            # Select nodes to expand
            to_expand = self.selection.select(self.tree, self.max_select)
            if not to_expand:
                break
                
            # Expand and score
            for node in to_expand:
                if node.depth >= self.termination.max_depth:
                    node.status = "terminal"
                    if node.id in self.tree.frontier:
                        self.tree.frontier.remove(node.id)
                        self.tree.open_status.pop(node.id, None)
                    continue
                self.step_expand_and_score(node, task, constraints)

            # Prune at each depth touched
            for depth in list(self.tree.depth_index.keys()):
                self.pruning.prune(self.tree, depth)

        best = self.tree.best_nodes(1)[0]
        return {
            "best_node_id": best.id,
            "best_score": best.score,
            "best_confidence": best.confidence,
            "best_path": [n.state.get("partial_solution", "") for n in best.path(self.tree)],
            "best_actions": [n.action for n in best.path(self.tree)],
            "best_state": best.state,
            "tree_snapshot": self.tree.snapshot(),
        }