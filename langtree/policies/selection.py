"""Selection policy for choosing nodes to expand."""

from typing import List

from ..models import ThoughtNode, TreeManager


class SelectionPolicy:
    """
    Priority = score + beta*(1 - confidence) + gamma*novelty
    For simplicity, novelty=components.get('novelty', 0)
    """
    
    def __init__(self, beta: float = 0.2, gamma: float = 0.1):
        self.beta = beta
        self.gamma = gamma

    def select(self, tree: TreeManager, k: int) -> List[ThoughtNode]:
        """Select k nodes from the frontier to expand."""
        candidates = [tree.nodes[nid] for nid in tree.frontier if tree.nodes[nid].status == "open"]
        
        # Compute priority
        def priority(n: ThoughtNode) -> float:
            novelty = float(n.components.get("novelty", 0.0))
            return n.score + self.beta * (1.0 - n.confidence) + self.gamma * novelty

        candidates.sort(key=priority, reverse=True)
        return candidates[:k]