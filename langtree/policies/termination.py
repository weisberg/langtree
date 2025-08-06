"""Termination policy for deciding when to stop the search."""

from ..models import TreeManager


class TerminationPolicy:
    """Determine when to stop the tree search."""
    
    def __init__(self, max_depth: int = 4, max_nodes: int = 50, target_score: float = 0.9):
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.target_score = target_score

    def should_stop(self, tree: TreeManager) -> bool:
        """Check if the search should terminate."""
        # Stop if max nodes reached
        if len(tree.nodes) >= self.max_nodes:
            return True
            
        # Stop if target score achieved
        best = tree.best_nodes(1)[0] if tree.nodes else None
        if best and best.score >= self.target_score:
            return True
            
        # Stop if no open nodes
        open_left = any(tree.nodes[nid].status == "open" for nid in tree.frontier)
        if not open_left:
            return True
            
        # Stop if all open nodes exceed max depth
        if all(tree.nodes[nid].depth >= self.max_depth for nid in tree.frontier):
            return True
            
        return False