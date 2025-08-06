"""Pruning policy for managing tree size."""

from ..models import TreeManager


class PruningPolicy:
    """Prune nodes based on beam width and minimum score thresholds."""
    
    def __init__(self, beam_per_depth: int = 3, min_score: float = 0.0, diversity_threshold: float = 0.0):
        self.beam = beam_per_depth
        self.min_score = min_score
        self.diversity_threshold = diversity_threshold  # placeholder if you add embeddings

    def prune(self, tree: TreeManager, depth: int) -> None:
        """Prune nodes at a given depth."""
        node_ids = tree.depth_index.get(depth, [])
        nodes = [tree.nodes[nid] for nid in node_ids if tree.nodes[nid].status in ("open", "expanded")]
        
        # Threshold prune
        nodes = [n for n in nodes if n.score >= self.min_score]
        
        # Beam prune
        nodes.sort(key=lambda n: n.score, reverse=True)
        keep = set([n.id for n in nodes[: self.beam]])
        
        for n in nodes[self.beam :]:
            n.status = "pruned"
            if n.id in tree.frontier:
                tree.frontier.remove(n.id)
                tree.open_status.pop(n.id, None)