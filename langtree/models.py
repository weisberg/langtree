"""Data models for LangTree."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ThoughtNode:
    """Represents a node in the reasoning tree."""
    
    id: str
    parent_id: Optional[str]
    depth: int
    state: Dict[str, Any] = field(default_factory=dict)  # reasoning_steps, scratchpad, partial_solution
    prompt_context: str = ""  # cached assembled prompt if needed
    action: Optional[str] = None
    score: float = 0.0
    confidence: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)  # breakdown for audits
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)
    status: str = "open"  # open, expanded, pruned, terminal

    def path(self, tree: "TreeManager") -> List["ThoughtNode"]:
        """Return the path from root to this node."""
        cur = self
        nodes = [cur]
        visited = {cur.id}  # Track visited nodes to prevent infinite loops
        
        while cur.parent_id and cur.parent_id not in visited:
            if cur.parent_id not in tree.nodes:
                break  # Parent doesn't exist, stop traversal
            cur = tree.nodes[cur.parent_id]
            nodes.append(cur)
            visited.add(cur.id)
        
        return list(reversed(nodes))


class TreeManager:
    """Manages the tree structure and operations."""
    
    def __init__(self):
        self.nodes: Dict[str, ThoughtNode] = {}
        self.frontier: List[str] = []  # list of node ids
        self.depth_index: Dict[int, List[str]] = {}
        self.open_status: Dict[str, bool] = {}

    def add_node(self, node: ThoughtNode) -> None:
        """Add a node to the tree."""
        # Validate node doesn't create circular reference
        if node.parent_id and node.parent_id == node.id:
            raise ValueError(f"Node {node.id} cannot be its own parent")
        
        # Check for circular references by traversing up the parent chain
        if node.parent_id and node.parent_id in self.nodes:
            visited = {node.id}
            current_parent = node.parent_id
            while current_parent and current_parent in self.nodes:
                if current_parent in visited:
                    raise ValueError(f"Adding node {node.id} would create circular reference")
                visited.add(current_parent)
                current_parent = self.nodes[current_parent].parent_id
        
        # If node already exists, ensure frontier consistency
        if node.id in self.nodes:
            old_node = self.nodes[node.id]
            if old_node.status == "open" and node.status != "open":
                # Remove from frontier if status changed from open
                if node.id in self.frontier:
                    self.frontier.remove(node.id)
                self.open_status.pop(node.id, None)
        
        self.nodes[node.id] = node
        self.depth_index.setdefault(node.depth, []).append(node.id)
        
        if node.status == "open":
            if node.id not in self.frontier:  # Avoid duplicates
                self.frontier.append(node.id)
            self.open_status[node.id] = True

    def set_expanded(self, node_id: str) -> None:
        """Mark a node as expanded."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found in tree")
            
        if node_id in self.open_status:
            self.open_status.pop(node_id, None)
            if node_id in self.frontier:
                self.frontier.remove(node_id)
        self.nodes[node_id].status = "expanded"
        
        # Ensure consistency
        self._validate_frontier_consistency()

    def add_children(self, parent_id: str, children: List[ThoughtNode]) -> None:
        """Add children to a parent node."""
        parent = self.nodes[parent_id]
        for c in children:
            parent.children.append(c.id)
            self.add_node(c)

    def best_nodes(self, k: int = 1) -> List[ThoughtNode]:
        """Return the k best nodes by score."""
        all_nodes = [n for n in self.nodes.values() if n.status in ("expanded", "terminal", "open")]
        all_nodes.sort(key=lambda n: n.score, reverse=True)
        return all_nodes[:k]

    def snapshot(self) -> Dict[str, Any]:
        """Return a snapshot of the tree state."""
        return {
            "nodes": {
                nid: {
                    "parent_id": n.parent_id,
                    "depth": n.depth,
                    "score": n.score,
                    "confidence": n.confidence,
                    "status": n.status,
                    "components": n.components,
                    "action": n.action,
                    "state": n.state,
                    "children": n.children,
                }
                for nid, n in self.nodes.items()
            }
        }
    
    def _validate_frontier_consistency(self) -> None:
        """Validate that frontier and open_status are consistent."""
        # Remove nodes from frontier that are not open
        frontier_to_remove = []
        for node_id in self.frontier:
            if node_id not in self.nodes or self.nodes[node_id].status != "open":
                frontier_to_remove.append(node_id)
        
        for node_id in frontier_to_remove:
            self.frontier.remove(node_id)
            self.open_status.pop(node_id, None)
        
        # Add open nodes that are missing from frontier
        for node_id, node in self.nodes.items():
            if node.status == "open" and node_id not in self.frontier:
                self.frontier.append(node_id)
                self.open_status[node_id] = True
    
    def validate_tree_integrity(self) -> List[str]:
        """Validate tree integrity and return list of issues found."""
        issues = []
        
        # Check frontier consistency
        for node_id in self.frontier:
            if node_id not in self.nodes:
                issues.append(f"Frontier contains non-existent node: {node_id}")
            elif self.nodes[node_id].status != "open":
                issues.append(f"Frontier contains non-open node: {node_id} (status: {self.nodes[node_id].status})")
        
        for node_id in self.open_status:
            if node_id not in self.nodes:
                issues.append(f"open_status contains non-existent node: {node_id}")
            elif node_id not in self.frontier:
                issues.append(f"open_status contains node not in frontier: {node_id}")
        
        # Check for circular references
        for node_id, node in self.nodes.items():
            visited = {node_id}
            current = node.parent_id
            while current:
                if current in visited:
                    issues.append(f"Circular reference detected involving node: {node_id}")
                    break
                if current not in self.nodes:
                    issues.append(f"Node {node_id} has non-existent parent: {current}")
                    break
                visited.add(current)
                current = self.nodes[current].parent_id
        
        # Check depth consistency
        for node_id, node in self.nodes.items():
            if node.parent_id:
                if node.parent_id in self.nodes:
                    parent = self.nodes[node.parent_id]
                    if node.depth != parent.depth + 1:
                        issues.append(f"Node {node_id} has inconsistent depth (depth: {node.depth}, parent depth: {parent.depth})")
        
        return issues