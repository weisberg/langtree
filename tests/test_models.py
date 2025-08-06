"""Tests for models module."""

import pytest
from langtree.models import ThoughtNode, TreeManager


class TestThoughtNode:
    """Test ThoughtNode functionality."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        node = ThoughtNode(
            id="test-1",
            parent_id=None,
            depth=0,
            state={"test": "data"},
            action="test_action",
        )
        
        assert node.id == "test-1"
        assert node.parent_id is None
        assert node.depth == 0
        assert node.state == {"test": "data"}
        assert node.action == "test_action"
        assert node.status == "open"
        assert node.score == 0.0
        assert node.confidence == 0.0
        assert node.components == {}
        assert node.metadata == {}
        assert node.children == []

    def test_node_path(self):
        """Test path reconstruction."""
        tree = TreeManager()
        
        # Create root
        root = ThoughtNode(id="root", parent_id=None, depth=0)
        tree.add_node(root)
        
        # Create child
        child = ThoughtNode(id="child", parent_id="root", depth=1)
        tree.add_node(child)
        
        # Create grandchild
        grandchild = ThoughtNode(id="grandchild", parent_id="child", depth=2)
        tree.add_node(grandchild)
        
        path = grandchild.path(tree)
        assert len(path) == 3
        assert path[0].id == "root"
        assert path[1].id == "child"
        assert path[2].id == "grandchild"


class TestTreeManager:
    """Test TreeManager functionality."""
    
    def test_add_node(self):
        """Test adding nodes to the tree."""
        tree = TreeManager()
        node = ThoughtNode(id="test-1", parent_id=None, depth=0, status="open")
        
        tree.add_node(node)
        
        assert "test-1" in tree.nodes
        assert tree.nodes["test-1"] == node
        assert "test-1" in tree.frontier
        assert tree.open_status["test-1"] is True
        assert 0 in tree.depth_index
        assert "test-1" in tree.depth_index[0]

    def test_add_non_open_node(self):
        """Test adding non-open nodes."""
        tree = TreeManager()
        node = ThoughtNode(id="test-1", parent_id=None, depth=0, status="expanded")
        
        tree.add_node(node)
        
        assert "test-1" in tree.nodes
        assert "test-1" not in tree.frontier
        assert "test-1" not in tree.open_status

    def test_set_expanded(self):
        """Test marking nodes as expanded."""
        tree = TreeManager()
        node = ThoughtNode(id="test-1", parent_id=None, depth=0, status="open")
        tree.add_node(node)
        
        # Verify initially open
        assert "test-1" in tree.frontier
        assert tree.nodes["test-1"].status == "open"
        
        tree.set_expanded("test-1")
        
        # Verify now expanded
        assert "test-1" not in tree.frontier
        assert "test-1" not in tree.open_status
        assert tree.nodes["test-1"].status == "expanded"

    def test_add_children(self):
        """Test adding children to a node."""
        tree = TreeManager()
        parent = ThoughtNode(id="parent", parent_id=None, depth=0)
        tree.add_node(parent)
        
        child1 = ThoughtNode(id="child1", parent_id="parent", depth=1)
        child2 = ThoughtNode(id="child2", parent_id="parent", depth=1)
        children = [child1, child2]
        
        tree.add_children("parent", children)
        
        # Check parent has children
        assert tree.nodes["parent"].children == ["child1", "child2"]
        
        # Check children are in tree
        assert "child1" in tree.nodes
        assert "child2" in tree.nodes
        assert tree.nodes["child1"].parent_id == "parent"
        assert tree.nodes["child2"].parent_id == "parent"

    def test_best_nodes(self):
        """Test getting best nodes by score."""
        tree = TreeManager()
        
        # Create nodes with different scores
        node1 = ThoughtNode(id="1", parent_id=None, depth=0, status="expanded")
        node1.score = 0.5
        tree.add_node(node1)
        
        node2 = ThoughtNode(id="2", parent_id=None, depth=0, status="expanded")
        node2.score = 0.8
        tree.add_node(node2)
        
        node3 = ThoughtNode(id="3", parent_id=None, depth=0, status="expanded")
        node3.score = 0.3
        tree.add_node(node3)
        
        # Test getting best nodes
        best = tree.best_nodes(2)
        assert len(best) == 2
        assert best[0].id == "2"  # Highest score
        assert best[1].id == "1"  # Second highest
        
        # Test single best
        best_one = tree.best_nodes(1)
        assert len(best_one) == 1
        assert best_one[0].id == "2"

    def test_best_nodes_filters_status(self):
        """Test that best_nodes only includes valid statuses."""
        tree = TreeManager()
        
        # Create nodes with different statuses
        open_node = ThoughtNode(id="open", parent_id=None, depth=0, status="open")
        open_node.score = 0.8
        tree.add_node(open_node)
        
        pruned_node = ThoughtNode(id="pruned", parent_id=None, depth=0, status="pruned")
        pruned_node.score = 0.9  # Higher score but pruned
        tree.add_node(pruned_node)
        
        best = tree.best_nodes(1)
        assert len(best) == 1
        assert best[0].id == "open"  # Pruned node excluded

    def test_snapshot(self):
        """Test tree snapshot functionality."""
        tree = TreeManager()
        
        # Create a simple tree
        root = ThoughtNode(id="root", parent_id=None, depth=0)
        root.score = 0.5
        root.confidence = 0.8
        root.components = {"accuracy": 0.7}
        root.state = {"test": "data"}
        tree.add_node(root)
        
        child = ThoughtNode(id="child", parent_id="root", depth=1)
        tree.add_children("root", [child])
        
        snapshot = tree.snapshot()
        
        assert "nodes" in snapshot
        assert "root" in snapshot["nodes"]
        assert "child" in snapshot["nodes"]
        
        root_data = snapshot["nodes"]["root"]
        assert root_data["parent_id"] is None
        assert root_data["depth"] == 0
        assert root_data["score"] == 0.5
        assert root_data["confidence"] == 0.8
        assert root_data["components"] == {"accuracy": 0.7}
        assert root_data["state"] == {"test": "data"}
        assert root_data["children"] == ["child"]
        
        child_data = snapshot["nodes"]["child"]
        assert child_data["parent_id"] == "root"
        assert child_data["depth"] == 1

    def test_depth_index(self):
        """Test depth indexing functionality."""
        tree = TreeManager()
        
        # Add nodes at different depths
        root = ThoughtNode(id="root", parent_id=None, depth=0)
        tree.add_node(root)
        
        child1 = ThoughtNode(id="child1", parent_id="root", depth=1)
        child2 = ThoughtNode(id="child2", parent_id="root", depth=1)
        tree.add_node(child1)
        tree.add_node(child2)
        
        grandchild = ThoughtNode(id="grandchild", parent_id="child1", depth=2)
        tree.add_node(grandchild)
        
        # Check depth indexing
        assert 0 in tree.depth_index
        assert 1 in tree.depth_index
        assert 2 in tree.depth_index
        
        assert tree.depth_index[0] == ["root"]
        assert set(tree.depth_index[1]) == {"child1", "child2"}
        assert tree.depth_index[2] == ["grandchild"]

    def test_frontier_management(self):
        """Test frontier node management."""
        tree = TreeManager()
        
        # Add open nodes
        node1 = ThoughtNode(id="1", parent_id=None, depth=0, status="open")
        node2 = ThoughtNode(id="2", parent_id=None, depth=0, status="open")
        tree.add_node(node1)
        tree.add_node(node2)
        
        assert len(tree.frontier) == 2
        assert set(tree.frontier) == {"1", "2"}
        
        # Expand one node
        tree.set_expanded("1")
        
        assert len(tree.frontier) == 1
        assert tree.frontier == ["2"]
        
        # Add non-open node
        node3 = ThoughtNode(id="3", parent_id=None, depth=0, status="expanded")
        tree.add_node(node3)
        
        assert len(tree.frontier) == 1  # Unchanged
        assert tree.frontier == ["2"]