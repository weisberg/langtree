"""Tests for policy modules."""

import pytest
from unittest.mock import Mock

from langtree.models import ThoughtNode, TreeManager
from langtree.policies import SelectionPolicy, PruningPolicy, TerminationPolicy


class TestSelectionPolicy:
    """Test SelectionPolicy functionality."""
    
    def test_select_basic(self):
        """Test basic selection by score."""
        tree = TreeManager()
        policy = SelectionPolicy(beta=0.0, gamma=0.0)  # Only score matters
        
        # Create nodes with different scores
        node1 = ThoughtNode(id="1", parent_id=None, depth=0, status="open")
        node1.score = 0.3
        node1.confidence = 0.8
        tree.add_node(node1)
        
        node2 = ThoughtNode(id="2", parent_id=None, depth=0, status="open")
        node2.score = 0.7
        node2.confidence = 0.5
        tree.add_node(node2)
        
        node3 = ThoughtNode(id="3", parent_id=None, depth=0, status="open")
        node3.score = 0.5
        node3.confidence = 0.9
        tree.add_node(node3)
        
        selected = policy.select(tree, k=2)
        
        assert len(selected) == 2
        assert selected[0].id == "2"  # Highest score
        assert selected[1].id == "3"  # Second highest

    def test_select_with_uncertainty_bonus(self):
        """Test selection with uncertainty (confidence) bonus."""
        tree = TreeManager()
        policy = SelectionPolicy(beta=0.5, gamma=0.0)  # Strong uncertainty bonus
        
        # High score, high confidence vs lower score, low confidence
        node1 = ThoughtNode(id="1", parent_id=None, depth=0, status="open")
        node1.score = 0.8
        node1.confidence = 0.9  # High confidence
        tree.add_node(node1)
        
        node2 = ThoughtNode(id="2", parent_id=None, depth=0, status="open")
        node2.score = 0.6
        node2.confidence = 0.1  # Low confidence -> high uncertainty bonus
        tree.add_node(node2)
        
        selected = policy.select(tree, k=1)
        
        # Priority of node1: 0.8 + 0.5 * (1 - 0.9) = 0.8 + 0.05 = 0.85
        # Priority of node2: 0.6 + 0.5 * (1 - 0.1) = 0.6 + 0.45 = 1.05
        assert selected[0].id == "2"  # Lower score but higher uncertainty bonus

    def test_select_with_novelty_bonus(self):
        """Test selection with novelty bonus."""
        tree = TreeManager()
        policy = SelectionPolicy(beta=0.0, gamma=0.5)  # Strong novelty bonus
        
        node1 = ThoughtNode(id="1", parent_id=None, depth=0, status="open")
        node1.score = 0.7
        node1.confidence = 0.8
        node1.components = {"novelty": 0.2}
        tree.add_node(node1)
        
        node2 = ThoughtNode(id="2", parent_id=None, depth=0, status="open")
        node2.score = 0.6
        node2.confidence = 0.8
        node2.components = {"novelty": 0.8}  # High novelty
        tree.add_node(node2)
        
        selected = policy.select(tree, k=1)
        
        # Priority of node1: 0.7 + 0.0 + 0.5 * 0.2 = 0.8
        # Priority of node2: 0.6 + 0.0 + 0.5 * 0.8 = 1.0
        assert selected[0].id == "2"  # Higher novelty bonus

    def test_select_filters_open_nodes(self):
        """Test that only open nodes are selected."""
        tree = TreeManager()
        policy = SelectionPolicy()
        
        # Add nodes with different statuses
        open_node = ThoughtNode(id="open", parent_id=None, depth=0, status="open")
        open_node.score = 0.5
        tree.add_node(open_node)
        
        # This node is in frontier but not open (shouldn't happen in practice)
        expanded_node = ThoughtNode(id="expanded", parent_id=None, depth=0, status="expanded")
        expanded_node.score = 0.9
        tree.nodes["expanded"] = expanded_node  # Add directly to avoid frontier management
        tree.frontier.append("expanded")  # Force into frontier
        
        selected = policy.select(tree, k=2)
        
        assert len(selected) == 1
        assert selected[0].id == "open"

    def test_select_respects_k_limit(self):
        """Test that selection respects the k parameter."""
        tree = TreeManager()
        policy = SelectionPolicy()
        
        # Add 5 open nodes
        for i in range(5):
            node = ThoughtNode(id=str(i), parent_id=None, depth=0, status="open")
            node.score = i * 0.1  # Different scores
            tree.add_node(node)
        
        selected = policy.select(tree, k=3)
        
        assert len(selected) == 3
        # Should get top 3 by score
        assert selected[0].id == "4"  # 0.4 score
        assert selected[1].id == "3"  # 0.3 score
        assert selected[2].id == "2"  # 0.2 score


class TestPruningPolicy:
    """Test PruningPolicy functionality."""
    
    def test_prune_by_beam(self):
        """Test pruning by beam width."""
        tree = TreeManager()
        policy = PruningPolicy(beam_per_depth=2, min_score=0.0)
        
        # Add 4 nodes at depth 1 with different scores
        nodes = []
        for i, score in enumerate([0.8, 0.6, 0.4, 0.2]):
            node = ThoughtNode(id=str(i), parent_id=None, depth=1, status="open")
            node.score = score
            tree.add_node(node)
            nodes.append(node)
        
        policy.prune(tree, depth=1)
        
        # Top 2 should remain open, bottom 2 should be pruned
        assert tree.nodes["0"].status == "open"   # 0.8 score
        assert tree.nodes["1"].status == "open"   # 0.6 score
        assert tree.nodes["2"].status == "pruned" # 0.4 score
        assert tree.nodes["3"].status == "pruned" # 0.2 score
        
        # Pruned nodes should be removed from frontier
        assert "0" in tree.frontier
        assert "1" in tree.frontier
        assert "2" not in tree.frontier
        assert "3" not in tree.frontier

    def test_prune_by_min_score(self):
        """Test pruning by minimum score threshold."""
        tree = TreeManager()
        policy = PruningPolicy(beam_per_depth=10, min_score=0.5)  # Large beam, focus on threshold
        
        # Add nodes with scores above and below threshold
        for i, score in enumerate([0.8, 0.6, 0.4, 0.3]):
            node = ThoughtNode(id=str(i), parent_id=None, depth=1, status="open")
            node.score = score
            tree.add_node(node)
        
        policy.prune(tree, depth=1)
        
        # Only nodes >= 0.5 should remain
        assert tree.nodes["0"].status == "open"   # 0.8 >= 0.5
        assert tree.nodes["1"].status == "open"   # 0.6 >= 0.5
        assert tree.nodes["2"].status == "pruned" # 0.4 < 0.5
        assert tree.nodes["3"].status == "pruned" # 0.3 < 0.5

    def test_prune_combined_beam_and_threshold(self):
        """Test pruning with both beam and threshold."""
        tree = TreeManager()
        policy = PruningPolicy(beam_per_depth=2, min_score=0.3)
        
        # Add nodes: some below threshold, some above
        scores_and_expected = [
            (0.8, "open"),    # Above threshold, top 2
            (0.6, "open"),    # Above threshold, top 2
            (0.4, "pruned"),  # Above threshold but outside beam
            (0.2, "pruned"),  # Below threshold
        ]
        
        for i, (score, expected_status) in enumerate(scores_and_expected):
            node = ThoughtNode(id=str(i), parent_id=None, depth=1, status="open")
            node.score = score
            tree.add_node(node)
        
        policy.prune(tree, depth=1)
        
        for i, (score, expected_status) in enumerate(scores_and_expected):
            assert tree.nodes[str(i)].status == expected_status

    def test_prune_only_affects_specified_depth(self):
        """Test that pruning only affects the specified depth."""
        tree = TreeManager()
        policy = PruningPolicy(beam_per_depth=1, min_score=0.0)
        
        # Add nodes at different depths
        depth0_node = ThoughtNode(id="d0", parent_id=None, depth=0, status="open")
        depth0_node.score = 0.2  # Would be pruned if at target depth
        tree.add_node(depth0_node)
        
        depth1_node1 = ThoughtNode(id="d1_1", parent_id=None, depth=1, status="open")
        depth1_node1.score = 0.8
        tree.add_node(depth1_node1)
        
        depth1_node2 = ThoughtNode(id="d1_2", parent_id=None, depth=1, status="open")
        depth1_node2.score = 0.4  # Should be pruned
        tree.add_node(depth1_node2)
        
        policy.prune(tree, depth=1)
        
        # Depth 0 node should be unchanged
        assert tree.nodes["d0"].status == "open"
        
        # Depth 1 nodes should be pruned according to policy
        assert tree.nodes["d1_1"].status == "open"
        assert tree.nodes["d1_2"].status == "pruned"

    def test_prune_handles_empty_depth(self):
        """Test pruning handles non-existent depth gracefully."""
        tree = TreeManager()
        policy = PruningPolicy(beam_per_depth=2, min_score=0.0)
        
        # No nodes at depth 5
        policy.prune(tree, depth=5)  # Should not raise exception


class TestTerminationPolicy:
    """Test TerminationPolicy functionality."""
    
    def test_should_stop_max_nodes(self):
        """Test termination by maximum nodes."""
        tree = TreeManager()
        policy = TerminationPolicy(max_depth=10, max_nodes=3, target_score=1.0)
        
        # Add nodes up to limit
        for i in range(2):
            node = ThoughtNode(id=str(i), parent_id=None, depth=0)
            tree.add_node(node)
        
        assert not policy.should_stop(tree)  # 2 < 3
        
        # Add one more to reach limit
        node = ThoughtNode(id="2", parent_id=None, depth=0)
        tree.add_node(node)
        
        assert policy.should_stop(tree)  # 3 >= 3

    def test_should_stop_target_score(self):
        """Test termination by target score."""
        tree = TreeManager()
        policy = TerminationPolicy(max_depth=10, max_nodes=100, target_score=0.8)
        
        # Add node with score below target
        node1 = ThoughtNode(id="1", parent_id=None, depth=0, status="expanded")
        node1.score = 0.7
        tree.add_node(node1)
        
        assert not policy.should_stop(tree)  # 0.7 < 0.8
        
        # Add node with score at target
        node2 = ThoughtNode(id="2", parent_id=None, depth=0, status="expanded")
        node2.score = 0.8
        tree.add_node(node2)
        
        assert policy.should_stop(tree)  # 0.8 >= 0.8

    def test_should_stop_no_open_nodes(self):
        """Test termination when no open nodes remain."""
        tree = TreeManager()
        policy = TerminationPolicy(max_depth=10, max_nodes=100, target_score=1.0)
        
        # Add nodes that are not open
        node1 = ThoughtNode(id="1", parent_id=None, depth=0, status="expanded")
        tree.add_node(node1)
        
        node2 = ThoughtNode(id="2", parent_id=None, depth=0, status="pruned")
        tree.add_node(node2)
        
        # No open nodes in frontier
        assert policy.should_stop(tree)

    def test_should_stop_max_depth_exceeded(self):
        """Test termination when all open nodes exceed max depth."""
        tree = TreeManager()
        policy = TerminationPolicy(max_depth=2, max_nodes=100, target_score=1.0)
        
        # Add open node at max depth
        node1 = ThoughtNode(id="1", parent_id=None, depth=2, status="open")
        tree.add_node(node1)
        
        # Add open node beyond max depth
        node2 = ThoughtNode(id="2", parent_id=None, depth=3, status="open")
        tree.add_node(node2)
        
        assert policy.should_stop(tree)  # All open nodes >= max_depth

    def test_should_not_stop_has_valid_open_nodes(self):
        """Test continuation when valid open nodes exist."""
        tree = TreeManager()
        policy = TerminationPolicy(max_depth=3, max_nodes=100, target_score=1.0)
        
        # Add open node within depth limit
        node1 = ThoughtNode(id="1", parent_id=None, depth=1, status="open")
        node1.score = 0.5  # Below target
        tree.add_node(node1)
        
        # Add node at max depth (should still allow continuation of node1)
        node2 = ThoughtNode(id="2", parent_id=None, depth=3, status="open")
        tree.add_node(node2)
        
        assert not policy.should_stop(tree)  # node1 is still expandable

    def test_should_stop_empty_tree(self):
        """Test termination with empty tree."""
        tree = TreeManager()
        policy = TerminationPolicy(max_depth=10, max_nodes=100, target_score=1.0)
        
        # Empty tree should stop (no nodes to work with)
        assert policy.should_stop(tree)

    def test_should_stop_priority_order(self):
        """Test that termination conditions are checked in correct priority."""
        tree = TreeManager()
        policy = TerminationPolicy(max_depth=5, max_nodes=2, target_score=0.9)
        
        # Add nodes that exceed max_nodes but have high scores
        node1 = ThoughtNode(id="1", parent_id=None, depth=0, status="expanded")
        node1.score = 0.95  # Above target
        tree.add_node(node1)
        
        node2 = ThoughtNode(id="2", parent_id=None, depth=0, status="expanded")
        node2.score = 0.85  # Below target  
        tree.add_node(node2)
        
        node3 = ThoughtNode(id="3", parent_id=None, depth=0, status="open")
        node3.score = 0.7
        tree.add_node(node3)  # Now at max_nodes + 1
        
        # Should stop due to max_nodes even though target score is met
        assert policy.should_stop(tree)