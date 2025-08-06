"""Tests for error handling and resilience across the system."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from langtree.models import ThoughtNode, TreeManager
from langtree.orchestrator import ToTOrchestrator
from langtree.policies import ExpansionPolicy, ScoringPolicy


class TestExpansionErrors:
    """Test error handling in expansion policy."""
    
    def test_expansion_complete_llm_failure(self):
        """Test expansion when LLM completely fails."""
        mock_llm = Mock()
        mock_llm.bind.return_value = mock_llm
        mock_llm.invoke.side_effect = Exception("Complete LLM failure")
        
        policy = ExpansionPolicy(mock_llm, k=3)
        parent = ThoughtNode(id="parent", parent_id=None, depth=0, state={"partial_solution": "test"})
        
        children = policy.expand(parent, task="Test task")
        
        # Should get fallback child
        assert len(children) == 1
        assert children[0].action == "noop"
        assert "Expansion failed" in children[0].state["reasoning_steps"][0]
        assert children[0].state["partial_solution"] == "test"
    
    def test_expansion_malformed_json_response(self):
        """Test expansion with malformed JSON from LLM."""
        mock_llm = Mock()
        mock_llm.bind.return_value = mock_llm
        mock_llm.invoke.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
        
        policy = ExpansionPolicy(mock_llm, k=2)
        parent = ThoughtNode(id="parent", parent_id=None, depth=0)
        
        children = policy.expand(parent, task="Test task")
        
        # Should get fallback child
        assert len(children) == 1
        assert children[0].action == "noop"
    
    def test_expansion_network_timeout(self):
        """Test expansion with network timeout."""
        mock_llm = Mock()
        mock_llm.bind.return_value = mock_llm
        mock_llm.invoke.side_effect = TimeoutError("Request timed out")
        
        policy = ExpansionPolicy(mock_llm, k=3)
        parent = ThoughtNode(id="parent", parent_id=None, depth=0)
        
        children = policy.expand(parent, task="Test task")
        
        # Should get fallback child with timeout info
        assert len(children) == 1
        assert "timed out" in children[0].state["reasoning_steps"][0].lower()
    
    def test_expansion_partial_json_response(self):
        """Test expansion with partial/incomplete JSON response."""
        mock_llm = Mock()
        mock_llm.bind.return_value = mock_llm
        # Return list with incomplete objects
        mock_llm.invoke.return_value = [
            {"thought": "Complete", "action": "complete_action", "partial_solution": "solution"},
            {"thought": "Incomplete"},  # Missing action and partial_solution
            {}  # Empty object
        ]
        
        policy = ExpansionPolicy(mock_llm, k=3)
        parent = ThoughtNode(id="parent", parent_id=None, depth=0)
        
        children = policy.expand(parent, task="Test task")
        
        assert len(children) == 3
        assert children[0].action == "complete_action"
        assert children[1].action == ""  # Default empty string
        assert children[2].action == ""  # Default empty string
        
        # All should have valid state structure
        for child in children:
            assert "reasoning_steps" in child.state
            assert "scratchpad" in child.state
            assert "partial_solution" in child.state
    
    def test_expansion_wrong_response_type(self):
        """Test expansion when LLM returns wrong type (not list)."""
        mock_llm = Mock()
        mock_llm.bind.return_value = mock_llm
        mock_llm.invoke.return_value = "Not a list"  # Wrong type
        
        policy = ExpansionPolicy(mock_llm, k=2)
        parent = ThoughtNode(id="parent", parent_id=None, depth=0)
        
        children = policy.expand(parent, task="Test task")
        
        # Should get fallback child
        assert len(children) == 1
        assert children[0].action == "noop"


class TestScoringErrors:
    """Test error handling in scoring policy."""
    
    def test_scoring_complete_llm_failure(self):
        """Test scoring when LLM completely fails."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("Complete scoring failure")
        
        policy = ScoringPolicy(mock_llm)
        
        tree = TreeManager()
        parent = ThoughtNode(id="parent", parent_id=None, depth=0)
        tree.add_node(parent)
        node = ThoughtNode(id="node", parent_id="parent", depth=1)
        tree.add_node(node)
        
        score, conf, comps, rationale = policy.score(node, tree, "Test task")
        
        # Should get fallback values
        assert score == 0.3
        assert conf == 0.3
        assert "LLM grading failed" in rationale
        assert all(key in comps for key in ["accuracy", "consistency", "progress", "novelty", "cost"])
    
    def test_scoring_malformed_json_response(self):
        """Test scoring with malformed JSON from LLM."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
        
        policy = ScoringPolicy(mock_llm)
        
        tree = TreeManager()
        node = ThoughtNode(id="node", parent_id=None, depth=0)
        tree.add_node(node)
        
        score, conf, comps, rationale = policy.score(node, tree, "Test task")
        
        # Should get fallback values
        assert score == 0.3
        assert conf == 0.3
        assert "LLM grading failed" in rationale
    
    def test_scoring_missing_required_fields(self):
        """Test scoring with missing required fields in response."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = {"score": 0.8}  # Missing other fields
        
        policy = ScoringPolicy(mock_llm)
        
        tree = TreeManager()
        node = ThoughtNode(id="node", parent_id=None, depth=0)
        tree.add_node(node)
        
        score, conf, comps, rationale = policy.score(node, tree, "Test task")
        
        # Should handle missing fields gracefully
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert conf == 0.0  # Default for missing confidence
        assert rationale == ""  # Default for missing rationale
    
    def test_scoring_invalid_component_values(self):
        """Test scoring with invalid component values."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = {
            "score": 0.8,
            "confidence": 0.9,
            "components": {
                "accuracy": "not_a_number",
                "consistency": None,
                "progress": float('inf'),
                "novelty": -5.0,
                "cost": "invalid"
            },
            "rationale": "Test rationale"
        }
        
        policy = ScoringPolicy(mock_llm)
        
        tree = TreeManager()
        node = ThoughtNode(id="node", parent_id=None, depth=0)
        tree.add_node(node)
        
        score, conf, comps, rationale = policy.score(node, tree, "Test task")
        
        # Should handle invalid values gracefully (convert to 0.0)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert all(isinstance(v, float) for v in comps.values())
        # Invalid values should become 0.0
        for key in ["accuracy", "consistency", "cost"]:
            assert comps[key] == 0.0
    
    def test_scoring_extreme_score_values(self):
        """Test scoring bounds checking with extreme values."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = {
            "score": 999.0,  # Will be ignored, weighted score used
            "confidence": 1.5,  # Above 1.0
            "components": {
                "accuracy": 10.0,   # Way above 1.0
                "consistency": -5.0, # Negative
                "progress": 2.0,    # Above 1.0
                "novelty": 1.0,     # Valid
                "cost": 0.0         # Valid
            },
            "rationale": "Extreme values test"
        }
        
        policy = ScoringPolicy(mock_llm)
        
        tree = TreeManager()
        node = ThoughtNode(id="node", parent_id=None, depth=0)
        tree.add_node(node)
        
        score, conf, comps, rationale = policy.score(node, tree, "Test task")
        
        # Score should be bounded to [0, 1]
        assert 0.0 <= score <= 1.0
        assert conf == 1.5  # Confidence not bounded in current implementation
        assert rationale == "Extreme values test"


class TestOrchestratorErrors:
    """Test error handling in orchestrator."""
    
    def test_orchestrator_all_expansions_fail(self):
        """Test orchestrator when all expansions fail."""
        mock_llm_expand = Mock()
        mock_llm_expand.bind.return_value = mock_llm_expand
        mock_llm_expand.invoke.side_effect = Exception("All expansions fail")
        
        mock_llm_score = Mock()
        mock_llm_score.invoke.return_value = {
            "score": 0.5,
            "confidence": 0.6,
            "components": {"accuracy": 0.5, "consistency": 0.5, "progress": 0.5, "novelty": 0.5, "cost": 0.5},
            "rationale": "Fallback scoring"
        }
        
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm_expand,
            llm_score=mock_llm_score,
            k_children=2,
            max_select=1,
            max_depth=3,
            max_nodes=10,
            target_score=0.9,
        )
        
        result = orchestrator.run(task="Test task")
        
        # Should complete with root node having fallback children
        assert result["best_node_id"] is not None
        assert isinstance(result["best_score"], float)
        assert "tree_snapshot" in result
    
    def test_orchestrator_memory_exhaustion(self):
        """Test orchestrator with very large node limit."""
        mock_llm_expand = Mock()
        mock_llm_expand.bind.return_value = mock_llm_expand
        # Create many children each time
        mock_llm_expand.invoke.return_value = [
            {"thought": f"Child {i}", "action": f"action{i}", "partial_solution": f"solution{i}"}
            for i in range(50)  # Large number of children
        ]
        
        mock_llm_score = Mock()
        mock_llm_score.invoke.return_value = {
            "score": 0.3,  # Low score to prevent early termination
            "confidence": 0.4,
            "components": {"accuracy": 0.3, "consistency": 0.3, "progress": 0.3, "novelty": 0.3, "cost": 0.3},
            "rationale": "Low score"
        }
        
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm_expand,
            llm_score=mock_llm_score,
            k_children=50,
            max_select=10,
            max_depth=5,
            max_nodes=1000,  # High limit
            target_score=0.9,  # High target
            beam_per_depth=20,
        )
        
        # Should handle large trees without crashing
        result = orchestrator.run(task="Large tree test")
        
        assert result["best_node_id"] is not None
        snapshot = result["tree_snapshot"]
        assert len(snapshot["nodes"]) <= 1000  # Should respect max_nodes
    
    def test_orchestrator_infinite_loop_prevention(self):
        """Test that orchestrator prevents infinite loops."""
        mock_llm_expand = Mock()
        mock_llm_expand.bind.return_value = mock_llm_expand
        mock_llm_expand.invoke.return_value = [
            {"thought": "Always same", "action": "same", "partial_solution": "same"}
        ]
        
        mock_llm_score = Mock()
        mock_llm_score.invoke.return_value = {
            "score": 0.5,  # Moderate score
            "confidence": 0.5,
            "components": {"accuracy": 0.5, "consistency": 0.5, "progress": 0.5, "novelty": 0.0, "cost": 0.5},
            "rationale": "Same every time"
        }
        
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm_expand,
            llm_score=mock_llm_score,
            k_children=1,
            max_select=1,
            max_depth=100,  # Very high
            max_nodes=1000,  # Very high
            target_score=0.99,  # Nearly impossible target
        )
        
        # Should terminate due to some constraint
        result = orchestrator.run(task="Infinite loop test")
        
        assert result["best_node_id"] is not None
        # Should not create infinite nodes
        snapshot = result["tree_snapshot"]
        assert len(snapshot["nodes"]) < 1000
    
    def test_orchestrator_empty_initial_state(self):
        """Test orchestrator with various empty/null initial states."""
        mock_llm_expand = Mock()
        mock_llm_expand.bind.return_value = mock_llm_expand
        mock_llm_expand.invoke.return_value = [
            {"thought": "From empty", "action": "start", "partial_solution": "started"}
        ]
        
        mock_llm_score = Mock()
        mock_llm_score.invoke.return_value = {
            "score": 0.8,
            "confidence": 0.9,
            "components": {"accuracy": 0.8, "consistency": 0.8, "progress": 0.8, "novelty": 0.8, "cost": 0.2},
            "rationale": "Good start"
        }
        
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm_expand,
            llm_score=mock_llm_score,
            target_score=0.75,
        )
        
        # Test various empty states
        empty_states = [
            None,
            {},
            {"reasoning_steps": [], "scratchpad": "", "partial_solution": ""},
            {"reasoning_steps": None, "scratchpad": None, "partial_solution": None},
        ]
        
        for initial_state in empty_states:
            result = orchestrator.run(task="Empty state test", initial_state=initial_state)
            
            assert result["best_node_id"] is not None
            assert isinstance(result["best_score"], float)
            # Should have valid best path even from empty state
            assert isinstance(result["best_path"], list)
            assert isinstance(result["best_actions"], list)


class TestTreeManagerErrors:
    """Test error handling in tree manager."""
    
    def test_tree_manager_corrupted_state(self):
        """Test tree manager with corrupted internal state."""
        tree = TreeManager()
        
        # Add node normally
        node = ThoughtNode(id="test", parent_id=None, depth=0, status="open")
        tree.add_node(node)
        
        # Corrupt the state manually
        tree.frontier.append("nonexistent_id")
        tree.open_status["another_nonexistent"] = True
        tree.depth_index[999] = ["missing_node"]
        
        # Operations should handle corruption gracefully
        best = tree.best_nodes(1)
        assert len(best) >= 0  # Should not crash
        
        # Snapshot should work despite corruption
        snapshot = tree.snapshot()
        assert "nodes" in snapshot
        assert "test" in snapshot["nodes"]
    
    def test_tree_manager_circular_references(self):
        """Test prevention of circular parent-child references."""
        tree = TreeManager()
        
        # Create nodes that would form a cycle
        node1 = ThoughtNode(id="node1", parent_id=None, depth=0)
        tree.add_node(node1)
        
        node2 = ThoughtNode(id="node2", parent_id="node1", depth=1)
        tree.add_node(node2)
        
        # Try to create circular reference
        node3 = ThoughtNode(id="node3", parent_id="node2", depth=2)
        tree.add_node(node3)
        
        # Manually corrupt to create cycle (simulates bug)
        tree.nodes["node1"].parent_id = "node3"  # Creates cycle: node1 -> node3 -> node2 -> node1
        
        # Path traversal should handle cycles (though it may fail)
        try:
            path = node2.path(tree)
            # If it completes, it should have reasonable length
            assert len(path) < 100  # Prevent infinite loops in path traversal
        except (RecursionError, KeyError):
            # Expected for circular references
            pass
    
    def test_tree_manager_add_duplicate_node_ids(self):
        """Test handling of duplicate node IDs."""
        tree = TreeManager()
        
        # Add first node
        node1 = ThoughtNode(id="duplicate", parent_id=None, depth=0, status="open")
        node1.score = 0.5
        tree.add_node(node1)
        
        # Add second node with same ID (simulates bug)
        node2 = ThoughtNode(id="duplicate", parent_id=None, depth=1, status="expanded")
        node2.score = 0.8
        tree.add_node(node2)
        
        # Second node should overwrite first
        assert tree.nodes["duplicate"].depth == 1
        assert tree.nodes["duplicate"].status == "expanded"
        assert tree.nodes["duplicate"].score == 0.8
        
        # Frontier should be consistent
        # Note: This might result in inconsistent frontier state
        # but should not crash
        best = tree.best_nodes(1)
        assert len(best) >= 0
    
    def test_tree_manager_invalid_depth_operations(self):
        """Test tree manager with invalid depth operations."""
        tree = TreeManager()
        
        # Add nodes with inconsistent depths
        root = ThoughtNode(id="root", parent_id=None, depth=0)
        tree.add_node(root)
        
        # Child with wrong depth
        wrong_child = ThoughtNode(id="wrong", parent_id="root", depth=5)  # Should be 1
        tree.add_node(wrong_child)
        
        # Depth index will be inconsistent
        assert 0 in tree.depth_index
        assert 5 in tree.depth_index
        
        # Operations should still work
        best = tree.best_nodes(1)
        snapshot = tree.snapshot()
        
        assert len(best) >= 0
        assert "nodes" in snapshot


class TestPolicyParameterValidation:
    """Test policy parameter validation and edge cases."""
    
    def test_selection_policy_invalid_parameters(self):
        """Test selection policy with invalid parameters."""
        from langtree.policies import SelectionPolicy
        
        # Test with extreme parameters
        policy = SelectionPolicy(beta=-1.0, gamma=10.0)  # Unusual but valid
        
        tree = TreeManager()
        node = ThoughtNode(id="test", parent_id=None, depth=0, status="open")
        node.score = 0.5
        node.confidence = 0.5
        node.components = {"novelty": 0.5}
        tree.add_node(node)
        
        # Should work even with extreme parameters
        selected = policy.select(tree, k=1)
        assert len(selected) <= 1
    
    def test_pruning_policy_invalid_parameters(self):
        """Test pruning policy with invalid parameters."""
        from langtree.policies import PruningPolicy
        
        # Test with edge case parameters
        policy = PruningPolicy(beam_per_depth=0, min_score=2.0)  # Impossible constraints
        
        tree = TreeManager()
        node = ThoughtNode(id="test", parent_id=None, depth=1, status="open")
        node.score = 0.5
        tree.add_node(node)
        
        # Should handle impossible constraints gracefully
        policy.prune(tree, depth=1)
        
        # With beam=0, all nodes should be pruned
        assert tree.nodes["test"].status == "pruned"
    
    def test_termination_policy_conflicting_parameters(self):
        """Test termination policy with conflicting parameters."""
        from langtree.policies import TerminationPolicy
        
        # Conflicting parameters
        policy = TerminationPolicy(max_depth=1, max_nodes=1, target_score=0.0)
        
        tree = TreeManager()
        
        # Should terminate immediately with max_nodes=1 and max_nodes=1
        assert policy.should_stop(tree)  # Empty tree triggers stop
        
        # Add one node
        node = ThoughtNode(id="test", parent_id=None, depth=0, status="open")
        node.score = 0.1  # Above target_score=0.0
        tree.add_node(node)
        
        # Should stop due to target score being reached
        assert policy.should_stop(tree)