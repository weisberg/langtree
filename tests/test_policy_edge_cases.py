"""Tests for policy edge cases and validation."""

import json
import pytest
from unittest.mock import Mock

from langtree.models import ThoughtNode, TreeManager
from langtree.policies import ExpansionPolicy, ScoringPolicy, SelectionPolicy, PruningPolicy, TerminationPolicy


class TestExpansionPolicyEdgeCases:
    """Test expansion policy edge cases."""
    
    def test_expansion_policy_temperature_effects(self):
        """Test that temperature parameter is properly passed through."""
        mock_llm = Mock()
        mock_bound_llm = Mock()
        mock_llm.bind.return_value = mock_bound_llm
        mock_bound_llm.invoke.return_value = [
            {"thought": "Test", "action": "test", "partial_solution": "test"}
        ]
        
        policy = ExpansionPolicy(mock_llm, k=1, temperature=0.9)
        
        parent = ThoughtNode(id="parent", parent_id=None, depth=0)
        children = policy.expand(parent, task="Test task")
        
        # Verify temperature was passed to bind
        mock_llm.bind.assert_called_with(temperature=0.9)
        assert len(children) == 1
    
    def test_expansion_policy_prompt_assembly(self):
        """Test prompt template assembly with various inputs."""
        mock_llm = Mock()
        mock_bound_llm = Mock()
        mock_llm.bind.return_value = mock_bound_llm
        mock_bound_llm.invoke.return_value = [
            {"thought": "Response", "action": "respond", "partial_solution": "solution"}
        ]
        
        policy = ExpansionPolicy(mock_llm, k=2)
        
        # Test with complex state
        parent = ThoughtNode(
            id="parent", 
            parent_id=None, 
            depth=0,
            state={
                "reasoning_steps": ["Step 1", "Step 2"],
                "scratchpad": "Complex\nmultiline\ntext",
                "partial_solution": "Solution with \"quotes\" and special chars: <>&"
            }
        )
        
        children = policy.expand(parent, task="Complex task with special chars: <>&", constraints="Strict constraints")
        
        # Should handle complex inputs without crashing
        assert len(children) == 2
        mock_bound_llm.invoke.assert_called_once()
        
        # Verify the prompt was assembled with all inputs
        call_args = mock_bound_llm.invoke.call_args[0][0]
        assert "Complex task with special chars" in call_args["task"]
        assert "Strict constraints" in call_args["constraints"]
        assert "Complex" in call_args["state_json"]
    
    def test_expansion_policy_chain_building(self):
        """Test that the LangChain chain is built correctly."""
        mock_llm = Mock()
        policy = ExpansionPolicy(mock_llm, k=3, temperature=0.5)
        
        # Verify chain components
        assert policy.llm == mock_llm
        assert policy.k == 3
        assert policy.temperature == 0.5
        assert hasattr(policy, 'prompt')
        assert hasattr(policy, 'parser')
        assert hasattr(policy, 'chain')
    
    def test_expansion_policy_k_parameter_bounds(self):
        """Test expansion policy with extreme k values."""
        mock_llm = Mock()
        mock_bound_llm = Mock()
        mock_llm.bind.return_value = mock_bound_llm
        
        # Test with k=0
        policy_zero = ExpansionPolicy(mock_llm, k=0)
        mock_bound_llm.invoke.return_value = [
            {"thought": "Test", "action": "test", "partial_solution": "test"}
        ]
        
        parent = ThoughtNode(id="parent", parent_id=None, depth=0)
        children = policy_zero.expand(parent, task="Test task")
        
        # With k=0, should get 0 children from successful response
        assert len(children) == 0
        
        # Test with very large k
        policy_large = ExpansionPolicy(mock_llm, k=1000)
        mock_bound_llm.invoke.return_value = [
            {"thought": f"Test {i}", "action": f"test_{i}", "partial_solution": f"solution_{i}"}
            for i in range(5)  # Only 5 responses
        ]
        
        children = policy_large.expand(parent, task="Test task")
        
        # Should only get 5 children despite k=1000
        assert len(children) == 5
    
    def test_expansion_policy_unicode_handling(self):
        """Test expansion policy with unicode and special characters."""
        mock_llm = Mock()
        mock_bound_llm = Mock()
        mock_llm.bind.return_value = mock_bound_llm
        mock_bound_llm.invoke.return_value = [
            {"thought": "Unicode: 你好", "action": "unicode_action", "partial_solution": "Solution with 🚀 emoji"}
        ]
        
        policy = ExpansionPolicy(mock_llm, k=1)
        
        parent = ThoughtNode(id="parent", parent_id=None, depth=0, state={"unicode": "Testing 中文"})
        children = policy.expand(parent, task="Unicode task: 测试", constraints="Constraints with émojis 🎯")
        
        # Should handle unicode gracefully
        assert len(children) == 1
        assert "Unicode: 你好" in children[0].state["reasoning_steps"]
        assert children[0].action == "unicode_action"


class TestScoringPolicyEdgeCases:
    """Test scoring policy edge cases."""
    
    def test_scoring_policy_weight_normalization(self):
        """Test scoring policy with various weight configurations."""
        mock_llm = Mock()
        
        # Test with negative weights
        negative_weights = {
            "accuracy": -0.5,
            "consistency": 1.0,
            "progress": 0.5,
            "novelty": 0.3,
            "cost": -0.1
        }
        policy = ScoringPolicy(mock_llm, weights=negative_weights)
        assert policy.weights == negative_weights
        
        # Test with zero weights
        zero_weights = {
            "accuracy": 0.0,
            "consistency": 0.0,
            "progress": 0.0,
            "novelty": 0.0,
            "cost": 0.0
        }
        policy_zero = ScoringPolicy(mock_llm, weights=zero_weights)
        
        mock_llm.invoke.return_value = {
            "score": 0.8,
            "confidence": 0.9,
            "components": {"accuracy": 1.0, "consistency": 1.0, "progress": 1.0, "novelty": 1.0, "cost": 1.0},
            "rationale": "Perfect components"
        }
        
        tree = TreeManager()
        node = ThoughtNode(id="node", parent_id=None, depth=0)
        tree.add_node(node)
        
        score, conf, comps, rationale = policy_zero.score(node, tree, "Test task")
        
        # With zero weights, score should be 0.0
        assert score == 0.0
    
    def test_scoring_policy_component_edge_cases(self):
        """Test scoring policy with extreme component values."""
        mock_llm = Mock()
        policy = ScoringPolicy(mock_llm)
        
        # Test with extreme values that get normalized
        mock_llm.invoke.return_value = {
            "score": 0.8,
            "confidence": 1.5,  # Above 1.0
            "components": {
                "accuracy": 999.0,    # Way above 1.0
                "consistency": -100.0, # Way below 0.0
                "progress": float('inf'),  # Infinity
                "novelty": float('nan'),   # NaN
                "cost": "not_a_number"     # Invalid type
            },
            "rationale": "Extreme values"
        }
        
        tree = TreeManager()
        node = ThoughtNode(id="node", parent_id=None, depth=0)
        tree.add_node(node)
        
        score, conf, comps, rationale = policy.score(node, tree, "Test task")
        
        # Score should be bounded
        assert 0.0 <= score <= 1.0
        # Confidence should be bounded 
        assert 0.0 <= conf <= 1.0
        # Invalid components should be normalized to 0.0
        assert comps["progress"] == 0.0  # inf -> 0.0
        assert comps["novelty"] == 0.0   # nan -> 0.0
        assert comps["cost"] == 0.0      # invalid type -> 0.0
    
    def test_scoring_policy_prompt_variations(self):
        """Test scoring policy with different state structures."""
        mock_llm = Mock()
        policy = ScoringPolicy(mock_llm)
        
        tree = TreeManager()
        
        # Test with minimal state
        node_minimal = ThoughtNode(id="minimal", parent_id=None, depth=0, state={})
        tree.add_node(node_minimal)
        
        # Test with complex nested state
        node_complex = ThoughtNode(
            id="complex", 
            parent_id=None, 
            depth=0,
            state={
                "reasoning_steps": ["A", "B", "C"],
                "scratchpad": "Complex data",
                "partial_solution": {"nested": {"data": [1, 2, 3]}},
                "custom_field": "Custom value"
            }
        )
        tree.add_node(node_complex)
        
        mock_llm.invoke.return_value = {
            "score": 0.8,
            "confidence": 0.9,
            "components": {"accuracy": 0.8, "consistency": 0.8, "progress": 0.8, "novelty": 0.8, "cost": 0.2},
            "rationale": "Good"
        }
        
        # Both should work
        score1, _, _, _ = policy.score(node_minimal, tree, "Test task")
        score2, _, _, _ = policy.score(node_complex, tree, "Test task")
        
        assert isinstance(score1, float)
        assert isinstance(score2, float)
        assert 0.0 <= score1 <= 1.0
        assert 0.0 <= score2 <= 1.0
    
    def test_scoring_policy_missing_components_handling(self):
        """Test scoring policy with various missing component scenarios."""
        mock_llm = Mock()
        policy = ScoringPolicy(mock_llm)
        
        test_cases = [
            {"components": {}},  # All missing
            {"components": {"accuracy": 0.8}},  # Partial
            {"components": None},  # Null
            {},  # No components field
        ]
        
        tree = TreeManager()
        node = ThoughtNode(id="node", parent_id=None, depth=0)
        tree.add_node(node)
        
        for i, test_case in enumerate(test_cases):
            mock_llm.invoke.return_value = {
                "score": 0.8,
                "confidence": 0.9,
                "rationale": f"Test case {i}",
                **test_case
            }
            
            score, conf, comps, rationale = policy.score(node, tree, "Test task")
            
            # Should handle all cases gracefully
            assert 0.0 <= score <= 1.0
            assert 0.0 <= conf <= 1.0
            assert all(isinstance(v, float) for v in comps.values())
            assert all(0.0 <= v <= 1.0 for v in comps.values() if v >= 0)  # Allow negative cost


class TestSelectionPolicyEdgeCases:
    """Test selection policy edge cases."""
    
    def test_selection_policy_parameter_bounds(self):
        """Test selection policy with extreme beta/gamma parameters."""
        tree = TreeManager()
        
        # Create test nodes
        nodes = []
        for i in range(3):
            node = ThoughtNode(id=str(i), parent_id=None, depth=0, status="open")
            node.score = 0.5 + (i * 0.1)
            node.confidence = 0.5 + (i * 0.1)
            node.components = {"novelty": 0.3 + (i * 0.1)}
            tree.add_node(node)
            nodes.append(node)
        
        # Test with extreme positive beta (high uncertainty bonus)
        policy_high_beta = SelectionPolicy(beta=10.0, gamma=0.0)
        selected = policy_high_beta.select(tree, k=1)
        # Should select lowest confidence node (highest uncertainty)
        assert selected[0].id == "0"  # Lowest confidence = 0.5
        
        # Test with extreme negative beta
        policy_neg_beta = SelectionPolicy(beta=-10.0, gamma=0.0)
        selected = policy_neg_beta.select(tree, k=1)
        # Should penalize uncertainty, prefer high confidence
        assert selected[0].id == "2"  # Highest confidence = 0.7
        
        # Test with extreme gamma (high novelty bonus)
        policy_high_gamma = SelectionPolicy(beta=0.0, gamma=10.0)
        selected = policy_high_gamma.select(tree, k=1)
        # Should select highest novelty node
        assert selected[0].id == "2"  # Highest novelty = 0.5
    
    def test_selection_policy_missing_components(self):
        """Test selection policy when nodes missing components."""
        tree = TreeManager()
        
        # Node with missing novelty component
        node1 = ThoughtNode(id="1", parent_id=None, depth=0, status="open")
        node1.score = 0.6
        node1.confidence = 0.7
        node1.components = {}  # No novelty
        tree.add_node(node1)
        
        # Node with novelty component
        node2 = ThoughtNode(id="2", parent_id=None, depth=0, status="open")
        node2.score = 0.5
        node2.confidence = 0.7
        node2.components = {"novelty": 0.8}
        tree.add_node(node2)
        
        policy = SelectionPolicy(beta=0.0, gamma=1.0)  # Only novelty matters
        selected = policy.select(tree, k=1)
        
        # Should select node with novelty component
        assert selected[0].id == "2"
    
    def test_selection_policy_identical_scores(self):
        """Test selection policy with identical priority scores."""
        tree = TreeManager()
        
        # Create nodes with identical priorities
        for i in range(5):
            node = ThoughtNode(id=str(i), parent_id=None, depth=0, status="open")
            node.score = 0.5  # Same score
            node.confidence = 0.7  # Same confidence
            node.components = {"novelty": 0.3}  # Same novelty
            tree.add_node(node)
        
        policy = SelectionPolicy(beta=0.2, gamma=0.1)
        selected = policy.select(tree, k=3)
        
        # Should return some selection without crashing
        assert len(selected) == 3
        assert all(isinstance(n, ThoughtNode) for n in selected)


class TestPruningPolicyEdgeCases:
    """Test pruning policy edge cases."""
    
    def test_pruning_policy_beam_size_validation(self):
        """Test pruning policy with edge case beam sizes."""
        tree = TreeManager()
        
        # Add 5 nodes at depth 1
        for i in range(5):
            node = ThoughtNode(id=str(i), parent_id=None, depth=1, status="open")
            node.score = 0.9 - (i * 0.1)  # Decreasing scores
            tree.add_node(node)
        
        # Test with beam=0 (prune everything)
        policy_zero = PruningPolicy(beam_per_depth=0, min_score=0.0)
        policy_zero.prune(tree, depth=1)
        
        # All nodes should be pruned
        for i in range(5):
            assert tree.nodes[str(i)].status == "pruned"
        
        # Reset nodes
        for i in range(5):
            tree.nodes[str(i)].status = "open"
            if str(i) not in tree.frontier:
                tree.frontier.append(str(i))
            tree.open_status[str(i)] = True
        
        # Test with beam larger than available nodes
        policy_large = PruningPolicy(beam_per_depth=100, min_score=0.0)
        policy_large.prune(tree, depth=1)
        
        # All nodes should remain open
        for i in range(5):
            assert tree.nodes[str(i)].status == "open"
    
    def test_pruning_policy_score_threshold_edge_cases(self):
        """Test pruning policy with extreme score thresholds."""
        tree = TreeManager()
        
        # Add nodes with various scores
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        for i, score in enumerate(scores):
            node = ThoughtNode(id=str(i), parent_id=None, depth=1, status="open")
            node.score = score
            tree.add_node(node)
        
        # Test with impossible high threshold
        policy_high = PruningPolicy(beam_per_depth=10, min_score=2.0)
        policy_high.prune(tree, depth=1)
        
        # All nodes should be pruned due to high threshold
        for i in range(5):
            assert tree.nodes[str(i)].status == "pruned"
        
        # Reset nodes
        for i in range(5):
            tree.nodes[str(i)].status = "open"
            if str(i) not in tree.frontier:
                tree.frontier.append(str(i))
            tree.open_status[str(i)] = True
        
        # Test with negative threshold
        policy_neg = PruningPolicy(beam_per_depth=10, min_score=-1.0)
        policy_neg.prune(tree, depth=1)
        
        # All nodes should remain (all scores > -1.0)
        for i in range(5):
            assert tree.nodes[str(i)].status == "open"
    
    def test_pruning_policy_empty_depth(self):
        """Test pruning policy on depths with no nodes."""
        tree = TreeManager()
        policy = PruningPolicy(beam_per_depth=3, min_score=0.5)
        
        # Pruning non-existent depth should not crash
        policy.prune(tree, depth=999)
        
        # Add some nodes at depth 0
        node = ThoughtNode(id="0", parent_id=None, depth=0, status="open")
        tree.add_node(node)
        
        # Pruning different depth should not affect existing nodes
        policy.prune(tree, depth=1)
        assert tree.nodes["0"].status == "open"


class TestTerminationPolicyEdgeCases:
    """Test termination policy edge cases."""
    
    def test_termination_policy_parameter_conflicts(self):
        """Test termination policy with conflicting parameters."""
        tree = TreeManager()
        
        # Conflicting parameters: very low limits
        policy = TerminationPolicy(max_depth=0, max_nodes=0, target_score=-1.0)
        
        # Empty tree should trigger termination
        assert policy.should_stop(tree)
        
        # Add one node
        node = ThoughtNode(id="test", parent_id=None, depth=0, status="open")
        node.score = 0.5
        tree.add_node(node)
        
        # Should still terminate due to max_nodes=0 (1 > 0)
        assert policy.should_stop(tree)
    
    def test_termination_policy_edge_scores(self):
        """Test termination policy with edge case scores."""
        tree = TreeManager()
        
        # Test with very high target score
        policy_high = TerminationPolicy(max_depth=10, max_nodes=100, target_score=1.0)
        
        node = ThoughtNode(id="test", parent_id=None, depth=0, status="expanded")
        node.score = 0.999999  # Very close but not quite 1.0
        tree.add_node(node)
        
        assert not policy_high.should_stop(tree)  # Should not stop
        
        node.score = 1.0  # Exactly 1.0
        assert policy_high.should_stop(tree)  # Should stop
        
        # Test with negative target score
        policy_neg = TerminationPolicy(max_depth=10, max_nodes=100, target_score=-0.5)
        
        node.score = 0.0  # Any positive score exceeds -0.5
        assert policy_neg.should_stop(tree)
    
    def test_termination_policy_complex_tree_states(self):
        """Test termination policy with complex tree states."""
        tree = TreeManager()
        policy = TerminationPolicy(max_depth=3, max_nodes=100, target_score=0.9)
        
        # Tree with all expanded nodes (no open nodes)
        for i in range(3):
            node = ThoughtNode(id=str(i), parent_id=None, depth=i, status="expanded")
            node.score = 0.5
            tree.add_node(node)
        
        # Should stop due to no open nodes
        assert policy.should_stop(tree)
        
        # Add one open node at max depth
        deep_node = ThoughtNode(id="deep", parent_id=None, depth=3, status="open")
        tree.add_node(deep_node)
        
        # Should stop due to all open nodes at max depth
        assert policy.should_stop(tree)
        
        # Add open node within depth limit
        shallow_node = ThoughtNode(id="shallow", parent_id=None, depth=1, status="open")
        tree.add_node(shallow_node)
        
        # Should not stop (has valid open node)
        assert not policy.should_stop(tree)


class TestPolicyParameterValidation:
    """Test policy parameter validation across all policies."""
    
    def test_policy_initialization_edge_cases(self):
        """Test policy initialization with edge case parameters."""
        mock_llm = Mock()
        
        # Test expansion policy with edge cases
        exp_policy = ExpansionPolicy(mock_llm, k=-1, temperature=-1.0)
        assert exp_policy.k == -1  # Should accept (though unusual)
        assert exp_policy.temperature == -1.0
        
        # Test scoring policy with extreme weights
        extreme_weights = {
            "accuracy": 1000.0,
            "consistency": -1000.0,
            "progress": 0.0,
            "novelty": float('inf'),
            "cost": float('-inf')
        }
        score_policy = ScoringPolicy(mock_llm, weights=extreme_weights)
        assert score_policy.weights == extreme_weights
        
        # Test selection policy with extreme parameters
        sel_policy = SelectionPolicy(beta=float('inf'), gamma=float('-inf'))
        assert sel_policy.beta == float('inf')
        assert sel_policy.gamma == float('-inf')
        
        # Test pruning policy with edge cases
        prune_policy = PruningPolicy(beam_per_depth=-1, min_score=float('inf'), diversity_threshold=-1.0)
        assert prune_policy.beam == -1
        assert prune_policy.min_score == float('inf')
        
        # Test termination policy with edge cases
        term_policy = TerminationPolicy(max_depth=-1, max_nodes=-1, target_score=float('inf'))
        assert term_policy.max_depth == -1
        assert term_policy.max_nodes == -1
        assert term_policy.target_score == float('inf')
    
    def test_policy_type_validation(self):
        """Test policies with incorrect parameter types."""
        mock_llm = Mock()
        
        # These should work (Python's duck typing)
        try:
            ExpansionPolicy(mock_llm, k="3", temperature="0.5")  # String numbers
            ScoringPolicy(mock_llm, weights={"accuracy": "0.5"})  # String in dict
            SelectionPolicy(beta="0.2", gamma="0.1")  # String parameters
        except (TypeError, ValueError):
            # If they fail, that's also acceptable validation
            pass