"""Advanced orchestrator scenarios and edge cases."""

import json
import pytest
from unittest.mock import Mock, MagicMock

from langtree.models import ThoughtNode, TreeManager
from langtree.orchestrator import ToTOrchestrator


class MockSequentialLLM:
    """Mock LLM that returns predefined sequences of responses."""
    
    def __init__(self, expansion_sequences, scoring_sequences):
        self.expansion_sequences = expansion_sequences
        self.scoring_sequences = scoring_sequences
        self.expansion_index = 0
        self.scoring_index = 0
    
    def bind(self, temperature=None, **kwargs):
        mock_bound = Mock()
        if temperature == 0.7:  # Expansion
            mock_bound.invoke = self._invoke_expansion
        else:  # Scoring
            mock_bound.invoke = self._invoke_scoring
        return mock_bound
    
    def _invoke_expansion(self, inputs):
        if self.expansion_index < len(self.expansion_sequences):
            response = self.expansion_sequences[self.expansion_index]
            self.expansion_index += 1
            return response
        return [{"thought": "fallback", "action": "fallback", "partial_solution": "fallback"}]
    
    def _invoke_scoring(self, inputs):
        if self.scoring_index < len(self.scoring_sequences):
            response = self.scoring_sequences[self.scoring_index]
            self.scoring_index += 1
            return response
        return {"score": 0.5, "confidence": 0.5, "components": {"accuracy": 0.5, "consistency": 0.5, "progress": 0.5, "novelty": 0.5, "cost": 0.5}, "rationale": "fallback"}


class TestAdvancedOrchestratorScenarios:
    """Test advanced orchestrator scenarios."""
    
    def test_orchestrator_immediate_max_nodes_reached(self):
        """Test orchestrator when max_nodes is reached immediately."""
        expansion_responses = [
            [{"thought": "Child 1", "action": "a1", "partial_solution": "s1"}]
        ]
        scoring_responses = [
            {"score": 0.8, "confidence": 0.9, "components": {"accuracy": 0.8, "consistency": 0.8, "progress": 0.8, "novelty": 0.8, "cost": 0.2}, "rationale": "Good"}
        ]
        
        mock_llm = MockSequentialLLM(expansion_responses, scoring_responses)
        
        # Set max_nodes to 2 (root + 1 child = immediate termination)
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm,
            llm_score=mock_llm,
            k_children=1,
            max_select=1,
            max_depth=10,
            max_nodes=2,  # Very low limit
            target_score=0.9,
        )
        
        result = orchestrator.run(task="Immediate max nodes test")
        
        # Should terminate after adding just one child
        snapshot = result["tree_snapshot"]
        assert len(snapshot["nodes"]) == 2  # Root + 1 child
    
    def test_orchestrator_no_valid_expansions(self):
        """Test orchestrator when no valid expansions are possible."""
        # Expansion returns empty list
        expansion_responses = [
            [],  # No children generated
            []   # Still no children
        ]
        scoring_responses = []  # No scoring needed
        
        mock_llm = MockSequentialLLM(expansion_responses, scoring_responses)
        
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm,
            llm_score=mock_llm,
            k_children=3,
            max_select=1,
            max_depth=5,
            max_nodes=100,
            target_score=0.9,
        )
        
        result = orchestrator.run(task="No expansions test")
        
        # Should terminate gracefully with just root node
        snapshot = result["tree_snapshot"]
        assert len(snapshot["nodes"]) == 1  # Just root
        
        # Best node should be the root
        root_nodes = [n for n in snapshot["nodes"].values() if n["parent_id"] is None]
        assert len(root_nodes) == 1
    
    def test_orchestrator_all_nodes_pruned(self):
        """Test orchestrator when all generated nodes get pruned."""
        expansion_responses = [
            [
                {"thought": "Low score 1", "action": "low1", "partial_solution": "low1"},
                {"thought": "Low score 2", "action": "low2", "partial_solution": "low2"},
                {"thought": "Low score 3", "action": "low3", "partial_solution": "low3"}
            ]
        ]
        
        # All children get very low scores
        scoring_responses = [
            {"score": 0.1, "confidence": 0.2, "components": {"accuracy": 0.1, "consistency": 0.1, "progress": 0.1, "novelty": 0.1, "cost": 0.9}, "rationale": "Very poor 1"},
            {"score": 0.05, "confidence": 0.1, "components": {"accuracy": 0.05, "consistency": 0.05, "progress": 0.05, "novelty": 0.05, "cost": 0.95}, "rationale": "Very poor 2"},
            {"score": 0.02, "confidence": 0.05, "components": {"accuracy": 0.02, "consistency": 0.02, "progress": 0.02, "novelty": 0.02, "cost": 0.98}, "rationale": "Very poor 3"}
        ]
        
        mock_llm = MockSequentialLLM(expansion_responses, scoring_responses)
        
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm,
            llm_score=mock_llm,
            k_children=3,
            max_select=1,
            max_depth=5,
            max_nodes=100,
            target_score=0.8,
            beam_per_depth=1,  # Keep only 1 per depth
            min_score=0.3,     # Prune all low scores
        )
        
        result = orchestrator.run(task="All pruned test")
        
        # Should terminate when no open nodes remain
        snapshot = result["tree_snapshot"]
        depth_1_nodes = [n for n in snapshot["nodes"].values() if n["depth"] == 1]
        
        # All depth 1 nodes should be pruned
        for node in depth_1_nodes:
            assert node["status"] == "pruned"
    
    def test_orchestrator_alternating_success_failure(self):
        """Test orchestrator with alternating success and failure patterns."""
        expansion_responses = [
            # First expansion succeeds
            [{"thought": "Good start", "action": "good", "partial_solution": "good"}],
            # Second expansion fails (empty list)
            [],
            # Third expansion succeeds  
            [{"thought": "Recovery", "action": "recover", "partial_solution": "recover"}]
        ]
        
        scoring_responses = [
            # Score for first child
            {"score": 0.7, "confidence": 0.8, "components": {"accuracy": 0.7, "consistency": 0.7, "progress": 0.8, "novelty": 0.6, "cost": 0.3}, "rationale": "Good start"},
            # Score for recovery child  
            {"score": 0.85, "confidence": 0.9, "components": {"accuracy": 0.85, "consistency": 0.85, "progress": 0.9, "novelty": 0.7, "cost": 0.2}, "rationale": "Good recovery"}
        ]
        
        mock_llm = MockSequentialLLM(expansion_responses, scoring_responses)
        
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm,
            llm_score=mock_llm,
            k_children=1,
            max_select=1,
            max_depth=5,
            max_nodes=100,
            target_score=0.8,
        )
        
        result = orchestrator.run(task="Alternating success/failure test")
        
        # Should complete successfully despite failures
        assert result["best_score"] >= 0.8
        snapshot = result["tree_snapshot"]
        assert len(snapshot["nodes"]) >= 2  # Root + at least one successful child
    
    def test_orchestrator_deep_tree_early_termination(self):
        """Test orchestrator with very deep tree but early termination."""
        # Create responses for deep tree
        expansion_responses = []
        scoring_responses = []
        
        for depth in range(10):  # Up to depth 10
            expansion_responses.append([
                {"thought": f"Depth {depth}", "action": f"action_{depth}", "partial_solution": f"solution_{depth}"}
            ])
            # Gradually increasing scores, reaching target at depth 5
            score = 0.5 + (depth * 0.1)
            scoring_responses.append({
                "score": score,
                "confidence": 0.8,
                "components": {"accuracy": score, "consistency": score, "progress": score, "novelty": 0.5, "cost": 0.2},
                "rationale": f"Depth {depth} scoring"
            })
        
        mock_llm = MockSequentialLLM(expansion_responses, scoring_responses)
        
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm,
            llm_score=mock_llm,
            k_children=1,
            max_select=1,
            max_depth=20,  # Allow deep trees
            max_nodes=100,
            target_score=0.9,  # Will be reached around depth 4-5
        )
        
        result = orchestrator.run(task="Deep tree early termination test")
        
        # Should terminate early due to target score
        assert result["best_score"] >= 0.9
        snapshot = result["tree_snapshot"]
        max_depth = max(n["depth"] for n in snapshot["nodes"].values())
        assert max_depth < 10  # Should terminate before reaching max depth
    
    def test_orchestrator_wide_tree_beam_pruning(self):
        """Test orchestrator with wide tree and aggressive beam pruning."""
        expansion_responses = [
            # Create many children in first expansion
            [
                {"thought": f"Wide child {i}", "action": f"wide_{i}", "partial_solution": f"wide_solution_{i}"}
                for i in range(10)  # 10 children
            ]
        ]
        
        # Give different scores to children
        scoring_responses = [
            {
                "score": 0.9 - (i * 0.05),  # Decreasing scores
                "confidence": 0.8,
                "components": {"accuracy": 0.9 - (i * 0.05), "consistency": 0.8, "progress": 0.7, "novelty": 0.6, "cost": 0.3},
                "rationale": f"Child {i} scoring"
            }
            for i in range(10)
        ]
        
        mock_llm = MockSequentialLLM(expansion_responses, scoring_responses)
        
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm,
            llm_score=mock_llm,
            k_children=10,
            max_select=1,
            max_depth=3,
            max_nodes=100,
            target_score=0.95,  # High target
            beam_per_depth=3,   # Keep only top 3
            min_score=0.0,
        )
        
        result = orchestrator.run(task="Wide tree beam pruning test")
        
        snapshot = result["tree_snapshot"]
        depth_1_nodes = [n for n in snapshot["nodes"].values() if n["depth"] == 1]
        
        # Should have created 10 nodes but kept only 3
        assert len(depth_1_nodes) == 10
        open_or_expanded = [n for n in depth_1_nodes if n["status"] in ("open", "expanded")]
        assert len(open_or_expanded) <= 3  # Beam size
        
        # Best 3 should be kept (highest scores)
        kept_scores = sorted([n["score"] for n in open_or_expanded], reverse=True)
        expected_top_scores = sorted([0.9 - (i * 0.05) for i in range(3)], reverse=True)
        assert kept_scores == expected_top_scores
    
    def test_orchestrator_concurrent_termination_conditions(self):
        """Test orchestrator when multiple termination conditions are met simultaneously."""
        expansion_responses = [
            [{"thought": "Perfect solution", "action": "perfect", "partial_solution": "perfect"}]
        ]
        
        scoring_responses = [
            {"score": 0.95, "confidence": 0.99, "components": {"accuracy": 0.95, "consistency": 0.95, "progress": 0.95, "novelty": 0.8, "cost": 0.1}, "rationale": "Perfect solution"}
        ]
        
        mock_llm = MockSequentialLLM(expansion_responses, scoring_responses)
        
        # Set up multiple termination conditions that will all be met
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm,
            llm_score=mock_llm,
            k_children=1,
            max_select=1,
            max_depth=2,      # Will be reached
            max_nodes=2,      # Will be reached (root + 1 child)
            target_score=0.9, # Will be reached
        )
        
        result = orchestrator.run(task="Multiple termination conditions test")
        
        # Should terminate successfully
        assert result["best_score"] >= 0.9
        snapshot = result["tree_snapshot"]
        assert len(snapshot["nodes"]) == 2  # Root + 1 child
    
    def test_orchestrator_dynamic_scoring_changes(self):
        """Test orchestrator with dramatically changing scores."""
        expansion_responses = [
            [{"thought": "First try", "action": "first", "partial_solution": "first"}],
            [{"thought": "Second try", "action": "second", "partial_solution": "second"}],
            [{"thought": "Third try", "action": "third", "partial_solution": "third"}]
        ]
        
        # Scores oscillate dramatically
        scoring_responses = [
            {"score": 0.9, "confidence": 0.9, "components": {"accuracy": 0.9, "consistency": 0.9, "progress": 0.9, "novelty": 0.8, "cost": 0.1}, "rationale": "Great start"},
            {"score": 0.1, "confidence": 0.2, "components": {"accuracy": 0.1, "consistency": 0.1, "progress": 0.1, "novelty": 0.1, "cost": 0.9}, "rationale": "Terrible follow-up"},
            {"score": 0.95, "confidence": 0.95, "components": {"accuracy": 0.95, "consistency": 0.95, "progress": 0.95, "novelty": 0.9, "cost": 0.05}, "rationale": "Amazing recovery"}
        ]
        
        mock_llm = MockSequentialLLM(expansion_responses, scoring_responses)
        
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm,
            llm_score=mock_llm,
            k_children=1,
            max_select=1,
            max_depth=5,
            max_nodes=100,
            target_score=0.85,
            beam_per_depth=2,  # Keep some diversity
            min_score=0.0,     # Don't prune based on score
        )
        
        result = orchestrator.run(task="Dynamic scoring test")
        
        # Should eventually find the high-scoring path
        assert result["best_score"] >= 0.85
        
        # Should have explored multiple paths
        snapshot = result["tree_snapshot"]
        assert len(snapshot["nodes"]) >= 3  # Root + multiple attempts
    
    def test_orchestrator_zero_confidence_handling(self):
        """Test orchestrator with zero confidence scores."""
        expansion_responses = [
            [
                {"thought": "Uncertain path 1", "action": "uncertain1", "partial_solution": "uncertain1"},
                {"thought": "Uncertain path 2", "action": "uncertain2", "partial_solution": "uncertain2"}
            ]
        ]
        
        scoring_responses = [
            {"score": 0.7, "confidence": 0.0, "components": {"accuracy": 0.7, "consistency": 0.7, "progress": 0.7, "novelty": 0.6, "cost": 0.3}, "rationale": "High uncertainty 1"},
            {"score": 0.6, "confidence": 0.0, "components": {"accuracy": 0.6, "consistency": 0.6, "progress": 0.6, "novelty": 0.7, "cost": 0.4}, "rationale": "High uncertainty 2"}
        ]
        
        mock_llm = MockSequentialLLM(expansion_responses, scoring_responses)
        
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm,
            llm_score=mock_llm,
            k_children=2,
            max_select=2,
            max_depth=3,
            max_nodes=100,
            target_score=0.8,  # Won't be reached
        )
        
        result = orchestrator.run(task="Zero confidence test")
        
        # Should handle zero confidence gracefully
        assert result["best_node_id"] is not None
        assert isinstance(result["best_confidence"], (int, float))
        
        # Zero confidence should give high uncertainty bonus in selection
        # Both nodes should be explored due to high uncertainty
        snapshot = result["tree_snapshot"]
        assert len(snapshot["nodes"]) >= 3  # Root + 2 children


class TestOrchestratorEdgeCases:
    """Test edge cases in orchestrator behavior."""
    
    def test_orchestrator_single_node_tree(self):
        """Test orchestrator that creates only single-node tree."""
        # Expansion fails immediately
        expansion_responses = []
        scoring_responses = []
        
        mock_llm = Mock()
        mock_llm.bind.return_value = mock_llm
        mock_llm.invoke.side_effect = Exception("No expansions possible")
        
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm,
            llm_score=mock_llm,
            k_children=3,
            max_select=1,
            max_depth=5,
            max_nodes=100,
            target_score=0.8,
        )
        
        result = orchestrator.run(task="Single node test")
        
        # Should complete with just root node
        snapshot = result["tree_snapshot"]
        assert len(snapshot["nodes"]) == 1
        
        # Root should be the best (and only) node
        root_node = list(snapshot["nodes"].values())[0]
        assert root_node["parent_id"] is None
        assert root_node["depth"] == 0
    
    def test_orchestrator_invalid_task_input(self):
        """Test orchestrator with various invalid task inputs."""
        mock_llm = Mock()
        mock_llm.bind.return_value = mock_llm
        mock_llm.invoke.return_value = [
            {"thought": "Handling invalid task", "action": "handle", "partial_solution": "handled"}
        ]
        
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm,
            llm_score=mock_llm,
            target_score=0.8,
        )
        
        # Test various invalid task inputs
        invalid_tasks = ["", None, 0, [], {}]
        
        for invalid_task in invalid_tasks:
            try:
                result = orchestrator.run(task=invalid_task)
                # Should complete without crashing
                assert result["best_node_id"] is not None
            except (TypeError, AttributeError):
                # Some invalid types might cause immediate failures, which is acceptable
                pass
    
    def test_orchestrator_extreme_parameter_combinations(self):
        """Test orchestrator with extreme parameter combinations."""
        mock_llm = Mock()
        mock_llm.bind.return_value = mock_llm
        mock_llm.invoke.return_value = [
            {"thought": "Extreme test", "action": "extreme", "partial_solution": "extreme"}
        ]
        
        # Extreme parameters
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm,
            llm_score=mock_llm,
            k_children=0,     # No children
            max_select=0,     # Select nothing
            max_depth=0,      # No depth
            max_nodes=1,      # Only root
            target_score=0.0, # Immediate target
            beam_per_depth=0, # Prune everything
            min_score=1.0,    # Impossible min score
        )
        
        result = orchestrator.run(task="Extreme parameters test")
        
        # Should handle extreme parameters gracefully
        assert result["best_node_id"] is not None
        snapshot = result["tree_snapshot"]
        assert len(snapshot["nodes"]) >= 1  # At least root node