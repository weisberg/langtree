"""Integration tests for the complete ToT orchestrator."""

import pytest
from unittest.mock import Mock

from langtree.orchestrator import ToTOrchestrator
from langtree.models import ThoughtNode


class MockLLM:
    """Mock LLM for integration testing."""
    
    def __init__(self, expansion_responses, scoring_responses):
        self.expansion_responses = expansion_responses
        self.scoring_responses = scoring_responses
        self.expansion_calls = 0
        self.scoring_calls = 0
        self.last_temperature = None
    
    def bind(self, temperature=None, **kwargs):
        """Mock bind method that tracks temperature."""
        mock_bound = Mock()
        mock_bound.invoke = self._get_invoke_for_temperature(temperature)
        return mock_bound
    
    def _get_invoke_for_temperature(self, temperature):
        """Return appropriate invoke function based on temperature."""
        if temperature == 0.7:  # Expansion LLM
            return self._invoke_expansion
        else:  # Scoring LLM (temperature=0.0)
            return self._invoke_scoring
    
    def _invoke_expansion(self, inputs):
        """Mock invoke for expansion."""
        if self.expansion_calls < len(self.expansion_responses):
            response = self.expansion_responses[self.expansion_calls]
            self.expansion_calls += 1
            return response
        else:
            # Fallback response
            return [{"thought": "No more responses", "action": "fallback", "partial_solution": "fallback"}]
    
    def _invoke_scoring(self, inputs):
        """Mock invoke for scoring."""
        if self.scoring_calls < len(self.scoring_responses):
            response = self.scoring_responses[self.scoring_calls]
            self.scoring_calls += 1
            return response
        else:
            # Fallback response
            return {
                "score": 0.3,
                "confidence": 0.3,
                "components": {"accuracy": 0.3, "consistency": 0.3, "progress": 0.3, "novelty": 0.3, "cost": 0.3},
                "rationale": "Fallback scoring"
            }


class TestToTOrchestrator:
    """Test complete ToT orchestrator integration."""
    
    def test_simple_task_completion(self):
        """Test orchestrator completes a simple task."""
        # Mock expansion responses (2 rounds)
        expansion_responses = [
            # Round 1: expand root
            [
                {"thought": "Start with analysis", "action": "analyze", "partial_solution": "Analysis phase"},
                {"thought": "Start with planning", "action": "plan", "partial_solution": "Planning phase"},
            ],
            # Round 2: expand best node from round 1
            [
                {"thought": "Deep analysis", "action": "deep_analyze", "partial_solution": "Detailed analysis"},
                {"thought": "Alternative analysis", "action": "alt_analyze", "partial_solution": "Alternative approach"},
            ]
        ]
        
        # Mock scoring responses (4 scores: 2 for round 1, 2 for round 2)
        scoring_responses = [
            # Scores for round 1 children
            {"score": 0.6, "confidence": 0.8, "components": {"accuracy": 0.6, "consistency": 0.6, "progress": 0.7, "novelty": 0.5, "cost": 0.4}, "rationale": "Good start"},
            {"score": 0.4, "confidence": 0.7, "components": {"accuracy": 0.4, "consistency": 0.5, "progress": 0.3, "novelty": 0.6, "cost": 0.5}, "rationale": "Okay approach"},
            # Scores for round 2 children  
            {"score": 0.8, "confidence": 0.9, "components": {"accuracy": 0.8, "consistency": 0.8, "progress": 0.9, "novelty": 0.7, "cost": 0.3}, "rationale": "Excellent progress"},
            {"score": 0.7, "confidence": 0.8, "components": {"accuracy": 0.7, "consistency": 0.7, "progress": 0.8, "novelty": 0.8, "cost": 0.4}, "rationale": "Good alternative"},
        ]
        
        mock_llm = MockLLM(expansion_responses, scoring_responses)
        
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm,
            llm_score=mock_llm,
            k_children=2,
            max_select=1,  # Expand one node at a time
            max_depth=3,
            max_nodes=10,
            target_score=0.75,  # Will be reached in round 2
            beam_per_depth=3,
            min_score=0.0,
        )
        
        result = orchestrator.run(task="Test task", constraints="Test constraints")
        
        # Should find the best solution
        assert result["best_score"] >= 0.75
        assert len(result["best_path"]) > 1  # Should have made progress
        assert len(result["best_actions"]) > 1
        assert result["best_node_id"] is not None
        
        # Check tree snapshot
        snapshot = result["tree_snapshot"]
        assert "nodes" in snapshot
        assert len(snapshot["nodes"]) > 1  # Should have multiple nodes

    def test_max_depth_termination(self):
        """Test orchestrator terminates at max depth."""
        # Simple responses that don't reach target score
        expansion_responses = [
            [{"thought": "Step 1", "action": "action1", "partial_solution": "Solution 1"}],
            [{"thought": "Step 2", "action": "action2", "partial_solution": "Solution 2"}],
            [{"thought": "Step 3", "action": "action3", "partial_solution": "Solution 3"}],
        ]
        
        scoring_responses = [
            {"score": 0.3, "confidence": 0.5, "components": {"accuracy": 0.3, "consistency": 0.3, "progress": 0.3, "novelty": 0.3, "cost": 0.3}, "rationale": "Low score"},
            {"score": 0.4, "confidence": 0.6, "components": {"accuracy": 0.4, "consistency": 0.4, "progress": 0.4, "novelty": 0.4, "cost": 0.4}, "rationale": "Still low"},
            {"score": 0.5, "confidence": 0.7, "components": {"accuracy": 0.5, "consistency": 0.5, "progress": 0.5, "novelty": 0.5, "cost": 0.5}, "rationale": "Getting better"},
        ]
        
        mock_llm = MockLLM(expansion_responses, scoring_responses)
        
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm,
            llm_score=mock_llm,
            k_children=1,
            max_select=1,
            max_depth=2,  # Low max depth
            max_nodes=100,
            target_score=0.9,  # High target that won't be reached
            beam_per_depth=5,
            min_score=0.0,
        )
        
        result = orchestrator.run(task="Test task")
        
        # Should terminate due to max depth
        best_path = result["best_path"]
        assert len(best_path) <= 3  # Root + max_depth levels
        
        # Tree should have nodes at depths 0, 1, 2
        snapshot = result["tree_snapshot"]
        depths = [node["depth"] for node in snapshot["nodes"].values()]
        assert max(depths) <= 2

    def test_max_nodes_termination(self):
        """Test orchestrator terminates at max nodes."""
        # Responses that would generate many nodes
        expansion_responses = [
            [{"thought": f"Child {i}", "action": f"action{i}", "partial_solution": f"Solution {i}"} for i in range(3)]
        ] * 10  # Many rounds
        
        scoring_responses = [
            {"score": 0.3, "confidence": 0.5, "components": {"accuracy": 0.3, "consistency": 0.3, "progress": 0.3, "novelty": 0.3, "cost": 0.3}, "rationale": "Low score"}
        ] * 100  # Many scores
        
        mock_llm = MockLLM(expansion_responses, scoring_responses)
        
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm,
            llm_score=mock_llm,
            k_children=3,
            max_select=3,
            max_depth=10,
            max_nodes=8,  # Low max nodes (1 root + 3 children + 4 more = 8)
            target_score=0.9,
            beam_per_depth=5,
            min_score=0.0,
        )
        
        result = orchestrator.run(task="Test task")
        
        # Should terminate due to max nodes
        snapshot = result["tree_snapshot"]
        assert len(snapshot["nodes"]) <= 8

    def test_target_score_termination(self):
        """Test orchestrator terminates when target score is reached."""
        expansion_responses = [
            [{"thought": "Great idea", "action": "excellent_action", "partial_solution": "Perfect solution"}]
        ]
        
        scoring_responses = [
            {"score": 0.95, "confidence": 0.95, "components": {"accuracy": 0.95, "consistency": 0.95, "progress": 0.95, "novelty": 0.8, "cost": 0.2}, "rationale": "Excellent work"}
        ]
        
        mock_llm = MockLLM(expansion_responses, scoring_responses)
        
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm,
            llm_score=mock_llm,
            k_children=1,
            max_select=1,
            max_depth=10,
            max_nodes=100,
            target_score=0.9,  # Will be reached
            beam_per_depth=5,
            min_score=0.0,
        )
        
        result = orchestrator.run(task="Test task")
        
        # Should terminate due to target score
        assert result["best_score"] >= 0.9
        
        # Should have minimal tree (just root + 1 child)
        snapshot = result["tree_snapshot"]
        assert len(snapshot["nodes"]) == 2

    def test_beam_pruning(self):
        """Test that beam pruning works correctly."""
        # Create responses for many children but small beam
        expansion_responses = [
            [
                {"thought": f"Child {i}", "action": f"action{i}", "partial_solution": f"Solution {i}"}
                for i in range(4)  # 4 children
            ]
        ]
        
        # Give different scores to children
        scoring_responses = [
            {"score": 0.8, "confidence": 0.8, "components": {"accuracy": 0.8, "consistency": 0.8, "progress": 0.8, "novelty": 0.8, "cost": 0.2}, "rationale": "Best"},
            {"score": 0.6, "confidence": 0.7, "components": {"accuracy": 0.6, "consistency": 0.6, "progress": 0.6, "novelty": 0.6, "cost": 0.4}, "rationale": "Good"},
            {"score": 0.4, "confidence": 0.6, "components": {"accuracy": 0.4, "consistency": 0.4, "progress": 0.4, "novelty": 0.4, "cost": 0.6}, "rationale": "Okay"},
            {"score": 0.2, "confidence": 0.5, "components": {"accuracy": 0.2, "consistency": 0.2, "progress": 0.2, "novelty": 0.2, "cost": 0.8}, "rationale": "Poor"},
        ]
        
        mock_llm = MockLLM(expansion_responses, scoring_responses)
        
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm,
            llm_score=mock_llm,
            k_children=4,
            max_select=1,
            max_depth=2,
            max_nodes=100,
            target_score=1.0,  # Won't be reached
            beam_per_depth=2,  # Only keep top 2 at each depth
            min_score=0.0,
        )
        
        result = orchestrator.run(task="Test task")
        
        # Check that pruning occurred
        snapshot = result["tree_snapshot"]
        depth_1_nodes = [n for n in snapshot["nodes"].values() if n["depth"] == 1]
        
        # Should have 4 nodes created but only 2 kept (beam=2)
        # Note: all nodes are in snapshot, but pruned ones have status="pruned"
        assert len(depth_1_nodes) == 4
        open_or_expanded = [n for n in depth_1_nodes if n["status"] in ("open", "expanded")]
        assert len(open_or_expanded) == 2
        
        # Best 2 should be kept
        kept_scores = [n["score"] for n in open_or_expanded]
        kept_scores.sort(reverse=True)
        assert kept_scores == [0.8, 0.6]  # Top 2 scores

    def test_with_initial_state(self):
        """Test orchestrator with custom initial state."""
        expansion_responses = [
            [{"thought": "Continue from initial", "action": "continue", "partial_solution": "Extended from initial"}]
        ]
        
        scoring_responses = [
            {"score": 0.7, "confidence": 0.8, "components": {"accuracy": 0.7, "consistency": 0.7, "progress": 0.8, "novelty": 0.6, "cost": 0.3}, "rationale": "Good continuation"}
        ]
        
        mock_llm = MockLLM(expansion_responses, scoring_responses)
        
        orchestrator = ToTOrchestrator(
            llm_expand=mock_llm,
            llm_score=mock_llm,
            k_children=1,
            max_select=1,
            max_depth=2,
            max_nodes=10,
            target_score=0.65,  # Will be reached
            beam_per_depth=3,
            min_score=0.0,
        )
        
        initial_state = {
            "reasoning_steps": ["Given initial context"],
            "scratchpad": "Initial notes",
            "partial_solution": "Starting point"
        }
        
        result = orchestrator.run(
            task="Continue from this state",
            constraints="Build on initial work",
            initial_state=initial_state
        )
        
        # Check that initial state was preserved in path
        assert result["best_path"][0] == "Starting point"  # Root partial solution
        assert "Extended from initial" in result["best_path"]  # Child extended it
        
        # Check root node has initial state
        snapshot = result["tree_snapshot"]
        root_nodes = [n for n in snapshot["nodes"].values() if n["parent_id"] is None]
        assert len(root_nodes) == 1
        root_state = root_nodes[0]["state"]
        assert root_state["scratchpad"] == "Initial notes"
        assert "Given initial context" in root_state["reasoning_steps"]