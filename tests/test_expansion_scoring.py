"""Tests for expansion and scoring policies with mocked LLMs."""

import json
import pytest
from unittest.mock import Mock, patch

from langtree.models import ThoughtNode, TreeManager
from langtree.policies import ExpansionPolicy, ScoringPolicy


class MockLLM:
    """Mock LLM for testing."""
    
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0
    
    def bind(self, **kwargs):
        """Mock bind method."""
        return self
    
    def invoke(self, inputs):
        """Mock invoke method."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        else:
            raise Exception("No more mock responses available")


class TestExpansionPolicy:
    """Test ExpansionPolicy with mocked LLM."""
    
    def test_expand_success(self):
        """Test successful expansion with valid JSON response."""
        mock_responses = [
            [
                {"thought": "Try approach A", "action": "analyze", "partial_solution": "Step 1: Analysis"},
                {"thought": "Try approach B", "action": "synthesize", "partial_solution": "Step 1: Synthesis"},
                {"thought": "Try approach C", "action": "verify", "partial_solution": "Step 1: Verification"}
            ]
        ]
        
        mock_llm = MockLLM(mock_responses)
        policy = ExpansionPolicy(mock_llm, k=3)
        
        parent = ThoughtNode(
            id="parent",
            parent_id=None,
            depth=0,
            state={"reasoning_steps": ["Initial thought"], "partial_solution": "Initial"}
        )
        
        children = policy.expand(parent, task="Test task", constraints="Test constraints")
        
        assert len(children) == 3
        
        # Check first child
        child1 = children[0]
        assert child1.parent_id == "parent"
        assert child1.depth == 1
        assert child1.action == "analyze"
        assert child1.status == "open"
        assert "Try approach A" in child1.state["reasoning_steps"]
        assert child1.state["partial_solution"] == "Step 1: Analysis"
        
        # Check all children have unique IDs
        child_ids = [c.id for c in children]
        assert len(set(child_ids)) == 3

    def test_expand_with_fallback(self):
        """Test expansion fallback when LLM fails."""
        mock_llm = Mock()
        mock_llm.bind.return_value = mock_llm
        mock_llm.invoke.side_effect = Exception("LLM failed")
        
        policy = ExpansionPolicy(mock_llm, k=3)
        
        parent = ThoughtNode(
            id="parent",
            parent_id=None,
            depth=0,
            state={"partial_solution": "Previous work"}
        )
        
        children = policy.expand(parent, task="Test task")
        
        # Should get fallback child
        assert len(children) == 1
        child = children[0]
        assert child.action == "noop"
        assert "Expansion failed" in child.state["reasoning_steps"][0]
        assert child.state["partial_solution"] == "Previous work"

    def test_expand_preserves_state(self):
        """Test that expansion preserves and extends parent state."""
        mock_responses = [
            [{"thought": "Next step", "action": "continue", "partial_solution": "Extended solution"}]
        ]
        
        mock_llm = MockLLM(mock_responses)
        policy = ExpansionPolicy(mock_llm, k=1)
        
        parent = ThoughtNode(
            id="parent",
            parent_id=None,
            depth=1,
            state={
                "reasoning_steps": ["Step 1", "Step 2"],
                "scratchpad": "Working notes",
                "partial_solution": "Previous solution"
            }
        )
        
        children = policy.expand(parent, task="Test task")
        
        child = children[0]
        assert child.depth == 2
        assert child.state["reasoning_steps"] == ["Step 1", "Step 2", "Next step"]
        assert child.state["scratchpad"] == "Working notes"  # Preserved
        assert child.state["partial_solution"] == "Extended solution"  # Updated

    def test_expand_limits_children(self):
        """Test that expansion respects k parameter even with more responses."""
        mock_responses = [
            [
                {"thought": "A", "action": "a", "partial_solution": "1"},
                {"thought": "B", "action": "b", "partial_solution": "2"},
                {"thought": "C", "action": "c", "partial_solution": "3"},
                {"thought": "D", "action": "d", "partial_solution": "4"},  # Should be ignored
            ]
        ]
        
        mock_llm = MockLLM(mock_responses)
        policy = ExpansionPolicy(mock_llm, k=2)  # Only want 2 children
        
        parent = ThoughtNode(id="parent", parent_id=None, depth=0)
        children = policy.expand(parent, task="Test task")
        
        assert len(children) == 2
        assert children[0].action == "a"
        assert children[1].action == "b"


class TestScoringPolicy:
    """Test ScoringPolicy with mocked LLM."""
    
    def test_score_success(self):
        """Test successful scoring with valid JSON response."""
        mock_responses = [
            {
                "score": 0.75,
                "confidence": 0.85,
                "components": {
                    "accuracy": 0.8,
                    "consistency": 0.7,
                    "progress": 0.9,
                    "novelty": 0.6,
                    "cost": 0.5
                },
                "rationale": "Good progress with minor issues"
            }
        ]
        
        mock_llm = MockLLM(mock_responses)
        policy = ScoringPolicy(mock_llm)
        
        tree = TreeManager()
        parent = ThoughtNode(id="parent", parent_id=None, depth=0, state={"prev": "data"})
        tree.add_node(parent)
        
        node = ThoughtNode(
            id="node",
            parent_id="parent",
            depth=1,
            state={"current": "data"}
        )
        tree.add_node(node)
        
        score, conf, comps, rationale = policy.score(node, tree, task="Test task")
        
        # Check that weighted score is computed correctly
        expected_score = (
            0.4 * 0.8 +    # accuracy
            0.2 * 0.7 +    # consistency  
            0.25 * 0.9 +   # progress
            0.1 * 0.6 +    # novelty
            -0.05 * 0.5    # cost (negative weight)
        )
        assert abs(score - expected_score) < 0.001
        assert conf == 0.85
        assert comps["accuracy"] == 0.8
        assert rationale == "Good progress with minor issues"

    def test_score_with_fallback(self):
        """Test scoring fallback when LLM fails."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM failed")
        
        policy = ScoringPolicy(mock_llm)
        
        tree = TreeManager()
        parent = ThoughtNode(id="parent", parent_id=None, depth=0)
        tree.add_node(parent)
        
        node = ThoughtNode(id="node", parent_id="parent", depth=1)
        tree.add_node(node)
        
        score, conf, comps, rationale = policy.score(node, tree, task="Test task")
        
        # Should get fallback values
        assert score == 0.3
        assert conf == 0.3
        assert "LLM grading failed" in rationale
        assert "accuracy" in comps
        assert "consistency" in comps

    def test_score_bounds_checking(self):
        """Test that scores are bounded to [0,1]."""
        # Test score above 1.0
        mock_responses = [
            {
                "score": 1.5,  # Will be ignored, weighted score used instead
                "confidence": 0.8,
                "components": {
                    "accuracy": 1.0,
                    "consistency": 1.0,
                    "progress": 1.0,
                    "novelty": 1.0,
                    "cost": 0.0  # This should result in score > 1.0 before bounding
                },
                "rationale": "Perfect score"
            }
        ]
        
        mock_llm = MockLLM(mock_responses)
        policy = ScoringPolicy(mock_llm)
        
        tree = TreeManager()
        parent = ThoughtNode(id="parent", parent_id=None, depth=0)
        tree.add_node(parent)
        
        node = ThoughtNode(id="node", parent_id="parent", depth=1)
        tree.add_node(node)
        
        score, conf, comps, rationale = policy.score(node, tree, task="Test task")
        
        # Score should be bounded to 1.0
        assert score <= 1.0
        assert score >= 0.0

    def test_score_custom_weights(self):
        """Test scoring with custom component weights."""
        mock_responses = [
            {
                "score": 0.5,  # Ignored
                "confidence": 0.8,
                "components": {
                    "accuracy": 1.0,
                    "consistency": 0.0,
                    "progress": 0.0,
                    "novelty": 0.0,
                    "cost": 0.0
                },
                "rationale": "Only accurate"
            }
        ]
        
        custom_weights = {
            "accuracy": 1.0,  # Only accuracy matters
            "consistency": 0.0,
            "progress": 0.0,
            "novelty": 0.0,
            "cost": 0.0
        }
        
        mock_llm = MockLLM(mock_responses)
        policy = ScoringPolicy(mock_llm, weights=custom_weights)
        
        tree = TreeManager()
        parent = ThoughtNode(id="parent", parent_id=None, depth=0)
        tree.add_node(parent)
        
        node = ThoughtNode(id="node", parent_id="parent", depth=1)
        tree.add_node(node)
        
        score, conf, comps, rationale = policy.score(node, tree, task="Test task")
        
        # Score should be 1.0 * 1.0 = 1.0 (only accuracy component)
        assert abs(score - 1.0) < 0.001

    def test_score_handles_missing_components(self):
        """Test scoring handles missing component values gracefully."""
        mock_responses = [
            {
                "score": 0.5,
                "confidence": 0.8,
                "components": {
                    "accuracy": 0.7,
                    # Missing consistency, progress, novelty, cost
                },
                "rationale": "Incomplete components"
            }
        ]
        
        mock_llm = MockLLM(mock_responses)
        policy = ScoringPolicy(mock_llm)
        
        tree = TreeManager()
        parent = ThoughtNode(id="parent", parent_id=None, depth=0)
        tree.add_node(parent)
        
        node = ThoughtNode(id="node", parent_id="parent", depth=1)
        tree.add_node(node)
        
        score, conf, comps, rationale = policy.score(node, tree, task="Test task")
        
        # Should handle missing values gracefully (default to 0.0)
        expected_score = 0.4 * 0.7  # Only accuracy component
        assert abs(score - expected_score) < 0.001
        assert comps["accuracy"] == 0.7
        assert comps["consistency"] == 0.0  # Default value

    def test_score_root_node(self):
        """Test scoring a root node (no parent)."""
        mock_responses = [
            {
                "score": 0.6,
                "confidence": 0.9,
                "components": {"accuracy": 0.6, "consistency": 0.6, "progress": 0.6, "novelty": 0.6, "cost": 0.6},
                "rationale": "Root evaluation"
            }
        ]
        
        mock_llm = MockLLM(mock_responses)
        policy = ScoringPolicy(mock_llm)
        
        tree = TreeManager()
        root = ThoughtNode(id="root", parent_id=None, depth=0, state={"initial": "state"})
        tree.add_node(root)
        
        score, conf, comps, rationale = policy.score(root, tree, task="Test task")
        
        # Should work even without parent
        assert isinstance(score, float)
        assert isinstance(conf, float)
        assert isinstance(comps, dict)
        assert isinstance(rationale, str)