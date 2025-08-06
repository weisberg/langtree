"""Tests for CLI module."""

import argparse
import json
import sys
from io import StringIO
from unittest.mock import Mock, patch, MagicMock
import pytest

from langtree.cli import main, build_llm, pretty_print_result


class TestArgumentParsing:
    """Test CLI argument parsing."""
    
    def test_argument_parsing_required_task(self):
        """Test that task argument is required."""
        with patch('sys.argv', ['langtree']):
            with pytest.raises(SystemExit):
                with patch('sys.stderr', new_callable=StringIO):
                    main()
    
    def test_argument_parsing_basic(self):
        """Test basic argument parsing with required task."""
        test_args = [
            'langtree',
            '--task', 'Test task description'
        ]
        
        with patch('sys.argv', test_args):
            with patch('langtree.cli.build_llm') as mock_build_llm:
                with patch('langtree.cli.ToTOrchestrator') as mock_orchestrator:
                    mock_llm = Mock()
                    mock_build_llm.return_value = mock_llm
                    mock_orch_instance = Mock()
                    mock_orch_instance.run.return_value = {
                        "best_score": 0.8,
                        "best_confidence": 0.9,
                        "best_state": {"result": "test"},
                        "best_actions": ["action1"],
                        "best_path": ["path1"]
                    }
                    mock_orchestrator.return_value = mock_orch_instance
                    
                    # Capture stdout
                    with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                        main()
                    
                    # Should have called orchestrator with task
                    mock_orch_instance.run.assert_called_once()
                    call_args = mock_orch_instance.run.call_args
                    assert call_args[1]['task'] == 'Test task description'
    
    def test_argument_parsing_all_options(self):
        """Test parsing all available CLI options."""
        test_args = [
            'langtree',
            '--task', 'Complex task',
            '--constraints', 'Test constraints',
            '--model-expand', 'gpt-4',
            '--model-score', 'gpt-3.5-turbo',
            '--k-children', '5',
            '--max-select', '2',
            '--max-depth', '6',
            '--max-nodes', '100',
            '--target-score', '0.85',
            '--beam', '4',
            '--min-score', '0.1',
            '--print-tree'
        ]
        
        with patch('sys.argv', test_args):
            with patch('langtree.cli.build_llm') as mock_build_llm:
                with patch('langtree.cli.ToTOrchestrator') as mock_orchestrator:
                    mock_llm = Mock()
                    mock_build_llm.return_value = mock_llm
                    mock_orch_instance = Mock()
                    mock_orch_instance.run.return_value = {
                        "best_score": 0.8,
                        "best_confidence": 0.9,
                        "best_state": {"result": "test"},
                        "best_actions": ["action1"],
                        "best_path": ["path1"],
                        "tree_snapshot": {"nodes": {"root": {"parent_id": None, "children": []}}}
                    }
                    mock_orchestrator.return_value = mock_orch_instance
                    
                    with patch('langtree.cli.pretty_print_result') as mock_pretty_print:
                        main()
                    
                    # Verify orchestrator created with correct parameters
                    mock_orchestrator.assert_called_once()
                    call_args = mock_orchestrator.call_args[1]
                    assert call_args['k_children'] == 5
                    assert call_args['max_select'] == 2
                    assert call_args['max_depth'] == 6
                    assert call_args['max_nodes'] == 100
                    assert call_args['target_score'] == 0.85
                    assert call_args['beam_per_depth'] == 4
                    assert call_args['min_score'] == 0.1
                    
                    # Verify pretty print was called (due to --print-tree)
                    mock_pretty_print.assert_called_once()
    
    def test_argument_parsing_invalid_values(self):
        """Test argument parsing with invalid values."""
        test_args = [
            'langtree',
            '--task', 'Test task',
            '--k-children', 'not-a-number'
        ]
        
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit):
                with patch('sys.stderr', new_callable=StringIO):
                    main()


class TestBuildLLM:
    """Test LLM building functionality."""
    
    @patch('langtree.cli.ChatOpenAI')
    def test_build_llm_defaults(self, mock_chat_openai):
        """Test build_llm with default parameters."""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        result = build_llm()
        
        mock_chat_openai.assert_called_once_with(model="gpt-4o-mini", temperature=0.2)
        assert result == mock_llm
    
    @patch('langtree.cli.ChatOpenAI')
    def test_build_llm_custom_params(self, mock_chat_openai):
        """Test build_llm with custom parameters."""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        result = build_llm("gpt-4", 0.8)
        
        mock_chat_openai.assert_called_once_with(model="gpt-4", temperature=0.8)
        assert result == mock_llm
    
    @patch('langtree.cli.ChatOpenAI')
    def test_build_llm_exception_handling(self, mock_chat_openai):
        """Test build_llm handles exceptions from ChatOpenAI."""
        mock_chat_openai.side_effect = Exception("API key missing")
        
        with pytest.raises(Exception, match="API key missing"):
            build_llm()


class TestPrettyPrintResult:
    """Test result formatting and printing."""
    
    def test_pretty_print_basic(self):
        """Test basic pretty printing functionality."""
        result = {
            "best_score": 0.75,
            "best_confidence": 0.85,
            "best_actions": ["analyze", "synthesize"],
            "best_path": ["initial", "analysis", "synthesis"],
            "tree_snapshot": {
                "nodes": {
                    "root": {
                        "parent_id": None,
                        "depth": 0,
                        "score": 0.75,
                        "confidence": 0.85,
                        "status": "expanded",
                        "children": ["child1"]
                    },
                    "child1": {
                        "parent_id": "root",
                        "depth": 1,
                        "score": 0.6,
                        "confidence": 0.7,
                        "status": "open",
                        "children": []
                    }
                }
            }
        }
        
        with patch('langtree.cli.rprint') as mock_rprint:
            with patch('langtree.cli.RichTree') as mock_rich_tree:
                mock_tree_instance = Mock()
                mock_rich_tree.return_value = mock_tree_instance
                
                pretty_print_result(result)
                
                # Should print score and confidence
                assert any("0.750" in str(call) for call in mock_rprint.call_args_list)
                assert any("0.85" in str(call) for call in mock_rprint.call_args_list)
                
                # Should print actions and paths
                assert any("analyze" in str(call) for call in mock_rprint.call_args_list)
                assert any("synthesis" in str(call) for call in mock_rprint.call_args_list)
    
    def test_pretty_print_without_rich_tree(self):
        """Test pretty printing when RichTree is not available."""
        result = {
            "best_score": 0.75,
            "best_confidence": 0.85,
            "best_actions": ["analyze"],
            "best_path": ["initial"],
            "tree_snapshot": {"nodes": {}}
        }
        
        with patch('langtree.cli.rprint') as mock_rprint:
            with patch('langtree.cli.RichTree', None):  # Simulate RichTree not available
                pretty_print_result(result)
                
                # Should still print basic info without tree
                assert mock_rprint.called
                assert any("0.750" in str(call) for call in mock_rprint.call_args_list)
    
    def test_pretty_print_complex_tree(self):
        """Test pretty printing with complex tree structure."""
        result = {
            "best_score": 0.9,
            "best_confidence": 0.95,
            "best_actions": ["a1", "a2", "a3"],
            "best_path": ["p1", "p2", "p3"],
            "tree_snapshot": {
                "nodes": {
                    "root": {"parent_id": None, "depth": 0, "score": 0.5, "confidence": 0.6, "status": "expanded", "children": ["c1", "c2"]},
                    "c1": {"parent_id": "root", "depth": 1, "score": 0.7, "confidence": 0.8, "status": "expanded", "children": ["c3"]},
                    "c2": {"parent_id": "root", "depth": 1, "score": 0.6, "confidence": 0.7, "status": "pruned", "children": []},
                    "c3": {"parent_id": "c1", "depth": 2, "score": 0.9, "confidence": 0.95, "status": "terminal", "children": []}
                }
            }
        }
        
        with patch('langtree.cli.rprint') as mock_rprint:
            with patch('langtree.cli.RichTree') as mock_rich_tree:
                mock_tree_instance = Mock()
                mock_child1 = Mock()
                mock_child2 = Mock()
                mock_tree_instance.add.side_effect = [mock_child1, mock_child2]
                mock_rich_tree.return_value = mock_tree_instance
                
                pretty_print_result(result)
                
                # Should create tree and add nodes
                mock_rich_tree.assert_called_once_with("Tree-of-Thought")
                assert mock_tree_instance.add.call_count >= 1


class TestMainExecution:
    """Test main function execution scenarios."""
    
    def test_main_json_output_mode(self):
        """Test main function with JSON output (no --print-tree)."""
        test_args = ['langtree', '--task', 'Test task']
        
        with patch('sys.argv', test_args):
            with patch('langtree.cli.build_llm') as mock_build_llm:
                with patch('langtree.cli.ToTOrchestrator') as mock_orchestrator:
                    mock_llm = Mock()
                    mock_build_llm.return_value = mock_llm
                    mock_orch_instance = Mock()
                    mock_result = {
                        "best_score": 0.8,
                        "best_confidence": 0.9,
                        "best_state": {"key": "value"},
                        "best_actions": ["action1", "action2"],
                        "best_path": ["path1", "path2"]
                    }
                    mock_orch_instance.run.return_value = mock_result
                    mock_orchestrator.return_value = mock_orch_instance
                    
                    with patch('builtins.print') as mock_print:
                        main()
                    
                    # Should print JSON output
                    mock_print.assert_called_once()
                    printed_json = mock_print.call_args[0][0]
                    parsed = json.loads(printed_json)
                    assert parsed["best_score"] == 0.8
                    assert parsed["best_confidence"] == 0.9
    
    def test_main_with_constraints(self):
        """Test main function with constraints parameter."""
        test_args = [
            'langtree',
            '--task', 'Test task',
            '--constraints', 'Test constraints'
        ]
        
        with patch('sys.argv', test_args):
            with patch('langtree.cli.build_llm') as mock_build_llm:
                with patch('langtree.cli.ToTOrchestrator') as mock_orchestrator:
                    mock_llm = Mock()
                    mock_build_llm.return_value = mock_llm
                    mock_orch_instance = Mock()
                    mock_orch_instance.run.return_value = {
                        "best_score": 0.8,
                        "best_confidence": 0.9,
                        "best_state": {},
                        "best_actions": [],
                        "best_path": []
                    }
                    mock_orchestrator.return_value = mock_orch_instance
                    
                    with patch('sys.stdout', new_callable=StringIO):
                        main()
                    
                    # Verify constraints passed to orchestrator
                    call_args = mock_orch_instance.run.call_args
                    assert call_args[1]['constraints'] == 'Test constraints'
    
    def test_main_different_models(self):
        """Test main function with different expand and score models."""
        test_args = [
            'langtree',
            '--task', 'Test task',
            '--model-expand', 'gpt-4',
            '--model-score', 'gpt-3.5-turbo'
        ]
        
        with patch('sys.argv', test_args):
            with patch('langtree.cli.build_llm') as mock_build_llm:
                with patch('langtree.cli.ToTOrchestrator') as mock_orchestrator:
                    mock_llm_expand = Mock()
                    mock_llm_score = Mock()
                    mock_build_llm.side_effect = [mock_llm_expand, mock_llm_score]
                    
                    mock_orch_instance = Mock()
                    mock_orch_instance.run.return_value = {
                        "best_score": 0.8,
                        "best_confidence": 0.9,
                        "best_state": {},
                        "best_actions": [],
                        "best_path": []
                    }
                    mock_orchestrator.return_value = mock_orch_instance
                    
                    with patch('sys.stdout', new_callable=StringIO):
                        main()
                    
                    # Verify different models were built
                    assert mock_build_llm.call_count == 2
                    mock_build_llm.assert_any_call('gpt-4', temperature=0.7)
                    mock_build_llm.assert_any_call('gpt-3.5-turbo', temperature=0.0)
                    
                    # Verify different LLMs passed to orchestrator
                    mock_orchestrator.assert_called_once()
                    call_args = mock_orchestrator.call_args[1]
                    assert call_args['llm_expand'] == mock_llm_expand
                    assert call_args['llm_score'] == mock_llm_score


class TestErrorHandling:
    """Test CLI error handling scenarios."""
    
    def test_main_missing_openai_key(self):
        """Test handling of missing OpenAI API key."""
        test_args = ['langtree', '--task', 'Test task']
        
        with patch('sys.argv', test_args):
            with patch('langtree.cli.build_llm') as mock_build_llm:
                mock_build_llm.side_effect = Exception("API key not found")
                
                with pytest.raises(Exception, match="API key not found"):
                    main()
    
    def test_main_orchestrator_exception(self):
        """Test handling of orchestrator exceptions."""
        test_args = ['langtree', '--task', 'Test task']
        
        with patch('sys.argv', test_args):
            with patch('langtree.cli.build_llm') as mock_build_llm:
                with patch('langtree.cli.ToTOrchestrator') as mock_orchestrator:
                    mock_llm = Mock()
                    mock_build_llm.return_value = mock_llm
                    mock_orch_instance = Mock()
                    mock_orch_instance.run.side_effect = Exception("Orchestration failed")
                    mock_orchestrator.return_value = mock_orch_instance
                    
                    with pytest.raises(Exception, match="Orchestration failed"):
                        main()
    
    def test_main_invalid_model_names(self):
        """Test handling of invalid model names."""
        test_args = [
            'langtree',
            '--task', 'Test task',
            '--model-expand', 'invalid-model'
        ]
        
        with patch('sys.argv', test_args):
            with patch('langtree.cli.build_llm') as mock_build_llm:
                mock_build_llm.side_effect = Exception("Invalid model name")
                
                with pytest.raises(Exception, match="Invalid model name"):
                    main()
    
    def test_pretty_print_missing_rich(self):
        """Test pretty printing gracefully handles missing rich library."""
        result = {
            "best_score": 0.8,
            "best_confidence": 0.9,
            "best_actions": ["test"],
            "best_path": ["test"],
            "tree_snapshot": {"nodes": {}}
        }
        
        # Mock rich being unavailable
        with patch('langtree.cli.RichTree', None):
            with patch('langtree.cli.rprint') as mock_rprint:
                pretty_print_result(result)
                
                # Should still work with regular print
                assert mock_rprint.called
    
    def test_main_json_serialization_error(self):
        """Test handling of JSON serialization errors."""
        test_args = ['langtree', '--task', 'Test task']
        
        with patch('sys.argv', test_args):
            with patch('langtree.cli.build_llm') as mock_build_llm:
                with patch('langtree.cli.ToTOrchestrator') as mock_orchestrator:
                    mock_llm = Mock()
                    mock_build_llm.return_value = mock_llm
                    mock_orch_instance = Mock()
                    # Return a result with non-serializable object
                    class NonSerializable:
                        pass
                    mock_orch_instance.run.return_value = {
                        "best_score": 0.8,
                        "best_confidence": 0.9,
                        "best_state": {"non_serializable": NonSerializable()},
                        "best_actions": [],
                        "best_path": []
                    }
                    mock_orchestrator.return_value = mock_orch_instance
                    
                    with patch('json.dumps') as mock_json_dumps:
                        mock_json_dumps.side_effect = TypeError("Object not serializable")
                        
                        with pytest.raises(TypeError, match="Object not serializable"):
                            main()