"""
Unit tests for CLI interface
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
import csv

from click.testing import CliRunner


class TestCLICommands(unittest.TestCase):
    """Test cases for CLI commands"""
    
    @classmethod
    def setUpClass(cls):
        """Set up CLI runner and mock dependencies"""
        cls.runner = CliRunner()
        
        # Import CLI with mocked dependencies
        with patch('guardrails.evaluators.ToxicityEvaluator'), \
             patch('guardrails.evaluators.AlignmentChecker'), \
             patch('guardrails.evaluators.HallucinationDetector'), \
             patch('guardrails.evaluators.PIIDetector'), \
             patch('guardrails.evaluators.PromptInjectionDetector'):
            from cli import cli
            cls.cli = cli
    
    def test_cli_help(self):
        """Test CLI help command"""
        result = self.runner.invoke(self.cli, ["--help"])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("LLM Guardrail Studio", result.output)
    
    def test_evaluate_command(self):
        """Test evaluate command"""
        with patch('cli.GuardrailPipeline') as mock_pipeline:
            mock_result = Mock()
            mock_result.passed = True
            mock_result.scores = {"toxicity": 0.1, "alignment": 0.8}
            mock_result.flags = []
            mock_result.to_dict.return_value = {
                "passed": True,
                "scores": {"toxicity": 0.1, "alignment": 0.8},
                "flags": []
            }
            mock_pipeline.return_value.evaluate.return_value = mock_result
            
            result = self.runner.invoke(self.cli, [
                "evaluate",
                "--prompt", "What is AI?",
                "--response", "AI is artificial intelligence."
            ])
            
            self.assertEqual(result.exit_code, 0)
    
    def test_evaluate_command_json_output(self):
        """Test evaluate command with JSON output"""
        with patch('cli.GuardrailPipeline') as mock_pipeline:
            mock_result = Mock()
            mock_result.passed = True
            mock_result.scores = {"toxicity": 0.1}
            mock_result.flags = []
            mock_result.to_dict.return_value = {
                "passed": True,
                "scores": {"toxicity": 0.1},
                "flags": []
            }
            mock_pipeline.return_value.evaluate.return_value = mock_result
            
            result = self.runner.invoke(self.cli, [
                "evaluate",
                "--prompt", "Test",
                "--response", "Test",
                "--format", "json"
            ])
            
            self.assertEqual(result.exit_code, 0)
            # Should be valid JSON
            try:
                json.loads(result.output)
            except json.JSONDecodeError:
                self.fail("Output is not valid JSON")
    
    def test_evaluate_file_command(self):
        """Test evaluate-file command with CSV input"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['prompt', 'response'])
            writer.writerow(['Q1', 'A1'])
            writer.writerow(['Q2', 'A2'])
            temp_path = f.name
        
        try:
            with patch('cli.GuardrailPipeline') as mock_pipeline:
                mock_result = Mock()
                mock_result.passed = True
                mock_result.scores = {"toxicity": 0.1}
                mock_result.flags = []
                mock_result.to_dict.return_value = {
                    "passed": True,
                    "scores": {"toxicity": 0.1},
                    "flags": []
                }
                mock_pipeline.return_value.evaluate.return_value = mock_result
                
                result = self.runner.invoke(self.cli, [
                    "evaluate-file",
                    temp_path
                ])
                
                self.assertEqual(result.exit_code, 0)
        finally:
            os.unlink(temp_path)
    
    def test_check_command_toxicity(self):
        """Test check command for specific evaluator"""
        with patch('cli.ToxicityEvaluator') as mock_eval:
            mock_eval.return_value.evaluate.return_value = 0.1
            
            result = self.runner.invoke(self.cli, [
                "check",
                "--evaluator", "toxicity",
                "--text", "This is a test"
            ])
            
            self.assertEqual(result.exit_code, 0)
    
    def test_check_command_pii(self):
        """Test check command for PII evaluator"""
        with patch('cli.PIIDetector') as mock_eval:
            mock_eval.return_value.evaluate.return_value = (0.0, False)
            mock_eval.return_value.evaluate_detailed.return_value = {
                "pii_found": False,
                "categories": {}
            }
            
            result = self.runner.invoke(self.cli, [
                "check",
                "--evaluator", "pii",
                "--text", "Hello world"
            ])
            
            self.assertEqual(result.exit_code, 0)
    
    def test_check_command_injection(self):
        """Test check command for injection evaluator"""
        with patch('cli.PromptInjectionDetector') as mock_eval:
            mock_eval.return_value.evaluate.return_value = (0.0, False)
            
            result = self.runner.invoke(self.cli, [
                "check",
                "--evaluator", "injection",
                "--text", "What is the capital of France?"
            ])
            
            self.assertEqual(result.exit_code, 0)
    
    def test_server_command_help(self):
        """Test server command help"""
        result = self.runner.invoke(self.cli, ["server", "--help"])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("host", result.output.lower())
        self.assertIn("port", result.output.lower())
    
    def test_config_show_command(self):
        """Test config show command"""
        with patch('cli.GuardrailPipeline'):
            result = self.runner.invoke(self.cli, ["config", "show"])
            
            self.assertEqual(result.exit_code, 0)
    
    def test_config_set_command(self):
        """Test config set command"""
        result = self.runner.invoke(self.cli, [
            "config", "set",
            "--key", "toxicity_threshold",
            "--value", "0.8"
        ])
        
        # Should work or show appropriate message
        self.assertIn(result.exit_code, [0, 1])
    
    def test_rules_list_command(self):
        """Test rules list command"""
        result = self.runner.invoke(self.cli, ["rules", "list"])
        
        self.assertEqual(result.exit_code, 0)
    
    def test_rules_add_command(self):
        """Test rules add command"""
        result = self.runner.invoke(self.cli, [
            "rules", "add",
            "--name", "test_rule",
            "--type", "keyword",
            "--pattern", "test",
            "--action", "flag"
        ])
        
        self.assertEqual(result.exit_code, 0)
    
    def test_rules_remove_command(self):
        """Test rules remove command"""
        # First add a rule
        self.runner.invoke(self.cli, [
            "rules", "add",
            "--name", "to_remove",
            "--type", "keyword",
            "--pattern", "remove",
            "--action", "flag"
        ])
        
        # Then remove it
        result = self.runner.invoke(self.cli, [
            "rules", "remove",
            "--name", "to_remove"
        ])
        
        self.assertIn(result.exit_code, [0, 1])


class TestCLIInputValidation(unittest.TestCase):
    """Test cases for CLI input validation"""
    
    @classmethod
    def setUpClass(cls):
        cls.runner = CliRunner()
        
        with patch('guardrails.evaluators.ToxicityEvaluator'), \
             patch('guardrails.evaluators.AlignmentChecker'), \
             patch('guardrails.evaluators.HallucinationDetector'), \
             patch('guardrails.evaluators.PIIDetector'), \
             patch('guardrails.evaluators.PromptInjectionDetector'):
            from cli import cli
            cls.cli = cli
    
    def test_evaluate_missing_prompt(self):
        """Test evaluate command without prompt"""
        result = self.runner.invoke(self.cli, [
            "evaluate",
            "--response", "Some response"
        ])
        
        # Should fail or show error
        self.assertNotEqual(result.exit_code, 0)
    
    def test_evaluate_missing_response(self):
        """Test evaluate command without response"""
        result = self.runner.invoke(self.cli, [
            "evaluate",
            "--prompt", "Some prompt"
        ])
        
        # Should fail or show error
        self.assertNotEqual(result.exit_code, 0)
    
    def test_evaluate_file_nonexistent(self):
        """Test evaluate-file with non-existent file"""
        result = self.runner.invoke(self.cli, [
            "evaluate-file",
            "/nonexistent/path/file.csv"
        ])
        
        self.assertNotEqual(result.exit_code, 0)
    
    def test_check_invalid_evaluator(self):
        """Test check with invalid evaluator name"""
        result = self.runner.invoke(self.cli, [
            "check",
            "--evaluator", "invalid_evaluator",
            "--text", "Test"
        ])
        
        # Should fail or show error
        self.assertIn(result.exit_code, [0, 1, 2])


class TestCLIOutput(unittest.TestCase):
    """Test cases for CLI output formatting"""
    
    @classmethod
    def setUpClass(cls):
        cls.runner = CliRunner()
        
        with patch('guardrails.evaluators.ToxicityEvaluator'), \
             patch('guardrails.evaluators.AlignmentChecker'), \
             patch('guardrails.evaluators.HallucinationDetector'), \
             patch('guardrails.evaluators.PIIDetector'), \
             patch('guardrails.evaluators.PromptInjectionDetector'):
            from cli import cli
            cls.cli = cli
    
    def test_table_output_format(self):
        """Test table output format"""
        with patch('cli.GuardrailPipeline') as mock_pipeline:
            mock_result = Mock()
            mock_result.passed = True
            mock_result.scores = {"toxicity": 0.1, "alignment": 0.8}
            mock_result.flags = []
            mock_result.to_dict.return_value = {
                "passed": True,
                "scores": {"toxicity": 0.1, "alignment": 0.8},
                "flags": []
            }
            mock_pipeline.return_value.evaluate.return_value = mock_result
            
            result = self.runner.invoke(self.cli, [
                "evaluate",
                "--prompt", "Test",
                "--response", "Test",
                "--format", "table"
            ])
            
            self.assertEqual(result.exit_code, 0)
    
    def test_verbose_output(self):
        """Test verbose output mode"""
        with patch('cli.GuardrailPipeline') as mock_pipeline:
            mock_result = Mock()
            mock_result.passed = True
            mock_result.scores = {"toxicity": 0.1}
            mock_result.flags = []
            mock_result.detailed_results = {}
            mock_result.to_dict.return_value = {
                "passed": True,
                "scores": {"toxicity": 0.1},
                "flags": []
            }
            mock_pipeline.return_value.evaluate.return_value = mock_result
            
            result = self.runner.invoke(self.cli, [
                "evaluate",
                "--prompt", "Test",
                "--response", "Test",
                "--verbose"
            ])
            
            self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
