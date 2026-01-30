"""
Unit tests for guardrail pipeline
"""

import unittest
from unittest.mock import Mock, patch

from guardrails.pipeline import GuardrailPipeline, GuardrailResult


class TestGuardrailResult(unittest.TestCase):
    """Test cases for GuardrailResult"""
    
    def test_initialization(self):
        """Test result initialization"""
        result = GuardrailResult()
        self.assertEqual(result.scores, {})
        self.assertEqual(result.flags, [])
        self.assertTrue(result.passed)
    
    def test_add_score(self):
        """Test adding scores"""
        result = GuardrailResult()
        result.add_score("toxicity", 0.5)
        self.assertEqual(result.scores["toxicity"], 0.5)
    
    def test_add_flag(self):
        """Test adding flags"""
        result = GuardrailResult()
        result.add_flag("High toxicity detected")
        self.assertIn("High toxicity detected", result.flags)
        self.assertFalse(result.passed)


class TestGuardrailPipeline(unittest.TestCase):
    """Test cases for GuardrailPipeline"""
    
    def setUp(self):
        """Set up test pipeline"""
        with patch('guardrails.pipeline.ToxicityEvaluator'), \
             patch('guardrails.pipeline.HallucinationDetector'), \
             patch('guardrails.pipeline.AlignmentChecker'):
            self.pipeline = GuardrailPipeline()
    
    def test_initialization(self):
        """Test pipeline initialization"""
        self.assertEqual(len(self.pipeline.evaluators), 3)
        self.assertIn('toxicity', self.pipeline.evaluators)
        self.assertIn('hallucination', self.pipeline.evaluators)
        self.assertIn('alignment', self.pipeline.evaluators)
    
    def test_initialization_selective_evaluators(self):
        """Test pipeline with selective evaluators"""
        with patch('guardrails.pipeline.ToxicityEvaluator'):
            pipeline = GuardrailPipeline(
                enable_toxicity=True,
                enable_hallucination=False,
                enable_alignment=False
            )
            self.assertEqual(len(pipeline.evaluators), 1)
            self.assertIn('toxicity', pipeline.evaluators)
    
    def test_evaluate_safe_content(self):
        """Test evaluation of safe content"""
        # Mock evaluators
        mock_toxicity = Mock()
        mock_toxicity.evaluate.return_value = 0.1
        
        mock_alignment = Mock()
        mock_alignment.evaluate.return_value = 0.8
        
        mock_hallucination = Mock()
        mock_hallucination.evaluate.return_value = 0.2
        
        self.pipeline.evaluators = {
            'toxicity': mock_toxicity,
            'alignment': mock_alignment,
            'hallucination': mock_hallucination
        }
        
        result = self.pipeline.evaluate("What is AI?", "AI is artificial intelligence.")
        
        self.assertTrue(result.passed)
        self.assertEqual(len(result.flags), 0)
        self.assertEqual(result.scores['toxicity'], 0.1)
        self.assertEqual(result.scores['alignment'], 0.8)
        self.assertEqual(result.scores['hallucination_risk'], 0.2)
    
    def test_evaluate_toxic_content(self):
        """Test evaluation of toxic content"""
        # Mock evaluators
        mock_toxicity = Mock()
        mock_toxicity.evaluate.return_value = 0.9  # High toxicity
        
        mock_alignment = Mock()
        mock_alignment.evaluate.return_value = 0.8
        
        mock_hallucination = Mock()
        mock_hallucination.evaluate.return_value = 0.2
        
        self.pipeline.evaluators = {
            'toxicity': mock_toxicity,
            'alignment': mock_alignment,
            'hallucination': mock_hallucination
        }
        
        result = self.pipeline.evaluate("Prompt", "Toxic response")
        
        self.assertFalse(result.passed)
        self.assertGreater(len(result.flags), 0)
        self.assertIn("High toxicity detected", result.flags[0])
    
    def test_evaluate_misaligned_content(self):
        """Test evaluation of misaligned content"""
        # Mock evaluators
        mock_toxicity = Mock()
        mock_toxicity.evaluate.return_value = 0.1
        
        mock_alignment = Mock()
        mock_alignment.evaluate.return_value = 0.2  # Low alignment
        
        mock_hallucination = Mock()
        mock_hallucination.evaluate.return_value = 0.2
        
        self.pipeline.evaluators = {
            'toxicity': mock_toxicity,
            'alignment': mock_alignment,
            'hallucination': mock_hallucination
        }
        
        result = self.pipeline.evaluate("What is AI?", "The weather is sunny.")
        
        self.assertFalse(result.passed)
        self.assertGreater(len(result.flags), 0)
        self.assertIn("Low prompt-response alignment", result.flags[0])
    
    def test_evaluate_hallucination_risk(self):
        """Test evaluation with hallucination risk"""
        # Mock evaluators
        mock_toxicity = Mock()
        mock_toxicity.evaluate.return_value = 0.1
        
        mock_alignment = Mock()
        mock_alignment.evaluate.return_value = 0.8
        
        mock_hallucination = Mock()
        mock_hallucination.evaluate.return_value = 0.8  # High hallucination risk
        
        self.pipeline.evaluators = {
            'toxicity': mock_toxicity,
            'alignment': mock_alignment,
            'hallucination': mock_hallucination
        }
        
        result = self.pipeline.evaluate("Question", "Uncertain response")
        
        self.assertFalse(result.passed)
        self.assertGreater(len(result.flags), 0)
        self.assertIn("Potential hallucination detected", result.flags[0])


if __name__ == '__main__':
    unittest.main()