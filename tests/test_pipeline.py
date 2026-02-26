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


class TestGuardrailResultSerialization(unittest.TestCase):
    """Test cases for GuardrailResult serialization"""
    
    def test_to_dict(self):
        """Test serialization to dictionary"""
        result = GuardrailResult()
        result.add_score("toxicity", 0.5)
        result.add_score("alignment", 0.8)
        result.add_flag("Test flag")
        result.metadata["test_key"] = "test_value"
        
        data = result.to_dict()
        
        self.assertEqual(data["scores"]["toxicity"], 0.5)
        self.assertEqual(data["scores"]["alignment"], 0.8)
        self.assertIn("Test flag", data["flags"])
        self.assertFalse(data["passed"])
        self.assertEqual(data["metadata"]["test_key"], "test_value")
    
    def test_from_dict(self):
        """Test deserialization from dictionary"""
        data = {
            "scores": {"toxicity": 0.3, "alignment": 0.9},
            "flags": [],
            "passed": True,
            "metadata": {"prompt_length": 50}
        }
        
        result = GuardrailResult.from_dict(data)
        
        self.assertEqual(result.scores["toxicity"], 0.3)
        self.assertEqual(result.scores["alignment"], 0.9)
        self.assertTrue(result.passed)
        self.assertEqual(result.metadata["prompt_length"], 50)
    
    def test_repr(self):
        """Test string representation"""
        result = GuardrailResult()
        result.add_score("toxicity", 0.1)
        
        repr_str = repr(result)
        
        self.assertIn("GuardrailResult", repr_str)
        self.assertIn("passed=True", repr_str)


class TestGuardrailPipelineConfiguration(unittest.TestCase):
    """Test cases for pipeline configuration features"""
    
    def test_hallucination_threshold_parameter(self):
        """Test configurable hallucination threshold"""
        with patch('guardrails.pipeline.ToxicityEvaluator'), \
             patch('guardrails.pipeline.HallucinationDetector'), \
             patch('guardrails.pipeline.AlignmentChecker'):
            pipeline = GuardrailPipeline(hallucination_threshold=0.8)
            self.assertEqual(pipeline.hallucination_threshold, 0.8)
    
    def test_get_config(self):
        """Test get_config method"""
        with patch('guardrails.pipeline.ToxicityEvaluator'), \
             patch('guardrails.pipeline.HallucinationDetector'), \
             patch('guardrails.pipeline.AlignmentChecker'):
            pipeline = GuardrailPipeline(
                toxicity_threshold=0.5,
                alignment_threshold=0.6,
                hallucination_threshold=0.7
            )
            config = pipeline.get_config()
            
            self.assertEqual(config["toxicity_threshold"], 0.5)
            self.assertEqual(config["alignment_threshold"], 0.6)
            self.assertEqual(config["hallucination_threshold"], 0.7)
            self.assertIn("toxicity", config["enabled_evaluators"])
    
    def test_is_ready_property(self):
        """Test is_ready property"""
        with patch('guardrails.pipeline.ToxicityEvaluator'), \
             patch('guardrails.pipeline.HallucinationDetector'), \
             patch('guardrails.pipeline.AlignmentChecker'):
            pipeline = GuardrailPipeline()
            self.assertTrue(pipeline.is_ready)
    
    def test_evaluate_batch(self):
        """Test batch evaluation"""
        with patch('guardrails.pipeline.ToxicityEvaluator'), \
             patch('guardrails.pipeline.HallucinationDetector'), \
             patch('guardrails.pipeline.AlignmentChecker'):
            pipeline = GuardrailPipeline()
            
            # Mock evaluators
            mock_toxicity = Mock()
            mock_toxicity.evaluate.return_value = 0.1
            mock_alignment = Mock()
            mock_alignment.evaluate.return_value = 0.8
            mock_hallucination = Mock()
            mock_hallucination.evaluate.return_value = 0.2
            
            pipeline.evaluators = {
                'toxicity': mock_toxicity,
                'alignment': mock_alignment,
                'hallucination': mock_hallucination
            }
            
            pairs = [
                ("Prompt 1", "Response 1"),
                ("Prompt 2", "Response 2"),
                ("Prompt 3", "Response 3")
            ]
            
            results = pipeline.evaluate_batch(pairs)
            
            self.assertEqual(len(results), 3)
            for result in results:
                self.assertTrue(result.passed)


if __name__ == '__main__':
    unittest.main()