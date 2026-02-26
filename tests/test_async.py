"""
Tests for async pipeline functionality
"""

import unittest
import asyncio
from unittest.mock import Mock, patch

from guardrails.pipeline import GuardrailPipeline, GuardrailResult


class TestAsyncPipeline(unittest.TestCase):
    """Test cases for async pipeline features"""
    
    def setUp(self):
        """Set up test pipeline"""
        with patch('guardrails.pipeline.ToxicityEvaluator'), \
             patch('guardrails.pipeline.HallucinationDetector'), \
             patch('guardrails.pipeline.AlignmentChecker'):
            self.pipeline = GuardrailPipeline()
    
    def test_async_evaluate(self):
        """Test async evaluate method"""
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
        
        # Run async test
        async def run_test():
            result = await self.pipeline.async_evaluate(
                "What is AI?",
                "AI is artificial intelligence."
            )
            self.assertTrue(result.passed)
            self.assertEqual(result.scores['toxicity'], 0.1)
            self.assertEqual(result.scores['alignment'], 0.8)
            self.assertEqual(result.scores['hallucination_risk'], 0.2)
        
        asyncio.run(run_test())
    
    def test_async_evaluate_batch(self):
        """Test async batch evaluation"""
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
        
        pairs = [
            ("Prompt 1", "Response 1"),
            ("Prompt 2", "Response 2"),
            ("Prompt 3", "Response 3")
        ]
        
        # Run async test
        async def run_test():
            results = await self.pipeline.async_evaluate_batch(pairs)
            self.assertEqual(len(results), 3)
            for result in results:
                self.assertTrue(result.passed)
        
        asyncio.run(run_test())
    
    def test_evaluate_batch_parallel(self):
        """Test parallel batch evaluation with ThreadPoolExecutor"""
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
        
        pairs = [
            ("Prompt 1", "Response 1"),
            ("Prompt 2", "Response 2"),
            ("Prompt 3", "Response 3")
        ]
        
        results = self.pipeline.evaluate_batch_parallel(pairs, max_workers=2)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertTrue(result.passed)


if __name__ == '__main__':
    unittest.main()
