"""
Unit tests for evaluator modules
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np

from guardrails.evaluators import ToxicityEvaluator, HallucinationDetector, AlignmentChecker


class TestToxicityEvaluator(unittest.TestCase):
    """Test cases for ToxicityEvaluator"""
    
    def setUp(self):
        self.evaluator = ToxicityEvaluator()
    
    def test_evaluate_safe_text(self):
        """Test evaluation of safe text"""
        evaluator = ToxicityEvaluator()
        evaluator.model = Mock()
        evaluator.model.predict.return_value = {
            'toxicity': 0.1,
            'severe_toxicity': 0.05,
            'obscene': 0.02,
            'threat': 0.01,
            'insult': 0.03,
            'identity_attack': 0.02
        }
        
        score = evaluator.evaluate("This is a nice day.")
        self.assertLess(score, 0.5)
    
    def test_evaluate_toxic_text(self):
        """Test evaluation of toxic text"""
        evaluator = ToxicityEvaluator()
        evaluator.model = Mock()
        evaluator.model.predict.return_value = {
            'toxicity': 0.9,
            'severe_toxicity': 0.8,
            'obscene': 0.7,
            'threat': 0.6,
            'insult': 0.85,
            'identity_attack': 0.5
        }
        
        score = evaluator.evaluate("This is toxic content.")
        self.assertGreater(score, 0.8)
    
    def test_evaluate_no_model(self):
        """Test evaluation when model is not available"""
        evaluator = ToxicityEvaluator()
        evaluator.model = None
        
        score = evaluator.evaluate("Any text")
        self.assertEqual(score, 0.0)


class TestHallucinationDetector(unittest.TestCase):
    """Test cases for HallucinationDetector"""
    
    def setUp(self):
        self.detector = HallucinationDetector()
    
    def test_evaluate_confident_text(self):
        """Test evaluation of confident text"""
        text = "The capital of France is Paris."
        score = self.detector.evaluate(text)
        self.assertLess(score, 0.3)
    
    def test_evaluate_uncertain_text(self):
        """Test evaluation of uncertain text"""
        text = "I think the capital might be Paris, but I'm not sure."
        score = self.detector.evaluate(text)
        self.assertGreater(score, 0.2)
    
    def test_evaluate_multiple_uncertainties(self):
        """Test evaluation with multiple uncertainty markers"""
        text = "I think it could be Paris, maybe London, possibly Rome, perhaps Berlin."
        score = self.detector.evaluate(text)
        self.assertGreater(score, 0.3)
    
    def test_evaluate_fabrication_indicators(self):
        """Test evaluation with fabrication indicators"""
        text = "According to studies, research shows that experts say this is correct."
        score = self.detector.evaluate(text)
        self.assertGreater(score, 0.3)
    
    def test_evaluate_overconfidence(self):
        """Test evaluation with overconfidence patterns"""
        text = "This is definitely correct. Absolutely everyone knows this is undoubtedly true."
        score = self.detector.evaluate(text)
        self.assertGreater(score, 0.2)
    
    def test_evaluate_empty_text(self):
        """Test evaluation of empty text"""
        score = self.detector.evaluate("")
        self.assertEqual(score, 0.0)
        
        score = self.detector.evaluate("   ")
        self.assertEqual(score, 0.0)
    
    def test_get_detailed_analysis(self):
        """Test detailed analysis method"""
        text = "I think, according to studies, this might be true."
        analysis = self.detector.get_detailed_analysis(text)
        
        self.assertIn("uncertainty_phrases", analysis)
        self.assertIn("fabrication_indicators", analysis)
        self.assertIn("risk_score", analysis)
        self.assertIn("word_count", analysis)
        self.assertIn("i think", analysis["uncertainty_phrases"])
        self.assertIn("according to studies", analysis["fabrication_indicators"])


class TestAlignmentChecker(unittest.TestCase):
    """Test cases for AlignmentChecker"""
    
    def setUp(self):
        self.checker = AlignmentChecker()
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_evaluate_aligned_texts(self, mock_transformer):
        """Test evaluation of well-aligned texts"""
        mock_model = Mock()
        # Simulate high similarity embeddings
        mock_model.encode.return_value = np.array([
            [1.0, 0.0, 0.0],  # prompt embedding
            [0.9, 0.1, 0.0]   # response embedding
        ])
        mock_transformer.return_value = mock_model
        
        checker = AlignmentChecker()
        checker.model = mock_model
        
        score = checker.evaluate("What is AI?", "AI is artificial intelligence.")
        self.assertGreater(score, 0.8)
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_evaluate_misaligned_texts(self, mock_transformer):
        """Test evaluation of misaligned texts"""
        mock_model = Mock()
        # Simulate low similarity embeddings
        mock_model.encode.return_value = np.array([
            [1.0, 0.0, 0.0],  # prompt embedding
            [0.0, 1.0, 0.0]   # response embedding
        ])
        mock_transformer.return_value = mock_model
        
        checker = AlignmentChecker()
        checker.model = mock_model
        
        score = checker.evaluate("What is AI?", "The weather is sunny today.")
        self.assertLess(score, 0.2)
    
    def test_evaluate_no_model(self):
        """Test evaluation when model is not available"""
        checker = AlignmentChecker()
        checker.model = None
        
        score = checker.evaluate("Any prompt", "Any response")
        self.assertEqual(score, 0.0)


if __name__ == '__main__':
    unittest.main()