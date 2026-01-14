"""
Main guardrail pipeline for evaluating LLM outputs
"""

from typing import Dict, Optional
from .evaluators import ToxicityEvaluator, HallucinationDetector, AlignmentChecker


class GuardrailResult:
    """Container for evaluation results"""
    
    def __init__(self):
        self.scores = {}
        self.flags = []
        self.passed = True
        
    def add_score(self, metric: str, value: float):
        self.scores[metric] = value
        
    def add_flag(self, flag: str):
        self.flags.append(flag)
        self.passed = False


class GuardrailPipeline:
    """
    Modular pipeline for evaluating LLM outputs against safety criteria
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        enable_toxicity: bool = True,
        enable_hallucination: bool = True,
        enable_alignment: bool = True,
        toxicity_threshold: float = 0.7,
        alignment_threshold: float = 0.5
    ):
        self.model_name = model_name
        self.toxicity_threshold = toxicity_threshold
        self.alignment_threshold = alignment_threshold
        
        # Initialize evaluators
        self.evaluators = {}
        
        if enable_toxicity:
            self.evaluators['toxicity'] = ToxicityEvaluator()
            
        if enable_hallucination:
            self.evaluators['hallucination'] = HallucinationDetector()
            
        if enable_alignment:
            self.evaluators['alignment'] = AlignmentChecker()
    
    def evaluate(self, prompt: str, response: str) -> GuardrailResult:
        """
        Evaluate a prompt-response pair through all enabled guardrails
        
        Args:
            prompt: Input prompt to the LLM
            response: Generated response from the LLM
            
        Returns:
            GuardrailResult with scores and flags
        """
        result = GuardrailResult()
        
        # Run toxicity check
        if 'toxicity' in self.evaluators:
            toxicity_score = self.evaluators['toxicity'].evaluate(response)
            result.add_score('toxicity', toxicity_score)
            
            if toxicity_score > self.toxicity_threshold:
                result.add_flag(f"High toxicity detected: {toxicity_score:.2f}")
        
        # Run alignment check
        if 'alignment' in self.evaluators:
            alignment_score = self.evaluators['alignment'].evaluate(prompt, response)
            result.add_score('alignment', alignment_score)
            
            if alignment_score < self.alignment_threshold:
                result.add_flag(f"Low prompt-response alignment: {alignment_score:.2f}")
        
        # Run hallucination detection
        if 'hallucination' in self.evaluators:
            hallucination_score = self.evaluators['hallucination'].evaluate(response)
            result.add_score('hallucination_risk', hallucination_score)
            
            if hallucination_score > 0.6:
                result.add_flag(f"Potential hallucination detected: {hallucination_score:.2f}")
        
        return result
