"""
LLM Guardrail Studio - Modular Trust Layer for Local LLMs
"""

from .pipeline import GuardrailPipeline
from .evaluators import ToxicityEvaluator, HallucinationDetector, AlignmentChecker

__version__ = "0.1.0"
__all__ = ["GuardrailPipeline", "ToxicityEvaluator", "HallucinationDetector", "AlignmentChecker"]
