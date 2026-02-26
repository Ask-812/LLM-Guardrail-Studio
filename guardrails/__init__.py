"""
LLM Guardrail Studio - Modular Trust Layer for Local LLMs

A comprehensive safety and moderation pipeline for open-source language models.

Core Features:
- Toxicity detection using ML models
- Hallucination detection using heuristics
- Prompt-response alignment checking
- PII (Personally Identifiable Information) detection
- Prompt injection and jailbreak detection
- Custom rule engine for domain-specific filtering

Example Usage:
    from guardrails import GuardrailPipeline
    
    pipeline = GuardrailPipeline(
        enable_toxicity=True,
        enable_hallucination=True,
        enable_alignment=True,
        enable_pii=True,
        enable_prompt_injection=True
    )
    
    result = pipeline.evaluate(
        prompt="What is machine learning?",
        response="Machine learning is a subset of AI..."
    )
    
    print(f"Passed: {result.passed}")
    print(f"Scores: {result.scores}")
"""

from .pipeline import GuardrailPipeline, GuardrailResult
from .evaluators import (
    BaseEvaluator,
    EvaluationResult,
    ToxicityEvaluator,
    HallucinationDetector,
    AlignmentChecker,
    PIIDetector,
    PromptInjectionDetector,
    CustomRuleEvaluator,
    EvaluatorRegistry
)
from .config import GuardrailConfig, ConfigManager

__version__ = "1.0.0"
__all__ = [
    # Pipeline
    "GuardrailPipeline",
    "GuardrailResult",
    # Evaluators
    "BaseEvaluator",
    "EvaluationResult",
    "ToxicityEvaluator",
    "HallucinationDetector",
    "AlignmentChecker",
    "PIIDetector",
    "PromptInjectionDetector",
    "CustomRuleEvaluator",
    "EvaluatorRegistry",
    # Config
    "GuardrailConfig",
    "ConfigManager"
]
