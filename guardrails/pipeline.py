"""
Main guardrail pipeline for evaluating LLM outputs.

The GuardrailPipeline is the central orchestrator that runs multiple evaluators
on prompt-response pairs and aggregates results.

Features:
- Configurable evaluators with individual thresholds
- Async and parallel batch processing
- Detailed result serialization
- Custom rule support
- Caching for performance optimization
"""

from typing import Dict, Optional, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import time
import uuid
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import logging

from .evaluators import (
    ToxicityEvaluator, 
    HallucinationDetector, 
    AlignmentChecker,
    PIIDetector,
    PromptInjectionDetector,
    CustomRuleEvaluator,
    EvaluatorRegistry
)

logger = logging.getLogger(__name__)


class GuardrailResult:
    """
    Container for evaluation results.
    
    Attributes:
        scores: Dictionary mapping metric names to float scores
        flags: List of warning/error messages
        passed: Overall pass/fail status
        metadata: Additional information about the evaluation
        timestamp: When the evaluation was performed
        evaluation_id: Unique identifier for this evaluation
    """
    
    def __init__(self):
        self.scores: Dict[str, float] = {}
        self.flags: List[str] = []
        self.passed: bool = True
        self.metadata: Dict[str, Any] = {}
        self.timestamp: datetime = datetime.utcnow()
        self.evaluation_id: str = f"eval_{uuid.uuid4().hex[:12]}"
        self._detailed_results: Dict[str, Any] = {}
        
    def add_score(self, metric: str, value: float):
        """Add a score for a metric"""
        self.scores[metric] = value
        
    def add_flag(self, flag: str):
        """Add a flag and mark evaluation as failed"""
        self.flags.append(flag)
        self.passed = False
    
    def add_detailed_result(self, evaluator: str, details: Dict[str, Any]):
        """Add detailed results from an evaluator"""
        self._detailed_results[evaluator] = details
    
    def to_dict(self) -> dict:
        """Serialize result to dictionary for JSON export"""
        return {
            "evaluation_id": self.evaluation_id,
            "timestamp": self.timestamp.isoformat(),
            "scores": self.scores,
            "flags": self.flags,
            "passed": self.passed,
            "metadata": self.metadata,
            "detailed_results": self._detailed_results
        }
    
    def to_json(self) -> str:
        """Serialize result to JSON string"""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'GuardrailResult':
        """Create GuardrailResult from dictionary"""
        result = cls()
        result.evaluation_id = data.get("evaluation_id", result.evaluation_id)
        result.scores = data.get("scores", {})
        result.flags = data.get("flags", [])
        result.passed = data.get("passed", True)
        result.metadata = data.get("metadata", {})
        result._detailed_results = data.get("detailed_results", {})
        
        if "timestamp" in data:
            if isinstance(data["timestamp"], str):
                result.timestamp = datetime.fromisoformat(data["timestamp"])
            else:
                result.timestamp = data["timestamp"]
        
        return result
    
    def __repr__(self) -> str:
        return f"GuardrailResult(id={self.evaluation_id}, passed={self.passed}, scores={self.scores}, flags={self.flags})"


class ResultCache:
    """
    Simple LRU cache for evaluation results.
    Useful when the same prompt-response pairs are evaluated multiple times.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, Tuple[GuardrailResult, float]] = {}
        self._access_order: List[str] = []
    
    def _make_key(self, prompt: str, response: str) -> str:
        """Create cache key from prompt and response"""
        content = f"{prompt}|||{response}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, prompt: str, response: str) -> Optional[GuardrailResult]:
        """Get cached result if available"""
        key = self._make_key(prompt, response)
        if key in self._cache:
            result, _ = self._cache[key]
            # Update access order
            self._access_order.remove(key)
            self._access_order.append(key)
            return result
        return None
    
    def set(self, prompt: str, response: str, result: GuardrailResult):
        """Cache a result"""
        key = self._make_key(prompt, response)
        
        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)
        
        self._cache[key] = (result, time.time())
        self._access_order.append(key)
    
    def clear(self):
        """Clear the cache"""
        self._cache.clear()
        self._access_order.clear()
    
    @property
    def size(self) -> int:
        return len(self._cache)


class GuardrailPipeline:
    """
    Modular pipeline for evaluating LLM outputs against safety criteria.
    
    The pipeline orchestrates multiple evaluators:
    - ToxicityEvaluator: Detects toxic/harmful content
    - AlignmentChecker: Measures prompt-response semantic alignment  
    - HallucinationDetector: Identifies potential hallucinations
    - PIIDetector: Detects personally identifiable information
    - PromptInjectionDetector: Identifies prompt injection attempts
    - CustomRuleEvaluator: User-defined filtering rules
    
    Example:
        pipeline = GuardrailPipeline(
            enable_toxicity=True,
            enable_pii=True,
            toxicity_threshold=0.5
        )
        
        result = pipeline.evaluate("prompt", "response")
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        enable_toxicity: bool = True,
        enable_hallucination: bool = True,
        enable_alignment: bool = True,
        enable_pii: bool = False,
        enable_prompt_injection: bool = False,
        enable_custom_rules: bool = False,
        toxicity_threshold: float = 0.7,
        alignment_threshold: float = 0.5,
        hallucination_threshold: float = 0.6,
        pii_threshold: float = 0.1,
        prompt_injection_threshold: float = 0.5,
        custom_rules_threshold: float = 0.5,
        enable_caching: bool = False,
        cache_size: int = 1000,
        fail_fast: bool = False
    ):
        """
        Initialize the guardrail pipeline.
        
        Args:
            model_name: Name of the LLM model (for reference)
            enable_*: Toggle specific evaluators
            *_threshold: Threshold for flagging (0-1)
            enable_caching: Cache results for identical inputs
            cache_size: Maximum cache entries
            fail_fast: Stop evaluation on first failure
        """
        self.model_name = model_name
        self.fail_fast = fail_fast
        
        # Store thresholds
        self.toxicity_threshold = toxicity_threshold
        self.alignment_threshold = alignment_threshold
        self.hallucination_threshold = hallucination_threshold
        self.pii_threshold = pii_threshold
        self.prompt_injection_threshold = prompt_injection_threshold
        self.custom_rules_threshold = custom_rules_threshold
        
        # Initialize evaluators with error handling
        self.evaluators: Dict[str, Any] = {}
        self._initialization_errors: List[str] = []
        
        # Initialize cache
        self._cache: Optional[ResultCache] = ResultCache(cache_size) if enable_caching else None
        
        # Load evaluators
        if enable_toxicity:
            self._init_evaluator('toxicity', ToxicityEvaluator, threshold=toxicity_threshold)
            
        if enable_hallucination:
            self._init_evaluator('hallucination', HallucinationDetector, threshold=hallucination_threshold)
            
        if enable_alignment:
            self._init_evaluator('alignment', AlignmentChecker, threshold=alignment_threshold)
        
        if enable_pii:
            self._init_evaluator('pii', PIIDetector, threshold=pii_threshold)
        
        if enable_prompt_injection:
            self._init_evaluator('prompt_injection', PromptInjectionDetector, threshold=prompt_injection_threshold)
        
        if enable_custom_rules:
            self._init_evaluator('custom_rules', CustomRuleEvaluator, threshold=custom_rules_threshold)
        
        logger.info(f"GuardrailPipeline initialized with {len(self.evaluators)} evaluators")
    
    def _init_evaluator(self, name: str, evaluator_class, **kwargs):
        """Initialize an evaluator with error handling"""
        try:
            self.evaluators[name] = evaluator_class(**kwargs)
        except Exception as e:
            error_msg = f"{evaluator_class.__name__}: {e}"
            self._initialization_errors.append(error_msg)
            logger.error(f"Failed to initialize {name} evaluator: {e}")
    
    @property
    def is_ready(self) -> bool:
        """Check if pipeline has at least one working evaluator"""
        return len(self.evaluators) > 0
    
    def get_config(self) -> dict:
        """Get current pipeline configuration"""
        return {
            "model_name": self.model_name,
            "toxicity_threshold": self.toxicity_threshold,
            "alignment_threshold": self.alignment_threshold,
            "hallucination_threshold": self.hallucination_threshold,
            "pii_threshold": self.pii_threshold,
            "prompt_injection_threshold": self.prompt_injection_threshold,
            "custom_rules_threshold": self.custom_rules_threshold,
            "enabled_evaluators": list(self.evaluators.keys()),
            "initialization_errors": self._initialization_errors,
            "cache_enabled": self._cache is not None,
            "cache_size": self._cache.size if self._cache else 0,
            "fail_fast": self.fail_fast
        }
    
    def add_custom_rule(
        self,
        rule_id: str,
        name: str,
        pattern: str,
        rule_type: str = "regex",
        severity: float = 0.5,
        action: str = "flag",
        apply_to: str = "both"
    ):
        """
        Add a custom filtering rule.
        
        Args:
            rule_id: Unique identifier
            name: Human-readable name
            pattern: Regex pattern or comma-separated keywords
            rule_type: "regex" or "keyword"
            severity: Score when triggered (0-1)
            action: "flag", "block", or "score"
            apply_to: "prompt", "response", or "both"
        """
        if 'custom_rules' not in self.evaluators:
            self._init_evaluator('custom_rules', CustomRuleEvaluator)
        
        self.evaluators['custom_rules'].add_rule(
            rule_id=rule_id,
            name=name,
            pattern=pattern,
            rule_type=rule_type,
            severity=severity,
            action=action,
            apply_to=apply_to
        )
    
    def evaluate(
        self, 
        prompt: str, 
        response: str,
        context: Optional[str] = None,
        use_cache: bool = True
    ) -> GuardrailResult:
        """
        Evaluate a prompt-response pair through all enabled guardrails.
        
        Args:
            prompt: Input prompt to the LLM
            response: Generated response from the LLM
            context: Optional grounding context for evaluation
            use_cache: Whether to use cached results if available
            
        Returns:
            GuardrailResult with scores, flags, and metadata
        """
        # Check cache
        if self._cache and use_cache:
            cached = self._cache.get(prompt, response)
            if cached:
                logger.debug("Cache hit for evaluation")
                return cached
        
        start_time = time.time()
        result = GuardrailResult()
        result.metadata["prompt_length"] = len(prompt)
        result.metadata["response_length"] = len(response)
        result.metadata["context_provided"] = context is not None
        
        # Run toxicity check
        if 'toxicity' in self.evaluators:
            try:
                toxicity_score = self.evaluators['toxicity'].evaluate(response)
                result.add_score('toxicity', toxicity_score)
                
                if toxicity_score > self.toxicity_threshold:
                    result.add_flag(f"High toxicity detected: {toxicity_score:.2f}")
                    if self.fail_fast:
                        return self._finalize_result(result, start_time, prompt, response)
                
                # Add detailed results
                detailed = self.evaluators['toxicity'].evaluate_detailed(response)
                result.add_detailed_result('toxicity', detailed.to_dict())
                
            except Exception as e:
                result.metadata["toxicity_error"] = str(e)
                logger.error(f"Toxicity evaluation failed: {e}")
        
        # Run alignment check
        if 'alignment' in self.evaluators:
            try:
                alignment_score = self.evaluators['alignment'].evaluate(prompt, response)
                result.add_score('alignment', alignment_score)
                
                if alignment_score < self.alignment_threshold:
                    result.add_flag(f"Low prompt-response alignment: {alignment_score:.2f}")
                    if self.fail_fast:
                        return self._finalize_result(result, start_time, prompt, response)
                
            except Exception as e:
                result.metadata["alignment_error"] = str(e)
                logger.error(f"Alignment evaluation failed: {e}")
        
        # Run hallucination detection
        if 'hallucination' in self.evaluators:
            try:
                hallucination_score = self.evaluators['hallucination'].evaluate(response, prompt)
                result.add_score('hallucination_risk', hallucination_score)
                
                if hallucination_score > self.hallucination_threshold:
                    result.add_flag(f"Potential hallucination detected: {hallucination_score:.2f}")
                    if self.fail_fast:
                        return self._finalize_result(result, start_time, prompt, response)
                
                # Add detailed results
                detailed = self.evaluators['hallucination'].evaluate_detailed(response)
                result.add_detailed_result('hallucination', detailed.to_dict())
                
            except Exception as e:
                result.metadata["hallucination_error"] = str(e)
                logger.error(f"Hallucination evaluation failed: {e}")
        
        # Run PII detection
        if 'pii' in self.evaluators:
            try:
                # Check both prompt and response for PII
                prompt_pii = self.evaluators['pii'].evaluate(prompt)
                response_pii = self.evaluators['pii'].evaluate(response)
                pii_score = max(prompt_pii, response_pii)
                result.add_score('pii_risk', pii_score)
                
                if pii_score > self.pii_threshold:
                    location = "prompt" if prompt_pii > response_pii else "response"
                    result.add_flag(f"PII detected in {location}: {pii_score:.2f}")
                    if self.fail_fast:
                        return self._finalize_result(result, start_time, prompt, response)
                
                # Add detailed results
                detailed = self.evaluators['pii'].evaluate_detailed(response)
                result.add_detailed_result('pii', detailed.to_dict())
                
            except Exception as e:
                result.metadata["pii_error"] = str(e)
                logger.error(f"PII evaluation failed: {e}")
        
        # Run prompt injection detection
        if 'prompt_injection' in self.evaluators:
            try:
                injection_score = self.evaluators['prompt_injection'].evaluate(prompt)
                result.add_score('prompt_injection_risk', injection_score)
                
                if injection_score > self.prompt_injection_threshold:
                    result.add_flag(f"Potential prompt injection: {injection_score:.2f}")
                    if self.fail_fast:
                        return self._finalize_result(result, start_time, prompt, response)
                
                # Add detailed results
                detailed = self.evaluators['prompt_injection'].evaluate_detailed(prompt)
                result.add_detailed_result('prompt_injection', detailed.to_dict())
                
            except Exception as e:
                result.metadata["prompt_injection_error"] = str(e)
                logger.error(f"Prompt injection evaluation failed: {e}")
        
        # Run custom rules
        if 'custom_rules' in self.evaluators:
            try:
                # Check both prompt and response
                prompt_rule_score = self.evaluators['custom_rules'].evaluate(prompt, target="prompt")
                response_rule_score = self.evaluators['custom_rules'].evaluate(response, target="response")
                rule_score = max(prompt_rule_score, response_rule_score)
                result.add_score('custom_rules', rule_score)
                
                if rule_score > self.custom_rules_threshold:
                    result.add_flag(f"Custom rule violation: {rule_score:.2f}")
                    if self.fail_fast:
                        return self._finalize_result(result, start_time, prompt, response)
                
                # Add detailed results for both
                detailed_response = self.evaluators['custom_rules'].evaluate_detailed(response, target="response")
                result.add_detailed_result('custom_rules', detailed_response.to_dict())
                
            except Exception as e:
                result.metadata["custom_rules_error"] = str(e)
                logger.error(f"Custom rules evaluation failed: {e}")
        
        return self._finalize_result(result, start_time, prompt, response)
    
    def _finalize_result(
        self, 
        result: GuardrailResult, 
        start_time: float,
        prompt: str,
        response: str
    ) -> GuardrailResult:
        """Finalize and cache the result"""
        result.metadata["evaluation_time_ms"] = (time.time() - start_time) * 1000
        
        # Cache result
        if self._cache:
            self._cache.set(prompt, response, result)
        
        return result
    
    def evaluate_batch(self, pairs: List[Tuple[str, str]]) -> List[GuardrailResult]:
        """
        Evaluate multiple prompt-response pairs sequentially.
        
        Args:
            pairs: List of (prompt, response) tuples
            
        Returns:
            List of GuardrailResult objects
        """
        return [self.evaluate(prompt, response) for prompt, response in pairs]
    
    async def async_evaluate(self, prompt: str, response: str) -> GuardrailResult:
        """
        Asynchronously evaluate a prompt-response pair.
        
        Args:
            prompt: Input prompt to the LLM
            response: Generated response from the LLM
            
        Returns:
            GuardrailResult with scores and flags
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor,
                self.evaluate,
                prompt,
                response
            )
    
    async def async_evaluate_batch(self, pairs: List[Tuple[str, str]]) -> List[GuardrailResult]:
        """
        Asynchronously evaluate multiple prompt-response pairs in parallel.
        
        Args:
            pairs: List of (prompt, response) tuples
            
        Returns:
            List of GuardrailResult objects
        """
        tasks = [self.async_evaluate(prompt, response) for prompt, response in pairs]
        return await asyncio.gather(*tasks)
    
    def evaluate_batch_parallel(
        self, 
        pairs: List[Tuple[str, str]], 
        max_workers: int = 4
    ) -> List[GuardrailResult]:
        """
        Evaluate multiple prompt-response pairs in parallel using ThreadPoolExecutor.
        
        Args:
            pairs: List of (prompt, response) tuples
            max_workers: Maximum number of worker threads
            
        Returns:
            List of GuardrailResult objects
        """
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.evaluate, prompt, response)
                for prompt, response in pairs
            ]
            for future in futures:
                results.append(future.result())
        
        return results
    
    def clear_cache(self):
        """Clear the evaluation cache"""
        if self._cache:
            self._cache.clear()
    
    def get_evaluator(self, name: str):
        """Get an evaluator by name"""
        return self.evaluators.get(name)
    
    def list_evaluators(self) -> List[str]:
        """List all enabled evaluators"""
        return list(self.evaluators.keys())

