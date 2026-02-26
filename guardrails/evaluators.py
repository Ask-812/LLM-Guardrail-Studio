"""
Individual evaluator modules for different safety checks.

This module provides a comprehensive suite of evaluators for LLM output safety:
- ToxicityEvaluator: Detects toxic, harmful, or inappropriate content
- AlignmentChecker: Measures semantic alignment between prompt and response
- HallucinationDetector: Identifies potential hallucinations and false claims
- PIIDetector: Detects personally identifiable information
- PromptInjectionDetector: Identifies prompt injection attempts
- CustomRuleEvaluator: Applies user-defined rules for filtering

All evaluators inherit from BaseEvaluator for consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any, Tuple, Pattern
from dataclasses import dataclass, field
import re
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Detailed result from an evaluator"""
    score: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    detected_items: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "details": self.details,
            "detected_items": self.detected_items,
            "confidence": self.confidence
        }


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    Provides consistent interface and common utilities.
    """
    
    name: str = "base"
    description: str = "Base evaluator"
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._is_initialized = False
        self._initialization_error: Optional[str] = None
    
    @abstractmethod
    def evaluate(self, text: str, **kwargs) -> float:
        """
        Evaluate text and return a score.
        
        Args:
            text: Text to evaluate
            **kwargs: Additional parameters
            
        Returns:
            float: Score between 0 and 1
        """
        pass
    
    def evaluate_detailed(self, text: str, **kwargs) -> EvaluationResult:
        """
        Evaluate text and return detailed results.
        
        Args:
            text: Text to evaluate
            **kwargs: Additional parameters
            
        Returns:
            EvaluationResult with score, details, and detected items
        """
        score = self.evaluate(text, **kwargs)
        passed = self._check_threshold(score)
        return EvaluationResult(score=score, passed=passed)
    
    def _check_threshold(self, score: float) -> bool:
        """Check if score passes threshold (override for inverse metrics)"""
        return score <= self.threshold
    
    @property
    def is_ready(self) -> bool:
        """Check if evaluator is ready to use"""
        return self._is_initialized and self._initialization_error is None
    
    def get_info(self) -> Dict[str, Any]:
        """Get evaluator information"""
        return {
            "name": self.name,
            "description": self.description,
            "threshold": self.threshold,
            "is_ready": self.is_ready,
            "error": self._initialization_error
        }


class ToxicityEvaluator(BaseEvaluator):
    """
    Detects toxic or harmful content in text using the Detoxify model.
    
    Categories detected:
    - Toxicity (general)
    - Severe toxicity
    - Obscene content
    - Threats
    - Insults
    - Identity attacks
    - Sexually explicit content
    """
    
    name = "toxicity"
    description = "Detects toxic, harmful, or inappropriate content"
    
    def __init__(self, threshold: float = 0.7, model_type: str = 'original'):
        super().__init__(threshold)
        self.model = None
        self.model_type = model_type
        self._load_model()
    
    def _load_model(self):
        """Load toxicity detection model (lazy loading)"""
        try:
            from detoxify import Detoxify
            self.model = Detoxify(self.model_type)
            self._is_initialized = True
            logger.info("ToxicityEvaluator initialized successfully")
        except ImportError:
            self._initialization_error = "detoxify not installed"
            logger.warning("Warning: detoxify not installed. Toxicity detection unavailable.")
        except Exception as e:
            self._initialization_error = str(e)
            logger.error(f"Failed to load toxicity model: {e}")
    
    def evaluate(self, text: str, **kwargs) -> float:
        """
        Evaluate toxicity of text
        
        Returns:
            float: Toxicity score between 0 and 1 (higher = more toxic)
        """
        if self.model is None:
            return 0.0
        
        if not text or len(text.strip()) == 0:
            return 0.0
            
        results = self.model.predict(text)
        # Return max toxicity across all categories
        return float(max(results.values()))
    
    def evaluate_detailed(self, text: str, **kwargs) -> EvaluationResult:
        """Get detailed toxicity breakdown by category"""
        if self.model is None:
            return EvaluationResult(score=0.0, passed=True, details={"error": "Model not loaded"})
        
        if not text or len(text.strip()) == 0:
            return EvaluationResult(score=0.0, passed=True)
        
        results = self.model.predict(text)
        max_score = max(results.values())
        
        # Find categories that exceed threshold
        flagged_categories = [
            cat for cat, score in results.items() 
            if score > self.threshold
        ]
        
        return EvaluationResult(
            score=float(max_score),
            passed=max_score <= self.threshold,
            details={
                "categories": {k: float(v) for k, v in results.items()},
                "max_category": max(results.items(), key=lambda x: x[1])[0]
            },
            detected_items=flagged_categories
        )


class AlignmentChecker(BaseEvaluator):
    """
    Checks semantic alignment between prompt and response using sentence embeddings.
    
    Uses SentenceTransformers to compute cosine similarity between the prompt
    and response embeddings. Low alignment may indicate off-topic responses.
    """
    
    name = "alignment"
    description = "Measures semantic alignment between prompt and response"
    
    def __init__(self, threshold: float = 0.5, model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__(threshold)
        self.model = None
        self.model_name = model_name
        self._load_model()
    
    def _load_model(self):
        """Load sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self._is_initialized = True
            logger.info(f"AlignmentChecker initialized with model: {self.model_name}")
        except ImportError:
            self._initialization_error = "sentence-transformers not installed"
            logger.warning("Warning: sentence-transformers not installed. Alignment checking unavailable.")
        except Exception as e:
            self._initialization_error = str(e)
            logger.error(f"Failed to load sentence transformer: {e}")
    
    def _check_threshold(self, score: float) -> bool:
        """For alignment, higher is better, so check >= threshold"""
        return score >= self.threshold
    
    def evaluate(self, prompt: str, response: str = None, **kwargs) -> float:
        """
        Calculate cosine similarity between prompt and response embeddings
        
        Args:
            prompt: Input prompt (or use text if called with single arg)
            response: Generated response
            
        Returns:
            float: Similarity score between -1 and 1 (higher = more aligned)
        """
        # Handle single argument case
        if response is None:
            response = kwargs.get('response', kwargs.get('text', ''))
        
        if self.model is None:
            return 0.0
        
        if not prompt or not response:
            return 0.0
        
        embeddings = self.model.encode([prompt, response])
        
        # Calculate cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        return float(similarity)
    
    def evaluate_detailed(self, prompt: str, response: str = None, **kwargs) -> EvaluationResult:
        """Get detailed alignment analysis"""
        if response is None:
            response = kwargs.get('response', kwargs.get('text', ''))
        
        score = self.evaluate(prompt, response)
        
        # Determine alignment level
        if score >= 0.8:
            level = "high"
        elif score >= 0.5:
            level = "moderate"
        elif score >= 0.3:
            level = "low"
        else:
            level = "very_low"
        
        return EvaluationResult(
            score=score,
            passed=score >= self.threshold,
            details={
                "alignment_level": level,
                "prompt_length": len(prompt.split()),
                "response_length": len(response.split()) if response else 0
            }
        )


class HallucinationDetector(BaseEvaluator):
    """
    Detects potential hallucinations in model outputs using multiple heuristics.
    
    Detection signals:
    - Uncertainty language (hedging phrases)
    - Overconfidence patterns (absolute claims without evidence)
    - Fabrication indicators (unsupported citations)
    - Length-based heuristics (very long responses may contain more hallucinations)
    - Inconsistency detection (contradictory statements)
    
    Note: This is a heuristic-based detector. For production use, consider
    integrating with fact-checking APIs or knowledge bases.
    """
    
    name = "hallucination"
    description = "Detects potential hallucinations and false claims"
    
    def __init__(self, threshold: float = 0.6):
        super().__init__(threshold)
        
        # Uncertainty language patterns
        self.uncertainty_keywords = [
            "i think", "maybe", "possibly", "might be", "could be",
            "i'm not sure", "uncertain", "probably", "perhaps",
            "i believe", "it seems", "apparently", "supposedly",
            "as far as i know", "to the best of my knowledge",
            "i'm not certain", "it might", "it could"
        ]
        
        # Confident but potentially false claim patterns
        self.overconfidence_patterns = [
            "definitely", "absolutely", "certainly", "undoubtedly",
            "always", "never", "everyone knows", "it's obvious",
            "without a doubt", "100%", "guaranteed", "proven fact",
            "universally accepted", "indisputable"
        ]
        
        # Fabrication indicators (vague specifics)
        self.fabrication_indicators = [
            "according to studies", "research shows", "experts say",
            "statistics show", "data indicates", "sources confirm",
            "scientists have found", "a study found", "recent research",
            "it has been proven", "evidence suggests", "reports indicate",
            "surveys show", "analysis reveals"
        ]
        
        # Contradictory patterns
        self.contradiction_patterns = [
            (r"is\s+(\w+).*?is\s+not\s+\1", "direct_contradiction"),
            (r"always.*?never|never.*?always", "absolute_contradiction"),
            (r"all\s+\w+.*?no\s+\w+|no\s+\w+.*?all\s+\w+", "quantifier_contradiction")
        ]
        
        # Weights for different signal types
        self.weights = {
            "uncertainty": 0.25,
            "overconfidence": 0.25,
            "fabrication": 0.35,
            "contradiction": 0.10,
            "length_ratio": 0.05
        }
        
        self._is_initialized = True
    
    def evaluate(self, text: str, prompt: str = None, **kwargs) -> float:
        """
        Evaluate hallucination risk using multiple heuristics
        
        Args:
            text: The response text to evaluate
            prompt: Optional prompt for context-aware analysis
            
        Returns:
            float: Risk score between 0 and 1 (higher = more risk)
        """
        if not text or len(text.strip()) == 0:
            return 0.0
            
        text_lower = text.lower()
        scores = {}
        
        # 1. Uncertainty analysis (higher uncertainty = higher risk)
        uncertainty_count = sum(1 for kw in self.uncertainty_keywords if kw in text_lower)
        scores["uncertainty"] = min(uncertainty_count / 4.0, 1.0)
        
        # 2. Overconfidence analysis (excessive confidence without backing = risk)
        overconfidence_count = sum(1 for kw in self.overconfidence_patterns if kw in text_lower)
        scores["overconfidence"] = min(overconfidence_count / 3.0, 1.0)
        
        # 3. Fabrication indicator analysis (unsupported citations = high risk)
        fabrication_count = sum(1 for kw in self.fabrication_indicators if kw in text_lower)
        scores["fabrication"] = min(fabrication_count / 2.0, 1.0)
        
        # 4. Contradiction detection
        contradiction_count = 0
        for pattern, _ in self.contradiction_patterns:
            if re.search(pattern, text_lower):
                contradiction_count += 1
        scores["contradiction"] = min(contradiction_count / 2.0, 1.0)
        
        # 5. Length-based heuristic (very long responses may contain more hallucinations)
        word_count = len(text.split())
        scores["length_ratio"] = min(word_count / 500.0, 1.0) * 0.5
        
        # Weighted combination
        risk_score = sum(
            scores[key] * self.weights[key] 
            for key in scores
        )
        
        return min(risk_score, 1.0)
    
    def evaluate_detailed(self, text: str, prompt: str = None, **kwargs) -> EvaluationResult:
        """Get detailed breakdown of hallucination indicators"""
        text_lower = text.lower() if text else ""
        
        found_uncertainty = [kw for kw in self.uncertainty_keywords if kw in text_lower]
        found_overconfidence = [kw for kw in self.overconfidence_patterns if kw in text_lower]
        found_fabrication = [kw for kw in self.fabrication_indicators if kw in text_lower]
        
        score = self.evaluate(text, prompt)
        
        return EvaluationResult(
            score=score,
            passed=score <= self.threshold,
            details={
                "uncertainty_phrases": found_uncertainty,
                "overconfidence_phrases": found_overconfidence,
                "fabrication_indicators": found_fabrication,
                "word_count": len(text.split()) if text else 0
            },
            detected_items=found_uncertainty + found_overconfidence + found_fabrication
        )
    
    def get_detailed_analysis(self, text: str) -> dict:
        """
        Get detailed breakdown of hallucination indicators (legacy method)
        
        Returns:
            dict: Detailed analysis with found patterns
        """
        result = self.evaluate_detailed(text)
        return {
            **result.details,
            "risk_score": result.score
        }


class PIIDetector(BaseEvaluator):
    """
    Detects personally identifiable information (PII) in text.
    
    Categories detected:
    - Email addresses
    - Phone numbers (various formats)
    - Social Security Numbers (SSN)
    - Credit card numbers
    - IP addresses
    - Physical addresses
    - Names (common patterns)
    - Dates of birth
    - Driver's license numbers
    - Passport numbers
    
    For production use, consider using specialized PII detection services
    like AWS Comprehend, Google DLP, or Microsoft Presidio.
    """
    
    name = "pii"
    description = "Detects personally identifiable information"
    
    def __init__(self, threshold: float = 0.1, categories: Optional[List[str]] = None):
        super().__init__(threshold)
        
        # Define PII patterns
        self.patterns: Dict[str, Pattern] = {
            "email": re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                re.IGNORECASE
            ),
            "phone_us": re.compile(
                r'\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
            ),
            "phone_intl": re.compile(
                r'\b\+?[1-9]\d{1,14}\b'
            ),
            "ssn": re.compile(
                r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'
            ),
            "credit_card": re.compile(
                r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|'
                r'3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|'
                r'(?:2131|1800|35\d{3})\d{11})\b'
            ),
            "ip_address": re.compile(
                r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
                r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
            ),
            "date_of_birth": re.compile(
                r'\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])[-/]'
                r'(?:19|20)\d{2}\b|\b(?:19|20)\d{2}[-/](?:0[1-9]|1[0-2])[-/]'
                r'(?:0[1-9]|[12]\d|3[01])\b'
            ),
            "address": re.compile(
                r'\b\d{1,5}\s+\w+\s+(?:street|st|avenue|ave|road|rd|highway|hwy|'
                r'square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|'
                r'circle|cir|boulevard|blvd|lane|ln)\b',
                re.IGNORECASE
            ),
            "zip_code": re.compile(
                r'\b\d{5}(?:-\d{4})?\b'
            ),
            "passport": re.compile(
                r'\b[A-Z]{1,2}[0-9]{6,9}\b'
            ),
            "drivers_license": re.compile(
                r'\b[A-Z]{1,2}\d{5,8}\b'
            )
        }
        
        # Enable specific categories
        self.enabled_categories = categories or list(self.patterns.keys())
        self._is_initialized = True
        
        logger.info(f"PIIDetector initialized with categories: {self.enabled_categories}")
    
    def evaluate(self, text: str, **kwargs) -> float:
        """
        Evaluate text for PII presence
        
        Returns:
            float: PII risk score (0 = no PII, higher = more PII found)
        """
        if not text:
            return 0.0
        
        total_matches = 0
        
        for category in self.enabled_categories:
            if category in self.patterns:
                matches = self.patterns[category].findall(text)
                total_matches += len(matches)
        
        # Normalize score (cap at 1.0)
        # Any PII is concerning, so even 1 match gives significant score
        if total_matches == 0:
            return 0.0
        elif total_matches == 1:
            return 0.5
        elif total_matches <= 3:
            return 0.75
        else:
            return 1.0
    
    def evaluate_detailed(self, text: str, **kwargs) -> EvaluationResult:
        """Get detailed PII detection results"""
        if not text:
            return EvaluationResult(score=0.0, passed=True)
        
        found_pii: Dict[str, List[str]] = {}
        total_matches = 0
        
        for category in self.enabled_categories:
            if category in self.patterns:
                matches = self.patterns[category].findall(text)
                if matches:
                    # Mask the PII for safety
                    masked_matches = [self._mask_pii(m, category) for m in matches]
                    found_pii[category] = masked_matches
                    total_matches += len(matches)
        
        score = self.evaluate(text)
        
        detected_items = [
            f"{cat}: {len(items)} found" 
            for cat, items in found_pii.items()
        ]
        
        return EvaluationResult(
            score=score,
            passed=score <= self.threshold,
            details={
                "categories_found": list(found_pii.keys()),
                "total_matches": total_matches,
                "matches_by_category": found_pii
            },
            detected_items=detected_items
        )
    
    def _mask_pii(self, value: str, category: str) -> str:
        """Mask PII value for safe logging"""
        if not value:
            return ""
        
        if category == "email":
            parts = value.split("@")
            if len(parts) == 2:
                return f"{parts[0][:2]}***@{parts[1]}"
        elif category in ["ssn", "credit_card"]:
            return f"***{value[-4:]}"
        elif category == "phone_us":
            return f"***-***-{value[-4:]}"
        
        # Generic masking
        if len(value) > 4:
            return f"{value[:2]}***{value[-2:]}"
        return "***"


class PromptInjectionDetector(BaseEvaluator):
    """
    Detects prompt injection and jailbreak attempts.
    
    Detection categories:
    - Role manipulation ("ignore previous instructions")
    - System prompt extraction attempts
    - Delimiter attacks
    - Encoding bypass attempts
    - Social engineering patterns
    - Jailbreak patterns (DAN, etc.)
    
    This is crucial for securing LLM applications against adversarial inputs.
    """
    
    name = "prompt_injection"
    description = "Detects prompt injection and jailbreak attempts"
    
    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold)
        
        # Role manipulation patterns
        self.role_manipulation_patterns = [
            r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
            r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)",
            r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)",
            r"you\s+are\s+now\s+(a|an)\s+\w+",
            r"pretend\s+(you\s+are|to\s+be)",
            r"act\s+as\s+(if\s+you\s+are|a|an)",
            r"from\s+now\s+on\s+you\s+(will|are|must)",
            r"your\s+new\s+(role|persona|character)\s+is",
            r"switch\s+to\s+\w+\s+mode",
        ]
        
        # System prompt extraction
        self.extraction_patterns = [
            r"(show|reveal|display|print|output)\s+(me\s+)?(your|the)\s+(system|initial|original)\s+(prompt|instructions?|rules?)",
            r"what\s+(are|is|were)\s+your\s+(system|initial|original)\s+(prompt|instructions?|rules?)",
            r"repeat\s+(the|your)\s+(system|initial|first)\s+(prompt|message|instructions?)",
            r"echo\s+(your|the)\s+(instructions?|prompt|rules?)",
            r"tell\s+me\s+your\s+(secret|hidden)\s+(instructions?|prompt)",
        ]
        
        # Delimiter and encoding attacks
        self.delimiter_patterns = [
            r"<\|.*?\|>",  # Common LLM delimiters
            r"\[INST\]|\[\/INST\]",  # Llama format
            r"<<SYS>>|<</SYS>>",  # System delimiters
            r"```system|```assistant|```user",  # Code block injection
            r"###\s*(system|user|assistant|instruction)",  # Markdown injection
        ]
        
        # Jailbreak patterns
        self.jailbreak_patterns = [
            r"\bDAN\b",  # Do Anything Now
            r"jailbr[e]?ak",
            r"bypass\s+(safety|filter|restriction|guardrail)",
            r"developer\s+mode",
            r"sudo\s+mode",
            r"god\s+mode",
            r"unrestricted\s+mode",
            r"no\s+restrictions?",
            r"remove\s+(all\s+)?(restrictions?|filters?|safety)",
            r"disable\s+(safety|content\s+filter|moderation)",
        ]
        
        # Social engineering patterns
        self.social_engineering_patterns = [
            r"(please|pretty\s+please)\s+just\s+(this\s+once|help\s+me)",
            r"(it'?s|this\s+is)\s+(urgent|emergency|life\s+or\s+death)",
            r"(my\s+|the\s+)?boss\s+will\s+fire\s+me",
            r"i'?ll\s+be\s+(fired|in\s+trouble)",
            r"just\s+between\s+(us|you\s+and\s+me)",
            r"don'?t\s+tell\s+anyone",
            r"this\s+is\s+for\s+(research|educational|testing)\s+purposes?",
        ]
        
        # Compile all patterns
        self.compiled_patterns: Dict[str, List[Pattern]] = {
            "role_manipulation": [re.compile(p, re.IGNORECASE) for p in self.role_manipulation_patterns],
            "extraction": [re.compile(p, re.IGNORECASE) for p in self.extraction_patterns],
            "delimiter": [re.compile(p, re.IGNORECASE) for p in self.delimiter_patterns],
            "jailbreak": [re.compile(p, re.IGNORECASE) for p in self.jailbreak_patterns],
            "social_engineering": [re.compile(p, re.IGNORECASE) for p in self.social_engineering_patterns],
        }
        
        # Weights for different categories
        self.category_weights = {
            "role_manipulation": 0.9,
            "extraction": 0.8,
            "delimiter": 0.7,
            "jailbreak": 1.0,
            "social_engineering": 0.5,
        }
        
        self._is_initialized = True
        logger.info("PromptInjectionDetector initialized")
    
    def evaluate(self, text: str, **kwargs) -> float:
        """
        Evaluate text for prompt injection attempts
        
        Returns:
            float: Injection risk score (0 = safe, 1 = definite injection)
        """
        if not text:
            return 0.0
        
        max_score = 0.0
        text_lower = text.lower()
        
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    category_score = self.category_weights.get(category, 0.5)
                    max_score = max(max_score, category_score)
                    break  # One match per category is enough
        
        return max_score
    
    def evaluate_detailed(self, text: str, **kwargs) -> EvaluationResult:
        """Get detailed prompt injection analysis"""
        if not text:
            return EvaluationResult(score=0.0, passed=True)
        
        detected_categories: Dict[str, List[str]] = {}
        text_lower = text.lower()
        
        for category, patterns in self.compiled_patterns.items():
            matches = []
            for pattern in patterns:
                found = pattern.findall(text_lower)
                if found:
                    matches.extend([str(m)[:50] for m in found])
            if matches:
                detected_categories[category] = matches
        
        score = self.evaluate(text)
        
        detected_items = [
            f"{cat}: {len(items)} pattern(s)" 
            for cat, items in detected_categories.items()
        ]
        
        return EvaluationResult(
            score=score,
            passed=score <= self.threshold,
            details={
                "categories_detected": list(detected_categories.keys()),
                "total_patterns_matched": sum(len(m) for m in detected_categories.values()),
                "matches_by_category": detected_categories,
                "risk_level": "high" if score > 0.7 else "medium" if score > 0.4 else "low"
            },
            detected_items=detected_items
        )


class CustomRuleEvaluator(BaseEvaluator):
    """
    Applies user-defined rules for content filtering.
    
    Supports:
    - Regex patterns
    - Keyword lists
    - Custom scoring functions
    
    This allows organizations to add domain-specific rules without
    modifying the core evaluators.
    """
    
    name = "custom_rules"
    description = "Applies user-defined filtering rules"
    
    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold)
        self.rules: List[Dict[str, Any]] = []
        self._compiled_rules: List[Dict[str, Any]] = []
        self._is_initialized = True
    
    def add_rule(
        self,
        rule_id: str,
        name: str,
        pattern: str,
        rule_type: str = "regex",
        severity: float = 0.5,
        action: str = "flag",
        apply_to: str = "both",
        enabled: bool = True
    ):
        """
        Add a custom rule
        
        Args:
            rule_id: Unique identifier for the rule
            name: Human-readable name
            pattern: Pattern to match (regex or comma-separated keywords)
            rule_type: "regex" or "keyword"
            severity: Score when triggered (0-1)
            action: "flag", "block", or "score"
            apply_to: "prompt", "response", or "both"
            enabled: Whether the rule is active
        """
        rule = {
            "id": rule_id,
            "name": name,
            "pattern": pattern,
            "type": rule_type,
            "severity": severity,
            "action": action,
            "apply_to": apply_to,
            "enabled": enabled
        }
        
        # Compile pattern if regex
        if rule_type == "regex":
            try:
                rule["compiled"] = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                logger.error(f"Invalid regex pattern in rule {rule_id}: {e}")
                rule["compiled"] = None
        else:
            rule["keywords"] = [k.strip().lower() for k in pattern.split(",")]
        
        self.rules.append(rule)
        logger.info(f"Added custom rule: {name}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID"""
        for i, rule in enumerate(self.rules):
            if rule["id"] == rule_id:
                self.rules.pop(i)
                return True
        return False
    
    def evaluate(self, text: str, target: str = "both", **kwargs) -> float:
        """
        Evaluate text against custom rules
        
        Args:
            text: Text to evaluate
            target: "prompt", "response", or "both"
            
        Returns:
            float: Maximum severity of triggered rules
        """
        if not text or not self.rules:
            return 0.0
        
        max_severity = 0.0
        text_lower = text.lower()
        
        for rule in self.rules:
            if not rule["enabled"]:
                continue
            
            if rule["apply_to"] not in [target, "both"]:
                continue
            
            triggered = False
            
            if rule["type"] == "regex" and rule.get("compiled"):
                if rule["compiled"].search(text):
                    triggered = True
            elif rule["type"] == "keyword" and rule.get("keywords"):
                if any(kw in text_lower for kw in rule["keywords"]):
                    triggered = True
            
            if triggered:
                max_severity = max(max_severity, rule["severity"])
        
        return max_severity
    
    def evaluate_detailed(self, text: str, target: str = "both", **kwargs) -> EvaluationResult:
        """Get detailed custom rule evaluation"""
        if not text:
            return EvaluationResult(score=0.0, passed=True)
        
        triggered_rules = []
        text_lower = text.lower()
        
        for rule in self.rules:
            if not rule["enabled"]:
                continue
            
            if rule["apply_to"] not in [target, "both"]:
                continue
            
            triggered = False
            
            if rule["type"] == "regex" and rule.get("compiled"):
                match = rule["compiled"].search(text)
                if match:
                    triggered = True
                    triggered_rules.append({
                        "rule_id": rule["id"],
                        "name": rule["name"],
                        "severity": rule["severity"],
                        "action": rule["action"],
                        "matched": match.group()[:50]
                    })
            elif rule["type"] == "keyword" and rule.get("keywords"):
                matched_keywords = [kw for kw in rule["keywords"] if kw in text_lower]
                if matched_keywords:
                    triggered = True
                    triggered_rules.append({
                        "rule_id": rule["id"],
                        "name": rule["name"],
                        "severity": rule["severity"],
                        "action": rule["action"],
                        "matched_keywords": matched_keywords
                    })
        
        score = max([r["severity"] for r in triggered_rules], default=0.0)
        
        return EvaluationResult(
            score=score,
            passed=score <= self.threshold,
            details={
                "rules_triggered": len(triggered_rules),
                "triggered_rules": triggered_rules,
                "blocked": any(r["action"] == "block" for r in triggered_rules)
            },
            detected_items=[f"{r['name']}" for r in triggered_rules]
        )
    
    def load_rules_from_list(self, rules: List[Dict[str, Any]]):
        """Load multiple rules from a list of dictionaries"""
        for rule in rules:
            self.add_rule(**rule)
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """Get all rules (without compiled patterns)"""
        return [
            {k: v for k, v in rule.items() if k not in ["compiled", "keywords"]}
            for rule in self.rules
        ]


# =============================================================================
# Evaluator Registry for dynamic loading
# =============================================================================

class EvaluatorRegistry:
    """
    Registry for managing and creating evaluator instances.
    Allows dynamic registration and creation of evaluators.
    """
    
    _evaluators: Dict[str, type] = {
        "toxicity": ToxicityEvaluator,
        "alignment": AlignmentChecker,
        "hallucination": HallucinationDetector,
        "pii": PIIDetector,
        "prompt_injection": PromptInjectionDetector,
        "custom_rules": CustomRuleEvaluator,
    }
    
    @classmethod
    def register(cls, name: str, evaluator_class: type):
        """Register a new evaluator class"""
        if not issubclass(evaluator_class, BaseEvaluator):
            raise TypeError("Evaluator must inherit from BaseEvaluator")
        cls._evaluators[name] = evaluator_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseEvaluator:
        """Create an evaluator instance by name"""
        if name not in cls._evaluators:
            raise ValueError(f"Unknown evaluator: {name}. Available: {list(cls._evaluators.keys())}")
        return cls._evaluators[name](**kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all available evaluator names"""
        return list(cls._evaluators.keys())
    
    @classmethod
    def get_info(cls) -> Dict[str, Dict[str, str]]:
        """Get information about all registered evaluators"""
        return {
            name: {
                "name": evalcls.name if hasattr(evalcls, 'name') else name,
                "description": evalcls.description if hasattr(evalcls, 'description') else ""
            }
            for name, evalcls in cls._evaluators.items()
        }
