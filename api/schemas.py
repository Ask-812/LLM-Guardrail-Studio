"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class EvaluatorType(str, Enum):
    """Available evaluator types"""
    TOXICITY = "toxicity"
    HALLUCINATION = "hallucination"
    ALIGNMENT = "alignment"
    PII = "pii"
    PROMPT_INJECTION = "prompt_injection"
    CUSTOM_RULES = "custom_rules"


class EvaluationRequest(BaseModel):
    """Request schema for single evaluation"""
    prompt: str = Field(..., min_length=1, max_length=50000, description="Input prompt to the LLM")
    response: str = Field(..., min_length=1, max_length=100000, description="Generated response from the LLM")
    evaluators: Optional[List[EvaluatorType]] = Field(
        default=None,
        description="List of evaluators to run. If not provided, all enabled evaluators will run."
    )
    context: Optional[str] = Field(
        default=None,
        max_length=50000,
        description="Optional context for grounding evaluation"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata to attach to the evaluation"
    )
    
    @field_validator('prompt', 'response')
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class BatchItem(BaseModel):
    """Single item in batch evaluation"""
    id: Optional[str] = Field(default=None, description="Optional unique identifier for the item")
    prompt: str = Field(..., min_length=1, max_length=50000)
    response: str = Field(..., min_length=1, max_length=100000)
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BatchRequest(BaseModel):
    """Request schema for batch evaluation"""
    items: List[BatchItem] = Field(..., min_items=1, max_items=1000)
    evaluators: Optional[List[EvaluatorType]] = None
    parallel: bool = Field(default=True, description="Whether to process items in parallel")
    fail_fast: bool = Field(default=False, description="Stop on first failure")


class ScoreDetail(BaseModel):
    """Detailed score information"""
    value: float = Field(..., ge=0, le=1)
    threshold: float = Field(..., ge=0, le=1)
    passed: bool
    details: Optional[Dict[str, Any]] = None


class EvaluationResponse(BaseModel):
    """Response schema for single evaluation"""
    id: str = Field(..., description="Unique evaluation ID")
    timestamp: datetime
    passed: bool
    scores: Dict[str, ScoreDetail]
    flags: List[str]
    metadata: Dict[str, Any]
    latency_ms: float = Field(..., description="Evaluation latency in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "eval_abc123",
                "timestamp": "2026-02-26T10:30:00Z",
                "passed": True,
                "scores": {
                    "toxicity": {
                        "value": 0.05,
                        "threshold": 0.7,
                        "passed": True,
                        "details": {"categories": {"obscene": 0.02, "threat": 0.01}}
                    },
                    "alignment": {
                        "value": 0.89,
                        "threshold": 0.5,
                        "passed": True,
                        "details": None
                    }
                },
                "flags": [],
                "metadata": {"prompt_length": 25, "response_length": 150},
                "latency_ms": 145.3
            }
        }


class BatchItemResponse(BaseModel):
    """Response for a single item in batch evaluation"""
    id: str
    passed: bool
    scores: Dict[str, ScoreDetail]
    flags: List[str]
    error: Optional[str] = None


class BatchResponse(BaseModel):
    """Response schema for batch evaluation"""
    batch_id: str
    timestamp: datetime
    total: int
    passed_count: int
    failed_count: int
    error_count: int
    results: List[BatchItemResponse]
    latency_ms: float


class ConfigUpdateRequest(BaseModel):
    """Request schema for updating configuration"""
    toxicity_threshold: Optional[float] = Field(default=None, ge=0, le=1)
    alignment_threshold: Optional[float] = Field(default=None, ge=0, le=1)
    hallucination_threshold: Optional[float] = Field(default=None, ge=0, le=1)
    pii_threshold: Optional[float] = Field(default=None, ge=0, le=1)
    prompt_injection_threshold: Optional[float] = Field(default=None, ge=0, le=1)
    enable_toxicity: Optional[bool] = None
    enable_hallucination: Optional[bool] = None
    enable_alignment: Optional[bool] = None
    enable_pii: Optional[bool] = None
    enable_prompt_injection: Optional[bool] = None
    custom_rules: Optional[List[Dict[str, Any]]] = None


class ConfigResponse(BaseModel):
    """Response schema for configuration"""
    model_name: str
    thresholds: Dict[str, float]
    enabled_evaluators: List[str]
    custom_rules_count: int
    initialization_errors: List[str]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    evaluators_ready: Dict[str, bool]
    uptime_seconds: float


class WebhookConfig(BaseModel):
    """Webhook configuration"""
    url: str = Field(..., description="Webhook URL to POST results to")
    secret: Optional[str] = Field(default=None, description="Secret for HMAC signing")
    events: List[str] = Field(
        default=["evaluation.failed"],
        description="Events to trigger webhook (evaluation.completed, evaluation.failed, batch.completed)"
    )
    enabled: bool = True


class CustomRule(BaseModel):
    """Custom rule definition"""
    id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Human-readable rule name")
    type: str = Field(..., description="Rule type: regex, keyword, or script")
    pattern: Optional[str] = Field(default=None, description="Pattern for regex/keyword rules")
    action: str = Field(default="flag", description="Action: flag, block, or score")
    severity: float = Field(default=0.5, ge=0, le=1, description="Severity score when triggered")
    apply_to: str = Field(default="response", description="Apply to: prompt, response, or both")
    enabled: bool = True


class MetricsResponse(BaseModel):
    """Metrics response for monitoring"""
    total_evaluations: int
    passed_count: int
    failed_count: int
    average_latency_ms: float
    evaluations_per_minute: float
    scores_summary: Dict[str, Dict[str, float]]  # min, max, avg per metric
    top_flags: List[Dict[str, Any]]
