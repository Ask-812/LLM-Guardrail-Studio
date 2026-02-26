"""
FastAPI REST API Server for LLM Guardrail Studio
Production-ready API with comprehensive endpoints for evaluation, configuration, and monitoring.
"""

import os
import time
import uuid
import asyncio
import hashlib
import hmac
import httpx
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from functools import lru_cache
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    EvaluationRequest, EvaluationResponse, BatchRequest, BatchResponse,
    BatchItemResponse, ConfigUpdateRequest, ConfigResponse, HealthResponse,
    WebhookConfig, CustomRule, MetricsResponse, ScoreDetail, EvaluatorType
)

# Import guardrails components
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guardrails import GuardrailPipeline
from guardrails.config import GuardrailConfig, ConfigManager


# Global state
class AppState:
    """Application state container"""
    def __init__(self):
        self.pipeline: Optional[GuardrailPipeline] = None
        self.config: Optional[GuardrailConfig] = None
        self.start_time: float = time.time()
        self.webhooks: List[WebhookConfig] = []
        self.custom_rules: List[CustomRule] = []
        self.metrics = MetricsCollector()
        
    def initialize(self):
        """Initialize the pipeline with current config"""
        self.config = ConfigManager.load_or_create()
        self.pipeline = GuardrailPipeline(
            model_name=self.config.model_name,
            enable_toxicity=self.config.enable_toxicity,
            enable_hallucination=self.config.enable_hallucination,
            enable_alignment=self.config.enable_alignment,
            toxicity_threshold=self.config.toxicity_threshold,
            alignment_threshold=self.config.alignment_threshold,
            hallucination_threshold=self.config.hallucination_threshold
        )


class MetricsCollector:
    """Collects and aggregates evaluation metrics"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.evaluations: List[Dict] = []
        self.lock = asyncio.Lock()
    
    async def record(self, result: Dict):
        """Record an evaluation result"""
        async with self.lock:
            self.evaluations.append({
                "timestamp": datetime.utcnow(),
                "passed": result.get("passed", False),
                "latency_ms": result.get("latency_ms", 0),
                "scores": result.get("scores", {}),
                "flags": result.get("flags", [])
            })
            
            # Trim old entries
            if len(self.evaluations) > self.max_history:
                self.evaluations = self.evaluations[-self.max_history:]
    
    def get_summary(self) -> Dict:
        """Get metrics summary"""
        if not self.evaluations:
            return {
                "total_evaluations": 0,
                "passed_count": 0,
                "failed_count": 0,
                "average_latency_ms": 0,
                "evaluations_per_minute": 0,
                "scores_summary": {},
                "top_flags": []
            }
        
        total = len(self.evaluations)
        passed = sum(1 for e in self.evaluations if e["passed"])
        failed = total - passed
        avg_latency = sum(e["latency_ms"] for e in self.evaluations) / total
        
        # Calculate evaluations per minute
        if len(self.evaluations) >= 2:
            time_span = (self.evaluations[-1]["timestamp"] - self.evaluations[0]["timestamp"]).total_seconds()
            epm = (total / time_span * 60) if time_span > 0 else 0
        else:
            epm = 0
        
        # Aggregate scores
        scores_data = defaultdict(list)
        for e in self.evaluations:
            for metric, score_data in e.get("scores", {}).items():
                if isinstance(score_data, dict):
                    scores_data[metric].append(score_data.get("value", 0))
                else:
                    scores_data[metric].append(score_data)
        
        scores_summary = {}
        for metric, values in scores_data.items():
            scores_summary[metric] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values)
            }
        
        # Count flags
        flag_counts = defaultdict(int)
        for e in self.evaluations:
            for flag in e.get("flags", []):
                # Extract flag type
                flag_type = flag.split(":")[0] if ":" in flag else flag[:50]
                flag_counts[flag_type] += 1
        
        top_flags = sorted(
            [{"flag": k, "count": v} for k, v in flag_counts.items()],
            key=lambda x: x["count"],
            reverse=True
        )[:10]
        
        return {
            "total_evaluations": total,
            "passed_count": passed,
            "failed_count": failed,
            "average_latency_ms": avg_latency,
            "evaluations_per_minute": epm,
            "scores_summary": scores_summary,
            "top_flags": top_flags
        }


# Application state
state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    state.initialize()
    yield
    # Shutdown
    pass


# Create FastAPI app
app = FastAPI(
    title="LLM Guardrail Studio API",
    description="""
    Production-ready REST API for LLM safety and moderation.
    
    ## Features
    - Real-time evaluation of prompt-response pairs
    - Batch processing with parallel execution
    - Multiple evaluators: toxicity, hallucination, alignment, PII, prompt injection
    - Custom rule engine for domain-specific filtering
    - Webhook notifications for integration
    - Prometheus-compatible metrics
    
    ## Authentication
    API key authentication via `X-API-Key` header (when enabled).
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# API Key dependency (optional)
async def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Verify API key if authentication is enabled"""
    required_key = os.getenv("GUARDRAIL_API_KEY")
    if required_key and x_api_key != required_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


# Helper functions
def generate_eval_id() -> str:
    """Generate unique evaluation ID"""
    return f"eval_{uuid.uuid4().hex[:12]}"


async def send_webhook(webhook: WebhookConfig, payload: Dict):
    """Send webhook notification"""
    try:
        headers = {"Content-Type": "application/json"}
        
        # Add HMAC signature if secret is configured
        if webhook.secret:
            import json
            body = json.dumps(payload)
            signature = hmac.new(
                webhook.secret.encode(),
                body.encode(),
                hashlib.sha256
            ).hexdigest()
            headers["X-Signature-256"] = f"sha256={signature}"
        
        async with httpx.AsyncClient() as client:
            await client.post(
                webhook.url,
                json=payload,
                headers=headers,
                timeout=10.0
            )
    except Exception as e:
        # Log but don't fail the request
        print(f"Webhook delivery failed: {e}")


def apply_custom_rules(text: str, rules: List[CustomRule], target: str) -> List[Dict]:
    """Apply custom rules to text"""
    import re
    
    violations = []
    
    for rule in rules:
        if not rule.enabled:
            continue
        
        if rule.apply_to not in [target, "both"]:
            continue
        
        matched = False
        
        if rule.type == "regex" and rule.pattern:
            try:
                if re.search(rule.pattern, text, re.IGNORECASE):
                    matched = True
            except re.error:
                pass
        
        elif rule.type == "keyword" and rule.pattern:
            keywords = [k.strip().lower() for k in rule.pattern.split(",")]
            text_lower = text.lower()
            if any(kw in text_lower for kw in keywords):
                matched = True
        
        if matched:
            violations.append({
                "rule_id": rule.id,
                "rule_name": rule.name,
                "severity": rule.severity,
                "action": rule.action
            })
    
    return violations


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs"""
    return {"message": "LLM Guardrail Studio API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    Returns the status of all evaluators and system uptime.
    """
    evaluators_ready = {}
    
    if state.pipeline:
        for name in ["toxicity", "hallucination", "alignment"]:
            evaluators_ready[name] = name in state.pipeline.evaluators
    
    return HealthResponse(
        status="healthy" if state.pipeline and state.pipeline.is_ready else "degraded",
        version="1.0.0",
        evaluators_ready=evaluators_ready,
        uptime_seconds=time.time() - state.start_time
    )


@app.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
async def evaluate(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Evaluate a single prompt-response pair through the guardrail pipeline.
    
    This is the primary endpoint for real-time evaluation of LLM outputs.
    Returns detailed scores from all enabled evaluators and any flags triggered.
    """
    if not state.pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    start_time = time.time()
    eval_id = generate_eval_id()
    
    try:
        # Run evaluation
        result = state.pipeline.evaluate(request.prompt, request.response)
        
        # Apply custom rules
        prompt_violations = apply_custom_rules(request.prompt, state.custom_rules, "prompt")
        response_violations = apply_custom_rules(request.response, state.custom_rules, "response")
        
        # Build detailed scores
        scores: Dict[str, ScoreDetail] = {}
        
        for metric, value in result.scores.items():
            threshold = getattr(state.config, f"{metric.replace('_risk', '')}_threshold", 0.5)
            
            if metric == "alignment":
                passed = value >= threshold
            else:
                passed = value <= threshold
            
            scores[metric] = ScoreDetail(
                value=value,
                threshold=threshold,
                passed=passed,
                details=None
            )
        
        # Add custom rule violations as flags
        all_flags = list(result.flags)
        for violation in prompt_violations + response_violations:
            if violation["action"] == "flag":
                all_flags.append(f"Custom rule violation: {violation['rule_name']}")
        
        latency_ms = (time.time() - start_time) * 1000
        
        response_data = {
            "id": eval_id,
            "timestamp": datetime.utcnow(),
            "passed": result.passed and len(prompt_violations + response_violations) == 0,
            "scores": scores,
            "flags": all_flags,
            "metadata": {
                **result.metadata,
                **(request.metadata or {}),
                "custom_rule_violations": len(prompt_violations) + len(response_violations)
            },
            "latency_ms": latency_ms
        }
        
        # Record metrics
        await state.metrics.record(response_data)
        
        # Send webhooks for failed evaluations
        if not response_data["passed"]:
            for webhook in state.webhooks:
                if webhook.enabled and "evaluation.failed" in webhook.events:
                    background_tasks.add_task(send_webhook, webhook, {
                        "event": "evaluation.failed",
                        "data": response_data
                    })
        
        return EvaluationResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.post("/evaluate/batch", response_model=BatchResponse, tags=["Evaluation"])
async def evaluate_batch(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Evaluate multiple prompt-response pairs in a single request.
    
    Supports parallel processing for improved throughput.
    Maximum 1000 items per batch.
    """
    if not state.pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    start_time = time.time()
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    
    results: List[BatchItemResponse] = []
    passed_count = 0
    failed_count = 0
    error_count = 0
    
    async def evaluate_item(item) -> BatchItemResponse:
        """Evaluate a single batch item"""
        item_id = item.id or f"item_{uuid.uuid4().hex[:8]}"
        
        try:
            result = state.pipeline.evaluate(item.prompt, item.response)
            
            scores = {}
            for metric, value in result.scores.items():
                threshold = getattr(state.config, f"{metric.replace('_risk', '')}_threshold", 0.5)
                if metric == "alignment":
                    passed = value >= threshold
                else:
                    passed = value <= threshold
                
                scores[metric] = ScoreDetail(
                    value=value,
                    threshold=threshold,
                    passed=passed,
                    details=None
                )
            
            return BatchItemResponse(
                id=item_id,
                passed=result.passed,
                scores=scores,
                flags=result.flags,
                error=None
            )
            
        except Exception as e:
            return BatchItemResponse(
                id=item_id,
                passed=False,
                scores={},
                flags=[],
                error=str(e)
            )
    
    if request.parallel:
        # Parallel processing
        tasks = [evaluate_item(item) for item in request.items]
        results = await asyncio.gather(*tasks)
    else:
        # Sequential processing
        for item in request.items:
            result = await evaluate_item(item)
            results.append(result)
            
            if request.fail_fast and not result.passed:
                break
    
    for r in results:
        if r.error:
            error_count += 1
        elif r.passed:
            passed_count += 1
        else:
            failed_count += 1
    
    latency_ms = (time.time() - start_time) * 1000
    
    response_data = BatchResponse(
        batch_id=batch_id,
        timestamp=datetime.utcnow(),
        total=len(results),
        passed_count=passed_count,
        failed_count=failed_count,
        error_count=error_count,
        results=results,
        latency_ms=latency_ms
    )
    
    # Send webhook for batch completion
    for webhook in state.webhooks:
        if webhook.enabled and "batch.completed" in webhook.events:
            background_tasks.add_task(send_webhook, webhook, {
                "event": "batch.completed",
                "data": {
                    "batch_id": batch_id,
                    "total": len(results),
                    "passed": passed_count,
                    "failed": failed_count
                }
            })
    
    return response_data


@app.get("/config", response_model=ConfigResponse, tags=["Configuration"])
async def get_config(api_key: str = Depends(verify_api_key)):
    """
    Get the current pipeline configuration.
    Returns thresholds, enabled evaluators, and any initialization errors.
    """
    if not state.pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    config = state.pipeline.get_config()
    
    return ConfigResponse(
        model_name=config["model_name"],
        thresholds={
            "toxicity": config["toxicity_threshold"],
            "alignment": config["alignment_threshold"],
            "hallucination": config["hallucination_threshold"]
        },
        enabled_evaluators=config["enabled_evaluators"],
        custom_rules_count=len(state.custom_rules),
        initialization_errors=config["initialization_errors"]
    )


@app.patch("/config", response_model=ConfigResponse, tags=["Configuration"])
async def update_config(
    request: ConfigUpdateRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Update pipeline configuration.
    Changes take effect immediately for new evaluations.
    """
    if not state.config:
        raise HTTPException(status_code=503, detail="Config not initialized")
    
    # Update config values
    update_data = request.model_dump(exclude_unset=True)
    
    for key, value in update_data.items():
        if hasattr(state.config, key):
            setattr(state.config, key, value)
        elif key == "custom_rules":
            state.custom_rules = [CustomRule(**r) for r in value]
    
    # Reinitialize pipeline with new config
    state.initialize()
    
    return await get_config(api_key)


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics(api_key: str = Depends(verify_api_key)):
    """
    Get aggregated evaluation metrics.
    Useful for monitoring dashboards and alerting systems.
    """
    summary = state.metrics.get_summary()
    return MetricsResponse(**summary)


@app.get("/metrics/prometheus", tags=["Monitoring"])
async def get_prometheus_metrics():
    """
    Get metrics in Prometheus format.
    Can be scraped by Prometheus server for monitoring.
    """
    summary = state.metrics.get_summary()
    
    lines = [
        "# HELP guardrail_evaluations_total Total number of evaluations",
        "# TYPE guardrail_evaluations_total counter",
        f'guardrail_evaluations_total {summary["total_evaluations"]}',
        "",
        "# HELP guardrail_evaluations_passed Total passed evaluations",
        "# TYPE guardrail_evaluations_passed counter",
        f'guardrail_evaluations_passed {summary["passed_count"]}',
        "",
        "# HELP guardrail_evaluations_failed Total failed evaluations",
        "# TYPE guardrail_evaluations_failed counter",
        f'guardrail_evaluations_failed {summary["failed_count"]}',
        "",
        "# HELP guardrail_latency_avg_ms Average evaluation latency in milliseconds",
        "# TYPE guardrail_latency_avg_ms gauge",
        f'guardrail_latency_avg_ms {summary["average_latency_ms"]:.2f}',
        "",
        "# HELP guardrail_evaluations_per_minute Evaluations per minute",
        "# TYPE guardrail_evaluations_per_minute gauge",
        f'guardrail_evaluations_per_minute {summary["evaluations_per_minute"]:.2f}',
    ]
    
    # Add score metrics
    for metric, stats in summary.get("scores_summary", {}).items():
        metric_name = metric.replace("_", "_").replace("-", "_")
        lines.extend([
            "",
            f"# HELP guardrail_score_{metric_name}_avg Average {metric} score",
            f"# TYPE guardrail_score_{metric_name}_avg gauge",
            f'guardrail_score_{metric_name}_avg {stats["avg"]:.4f}'
        ])
    
    return "\n".join(lines)


@app.post("/webhooks", tags=["Webhooks"])
async def register_webhook(
    webhook: WebhookConfig,
    api_key: str = Depends(verify_api_key)
):
    """
    Register a webhook for event notifications.
    Webhooks are called for evaluation failures and batch completions.
    """
    state.webhooks.append(webhook)
    return {"status": "registered", "total_webhooks": len(state.webhooks)}


@app.get("/webhooks", tags=["Webhooks"])
async def list_webhooks(api_key: str = Depends(verify_api_key)):
    """List all registered webhooks"""
    return {"webhooks": [w.model_dump() for w in state.webhooks]}


@app.delete("/webhooks/{index}", tags=["Webhooks"])
async def delete_webhook(index: int, api_key: str = Depends(verify_api_key)):
    """Delete a webhook by index"""
    if 0 <= index < len(state.webhooks):
        state.webhooks.pop(index)
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Webhook not found")


@app.post("/rules", tags=["Custom Rules"])
async def add_custom_rule(
    rule: CustomRule,
    api_key: str = Depends(verify_api_key)
):
    """
    Add a custom rule for content filtering.
    Supports regex, keyword, and script-based rules.
    """
    # Check for duplicate ID
    if any(r.id == rule.id for r in state.custom_rules):
        raise HTTPException(status_code=400, detail="Rule ID already exists")
    
    state.custom_rules.append(rule)
    return {"status": "added", "total_rules": len(state.custom_rules)}


@app.get("/rules", tags=["Custom Rules"])
async def list_rules(api_key: str = Depends(verify_api_key)):
    """List all custom rules"""
    return {"rules": [r.model_dump() for r in state.custom_rules]}


@app.delete("/rules/{rule_id}", tags=["Custom Rules"])
async def delete_rule(rule_id: str, api_key: str = Depends(verify_api_key)):
    """Delete a custom rule by ID"""
    for i, rule in enumerate(state.custom_rules):
        if rule.id == rule_id:
            state.custom_rules.pop(i)
            return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Rule not found")


# =============================================================================
# GuardrailAPI class for programmatic use
# =============================================================================

class GuardrailAPI:
    """
    Wrapper class for using the API programmatically within Python.
    Useful for embedding the guardrail in other applications.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client = httpx.Client(timeout=30.0)
    
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers
    
    def evaluate(self, prompt: str, response: str, **kwargs) -> Dict[str, Any]:
        """Evaluate a single prompt-response pair"""
        resp = self.client.post(
            f"{self.base_url}/evaluate",
            json={"prompt": prompt, "response": response, **kwargs},
            headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()
    
    def evaluate_batch(self, items: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Evaluate multiple items"""
        resp = self.client.post(
            f"{self.base_url}/evaluate/batch",
            json={"items": items, **kwargs},
            headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        resp = self.client.get(
            f"{self.base_url}/config",
            headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        resp = self.client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()
    
    def close(self):
        """Close the HTTP client"""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# =============================================================================
# Run server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        workers=int(os.getenv("WORKERS", "1"))
    )
