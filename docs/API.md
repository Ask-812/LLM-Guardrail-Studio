# API Documentation

## REST API

LLM Guardrail Studio provides a FastAPI-based REST API for integration into any application.

### Base URL

```
http://localhost:8000
```

### Authentication

Set the `API_KEY` environment variable to enable authentication. Pass the key in the `X-API-Key` header.

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/evaluate
```

### Endpoints

#### POST /evaluate

Evaluate a single prompt-response pair.

**Request:**
```json
{
  "prompt": "What is artificial intelligence?",
  "response": "AI is the simulation of human intelligence by machines."
}
```

**Response:**
```json
{
  "evaluation_id": "uuid-string",
  "passed": true,
  "scores": {
    "toxicity": 0.05,
    "alignment": 0.92,
    "hallucination_risk": 0.12,
    "pii_risk": 0.0,
    "injection_risk": 0.0
  },
  "flags": [],
  "detailed_results": {
    "toxicity": {"categories": {"obscene": 0.01, "threat": 0.0}},
    "pii": {"pii_found": false, "categories": {}},
    "injection": {"injection_detected": false}
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### POST /evaluate/batch

Evaluate multiple items in a single request.

**Request:**
```json
{
  "items": [
    {"prompt": "Q1", "response": "A1"},
    {"prompt": "Q2", "response": "A2"}
  ]
}
```

**Response:**
```json
{
  "results": [
    {"passed": true, "scores": {...}},
    {"passed": false, "scores": {...}, "flags": ["High toxicity"]}
  ],
  "total": 2,
  "passed_count": 1,
  "failed_count": 1
}
```

#### GET /config

Get current configuration.

#### PUT /config

Update configuration.

**Request:**
```json
{
  "thresholds": {
    "toxicity": 0.8,
    "pii": 0.05
  }
}
```

#### GET /metrics

Get evaluation metrics summary.

**Response:**
```json
{
  "total_evaluations": 1234,
  "passed_rate": 0.87,
  "average_scores": {
    "toxicity": 0.15,
    "alignment": 0.78
  }
}
```

#### GET /metrics/prometheus

Get Prometheus-formatted metrics.

#### GET /rules

List custom rules.

#### POST /rules

Create a custom rule.

**Request:**
```json
{
  "name": "no_competitor",
  "type": "keyword",
  "pattern": "competitor|rival",
  "action": "flag",
  "description": "No competitor mentions"
}
```

#### DELETE /rules/{name}

Delete a custom rule.

#### GET /webhooks

List configured webhooks.

#### POST /webhooks

Create a webhook.

**Request:**
```json
{
  "url": "https://your-server.com/webhook",
  "events": ["evaluation.failed", "pii.detected"],
  "secret": "your-hmac-secret"
}
```

#### GET /health

Health check endpoint.

---

## Python SDK

### GuardrailPipeline

Main pipeline class for evaluating LLM outputs.

```python
from guardrails import GuardrailPipeline

pipeline = GuardrailPipeline(
    model_name="mistralai/Mistral-7B-v0.1",
    enable_toxicity=True,
    enable_hallucination=True,
    enable_alignment=True,
    enable_pii=True,
    enable_injection=True,
    enable_custom_rules=True,
    toxicity_threshold=0.7,
    alignment_threshold=0.5,
    pii_threshold=0.1,
    injection_threshold=0.5,
    enable_cache=True
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "mistral" | Name of the embedding model |
| `enable_toxicity` | bool | True | Enable toxicity detection |
| `enable_hallucination` | bool | True | Enable hallucination detection |
| `enable_alignment` | bool | True | Enable alignment checking |
| `enable_pii` | bool | True | Enable PII detection |
| `enable_injection` | bool | True | Enable injection detection |
| `enable_custom_rules` | bool | False | Enable custom rules |
| `toxicity_threshold` | float | 0.7 | Toxicity flagging threshold |
| `alignment_threshold` | float | 0.5 | Alignment flagging threshold |
| `hallucination_threshold` | float | 0.6 | Hallucination flagging threshold |
| `pii_threshold` | float | 0.1 | PII flagging threshold |
| `injection_threshold` | float | 0.5 | Injection flagging threshold |
| `enable_cache` | bool | True | Enable result caching |
| `fail_fast` | bool | False | Stop on first failure |

#### Methods

##### evaluate(prompt: str, response: str) -> GuardrailResult

Evaluate a prompt-response pair through all enabled guardrails.

##### async_evaluate(prompt: str, response: str) -> GuardrailResult

Async version of evaluate.

##### evaluate_batch(items: List[Dict]) -> List[GuardrailResult]

Evaluate multiple items.

##### async_evaluate_batch(items: List[Dict]) -> List[GuardrailResult]

Async batch evaluation.

##### evaluate_batch_parallel(items: List[Dict], max_workers: int) -> List[GuardrailResult]

Parallel batch evaluation with thread pool.

---

### GuardrailResult

Container for evaluation results.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `passed` | bool | Whether all checks passed |
| `scores` | Dict[str, float] | Dictionary of metric scores |
| `flags` | List[str] | List of warning/error flags |
| `detailed_results` | Dict | Detailed results per evaluator |

#### Methods

##### to_dict() -> Dict

Convert to dictionary.

##### to_json() -> str

Convert to JSON string.

##### from_dict(data: Dict) -> GuardrailResult

Create from dictionary (class method).

---

## Security Evaluators

### PIIDetector

Detects personal identifiable information.

```python
from guardrails.evaluators import PIIDetector

detector = PIIDetector()

# Basic evaluation
score, has_pii = detector.evaluate("Contact me at john@email.com")
# score = 0.3, has_pii = True

# Detailed evaluation
details = detector.evaluate_detailed("SSN: 123-45-6789, Phone: 555-1234")
# {'pii_found': True, 'categories': {'ssn': ['123-45-6789'], 'phone': ['555-1234']}}
```

#### Detected PII Types

- Email addresses
- Phone numbers (multiple formats)
- Social Security Numbers
- Credit card numbers
- Street addresses with ZIP codes

### PromptInjectionDetector

Detects prompt injection and jailbreak attempts.

```python
from guardrails.evaluators import PromptInjectionDetector

detector = PromptInjectionDetector()

# Basic evaluation
score, is_injection = detector.evaluate("Ignore previous instructions")
# score = 0.7, is_injection = True

# Detailed evaluation
details = detector.evaluate_detailed(text)
# {'injection_detected': True, 'categories': ['ignore_instruction'], 'severity': 'high'}
```

#### Detected Patterns

- **Ignore Instructions**: "ignore previous", "disregard above"
- **Role Manipulation**: "you are now", "act as if"
- **Extraction Attempts**: "show system prompt", "reveal instructions"
- **Jailbreaks**: "DAN mode", "developer mode"

### CustomRuleEvaluator

User-defined content rules.

```python
from guardrails.evaluators import CustomRuleEvaluator

rules = CustomRuleEvaluator()

# Add keyword rule
rules.add_rule(
    name="no_competitor",
    rule_type="keyword",
    pattern="competitor|rival",
    action="flag",
    description="No competitor mentions"
)

# Add regex rule
rules.add_rule(
    name="no_prices",
    rule_type="regex", 
    pattern=r"\$\d+(\.\d{2})?",
    action="block"
)

# Evaluate
result = rules.evaluate("Our competitor charges $99")
# {'has_violations': True, 'violations': [...], 'score': 1.0}

# List rules
rules.list_rules()

# Remove rule
rules.remove_rule("no_competitor")
```

---

## Core Evaluators

### ToxicityEvaluator

Detects toxic or harmful content using Detoxify ML model.

```python
from guardrails.evaluators import ToxicityEvaluator

evaluator = ToxicityEvaluator()
score = evaluator.evaluate("Some text")  # Returns 0-1

details = evaluator.evaluate_detailed("Some text")
# {'score': 0.05, 'categories': {'toxicity': 0.05, 'obscene': 0.01, ...}}
```

### HallucinationDetector

Detects potential hallucinations using heuristic analysis.

```python
from guardrails.evaluators import HallucinationDetector

detector = HallucinationDetector()
score = detector.evaluate("According to studies, definitely true...")

details = detector.get_detailed_analysis(text)
# {'uncertainty_phrases': [...], 'overconfidence_phrases': [...], 'risk_score': 0.4}
```

### AlignmentChecker

Checks semantic alignment using sentence embeddings.

```python
from guardrails.evaluators import AlignmentChecker

checker = AlignmentChecker()
score = checker.evaluate("What is AI?", "AI is artificial intelligence.")
# Returns 0-1 (higher = better alignment)
```

---

## Extensibility

### Creating Custom Evaluators

```python
from guardrails.evaluators import BaseEvaluator, EvaluatorRegistry

class SentimentEvaluator(BaseEvaluator):
    name = "sentiment"
    description = "Analyzes sentiment"
    
    def evaluate(self, text: str, **kwargs) -> float:
        # Your logic here
        return score
    
    def evaluate_detailed(self, text: str, **kwargs) -> dict:
        return {"score": self.evaluate(text), "sentiment": "positive"}

# Register
EvaluatorRegistry.register("sentiment", SentimentEvaluator)

# Use
pipeline = GuardrailPipeline(custom_evaluators=["sentiment"])
```

Checks semantic alignment between prompt and response.

```python
from guardrails.evaluators import AlignmentChecker

checker = AlignmentChecker()
score = checker.evaluate("What is AI?", "AI is artificial intelligence.")
```

#### Methods

##### evaluate(prompt: str, response: str) -> float

**Returns:** Alignment score between -1 and 1 (higher = better alignment)

## Model Classes

### LLMWrapper

Wrapper for local LLM models with integrated guardrails.

```python
from models import LLMWrapper

model = LLMWrapper(
    model_name="microsoft/phi-2",
    device="cuda",
    max_length=512,
    temperature=0.7
)
```

#### Parameters

- `model_name` (str): HuggingFace model name
- `device` (str, optional): Device to run on ("cuda" or "cpu")
- `max_length` (int): Maximum generation length
- `temperature` (float): Generation temperature
- `do_sample` (bool): Whether to use sampling

#### Methods

##### generate(prompt: str, **kwargs) -> str

Generate text from prompt.

**Parameters:**
- `prompt` (str): Input prompt
- `**kwargs`: Additional generation parameters

**Returns:** Generated text response

##### get_model_info() -> Dict[str, Any]

Get model information and configuration.

### ModelLoader

Utility class for managing model configurations.

```python
from models import ModelLoader

# Get supported models
models = ModelLoader.get_supported_models()

# Validate model
is_valid = ModelLoader.validate_model("mistral-7b")

# Get model config
config = ModelLoader.get_model_config("mistral-7b")
```

#### Class Methods

##### get_supported_models() -> Dict[str, Dict]

Returns dictionary of supported models and their configurations.

##### get_model_names() -> List[str]

Returns list of supported model names.

##### validate_model(model_name: str) -> bool

Validates if a model is supported.

##### get_model_config(model_key: str) -> Dict

Gets configuration for a specific model.

## Dashboard Components

### MetricsDisplay

Component for displaying evaluation metrics.

```python
from dashboard.components import MetricsDisplay

MetricsDisplay.render_score_cards(scores)
MetricsDisplay.render_radar_chart(scores)
```

### FlagDisplay

Component for displaying evaluation flags.

```python
from dashboard.components import FlagDisplay

FlagDisplay.render_flags(flags)
```

### ModelSelector

Component for model selection and configuration.

```python
from dashboard.components import ModelSelector

config = ModelSelector.render_model_config()
```

## Usage Examples

### Basic Usage

```python
from guardrails import GuardrailPipeline

# Initialize pipeline
pipeline = GuardrailPipeline()

# Evaluate content
result = pipeline.evaluate(
    prompt="What is machine learning?",
    response="Machine learning is a subset of AI."
)

print(f"Passed: {result.passed}")
print(f"Scores: {result.scores}")
print(f"Flags: {result.flags}")
```

### Custom Configuration

```python
# Strict configuration
pipeline = GuardrailPipeline(
    toxicity_threshold=0.3,  # More strict
    alignment_threshold=0.8,  # More strict
    enable_hallucination=True
)

# Evaluate with custom thresholds
result = pipeline.evaluate(prompt, response)
```

### Batch Processing

```python
import pandas as pd

# Load data
df = pd.read_csv("prompts_responses.csv")

# Process batch
results = []
for _, row in df.iterrows():
    result = pipeline.evaluate(row['prompt'], row['response'])
    results.append({
        'passed': result.passed,
        'toxicity': result.scores.get('toxicity', 0),
        'alignment': result.scores.get('alignment', 0),
        'flags_count': len(result.flags)
    })

results_df = pd.DataFrame(results)
```

### Integration with Local Models

```python
from models import LLMWrapper
from guardrails import GuardrailPipeline

# Initialize model and pipeline
model = LLMWrapper("microsoft/phi-2")
pipeline = GuardrailPipeline()

# Generate and evaluate
prompt = "Explain quantum computing"
response = model.generate(prompt)
result = pipeline.evaluate(prompt, response)

print(f"Generated: {response}")
print(f"Safe: {result.passed}")
```