# API Documentation

## Core Classes

### GuardrailPipeline

Main pipeline class for evaluating LLM outputs.

```python
from guardrails import GuardrailPipeline

pipeline = GuardrailPipeline(
    model_name="mistralai/Mistral-7B-v0.1",
    enable_toxicity=True,
    enable_hallucination=True,
    enable_alignment=True,
    toxicity_threshold=0.7,
    alignment_threshold=0.5
)
```

#### Parameters

- `model_name` (str): Name of the LLM model
- `enable_toxicity` (bool): Enable toxicity detection
- `enable_hallucination` (bool): Enable hallucination detection  
- `enable_alignment` (bool): Enable prompt-response alignment checking
- `toxicity_threshold` (float): Threshold for toxicity flagging (0-1)
- `alignment_threshold` (float): Threshold for alignment flagging (0-1)

#### Methods

##### evaluate(prompt: str, response: str) -> GuardrailResult

Evaluate a prompt-response pair through all enabled guardrails.

**Parameters:**
- `prompt` (str): Input prompt to the LLM
- `response` (str): Generated response from the LLM

**Returns:**
- `GuardrailResult`: Object containing scores and flags

### GuardrailResult

Container for evaluation results.

#### Attributes

- `scores` (Dict[str, float]): Dictionary of metric scores
- `flags` (List[str]): List of warning/error flags
- `passed` (bool): Whether all checks passed

#### Methods

##### add_score(metric: str, value: float)

Add a metric score to the result.

##### add_flag(flag: str)

Add a warning/error flag to the result.

## Evaluator Classes

### ToxicityEvaluator

Detects toxic or harmful content in text.

```python
from guardrails.evaluators import ToxicityEvaluator

evaluator = ToxicityEvaluator()
score = evaluator.evaluate("Some text to evaluate")
```

#### Methods

##### evaluate(text: str) -> float

**Returns:** Toxicity score between 0 and 1 (higher = more toxic)

### HallucinationDetector

Detects potential hallucinations in model outputs.

```python
from guardrails.evaluators import HallucinationDetector

detector = HallucinationDetector()
score = detector.evaluate("Model response text")
```

#### Methods

##### evaluate(text: str) -> float

**Returns:** Hallucination risk score between 0 and 1 (higher = more risk)

### AlignmentChecker

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