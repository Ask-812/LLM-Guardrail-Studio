# LLM Guardrail Studio

A modular trust layer for local LLMs that provides safety and moderation capabilities for open-source language models.

## Overview

LLM Guardrail Studio is a plug-and-play safety pipeline designed to enhance the reliability of local LLM deployments. It detects hallucinations, toxicity, and prompt-response mismatches in real-time, providing comprehensive analysis through an interactive dashboard.

## Features

- **Hallucination Detection**: Identifies inconsistencies and factual errors in model outputs
- **Toxicity Filtering**: Detects and flags harmful or inappropriate content
- **Prompt-Response Alignment**: Measures semantic similarity between inputs and outputs
- **Real-time Dashboard**: Interactive Streamlit interface for monitoring and analysis
- **Modular Architecture**: Easy to extend with custom evaluation modules
- **Multi-Model Support**: Compatible with Mistral, Zephyr, and other open-source LLMs

## Metrics & Evaluation

- Cosine similarity scoring for semantic alignment
- Toxicity probability classification
- Rule-based violation detection
- Comprehensive logging and scoring system

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-guardrail-studio.git
cd llm-guardrail-studio

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from guardrails import GuardrailPipeline

# Initialize the pipeline
pipeline = GuardrailPipeline(
    model_name="mistralai/Mistral-7B-v0.1",
    enable_toxicity=True,
    enable_hallucination=True
)

# Evaluate a response
result = pipeline.evaluate(
    prompt="What is the capital of France?",
    response="The capital of France is Paris."
)

print(result.scores)
```

## Dashboard

Launch the interactive dashboard:

```bash
streamlit run app.py
```

## Project Structure

```
llm-guardrail-studio/
├── guardrails/          # Core evaluation modules
├── models/              # Model wrappers and utilities
├── dashboard/           # Streamlit dashboard components
├── tests/               # Unit tests
├── examples/            # Usage examples
└── app.py              # Main dashboard application
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- SentenceTransformers
- Streamlit
- NumPy, Pandas

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Acknowledgments

Built with open-source tools including Hugging Face Transformers, SentenceTransformers, and Streamlit.
