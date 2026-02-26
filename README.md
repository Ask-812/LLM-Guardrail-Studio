# LLM Guardrail Studio

A production-ready safety pipeline for local LLMs with comprehensive moderation, security evaluation, and real-time monitoring capabilities.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

## 🚀 Overview

LLM Guardrail Studio is a **production-ready** trust layer for LLM applications. It provides:

- **6 Safety Evaluators**: Toxicity, hallucination, alignment, PII detection, prompt injection, and custom rules
- **Multiple Interfaces**: REST API, CLI, Python SDK, and Interactive Dashboard
- **Enterprise Features**: Audit logging, webhooks, Prometheus metrics, and SQL persistence
- **Easy Deployment**: Docker Compose setup with monitoring stack (Prometheus/Grafana)

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       LLM Guardrail Studio                       │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────┤
│   REST API  │     CLI     │  Dashboard  │  Python SDK │ Webhooks│
├─────────────┴─────────────┴─────────────┴─────────────┴─────────┤
│                      Guardrail Pipeline                          │
├─────────┬─────────┬─────────┬─────────┬─────────┬───────────────┤
│Toxicity │Halluc.  │Alignment│   PII   │Injection│ Custom Rules  │
├─────────┴─────────┴─────────┴─────────┴─────────┴───────────────┤
│                    Persistence Layer                             │
├─────────────────────────────────────────────────────────────────┤
│         SQLite/PostgreSQL  │  Redis Cache  │  Audit Logs        │
└─────────────────────────────────────────────────────────────────┘
```

## ✨ Features

### Safety Evaluators

| Evaluator | Description | Method |
|-----------|-------------|--------|
| **Toxicity Detection** | Identifies harmful, offensive, or inappropriate content | Detoxify ML model |
| **Hallucination Detection** | Detects fabricated information and overconfidence | Heuristic analysis |
| **Alignment Check** | Measures semantic relevance between prompt and response | Sentence embeddings |
| **PII Detection** | Finds personal data (emails, SSN, credit cards, phones) | Regex patterns |
| **Prompt Injection** | Detects jailbreak attempts and role manipulation | Pattern matching |
| **Custom Rules** | User-defined content policies | Keyword/Regex rules |

### Integration Options

- **REST API**: FastAPI server with OpenAPI documentation
- **CLI**: Command-line interface for automation and scripting
- **Python SDK**: Direct integration into Python applications
- **Dashboard**: Interactive Streamlit web interface
- **Webhooks**: Real-time notifications for failed evaluations

### Enterprise Features

- **Persistence**: SQLAlchemy with SQLite/PostgreSQL support
- **Caching**: In-memory LRU cache with optional Redis
- **Metrics**: Prometheus-compatible metrics endpoint
- **Monitoring**: Grafana dashboards for visualization
- **Audit Logging**: Complete evaluation history with timestamps

## 📦 Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/llm-guardrail-studio.git
cd llm-guardrail-studio

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
```

### Production Installation

```bash
# Install production dependencies
pip install -r requirements-prod.txt

# Start API server
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f guardrail-api

# Stop services
docker-compose down
```

## 🔧 Usage

### Python SDK

```python
from guardrails import GuardrailPipeline

# Initialize with all evaluators
pipeline = GuardrailPipeline(
    enable_toxicity=True,
    enable_hallucination=True,
    enable_alignment=True,
    enable_pii=True,
    enable_injection=True,
    enable_custom_rules=True,
    toxicity_threshold=0.7,
    alignment_threshold=0.5
)

# Evaluate content
result = pipeline.evaluate(
    prompt="What is your email address?",
    response="My email is john.doe@example.com"
)

print(f"Passed: {result.passed}")
print(f"Scores: {result.scores}")
print(f"Flags: {result.flags}")

# Access detailed results
for evaluator, details in result.detailed_results.items():
    print(f"{evaluator}: {details}")
```

### REST API

```bash
# Start server
uvicorn api.server:app --reload

# Evaluate content
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "response": "Hi there!"}'

# Batch evaluation
curl -X POST http://localhost:8000/evaluate/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{"prompt": "Q1", "response": "A1"}, {"prompt": "Q2", "response": "A2"}]}'

# Get metrics
curl http://localhost:8000/metrics

# API documentation at http://localhost:8000/docs
```

### CLI

```bash
# Single evaluation
guardrails evaluate --prompt "What is AI?" --response "AI is artificial intelligence."

# Evaluate with specific evaluator
guardrails check --evaluator toxicity --text "Your text here"

# Batch evaluation from CSV
guardrails evaluate-file data.csv --output results.csv

# Start API server
guardrails server --host 0.0.0.0 --port 8000

# Manage custom rules
guardrails rules list
guardrails rules add --name no_spam --type keyword --pattern "spam" --action flag
```

### Dashboard

```bash
streamlit run app.py
```

Features: Single/batch evaluation, analytics, custom rules management, API integration.

## 🔒 Security Evaluators

### PII Detection

Detects: emails, phone numbers, SSN, credit cards, addresses.

```python
from guardrails.evaluators import PIIDetector

detector = PIIDetector()
score, has_pii = detector.evaluate("Contact me at john@email.com")
# has_pii = True
```

### Prompt Injection Detection

Detects: ignore instructions, role manipulation, extraction attempts, jailbreaks.

```python
from guardrails.evaluators import PromptInjectionDetector

detector = PromptInjectionDetector()
score, is_injection = detector.evaluate("Ignore all previous instructions")
# is_injection = True
```

### Custom Rules

```python
from guardrails.evaluators import CustomRuleEvaluator

rules = CustomRuleEvaluator()
rules.add_rule("no_competitor", "keyword", "competitor|rival", "flag")
result = rules.evaluate("Our competitor is better")
# result['has_violations'] = True
```

## ⚙️ Configuration

Environment variables (see `.env.example`):

```bash
API_KEY=your-secret-key
DATABASE_URL=sqlite:///guardrails.db
TOXICITY_THRESHOLD=0.7
ENABLE_PII=true
ENABLE_INJECTION=true
```

## 📊 Monitoring

### Endpoints

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | REST API |
| Docs | http://localhost:8000/docs | OpenAPI |
| Dashboard | http://localhost:8501 | Streamlit |
| Prometheus | http://localhost:9090 | Metrics |
| Grafana | http://localhost:3000 | Dashboards |

## 🧪 Testing

```bash
pytest                              # Run all tests
pytest --cov=guardrails            # With coverage
pytest tests/test_security_evaluators.py  # Specific tests
```

## 📁 Project Structure

```
llm-guardrail-studio/
├── guardrails/           # Core library
│   ├── evaluators.py    # All evaluator classes
│   ├── pipeline.py      # GuardrailPipeline
├── api/                  # REST API (FastAPI)
├── cli.py               # CLI interface
├── persistence/         # Database layer
├── dashboard/           # Streamlit components
├── app.py               # Dashboard entry point
├── tests/               # Test suite
├── Dockerfile           # Container image
└── docker-compose.yml   # Multi-service deployment
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests
4. Submit a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE).

---

**Built with ❤️ for safer AI applications**
