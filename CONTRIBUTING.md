# Contributing to LLM Guardrail Studio

Thank you for your interest in contributing to LLM Guardrail Studio! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ask-812/LLM-Guardrail-Studio.git
   cd LLM-Guardrail-Studio
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   make install-dev
   # or
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

## Code Style

We use the following tools for code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run formatting and linting:
```bash
make format
make lint
```

## Testing

We use pytest for testing. Run tests with:

```bash
make test
```

This will run tests with coverage reporting.

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Include both positive and negative test cases
- Mock external dependencies when appropriate

Example test structure:
```python
def test_evaluator_functionality():
    """Test that evaluator works correctly"""
    evaluator = SomeEvaluator()
    result = evaluator.evaluate("test input")
    assert result >= 0.0
    assert result <= 1.0
```

## Adding New Evaluators

To add a new evaluator:

1. **Create the evaluator class** in `guardrails/evaluators.py`:
   ```python
   class NewEvaluator:
       def __init__(self):
           # Initialize your evaluator
           pass
       
       def evaluate(self, text: str) -> float:
           # Return score between 0 and 1
           return 0.0
   ```

2. **Add to pipeline** in `guardrails/pipeline.py`:
   - Import the evaluator
   - Add initialization logic
   - Add evaluation logic

3. **Write tests** in `tests/test_evaluators.py`

4. **Update documentation** in README.md

## Adding Dashboard Components

Dashboard components go in `dashboard/components.py`. Follow the existing pattern:

```python
class NewComponent:
    @staticmethod
    def render_something(data):
        """Render component with Streamlit"""
        st.write("Component content")
```

## Submitting Changes

1. **Fork the repository** on GitHub

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**
   ```bash
   make test
   make lint
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

## Pull Request Guidelines

- **Clear title and description**: Explain what your PR does and why
- **Reference issues**: Link to any related issues
- **Small, focused changes**: Keep PRs manageable in size
- **Tests included**: Add tests for new functionality
- **Documentation updated**: Update README or other docs as needed

## Issue Reporting

When reporting issues:

- **Use a clear title** that describes the problem
- **Provide steps to reproduce** the issue
- **Include error messages** and stack traces
- **Specify your environment** (OS, Python version, etc.)
- **Add relevant code samples** if applicable

## Feature Requests

For feature requests:

- **Describe the use case** and why it's needed
- **Provide examples** of how it would be used
- **Consider implementation** if you have ideas

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain a positive environment

## Questions?

If you have questions about contributing:

- Check existing issues and discussions
- Create a new issue with the "question" label
- Reach out to maintainers

Thank you for contributing to LLM Guardrail Studio! 🚀