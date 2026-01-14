"""
Basic usage example for LLM Guardrail Studio
"""

from guardrails import GuardrailPipeline


def main():
    # Initialize the pipeline
    pipeline = GuardrailPipeline(
        model_name="mistralai/Mistral-7B-v0.1",
        enable_toxicity=True,
        enable_hallucination=True,
        enable_alignment=True
    )
    
    # Example 1: Safe response
    print("Example 1: Safe Response")
    print("-" * 50)
    result = pipeline.evaluate(
        prompt="What is the capital of France?",
        response="The capital of France is Paris."
    )
    print(f"Scores: {result.scores}")
    print(f"Passed: {result.passed}")
    print(f"Flags: {result.flags}\n")
    
    # Example 2: Toxic response
    print("Example 2: Toxic Response")
    print("-" * 50)
    result = pipeline.evaluate(
        prompt="Tell me about politics",
        response="I hate all politicians, they are terrible people."
    )
    print(f"Scores: {result.scores}")
    print(f"Passed: {result.passed}")
    print(f"Flags: {result.flags}\n")
    
    # Example 3: Misaligned response
    print("Example 3: Misaligned Response")
    print("-" * 50)
    result = pipeline.evaluate(
        prompt="What is machine learning?",
        response="The weather today is sunny and warm."
    )
    print(f"Scores: {result.scores}")
    print(f"Passed: {result.passed}")
    print(f"Flags: {result.flags}\n")


if __name__ == "__main__":
    main()
