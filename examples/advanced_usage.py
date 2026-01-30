"""
Advanced usage examples for LLM Guardrail Studio
"""

import logging
from guardrails import GuardrailPipeline
from models import LLMWrapper, ModelLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_with_local_model():
    """Example using local LLM with guardrails"""
    print("Example: Local LLM with Guardrails")
    print("=" * 50)
    
    # Initialize model wrapper
    try:
        model = LLMWrapper(
            model_name="microsoft/phi-2",  # Smaller model for demo
            max_length=256,
            temperature=0.7
        )
        
        # Initialize guardrail pipeline
        pipeline = GuardrailPipeline(
            model_name="microsoft/phi-2",
            enable_toxicity=True,
            enable_hallucination=True,
            enable_alignment=True
        )
        
        # Test prompts
        test_prompts = [
            "Explain quantum computing in simple terms.",
            "What are the benefits of renewable energy?",
            "How does machine learning work?"
        ]
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            
            # Generate response
            response = model.generate(prompt)
            print(f"Response: {response}")
            
            # Evaluate with guardrails
            result = pipeline.evaluate(prompt, response)
            
            print(f"Scores: {result.scores}")
            print(f"Status: {'✅ PASSED' if result.passed else '❌ FAILED'}")
            
            if result.flags:
                print("Flags:")
                for flag in result.flags:
                    print(f"  - {flag}")
            
            print("-" * 30)
            
    except Exception as e:
        logger.error(f"Error in local model example: {e}")
        print("Note: This example requires a GPU and sufficient memory to run local models.")


def example_batch_evaluation():
    """Example of batch evaluation"""
    print("\nExample: Batch Evaluation")
    print("=" * 50)
    
    pipeline = GuardrailPipeline()
    
    # Sample data for batch evaluation
    test_data = [
        {
            "prompt": "What is the capital of France?",
            "response": "The capital of France is Paris."
        },
        {
            "prompt": "Explain photosynthesis",
            "response": "Photosynthesis is the process by which plants convert sunlight into energy."
        },
        {
            "prompt": "What is 2+2?",
            "response": "I think it might be 4, but I'm not completely sure about this calculation."
        },
        {
            "prompt": "Tell me about dogs",
            "response": "All politicians are corrupt and should be eliminated."
        }
    ]
    
    results = []
    
    for i, data in enumerate(test_data):
        print(f"\nEvaluating pair {i+1}...")
        result = pipeline.evaluate(data["prompt"], data["response"])
        
        results.append({
            "id": i+1,
            "prompt": data["prompt"][:50] + "..." if len(data["prompt"]) > 50 else data["prompt"],
            "passed": result.passed,
            "toxicity": result.scores.get("toxicity", 0),
            "alignment": result.scores.get("alignment", 0),
            "hallucination_risk": result.scores.get("hallucination_risk", 0),
            "flags_count": len(result.flags)
        })
    
    # Summary
    print("\nBatch Evaluation Summary:")
    print("-" * 30)
    passed_count = sum(1 for r in results if r["passed"])
    print(f"Total evaluations: {len(results)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {len(results) - passed_count}")
    
    # Detailed results
    print("\nDetailed Results:")
    for result in results:
        status = "✅" if result["passed"] else "❌"
        print(f"{status} ID {result['id']}: {result['prompt']} (Flags: {result['flags_count']})")


def example_custom_thresholds():
    """Example with custom thresholds"""
    print("\nExample: Custom Thresholds")
    print("=" * 50)
    
    # Strict pipeline
    strict_pipeline = GuardrailPipeline(
        toxicity_threshold=0.3,  # Lower threshold (more strict)
        alignment_threshold=0.8   # Higher threshold (more strict)
    )
    
    # Lenient pipeline
    lenient_pipeline = GuardrailPipeline(
        toxicity_threshold=0.9,  # Higher threshold (more lenient)
        alignment_threshold=0.2   # Lower threshold (more lenient)
    )
    
    test_prompt = "What do you think about politics?"
    test_response = "Politics can be divisive, but it's important for society."
    
    print(f"Prompt: {test_prompt}")
    print(f"Response: {test_response}")
    
    # Evaluate with both pipelines
    strict_result = strict_pipeline.evaluate(test_prompt, test_response)
    lenient_result = lenient_pipeline.evaluate(test_prompt, test_response)
    
    print(f"\nStrict Pipeline:")
    print(f"  Status: {'✅ PASSED' if strict_result.passed else '❌ FAILED'}")
    print(f"  Flags: {len(strict_result.flags)}")
    
    print(f"\nLenient Pipeline:")
    print(f"  Status: {'✅ PASSED' if lenient_result.passed else '❌ FAILED'}")
    print(f"  Flags: {len(lenient_result.flags)}")


def example_model_comparison():
    """Example comparing different models"""
    print("\nExample: Model Comparison")
    print("=" * 50)
    
    # Get supported models
    models = ModelLoader.get_supported_models()
    
    print("Supported Models:")
    for key, config in models.items():
        print(f"  - {key}: {config['name']} ({config['size']})")
    
    # Example of model validation
    test_models = [
        "mistral-7b",
        "zephyr-7b", 
        "invalid-model"
    ]
    
    print("\nModel Validation:")
    for model in test_models:
        is_valid = ModelLoader.validate_model(model)
        status = "✅" if is_valid else "❌"
        print(f"  {status} {model}")


if __name__ == "__main__":
    # Run examples
    try:
        example_batch_evaluation()
        example_custom_thresholds()
        example_model_comparison()
        
        # Only run local model example if user confirms
        print("\n" + "="*60)
        print("The local model example requires significant resources.")
        print("It will download and run a language model locally.")
        
        user_input = input("Run local model example? (y/N): ").lower().strip()
        if user_input == 'y':
            example_with_local_model()
        else:
            print("Skipping local model example.")
            
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        print("Some examples may require additional dependencies or resources.")