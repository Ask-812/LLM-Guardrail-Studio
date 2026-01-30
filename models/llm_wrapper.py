"""
Wrapper for local LLM models with guardrail integration
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LLMWrapper:
    """
    Wrapper class for local LLM models with integrated guardrails
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.temperature = temperature
        self.do_sample = do_sample
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded")
        
        generation_kwargs = {
            "max_length": kwargs.get("max_length", self.max_length),
            "temperature": kwargs.get("temperature", self.temperature),
            "do_sample": kwargs.get("do_sample", self.do_sample),
            "pad_token_id": self.tokenizer.eos_token_id,
            "return_full_text": False
        }
        
        try:
            outputs = self.pipeline(prompt, **generation_kwargs)
            response = outputs[0]["generated_text"].strip()
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "parameters": self.model.num_parameters() if self.model else None
        }