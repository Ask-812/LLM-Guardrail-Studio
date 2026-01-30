"""
Model loading utilities and configurations
"""

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Utility class for managing model configurations and loading"""
    
    SUPPORTED_MODELS = {
        "mistral-7b": {
            "name": "mistralai/Mistral-7B-Instruct-v0.2",
            "type": "instruct",
            "size": "7B",
            "description": "Mistral 7B Instruct model"
        },
        "zephyr-7b": {
            "name": "HuggingFaceH4/zephyr-7b-beta",
            "type": "chat",
            "size": "7B", 
            "description": "Zephyr 7B Beta chat model"
        },
        "llama2-7b": {
            "name": "meta-llama/Llama-2-7b-chat-hf",
            "type": "chat",
            "size": "7B",
            "description": "Llama 2 7B Chat model"
        },
        "phi-2": {
            "name": "microsoft/phi-2",
            "type": "base",
            "size": "2.7B",
            "description": "Microsoft Phi-2 model"
        }
    }
    
    @classmethod
    def get_supported_models(cls) -> Dict[str, Dict]:
        """Get list of supported models"""
        return cls.SUPPORTED_MODELS
    
    @classmethod
    def get_model_names(cls) -> List[str]:
        """Get list of model names"""
        return list(cls.SUPPORTED_MODELS.keys())
    
    @classmethod
    def get_model_config(cls, model_key: str) -> Dict:
        """Get configuration for a specific model"""
        if model_key not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_key} not supported. Available: {cls.get_model_names()}")
        
        return cls.SUPPORTED_MODELS[model_key]
    
    @classmethod
    def validate_model(cls, model_name: str) -> bool:
        """Validate if model is supported"""
        return model_name in cls.SUPPORTED_MODELS or any(
            config["name"] == model_name for config in cls.SUPPORTED_MODELS.values()
        )