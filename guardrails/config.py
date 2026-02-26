"""
Configuration management for LLM Guardrail Studio
Handles loading, validating, and managing pipeline configurations
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging


logger = logging.getLogger(__name__)


@dataclass
class GuardrailConfig:
    """Configuration dataclass for the guardrail pipeline"""
    
    # Model configuration
    model_name: str = "mistralai/Mistral-7B-v0.1"
    
    # Evaluator toggles
    enable_toxicity: bool = True
    enable_hallucination: bool = True
    enable_alignment: bool = True
    
    # Thresholds
    toxicity_threshold: float = 0.7
    alignment_threshold: float = 0.5
    hallucination_threshold: float = 0.6
    
    # Performance settings
    batch_size: int = 32
    max_workers: int = 4
    timeout: int = 30
    
    # Logging
    enable_logging: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/guardrails.log"
    
    def validate(self) -> bool:
        """
        Validate configuration values
        
        Returns:
            bool: True if valid, raises ValueError otherwise
        """
        errors = []
        
        # Validate thresholds are between 0 and 1
        if not 0 <= self.toxicity_threshold <= 1:
            errors.append("toxicity_threshold must be between 0 and 1")
        
        if not 0 <= self.alignment_threshold <= 1:
            errors.append("alignment_threshold must be between 0 and 1")
        
        if not 0 <= self.hallucination_threshold <= 1:
            errors.append("hallucination_threshold must be between 0 and 1")
        
        # Validate positive integers
        if self.batch_size < 1:
            errors.append("batch_size must be positive")
        
        if self.max_workers < 1:
            errors.append("max_workers must be positive")
        
        if self.timeout < 1:
            errors.append("timeout must be positive")
        
        # Validate log level
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if self.log_level.upper() not in valid_levels:
            errors.append(f"log_level must be one of {valid_levels}")
        
        if errors:
            raise ValueError("Configuration validation failed: " + "; ".join(errors))
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GuardrailConfig':
        """Create configuration from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def save(self, path: str) -> None:
        """
        Save configuration to JSON file
        
        Args:
            path: Path to save configuration
        """
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'GuardrailConfig':
        """
        Load configuration from JSON file
        
        Args:
            path: Path to configuration file
            
        Returns:
            GuardrailConfig instance
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = cls.from_dict(data)
        config.validate()
        logger.info(f"Configuration loaded from {path}")
        return config


class ConfigManager:
    """Manages configuration with environment variable overrides"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Path to configuration file, uses defaults if None
        """
        if config_file and Path(config_file).exists():
            self.config = GuardrailConfig.load(config_file)
            logger.info(f"Loaded config from {config_file}")
        else:
            self.config = GuardrailConfig()
            logger.info("Using default configuration")
        
        # Apply environment variable overrides
        self._apply_env_overrides()
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration"""
        env_mappings = {
            'GUARDRAILS_MODEL': 'model_name',
            'GUARDRAILS_TOXICITY_THRESHOLD': 'toxicity_threshold',
            'GUARDRAILS_ALIGNMENT_THRESHOLD': 'alignment_threshold',
            'GUARDRAILS_HALLUCINATION_THRESHOLD': 'hallucination_threshold',
            'GUARDRAILS_LOG_LEVEL': 'log_level',
            'GUARDRAILS_BATCH_SIZE': 'batch_size',
            'GUARDRAILS_MAX_WORKERS': 'max_workers',
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                # Convert to appropriate type
                if config_key in ['batch_size', 'max_workers', 'timeout']:
                    value = int(value)
                elif config_key in ['toxicity_threshold', 'alignment_threshold', 'hallucination_threshold']:
                    value = float(value)
                
                setattr(self.config, config_key, value)
                logger.debug(f"Applied environment override: {env_var}={value}")
    
    def get(self) -> GuardrailConfig:
        """Get the current configuration"""
        return self.config
    
    def update(self, **kwargs) -> None:
        """
        Update configuration values
        
        Args:
            **kwargs: Configuration key-value pairs to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.debug(f"Updated config: {key}={value}")
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        self.config.validate()


# Default configuration manager instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get or create the global configuration manager"""
    global _config_manager
    if _config_manager is None:
        config_file = os.getenv('GUARDRAILS_CONFIG_FILE')
        _config_manager = ConfigManager(config_file)
    return _config_manager


def get_config() -> GuardrailConfig:
    """Get the current configuration"""
    return get_config_manager().get()


__all__ = [
    'GuardrailConfig',
    'ConfigManager',
    'get_config_manager',
    'get_config'
]
