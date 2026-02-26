"""
Logging configuration for LLM Guardrail Studio
Provides structured logging with console and file handlers
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from functools import wraps
import time


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        log_level = record.levelname
        color = self.COLORS.get(log_level, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format the message
        formatted = super().format(record)
        
        # Add color to the level name
        formatted = formatted.replace(
            log_level,
            f"{color}{log_level}{reset}"
        )
        
        return formatted


def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_execution_time(logger: logging.Logger):
    """
    Decorator to log function execution time
    
    Args:
        logger: Logger instance
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.debug(f"Starting {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                logger.info(f"{func.__name__} completed in {elapsed_time:.3f}s")
                return result
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {elapsed_time:.3f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator


def log_evaluation_result(logger: logging.Logger, result):
    """
    Log evaluation result details
    
    Args:
        logger: Logger instance
        result: GuardrailResult object
    """
    logger.info(f"Evaluation result: {'PASSED' if result.passed else 'FAILED'}")
    
    if result.scores:
        scores_str = ', '.join(f"{k}={v:.3f}" for k, v in result.scores.items())
        logger.debug(f"Scores: {scores_str}")
    
    if result.flags:
        for flag in result.flags:
            logger.warning(f"Flag: {flag}")
    
    if result.metadata:
        logger.debug(f"Metadata: {result.metadata}")


# Default logger instance
logger = setup_logger(
    'guardrails',
    log_file='logs/guardrails.log',
    level=logging.INFO
)


__all__ = [
    'setup_logger',
    'log_execution_time',
    'log_evaluation_result',
    'ColoredFormatter',
    'logger'
]
