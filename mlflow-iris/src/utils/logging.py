"""
Logging Utilities
=================

Standardized logging configuration for the project.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format_string: str = None,
    log_file: str = None,
) -> logging.Logger:
    """
    Setup standardized logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_string: Custom format string
        log_file: Optional file to write logs to
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Initialize default logging
logger = get_logger(__name__)