"""
Configuration Module for RBS Analysis

This module centralizes all configuration settings for the RBS analysis system,
including paths, constants, logging configuration, and environment setup.
"""

import os
import sys
import logging
import json
from pathlib import Path
import warnings
from typing import Dict, List, Optional, Union, Any

# Determine if we're running in a development, test, or production environment
ENVIRONMENT = os.environ.get('RBS_ENVIRONMENT', 'development')

# Base directories
ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
CACHE_DIR = os.path.join(ROOT_DIR, 'cache')

# Create directories if they don't exist
for directory in [DATA_DIR, RESULTS_DIR, LOG_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Input and output paths
DEFAULT_INPUT_PATH = os.path.join(DATA_DIR, 'input_data.csv')
DEFAULT_OUTPUT_PATH = os.path.join(RESULTS_DIR, 'processed_data.csv')

# Default values for missing data
DEFAULT_POWER_WATTS = 25.0         # Default transmitter power in Watts
DEFAULT_ANTENNA_GAIN = 15.0        # Default antenna gain in dBi
DEFAULT_FREQUENCY_MHZ = 850.0      # Default frequency in MHz
DEFAULT_AZIMUTHS = [0, 120, 240]   # Default azimuths for 3-sector sites

# Region settings
DEFAULT_REGION_BBOX = [-23.0, -22.5, -43.5, -43.0]  # Default bounding box [lat_min, lat_max, lon_min, lon_max]

# Performance settings
BATCH_SIZE = 1000                 # Default batch size for processing
MULTIPROCESSING_THRESHOLD = 10000  # Threshold for switching to multiprocessing
GPU_MEMORY_LIMIT = 0.8            # Maximum fraction of GPU memory to use (0.0-1.0)
CACHE_ENABLED = True              # Whether to use result caching

# Operator name standardization mapping
OPERATOR_MAPPING = {
    'CLARO': 'CLARO',
    'AMERICEL': 'CLARO',
    'BCP': 'CLARO',
    'TELECOM AMERICAS': 'CLARO',
    
    'TIM': 'TIM',
    'TIM CELULAR': 'TIM',
    'TIM NORDESTE': 'TIM',
    
    'VIVO': 'VIVO',
    'TELEFONICA': 'VIVO',
    'TELEMIG': 'VIVO',
    
    'OI': 'OI',
    'TNL': 'OI',
    'TELEMAR': 'OI',
    'BRASIL TELECOM': 'OI'
}

# Logging configuration
def setup_logging(log_filename: Optional[str] = None, 
                 console_level: int = logging.INFO,
                 file_level: int = logging.DEBUG) -> logging.Logger:
    """
    Configure logging with both console and file handlers.
    
    Args:
        log_filename: Name of the log file (if None, a default will be used)
        console_level: Logging level for console output
        file_level: Logging level for file output
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger('rbs_analysis')
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # Create file handler if requested
    if log_filename:
        if not log_filename.endswith('.log'):
            log_filename += '.log'
        
        log_path = os.path.join(LOG_DIR, log_filename)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(file_level)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger

# Create default logger
logger = setup_logging('rbs_analysis.log')

# Function to load environment-specific configuration
def load_env_config() -> Dict[str, Any]:
    """
    Load environment-specific configuration from JSON file
    
    Returns:
        dict: Configuration settings
    """
    config_file = os.path.join(ROOT_DIR, f'config_{ENVIRONMENT}.json')
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading config file {config_file}: {e}")
            logger.warning("Using default configuration.")
    else:
        logger.info(f"No environment-specific config file found at {config_file}.")
        logger.info("Using default configuration.")
    
    return {}

# Load environment-specific configuration
ENV_CONFIG = load_env_config()

# Override defaults with environment-specific settings
for key, value in ENV_CONFIG.items():
    if key in globals():
        globals()[key] = value
    else:
        logger.warning(f"Unknown configuration key in environment config: {key}")

# GPU configuration
# This will be replaced with proper environment variable by main.py
USE_GPU = os.environ.get('USE_GPU', '').lower() == 'true'

def user_config_path(config_name: str = 'user_config.json') -> str:
    """
    Get the path to a user configuration file
    
    Args:
        config_name: Name of the configuration file
        
    Returns:
        str: Full path to the configuration file
    """
    return os.path.join(ROOT_DIR, config_name)

def save_user_config(config: Dict[str, Any], config_name: str = 'user_config.json') -> bool:
    """
    Save user configuration to file
    
    Args:
        config: Configuration dictionary to save
        config_name: Name of the configuration file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        config_path = user_config_path(config_name)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving user configuration: {e}")
        return False

def load_user_config(config_name: str = 'user_config.json') -> Dict[str, Any]:
    """
    Load user configuration from file
    
    Args:
        config_name: Name of the configuration file
        
    Returns:
        dict: User configuration settings
    """
    config_path = user_config_path(config_name)
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading user config file {config_path}: {e}")
    
    return {}

# Function to get configuration value with fallback
def get_config(key: str, default: Any = None) -> Any:
    """
    Get a configuration value with fallback to default
    
    Args:
        key: Configuration key
        default: Default value if key is not found
        
    Returns:
        Any: Configuration value
    """
    # First check user config
    user_cfg = load_user_config()
    if key in user_cfg:
        return user_cfg[key]
    
    # Then check environment config
    if key in ENV_CONFIG:
        return ENV_CONFIG[key]
    
    # Then check globals
    if key in globals():
        return globals()[key]
    
    # Finally use provided default
    return default

def print_config_summary():
    """Print a summary of the configuration settings"""
    print("\n=== RBS Analysis Configuration ===")
    print(f"Environment: {ENVIRONMENT}")
    print(f"Root directory: {ROOT_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Log directory: {LOG_DIR}")
    print(f"GPU acceleration enabled: {USE_GPU}")
    print(f"Cache enabled: {CACHE_ENABLED}")
    print("==================================\n")

if __name__ == "__main__":
    # If run directly, print configuration summary
    print_config_summary()
    
    # And save a default user config if it doesn't exist
    if not os.path.exists(user_config_path()):
        default_user_config = {
            "DEFAULT_REGION_BBOX": DEFAULT_REGION_BBOX,
            "USE_GPU": USE_GPU,
            "CACHE_ENABLED": CACHE_ENABLED
        }
        save_user_config(default_user_config)
        print(f"Default user configuration saved to {user_config_path()}") 