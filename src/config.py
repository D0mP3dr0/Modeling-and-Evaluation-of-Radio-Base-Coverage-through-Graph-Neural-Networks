"""
Configuration module for the RBS Analysis project.

This module centralizes all configuration parameters used throughout the project,
making it easier to modify settings without changing the code.
"""

import os
import logging
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Input/Output files
DEFAULT_INPUT_PATH = os.path.join(DATA_DIR, "csv_licenciamento_bruto.csv")
DEFAULT_OUTPUT_PATH = os.path.join(DATA_DIR, "erb_processed.csv")

# Create necessary directories
for directory in [DATA_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Region of interest (Sorocaba bbox)
# Format: [lat_min, lat_max, lon_min, lon_max]
DEFAULT_REGION_BBOX = [-23.60, -23.30, -47.65, -47.25]

# Operator mapping for standardization
OPERATOR_MAPPING = {
    'CLARO': 'CLARO',
    'CLARO S.A.': 'CLARO',
    'OI': 'OI',
    'OI MÓVEL S.A.': 'OI',
    'VIVO': 'VIVO',
    'TELEFÔNICA BRASIL S.A.': 'VIVO',
    'TIM': 'TIM',
    'TIM S.A.': 'TIM'
}

# Default values for missing data
DEFAULT_POWER_WATTS = 20.0
DEFAULT_ANTENNA_GAIN = 16.0
DEFAULT_FREQUENCY_MHZ = 850.0
DEFAULT_AZIMUTHS = [0, 120, 240]  # Standard for 3 sectors

# Visualization settings
VIZ_DEFAULTS = {
    'figsize_small': (10, 6),
    'figsize_medium': (12, 8),
    'figsize_large': (16, 10),
    'dpi': 300,
    'cmap': 'viridis',
    'marker_size': 50,
    'line_width': 1.5,
    'alpha': 0.7
}

# Colors for operators
OPERATOR_COLORS = {
    'CLARO': '#E02020',  # Red
    'OI': '#FFD700',     # Yellow
    'VIVO': '#9932CC',   # Purple
    'TIM': '#0000CD',    # Blue
    'OTHER': '#CCCCCC'   # Gray
}

# Colors for technologies
TECHNOLOGY_COLORS = {
    'GSM': '#66c2a5',
    '2G': '#66c2a5',
    '3G': '#fc8d62',
    '4G': '#8da0cb',
    '4G+': '#e78ac3',
    '5G': '#a6d854',
    'N/A': '#cccccc'
}

# Graph analysis parameters
GRAPH_PARAMS = {
    'max_edge_distance_km': 5.0,  # Maximum distance between connected nodes
    'node_size': 100,
    'edge_width': 0.7,
    'community_resolution': 1.0,  # Resolution parameter for community detection
}

# Coverage model parameters
COVERAGE_PARAMS = {
    'urban_factor': 1.0,
    'suburban_factor': 1.4,
    'rural_factor': 2.0,
    'urban_density_threshold': 10,  # RBS per km²
    'suburban_density_threshold': 3  # RBS per km²
}

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Setup logging
def setup_logging(log_file=None):
    """Configure logging for the project."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = os.path.join(LOGS_DIR, log_file)
        handlers.append(logging.FileHandler(log_path))
    
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers
    )
    
    # Reduce verbosity of matplotlib and other libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    return logging.getLogger('rbs_analysis') 