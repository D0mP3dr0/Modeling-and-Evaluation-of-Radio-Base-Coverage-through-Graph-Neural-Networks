"""
Radio Base Stations (RBS) Analysis Package.

This package provides tools for analyzing radio base station data,
including coverage analysis, network topology, and visualization.
"""

# Version information
__version__ = '1.0.0'

# Import main modules for easier access
from .data_processing import load_and_process_data
from .analysis import run_basic_analysis
from .visualization import create_visualizations
from .graph_analysis import run_graph_analysis
from .coverage_models import estimate_coverage

# Import advanced modules
try:
    from .tech_frequency_analysis import run_tech_frequency_analysis
    from .advanced_temporal_analysis import run_temporal_analysis
    from .correlation_analysis import run_correlation_analysis
    from .spatial_analysis import run_spatial_analysis
    from .integration_analysis import run_integration_analysis
    from .prediction_module import run_prediction_analysis
    from .advanced_coverage_visualization import run_advanced_coverage_visualization
    from .coverage_quality_analysis import run_coverage_quality_analysis
    from .coverage_prediction import run_coverage_prediction
    from .advanced_graph_analysis import run_advanced_graph_analysis
    from .educational_documentation import create_educational_documentation
except ImportError as e:
    import logging
    logging.warning(f"Some advanced modules could not be imported: {e}")

# Setup logging
from .config import setup_logging
logger = setup_logging('rbs_analysis.log')
