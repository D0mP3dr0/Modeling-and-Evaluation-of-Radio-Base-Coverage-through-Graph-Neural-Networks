"""
Radio Base Station Network Analysis and Modeling

A comprehensive toolkit for analyzing radio base station networks using
graph theory, coverage modeling, and advanced visualization techniques.
"""

__version__ = '1.2.0'
__author__ = 'RBS Analysis Team'

# Import main modules for easier access
from .data_processing import load_and_process_data
from .config import setup_logging

# Setup default logger
logger = setup_logging('rbs_analysis.log')
