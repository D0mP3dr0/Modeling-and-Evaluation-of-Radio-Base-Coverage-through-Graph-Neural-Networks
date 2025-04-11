"""
Pytest configuration and fixtures for RBS Analysis tests
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Ensure src directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules
from src.config import CACHE_DIR

# Create cache directory for tests if needed
os.makedirs(CACHE_DIR, exist_ok=True)

@pytest.fixture
def sample_rbs_data():
    """
    Fixture providing sample RBS data for tests
    
    Returns:
        DataFrame: Sample data with RBS information
    """
    # Create a small sample dataset
    data = {
        'Latitude': [-23.45, -23.42, -23.40, -23.44, -23.47],
        'Longitude': [-46.65, -46.63, -46.60, -46.68, -46.66],
        'Operator': ['CLARO', 'TIM', 'VIVO', 'OI', 'CLARO'],
        'Tecnologia': ['4G', '3G', '4G', '3G', '5G'],
        'FreqTxMHz': [2600, 850, 1800, 850, 3500],
        'PotenciaTransmissorWatts': [40, 20, 30, 20, 15],
        'GanhoAntena': [15, 18, 16, 14, 20],
        'Azimute': [0, 120, 240, 0, 120]
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_geo_rbs_data(sample_rbs_data):
    """
    Fixture providing sample geopandas GeoDataFrame for tests
    
    Returns:
        GeoDataFrame: Sample GeoDataFrame with RBS information
    """
    # Create a copy of the sample data
    df = sample_rbs_data.copy()
    
    # Convert to GeoDataFrame
    geometry = [Point(lon, lat) for lon, lat in zip(df['Longitude'], df['Latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    return gdf

@pytest.fixture
def sample_graph():
    """
    Fixture providing a sample NetworkX graph for tests
    
    Returns:
        nx.Graph: Sample graph with RBS nodes
    """
    import networkx as nx
    
    G = nx.Graph()
    
    # Add nodes
    nodes_data = [
        (0, {'pos': (-46.65, -23.45), 'operator': 'CLARO', 'technology': '4G', 'power': 40}),
        (1, {'pos': (-46.63, -23.42), 'operator': 'TIM', 'technology': '3G', 'power': 20}),
        (2, {'pos': (-46.60, -23.40), 'operator': 'VIVO', 'technology': '4G', 'power': 30}),
        (3, {'pos': (-46.68, -23.44), 'operator': 'OI', 'technology': '3G', 'power': 20}),
        (4, {'pos': (-46.66, -23.47), 'operator': 'CLARO', 'technology': '5G', 'power': 15})
    ]
    
    G.add_nodes_from(nodes_data)
    
    # Add edges
    edges_data = [
        (0, 1, {'weight': 1.0, 'distance': 0.03}),
        (0, 3, {'weight': 1.5, 'distance': 0.04}),
        (0, 4, {'weight': 2.0, 'distance': 0.02}),
        (1, 2, {'weight': 1.0, 'distance': 0.03}),
        (2, 3, {'weight': 0.5, 'distance': 0.08})
    ]
    
    G.add_edges_from(edges_data)
    
    return G

@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Fixture providing a temporary directory for test outputs
    
    Returns:
        Path: Path to temporary directory
    """
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir

@pytest.fixture
def gpu_available():
    """
    Fixture to check if GPU is available
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    from src.gpu_utils import GPU_AVAILABLE
    return GPU_AVAILABLE 