"""
Utility functions for the RBS Analysis project.

This module contains shared utility functions that are used across
multiple modules in the project.
"""

import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# Import configuration
from .config import (
    VIZ_DEFAULTS, OPERATOR_COLORS, TECHNOLOGY_COLORS,
    setup_logging
)

# Setup logging
logger = setup_logging('utils.log')

def ensure_directory(directory_path: str) -> str:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        str: Path to the directory
    """
    directory = os.path.abspath(directory_path)
    os.makedirs(directory, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory}")
    return directory

def safe_filename(name: str) -> str:
    """
    Convert a string to a safe filename by removing or replacing invalid characters.
    
    Args:
        name: Input string
        
    Returns:
        str: Safe filename
    """
    # Replace spaces with underscores
    safe_name = name.replace(' ', '_')
    
    # Remove invalid characters
    safe_name = re.sub(r'[^\w\-\.]', '', safe_name)
    
    # Ensure filename is not empty
    if not safe_name:
        safe_name = f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return safe_name

def save_dataframe(df: pd.DataFrame, output_path: str, format: str = 'auto') -> str:
    """
    Save a DataFrame to file in the specified format.
    
    Args:
        df: DataFrame to save
        output_path: Path to save the file
        format: File format ('csv', 'geojson', 'excel', or 'auto')
        
    Returns:
        str: Path to the saved file
    """
    # Ensure the directory exists
    ensure_directory(os.path.dirname(os.path.abspath(output_path)))
    
    # Determine format if auto
    if format == 'auto':
        ext = os.path.splitext(output_path)[1].lower()
        if ext in ['.geojson', '.json']:
            format = 'geojson'
        elif ext in ['.xlsx', '.xls']:
            format = 'excel'
        else:
            format = 'csv'
    
    try:
        logger.debug(f"Saving DataFrame to {output_path} in {format} format")
        
        # Save in the appropriate format
        if format == 'geojson':
            # Check if this is a GeoDataFrame
            if hasattr(df, 'crs') and hasattr(df, 'geometry'):
                df.to_file(output_path, driver='GeoJSON')
            else:
                logger.warning(f"Cannot save as GeoJSON: Not a GeoDataFrame. Saving as CSV instead.")
                df.to_csv(output_path.replace('.geojson', '.csv'), index=False)
                return output_path.replace('.geojson', '.csv')
        elif format == 'excel':
            df.to_excel(output_path, index=False)
        else:  # Default to CSV
            df.to_csv(output_path, index=False)
        
        logger.info(f"Successfully saved DataFrame with {len(df)} rows to {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error saving DataFrame to {output_path}: {e}")
        # Try to save to a backup location
        backup_path = f"{os.path.splitext(output_path)[0]}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            df.to_csv(backup_path, index=False)
            logger.info(f"Saved backup to {backup_path}")
            return backup_path
        except Exception as backup_e:
            logger.error(f"Failed to save backup: {backup_e}")
            return ""

def get_color_for_operator(operator: str) -> str:
    """
    Get a consistent color for a specific operator.
    
    Args:
        operator: Operator name
        
    Returns:
        str: Color code (hex)
    """
    # Handle None or empty string
    if not operator:
        return OPERATOR_COLORS.get('N/A', '#CCCCCC')
    
    # Standardize operator name
    operator_upper = operator.upper()
    
    # Direct match
    if operator_upper in OPERATOR_COLORS:
        return OPERATOR_COLORS[operator_upper]
    
    # Partial match
    for key, color in OPERATOR_COLORS.items():
        if key in operator_upper:
            return color
    
    # Default
    return OPERATOR_COLORS.get('OTHER', '#CCCCCC')

def get_color_for_technology(technology: str) -> str:
    """
    Get a consistent color for a specific technology.
    
    Args:
        technology: Technology name
        
    Returns:
        str: Color code (hex)
    """
    # Handle None or empty string
    if not technology:
        return TECHNOLOGY_COLORS.get('N/A', '#CCCCCC')
    
    # Standardize technology name
    tech_upper = technology.upper()
    
    # Direct match (case insensitive)
    for key, color in TECHNOLOGY_COLORS.items():
        if key.upper() == tech_upper:
            return color
    
    # Partial match
    for key, color in TECHNOLOGY_COLORS.items():
        if key.upper() in tech_upper:
            return color
    
    # Default
    return TECHNOLOGY_COLORS.get('N/A', '#CCCCCC')

def setup_plot_style(theme: str = 'default') -> None:
    """
    Set up the plotting style for consistent visualizations.
    
    Args:
        theme: Name of the theme to use
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    if theme == 'dark':
        plt.rcParams.update({
            'figure.facecolor': '#1f1f1f',
            'axes.facecolor': '#1f1f1f',
            'axes.edgecolor': 'white',
            'axes.labelcolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'text.color': 'white',
            'grid.color': '#444444'
        })
    else:
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.labelcolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black',
            'text.color': 'black',
            'grid.color': '#cccccc'
        })
    
    # Set font sizes
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    logger.debug(f"Plot style set to '{theme}'")

def format_filename_with_timestamp(base_name: str, extension: str = '.png') -> str:
    """
    Format a filename with a timestamp to ensure uniqueness.
    
    Args:
        base_name: Base name for the file
        extension: File extension
        
    Returns:
        str: Formatted filename with timestamp
    """
    # Clean the base name
    clean_name = safe_filename(base_name)
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Ensure extension starts with a dot
    if not extension.startswith('.'):
        extension = f".{extension}"
    
    return f"{clean_name}_{timestamp}{extension}"

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on earth (haversine formula).
    
    Args:
        lat1: Latitude of point 1 (degrees)
        lon1: Longitude of point 1 (degrees)
        lat2: Latitude of point 2 (degrees)
        lon2: Longitude of point 2 (degrees)
        
    Returns:
        float: Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    
    return c * r

def save_dict_to_json(data: Dict, output_path: str) -> str:
    """
    Save a dictionary to a JSON file.
    
    Args:
        data: Dictionary to save
        output_path: Path to save the JSON file
        
    Returns:
        str: Path to the saved file
    """
    # Ensure the directory exists
    ensure_directory(os.path.dirname(os.path.abspath(output_path)))
    
    try:
        # Convert numpy types to Python types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, default=convert_numpy, indent=2)
        
        logger.info(f"Successfully saved dictionary to {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error saving dictionary to {output_path}: {e}")
        return ""

def load_json_to_dict(input_path: str) -> Dict:
    """
    Load a JSON file into a dictionary.
    
    Args:
        input_path: Path to the JSON file
        
    Returns:
        dict: Loaded dictionary
    """
    try:
        if not os.path.exists(input_path):
            logger.error(f"JSON file not found: {input_path}")
            return {}
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Successfully loaded dictionary from {input_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading dictionary from {input_path}: {e}")
        return {} 