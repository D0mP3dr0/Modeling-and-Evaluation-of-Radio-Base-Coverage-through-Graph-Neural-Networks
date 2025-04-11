"""
Data processing module for RBS analysis.

This module handles loading, cleaning, and preprocessing data for RBS analysis.
It includes functions for data validation, cleaning, and transformation.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os
import logging
from shapely.geometry import Point
from typing import Dict, List, Optional, Tuple, Union

# Import configuration
from .config import (
    DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_PATH, OPERATOR_MAPPING,
    DEFAULT_POWER_WATTS, DEFAULT_ANTENNA_GAIN, DEFAULT_FREQUENCY_MHZ,
    DEFAULT_AZIMUTHS, DEFAULT_REGION_BBOX, setup_logging
)

# Setup logging
logger = setup_logging('data_processing.log')

def validate_input_file(file_path: str) -> bool:
    """
    Validate that the input file exists and has the expected format.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"Input file not found: {file_path}")
        return False
    
    # Check file extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ['.csv', '.geojson', '.shp']:
        logger.error(f"Unsupported file format: {ext}. Expected .csv, .geojson, or .shp")
        return False
    
    try:
        # Try to read the first few rows to validate format
        if ext == '.csv':
            df = pd.read_csv(file_path, nrows=5)
        else:
            df = gpd.read_file(file_path, rows=5)
        
        # Check for required columns
        required_columns = ['Latitude', 'Longitude']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {', '.join(missing_columns)}")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error validating input file: {e}")
        return False

def load_data(input_path: str = DEFAULT_INPUT_PATH) -> Optional[pd.DataFrame]:
    """
    Load data from the specified input file.
    
    Args:
        input_path: Path to the input file
        
    Returns:
        DataFrame or None: Loaded data or None if loading failed
    """
    # Validate input file
    if not validate_input_file(input_path):
        return None
    
    try:
        # Determine file type and load accordingly
        ext = os.path.splitext(input_path)[1].lower()
        
        if ext == '.csv':
            logger.info(f"Loading CSV data from {input_path}")
            df = pd.read_csv(input_path, encoding='utf-8')
        elif ext in ['.geojson', '.shp']:
            logger.info(f"Loading GeoSpatial data from {input_path}")
            df = gpd.read_file(input_path)
        else:
            logger.error(f"Unsupported file format: {ext}")
            return None
        
        logger.info(f"Successfully loaded {len(df)} records")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by removing unnecessary columns and handling missing values.
    
    Args:
        df: DataFrame to clean
        
    Returns:
        DataFrame: Cleaned DataFrame
    """
    if df is None or df.empty:
        logger.error("No data to clean")
        return pd.DataFrame()
    
    logger.info("Cleaning data...")
    
    # Make a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    # List of columns that can be safely excluded
    columns_to_exclude = [
        'NumFistel', 'NumServico', 'NumAto', 'CodDebitoTFI', '_id',
        'NumFistelAssociado', 'NumRede', 'NumEstacao',
        'DataLicenciamento', 'DataPrimeiroLicenciamento', 'DataValidade',
        'CodTipoAntena', 'CodEquipamentoAntena', 'CodEquipamentoTransmissor',
        'CodTipoClasseEstacao', 'DesignacaoEmissao', 'ClassInfraFisica',
        'CompartilhamentoInfraFisica', 'NomeEntidadeAssociado',
        'FrenteCostaAntena', 'AnguloMeiaPotenciaAntena'
    ]
    
    # Remove columns if they exist
    existing_columns = [col for col in columns_to_exclude if col in df_cleaned.columns]
    if existing_columns:
        logger.info(f"Removing {len(existing_columns)} unnecessary columns")
        df_cleaned = df_cleaned.drop(columns=existing_columns)
    
    # Remove duplicate records
    initial_count = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    duplicate_count = initial_count - len(df_cleaned)
    if duplicate_count > 0:
        logger.info(f"Removed {duplicate_count} duplicate records")
    
    # Handle missing values in critical columns
    for col in ['Latitude', 'Longitude']:
        if col in df_cleaned.columns and df_cleaned[col].isna().any():
            logger.warning(f"Removing {df_cleaned[col].isna().sum()} records with missing {col}")
            df_cleaned = df_cleaned.dropna(subset=[col])
    
    logger.info(f"Data cleaning complete. {len(df_cleaned)} records remaining")
    return df_cleaned

def convert_to_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert DataFrame to GeoDataFrame using latitude and longitude.
    
    Args:
        df: DataFrame with latitude and longitude columns
        
    Returns:
        GeoDataFrame: GeoDataFrame with geometry
    """
    if df is None or df.empty:
        logger.error("No data to convert to GeoDataFrame")
        return gpd.GeoDataFrame()
    
    # Check if already a GeoDataFrame
    if isinstance(df, gpd.GeoDataFrame) and df.geometry.name == 'geometry':
        logger.info("Input is already a GeoDataFrame")
        return df
    
    logger.info("Converting to GeoDataFrame...")
    
    # Check for required columns
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        logger.error("Missing Latitude or Longitude columns")
        return gpd.GeoDataFrame()
    
    try:
        # Convert to numeric to ensure valid coordinates
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        
        # Drop rows with invalid coordinates
        initial_count = len(df)
        df = df.dropna(subset=['Latitude', 'Longitude'])
        invalid_count = initial_count - len(df)
        
        if invalid_count > 0:
            logger.warning(f"Removed {invalid_count} records with invalid coordinates")
        
        # Create geometry column
        geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        
        logger.info(f"Successfully converted {len(gdf)} records to GeoDataFrame")
        return gdf
    
    except Exception as e:
        logger.error(f"Error converting to GeoDataFrame: {e}")
        return gpd.GeoDataFrame()

def standardize_operator_names(df: pd.DataFrame, column: str = 'NomeEntidade') -> pd.DataFrame:
    """
    Standardize operator names for consistency.
    
    Args:
        df: DataFrame with operator column
        column: Name of the column containing operator information
        
    Returns:
        DataFrame: DataFrame with standardized operator names
    """
    if df is None or df.empty or column not in df.columns:
        logger.warning(f"Cannot standardize operators: missing column '{column}'")
        return df
    
    logger.info("Standardizing operator names...")
    
    # Make a copy to avoid modifying the original
    df_std = df.copy()
    
    # Map operator names
    def map_operator(name):
        if pd.isna(name):
            return "N/A"
        
        name_upper = str(name).upper()
        for pattern, standardized in OPERATOR_MAPPING.items():
            if pattern in name_upper:
                return standardized
        return "OTHER"
    
    # Apply mapping
    df_std['Operator'] = df_std[column].apply(map_operator)
    
    # Log operator distribution
    operator_counts = df_std['Operator'].value_counts()
    logger.info(f"Operator distribution: {operator_counts.to_dict()}")
    
    return df_std

def filter_by_region(df: pd.DataFrame, bbox: List[float] = DEFAULT_REGION_BBOX) -> pd.DataFrame:
    """
    Filter data to only include records within the specified bounding box.
    
    Args:
        df: DataFrame with latitude and longitude columns
        bbox: Bounding box [lat_min, lat_max, lon_min, lon_max]
        
    Returns:
        DataFrame: Filtered DataFrame
    """
    if df is None or df.empty:
        logger.warning("No data to filter by region")
        return df
    
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        logger.error("Missing Latitude or Longitude columns for region filtering")
        return df
    
    logger.info(f"Filtering data by region: {bbox}")
    
    # Make a copy to avoid modifying the original
    df_filtered = df.copy()
    
    # Extract bbox parameters
    lat_min, lat_max, lon_min, lon_max = bbox
    
    # Apply filter
    initial_count = len(df_filtered)
    df_filtered = df_filtered[
        (df_filtered['Latitude'] >= lat_min) &
        (df_filtered['Latitude'] <= lat_max) &
        (df_filtered['Longitude'] >= lon_min) &
        (df_filtered['Longitude'] <= lon_max)
    ]
    
    filtered_count = initial_count - len(df_filtered)
    logger.info(f"Removed {filtered_count} records outside the region")
    logger.info(f"{len(df_filtered)} records remaining after region filtering")
    
    return df_filtered

def handle_missing_numeric_values(df: pd.DataFrame, column: str, default_value: float) -> pd.DataFrame:
    """
    Handle missing values in numeric columns.
    
    Args:
        df: DataFrame to process
        column: Column to handle
        default_value: Default value to use if median cannot be computed
        
    Returns:
        DataFrame: DataFrame with handled missing values
    """
    if df is None or df.empty:
        return df
    
    # If column doesn't exist, add it with default value
    if column not in df.columns:
        logger.info(f"Adding missing column '{column}' with default value {default_value}")
        df[column] = default_value
        return df
    
    # Convert to numeric
    df[column] = pd.to_numeric(df[column], errors='coerce')
    
    # Count missing values
    missing_count = df[column].isna().sum()
    
    if missing_count > 0:
        # Try to use median for filling
        median_value = df[column].median()
        
        # If median is not available or invalid, use default
        if pd.isna(median_value) or median_value <= 0:
            logger.info(f"Using default value {default_value} for '{column}'")
            fill_value = default_value
        else:
            logger.info(f"Using median value {median_value:.2f} for '{column}'")
            fill_value = median_value
        
        # Fill missing values
        df.loc[pd.isna(df[column]) | (df[column] <= 0), column] = fill_value
        logger.info(f"Filled {missing_count} missing values in '{column}'")
    
    return df

def process_data(df: pd.DataFrame, output_path: str = DEFAULT_OUTPUT_PATH) -> gpd.GeoDataFrame:
    """
    Process the data by applying all necessary transformations.
    
    Args:
        df: DataFrame to process
        output_path: Path to save the processed data
        
    Returns:
        GeoDataFrame: Processed GeoDataFrame
    """
    if df is None or df.empty:
        logger.error("No data to process")
        return gpd.GeoDataFrame()
    
    logger.info("Processing data...")
    
    # Clean data
    df_clean = clean_data(df)
    
    # Standardize operator names
    if 'NomeEntidade' in df_clean.columns:
        df_clean = standardize_operator_names(df_clean)
    
    # Filter by region if coordinates are available
    if 'Latitude' in df_clean.columns and 'Longitude' in df_clean.columns:
        df_clean = filter_by_region(df_clean)
    
    # Handle missing numeric values
    df_clean = handle_missing_numeric_values(df_clean, 'PotenciaTransmissorWatts', DEFAULT_POWER_WATTS)
    df_clean = handle_missing_numeric_values(df_clean, 'GanhoAntena', DEFAULT_ANTENNA_GAIN)
    df_clean = handle_missing_numeric_values(df_clean, 'FreqTxMHz', DEFAULT_FREQUENCY_MHZ)
    
    # Handle Azimuth values
    if 'Azimute' not in df_clean.columns:
        logger.info(f"Adding missing column 'Azimute' with default values")
        df_clean['Azimute'] = 0
    
    # Fill missing azimuth values with 0, 120, 240 (standard for 3 sectors)
    if 'Azimute' in df_clean.columns and df_clean['Azimute'].isna().sum() > 0:
        missing_count = df_clean['Azimute'].isna().sum()
        logger.info(f"Filling {missing_count} missing Azimute values with standard sector values")
        
        # Convert to numeric first
        df_clean['Azimute'] = pd.to_numeric(df_clean['Azimute'], errors='coerce')
        
        # Generate missing values
        for i, (idx, row) in enumerate(df_clean[df_clean['Azimute'].isna()].iterrows()):
            df_clean.loc[idx, 'Azimute'] = DEFAULT_AZIMUTHS[i % len(DEFAULT_AZIMUTHS)]
    
    # Convert to GeoDataFrame
    gdf = convert_to_geodataframe(df_clean)
    
    # Save processed data if output path is provided
    if output_path and len(gdf) > 0:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save as GeoJSON for better compatibility
            output_geojson = os.path.splitext(output_path)[0] + '.geojson'
            gdf.to_file(output_geojson, driver='GeoJSON')
            logger.info(f"Saved processed data to {output_geojson}")
            
            # If original path was CSV, also save as CSV
            if os.path.splitext(output_path)[1].lower() == '.csv':
                # Remove geometry column for CSV
                gdf_csv = gdf.copy()
                gdf_csv = pd.DataFrame(gdf_csv.drop(columns=['geometry']))
                gdf_csv.to_csv(output_path, index=False)
                logger.info(f"Saved processed data to {output_path}")
        
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
    
    return gdf

def load_and_process_data(input_path: str = DEFAULT_INPUT_PATH, 
                          output_path: str = DEFAULT_OUTPUT_PATH) -> gpd.GeoDataFrame:
    """
    Load and process data in one step.
    
    Args:
        input_path: Path to the input file
        output_path: Path to save the processed data
        
    Returns:
        GeoDataFrame: Processed GeoDataFrame
    """
    logger.info(f"Loading and processing data from {input_path}")
    
    # Load data
    df = load_data(input_path)
    
    # Return empty GeoDataFrame if loading failed
    if df is None or df.empty:
        logger.error("Failed to load data")
        return gpd.GeoDataFrame()
    
    # Process data
    gdf = process_data(df, output_path)
    
    logger.info(f"Data loading and processing complete. {len(gdf)} records processed")
    return gdf

# For direct execution
if __name__ == "__main__":
    gdf = load_and_process_data()
    print(f"Processed {len(gdf)} records")
