"""
Module for radio signal coverage calculations of RBS (Radio Base Stations).
Contains functions to calculate Effective Isotropic Radiated Power (EIRP),
coverage radius, coverage area, and coverage sector geometries.
"""

import numpy as np
from shapely.geometry import Polygon, Point
import geopandas as gpd
import pandas as pd

# Default constants
RECEIVER_SENSITIVITY = -100  # dBm
SECTOR_ANGLE = 120  # degrees

def calculate_eirp(power_watts, antenna_gain):
    """
    Calculates the Effective Isotropic Radiated Power (EIRP) in dBm.
    
    Args:
        power_watts (float): Transmitter power in Watts
        antenna_gain (float): Antenna gain in dBi
        
    Returns:
        float: EIRP in dBm or np.nan if power is invalid
    """
    try:
        power = float(power_watts)
        if power <= 0:
            return np.nan
    except (ValueError, TypeError):
        return np.nan
    
    # Conversion from Watts to dBm and adding gain
    return 10 * np.log10(power * 1000) + antenna_gain

def calculate_improved_coverage_radius(eirp, freq_mhz, area_type='urban'):
    """
    Calculates the coverage radius based on EIRP, frequency and area type.
    
    Args:
        eirp (float): Effective isotropic radiated power in dBm
        freq_mhz (float): Frequency in MHz
        area_type (str): Area type ('dense_urban', 'urban', 'suburban', 'rural')
        
    Returns:
        float: Coverage radius in kilometers or np.nan if values are invalid
    """
    if np.isnan(eirp) or np.isnan(freq_mhz) or freq_mhz <= 0:
        return np.nan
    
    # Attenuation factors by area type
    attenuation = {'dense_urban': 22, 'urban': 16, 'suburban': 12, 'rural': 8}
    
    # Base radius calculation (simplified Friis formula)
    base_radius = 10 ** ((eirp - RECEIVER_SENSITIVITY - 32.44 - 20 * np.log10(freq_mhz)) / 20)
    
    # Adjustment by area type
    adjusted_radius = base_radius * 0.7 / (attenuation.get(area_type, 16) / 10)
    
    # Limit according to frequency (higher frequencies have shorter range)
    freq_limit = min(7, 15000 / freq_mhz) if freq_mhz > 0 else 5
    
    return min(adjusted_radius, freq_limit)

def calculate_coverage_area(radius, angle=SECTOR_ANGLE):
    """
    Calculates the coverage area based on radius and sector angle.
    
    Args:
        radius (float): Coverage radius in kilometers
        angle (float): Sector angle in degrees (default = 120)
        
    Returns:
        float: Coverage area in square kilometers
    """
    if np.isnan(radius):
        return np.nan
    
    # Sector area as a fraction of a full circle
    return (np.pi * radius**2 * angle) / 360

def create_precise_sector(lat, lon, radius, azimuth, angle=SECTOR_ANGLE, resolution=30):
    """
    Creates a polygon representing the coverage sector of the antenna.
    
    Args:
        lat (float): Latitude of the central point (RBS)
        lon (float): Longitude of the central point (RBS)
        radius (float): Coverage radius in kilometers
        azimuth (float): Antenna direction in degrees (0-360)
        angle (float): Sector angle in degrees (default = 120)
        resolution (int): Number of points to use for the sector arc
        
    Returns:
        Polygon or None: Sector geometry or None if radius/azimuth is invalid
    """
    if np.isnan(radius) or np.isnan(azimuth) or radius <= 0:
        return None
    
    # Adjust azimuth for the geographic coordinate system
    azimuth_rad = np.radians((450 - float(azimuth)) % 360)
    half_angle = np.radians(angle / 2)
    
    # Create points for the sector polygon
    points = [(lon, lat)]  # Central point
    
    # Create arc with multiple points
    for i in range(resolution + 1):
        current_angle = azimuth_rad - half_angle + (i * 2 * half_angle / resolution)
        
        # Add points at different distances for better definition
        for j in [0.8, 0.9, 0.95, 1.0]:
            dist = radius * j
            # Convert distance to decimal degrees
            dx = dist * np.cos(current_angle) / 111.32  # 111.32 km = 1 degree of longitude at the equator
            dy = dist * np.sin(current_angle) / (111.32 * np.cos(np.radians(lat)))  # Adjustment for latitude
            points.append((lon + dx, lat + dy))
    
    # Close the polygon
    points.append((lon, lat))
    
    try:
        return Polygon(points)
    except:
        return None

def estimate_area_type(gdf_rbs, radius=0.01):
    """
    Estimates the area type (urban, rural, etc.) based on RBS density.
    
    Args:
        gdf_rbs (GeoDataFrame): GeoDataFrame with the RBS
        radius (float): Search radius (in decimal degrees)
        
    Returns:
        GeoDataFrame: The same GeoDataFrame with 'area_type' column added
    """
    # Function to count nearby RBS
    def calculate_rbs_density(point, gdf, radius):
        buffer = point.buffer(radius)
        return len(gdf[gdf.geometry.intersects(buffer)])
    
    # Calculate density for each RBS
    gdf_rbs = gdf_rbs.copy()
    gdf_rbs['rbs_density'] = gdf_rbs.geometry.apply(lambda p: calculate_rbs_density(p, gdf_rbs, radius))
    
    # Classify area type based on density
    gdf_rbs['area_type'] = pd.cut(
        gdf_rbs['rbs_density'], 
        bins=[0, 3, 6, 10, float('inf')], 
        labels=['rural', 'suburban', 'urban', 'dense_urban']
    )
    gdf_rbs['area_type'] = gdf_rbs['area_type'].fillna('urban')
    
    return gdf_rbs
