import os
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np

# Path to the DEM file
dem_path = 'data/dem.tif'

# Check if the file exists
if not os.path.exists(dem_path):
    print(f"Error: File {dem_path} does not exist.")
    exit(1)

# Read the DEM file
try:
    with rasterio.open(dem_path) as src:
        # Get basic information
        print(f"DEM file: {dem_path}")
        print(f"Width: {src.width}, Height: {src.height}")
        print(f"Number of bands: {src.count}")
        print(f"Coordinate Reference System (CRS): {src.crs}")
        print(f"Bounds: {src.bounds}")
        print(f"Resolution: {src.res}")
        
        # Read the DEM data
        dem_data = src.read(1)  # Read the first band
        
        # Calculate statistics
        min_elevation = dem_data.min()
        max_elevation = dem_data.max()
        mean_elevation = dem_data.mean()
        std_elevation = dem_data.std()
        
        print(f"\nElevation Statistics:")
        print(f"Min elevation: {min_elevation} meters")
        print(f"Max elevation: {max_elevation} meters")
        print(f"Mean elevation: {mean_elevation:.2f} meters")
        print(f"Standard deviation: {std_elevation:.2f} meters")
except Exception as e:
    print(f"Error reading DEM file: {e}") 