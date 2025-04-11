import os
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point

# File paths
csv_path = 'data/csv_licenciamento_bruto.csv.csv'
dem_path = 'data/dem.tif'

# Check if files exist
if not os.path.exists(csv_path):
    print(f"Error: RBS data file {csv_path} does not exist.")
    exit(1)
    
if not os.path.exists(dem_path):
    print(f"Error: DEM file {dem_path} does not exist.")
    exit(1)

print("Loading RBS data...")
# Load and prepare RBS data
df = pd.read_csv(csv_path, low_memory=False)
print(f"Loaded {len(df)} RBS records")

# Filter records with valid coordinates
df_valid = df.dropna(subset=['Latitude', 'Longitude'])
print(f"Records with valid coordinates: {len(df_valid)}")

# Create GeoDataFrame
geometry = [Point(lon, lat) for lon, lat in zip(df_valid['Longitude'], df_valid['Latitude'])]
gdf = gpd.GeoDataFrame(df_valid, geometry=geometry, crs="EPSG:4326")

# Transform to the DEM's CRS
print("Loading DEM data...")
with rasterio.open(dem_path) as src:
    dem_crs = src.crs
    dem_data = src.read(1)
    
    # Transform GeoDataFrame to match DEM CRS
    print(f"Transforming coordinates from EPSG:4326 to {dem_crs}")
    gdf = gdf.to_crs(dem_crs)
    
    # Sample DEM at RBS locations
    print("Sampling elevation at RBS locations...")
    elevations = []
    xy_coords = [(point.x, point.y) for point in gdf.geometry]
    
    for point in xy_coords:
        row, col = src.index(point[0], point[1])
        try:
            elevation = dem_data[row, col]
            elevations.append(elevation)
        except IndexError:
            elevations.append(np.nan)
    
    gdf['elevation'] = elevations
    
    # Filter out points with NaN elevations (outside DEM bounds)
    gdf = gdf.dropna(subset=['elevation'])
    print(f"RBS stations within DEM area: {len(gdf)}")
    
    # Basic statistics by operator
    if 'NomeEntidade' in gdf.columns:
        print("\nElevation statistics by operator:")
        operator_stats = gdf.groupby('NomeEntidade')['elevation'].agg(['count', 'min', 'max', 'mean', 'std'])
        print(operator_stats.sort_values('count', ascending=False).head(10))
    
    # Plot histogram of elevations
    plt.figure(figsize=(10, 6))
    plt.hist(gdf['elevation'], bins=30, alpha=0.7, color='blue')
    plt.title('Distribution of RBS Stations by Elevation')
    plt.xlabel('Elevation (meters)')
    plt.ylabel('Number of RBS Stations')
    plt.grid(True, alpha=0.3)
    plt.savefig('rbs_elevation_distribution.png')
    print("Saved elevation distribution plot to rbs_elevation_distribution.png")
    
    # Plot RBS locations on DEM
    fig, ax = plt.subplots(figsize=(12, 8))
    show(dem_data, ax=ax, cmap='terrain', title='RBS Stations on Elevation Map')
    
    # Plot RBS stations
    gdf.plot(ax=ax, marker='o', color='red', markersize=5, alpha=0.5)
    plt.savefig('rbs_on_dem.png')
    print("Saved RBS stations on DEM map to rbs_on_dem.png")

# Save processed data
gdf.to_file('rbs_with_elevation.gpkg', driver='GPKG')
print("Saved GeoPackage with RBS and elevation data to rbs_with_elevation.gpkg") 