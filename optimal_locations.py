import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from shapely.geometry import Point, Polygon
import seaborn as sns
import matplotlib.colors as colors
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.ndimage import gaussian_filter

# Load the RBS data with elevation and propagation information
gdf_path = 'rbs_with_elevation.gpkg'
dem_path = 'data/dem.tif'

if not os.path.exists(gdf_path):
    print(f"Error: {gdf_path} does not exist. Run integrated_analysis.py first.")
    exit(1)

if not os.path.exists(dem_path):
    print(f"Error: DEM file {dem_path} does not exist.")
    exit(1)

# Load the GeoDataFrame
print(f"Loading RBS data with elevation from {gdf_path}...")
gdf = gpd.read_file(gdf_path)
print(f"Loaded {len(gdf)} RBS stations with elevation data")

# Convert columns to numeric
for col in ['AlturaAntena', 'PotenciaTransmissorWatts', 'GanhoAntena', 'FreqTxMHz', 'propagation_est_km']:
    if col in gdf.columns:
        gdf[col] = pd.to_numeric(gdf[col], errors='coerce')
    else:
        # If propagation_est_km doesn't exist, create a placeholder
        if col == 'propagation_est_km':
            # Assume a default propagation distance of 5km
            gdf['propagation_est_km'] = 5.0

# Load DEM for analysis
print("Loading DEM data...")
with rasterio.open(dem_path) as src:
    dem_data = src.read(1)
    dem_meta = src.meta
    dem_bounds = src.bounds
    dem_transform = src.transform
    dem_crs = src.crs
    dem_width = src.width
    dem_height = src.height
    
    # Create a grid of points for analysis
    print("Creating analysis grid...")
    # Use a lower resolution grid to make computation feasible
    grid_factor = 50  # Sample 1 out of every grid_factor pixels
    y_indices, x_indices = np.mgrid[0:dem_height:grid_factor, 0:dem_width:grid_factor]
    
    # Get the real-world coordinates for each grid point
    xs, ys = rasterio.transform.xy(dem_transform, y_indices, x_indices)
    
    # Convert to arrays
    xs = np.array(xs)
    ys = np.array(ys)
    
    # Flatten for easier processing
    points_x = xs.flatten()
    points_y = ys.flatten()
    
    # Sample elevation at each point
    point_elevations = []
    for i in range(len(points_y)):
        row, col = src.index(points_x[i], points_y[i])
        try:
            if 0 <= row < dem_height and 0 <= col < dem_width:
                elevation = dem_data[row, col]
                point_elevations.append(elevation)
            else:
                point_elevations.append(np.nan)
        except IndexError:
            point_elevations.append(np.nan)
    
    # Create geodataframe for the analysis grid
    grid_points = [Point(x, y) for x, y in zip(points_x, points_y)]
    grid_gdf = gpd.GeoDataFrame(
        {'elevation': point_elevations},
        geometry=grid_points,
        crs=dem_crs
    )
    
    # Remove points with NaN elevations
    grid_gdf = grid_gdf.dropna(subset=['elevation'])
    print(f"Analysis grid created with {len(grid_gdf)} points")

# --- Analysis Functions ---
def calculate_coverage_score(grid_point, rbs_stations):
    """
    Calculate coverage score for a grid point:
    - Higher score = better coverage from existing stations
    - Lower score = area might need a new station
    """
    # Calculate distance to each RBS station
    distances = [grid_point.distance(station.geometry) for _, station in rbs_stations.iterrows()]
    
    # Convert to kilometers (assuming the CRS is in meters)
    distances_km = [d / 1000 for d in distances]
    
    # Get propagation ranges
    propagation_ranges = rbs_stations['propagation_est_km'].values
    
    # Calculate coverage ratio (distance/range)
    # Values < 1 mean the point is within coverage range
    # Values > 1 mean the point is outside coverage range
    coverage_ratios = [d/r if r > 0 else float('inf') for d, r in zip(distances_km, propagation_ranges)]
    
    # Find the minimum coverage ratio (best coverage)
    min_ratio = min(coverage_ratios) if coverage_ratios else float('inf')
    
    # Convert to a score:
    # 1.0 = exactly at the edge of coverage
    # > 1.0 = outside coverage (higher = worse coverage)
    # < 1.0 = inside coverage (lower = better coverage)
    return min_ratio

def calculate_elevation_score(elevation, mean_elevation, std_elevation):
    """
    Calculate elevation score:
    - Higher score = better elevation for RBS placement
    - Score is normalized relative to existing stations
    """
    # Calculate z-score
    z_score = (elevation - mean_elevation) / std_elevation if std_elevation > 0 else 0
    
    # Apply sigmoid-like function that favors higher elevations
    # but not extremely high (diminishing returns)
    if z_score > 0:
        # For positive z-scores (above mean): favor higher elevations
        return 1 + (1 / (1 + np.exp(-z_score + 3)))
    else:
        # For negative z-scores (below mean): penalize lower elevations
        return 1 / (1 + np.exp(z_score + 1))

# --- Calculate Scores ---
print("Calculating coverage and elevation scores...")

# Get elevation stats from existing stations
mean_elevation = gdf['elevation'].mean()
std_elevation = gdf['elevation'].std()

# Calculate scores for each grid point
coverage_scores = []
elevation_scores = []
combined_scores = []

# Process in chunks to avoid memory issues
chunk_size = 1000
for i in range(0, len(grid_gdf), chunk_size):
    chunk = grid_gdf.iloc[i:i+chunk_size]
    
    chunk_coverage_scores = []
    chunk_elevation_scores = []
    
    for idx, point in chunk.iterrows():
        # Calculate coverage score
        cov_score = calculate_coverage_score(point.geometry, gdf)
        chunk_coverage_scores.append(cov_score)
        
        # Calculate elevation score
        elev_score = calculate_elevation_score(point['elevation'], mean_elevation, std_elevation)
        chunk_elevation_scores.append(elev_score)
    
    coverage_scores.extend(chunk_coverage_scores)
    elevation_scores.extend(chunk_elevation_scores)
    
    # Print progress
    if i % (chunk_size * 10) == 0:
        print(f"  Processed {i}/{len(grid_gdf)} points...")

# Add scores to the grid GeoDataFrame
grid_gdf['coverage_score'] = coverage_scores
grid_gdf['elevation_score'] = elevation_scores

# Calculate combined score (higher = better location for new RBS)
# For coverage: we want areas with high scores (less coverage)
# For elevation: we want areas with high scores (better elevation)
# Combined formula: coverage_weight * coverage_score + elevation_weight * elevation_score
coverage_weight = 0.7  # Areas with poor coverage are prioritized
elevation_weight = 0.3  # Elevation is important but secondary

# Normalize coverage scores: higher score = worse coverage = better for new RBS
max_coverage = grid_gdf['coverage_score'].max()
min_coverage = grid_gdf['coverage_score'].min()
grid_gdf['norm_coverage_score'] = (grid_gdf['coverage_score'] - min_coverage) / (max_coverage - min_coverage) if max_coverage > min_coverage else 0.5

# Combined score calculation
grid_gdf['combined_score'] = coverage_weight * grid_gdf['norm_coverage_score'] + elevation_weight * grid_gdf['elevation_score']

# --- Identify Optimal Locations ---
print("Identifying optimal locations for new RBS stations...")

# Find top 10 points with highest combined score
top_locations = grid_gdf.sort_values('combined_score', ascending=False).head(10)
print("\nTop 10 recommended locations for new RBS stations:")
print(top_locations[['elevation', 'coverage_score', 'elevation_score', 'combined_score']])

# --- Visualizations ---
print("Creating visualizations...")

# Plot 1: Coverage map
plt.figure(figsize=(12, 10))
ax = plt.subplot(111)

# Plot DEM as background
with rasterio.open(dem_path) as src:
    show(src.read(1), transform=src.transform, ax=ax, cmap='terrain', alpha=0.7)

# Plot coverage score (interpolated for better visualization)
# Create a heatmap using a scatter plot with color mapping
scatter = ax.scatter(
    grid_gdf.geometry.x, 
    grid_gdf.geometry.y, 
    c=grid_gdf['norm_coverage_score'],
    cmap='YlOrRd', 
    alpha=0.5,
    s=10
)
plt.colorbar(scatter, label='Coverage Need Score (Higher = Less Coverage)')

# Plot existing RBS stations
gdf.plot(ax=ax, color='blue', markersize=10, alpha=0.7, label='Existing RBS')

# Plot recommended locations
top_locations.plot(ax=ax, color='green', markersize=50, alpha=0.7, label='Recommended Locations')

plt.title('RBS Coverage Analysis and Recommended New Locations')
plt.legend()
plt.tight_layout()
plt.savefig('recommended_rbs_locations.png', dpi=300)
print("Saved recommended locations map to recommended_rbs_locations.png")

# Plot 2: Combined score heatmap
plt.figure(figsize=(12, 10))
ax = plt.subplot(111)

# Plot DEM as background
with rasterio.open(dem_path) as src:
    show(src.read(1), transform=src.transform, ax=ax, cmap='terrain', alpha=0.6)

# Plot combined score heatmap
scatter = ax.scatter(
    grid_gdf.geometry.x, 
    grid_gdf.geometry.y, 
    c=grid_gdf['combined_score'],
    cmap='viridis', 
    alpha=0.5,
    s=10
)
plt.colorbar(scatter, label='Optimal Location Score')

# Plot existing RBS stations
gdf.plot(ax=ax, color='red', markersize=10, alpha=0.7, label='Existing RBS')

# Plot recommended locations
top_locations.plot(ax=ax, color='yellow', markersize=50, alpha=0.7, label='Recommended Locations')

plt.title('Optimal Location Score (Combined Coverage and Elevation)')
plt.legend()
plt.tight_layout()
plt.savefig('optimal_location_heatmap.png', dpi=300)
print("Saved optimal location heatmap to optimal_location_heatmap.png")

# Save the top locations to a GeoPackage for further analysis
top_locations.to_file('recommended_rbs_locations.gpkg', driver='GPKG')
print("Saved recommended locations to recommended_rbs_locations.gpkg")

print("\nAnalysis completed successfully!") 