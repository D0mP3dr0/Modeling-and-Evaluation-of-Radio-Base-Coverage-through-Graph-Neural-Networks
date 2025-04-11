"""
Module for predictive coverage modeling, including RBS position optimization
and expansion area prediction.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
import matplotlib.patheffects as PathEffects
from shapely.geometry import Point, Polygon, box, LineString
from shapely.ops import unary_union
import folium
from folium.plugins import HeatMap, MarkerCluster
import os
import rasterio
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.optimize import minimize
from scipy.interpolate import griddata
import random
import networkx as nx

# Constants for optimization
MAX_ITERATIONS = 100
COVERAGE_THRESHOLD = 0.85  # Target: 85% coverage
MIN_DISTANCE_BETWEEN_RBS = 1000  # Minimum distance between RBS in meters
POPULATION_WEIGHT = 0.7  # Weight for population density in optimization

def simulate_coverage_optimization(gdf_rbs, gdf_sectors, grid_size=0.01, 
                                  optimization_mode='adjust', output_path=None):
    """
    Simulates how coverage would change with optimized RBS positions.
    
    Args:
        gdf_rbs: GeoDataFrame with RBS locations
        gdf_sectors: GeoDataFrame with coverage sectors
        grid_size: Size of the grid cells for the analysis (in CRS units)
        optimization_mode: 'adjust' to adjust existing RBS or 'add' to add new ones
        output_path: Path to save the visualization
        
    Returns:
        tuple: (GeoDataFrame with optimized RBS, GeoDataFrame with optimized coverage)
    """
    print(f"Simulating coverage optimization ({optimization_mode} mode)...")
    
    # Ensure correct CRS (Web Mercator for distance calculations)
    gdf_rbs = gdf_rbs.to_crs(epsg=3857)
    gdf_sectors = gdf_sectors.to_crs(epsg=3857)
    
    # Get study area bounds
    bounds = gdf_sectors.total_bounds
    
    # Create a coverage grid for analysis
    coverage_grid = create_coverage_grid(gdf_sectors, grid_size)
    
    # Calculate initial coverage metrics
    initial_coverage_ratio = calculate_coverage_ratio(coverage_grid)
    print(f"Initial coverage ratio: {initial_coverage_ratio:.2f}")
    
    # Optimize based on selected mode
    if optimization_mode == 'adjust':
        # Adjust existing RBS positions to optimize coverage
        optimized_rbs = optimize_existing_rbs_positions(gdf_rbs, gdf_sectors, coverage_grid)
    else:  # 'add' mode
        # Find optimal locations for additional RBS
        optimized_rbs = add_optimal_new_rbs(gdf_rbs, gdf_sectors, coverage_grid)
    
    # Regenerate coverage with optimized RBS
    optimized_sectors = create_optimized_sectors(optimized_rbs, gdf_sectors)
    
    # Calculate new coverage grid
    optimized_coverage_grid = create_coverage_grid(optimized_sectors, grid_size)
    
    # Calculate final coverage metrics
    final_coverage_ratio = calculate_coverage_ratio(optimized_coverage_grid)
    print(f"Optimized coverage ratio: {final_coverage_ratio:.2f}")
    print(f"Coverage improvement: {(final_coverage_ratio - initial_coverage_ratio) * 100:.2f}%")
    
    # Create visualization if output path is provided
    if output_path:
        create_optimization_comparison_map(
            gdf_rbs, gdf_sectors, optimized_rbs, optimized_sectors, 
            coverage_grid, optimized_coverage_grid, output_path
        )
    
    return optimized_rbs, optimized_sectors

def create_coverage_grid(gdf_sectors, grid_size):
    """
    Creates a grid to analyze coverage over the study area.
    
    Args:
        gdf_sectors: GeoDataFrame with coverage sectors
        grid_size: Size of the grid cells
        
    Returns:
        GeoDataFrame: Grid cells with coverage information
    """
    bounds = gdf_sectors.total_bounds
    
    # Create grid cells
    x_points = np.arange(bounds[0], bounds[2], grid_size)
    y_points = np.arange(bounds[1], bounds[3], grid_size)
    
    grid_cells = []
    
    for x in x_points:
        for y in y_points:
            # Create grid cell
            cell = box(x, y, x + grid_size, y + grid_size)
            
            # Count overlapping sectors
            overlapping_sectors = sum([1 for idx, sector in gdf_sectors.iterrows() 
                                     if sector.geometry is not None and sector.geometry.intersects(cell)])
            
            # Calculate coverage ratio (simplification for efficiency)
            covered = overlapping_sectors > 0
            
            # Add to grid cells
            grid_cells.append({
                'geometry': cell,
                'overlapping_sectors': overlapping_sectors,
                'is_covered': covered
            })
    
    # Convert to GeoDataFrame
    return gpd.GeoDataFrame(grid_cells, crs=gdf_sectors.crs)

def calculate_coverage_ratio(coverage_grid):
    """
    Calculates the percentage of area that is covered.
    
    Args:
        coverage_grid: GeoDataFrame with grid cells
        
    Returns:
        float: Coverage ratio (0-1)
    """
    if len(coverage_grid) == 0:
        return 0
    
    return coverage_grid['is_covered'].sum() / len(coverage_grid)

def optimize_existing_rbs_positions(gdf_rbs, gdf_sectors, coverage_grid):
    """
    Optimizes the positions of existing RBS to improve coverage.
    
    Args:
        gdf_rbs: GeoDataFrame with RBS locations
        gdf_sectors: GeoDataFrame with coverage sectors
        coverage_grid: GeoDataFrame with coverage grid
        
    Returns:
        GeoDataFrame: Optimized RBS locations
    """
    print("Optimizing existing RBS positions...")
    
    # Create a deep copy to avoid modifying the original
    optimized_rbs = gdf_rbs.copy()
    
    # Identify critical areas (uncovered areas)
    critical_areas = coverage_grid[~coverage_grid['is_covered']]
    
    if len(critical_areas) == 0:
        print("No critical areas found. Coverage is already optimal.")
        return optimized_rbs
    
    # For each RBS, try to optimize its position
    for idx, rbs in optimized_rbs.iterrows():
        # Get the initial position
        initial_pos = np.array([rbs.geometry.x, rbs.geometry.y])
        
        # Get coverage radius
        radius = rbs['Coverage_Radius_km'] * 1000 if 'Coverage_Radius_km' in rbs else 5000
        
        # Setup optimization constraints
        # We don't want to move RBS too far from original position
        max_shift = radius * 0.5  # Allow shifting up to 50% of the coverage radius
        
        # Find the nearest critical areas
        nearest_critical = find_nearest_critical_areas(rbs.geometry, critical_areas, max_points=10)
        
        if len(nearest_critical) == 0:
            continue  # Skip if no nearby critical areas
            
        # Calculate centroid of critical areas
        critical_points = np.array([[point.x, point.y] for point in nearest_critical.geometry.centroid])
        target = np.mean(critical_points, axis=0)
        
        # Move the RBS towards the target, but not too far
        vector = target - initial_pos
        distance = np.linalg.norm(vector)
        
        if distance > 0:
            # Limit the movement to max_shift
            shift = min(distance, max_shift)
            unit_vector = vector / distance
            new_pos = initial_pos + unit_vector * shift
            
            # Update the RBS position
            optimized_rbs.loc[idx, 'geometry'] = Point(new_pos[0], new_pos[1])
    
    return optimized_rbs

def find_nearest_critical_areas(point, critical_areas, max_points=10):
    """
    Finds the nearest critical areas to a given point.
    
    Args:
        point: Shapely Point object
        critical_areas: GeoDataFrame with critical areas
        max_points: Maximum number of points to return
        
    Returns:
        GeoDataFrame: Nearest critical areas
    """
    if len(critical_areas) == 0:
        return gpd.GeoDataFrame(geometry=[])
    
    # Calculate distances
    critical_areas['distance'] = critical_areas.geometry.centroid.apply(lambda p: point.distance(p))
    
    # Sort by distance and get the nearest ones
    return critical_areas.sort_values('distance').head(max_points)

def add_optimal_new_rbs(gdf_rbs, gdf_sectors, coverage_grid):
    """
    Adds new RBS at optimal locations to improve coverage.
    
    Args:
        gdf_rbs: GeoDataFrame with RBS locations
        gdf_sectors: GeoDataFrame with coverage sectors
        coverage_grid: GeoDataFrame with coverage grid
        
    Returns:
        GeoDataFrame: GeoDataFrame with original and new RBS locations
    """
    print("Finding optimal locations for new RBS...")
    
    # Create a deep copy to avoid modifying the original
    optimized_rbs = gdf_rbs.copy()
    
    # Identify critical areas (uncovered areas)
    critical_areas = coverage_grid[~coverage_grid['is_covered']]
    
    if len(critical_areas) == 0:
        print("No critical areas found. Coverage is already optimal.")
        return optimized_rbs
    
    # Cluster critical areas to find centers for new RBS
    clusters = cluster_critical_areas(critical_areas)
    
    # For each cluster centroid, add a new RBS
    for i, centroid in enumerate(clusters):
        # Skip if too close to existing RBS
        too_close = False
        for idx, rbs in optimized_rbs.iterrows():
            if Point(centroid).distance(rbs.geometry) < MIN_DISTANCE_BETWEEN_RBS:
                too_close = True
                break
                
        if too_close:
            continue
            
        # Create a new RBS with default parameters
        new_rbs = optimized_rbs.iloc[0].copy() if len(optimized_rbs) > 0 else None
        
        if new_rbs is not None:
            # Update geometry and ID
            new_rbs['geometry'] = Point(centroid[0], centroid[1])
            new_idx = len(optimized_rbs)
            
            # Add to the optimized RBS DataFrame
            optimized_rbs.loc[new_idx] = new_rbs
    
    return optimized_rbs

def cluster_critical_areas(critical_areas, n_clusters=None):
    """
    Clusters critical areas to find centers for new RBS.
    
    Args:
        critical_areas: GeoDataFrame with critical areas
        n_clusters: Number of clusters (if None, determined automatically)
        
    Returns:
        list: List of cluster centroids (x, y coordinates)
    """
    if len(critical_areas) == 0:
        return []
    
    # Extract centroids of critical areas
    points = np.array([[p.x, p.y] for p in critical_areas.geometry.centroid])
    
    # Determine number of clusters if not specified
    if n_clusters is None:
        # Simple rule: 1 cluster per X critical areas
        n_clusters = max(1, len(critical_areas) // 100)
    
    try:
        # Use a simple K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(points)
        return kmeans.cluster_centers_
    except:
        # Fallback: just return centroids of some random points
        indices = np.random.choice(len(points), size=min(n_clusters, len(points)), replace=False)
        return points[indices]

def create_optimized_sectors(optimized_rbs, original_sectors):
    """
    Creates coverage sectors for optimized RBS positions.
    
    Args:
        optimized_rbs: GeoDataFrame with optimized RBS locations
        original_sectors: GeoDataFrame with original coverage sectors
        
    Returns:
        GeoDataFrame: Coverage sectors for optimized RBS
    """
    # Create a new GeoDataFrame for optimized sectors
    optimized_sectors = gpd.GeoDataFrame(columns=original_sectors.columns)
    
    # For each RBS, create a new sector
    for idx, rbs in optimized_rbs.iterrows():
        # Find the original sector for this RBS if it exists
        original_idx = None
        for i, sector in original_sectors.iterrows():
            # This is an approximate matching - in a real scenario, 
            # you would need a proper ID to match RBS to sectors
            if idx < len(original_sectors) and i == idx:
                original_idx = i
                break
        
        if original_idx is not None:
            # Copy the original sector
            new_sector = original_sectors.loc[original_idx].copy()
            
            # Update geometry based on new RBS position
            if 'Coverage_Radius_km' in rbs and 'Azimute' in rbs:
                # Create a new sector geometry
                from src.coverage_models import create_precise_sector
                
                new_sector['geometry'] = create_precise_sector(
                    rbs.geometry.y, rbs.geometry.x, 
                    rbs['Coverage_Radius_km'], rbs['Azimute']
                )
            else:
                # Simple circular coverage
                radius = 5000  # Default radius in meters
                new_sector['geometry'] = rbs.geometry.buffer(radius)
        else:
            # Create a new sector with default parameters
            new_sector = pd.Series({
                'geometry': rbs.geometry.buffer(5000),  # Default 5km radius
                'Operator': rbs['Operator'] if 'Operator' in rbs else 'NEW',
                'EIRP_dBm': rbs['EIRP_dBm'] if 'EIRP_dBm' in rbs else 50,
                'Coverage_Radius_km': rbs['Coverage_Radius_km'] if 'Coverage_Radius_km' in rbs else 5
            })
        
        # Add to optimized sectors
        optimized_sectors = pd.concat([optimized_sectors, pd.DataFrame([new_sector])])
    
    # Reset index and ensure it's a GeoDataFrame
    optimized_sectors = gpd.GeoDataFrame(optimized_sectors, crs=original_sectors.crs)
    
    return optimized_sectors

def create_optimization_comparison_map(original_rbs, original_sectors, 
                                     optimized_rbs, optimized_sectors,
                                     original_grid, optimized_grid, output_path):
    """
    Creates a comparison map showing original and optimized coverage.
    
    Args:
        original_rbs: GeoDataFrame with original RBS locations
        original_sectors: GeoDataFrame with original sectors
        optimized_rbs: GeoDataFrame with optimized RBS
        optimized_sectors: GeoDataFrame with optimized sectors
        original_grid: GeoDataFrame with original coverage grid
        optimized_grid: GeoDataFrame with optimized coverage grid
        output_path: Path to save the visualization
    """
    print("Creating optimization comparison map...")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # First subplot: Original coverage
    # Plot uncovered areas
    original_grid[~original_grid['is_covered']].plot(
        ax=ax1,
        color='red',
        alpha=0.5,
        edgecolor='none',
        label='Uncovered'
    )
    
    # Plot original sectors
    original_sectors.plot(
        ax=ax1,
        color='blue',
        alpha=0.3,
        edgecolor='none'
    )
    
    # Plot original RBS
    original_rbs.plot(
        ax=ax1,
        markersize=30,
        marker='o',
        color='blue',
        edgecolor='white',
        linewidth=0.5,
        alpha=0.8,
        label='Original RBS'
    )
    
    # Add basemap
    ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron)
    
    # Add title
    ax1.set_title('Original Coverage', fontsize=14)
    
    # Second subplot: Optimized coverage
    # Plot uncovered areas in optimized scenario
    optimized_grid[~optimized_grid['is_covered']].plot(
        ax=ax2,
        color='red',
        alpha=0.5,
        edgecolor='none',
        label='Uncovered'
    )
    
    # Plot optimized sectors
    optimized_sectors.plot(
        ax=ax2,
        color='green',
        alpha=0.3,
        edgecolor='none'
    )
    
    # Plot optimized RBS
    optimized_rbs.plot(
        ax=ax2,
        markersize=30,
        marker='o',
        color='green',
        edgecolor='white',
        linewidth=0.5,
        alpha=0.8,
        label='Optimized RBS'
    )
    
    # Plot movement arrows for existing RBS
    for i in range(min(len(original_rbs), len(optimized_rbs))):
        orig_point = original_rbs.iloc[i].geometry
        opt_point = optimized_rbs.iloc[i].geometry
        
        # Only draw arrows if the position changed
        if orig_point.distance(opt_point) > 0:
            # Create a line connecting original and optimized positions
            line = LineString([
                (orig_point.x, orig_point.y),
                (opt_point.x, opt_point.y)
            ])
            
            # Plot line
            x, y = line.xy
            ax2.plot(x, y, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    # Add basemap
    ctx.add_basemap(ax2, source=ctx.providers.CartoDB.Positron)
    
    # Add title
    ax2.set_title('Optimized Coverage', fontsize=14)
    
    # Add coverage statistics
    original_coverage = calculate_coverage_ratio(original_grid)
    optimized_coverage = calculate_coverage_ratio(optimized_grid)
    improvement = (optimized_coverage - original_coverage) * 100
    
    ax1.text(
        0.02, 0.02,
        f"Coverage: {original_coverage:.1%}\nRBS count: {len(original_rbs)}",
        transform=ax1.transAxes,
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    ax2.text(
        0.02, 0.02,
        f"Coverage: {optimized_coverage:.1%}\nRBS count: {len(optimized_rbs)}\nImprovement: +{improvement:.1f}%",
        transform=ax2.transAxes,
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create interactive version
    create_interactive_optimization_map(
        original_rbs, original_sectors, optimized_rbs, optimized_sectors, 
        original_grid, optimized_grid, output_path.replace('.png', '_interactive.html')
    )
    
    print(f"Optimization comparison map saved at {output_path}")

def create_interactive_optimization_map(original_rbs, original_sectors, 
                                      optimized_rbs, optimized_sectors,
                                      original_grid, optimized_grid, output_path):
    """
    Creates an interactive map comparing original and optimized coverage.
    
    Args:
        original_rbs: GeoDataFrame with original RBS locations
        original_sectors: GeoDataFrame with original sectors
        optimized_rbs: GeoDataFrame with optimized RBS
        optimized_sectors: GeoDataFrame with optimized sectors
        original_grid: GeoDataFrame with original coverage grid
        optimized_grid: GeoDataFrame with optimized coverage grid
        output_path: Path to save the HTML file
    """
    # Convert to WGS84 for Folium
    original_rbs_wgs84 = original_rbs.to_crs(epsg=4326)
    original_sectors_wgs84 = original_sectors.to_crs(epsg=4326)
    optimized_rbs_wgs84 = optimized_rbs.to_crs(epsg=4326)
    optimized_sectors_wgs84 = optimized_sectors.to_crs(epsg=4326)
    
    # Calculate center of the map
    center = original_rbs_wgs84.unary_union.centroid
    
    # Create map
    m = folium.Map(location=[center.y, center.x], zoom_start=11,
                 tiles='CartoDB positron')
    
    # Create layer groups
    original_group = folium.FeatureGroup(name='Original Coverage')
    optimized_group = folium.FeatureGroup(name='Optimized Coverage')
    
    # Add original RBS and sectors to original group
    for idx, rbs in original_rbs_wgs84.iterrows():
        folium.CircleMarker(
            location=[rbs.geometry.y, rbs.geometry.x],
            radius=6,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.7,
            popup=f"RBS #{idx}<br>Original Position",
        ).add_to(original_group)
    
    # Add original sectors
    folium.GeoJson(
        original_sectors_wgs84,
        style_function=lambda x: {
            'fillColor': 'blue',
            'color': 'blue',
            'weight': 1,
            'fillOpacity': 0.2
        }
    ).add_to(original_group)
    
    # Add optimized RBS and sectors to optimized group
    for idx, rbs in optimized_rbs_wgs84.iterrows():
        # Check if this is a new or moved RBS
        is_new = idx >= len(original_rbs_wgs84)
        
        if not is_new:
            # Calculate distance from original position
            orig_pos = original_rbs_wgs84.iloc[idx].geometry
            opt_pos = rbs.geometry
            
            # Skip if position hasn't changed
            if orig_pos.distance(opt_pos) < 0.0001:  # Small threshold for floating point comparison
                continue
                
            # Add a line showing the move
            folium.PolyLine(
                locations=[[orig_pos.y, orig_pos.x], [opt_pos.y, opt_pos.x]],
                color='red',
                weight=2,
                opacity=0.7,
                dash_array='5,5',
                popup=f"RBS #{idx} moved"
            ).add_to(optimized_group)
        
        # Add marker
        folium.CircleMarker(
            location=[rbs.geometry.y, rbs.geometry.x],
            radius=6,
            color='green' if not is_new else 'red',
            fill=True,
            fill_color='green' if not is_new else 'red',
            fill_opacity=0.7,
            popup=f"RBS #{idx}<br>{'New' if is_new else 'Optimized'} Position",
        ).add_to(optimized_group)
    
    # Add optimized sectors
    folium.GeoJson(
        optimized_sectors_wgs84,
        style_function=lambda x: {
            'fillColor': 'green',
            'color': 'green',
            'weight': 1,
            'fillOpacity': 0.2
        }
    ).add_to(optimized_group)
    
    # Add layer groups to map
    original_group.add_to(m)
    optimized_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add coverage statistics
    original_coverage = calculate_coverage_ratio(original_grid)
    optimized_coverage = calculate_coverage_ratio(optimized_grid)
    improvement = (optimized_coverage - original_coverage) * 100
    
    stats_html = f"""
    <div style="position: fixed; 
                bottom: 50px; left: 50px; 
                border:2px solid grey; z-index:9999; font-size:12px;
                background-color:white; 
                padding: 10px;
                opacity: 0.8;">
        <p><b>Coverage Statistics</b></p>
        <p>Original: {original_coverage:.1%}<br>
        Optimized: {optimized_coverage:.1%}<br>
        Improvement: +{improvement:.1f}%</p>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(stats_html))
    
    # Save to HTML file
    m.save(output_path)
    print(f"Interactive optimization map saved at {output_path}")

def predict_expansion_areas(gdf_rbs, gdf_sectors, population_data=None, road_network=None, 
                           output_path=None, n_suggested_areas=5):
    """
    Predicts and visualizes priority areas for network expansion.
    
    Args:
        gdf_rbs: GeoDataFrame with RBS locations
        gdf_sectors: GeoDataFrame with coverage sectors
        population_data: GeoDataFrame with population density (optional)
        road_network: GeoDataFrame with road network (optional)
        output_path: Path to save the visualization
        n_suggested_areas: Number of suggested areas to return
        
    Returns:
        GeoDataFrame: Suggested areas for new RBS deployment
    """
    print("Predicting priority areas for network expansion...")
    
    # Ensure correct CRS
    gdf_rbs = gdf_rbs.to_crs(epsg=3857)
    gdf_sectors = gdf_sectors.to_crs(epsg=3857)
    
    # Create a grid covering the study area
    bounds = gdf_sectors.total_bounds
    
    # Expand bounds slightly to include areas just outside current coverage
    margin = 0.2 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    analysis_bounds = [bounds[0] - margin, bounds[2] + margin, 
                      bounds[1] - margin, bounds[3] + margin]
    
    # Create grid with smaller cells
    grid_size = 0.005  # Smaller grid for more precise analysis
    x_points = np.arange(analysis_bounds[0], analysis_bounds[2], grid_size)
    y_points = np.arange(analysis_bounds[1], analysis_bounds[3], grid_size)
    
    grid_cells = []
    
    print(f"Analyzing {len(x_points) * len(y_points)} grid cells for expansion prediction...")
    
    for x in x_points:
        for y in y_points:
            # Create grid cell
            cell = box(x, y, x + grid_size, y + grid_size)
            
            # Check if this cell is already covered
            is_covered = any(sector.geometry is not None and sector.geometry.intersects(cell) 
                            for idx, sector in gdf_sectors.iterrows())
            
            # Calculate distance to nearest RBS
            nearest_rbs_dist = min([rbs.geometry.distance(cell.centroid) for idx, rbs in gdf_rbs.iterrows()], 
                                 default=float('inf'))
            
            # Calculate population score (if data available)
            population_score = 0
            if population_data is not None:
                population_data = population_data.to_crs(epsg=3857)
                for idx, pop_area in population_data.iterrows():
                    if pop_area.geometry.intersects(cell):
                        intersection_area = pop_area.geometry.intersection(cell).area
                        population_score += pop_area['population'] * (intersection_area / pop_area.geometry.area)
            
            # Calculate road access score (if data available)
            road_score = 0
            if road_network is not None:
                road_network = road_network.to_crs(epsg=3857)
                for idx, road in road_network.iterrows():
                    if road.geometry.intersects(cell):
                        road_score += road.geometry.intersection(cell).length
            
            # Add to grid cells
            grid_cells.append({
                'geometry': cell,
                'is_covered': is_covered,
                'nearest_rbs_dist': nearest_rbs_dist,
                'population_score': population_score,
                'road_score': road_score
            })
    
    # Convert to GeoDataFrame
    gdf_grid = gpd.GeoDataFrame(grid_cells, crs=gdf_sectors.crs)
    
    # Filter out already covered areas
    expansion_candidates = gdf_grid[~gdf_grid['is_covered']]
    
    if len(expansion_candidates) == 0:
        print("No uncovered areas found for expansion.")
        return gpd.GeoDataFrame(geometry=[])
    
    # Normalize scores for ranking
    max_dist = expansion_candidates['nearest_rbs_dist'].max()
    if max_dist > 0:
        expansion_candidates['dist_score'] = 1 - (expansion_candidates['nearest_rbs_dist'] / max_dist)
    else:
        expansion_candidates['dist_score'] = 0
    
    if 'population_score' in expansion_candidates.columns and expansion_candidates['population_score'].max() > 0:
        max_pop = expansion_candidates['population_score'].max()
        expansion_candidates['population_score_norm'] = expansion_candidates['population_score'] / max_pop
    else:
        expansion_candidates['population_score_norm'] = 0
    
    if 'road_score' in expansion_candidates.columns and expansion_candidates['road_score'].max() > 0:
        max_road = expansion_candidates['road_score'].max()
        expansion_candidates['road_score_norm'] = expansion_candidates['road_score'] / max_road
    else:
        expansion_candidates['road_score_norm'] = 0
    
    # Calculate composite score (weighted sum)
    # Adjust weights based on importance of each factor
    expansion_candidates['expansion_score'] = (
        0.3 * expansion_candidates['dist_score'] +
        0.5 * expansion_candidates['population_score_norm'] +
        0.2 * expansion_candidates['road_score_norm']
    )
    
    # Sort by expansion score
    expansion_candidates = expansion_candidates.sort_values('expansion_score', ascending=False)
    
    # Get top N areas
    top_areas = expansion_candidates.head(n_suggested_areas * 10)  # Get more than needed for clustering
    
    # Cluster nearby cells to group into expansion areas
    expansion_areas = cluster_expansion_areas(top_areas, n_clusters=n_suggested_areas)
    
    # Create visualization if output path is provided
    if output_path:
        create_expansion_prediction_map(gdf_rbs, gdf_sectors, expansion_areas, output_path)
    
    return expansion_areas

def cluster_expansion_areas(candidate_cells, n_clusters=5):
    """
    Clusters nearby candidate cells into expansion areas.
    
    Args:
        candidate_cells: GeoDataFrame with candidate cells
        n_clusters: Number of expansion areas to identify
        
    Returns:
        GeoDataFrame: Clustered expansion areas
    """
    if len(candidate_cells) == 0:
        return gpd.GeoDataFrame(geometry=[])
    
    # If we have very few candidates, just return them as is
    if len(candidate_cells) <= n_clusters:
        return candidate_cells
    
    # Extract centroids of candidate cells
    points = np.array([[p.x, p.y] for p in candidate_cells.geometry.centroid])
    
    try:
        # Use K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(points)
        
        # Add cluster ID to the DataFrame
        candidate_cells['cluster'] = clusters
        
        # For each cluster, create a merged area
        cluster_areas = []
        
        for cluster_id in range(n_clusters):
            # Get cells in this cluster
            cluster_cells = candidate_cells[candidate_cells['cluster'] == cluster_id]
            
            if len(cluster_cells) == 0:
                continue
                
            # Merge geometries
            merged_geometry = unary_union(cluster_cells.geometry)
            
            # Calculate average score
            avg_score = cluster_cells['expansion_score'].mean()
            
            # Add to output
            cluster_areas.append({
                'geometry': merged_geometry,
                'expansion_score': avg_score,
                'n_cells': len(cluster_cells),
                'cluster_id': cluster_id
            })
        
        # Convert to GeoDataFrame
        gdf_clusters = gpd.GeoDataFrame(cluster_areas, crs=candidate_cells.crs)
        
        # Sort by score
        return gdf_clusters.sort_values('expansion_score', ascending=False)
        
    except Exception as e:
        print(f"Error clustering expansion areas: {e}")
        # Fallback: just return top N cells
        return candidate_cells.head(n_clusters)

def create_expansion_prediction_map(gdf_rbs, gdf_sectors, expansion_areas, output_path):
    """
    Creates a map showing predicted expansion areas.
    
    Args:
        gdf_rbs: GeoDataFrame with RBS locations
        gdf_sectors: GeoDataFrame with coverage sectors
        expansion_areas: GeoDataFrame with suggested expansion areas
        output_path: Path to save the visualization
    """
    print("Creating expansion prediction map...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Plot coverage sectors
    gdf_sectors.plot(
        ax=ax,
        color='blue',
        alpha=0.2,
        edgecolor='none'
    )
    
    # Plot RBS locations
    gdf_rbs.plot(
        ax=ax,
        markersize=30,
        marker='o',
        color='blue',
        edgecolor='white',
        linewidth=0.5,
        alpha=0.8,
        label='Existing RBS'
    )
    
    # Plot expansion areas with color based on score
    if len(expansion_areas) > 0:
        # Create colormap for expansion scores
        norm = Normalize(
            vmin=expansion_areas['expansion_score'].min(),
            vmax=expansion_areas['expansion_score'].max()
        )
        
        # Plot areas
        expansion_areas.plot(
            ax=ax,
            column='expansion_score',
            cmap='Reds',
            alpha=0.7,
            edgecolor='darkred',
            linewidth=0.5,
            legend=True,
            legend_kwds={'label': 'Expansion Priority Score'}
        )
        
        # Add suggested RBS locations (at centroids of expansion areas)
        for idx, area in expansion_areas.iterrows():
            # Get centroid
            centroid = area.geometry.centroid
            
            # Add marker
            ax.scatter(centroid.x, centroid.y, 
                     s=100, marker='*', color='darkred', 
                     edgecolor='white', linewidth=0.5, zorder=5)
            
            # Add label
            ax.annotate(
                f"Area {idx+1}",
                xy=(centroid.x, centroid.y),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=10,
                color='black',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8)
            )
    
    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    # Add title
    ax.set_title('Predicted Priority Areas for Network Expansion', fontsize=16)
    
    # Add legend for RBS markers
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Existing RBS'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='darkred', markersize=10, label='Suggested New RBS')
    ]
    ax.legend(handles=custom_lines, loc='upper right')
    
    # Add statistics
    coverage_area = unary_union([sector.geometry for sector in gdf_sectors.geometry if sector is not None])
    coverage_area_km2 = coverage_area.area / 1_000_000  # Convert to km²
    
    stats_text = (
        f"Existing RBS: {len(gdf_rbs)}\n"
        f"Current Coverage: {coverage_area_km2:.2f} km²\n"
        f"Suggested New Areas: {len(expansion_areas)}"
    )
    
    ax.text(
        0.02, 0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create interactive version
    create_interactive_expansion_map(
        gdf_rbs, gdf_sectors, expansion_areas,
        output_path.replace('.png', '_interactive.html')
    )
    
    print(f"Expansion prediction map saved at {output_path}")

def create_interactive_expansion_map(gdf_rbs, gdf_sectors, expansion_areas, output_path):
    """
    Creates an interactive map showing predicted expansion areas.
    
    Args:
        gdf_rbs: GeoDataFrame with RBS locations
        gdf_sectors: GeoDataFrame with coverage sectors
        expansion_areas: GeoDataFrame with suggested expansion areas
        output_path: Path to save the HTML file
    """
    # Convert to WGS84 for Folium
    gdf_rbs_wgs84 = gdf_rbs.to_crs(epsg=4326)
    gdf_sectors_wgs84 = gdf_sectors.to_crs(epsg=4326)
    
    if len(expansion_areas) > 0:
        expansion_areas_wgs84 = expansion_areas.to_crs(epsg=4326)
    
    # Calculate center of the map
    center = gdf_rbs_wgs84.unary_union.centroid
    
    # Create map
    m = folium.Map(location=[center.y, center.x], zoom_start=11,
                 tiles='CartoDB positron')
    
    # Add existing RBS and coverage
    folium.GeoJson(
        gdf_sectors_wgs84,
        name='Current Coverage',
        style_function=lambda x: {
            'fillColor': 'blue',
            'color': 'blue',
            'weight': 1,
            'fillOpacity': 0.2
        }
    ).add_to(m)
    
    # Add RBS markers
    for idx, rbs in gdf_rbs_wgs84.iterrows():
        folium.CircleMarker(
            location=[rbs.geometry.y, rbs.geometry.x],
            radius=6,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.7,
            popup=f"RBS #{idx}<br>Existing"
        ).add_to(m)
    
    # Add expansion areas
    if len(expansion_areas) > 0:
        # Create a choropleth layer for expansion areas
        folium.GeoJson(
            expansion_areas_wgs84,
            name='Expansion Areas',
            style_function=lambda x: {
                'fillColor': get_color_for_expansion_score(x['properties']['expansion_score']),
                'color': 'darkred',
                'weight': 1,
                'fillOpacity': 0.6
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['expansion_score', 'n_cells', 'cluster_id'],
                aliases=['Priority Score:', 'Grid Cells:', 'Area ID:'],
                localize=True
            )
        ).add_to(m)
        
        # Add suggested new RBS locations
        for idx, area in expansion_areas_wgs84.iterrows():
            centroid = area.geometry.centroid
            
            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=folium.Icon(color='red', icon='star', prefix='fa'),
                popup=f"Suggested RBS<br>Area {idx+1}<br>Score: {area['expansion_score']:.2f}"
            ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; 
                border:2px solid grey; z-index:9999; font-size:12px;
                background-color:white; 
                padding: 10px;
                opacity: 0.8;">
        <p><b>Legend</b></p>
        <div>
            <i class="fa fa-circle" style="color:blue"></i> Existing RBS
        </div>
        <div>
            <i class="fa fa-star" style="color:red"></i> Suggested New RBS
        </div>
        <p><b>Priority Score</b></p>
        <div style="width:20px;height:20px;background-color:#bd0026;display:inline-block"></div> High
        <div style="width:20px;height:20px;background-color:#f03b20;display:inline-block"></div> Medium-High
        <div style="width:20px;height:20px;background-color:#fd8d3c;display:inline-block"></div> Medium
        <div style="width:20px;height:20px;background-color:#feb24c;display:inline-block"></div> Medium-Low
        <div style="width:20px;height:20px;background-color:#fed976;display:inline-block"></div> Low
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save to HTML file
    m.save(output_path)
    print(f"Interactive expansion map saved at {output_path}")

def get_color_for_expansion_score(score):
    """
    Returns a color based on the expansion score.
    
    Args:
        score: Expansion priority score
        
    Returns:
        str: Color in hex format
    """
    if score >= 0.8:
        return '#bd0026'  # Very high priority (dark red)
    elif score >= 0.6:
        return '#f03b20'  # High priority
    elif score >= 0.4:
        return '#fd8d3c'  # Medium priority
    elif score >= 0.2:
        return '#feb24c'  # Low priority
    else:
        return '#fed976'  # Very low priority 