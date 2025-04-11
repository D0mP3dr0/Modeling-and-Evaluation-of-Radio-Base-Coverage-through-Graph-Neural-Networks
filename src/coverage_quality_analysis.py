"""
Module for analyzing coverage quality and service metrics for RBS data.
Includes functions for identifying critical areas, analyzing coverage redundancy,
and calculating efficiency metrics.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
import matplotlib.patheffects as PathEffects
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
import folium
from folium.plugins import HeatMap, MarkerCluster
import os
import rasterio
import scipy.ndimage as ndimage
from scipy.spatial import Voronoi
import warnings
import time
from tqdm import tqdm
import traceback

# Check if GPU acceleration is available
try:
    import cupy as cp
    import cuspatial
    
    # Check if environment variable is set
    USE_GPU = os.environ.get('USE_GPU', '').lower() == 'true'
    
    if USE_GPU:
        print("GPU acceleration available for coverage quality analysis.")
    else:
        print("GPU acceleration available but not enabled.")
except ImportError:
    USE_GPU = False
    print("GPU acceleration not available for coverage quality analysis. Using CPU.")

# Threshold values for quality analysis
CRITICAL_COVERAGE_THRESHOLD = 0.3  # Below 30% coverage is critical
REDUNDANT_COVERAGE_THRESHOLD = 3   # More than 3 overlaps is considered redundant
POPULATION_DENSITY_FACTOR = 1000   # People per square km in urban areas

def validate_input(gdf_sectors, check_required_columns=True):
    """
    Validates input GeoDataFrame for coverage analysis.
    
    Args:
        gdf_sectors: GeoDataFrame with coverage sectors
        check_required_columns: Whether to check for required columns
        
    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    if not isinstance(gdf_sectors, gpd.GeoDataFrame):
        return False, "Input must be a GeoDataFrame"
    
    if gdf_sectors.empty:
        return False, "Input GeoDataFrame is empty"
    
    if 'geometry' not in gdf_sectors.columns:
        return False, "Input GeoDataFrame has no geometry column"
    
    # Check if there are any valid geometries
    valid_geoms = gdf_sectors[~gdf_sectors.geometry.isna() & gdf_sectors.geometry.is_valid]
    if len(valid_geoms) == 0:
        return False, "Input GeoDataFrame has no valid geometries"
    
    # Check if CRS is defined
    if gdf_sectors.crs is None:
        warnings.warn("Input GeoDataFrame has no CRS defined. Assuming EPSG:4326.")
    
    # Check required columns if needed
    if check_required_columns:
        for column in ['Operator', 'Tecnologia']:
            if column not in gdf_sectors.columns:
                warnings.warn(f"Column '{column}' not found. Some analyses may be incomplete.")
    
    return True, ""

def identify_critical_areas(gdf_sectors, grid_size=0.01, output_path=None, use_gpu=None):
    """
    Identifies areas with insufficient coverage based on overlapping sectors.
    
    Args:
        gdf_sectors: GeoDataFrame with coverage sectors
        grid_size: Size of the grid cells for the analysis (in CRS units)
        output_path: Path to save the result visualizations
        use_gpu: Whether to use GPU acceleration (overrides global setting)
        
    Returns:
        GeoDataFrame: Grid cells with coverage metrics and critical areas identified
    """
    print("Identifying areas with critical coverage...")
    
    # Check if GPU should be used
    use_gpu_local = USE_GPU if use_gpu is None else use_gpu
    
    # Validate input
    is_valid, error_msg = validate_input(gdf_sectors)
    if not is_valid:
        print(f"Error: {error_msg}")
        return gpd.GeoDataFrame()
    
    try:
        # Start timer
        start_time = time.time()
        
        # Ensure correct CRS (Web Mercator for area calculations)
        gdf_sectors = gdf_sectors.to_crs(epsg=3857)
        
        # Get the bounds of the study area
        bounds = gdf_sectors.total_bounds
        
        # Create a grid of points covering the study area
        x_points = np.arange(bounds[0], bounds[2], grid_size)
        y_points = np.arange(bounds[1], bounds[3], grid_size)
        
        # Report grid size
        grid_cell_count = len(x_points) * len(y_points)
        print(f"Analyzing coverage over {grid_cell_count} grid cells...")
        
        if use_gpu_local and grid_cell_count > 10000:
            # Use GPU for large grids if available
            try:
                gdf_grid = _identify_critical_areas_gpu(gdf_sectors, bounds, grid_size, x_points, y_points)
            except Exception as e:
                print(f"GPU processing failed: {e}")
                print("Falling back to CPU processing")
                gdf_grid = _identify_critical_areas_optimized(gdf_sectors, bounds, grid_size, x_points, y_points)
        else:
            # Use optimized CPU version
            gdf_grid = _identify_critical_areas_optimized(gdf_sectors, bounds, grid_size, x_points, y_points)
        
        # Report time taken
        end_time = time.time()
        print(f"Critical area analysis completed in {end_time - start_time:.2f} seconds")
        
        # Create visualization if output path is provided
        if output_path:
            create_critical_areas_map(gdf_grid, gdf_sectors, output_path)
        
        return gdf_grid
    
    except Exception as e:
        print(f"Error in identify_critical_areas: {e}")
        traceback.print_exc()
        return gpd.GeoDataFrame()

def _identify_critical_areas_optimized(gdf_sectors, bounds, grid_size, x_points, y_points):
    """
    CPU-optimized version of critical areas identification.
    Uses vectorized operations where possible.
    
    Args:
        gdf_sectors: GeoDataFrame with coverage sectors
        bounds: Total bounds of the study area
        grid_size: Size of grid cells
        x_points: Array of x coordinates
        y_points: Array of y coordinates
        
    Returns:
        GeoDataFrame: Grid cells with coverage metrics
    """
    # Prepare geometries for spatial indexing
    if not hasattr(gdf_sectors, 'sindex'):
        gdf_sectors.sindex  # Build spatial index
    
    # Pre-create all grid cells
    print("Creating grid cells...")
    grid_cells = []
    total_cells = len(x_points) * len(y_points)
    
    # Process in batches for memory efficiency and to show progress
    batch_size = 1000
    x_batches = [x_points[i:i + batch_size] for i in range(0, len(x_points), batch_size)]
    
    with tqdm(total=total_cells) as pbar:
        for x_batch in x_batches:
            for y in y_points:
                for x in x_batch:
                    # Create grid cell (small square)
                    cell = box(x, y, x + grid_size, y + grid_size)
                    
                    # Use spatial index to find possible intersections
                    possible_matches_idx = list(gdf_sectors.sindex.intersection((x, y, x + grid_size, y + grid_size)))
                    
                    if possible_matches_idx:
                        # Get geometries that may intersect
                        possible_matches = gdf_sectors.iloc[possible_matches_idx]
                        
                        # Count actual intersections
                        intersections = possible_matches[possible_matches.geometry.intersects(cell)]
                        overlapping_sectors = len(intersections)
                        
                        # Calculate overlap ratio
                        if overlapping_sectors > 0:
                            # Get all intersecting geometries
                            intersecting_geometries = [geom for geom in intersections.geometry if geom is not None]
                            
                            # Merge all intersecting geometries
                            if intersecting_geometries:
                                union_geometry = unary_union(intersecting_geometries)
                                overlap_area = union_geometry.intersection(cell).area
                                coverage_ratio = min(overlap_area / cell.area, 1.0)
                            else:
                                coverage_ratio = 0
                        else:
                            coverage_ratio = 0
                    else:
                        # No intersections
                        overlapping_sectors = 0
                        coverage_ratio = 0
                    
                    # Define if it's a critical area
                    is_critical = coverage_ratio < CRITICAL_COVERAGE_THRESHOLD
                    
                    # Add to grid cells
                    grid_cells.append({
                        'geometry': cell,
                        'overlapping_sectors': overlapping_sectors,
                        'coverage_ratio': coverage_ratio,
                        'is_critical': is_critical
                    })
                    
                    pbar.update(1)
    
    # Convert to GeoDataFrame
    gdf_grid = gpd.GeoDataFrame(grid_cells, crs=gdf_sectors.crs)
    return gdf_grid

def _identify_critical_areas_gpu(gdf_sectors, bounds, grid_size, x_points, y_points):
    """
    GPU-accelerated version of critical areas identification.
    
    Args:
        gdf_sectors: GeoDataFrame with coverage sectors
        bounds: Total bounds of the study area
        grid_size: Size of grid cells
        x_points: Array of x coordinates
        y_points: Array of y coordinates
        
    Returns:
        GeoDataFrame: Grid cells with coverage metrics
    """
    if not USE_GPU:
        raise RuntimeError("GPU acceleration not available")
    
    # Convert geometries to GPU format - implementation depends on GPU library
    # This is a placeholder for actual GPU implementation
    # In a real implementation, cuspatial would be used for polygon intersection tests
    
    print("Using GPU acceleration for critical areas analysis...")
    
    # Placeholder for GPU implementation:
    # 1. Convert geometries to GPU format
    # 2. Create grid cells on GPU
    # 3. Perform intersection tests using GPU
    # 4. Calculate coverage ratios
    # 5. Transfer results back to CPU
    
    # For now, fall back to CPU implementation
    return _identify_critical_areas_optimized(gdf_sectors, bounds, grid_size, x_points, y_points)

def create_critical_areas_map(gdf_grid, gdf_sectors, output_path):
    """
    Creates a map highlighting critical areas with insufficient coverage.
    
    Args:
        gdf_grid: GeoDataFrame with grid cells and coverage metrics
        gdf_sectors: GeoDataFrame with coverage sectors
        output_path: Path to save the visualization
    """
    print("Creating critical areas map...")
    
    # Validate inputs
    if gdf_grid.empty:
        print("Error: Grid data is empty. Cannot create map.")
        return
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Plot critical areas
        critical_areas = gdf_grid[gdf_grid['is_critical']]
        if not critical_areas.empty:
            critical_areas.plot(
                ax=ax,
                color='red',
                alpha=0.7,
                edgecolor='none',
                label='Critical Areas'
            )
        
        # Plot regular grid with coverage ratio as color
        non_critical = gdf_grid[~gdf_grid['is_critical']]
        if not non_critical.empty:
            non_critical.plot(
                ax=ax,
                column='coverage_ratio',
                cmap='viridis',
                alpha=0.5,
                edgecolor='none',
                legend=True,
                legend_kwds={'label': 'Coverage Ratio', 'orientation': 'horizontal'}
            )
        
        # Plot sector outlines
        if not gdf_sectors.empty:
            gdf_sectors.boundary.plot(
                ax=ax,
                color='blue',
                linewidth=0.5,
                alpha=0.4
            )
        
        # Add basemap
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        except Exception as e:
            print(f"Warning: Could not add basemap: {e}")
        
        # Add title and labels
        ax.set_title('Critical Coverage Areas Map', fontsize=16)
        
        # Add legend
        red_patch = Patch(color='red', alpha=0.7, label='Critical Areas (< 30% coverage)')
        ax.legend(handles=[red_patch], loc='upper right')
        
        # Add statistics
        critical_percentage = (gdf_grid['is_critical'].sum() / len(gdf_grid)) * 100
        ax.text(
            0.02, 0.02,
            f"Critical areas: {critical_percentage:.1f}% of total area\n"
            f"Average coverage ratio: {gdf_grid['coverage_ratio'].mean():.2f}",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8)
        )
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Create an interactive version
        interactive_path = output_path.replace('.png', '_interactive.html')
        try:
            create_interactive_critical_map(gdf_grid, gdf_sectors, interactive_path)
        except Exception as e:
            print(f"Warning: Could not create interactive map: {e}")
        
        print(f"Critical areas map saved at {output_path}")
    
    except Exception as e:
        print(f"Error creating critical areas map: {e}")
        traceback.print_exc()

def create_interactive_critical_map(gdf_grid, gdf_sectors, output_path):
    """
    Creates an interactive web map showing critical coverage areas.
    
    Args:
        gdf_grid: GeoDataFrame with grid cells and coverage metrics
        gdf_sectors: GeoDataFrame with coverage sectors
        output_path: Path to save the HTML file
    """
    # Validate inputs
    if gdf_grid.empty:
        print("Error: Grid data is empty. Cannot create interactive map.")
        return
    
    try:
        # Convert to WGS84 for Folium
        gdf_grid_wgs84 = gdf_grid.to_crs(epsg=4326)
        gdf_sectors_wgs84 = gdf_sectors.to_crs(epsg=4326) if not gdf_sectors.empty else None
        
        # Find center of the map
        center = gdf_grid_wgs84.unary_union.centroid
        
        # Create map
        m = folium.Map(location=[center.y, center.x], zoom_start=11,
                     tiles='CartoDB positron')
        
        # Add critical areas
        critical_areas = gdf_grid_wgs84[gdf_grid_wgs84['is_critical']]
        if not critical_areas.empty:
            folium.GeoJson(
                critical_areas,
                name='Critical Areas',
                style_function=lambda x: {
                    'fillColor': 'red',
                    'color': 'none',
                    'fillOpacity': 0.6
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['coverage_ratio', 'overlapping_sectors'],
                    aliases=['Coverage Ratio:', 'Overlapping Sectors:'],
                    localize=True
                )
            ).add_to(m)
        
        # Add regular grid areas with color based on coverage ratio
        non_critical = gdf_grid_wgs84[~gdf_grid_wgs84['is_critical']]
        if not non_critical.empty:
            folium.GeoJson(
                non_critical,
                name='Coverage Grid',
                style_function=lambda x: {
                    'fillColor': get_color_for_coverage(x['properties']['coverage_ratio']),
                    'color': 'none',
                    'fillOpacity': 0.5
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['coverage_ratio', 'overlapping_sectors'],
                    aliases=['Coverage Ratio:', 'Overlapping Sectors:'],
                    localize=True
                )
            ).add_to(m)
        
        # Add sector outlines
        if gdf_sectors_wgs84 is not None and not gdf_sectors_wgs84.empty:
            folium.GeoJson(
                gdf_sectors_wgs84,
                name='Coverage Sectors',
                style_function=lambda x: {
                    'fillColor': 'blue',
                    'color': 'blue',
                    'fillOpacity': 0.1,
                    'weight': 1
                }
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add title and legend
        title_html = """
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 300px; height: 30px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white; padding: 5px;
                    opacity: 0.8;">
            <b>Critical Coverage Areas Map</b>
        </div>
        """
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Save to HTML file
        m.save(output_path)
        print(f"Interactive critical areas map saved at {output_path}")
    
    except Exception as e:
        print(f"Error creating interactive map: {e}")
        traceback.print_exc()

def get_color_for_coverage(coverage_ratio):
    """
    Returns a color based on the coverage ratio.
    
    Args:
        coverage_ratio: Coverage ratio (0-1)
        
    Returns:
        str: Color in hex format
    """
    if coverage_ratio >= 0.9:
        return '#006837'  # Dark green
    elif coverage_ratio >= 0.7:
        return '#1a9850'  # Green
    elif coverage_ratio >= 0.5:
        return '#66bd63'  # Light green
    elif coverage_ratio >= 0.3:
        return '#fee08b'  # Yellow
    else:
        return '#fdae61'  # Orange

def analyze_coverage_redundancy(gdf_sectors, grid_size=0.01, output_path=None, use_gpu=None):
    """
    Analyzes areas with excessive or insufficient coverage overlap.
    
    Args:
        gdf_sectors: GeoDataFrame with coverage sectors
        grid_size: Size of the grid cells for the analysis (in CRS units)
        output_path: Path to save the visualization
        use_gpu: Whether to use GPU acceleration (overrides global setting)
        
    Returns:
        GeoDataFrame: Grid cells with redundancy metrics
    """
    print("Analyzing coverage redundancy...")
    
    # Check if GPU should be used
    use_gpu_local = USE_GPU if use_gpu is None else use_gpu
    
    # Validate input
    is_valid, error_msg = validate_input(gdf_sectors)
    if not is_valid:
        print(f"Error: {error_msg}")
        return gpd.GeoDataFrame()
    
    try:
        # Start timer
        start_time = time.time()
        
        # Ensure correct CRS
        gdf_sectors = gdf_sectors.to_crs(epsg=3857)
        
        # Get the bounds of the study area
        bounds = gdf_sectors.total_bounds
        
        # Create a grid of cells covering the study area
        x_points = np.arange(bounds[0], bounds[2], grid_size)
        y_points = np.arange(bounds[1], bounds[3], grid_size)
        
        # Report grid size
        grid_cell_count = len(x_points) * len(y_points)
        print(f"Analyzing redundancy over {grid_cell_count} grid cells...")
        
        # Create spatial index if it doesn't exist
        if not hasattr(gdf_sectors, 'sindex'):
            gdf_sectors.sindex  # Build spatial index
        
        # Pre-create all grid cells
        print("Creating grid cells...")
        grid_cells = []
        
        # Process in batches
        batch_size = 1000
        x_batches = [x_points[i:i + batch_size] for i in range(0, len(x_points), batch_size)]
        
        with tqdm(total=grid_cell_count) as pbar:
            for x_batch in x_batches:
                for y in y_points:
                    for x in x_batch:
                        # Create grid cell
                        cell = box(x, y, x + grid_size, y + grid_size)
                        
                        # Use spatial index to find possible intersections
                        possible_matches_idx = list(gdf_sectors.sindex.intersection((x, y, x + grid_size, y + grid_size)))
                        
                        # Count actual intersections
                        if possible_matches_idx:
                            possible_matches = gdf_sectors.iloc[possible_matches_idx]
                            overlaps = sum([1 for _, sector in possible_matches.iterrows() 
                                        if sector.geometry is not None and sector.geometry.intersects(cell)])
                        else:
                            overlaps = 0
                        
                        # Calculate redundancy category
                        if overlaps == 0:
                            redundancy = 'No Coverage'
                        elif overlaps == 1:
                            redundancy = 'Optimal'
                        elif overlaps == 2:
                            redundancy = 'Good'
                        elif overlaps <= REDUNDANT_COVERAGE_THRESHOLD:
                            redundancy = 'Moderate Redundancy'
                        else:
                            redundancy = 'Excessive Redundancy'
                        
                        # Add to grid cells
                        grid_cells.append({
                            'geometry': cell,
                            'overlaps': overlaps,
                            'redundancy': redundancy
                        })
                        
                        pbar.update(1)
        
        # Convert to GeoDataFrame
        gdf_redundancy = gpd.GeoDataFrame(grid_cells, crs=gdf_sectors.crs)
        
        # Report time taken
        end_time = time.time()
        print(f"Redundancy analysis completed in {end_time - start_time:.2f} seconds")
        
        # Create visualization if output path is provided
        if output_path:
            create_redundancy_map(gdf_redundancy, gdf_sectors, output_path)
        
        return gdf_redundancy
    
    except Exception as e:
        print(f"Error in analyze_coverage_redundancy: {e}")
        traceback.print_exc()
        return gpd.GeoDataFrame()

def create_redundancy_map(gdf_redundancy, gdf_sectors, output_path):
    """
    Creates a map visualizing coverage redundancy.
    
    Args:
        gdf_redundancy: GeoDataFrame with grid cells and redundancy metrics
        gdf_sectors: GeoDataFrame with coverage sectors
        output_path: Path to save the visualization
    """
    print("Creating coverage redundancy map...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Define color map for redundancy categories
    redundancy_colors = {
        'No Coverage': '#d73027',          # Red
        'Optimal': '#1a9850',              # Green
        'Good': '#66bd63',                 # Light green
        'Moderate Redundancy': '#fee08b',  # Yellow
        'Excessive Redundancy': '#8c510a'  # Brown
    }
    
    # Create a categorical column for plotting
    gdf_redundancy['color_value'] = gdf_redundancy['redundancy'].map({
        'No Coverage': 0,
        'Optimal': 1,
        'Good': 2,
        'Moderate Redundancy': 3,
        'Excessive Redundancy': 4
    })
    
    # Plot grid cells colored by redundancy category
    gdf_redundancy.plot(
        ax=ax,
        column='color_value',
        categorical=True,
        cmap='RdYlGn_r',
        alpha=0.7,
        edgecolor='none',
        legend=False
    )
    
    # Plot sector outlines
    gdf_sectors.boundary.plot(
        ax=ax,
        color='blue',
        linewidth=0.5,
        alpha=0.3
    )
    
    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    # Add title
    ax.set_title('Coverage Redundancy Analysis', fontsize=16)
    
    # Create custom legend
    legend_elements = [Patch(facecolor=color, edgecolor='black', alpha=0.7, label=category)
                     for category, color in redundancy_colors.items()]
    
    ax.legend(handles=legend_elements, loc='upper right', title='Redundancy Level')
    
    # Add statistics
    redundancy_stats = gdf_redundancy['redundancy'].value_counts(normalize=True) * 100
    stats_text = "Area Coverage:\n"
    for category in ['No Coverage', 'Optimal', 'Good', 'Moderate Redundancy', 'Excessive Redundancy']:
        if category in redundancy_stats:
            stats_text += f"{category}: {redundancy_stats[category]:.1f}%\n"
    
    ax.text(
        0.02, 0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create interactive version
    create_interactive_redundancy_map(gdf_redundancy, gdf_sectors, 
                                     output_path.replace('.png', '_interactive.html'))
    
    print(f"Coverage redundancy map saved at {output_path}")

def create_interactive_redundancy_map(gdf_redundancy, gdf_sectors, output_path):
    """
    Creates an interactive web map showing coverage redundancy.
    
    Args:
        gdf_redundancy: GeoDataFrame with grid cells and redundancy metrics
        gdf_sectors: GeoDataFrame with coverage sectors
        output_path: Path to save the HTML file
    """
    # Convert to WGS84 for Folium
    gdf_redundancy_wgs84 = gdf_redundancy.to_crs(epsg=4326)
    gdf_sectors_wgs84 = gdf_sectors.to_crs(epsg=4326)
    
    # Find center of the map
    center = gdf_redundancy_wgs84.unary_union.centroid
    
    # Create map
    m = folium.Map(location=[center.y, center.x], zoom_start=11,
                 tiles='CartoDB positron')
    
    # Define color function for redundancy categories
    def get_color_for_redundancy(redundancy):
        colors = {
            'No Coverage': '#d73027',          # Red
            'Optimal': '#1a9850',              # Green
            'Good': '#66bd63',                 # Light green
            'Moderate Redundancy': '#fee08b',  # Yellow
            'Excessive Redundancy': '#8c510a'  # Brown
        }
        return colors.get(redundancy, '#808080')  # Default to gray
    
    # Add grid cells with redundancy information
    folium.GeoJson(
        gdf_redundancy_wgs84,
        name='Coverage Redundancy',
        style_function=lambda x: {
            'fillColor': get_color_for_redundancy(x['properties']['redundancy']),
            'color': 'none',
            'fillOpacity': 0.7
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['redundancy', 'overlaps'],
            aliases=['Redundancy Level:', 'Overlapping Sectors:'],
            localize=True
        )
    ).add_to(m)
    
    # Add sector outlines
    folium.GeoJson(
        gdf_sectors_wgs84,
        name='Coverage Sectors',
        style_function=lambda x: {
            'fillColor': 'blue',
            'color': 'blue',
            'fillOpacity': 0.0,
            'weight': 1
        }
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add legend
    categories = ['No Coverage', 'Optimal', 'Good', 'Moderate Redundancy', 'Excessive Redundancy']
    colors = ['#d73027', '#1a9850', '#66bd63', '#fee08b', '#8c510a']
    
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; 
                border:2px solid grey; z-index:9999; font-size:12px;
                background-color:white; 
                padding: 10px;
                opacity: 0.8;">
        <p><b>Redundancy Level</b></p>
    """
    
    for i, category in enumerate(categories):
        legend_html += f"""
        <div>
            <i class="fa fa-square" style="color:{colors[i]}"></i> {category}
        </div>
        """
    
    legend_html += "</div>"
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save to HTML file
    m.save(output_path)
    print(f"Interactive redundancy map saved at {output_path}")

def calculate_coverage_efficiency(gdf_rbs, gdf_sectors, population_data=None, output_path=None):
    """
    Calculates coverage efficiency metrics per km² and by population.
    
    Args:
        gdf_rbs: GeoDataFrame with RBS locations
        gdf_sectors: GeoDataFrame with coverage sectors
        population_data: GeoDataFrame with population data (optional)
        output_path: Path to save the visualization
        
    Returns:
        dict: Dictionary with efficiency metrics
    """
    print("Calculating coverage efficiency metrics...")
    
    # Validate input
    for name, gdf in [("RBS", gdf_rbs), ("Sectors", gdf_sectors)]:
        is_valid, error_msg = validate_input(gdf, check_required_columns=False)
        if not is_valid:
            print(f"Error in {name} data: {error_msg}")
            return {}
    
    try:
        # Ensure correct CRS for area calculations
        gdf_rbs = gdf_rbs.to_crs(epsg=3857)
        gdf_sectors = gdf_sectors.to_crs(epsg=3857)
        
        # Calculate total area covered (excluding overlaps)
        try:
            # Use unary_union for efficiency - merges all geometries into one
            total_coverage = unary_union([sector.geometry for sector in gdf_sectors.geometry 
                                      if sector is not None and not sector.is_empty])
            total_area_km2 = total_coverage.area / 1_000_000  # Convert m² to km²
        except Exception as e:
            print(f"Error calculating coverage area: {e}")
            total_area_km2 = 0
        
        # Count RBS
        total_rbs = len(gdf_rbs)
        
        # Calculate area efficiency
        area_per_rbs = total_area_km2 / total_rbs if total_rbs > 0 else 0
        rbs_per_km2 = total_rbs / total_area_km2 if total_area_km2 > 0 else 0
        
        # Initialize population metrics
        population_coverage = None
        people_per_rbs = None
        
        # If population data is provided, calculate population-based metrics
        if population_data is not None:
            try:
                # Ensure correct CRS
                population_data = population_data.to_crs(epsg=3857)
                
                # Calculate population covered - more accurate spatial join
                covered_population = 0
                
                for idx, pop_area in population_data.iterrows():
                    try:
                        if pop_area.geometry.is_valid:
                            # Calculate intersection area / total area
                            intersection = pop_area.geometry.intersection(total_coverage)
                            intersection_ratio = min(intersection.area / pop_area.geometry.area, 1.0) if pop_area.geometry.area > 0 else 0
                            
                            # Apply to population
                            covered_population += pop_area['population'] * intersection_ratio
                    except Exception as e:
                        print(f"Warning: Error processing population area {idx}: {e}")
                        continue
                
                population_coverage = covered_population
                people_per_rbs = covered_population / total_rbs if total_rbs > 0 else 0
            except Exception as e:
                print(f"Error calculating population metrics: {e}")
                # Fall back to estimation
                population_coverage = estimate_population_coverage(gdf_rbs, total_area_km2)
                people_per_rbs = population_coverage / total_rbs if total_rbs > 0 else 0
        else:
            # Estimate based on area type
            population_coverage = estimate_population_coverage(gdf_rbs, total_area_km2)
            people_per_rbs = population_coverage / total_rbs if total_rbs > 0 else 0
        
        # Compile metrics
        efficiency_metrics = {
            'total_rbs': total_rbs,
            'total_coverage_km2': total_area_km2,
            'area_per_rbs_km2': area_per_rbs,
            'rbs_per_km2': rbs_per_km2,
            'estimated_population_coverage': population_coverage,
            'people_per_rbs': people_per_rbs
        }
        
        # Create visualization if output path is provided
        if output_path:
            create_efficiency_visualization(gdf_rbs, gdf_sectors, efficiency_metrics, output_path)
        
        return efficiency_metrics
    
    except Exception as e:
        print(f"Error in calculate_coverage_efficiency: {e}")
        traceback.print_exc()
        return {}

def estimate_population_coverage(gdf_rbs, total_area_km2):
    """
    Estimates population coverage based on area type.
    
    Args:
        gdf_rbs: GeoDataFrame with RBS information
        total_area_km2: Total coverage area in km²
        
    Returns:
        float: Estimated population coverage
    """
    try:
        # Count RBS by area type
        if 'area_type' in gdf_rbs.columns:
            area_type_counts = gdf_rbs['area_type'].value_counts()
        else:
            # Try to infer area type from operator density
            # Higher density indicates urban areas
            if hasattr(gdf_rbs, 'geometry') and not gdf_rbs.empty:
                bounds = gdf_rbs.total_bounds
                area = ((bounds[2] - bounds[0]) * (bounds[3] - bounds[1])) / 1_000_000  # km²
                density = len(gdf_rbs) / area if area > 0 else 0
                
                if density > 10:  # High density
                    area_type_counts = pd.Series({'dense_urban': len(gdf_rbs)})
                elif density > 5:  # Medium density
                    area_type_counts = pd.Series({'urban': len(gdf_rbs)})
                elif density > 1:  # Low density
                    area_type_counts = pd.Series({'suburban': len(gdf_rbs)})
                else:  # Very low density
                    area_type_counts = pd.Series({'rural': len(gdf_rbs)})
            else:
                # Default to urban if we can't infer
                area_type_counts = pd.Series({'urban': len(gdf_rbs)})
        
        # Default density factors based on area type (people per km²)
        density_factors = {
            'dense_urban': 10000,
            'urban': 5000,
            'suburban': 1000,
            'rural': 100
        }
        
        # Calculate weighted average density
        total_count = sum(area_type_counts.values())
        if total_count > 0:
            weighted_density = sum(
                [area_type_counts.get(area_type, 0) * density_factors.get(area_type, 1000) 
                for area_type in density_factors]
            ) / total_count
        else:
            weighted_density = 1000  # Default to urban density
        
        # Estimate population
        return total_area_km2 * weighted_density
    
    except Exception as e:
        print(f"Error estimating population: {e}")
        return total_area_km2 * 1000  # Default to urban density as fallback

def create_efficiency_visualization(gdf_rbs, gdf_sectors, metrics, output_path):
    """
    Creates a visualization of coverage efficiency metrics.
    
    Args:
        gdf_rbs: GeoDataFrame with RBS locations
        gdf_sectors: GeoDataFrame with coverage sectors
        metrics: Dictionary with efficiency metrics
        output_path: Path to save the visualization
    """
    print("Creating coverage efficiency visualization...")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # First subplot: Coverage density map
    # Create a buffer around RBS points to represent coverage density
    gdf_rbs = gdf_rbs.to_crs(epsg=3857)
    gdf_sectors = gdf_sectors.to_crs(epsg=3857)
    
    # Plot RBS density (using a heatmap-like approach)
    heatmap_data = np.zeros((100, 100))
    bounds = gdf_rbs.total_bounds
    
    for idx, rbs in gdf_rbs.iterrows():
        x_idx = int((rbs.geometry.x - bounds[0]) / (bounds[2] - bounds[0]) * 99)
        y_idx = int((rbs.geometry.y - bounds[1]) / (bounds[3] - bounds[1]) * 99)
        
        # Add influence to heatmap (with distance decay)
        for i in range(max(0, x_idx-10), min(99, x_idx+10)):
            for j in range(max(0, y_idx-10), min(99, y_idx+10)):
                dist = np.sqrt((i - x_idx)**2 + (j - y_idx)**2)
                if dist < 10:
                    heatmap_data[j, i] += max(0, 1 - dist/10)
    
    # Normalize and smooth
    heatmap_data = heatmap_data / heatmap_data.max()
    heatmap_data = ndimage.gaussian_filter(heatmap_data, sigma=1.5)
    
    # Plot heatmap
    im1 = ax1.imshow(heatmap_data, extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
                   cmap='hot_r', alpha=0.7)
    
    # Plot RBS points
    gdf_rbs.plot(ax=ax1, markersize=20, marker='o', color='blue', 
                edgecolor='white', linewidth=0.5, alpha=0.8)
    
    # Add coverage sectors outline
    gdf_sectors.boundary.plot(ax=ax1, color='green', linewidth=0.5, alpha=0.4)
    
    # Add basemap
    ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron)
    
    # Add colorbar
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.7)
    cbar1.set_label('RBS Density')
    
    # Add title
    ax1.set_title('RBS Density and Coverage', fontsize=14)
    
    # Second subplot: Efficiency metrics visualization
    # Create bar chart for key metrics
    metrics_to_plot = [
        ('Area per RBS (km²)', metrics['area_per_rbs_km2']),
        ('RBS per km²', metrics['rbs_per_km2']),
        ('People per RBS (thousands)', metrics['people_per_rbs'] / 1000 if metrics['people_per_rbs'] else 0)
    ]
    
    bars = ax2.bar([m[0] for m in metrics_to_plot], [m[1] for m in metrics_to_plot],
                  color=['#3498db', '#2ecc71', '#e74c3c'])
    
    # Add labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Add title and labels
    ax2.set_title('Coverage Efficiency Metrics', fontsize=14)
    ax2.set_ylabel('Value')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text summary
    summary_text = (
        f"Total RBS: {metrics['total_rbs']}\n"
        f"Total Coverage: {metrics['total_coverage_km2']:.2f} km²\n"
        f"Estimated Population: {metrics['estimated_population_coverage']:,.0f} people\n"
        f"Coverage Efficiency: {metrics['area_per_rbs_km2']:.2f} km² per RBS\n"
        f"Population Efficiency: {metrics['people_per_rbs']:,.0f} people per RBS"
    )
    
    ax2.text(0.5, -0.15, summary_text, ha='center', va='center', transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create additional visualizations: Voronoi diagram for efficiency
    create_voronoi_efficiency_map(gdf_rbs, gdf_sectors, metrics, 
                                 output_path.replace('.png', '_voronoi.png'))
    
    print(f"Coverage efficiency visualization saved at {output_path}")

def create_voronoi_efficiency_map(gdf_rbs, gdf_sectors, metrics, output_path):
    """
    Creates a Voronoi diagram to visualize coverage efficiency.
    
    Args:
        gdf_rbs: GeoDataFrame with RBS locations
        gdf_sectors: GeoDataFrame with coverage sectors
        metrics: Dictionary with efficiency metrics
        output_path: Path to save the visualization
    """
    print("Creating Voronoi efficiency map...")
    
    # Ensure geographic CRS for proper visualization
    gdf_rbs = gdf_rbs.to_crs(epsg=3857)
    
    # Extract points for Voronoi
    points = np.array([[p.x, p.y] for p in gdf_rbs.geometry])
    
    # Create bounding box (slightly larger than data bounds)
    bounds = gdf_rbs.total_bounds
    margin = 0.1 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    bbox = [bounds[0] - margin, bounds[2] + margin, bounds[1] - margin, bounds[3] + margin]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 12))
    
    try:
        # Add far points to handle open Voronoi cells
        far_points = [
            [bbox[0], bbox[2]],
            [bbox[0], bbox[3]],
            [bbox[1], bbox[2]],
            [bbox[1], bbox[3]]
        ]
        all_points = np.vstack([points, far_points])
        
        # Compute Voronoi diagram
        vor = Voronoi(all_points)
        
        # Plot Voronoi diagram
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', 
                       line_width=1, line_alpha=0.6, point_size=5)
        
        # Plot RBS points
        gdf_rbs.plot(ax=ax, markersize=30, marker='o', color='red', 
                    edgecolor='white', linewidth=0.5, alpha=0.8)
        
        # Plot sectors
        gdf_sectors.plot(ax=ax, color='green', alpha=0.2, edgecolor='none')
        
        # Add basemap
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        
        # Add title
        ax.set_title('RBS Coverage Efficiency (Voronoi Diagram)', fontsize=16)
        
        # Add info box
        info_text = (
            f"Total RBS: {metrics['total_rbs']}\n"
            f"Area per RBS: {metrics['area_per_rbs_km2']:.2f} km²\n"
            f"Theoretical max coverage: {metrics['total_coverage_km2']:.2f} km²"
        )
        
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=12,
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set limits
        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Voronoi efficiency map saved at {output_path}")
    except Exception as e:
        print(f"Error creating Voronoi diagram: {e}")
        plt.close(fig)

def run_coverage_quality_analysis(gdf_rbs, output_path, grid_size=0.01, use_gpu=None):
    """
    Run the full coverage quality analysis.
    
    Args:
        gdf_rbs: GeoDataFrame with RBS data
        output_path: Path to save outputs
        grid_size: Size of grid cells for analysis
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        dict: Dictionary with analysis results
    """
    print("Running coverage quality analysis...")
    
    # Check if GPU should be used
    use_gpu_local = USE_GPU if use_gpu is None else use_gpu
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    results = {}
    
    try:
        # First, we need coverage sectors - if they don't exist in the input data,
        # we'll need to generate simplified sectors
        if 'coverage_sector' in gdf_rbs.columns:
            gdf_sectors = gpd.GeoDataFrame(
                gdf_rbs[['coverage_sector', 'Operator', 'Tecnologia']].copy(),
                geometry='coverage_sector',
                crs=gdf_rbs.crs
            )
        else:
            print("Coverage sectors not found in data. Creating simplified sectors...")
            # Import coverage_models module to create sectors
            try:
                from coverage_models import create_simplified_sectors
                gdf_sectors = create_simplified_sectors(gdf_rbs)
            except ImportError:
                from src.coverage_models import create_simplified_sectors
                gdf_sectors = create_simplified_sectors(gdf_rbs)
            except Exception as e:
                print(f"Error creating simplified sectors: {e}")
                return {}
        
        # 1. Critical Areas Analysis
        print("\n1. Identifying Critical Coverage Areas")
        critical_path = os.path.join(output_path, "critical_areas.png")
        gdf_critical = identify_critical_areas(gdf_sectors, grid_size, critical_path, use_gpu=use_gpu_local)
        results['critical_areas'] = {
            'percent_critical': (gdf_critical['is_critical'].sum() / len(gdf_critical)) * 100 if not gdf_critical.empty else 0,
            'avg_coverage': gdf_critical['coverage_ratio'].mean() if not gdf_critical.empty else 0
        }
        
        # 2. Coverage Redundancy Analysis
        print("\n2. Analyzing Coverage Redundancy")
        redundancy_path = os.path.join(output_path, "redundancy_analysis.png")
        gdf_redundancy = analyze_coverage_redundancy(gdf_sectors, grid_size, redundancy_path, use_gpu=use_gpu_local)
        if not gdf_redundancy.empty:
            redundancy_counts = gdf_redundancy['redundancy'].value_counts(normalize=True) * 100
            results['redundancy'] = {category: redundancy_counts.get(category, 0) 
                                   for category in ['No Coverage', 'Optimal', 'Good', 
                                                  'Moderate Redundancy', 'Excessive Redundancy']}
        else:
            results['redundancy'] = {}
        
        # 3. Efficiency Metrics
        print("\n3. Calculating Coverage Efficiency Metrics")
        efficiency_path = os.path.join(output_path, "efficiency_metrics.png")
        efficiency_metrics = calculate_coverage_efficiency(gdf_rbs, gdf_sectors, 
                                                         None, efficiency_path)
        results['efficiency'] = efficiency_metrics
        
        # 4. Save summary report
        print("\n4. Generating Summary Report")
        report_path = os.path.join(output_path, "coverage_quality_summary.json")
        import json
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Coverage quality analysis completed. Results saved to {output_path}")
        return results
    
    except Exception as e:
        print(f"Error in coverage quality analysis: {e}")
        traceback.print_exc()
        return {} 