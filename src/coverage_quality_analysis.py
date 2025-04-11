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
from shapely.ops import unary_union, cascaded_union
import folium
from folium.plugins import HeatMap, MarkerCluster
import os
import rasterio
import scipy.ndimage as ndimage
from scipy.spatial import Voronoi, voronoi_plot_2d

# Threshold values for quality analysis
CRITICAL_COVERAGE_THRESHOLD = 0.3  # Below 30% coverage is critical
REDUNDANT_COVERAGE_THRESHOLD = 3   # More than 3 overlaps is considered redundant
POPULATION_DENSITY_FACTOR = 1000   # People per square km in urban areas

def identify_critical_areas(gdf_sectors, grid_size=0.01, output_path=None):
    """
    Identifies areas with insufficient coverage based on overlapping sectors.
    
    Args:
        gdf_sectors: GeoDataFrame with coverage sectors
        grid_size: Size of the grid cells for the analysis (in CRS units)
        output_path: Path to save the result visualizations
        
    Returns:
        GeoDataFrame: Grid cells with coverage metrics and critical areas identified
    """
    print("Identifying areas with critical coverage...")
    
    # Ensure correct CRS (Web Mercator for area calculations)
    gdf_sectors = gdf_sectors.to_crs(epsg=3857)
    
    # Get the bounds of the study area
    bounds = gdf_sectors.total_bounds
    
    # Create a grid of points covering the study area
    x_points = np.arange(bounds[0], bounds[2], grid_size)
    y_points = np.arange(bounds[1], bounds[3], grid_size)
    grid_points = []
    
    # For each grid cell, create a point and check coverage
    grid_cells = []
    
    print(f"Analyzing coverage over {len(x_points) * len(y_points)} grid cells...")
    
    for x in x_points:
        for y in y_points:
            # Create grid cell (small square)
            cell = box(x, y, x + grid_size, y + grid_size)
            
            # Count overlapping sectors
            overlapping_sectors = sum([1 for idx, sector in gdf_sectors.iterrows() 
                                    if sector.geometry is not None and sector.geometry.intersects(cell)])
            
            # Calculate overlap ratio (percentage of the area covered)
            if overlapping_sectors > 0:
                # Get all intersecting sectors
                intersecting_geometries = [sector.geometry for idx, sector in gdf_sectors.iterrows() 
                                          if sector.geometry is not None and sector.geometry.intersects(cell)]
                
                # Merge all intersecting geometries
                if intersecting_geometries:
                    union_geometry = unary_union(intersecting_geometries)
                    overlap_area = union_geometry.intersection(cell).area
                    coverage_ratio = min(overlap_area / cell.area, 1.0)
                else:
                    coverage_ratio = 0
            else:
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
    
    # Convert to GeoDataFrame
    gdf_grid = gpd.GeoDataFrame(grid_cells, crs=gdf_sectors.crs)
    
    # Create visualization if output path is provided
    if output_path:
        create_critical_areas_map(gdf_grid, gdf_sectors, output_path)
    
    return gdf_grid

def create_critical_areas_map(gdf_grid, gdf_sectors, output_path):
    """
    Creates a map highlighting critical areas with insufficient coverage.
    
    Args:
        gdf_grid: GeoDataFrame with grid cells and coverage metrics
        gdf_sectors: GeoDataFrame with coverage sectors
        output_path: Path to save the visualization
    """
    print("Creating critical areas map...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Plot critical areas
    gdf_grid[gdf_grid['is_critical']].plot(
        ax=ax,
        color='red',
        alpha=0.7,
        edgecolor='none',
        label='Critical Areas'
    )
    
    # Plot regular grid with coverage ratio as color
    gdf_grid[~gdf_grid['is_critical']].plot(
        ax=ax,
        column='coverage_ratio',
        cmap='viridis',
        alpha=0.5,
        edgecolor='none',
        legend=True,
        legend_kwds={'label': 'Coverage Ratio', 'orientation': 'horizontal'}
    )
    
    # Plot sector outlines
    gdf_sectors.boundary.plot(
        ax=ax,
        color='blue',
        linewidth=0.5,
        alpha=0.4
    )
    
    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
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
    create_interactive_critical_map(gdf_grid, gdf_sectors, 
                                  output_path.replace('.png', '_interactive.html'))
    
    print(f"Critical areas map saved at {output_path}")

def create_interactive_critical_map(gdf_grid, gdf_sectors, output_path):
    """
    Creates an interactive web map showing critical coverage areas.
    
    Args:
        gdf_grid: GeoDataFrame with grid cells and coverage metrics
        gdf_sectors: GeoDataFrame with coverage sectors
        output_path: Path to save the HTML file
    """
    # Convert to WGS84 for Folium
    gdf_grid_wgs84 = gdf_grid.to_crs(epsg=4326)
    gdf_sectors_wgs84 = gdf_sectors.to_crs(epsg=4326)
    
    # Find center of the map
    center = gdf_grid_wgs84.unary_union.centroid
    
    # Create map
    m = folium.Map(location=[center.y, center.x], zoom_start=11,
                 tiles='CartoDB positron')
    
    # Add critical areas
    folium.GeoJson(
        gdf_grid_wgs84[gdf_grid_wgs84['is_critical']],
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
    folium.GeoJson(
        gdf_grid_wgs84[~gdf_grid_wgs84['is_critical']],
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

def analyze_coverage_redundancy(gdf_sectors, grid_size=0.01, output_path=None):
    """
    Analyzes areas with excessive or insufficient coverage overlap.
    
    Args:
        gdf_sectors: GeoDataFrame with coverage sectors
        grid_size: Size of the grid cells for the analysis (in CRS units)
        output_path: Path to save the visualization
        
    Returns:
        GeoDataFrame: Grid cells with redundancy metrics
    """
    print("Analyzing coverage redundancy...")
    
    # Ensure correct CRS
    gdf_sectors = gdf_sectors.to_crs(epsg=3857)
    
    # Get the bounds of the study area
    bounds = gdf_sectors.total_bounds
    
    # Create a grid of cells covering the study area
    x_points = np.arange(bounds[0], bounds[2], grid_size)
    y_points = np.arange(bounds[1], bounds[3], grid_size)
    
    # For each grid cell, create a polygon and check coverage
    grid_cells = []
    
    print(f"Analyzing redundancy over {len(x_points) * len(y_points)} grid cells...")
    
    for x in x_points:
        for y in y_points:
            # Create grid cell
            cell = box(x, y, x + grid_size, y + grid_size)
            
            # Count overlapping sectors
            overlaps = sum([1 for idx, sector in gdf_sectors.iterrows() 
                          if sector.geometry is not None and sector.geometry.intersects(cell)])
            
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
    
    # Convert to GeoDataFrame
    gdf_redundancy = gpd.GeoDataFrame(grid_cells, crs=gdf_sectors.crs)
    
    # Create visualization if output path is provided
    if output_path:
        create_redundancy_map(gdf_redundancy, gdf_sectors, output_path)
    
    return gdf_redundancy

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
    
    # Ensure correct CRS for area calculations
    gdf_rbs = gdf_rbs.to_crs(epsg=3857)
    gdf_sectors = gdf_sectors.to_crs(epsg=3857)
    
    # Calculate total area covered (excluding overlaps)
    total_coverage = unary_union([sector.geometry for sector in gdf_sectors.geometry if sector is not None])
    total_area_km2 = total_coverage.area / 1_000_000  # Convert m² to km²
    
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
        # Ensure correct CRS
        population_data = population_data.to_crs(epsg=3857)
        
        # Calculate population covered
        # This is a simplified approach - in reality, we would need to do a proper spatial join
        # and calculate the exact population within coverage areas
        covered_population = sum([
            pop_area['population'] * min(
                pop_area.geometry.intersection(total_coverage).area / pop_area.geometry.area, 
                1.0
            )
            for idx, pop_area in population_data.iterrows()
        ])
        
        population_coverage = covered_population
        people_per_rbs = covered_population / total_rbs if total_rbs > 0 else 0
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

def estimate_population_coverage(gdf_rbs, total_area_km2):
    """
    Estimates population coverage based on area type.
    
    Args:
        gdf_rbs: GeoDataFrame with RBS information
        total_area_km2: Total coverage area in km²
        
    Returns:
        float: Estimated population coverage
    """
    # Count RBS by area type
    area_type_counts = gdf_rbs['area_type'].value_counts() if 'area_type' in gdf_rbs.columns else {}
    
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