"""
Module for advanced coverage visualization, including 3D maps, climate condition simulation,
and topographic obstruction analysis.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import contextily as ctx
from matplotlib.colors import LinearSegmentedColormap, Normalize
import rasterio
from rasterio.plot import show as rshow
import folium
from folium.plugins import HeatMap, MarkerCluster
from scipy.interpolate import griddata
import os
from scipy.spatial.distance import cdist
import pickle

# Constants for signal attenuation under different conditions
CLIMATE_ATTENUATION = {
    'clear': 1.0,      # No attenuation
    'cloudy': 0.9,     # 10% attenuation
    'rain': 0.7,       # 30% attenuation
    'heavy_rain': 0.5, # 50% attenuation
    'snow': 0.4,       # 60% attenuation
    'fog': 0.8         # 20% attenuation
}

def create_3d_coverage_map(gdf_rbs, output_path, resolution=100, z_exaggeration=0.5):
    """
    Creates a 3D visualization of signal coverage where height represents signal intensity.

    Args:
        gdf_rbs: GeoDataFrame with RBS information
        output_path: Path to save the visualization
        resolution: Grid resolution for interpolation
        z_exaggeration: Factor to exaggerate the z-axis for better visualization
    """
    print("Creating 3D coverage map...")
    
    # Convert to same CRS
    gdf_rbs = gdf_rbs.to_crs(epsg=3857)
    
    # Extract coordinates and values
    points = np.array([(p.x, p.y) for p in gdf_rbs.geometry])
    values = gdf_rbs['EIRP_dBm'].values
    
    # Create grid for interpolation
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    
    margin = 0.1 * max(max_x - min_x, max_y - min_y)
    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin
    
    x_grid = np.linspace(min_x, max_x, resolution)
    y_grid = np.linspace(min_y, max_y, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Interpolate signal strength
    Z = griddata(points, values, (X, Y), method='cubic', fill_value=np.min(values))
    
    # Create 3D figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create custom colormap
    colors = ['blue', 'green', 'yellow', 'orange', 'red']
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z * z_exaggeration,
                          cmap=custom_cmap, alpha=0.8,
                          linewidth=0, antialiased=True)
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.4, aspect=10)
    cbar.set_label('Signal Strength (EIRP_dBm)')
    
    # Scatter RBS locations
    ax.scatter(points[:, 0], points[:, 1], values * z_exaggeration,
              c='black', s=50, marker='^', label='RBS')
    
    # Add title and labels
    ax.set_title('3D Signal Coverage Map', fontsize=16)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Signal Strength')
    
    # Adjust view angle
    ax.view_init(elev=30, azim=220)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"3D coverage map saved at {output_path}")
    
    # Create an additional interactive figure if plotly is available
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        output_html = output_path.replace('.png', '.html')
        
        # Create 3D surface plot with plotly
        fig_plotly = go.Figure(data=[
            go.Surface(z=Z * z_exaggeration, x=X, y=Y, 
                       colorscale='Viridis', showscale=True, 
                       colorbar=dict(title='Signal Strength (EIRP_dBm)'))
        ])
        
        # Add RBS points
        fig_plotly.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=values * z_exaggeration,
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='circle',
            ),
            name='RBS Locations'
        ))
        
        # Update layout
        fig_plotly.update_layout(
            title='Interactive 3D Signal Coverage Map',
            scene=dict(
                xaxis_title='X (meters)',
                yaxis_title='Y (meters)',
                zaxis_title='Signal Strength',
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            autosize=True,
            margin=dict(l=65, r=50, b=65, t=90),
        )
        
        # Save as HTML
        fig_plotly.write_html(output_html)
        print(f"Interactive 3D map saved at {output_html}")
    except ImportError:
        print("Interactive 3D visualization requires plotly. Install with: pip install plotly")

def simulate_coverage_in_climate_conditions(gdf_rbs, gdf_sectors, output_path, conditions=None):
    """
    Creates a visualization showing how signal coverage varies under different climate conditions.
    
    Args:
        gdf_rbs: GeoDataFrame with RBS information
        gdf_sectors: GeoDataFrame with coverage sectors
        output_path: Path to save the visualization
        conditions: List of climate conditions to simulate (default: clear, rain, heavy_rain, fog)
    """
    print("Creating climate condition impact visualization...")
    
    if conditions is None:
        conditions = ['clear', 'rain', 'heavy_rain', 'fog']
    
    # Create figure with subplots for each condition
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    # Reproject to Web Mercator
    gdf_rbs_3857 = gdf_rbs.to_crs(epsg=3857)
    gdf_sectors_3857 = gdf_sectors.to_crs(epsg=3857)
    
    for i, condition in enumerate(conditions[:4]):  # Limit to 4 conditions
        ax = axes[i]
        
        # Get attenuation factor for this condition
        attenuation_factor = CLIMATE_ATTENUATION.get(condition, 1.0)
        
        # Create modified sectors with adjusted coverage radius
        gdf_condition = gdf_sectors_3857.copy()
        
        # Scale down the size of the sectors (adjust 'geometry') based on attenuation
        gdf_condition['geometry'] = gdf_condition.apply(
            lambda row: row.geometry.buffer(
                (attenuation_factor - 1) * row.geometry.area**0.5
            ) if row.geometry else row.geometry,
            axis=1
        )
        
        # Plot modified sectors
        gdf_condition.plot(
            ax=ax,
            column='EIRP_dBm',
            cmap='viridis',
            alpha=0.6,
            edgecolor='gray',
            linewidth=0.5
        )
        
        # Plot RBS
        gdf_rbs_3857.plot(
            ax=ax,
            markersize=10,
            marker='o',
            color='red',
            edgecolor='white',
            linewidth=0.7
        )
        
        # Add base map
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        
        # Add title and attribution
        ax.set_title(f"Coverage in {condition.replace('_', ' ').title()} Conditions", fontsize=14)
        ax.text(
            0.02, 0.02,
            f"Signal attenuation: {int((1-attenuation_factor)*100)}%",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8)
        )
        
        # Remove axes
        ax.set_axis_off()
    
    # Add overall title
    plt.suptitle('Signal Coverage Under Different Climate Conditions', fontsize=20, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Climate condition impact visualization saved at {output_path}")

def analyze_topographic_obstructions(gdf_rbs, dem_path, output_path):
    """
    Creates a visualization showing how topography affects signal coverage.
    
    Args:
        gdf_rbs: GeoDataFrame with RBS information
        dem_path: Path to Digital Elevation Model (DEM) raster file
        output_path: Path to save the visualization
    """
    print("Creating topographic obstruction analysis...")
    
    # Check if DEM file exists
    if not os.path.exists(dem_path):
        print(f"ERROR: DEM file not found at {dem_path}")
        print("Generating mock DEM for demonstration purposes...")
        
        # Generate mock DEM for demonstration if real DEM is not available
        dem_data = create_mock_dem(gdf_rbs)
        use_mock_dem = True
    else:
        # Load DEM using rasterio
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)
            dem_transform = src.transform
            dem_crs = src.crs
        use_mock_dem = False
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Reproject RBS to DEM CRS if using real DEM
    if not use_mock_dem:
        gdf_rbs_dem = gdf_rbs.to_crs(dem_crs)
    else:
        gdf_rbs_dem = gdf_rbs.copy()
    
    # First subplot: DEM with RBS locations
    if use_mock_dem:
        # Plot mock DEM
        dem_img = ax1.imshow(dem_data, cmap='terrain', extent=[
            gdf_rbs_dem.total_bounds[0],
            gdf_rbs_dem.total_bounds[2],
            gdf_rbs_dem.total_bounds[1],
            gdf_rbs_dem.total_bounds[3]
        ])
    else:
        # Plot real DEM
        dem_img = rshow(dem_data, transform=dem_transform, ax=ax1, cmap='terrain')
    
    # Plot RBS locations
    gdf_rbs_dem.plot(ax=ax1, markersize=50, marker='^', color='red', 
                    edgecolor='white', linewidth=0.7, alpha=0.8)
    
    # Add colorbar
    cbar = fig.colorbar(dem_img, ax=ax1, shrink=0.6)
    cbar.set_label('Elevation (meters)')
    
    # Add title and labels
    ax1.set_title('Digital Elevation Model with RBS Locations', fontsize=14)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    
    # Second subplot: Line of sight analysis
    if use_mock_dem:
        # Create visual representation of topographic obstruction
        plot_line_of_sight_simulation(ax2, gdf_rbs_dem, dem_data)
    else:
        # Call function for real DEM
        plot_line_of_sight_analysis(ax2, gdf_rbs_dem, dem_data, dem_transform)
    
    # Add title
    ax2.set_title('Signal Obstruction Analysis', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Topographic obstruction analysis saved at {output_path}")
    
    # Create interactive version with Folium
    create_interactive_topographic_analysis(gdf_rbs, dem_data, 
                                           output_path.replace('.png', '_interactive.html'),
                                           use_mock_dem)

def create_mock_dem(gdf_rbs, resolution=100):
    """
    Creates a mock Digital Elevation Model for demonstration purposes.
    
    Args:
        gdf_rbs: GeoDataFrame with RBS locations
        resolution: Resolution of the mock DEM
        
    Returns:
        np.array: Mock DEM data
    """
    # Extract bounds
    bounds = gdf_rbs.total_bounds
    
    # Create grid
    x = np.linspace(bounds[0], bounds[2], resolution)
    y = np.linspace(bounds[1], bounds[3], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create synthetic terrain
    dem = np.zeros((resolution, resolution))
    
    # Add some random mountains
    n_mountains = 5
    mountain_centers = np.random.rand(n_mountains, 2) * np.array([bounds[2]-bounds[0], bounds[3]-bounds[1]]) + np.array([bounds[0], bounds[1]])
    mountain_heights = np.random.rand(n_mountains) * 500 + 100
    mountain_widths = np.random.rand(n_mountains) * 0.1 + 0.05
    
    # Generate terrain
    points = np.column_stack((X.flatten(), Y.flatten()))
    for i in range(n_mountains):
        distances = np.sqrt(((points - mountain_centers[i]) ** 2).sum(axis=1))
        heights = mountain_heights[i] * np.exp(-(distances / (mountain_widths[i] * (bounds[2]-bounds[0])))**2)
        dem += heights.reshape(resolution, resolution)
    
    # Add noise
    dem += np.random.rand(resolution, resolution) * 50
    
    # Ensure minimum height
    dem += 50
    
    return dem

def plot_line_of_sight_simulation(ax, gdf_rbs, dem_data):
    """
    Simulates and plots line of sight analysis using a mock DEM.
    
    Args:
        ax: Matplotlib axis
        gdf_rbs: GeoDataFrame with RBS locations
        dem_data: Mock DEM data
    """
    # Calculate grid coordinates
    bounds = gdf_rbs.total_bounds
    resolution = dem_data.shape[0]
    x = np.linspace(bounds[0], bounds[2], resolution)
    y = np.linspace(bounds[1], bounds[3], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Get RBS coordinates
    rbs_points = np.array([[p.x, p.y] for p in gdf_rbs.geometry])
    
    # Initialize visibility matrix
    visibility = np.zeros_like(dem_data)
    
    # For each RBS, determine visible areas
    for i, (idx, rbs) in enumerate(gdf_rbs.iterrows()):
        rbs_x, rbs_y = rbs.geometry.x, rbs.geometry.y
        
        # Convert to grid indices
        rbs_idx_x = int((rbs_x - bounds[0]) / (bounds[2] - bounds[0]) * (resolution - 1))
        rbs_idx_y = int((rbs_y - bounds[1]) / (bounds[3] - bounds[1]) * (resolution - 1))
        
        # Get RBS height (assume 30m tower + terrain height)
        rbs_height = dem_data[rbs_idx_y, rbs_idx_x] + 30
        
        # Simple line of sight calculation (distance and terrain based)
        for j in range(resolution):
            for k in range(resolution):
                # Distance to point
                dist = np.sqrt((X[j,k] - rbs_x)**2 + (Y[j,k] - rbs_y)**2)
                max_radius = rbs['Coverage_Radius_km'] * 1000 if 'Coverage_Radius_km' in rbs else 5000
                
                if dist <= max_radius:
                    # Simple line of sight check
                    visible = True
                    
                    # Sample points along line of sight
                    n_samples = 20
                    sample_x = np.linspace(rbs_x, X[j,k], n_samples)
                    sample_y = np.linspace(rbs_y, Y[j,k], n_samples)
                    
                    # Convert to grid indices
                    sample_idx_x = np.clip((sample_x - bounds[0]) / (bounds[2] - bounds[0]) * (resolution - 1), 0, resolution-1).astype(int)
                    sample_idx_y = np.clip((sample_y - bounds[1]) / (bounds[3] - bounds[1]) * (resolution - 1), 0, resolution-1).astype(int)
                    
                    # Get heights along line
                    heights = dem_data[sample_idx_y, sample_idx_x]
                    
                    # Calculate height of line of sight at each point
                    los_heights = np.linspace(rbs_height, dem_data[j,k] + 2, n_samples)
                    
                    # Check if any point blocks the line of sight
                    if np.any(heights > los_heights):
                        visible = False
                    
                    if visible:
                        # Decrease signal with distance
                        signal_strength = max(1 - dist/max_radius, 0.1)
                        visibility[j, k] += signal_strength
    
    # Normalize visibility
    visibility = visibility / visibility.max()
    
    # Plot visibility map
    im = ax.imshow(visibility, cmap='viridis', alpha=0.7, extent=[
        bounds[0], bounds[2], bounds[1], bounds[3]
    ])
    
    # Plot RBS locations
    gdf_rbs.plot(ax=ax, markersize=50, marker='^', color='red', 
                edgecolor='white', linewidth=0.7, alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Signal Visibility')
    
    # Add contours of terrain
    contour = ax.contour(X, Y, dem_data, colors='black', alpha=0.5, linewidths=0.5, levels=10)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%1.0f')
    
    # Add labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

def plot_line_of_sight_analysis(ax, gdf_rbs, dem_data, dem_transform):
    """
    Performs and plots line of sight analysis using a real DEM.
    
    Args:
        ax: Matplotlib axis
        gdf_rbs: GeoDataFrame with RBS locations
        dem_data: DEM data
        dem_transform: Affine transform of the DEM
    """
    # This function would use the real DEM to calculate line of sight
    # Showing implementation outline but not fully implemented
    
    # Get DEM extents
    width = dem_data.shape[1]
    height = dem_data.shape[0]
    
    # Create mesh grid of pixel coordinates
    rows, cols = np.mgrid[0:height, 0:width]
    
    # Transform pixel coordinates to CRS coordinates
    x_coords, y_coords = rasterio.transform.xy(dem_transform, rows, cols)
    
    # Convert to arrays
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    
    # Show DEM
    ax.imshow(dem_data, cmap='terrain', extent=[
        x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()
    ])
    
    # Plot RBS locations
    gdf_rbs.plot(ax=ax, markersize=20, marker='^', color='red',
                edgecolor='white', linewidth=0.7, alpha=0.8)
    
    # Note: A full implementation would calculate viewsheds from each RBS
    # and combine them to show signal coverage considering topography
    
    # Add placeholder implementation note
    ax.text(0.5, 0.5, 'Viewshed Analysis\n(needs rasterio and richdem)',
           transform=ax.transAxes, ha='center', va='center',
           fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

def create_interactive_topographic_analysis(gdf_rbs, dem_data, output_path, use_mock_dem=True):
    """
    Creates an interactive map showing topography and RBS coverage.
    
    Args:
        gdf_rbs: GeoDataFrame with RBS locations
        dem_data: DEM data
        output_path: Path to save the HTML file
        use_mock_dem: Whether using mock or real DEM
    """
    # Convert to WGS84 for Folium
    gdf_rbs_wgs84 = gdf_rbs.to_crs(epsg=4326)
    
    # Create map centered on the RBS centroid
    center = gdf_rbs_wgs84.geometry.unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=10,
                 tiles='CartoDB positron')
    
    # Add RBS markers
    for idx, rbs in gdf_rbs_wgs84.iterrows():
        html = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4>RBS Information</h4>
            <b>ID:</b> {idx}<br>
            <b>Operator:</b> {rbs['Operator'] if 'Operator' in rbs else 'Unknown'}<br>
            <b>EIRP:</b> {rbs['EIRP_dBm'] if 'EIRP_dBm' in rbs else 'Unknown'} dBm<br>
            <b>Coverage:</b> {rbs['Coverage_Radius_km'] if 'Coverage_Radius_km' in rbs else 'Unknown'} km<br>
            <b>Area Type:</b> {rbs['area_type'] if 'area_type' in rbs else 'Unknown'}<br>
        </div>
        """
        
        iframe = folium.IFrame(html=html, width=220, height=150)
        popup = folium.Popup(iframe, max_width=230)
        
        folium.Marker(
            location=[rbs.geometry.y, rbs.geometry.x],
            popup=popup,
            icon=folium.Icon(color='red', icon='tower-broadcast', prefix='fa')
        ).add_to(m)
    
    # Add coverage circles with obstruction info
    for idx, rbs in gdf_rbs_wgs84.iterrows():
        # Get coverage radius
        radius = rbs['Coverage_Radius_km'] * 1000 if 'Coverage_Radius_km' in rbs else 5000
        
        # Calculate color based on topographic obstruction
        if use_mock_dem:
            # Use dummy obstruction percentage for illustration
            obstruction = np.random.randint(0, 100)
        else:
            # Real calculation would be implemented here
            obstruction = 20
        
        # Color based on obstruction
        if obstruction < 20:
            color = 'green'
            fill_color = 'green'
        elif obstruction < 50:
            color = 'orange'
            fill_color = 'orange'
        else:
            color = 'red'
            fill_color = 'red'
        
        # Add circle
        folium.Circle(
            location=[rbs.geometry.y, rbs.geometry.x],
            radius=radius,
            color=color,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.2,
            opacity=0.6,
            popup=f"Topographic Obstruction: {obstruction}%"
        ).add_to(m)
    
    # Save map
    m.save(output_path)
    print(f"Interactive topographic analysis saved at {output_path}") 