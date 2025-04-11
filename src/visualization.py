"""
Module for advanced visualizations of RBS data.
Contains functions to create thematic maps, signal coverage visualizations,
heat maps and other visual representations of RBS and their characteristics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
import contextily as ctx
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
import matplotlib.patheffects as PathEffects
from matplotlib.offsetbox import AnchoredSizeBar
import matplotlib.font_manager as fm
from datetime import datetime
import os
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.interpolate import griddata

# Default color configuration for operators
OPERATOR_COLORS = {
    'CLARO': '#E02020',
    'OI': '#FFD700',
    'VIVO': '#9932CC',
    'TIM': '#0000CD'
}

def configure_visualization_style():
    """Configures the default style for all visualizations."""
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.figsize': (12, 8),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2
    })

def add_cartographic_elements(ax, title, source="Data: Anatel (RBS), OpenStreetMap (Base)"):
    """
    Adds standard cartographic elements to a matplotlib map.
    
    Args:
        ax: Matplotlib axis where elements will be added
        title: Map title
        source: Text with data source
    """
    # Title
    ax.set_title(title, fontweight='bold', pad=20)
    
    # Scale
    scalebar = AnchoredSizeBar(ax.transData, 5000, '5 km', 'lower right', pad=0.5,
                              color='black', frameon=True, size_vertical=100)
    ax.add_artist(scalebar)
    
    # North arrow
    x, y, arrow_length = 0.06, 0.12, 0.08
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='black', width=2, headwidth=8),
                ha='center', va='center', fontsize=14,
                xycoords='figure fraction',
                fontweight='bold',
                bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    # Source and date
    ax.annotate(f"{source}\nGenerated on: {datetime.now().strftime('%d/%m/%Y')}",
                xy=(0.01, 0.01), xycoords='figure fraction', fontsize=8,
                ha='left', va='bottom',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    # Grid and other elements
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.title.set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground='white')])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

def create_custom_legend(ax, colors, title="Legend"):
    """
    Creates a custom legend with operator colors.
    
    Args:
        ax: Matplotlib axis where the legend will be added
        colors: Dictionary with operator names and their colors
        title: Legend title
    """
    legend_elements = [Patch(facecolor=color, edgecolor='k', alpha=0.7, label=operator)
                         for operator, color in colors.items()]
    legend = ax.legend(handles=legend_elements, title=title, loc='upper right',
                      frameon=True, framealpha=0.8, edgecolor='k')
    legend.get_frame().set_linewidth(0.8)
    plt.setp(legend.get_title(), fontweight='bold')

def create_positioning_map(gdf_rbs, output_path, operator_colors=OPERATOR_COLORS):
    """
    Creates a map showing the position of RBS by operator.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS
        output_path: Path to save the map
        operator_colors: Dictionary of colors for each operator
    """
    print("Creating RBS positioning map...")
    
    # Reproject to Web Mercator (EPSG:3857) for use with contextily
    gdf_rbs_3857 = gdf_rbs.to_crs(epsg=3857)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Calculate sizes based on EIRP
    min_eirp = gdf_rbs_3857['EIRP_dBm'].min()
    max_eirp = gdf_rbs_3857['EIRP_dBm'].max()
    sizes = ((gdf_rbs_3857['EIRP_dBm'] - min_eirp) / (max_eirp - min_eirp) * 120 + 30)
    
    # Plot each operator with a different color
    for operator, color in operator_colors.items():
        subset = gdf_rbs_3857[gdf_rbs_3857['Operator'] == operator]
        if subset.empty:
            continue
            
        subset_sizes = sizes.loc[subset.index]
        
        # Add glow effect
        ax.scatter(subset.geometry.x, subset.geometry.y,
                  s=subset_sizes * 1.5, color=color, alpha=0.2, edgecolor='none')
                  
        # Add points
        ax.scatter(subset.geometry.x, subset.geometry.y,
                  s=subset_sizes, color=color, alpha=0.8, edgecolor='white', 
                  linewidth=0.5, label=operator)
    
    # Add OpenStreetMap base map
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    
    # Add cartographic elements
    add_cartographic_elements(ax, 'Radio Base Stations (RBS) Positioning by Operator')
    
    # Add legend
    create_custom_legend(ax, operator_colors, "Operators")
    
    # Add note about point sizes
    ax.annotate('Point size proportional to radiated power (EIRP)',
                xy=(0.5, 0.02), xycoords='figure fraction', ha='center',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Positioning map saved at {output_path}")

def create_coverage_map_by_operator(gdf_rbs, gdf_sectors, output_path, operator_colors=OPERATOR_COLORS):
    """
    Creates a map with 4 subplots showing the coverage of each operator.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS
        gdf_sectors: GeoDataFrame with the coverage sectors
        output_path: Path to save the map
        operator_colors: Dictionary of colors for each operator
    """
    print("Creating coverage map by operator...")
    
    # Reproject to Web Mercator
    gdf_rbs_3857 = gdf_rbs.to_crs(epsg=3857)
    gdf_sectors_3857 = gdf_sectors.to_crs(epsg=3857)
    
    # Create figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.flatten()
    
    operators = ['CLARO', 'OI', 'VIVO', 'TIM']
    
    for i, operator in enumerate(operators):
        ax = axes[i]
        
        # Filter data for the current operator
        subset_rbs = gdf_rbs_3857[gdf_rbs_3857['Operator'] == operator]
        subset_sectors = gdf_sectors_3857[gdf_sectors_3857['Operator'] == operator]
        
        if subset_sectors.empty:
            ax.set_title(f'No data for {operator}', fontsize=16)
            continue
            
        # Get color for the operator
        base_color = operator_colors[operator]
        sectors_color = f"{base_color}66"  # Color with transparency (alpha = 66)
        
        # Plot coverage sectors
        subset_sectors.plot(ax=ax, color=sectors_color, edgecolor=base_color, linewidth=0.3, alpha=0.6)
        
        # Plot RBS
        subset_rbs.plot(ax=ax, color=base_color, markersize=50, marker='o',
                       edgecolor='white', linewidth=0.7, alpha=0.9)
        
        # Add base map
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        
        # Add cartographic elements
        add_cartographic_elements(
            ax, 
            f'Coverage of Operator {operator}'
        )
        
        # Add statistical information
        n_rbs = len(subset_rbs)
        average_coverage = subset_rbs['Coverage_Radius_km'].mean()
        coverage_density = n_rbs / 325  # Approximation of the area in km²
        
        info_text = (f"Total RBS: {n_rbs}\n"
                    f"Average radius: {average_coverage:.2f} km\n"
                    f"Density: {coverage_density:.2f} RBS/km²")
                    
        ax.annotate(info_text, xy=(0.02, 0.96), xycoords='axes fraction',
                   fontsize=11, ha='left', va='top',
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=base_color, alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.15)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Coverage map by operator saved at {output_path}")

def create_overlap_map(gdf_rbs, gdf_sectors, output_path, operator_colors=OPERATOR_COLORS):
    """
    Creates a map showing the coverage overlap between operators.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS
        gdf_sectors: GeoDataFrame with the coverage sectors
        output_path: Path to save the map
        operator_colors: Dictionary of colors for each operator
    """
    print("Creating coverage overlap map...")
    
    try:
        # Reproject to Web Mercator
        gdf_rbs_3857 = gdf_rbs.to_crs(epsg=3857)
        gdf_sectors_3857 = gdf_sectors.to_crs(epsg=3857)
        
        # Get limits (bounding box)
        bbox = gdf_rbs_3857.total_bounds
        x_min, y_min, x_max, y_max = bbox
        
        # Define grid size for rasterization
        grid_size = 500
        
        # Define affine transformation for raster
        transform = from_bounds(x_min, y_min, x_max, y_max, grid_size, grid_size)
        
        # Initialize array of counts with zeros
        contagem_overlap = np.zeros((grid_size, grid_size), dtype=np.uint8)
        
        # For each operator, rasterize their coverage polygons and sum the results
        for operator in operator_colors.keys():
            subset = gdf_sectors_3857[gdf_sectors_3857['Operator'] == operator]
            if not subset.empty:
                shapes = [(geom, 1) for geom in subset.geometry if geom.is_valid]
                if shapes:
                    mask = rasterize(
                        shapes=shapes,
                        out_shape=(grid_size, grid_size),
                        transform=transform,
                        all_touched=True,
                        fill=0,
                        dtype=np.uint8
                    )
                    contagem_overlap += mask
        
        # Plot the result
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Plot the overlap matrix as an image
        im = ax.imshow(contagem_overlap, extent=[x_min, x_max, y_min, y_max],
                       cmap=plt.cm.viridis, origin='lower', alpha=0.7, interpolation='bilinear')
        
        # Optional: plot the boundaries of the sectors for each operator
        for operator in operator_colors.keys():
            subset = gdf_sectors_3857[gdf_sectors_3857['Operator'] == operator]
            if not subset.empty:
                subset.boundary.plot(ax=ax, color=operator_colors[operator], linewidth=1, alpha=0.8)
        
        # Plot the RBS
        for operator, cor in operator_colors.items():
            subset = gdf_rbs_3857[gdf_rbs_3857['Operator'] == operator]
            if not subset.empty:
                subset.plot(ax=ax, color=cor, markersize=20, edgecolor='white', linewidth=0.5)
        
        # Add base map
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        
        # Add cartographic elements
        add_cartographic_elements(
            ax, 
            'Coverage Overlap between Operators',
            "Overlapping Operators: CLARO, OI, TIM, VIVO"
        )
        
        # Add color bar
        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label('Number of Operators with Coverage', fontsize=12)
        cbar.set_ticks(range(5))
        cbar.set_ticklabels(['No coverage', '1 operator', '2 operators', '3 operators', '4 operators'])
        
        # Add legend
        create_custom_legend(ax, operator_colors, "Operators")
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Coverage overlap map saved at {output_path}")
        
    except Exception as e:
        print(f"Error creating coverage overlap map: {e}")
        plt.close('all')

def create_heat_map_power(gdf_rbs, output_path):
    """
    Creates a heat map showing the intensity of the power of the RBS.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS
        output_path: Path to save the map
    """
    print("Creating heat map of power...")
    
    # Reproject to Web Mercator
    gdf_rbs_3857 = gdf_rbs.to_crs(epsg=3857)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Get limits (bounding box) with margin
    bbox = gdf_rbs_3857.total_bounds
    x_min, y_min, x_max, y_max = bbox
    margin = 0.01
    x_min -= margin; y_min -= margin; x_max += margin; y_max += margin
    
    # Create grid for interpolation
    grid_size = 500
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    xi, yi = np.meshgrid(xi, yi)
    
    # Get points and values for interpolation
    points = np.array([(p.x, p.y) for p in gdf_rbs_3857.geometry])
    values = gdf_rbs_3857['EIRP_dBm'].values
    
    # Interpolate values on the grid
    grid_power = griddata(points, values, (xi, yi), method='cubic', fill_value=np.min(values))
    
    # Limit values for better visualization (5th and 95th percentiles)
    vmin = np.percentile(grid_power, 5)
    vmax = np.percentile(grid_power, 95)
    
    # Create custom color map
    cmap = LinearSegmentedColormap.from_list('power',
                ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1',
                 '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'], N=256)
    
    # Plot the interpolation as an image
    im = ax.imshow(grid_power, extent=[x_min, x_max, y_min, y_max],
                  origin='lower', cmap=cmap, alpha=0.8, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Add contours
    contours = ax.contour(xi, yi, grid_power, levels=5, colors='white', alpha=0.6, linewidths=0.8)
    plt.clabel(contours, inline=1, fontsize=8, fmt='%.1f dBm')
    
    # Plot the RBS
    scatter = ax.scatter(gdf_rbs_3857.geometry.x, gdf_rbs_3857.geometry.y,
                        c=gdf_rbs_3857['EIRP_dBm'], cmap=cmap, s=50, edgecolor='white',
                        linewidth=0.5, alpha=0.9, vmin=vmin, vmax=vmax)
    
    # Add base map
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    # Add cartographic elements
    add_cartographic_elements(ax, 'Effective Radiated Power (EIRP) of RBS')
    
    # Add color bar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('EIRP (dBm)', fontsize=12)
    
    # Add statistics
    eirp_min = gdf_rbs_3857['EIRP_dBm'].min()
    eirp_max = gdf_rbs_3857['EIRP_dBm'].max()
    eirp_media = gdf_rbs_3857['EIRP_dBm'].mean()
    
    info_text = (f"EIRP Average: {eirp_media:.1f} dBm\n"
                f"EIRP Minimum: {eirp_min:.1f} dBm\n"
                f"EIRP Maximum: {eirp_max:.1f} dBm")
                
    ax.annotate(info_text, xy=(0.02, 0.96), xycoords='axes fraction',
                fontsize=11, ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Heat map of power saved at {output_path}")

def create_folium_map(gdf_rbs, output_path):
    """
    Creates an interactive Folium map with the RBS.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS
        output_path: Path to save the HTML map
    """
    print("Creating interactive Folium map...")
    
    # Filter valid data
    geo_df = gdf_rbs.dropna(subset=['Latitude', 'Longitude'])
    geo_df = geo_df[(geo_df['Latitude'] != 0) & (geo_df['Longitude'] != 0)]
    
    if geo_df.empty:
        print("No valid data to generate Folium map.")
        return
    
    # Map center (average of coordinates)
    map_center = [geo_df['Latitude'].mean(), geo_df['Longitude'].mean()]
    
    # Create map
    m = folium.Map(location=map_center, zoom_start=10, tiles='CartoDB positron')
    
    # Add marker cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    # For each operator, use a different color
    cores_html = {
        'CLARO': 'red',
        'OI': 'orange',
        'VIVO': 'purple',
        'TIM': 'blue',
        'OUTRA': 'gray'
    }
    
    # Add markers for each RBS
    for idx, row in geo_df.iterrows():
        cor = cores_html.get(row.get('Operator', 'OUTRA'), 'gray')
        
        # Create pop-up text
        popup_text = f"""
        <b>Operator:</b> {row.get('NomeEntidade', 'N/A')}<br>
        <b>Technology:</b> {row.get('Tecnologia', 'N/A')}<br>
        <b>Tx Freq:</b> {row.get('FreqTxMHz', 'N/A')} MHz<br>
        <b>EIRP:</b> {row.get('EIRP_dBm', 'N/A')} dBm<br>
        <b>Radius:</b> {row.get('Coverage_Radius_km', 'N/A')} km<br>
        <b>Azimuth:</b> {row.get('Azimute', 'N/A')}°<br>
        <b>Height:</b> {row.get('AlturaAntena', 'N/A')} m<br>
        """
        
        # Create tooltip (text that appears when mouse hovers over)
        tooltip = f"{row.get('Operator', 'RBS')}: {row.get('Tecnologia', '')}"
        
        # Add marker
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=tooltip,
            icon=folium.Icon(color=cor, icon='signal', prefix='fa')
        ).add_to(marker_cluster)
    
    # Save map
    m.save(output_path)
    print(f"Interactive Folium map saved at: {output_path}")
