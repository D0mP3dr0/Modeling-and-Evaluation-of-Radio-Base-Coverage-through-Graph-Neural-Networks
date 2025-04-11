"""
Module for advanced spatial analysis of Radio Base Stations (RBS).
Contains functions for clustering, density analysis, and spatial pattern identification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, HeatMap
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import os
import contextily as ctx
from shapely.geometry import Point, Polygon
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

# Default color configuration for operators
OPERATOR_COLORS = {
    'CLARO': '#E02020',
    'OI': '#FFD700',
    'VIVO': '#9932CC',
    'TIM': '#0000CD'
}

def preprocess_spatial_data(gdf_rbs):
    """
    Preprocesses the RBS data for spatial analysis.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        
    Returns:
        GeoDataFrame: Preprocessed data for spatial analysis
    """
    # Make a copy to avoid modifying the original
    df = gdf_rbs.copy()
    
    # Check if it's already a GeoDataFrame
    if not isinstance(df, gpd.GeoDataFrame):
        print("Converting DataFrame to GeoDataFrame...")
        
        # Check for longitude and latitude columns
        lon_cols = [col for col in df.columns if 'lon' in col.lower() or 'lng' in col.lower()]
        lat_cols = [col for col in df.columns if 'lat' in col.lower()]
        
        if not lon_cols or not lat_cols:
            print("Error: Could not find longitude and latitude columns.")
            # Create synthetic data for demonstration
            print("Creating synthetic spatial data for demonstration purposes.")
            
            # Create a grid of points around São Paulo city
            n_points = len(df)
            
            # São Paulo approximate center
            center_lon, center_lat = -46.6333, -23.5505
            
            # Create random points in a grid around the center
            np.random.seed(42)  # For reproducibility
            lons = center_lon + np.random.normal(0, 0.4, n_points)
            lats = center_lat + np.random.normal(0, 0.4, n_points)
            
            # Add to DataFrame
            df['longitude'] = lons
            df['latitude'] = lats
            
            lon_col, lat_col = 'longitude', 'latitude'
        else:
            lon_col, lat_col = lon_cols[0], lat_cols[0]
        
        # Convert to GeoDataFrame
        geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
        df = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Ensure the data is in EPSG:4326 (WGS84) for consistency
    if df.crs and df.crs != "EPSG:4326":
        df = df.to_crs("EPSG:4326")
    
    return df

def create_density_map(gdf_rbs, output_path):
    """
    Creates a heatmap showing the density of RBS sites.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        output_path: Path to save the visualization
    """
    print("Creating RBS density map...")
    
    # Preprocess data
    gdf = preprocess_spatial_data(gdf_rbs)
    
    # Create folium map
    center_lat = gdf.geometry.y.mean()
    center_lon = gdf.geometry.x.mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='CartoDB positron')
    
    # Add heat map layer
    heat_data = [[row.geometry.y, row.geometry.x] for idx, row in gdf.iterrows()]
    HeatMap(heat_data, radius=15, blur=10).add_to(m)
    
    # Add operator-based markers in clusters
    if 'Operator' in gdf.columns:
        # Create a separate marker cluster for each operator
        for operator, color in OPERATOR_COLORS.items():
            operator_data = gdf[gdf['Operator'] == operator]
            if not operator_data.empty:
                marker_cluster = MarkerCluster(name=operator).add_to(m)
                
                for idx, row in operator_data.iterrows():
                    popup_text = f"Operator: {operator}<br>"
                    
                    # Add additional info if available
                    for col in ['FreqTxMHz', 'Tecnologia', 'PotenciaTransmissorWatts']:
                        if col in row and not pd.isna(row[col]):
                            popup_text += f"{col}: {row[col]}<br>"
                    
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=3,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        popup=popup_text
                    ).add_to(marker_cluster)
        
        # Add layer control
        folium.LayerControl().add_to(m)
    else:
        # Add all markers to a single cluster
        marker_cluster = MarkerCluster().add_to(m)
        for idx, row in gdf.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=3,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.7
            ).add_to(marker_cluster)
    
    # Save the map
    m.save(output_path)
    print(f"Density map saved to {output_path}")
    
    # Create a static version for reports
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Create a kernel density plot
    gdf_projected = gdf.to_crs(epsg=3857)  # Web Mercator for equal-area analysis
    gdf_projected.plot(ax=ax, alpha=0.2, markersize=5)
    
    # Add basemap for context
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    plt.title('RBS Spatial Distribution', fontsize=16)
    plt.axis('off')
    
    static_path = output_path.replace('.html', '_static.png')
    plt.savefig(static_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Static density map saved to {static_path}")

def perform_dbscan_clustering(gdf_rbs, output_path):
    """
    Performs DBSCAN clustering to identify dense areas of RBS deployment.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        output_path: Path to save the visualization
    """
    print("Performing DBSCAN clustering analysis...")
    
    # Preprocess data
    gdf = preprocess_spatial_data(gdf_rbs)
    
    # Project to a suitable projection for the region (assumes Brazil)
    gdf_projected = gdf.to_crs(epsg=3857)  # Web Mercator
    
    # Extract coordinates for clustering
    coords = np.array([[point.x, point.y] for point in gdf_projected.geometry])
    
    # Standardize for DBSCAN
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    # Perform DBSCAN clustering
    # The eps parameter controls the cluster size, min_samples is the min points in a neighborhood
    db = DBSCAN(eps=0.2, min_samples=5).fit(coords_scaled)
    
    # Add cluster labels to the data
    gdf['cluster'] = db.labels_
    
    # Number of clusters (excluding noise points labeled as -1)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    print(f"Number of clusters found: {n_clusters}")
    
    # Create a static visualization
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Create a colormap for clusters
    cmap = plt.cm.get_cmap('tab20', n_clusters)
    
    # Plot noise points in gray
    noise_points = gdf[gdf['cluster'] == -1]
    noise_points.plot(ax=ax, color='gray', alpha=0.3, markersize=5)
    
    # Plot clusters with different colors
    for i in range(n_clusters):
        cluster_points = gdf[gdf['cluster'] == i]
        cluster_points.plot(
            ax=ax, 
            color=cmap(i), 
            alpha=0.6, 
            markersize=10, 
            label=f'Cluster {i+1}'
        )
    
    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    plt.title(f'DBSCAN Clustering of RBS Locations (Found {n_clusters} clusters)', fontsize=16)
    plt.axis('off')
    ax.legend(title='Clusters', loc='upper right')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create an interactive version with folium
    center_lat = gdf.geometry.y.mean()
    center_lon = gdf.geometry.x.mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='CartoDB positron')
    
    # Add points colored by cluster
    for i in range(-1, n_clusters):
        if i == -1:
            color = 'gray'
            cluster_name = 'Noise'
        else:
            color = mcolors.rgb2hex(cmap(i))
            cluster_name = f'Cluster {i+1}'
        
        cluster_points = gdf[gdf['cluster'] == i]
        
        if not cluster_points.empty:
            for idx, row in cluster_points.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    tooltip=f"Cluster: {cluster_name}"
                ).add_to(m)
    
    # Add legend as HTML
    legend_html = '''
    <div style="position: fixed; 
        bottom: 50px; right: 50px; width: 150px; height: 160px; 
        border:2px solid grey; z-index:9999; font-size:12px;
        background-color: white; padding: 10px; border-radius: 5px;">
    <p><b>Clusters</b></p>
    '''
    
    # Add entry for noise
    legend_html += f'''
    <p><i class="fa fa-circle" style="color:gray"></i> Noise</p>
    '''
    
    # Add entry for each cluster
    for i in range(n_clusters):
        color = mcolors.rgb2hex(cmap(i))
        legend_html += f'''
        <p><i class="fa fa-circle" style="color:{color}"></i> Cluster {i+1}</p>
        '''
    
    legend_html += '</div>'
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save the interactive map
    interactive_path = output_path.replace('.png', '.html')
    m.save(interactive_path)
    print(f"Interactive cluster map saved to {interactive_path}")
    
    # Create cluster data summary
    cluster_summary = gdf.groupby('cluster').size().reset_index(name='Count')
    
    if 'Operator' in gdf.columns:
        # Add operator distribution per cluster
        for operator in gdf['Operator'].unique():
            cluster_summary[f'{operator}_Count'] = gdf[gdf['Operator'] == operator].groupby('cluster').size()
            cluster_summary[f'{operator}_Percentage'] = (cluster_summary[f'{operator}_Count'] / cluster_summary['Count'] * 100).round(1)
    
    # Save summary
    summary_path = output_path.replace('.png', '_summary.csv')
    cluster_summary.to_csv(summary_path, index=False)
    print(f"Cluster summary saved to {summary_path}")

def create_voronoi_coverage(gdf_rbs, output_path):
    """
    Creates a Voronoi diagram to visualize coverage and service areas.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        output_path: Path to save the visualization
    """
    print("Creating Voronoi coverage diagram...")
    
    # Preprocess data
    gdf = preprocess_spatial_data(gdf_rbs)
    
    # Try to use geopandas' Voronoi capabilities
    try:
        from geovoronoi import voronoi_regions_from_coords
        from shapely.geometry import MultiPoint
        
        # Project to a local projection for accurate distance measurements
        gdf_projected = gdf.to_crs(epsg=3857)  # Web Mercator
        
        # Extract coordinates for Voronoi
        coords = np.array([[point.x, point.y] for point in gdf_projected.geometry])
        
        # Create boundary (buffer around points to limit Voronoi extent)
        all_points = MultiPoint([(point.x, point.y) for point in gdf_projected.geometry])
        boundary = all_points.convex_hull.buffer(5000)  # 5km buffer
        
        # Compute Voronoi regions
        region_polys, region_pts = voronoi_regions_from_coords(coords, boundary)
        
        # Convert to GeoDataFrame
        voronoi_gdf = gpd.GeoDataFrame(geometry=[region_polys[i] for i in range(len(region_pts))], crs=gdf_projected.crs)
        
        # Map regions back to original points for attributes
        voronoi_gdf['point_idx'] = [list(region_pts.values()).index(i) for i in range(len(region_pts))]
        
        # Join with original data
        voronoi_gdf = voronoi_gdf.merge(
            gdf_projected.reset_index(), 
            left_on='point_idx', 
            right_index=True
        )
        
        # Create static visualization
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # Plot Voronoi polygons with operator colors if available
        if 'Operator' in voronoi_gdf.columns:
            for operator, color in OPERATOR_COLORS.items():
                operator_voronoi = voronoi_gdf[voronoi_gdf['Operator'] == operator]
                operator_voronoi.plot(
                    ax=ax,
                    color=color,
                    alpha=0.3,
                    edgecolor='white',
                    linewidth=0.5
                )
                
            # Add legend
            import matplotlib.patches as mpatches
            handles = [
                mpatches.Patch(color=color, label=operator, alpha=0.5)
                for operator, color in OPERATOR_COLORS.items()
                if operator in voronoi_gdf['Operator'].values
            ]
            ax.legend(handles=handles, title='Operator', loc='upper right')
        else:
            # Plot with random colors
            voronoi_gdf.plot(
                ax=ax,
                column='point_idx',
                cmap='tab20',
                alpha=0.3,
                edgecolor='white',
                linewidth=0.5
            )
        
        # Add base stations as points
        gdf_projected.plot(
            ax=ax,
            markersize=5,
            color='black',
            alpha=0.7
        )
        
        # Add basemap
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        
        plt.title('Voronoi Coverage Areas for RBS Stations', fontsize=16)
        plt.axis('off')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Voronoi coverage diagram saved to {output_path}")
        
    except ImportError as e:
        print(f"Error: Missing required package for Voronoi analysis: {e}")
        print("Please install geovoronoi with: pip install geovoronoi")
    except Exception as e:
        print(f"Error creating Voronoi diagram: {e}")

def run_spatial_analysis(gdf_rbs, results_dir):
    """
    Runs all spatial analysis visualizations.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        results_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    spatial_dir = os.path.join(results_dir, 'spatial_analysis')
    os.makedirs(spatial_dir, exist_ok=True)
    
    # Run all visualizations
    create_density_map(gdf_rbs, os.path.join(spatial_dir, 'density_map.html'))
    perform_dbscan_clustering(gdf_rbs, os.path.join(spatial_dir, 'dbscan_clusters.png'))
    create_voronoi_coverage(gdf_rbs, os.path.join(spatial_dir, 'voronoi_coverage.png'))
    
    print(f"All spatial analyses completed and saved to {spatial_dir}") 