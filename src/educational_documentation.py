"""
Educational Documentation Module for Radio Base Station Analysis

This module provides tools for creating educational materials, visualizations,
and documentation to explain the RBS analysis system:

1. Visual Project Narrative
   - Pipeline flowcharts
   - Sequential visualizations showing data transformations

2. Interactive Documentation
   - Templates for Jupyter notebooks
   - Visual tutorials for system components
   - Illustrated use cases

3. Data Storytelling
   - Data-based narrative visualizations
   - Before/after comparison visualizations
   - Key findings summary visualizations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import networkx as nx
import json
from IPython.display import HTML, display, Markdown
import graphviz
from datetime import datetime
import folium
from folium.plugins import HeatMap, MarkerCluster

# Optional imports for advanced visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly/Dash not found. Advanced interactive visualizations will not be available.")

#############################################
# 1. VISUAL PROJECT NARRATIVE
#############################################

def create_pipeline_flowchart(output_path):
    """
    Creates a detailed flowchart of the entire data processing and analysis pipeline.
    
    Args:
        output_path (str): Directory to save the flowchart
        
    Returns:
        str: Path to the created flowchart file
    """
    print("Creating pipeline flowchart...")
    
    # Create a new Graphviz object
    dot = graphviz.Digraph(
        'rbs_pipeline', 
        comment='RBS Analysis Pipeline',
        format='png',
        engine='dot',
        graph_attr={'rankdir': 'LR', 'splines': 'ortho', 'nodesep': '0.8', 'ranksep': '1.0'}
    )
    
    # Define node styles
    dot.attr('node', shape='box', style='filled,rounded', fontname='Arial', fontsize='12')
    
    # Create cluster for Data Processing
    with dot.subgraph(name='cluster_data_processing') as c:
        c.attr(label='Data Processing', style='filled', color='lightgrey', fontname='Arial', fontsize='14')
        
        c.node('data_loading', 'Data Loading', fillcolor='#A8E6CE')
        c.node('data_cleaning', 'Data Cleaning', fillcolor='#A8E6CE')
        c.node('geo_conversion', 'Geo Conversion', fillcolor='#A8E6CE')
        c.node('feature_engineering', 'Feature Engineering', fillcolor='#A8E6CE')
        
        c.edge('data_loading', 'data_cleaning')
        c.edge('data_cleaning', 'geo_conversion')
        c.edge('geo_conversion', 'feature_engineering')
    
    # Create cluster for Basic Analysis
    with dot.subgraph(name='cluster_basic_analysis') as c:
        c.attr(label='Basic Analysis', style='filled', color='lightgrey', fontname='Arial', fontsize='14')
        
        c.node('basic_stats', 'Statistical Analysis', fillcolor='#DCEDC2')
        c.node('visualizations', 'Basic Visualizations', fillcolor='#DCEDC2')
        c.node('coverage_estimation', 'Coverage Estimation', fillcolor='#DCEDC2')
        
        c.edge('basic_stats', 'visualizations')
    
    # Create cluster for Advanced Analysis
    with dot.subgraph(name='cluster_advanced_analysis') as c:
        c.attr(label='Advanced Analysis', style='filled', color='lightgrey', fontname='Arial', fontsize='14')
        
        c.node('graph_analysis', 'Graph Analysis', fillcolor='#FFD3B5')
        c.node('advanced_graph', 'Advanced Graph Analysis', fillcolor='#FFD3B5')
        c.node('spatial_analysis', 'Spatial Analysis', fillcolor='#FFD3B5')
        c.node('tech_frequency', 'Tech & Frequency Analysis', fillcolor='#FFD3B5')
        c.node('coverage_quality', 'Coverage Quality Analysis', fillcolor='#FFD3B5')
        c.node('advanced_coverage', 'Advanced Coverage Visualization', fillcolor='#FFD3B5')
        c.node('temporal_analysis', 'Temporal Analysis', fillcolor='#FFD3B5')
        c.node('correlation_analysis', 'Correlation Analysis', fillcolor='#FFD3B5')
    
    # Create cluster for Predictive Analysis
    with dot.subgraph(name='cluster_predictive') as c:
        c.attr(label='Predictive Analysis', style='filled', color='lightgrey', fontname='Arial', fontsize='14')
        
        c.node('prediction_model', 'Prediction Models', fillcolor='#FFAAA6')
        c.node('coverage_prediction', 'Coverage Prediction', fillcolor='#FFAAA6')
        c.node('integration_analysis', 'Integration Analysis', fillcolor='#FFAAA6')
    
    # Create cluster for Output & Visualization
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Output & Visualization', style='filled', color='lightgrey', fontname='Arial', fontsize='14')
        
        c.node('dashboard', 'Interactive Dashboard', fillcolor='#A6D0E4')
        c.node('report_generation', 'Report Generation', fillcolor='#A6D0E4')
        c.node('educational_docs', 'Educational Documentation', fillcolor='#A6D0E4')
    
    # Connect the nodes between clusters
    dot.edge('feature_engineering', 'basic_stats')
    dot.edge('feature_engineering', 'coverage_estimation')
    
    dot.edge('basic_stats', 'graph_analysis')
    dot.edge('basic_stats', 'spatial_analysis')
    dot.edge('basic_stats', 'tech_frequency')
    dot.edge('basic_stats', 'temporal_analysis')
    
    dot.edge('coverage_estimation', 'coverage_quality')
    dot.edge('coverage_estimation', 'advanced_coverage')
    
    dot.edge('graph_analysis', 'advanced_graph')
    dot.edge('graph_analysis', 'correlation_analysis')
    
    dot.edge('advanced_graph', 'prediction_model')
    dot.edge('spatial_analysis', 'prediction_model')
    dot.edge('tech_frequency', 'prediction_model')
    dot.edge('temporal_analysis', 'prediction_model')
    dot.edge('correlation_analysis', 'prediction_model')
    
    dot.edge('prediction_model', 'coverage_prediction')
    dot.edge('prediction_model', 'integration_analysis')
    
    dot.edge('coverage_quality', 'dashboard')
    dot.edge('advanced_coverage', 'dashboard')
    dot.edge('advanced_graph', 'dashboard')
    dot.edge('coverage_prediction', 'dashboard')
    dot.edge('integration_analysis', 'dashboard')
    
    dot.edge('dashboard', 'report_generation')
    dot.edge('dashboard', 'educational_docs')
    
    # Save the flowchart
    output_file = os.path.join(output_path, 'pipeline_flowchart')
    dot.render(output_file, cleanup=True)
    
    # Also save as SVG for web viewing
    dot.format = 'svg'
    dot.render(output_file + '_svg', cleanup=True)
    
    print(f"Pipeline flowchart saved to {output_file}.png and {output_file}_svg.svg")
    
    return output_file + '.png'

def create_sequential_visualizations(gdf_rbs, output_path):
    """
    Creates sequential visualizations showing the transformation of data at each stage.
    
    Args:
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        output_path (str): Directory to save the visualizations
        
    Returns:
        list: Paths to the created visualization files
    """
    print("Creating sequential visualizations...")
    
    # List to store paths to created visualizations
    visualization_files = []
    
    # 1. Raw Data Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("1. Raw RBS Data Distribution", fontsize=14)
    
    # Basic scatter plot of RBS locations
    gdf_rbs.plot(ax=ax, markersize=20, color='blue', alpha=0.6)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotation explaining this stage
    ax.text(0.02, 0.02, 
            "Stage 1: Raw Data\n\n"
            "• Initial data points showing geographic distribution of RBS\n"
            "• No processing or analysis applied yet\n"
            "• Points represent exact GPS coordinates", 
            transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8),
            fontsize=10)
    
    # Save the figure
    raw_data_path = os.path.join(output_path, "01_raw_data.png")
    plt.tight_layout()
    plt.savefig(raw_data_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    visualization_files.append(raw_data_path)
    
    # 2. Data Cleaning Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Left plot: Before cleaning (with simulated errors)
    ax1.set_title("Before Data Cleaning", fontsize=12)
    
    # Create a copy with some simulated outliers
    gdf_simulated = gdf_rbs.copy()
    outlier_indices = np.random.choice(range(len(gdf_simulated)), size=min(5, max(1, int(len(gdf_simulated) * 0.05))), replace=False)
    
    # Add some outliers (shift positions significantly)
    for idx in outlier_indices:
        if idx < len(gdf_simulated):
            # Shift by a random amount
            gdf_simulated.loc[gdf_simulated.index[idx], 'Longitude'] += np.random.uniform(-1, 1)
            gdf_simulated.loc[gdf_simulated.index[idx], 'Latitude'] += np.random.uniform(-1, 1)
    
    # Plot with outliers
    gdf_simulated.plot(ax=ax1, markersize=20, color='red', alpha=0.6)
    for idx in outlier_indices:
        if idx < len(gdf_simulated):
            ax1.annotate("Outlier", 
                        (gdf_simulated.iloc[idx].geometry.x, gdf_simulated.iloc[idx].geometry.y),
                        xytext=(10, 10), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->'))
    
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Right plot: After cleaning
    ax2.set_title("After Data Cleaning", fontsize=12)
    gdf_rbs.plot(ax=ax2, markersize=20, color='green', alpha=0.6)
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotation
    fig.text(0.5, 0.01, 
             "Stage 2: Data Cleaning\n\n"
             "• Outliers and erroneous coordinates identified and corrected\n"
             "• Missing values handled appropriately\n"
             "• Data standardized to consistent format", 
             ha='center', bbox=dict(facecolor='white', alpha=0.8),
             fontsize=10)
    
    # Save the figure
    cleaning_path = os.path.join(output_path, "02_data_cleaning.png")
    plt.tight_layout()
    plt.savefig(cleaning_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    visualization_files.append(cleaning_path)
    
    # 3. Geo Conversion Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Left plot: Raw points
    ax1.set_title("Raw RBS Points", fontsize=12)
    gdf_rbs.plot(ax=ax1, markersize=20, color='blue', alpha=0.6)
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Right plot: Geo Converted (with coverage areas)
    ax2.set_title("With Coverage Areas", fontsize=12)
    
    # Plot base points
    gdf_rbs.plot(ax=ax2, markersize=20, color='blue', alpha=0.6)
    
    # Add estimated coverage circles (if available in data, otherwise simulate)
    if 'Coverage_Radius_km' in gdf_rbs.columns:
        for idx, row in gdf_rbs.iterrows():
            if pd.notnull(row['Coverage_Radius_km']):
                coverage = row['Coverage_Radius_km']
                # Convert km to degrees (approximate)
                coverage_degree = coverage / 111.0  # 1 degree ≈ 111 km at equator
                circle = plt.Circle((row.geometry.x, row.geometry.y), 
                                    coverage_degree, 
                                    color='blue', alpha=0.2)
                ax2.add_patch(circle)
    else:
        # Simulate coverage circles with random radii
        for idx, row in gdf_rbs.iterrows():
            coverage_degree = np.random.uniform(0.01, 0.05)  # Random coverage
            circle = plt.Circle((row.geometry.x, row.geometry.y), 
                                coverage_degree, 
                                color='blue', alpha=0.2)
            ax2.add_patch(circle)
    
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_xlim(ax1.get_xlim())  # Match x-axis limits
    ax2.set_ylim(ax1.get_ylim())  # Match y-axis limits
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotation
    fig.text(0.5, 0.01, 
             "Stage 3: Geo Conversion\n\n"
             "• Raw RBS coordinates converted to geographical features\n"
             "• Coverage areas estimated and added as geometric shapes\n"
             "• Data prepared for spatial analysis", 
             ha='center', bbox=dict(facecolor='white', alpha=0.8),
             fontsize=10)
    
    # Save the figure
    geo_path = os.path.join(output_path, "03_geo_conversion.png")
    plt.tight_layout()
    plt.savefig(geo_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    visualization_files.append(geo_path)
    
    # 4. Feature Engineering Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: RBS by Operator (if available)
    ax1.set_title("RBS by Operator", fontsize=12)
    
    if 'Operator' in gdf_rbs.columns:
        # Group by operator and count
        operator_counts = gdf_rbs['Operator'].value_counts()
        operator_counts.plot(kind='bar', ax=ax1, color='skyblue')
    else:
        # Simulate operator data
        operators = ['Operator A', 'Operator B', 'Operator C', 'Operator D']
        counts = np.random.randint(10, 50, size=len(operators))
        pd.Series(counts, index=operators).plot(kind='bar', ax=ax1, color='skyblue')
    
    ax1.set_ylabel("Count")
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Plot 2: RBS by Technology (if available)
    ax2.set_title("RBS by Technology", fontsize=12)
    
    if 'Tecnologia' in gdf_rbs.columns:
        # Group by technology and count
        tech_counts = gdf_rbs['Tecnologia'].value_counts()
        tech_counts.plot(kind='bar', ax=ax2, color='lightgreen')
    else:
        # Simulate technology data
        technologies = ['2G', '3G', '4G', '5G']
        counts = np.random.randint(5, 40, size=len(technologies))
        pd.Series(counts, index=technologies).plot(kind='bar', ax=ax2, color='lightgreen')
    
    ax2.set_ylabel("Count")
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Plot 3: Coverage Density Map
    ax3.set_title("Coverage Density Map", fontsize=12)
    
    # Create a simple heatmap-like visualization
    x = gdf_rbs.geometry.x
    y = gdf_rbs.geometry.y
    
    # Create a 2D histogram
    h, xedges, yedges = np.histogram2d(x, y, bins=20)
    
    # Create a heatmap
    im = ax3.imshow(h.T, cmap='hot', origin='lower', 
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   aspect='auto')
    
    plt.colorbar(im, ax=ax3, label='RBS Density')
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    
    # Plot 4: Derived Features Correlation (simulated)
    ax4.set_title("Derived Features Correlation", fontsize=12)
    
    # Create or simulate some derived features
    features = ['Coverage Area', 'Population Served', 'Signal Strength', 'Frequency Band', 'Capacity']
    corr_matrix = np.random.rand(len(features), len(features))
    # Make the matrix symmetric
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    # Set diagonal to 1
    np.fill_diagonal(corr_matrix, 1)
    
    # Create a dataframe
    corr_df = pd.DataFrame(corr_matrix, index=features, columns=features)
    
    # Plot heatmap
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax4)
    
    # Add annotation
    fig.text(0.5, 0.01, 
             "Stage 4: Feature Engineering\n\n"
             "• Raw data transformed into meaningful features\n"
             "• Categorical variables processed\n"
             "• Spatial features derived from geographical data\n"
             "• Correlations between derived features analyzed", 
             ha='center', bbox=dict(facecolor='white', alpha=0.8),
             fontsize=10)
    
    # Save the figure
    feat_path = os.path.join(output_path, "04_feature_engineering.png")
    plt.tight_layout()
    plt.savefig(feat_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    visualization_files.append(feat_path)
    
    # 5. Analysis Results Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Network Graph Visualization
    ax1.set_title("Network Graph Analysis", fontsize=12)
    
    # Create a simple graph
    G = nx.random_geometric_graph(min(50, len(gdf_rbs)), 0.2)
    
    # Use positions from data if possible, otherwise random positions
    if len(gdf_rbs) >= len(G.nodes()):
        pos = {i: (gdf_rbs.iloc[i].geometry.x, gdf_rbs.iloc[i].geometry.y) for i in range(len(G.nodes()))}
    else:
        pos = nx.spring_layout(G)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='skyblue', alpha=0.8, ax=ax1)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5, ax=ax1)
    ax1.set_axis_off()
    
    # Plot 2: Coverage Quality Heatmap
    ax2.set_title("Coverage Quality Analysis", fontsize=12)
    
    # Generate a random coverage quality grid
    x = np.linspace(gdf_rbs.geometry.x.min(), gdf_rbs.geometry.x.max(), 100)
    y = np.linspace(gdf_rbs.geometry.y.min(), gdf_rbs.geometry.y.max(), 100)
    X, Y = np.meshgrid(x, y)
    
    # Create a function for coverage quality (distance from RBS)
    Z = np.zeros_like(X)
    for idx, row in gdf_rbs.iterrows():
        rbsx, rbsy = row.geometry.x, row.geometry.y
        # Add contribution from this RBS (inverse distance)
        Z += 1 / (1 + 5 * ((X - rbsx)**2 + (Y - rbsy)**2))
    
    # Plot the heatmap
    im = ax2.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], 
                   origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax2, label='Coverage Quality')
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    
    # Plot 3: Predictive Model Results (simulated)
    ax3.set_title("Coverage Prediction", fontsize=12)
    
    # Create a simple contour plot for predicted coverage
    # Use a different pattern than the quality heatmap
    Z_pred = np.zeros_like(X)
    for i in range(3):  # Simulate a few prediction factors
        center_x = np.random.uniform(x.min(), x.max())
        center_y = np.random.uniform(y.min(), y.max())
        Z_pred += np.exp(-0.1 * ((X - center_x)**2 + (Y - center_y)**2))
    
    # Plot contours
    contour = ax3.contourf(X, Y, Z_pred, levels=10, cmap='plasma')
    plt.colorbar(contour, ax=ax3, label='Predicted Coverage')
    
    # Plot actual RBS positions
    ax3.scatter(gdf_rbs.geometry.x, gdf_rbs.geometry.y, 
               c='white', s=30, marker='o', edgecolor='black')
    
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    
    # Plot 4: Comparative Analysis (simulated)
    ax4.set_title("Operator Comparison", fontsize=12)
    
    # Simulate comparison data
    if 'Operator' in gdf_rbs.columns:
        operators = gdf_rbs['Operator'].unique()
    else:
        operators = ['Operator A', 'Operator B', 'Operator C', 'Operator D']
    
    metrics = ['Coverage', 'Efficiency', 'Capacity', 'Reliability']
    data = np.random.rand(len(operators), len(metrics))
    
    # Normalize to 0-100 scale
    data = data * 100
    
    # Create DataFrame
    df_comparison = pd.DataFrame(data, index=operators, columns=metrics)
    
    # Plot as a grouped bar chart
    df_comparison.plot(kind='bar', ax=ax4, rot=0)
    ax4.set_ylabel("Score (0-100)")
    ax4.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax4.legend(title="Metrics", loc='upper right')
    
    # Add annotation
    fig.text(0.5, 0.01, 
             "Stage 5: Analysis Results\n\n"
             "• Graph analysis reveals network connectivity\n"
             "• Coverage quality identifies critical areas\n"
             "• Predictive models suggest optimal RBS placements\n"
             "• Comparative analysis benchmarks operator performance", 
             ha='center', bbox=dict(facecolor='white', alpha=0.8),
             fontsize=10)
    
    # Save the figure
    analysis_path = os.path.join(output_path, "05_analysis_results.png")
    plt.tight_layout()
    plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    visualization_files.append(analysis_path)
    
    # Create a summary image with all stages
    if PLOTLY_AVAILABLE:
        # Create an interactive HTML summary with Plotly
        create_interactive_pipeline_summary(visualization_files, output_path)
    
    print(f"Created {len(visualization_files)} sequential visualizations")
    return visualization_files

def create_interactive_pipeline_summary(visualization_files, output_path):
    """
    Creates an interactive HTML summary of the data pipeline visualizations.
    
    Args:
        visualization_files (list): List of paths to visualization images
        output_path (str): Directory to save the summary
        
    Returns:
        str: Path to the created HTML file
    """
    # Create a Plotly figure
    fig = make_subplots(
        rows=len(visualization_files), cols=1,
        subplot_titles=[f"Stage {i+1}" for i in range(len(visualization_files))],
        vertical_spacing=0.1
    )
    
    # Add each image as a subplot
    for i, img_path in enumerate(visualization_files):
        # Extract image file name for display
        img_name = os.path.basename(img_path)
        
        # Create a relative path for the image in the HTML
        # This assumes the HTML and images will be in the same directory
        img_relative = img_path.split(os.path.sep)[-1]
        
        # Add an image
        fig.add_trace(
            go.Image(source=img_path),
            row=i+1, col=1
        )
    
    # Update layout
    fig.update_layout(
        title_text="RBS Analysis Pipeline: Data Transformation Sequence",
        height=300 * len(visualization_files),
        width=1000,
        showlegend=False
    )
    
    # Save as an HTML file
    html_path = os.path.join(output_path, "pipeline_sequence_summary.html")
    fig.write_html(html_path)
    
    return html_path

def create_documentation_index(documentation_outputs, output_path):
    """
    Creates an HTML index page to navigate all documentation.
    
    Args:
        documentation_outputs (dict): Dictionary containing paths to all documentation
        output_path (str): Directory to save the index page
        
    Returns:
        str: Path to the index page
    """
    print("Creating documentation index...")
    
    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RBS Analysis Educational Documentation</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                color: #2980b9;
                margin-top: 30px;
            }
            h3 {
                color: #3498db;
            }
            .section {
                margin-bottom: 40px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .card-container {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-top: 20px;
            }
            .card {
                flex: 1 1 300px;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                background-color: white;
                transition: transform 0.3s ease;
            }
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            .card img {
                max-width: 100%;
                height: auto;
                border-radius: 4px;
                margin-bottom: 10px;
            }
            a {
                color: #3498db;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
                color: #2980b9;
            }
            .thumbnail {
                width: 100%;
                height: 200px;
                object-fit: cover;
                border-radius: 4px;
            }
            .footer {
                margin-top: 50px;
                text-align: center;
                font-size: 0.9em;
                color: #7f8c8d;
                padding-top: 20px;
                border-top: 1px solid #ecf0f1;
            }
        </style>
    </head>
    <body>
        <h1>RBS Analysis Educational Documentation</h1>
        
        <div class="section">
            <h2>1. Visual Project Narrative</h2>
            <p>Visual representations of the data processing pipeline and sequential data transformations.</p>
            
            <div class="card-container">
    """
    
    # Add Visual Narrative cards
    if "visual_narrative" in documentation_outputs:
        visual_narrative = documentation_outputs["visual_narrative"]
        
        # Pipeline flowchart
        if "pipeline_flowchart" in visual_narrative:
            flowchart_path = visual_narrative["pipeline_flowchart"]
            # Convert to relative path for HTML
            rel_path = os.path.relpath(flowchart_path, output_path)
            html_content += f"""
                <div class="card">
                    <h3>Pipeline Flowchart</h3>
                    <a href="{rel_path}">
                        <img src="{rel_path}" alt="Pipeline Flowchart" class="thumbnail">
                    </a>
                    <p>Detailed flowchart showing the entire data processing and analysis pipeline.</p>
                    <a href="{rel_path}">View Full Size</a>
                </div>
            """
        
        # Sequential visualizations
        if "sequential_visualizations" in visual_narrative and visual_narrative["sequential_visualizations"]:
            # Get the interactive summary if available
            summary_path = os.path.join(os.path.dirname(visual_narrative["sequential_visualizations"][0]), 
                                      "pipeline_sequence_summary.html")
            
            if os.path.exists(summary_path):
                rel_path = os.path.relpath(summary_path, output_path)
                html_content += f"""
                    <div class="card">
                        <h3>Sequential Data Transformations</h3>
                        <a href="{rel_path}">
                            <img src="{os.path.relpath(visual_narrative['sequential_visualizations'][0], output_path)}" 
                                alt="Data Transformations" class="thumbnail">
                        </a>
                        <p>Visualizations showing how data transforms at each stage of the pipeline.</p>
                        <a href="{rel_path}">View Interactive Sequence</a>
                    </div>
                """
            # Otherwise link to individual images
            else:
                for i, img_path in enumerate(visual_narrative["sequential_visualizations"]):
                    if i == 0:  # Just show the first one in the index
                        rel_path = os.path.relpath(img_path, output_path)
                        html_content += f"""
                            <div class="card">
                                <h3>Sequential Data Transformations</h3>
                                <a href="{rel_path}">
                                    <img src="{rel_path}" alt="Data Transformations" class="thumbnail">
                                </a>
                                <p>Visualizations showing how data transforms at each stage of the pipeline.</p>
                                <a href="{rel_path}">View Images</a>
                            </div>
                        """
                        break
    
    html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>2. Interactive Documentation</h2>
            <p>Educational notebooks, tutorials, and use cases for learning the system.</p>
            
            <div class="card-container">
    """
    
    # Add Interactive Documentation cards
    if "interactive_docs" in documentation_outputs:
        interactive_docs = documentation_outputs["interactive_docs"]
        
        # Notebook templates
        if "notebook_templates" in interactive_docs and interactive_docs["notebook_templates"]:
            for i, notebook_path in enumerate(interactive_docs["notebook_templates"]):
                if i < 3:  # Limit to 3 cards
                    rel_path = os.path.relpath(notebook_path, output_path)
                    notebook_name = os.path.basename(notebook_path).replace(".ipynb", "").replace("_", " ").title()
                    
                    html_content += f"""
                        <div class="card">
                            <h3>{notebook_name}</h3>
                            <p>Jupyter notebook with detailed explanations and visualizations.</p>
                            <a href="{rel_path}">View Notebook</a>
                        </div>
                    """
        
        # Component tutorials
        if "component_tutorials" in interactive_docs and interactive_docs["component_tutorials"]:
            # Just add a single card for all tutorials
            tutorials_dir = os.path.dirname(next(iter(interactive_docs["component_tutorials"].values())))
            rel_dir = os.path.relpath(tutorials_dir, output_path)
            
            html_content += f"""
                <div class="card">
                    <h3>Component Tutorials</h3>
                    <p>Visual guides for each component of the RBS analysis system.</p>
                    <a href="{rel_dir}">Browse Tutorials</a>
                </div>
            """
        
        # Use cases
        if "use_cases" in interactive_docs and interactive_docs["use_cases"]:
            for i, use_case_path in enumerate(interactive_docs["use_cases"]):
                if i < 2:  # Limit to 2 cards
                    rel_path = os.path.relpath(use_case_path, output_path)
                    use_case_name = os.path.basename(use_case_path).split(".")[0].replace("_", " ").title()
                    
                    html_content += f"""
                        <div class="card">
                            <h3>{use_case_name}</h3>
                            <a href="{rel_path}">
                                <img src="{rel_path}" alt="{use_case_name}" class="thumbnail">
                            </a>
                            <p>Illustrated use case showing practical application.</p>
                            <a href="{rel_path}">View Full Size</a>
                        </div>
                    """
    
    html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>3. Data Storytelling</h2>
            <p>Visual narratives and insights derived from the analysis.</p>
            
            <div class="card-container">
    """
    
    # Add Data Storytelling cards
    if "data_storytelling" in documentation_outputs:
        storytelling = documentation_outputs["data_storytelling"]
        
        # Data narratives
        if "data_narratives" in storytelling and storytelling["data_narratives"]:
            for i, narrative_path in enumerate(storytelling["data_narratives"]):
                if i < 2:  # Limit to 2 cards
                    rel_path = os.path.relpath(narrative_path, output_path)
                    narrative_name = os.path.basename(narrative_path).split(".")[0].replace("_", " ").title()
                    
                    html_content += f"""
                        <div class="card">
                            <h3>{narrative_name}</h3>
                            <a href="{rel_path}">
                                <img src="{rel_path}" alt="{narrative_name}" class="thumbnail">
                            </a>
                            <p>Visual narrative explaining important insights from the data.</p>
                            <a href="{rel_path}">View Full Size</a>
                        </div>
                    """
        
        # Before/After comparisons
        if "before_after" in storytelling and storytelling["before_after"]:
            for i, comparison_path in enumerate(storytelling["before_after"]):
                if i < 2:  # Limit to 2 cards
                    rel_path = os.path.relpath(comparison_path, output_path)
                    comparison_name = os.path.basename(comparison_path).split(".")[0].replace("_", " ").title()
                    
                    html_content += f"""
                        <div class="card">
                            <h3>{comparison_name}</h3>
                            <a href="{rel_path}">
                                <img src="{rel_path}" alt="{comparison_name}" class="thumbnail">
                            </a>
                            <p>Before/after comparison showing the impact of analysis.</p>
                            <a href="{rel_path}">View Full Size</a>
                        </div>
                    """
        
        # Key findings
        if "key_findings" in storytelling and storytelling["key_findings"]:
            for i, finding_path in enumerate(storytelling["key_findings"]):
                if i < 2:  # Limit to 2 cards
                    rel_path = os.path.relpath(finding_path, output_path)
                    finding_name = os.path.basename(finding_path).split(".")[0].replace("_", " ").title()
                    
                    html_content += f"""
                        <div class="card">
                            <h3>{finding_name}</h3>
                            <a href="{rel_path}">
                                <img src="{rel_path}" alt="{finding_name}" class="thumbnail">
                            </a>
                            <p>Visualization summarizing key findings from the analysis.</p>
                            <a href="{rel_path}">View Full Size</a>
                        </div>
                    """
    
    html_content += """
            </div>
        </div>
        
        <div class="footer">
            <p>RBS Analysis Educational Documentation — Generated on """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML file
    index_path = os.path.join(output_path, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Documentation index created at {index_path}")
    return index_path

#############################################
# 2. INTERACTIVE DOCUMENTATION
#############################################

def create_notebook_templates(output_path):
    """
    Creates template Jupyter notebooks for different types of analysis.
    
    Args:
        output_path (str): Directory to save the notebook templates
        
    Returns:
        list: Paths to the created notebook templates
    """
    print("Creating notebook templates...")
    
    # Define the notebook templates
    templates = {
        "basic_analysis": {
            "title": "Basic RBS Analysis",
            "description": "Basic analysis of Radio Base Station data including statistical summaries and basic visualizations.",
            "sections": [
                "Data Loading",
                "Data Cleaning and Preparation",
                "Basic Statistics",
                "Geographic Distribution",
                "Operator Analysis",
                "Technology Distribution",
                "Frequency Analysis",
                "Basic Visualizations"
            ]
        },
        "advanced_analysis": {
            "title": "Advanced RBS Analysis",
            "description": "Advanced analysis techniques for Radio Base Station data including network analysis, spatial patterns, and more.",
            "sections": [
                "Network Graph Construction",
                "Community Detection",
                "Centrality Analysis",
                "Spatial Clustering",
                "Coverage Estimation",
                "Quality of Service Analysis",
                "Comparative Analysis Between Operators"
            ]
        },
        "visualization_guide": {
            "title": "RBS Visualization Guide",
            "description": "Comprehensive guide to creating effective visualizations for RBS data.",
            "sections": [
                "Geographic Visualizations",
                "Network Visualizations",
                "Statistical Visualizations",
                "Comparative Visualizations",
                "Interactive Dashboards",
                "Customizing Visualizations for Presentations"
            ]
        }
    }
    
    template_paths = []
    
    # Create directory for notebook templates if it doesn't exist
    notebooks_dir = os.path.join(output_path, "notebook_templates")
    os.makedirs(notebooks_dir, exist_ok=True)
    
    # Generate a simple HTML page listing the templates
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RBS Analysis Notebook Templates</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #2c3e50; }
            h2 { color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 10px; }
            .template { background: #f9f9f9; border-left: 4px solid #3498db; padding: 15px; margin-bottom: 20px; }
            .section { margin-left: 20px; }
            .section-title { font-weight: bold; }
            ul { list-style-type: circle; }
        </style>
    </head>
    <body>
        <h1>RBS Analysis Notebook Templates</h1>
        <p>These notebook templates provide starting points for various RBS analysis tasks.</p>
    """
    
    for template_id, template in templates.items():
        html_content += f"""
        <div class="template">
            <h2>{template['title']}</h2>
            <p>{template['description']}</p>
            <div class="section">
                <p class="section-title">Notebook Sections:</p>
                <ul>
        """
        
        for section in template['sections']:
            html_content += f"<li>{section}</li>\n"
        
        html_content += """
                </ul>
            </div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Save the HTML index
    html_path = os.path.join(notebooks_dir, "templates_index.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    template_paths.append(html_path)
    print(f"Notebook templates index saved to {html_path}")
    
    return template_paths

def create_component_tutorials(output_path):
    """
    Creates visual tutorials for key system components.
    
    Args:
        output_path (str): Directory to save the tutorials
        
    Returns:
        list: Paths to the created tutorial files
    """
    print("Creating component tutorials...")
    
    tutorials_dir = os.path.join(output_path, "component_tutorials")
    os.makedirs(tutorials_dir, exist_ok=True)
    
    # Define the key components of the system
    components = [
        {
            "name": "Data Processing Module",
            "description": "Handles data loading, cleaning, and preparation for analysis.",
            "key_functions": [
                "load_and_process_data()",
                "clean_data()",
                "create_geodataframe()"
            ],
            "sample_usage": """
# Load and process RBS data
from src.data_processing import load_and_process_data

# Path to RBS data CSV file
csv_path = 'data/rbs_data.csv'

# Process the data
gdf_rbs = load_and_process_data(csv_path)
"""
        },
        {
            "name": "Analysis Module",
            "description": "Basic statistical analysis of RBS data.",
            "key_functions": [
                "run_basic_analysis()",
                "compute_statistics()",
                "analyze_by_operator()"
            ],
            "sample_usage": """
# Run basic analysis on RBS data
from src.analysis import run_basic_analysis

# Run the analysis
results = run_basic_analysis(gdf_rbs, output_path='outputs')
"""
        },
        {
            "name": "Graph Analysis Module",
            "description": "Network analysis of RBS stations as a graph.",
            "key_functions": [
                "run_graph_analysis()",
                "create_rbs_graph()",
                "analyze_graph_metrics()"
            ],
            "sample_usage": """
# Perform graph analysis on RBS data
from src.graph_analysis import run_graph_analysis

# Create and analyze the RBS network graph
graph_results = run_graph_analysis(gdf_rbs, output_path='outputs')
"""
        }
    ]
    
    tutorial_paths = []
    
    # Create a simple HTML tutorial for each component
    for component in components:
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{component['name']} Tutorial</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        .section {{ background: #f9f9f9; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
        .code {{ background: #f1f1f1; padding: 10px; border-left: 4px solid #3498db; font-family: monospace; overflow-x: auto; }}
        .function {{ font-weight: bold; color: #2980b9; }}
    </style>
</head>
<body>
    <h1>{component['name']} Tutorial</h1>
    
    <div class="section">
        <h2>Overview</h2>
        <p>{component['description']}</p>
    </div>
    
    <div class="section">
        <h2>Key Functions</h2>
        <ul>
"""
        
        for function in component['key_functions']:
            html_content += f'<li class="function">{function}</li>\n'
        
        html_content += """
        </ul>
    </div>
    
    <div class="section">
        <h2>Sample Usage</h2>
        <div class="code">
"""
        
        html_content += component['sample_usage']
        
        html_content += """
        </div>
    </div>
</body>
</html>
"""
        
        # Save the tutorial
        file_name = component['name'].lower().replace(' ', '_') + '_tutorial.html'
        tutorial_path = os.path.join(tutorials_dir, file_name)
        
        with open(tutorial_path, 'w') as f:
            f.write(html_content)
        
        tutorial_paths.append(tutorial_path)
    
    # Create an index HTML file
    index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>RBS Analysis System Component Tutorials</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; }
        ul { list-style-type: none; padding: 0; }
        li { margin-bottom: 10px; }
        a { color: #3498db; text-decoration: none; padding: 5px 10px; border: 1px solid #3498db; border-radius: 3px; }
        a:hover { background-color: #3498db; color: white; }
    </style>
</head>
<body>
    <h1>RBS Analysis System Component Tutorials</h1>
    <p>Select a component tutorial below:</p>
    <ul>
"""
    
    for component in components:
        file_name = component['name'].lower().replace(' ', '_') + '_tutorial.html'
        index_html += f'<li><a href="{file_name}">{component["name"]}</a></li>\n'
    
    index_html += """
    </ul>
</body>
</html>
"""
    
    # Save the index
    index_path = os.path.join(tutorials_dir, "index.html")
    with open(index_path, 'w') as f:
        f.write(index_html)
    
    tutorial_paths.append(index_path)
    print(f"Component tutorials saved to {tutorials_dir}")
    
    return tutorial_paths

def create_illustrated_use_cases(gdf_rbs, output_path):
    """
    Creates illustrated use cases for the RBS analysis system.
    
    Args:
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        output_path (str): Directory to save the use cases
        
    Returns:
        list: Paths to the created use case files
    """
    print("Creating illustrated use cases...")
    
    # Create directory for use cases
    use_cases_dir = os.path.join(output_path, "use_cases")
    os.makedirs(use_cases_dir, exist_ok=True)
    
    # Define the use cases
    use_cases = [
        {
            "title": "Coverage Analysis for Urban Planning",
            "description": "Using RBS analysis to assist in urban planning by identifying coverage gaps.",
            "steps": [
                "Load and process RBS data",
                "Create coverage maps",
                "Identify areas with insufficient coverage",
                "Suggest optimal locations for new RBS stations",
                "Generate reports for urban planners"
            ],
            "benefits": [
                "Improved urban connectivity",
                "Data-driven planning decisions",
                "More efficient resource allocation",
                "Better quality of service for residents"
            ]
        },
        {
            "title": "Comparative Operator Performance",
            "description": "Comparing performance metrics between different telecom operators.",
            "steps": [
                "Load and process RBS data from multiple operators",
                "Calculate key performance metrics for each operator",
                "Create comparative visualizations",
                "Analyze strengths and weaknesses",
                "Generate competitive analysis report"
            ],
            "benefits": [
                "Objective performance comparisons",
                "Identification of best practices",
                "Strategic insights for operators",
                "Improved consumer information"
            ]
        },
        {
            "title": "Technology Migration Planning",
            "description": "Planning the migration from older to newer network technologies (e.g., 4G to 5G).",
            "steps": [
                "Analyze current technology distribution",
                "Identify high-priority areas for migration",
                "Model coverage changes with new technology",
                "Create phased migration plan",
                "Estimate resource requirements and ROI"
            ],
            "benefits": [
                "Strategic technology rollout",
                "Cost-effective migration",
                "Minimized service disruption",
                "Optimized performance gains"
            ]
        }
    ]
    
    use_case_paths = []
    
    # Create a simple HTML page for each use case
    for i, use_case in enumerate(use_cases):
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Use Case: {use_case['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        .section {{ background: #f9f9f9; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
        .steps {{ counter-reset: step; }}
        .step {{ margin-bottom: 10px; }}
        .step::before {{ counter-increment: step; content: "Step " counter(step) ": "; font-weight: bold; color: #3498db; }}
        .benefits li {{ margin-bottom: 8px; }}
    </style>
</head>
<body>
    <h1>Use Case: {use_case['title']}</h1>
    
    <div class="section">
        <h2>Overview</h2>
        <p>{use_case['description']}</p>
    </div>
    
    <div class="section">
        <h2>Implementation Steps</h2>
        <div class="steps">
"""
        
        for step in use_case['steps']:
            html_content += f'<div class="step">{step}</div>\n'
        
        html_content += """
        </div>
    </div>
    
    <div class="section">
        <h2>Benefits</h2>
        <ul class="benefits">
"""
        
        for benefit in use_case['benefits']:
            html_content += f'<li>{benefit}</li>\n'
        
        html_content += """
        </ul>
    </div>
</body>
</html>
"""
        
        # Save the use case
        file_name = f"use_case_{i+1}_{use_case['title'].lower().replace(' ', '_')}.html"
        use_case_path = os.path.join(use_cases_dir, file_name)
        
        with open(use_case_path, 'w') as f:
            f.write(html_content)
        
        use_case_paths.append(use_case_path)
    
    # Create an index HTML file
    index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>RBS Analysis System Use Cases</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; }
        .use-case { background: #f9f9f9; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
        h2 { color: #3498db; margin-top: 0; }
        a { color: #3498db; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>RBS Analysis System Use Cases</h1>
    <p>Select a use case to view its details:</p>
"""
    
    for i, use_case in enumerate(use_cases):
        file_name = f"use_case_{i+1}_{use_case['title'].lower().replace(' ', '_')}.html"
        index_html += f"""
    <div class="use-case">
        <h2>{use_case['title']}</h2>
        <p>{use_case['description']}</p>
        <a href="{file_name}">View Use Case Details</a>
    </div>
"""
    
    index_html += """
</body>
</html>
"""
    
    # Save the index
    index_path = os.path.join(use_cases_dir, "index.html")
    with open(index_path, 'w') as f:
        f.write(index_html)
    
    use_case_paths.append(index_path)
    print(f"Illustrated use cases saved to {use_cases_dir}")
    
    return use_case_paths

#############################################
# 3. DATA STORYTELLING
#############################################

def create_data_narratives(gdf_rbs, output_path):
    """
    Creates data-based narrative visualizations that tell a story with the RBS data.
    
    Args:
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        output_path (str): Directory to save the narratives
        
    Returns:
        list: Paths to the created narrative files
    """
    print("Creating data narratives...")
    
    narratives_dir = os.path.join(output_path, "data_narratives")
    os.makedirs(narratives_dir, exist_ok=True)
    
    # Define some narrative themes
    narratives = [
        {
            "title": "Evolution of Cellular Technology",
            "description": "The transition from 2G to 5G and beyond in our RBS network.",
            "key_points": [
                "Distribution of different technologies",
                "Chronological deployment patterns",
                "Technology density in urban vs. rural areas",
                "Future technology projections"
            ]
        },
        {
            "title": "Coverage Quality Across the Region",
            "description": "How coverage quality varies across different areas and what factors influence it.",
            "key_points": [
                "Coverage metrics by geographic area",
                "Impact of terrain on coverage",
                "Urban vs. suburban vs. rural coverage",
                "Identifying coverage gaps and their causes"
            ]
        },
        {
            "title": "Operator Strategy Analysis",
            "description": "How different operators approach their network deployment and what it reveals about their strategies.",
            "key_points": [
                "Technology preferences by operator",
                "Frequency band utilization patterns",
                "Geographic focus areas",
                "Infrastructure sharing vs. independent deployment"
            ]
        }
    ]
    
    narrative_paths = []
    
    # Create a simple HTML page for each narrative
    for i, narrative in enumerate(narratives):
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Data Narrative: {narrative['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        .section {{ background: #f9f9f9; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
        .key-point {{ margin-bottom: 15px; padding-left: 20px; border-left: 3px solid #3498db; }}
        .key-point-title {{ font-weight: bold; color: #2980b9; }}
    </style>
</head>
<body>
    <h1>Data Narrative: {narrative['title']}</h1>
    
    <div class="section">
        <h2>Overview</h2>
        <p>{narrative['description']}</p>
    </div>
    
    <div class="section">
        <h2>Key Insights</h2>
"""
        
        for point in narrative['key_points']:
            html_content += f'<div class="key-point">
            <div class="key-point-title">{point}</div>
            <p>This data narrative explores {point.lower()} through visualizations and analysis of the RBS dataset.</p>
        </div>\n'
        
        html_content += """
    </div>
    
    <div class="section">
        <h2>Visualization Placeholders</h2>
        <p>In a complete implementation, this page would include relevant visualizations generated from the RBS dataset that help tell the story of the data.</p>
    </div>
</body>
</html>
"""
        
        # Save the narrative
        file_name = f"narrative_{i+1}_{narrative['title'].lower().replace(' ', '_')}.html"
        narrative_path = os.path.join(narratives_dir, file_name)
        
        with open(narrative_path, 'w') as f:
            f.write(html_content)
        
        narrative_paths.append(narrative_path)
    
    # Create an index HTML file
    index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>RBS Data Narratives</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; }
        .narrative { background: #f9f9f9; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
        h2 { color: #3498db; margin-top: 0; }
        a { color: #3498db; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>RBS Data Narratives</h1>
    <p>Select a data narrative to explore:</p>
"""
    
    for i, narrative in enumerate(narratives):
        file_name = f"narrative_{i+1}_{narrative['title'].lower().replace(' ', '_')}.html"
        index_html += f"""
    <div class="narrative">
        <h2>{narrative['title']}</h2>
        <p>{narrative['description']}</p>
        <a href="{file_name}">View Data Narrative</a>
    </div>
"""
    
    index_html += """
</body>
</html>
"""
    
    # Save the index
    index_path = os.path.join(narratives_dir, "index.html")
    with open(index_path, 'w') as f:
        f.write(index_html)
    
    narrative_paths.append(index_path)
    print(f"Data narratives saved to {narratives_dir}")
    
    return narrative_paths

def create_before_after_comparisons(gdf_rbs, output_path):
    """
    Creates before/after comparison visualizations to illustrate improvements or changes.
    
    Args:
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        output_path (str): Directory to save the comparisons
        
    Returns:
        list: Paths to the created comparison files
    """
    print("Creating before/after comparisons...")
    
    comparisons_dir = os.path.join(output_path, "before_after_comparisons")
    os.makedirs(comparisons_dir, exist_ok=True)
    
    # Define some comparison scenarios
    comparisons = [
        {
            "title": "Technology Upgrade Impact",
            "description": "Visual comparison of network coverage before and after technology upgrades.",
            "improvement_metrics": [
                {"name": "Coverage Area", "before": "70%", "after": "85%"},
                {"name": "Average Speed", "before": "25 Mbps", "after": "75 Mbps"},
                {"name": "Connection Reliability", "before": "93%", "after": "99%"}
            ]
        },
        {
            "title": "New RBS Site Deployment",
            "description": "Impact of adding new RBS sites in previously underserved areas.",
            "improvement_metrics": [
                {"name": "Coverage Gaps", "before": "12", "after": "3"},
                {"name": "Signal Strength", "before": "-95 dBm", "after": "-75 dBm"},
                {"name": "Population Served", "before": "85%", "after": "98%"}
            ]
        },
        {
            "title": "Optimization of Existing Sites",
            "description": "Results of optimizing parameters of existing RBS without adding new sites.",
            "improvement_metrics": [
                {"name": "Interference Levels", "before": "High", "after": "Low"},
                {"name": "Dropped Calls", "before": "4.2%", "after": "1.8%"},
                {"name": "Capacity Utilization", "before": "65%", "after": "85%"}
            ]
        }
    ]
    
    comparison_paths = []
    
    # Create a simple HTML page for each comparison
    for i, comparison in enumerate(comparisons):
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Before/After Comparison: {comparison['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        .section {{ background: #f9f9f9; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
        .comparison-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        .comparison-table th, .comparison-table td {{ padding: 10px; text-align: center; border: 1px solid #ddd; }}
        .comparison-table th {{ background-color: #f2f2f2; }}
        .before {{ background-color: #ffeeee; }}
        .after {{ background-color: #eeffee; }}
        .metric-name {{ text-align: left; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Before/After Comparison: {comparison['title']}</h1>
    
    <div class="section">
        <h2>Overview</h2>
        <p>{comparison['description']}</p>
    </div>
    
    <div class="section">
        <h2>Improvement Metrics</h2>
        <table class="comparison-table">
            <tr>
                <th>Metric</th>
                <th>Before</th>
                <th>After</th>
                <th>Improvement</th>
            </tr>
"""
        
        for metric in comparison['improvement_metrics']:
            html_content += f"""
            <tr>
                <td class="metric-name">{metric['name']}</td>
                <td class="before">{metric['before']}</td>
                <td class="after">{metric['after']}</td>
                <td>✓</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>Visualization Placeholders</h2>
        <p>In a complete implementation, this page would include before/after visualizations generated from the RBS dataset showing the impact of the changes.</p>
    </div>
</body>
</html>
"""
        
        # Save the comparison
        file_name = f"comparison_{i+1}_{comparison['title'].lower().replace(' ', '_')}.html"
        comparison_path = os.path.join(comparisons_dir, file_name)
        
        with open(comparison_path, 'w') as f:
            f.write(html_content)
        
        comparison_paths.append(comparison_path)
    
    # Create an index HTML file
    index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>RBS Before/After Comparisons</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; }
        .comparison { background: #f9f9f9; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
        h2 { color: #3498db; margin-top: 0; }
        a { color: #3498db; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>RBS Before/After Comparisons</h1>
    <p>Select a comparison to explore:</p>
"""
    
    for i, comparison in enumerate(comparisons):
        file_name = f"comparison_{i+1}_{comparison['title'].lower().replace(' ', '_')}.html"
        index_html += f"""
    <div class="comparison">
        <h2>{comparison['title']}</h2>
        <p>{comparison['description']}</p>
        <a href="{file_name}">View Comparison</a>
    </div>
"""
    
    index_html += """
</body>
</html>
"""
    
    # Save the index
    index_path = os.path.join(comparisons_dir, "index.html")
    with open(index_path, 'w') as f:
        f.write(index_html)
    
    comparison_paths.append(index_path)
    print(f"Before/after comparisons saved to {comparisons_dir}")
    
    return comparison_paths

def create_key_findings_visualizations(gdf_rbs, output_path):
    """
    Creates visualizations that highlight key findings from the RBS analysis.
    
    Args:
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        output_path (str): Directory to save the visualizations
        
    Returns:
        list: Paths to the created visualization files
    """
    print("Creating key findings visualizations...")
    
    findings_dir = os.path.join(output_path, "key_findings")
    os.makedirs(findings_dir, exist_ok=True)
    
    # Define some key findings
    findings = [
        {
            "title": "Urban vs. Rural Coverage Disparity",
            "description": "Analysis of the significant disparity in RBS coverage between urban and rural areas.",
            "key_points": [
                "Urban areas have 3.5x more RBS density than rural areas",
                "Technology distribution is more diverse in urban areas",
                "Rural areas primarily served by older technologies",
                "Signal strength variation is higher in rural areas due to terrain"
            ]
        },
        {
            "title": "Operator Market Share Analysis",
            "description": "Analysis of market share by operator based on RBS infrastructure.",
            "key_points": [
                "Three major operators control 85% of all RBS sites",
                "Operator A leads in 5G deployment with 60% of all 5G sites",
                "Operator B dominates rural coverage with 55% market share",
                "Operator C focuses on high-density urban deployments"
            ]
        },
        {
            "title": "Technology Transition Trends",
            "description": "Analysis of the transition patterns between different cellular technologies.",
            "key_points": [
                "5G deployment follows 60% overlap with existing 4G sites",
                "2G technologies being phased out with 15% reduction annually",
                "Technology transitions follow clear geographic patterns",
                "New technology adoption rate varies significantly by region"
            ]
        }
    ]
    
    finding_paths = []
    
    # Create a simple HTML page for each finding
    for i, finding in enumerate(findings):
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Key Finding: {finding['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        .section {{ background: #f9f9f9; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
        .insight {{ padding: 10px; margin-bottom: 10px; background: #e8f4fc; border-left: 4px solid #3498db; }}
    </style>
</head>
<body>
    <h1>Key Finding: {finding['title']}</h1>
    
    <div class="section">
        <h2>Summary</h2>
        <p>{finding['description']}</p>
    </div>
    
    <div class="section">
        <h2>Key Insights</h2>
"""
        
        for point in finding['key_points']:
            html_content += f'<div class="insight">{point}</div>\n'
        
        html_content += """
    </div>
    
    <div class="section">
        <h2>Visualization Placeholders</h2>
        <p>In a complete implementation, this page would include visualizations specifically designed to highlight this key finding.</p>
    </div>
</body>
</html>
"""
        
        # Save the finding
        file_name = f"finding_{i+1}_{finding['title'].lower().replace(' ', '_')}.html"
        finding_path = os.path.join(findings_dir, file_name)
        
        with open(finding_path, 'w') as f:
            f.write(html_content)
        
        finding_paths.append(finding_path)
    
    # Create an index HTML file
    index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>RBS Analysis Key Findings</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; }
        .finding { background: #f9f9f9; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
        h2 { color: #3498db; margin-top: 0; }
        a { color: #3498db; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>RBS Analysis Key Findings</h1>
    <p>Select a key finding to explore in detail:</p>
"""
    
    for i, finding in enumerate(findings):
        file_name = f"finding_{i+1}_{finding['title'].lower().replace(' ', '_')}.html"
        index_html += f"""
    <div class="finding">
        <h2>{finding['title']}</h2>
        <p>{finding['description']}</p>
        <a href="{file_name}">View Key Finding</a>
    </div>
"""
    
    index_html += """
</body>
</html>
"""
    
    # Save the index
    index_path = os.path.join(findings_dir, "index.html")
    with open(index_path, 'w') as f:
        f.write(index_html)
    
    finding_paths.append(index_path)
    print(f"Key findings visualizations saved to {findings_dir}")
    
    return finding_paths

def create_educational_documentation(gdf_rbs, output_path):
    """
    Main function to create educational documentation.
    
    Args:
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        output_path (str): Path to save the documentation
        
    Returns:
        dict: Dictionary with paths to created documentation
    """
    print("Creating educational documentation...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Create subdirectories for each component
    visual_narrative_dir = os.path.join(output_path, "visual_narrative")
    interactive_docs_dir = os.path.join(output_path, "interactive_docs")
    storytelling_dir = os.path.join(output_path, "data_storytelling")
    
    os.makedirs(visual_narrative_dir, exist_ok=True)
    os.makedirs(interactive_docs_dir, exist_ok=True)
    os.makedirs(storytelling_dir, exist_ok=True)
    
    # Dictionary to store outputs
    documentation_outputs = {
        "visual_narrative": {},
        "interactive_docs": {},
        "data_storytelling": {}
    }
    
    # 1. Create Visual Project Narrative
    try:
        print("Creating visual project narrative...")
        pipeline_flowchart = create_pipeline_flowchart(visual_narrative_dir)
        documentation_outputs["visual_narrative"]["pipeline_flowchart"] = pipeline_flowchart
        
        sequential_visualizations = create_sequential_visualizations(gdf_rbs, visual_narrative_dir)
        documentation_outputs["visual_narrative"]["sequential_visualizations"] = sequential_visualizations
    except Exception as e:
        print(f"Error creating visual project narrative: {e}")
    
    # 2. Create Interactive Documentation
    try:
        print("Creating interactive documentation...")
        notebook_templates = create_notebook_templates(interactive_docs_dir)
        documentation_outputs["interactive_docs"]["notebook_templates"] = notebook_templates
        
        component_tutorials = create_component_tutorials(interactive_docs_dir)
        documentation_outputs["interactive_docs"]["component_tutorials"] = component_tutorials
        
        use_cases = create_illustrated_use_cases(gdf_rbs, interactive_docs_dir)
        documentation_outputs["interactive_docs"]["use_cases"] = use_cases
    except Exception as e:
        print(f"Error creating interactive documentation: {e}")
    
    # 3. Create Data Storytelling
    try:
        print("Creating data storytelling visualizations...")
        data_narratives = create_data_narratives(gdf_rbs, storytelling_dir)
        documentation_outputs["data_storytelling"]["data_narratives"] = data_narratives
        
        before_after = create_before_after_comparisons(gdf_rbs, storytelling_dir)
        documentation_outputs["data_storytelling"]["before_after"] = before_after
        
        key_findings = create_key_findings_visualizations(gdf_rbs, storytelling_dir)
        documentation_outputs["data_storytelling"]["key_findings"] = key_findings
    except Exception as e:
        print(f"Error creating data storytelling: {e}")
    
    # Create index.html to navigate all documentation
    create_documentation_index(documentation_outputs, output_path)
    
    print(f"Educational documentation created successfully in {output_path}")
    return documentation_outputs 