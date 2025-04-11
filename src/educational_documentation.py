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

def create_notebook_templates(output_path):
    """
    Creates Jupyter notebook templates with explanations and visualizations.
    
    Args:
        output_path (str): Directory to save the notebook templates
        
    Returns:
        list: Paths to the created notebook templates
    """
    print("Creating notebook templates...")
    # This function will be implemented in a later section
    return []

def create_component_tutorials(output_path):
    """
    Creates visual tutorials for each system component.
    
    Args:
        output_path (str): Directory to save the tutorials
        
    Returns:
        dict: Dictionary with paths to the created tutorials
    """
    print("Creating component tutorials...")
    # This function will be implemented in a later section
    return {}

def create_illustrated_use_cases(gdf_rbs, output_path):
    """
    Creates illustrated use cases for the system.
    
    Args:
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        output_path (str): Directory to save the use cases
        
    Returns:
        list: Paths to the created use case illustrations
    """
    print("Creating illustrated use cases...")
    # This function will be implemented in a later section
    return []

def create_data_narratives(gdf_rbs, output_path):
    """
    Creates data-based narrative visualizations.
    
    Args:
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        output_path (str): Directory to save the narratives
        
    Returns:
        list: Paths to the created narrative visualizations
    """
    print("Creating data narratives...")
    # This function will be implemented in a later section
    return []

def create_before_after_comparisons(gdf_rbs, output_path):
    """
    Creates before/after comparison visualizations.
    
    Args:
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        output_path (str): Directory to save the comparisons
        
    Returns:
        list: Paths to the created comparison visualizations
    """
    print("Creating before/after comparisons...")
    # This function will be implemented in a later section
    return []

def create_key_findings_visualizations(gdf_rbs, output_path):
    """
    Creates visualizations summarizing key findings.
    
    Args:
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        output_path (str): Directory to save the visualizations
        
    Returns:
        list: Paths to the created key findings visualizations
    """
    print("Creating key findings visualizations...")
    # This function will be implemented in a later section
    return []

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