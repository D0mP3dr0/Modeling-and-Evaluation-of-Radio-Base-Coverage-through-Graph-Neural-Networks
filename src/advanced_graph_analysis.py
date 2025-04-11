"""
Advanced Graph Analysis Module for Radio Base Station Networks

This module extends basic graph analysis with:
1. Advanced Graph Visualizations - Telecom-specific layouts, dynamic and multilayer graphs
2. Advanced Network Metrics - Community detection, vulnerability analysis, telecom-specific centrality
3. Comparative Analysis Between Operators - Side-by-side comparisons, efficiency metrics, and overlap visualization

This module requires additional dependencies:
- python-louvain: For community detection
- folium: For interactive maps
- plotly: For 3D visualizations
- torch and torch-geometric: Optional, for advanced GNN capabilities

Usage:
    from advanced_graph_analysis import run_advanced_graph_analysis
    results = run_advanced_graph_analysis(gdf_rbs, output_path, time_field='InstallDate')
"""

import os
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from sklearn.cluster import SpectralClustering
from community import community_louvain
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import folium
from folium.plugins import HeatMap, MarkerCluster
import matplotlib.patches as mpatches
from itertools import combinations
from datetime import datetime
import json

# Optional imports for 3D visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not found. 3D visualizations will not be available.")

#############################################
# 1. ADVANCED GRAPH VISUALIZATIONS
#############################################

def create_telecom_layout(G, gdf_rbs, output_path, title="Telecom-Specific Network Layout"):
    """
    Creates a specialized graph visualization for telecommunication networks 
    that preserves both geographic and topological properties.
    
    Args:
        G (nx.Graph): NetworkX graph of RBS
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        output_path (str): Path to save the visualization
        title (str): Chart title
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    print("Creating telecommunication-specific network layout...")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle(title, fontsize=16)
    
    # Get node positions from graph
    geo_pos = nx.get_node_attributes(G, 'pos')
    
    # Get operator information
    operators = nx.get_node_attributes(G, 'operator')
    
    # Define colors for operators
    operator_colors = {
        'CLARO': '#E02020',  # Red
        'OI': '#FFD700',     # Yellow
        'VIVO': '#9932CC',   # Purple
        'TIM': '#0000CD',    # Blue
        'N/A': '#CCCCCC'     # Gray
    }
    
    # 1. Geographic Layout (Left subplot)
    ax1.set_title("Geographic Layout")
    
    # Draw nodes by operator
    for operator in set(operators.values()):
        # Get nodes for this operator
        nodes = [n for n, op in operators.items() if op == operator]
        # Get node positions 
        pos_list = {n: geo_pos[n] for n in nodes if n in geo_pos}
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos_list, 
            nodelist=nodes,
            node_color=operator_colors.get(operator, '#CCCCCC'),
            node_size=100,
            alpha=0.8,
            ax=ax1,
            label=operator
        )
    
    # Draw edges with width based on signal strength or capacity
    edge_widths = []
    for u, v, d in G.edges(data=True):
        if 'capacity' in d:
            width = d['capacity'] / 10  # Scale appropriately
        elif 'weight' in d:
            width = d['weight'] 
        else:
            width = 0.5
        edge_widths.append(width)
    
    nx.draw_networkx_edges(
        G, geo_pos, 
        width=edge_widths,
        alpha=0.4, 
        edge_color='gray',
        ax=ax1
    )
    
    # Add background map or grid
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    
    # 2. Topological Layout (Right subplot)
    ax2.set_title("Topological Layout")
    
    # Use force-directed layout for topology view
    topo_pos = nx.spring_layout(G, seed=42)
    
    # Group by technology
    technologies = nx.get_node_attributes(G, 'technology')
    tech_groups = {}
    
    for node, tech in technologies.items():
        if tech not in tech_groups:
            tech_groups[tech] = []
        tech_groups[tech].append(node)
    
    # Define colors for technologies
    tech_colors = {
        'GSM': '#66c2a5',
        '3G': '#fc8d62',
        '4G': '#8da0cb',
        '5G': '#e78ac3',
        'N/A': '#cccccc'
    }
    
    # Draw nodes grouped by technology
    for tech, nodes in tech_groups.items():
        nx.draw_networkx_nodes(
            G, topo_pos, 
            nodelist=nodes,
            node_color=tech_colors.get(tech, '#cccccc'),
            node_size=100,
            alpha=0.8,
            ax=ax2,
            label=tech
        )
    
    # Draw edges, with edge width representing connection strength
    nx.draw_networkx_edges(
        G, topo_pos, 
        width=0.7,
        alpha=0.4, 
        edge_color='gray',
        ax=ax2
    )
    
    # Create legends for both plots
    ax1.legend(title="Operators")
    ax2.legend(title="Technologies")
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    plt.savefig(os.path.join(output_path, "telecom_network_layout.png"), dpi=300, bbox_inches='tight')
    
    return fig

def create_dynamic_network_visualization(G, gdf_rbs, time_field, output_path):
    """
    Creates a dynamic visualization showing network evolution over time.
    
    Args:
        G (nx.Graph): NetworkX graph of RBS
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        time_field (str): Name of the timestamp field in the data
        output_path (str): Path to save the visualization
    """
    print("Creating dynamic network visualization...")
    
    # Check if the time field exists
    if time_field not in gdf_rbs.columns:
        print(f"Error: Time field '{time_field}' not found in the data.")
        return None
    
    # Convert timestamps to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(gdf_rbs[time_field]):
        gdf_rbs[time_field] = pd.to_datetime(gdf_rbs[time_field], errors='coerce')
    
    # Sort data by time
    gdf_sorted = gdf_rbs.sort_values(by=time_field)
    
    # Create a list of unique time periods (e.g., years)
    time_periods = sorted(gdf_sorted[time_field].dt.year.unique())
    
    # Create subgraphs for each time period
    subgraphs = {}
    for period in time_periods:
        # Filter data for this time period
        period_data = gdf_sorted[gdf_sorted[time_field].dt.year <= period]
        
        # Create a new graph for this period
        subgraph = nx.Graph()
        
        # Add nodes
        for idx, row in period_data.iterrows():
            subgraph.add_node(
                idx,
                pos=(row['Longitude'], row['Latitude']),
                operator=row.get('Operator', 'N/A'),
                technology=row.get('Tecnologia', 'N/A'),
                year=row[time_field].year
            )
        
        # Add edges (connections between nodes)
        for node1, node2 in combinations(subgraph.nodes(), 2):
            # Calculate Euclidean distance
            pos1 = subgraph.nodes[node1]['pos']
            pos2 = subgraph.nodes[node2]['pos']
            
            # Simplified distance calculation for animation purposes
            distance = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
            
            # Add edge if distance is below threshold (in degrees)
            if distance < 0.05:  # Adjust threshold as needed
                subgraph.add_edge(node1, node2, weight=1.0/distance)
        
        subgraphs[period] = subgraph
    
    # Create animation
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Animation function for each frame
    def update(frame):
        ax.clear()
        period = time_periods[frame]
        G_period = subgraphs[period]
        
        # Add title with current period
        ax.set_title(f"Network Evolution - Year {period}", fontsize=16)
        
        # Get positions
        pos = nx.get_node_attributes(G_period, 'pos')
        
        # Get operator information
        operators = nx.get_node_attributes(G_period, 'operator')
        
        # Define colors for operators
        operator_colors = {
            'CLARO': '#E02020',
            'OI': '#FFD700',
            'VIVO': '#9932CC',
            'TIM': '#0000CD',
            'N/A': '#CCCCCC'
        }
        
        # Group nodes by operator
        operator_groups = {}
        for node, operator in operators.items():
            if operator not in operator_groups:
                operator_groups[operator] = []
            operator_groups[operator].append(node)
        
        # Draw nodes by operator
        for operator, nodes in operator_groups.items():
            nx.draw_networkx_nodes(
                G_period, pos, 
                nodelist=nodes,
                node_color=operator_colors.get(operator, '#CCCCCC'),
                node_size=100,
                alpha=0.8,
                label=operator if frame == 0 else ""  # Only show legend on first frame
            )
        
        # Draw edges
        nx.draw_networkx_edges(G_period, pos, width=0.5, alpha=0.5, edge_color='gray')
        
        # Show node count
        nodes_count = G_period.number_of_nodes()
        ax.text(0.02, 0.02, f"Total RBS: {nodes_count}", transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7))
        
        # Show legend only on first frame
        if frame == 0:
            plt.legend(title="Operators")
        
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle='--', alpha=0.4)
        
        return ax
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(time_periods), interval=1000, blit=False)
    
    # Save animation
    ani.save(os.path.join(output_path, 'network_evolution.gif'), writer='pillow', fps=1)
    
    return ani

def create_multilayer_graph(G, gdf_rbs, output_path):
    """
    Creates a multilayer graph visualization showing different aspects 
    of the network (geographical, topological, technological).
    
    Args:
        G (nx.Graph): NetworkX graph of RBS
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        output_path (str): Path to save the visualization
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    print("Creating multilayer graph visualization...")
    
    if PLOTLY_AVAILABLE:
        # Create a more advanced 3D multilayer visualization with Plotly
        return _create_plotly_multilayer_graph(G, gdf_rbs, output_path)
    else:
        # Fallback to matplotlib for simpler multilayer visualization
        return _create_matplotlib_multilayer_graph(G, gdf_rbs, output_path)

def _create_matplotlib_multilayer_graph(G, gdf_rbs, output_path):
    """
    Creates a multilayer graph visualization using Matplotlib.
    
    Args:
        G (nx.Graph): NetworkX graph of RBS
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        output_path (str): Path to save the visualization
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Get positions and attributes
    pos = nx.get_node_attributes(G, 'pos')
    operators = nx.get_node_attributes(G, 'operator')
    technologies = nx.get_node_attributes(G, 'technology')
    
    # Define colors
    operator_colors = {
        'CLARO': '#E02020', 'OI': '#FFD700', 
        'VIVO': '#9932CC', 'TIM': '#0000CD', 'N/A': '#CCCCCC'
    }
    
    tech_colors = {
        'GSM': '#66c2a5', '3G': '#fc8d62', 
        '4G': '#8da0cb', '5G': '#e78ac3', 'N/A': '#cccccc'
    }
    
    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Layer 1: Geographical Network (top left)
    ax1 = plt.subplot(gs[0, 0])
    ax1.set_title("Geographical Network Layer", fontsize=14)
    
    for operator, color in operator_colors.items():
        # Get nodes for this operator
        nodes = [n for n, op in operators.items() if op == operator]
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes, node_color=color, 
            node_size=100, alpha=0.8, ax=ax1, label=operator
        )
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, edge_color='gray', ax=ax1)
    ax1.legend(title="Operators", loc="upper right", fontsize=10)
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # Layer 2: Technological Network (top right)
    ax2 = plt.subplot(gs[0, 1])
    ax2.set_title("Technological Network Layer", fontsize=14)
    
    for tech, color in tech_colors.items():
        # Get nodes for this technology
        nodes = [n for n, t in technologies.items() if t == tech]
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes, node_color=color, 
            node_size=100, alpha=0.8, ax=ax2, label=tech
        )
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, edge_color='gray', ax=ax2)
    ax2.legend(title="Technologies", loc="upper right", fontsize=10)
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    # Layer 3: Topological Network (bottom left)
    ax3 = plt.subplot(gs[1, 0])
    ax3.set_title("Topological Network Layer", fontsize=14)
    
    # Use force-directed layout for topology
    topo_pos = nx.spring_layout(G, seed=42)
    
    # Calculate node centrality
    betweenness = nx.betweenness_centrality(G)
    centrality_values = list(betweenness.values())
    
    # Draw nodes colored by centrality
    nodes = nx.draw_networkx_nodes(
        G, topo_pos, node_color=centrality_values, 
        node_size=100, alpha=0.8, ax=ax3, cmap=plt.cm.viridis
    )
    
    # Add colorbar
    plt.colorbar(nodes, ax=ax3, label="Betweenness Centrality")
    
    # Draw edges based on weight
    weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, topo_pos, width=weights, alpha=0.4, edge_color='gray', ax=ax3
    )
    
    # Layer 4: Community Network (bottom right)
    ax4 = plt.subplot(gs[1, 1])
    ax4.set_title("Community Structure Layer", fontsize=14)
    
    # Get communities 
    communities = community_louvain.best_partition(G)
    
    # Convert community dict to list for node coloring
    community_values = [communities[n] for n in G.nodes()]
    
    # Draw nodes colored by community
    nodes = nx.draw_networkx_nodes(
        G, pos, node_color=community_values, 
        node_size=100, alpha=0.8, ax=ax4, cmap=plt.cm.tab20
    )
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, edge_color='gray', ax=ax4)
    
    # Add colorbar for communities
    plt.colorbar(nodes, ax=ax4, label="Community ID")
    
    ax4.set_xlabel("Longitude")
    ax4.set_ylabel("Latitude")
    ax4.grid(True, linestyle='--', alpha=0.4)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "multilayer_network.png"), dpi=300, bbox_inches='tight')
    
    return fig

def _create_plotly_multilayer_graph(G, gdf_rbs, output_path):
    """
    Creates an interactive 3D multilayer graph visualization using Plotly.
    
    Args:
        G (nx.Graph): NetworkX graph of RBS
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        output_path (str): Path to save the visualization
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Get node attributes
    pos = nx.get_node_attributes(G, 'pos')
    operators = nx.get_node_attributes(G, 'operator')
    technologies = nx.get_node_attributes(G, 'technology')
    
    # Calculate communities
    communities = community_louvain.best_partition(G)
    
    # Get edges
    edge_x = []
    edge_y = []
    edge_z = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Layer 1 (z=0): Geographic
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([0, 0, None])
        
        # Layer 2 (z=1): Technological
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([1, 1, None])
        
        # Layer 3 (z=2): Community
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([2, 2, None])
        
        # Vertical connections between layers
        edge_x.extend([x0, x0, None])
        edge_y.extend([y0, y0, None])
        edge_z.extend([0, 1, None])
        
        edge_x.extend([x0, x0, None])
        edge_y.extend([y0, y0, None])
        edge_z.extend([1, 2, None])
    
    # Create edges trace
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces for each layer
    node_traces = []
    
    # Layer 1: Geographic by Operator
    operator_colors = {
        'CLARO': '#E02020', 'OI': '#FFD700', 
        'VIVO': '#9932CC', 'TIM': '#0000CD', 'N/A': '#CCCCCC'
    }
    
    for operator, color in operator_colors.items():
        x = []
        y = []
        z = []
        text = []
        
        for node in G.nodes():
            if operators.get(node, 'N/A') == operator:
                x.append(pos[node][0])
                y.append(pos[node][1])
                z.append(0)  # Layer 1
                text.append(f"Node: {node}<br>Operator: {operator}")
        
        node_trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=6,
                color=color,
                opacity=0.8
            ),
            text=text,
            hoverinfo='text',
            name=f"{operator} (Layer 1)"
        )
        
        node_traces.append(node_trace)
    
    # Layer 2: Technology
    tech_colors = {
        'GSM': '#66c2a5', '3G': '#fc8d62', 
        '4G': '#8da0cb', '5G': '#e78ac3', 'N/A': '#cccccc'
    }
    
    for tech, color in tech_colors.items():
        x = []
        y = []
        z = []
        text = []
        
        for node in G.nodes():
            if technologies.get(node, 'N/A') == tech:
                x.append(pos[node][0])
                y.append(pos[node][1])
                z.append(1)  # Layer 2
                text.append(f"Node: {node}<br>Technology: {tech}")
        
        node_trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=6,
                color=color,
                opacity=0.8
            ),
            text=text,
            hoverinfo='text',
            name=f"{tech} (Layer 2)"
        )
        
        node_traces.append(node_trace)
    
    # Layer 3: Communities
    x = []
    y = []
    z = []
    color = []
    text = []
    
    for node in G.nodes():
        x.append(pos[node][0])
        y.append(pos[node][1])
        z.append(2)  # Layer 3
        color.append(communities.get(node, 0))
        text.append(f"Node: {node}<br>Community: {communities.get(node, 0)}")
    
    node_trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=6,
            color=color,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(
                title='Community'
            )
        ),
        text=text,
        hoverinfo='text',
        name='Communities (Layer 3)'
    )
    
    node_traces.append(node_trace)
    
    # Create figure
    fig = go.Figure(data=[edge_trace] + node_traces)
    
    # Update layout
    fig.update_layout(
        title='Multilayer Network Visualization',
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Layer',
            zaxis=dict(
                tickvals=[0, 1, 2],
                ticktext=['Geographic', 'Technological', 'Community']
            )
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        width=1000,
        height=800
    )
    
    # Save as HTML
    fig.write_html(os.path.join(output_path, "interactive_multilayer_network.html"))
    
    return fig 

#############################################
# 2. ADVANCED NETWORK METRICS
#############################################

def analyze_communities(G, gdf_rbs, output_path):
    """
    Detects and analyzes communities in RBS networks using Louvain algorithm.
    
    Args:
        G (nx.Graph): NetworkX graph of RBS
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        output_path (str): Path to save the visualization
        
    Returns:
        dict: Dictionary with community analysis results
    """
    print("Analyzing communities in RBS network...")
    
    # Detect communities using Louvain method
    communities = community_louvain.best_partition(G)
    
    # Add community information to graph
    nx.set_node_attributes(G, communities, 'community')
    
    # Count number of communities
    num_communities = len(set(communities.values()))
    
    # Create output dictionary with results
    results = {
        'num_communities': num_communities,
        'community_sizes': {},
        'community_compositions': {},
        'community_metrics': {}
    }
    
    # Analyze each community
    for community_id in set(communities.values()):
        # Get nodes in this community
        community_nodes = [n for n, c in communities.items() if c == community_id]
        
        # Store community size
        results['community_sizes'][community_id] = len(community_nodes)
        
        # Extract subgraph for this community
        community_subgraph = G.subgraph(community_nodes).copy()
        
        # Calculate metrics for community
        metrics = {
            'avg_degree': sum(dict(community_subgraph.degree()).values()) / len(community_nodes),
            'density': nx.density(community_subgraph),
            'diameter': nx.diameter(community_subgraph) if nx.is_connected(community_subgraph) else float('inf'),
            'modularity': community_louvain.modularity(communities, G)
        }
        
        results['community_metrics'][community_id] = metrics
        
        # Analyze community composition by operator
        operators = {}
        for node in community_nodes:
            op = G.nodes[node].get('operator', 'N/A')
            operators[op] = operators.get(op, 0) + 1
        
        results['community_compositions'][community_id] = operators
    
    # Create visualizations
    _visualize_communities(G, communities, output_path)
    
    # Save results to file
    with open(os.path.join(output_path, 'community_analysis.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def _visualize_communities(G, communities, output_path):
    """
    Creates visualizations of detected communities.
    
    Args:
        G (nx.Graph): NetworkX graph
        communities (dict): Community assignments
        output_path (str): Path to save visualizations
    """
    # Get positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Get unique community IDs and set up colormap
    community_ids = sorted(set(communities.values()))
    cmap = plt.cm.get_cmap('tab20', len(community_ids))
    
    # Draw nodes colored by community
    for i, comm_id in enumerate(community_ids):
        nodes = [n for n, c in communities.items() if c == comm_id]
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=nodes,
            node_color=[cmap(i)] * len(nodes),
            node_size=100,
            alpha=0.8,
            label=f"Community {comm_id}"
        )
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, edge_color='gray')
    
    # Add legend with limited items if there are too many communities
    if len(community_ids) > 10:
        plt.legend(title="Communities", loc="upper right", ncol=2, fontsize=10)
    else:
        plt.legend(title="Communities", loc="upper right", fontsize=10)
    
    plt.title("Network Communities", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_path, "community_structure.png"), dpi=300, bbox_inches='tight')

def analyze_vulnerability(G, gdf_rbs, output_path):
    """
    Analyzes network vulnerability by identifying critical nodes 
    whose removal would significantly affect network connectivity.
    
    Args:
        G (nx.Graph): NetworkX graph of RBS
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        output_path (str): Path to save the visualization
        
    Returns:
        dict: Dictionary with vulnerability analysis results
    """
    print("Analyzing network vulnerability...")
    
    # Get the largest connected component
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    G_cc = G.subgraph(largest_cc).copy()
    
    # Calculate node centrality metrics
    betweenness = nx.betweenness_centrality(G_cc, weight='distance')
    closeness = nx.closeness_centrality(G_cc, distance='distance')
    eigenvector = nx.eigenvector_centrality_numpy(G_cc, weight='weight')
    
    # Add centrality values as node attributes
    nx.set_node_attributes(G, betweenness, 'betweenness')
    nx.set_node_attributes(G, closeness, 'closeness')
    nx.set_node_attributes(G, eigenvector, 'eigenvector')
    
    # Calculate additional node-level vulnerability metrics
    vulnerability = {}
    for node in G_cc.nodes():
        # Count number of adjacent components created when node is removed
        G_test = G_cc.copy()
        neighbors = list(G_test.neighbors(node))
        G_test.remove_node(node)
        
        num_components = nx.number_connected_components(G_test)
        vulnerability[node] = {
            'num_components_after_removal': num_components,
            'betweenness': betweenness[node],
            'closeness': closeness[node],
            'eigenvector': eigenvector[node],
            'degree': G_cc.degree(node),
            'vulnerability_score': betweenness[node] * G_cc.degree(node)  # Combined score
        }
    
    # Identify top critical nodes
    sorted_nodes = sorted(vulnerability.items(), key=lambda x: x[1]['vulnerability_score'], reverse=True)
    critical_nodes = [node for node, _ in sorted_nodes[:10]]  # Top 10 critical nodes
    
    # Create vulnerability map visualization
    _visualize_vulnerability(G, vulnerability, critical_nodes, output_path)
    
    # Calculate network-level vulnerability metrics
    network_metrics = {
        'node_connectivity': nx.node_connectivity(G_cc),
        'edge_connectivity': nx.edge_connectivity(G_cc),
        'algebraic_connectivity': nx.algebraic_connectivity(G_cc, weight='weight'),
        'average_shortest_path': nx.average_shortest_path_length(G_cc, weight='distance'),
        'critical_nodes': [int(n) for n in critical_nodes]  # Convert to int for JSON serialization
    }
    
    # Save vulnerability results
    results = {
        'network_vulnerability': network_metrics,
        'node_vulnerability': {int(k): v for k, v in vulnerability.items()}  # Convert keys to int for JSON
    }
    
    with open(os.path.join(output_path, 'vulnerability_analysis.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def _visualize_vulnerability(G, vulnerability, critical_nodes, output_path):
    """
    Creates visualizations of network vulnerability.
    
    Args:
        G (nx.Graph): NetworkX graph
        vulnerability (dict): Node vulnerability metrics
        critical_nodes (list): List of critical nodes
        output_path (str): Path to save visualizations
    """
    # Get positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Get vulnerability scores
    vuln_scores = [vulnerability.get(n, {}).get('vulnerability_score', 0) for n in G.nodes()]
    
    # Draw nodes with size and color based on vulnerability
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=vuln_scores,
        node_size=[max(50, v.get('vulnerability_score', 0) * 1000) if n in vulnerability else 50 for n, v in zip(G.nodes(), [vulnerability.get(n, {}) for n in G.nodes()])],
        alpha=0.8,
        cmap=plt.cm.Reds
    )
    
    # Highlight critical nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=critical_nodes,
        node_color='yellow',
        node_shape='*',
        node_size=300,
        alpha=1.0,
        label="Critical Nodes"
    )
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, edge_color='gray')
    
    # Add colorbar
    plt.colorbar(nodes, label="Vulnerability Score")
    
    # Add labels for critical nodes
    labels = {node: f"{node}" for node in critical_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')
    
    plt.title("Network Vulnerability Analysis", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_path, "vulnerability_map.png"), dpi=300, bbox_inches='tight')
    
    # Create a second visualization showing the effect of removing critical nodes
    plt.figure(figsize=(14, 10))
    
    # Create a copy of the graph without critical nodes
    G_reduced = G.copy()
    G_reduced.remove_nodes_from(critical_nodes)
    
    # Get connected components in reduced graph
    components = list(nx.connected_components(G_reduced))
    
    # Draw each component with a different color
    for i, comp in enumerate(components):
        color = plt.cm.tab20(i % 20)
        nx.draw_networkx_nodes(
            G_reduced, pos,
            nodelist=list(comp),
            node_color=[color] * len(comp),
            node_size=100,
            alpha=0.8,
            label=f"Component {i+1}" if i < 5 else ""  # Only label first 5 components
        )
    
    # Draw edges
    nx.draw_networkx_edges(G_reduced, pos, width=0.5, alpha=0.3, edge_color='gray')
    
    # Show removed nodes positions with X markers
    x_pos = [pos[node][0] for node in critical_nodes if node in pos]
    y_pos = [pos[node][1] for node in critical_nodes if node in pos]
    plt.scatter(x_pos, y_pos, s=200, c='red', marker='x', label="Removed Nodes", zorder=10)
    
    if len(components) > 5:
        plt.legend(title=f"Network splits into {len(components)} components", loc="upper right", ncol=2, fontsize=8)
    else:
        plt.legend(title=f"Network splits into {len(components)} components", loc="upper right", fontsize=10)
    
    plt.title("Network After Removing Critical Nodes", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_path, "network_after_removal.png"), dpi=300, bbox_inches='tight')

def calculate_telecom_centrality(G, gdf_rbs, output_path):
    """
    Calculates telecom-specific centrality metrics for RBS networks.
    These are adapted versions of standard centrality metrics that
    account for telecom-specific properties like coverage, capacity,
    and technology generation.
    
    Args:
        G (nx.Graph): NetworkX graph of RBS
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        output_path (str): Path to save results
        
    Returns:
        dict: Dictionary with telecom-specific centrality results
    """
    print("Calculating telecom-specific centrality metrics...")
    
    # Create coverage-weighted graph
    G_weighted = G.copy()
    
    # Assign weights based on coverage and technology capabilities
    tech_weights = {
        'GSM': 1.0,
        '3G': 2.0,
        '4G': 3.0,
        '5G': 4.0,
        'N/A': 1.0
    }
    
    # Calculate capacity-weighted betweenness centrality
    # First, assign capacity weights to edges based on technology
    for u, v, data in G_weighted.edges(data=True):
        # Get node technologies
        tech_u = G.nodes[u].get('technology', 'N/A')
        tech_v = G.nodes[v].get('technology', 'N/A')
        
        # Get coverage radii
        coverage_u = G.nodes[u].get('coverage_radius', 1.0)
        coverage_v = G.nodes[v].get('coverage_radius', 1.0)
        
        # Calculate edge weight as inverse of distance multiplied by technology factors
        tech_factor = (tech_weights.get(tech_u, 1.0) + tech_weights.get(tech_v, 1.0)) / 2
        coverage_factor = (coverage_u + coverage_v) / 2
        
        # Update edge weight (higher is better for capacity)
        if 'distance' in data:
            # Ensure distance is not zero to avoid division by zero
            distance = max(0.001, data['distance'])
            # Higher weight = better connection
            capacity_weight = (tech_factor * coverage_factor) / distance
            G_weighted[u][v]['capacity_weight'] = capacity_weight
        else:
            G_weighted[u][v]['capacity_weight'] = tech_factor * coverage_factor
    
    # Calculate capacity-weighted betweenness
    capacity_betweenness = nx.betweenness_centrality(
        G_weighted, 
        weight='capacity_weight', 
        normalized=True
    )
    
    # Calculate coverage-weighted eigenvector centrality
    # First, create adjacency matrix with coverage weighting
    for u, v, data in G_weighted.edges(data=True):
        coverage_u = G.nodes[u].get('coverage_radius', 1.0)
        coverage_v = G.nodes[v].get('coverage_radius', 1.0)
        
        # Weight by combined coverage
        coverage_weight = coverage_u * coverage_v
        G_weighted[u][v]['coverage_weight'] = coverage_weight
    
    # Calculate coverage-weighted eigenvector centrality
    try:
        coverage_eigenvector = nx.eigenvector_centrality_numpy(
            G_weighted,
            weight='coverage_weight',
            max_iter=1000
        )
    except:
        # Fallback if it doesn't converge
        coverage_eigenvector = {node: 0.0 for node in G_weighted.nodes()}
        print("Warning: Eigenvector centrality calculation did not converge.")
    
    # Calculate service area centrality (based on coverage area served)
    service_centrality = {}
    for node in G.nodes():
        # Get coverage radius
        coverage = G.nodes[node].get('coverage_radius', 1.0)
        # Approximate service area as π*r²
        service_area = np.pi * (coverage ** 2)
        # Weight by technology capability
        tech = G.nodes[node].get('technology', 'N/A')
        tech_factor = tech_weights.get(tech, 1.0)
        # Calculate service centrality
        service_centrality[node] = service_area * tech_factor
    
    # Normalize service centrality
    if service_centrality:
        max_service = max(service_centrality.values())
        if max_service > 0:
            service_centrality = {k: v/max_service for k, v in service_centrality.items()}
    
    # Add centrality values as node attributes
    nx.set_node_attributes(G, capacity_betweenness, 'capacity_betweenness')
    nx.set_node_attributes(G, coverage_eigenvector, 'coverage_eigenvector')
    nx.set_node_attributes(G, service_centrality, 'service_centrality')
    
    # Create telecom-specific centrality visualization
    _visualize_telecom_centrality(G, capacity_betweenness, coverage_eigenvector, service_centrality, output_path)
    
    # Prepare output results
    results = {
        'capacity_betweenness': {int(k): v for k, v in capacity_betweenness.items()},
        'coverage_eigenvector': {int(k): v for k, v in coverage_eigenvector.items()},
        'service_centrality': {int(k): v for k, v in service_centrality.items()}
    }
    
    # Save results to file
    with open(os.path.join(output_path, 'telecom_centrality.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def _visualize_telecom_centrality(G, capacity_betweenness, coverage_eigenvector, service_centrality, output_path):
    """
    Creates visualizations of telecom-specific centrality metrics.
    
    Args:
        G (nx.Graph): NetworkX graph
        capacity_betweenness (dict): Capacity-weighted betweenness centrality
        coverage_eigenvector (dict): Coverage-weighted eigenvector centrality
        service_centrality (dict): Service area centrality
        output_path (str): Path to save visualizations
    """
    # Get positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Capacity-weighted betweenness
    betweenness_values = list(capacity_betweenness.values())
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=betweenness_values,
        node_size=[v*1000+50 for v in betweenness_values],
        alpha=0.8,
        cmap=plt.cm.viridis,
        ax=ax1
    )
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, edge_color='gray', ax=ax1)
    ax1.set_title("Capacity-Weighted Betweenness", fontsize=12)
    plt.colorbar(nodes, ax=ax1, label="Centrality Value")
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # 2. Coverage-weighted eigenvector
    eigenvector_values = list(coverage_eigenvector.values())
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=eigenvector_values,
        node_size=[v*1000+50 for v in eigenvector_values],
        alpha=0.8,
        cmap=plt.cm.plasma,
        ax=ax2
    )
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, edge_color='gray', ax=ax2)
    ax2.set_title("Coverage-Weighted Eigenvector", fontsize=12)
    plt.colorbar(nodes, ax=ax2, label="Centrality Value")
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    # 3. Service area centrality
    service_values = list(service_centrality.values())
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=service_values,
        node_size=[v*1000+50 for v in service_values],
        alpha=0.8,
        cmap=plt.cm.inferno,
        ax=ax3
    )
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, edge_color='gray', ax=ax3)
    ax3.set_title("Service Area Centrality", fontsize=12)
    plt.colorbar(nodes, ax=ax3, label="Centrality Value")
    ax3.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_path, "telecom_centrality.png"), dpi=300, bbox_inches='tight') 

#############################################
# 3. COMPARATIVE ANALYSIS BETWEEN OPERATORS
#############################################

def compare_operator_networks(G, gdf_rbs, output_path):
    """
    Creates comparative visualizations showing differences in network
    strategies between different operators.
    
    Args:
        G (nx.Graph): NetworkX graph of RBS
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        output_path (str): Path to save the visualization
        
    Returns:
        dict: Dictionary with comparative metrics
    """
    print("Creating comparative analysis between operators...")
    
    # Check if operator field exists
    if 'Operator' not in gdf_rbs.columns:
        print("Error: 'Operator' field not found in the RBS data.")
        return None
    
    # Get list of operators
    operators = sorted(gdf_rbs['Operator'].unique())
    
    # Create separate graphs for each operator
    operator_graphs = {}
    
    for operator in operators:
        # Filter data for this operator
        operator_data = gdf_rbs[gdf_rbs['Operator'] == operator]
        
        # Get nodes for this operator
        operator_nodes = operator_data.index.tolist()
        
        # Create subgraph
        if operator_nodes:
            subgraph = G.subgraph(operator_nodes).copy()
            operator_graphs[operator] = subgraph
    
    # Calculate metrics for each operator
    comparative_metrics = {}
    
    for operator, graph in operator_graphs.items():
        # Skip if graph is empty
        if graph.number_of_nodes() == 0:
            continue
        
        # Calculate basic metrics
        metrics = {
            'number_of_rbs': graph.number_of_nodes(),
            'number_of_connections': graph.number_of_edges(),
            'graph_density': nx.density(graph),
            'average_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
        }
        
        # Calculate connected components
        components = list(nx.connected_components(graph))
        metrics['number_of_components'] = len(components)
        
        if components:
            largest_cc = max(components, key=len)
            largest_cc_graph = graph.subgraph(largest_cc).copy()
            
            # Calculate metrics on largest component
            if nx.is_connected(largest_cc_graph) and largest_cc_graph.number_of_nodes() > 1:
                try:
                    metrics['average_path_length'] = nx.average_shortest_path_length(largest_cc_graph)
                except:
                    metrics['average_path_length'] = float('nan')
                
                metrics['clustering_coefficient'] = nx.average_clustering(largest_cc_graph)
        
        # Count technology distribution
        tech_field = 'Tecnologia' if 'Tecnologia' in gdf_rbs.columns else 'Technology'
        if tech_field in gdf_rbs.columns:
            tech_counts = gdf_rbs[gdf_rbs['Operator'] == operator][tech_field].value_counts().to_dict()
            metrics['technology_distribution'] = tech_counts
        
        # Store metrics
        comparative_metrics[operator] = metrics
    
    # Create comparative visualizations
    _create_side_by_side_networks(operator_graphs, output_path)
    _create_efficiency_comparison(comparative_metrics, operator_graphs, gdf_rbs, output_path)
    _create_network_overlap_visualization(G, gdf_rbs, operators, output_path)
    
    # Save metrics to file
    with open(os.path.join(output_path, 'operator_comparative_metrics.json'), 'w') as f:
        # Convert any non-serializable values to strings
        serializable_metrics = {}
        for op, metrics in comparative_metrics.items():
            serializable_metrics[op] = {k: str(v) if not isinstance(v, (int, float, str, list, dict)) else v 
                                       for k, v in metrics.items()}
        json.dump(serializable_metrics, f, indent=4)
    
    return comparative_metrics

def _create_side_by_side_networks(operator_graphs, output_path):
    """
    Creates a side-by-side visualization of operator networks.
    
    Args:
        operator_graphs (dict): Dictionary of NetworkX graphs by operator
        output_path (str): Path to save the visualization
    """
    # Get number of operators
    num_operators = len(operator_graphs)
    
    if num_operators == 0:
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, num_operators, figsize=(5*num_operators, 8))
    
    # Handle case of single operator
    if num_operators == 1:
        axes = [axes]
    
    # Define colors for operators
    operator_colors = {
        'CLARO': '#E02020',
        'OI': '#FFD700',
        'VIVO': '#9932CC',
        'TIM': '#0000CD',
        'N/A': '#CCCCCC'
    }
    
    # Plot each operator's network
    for i, (operator, graph) in enumerate(operator_graphs.items()):
        ax = axes[i]
        
        # Get positions
        pos = nx.get_node_attributes(graph, 'pos')
        
        # If no positions, use spring layout
        if not pos:
            pos = nx.spring_layout(graph, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            graph, pos,
            node_color=operator_colors.get(operator, '#CCCCCC'),
            node_size=80,
            alpha=0.8,
            ax=ax
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            graph, pos,
            width=0.5,
            alpha=0.4,
            edge_color='gray',
            ax=ax
        )
        
        # Set title and grid
        ax.set_title(f"{operator} Network\n({graph.number_of_nodes()} RBS)", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Get technology distribution
        technologies = nx.get_node_attributes(graph, 'technology')
        tech_counts = {}
        
        for tech in technologies.values():
            tech_counts[tech] = tech_counts.get(tech, 0) + 1
        
        # Add tech distribution annotation
        tech_text = "\n".join([f"{tech}: {count}" for tech, count in tech_counts.items()])
        ax.text(0.05, 0.05, tech_text, transform=ax.transAxes, fontsize=10,
               bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "operator_network_comparison.png"), dpi=300, bbox_inches='tight')

def _create_efficiency_comparison(comparative_metrics, operator_graphs, gdf_rbs, output_path):
    """
    Creates a comparison of efficiency metrics between operators.
    
    Args:
        comparative_metrics (dict): Dictionary of metrics by operator
        operator_graphs (dict): Dictionary of NetworkX graphs by operator
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        output_path (str): Path to save the visualization
    """
    # Get operators with valid metrics
    valid_operators = [op for op in comparative_metrics.keys() if comparative_metrics[op]]
    
    if not valid_operators:
        return
    
    # Create figure for bar charts
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Define colors for operators
    operator_colors = {
        'CLARO': '#E02020',
        'OI': '#FFD700',
        'VIVO': '#9932CC',
        'TIM': '#0000CD',
        'N/A': '#CCCCCC'
    }
    
    # 1. Number of RBS
    ax = axes[0]
    values = [comparative_metrics[op].get('number_of_rbs', 0) for op in valid_operators]
    bars = ax.bar(valid_operators, values, color=[operator_colors.get(op, '#CCCCCC') for op in valid_operators])
    ax.set_title("Number of Radio Base Stations", fontsize=12)
    ax.set_ylabel("Count")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f"{int(height)}", ha='center', va='bottom', fontsize=10)
    
    # 2. Average Degree (Connectivity)
    ax = axes[1]
    values = [comparative_metrics[op].get('average_degree', 0) for op in valid_operators]
    bars = ax.bar(valid_operators, values, color=[operator_colors.get(op, '#CCCCCC') for op in valid_operators])
    ax.set_title("Average Connectivity (Degree)", fontsize=12)
    ax.set_ylabel("Average Connections per RBS")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f"{height:.2f}", ha='center', va='bottom', fontsize=10)
    
    # 3. Coverage Efficiency (if area data available)
    ax = axes[2]
    
    # Calculate coverage per km²
    coverage_values = []
    for operator in valid_operators:
        # Get RBS count
        rbs_count = comparative_metrics[operator].get('number_of_rbs', 0)
        
        # Estimate covered area from operator graph
        graph = operator_graphs[operator]
        
        # Get coverage radii
        coverage_radii = [graph.nodes[n].get('coverage_radius', 1.0) for n in graph.nodes()]
        
        # Estimate total coverage area (simple sum, not accounting for overlap)
        total_coverage = sum([np.pi * (r**2) for r in coverage_radii])
        
        # Coverage efficiency: RBS per 100 km²
        if total_coverage > 0:
            efficiency = rbs_count / (total_coverage / 100)
        else:
            efficiency = 0
        
        coverage_values.append(efficiency)
    
    bars = ax.bar(valid_operators, coverage_values, color=[operator_colors.get(op, '#CCCCCC') for op in valid_operators])
    ax.set_title("Coverage Efficiency", fontsize=12)
    ax.set_ylabel("RBS per 100 km² of Coverage")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f"{height:.2f}", ha='center', va='bottom', fontsize=10)
    
    # 4. Technology Distribution
    ax = axes[3]
    
    # Get technology distribution for each operator
    tech_distribution = {}
    for operator in valid_operators:
        if 'technology_distribution' in comparative_metrics[operator]:
            tech_distribution[operator] = comparative_metrics[operator]['technology_distribution']
    
    # Organize data for stacked bar chart
    if tech_distribution:
        # Get all unique technologies
        all_techs = set()
        for dist in tech_distribution.values():
            all_techs.update(dist.keys())
        
        all_techs = sorted(all_techs)
        
        # Create data for each technology
        tech_data = {}
        for tech in all_techs:
            tech_data[tech] = [tech_distribution.get(op, {}).get(tech, 0) for op in valid_operators]
        
        # Create stacked bar chart
        bottom = np.zeros(len(valid_operators))
        for tech in all_techs:
            ax.bar(valid_operators, tech_data[tech], bottom=bottom, label=tech)
            bottom += tech_data[tech]
        
        ax.set_title("Technology Distribution", fontsize=12)
        ax.set_ylabel("Count")
        ax.legend(title="Technology")
    else:
        ax.text(0.5, 0.5, "Technology data not available", ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "operator_efficiency_comparison.png"), dpi=300, bbox_inches='tight')

def _create_network_overlap_visualization(G, gdf_rbs, operators, output_path):
    """
    Creates a visualization showing how networks of different operators
    overlap or complement each other.
    
    Args:
        G (nx.Graph): NetworkX graph of RBS
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        operators (list): List of operators to compare
        output_path (str): Path to save the visualization
    """
    # Create a combined visualization showing areas of overlap
    if PLOTLY_AVAILABLE:
        # Create interactive folium map for better overlap visualization
        _create_interactive_overlap_map(gdf_rbs, operators, output_path)
    
    # Create a static matplotlib visualization
    plt.figure(figsize=(14, 10))
    
    # Define colors for operators
    operator_colors = {
        'CLARO': '#E02020',
        'OI': '#FFD700',
        'VIVO': '#9932CC',
        'TIM': '#0000CD',
        'N/A': '#CCCCCC'
    }
    
    # Create a custom colormap for overlap areas
    overlap_cmap = LinearSegmentedColormap.from_list(
        'overlap_map', ['#FFFFFF', '#FF00FF', '#800080'], N=100
    )
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Group nodes by operator
    operator_nodes = {}
    for operator in operators:
        operator_nodes[operator] = [n for n, d in G.nodes(data=True) if d.get('operator') == operator]
    
    # Draw nodes by operator with transparent circles for coverage
    for operator, nodes in operator_nodes.items():
        if not nodes:
            continue
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodes,
            node_color=operator_colors.get(operator, '#CCCCCC'),
            node_size=80,
            alpha=0.8,
            label=operator
        )
        
        # Draw coverage circles with transparency
        for node in nodes:
            if node in pos:
                x, y = pos[node]
                coverage = G.nodes[node].get('coverage_radius', 1.0)
                
                # Convert from km to degrees (approximate)
                coverage_degree = coverage / 111.0  # Rough conversion: 1 degree ≈ 111 km
                
                # Draw coverage circle
                circle = plt.Circle((x, y), coverage_degree, color=operator_colors.get(operator, '#CCCCCC'), 
                                 alpha=0.2, fill=True)
                plt.gca().add_patch(circle)
    
    # Draw edges with light colors
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.2, edge_color='gray')
    
    # Add legend
    plt.legend(title="Operators")
    
    # Calculate overlap areas and annotate
    overlap_text = "Estimated Coverage Overlap:\n"
    for op1, op2 in combinations(operators, 2):
        if op1 not in operator_nodes or op2 not in operator_nodes:
            continue
            
        nodes1 = operator_nodes[op1]
        nodes2 = operator_nodes[op2]
        
        if not nodes1 or not nodes2:
            continue
        
        # Count edges between operators (indication of overlap)
        overlap_count = 0
        for n1 in nodes1:
            for n2 in nodes2:
                if G.has_edge(n1, n2):
                    overlap_count += 1
        
        overlap_text += f"{op1}-{op2}: {overlap_count} overlapping connections\n"
    
    # Add overlap annotation
    plt.text(0.02, 0.02, overlap_text, transform=plt.gca().transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title("Network Overlap Between Operators", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_path, "operator_network_overlap.png"), dpi=300, bbox_inches='tight')

def _create_interactive_overlap_map(gdf_rbs, operators, output_path):
    """
    Creates an interactive map showing coverage overlap between operators.
    
    Args:
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        operators (list): List of operators to compare
        output_path (str): Path to save the visualization
    """
    # Calculate center of the map
    center_lat = gdf_rbs['Latitude'].mean()
    center_lon = gdf_rbs['Longitude'].mean()
    
    # Create base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Define colors for operators
    operator_colors = {
        'CLARO': '#E02020',
        'OI': '#FFD700',
        'VIVO': '#9932CC',
        'TIM': '#0000CD',
        'N/A': '#CCCCCC'
    }
    
    # Create feature groups for each operator
    operator_groups = {}
    for operator in operators:
        operator_groups[operator] = folium.FeatureGroup(name=operator)
    
    # Add RBS points and coverage circles for each operator
    for operator in operators:
        operator_data = gdf_rbs[gdf_rbs['Operator'] == operator]
        
        # Skip if no data for this operator
        if operator_data.empty:
            continue
        
        # Get color for this operator
        color = operator_colors.get(operator, '#CCCCCC')
        
        # Add points and circles
        for idx, row in operator_data.iterrows():
            # Add marker for RBS
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=f"ID: {idx}<br>Operator: {operator}<br>Technology: {row.get('Tecnologia', 'N/A')}"
            ).add_to(operator_groups[operator])
            
            # Add coverage circle
            coverage_radius = row.get('Coverage_Radius_km', 1.0) * 1000  # Convert to meters
            folium.Circle(
                location=[row['Latitude'], row['Longitude']],
                radius=coverage_radius,
                color=color,
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.1,
                popup=f"Coverage for RBS {idx}"
            ).add_to(operator_groups[operator])
    
    # Add feature groups to map
    for operator, group in operator_groups.items():
        group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    m.save(os.path.join(output_path, "interactive_operator_overlap.html"))

def run_advanced_graph_analysis(gdf_rbs, output_path, time_field=None):
    """
    Main function to run all advanced graph analysis methods.
    
    Args:
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        output_path (str): Path to save the results
        time_field (str, optional): Field containing timestamp for temporal analysis
        
    Returns:
        dict: Dictionary with analysis results
    """
    print("Running advanced graph analysis...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Create RBS graph
    from graph_analysis import create_rbs_graph
    G = create_rbs_graph(gdf_rbs, connection_radius=3.0, weighted=True)
    
    results = {}
    
    # 1. Advanced Graph Visualizations
    try:
        print("Creating advanced graph visualizations...")
        telecom_layout = create_telecom_layout(G, gdf_rbs, output_path)
        results['telecom_layout_created'] = True
        
        # Create multilayer visualization
        multilayer_graph = create_multilayer_graph(G, gdf_rbs, output_path)
        results['multilayer_graph_created'] = True
        
        # Create dynamic visualization if time field is provided
        if time_field and time_field in gdf_rbs.columns:
            dynamic_viz = create_dynamic_network_visualization(G, gdf_rbs, time_field, output_path)
            results['dynamic_visualization_created'] = True
    except Exception as e:
        print(f"Error in advanced graph visualizations: {e}")
        results['visualization_error'] = str(e)
    
    # 2. Advanced Network Metrics
    try:
        print("Calculating advanced network metrics...")
        
        # Community detection
        community_results = analyze_communities(G, gdf_rbs, output_path)
        results['community_analysis'] = {
            'num_communities': community_results['num_communities']
        }
        
        # Vulnerability analysis
        vulnerability_results = analyze_vulnerability(G, gdf_rbs, output_path)
        results['vulnerability_analysis'] = {
            'critical_nodes': vulnerability_results['network_vulnerability']['critical_nodes'][:3]  # First 3 critical nodes
        }
        
        # Telecom-specific centrality
        centrality_results = calculate_telecom_centrality(G, gdf_rbs, output_path)
        results['telecom_centrality'] = True
    except Exception as e:
        print(f"Error in advanced network metrics: {e}")
        results['metrics_error'] = str(e)
    
    # 3. Comparative Analysis Between Operators
    try:
        print("Creating operator comparative analysis...")
        comparative_results = compare_operator_networks(G, gdf_rbs, output_path)
        
        if comparative_results:
            results['comparative_analysis'] = {
                'operators_analyzed': list(comparative_results.keys())
            }
    except Exception as e:
        print(f"Error in comparative analysis: {e}")
        results['comparative_error'] = str(e)
    
    print("Advanced graph analysis completed.")
    return results 