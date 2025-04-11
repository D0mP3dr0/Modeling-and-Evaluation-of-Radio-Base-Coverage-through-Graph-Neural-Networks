"""
Module for modeling and analysis of RBS using graph theory and GNNs.
This module allows transforming RBS data into graphs, calculating graph metrics and
analyzing connectivity between stations.
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString
import seaborn as sns
import os
from scipy.spatial import Voronoi

# Check if PyTorch and PyG are available
try:
    import torch
    import torch_geometric
    from torch_geometric.data import Data as PyGData
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch and/or PyTorch Geometric not found. GNN functionalities will not be available.")

def create_rbs_graph(gdf_rbs, connection_radius=3.0, weighted=True):
    """
    Creates a NetworkX graph where nodes are RBS and edges represent connectivity.
    
    Args:
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        connection_radius (float): Maximum radius (km) to consider two RBS connected
        weighted (bool): If True, adds weights to edges based on distance
        
    Returns:
        G (nx.Graph): NetworkX graph representing the RBS network
    """
    print(f"Creating RBS graph with connection radius of {connection_radius} km...")
    
    # Create empty graph
    G = nx.Graph()
    
    # Add nodes (RBS)
    for idx, row in gdf_rbs.iterrows():
        # Define node attributes
        node_attrs = {
            'pos': (row['Longitude'], row['Latitude']),
            'operator': row.get('Operator', 'N/A'),
            'technology': row.get('Tecnologia', 'N/A'),
            'tx_freq': row.get('FreqTxMHz', 0),
            'power': row.get('PotenciaTransmissorWatts', 0),
            'gain': row.get('GanhoAntena', 0),
            'coverage_radius': row.get('Coverage_Radius_km', 0)
        }
        
        # Add to graph
        G.add_node(idx, **node_attrs)
    
    # Add edges (connections between RBS)
    nodes = list(G.nodes())
    edge_counter = 0
    
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            node_i = nodes[i]
            node_j = nodes[j]
            
            # Get positions
            pos_i = G.nodes[node_i]['pos']
            pos_j = G.nodes[node_j]['pos']
            
            # Calculate approximate distance in km (Haversine distance)
            lon1, lat1 = pos_i
            lon2, lat2 = pos_j
            
            # Convert to radians
            lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distance = 6371 * c  # Earth radius in km
            
            # If distance is less than connection radius, add edge
            if distance <= connection_radius:
                # Define edge weight (inverse of distance)
                if weighted:
                    weight = 1.0 / max(0.1, distance)  # Avoid division by zero
                else:
                    weight = 1.0
                
                # Get coverage radii of nodes
                radius_i = G.nodes[node_i]['coverage_radius']
                radius_j = G.nodes[node_j]['coverage_radius']
                
                # Check coverage overlap
                overlap = False
                if radius_i > 0 and radius_j > 0:
                    overlap = distance < (radius_i + radius_j)
                
                # Add edge with attributes
                G.add_edge(node_i, node_j, weight=weight, distance=distance, overlap=overlap)
                edge_counter += 1
    
    print(f"Graph created with {len(G.nodes())} nodes and {edge_counter} edges.")
    return G

def calculate_graph_metrics(G):
    """
    Calculates network metrics for the RBS graph.
    
    Args:
        G (nx.Graph): NetworkX graph of RBS
        
    Returns:
        dict: Dictionary with calculated metrics
    """
    print("Calculating graph metrics...")
    
    metrics = {}
    
    # Number of nodes and edges
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    
    # Graph density
    metrics['density'] = nx.density(G)
    
    # Connected components
    components = list(nx.connected_components(G))
    metrics['num_components'] = len(components)
    metrics['largest_component_size'] = len(max(components, key=len))
    
    # Average distance and diameter
    if nx.is_connected(G):
        metrics['average_distance'] = nx.average_shortest_path_length(G, weight='distance')
        metrics['diameter'] = nx.diameter(G, e=None, weight='distance')
    else:
        # Calculate only for the largest component
        largest_component = max(components, key=len)
        subgraph = G.subgraph(largest_component).copy()
        metrics['average_distance'] = nx.average_shortest_path_length(subgraph, weight='distance')
        metrics['diameter'] = nx.diameter(subgraph, e=None, weight='distance')
    
    # Clustering coefficient
    metrics['average_clustering'] = nx.average_clustering(G)
    
    # Centrality
    betweenness = nx.betweenness_centrality(G, weight='distance')
    metrics['betweenness_max'] = max(betweenness.values())
    metrics['betweenness_median'] = np.median(list(betweenness.values()))
    
    # Add centralities as node attributes
    nx.set_node_attributes(G, betweenness, 'betweenness')
    
    # Degree distribution
    degrees = [d for n, d in G.degree()]
    metrics['min_degree'] = min(degrees)
    metrics['max_degree'] = max(degrees)
    metrics['average_degree'] = sum(degrees) / len(degrees)
    
    print("Metrics calculated successfully.")
    return metrics

def visualize_graph(G, output_path, title="Connectivity Graph between RBS", 
                     by_operator=True, show_weights=False):
    """
    Visualizes the RBS graph.
    
    Args:
        G (nx.Graph): NetworkX graph of RBS
        output_path (str): Path to save the visualization
        title (str): Chart title
        by_operator (bool): If True, color nodes by operator
        show_weights (bool): If True, show edge weights
    """
    print("Creating graph visualization...")
    
    plt.figure(figsize=(14, 10))
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Define colors for operators
    operator_colors = {
        'CLARO': '#E02020',
        'OI': '#FFD700',
        'VIVO': '#9932CC',
        'TIM': '#0000CD',
        'N/A': '#CCCCCC'
    }
    
    if by_operator:
        # Group nodes by operator
        operators = nx.get_node_attributes(G, 'operator')
        operator_groups = {}
        
        for node, operator in operators.items():
            if operator not in operator_groups:
                operator_groups[operator] = []
            operator_groups[operator].append(node)
        
        # Draw nodes by operator
        for operator, nodes in operator_groups.items():
            color = operator_colors.get(operator, '#CCCCCC')
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, 
                                  node_size=100, alpha=0.8, label=operator)
    else:
        # Color by centrality
        betweenness = nx.get_node_attributes(G, 'betweenness')
        
        if betweenness:
            node_colors = [betweenness.get(node, 0) for node in G.nodes()]
            nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                         node_size=100, alpha=0.8, cmap=plt.cm.viridis)
            plt.colorbar(nodes, label='Betweenness Centrality')
        else:
            nx.draw_networkx_nodes(G, pos, node_color='skyblue', 
                                  node_size=100, alpha=0.8)
    
    # Draw edges
    if show_weights:
        # Get edge weights
        weights = nx.get_edge_attributes(G, 'weight').values()
        # Normalize thicknesses
        edge_widths = [w * 2 for w in weights]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color='gray')
    else:
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, edge_color='gray')
    
    # Add labels only for important nodes
    betweenness = nx.get_node_attributes(G, 'betweenness')
    important_nodes = []
    
    if betweenness:
        # Select the 5% most important nodes
        threshold = np.percentile(list(betweenness.values()), 95)
        important_nodes = [n for n, c in betweenness.items() if c >= threshold]
        
        labels = {n: f"{n}" for n in important_nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')
    
    # Add legend for operators
    if by_operator:
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, 
                            label=operator) for operator, color in operator_colors.items()]
        plt.legend(handles=handles, title="Operators", loc='best')
    
    # Add title and information
    plt.title(title, fontsize=16)
    plt.text(0.01, 0.01, f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}", 
            transform=plt.gca().transAxes, fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graph visualization saved to {output_path}")

def convert_to_pyg(G):
    """
    Converts a NetworkX graph to PyTorch Geometric format.
    
    Args:
        G (nx.Graph): NetworkX graph of RBS
        
    Returns:
        PyGData or None: PyG Data object if PyTorch is available, otherwise None
    """
    if not TORCH_AVAILABLE:
        print("PyTorch or PyTorch Geometric not found. Unable to convert to PyG format.")
        return None
    
    print("Converting graph to PyTorch Geometric format...")
    
    # Map nodes to contiguous indices
    nodes_to_idx = {n: i for i, n in enumerate(G.nodes())}
    
    # Create edge list
    edge_index = []
    edge_attr = []
    
    for u, v, data in G.edges(data=True):
        edge_index.append([nodes_to_idx[u], nodes_to_idx[v]])
        edge_index.append([nodes_to_idx[v], nodes_to_idx[u]])  # Add edge in both directions
        
        # Edge attributes
        weight = data.get('weight', 1.0)
        distance = data.get('distance', 0.0)
        overlap = 1.0 if data.get('overlap', False) else 0.0
        
        edge_attr.append([weight, distance, overlap])
        edge_attr.append([weight, distance, overlap])  # Duplicate for reverse edge
    
    # Convert to PyTorch tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create features for nodes
    features = []
    y = []  # Classes/labels (example: operator as label)
    
    operator_map = {'CLARO': 0, 'OI': 1, 'VIVO': 2, 'TIM': 3, 'N/A': 4}
    
    for node in G.nodes():
        attrs = G.nodes[node]
        
        # Feature vector: [power, gain, coverage_radius, tx_freq, centrality]
        power = float(attrs.get('power', 0))
        gain = float(attrs.get('gain', 0))
        radius = float(attrs.get('coverage_radius', 0))
        freq = float(attrs.get('tx_freq', 0))
        betweenness = float(attrs.get('betweenness', 0))
        
        features.append([power, gain, radius, freq, betweenness])
        
        # Use operator as example label
        operator = attrs.get('operator', 'N/A')
        y.append(operator_map.get(operator, 4))
    
    # Convert to PyTorch tensors
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    
    # Create PyG Data object
    data = PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    print(f"Graph converted to PyG: {data}")
    return data

def create_voronoi_rbs_graph(gdf_rbs, output_path, bound_factor=0.1):
    """
    Creates a graph based on the Voronoi diagram of RBS.
    
    Args:
        gdf_rbs (GeoDataFrame): GeoDataFrame containing RBS data
        output_path (str): Path to save the visualization
        bound_factor (float): Factor to extend Voronoi diagram boundaries
        
    Returns:
        nx.Graph: NetworkX graph based on Voronoi cells
    """
    print("Creating graph based on Voronoi diagram...")
    
    # Extract points
    points = np.array([(r['Longitude'], r['Latitude']) for _, r in gdf_rbs.iterrows()])
    
    # Calculate limits with margin
    x_min, y_min = np.min(points, axis=0) - bound_factor
    x_max, y_max = np.max(points, axis=0) + bound_factor
    
    # Add points at corners to close the diagram
    far_points = np.array([
        [x_min, y_min],
        [x_min, y_max],
        [x_max, y_min],
        [x_max, y_max]
    ])
    
    all_points = np.vstack([points, far_points])
    
    # Calculate Voronoi diagram
    vor = Voronoi(all_points)
    
    # Create graph
    G = nx.Graph()
    
    # Add RBS nodes
    for i, (_, row) in enumerate(gdf_rbs.iterrows()):
        G.add_node(i, pos=(row['Longitude'], row['Latitude']), 
                  operator=row.get('Operator', 'N/A'),
                  coverage_radius=row.get('Coverage_Radius_km', 0))
    
    # Add edges based on Voronoi cell adjacency
    for i, j in vor.ridge_points:
        # Ignore edges connected to points at corners
        if i >= len(points) or j >= len(points):
            continue
        
        # Calculate edge length (distance between RBS)
        p1 = all_points[i]
        p2 = all_points[j]
        
        # Convert to radians
        lon1, lat1 = p1
        lon2, lat2 = p2
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = 6371 * c  # Earth radius in km
        
        # Add edge
        G.add_edge(i, j, weight=1.0/max(0.1, distance), distance=distance)
    
    # Visualize
    plt.figure(figsize=(14, 10))
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Define colors for operators
    operator_colors = {
        'CLARO': '#E02020',
        'OI': '#FFD700',
        'VIVO': '#9932CC',
        'TIM': '#0000CD',
        'N/A': '#CCCCCC'
    }
    
    # Group nodes by operator
    operators = nx.get_node_attributes(G, 'operator')
    operator_groups = {}
    
    for node, operator in operators.items():
        if operator not in operator_groups:
            operator_groups[operator] = []
        operator_groups[operator].append(node)
    
    # Draw Voronoi cells
    voronoi_plot_2d(vor, show_points=False, show_vertices=False, 
                   line_colors='gray', line_width=0.5, line_alpha=0.4, ax=plt.gca())
    
    # Draw nodes by operator
    for operator, nodes in operator_groups.items():
        color = operator_colors.get(operator, '#CCCCCC')
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, 
                              node_size=100, alpha=0.8, label=operator)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.8, alpha=0.4, edge_color='gray')
    
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, 
                        label=operator) for operator, color in operator_colors.items()]
    plt.legend(handles=handles, title="Operators", loc='best')
    
    plt.title("RBS Graph based on Voronoi Cells", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Voronoi graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print(f"Voronoi graph visualization saved to {output_path}")
    
    return G
