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
import warnings

# Try to import voronoi_plot_2d, but don't fail if not available
try:
    from scipy.spatial import voronoi_plot_2d
    VORONOI_PLOT_AVAILABLE = True
except ImportError:
    VORONOI_PLOT_AVAILABLE = False
    warnings.warn("scipy.spatial.voronoi_plot_2d not available. Voronoi visualization will be limited.")

# Check if PyTorch and PyG are available
try:
    import torch
    import torch_geometric
    from torch_geometric.data import Data as PyGData
    TORCH_AVAILABLE = True
    
    # Check if GPU is available and set up accordingly
    if torch.cuda.is_available() and os.environ.get('USE_GPU', '').lower() == 'true':
        DEVICE = torch.device('cuda')
        print(f"PyTorch will use GPU: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device('cpu')
        print("PyTorch will use CPU")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch and/or PyTorch Geometric not found. GNN functionalities will not be available.")

def validate_geometry(gdf):
    """
    Validates and cleans geometry in a GeoDataFrame.
    
    Args:
        gdf (GeoDataFrame): GeoDataFrame to validate
        
    Returns:
        GeoDataFrame: Validated GeoDataFrame
    """
    # Check if gdf is a GeoDataFrame
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("Input must be a GeoDataFrame")
    
    # Check if geometry column exists
    if 'geometry' not in gdf.columns:
        raise ValueError("GeoDataFrame has no 'geometry' column")
    
    # Remove rows with invalid or None geometries
    valid_geom = gdf[~gdf.geometry.isna() & gdf.geometry.is_valid].copy()
    
    # Check if any rows were removed
    if len(valid_geom) < len(gdf):
        warnings.warn(f"Removed {len(gdf) - len(valid_geom)} rows with invalid or missing geometries")
    
    return valid_geom

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
    
    # Validate geometry
    try:
        gdf_rbs = validate_geometry(gdf_rbs)
        if len(gdf_rbs) == 0:
            raise ValueError("No valid geometries found in input data")
    except (TypeError, ValueError) as e:
        print(f"Error: {e}")
        return nx.Graph()
    
    # Create empty graph
    G = nx.Graph()
    
    # Add nodes (RBS)
    for idx, row in gdf_rbs.iterrows():
        try:
            # Get coordinates from geometry
            if hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
                lon, lat = row.geometry.x, row.geometry.y
            # Alternatively try to get from Longitude/Latitude columns
            elif 'Longitude' in row and 'Latitude' in row:
                lon, lat = row['Longitude'], row['Latitude']
            else:
                print(f"Warning: Cannot extract coordinates for node {idx}. Skipping.")
                continue
                
            # Define node attributes
            node_attrs = {
                'pos': (lon, lat),
                'operator': row.get('Operator', 'N/A'),
                'technology': row.get('Tecnologia', 'N/A'),
                'tx_freq': row.get('FreqTxMHz', 0),
                'power': row.get('PotenciaTransmissorWatts', 0),
                'gain': row.get('GanhoAntena', 0),
                'coverage_radius': row.get('Coverage_Radius_km', 0)
            }
            
            # Add to graph
            G.add_node(idx, **node_attrs)
        except Exception as e:
            print(f"Error adding node {idx}: {e}")
    
    # Add edges (connections between RBS)
    nodes = list(G.nodes())
    edge_counter = 0
    
    # Vectorized approach to calculate all pairwise distances
    # This is more efficient than nested loops for large datasets
    if len(nodes) > 1000:
        # For very large datasets, use a more memory-efficient approach
        edge_counter = _add_edges_batch_processing(G, nodes, connection_radius, weighted)
    else:
        # For smaller datasets, we can calculate all pairs at once
        edge_counter = _add_edges_all_pairs(G, nodes, connection_radius, weighted)
    
    print(f"Graph created with {len(G.nodes())} nodes and {edge_counter} edges.")
    return G

def _add_edges_all_pairs(G, nodes, connection_radius, weighted):
    """
    Add edges between all pairs of nodes in a single calculation.
    More efficient for smaller datasets.
    
    Args:
        G (nx.Graph): NetworkX graph
        nodes (list): List of node IDs
        connection_radius (float): Maximum radius (km) for connection
        weighted (bool): Whether to add weights to edges
        
    Returns:
        int: Number of edges added
    """
    edge_counter = 0
    
    # Get all positions
    positions = np.array([G.nodes[n]['pos'] for n in nodes])
    
    # Convert to radians for Haversine formula
    lon_rad = np.radians(positions[:, 0]).reshape(-1, 1)
    lat_rad = np.radians(positions[:, 1]).reshape(-1, 1)
    
    # Calculate differences for all pairs
    dlon = lon_rad - lon_rad.T
    dlat = lat_rad - lat_rad.T
    
    # Haversine formula (vectorized)
    a = np.sin(dlat/2)**2 + np.cos(lat_rad) * np.cos(lat_rad.T) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    distances = 6371 * c  # Earth radius in km
    
    # Find pairs within connection radius
    connected_pairs = np.where((distances <= connection_radius) & (distances > 0))
    
    # Add edges
    for i, j in zip(connected_pairs[0], connected_pairs[1]):
        if i < j:  # Only add each edge once
            node_i = nodes[i]
            node_j = nodes[j]
            distance = distances[i, j]
            
            # Calculate weight
            if weighted:
                weight = 1.0 / max(0.1, distance)  # Avoid division by zero
            else:
                weight = 1.0
            
            # Get coverage radii of nodes
            radius_i = G.nodes[node_i].get('coverage_radius', 0)
            radius_j = G.nodes[node_j].get('coverage_radius', 0)
            
            # Check coverage overlap
            overlap = False
            if radius_i > 0 and radius_j > 0:
                overlap = distance < (radius_i + radius_j)
            
            # Add edge with attributes
            G.add_edge(node_i, node_j, weight=weight, distance=distance, overlap=overlap)
            edge_counter += 1
    
    return edge_counter

def _add_edges_batch_processing(G, nodes, connection_radius, weighted):
    """
    Add edges in batches to save memory for large datasets.
    
    Args:
        G (nx.Graph): NetworkX graph
        nodes (list): List of node IDs
        connection_radius (float): Maximum radius (km) for connection
        weighted (bool): Whether to add weights to edges
        
    Returns:
        int: Number of edges added
    """
    edge_counter = 0
    batch_size = 500
    
    for i in range(0, len(nodes), batch_size):
        batch_i = nodes[i:i+batch_size]
        positions_i = np.array([G.nodes[n]['pos'] for n in batch_i])
        lon_i_rad = np.radians(positions_i[:, 0])
        lat_i_rad = np.radians(positions_i[:, 1])
        
        # For each batch i, compare with all nodes j where j > i
        for j in range(i, len(nodes), batch_size):
            batch_j = nodes[j:j+batch_size]
            positions_j = np.array([G.nodes[n]['pos'] for n in batch_j])
            lon_j_rad = np.radians(positions_j[:, 0])
            lat_j_rad = np.radians(positions_j[:, 1])
            
            # Calculate distances between batch i and batch j
            for idx_i, node_i in enumerate(batch_i):
                start_j = idx_i + 1 if i == j else 0  # Avoid duplicate edges
                
                for idx_j in range(start_j, len(batch_j)):
                    node_j = batch_j[idx_j]
                    
                    # Calculate Haversine distance
                    dlon = lon_j_rad[idx_j] - lon_i_rad[idx_i]
                    dlat = lat_j_rad[idx_j] - lat_i_rad[idx_i]
                    a = np.sin(dlat/2)**2 + np.cos(lat_i_rad[idx_i]) * np.cos(lat_j_rad[idx_j]) * np.sin(dlon/2)**2
                    c = 2 * np.arcsin(np.sqrt(a))
                    distance = 6371 * c  # Earth radius in km
                    
                    # If distance is less than connection radius, add edge
                    if distance <= connection_radius:
                        # Define edge weight
                        if weighted:
                            weight = 1.0 / max(0.1, distance)  # Avoid division by zero
                        else:
                            weight = 1.0
                        
                        # Get coverage radii of nodes
                        radius_i = G.nodes[node_i].get('coverage_radius', 0)
                        radius_j = G.nodes[node_j].get('coverage_radius', 0)
                        
                        # Check coverage overlap
                        overlap = False
                        if radius_i > 0 and radius_j > 0:
                            overlap = distance < (radius_i + radius_j)
                        
                        # Add edge with attributes
                        G.add_edge(node_i, node_j, weight=weight, distance=distance, overlap=overlap)
                        edge_counter += 1
    
    return edge_counter

def calculate_graph_metrics(G):
    """
    Calculates network metrics for the RBS graph.
    
    Args:
        G (nx.Graph): NetworkX graph of RBS
        
    Returns:
        dict: Dictionary with calculated metrics
    """
    print("Calculating graph metrics...")
    
    if G.number_of_nodes() == 0:
        print("Warning: Graph has no nodes. Cannot calculate metrics.")
        return {}
    
    metrics = {}
    
    # Number of nodes and edges
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    
    # Graph density
    metrics['density'] = nx.density(G)
    
    # Connected components
    components = list(nx.connected_components(G))
    metrics['num_components'] = len(components)
    metrics['largest_component_size'] = len(max(components, key=len)) if components else 0
    metrics['largest_component_percentage'] = (metrics['largest_component_size'] / metrics['num_nodes'] * 100) if metrics['num_nodes'] > 0 else 0
    
    # Process the largest component
    largest_component = max(components, key=len) if components else set()
    largest_subgraph = G.subgraph(largest_component).copy()
    
    # Average distance and diameter - only for the largest component
    if nx.is_connected(largest_subgraph) and largest_subgraph.number_of_nodes() > 1:
        try:
            metrics['average_distance'] = nx.average_shortest_path_length(largest_subgraph, weight='distance')
            metrics['diameter'] = nx.diameter(largest_subgraph, e=None, weight='distance')
        except nx.NetworkXError as e:
            print(f"Warning: Error calculating path metrics: {e}")
            metrics['average_distance'] = float('nan')
            metrics['diameter'] = float('nan')
    else:
        metrics['average_distance'] = float('nan')
        metrics['diameter'] = float('nan')
    
    # Clustering coefficient
    metrics['average_clustering'] = nx.average_clustering(G)
    
    # Centrality - calculated per component for better accuracy
    try:
        # Initialize dictionaries for centrality measures
        betweenness = {}
        eigenvector = {}
        degree_centrality = {}
        
        # Calculate for each connected component
        for component in components:
            if len(component) > 1:  # Only meaningful for components with at least 2 nodes
                subgraph = G.subgraph(component).copy()
                
                # Betweenness centrality
                comp_betweenness = nx.betweenness_centrality(subgraph, weight='distance')
                betweenness.update(comp_betweenness)
                
                # Eigenvector centrality
                try:
                    comp_eigenvector = nx.eigenvector_centrality(subgraph, weight='weight', max_iter=1000)
                    eigenvector.update(comp_eigenvector)
                except nx.PowerIterationFailedConvergence:
                    # Fall back to simpler method if eigenvector calculation fails
                    print("Warning: Eigenvector centrality calculation failed to converge.")
                    for node in component:
                        eigenvector[node] = 0
                
                # Degree centrality
                comp_degree = nx.degree_centrality(subgraph)
                degree_centrality.update(comp_degree)
            else:
                # For isolated nodes, set centrality to 0
                for node in component:
                    betweenness[node] = 0
                    eigenvector[node] = 0
                    degree_centrality[node] = 0
        
        # Add centrality metrics to results
        if betweenness:
            metrics['betweenness_max'] = max(betweenness.values())
            metrics['betweenness_median'] = np.median(list(betweenness.values()))
            metrics['betweenness_mean'] = np.mean(list(betweenness.values()))
            
            # Add node attributes
            nx.set_node_attributes(G, betweenness, 'betweenness')
            nx.set_node_attributes(G, eigenvector, 'eigenvector')
            nx.set_node_attributes(G, degree_centrality, 'degree_centrality')
    except Exception as e:
        print(f"Warning: Error calculating centrality metrics: {e}")
        metrics['betweenness_max'] = float('nan')
        metrics['betweenness_median'] = float('nan')
        metrics['betweenness_mean'] = float('nan')
    
    # Degree distribution
    degrees = [d for n, d in G.degree()]
    if degrees:
        metrics['min_degree'] = min(degrees)
        metrics['max_degree'] = max(degrees)
        metrics['average_degree'] = sum(degrees) / len(degrees)
    else:
        metrics['min_degree'] = 0
        metrics['max_degree'] = 0
        metrics['average_degree'] = 0
    
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
    
    if G.number_of_nodes() == 0:
        print("Warning: Graph has no nodes. Cannot create visualization.")
        return
    
    plt.figure(figsize=(14, 10))
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        print("Warning: Graph does not have position attributes. Using spring layout.")
        pos = nx.spring_layout(G)
    
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
        weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
        # Normalize thicknesses
        max_weight = max(weights) if weights else 1.0
        edge_widths = [w / max_weight * 3 for w in weights]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color='gray')
    else:
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, edge_color='gray')
    
    # Add labels only for important nodes
    betweenness = nx.get_node_attributes(G, 'betweenness')
    important_nodes = []
    
    if betweenness and G.number_of_nodes() > 5:
        # Select the 5% most important nodes
        threshold = np.percentile(list(betweenness.values()), 95)
        important_nodes = [n for n, c in betweenness.items() if c >= threshold]
        
        labels = {n: f"{n}" for n in important_nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')
    elif G.number_of_nodes() <= 5:
        # For small graphs, label all nodes
        labels = {n: f"{n}" for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')
    
    # Add legend for operators
    if by_operator:
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, 
                            label=operator) for operator, color in operator_colors.items() 
                  if operator in operator_groups]
        if handles:
            plt.legend(handles=handles, title="Operators", loc='best')
    
    # Add title and information
    plt.title(title, fontsize=16)
    plt.text(0.01, 0.01, f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}", 
            transform=plt.gca().transAxes, fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis('off')
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
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
    
    if G.number_of_nodes() == 0:
        print("Warning: Graph has no nodes. Cannot convert to PyG format.")
        return None
    
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
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        # Handle empty edge case
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float)
    
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
    
    # Move to GPU if available
    if DEVICE.type == 'cuda':
        data = data.to(DEVICE)
    
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
    
    try:
        # Validate geometry
        gdf_rbs = validate_geometry(gdf_rbs)
        if len(gdf_rbs) < 3:
            print("Warning: Voronoi diagram requires at least 3 points.")
            return nx.Graph()
    except (TypeError, ValueError) as e:
        print(f"Error: {e}")
        return nx.Graph()
    
    # Extract points
    points = []
    for _, row in gdf_rbs.iterrows():
        try:
            # Try to get coordinates from geometry
            if hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
                points.append((row.geometry.x, row.geometry.y))
            # Alternatively try to get from Longitude/Latitude columns
            elif 'Longitude' in row and 'Latitude' in row:
                points.append((row['Longitude'], row['Latitude']))
        except Exception as e:
            print(f"Warning: Error extracting coordinates: {e}")
    
    if len(points) < 3:
        print("Error: Not enough valid points for Voronoi diagram.")
        return nx.Graph()
    
    points = np.array(points)
    
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
    
    try:
        # Calculate Voronoi diagram
        vor = Voronoi(all_points)
    except Exception as e:
        print(f"Error calculating Voronoi diagram: {e}")
        return nx.Graph()
    
    # Create graph
    G = nx.Graph()
    
    # Add RBS nodes
    valid_idx_map = {}  # Map to keep track of valid indices
    valid_idx = 0
    
    for i, (idx, row) in enumerate(gdf_rbs.iterrows()):
        try:
            # Check if we have coordinates for this point
            if i < len(points):
                G.add_node(valid_idx, 
                          pos=(points[i][0], points[i][1]), 
                          original_idx=idx,
                          operator=row.get('Operator', 'N/A'),
                          coverage_radius=row.get('Coverage_Radius_km', 0))
                valid_idx_map[i] = valid_idx
                valid_idx += 1
        except Exception as e:
            print(f"Warning: Error adding node {idx}: {e}")
    
    # Add edges based on Voronoi cell adjacency
    try:
        for i, j in vor.ridge_points:
            # Ignore edges connected to points at corners
            if i >= len(points) or j >= len(points):
                continue
            
            # Check if both endpoints are valid
            if i in valid_idx_map and j in valid_idx_map:
                node_i = valid_idx_map[i]
                node_j = valid_idx_map[j]
                
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
                G.add_edge(node_i, node_j, weight=1.0/max(0.1, distance), distance=distance)
    except Exception as e:
        print(f"Error adding edges: {e}")
    
    # Visualize if output path is specified
    if output_path:
        _visualize_voronoi_graph(G, vor, output_path)
    
    print(f"Voronoi graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def _visualize_voronoi_graph(G, vor, output_path):
    """
    Creates a visualization of the Voronoi-based graph.
    
    Args:
        G (nx.Graph): NetworkX graph
        vor (Voronoi): Scipy Voronoi object
        output_path (str): Path to save the visualization
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
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
    
    # Draw Voronoi cells if voronoi_plot_2d is available
    if VORONOI_PLOT_AVAILABLE:
        voronoi_plot_2d(vor, show_points=False, show_vertices=False, 
                       line_colors='gray', line_width=0.5, line_alpha=0.4, ax=plt.gca())
    else:
        # Simple fallback: Draw the Voronoi edges manually
        for simplex in vor.ridge_vertices:
            if -1 not in simplex:  # -1 indicates a ridge extending to infinity
                vertices = vor.vertices[simplex]
                plt.plot(vertices[:, 0], vertices[:, 1], 'k-', lw=0.5, alpha=0.4)
    
    # Draw nodes by operator
    for operator, nodes in operator_groups.items():
        color = operator_colors.get(operator, '#CCCCCC')
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, 
                              node_size=100, alpha=0.8, label=operator)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.8, alpha=0.4, edge_color='gray')
    
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, 
                        label=operator) for operator, color in operator_colors.items() 
              if operator in operator_groups]
    if handles:
        plt.legend(handles=handles, title="Operators", loc='best')
    
    plt.title("RBS Graph based on Voronoi Cells", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Voronoi graph visualization saved to {output_path}")

def run_graph_analysis(gdf_rbs, output_path, connection_radius=3.0):
    """
    Run full graph analysis on RBS data.
    
    Args:
        gdf_rbs (GeoDataFrame): GeoDataFrame with RBS data
        output_path (str): Path to save output files
        connection_radius (float): Maximum radius (km) to consider two RBS connected
        
    Returns:
        dict: Dictionary with analysis results
    """
    print("Running graph analysis on RBS data...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Create RBS graph
    G = create_rbs_graph(gdf_rbs, connection_radius=connection_radius)
    
    # Calculate graph metrics
    metrics = calculate_graph_metrics(G)
    
    # Save metrics to file
    metrics_file = os.path.join(output_path, "graph_metrics.json")
    with open(metrics_file, 'w') as f:
        import json
        json.dump(metrics, f, indent=2)
    
    # Create visualizations
    visualize_graph(G, os.path.join(output_path, "rbs_graph.png"), 
                   title="Connectivity Graph between RBS", by_operator=True)
    
    visualize_graph(G, os.path.join(output_path, "rbs_graph_centrality.png"), 
                   title="RBS Centrality Graph", by_operator=False)
    
    # Create Voronoi-based graph
    G_voronoi = create_voronoi_rbs_graph(gdf_rbs, os.path.join(output_path, "voronoi_graph.png"))
    
    # Convert to PyG if available
    pyg_data = None
    if TORCH_AVAILABLE:
        pyg_data = convert_to_pyg(G)
    
    # Return results
    results = {
        'graph': G,
        'voronoi_graph': G_voronoi,
        'metrics': metrics,
        'pyg_data': pyg_data
    }
    
    print("Graph analysis completed.")
    return results
