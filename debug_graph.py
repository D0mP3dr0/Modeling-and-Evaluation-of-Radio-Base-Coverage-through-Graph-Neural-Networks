#!/usr/bin/env python3
"""
Debug script for graph analysis in the RBS Analysis Tool.
This script provides a step-by-step execution of the graph analysis process.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project directory to the path if running as script
project_dir = Path(__file__).resolve().parent
if project_dir not in sys.path:
    sys.path.append(str(project_dir))

# Import configuration
from src.config import setup_logging, RESULTS_DIR

# Setup logging with high verbosity
logger = setup_logging('debug_graph.log', console_level=logging.DEBUG, file_level=logging.DEBUG)

def debug_graph_analysis():
    """
    Debug the graph analysis process step by step.
    """
    input_file = "data/csv_licenciamento_bruto.csv.csv"
    
    # Create output directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(RESULTS_DIR, f"debug_analysis_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("STARTING GRAPH ANALYSIS DEBUG")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Results directory: {results_dir}")
    logger.info("=" * 80)
    
    # Step 1: Load and process data
    logger.info("STEP 1: Loading and processing data...")
    try:
        from src.data_processing import load_and_process_data
        gdf_rbs = load_and_process_data(input_file)
        
        if gdf_rbs is None or gdf_rbs.empty:
            logger.error("No data loaded. Exiting.")
            return
        
        logger.info(f"Successfully loaded {len(gdf_rbs)} RBS records.")
        logger.info(f"Columns: {list(gdf_rbs.columns)}")
        logger.info(f"First few records: \n{gdf_rbs.head().to_string()}")
    except Exception as e:
        logger.error(f"Error in data loading and processing: {e}", exc_info=True)
        return
    
    # Step 2: Import graph analysis module
    logger.info("STEP 2: Importing graph analysis module...")
    try:
        from src.graph_analysis import create_rbs_graph, calculate_graph_metrics, visualize_graph, create_voronoi_rbs_graph, convert_to_pyg
        logger.info("Graph analysis module imported successfully.")
    except Exception as e:
        logger.error(f"Error importing graph analysis module: {e}", exc_info=True)
        return
    
    # Step 3: Create RBS graph
    logger.info("STEP 3: Creating RBS graph...")
    try:
        G = create_rbs_graph(gdf_rbs, connection_radius=3.0)
        logger.info(f"RBS graph created successfully with {len(G.nodes())} nodes and {len(G.edges())} edges.")
    except Exception as e:
        logger.error(f"Error creating RBS graph: {e}", exc_info=True)
        return
    
    # Step 4: Calculate graph metrics
    logger.info("STEP 4: Calculating graph metrics...")
    try:
        metrics = calculate_graph_metrics(G)
        logger.info(f"Graph metrics calculated: {metrics}")
        
        # Save metrics to file
        metrics_file = os.path.join(results_dir, "graph_metrics.json")
        import json
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")
    except Exception as e:
        logger.error(f"Error calculating graph metrics: {e}", exc_info=True)
    
    # Step 5: Create visualizations
    logger.info("STEP 5: Creating visualizations...")
    try:
        # Visualization 1: RBS Graph by operator
        viz_path_1 = os.path.join(results_dir, "rbs_graph.png")
        visualize_graph(G, viz_path_1, title="Connectivity Graph between RBS", by_operator=True)
        logger.info(f"Created visualization: {viz_path_1}")
        
        # Visualization 2: RBS Graph by centrality
        viz_path_2 = os.path.join(results_dir, "rbs_graph_centrality.png")
        visualize_graph(G, viz_path_2, title="RBS Centrality Graph", by_operator=False)
        logger.info(f"Created visualization: {viz_path_2}")
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}", exc_info=True)
    
    # Step 6: Create Voronoi-based graph
    logger.info("STEP 6: Creating Voronoi-based graph...")
    try:
        viz_path_3 = os.path.join(results_dir, "voronoi_graph.png")
        G_voronoi = create_voronoi_rbs_graph(gdf_rbs, viz_path_3)
        logger.info(f"Voronoi graph created with {G_voronoi.number_of_nodes()} nodes and {G_voronoi.number_of_edges()} edges.")
    except Exception as e:
        logger.error(f"Error creating Voronoi graph: {e}", exc_info=True)
    
    # Step 7: Convert to PyG (if available)
    logger.info("STEP 7: Converting to PyG data format...")
    try:
        # Check if PyTorch is available
        import importlib
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is not None:
            pyg_data = convert_to_pyg(G)
            logger.info(f"Converted to PyG data format: {pyg_data}")
        else:
            logger.warning("PyTorch/PyG not available. Skipping conversion.")
    except Exception as e:
        logger.error(f"Error converting to PyG: {e}", exc_info=True)
    
    logger.info("=" * 80)
    logger.info("GRAPH ANALYSIS DEBUG COMPLETED")
    logger.info(f"Results saved to {results_dir}")
    logger.info("=" * 80)

if __name__ == "__main__":
    debug_graph_analysis() 