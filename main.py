import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

from src.data_processing import load_and_clean_data
from src.analysis import setup_visualization_options, exploratory_analysis_rbs
from src.coverage_models import (
    calculate_eirp, calculate_improved_coverage_radius, calculate_coverage_area,
    create_precise_sector, estimate_area_type
)
from src.visualization import (
    configure_visualization_style, create_positioning_map, 
    create_coverage_map_by_operator, create_overlap_map,
    create_heat_map_power, create_folium_map
)
from src.graph_analysis import (
    create_rbs_graph, calculate_graph_metrics, visualize_graph,
    create_voronoi_rbs_graph, convert_to_pyg
)

# --- Configuration --- 
# Define paths here. Use relative paths for better portability.
INPUT_CSV_PATH = "data/csv_licenciamento_bruto.csv"  # Replace with the real file path
OUTPUT_CSV_PATH = "data/erb_sorocaba_clean.csv"
RESULTS_DIR = "results"

# Create directories if they don't exist
for directory in [RESULTS_DIR, "data"]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# --- Step 1: Load and Clean Data --- 
def step_1_loading():
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING AND CLEANING")
    print("="*80 + "\n")
    
    print(f"Loading raw data from: {INPUT_CSV_PATH}")
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"WARNING: File {INPUT_CSV_PATH} not found.")
        print("You need to add the original CSV file in the 'data' folder.")
        print("Skipping this step.")
        
        # For testing purposes, try to use already processed file
        if os.path.exists(OUTPUT_CSV_PATH):
            print(f"Found existing clean file: {OUTPUT_CSV_PATH}")
            df = pd.read_csv(OUTPUT_CSV_PATH)
            print(f"Loaded {len(df)} records.")
            return df
        else:
            print("No data file found. Terminating.")
            return None
    
    clean_df = load_and_clean_data(INPUT_CSV_PATH, OUTPUT_CSV_PATH)
    return clean_df

# --- Step 2: Exploratory Analysis --- 
def step_2_exploratory_analysis(df):
    if df is None or df.empty:
        print("No data for exploratory analysis. Skipping step.")
        return df
    
    print("\n" + "="*80)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("="*80 + "\n")
    
    # Set up visualization options before calling the analysis
    setup_visualization_options() 
    
    # Call the analysis function, passing the clean DataFrame and the results directory
    exploratory_analysis_rbs(df, RESULTS_DIR)
    
    return df

# --- Step 3: Coverage Processing --- 
def step_3_coverage_processing(df):
    if df is None or df.empty:
        print("No data for coverage processing. Skipping step.")
        return None, None
    
    print("\n" + "="*80)
    print("STEP 3: SIGNAL COVERAGE PROCESSING")
    print("="*80 + "\n")
    
    # Configure visual style
    configure_visualization_style()
    
    # Copy dataframe to avoid modifying the original
    df_coverage = df.copy()
    
    # Convert data types
    print("Converting data types...")
    numeric_columns = ['PotenciaTransmissorWatts', 'FreqTxMHz', 'GanhoAntena', 'Azimute']
    for column in numeric_columns:
        if column in df_coverage.columns:
            df_coverage[column] = pd.to_numeric(df_coverage[column], errors='coerce')
    
    # Filter for Sorocaba region and main operators
    print("Filtering data for region of interest...")
    
    # Check if we have coordinates
    if 'Latitude' not in df_coverage.columns or 'Longitude' not in df_coverage.columns:
        print("ERROR: Data does not contain coordinates (Latitude/Longitude). Unable to continue.")
        return None, None
    
    # Define region to filter (Sorocaba bbox)
    sorocaba_bbox = [-23.60, -23.30, -47.65, -47.25]  # [lat_min, lat_max, lon_min, lon_max]
    
    # Filter by region
    df_coverage = df_coverage[
        (df_coverage['Latitude'] >= sorocaba_bbox[0]) &
        (df_coverage['Latitude'] <= sorocaba_bbox[1]) &
        (df_coverage['Longitude'] >= sorocaba_bbox[2]) &
        (df_coverage['Longitude'] <= sorocaba_bbox[3])
    ]
    
    # Normalize operator names (if the column exists)
    if 'NomeEntidade' in df_coverage.columns:
        print("Standardizing operator names...")
        
        # Mapping to standardized names
        operator_mapping = {
            'CLARO': 'CLARO',
            'CLARO S.A.': 'CLARO',
            'OI': 'OI',
            'OI MÓVEL S.A.': 'OI',
            'VIVO': 'VIVO',
            'TELEFÔNICA BRASIL S.A.': 'VIVO',
            'TIM': 'TIM',
            'TIM S.A.': 'TIM'
        }
        
        # Function to map operator
        def map_operator(name):
            for pattern, standardized in operator_mapping.items():
                if pattern in name.upper():
                    return standardized
            return "OTHER"
        
        # Create Operator column
        df_coverage['Operator'] = df_coverage['NomeEntidade'].apply(map_operator)
        
        # Filter main operators
        df_coverage = df_coverage[df_coverage['Operator'].isin(['CLARO', 'OI', 'VIVO', 'TIM'])]
    
    print(f"After filtering: {len(df_coverage)} RBS in the region of interest")
    
    # Fill missing values
    print("Filling missing values...")
    
    # Power
    if 'PotenciaTransmissorWatts' in df_coverage.columns:
        median_power = df_coverage['PotenciaTransmissorWatts'].median()
        if pd.isna(median_power) or median_power <= 0:
            median_power = 20.0
        df_coverage.loc[pd.isna(df_coverage['PotenciaTransmissorWatts']) | 
                         (df_coverage['PotenciaTransmissorWatts'] <= 0), 
                         'PotenciaTransmissorWatts'] = median_power
    else:
        df_coverage['PotenciaTransmissorWatts'] = 20.0  # Default value
    
    # Gain
    if 'GanhoAntena' in df_coverage.columns:
        median_gain = df_coverage['GanhoAntena'].median()
        if pd.isna(median_gain) or median_gain <= 0:
            median_gain = 16.0
        df_coverage.loc[pd.isna(df_coverage['GanhoAntena']) | 
                         (df_coverage['GanhoAntena'] <= 0), 
                         'GanhoAntena'] = median_gain
    else:
        df_coverage['GanhoAntena'] = 16.0  # Default value
    
    # Frequency
    if 'FreqTxMHz' in df_coverage.columns:
        median_freq = df_coverage['FreqTxMHz'].median()
        if pd.isna(median_freq) or median_freq <= 0:
            median_freq = 850.0
        df_coverage.loc[pd.isna(df_coverage['FreqTxMHz']) | 
                         (df_coverage['FreqTxMHz'] <= 0), 
                         'FreqTxMHz'] = median_freq
    else:
        df_coverage['FreqTxMHz'] = 850.0  # Default value
    
    # Azimuth
    if 'Azimute' not in df_coverage.columns or df_coverage['Azimute'].isna().sum() > 0:
        print("Generating random azimuths for missing values...")
        if 'Azimute' not in df_coverage.columns:
            df_coverage['Azimute'] = 0
        
        # Fill missing azimuth values with 0, 120, 240 (standard for 3 sectors)
        default_azimuths = [0, 120, 240]
        for i, row in df_coverage[df_coverage['Azimute'].isna()].iterrows():
            df_coverage.loc[i, 'Azimute'] = default_azimuths[i % len(default_azimuths)]
    
    # Transform data to GeoDataFrame
    print("Converting to GeoDataFrame...")
    geometry = [Point(xy) for xy in zip(df_coverage['Longitude'], df_coverage['Latitude'])]
    gdf_rbs = gpd.GeoDataFrame(df_coverage, geometry=geometry, crs="EPSG:4326")
    
    # Calculate area type based on RBS density
    print("Calculating area type (urban, suburban, rural)...")
    gdf_rbs = estimate_area_type(gdf_rbs, radius=0.01)
    
    # Calculate EIRP (Effective Isotropic Radiated Power)
    print("Calculating EIRP (Effective Isotropic Radiated Power)...")
    gdf_rbs['EIRP_dBm'] = gdf_rbs.apply(
        lambda row: calculate_eirp(row['PotenciaTransmissorWatts'], row['GanhoAntena']), 
        axis=1
    )
    
    # Calculate coverage radius
    print("Calculating coverage radius...")
    gdf_rbs['Coverage_Radius_km'] = gdf_rbs.apply(
        lambda row: calculate_improved_coverage_radius(
            row['EIRP_dBm'], row['FreqTxMHz'], row['area_type']
        ), 
        axis=1
    )
    
    # Calculate coverage area
    print("Calculating coverage area...")
    gdf_rbs['Coverage_Area_km2'] = gdf_rbs.apply(
        lambda row: calculate_coverage_area(row['Coverage_Radius_km']), 
        axis=1
    )
    
    # Create sector geometries
    print("Creating coverage sector geometries...")
    gdf_rbs['sector_geometry'] = gdf_rbs.apply(
        lambda row: create_precise_sector(
            row['Latitude'], row['Longitude'], 
            row['Coverage_Radius_km'], row['Azimute']
        ), 
        axis=1
    )
    
    # Create GeoDataFrame with just sectors
    gdf_sectors = gpd.GeoDataFrame(
        gdf_rbs[['Operator', 'EIRP_dBm', 'Coverage_Radius_km', 'Coverage_Area_km2', 'area_type']],
        geometry=gdf_rbs['sector_geometry'],
        crs="EPSG:4326"
    ).dropna(subset=['geometry'])
    
    print(f"Processing completed: {len(gdf_rbs)} RBS and {len(gdf_sectors)} coverage sectors.")
    
    return gdf_rbs, gdf_sectors

# --- Step 4: Advanced Visualizations --- 
def step_4_advanced_visualizations(gdf_rbs, gdf_sectors):
    if gdf_rbs is None or gdf_rbs.empty:
        print("No data for advanced visualizations. Skipping step.")
        return
    
    print("\n" + "="*80)
    print("STEP 4: ADVANCED VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Create positioning map of RBS
    print("\n4.1. RBS Positioning Map")
    positioning_map = os.path.join(RESULTS_DIR, "positioning_map_rbs.png")
    create_positioning_map(gdf_rbs, positioning_map)
    
    # Create coverage map by operator
    if gdf_sectors is not None and not gdf_sectors.empty:
        print("\n4.2. Coverage Map by Operator")
        coverage_map = os.path.join(RESULTS_DIR, "coverage_map_by_operator.png")
        create_coverage_map_by_operator(gdf_rbs, gdf_sectors, coverage_map)
        
        print("\n4.3. Overlap Map")
        overlap_map = os.path.join(RESULTS_DIR, "overlap_map.png")
        create_overlap_map(gdf_rbs, gdf_sectors, overlap_map)
    
    # Create heat map of power
    print("\n4.4. Heat Map of EIRP Power")
    heat_map = os.path.join(RESULTS_DIR, "heat_map_power.png")
    create_heat_map_power(gdf_rbs, heat_map)
    
    # Create interactive Folium map
    print("\n4.5. Interactive Folium Map")
    folium_map = os.path.join(RESULTS_DIR, "interactive_rbs_map.html")
    create_folium_map(gdf_rbs, folium_map)
    
    print("\nAll visualizations generated successfully.")

# --- Step 5: Graph Analysis --- 
def step_5_graph_analysis(gdf_rbs):
    if gdf_rbs is None or gdf_rbs.empty:
        print("No data for graph analysis. Skipping step.")
        return
    
    print("\n" + "="*80)
    print("STEP 5: GRAPH ANALYSIS AND NETWORKS")
    print("="*80 + "\n")
    
    # Create connectivity graph
    print("\n5.1. Creating connectivity graph...")
    graph = create_rbs_graph(gdf_rbs, connection_radius=2.0, weighted=True)
    
    # Calculate metrics
    print("\n5.2. Calculating graph metrics...")
    metrics = calculate_graph_metrics(graph)
    
    # Print metrics
    print("\nConnectivity Graph Metrics:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    # Create visualization
    print("\n5.3. Creating graph visualization...")
    graph_viz_path = os.path.join(RESULTS_DIR, "rbs_connectivity_graph.png")
    visualize_graph(graph, graph_viz_path, title="RBS Connectivity Graph", by_operator=True)
    
    # Create Voronoi diagram
    print("\n5.4. Creating Voronoi diagram...")
    voronoi_path = os.path.join(RESULTS_DIR, "rbs_voronoi_diagram.png")
    voronoi_graph = create_voronoi_rbs_graph(gdf_rbs, voronoi_path)
    
    # Create PyG Data object if PyTorch is available
    print("\n5.5. Converting to PyTorch Geometric format...")
    pyg_data = convert_to_pyg(graph)
    if pyg_data is not None:
        print(f"PyG data created with {pyg_data.num_nodes} nodes and {pyg_data.num_edges} edges.")
    
    print("\nAll graph analysis completed successfully.")

# --- Função Principal --- 
def main():
    print("="*80)
    print("RADIO BASE STATION (RBS) ANALYSIS SYSTEM")
    print("="*80)
    print("\nThis program analyzes RBS data to extract insights about signal coverage.")
    
    # Step 1: Loading data
    df = step_1_loading()
    
    # Step 2: Exploratory analysis
    df = step_2_exploratory_analysis(df)
    
    # Step 3: Coverage processing
    gdf_rbs, gdf_sectors = step_3_coverage_processing(df)
    
    # Step 4: Advanced visualizations
    step_4_advanced_visualizations(gdf_rbs, gdf_sectors)
    
    # Step 5: Graph analysis
    step_5_graph_analysis(gdf_rbs)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nResults saved in the '{RESULTS_DIR}' directory.")

if __name__ == "__main__":
    main()
