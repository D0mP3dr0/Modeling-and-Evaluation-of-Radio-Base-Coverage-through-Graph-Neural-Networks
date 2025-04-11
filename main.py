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
# Importando novos módulos avançados
from src.tech_frequency_analysis import run_tech_frequency_analysis
from src.advanced_temporal_analysis import run_temporal_analysis
from src.correlation_analysis import run_correlation_analysis
from src.spatial_analysis import run_spatial_analysis
from src.integration_analysis import run_integration_analysis
from src.prediction_module import run_prediction_analysis
from src.dashboard_interactive import run_dashboard
from src.report_generator import run_report_generation

# Import new advanced modules
from src.advanced_coverage_visualization import (
    create_3d_coverage_map, simulate_coverage_in_climate_conditions,
    analyze_topographic_obstructions
)
from src.coverage_quality_analysis import (
    identify_critical_areas, analyze_coverage_redundancy,
    calculate_coverage_efficiency
)
from src.coverage_prediction import (
    simulate_coverage_optimization, predict_expansion_areas
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
        crs=gdf_rbs.crs
    )
    
    print(f"Processed {len(gdf_rbs)} RBS with coverage information")
    
    return gdf_rbs, gdf_sectors

def step_4_advanced_visualizations(gdf_rbs, gdf_sectors):
    if gdf_rbs is None or gdf_sectors is None:
        print("No data for advanced visualizations. Skipping step.")
        return gdf_rbs, gdf_sectors
    
    print("\n" + "="*80)
    print("STEP 4: ADVANCED COVERAGE VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Create a directory for visualization results
    vis_dir = os.path.join(RESULTS_DIR, "visualizations")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Create a positioning map
    print("Creating positioning map...")
    create_positioning_map(
        gdf_rbs, 
        os.path.join(vis_dir, "positioning_map.png")
    )
    
    # Create a coverage map by operator
    print("Creating coverage map by operator...")
    create_coverage_map_by_operator(
        gdf_rbs, 
        gdf_sectors,
        os.path.join(vis_dir, "coverage_by_operator.png")
    )
    
    # Create an overlap map
    print("Creating coverage overlap map...")
    create_overlap_map(
        gdf_rbs,
        gdf_sectors,
        os.path.join(vis_dir, "coverage_overlap.png")
    )
    
    # Create a heat map of power
    print("Creating power heat map...")
    create_heat_map_power(
        gdf_rbs,
        os.path.join(vis_dir, "power_heat_map.png")
    )
    
    # Create an interactive Folium map
    print("Creating interactive map...")
    create_folium_map(
        gdf_rbs,
        os.path.join(vis_dir, "interactive_map.html")
    )
    
    # Add new advanced visualizations
    print("\nRunning advanced coverage visualizations...")
    
    # 3D Coverage Map
    create_3d_coverage_map(
        gdf_rbs,
        os.path.join(vis_dir, "3d_coverage_map.png"),
        resolution=100
    )
    
    # Simulate coverage in different climate conditions
    simulate_coverage_in_climate_conditions(
        gdf_rbs,
        gdf_sectors,
        os.path.join(vis_dir, "climate_coverage_simulation.png"),
        conditions=['clear', 'rain', 'heavy_rain', 'fog']
    )
    
    # Analyze topographic obstructions
    # Note: This requires a DEM file, we'll use a mock one for demonstration
    dem_path = os.path.join("data", "dem.tif")
    analyze_topographic_obstructions(
        gdf_rbs,
        dem_path,
        os.path.join(vis_dir, "topographic_obstruction_analysis.png")
    )
    
    return gdf_rbs, gdf_sectors

def step_5_graph_analysis(gdf_rbs):
    if gdf_rbs is None or gdf_rbs.empty:
        print("No data for graph analysis. Skipping step.")
        return None
    
    print("\n" + "="*80)
    print("STEP 5: GRAPH ANALYSIS OF RBS NETWORK")
    print("="*80 + "\n")
    
    # Create a directory for graph analysis results
    graph_dir = os.path.join(RESULTS_DIR, "graph_analysis")
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    
    # Create a graph from RBS points
    print("Creating RBS graph...")
    G = create_rbs_graph(gdf_rbs)
    
    # Calculate graph metrics
    print("Calculating graph metrics...")
    G, metrics = calculate_graph_metrics(G)
    
    # Create graph visualizations
    print("Creating graph visualizations...")
    visualize_graph(
        G, 
        gdf_rbs,
        metrics,
        os.path.join(graph_dir, "rbs_graph.png")
    )
    
    # Create Voronoi graph
    print("Creating Voronoi analysis...")
    voronoi_gdf = create_voronoi_rbs_graph(
        gdf_rbs,
        os.path.join(graph_dir, "voronoi_graph.png")
    )
    
    # Convert to PyG format if available
    try:
        print("Converting to PyG format for Machine Learning...")
        pyg_graph = convert_to_pyg(G, gdf_rbs)
        print("PyG conversion successful.")
    except Exception as e:
        print(f"PyG conversion error: {e}")
    
    return G

def step_6_tech_frequency_analysis(gdf_rbs):
    if gdf_rbs is None or gdf_rbs.empty:
        print("No data for technology and frequency analysis. Skipping step.")
        return
    
    print("\n" + "="*80)
    print("STEP 6: TECHNOLOGY AND FREQUENCY ANALYSIS")
    print("="*80 + "\n")
    
    # Create directory for tech analysis results
    tech_dir = os.path.join(RESULTS_DIR, "tech_analysis")
    
    # Run comprehensive technology and frequency analysis
    run_tech_frequency_analysis(gdf_rbs, tech_dir)

def step_7_temporal_analysis(gdf_rbs):
    if gdf_rbs is None or gdf_rbs.empty:
        print("No data for temporal analysis. Skipping step.")
        return
    
    print("\n" + "="*80)
    print("STEP 7: ADVANCED TEMPORAL ANALYSIS")
    print("="*80 + "\n")
    
    # Create directory for temporal analysis results
    temporal_dir = os.path.join(RESULTS_DIR, "temporal_analysis")
    
    # Run comprehensive temporal analysis
    run_temporal_analysis(gdf_rbs, temporal_dir)

def step_8_advanced_integrations(gdf_rbs):
    if gdf_rbs is None or gdf_rbs.empty:
        print("No data for advanced integrations. Skipping step.")
        return
    
    print("\n" + "="*80)
    print("STEP 8: ADVANCED INTEGRATIONS")
    print("="*80 + "\n")
    
    # Create directories for analysis results
    correlation_dir = os.path.join(RESULTS_DIR, "correlation_analysis")
    spatial_dir = os.path.join(RESULTS_DIR, "spatial_analysis")
    integration_dir = os.path.join(RESULTS_DIR, "integration_analysis")
    prediction_dir = os.path.join(RESULTS_DIR, "prediction_analysis")
    dashboard_dir = os.path.join(RESULTS_DIR, "dashboard")
    reports_dir = os.path.join(RESULTS_DIR, "reports")
    
    for directory in [correlation_dir, spatial_dir, integration_dir, prediction_dir, dashboard_dir, reports_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Run correlation analysis
    print("Running correlation analysis...")
    run_correlation_analysis(gdf_rbs, correlation_dir)
    
    # Run spatial analysis
    print("Running spatial analysis...")
    run_spatial_analysis(gdf_rbs, spatial_dir)
    
    # Run integration analysis (combining temporal, spatial, and tech)
    print("Running integration analysis...")
    run_integration_analysis(gdf_rbs, integration_dir)
    
    # Run prediction analysis
    print("Running prediction analysis...")
    run_prediction_analysis(gdf_rbs, prediction_dir)
    
    # Run interactive dashboard
    print("Setting up interactive dashboard...")
    try:
        run_dashboard(gdf_rbs, dashboard_dir)
    except Exception as e:
        print(f"Error creating dashboard: {e}")
        print("Dashboard creation skipped. You may need to install additional dependencies.")
    
    # Generate comprehensive reports
    print("Generating reports...")
    try:
        run_report_generation(gdf_rbs, reports_dir)
    except Exception as e:
        print(f"Error generating reports: {e}")
        print("Report generation skipped. You may need to install additional dependencies.")

# --- Step 9: Coverage Quality Analysis ---
def step_9_coverage_quality_analysis(gdf_rbs, gdf_sectors):
    if gdf_rbs is None or gdf_sectors is None:
        print("No data for coverage quality analysis. Skipping step.")
        return gdf_rbs, gdf_sectors
    
    print("\n" + "="*80)
    print("STEP 9: COVERAGE QUALITY ANALYSIS")
    print("="*80 + "\n")
    
    # Create directory for quality analysis results
    quality_dir = os.path.join(RESULTS_DIR, "quality_analysis")
    if not os.path.exists(quality_dir):
        os.makedirs(quality_dir)
    
    # Identify critical areas with insufficient coverage
    print("Identifying critical coverage areas...")
    identify_critical_areas(
        gdf_sectors,
        grid_size=0.01,
        output_path=os.path.join(quality_dir, "critical_areas_map.png")
    )
    
    # Analyze coverage redundancy
    print("Analyzing coverage redundancy...")
    analyze_coverage_redundancy(
        gdf_sectors,
        grid_size=0.01,
        output_path=os.path.join(quality_dir, "coverage_redundancy_map.png")
    )
    
    # Calculate coverage efficiency metrics
    print("Calculating coverage efficiency metrics...")
    calculate_coverage_efficiency(
        gdf_rbs,
        gdf_sectors,
        population_data=None,  # We don't have population data for this example
        output_path=os.path.join(quality_dir, "coverage_efficiency_metrics.png")
    )
    
    return gdf_rbs, gdf_sectors

# --- Step 10: Coverage Prediction and Optimization ---
def step_10_coverage_prediction(gdf_rbs, gdf_sectors):
    if gdf_rbs is None or gdf_sectors is None:
        print("No data for coverage prediction. Skipping step.")
        return gdf_rbs, gdf_sectors
    
    print("\n" + "="*80)
    print("STEP 10: COVERAGE PREDICTION AND OPTIMIZATION")
    print("="*80 + "\n")
    
    # Create directory for prediction results
    prediction_dir = os.path.join(RESULTS_DIR, "prediction")
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
    
    # Simulate optimization of RBS positions
    print("Simulating optimization of RBS positions...")
    simulate_coverage_optimization(
        gdf_rbs,
        gdf_sectors,
        grid_size=0.01,
        optimization_mode='adjust',
        output_path=os.path.join(prediction_dir, "position_optimization_map.png")
    )
    
    # Simulate adding new RBS for better coverage
    print("Simulating addition of new RBS...")
    simulate_coverage_optimization(
        gdf_rbs,
        gdf_sectors,
        grid_size=0.01,
        optimization_mode='add',
        output_path=os.path.join(prediction_dir, "new_rbs_optimization_map.png")
    )
    
    # Predict areas for network expansion
    print("Predicting priority areas for network expansion...")
    predict_expansion_areas(
        gdf_rbs,
        gdf_sectors,
        population_data=None,  # We don't have population data for this example
        road_network=None,     # We don't have road network data for this example
        output_path=os.path.join(prediction_dir, "expansion_prediction_map.png"),
        n_suggested_areas=5
    )
    
    return gdf_rbs, gdf_sectors

def main():
    print("\n" + "="*80)
    print("RADIO BASE STATIONS (RBS) ANALYSIS TOOL")
    print("="*80)
    print("\nStarting comprehensive analysis of radio base stations data...\n")
    
    # Step 1: Load and clean data
    df = step_1_loading()
    
    # Step 2: Exploratory analysis
    df = step_2_exploratory_analysis(df)
    
    # Step 3: Coverage processing
    gdf_rbs, gdf_sectors = step_3_coverage_processing(df)
    
    # Step 4: Advanced visualizations
    gdf_rbs, gdf_sectors = step_4_advanced_visualizations(gdf_rbs, gdf_sectors)
    
    # Step 5: Graph analysis
    g_rbs = step_5_graph_analysis(gdf_rbs)
    
    # Step 6: Technology and Frequency analysis
    step_6_tech_frequency_analysis(gdf_rbs)
    
    # Step 7: Temporal analysis
    step_7_temporal_analysis(gdf_rbs)
    
    # Step 8: Advanced integrations (temporal+spatial+tech)
    step_8_advanced_integrations(gdf_rbs)
    
    # Step 9: Coverage Quality Analysis (new)
    gdf_rbs, gdf_sectors = step_9_coverage_quality_analysis(gdf_rbs, gdf_sectors)
    
    # Step 10: Coverage Prediction and Optimization (new)
    gdf_rbs, gdf_sectors = step_10_coverage_prediction(gdf_rbs, gdf_sectors)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("Results saved in:", RESULTS_DIR)
    print("="*80)
    
    return 0

if __name__ == "__main__":
    main()
