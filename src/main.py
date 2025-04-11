"""
Main module for Radio Base Stations (RBS) analysis.
Provides a unified interface to run various types of analyses on RBS data.
"""

import pandas as pd
import geopandas as gpd
import os
import argparse
import sys
from datetime import datetime

# Import analysis modules
try:
    from data_processing import load_and_process_data
    from analysis import run_basic_analysis
    from visualization import create_visualizations
    from graph_analysis import run_graph_analysis
    from coverage_models import estimate_coverage
    from tech_frequency_analysis import run_tech_frequency_analysis
    from advanced_temporal_analysis import run_temporal_analysis
    from correlation_analysis import run_correlation_analysis
    from spatial_analysis import run_spatial_analysis
    from integration_analysis import run_integration_analysis
    from prediction_module import run_prediction_analysis
    from dashboard_interactive import run_dashboard
    from report_generator import run_report_generation
    from unit_tests import run_tests
except ImportError:
    # Fallback for direct execution from project root
    from src.data_processing import load_and_process_data
    from src.analysis import run_basic_analysis
    from src.visualization import create_visualizations
    from src.graph_analysis import run_graph_analysis
    from src.coverage_models import estimate_coverage
    from src.tech_frequency_analysis import run_tech_frequency_analysis
    from src.advanced_temporal_analysis import run_temporal_analysis
    from src.correlation_analysis import run_correlation_analysis
    from src.spatial_analysis import run_spatial_analysis
    from src.integration_analysis import run_integration_analysis
    from src.prediction_module import run_prediction_analysis
    from src.dashboard_interactive import run_dashboard
    from src.report_generator import run_report_generation
    from src.unit_tests import run_tests

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Radio Base Stations (RBS) Analysis Tool')
    
    # Input/output arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input data file (CSV or GeoJSON)')
    parser.add_argument('--output', '-o', type=str, default='results',
                       help='Path to output directory')
    
    # Analysis selection arguments
    parser.add_argument('--all', '-a', action='store_true',
                       help='Run all available analyses')
    parser.add_argument('--basic', '-b', action='store_true',
                       help='Run basic analysis')
    parser.add_argument('--visualization', '-v', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--graph', '-g', action='store_true',
                       help='Run graph analysis')
    parser.add_argument('--coverage', '-c', action='store_true',
                       help='Estimate coverage')
    parser.add_argument('--tech-frequency', '-tf', action='store_true',
                       help='Run technology and frequency analysis')
    parser.add_argument('--temporal', '-t', action='store_true',
                       help='Run advanced temporal analysis')
    parser.add_argument('--correlation', '-cr', action='store_true',
                       help='Run correlation analysis')
    parser.add_argument('--spatial', '-s', action='store_true',
                       help='Run spatial analysis')
    parser.add_argument('--integration', '-int', action='store_true',
                       help='Run integration analysis')
    parser.add_argument('--prediction', '-p', action='store_true',
                       help='Run prediction analysis')
    parser.add_argument('--dashboard', '-d', action='store_true',
                       help='Run interactive dashboard')
    parser.add_argument('--report', '-r', action='store_true',
                       help='Generate comprehensive report')
    parser.add_argument('--test', action='store_true',
                       help='Run unit tests')
    
    # Additional options
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    return parser.parse_args()

def main():
    """Main function to run the RBS analysis."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Run unit tests if requested
        if args.test:
            print("Running unit tests...")
            success = run_tests()
            if not success:
                print("Unit tests failed. Exiting...")
                sys.exit(1)
            else:
                print("All unit tests passed.")
                # If only testing was requested, exit
                if not any([args.all, args.basic, args.visualization, args.graph, 
                          args.coverage, args.tech_frequency, args.temporal,
                          args.correlation, args.spatial, args.integration,
                          args.prediction, args.dashboard, args.report]):
                    sys.exit(0)
        
        # Load and process data
        print(f"Loading data from {args.input}...")
        try:
            gdf_rbs = load_and_process_data(args.input)
            print(f"Loaded {len(gdf_rbs)} RBS records.")
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
        
        # Create a timestamp for output subdirectories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(args.output, f"analysis_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Run requested analyses
        if args.all or args.basic:
            print("Running basic analysis...")
            try:
                run_basic_analysis(gdf_rbs, results_dir)
            except Exception as e:
                print(f"Error in basic analysis: {e}")
        
        if args.all or args.visualization:
            print("Creating visualizations...")
            try:
                create_visualizations(gdf_rbs, results_dir)
            except Exception as e:
                print(f"Error in visualization: {e}")
        
        if args.all or args.graph:
            print("Running graph analysis...")
            try:
                run_graph_analysis(gdf_rbs, results_dir)
            except Exception as e:
                print(f"Error in graph analysis: {e}")
        
        if args.all or args.coverage:
            print("Estimating coverage...")
            try:
                gdf_with_coverage = estimate_coverage(gdf_rbs)
                output_file = os.path.join(results_dir, 'coverage_estimates.geojson')
                gdf_with_coverage.to_file(output_file, driver='GeoJSON')
                print(f"Coverage estimates saved to {output_file}")
            except Exception as e:
                print(f"Error in coverage estimation: {e}")
        
        if args.all or args.tech_frequency:
            print("Running technology and frequency analysis...")
            try:
                run_tech_frequency_analysis(gdf_rbs, results_dir)
            except Exception as e:
                print(f"Error in technology and frequency analysis: {e}")
        
        if args.all or args.temporal:
            print("Running advanced temporal analysis...")
            try:
                run_temporal_analysis(gdf_rbs, results_dir)
            except Exception as e:
                print(f"Error in advanced temporal analysis: {e}")
        
        if args.all or args.correlation:
            print("Running correlation analysis...")
            try:
                run_correlation_analysis(gdf_rbs, results_dir)
            except Exception as e:
                print(f"Error in correlation analysis: {e}")
        
        if args.all or args.spatial:
            print("Running spatial analysis...")
            try:
                run_spatial_analysis(gdf_rbs, results_dir)
            except Exception as e:
                print(f"Error in spatial analysis: {e}")
        
        if args.all or args.integration:
            print("Running integration analysis...")
            try:
                run_integration_analysis(gdf_rbs, results_dir)
            except Exception as e:
                print(f"Error in integration analysis: {e}")
        
        if args.all or args.prediction:
            print("Running prediction analysis...")
            try:
                run_prediction_analysis(gdf_rbs, results_dir)
            except Exception as e:
                print(f"Error in prediction analysis: {e}")
        
        if args.all or args.report:
            print("Generating comprehensive report...")
            try:
                reports_dir = os.path.join(results_dir, 'reports')
                os.makedirs(reports_dir, exist_ok=True)
                pdf_path, excel_path = run_report_generation(gdf_rbs, reports_dir)
                print(f"Reports generated at:\n  - PDF: {pdf_path}\n  - Excel: {excel_path}")
            except Exception as e:
                print(f"Error in report generation: {e}")
        
        if args.dashboard:
            print("Starting interactive dashboard...")
            try:
                run_dashboard(gdf_rbs, debug=args.debug)
            except Exception as e:
                print(f"Error in dashboard: {e}")
        
        print(f"All analyses completed. Results saved to {results_dir}")
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 