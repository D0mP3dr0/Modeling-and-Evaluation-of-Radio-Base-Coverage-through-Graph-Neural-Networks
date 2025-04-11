#!/usr/bin/env python3
"""
Main entry point for the Radio Base Stations (RBS) Analysis Tool.

This script provides a command-line interface to run various analyses on RBS data.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add the project directory to the path if running as script
project_dir = Path(__file__).resolve().parent
if project_dir not in sys.path:
    sys.path.append(str(project_dir))

# Import configuration
from src.config import DEFAULT_INPUT_PATH, RESULTS_DIR, setup_logging

# Setup logging
logger = setup_logging('main.log')

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Radio Base Stations (RBS) Analysis Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/output arguments
    parser.add_argument('--input', '-i', type=str, default=DEFAULT_INPUT_PATH,
                       help='Path to input data file (CSV or GeoJSON)')
    parser.add_argument('--output', '-o', type=str, default=RESULTS_DIR,
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
    parser.add_argument('--advanced-coverage', '-ac', action='store_true',
                      help='Run advanced coverage visualization')
    parser.add_argument('--coverage-quality', '-cq', action='store_true',
                      help='Run coverage quality analysis')
    parser.add_argument('--coverage-prediction', '-cp', action='store_true',
                      help='Run coverage prediction')
    parser.add_argument('--advanced-graph', '-ag', action='store_true',
                      help='Run advanced graph analysis')
    parser.add_argument('--educational-docs', '-ed', action='store_true',
                      help='Create educational documentation')
    
    # Additional options
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--time-field', type=str, default='installation_date',
                      help='Field name containing timestamp for temporal analyses')
    
    return parser.parse_args()

def run_analyses(args):
    """
    Run the requested analyses based on command line arguments.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    # Import modules here to avoid slow startup time
    from src.data_processing import load_and_process_data
    from src.unit_tests import run_tests
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    logger.info(f"Output directory: {args.output}")
    
    # Run unit tests if requested
    if args.test:
        logger.info("Running unit tests...")
        success = run_tests()
        if not success:
            logger.error("Unit tests failed. Exiting...")
            sys.exit(1)
        else:
            logger.info("All unit tests passed.")
            # If only testing was requested, exit
            if not any([args.all, args.basic, args.visualization, args.graph, 
                      args.coverage, args.tech_frequency, args.temporal,
                      args.correlation, args.spatial, args.integration,
                      args.prediction, args.dashboard, args.report,
                      args.advanced_coverage, args.coverage_quality, 
                      args.coverage_prediction, args.advanced_graph,
                      args.educational_docs]):
                return
    
    # Load and process data
    logger.info(f"Loading data from {args.input}...")
    try:
        gdf_rbs = load_and_process_data(args.input)
        if gdf_rbs is None or gdf_rbs.empty:
            logger.error("No valid data loaded. Exiting...")
            sys.exit(1)
        logger.info(f"Loaded {len(gdf_rbs)} RBS records.")
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=args.debug)
        sys.exit(1)
    
    # Create a timestamp for output subdirectories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.output, f"analysis_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results will be saved to {results_dir}")
    
    # Run basic analysis if requested
    if args.all or args.basic:
        try:
            from src.analysis import run_basic_analysis
            logger.info("Running basic analysis...")
            run_basic_analysis(gdf_rbs, results_dir)
            logger.info("Basic analysis complete.")
        except Exception as e:
            logger.error(f"Error in basic analysis: {e}", exc_info=args.debug)
    
    # Create visualizations if requested
    if args.all or args.visualization:
        try:
            from src.visualization import create_visualizations
            logger.info("Creating visualizations...")
            create_visualizations(gdf_rbs, results_dir)
            logger.info("Visualizations created successfully.")
        except Exception as e:
            logger.error(f"Error in visualization: {e}", exc_info=args.debug)
    
    # Run graph analysis if requested
    if args.all or args.graph:
        try:
            from src.graph_analysis import run_graph_analysis
            logger.info("Running graph analysis...")
            run_graph_analysis(gdf_rbs, results_dir)
            logger.info("Graph analysis complete.")
        except Exception as e:
            logger.error(f"Error in graph analysis: {e}", exc_info=args.debug)
    
    # Estimate coverage if requested
    if args.all or args.coverage:
        try:
            from src.coverage_models import estimate_coverage
            logger.info("Estimating coverage...")
            gdf_with_coverage = estimate_coverage(gdf_rbs)
            output_file = os.path.join(results_dir, 'coverage_estimates.geojson')
            gdf_with_coverage.to_file(output_file, driver='GeoJSON')
            logger.info(f"Coverage estimates saved to {output_file}")
        except Exception as e:
            logger.error(f"Error in coverage estimation: {e}", exc_info=args.debug)
    
    # Run technology and frequency analysis if requested
    if args.all or args.tech_frequency:
        try:
            from src.tech_frequency_analysis import run_tech_frequency_analysis
            logger.info("Running technology and frequency analysis...")
            run_tech_frequency_analysis(gdf_rbs, results_dir)
            logger.info("Technology and frequency analysis complete.")
        except Exception as e:
            logger.error(f"Error in technology and frequency analysis: {e}", exc_info=args.debug)
    
    # Run temporal analysis if requested
    if args.all or args.temporal:
        try:
            from src.advanced_temporal_analysis import run_temporal_analysis
            logger.info("Running advanced temporal analysis...")
            run_temporal_analysis(gdf_rbs, results_dir)
            logger.info("Advanced temporal analysis complete.")
        except Exception as e:
            logger.error(f"Error in advanced temporal analysis: {e}", exc_info=args.debug)
    
    # Run correlation analysis if requested
    if args.all or args.correlation:
        try:
            from src.correlation_analysis import run_correlation_analysis
            logger.info("Running correlation analysis...")
            run_correlation_analysis(gdf_rbs, results_dir)
            logger.info("Correlation analysis complete.")
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}", exc_info=args.debug)
    
    # Run spatial analysis if requested
    if args.all or args.spatial:
        try:
            from src.spatial_analysis import run_spatial_analysis
            logger.info("Running spatial analysis...")
            run_spatial_analysis(gdf_rbs, results_dir)
            logger.info("Spatial analysis complete.")
        except Exception as e:
            logger.error(f"Error in spatial analysis: {e}", exc_info=args.debug)
    
    # Run integration analysis if requested
    if args.all or args.integration:
        try:
            from src.integration_analysis import run_integration_analysis
            logger.info("Running integration analysis...")
            run_integration_analysis(gdf_rbs, results_dir)
            logger.info("Integration analysis complete.")
        except Exception as e:
            logger.error(f"Error in integration analysis: {e}", exc_info=args.debug)
    
    # Run prediction analysis if requested
    if args.all or args.prediction:
        try:
            from src.prediction_module import run_prediction_analysis
            logger.info("Running prediction analysis...")
            run_prediction_analysis(gdf_rbs, results_dir)
            logger.info("Prediction analysis complete.")
        except Exception as e:
            logger.error(f"Error in prediction analysis: {e}", exc_info=args.debug)
    
    # Run advanced coverage visualization if requested
    if args.all or args.advanced_coverage:
        try:
            from src.advanced_coverage_visualization import run_advanced_coverage_visualization
            logger.info("Running advanced coverage visualization...")
            advanced_coverage_dir = os.path.join(results_dir, 'advanced_coverage')
            os.makedirs(advanced_coverage_dir, exist_ok=True)
            run_advanced_coverage_visualization(gdf_rbs, advanced_coverage_dir)
            logger.info(f"Advanced coverage visualizations saved to {advanced_coverage_dir}")
        except Exception as e:
            logger.error(f"Error in advanced coverage visualization: {e}", exc_info=args.debug)
    
    # Run coverage quality analysis if requested
    if args.all or args.coverage_quality:
        try:
            from src.coverage_quality_analysis import run_coverage_quality_analysis
            logger.info("Running coverage quality analysis...")
            coverage_quality_dir = os.path.join(results_dir, 'coverage_quality')
            os.makedirs(coverage_quality_dir, exist_ok=True)
            run_coverage_quality_analysis(gdf_rbs, coverage_quality_dir)
            logger.info(f"Coverage quality analysis saved to {coverage_quality_dir}")
        except Exception as e:
            logger.error(f"Error in coverage quality analysis: {e}", exc_info=args.debug)
    
    # Run coverage prediction if requested
    if args.all or args.coverage_prediction:
        try:
            from src.coverage_prediction import run_coverage_prediction
            logger.info("Running coverage prediction...")
            coverage_prediction_dir = os.path.join(results_dir, 'coverage_prediction')
            os.makedirs(coverage_prediction_dir, exist_ok=True)
            run_coverage_prediction(gdf_rbs, coverage_prediction_dir)
            logger.info(f"Coverage prediction results saved to {coverage_prediction_dir}")
        except Exception as e:
            logger.error(f"Error in coverage prediction: {e}", exc_info=args.debug)
    
    # Run advanced graph analysis if requested
    if args.all or args.advanced_graph:
        try:
            from src.advanced_graph_analysis import run_advanced_graph_analysis
            logger.info("Running advanced graph analysis...")
            advanced_graph_dir = os.path.join(results_dir, 'advanced_graph')
            os.makedirs(advanced_graph_dir, exist_ok=True)
            run_advanced_graph_analysis(gdf_rbs, advanced_graph_dir, args.time_field)
            logger.info(f"Advanced graph analysis saved to {advanced_graph_dir}")
        except Exception as e:
            logger.error(f"Error in advanced graph analysis: {e}", exc_info=args.debug)
    
    # Create educational documentation if requested
    if args.all or args.educational_docs:
        try:
            from src.educational_documentation import create_educational_documentation
            logger.info("Creating educational documentation...")
            educational_docs_dir = os.path.join(results_dir, 'educational_docs')
            os.makedirs(educational_docs_dir, exist_ok=True)
            create_educational_documentation(gdf_rbs, educational_docs_dir)
            logger.info(f"Educational documentation created at {educational_docs_dir}")
            logger.info(f"Open {os.path.join(educational_docs_dir, 'index.html')} in a web browser to view.")
        except Exception as e:
            logger.error(f"Error in educational documentation: {e}", exc_info=args.debug)
    
    # Run interactive dashboard if requested
    if args.all or args.dashboard:
        try:
            from src.dashboard_interactive import run_dashboard
            logger.info("Running interactive dashboard...")
            dashboard_path = run_dashboard(gdf_rbs, results_dir)
            logger.info(f"Interactive dashboard saved to {dashboard_path}")
        except Exception as e:
            logger.error(f"Error in dashboard generation: {e}", exc_info=args.debug)
    
    # Generate comprehensive report if requested
    if args.all or args.report:
        try:
            from src.report_generator import run_report_generation
            logger.info("Generating comprehensive report...")
            report_path = run_report_generation(gdf_rbs, results_dir)
            logger.info(f"Report generated at {report_path}")
        except Exception as e:
            logger.error(f"Error in report generation: {e}", exc_info=args.debug)
    
    logger.info(f"Analysis completed. Results saved to {results_dir}")

def main():
    """Main entry point for the application."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Run requested analyses
        run_analyses(args)
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 