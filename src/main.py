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
import importlib
import pkg_resources

# Define required packages with minimum versions
REQUIRED_PACKAGES = {
    'numpy': '1.20.0',
    'pandas': '1.3.0',
    'geopandas': '0.10.0',
    'matplotlib': '3.5.0',
    'networkx': '2.6.0',
    'shapely': '1.8.0'
}

def check_dependencies():
    """Check if all required dependencies are installed with correct versions."""
    missing_packages = []
    outdated_packages = []
    
    for package, min_version in REQUIRED_PACKAGES.items():
        try:
            installed_version = pkg_resources.get_distribution(package).version
            if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(min_version):
                outdated_packages.append((package, installed_version, min_version))
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package)
    
    if missing_packages or outdated_packages:
        print("Dependency check failed!")
        
        if missing_packages:
            print("\nMissing packages:")
            for package in missing_packages:
                print(f"  - {package}")
        
        if outdated_packages:
            print("\nOutdated packages:")
            for package, installed_version, min_version in outdated_packages:
                print(f"  - {package}: installed {installed_version}, required {min_version}")
        
        print("\nPlease install required dependencies using:")
        print("pip install -r requirements.txt")
        return False
    
    return True

# Function to safely import modules
def safe_import(module_name):
    """Safely import a module and return None if not available."""
    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError:
        return None

# Import analysis modules
try:
    from data_processing import load_and_process_data
    analysis_module = safe_import('analysis')
    visualization_module = safe_import('visualization')
    graph_analysis_module = safe_import('graph_analysis')
    coverage_models_module = safe_import('coverage_models')
    tech_frequency_analysis_module = safe_import('tech_frequency_analysis')
    advanced_temporal_analysis_module = safe_import('advanced_temporal_analysis')
    correlation_analysis_module = safe_import('correlation_analysis')
    spatial_analysis_module = safe_import('spatial_analysis')
    integration_analysis_module = safe_import('integration_analysis')
    prediction_module_module = safe_import('prediction_module')
    dashboard_interactive_module = safe_import('dashboard_interactive')
    report_generator_module = safe_import('report_generator')
    unit_tests_module = safe_import('unit_tests')
    advanced_coverage_visualization_module = safe_import('advanced_coverage_visualization')
    coverage_quality_analysis_module = safe_import('coverage_quality_analysis')
    coverage_prediction_module = safe_import('coverage_prediction')
    advanced_graph_analysis_module = safe_import('advanced_graph_analysis')
    educational_documentation_module = safe_import('educational_documentation')
except ImportError:
    # Fallback for direct execution from project root
    try:
        from src.data_processing import load_and_process_data
        analysis_module = safe_import('src.analysis')
        visualization_module = safe_import('src.visualization')
        graph_analysis_module = safe_import('src.graph_analysis')
        coverage_models_module = safe_import('src.coverage_models')
        tech_frequency_analysis_module = safe_import('src.tech_frequency_analysis')
        advanced_temporal_analysis_module = safe_import('src.advanced_temporal_analysis')
        correlation_analysis_module = safe_import('src.correlation_analysis')
        spatial_analysis_module = safe_import('src.spatial_analysis')
        integration_analysis_module = safe_import('src.integration_analysis')
        prediction_module_module = safe_import('src.prediction_module')
        dashboard_interactive_module = safe_import('src.dashboard_interactive')
        report_generator_module = safe_import('src.report_generator')
        unit_tests_module = safe_import('src.unit_tests')
        advanced_coverage_visualization_module = safe_import('src.advanced_coverage_visualization')
        coverage_quality_analysis_module = safe_import('src.coverage_quality_analysis')
        coverage_prediction_module = safe_import('src.coverage_prediction')
        advanced_graph_analysis_module = safe_import('src.advanced_graph_analysis')
        educational_documentation_module = safe_import('src.educational_documentation')
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Please ensure you are running the script from the project root or install the package.")
        sys.exit(1)

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
                       help='Enable debug mode')
    parser.add_argument('--time-field', type=str, default='InstallDate',
                      help='Field name containing timestamp for temporal analyses')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU acceleration for intensive computations')
    
    return parser.parse_args()

def run_analysis_safely(name, func, *args, **kwargs):
    """
    Run an analysis function safely, with proper error handling.
    
    Args:
        name: Name of the analysis for display
        func: Function to run
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        The function result or None on error
    """
    print(f"Running {name}...")
    try:
        result = func(*args, **kwargs)
        return result
    except Exception as e:
        print(f"Error in {name}: {e}")
        if 'debug' in kwargs and kwargs['debug']:
            import traceback
            print(traceback.format_exc())
        return None

def main():
    """Main function to run the RBS analysis."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Check dependencies if not in debug mode
        if not args.debug and not check_dependencies():
            sys.exit(1)
        
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Run unit tests if requested
        if args.test:
            if unit_tests_module is None:
                print("Unit tests module not available.")
                sys.exit(1)
                
            print("Running unit tests...")
            success = unit_tests_module.run_tests()
            if not success:
                print("Unit tests failed. Exiting...")
                sys.exit(1)
            else:
                print("All unit tests passed.")
                # If only testing was requested, exit
                if not any([args.all, args.basic, args.visualization, args.graph, 
                          args.coverage, args.tech_frequency, args.temporal,
                          args.correlation, args.spatial, args.integration,
                          args.prediction, args.dashboard, args.report,
                          args.advanced_coverage, args.coverage_quality, 
                          args.coverage_prediction, args.advanced_graph,
                          args.educational_docs]):
                    sys.exit(0)
        
        # Load and process data
        print(f"Loading data from {args.input}...")
        try:
            gdf_rbs = load_and_process_data(args.input)
            print(f"Loaded {len(gdf_rbs)} RBS records.")
        except Exception as e:
            print(f"Error loading data: {e}")
            if args.debug:
                import traceback
                print(traceback.format_exc())
            sys.exit(1)
        
        # Create a timestamp for output subdirectories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(args.output, f"analysis_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Configure GPU use if requested
        if args.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
                    # Set environment variable for other modules to check
                    os.environ['USE_GPU'] = 'true'
                else:
                    print("GPU requested but not available. Using CPU instead.")
                    args.use_gpu = False
            except ImportError:
                print("PyTorch not installed. Cannot use GPU acceleration.")
                args.use_gpu = False
        
        # Run requested analyses
        if args.all or args.basic:
            if analysis_module is not None and hasattr(analysis_module, 'run_basic_analysis'):
                run_analysis_safely("basic analysis", analysis_module.run_basic_analysis, 
                                    gdf_rbs, results_dir, debug=args.debug)
            else:
                print("Basic analysis module not available.")
        
        if args.all or args.visualization:
            if visualization_module is not None and hasattr(visualization_module, 'create_visualizations'):
                run_analysis_safely("visualization", visualization_module.create_visualizations, 
                                    gdf_rbs, results_dir, debug=args.debug)
            else:
                print("Visualization module not available.")
        
        if args.all or args.graph:
            if graph_analysis_module is not None and hasattr(graph_analysis_module, 'run_graph_analysis'):
                run_analysis_safely("graph analysis", graph_analysis_module.run_graph_analysis, 
                                     gdf_rbs, results_dir, debug=args.debug)
            else:
                print("Graph analysis module not available.")
        
        if args.all or args.coverage:
            if coverage_models_module is not None and hasattr(coverage_models_module, 'estimate_coverage'):
                gdf_with_coverage = run_analysis_safely("coverage estimation", 
                                                      coverage_models_module.estimate_coverage, 
                                                      gdf_rbs, debug=args.debug)
                if gdf_with_coverage is not None:
                    output_file = os.path.join(results_dir, 'coverage_estimates.geojson')
                    gdf_with_coverage.to_file(output_file, driver='GeoJSON')
                    print(f"Coverage estimates saved to {output_file}")
            else:
                print("Coverage models module not available.")
        
        if args.all or args.tech_frequency:
            if tech_frequency_analysis_module is not None and hasattr(tech_frequency_analysis_module, 'run_tech_frequency_analysis'):
                run_analysis_safely("technology and frequency analysis", 
                                    tech_frequency_analysis_module.run_tech_frequency_analysis, 
                                    gdf_rbs, results_dir, debug=args.debug)
            else:
                print("Technology and frequency analysis module not available.")
        
        if args.all or args.temporal:
            if advanced_temporal_analysis_module is not None and hasattr(advanced_temporal_analysis_module, 'run_temporal_analysis'):
                run_analysis_safely("advanced temporal analysis", 
                                    advanced_temporal_analysis_module.run_temporal_analysis, 
                                    gdf_rbs, results_dir, time_field=args.time_field, debug=args.debug)
            else:
                print("Advanced temporal analysis module not available.")
        
        if args.all or args.correlation:
            if correlation_analysis_module is not None and hasattr(correlation_analysis_module, 'run_correlation_analysis'):
                run_analysis_safely("correlation analysis", 
                                    correlation_analysis_module.run_correlation_analysis, 
                                    gdf_rbs, results_dir, debug=args.debug)
            else:
                print("Correlation analysis module not available.")
        
        if args.all or args.spatial:
            if spatial_analysis_module is not None and hasattr(spatial_analysis_module, 'run_spatial_analysis'):
                run_analysis_safely("spatial analysis", 
                                    spatial_analysis_module.run_spatial_analysis, 
                                    gdf_rbs, results_dir, debug=args.debug)
            else:
                print("Spatial analysis module not available.")
        
        if args.all or args.integration:
            if integration_analysis_module is not None and hasattr(integration_analysis_module, 'run_integration_analysis'):
                run_analysis_safely("integration analysis", 
                                    integration_analysis_module.run_integration_analysis, 
                                    gdf_rbs, results_dir, debug=args.debug)
            else:
                print("Integration analysis module not available.")
        
        if args.all or args.prediction:
            if prediction_module_module is not None and hasattr(prediction_module_module, 'run_prediction_analysis'):
                run_analysis_safely("prediction analysis", 
                                   prediction_module_module.run_prediction_analysis, 
                                   gdf_rbs, results_dir, use_gpu=args.use_gpu, debug=args.debug)
            else:
                print("Prediction module not available.")
        
        if args.all or args.advanced_coverage:
            if advanced_coverage_visualization_module is not None and hasattr(advanced_coverage_visualization_module, 'run_advanced_coverage_visualization'):
                advanced_coverage_dir = os.path.join(results_dir, 'advanced_coverage')
                os.makedirs(advanced_coverage_dir, exist_ok=True)
                run_analysis_safely("advanced coverage visualization", 
                                   advanced_coverage_visualization_module.run_advanced_coverage_visualization, 
                                   gdf_rbs, advanced_coverage_dir, debug=args.debug)
                print(f"Advanced coverage visualizations saved to {advanced_coverage_dir}")
            else:
                print("Advanced coverage visualization module not available.")
        
        if args.all or args.coverage_quality:
            if coverage_quality_analysis_module is not None and hasattr(coverage_quality_analysis_module, 'run_coverage_quality_analysis'):
                coverage_quality_dir = os.path.join(results_dir, 'coverage_quality')
                os.makedirs(coverage_quality_dir, exist_ok=True)
                run_analysis_safely("coverage quality analysis", 
                                   coverage_quality_analysis_module.run_coverage_quality_analysis, 
                                   gdf_rbs, coverage_quality_dir, debug=args.debug)
                print(f"Coverage quality analysis saved to {coverage_quality_dir}")
            else:
                print("Coverage quality analysis module not available.")
        
        if args.all or args.coverage_prediction:
            if coverage_prediction_module is not None and hasattr(coverage_prediction_module, 'run_coverage_prediction'):
                coverage_prediction_dir = os.path.join(results_dir, 'coverage_prediction')
                os.makedirs(coverage_prediction_dir, exist_ok=True)
                run_analysis_safely("coverage prediction", 
                                   coverage_prediction_module.run_coverage_prediction, 
                                   gdf_rbs, coverage_prediction_dir, use_gpu=args.use_gpu, debug=args.debug)
                print(f"Coverage prediction results saved to {coverage_prediction_dir}")
            else:
                print("Coverage prediction module not available.")
        
        if args.all or args.advanced_graph:
            if advanced_graph_analysis_module is not None and hasattr(advanced_graph_analysis_module, 'run_advanced_graph_analysis'):
                advanced_graph_dir = os.path.join(results_dir, 'advanced_graph')
                os.makedirs(advanced_graph_dir, exist_ok=True)
                run_analysis_safely("advanced graph analysis", 
                                   advanced_graph_analysis_module.run_advanced_graph_analysis, 
                                   gdf_rbs, advanced_graph_dir, time_field=args.time_field, debug=args.debug)
                print(f"Advanced graph analysis saved to {advanced_graph_dir}")
            else:
                print("Advanced graph analysis module not available.")
        
        if args.all or args.educational_docs:
            if educational_documentation_module is not None and hasattr(educational_documentation_module, 'create_educational_documentation'):
                educational_docs_dir = os.path.join(results_dir, 'educational_docs')
                os.makedirs(educational_docs_dir, exist_ok=True)
                run_analysis_safely("educational documentation", 
                                   educational_documentation_module.create_educational_documentation, 
                                   gdf_rbs, educational_docs_dir, debug=args.debug)
                print(f"Educational documentation created at {educational_docs_dir}")
                print(f"Open {os.path.join(educational_docs_dir, 'index.html')} in a web browser to view.")
            else:
                print("Educational documentation module not available.")
        
        if args.all or args.dashboard:
            if dashboard_interactive_module is not None and hasattr(dashboard_interactive_module, 'run_dashboard'):
                dashboard_path = run_analysis_safely("interactive dashboard", 
                                                   dashboard_interactive_module.run_dashboard, 
                                                   gdf_rbs, results_dir, debug=args.debug)
                if dashboard_path:
                    print(f"Interactive dashboard saved to {dashboard_path}")
            else:
                print("Dashboard module not available.")
        
        if args.all or args.report:
            if report_generator_module is not None and hasattr(report_generator_module, 'run_report_generation'):
                report_path = run_analysis_safely("report generation", 
                                                report_generator_module.run_report_generation, 
                                                gdf_rbs, results_dir, debug=args.debug)
                if report_path:
                    print(f"Report generated at {report_path}")
            else:
                print("Report generator module not available.")
        
        print(f"Analysis completed. Results saved to {results_dir}")
    
    except KeyboardInterrupt:
        print("Analysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.debug:
            import traceback
            print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 