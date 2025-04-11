"""
Module for unit tests and validation of the Radio Base Stations (RBS) analysis code.
Contains tests for data preprocessing, analysis functions, and visualizations.
"""

import unittest
import pandas as pd
import numpy as np
import geopandas as gpd
import os
import sys
import matplotlib.pyplot as plt
from shapely.geometry import Point
import tempfile
import shutil

# Import modules to test
from tech_frequency_analysis import preprocess_tech_frequency_data, create_tech_scatter_plot
from advanced_temporal_analysis import preprocess_temporal_data, create_deployment_timeline
from correlation_analysis import preprocess_correlation_data, create_correlation_matrix
from spatial_analysis import preprocess_spatial_data, create_density_map
from integration_analysis import preprocess_integrated_data
from prediction_module import preprocess_prediction_data
from report_generator import generate_summary_statistics

# Create a test fixture with synthetic RBS data
def create_test_data(n_samples=100):
    """
    Creates synthetic RBS data for testing.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        GeoDataFrame: Synthetic RBS data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate random dates
    start_date = pd.Timestamp('2010-01-01')
    end_date = pd.Timestamp('2022-12-31')
    date_range = (end_date - start_date).days
    random_dates = [start_date + pd.Timedelta(days=np.random.randint(0, date_range)) 
                   for _ in range(n_samples)]
    
    # Generate random coordinates around São Paulo
    lats = -23.55 + np.random.normal(0, 0.2, n_samples)
    lons = -46.63 + np.random.normal(0, 0.2, n_samples)
    
    # Generate random frequencies
    frequencies = np.random.choice([850, 900, 1800, 2100, 2600, 3500], size=n_samples)
    
    # Generate random power values
    power_values = np.random.uniform(10, 50, size=n_samples)
    
    # Generate random antenna heights
    antenna_heights = np.random.uniform(15, 120, size=n_samples)
    
    # Generate random operators
    operators = np.random.choice(['CLARO', 'VIVO', 'TIM', 'OI'], size=n_samples)
    
    # Generate random technologies
    technologies = []
    for freq in frequencies:
        if freq < 1000:
            tech = np.random.choice(['2G', '3G'], p=[0.3, 0.7])
        elif freq < 2200:
            tech = np.random.choice(['3G', '4G'], p=[0.6, 0.4])
        elif freq < 3000:
            tech = np.random.choice(['4G', '4G+'], p=[0.7, 0.3])
        else:
            tech = np.random.choice(['4G+', '5G'], p=[0.2, 0.8])
        technologies.append(tech)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'installation_date': random_dates,
        'FreqTxMHz': frequencies,
        'PotenciaTransmissorWatts': power_values,
        'AlturaAntena': antenna_heights,
        'Operator': operators,
        'Tecnologia': technologies,
        'latitude': lats,
        'longitude': lons
    })
    
    # Create a GeoDataFrame
    geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    return gdf

class TestPreprocessingFunctions(unittest.TestCase):
    """Test cases for data preprocessing functions."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = create_test_data()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_tech_frequency_preprocessing(self):
        """Test preprocessing for technology and frequency analysis."""
        df = preprocess_tech_frequency_data(self.test_data)
        
        # Check if required columns are present
        self.assertIn('FreqTxMHz', df.columns)
        self.assertIn('PotenciaTransmissorWatts', df.columns)
        self.assertIn('Operator', df.columns)
        self.assertIn('Tecnologia', df.columns)
        self.assertIn('FrequencyBand', df.columns)
        
        # Check if frequency bands are created correctly
        self.assertTrue(all(isinstance(band, str) for band in df['FrequencyBand'].dropna()))
        
        # Check if numeric columns are numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(df['FreqTxMHz']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['PotenciaTransmissorWatts']))
    
    def test_temporal_preprocessing(self):
        """Test preprocessing for temporal analysis."""
        df = preprocess_temporal_data(self.test_data)
        
        # Check if required columns are present
        self.assertIn('installation_date', df.columns)
        self.assertIn('year', df.columns)
        self.assertIn('month', df.columns)
        self.assertIn('quarter', df.columns)
        
        # Check if date components are extracted correctly
        self.assertTrue(pd.api.types.is_numeric_dtype(df['year']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['month']))
        self.assertTrue(all(1 <= month <= 12 for month in df['month']))
        self.assertTrue(all(1 <= quarter <= 4 for quarter in df['quarter']))
    
    def test_correlation_preprocessing(self):
        """Test preprocessing for correlation analysis."""
        df = preprocess_correlation_data(self.test_data)
        
        # Check if required columns are present
        self.assertIn('FreqTxMHz', df.columns)
        self.assertIn('PotenciaTransmissorWatts', df.columns)
        self.assertIn('AlturaAntena', df.columns)
        self.assertIn('Coverage_Radius_km', df.columns)
        
        # Check if coverage radius is calculated
        self.assertTrue(pd.api.types.is_numeric_dtype(df['Coverage_Radius_km']))
        self.assertTrue(all(df['Coverage_Radius_km'].dropna() > 0))
    
    def test_spatial_preprocessing(self):
        """Test preprocessing for spatial analysis."""
        gdf = preprocess_spatial_data(self.test_data)
        
        # Check if it's a GeoDataFrame
        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        
        # Check if geometry is present
        self.assertTrue(all(isinstance(geom, Point) for geom in gdf.geometry))
        
        # Check if CRS is set
        self.assertEqual(gdf.crs, "EPSG:4326")
    
    def test_integrated_preprocessing(self):
        """Test preprocessing for integrated analysis."""
        df = preprocess_integrated_data(self.test_data)
        
        # Check if it contains columns from both temporal and tech preprocessing
        self.assertIn('year', df.columns)
        self.assertIn('month', df.columns)
        self.assertIn('FreqTxMHz', df.columns)
        self.assertIn('Tecnologia', df.columns)
        self.assertIn('FrequencyBand', df.columns)
    
    def test_prediction_preprocessing(self):
        """Test preprocessing for prediction analysis."""
        data_dict = preprocess_prediction_data(self.test_data)
        
        # Check if it returns a dictionary with time series data
        self.assertIsInstance(data_dict, dict)
        self.assertIn('daily', data_dict)
        self.assertIn('monthly', data_dict)
        self.assertIn('yearly', data_dict)
        
        # Check if time series data has the expected format
        self.assertIn('date', data_dict['monthly'].columns)
        self.assertIn('installations', data_dict['monthly'].columns)

class TestVisualizationFunctions(unittest.TestCase):
    """Test cases for visualization functions."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = create_test_data()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create output directory
        os.makedirs(os.path.join(self.temp_dir, 'viz_test'), exist_ok=True)
    
    def tearDown(self):
        """Clean up temporary files."""
        plt.close('all')  # Close any open figures
        shutil.rmtree(self.temp_dir)
    
    def test_tech_scatter_plot(self):
        """Test tech scatter plot creation."""
        output_path = os.path.join(self.temp_dir, 'viz_test/tech_scatter.png')
        
        # Create the plot
        create_tech_scatter_plot(self.test_data, output_path)
        
        # Check if the file was created
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.getsize(output_path) > 0)
    
    def test_deployment_timeline(self):
        """Test deployment timeline creation."""
        output_path = os.path.join(self.temp_dir, 'viz_test/timeline.png')
        
        # Create the plot
        create_deployment_timeline(self.test_data, output_path)
        
        # Check if the file was created
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.getsize(output_path) > 0)
    
    def test_correlation_matrix(self):
        """Test correlation matrix creation."""
        output_path = os.path.join(self.temp_dir, 'viz_test/correlation.png')
        
        # Create the plot
        create_correlation_matrix(self.test_data, output_path)
        
        # Check if the file was created
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.getsize(output_path) > 0)
    
    def test_density_map(self):
        """Test density map creation."""
        output_path = os.path.join(self.temp_dir, 'viz_test/density.html')
        
        # Create the map
        create_density_map(self.test_data, output_path)
        
        # Check if the file was created
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.getsize(output_path) > 0)
        
        # Also check for static version
        static_path = output_path.replace('.html', '_static.png')
        self.assertTrue(os.path.exists(static_path))
        self.assertTrue(os.path.getsize(static_path) > 0)

class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility and helper functions."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = create_test_data()
    
    def test_summary_statistics(self):
        """Test summary statistics generation."""
        stats_df = generate_summary_statistics(self.test_data)
        
        # Check if it returns a DataFrame
        self.assertIsInstance(stats_df, pd.DataFrame)
        
        # Check if it has the expected columns
        self.assertIn('Metric', stats_df.columns)
        self.assertIn('Value', stats_df.columns)
        
        # Check if it contains total count
        self.assertTrue(any('Total RBS Stations' in str(metric) for metric in stats_df['Metric']))
        
        # Check if it contains operator information
        self.assertTrue(any('Operator' in str(metric) for metric in stats_df['Metric']))
        
        # Check if it contains technology information
        self.assertTrue(any('Technology' in str(metric) for metric in stats_df['Metric']))

def run_tests():
    """Run all unit tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessingFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualizationFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 