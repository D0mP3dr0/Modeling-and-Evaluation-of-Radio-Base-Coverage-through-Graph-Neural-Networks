"""
Module for generating automated reports from Radio Base Stations (RBS) analysis.
Combines visualizations from various analysis modules into PDF and Excel reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import io
from fpdf import FPDF
import base64
from PIL import Image

# Import from other modules
from tech_frequency_analysis import preprocess_tech_frequency_data, run_tech_frequency_analysis
from advanced_temporal_analysis import preprocess_temporal_data, run_temporal_analysis
from correlation_analysis import preprocess_correlation_data, run_correlation_analysis
from spatial_analysis import preprocess_spatial_data, run_spatial_analysis
from integration_analysis import run_integration_analysis
from prediction_module import preprocess_prediction_data, run_prediction_analysis

class RBSReportPDF(FPDF):
    """Custom PDF class for RBS reports."""
    
    def __init__(self):
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297
        
    def header(self):
        # Set up the header
        self.set_font('Arial', 'B', 12)
        self.cell(self.WIDTH - 20, 10, 'Radio Base Stations Analysis Report', 0, 0, 'R')
        self.ln(20)
        
    def footer(self):
        # Set up the footer
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'R')
        
    def chapter_title(self, title):
        # Add a chapter title
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(5)
        
    def chapter_body(self, body):
        # Add chapter body text
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()
        
    def add_image(self, img_path, caption=None, w=180):
        """Add an image to the PDF with optional caption."""
        if os.path.exists(img_path):
            # Calculate height while maintaining aspect ratio
            img = Image.open(img_path)
            width, height = img.size
            h = w * height / width
            
            # Add the image
            self.image(img_path, x=15, y=None, w=w, h=h)
            
            # Add caption if provided
            if caption:
                self.ln(2)
                self.set_font('Arial', 'I', 10)
                self.cell(0, 10, caption, 0, 1, 'C')
                
            self.ln(10)
        else:
            self.cell(0, 10, f"Image not found: {img_path}", 0, 1)
            self.ln(5)

def generate_summary_statistics(gdf_rbs):
    """
    Generates a DataFrame with summary statistics about the RBS data.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        
    Returns:
        DataFrame: Summary statistics
    """
    stats = []
    
    # Total number of stations
    stats.append({
        'Metric': 'Total RBS Stations',
        'Value': len(gdf_rbs)
    })
    
    # Distribution by operator
    if 'Operator' in gdf_rbs.columns:
        operator_counts = gdf_rbs['Operator'].value_counts()
        for operator, count in operator_counts.items():
            stats.append({
                'Metric': f'Operator: {operator}',
                'Value': count,
                'Percentage': f'{count/len(gdf_rbs)*100:.1f}%'
            })
    
    # Distribution by technology
    if 'Tecnologia' in gdf_rbs.columns:
        tech_counts = gdf_rbs['Tecnologia'].value_counts()
        for tech, count in tech_counts.items():
            stats.append({
                'Metric': f'Technology: {tech}',
                'Value': count,
                'Percentage': f'{count/len(gdf_rbs)*100:.1f}%'
            })
    
    # Distribution by frequency bands (if available)
    if 'FrequencyBand' in gdf_rbs.columns:
        band_counts = gdf_rbs['FrequencyBand'].value_counts()
        for band, count in band_counts.items():
            stats.append({
                'Metric': f'Frequency Band: {band}',
                'Value': count,
                'Percentage': f'{count/len(gdf_rbs)*100:.1f}%'
            })
    elif 'FreqTxMHz' in gdf_rbs.columns:
        # Create frequency bands
        df_tech = preprocess_tech_frequency_data(gdf_rbs)
        if 'FrequencyBand' in df_tech.columns:
            band_counts = df_tech['FrequencyBand'].value_counts()
            for band, count in band_counts.items():
                stats.append({
                    'Metric': f'Frequency Band: {band}',
                    'Value': count,
                    'Percentage': f'{count/len(df_tech)*100:.1f}%'
                })
    
    # Temporal statistics if available
    if any(col in gdf_rbs.columns for col in ['installation_date', 'data_licenciamento']):
        df_temporal = preprocess_temporal_data(gdf_rbs)
        if 'installation_date' in df_temporal.columns:
            stats.append({
                'Metric': 'Earliest Installation Date',
                'Value': df_temporal['installation_date'].min().strftime('%Y-%m-%d')
            })
            stats.append({
                'Metric': 'Latest Installation Date',
                'Value': df_temporal['installation_date'].max().strftime('%Y-%m-%d')
            })
            
            # Installations by year
            yearly_counts = df_temporal.groupby(df_temporal['installation_date'].dt.year).size()
            for year, count in yearly_counts.items():
                stats.append({
                    'Metric': f'Installations in {year}',
                    'Value': count
                })
    
    # Create DataFrame
    return pd.DataFrame(stats)

def create_pdf_report(gdf_rbs, output_path, run_all_analyses=False):
    """
    Creates a comprehensive PDF report with visualizations and statistics.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        output_path: Path to save the PDF report
        run_all_analyses: Whether to run all analyses and generate new visualizations
    
    Returns:
        str: Path to the generated PDF report
    """
    # Create output directories
    report_dir = os.path.dirname(output_path)
    os.makedirs(report_dir, exist_ok=True)
    
    temp_dir = os.path.join(report_dir, 'temp_visualizations')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Run analyses if requested
    if run_all_analyses:
        print("Running all analyses to generate visualizations...")
        run_tech_frequency_analysis(gdf_rbs, temp_dir)
        run_temporal_analysis(gdf_rbs, temp_dir)
        run_correlation_analysis(gdf_rbs, temp_dir)
        run_spatial_analysis(gdf_rbs, temp_dir)
        run_integration_analysis(gdf_rbs, temp_dir)
        run_prediction_analysis(gdf_rbs, temp_dir)
    
    # Generate summary statistics
    summary_stats = generate_summary_statistics(gdf_rbs)
    
    # Create PDF
    pdf = RBSReportPDF()
    pdf.add_page()
    
    # Title page
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 40, 'Radio Base Stations', 0, 1, 'C')
    pdf.cell(0, 20, 'Analysis Report', 0, 1, 'C')
    pdf.set_font('Arial', 'I', 14)
    pdf.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'C')
    
    # Summary statistics
    pdf.add_page()
    pdf.chapter_title('1. Dataset Overview')
    pdf.chapter_body('This section provides an overview of the RBS dataset, including basic statistics and distribution by operator and technology.')
    
    # Create a table for summary statistics
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(100, 10, 'Metric', 1, 0, 'C')
    pdf.cell(40, 10, 'Value', 1, 0, 'C')
    pdf.cell(40, 10, 'Percentage', 1, 1, 'C')
    
    pdf.set_font('Arial', '', 10)
    for _, row in summary_stats.iterrows():
        pdf.cell(100, 10, str(row.get('Metric', '')), 1, 0)
        pdf.cell(40, 10, str(row.get('Value', '')), 1, 0)
        pdf.cell(40, 10, str(row.get('Percentage', '')), 1, 1)
    
    # Temporal analysis
    pdf.add_page()
    pdf.chapter_title('2. Temporal Analysis')
    pdf.chapter_body('This section shows the evolution of RBS deployments over time, including seasonality patterns and geographical expansion.')
    
    # Add temporal visualizations
    timeline_path = os.path.join(temp_dir, 'advanced_temporal_analysis/deployment_timeline.png')
    pdf.add_image(timeline_path, 'Figure 2.1: RBS Deployment Timeline by Operator')
    
    seasonality_path = os.path.join(temp_dir, 'advanced_temporal_analysis/seasonality_analysis.png')
    pdf.add_image(seasonality_path, 'Figure 2.2: Seasonality Patterns in RBS Installations')
    
    heatmap_path = os.path.join(temp_dir, 'advanced_temporal_analysis/chronological_heatmap.png')
    pdf.add_image(heatmap_path, 'Figure 2.3: Chronological Heatmap of RBS Deployments')
    
    # Technology and frequency analysis
    pdf.add_page()
    pdf.chapter_title('3. Technology and Frequency Analysis')
    pdf.chapter_body('This section examines the relationship between transmission frequency, power, and technology, as well as the distribution across operators.')
    
    # Add technology visualizations
    scatter_path = os.path.join(temp_dir, 'tech_frequency_analysis/tech_scatter_plot.png')
    pdf.add_image(scatter_path, 'Figure 3.1: Relationship Between Frequency, Power, and Technology')
    
    histogram_path = os.path.join(temp_dir, 'tech_frequency_analysis/frequency_histograms.png')
    pdf.add_image(histogram_path, 'Figure 3.2: Frequency Distribution by Operator')
    
    sunburst_path = os.path.join(temp_dir, 'tech_frequency_analysis/sunburst_chart.png')
    pdf.add_image(sunburst_path, 'Figure 3.3: Hierarchical Visualization of Operators, Technologies, and Frequency Bands')
    
    # Spatial analysis
    pdf.add_page()
    pdf.chapter_title('4. Spatial Analysis')
    pdf.chapter_body('This section analyzes the geographical distribution of RBS sites, including clustering and coverage patterns.')
    
    # Add spatial visualizations
    density_path = os.path.join(temp_dir, 'spatial_analysis/density_map_static.png')
    pdf.add_image(density_path, 'Figure 4.1: RBS Density Map')
    
    cluster_path = os.path.join(temp_dir, 'spatial_analysis/dbscan_clusters.png')
    pdf.add_image(cluster_path, 'Figure 4.2: DBSCAN Clustering of RBS Locations')
    
    voronoi_path = os.path.join(temp_dir, 'spatial_analysis/voronoi_coverage.png')
    pdf.add_image(voronoi_path, 'Figure 4.3: Voronoi Coverage Areas for RBS Stations')
    
    # Correlation analysis
    pdf.add_page()
    pdf.chapter_title('5. Correlation Analysis')
    pdf.chapter_body('This section examines the relationships between technical variables such as frequency, power, antenna height, and coverage radius.')
    
    # Add correlation visualizations
    correlation_path = os.path.join(temp_dir, 'correlation_analysis/correlation_matrix.png')
    pdf.add_image(correlation_path, 'Figure 5.1: Correlation Matrix of Technical Variables')
    
    pairplot_path = os.path.join(temp_dir, 'correlation_analysis/technical_pairplot.png')
    pdf.add_image(pairplot_path, 'Figure 5.2: Relationships Between Technical Variables')
    
    regression_path = os.path.join(temp_dir, 'correlation_analysis/regression_results.png')
    pdf.add_image(regression_path, 'Figure 5.3: Predictive Power for Coverage Radius')
    
    # Integration analysis
    pdf.add_page()
    pdf.chapter_title('6. Integrated Analysis')
    pdf.chapter_body('This section combines temporal and technological data to provide deeper insights into the evolution of RBS technologies and frequency bands.')
    
    # Add integration visualizations
    tech_evolution_path = os.path.join(temp_dir, 'integration_analysis/technology_evolution.png')
    pdf.add_image(tech_evolution_path, 'Figure 6.1: Evolution of RBS Technologies Over Time')
    
    freq_migration_path = os.path.join(temp_dir, 'integration_analysis/frequency_migration.png')
    pdf.add_image(freq_migration_path, 'Figure 6.2: Migration of Frequency Bands Over Time')
    
    operator_tech_path = os.path.join(temp_dir, 'integration_analysis/operator_tech_timeline.png')
    pdf.add_image(operator_tech_path, 'Figure 6.3: Timeline of Technology Adoption by Operator')
    
    # Prediction analysis
    pdf.add_page()
    pdf.chapter_title('7. Future Trends')
    pdf.chapter_body('This section presents forecasts for future RBS deployments and technology adoption trends based on historical data.')
    
    # Add prediction visualizations
    arima_path = os.path.join(temp_dir, 'prediction_analysis/arima_forecast_monthly.png')
    pdf.add_image(arima_path, 'Figure 7.1: ARIMA Forecast of Monthly RBS Installations')
    
    prophet_path = os.path.join(temp_dir, 'prediction_analysis/prophet_forecast_monthly.png')
    pdf.add_image(prophet_path, 'Figure 7.2: Prophet Forecast of RBS Installations')
    
    tech_trends_path = os.path.join(temp_dir, 'prediction_analysis/technology_trends_forecast.png')
    pdf.add_image(tech_trends_path, 'Figure 7.3: Technology Adoption Forecast')
    
    # Conclusions
    pdf.add_page()
    pdf.chapter_title('8. Conclusions')
    
    # Generate some basic conclusions based on the data
    conclusions = "Based on the analysis of the Radio Base Stations dataset, the following conclusions can be drawn:\n\n"
    
    # Add some operator-specific conclusions if available
    if 'Operator' in gdf_rbs.columns:
        dominant_operator = gdf_rbs['Operator'].value_counts().idxmax()
        operator_share = gdf_rbs['Operator'].value_counts().max() / len(gdf_rbs) * 100
        conclusions += f"- {dominant_operator} is the dominant operator with approximately {operator_share:.1f}% of the RBS stations.\n\n"
    
    # Add technology-specific conclusions if available
    if 'Tecnologia' in gdf_rbs.columns:
        dominant_tech = gdf_rbs['Tecnologia'].value_counts().idxmax()
        tech_share = gdf_rbs['Tecnologia'].value_counts().max() / len(gdf_rbs) * 100
        conclusions += f"- {dominant_tech} is the most deployed technology, representing about {tech_share:.1f}% of all installations.\n\n"
    
    # Add temporal conclusions if available
    df_temporal = preprocess_temporal_data(gdf_rbs)
    if 'installation_date' in df_temporal.columns:
        yearly_counts = df_temporal.groupby(df_temporal['installation_date'].dt.year).size()
        if len(yearly_counts) > 1:
            growth_rate = (yearly_counts.iloc[-1] / yearly_counts.iloc[0] - 1) * 100
            conclusions += f"- The number of RBS installations has {'increased' if growth_rate > 0 else 'decreased'} by approximately {abs(growth_rate):.1f}% from {yearly_counts.index[0]} to {yearly_counts.index[-1]}.\n\n"
    
    # General conclusions
    conclusions += "- The spatial analysis reveals clusters of RBS deployments in densely populated areas, with some regions showing potential coverage gaps.\n\n"
    conclusions += "- Correlation analysis suggests a significant relationship between transmission frequency and coverage radius, with lower frequencies providing wider coverage.\n\n"
    conclusions += "- Future trends indicate continued growth in deployment of newer technologies, with a gradual migration towards higher frequency bands."
    
    pdf.chapter_body(conclusions)
    
    # Save the PDF
    pdf.output(output_path)
    print(f"PDF report saved to {output_path}")
    
    return output_path

def export_to_excel(gdf_rbs, output_path, include_statistics=True):
    """
    Exports the RBS data and analysis results to an Excel file.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        output_path: Path to save the Excel file
        include_statistics: Whether to include summary statistics
        
    Returns:
        str: Path to the generated Excel file
    """
    # Create a writer
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    
    # Write the main data
    gdf_rbs.drop(columns=['geometry'], errors='ignore').to_excel(writer, sheet_name='RBS_Data', index=False)
    
    # Write summary statistics if requested
    if include_statistics:
        summary_stats = generate_summary_statistics(gdf_rbs)
        summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    
    # Add operator-specific sheets if available
    if 'Operator' in gdf_rbs.columns:
        for operator in gdf_rbs['Operator'].unique():
            operator_data = gdf_rbs[gdf_rbs['Operator'] == operator]
            operator_data.drop(columns=['geometry'], errors='ignore').to_excel(
                writer, 
                sheet_name=f'{operator}_Data'[:31],  # Excel has 31 character limit for sheet names
                index=False
            )
    
    # Add technology-specific sheets if available
    if 'Tecnologia' in gdf_rbs.columns:
        for tech in gdf_rbs['Tecnologia'].unique():
            tech_data = gdf_rbs[gdf_rbs['Tecnologia'] == tech]
            tech_data.drop(columns=['geometry'], errors='ignore').to_excel(
                writer, 
                sheet_name=f'{tech}_Data'[:31],
                index=False
            )
    
    # Add temporal analysis sheet if available
    df_temporal = preprocess_temporal_data(gdf_rbs)
    if 'installation_date' in df_temporal.columns:
        # Create yearly summary
        yearly_summary = df_temporal.groupby(df_temporal['installation_date'].dt.year).size().reset_index()
        yearly_summary.columns = ['Year', 'Installations']
        yearly_summary['Cumulative'] = yearly_summary['Installations'].cumsum()
        
        # If operator data is available, add breakdown by operator
        if 'Operator' in df_temporal.columns:
            operator_pivot = df_temporal.pivot_table(
                index=df_temporal['installation_date'].dt.year,
                columns='Operator',
                aggfunc='size',
                fill_value=0
            ).reset_index()
            operator_pivot.columns.name = None
            
            # Merge with yearly summary
            yearly_summary = yearly_summary.merge(
                operator_pivot,
                left_on='Year',
                right_on='installation_date',
                how='left'
            ).drop(columns=['installation_date'])
        
        yearly_summary.to_excel(writer, sheet_name='Temporal_Analysis', index=False)
    
    # Add frequency analysis sheet if available
    df_tech = preprocess_tech_frequency_data(gdf_rbs)
    if 'FrequencyBand' in df_tech.columns:
        # Create frequency band summary
        band_summary = df_tech['FrequencyBand'].value_counts().reset_index()
        band_summary.columns = ['Frequency Band', 'Count']
        band_summary['Percentage'] = band_summary['Count'] / len(df_tech) * 100
        
        # If operator data is available, add breakdown by operator
        if 'Operator' in df_tech.columns:
            band_operator_pivot = pd.pivot_table(
                df_tech,
                index='FrequencyBand',
                columns='Operator',
                aggfunc='size',
                fill_value=0
            ).reset_index()
            
            # Merge with band summary
            band_summary = band_summary.merge(
                band_operator_pivot,
                left_on='Frequency Band',
                right_on='FrequencyBand',
                how='left'
            ).drop(columns=['FrequencyBand'])
        
        band_summary.to_excel(writer, sheet_name='Frequency_Analysis', index=False)
    
    # Close the writer
    writer.close()
    print(f"Excel report saved to {output_path}")
    
    return output_path

def run_report_generation(gdf_rbs, output_dir, file_prefix='rbs_analysis'):
    """
    Runs the report generation process, creating both PDF and Excel reports.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        output_dir: Directory to save the reports
        file_prefix: Prefix for the report filenames
        
    Returns:
        tuple: Paths to the generated PDF and Excel reports
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define output paths
    pdf_path = os.path.join(output_dir, f"{file_prefix}_{timestamp}.pdf")
    excel_path = os.path.join(output_dir, f"{file_prefix}_{timestamp}.xlsx")
    
    # Create reports
    pdf_report = create_pdf_report(gdf_rbs, pdf_path, run_all_analyses=True)
    excel_report = export_to_excel(gdf_rbs, excel_path)
    
    print(f"Report generation completed. Files saved to {output_dir}")
    return pdf_report, excel_report 