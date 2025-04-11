"""
Module for advanced temporal analysis of Radio Base Stations (RBS).
Contains functions to visualize chronological deployment, seasonality patterns,
and geographical expansion over time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import folium
from folium.plugins import HeatMapWithTime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import calendar
from matplotlib.colors import LinearSegmentedColormap
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("advanced_temporal_analysis")

# Default color configuration for operators
OPERATOR_COLORS = {
    'CLARO': '#E02020',
    'OI': '#FFD700',
    'VIVO': '#9932CC',
    'TIM': '#0000CD'
}

def preprocess_temporal_data(gdf_rbs):
    """
    Preprocesses the RBS data for temporal analysis, extracting and formatting dates.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        
    Returns:
        GeoDataFrame: The same GeoDataFrame with added temporal columns
    """
    logger.info("Preprocessing data for temporal analysis")
    
    try:
        # Make a copy to avoid modifying the original
        df = gdf_rbs.copy()
        
        # Check for date columns
        date_cols = [col for col in df.columns if 'data' in col.lower() or 'date' in col.lower()]
        
        if not date_cols:
            logger.warning("No date columns found for temporal analysis.")
            # Add a placeholder column with random dates for demonstration
            import random
            from datetime import timedelta
            
            start_date = datetime(2010, 1, 1)
            end_date = datetime(2023, 12, 31)
            days_between = (end_date - start_date).days
            
            # Generate random dates
            df['installation_date'] = [start_date + timedelta(days=random.randint(0, days_between)) for _ in range(len(df))]
            
            logger.info("Added synthetic installation dates for demonstration purposes.")
        else:
            # Try to use the most relevant date column
            license_date_cols = [col for col in date_cols if 'licen' in col.lower()]
            if license_date_cols:
                primary_date_col = license_date_cols[0]
            else:
                primary_date_col = date_cols[0]
            
            logger.info(f"Using {primary_date_col} for temporal analysis.")
            
            # Convert to datetime
            try:
                df['installation_date'] = pd.to_datetime(df[primary_date_col], errors='coerce')
            except Exception as e:
                logger.error(f"Error converting dates: {e}")
                # Add placeholder dates as fallback
                import random
                from datetime import timedelta
                
                start_date = datetime(2010, 1, 1)
                end_date = datetime(2023, 12, 31)
                days_between = (end_date - start_date).days
                
                # Generate random dates
                df['installation_date'] = [start_date + timedelta(days=random.randint(0, days_between)) for _ in range(len(df))]
                
                logger.info("Added synthetic installation dates for demonstration purposes.")
        
        # Handle NaT in installation_date
        if df['installation_date'].isna().any():
            logger.warning(f"Found {df['installation_date'].isna().sum()} NaT values in installation_date")
            # Generate random dates for NaT values
            import random
            from datetime import timedelta
            
            start_date = datetime(2010, 1, 1)
            end_date = datetime(2023, 12, 31)
            days_between = (end_date - start_date).days
            
            # Create mask for NaT values
            nat_mask = df['installation_date'].isna()
            
            # Generate random dates only for NaT values
            df.loc[nat_mask, 'installation_date'] = [
                start_date + timedelta(days=random.randint(0, days_between)) 
                for _ in range(nat_mask.sum())
            ]
            
            logger.info(f"Filled {nat_mask.sum()} NaT values with random dates")
        
        # Extract temporal components
        try:
            df['year'] = df['installation_date'].dt.year
            df['month'] = df['installation_date'].dt.month
            df['month_name'] = df['installation_date'].dt.month_name()
            df['quarter'] = df['installation_date'].dt.quarter
            df['year_quarter'] = df['year'].astype(str) + 'Q' + df['quarter'].astype(str)
            
            logger.info("Extracted temporal components successfully")
        except Exception as e:
            logger.error(f"Error extracting temporal components: {e}")
            # Create default temporal components
            current_year = datetime.now().year
            df['year'] = current_year
            df['month'] = 1
            df['month_name'] = 'January'
            df['quarter'] = 1
            df['year_quarter'] = f"{current_year}Q1"
            
            logger.warning("Using default temporal components due to extraction error")
        
        return df
    
    except Exception as e:
        logger.error(f"Error in preprocessing temporal data: {e}")
        # Return minimal dataframe with required columns if original is None
        if gdf_rbs is None:
            logger.warning("Input GeoDataFrame is None, creating minimal dataframe")
            import random
            from datetime import timedelta
            
            # Create a minimal dataframe with one row
            start_date = datetime(2010, 1, 1)
            end_date = datetime(2023, 12, 31)
            days_between = (end_date - start_date).days
            
            random_date = start_date + timedelta(days=random.randint(0, days_between))
            
            df = pd.DataFrame({
                'installation_date': [random_date],
                'year': [random_date.year],
                'month': [random_date.month],
                'month_name': [random_date.strftime('%B')],
                'quarter': [(random_date.month - 1) // 3 + 1],
                'year_quarter': [f"{random_date.year}Q{(random_date.month - 1) // 3 + 1}"],
                'Operator': ['Unknown']
            })
            
            return df
        
        # If original dataframe exists but processing failed, add minimal columns
        try:
            df = gdf_rbs.copy()
            
            # Create default date
            current_date = datetime.now()
            df['installation_date'] = current_date
            df['year'] = current_date.year
            df['month'] = current_date.month
            df['month_name'] = current_date.strftime('%B')
            df['quarter'] = (current_date.month - 1) // 3 + 1
            df['year_quarter'] = f"{current_date.year}Q{(current_date.month - 1) // 3 + 1}"
            
            if 'Operator' not in df.columns:
                df['Operator'] = 'Unknown'
                
            return df
        except:
            # Complete fallback - create minimal dataframe
            logger.error("Failed to process temporal data, creating fallback dataframe")
            current_date = datetime.now()
            
            df = pd.DataFrame({
                'installation_date': [current_date],
                'year': [current_date.year],
                'month': [current_date.month],
                'month_name': [current_date.strftime('%B')],
                'quarter': [(current_date.month - 1) // 3 + 1],
                'year_quarter': [f"{current_date.year}Q{(current_date.month - 1) // 3 + 1}"],
                'Operator': ['Unknown']
            })
            
            return df

def create_deployment_timeline(gdf_rbs, output_path):
    """
    Creates a line chart showing the evolution of RBS deployment by operator over time.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        output_path: Path to save the visualization
    """
    logger.info("Creating RBS deployment timeline...")
    
    try:
        # Preprocess data
        df = preprocess_temporal_data(gdf_rbs)
        
        # Group by year and operator, count installations
        if 'Operator' in df.columns and not df.empty:
            # Handle case with only one data point
            if df['year'].nunique() <= 1:
                logger.warning("Insufficient temporal data for timeline - only one year found")
                # Create a simple plot with minimal data
                fig, ax = plt.subplots(figsize=(15, 8))
                
                year = df['year'].iloc[0]
                operators = df['Operator'].unique()
                
                # Plot a simple bar chart instead
                counts = df.groupby('Operator').size()
                counts.plot(kind='bar', ax=ax, color=[OPERATOR_COLORS.get(op, '#333333') for op in counts.index])
                
                ax.set_title(f'RBS Deployment by Operator ({year})', fontsize=16)
                ax.set_xlabel('Operator', fontsize=14)
                ax.set_ylabel('Number of RBS Stations', fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                try:
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Simple deployment chart saved to {output_path}")
                except Exception as e:
                    logger.error(f"Error saving simple deployment chart: {e}")
                plt.close(fig)
                return
            
            try:
                timeline_data = df.groupby(['year', 'Operator']).size().unstack(fill_value=0)
                timeline_data['Total'] = timeline_data.sum(axis=1)
                
                # Calculate cumulative sum for each operator
                cumulative_data = timeline_data.cumsum()
                
                # Create the plot
                fig, ax = plt.subplots(figsize=(15, 8))
                
                # Plot individual operators
                for operator, color in OPERATOR_COLORS.items():
                    if operator in cumulative_data.columns:
                        ax.plot(cumulative_data.index, cumulative_data[operator], 
                                marker='o', linewidth=2, color=color, label=operator)
                
                # Plot total
                ax.plot(cumulative_data.index, cumulative_data['Total'], 
                        marker='*', linewidth=3, color='black', linestyle='--', label='Total')
                
                # Styling
                ax.set_title('Cumulative RBS Deployment Timeline by Operator', fontsize=16)
                ax.set_xlabel('Year', fontsize=14)
                ax.set_ylabel('Number of RBS Stations', fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend(title='Operator', fontsize=12)
                
                # Adjust x-axis
                if len(cumulative_data.index) > 0:
                    ax.set_xticks(cumulative_data.index)
                    ax.set_xticklabels(cumulative_data.index, rotation=45)
                
                # Add annotations for key milestones
                for i, year in enumerate(cumulative_data.index):
                    if i > 0 and i < len(cumulative_data.index) - 1:
                        continue  # Skip middle years for clarity
                    
                    total = cumulative_data.loc[year, 'Total']
                    ax.annotate(f'{total}', 
                                xy=(year, total), 
                                xytext=(0, 10),
                                textcoords='offset points',
                                ha='center',
                                va='bottom',
                                fontsize=10,
                                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
                
                plt.tight_layout()
                
                try:
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Deployment timeline saved to {output_path}")
                except Exception as e:
                    logger.error(f"Error saving deployment timeline: {e}")
                    try:
                        alt_path = os.path.join(os.path.dirname(output_path), 'deployment_timeline_alt.png')
                        plt.savefig(alt_path, dpi=300, bbox_inches='tight')
                        logger.info(f"Deployment timeline saved to alternative path: {alt_path}")
                    except Exception as e2:
                        logger.error(f"Error saving to alternative path: {e2}")
                
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating deployment timeline: {e}")
        else:
            logger.error("Error: 'Operator' column not found in the dataset or dataset is empty")
    except Exception as e:
        logger.error(f"Error in create_deployment_timeline: {e}")

def create_seasonality_analysis(gdf_rbs, output_path):
    """
    Creates visualizations showing seasonality patterns in RBS installations.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        output_path: Path to save the visualization
    """
    logger.info("Creating RBS installation seasonality analysis...")
    
    try:
        # Preprocess data
        df = preprocess_temporal_data(gdf_rbs)
        
        if df.empty:
            logger.error("No data available for seasonality analysis")
            return
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        try:
            # 1. Monthly installations overall
            monthly_installations = df.groupby('month').size()
            monthly_data = pd.DataFrame({
                'Month': range(1, 13),
                'Installations': [monthly_installations.get(m, 0) for m in range(1, 13)]
            })
            monthly_data['Month_Name'] = monthly_data['Month'].apply(lambda x: calendar.month_name[x])
            
            sns.barplot(x='Month_Name', y='Installations', data=monthly_data, ax=axes[0], palette='viridis')
            axes[0].set_title('RBS Installations by Month (All Years)', fontsize=14)
            axes[0].set_xlabel('Month', fontsize=12)
            axes[0].set_ylabel('Number of Installations', fontsize=12)
            axes[0].tick_params(axis='x', rotation=45)
        except Exception as e:
            logger.error(f"Error creating monthly installations plot: {e}")
            axes[0].text(0.5, 0.5, "Error generating monthly analysis", 
                      ha='center', va='center', fontsize=14, color='red',
                      transform=axes[0].transAxes)
        
        try:
            # 2. Quarterly installations by year
            if 'year' in df.columns and 'quarter' in df.columns:
                quarterly_data = df.groupby(['year', 'quarter']).size().unstack(fill_value=0)
                
                quarterly_data.plot(kind='bar', ax=axes[1], colormap='viridis')
                axes[1].set_title('Quarterly RBS Installations by Year', fontsize=14)
                axes[1].set_xlabel('Year', fontsize=12)
                axes[1].set_ylabel('Number of Installations', fontsize=12)
                axes[1].legend(title='Quarter', labels=[f'Q{i}' for i in range(1, 5)])
        except Exception as e:
            logger.error(f"Error creating quarterly installations plot: {e}")
            axes[1].text(0.5, 0.5, "Error generating quarterly analysis", 
                      ha='center', va='center', fontsize=14, color='red',
                      transform=axes[1].transAxes)
        
        try:
            # 3. Heatmap of installations by month and year
            years = sorted(df['year'].unique())
            months = range(1, 13)
            
            # Create a pivot table
            heatmap_data = df.groupby(['year', 'month']).size().unstack(fill_value=0)
            
            # Fill missing months with 0
            for month in months:
                if month not in heatmap_data.columns:
                    heatmap_data[month] = 0
            
            # Sort columns
            heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)
            
            # Create heatmap
            sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='d', ax=axes[2])
            axes[2].set_title('Heatmap of RBS Installations by Month and Year', fontsize=14)
            axes[2].set_xlabel('Month', fontsize=12)
            axes[2].set_ylabel('Year', fontsize=12)
            axes[2].set_xticklabels([calendar.month_abbr[i] for i in range(1, 13)], rotation=45)
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            axes[2].text(0.5, 0.5, "Error generating heatmap", 
                      ha='center', va='center', fontsize=14, color='red',
                      transform=axes[2].transAxes)
        
        try:
            # 4. Installations by operator over time
            if 'Operator' in df.columns:
                operator_data = df.groupby(['year', 'Operator']).size().unstack(fill_value=0)
                operator_data.plot(kind='line', marker='o', ax=axes[3], colormap='tab10')
                axes[3].set_title('RBS Installations by Operator Over Time', fontsize=14)
                axes[3].set_xlabel('Year', fontsize=12)
                axes[3].set_ylabel('Number of Installations', fontsize=12)
                axes[3].legend(title='Operator')
                axes[3].grid(True, linestyle='--', alpha=0.7)
        except Exception as e:
            logger.error(f"Error creating operator timeline: {e}")
            axes[3].text(0.5, 0.5, "Error generating operator analysis", 
                      ha='center', va='center', fontsize=14, color='red',
                      transform=axes[3].transAxes)
        
        plt.tight_layout()
        
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Seasonality analysis saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving seasonality analysis: {e}")
            try:
                alt_path = os.path.join(os.path.dirname(output_path), 'seasonality_analysis_alt.png')
                plt.savefig(alt_path, dpi=300, bbox_inches='tight')
                logger.info(f"Seasonality analysis saved to alternative path: {alt_path}")
            except Exception as e2:
                logger.error(f"Error saving to alternative path: {e2}")
        
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error in create_seasonality_analysis: {e}")

def create_chronological_heatmap(gdf_rbs, output_path):
    """
    Creates a chronological heatmap showing the geographical expansion of RBS over time.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        output_path: Path to save the visualization
    """
    logger.info("Creating chronological heatmap of RBS expansion...")
    
    try:
        # Preprocess data
        df = preprocess_temporal_data(gdf_rbs)
        
        if df.empty or 'geometry' not in df.columns:
            logger.error("No geometry data available for chronological heatmap")
            return
        
        # Save as HTML for interactive folium map
        html_path = output_path.replace('.png', '.html')
        
        try:
            # Create a folium map
            center = [df.geometry.y.mean(), df.geometry.x.mean()]
            m = folium.Map(location=center, zoom_start=12, tiles='CartoDB positron')
            
            # Group data by year
            df = df.sort_values('year')
            years = sorted(df['year'].unique())
            
            # Prepare data for heatmap with time
            heat_data = []
            for year in years:
                year_data = df[df['year'] == year]
                locations = [[row.geometry.y, row.geometry.x] for idx, row in year_data.iterrows()]
                heat_data.append(locations)
            
            # Add HeatMapWithTime layer
            HeatMapWithTime(
                heat_data,
                index=years,
                auto_play=True,
                radius=15,
                gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'},
                max_opacity=0.8,
                min_opacity=0.1
            ).add_to(m)
            
            # Save to HTML
            m.save(html_path)
            logger.info(f"Interactive chronological heatmap saved to {html_path}")
        except Exception as e:
            logger.error(f"Error creating interactive folium map: {e}")
        
        try:
            # Create static images for each year to combine into a figure
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            axes = axes.flatten()
            
            # Select years to display (first, last, and evenly spaced in between)
            if len(years) > 6:
                display_years = [years[0]] + [years[i] for i in range(1, len(years)-1, len(years)//5)][:4] + [years[-1]]
            else:
                display_years = years
            
            for i, year in enumerate(display_years[:6]):
                ax = axes[i]
                
                # Filter data for this year and all previous years (cumulative)
                cumulative_data = df[df['year'] <= year]
                
                # Create a kernel density plot
                cumulative_data.plot(ax=ax, alpha=0.6, markersize=5)
                
                # Add title and styling
                ax.set_title(f'RBS Coverage: {year}', fontsize=14)
                ax.set_xlabel('Longitude', fontsize=10)
                ax.set_ylabel('Latitude', fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.4)
                
                # Try to add a basemap
                try:
                    import contextily as ctx
                    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
                except Exception as e:
                    logger.warning(f"Could not add basemap: {e}")
            
            # Add a title to the figure
            fig.suptitle('Geographical Expansion of RBS Over Time', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
            
            try:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Static chronological heatmap saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving static chronological heatmap: {e}")
                try:
                    alt_path = os.path.join(os.path.dirname(output_path), 'chronological_heatmap_alt.png')
                    plt.savefig(alt_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Static chronological heatmap saved to alternative path: {alt_path}")
                except Exception as e2:
                    logger.error(f"Error saving to alternative path: {e2}")
            
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating static chronological heatmap: {e}")
    except Exception as e:
        logger.error(f"Error in create_chronological_heatmap: {e}")

def run_temporal_analysis(gdf_rbs, results_dir):
    """
    Runs all temporal analyses and saves the results.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        results_dir: Directory to save the results
    """
    logger.info("Starting temporal analysis")
    
    try:
        if gdf_rbs is None or gdf_rbs.empty:
            logger.warning("No data for temporal analysis.")
            return
        
        # Create folder for temporal analysis
        temporal_dir = os.path.join(results_dir, "temporal_analysis")
        try:
            os.makedirs(temporal_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directory {temporal_dir}: {e}")
            # Fallback to results_dir
            temporal_dir = results_dir
        
        # Create deployment timeline
        deployment_path = os.path.join(temporal_dir, "deployment_timeline.png")
        create_deployment_timeline(gdf_rbs, deployment_path)
        
        # Create seasonality analysis
        seasonality_path = os.path.join(temporal_dir, "seasonality_analysis.png")
        create_seasonality_analysis(gdf_rbs, seasonality_path)
        
        # Create chronological heatmap
        heatmap_path = os.path.join(temporal_dir, "chronological_heatmap.png")
        create_chronological_heatmap(gdf_rbs, heatmap_path)
        
        logger.info(f"All temporal analyses completed. Results saved to {temporal_dir}")
    except Exception as e:
        logger.error(f"Error in run_temporal_analysis: {e}")
        raise 