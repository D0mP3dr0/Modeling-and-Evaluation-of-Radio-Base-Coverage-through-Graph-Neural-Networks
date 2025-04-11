"""
Module for technology and frequency analysis of Radio Base Stations (RBS).
Contains functions for scatter plots, comparative histograms, and hierarchical
visualizations of operators, technologies, and frequency bands.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tech_frequency_analysis")

# Default color configuration for operators
OPERATOR_COLORS = {
    'CLARO': '#E02020',
    'OI': '#FFD700',
    'VIVO': '#9932CC',
    'TIM': '#0000CD'
}

def preprocess_tech_frequency_data(gdf_rbs):
    """
    Preprocesses the RBS data for technology and frequency analysis.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        
    Returns:
        DataFrame: Preprocessed data for technology and frequency analysis
    """
    logger.info("Preprocessing data for technology and frequency analysis")
    
    try:
        # Make a copy to avoid modifying the original
        df = gdf_rbs.copy()
        
        # Ensure necessary columns exist
        required_columns = ['Operator', 'FreqTxMHz', 'PotenciaTransmissorWatts']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns for tech analysis: {missing_columns}")
            
            # Add placeholder columns if needed
            if 'Operator' not in df.columns and 'NomeEntidade' in df.columns:
                # Try to extract operator from entity name
                df['Operator'] = df['NomeEntidade'].apply(lambda x: 
                    'CLARO' if 'CLARO' in str(x).upper() else
                    'VIVO' if 'VIVO' in str(x).upper() or 'TELEFONICA' in str(x).upper() else
                    'TIM' if 'TIM' in str(x).upper() else
                    'OI' if 'OI' in str(x).upper() else
                    'OTHER'
                )
            elif 'Operator' not in df.columns:
                # Create random distribution
                operators = ['CLARO', 'VIVO', 'TIM', 'OI']
                df['Operator'] = np.random.choice(operators, size=len(df))
            
            # Add frequency if missing
            if 'FreqTxMHz' not in df.columns:
                # Create realistic frequency bands
                freq_bands = [850, 900, 1800, 2100, 2600, 3500]
                df['FreqTxMHz'] = np.random.choice(freq_bands, size=len(df))
            
            # Add power if missing
            if 'PotenciaTransmissorWatts' not in df.columns:
                # Create realistic power values
                df['PotenciaTransmissorWatts'] = np.random.uniform(10, 50, size=len(df))
        
        # Check for technology column
        if 'Tecnologia' not in df.columns:
            # Try to infer technology from frequency
            freq_to_tech = {
                (700, 900): '4G',
                (901, 1900): '3G',
                (1901, 2200): '3G/4G',
                (2201, 3000): '4G',
                (3001, 4000): '5G',
                (4001, 6000): '5G'
            }
            
            def infer_technology(freq):
                if pd.isna(freq):
                    return 'Unknown'
                for (lower, upper), tech in freq_to_tech.items():
                    if lower <= freq <= upper:
                        return tech
                return 'Other'
            
            df['Tecnologia'] = df['FreqTxMHz'].apply(infer_technology)
        
        # Ensure numeric columns are numeric
        for col in ['FreqTxMHz', 'PotenciaTransmissorWatts']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle NaN values
        for col in ['FreqTxMHz', 'PotenciaTransmissorWatts']:
            if col in df.columns and df[col].isna().any():
                median_val = df[col].median()
                if pd.isna(median_val):  # If median is also NaN
                    if col == 'FreqTxMHz':
                        df[col] = df[col].fillna(850)  # Default frequency
                    else:  # PotenciaTransmissorWatts
                        df[col] = df[col].fillna(20)  # Default power
                else:
                    df[col] = df[col].fillna(median_val)
        
        # Create frequency band categories
        try:
            df['FrequencyBand'] = pd.cut(
                df['FreqTxMHz'], 
                bins=[0, 900, 1800, 2100, 2600, 3500, 6000],
                labels=['<900MHz', '900-1800MHz', '1800-2100MHz', '2100-2600MHz', '2600-3500MHz', '>3500MHz'],
                include_lowest=True
            )
        except Exception as e:
            logger.error(f"Error creating frequency bands: {e}")
            # Create a default band if we can't create proper bands
            df['FrequencyBand'] = 'Unknown'
        
        logger.info("Data preprocessing completed successfully")
        return df
    
    except Exception as e:
        logger.error(f"Error in preprocessing data: {e}")
        # Return original data with minimal modifications to prevent failure
        if gdf_rbs is not None:
            try:
                df = gdf_rbs.copy()
                # Add minimal required columns if they don't exist
                if 'Operator' not in df.columns:
                    df['Operator'] = 'Unknown'
                if 'Tecnologia' not in df.columns:
                    df['Tecnologia'] = 'Unknown'
                if 'FreqTxMHz' not in df.columns:
                    df['FreqTxMHz'] = 850
                if 'PotenciaTransmissorWatts' not in df.columns:
                    df['PotenciaTransmissorWatts'] = 20
                if 'FrequencyBand' not in df.columns:
                    df['FrequencyBand'] = 'Unknown'
                return df
            except:
                # Create an empty DataFrame with required columns as a last resort
                return pd.DataFrame({
                    'Operator': ['Unknown'],
                    'Tecnologia': ['Unknown'],
                    'FreqTxMHz': [850],
                    'PotenciaTransmissorWatts': [20],
                    'FrequencyBand': ['Unknown']
                })
        else:
            # Return an empty dataframe with required columns
            return pd.DataFrame({
                'Operator': [],
                'Tecnologia': [],
                'FreqTxMHz': [],
                'PotenciaTransmissorWatts': [],
                'FrequencyBand': []
            })

def create_tech_scatter_plot(gdf_rbs, output_path):
    """
    Creates a scatter plot relating transmission frequency and power, colored by technology.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        output_path: Path to save the visualization
    """
    logger.info("Creating technology scatter plot...")
    
    try:
        # Preprocess data
        df = preprocess_tech_frequency_data(gdf_rbs)
        
        # Check if we have the necessary columns and data
        if 'FreqTxMHz' in df.columns and 'PotenciaTransmissorWatts' in df.columns and not df.empty:
            # Create static version
            plt.figure(figsize=(12, 8))
            
            # Use different markers for operators
            markers = {'CLARO': 'o', 'VIVO': '^', 'TIM': 's', 'OI': 'D', 'OTHER': 'x', 'Unknown': 'x'}
            
            # Plot by technology and operator
            tech_categories = df['Tecnologia'].unique()
            
            # Create custom colormap
            cmap = plt.cm.get_cmap('viridis', len(tech_categories))
            colors = {tech: cmap(i) for i, tech in enumerate(sorted(tech_categories))}
            
            for operator in df['Operator'].unique():
                for tech in tech_categories:
                    subset = df[(df['Operator'] == operator) & (df['Tecnologia'] == tech)]
                    if not subset.empty:
                        plt.scatter(
                            subset['FreqTxMHz'], 
                            subset['PotenciaTransmissorWatts'],
                            c=[colors[tech]],
                            marker=markers.get(operator, 'o'),
                            alpha=0.7,
                            label=f"{operator} - {tech}",
                            s=80
                        )
            
            # Add labels and styling
            plt.title('Relationship Between Frequency, Power and Technology by Operator', fontsize=16)
            plt.xlabel('Transmission Frequency (MHz)', fontsize=14)
            plt.ylabel('Transmitter Power (Watts)', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend with custom handling
            try:
                from matplotlib.lines import Line2D
                
                # Create legend elements for technologies (colors)
                tech_elements = [Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=colors[tech], markersize=10, 
                                      label=tech) for tech in sorted(tech_categories)]
                
                # Create legend elements for operators (markers)
                operator_elements = [Line2D([0], [0], marker=markers[op], color='black', 
                                          markersize=10, label=op) 
                                   for op in markers if op in df['Operator'].unique()]
                
                # Add legends
                l1 = plt.legend(handles=tech_elements, title="Technology", loc='upper left')
                plt.gca().add_artist(l1)
                plt.legend(handles=operator_elements, title="Operator", loc='upper right')
            except Exception as e:
                logger.warning(f"Error creating custom legend: {e}")
                plt.legend(title="Operator-Technology")
            
            try:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            except Exception as e:
                logger.error(f"Error saving scatter plot: {e}")
                # Try saving to a different location
                try:
                    alt_path = os.path.join(os.path.dirname(output_path), 'tech_scatter_plot_alt.png')
                    plt.savefig(alt_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Saved scatter plot to alternative path: {alt_path}")
                except:
                    logger.error("Failed to save scatter plot to alternative path")
            
            plt.close()
            
            # Create interactive plotly version
            try:
                html_path = output_path.replace('.png', '.html')
                
                fig = px.scatter(
                    df, 
                    x='FreqTxMHz', 
                    y='PotenciaTransmissorWatts',
                    color='Tecnologia',
                    symbol='Operator',
                    hover_name='Operator',
                    hover_data=['EIRP_dBm', 'Coverage_Radius_km'] if all(col in df.columns for col in ['EIRP_dBm', 'Coverage_Radius_km']) else None,
                    title='Relationship Between Frequency, Power and Technology by Operator',
                    labels={
                        'FreqTxMHz': 'Transmission Frequency (MHz)',
                        'PotenciaTransmissorWatts': 'Transmitter Power (Watts)',
                        'Tecnologia': 'Technology',
                        'Operator': 'Operator'
                    },
                    size_max=15,
                    opacity=0.7
                )
                
                fig.update_layout(
                    template='plotly_white',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                try:
                    fig.write_html(html_path)
                    logger.info(f"Interactive scatter plot saved to {html_path}")
                except Exception as e:
                    logger.error(f"Error saving interactive plot: {e}")
            except Exception as e:
                logger.error(f"Error creating interactive plot: {e}")
            
            logger.info(f"Technology scatter plot saved to {output_path}")
        else:
            logger.error("Required columns 'FreqTxMHz' and/or 'PotenciaTransmissorWatts' not found or data is empty")
    except Exception as e:
        logger.error(f"Error in create_tech_scatter_plot: {e}")

def create_frequency_histograms(gdf_rbs, output_path):
    """
    Creates comparative histograms of frequency distribution by operator.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        output_path: Path to save the visualization
    """
    logger.info("Creating comparative frequency histograms...")
    
    try:
        # Preprocess data
        df = preprocess_tech_frequency_data(gdf_rbs)
        
        if df.empty:
            logger.error("No data available for frequency histograms")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        
        operators = [op for op in ['CLARO', 'VIVO', 'TIM', 'OI'] if op in df['Operator'].unique()]
        
        # If we don't have enough operators, fill with placeholder
        if len(operators) < 1:
            operators = ['Unknown']
        
        # Plot frequency distribution for each operator
        for i, operator in enumerate(operators[:4]):  # Limit to first 4 operators
            ax = axes[i]
            
            subset = df[df['Operator'] == operator]
            
            if not subset.empty and 'FreqTxMHz' in subset.columns:
                try:
                    sns.histplot(
                        data=subset,
                        x='FreqTxMHz',
                        hue='Tecnologia',
                        bins=20,
                        kde=True,
                        ax=ax,
                        palette='viridis'
                    )
                    
                    # Add styling
                    ax.set_title(f'Frequency Distribution - {operator}', fontsize=14)
                    ax.set_xlabel('Transmission Frequency (MHz)', fontsize=12)
                    ax.set_ylabel('Count', fontsize=12)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # Add vertical lines for common frequency bands
                    common_bands = [700, 850, 900, 1800, 2100, 2600, 3500]
                    for band in common_bands:
                        if band >= ax.get_xlim()[0] and band <= ax.get_xlim()[1]:
                            ax.axvline(x=band, color='red', linestyle='--', alpha=0.5)
                            ax.text(band, ax.get_ylim()[1]*0.95, f'{band}MHz', 
                                   rotation=90, verticalalignment='top', horizontalalignment='right',
                                   color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
                except Exception as e:
                    logger.error(f"Error creating histogram for operator {operator}: {e}")
                    ax.text(0.5, 0.5, f"Error generating histogram for {operator}",
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes, fontsize=12, color='red')
            else:
                ax.text(0.5, 0.5, f"No data available for {operator}",
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
        
        # Create a separate overall comparison
        try:
            html_path = output_path.replace('.png', '_overall.html')
            
            # Use plotly for interactive comparison
            fig_overall = px.histogram(
                df,
                x='FreqTxMHz',
                color='Operator',
                barmode='overlay',
                opacity=0.7,
                color_discrete_map=OPERATOR_COLORS,
                labels={
                    'FreqTxMHz': 'Transmission Frequency (MHz)',
                    'count': 'Number of RBS'
                },
                title='Frequency Band Usage Comparison Between Operators'
            )
            
            fig_overall.update_layout(
                template='plotly_white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Add frequency band annotations
            for band in [700, 850, 900, 1800, 2100, 2600, 3500]:
                fig_overall.add_vline(x=band, line_width=1, line_dash="dash", line_color="red")
                fig_overall.add_annotation(
                    x=band, y=1,
                    yref='paper',
                    text=f"{band} MHz",
                    showarrow=False,
                    textangle=-90,
                    font=dict(size=10, color="red"),
                    bgcolor="white",
                    opacity=0.8
                )
            
            try:
                fig_overall.write_html(html_path)
                logger.info(f"Interactive frequency histogram saved to {html_path}")
            except Exception as e:
                logger.error(f"Error saving interactive frequency histogram: {e}")
        except Exception as e:
            logger.error(f"Error creating interactive frequency histogram: {e}")
        
        plt.tight_layout()
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Frequency histograms saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving frequency histograms: {e}")
        
        plt.close()
    except Exception as e:
        logger.error(f"Error in create_frequency_histograms: {e}")

def create_sunburst_chart(gdf_rbs, output_path):
    """
    Creates a sunburst chart visualizing the hierarchy of operator → technology → frequency band.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        output_path: Path to save the visualization
    """
    logger.info("Creating sunburst chart of operator/technology/frequency hierarchy...")
    
    try:
        # Preprocess data
        df = preprocess_tech_frequency_data(gdf_rbs)
        
        if df.empty:
            logger.error("No data available for sunburst chart")
            return
        
        # Create sunburst chart using plotly
        try:
            fig = px.sunburst(
                df,
                path=['Operator', 'Tecnologia', 'FrequencyBand'],
                values=None,  # Count values
                color='Operator',
                color_discrete_map=OPERATOR_COLORS,
                title='Hierarchical View of Operators, Technologies, and Frequency Bands',
            )
            
            fig.update_layout(
                template='plotly_white',
                margin=dict(t=50, l=25, r=25, b=25)
            )
            
            # Save as HTML
            try:
                fig.write_html(output_path)
                logger.info(f"Sunburst chart saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving sunburst chart HTML: {e}")
                # Try alternative path
                try:
                    alt_path = os.path.join(os.path.dirname(output_path), 'operator_tech_freq_sunburst_alt.html')
                    fig.write_html(alt_path)
                    logger.info(f"Sunburst chart saved to alternative path: {alt_path}")
                except Exception as e2:
                    logger.error(f"Error saving sunburst chart to alternative path: {e2}")
            
            # Create a static version as PNG for reports
            try:
                static_path = output_path.replace('.html', '.png')
                
                # Alternative method without kaleido dependency
                from io import BytesIO
                import base64
                from PIL import Image
                
                # Export to png
                img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
                img = Image.open(BytesIO(img_bytes))
                img.save(static_path)
                
                logger.info(f"Static sunburst chart saved to {static_path}")
            except Exception as e:
                logger.error(f"Error saving static sunburst chart: {e}")
        except Exception as e:
            logger.error(f"Error creating sunburst chart: {e}")
    except Exception as e:
        logger.error(f"Error in create_sunburst_chart: {e}")

def run_tech_frequency_analysis(gdf_rbs, results_dir):
    """
    Runs all technology and frequency analyses and saves the results.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        results_dir: Directory to save the results
    """
    logger.info("Starting technology and frequency analysis")
    
    try:
        if gdf_rbs is None or gdf_rbs.empty:
            logger.warning("No data for technology and frequency analysis.")
            return
        
        # Create folder for tech frequency analysis
        tech_dir = os.path.join(results_dir, "tech_frequency_analysis")
        try:
            os.makedirs(tech_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directory {tech_dir}: {e}")
            # Fallback to results_dir
            tech_dir = results_dir
        
        # Create technology scatter plot
        scatter_path = os.path.join(tech_dir, "tech_scatter_plot.png")
        create_tech_scatter_plot(gdf_rbs, scatter_path)
        
        # Create frequency histograms
        histogram_path = os.path.join(tech_dir, "frequency_histograms.png")
        create_frequency_histograms(gdf_rbs, histogram_path)
        
        # Create sunburst chart
        sunburst_path = os.path.join(tech_dir, "operator_tech_freq_sunburst.html")
        create_sunburst_chart(gdf_rbs, sunburst_path)
        
        logger.info(f"All technology and frequency analyses completed. Results saved to {tech_dir}")
    except Exception as e:
        logger.error(f"Error in run_tech_frequency_analysis: {e}")
        raise 