"""
Module for integrated analysis between temporal and technological aspects of Radio Base Stations (RBS).
Combines data from temporal and technology analyses to provide deeper insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import calendar

# Import from other modules
from src.advanced_temporal_analysis import preprocess_temporal_data
from src.tech_frequency_analysis import preprocess_tech_frequency_data

# Default color configuration for operators
OPERATOR_COLORS = {
    'CLARO': '#E02020',
    'OI': '#FFD700',
    'VIVO': '#9932CC',
    'TIM': '#0000CD'
}

def preprocess_integrated_data(gdf_rbs):
    """
    Combines preprocessing from both temporal and technology analyses.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        
    Returns:
        DataFrame: Preprocessed data for integrated analysis
    """
    # Apply both preprocessing functions
    df_temporal = preprocess_temporal_data(gdf_rbs)
    df_tech = preprocess_tech_frequency_data(gdf_rbs)
    
    # Merge the results if needed
    if id(df_temporal) != id(df_tech):
        # They are different objects, need to merge
        # Using the index as the common key
        df = df_temporal.merge(df_tech, left_index=True, right_index=True, suffixes=('', '_tech'))
        
        # Clean up duplicate columns
        duplicate_cols = [col for col in df.columns if col.endswith('_tech')]
        df.drop(columns=duplicate_cols, inplace=True)
    else:
        # They are the same object
        df = df_temporal
    
    return df

def create_technology_evolution_chart(gdf_rbs, output_path):
    """
    Creates a stacked area chart showing the evolution of technologies over time.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        output_path: Path to save the visualization
    """
    print("Creating technology evolution chart...")
    
    # Preprocess data
    df = preprocess_integrated_data(gdf_rbs)
    
    # Check if we have the necessary columns
    if 'year' in df.columns and 'Tecnologia' in df.columns:
        # Group by year and technology, count installations
        tech_evolution = df.groupby(['year', 'Tecnologia']).size().unstack(fill_value=0)
        
        # Calculate cumulative sum for each technology
        cumulative_tech = tech_evolution.cumsum()
        
        # Create the plot with plotly for interactivity
        fig = go.Figure()
        
        # Create a color scale for technologies
        tech_categories = sorted(df['Tecnologia'].unique())
        cmap = plt.cm.get_cmap('viridis', len(tech_categories))
        colors = {tech: f'rgba({int(255*cmap(i)[0])}, {int(255*cmap(i)[1])}, {int(255*cmap(i)[2])}, 0.8)' 
                 for i, tech in enumerate(tech_categories)}
        
        # Add traces for each technology
        for tech in cumulative_tech.columns:
            fig.add_trace(go.Scatter(
                x=cumulative_tech.index, 
                y=cumulative_tech[tech],
                mode='lines',
                stackgroup='one',
                name=tech,
                line=dict(width=0.5, color=colors.get(tech, 'rgba(0,0,0,0.8)')),
                hovertemplate='Year: %{x}<br>Cumulative %{y}<br>Technology: ' + tech
            ))
        
        # Update layout
        fig.update_layout(
            title='Evolution of RBS Technologies Over Time',
            xaxis_title='Year',
            yaxis_title='Cumulative Number of Installations',
            legend_title='Technology',
            template='plotly_white',
            hovermode='closest'
        )
        
        # Save interactive HTML version
        fig.write_html(output_path.replace('.png', '.html'))
        
        # Create static version for presentations
        plt.figure(figsize=(15, 8))
        
        # Create stacked area chart
        cumulative_tech.plot.area(stacked=True, alpha=0.7, figsize=(15, 8), colormap='viridis')
        
        plt.title('Evolution of RBS Technologies Over Time', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Cumulative Number of Installations', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Technology', fontsize=12)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Technology evolution chart saved to {output_path}")
    else:
        print("Error: Required columns ('year', 'Tecnologia') not found in the dataset.")

def create_frequency_migration_chart(gdf_rbs, output_path):
    """
    Creates a visualization showing the migration of frequency bands over time.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        output_path: Path to save the visualization
    """
    print("Creating frequency migration chart...")
    
    # Preprocess data
    df = preprocess_integrated_data(gdf_rbs)
    
    # Check if we have the necessary columns
    if all(col in df.columns for col in ['year', 'FrequencyBand']):
        # Group by year and frequency band, count installations
        freq_evolution = df.groupby(['year', 'FrequencyBand']).size().unstack(fill_value=0)
        
        # Calculate percentage for each year to show distribution changes
        freq_percentage = freq_evolution.apply(lambda x: x / x.sum() * 100, axis=1)
        
        # Create static version
        plt.figure(figsize=(15, 8))
        
        # Create heatmap
        sns.heatmap(freq_percentage.T, annot=True, cmap='YlGnBu', fmt='.1f', cbar_kws={'label': 'Percentage (%)'})
        
        plt.title('Migration of Frequency Bands Over Time', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Frequency Band', fontsize=14)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create interactive plotly version
        fig = px.imshow(
            freq_percentage.T,
            labels=dict(x="Year", y="Frequency Band", color="Percentage (%)"),
            x=freq_percentage.index,
            y=freq_percentage.columns,
            color_continuous_scale="YlGnBu",
            title='Migration of Frequency Bands Over Time (% of new installations)'
        )
        
        fig.update_layout(
            template='plotly_white',
            xaxis={'side': 'bottom'}
        )
        
        # Add text annotations
        for y in range(len(freq_percentage.columns)):
            for x in range(len(freq_percentage.index)):
                fig.add_annotation(
                    x=freq_percentage.index[x],
                    y=freq_percentage.columns[y],
                    text=f"{freq_percentage.iloc[x, y]:.1f}%",
                    showarrow=False,
                    font=dict(color="black")
                )
        
        fig.write_html(output_path.replace('.png', '.html'))
        
        print(f"Frequency migration chart saved to {output_path}")
    else:
        print("Error: Required columns ('year', 'FrequencyBand') not found in the dataset.")

def create_operator_tech_timeline(gdf_rbs, output_path):
    """
    Creates a visualization showing which operators adopted which technologies over time.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        output_path: Path to save the visualization
    """
    print("Creating operator technology adoption timeline...")
    
    # Preprocess data
    df = preprocess_integrated_data(gdf_rbs)
    
    # Check if we have the necessary columns
    if all(col in df.columns for col in ['year', 'Tecnologia', 'Operator']):
        # For each operator and technology, find the first year of adoption
        first_adoption = df.groupby(['Operator', 'Tecnologia'])['year'].min().unstack()
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot as a bubble chart
        operators = first_adoption.index
        technologies = first_adoption.columns
        
        for i, op in enumerate(operators):
            for j, tech in enumerate(technologies):
                if not pd.isna(first_adoption.loc[op, tech]):
                    year = first_adoption.loc[op, tech]
                    ax.scatter(
                        year, 
                        i, 
                        s=200, 
                        alpha=0.7,
                        color=OPERATOR_COLORS.get(op, 'gray'),
                        edgecolor='black',
                        linewidth=1.5,
                        label=f"{op}-{tech}" if j == 0 else ""
                    )
                    ax.annotate(
                        tech, 
                        xy=(year, i), 
                        xytext=(0, 0),
                        textcoords='offset points',
                        ha='center',
                        va='center',
                        fontsize=8,
                        color='white',
                        fontweight='bold'
                    )
        
        # Styling
        ax.set_yticks(range(len(operators)))
        ax.set_yticklabels(operators)
        ax.set_xlabel('Year of First Adoption', fontsize=14)
        ax.set_title('Timeline of Technology Adoption by Operator', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add custom legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                             markersize=15, label=op) 
                  for op, color in OPERATOR_COLORS.items() if op in operators]
        ax.legend(handles=handles, title='Operator', loc='upper left')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create interactive plotly version
        # Reshape data for plotly
        adoption_data = []
        for op in operators:
            for tech in technologies:
                if not pd.isna(first_adoption.loc[op, tech]):
                    adoption_data.append({
                        'Operator': op,
                        'Technology': tech,
                        'Year': first_adoption.loc[op, tech]
                    })
        
        adoption_df = pd.DataFrame(adoption_data)
        
        fig = px.scatter(
            adoption_df,
            x='Year',
            y='Operator',
            color='Operator',
            text='Technology',
            size=[40] * len(adoption_df),
            title='Timeline of Technology Adoption by Operator',
            color_discrete_map=OPERATOR_COLORS
        )
        
        fig.update_traces(
            textposition='middle center',
            marker=dict(line=dict(width=1, color='DarkSlateGrey')),
        )
        
        fig.update_layout(
            template='plotly_white',
            xaxis_title='Year of First Adoption',
            showlegend=True
        )
        
        fig.write_html(output_path.replace('.png', '.html'))
        
        print(f"Operator technology adoption timeline saved to {output_path}")
    else:
        print("Error: Required columns ('year', 'Tecnologia', 'Operator') not found in the dataset.")

def run_integration_analysis(gdf_rbs, results_dir):
    """
    Runs all integrated analysis visualizations.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        results_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    integration_dir = os.path.join(results_dir, 'integration_analysis')
    os.makedirs(integration_dir, exist_ok=True)
    
    # Run all visualizations
    create_technology_evolution_chart(gdf_rbs, os.path.join(integration_dir, 'technology_evolution.png'))
    create_frequency_migration_chart(gdf_rbs, os.path.join(integration_dir, 'frequency_migration.png'))
    create_operator_tech_timeline(gdf_rbs, os.path.join(integration_dir, 'operator_tech_timeline.png'))
    
    print(f"All integration analyses completed and saved to {integration_dir}") 