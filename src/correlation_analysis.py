"""
Module for correlation analysis of Radio Base Stations (RBS) data.
Contains functions to analyze relationships between technical variables such as
frequency, power, antenna height, and coverage radius.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from scipy.stats import pearsonr, spearmanr

# Import from other modules if needed
from tech_frequency_analysis import preprocess_tech_frequency_data

def preprocess_correlation_data(gdf_rbs):
    """
    Preprocesses the RBS data for correlation analysis.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        
    Returns:
        DataFrame: Preprocessed data for correlation analysis
    """
    # Start with tech preprocessing
    df = preprocess_tech_frequency_data(gdf_rbs)
    
    # Define key technical columns for correlation
    tech_columns = [
        'FreqTxMHz', 
        'PotenciaTransmissorWatts',
        'AlturaAntena'  # Antenna height, if available
    ]
    
    # Check for 'AlturaAntena' column, create if missing
    if 'AlturaAntena' not in df.columns:
        print("Warning: 'AlturaAntena' column not found. Creating synthetic data for demonstration.")
        # Create realistic antenna heights (typically between 15-120 meters)
        df['AlturaAntena'] = np.random.uniform(15, 120, size=len(df))
    
    # Add coverage radius if not present
    if 'Coverage_Radius_km' not in df.columns:
        print("Adding estimated coverage radius based on frequency and power...")
        
        # Simple formula for coverage radius estimation (simplified model)
        # R = k * sqrt(P) / f
        # where R is radius, P is power, f is frequency, k is a constant
        
        def estimate_coverage(row):
            # This is a very simplified model
            k = 50  # Constant factor
            power = row['PotenciaTransmissorWatts']
            freq = row['FreqTxMHz']
            height = row['AlturaAntena']
            
            if pd.isna(power) or pd.isna(freq) or pd.isna(height) or freq == 0:
                return np.nan
            
            # Basic Hata model inspired formula 
            radius = k * np.sqrt(power) * np.sqrt(height) / np.sqrt(freq)
            return np.clip(radius, 0.5, 30)  # Reasonable range in km
            
        df['Coverage_Radius_km'] = df.apply(estimate_coverage, axis=1)
    
    # Ensure all technical columns are numeric
    for col in tech_columns + ['Coverage_Radius_km']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def create_correlation_matrix(gdf_rbs, output_path):
    """
    Creates a correlation matrix visualization between key technical variables.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        output_path: Path to save the visualization
    """
    print("Creating correlation matrix...")
    
    # Preprocess data
    df = preprocess_correlation_data(gdf_rbs)
    
    # Define variables to correlate
    corr_vars = [
        'FreqTxMHz', 
        'PotenciaTransmissorWatts', 
        'AlturaAntena',
        'Coverage_Radius_km'
    ]
    
    # Check if we have all required columns
    missing_cols = [col for col in corr_vars if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for correlation analysis: {missing_cols}")
        corr_vars = [col for col in corr_vars if col in df.columns]
    
    if len(corr_vars) < 2:
        print("Error: Not enough variables for correlation analysis.")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[corr_vars].corr(method='pearson')
    
    # Create static visualization
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        fmt='.2f',
        cmap=cmap,
        vmin=-1, 
        vmax=1, 
        center=0,
        square=True, 
        linewidths=.5,
        cbar_kws={'shrink': .7}
    )
    
    plt.title('Correlation Matrix of Technical Variables', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create interactive plotly version
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title='Correlation Matrix of Technical Variables'
    )
    
    fig.update_layout(
        template='plotly_white',
        width=700,
        height=700
    )
    
    fig.write_html(output_path.replace('.png', '.html'))
    
    print(f"Correlation matrix saved to {output_path}")
    
    # Also return p-values for correlations
    p_values = pd.DataFrame(np.zeros_like(corr_matrix), 
                          index=corr_matrix.index, 
                          columns=corr_matrix.columns)
    
    for i, row in enumerate(corr_matrix.index):
        for j, col in enumerate(corr_matrix.columns):
            if i != j:  # Skip diagonal
                valid_data = df[[row, col]].dropna()
                if len(valid_data) > 1:
                    _, p_val = pearsonr(valid_data[row], valid_data[col])
                    p_values.loc[row, col] = p_val
    
    # Save p-values
    p_values_path = output_path.replace('.png', '_pvalues.csv')
    p_values.to_csv(p_values_path)
    print(f"Correlation p-values saved to {p_values_path}")

def create_pairplot(gdf_rbs, output_path):
    """
    Creates a pair plot showing relationships between all pairs of technical variables.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        output_path: Path to save the visualization
    """
    print("Creating technical variables pairplot...")
    
    # Preprocess data
    df = preprocess_correlation_data(gdf_rbs)
    
    # Define variables to include in pairplot
    pair_vars = [
        'FreqTxMHz', 
        'PotenciaTransmissorWatts', 
        'AlturaAntena',
        'Coverage_Radius_km'
    ]
    
    # Check if we have all required columns
    missing_cols = [col for col in pair_vars if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for pairplot: {missing_cols}")
        pair_vars = [col for col in pair_vars if col in df.columns]
    
    if len(pair_vars) < 2:
        print("Error: Not enough variables for pairplot.")
        return
    
    # Create pairplot
    if 'Operator' in df.columns:
        # Create with operator color coding
        g = sns.pairplot(
            df[pair_vars + ['Operator']],
            hue='Operator',
            palette=dict(CLARO='#E02020', OI='#FFD700', VIVO='#9932CC', TIM='#0000CD'),
            diag_kind='kde',
            plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'none'},
            height=2.5
        )
    else:
        # Create without operator distinction
        g = sns.pairplot(
            df[pair_vars],
            diag_kind='kde',
            plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'none'},
            height=2.5
        )
    
    g.fig.suptitle('Relationships Between Technical Variables', y=1.02, fontsize=16)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create interactive plotly version for selected pairs
    if len(pair_vars) >= 2:
        key_pairs = [
            ('FreqTxMHz', 'Coverage_Radius_km'),
            ('PotenciaTransmissorWatts', 'Coverage_Radius_km'),
            ('AlturaAntena', 'Coverage_Radius_km')
        ]
        
        # Filter to pairs that exist in our data
        key_pairs = [(x, y) for x, y in key_pairs 
                     if x in pair_vars and y in pair_vars]
        
        # Create a subplot for each pair
        if key_pairs:
            fig = make_subplots(
                rows=1, 
                cols=len(key_pairs),
                subplot_titles=[f"{x} vs {y}" for x, y in key_pairs]
            )
            
            for i, (x, y) in enumerate(key_pairs):
                if 'Operator' in df.columns and 'Tecnologia' in df.columns:
                    # Color by operator, symbol by technology
                    for op in df['Operator'].unique():
                        op_data = df[df['Operator'] == op]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=op_data[x],
                                y=op_data[y],
                                mode='markers',
                                marker=dict(
                                    size=8,
                                    opacity=0.7,
                                ),
                                name=op,
                                legendgroup=op,
                                showlegend=i == 0,  # Show legend only for first subplot
                                hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<br>Operator: {op}"
                            ),
                            row=1, 
                            col=i+1
                        )
                else:
                    # Simple scatter without categories
                    fig.add_trace(
                        go.Scatter(
                            x=df[x],
                            y=df[y],
                            mode='markers',
                            marker=dict(
                                size=8,
                                opacity=0.7,
                                color='blue'
                            ),
                            showlegend=False,
                            hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}"
                        ),
                        row=1, 
                        col=i+1
                    )
                
                # Add trend line
                valid_data = df[[x, y]].dropna()
                if len(valid_data) > 1:
                    # Linear regression
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        valid_data[x], valid_data[y]
                    )
                    
                    x_range = np.linspace(valid_data[x].min(), valid_data[x].max(), 100)
                    y_pred = slope * x_range + intercept
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=y_pred,
                            mode='lines',
                            line=dict(color='red', width=2, dash='dash'),
                            name=f'Trend (r={r_value:.2f})',
                            showlegend=i == 0,
                            hovertemplate=f"r²={r_value**2:.3f}<br>p={p_value:.3e}"
                        ),
                        row=1, 
                        col=i+1
                    )
            
            fig.update_layout(
                title='Key Technical Relationships',
                template='plotly_white',
                height=500,
                width=300 * len(key_pairs),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            html_path = output_path.replace('.png', '_interactive.html')
            fig.write_html(html_path)
            print(f"Interactive relationship plots saved to {html_path}")
    
    print(f"Technical variables pairplot saved to {output_path}")

def create_regression_analysis(gdf_rbs, output_path):
    """
    Performs regression analysis to quantify relationships between variables.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        output_path: Path to save the results
    """
    print("Performing regression analysis...")
    
    # Preprocess data
    df = preprocess_correlation_data(gdf_rbs)
    
    # Define target and predictor variables
    target = 'Coverage_Radius_km'
    predictors = ['FreqTxMHz', 'PotenciaTransmissorWatts', 'AlturaAntena']
    
    # Check if we have all required columns
    missing_cols = [col for col in predictors + [target] if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for regression analysis: {missing_cols}")
        predictors = [col for col in predictors if col in df.columns]
    
    if not predictors or target not in df.columns:
        print("Error: Not enough variables for regression analysis.")
        return
    
    # Drop rows with missing values
    regression_data = df[predictors + [target]].dropna()
    
    if len(regression_data) < 10:
        print("Error: Not enough data for regression analysis.")
        return
    
    # Perform regression analysis
    results = []
    
    for predictor in predictors:
        # Calculate Pearson and Spearman correlations
        pearson_r, pearson_p = pearsonr(regression_data[predictor], regression_data[target])
        spearman_r, spearman_p = spearmanr(regression_data[predictor], regression_data[target])
        
        # Linear regression
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            regression_data[predictor], regression_data[target]
        )
        
        results.append({
            'Predictor': predictor,
            'Target': target,
            'Pearson_r': pearson_r,
            'Pearson_p': pearson_p,
            'Spearman_r': spearman_r,
            'Spearman_p': spearman_p,
            'Regression_slope': slope,
            'Regression_intercept': intercept,
            'R_squared': r_value**2,
            'Regression_p': p_value
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv(output_path, index=False)
    
    # Create visualization summary
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot R-squared values
    bar_width = 0.35
    x = np.arange(len(predictors))
    
    rects1 = ax.bar(x - bar_width/2, results_df['Pearson_r']**2, bar_width, 
                   label='Pearson R²', alpha=0.7, color='blue')
    rects2 = ax.bar(x + bar_width/2, results_df['R_squared'], bar_width, 
                   label='Linear Regression R²', alpha=0.7, color='red')
    
    # Add labels and formatting
    ax.set_xlabel('Predictor Variable', fontsize=14)
    ax.set_ylabel('R² Value', fontsize=14)
    ax.set_title(f'Predictive Power for {target}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(predictors, rotation=45)
    ax.legend()
    
    # Add significance markers
    for i, p in enumerate(results_df['Regression_p']):
        significance = ''
        if p < 0.001:
            significance = '***'
        elif p < 0.01:
            significance = '**'
        elif p < 0.05:
            significance = '*'
            
        if significance:
            ax.text(i + bar_width/2, results_df.iloc[i]['R_squared'] + 0.02, 
                   significance, ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path.replace('.csv', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Regression analysis results saved to {output_path}")
    print(f"Regression visualization saved to {output_path.replace('.csv', '.png')}")

def run_correlation_analysis(gdf_rbs, results_dir):
    """
    Runs all correlation analysis visualizations.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        results_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    correlation_dir = os.path.join(results_dir, 'correlation_analysis')
    os.makedirs(correlation_dir, exist_ok=True)
    
    # Run all visualizations
    create_correlation_matrix(gdf_rbs, os.path.join(correlation_dir, 'correlation_matrix.png'))
    create_pairplot(gdf_rbs, os.path.join(correlation_dir, 'technical_pairplot.png'))
    create_regression_analysis(gdf_rbs, os.path.join(correlation_dir, 'regression_results.csv'))
    
    print(f"All correlation analyses completed and saved to {correlation_dir}") 