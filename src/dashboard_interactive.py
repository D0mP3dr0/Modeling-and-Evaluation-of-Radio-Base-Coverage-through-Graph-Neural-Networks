"""
Module for creating an interactive dashboard for Radio Base Stations (RBS) analysis.
This dashboard integrates visualizations from various analysis modules.
"""

import pandas as pd
import numpy as np
import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import base64
import io
from datetime import datetime

# Import from other modules
from tech_frequency_analysis import preprocess_tech_frequency_data
from advanced_temporal_analysis import preprocess_temporal_data
from correlation_analysis import preprocess_correlation_data
from spatial_analysis import preprocess_spatial_data
from prediction_module import preprocess_prediction_data

# Define styling constants
COLORS = {
    'background': '#F8F9F9',
    'text': '#333333',
    'primary': '#3498DB',
    'secondary': '#2ECC71',
    'accent': '#9B59B6',
    'warning': '#F39C12',
    'danger': '#E74C3C'
}

OPERATOR_COLORS = {
    'CLARO': '#E02020',
    'OI': '#FFD700',
    'VIVO': '#9932CC',
    'TIM': '#0000CD'
}

def init_dashboard(gdf_rbs=None):
    """
    Initializes the Dash application for the RBS analysis dashboard.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data (optional)
        
    Returns:
        Dash application instance
    """
    # Create Dash app
    app = dash.Dash(
        __name__,
        title='RBS Analysis Dashboard',
        meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}]
    )
    
    # Define app layout
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1('Radio Base Stations (RBS) Analysis Dashboard', style={'textAlign': 'center'}),
            html.P('Interactive visualization and analysis of RBS deployment data', 
                  style={'textAlign': 'center', 'fontSize': '1.2em'})
        ], style={'backgroundColor': COLORS['primary'], 'color': 'white', 'padding': '20px'}),
        
        # Data upload section (if no data is provided)
        html.Div([
            html.H3('Upload Data', style={'textAlign': 'center', 'marginTop': '20px'}),
            html.P('Upload a CSV or GeoJSON file with RBS data:', style={'textAlign': 'center'}),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '80%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px auto'
                },
                multiple=False
            ),
            html.Div(id='upload-message', style={'textAlign': 'center', 'margin': '10px'})
        ]) if gdf_rbs is None else html.Div(),
        
        # Main content area with tabs
        html.Div([
            dcc.Tabs(id='tabs', value='tab-overview', children=[
                # Overview tab
                dcc.Tab(label='Overview', value='tab-overview', children=[
                    html.Div([
                        html.H3('RBS Dataset Overview', style={'textAlign': 'center', 'marginTop': '20px'}),
                        
                        # Summary statistics
                        html.Div(id='data-summary', style={'margin': '20px'}),
                        
                        # Distribution by operator
                        html.Div([
                            html.H4('Distribution by Operator', style={'textAlign': 'center'}),
                            dcc.Graph(id='operator-pie')
                        ], style={'width': '48%', 'display': 'inline-block'}),
                        
                        # Distribution by technology
                        html.Div([
                            html.H4('Distribution by Technology', style={'textAlign': 'center'}),
                            dcc.Graph(id='technology-pie')
                        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                    ])
                ]),
                
                # Temporal analysis tab
                dcc.Tab(label='Temporal Analysis', value='tab-temporal', children=[
                    html.Div([
                        html.H3('Temporal Deployment Analysis', style={'textAlign': 'center', 'marginTop': '20px'}),
                        
                        # Deployment timeline
                        html.Div([
                            html.H4('RBS Deployment Timeline', style={'textAlign': 'center'}),
                            dcc.Graph(id='deployment-timeline')
                        ]),
                        
                        # Seasonality analysis
                        html.Div([
                            html.H4('Seasonality Analysis', style={'textAlign': 'center'}),
                            dcc.Graph(id='seasonality-chart')
                        ])
                    ])
                ]),
                
                # Technology analysis tab
                dcc.Tab(label='Technology Analysis', value='tab-technology', children=[
                    html.Div([
                        html.H3('Technology and Frequency Analysis', style={'textAlign': 'center', 'marginTop': '20px'}),
                        
                        # Tech-frequency scatter plot
                        html.Div([
                            html.H4('Frequency vs Power by Technology', style={'textAlign': 'center'}),
                            dcc.Graph(id='tech-scatter')
                        ]),
                        
                        # Frequency distribution histograms
                        html.Div([
                            html.H4('Frequency Distribution by Operator', style={'textAlign': 'center'}),
                            dcc.Graph(id='freq-histogram')
                        ])
                    ])
                ]),
                
                # Spatial analysis tab
                dcc.Tab(label='Spatial Analysis', value='tab-spatial', children=[
                    html.Div([
                        html.H3('Spatial Distribution Analysis', style={'textAlign': 'center', 'marginTop': '20px'}),
                        
                        # Map visualization
                        html.Div([
                            html.H4('RBS Spatial Distribution', style={'textAlign': 'center'}),
                            dcc.Graph(id='spatial-map')
                        ]),
                        
                        # Cluster visualization
                        html.Div([
                            html.H4('Clustering Analysis', style={'textAlign': 'center'}),
                            dcc.Graph(id='cluster-map')
                        ])
                    ])
                ]),
                
                # Correlation analysis tab
                dcc.Tab(label='Correlation Analysis', value='tab-correlation', children=[
                    html.Div([
                        html.H3('Variable Correlation Analysis', style={'textAlign': 'center', 'marginTop': '20px'}),
                        
                        # Correlation matrix
                        html.Div([
                            html.H4('Correlation Matrix', style={'textAlign': 'center'}),
                            dcc.Graph(id='correlation-matrix')
                        ]),
                        
                        # Scatter plot matrix
                        html.Div([
                            html.H4('Relationship between Variables', style={'textAlign': 'center'}),
                            dcc.Graph(id='scatter-matrix')
                        ])
                    ])
                ]),
                
                # Prediction tab
                dcc.Tab(label='Prediction', value='tab-prediction', children=[
                    html.Div([
                        html.H3('Future Deployment Prediction', style={'textAlign': 'center', 'marginTop': '20px'}),
                        
                        # Time series forecast
                        html.Div([
                            html.H4('RBS Deployment Forecast', style={'textAlign': 'center'}),
                            dcc.Graph(id='time-forecast')
                        ]),
                        
                        # Technology adoption forecast
                        html.Div([
                            html.H4('Technology Adoption Forecast', style={'textAlign': 'center'}),
                            dcc.Graph(id='tech-forecast')
                        ])
                    ])
                ])
            ]),
        ], style={'padding': '20px', 'backgroundColor': COLORS['background']}),
        
        # Footer
        html.Div([
            html.P('Created with Dash - RBS Analysis Dashboard', style={'textAlign': 'center'})
        ], style={'backgroundColor': '#333', 'color': 'white', 'padding': '10px', 'marginTop': '20px'})
    ], style={'fontFamily': 'Arial, sans-serif'})
    
    # Define callbacks
    if gdf_rbs is not None:
        # Pre-load the data
        register_callbacks_with_data(app, gdf_rbs)
    else:
        # Load data through upload
        register_callbacks_with_upload(app)
    
    return app

def parse_contents(contents, filename):
    """
    Parse uploaded file contents into a DataFrame.
    
    Args:
        contents: Contents of the uploaded file
        filename: Name of the uploaded file
        
    Returns:
        DataFrame or GeoDataFrame with the uploaded data
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename.lower():
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename.lower():
            # Assume that the user uploaded an Excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'json' in filename.lower() or 'geojson' in filename.lower():
            # Assume that the user uploaded a JSON/GeoJSON file
            import geopandas as gpd
            df = gpd.read_file(io.BytesIO(decoded))
        else:
            return None, "Unsupported file format. Please upload a CSV, Excel, or GeoJSON file."
        
        return df, f"Successfully loaded {filename}"
    except Exception as e:
        print(e)
        return None, f"Error processing {filename}: {str(e)}"

def register_callbacks_with_upload(app):
    """
    Register callbacks for the dashboard app that loads data through upload.
    
    Args:
        app: Dash application instance
    """
    # Create a global variable to store the uploaded data
    app.uploaded_data = None
    
    @app.callback(
        [Output('upload-message', 'children'),
         Output('data-summary', 'children'),
         Output('operator-pie', 'figure'),
         Output('technology-pie', 'figure')],
        Input('upload-data', 'contents'),
        State('upload-data', 'filename')
    )
    def update_output(contents, filename):
        if contents is None:
            return (
                "No file uploaded yet.", 
                "Upload a file to see summary statistics.", 
                empty_figure("Upload data to see operator distribution"), 
                empty_figure("Upload data to see technology distribution")
            )
        
        # Parse and store the data
        df, message = parse_contents(contents, filename)
        app.uploaded_data = df
        
        if df is None:
            return message, "Error loading data.", empty_figure(), empty_figure()
        
        # Generate summary statistics
        summary = generate_summary_stats(df)
        
        # Generate initial visualizations
        operator_fig = create_operator_pie(df)
        technology_fig = create_technology_pie(df)
        
        return message, summary, operator_fig, technology_fig
    
    # Register the rest of the callbacks for each tab
    register_tab_callbacks(app)

def register_callbacks_with_data(app, gdf_rbs):
    """
    Register callbacks for the dashboard app with pre-loaded data.
    
    Args:
        app: Dash application instance
        gdf_rbs: GeoDataFrame with the RBS data
    """
    # Store the data
    app.uploaded_data = gdf_rbs
    
    # Set initial outputs
    @app.callback(
        [Output('data-summary', 'children'),
         Output('operator-pie', 'figure'),
         Output('technology-pie', 'figure')],
        Input('tabs', 'value')
    )
    def initialize_dashboard(tab_value):
        if tab_value != 'tab-overview':
            raise dash.exceptions.PreventUpdate
        
        # Generate summary statistics
        summary = generate_summary_stats(app.uploaded_data)
        
        # Generate initial visualizations
        operator_fig = create_operator_pie(app.uploaded_data)
        technology_fig = create_technology_pie(app.uploaded_data)
        
        return summary, operator_fig, technology_fig
    
    # Register the rest of the callbacks for each tab
    register_tab_callbacks(app)

def register_tab_callbacks(app):
    """
    Register callbacks for all dashboard tabs.
    
    Args:
        app: Dash application instance
    """
    # Temporal analysis tab callbacks
    @app.callback(
        [Output('deployment-timeline', 'figure'),
         Output('seasonality-chart', 'figure')],
        Input('tabs', 'value')
    )
    def update_temporal_tab(tab_value):
        if tab_value != 'tab-temporal' or app.uploaded_data is None:
            raise dash.exceptions.PreventUpdate
        
        # Preprocess data for temporal analysis
        df = preprocess_temporal_data(app.uploaded_data)
        
        # Create timeline visualization
        timeline_fig = create_deployment_timeline_viz(df)
        
        # Create seasonality visualization
        seasonality_fig = create_seasonality_viz(df)
        
        return timeline_fig, seasonality_fig
    
    # Technology analysis tab callbacks
    @app.callback(
        [Output('tech-scatter', 'figure'),
         Output('freq-histogram', 'figure')],
        Input('tabs', 'value')
    )
    def update_technology_tab(tab_value):
        if tab_value != 'tab-technology' or app.uploaded_data is None:
            raise dash.exceptions.PreventUpdate
        
        # Preprocess data for technology analysis
        df = preprocess_tech_frequency_data(app.uploaded_data)
        
        # Create scatter plot
        scatter_fig = create_tech_scatter_viz(df)
        
        # Create histogram
        histogram_fig = create_freq_histogram_viz(df)
        
        return scatter_fig, histogram_fig
    
    # Spatial analysis tab callbacks
    @app.callback(
        [Output('spatial-map', 'figure'),
         Output('cluster-map', 'figure')],
        Input('tabs', 'value')
    )
    def update_spatial_tab(tab_value):
        if tab_value != 'tab-spatial' or app.uploaded_data is None:
            raise dash.exceptions.PreventUpdate
        
        # Preprocess data for spatial analysis
        gdf = preprocess_spatial_data(app.uploaded_data)
        
        # Create spatial map
        map_fig = create_spatial_map_viz(gdf)
        
        # Create cluster map
        cluster_fig = create_cluster_map_viz(gdf)
        
        return map_fig, cluster_fig
    
    # Correlation analysis tab callbacks
    @app.callback(
        [Output('correlation-matrix', 'figure'),
         Output('scatter-matrix', 'figure')],
        Input('tabs', 'value')
    )
    def update_correlation_tab(tab_value):
        if tab_value != 'tab-correlation' or app.uploaded_data is None:
            raise dash.exceptions.PreventUpdate
        
        # Preprocess data for correlation analysis
        df = preprocess_correlation_data(app.uploaded_data)
        
        # Create correlation matrix
        matrix_fig = create_correlation_matrix_viz(df)
        
        # Create scatter matrix
        scatter_matrix_fig = create_scatter_matrix_viz(df)
        
        return matrix_fig, scatter_matrix_fig
    
    # Prediction tab callbacks
    @app.callback(
        [Output('time-forecast', 'figure'),
         Output('tech-forecast', 'figure')],
        Input('tabs', 'value')
    )
    def update_prediction_tab(tab_value):
        if tab_value != 'tab-prediction' or app.uploaded_data is None:
            raise dash.exceptions.PreventUpdate
        
        # Preprocess data for prediction analysis
        time_series_data = preprocess_prediction_data(app.uploaded_data)
        
        if time_series_data is None:
            return empty_figure("Insufficient data for forecasting"), empty_figure("Insufficient data for technology forecast")
        
        # Create time series forecast
        forecast_fig = create_time_forecast_viz(time_series_data)
        
        # Create technology forecast
        tech_forecast_fig = create_tech_forecast_viz(app.uploaded_data)
        
        return forecast_fig, tech_forecast_fig

# Visualization functions
def empty_figure(message="No data available"):
    """Creates an empty figure with a message."""
    fig = go.Figure()
    fig.update_layout(
        title=message,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(
            text=message,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=20)
        )]
    )
    return fig

def generate_summary_stats(df):
    """Generates HTML with summary statistics for the dataset."""
    try:
        # Basic stats
        total_stations = len(df)
        
        # Try to extract operators count
        operator_counts = {}
        if 'Operator' in df.columns:
            operator_counts = df['Operator'].value_counts().to_dict()
        
        # Try to extract technology count
        tech_counts = {}
        if 'Tecnologia' in df.columns:
            tech_counts = df['Tecnologia'].value_counts().to_dict()
        
        # Create summary table
        summary_html = html.Div([
            html.H4("Dataset Summary", style={'textAlign': 'center'}),
            html.Table([
                html.Tr([html.Th("Metric"), html.Th("Value")]),
                html.Tr([html.Td("Total RBS Stations"), html.Td(total_stations)]),
                *[html.Tr([html.Td(f"{op} Stations"), html.Td(count)]) 
                  for op, count in operator_counts.items()],
                *[html.Tr([html.Td(f"{tech} Deployments"), html.Td(count)]) 
                  for tech, count in tech_counts.items()]
            ], style={
                'margin': 'auto',
                'border': '1px solid #ddd',
                'borderCollapse': 'collapse',
                'width': '50%'
            })
        ])
        
        return summary_html
    
    except Exception as e:
        return html.Div([
            html.P(f"Error generating summary statistics: {str(e)}"),
            html.P("Please ensure your data contains the expected columns.")
        ])

def create_operator_pie(df):
    """Creates a pie chart showing distribution by operator."""
    try:
        if 'Operator' not in df.columns:
            return empty_figure("No operator data found")
        
        # Count by operator
        operator_counts = df['Operator'].value_counts().reset_index()
        operator_counts.columns = ['Operator', 'Count']
        
        # Create color map
        color_map = {op: OPERATOR_COLORS.get(op, '#808080') for op in operator_counts['Operator']}
        colors = [color_map.get(op, '#808080') for op in operator_counts['Operator']]
        
        # Create pie chart
        fig = px.pie(
            operator_counts, 
            values='Count', 
            names='Operator',
            title='RBS Distribution by Operator',
            color_discrete_sequence=colors
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(template='plotly_white')
        
        return fig
    
    except Exception as e:
        return empty_figure(f"Error creating operator pie chart: {str(e)}")

def create_technology_pie(df):
    """Creates a pie chart showing distribution by technology."""
    try:
        if 'Tecnologia' not in df.columns:
            return empty_figure("No technology data found")
        
        # Count by technology
        tech_counts = df['Tecnologia'].value_counts().reset_index()
        tech_counts.columns = ['Technology', 'Count']
        
        # Create pie chart
        fig = px.pie(
            tech_counts, 
            values='Count', 
            names='Technology',
            title='RBS Distribution by Technology',
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(template='plotly_white')
        
        return fig
    
    except Exception as e:
        return empty_figure(f"Error creating technology pie chart: {str(e)}")

def create_deployment_timeline_viz(df):
    """Creates a timeline visualization of RBS deployments."""
    try:
        # Check if we have the necessary data
        if 'installation_date' not in df.columns:
            return empty_figure("No installation date data found")
        
        # Group by year and operator
        if 'Operator' in df.columns:
            yearly_data = df.groupby([df['installation_date'].dt.year, 'Operator']).size().unstack(fill_value=0)
            yearly_data['Total'] = yearly_data.sum(axis=1)
            
            # Calculate cumulative sum
            cumulative_data = yearly_data.cumsum()
            
            # Create the figure
            fig = go.Figure()
            
            # Add traces for each operator
            for operator in cumulative_data.columns:
                if operator != 'Total':
                    fig.add_trace(go.Scatter(
                        x=cumulative_data.index,
                        y=cumulative_data[operator],
                        mode='lines+markers',
                        name=operator,
                        line=dict(width=2, color=OPERATOR_COLORS.get(operator, '#808080'))
                    ))
            
            # Add total trace
            fig.add_trace(go.Scatter(
                x=cumulative_data.index,
                y=cumulative_data['Total'],
                mode='lines+markers',
                name='Total',
                line=dict(width=3, color='black', dash='dash')
            ))
            
            # Update layout
            fig.update_layout(
                title='Cumulative RBS Deployment Over Time',
                xaxis_title='Year',
                yaxis_title='Number of Stations',
                template='plotly_white',
                hovermode='x unified'
            )
            
            return fig
        else:
            # No operator data, just show total by year
            yearly_data = df.groupby(df['installation_date'].dt.year).size()
            cumulative_data = yearly_data.cumsum()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=cumulative_data.index,
                y=cumulative_data.values,
                mode='lines+markers',
                name='Total',
                line=dict(width=3, color='blue')
            ))
            
            # Update layout
            fig.update_layout(
                title='Cumulative RBS Deployment Over Time',
                xaxis_title='Year',
                yaxis_title='Number of Stations',
                template='plotly_white'
            )
            
            return fig
    
    except Exception as e:
        return empty_figure(f"Error creating deployment timeline: {str(e)}")

def create_seasonality_viz(df):
    """Creates a visualization showing seasonal patterns in RBS installations."""
    try:
        # Check if we have the necessary data
        if 'installation_date' not in df.columns:
            return empty_figure("No installation date data found")
        
        # Create monthly installation counts
        monthly_counts = df.groupby(df['installation_date'].dt.month).size()
        
        # Ensure all months are represented
        all_months = {i: monthly_counts.get(i, 0) for i in range(1, 13)}
        monthly_df = pd.DataFrame({
            'Month': list(all_months.keys()),
            'Installations': list(all_months.values())
        })
        
        # Add month names
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        monthly_df['Month_Name'] = monthly_df['Month'].map(month_names)
        
        # Create the figure
        fig = px.bar(
            monthly_df,
            x='Month_Name',
            y='Installations',
            title='Monthly Installation Patterns',
            color='Installations',
            color_continuous_scale='Viridis'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title='Number of Installations',
            template='plotly_white'
        )
        
        return fig
    
    except Exception as e:
        return empty_figure(f"Error creating seasonality chart: {str(e)}")

def create_tech_scatter_viz(df):
    """Creates a scatter plot of frequency vs power by technology."""
    try:
        # Check if we have the necessary columns
        if 'FreqTxMHz' not in df.columns or 'PotenciaTransmissorWatts' not in df.columns:
            return empty_figure("Required frequency or power data not found")
        
        # Create the figure
        if 'Tecnologia' in df.columns and 'Operator' in df.columns:
            fig = px.scatter(
                df,
                x='FreqTxMHz',
                y='PotenciaTransmissorWatts',
                color='Tecnologia',
                symbol='Operator',
                hover_name='Operator',
                title='Relationship Between Frequency and Power by Technology',
                labels={
                    'FreqTxMHz': 'Transmission Frequency (MHz)',
                    'PotenciaTransmissorWatts': 'Transmitter Power (Watts)',
                    'Tecnologia': 'Technology',
                    'Operator': 'Operator'
                }
            )
        elif 'Tecnologia' in df.columns:
            fig = px.scatter(
                df,
                x='FreqTxMHz',
                y='PotenciaTransmissorWatts',
                color='Tecnologia',
                title='Relationship Between Frequency and Power by Technology',
                labels={
                    'FreqTxMHz': 'Transmission Frequency (MHz)',
                    'PotenciaTransmissorWatts': 'Transmitter Power (Watts)',
                    'Tecnologia': 'Technology'
                }
            )
        else:
            fig = px.scatter(
                df,
                x='FreqTxMHz',
                y='PotenciaTransmissorWatts',
                title='Relationship Between Frequency and Power',
                labels={
                    'FreqTxMHz': 'Transmission Frequency (MHz)',
                    'PotenciaTransmissorWatts': 'Transmitter Power (Watts)'
                }
            )
        
        # Update layout
        fig.update_layout(template='plotly_white')
        
        return fig
    
    except Exception as e:
        return empty_figure(f"Error creating technology scatter plot: {str(e)}")

def create_freq_histogram_viz(df):
    """Creates histograms of frequency distribution by operator."""
    try:
        # Check if we have the necessary columns
        if 'FreqTxMHz' not in df.columns:
            return empty_figure("No frequency data found")
        
        # Create the figure
        if 'Operator' in df.columns:
            fig = px.histogram(
                df,
                x='FreqTxMHz',
                color='Operator',
                barmode='overlay',
                opacity=0.7,
                nbins=30,
                color_discrete_map=OPERATOR_COLORS,
                title='Frequency Distribution by Operator',
                labels={'FreqTxMHz': 'Transmission Frequency (MHz)'}
            )
        else:
            fig = px.histogram(
                df,
                x='FreqTxMHz',
                nbins=30,
                title='Frequency Distribution',
                labels={'FreqTxMHz': 'Transmission Frequency (MHz)'}
            )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Frequency (MHz)',
            yaxis_title='Count',
            bargap=0.05,
            template='plotly_white'
        )
        
        return fig
    
    except Exception as e:
        return empty_figure(f"Error creating frequency histogram: {str(e)}")

def create_spatial_map_viz(gdf):
    """Creates a map visualization of RBS locations."""
    try:
        # Check if we have geometry
        if not hasattr(gdf, 'geometry') or gdf.geometry.isna().all():
            return empty_figure("No valid geometry data found")
        
        # Create the map
        if 'Operator' in gdf.columns:
            fig = px.scatter_mapbox(
                gdf,
                lat=gdf.geometry.y,
                lon=gdf.geometry.x,
                color='Operator',
                color_discrete_map=OPERATOR_COLORS,
                zoom=10,
                title='RBS Spatial Distribution',
                mapbox_style='carto-positron',
                hover_name='Operator' if 'Operator' in gdf.columns else None,
                hover_data=['Tecnologia', 'FreqTxMHz'] if all(col in gdf.columns for col in ['Tecnologia', 'FreqTxMHz']) else None
            )
        else:
            fig = px.scatter_mapbox(
                gdf,
                lat=gdf.geometry.y,
                lon=gdf.geometry.x,
                zoom=10,
                title='RBS Spatial Distribution',
                mapbox_style='carto-positron'
            )
        
        # Update layout
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            mapbox=dict(
                center=dict(
                    lat=gdf.geometry.y.mean(),
                    lon=gdf.geometry.x.mean()
                )
            )
        )
        
        return fig
    
    except Exception as e:
        return empty_figure(f"Error creating spatial map: {str(e)}")

def create_cluster_map_viz(gdf):
    """Creates a visualization of DBSCAN clustering on RBS locations."""
    try:
        # Check if we have geometry
        if not hasattr(gdf, 'geometry') or gdf.geometry.isna().all():
            return empty_figure("No valid geometry data found")
        
        # Perform DBSCAN clustering
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        
        # Extract coordinates
        coords = np.array([[point.x, point.y] for point in gdf.geometry])
        
        # Standardize
        coords_scaled = StandardScaler().fit_transform(coords)
        
        # Cluster
        db = DBSCAN(eps=0.2, min_samples=5).fit(coords_scaled)
        
        # Add cluster labels
        gdf = gdf.copy()
        gdf['cluster'] = db.labels_
        
        # Count clusters (excluding noise points labeled as -1)
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        
        # Create the map
        fig = px.scatter_mapbox(
            gdf,
            lat=gdf.geometry.y,
            lon=gdf.geometry.x,
            color='cluster',
            color_continuous_scale='Viridis',
            zoom=10,
            title=f'DBSCAN Clustering of RBS Locations (Found {n_clusters} clusters)',
            mapbox_style='carto-positron',
            hover_name='cluster',
            hover_data=['Operator'] if 'Operator' in gdf.columns else None
        )
        
        # Update layout
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            mapbox=dict(
                center=dict(
                    lat=gdf.geometry.y.mean(),
                    lon=gdf.geometry.x.mean()
                )
            )
        )
        
        return fig
    
    except Exception as e:
        return empty_figure(f"Error creating cluster map: {str(e)}")

def create_correlation_matrix_viz(df):
    """Creates a correlation matrix visualization of technical variables."""
    try:
        # Define key technical columns for correlation
        tech_columns = [
            'FreqTxMHz', 
            'PotenciaTransmissorWatts',
            'AlturaAntena',
            'Coverage_Radius_km'
        ]
        
        # Check which columns exist
        available_cols = [col for col in tech_columns if col in df.columns]
        
        if len(available_cols) < 2:
            return empty_figure("Not enough technical variables for correlation analysis")
        
        # Calculate correlation matrix
        corr_matrix = df[available_cols].corr(method='pearson')
        
        # Create the heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            title='Correlation Matrix of Technical Variables'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Variable',
            yaxis_title='Variable',
            template='plotly_white'
        )
        
        return fig
    
    except Exception as e:
        return empty_figure(f"Error creating correlation matrix: {str(e)}")

def create_scatter_matrix_viz(df):
    """Creates a scatter matrix visualization of technical variables."""
    try:
        # Define key technical columns for the scatter matrix
        tech_columns = [
            'FreqTxMHz', 
            'PotenciaTransmissorWatts',
            'AlturaAntena',
            'Coverage_Radius_km'
        ]
        
        # Check which columns exist
        available_cols = [col for col in tech_columns if col in df.columns]
        
        if len(available_cols) < 2:
            return empty_figure("Not enough technical variables for scatter matrix")
        
        # Create the scatter matrix
        if 'Operator' in df.columns:
            fig = px.scatter_matrix(
                df[available_cols + ['Operator']],
                dimensions=available_cols,
                color='Operator',
                color_discrete_map=OPERATOR_COLORS,
                title='Relationships Between Technical Variables',
                opacity=0.7
            )
        else:
            fig = px.scatter_matrix(
                df[available_cols],
                dimensions=available_cols,
                title='Relationships Between Technical Variables',
                opacity=0.7
            )
        
        # Update layout
        fig.update_layout(
            template='plotly_white',
            height=700
        )
        
        # Update traces
        fig.update_traces(
            diagonal_visible=False,
            showupperhalf=False
        )
        
        return fig
    
    except Exception as e:
        return empty_figure(f"Error creating scatter matrix: {str(e)}")

def create_time_forecast_viz(time_series_data):
    """Creates a time series forecast visualization."""
    try:
        # Check if we have the necessary data
        if 'monthly' not in time_series_data:
            return empty_figure("No monthly data available for forecasting")
        
        monthly_df = time_series_data['monthly']
        
        # Fit simple ARIMA model
        from statsmodels.tsa.arima.model import ARIMA
        
        # Set the date as index
        ts = monthly_df.set_index('date')['installations']
        
        # Fit ARIMA model (p, d, q) - using simple parameters
        model = ARIMA(ts, order=(1, 1, 1))
        fitted_model = model.fit()
        
        # Forecast future values (next 12 months)
        forecast_steps = 12
        forecast = fitted_model.forecast(steps=forecast_steps)
        
        # Create future dates
        last_date = ts.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), 
                                   periods=forecast_steps, 
                                   freq='M')
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecast': forecast.values
        })
        
        # Create the figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=ts.index, 
            y=ts.values,
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title='RBS Deployment Forecast (ARIMA)',
            xaxis_title='Date',
            yaxis_title='Number of Installations',
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    except Exception as e:
        return empty_figure(f"Error creating time forecast: {str(e)}")

def create_tech_forecast_viz(gdf_rbs):
    """Creates a technology trend forecast visualization."""
    try:
        # Check if we have the necessary columns
        if 'installation_date' not in gdf_rbs.columns or 'Tecnologia' not in gdf_rbs.columns:
            return empty_figure("Missing required data for technology forecast")
        
        # Preprocess data for temporal analysis
        df = preprocess_temporal_data(gdf_rbs)
        
        # Group by date and technology
        tech_time_series = df.groupby([pd.Grouper(key='installation_date', freq='Q'), 'Tecnologia']).size().unstack(fill_value=0)
        
        # Get list of technologies
        technologies = tech_time_series.columns
        
        # Create forecast for each technology
        tech_forecasts = {}
        forecast_periods = 8  # 2 years of quarterly forecasts
        
        for tech in technologies:
            # Extract time series for this technology
            tech_ts = tech_time_series[tech]
            
            # Use simple trend continuation as fallback
            if len(tech_ts) > 3:
                last_values = tech_ts.values[-3:]
                avg_growth = np.mean(np.diff(last_values))
            else:
                avg_growth = 0
            
            future_values = [tech_ts.values[-1]]
            for _ in range(forecast_periods - 1):
                future_values.append(future_values[-1] + avg_growth)
            
            last_date = tech_ts.index[-1]
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=3), 
                                        periods=forecast_periods, 
                                        freq='Q')
            
            tech_forecasts[tech] = pd.Series(future_values, index=future_dates)
        
        # Create the figure
        fig = go.Figure()
        
        # Add historical data
        for tech in technologies:
            historical = tech_time_series[tech]
            fig.add_trace(go.Scatter(
                x=historical.index,
                y=historical.values,
                mode='lines',
                name=f'{tech} (Historical)',
                line=dict(width=2)
            ))
        
        # Add forecast data
        for tech, forecast in tech_forecasts.items():
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast.values,
                mode='lines',
                name=f'{tech} (Forecast)',
                line=dict(width=2, dash='dash')
            ))
        
        # Add vertical line to separate historical and forecast
        last_historical_date = tech_time_series.index[-1]
        fig.add_vline(
            x=last_historical_date,
            line_width=2,
            line_dash="dash",
            line_color="black",
            annotation_text="Forecast Start",
            annotation_position="top right"
        )
        
        # Update layout
        fig.update_layout(
            title='Technology Adoption Forecast',
            xaxis_title='Date',
            yaxis_title='Number of Installations',
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        return fig
    
    except Exception as e:
        return empty_figure(f"Error creating technology forecast: {str(e)}")

def run_dashboard(gdf_rbs=None, debug=False, port=8050):
    """
    Run the RBS analysis dashboard.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data (optional)
        debug: Whether to run the app in debug mode
        port: Port to run the app on
    """
    app = init_dashboard(gdf_rbs)
    app.run_server(debug=debug, port=port) 