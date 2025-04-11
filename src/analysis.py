import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from mpl_toolkits.basemap import Basemap # Commented due to installation complexity
import folium
from folium.plugins import HeatMap, MarkerCluster
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def setup_visualization_options():
    """Configures global options for visualization libraries."""
    plt.style.use('ggplot')
    sns.set(style="whitegrid")
    pd.set_option('display.max_columns', None)
    plt.rcParams['figure.figsize'] = (12, 8)

def exploratory_analysis_rbs(df: pd.DataFrame, results_path: str):
    """Performs exploratory analysis and generates visualizations for the RBS dataset."""

    if df is None or df.empty:
        print("Empty or invalid DataFrame. Analysis cannot be performed.")
        return

    print("=" * 80)
    print("EXPLORATORY ANALYSIS - RADIO BASE STATIONS")
    print("=" * 80)

    # --- 1. OVERVIEW --- 
    print("\n1. DATA OVERVIEW")
    print("-" * 50)
    print(f"Total stations: {df.shape[0]}")
    print(f"Available variables: {df.shape[1]}")
    print("\nFirst rows of the dataset:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)

    # --- 2. MISSING VALUES --- 
    print("\n\n2. MISSING VALUES ANALYSIS")
    print("-" * 50)
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_data = pd.DataFrame({
        'Missing values': missing_values,
        'Percentage (%)': missing_percent.round(2)
    })
    missing_data_sorted = missing_data[missing_data['Missing values'] > 0].sort_values('Percentage (%)', ascending=False)
    if not missing_data_sorted.empty:
      print(missing_data_sorted)
    else:
      print("There are no missing values in the dataset.")

    # --- 3. DESCRIPTIVE STATISTICS --- 
    print("\n\n3. DESCRIPTIVE STATISTICS")
    print("-" * 50)
    
    # Try to convert potentially numeric columns that may be as object type
    potential_numeric_cols = df.select_dtypes(include=['object']).columns
    for col in potential_numeric_cols:
        try:
            # Try to convert to numeric after treating commas as decimal separators
            converted_col = pd.to_numeric(df[col].str.replace(',', '.', regex=False), errors='coerce')
            # If most non-null values could be converted, update the column
            if converted_col.notna().sum() / df[col].notna().sum() > 0.8: # 80% threshold
                df[col] = converted_col
                print(f"Column '{col}' converted to numeric type.")
        except AttributeError:
            # Ignore columns that are not strings
            pass 
        except Exception as e:
            print(f"Could not process column '{col}' for numeric conversion: {e}")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        print("\nNumeric variables statistics:")
        numeric_stats = df[numeric_cols].describe().T
        numeric_stats['median'] = df[numeric_cols].median()
        numeric_stats['variance'] = df[numeric_cols].var()
        print(numeric_stats)
    else:
        print("No numeric columns found for statistics.")

    # --- 4. GEOGRAPHIC ANALYSIS --- 
    print("\n\n4. GEOGRAPHIC ANALYSIS")
    print("-" * 50)
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        print(f"Lat Range: {df['Latitude'].min()} to {df['Latitude'].max()}")
        print(f"Lon Range: {df['Longitude'].min()} to {df['Longitude'].max()}")
        if 'Municipio.NomeMunicipio' in df.columns:
            print("\nTop 10 Municipalities by number of RBS:")
            print(df['Municipio.NomeMunicipio'].value_counts().head(10))
        
        # Generate Folium Map (based on the code seen)
        try:
            geo_df = df.dropna(subset=['Latitude', 'Longitude'])
            geo_df = geo_df[(geo_df['Latitude'] != 0) & (geo_df['Longitude'] != 0)]
            
            if not geo_df.empty:
                map_center = [geo_df['Latitude'].mean(), geo_df['Longitude'].mean()]
                m = folium.Map(location=map_center, zoom_start=10)
                marker_cluster = MarkerCluster().add_to(m)

                for idx, row in geo_df.iterrows():
                    popup_text = f"""
                    <b>Operator:</b> {row.get('NomeEntidade', 'N/A')}<br>
                    <b>Technology:</b> {row.get('Tecnologia', 'N/A')}<br>
                    <b>Tx Freq:</b> {row.get('FreqTxMHz', 'N/A')} MHz<br>
                    <b>Height:</b> {row.get('AlturaAntena', 'N/A')} m<br>
                    """
                    folium.Marker(
                        location=[row['Latitude'], row['Longitude']],
                        popup=folium.Popup(popup_text, max_width=300),
                        tooltip=f"{row.get('NomeEntidade', 'RBS')}: {row.get('Tecnologia', '')}"
                    ).add_to(marker_cluster)
                
                map_file_path = f"{results_path}/rbs_distribution_map.html"
                m.save(map_file_path)
                print(f"\nInteractive distribution map saved at: {map_file_path}")
            else:
                 print("\nThere's no valid geolocation data to generate the map.")
                 
        except Exception as e:
            print(f"\nError generating Folium map: {e}")
            
    else:
        print("Columns 'Latitude' and/or 'Longitude' not found. Limited geographic analysis.")
        
    # --- 5. TECHNOLOGY ANALYSIS --- 
    print("\n\n5. TECHNOLOGY ANALYSIS")
    print("-" * 50)
    if 'Tecnologia' in df.columns:
        print("\nTechnology distribution:")
        print(df['Tecnologia'].value_counts())
    if 'tipoTecnologia' in df.columns:
        print("\nTechnology type distribution:")
        print(df['tipoTecnologia'].value_counts())
        
    # --- 6. FREQUENCY ANALYSIS --- 
    print("\n\n6. FREQUENCY ANALYSIS")
    print("-" * 50)
    if 'FreqTxMHz' in df.columns:
        print(f"Tx Freq: {df['FreqTxMHz'].min()} MHz to {df['FreqTxMHz'].max()} MHz")
    if 'FreqRxMHz' in df.columns:
        print(f"Rx Freq: {df['FreqRxMHz'].min()} MHz to {df['FreqRxMHz'].max()} MHz")
        
    # --- 7. ANALYSIS BY OPERATOR --- 
    print("\n\n7. ANALYSIS BY OPERATOR")
    print("-" * 50)
    if 'NomeEntidade' in df.columns:
        print("\nTop 10 Operators by number of RBS:")
        print(df['NomeEntidade'].value_counts().head(10))

    # --- 8. ADDITIONAL VISUALIZATIONS (Static Charts) ---
    print("\n\n8. GENERATING ADDITIONAL CHARTS")
    print("-" * 50)
    
    try:
        plt.figure(figsize=(18, 6))
        
        # Antenna Height Histogram
        if 'AlturaAntena' in df.columns and pd.api.types.is_numeric_dtype(df['AlturaAntena']):
            plt.subplot(1, 2, 1)
            sns.histplot(df['AlturaAntena'].dropna(), bins=30, kde=True)
            plt.title('Antenna Height Distribution')
            plt.xlabel('Height (m)')
            plt.ylabel('Frequency')
        else:
             print("Column 'AlturaAntena' not found or not numeric for histogram.")
             
        # Technologies Bar Chart
        if 'Tecnologia' in df.columns:
            plt.subplot(1, 2, 2)
            tech_counts = df['Tecnologia'].value_counts()
            sns.barplot(x=tech_counts.index, y=tech_counts.values)
            plt.title('Technology Distribution')
            plt.xlabel('Technology')
            plt.ylabel('Quantity')
            plt.xticks(rotation=45, ha='right')
        else:
             print("Column 'Tecnologia' not found for bar chart.")

        plt.tight_layout()
        plot_file_path = f"{results_path}/height_technology_distribution_plots.png"
        plt.savefig(plot_file_path)
        print(f"Distribution charts saved at: {plot_file_path}")
        plt.close() # Close the figure to release memory

    except Exception as e:
        print(f"Error generating static charts: {e}")

    # Add here the code for other visualizations that were in the notebook
    # (ex: Gain x Power), adapting to save in `results_path`
    print("\nExploratory analysis completed.")

# Example of how to call (will be done in main.py):
# setup_visualization_options()
# df_analysis = pd.read_csv('data/erb_sorocaba_clean.csv') # Assuming the clean file is in data/
# exploratory_analysis_rbs(df_analysis, 'results')
