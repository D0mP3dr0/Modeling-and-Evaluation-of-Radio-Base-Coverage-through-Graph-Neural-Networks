import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
import math

# Load the RBS data with elevation information (created by integrated_analysis.py)
gdf_path = 'rbs_with_elevation.gpkg'

if not os.path.exists(gdf_path):
    print(f"Error: {gdf_path} does not exist. Run integrated_analysis.py first.")
    exit(1)

# Load the GeoDataFrame
print(f"Loading RBS data with elevation from {gdf_path}...")
gdf = gpd.read_file(gdf_path)
print(f"Loaded {len(gdf)} RBS stations with elevation data")

# --- Data cleaning and preparation ---
# Convert antenna height and power to numeric
for col in ['AlturaAntena', 'PotenciaTransmissorWatts', 'GanhoAntena', 'FreqTxMHz']:
    if col in gdf.columns:
        gdf[col] = pd.to_numeric(gdf[col], errors='coerce')

# Filter out rows with missing key values
required_cols = ['AlturaAntena', 'PotenciaTransmissorWatts', 'elevation']
gdf_clean = gdf.dropna(subset=required_cols)
print(f"RBS stations with complete data: {len(gdf_clean)}")

# --- Add derived columns for analysis ---
# Calculate total height (antenna height + elevation)
gdf_clean['total_height'] = gdf_clean['AlturaAntena'] + gdf_clean['elevation']

# Simple propagation distance estimate using free space path loss model
# d = 10^((EIRP - Rx_Sensitivity - 20*log10(f) + 147.55) / 20)
# EIRP = Tx_Power_dBm + Antenna_Gain_dBi

def calculate_propagation_estimate(row):
    # Skip if frequency is missing
    if pd.isna(row['FreqTxMHz']):
        return np.nan
        
    # Convert Watts to dBm: P(dBm) = 10 * log10(P(W) * 1000)
    try:
        power_dbm = 10 * np.log10(row['PotenciaTransmissorWatts'] * 1000)
        
        # Add gain if available
        gain_dbi = row['GanhoAntena'] if not pd.isna(row['GanhoAntena']) else 0
        
        # Calculate EIRP
        eirp = power_dbm + gain_dbi
        
        # Assume a typical receiver sensitivity of -100 dBm
        rx_sensitivity = -100
        
        # Frequency in MHz
        freq_mhz = row['FreqTxMHz']
        
        # Free space path loss formula
        path_loss_exponent = (eirp - rx_sensitivity - 20 * np.log10(freq_mhz) + 147.55) / 20
        
        # Calculate distance in km
        distance_km = 10 ** path_loss_exponent
        
        # Apply height/elevation boost factor
        # Simple approximation: higher stations have better propagation
        # This is a simplification - more complex models would account for terrain properly
        height_factor = 1 + np.log10(row['total_height'] / 10) if row['total_height'] > 10 else 1
        
        return distance_km * height_factor
    except:
        return np.nan

# Apply propagation estimate calculation
print("Calculating propagation estimates...")
gdf_clean['propagation_est_km'] = gdf_clean.apply(calculate_propagation_estimate, axis=1)

# --- Analysis and Visualizations ---
# 1. Correlation between elevation and propagation
print("\nAnalyzing relationship between elevation and estimated propagation...")
correlation = gdf_clean['elevation'].corr(gdf_clean['propagation_est_km'])
print(f"Correlation between elevation and propagation estimate: {correlation:.4f}")

# 2. Create visualization - Elevation vs. Propagation
plt.figure(figsize=(10, 6))
plt.scatter(gdf_clean['elevation'], gdf_clean['propagation_est_km'], 
            alpha=0.5, c=gdf_clean['PotenciaTransmissorWatts'], cmap='viridis')
plt.colorbar(label='Transmitter Power (Watts)')
plt.title('Relationship between Elevation and Estimated Propagation Distance')
plt.xlabel('Elevation (meters)')
plt.ylabel('Estimated Propagation Distance (km)')
plt.grid(True, alpha=0.3)
plt.savefig('elevation_vs_propagation.png')
print("Saved elevation vs propagation plot to elevation_vs_propagation.png")

# 3. Propagation by Operator
if 'NomeEntidade' in gdf_clean.columns:
    # Get top 5 operators by count
    top_operators = gdf_clean['NomeEntidade'].value_counts().head(5).index.tolist()
    
    # Filter data for top operators
    top_op_data = gdf_clean[gdf_clean['NomeEntidade'].isin(top_operators)]
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='NomeEntidade', y='propagation_est_km', data=top_op_data)
    plt.title('Estimated Propagation Distance by Top Operators')
    plt.xlabel('Operator')
    plt.ylabel('Estimated Propagation Distance (km)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('propagation_by_operator.png')
    print("Saved propagation by operator plot to propagation_by_operator.png")

# 4. Scatter plot of total height vs propagation with density
plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_height', y='propagation_est_km', hue='FreqTxMHz', 
                data=gdf_clean, palette='viridis', size='PotenciaTransmissorWatts', 
                sizes=(20, 200), alpha=0.6)
plt.title('Propagation Distance vs Total Height (Antenna + Elevation)')
plt.xlabel('Total Height (meters)')
plt.ylabel('Estimated Propagation Distance (km)')
plt.legend(title='Frequency (MHz)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('height_vs_propagation.png')
print("Saved height vs propagation plot to height_vs_propagation.png")

# Create a summary dataframe for export
summary = gdf_clean.groupby('NomeEntidade').agg({
    'propagation_est_km': ['count', 'mean', 'std', 'min', 'max'],
    'elevation': ['mean', 'min', 'max'],
    'AlturaAntena': ['mean'],
    'PotenciaTransmissorWatts': ['mean']
}).reset_index()

# Flatten the column names
summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

# Sort by number of stations
summary = summary.sort_values('propagation_est_km_count', ascending=False)

# Save the summary to CSV
summary.to_csv('propagation_summary_by_operator.csv', index=False)
print("Saved summary statistics to propagation_summary_by_operator.csv")

print("\nAnalysis completed successfully!") 