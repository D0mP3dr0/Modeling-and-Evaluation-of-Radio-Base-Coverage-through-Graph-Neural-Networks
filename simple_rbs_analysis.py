import os
import csv
import math
import matplotlib.pyplot as plt

# File paths
csv_path = 'data/csv_licenciamento_bruto.csv.csv'

# Check if file exists
if not os.path.exists(csv_path):
    print(f"Error: RBS data file {csv_path} does not exist.")
    exit(1)

print(f"Loading RBS data from {csv_path}...")

# Function to read CSV and extract relevant data
def read_rbs_data(filepath):
    data = []
    operators = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Extract basic information
                operator = row.get('NomeEntidade', 'Unknown')
                lat = float(row.get('Latitude', 0))
                lon = float(row.get('Longitude', 0))
                
                # Extract technical specifications
                try:
                    antenna_height = float(row.get('AlturaAntena', 0))
                except (ValueError, TypeError):
                    antenna_height = 0
                    
                try:
                    power = float(row.get('PotenciaTransmissorWatts', 0))
                except (ValueError, TypeError):
                    power = 0
                    
                try:
                    gain = float(row.get('GanhoAntena', 0))
                except (ValueError, TypeError):
                    gain = 0
                    
                try:
                    freq = float(row.get('FreqTxMHz', 0))
                except (ValueError, TypeError):
                    freq = 0
                
                # Only include records with valid coordinates
                if lat != 0 and lon != 0:
                    # Create a record
                    record = {
                        'operator': operator,
                        'lat': lat,
                        'lon': lon,
                        'antenna_height': antenna_height,
                        'power': power,
                        'gain': gain,
                        'freq': freq
                    }
                    
                    data.append(record)
                    
                    # Count by operator
                    if operator in operators:
                        operators[operator] += 1
                    else:
                        operators[operator] = 1
            except Exception as e:
                # Skip problematic rows
                continue
    
    return data, operators

# Read the data
rbs_data, operators = read_rbs_data(csv_path)
print(f"Loaded {len(rbs_data)} RBS records with valid coordinates")

# Basic statistics
print("\nTop 10 operators by number of stations:")
top_operators = sorted(operators.items(), key=lambda x: x[1], reverse=True)[:10]
for operator, count in top_operators:
    print(f"{operator}: {count} stations")

# Calculate simple statistics for antenna height and power
heights = [r['antenna_height'] for r in rbs_data if r['antenna_height'] > 0]
powers = [r['power'] for r in rbs_data if r['power'] > 0]

# Calculate statistics
avg_height = sum(heights) / len(heights) if heights else 0
avg_power = sum(powers) / len(powers) if powers else 0

print(f"\nAverage antenna height: {avg_height:.2f} meters")
print(f"Average transmitter power: {avg_power:.2f} watts")

# Create visualizations
print("\nCreating visualizations...")

# 1. Plot RBS distribution on a simple map
plt.figure(figsize=(12, 8))
plt.scatter([r['lon'] for r in rbs_data], [r['lat'] for r in rbs_data], 
            s=5, alpha=0.5, c='blue')
plt.title('RBS Distribution')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rbs_distribution.png')
print("Saved RBS distribution map to rbs_distribution.png")

# 2. Histogram of antenna heights
plt.figure(figsize=(10, 6))
plt.hist(heights, bins=30, alpha=0.7)
plt.title('Distribution of Antenna Heights')
plt.xlabel('Height (meters)')
plt.ylabel('Number of Stations')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('antenna_height_distribution.png')
print("Saved antenna height distribution to antenna_height_distribution.png")

# 3. Histogram of transmitter powers
plt.figure(figsize=(10, 6))
plt.hist(powers, bins=30, alpha=0.7)
plt.title('Distribution of Transmitter Power')
plt.xlabel('Power (watts)')
plt.ylabel('Number of Stations')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('transmitter_power_distribution.png')
print("Saved transmitter power distribution to transmitter_power_distribution.png")

# 4. Scatter plot of height vs. power
plt.figure(figsize=(10, 6))
valid_data = [(r['antenna_height'], r['power']) for r in rbs_data 
              if r['antenna_height'] > 0 and r['power'] > 0]
if valid_data:
    heights_plot, powers_plot = zip(*valid_data)
    plt.scatter(heights_plot, powers_plot, alpha=0.5)
    plt.title('Antenna Height vs. Transmitter Power')
    plt.xlabel('Height (meters)')
    plt.ylabel('Power (watts)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('height_vs_power.png')
    print("Saved height vs. power plot to height_vs_power.png")

print("\nAnalysis completed successfully!") 