import os
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

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
                tech = row.get('Tecnologia', 'Unknown')
                
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
                
                try:
                    azimuth = float(row.get('Azimute', 0))
                except (ValueError, TypeError):
                    azimuth = 0
                
                # Get municipality
                municipality = row.get('Municipio.NomeMunicipio', 'Unknown')
                
                # Only include records with valid coordinates
                if lat != 0 and lon != 0:
                    # Create a record
                    record = {
                        'operator': operator,
                        'lat': lat,
                        'lon': lon,
                        'tech': tech,
                        'antenna_height': antenna_height,
                        'power': power,
                        'gain': gain,
                        'freq': freq,
                        'azimuth': azimuth,
                        'municipality': municipality
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

# Filter for major telecom operators only
major_operators = ['TELEFONICA BRASIL S.A.', 'CLARO S.A.', 'TIM S A']
telecom_data = [r for r in rbs_data if r['operator'] in major_operators]
print(f"Selected {len(telecom_data)} stations from major telecom operators")

# Group data by operator
operator_data = defaultdict(list)
for record in telecom_data:
    operator_data[record['operator']].append(record)

# Calculate operator statistics
operator_stats = {}
for operator, records in operator_data.items():
    # Calculate stats
    heights = [r['antenna_height'] for r in records if r['antenna_height'] > 0]
    powers = [r['power'] for r in records if r['power'] > 0]
    freqs = [r['freq'] for r in records if r['freq'] > 0]
    
    # Technology distribution
    tech_counts = defaultdict(int)
    for r in records:
        tech_counts[r['tech']] += 1
    
    # Municipality coverage
    municipalities = set(r['municipality'] for r in records if r['municipality'] != 'Unknown')
    
    # Store stats
    operator_stats[operator] = {
        'count': len(records),
        'avg_height': sum(heights) / len(heights) if heights else 0,
        'avg_power': sum(powers) / len(powers) if powers else 0,
        'avg_freq': sum(freqs) / len(freqs) if freqs else 0,
        'tech_distribution': dict(tech_counts),
        'municipality_count': len(municipalities)
    }

# Print operator comparison
print("\nOperator Comparison:")
print("-" * 80)
print(f"{'Operator':<30} {'Stations':<10} {'Avg Height':<15} {'Avg Power':<15} {'Municipalities':<15}")
print("-" * 80)
for operator, stats in operator_stats.items():
    print(f"{operator:<30} {stats['count']:<10} {stats['avg_height']:.2f} m{'':<7} {stats['avg_power']:.2f} W{'':<7} {stats['municipality_count']:<15}")

# Technology distribution by operator
print("\nTechnology Distribution by Operator:")
print("-" * 80)
for operator, stats in operator_stats.items():
    print(f"\n{operator}:")
    for tech, count in sorted(stats['tech_distribution'].items(), key=lambda x: x[1], reverse=True):
        if tech != 'Unknown' and tech is not None and tech != '':
            percentage = (count / stats['count']) * 100
            print(f"  {tech:<10}: {count} stations ({percentage:.1f}%)")

# Create visualizations
print("\nCreating visualizations...")

# 1. Plot RBS distribution by operator
plt.figure(figsize=(12, 8))
colors = {'TELEFONICA BRASIL S.A.': 'blue', 'CLARO S.A.': 'red', 'TIM S A': 'green'}
for operator, records in operator_data.items():
    lons = [r['lon'] for r in records]
    lats = [r['lat'] for r in records]
    plt.scatter(lons, lats, s=5, alpha=0.5, c=colors[operator], label=operator)

plt.title('RBS Distribution by Major Telecom Operators')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('operator_rbs_distribution.png')
print("Saved operator RBS distribution map to operator_rbs_distribution.png")

# 2. Bar chart comparing operator statistics
operators_list = list(operator_stats.keys())
heights = [operator_stats[op]['avg_height'] for op in operators_list]
powers = [operator_stats[op]['avg_power'] for op in operators_list]

# Set up bar chart
plt.figure(figsize=(12, 6))
x = np.arange(len(operators_list))
width = 0.35

fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

# Plot height bars
bars1 = ax1.bar(x - width/2, heights, width, label='Avg Antenna Height (m)', color='lightblue')
ax1.set_ylabel('Average Antenna Height (m)')

# Plot power bars
bars2 = ax2.bar(x + width/2, powers, width, label='Avg Transmitter Power (W)', color='salmon')
ax2.set_ylabel('Average Transmitter Power (W)')

# Set x-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels([op.split()[0] for op in operators_list])  # Use first word of operator name for clarity
ax1.set_xlabel('Operator')
ax1.set_title('Comparison of Antenna Height and Transmitter Power by Operator')

# Add legend
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('operator_stats_comparison.png')
print("Saved operator statistics comparison to operator_stats_comparison.png")

# 3. Technology distribution pie charts
plt.figure(figsize=(15, 5))
for i, (operator, stats) in enumerate(operator_stats.items()):
    # Filter out unknown or empty tech values
    tech_data = {k: v for k, v in stats['tech_distribution'].items() 
                if k != 'Unknown' and k is not None and k != ''}
    
    # Sort by count
    labels = [k for k, v in sorted(tech_data.items(), key=lambda x: x[1], reverse=True)]
    sizes = [tech_data[k] for k in labels]
    
    # Only show top 5 technologies
    if len(labels) > 5:
        other_count = sum(sizes[5:])
        labels = labels[:5] + ['Other']
        sizes = sizes[:5] + [other_count]
    
    # Create pie chart
    plt.subplot(1, 3, i+1)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title(f"{operator.split()[0]} Technology Distribution")

plt.tight_layout()
plt.savefig('operator_technology_distribution.png')
print("Saved operator technology distribution charts to operator_technology_distribution.png")

print("\nAnalysis completed successfully!") 